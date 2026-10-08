// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `sdm_core._native`, the core as Python loads it.
//!
//! Arrow data crosses through the Arrow C Data Interface without a copy, by
//! way of `_export_to_c` and `_import_from_c` where pyarrow lacks the
//! PyCapsule interface, so pyarrow 8.0.0 works. A diagnostic from the core is
//! raised as `CoreError`; nothing here panics on any input.

mod driver;
mod labels;
pub(crate) mod plain;
mod random;
mod request;
mod sampling;
mod schema;

use std::sync::{Arc, Mutex, PoisonError};

use arrow_array::ffi_stream::ArrowArrayStreamReader;
use arrow_array::{RecordBatch, RecordBatchReader};
use arrow_pyarrow::{FromPyArrow, IntoPyArrow, Table};
use arrow_schema::{Schema, SchemaRef};
use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyList;
use sdm_core::{
    Diagnostic, Endpoint, ModelCall, Outcome, Param, Query, QueryId, QueryResult, Run, Step, codes,
};

// Named for the module it is importable from, so that pickle can find it:
// Spark pickles what a Python worker raises back to the driver.
create_exception!(
    sdm_core._native,
    CoreError,
    PyException,
    "A problem the core reports; `code` names its kind."
);

create_exception!(
    sdm_core._native,
    CorePanic,
    PyValueError,
    "The core panicked: a bug in it, never a refusal of what it was given. \
     A `ValueError`, as every panic was before it had a class of its own."
);

/// A panic in `what`, as the `CorePanic` a host can tell from a refusal.
pub(crate) fn panicked(what: &str) -> PyErr {
    CorePanic::new_err(format!("{what} panicked; this is a bug in it"))
}

/// `diagnostic` as a `CoreError`, after `warnings` were given. Its
/// attributes, which pickling keeps along with the message, are plain data:
/// `code` (`None` where it has none), `error_class` (the class a host raises
/// it as, `None` for the run loop's own problems), `message`, `details`,
/// `status_code` and `warnings`, each `{"category", "message"}`.
pub(crate) fn core_error_after(
    py: Python<'_>,
    diagnostic: &Diagnostic,
    warnings: &[Diagnostic],
) -> PyErr {
    let error = CoreError::new_err(diagnostic.to_string());
    let set = || -> PyResult<()> {
        let value = error.value(py);
        value.setattr("code", diagnostic.code.as_deref())?;
        value.setattr("error_class", diagnostic.class.as_deref())?;
        value.setattr("message", diagnostic.message.as_str())?;
        let details = serde_json::Value::Object((*diagnostic.details).clone());
        value.setattr("details", plain::json_to_py(py, &details)?)?;
        value.setattr("status_code", diagnostic.status)?;
        value.setattr("warnings", warning_list(py, warnings)?)?;
        if !diagnostic.chain.is_empty() {
            let chain = serde_json::Value::Array((*diagnostic.chain).clone());
            value.setattr("chain", plain::json_to_py(py, &chain)?)?;
        }
        Ok(())
    };
    match set() {
        Ok(()) => error,
        Err(failed) => failed,
    }
}

fn core_error(py: Python<'_>, diagnostic: &Diagnostic) -> PyErr {
    core_error_after(py, diagnostic, &[])
}

/// Warnings as a list of `{"category", "message"}`.
pub(crate) fn warning_list<'py>(
    py: Python<'py>,
    warnings: &[Diagnostic],
) -> PyResult<Bound<'py, PyList>> {
    let items = warnings
        .iter()
        .map(|w| {
            let item = pyo3::types::PyDict::new(py);
            item.set_item(
                "category",
                w.category.as_deref().unwrap_or("builtins.UserWarning"),
            )?;
            item.set_item("message", w.message.as_str())?;
            if w.details
                .get("masked_float32_cast")
                .and_then(serde_json::Value::as_bool)
                == Some(true)
            {
                item.set_item("masked_float32_cast", true)?;
            }
            Ok(item)
        })
        .collect::<PyResult<Vec<_>>>()?;
    PyList::new(py, items)
}

/// `f`'s answer, with a panic as a `CorePanic`. A panic would be a bug in
/// the core; uncaught, pyo3 raises it as `PanicException`, which derives
/// from `BaseException`, so `except Exception` would miss it. Every entry
/// point Phase 7 added runs its whole body under this.
pub(crate) fn caught<T>(f: impl FnOnce() -> PyResult<T>) -> PyResult<T> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f))
        .unwrap_or_else(|_| Err(panicked("the core")))
}

/// The installed version, the crate's.
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

// --- Arrow ---------------------------------------------------------------

fn pyarrow_class<'py>(py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyAny>> {
    py.import("pyarrow")?.getattr(name)
}

/// A `pyarrow.Table` as its schema and batches, sharing its buffers.
///
/// Read batch by batch rather than as a stream, because a pyarrow without
/// the PyCapsule interface can export a batch but not a table.
pub(crate) fn import_table(table: &Bound<'_, PyAny>) -> PyResult<(SchemaRef, Vec<RecordBatch>)> {
    let schema = Arc::new(Schema::from_pyarrow_bound(&table.getattr("schema")?)?);
    let batches = table
        .call_method0("to_batches")?
        .try_iter()?
        .map(|batch| RecordBatch::from_pyarrow_bound(&batch?))
        .collect::<PyResult<Vec<_>>>()?;
    Ok((schema, batches))
}

pub(crate) fn export_table(
    py: Python<'_>,
    schema: SchemaRef,
    batches: Vec<RecordBatch>,
) -> PyResult<Bound<'_, PyAny>> {
    Table::try_new(batches, schema)
        .map_err(|e| PyValueError::new_err(e.to_string()))?
        .into_pyarrow(py)
}

/// `data`, a `pyarrow.Table` or `pyarrow.RecordBatchReader`, through Rust
/// and back, with every buffer where it was.
///
/// A reader's batches come back as they were. A table comes back equal, but
/// chunked as `to_batches()` splits it: where its columns' chunks end in
/// different places, each column is cut wherever any of them ends. That
/// follows from importing a table batch by batch, which pyarrow 8 needs.
#[pyfunction]
fn roundtrip_arrow<'py>(py: Python<'py>, data: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    if data.is_instance(&pyarrow_class(py, "Table")?)? {
        let (schema, batches) = import_table(data)?;
        return export_table(py, schema, batches);
    }
    if data.is_instance(&pyarrow_class(py, "RecordBatchReader")?)? {
        let reader: Box<dyn RecordBatchReader + Send> =
            Box::new(ArrowArrayStreamReader::from_pyarrow_bound(data)?);
        return reader.into_pyarrow(py);
    }
    Err(PyTypeError::new_err(format!(
        "roundtrip_arrow takes a pyarrow.Table or pyarrow.RecordBatchReader, not {}",
        data.get_type().name()?
    )))
}

// --- Payloads ------------------------------------------------------------

/// `json`, the body of `POST /v1/predictions`, parsed into the wire types
/// and serialized back; `ValueError` if it is not one.
#[pyfunction]
fn roundtrip_payload(json: &str) -> PyResult<String> {
    let request = sdm_wire::parse_prediction_request(json)
        .map_err(|e| PyValueError::new_err(format!("not a prediction request: {e}")))?;
    serde_json::to_string(&request)
        .map_err(|e| PyValueError::new_err(format!("cannot serialize the request: {e}")))
}

// --- The run loop ----------------------------------------------------------

/// What came of running a query: `Outcome.data(table)` or
/// `Outcome.error(sqlstate, vendor_code, message)`.
#[pyclass(name = "Outcome", module = "sdm_core._native", frozen)]
struct PyOutcome {
    inner: Outcome,
}

#[pymethods]
impl PyOutcome {
    /// The query's result, a `pyarrow.Table`.
    #[staticmethod]
    fn data(table: &Bound<'_, PyAny>) -> PyResult<Self> {
        if !table.is_instance(&pyarrow_class(table.py(), "Table")?)? {
            return Err(PyTypeError::new_err(format!(
                "Outcome.data takes a pyarrow.Table, not {}",
                table.get_type().name()?
            )));
        }
        let (schema, batches) = import_table(table)?;
        let result = QueryResult::try_new(schema, batches)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            inner: Outcome::Data(result),
        })
    }

    /// The query failed, with the driver's SQLSTATE and code, if it gave any.
    #[staticmethod]
    #[pyo3(signature = (sqlstate, vendor_code, message))]
    fn error(sqlstate: Option<String>, vendor_code: Option<i64>, message: String) -> Self {
        Self {
            inner: Outcome::Error {
                sqlstate,
                vendor_code,
                message,
            },
        }
    }

    /// The model's answer to a call, as JSON text.
    #[staticmethod]
    fn response(text: String) -> Self {
        Self {
            inner: Outcome::Response(text),
        }
    }

    /// The call failed: a JSON object of the error, with its `class`,
    /// `status`, `code`, `message` and `details`.
    #[staticmethod]
    fn model_error(text: String) -> Self {
        Self {
            inner: Outcome::ModelError(text),
        }
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            Outcome::Data(result) => format!("Outcome.data(<{} rows>)", result.num_rows()),
            Outcome::Error {
                sqlstate,
                vendor_code,
                message,
            } => format!("Outcome.error({sqlstate:?}, {vendor_code:?}, {message:?})"),
            Outcome::Response(text) => format!("Outcome.response(<{} characters>)", text.len()),
            Outcome::ModelError(text) => format!("Outcome.model_error({text:?})"),
        }
    }
}

/// A query a run needs the result of.
#[pyclass(name = "Query", module = "sdm_core._native", frozen)]
struct PyQuery {
    /// What `Run.submit` takes back with the query's outcome.
    #[pyo3(get)]
    id: u64,
    /// The statement.
    #[pyo3(get)]
    sql: String,
    params: Vec<Param>,
    /// The columns the run reads from the result, by name, in order.
    #[pyo3(get)]
    columns: Vec<String>,
}

#[pymethods]
impl PyQuery {
    /// The values of the statement's placeholders, in order.
    #[getter]
    fn params<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let values = self
            .params
            .iter()
            .map(|param| match param {
                Param::Null => Ok(py.None().into_bound(py)),
                Param::Bool(v) => Ok(v.into_pyobject(py)?.to_owned().into_any()),
                Param::Int(v) => Ok(v.into_pyobject(py)?.into_any()),
                Param::Float(v) => Ok(v.into_pyobject(py)?.into_any()),
                Param::Text(v) => Ok(v.into_pyobject(py)?.into_any()),
            })
            .collect::<PyResult<Vec<_>>>()?;
        PyList::new(py, values)
    }

    fn __repr__(&self) -> String {
        format!("Query(id={}, sql={:?})", self.id, self.sql)
    }
}

impl From<Query> for PyQuery {
    fn from(query: Query) -> Self {
        Self {
            id: query.id.0,
            sql: query.sql,
            params: query.params,
            columns: query.columns,
        }
    }
}

/// What a run does next: `kind` is `'run_queries'`, with `queries`,
/// `'call_model'`, with `id`, `endpoint`, `session_id` and `body`, or
/// `'done'`, with `value`. Every step has `warnings`: those the run gave
/// since its previous step, in order, each `{"category", "message"}`.
#[pyclass(name = "Step", module = "sdm_core._native", frozen)]
struct PyStep {
    #[pyo3(get)]
    kind: &'static str,
    #[pyo3(get)]
    warnings: Py<PyList>,
    #[pyo3(get)]
    queries: Option<Py<PyList>>,
    #[pyo3(get)]
    id: Option<u64>,
    #[pyo3(get)]
    endpoint: Option<&'static str>,
    #[pyo3(get)]
    session_id: Option<String>,
    #[pyo3(get)]
    body: Option<String>,
    #[pyo3(get)]
    value: Option<Py<PyAny>>,
}

impl PyStep {
    fn new(kind: &'static str, warnings: Py<PyList>) -> Self {
        Self {
            kind,
            warnings,
            queries: None,
            id: None,
            endpoint: None,
            session_id: None,
            body: None,
            value: None,
        }
    }
}

#[pymethods]
impl PyStep {
    fn __repr__(&self) -> String {
        format!("Step(kind={:?})", self.kind)
    }
}

/// What a run ends with, as Python is handed it.
pub(crate) enum Done {
    Count(u64),
    Texts(String, String),
    Target(Box<sdm_sampling::driver::TargetDone>),
    Subgraph(Box<sdm_sampling::driver::SubgraphDone>),
    Lagged(Box<sdm_sampling::driver::lagged::LaggedDone>),
    Task(Box<sdm_sampling::driver::task::TaskDone>),
    Context(Box<sdm_sampling::driver::task::ContextDone>),
    MissingIndices(
        sdm_sampling::driver::init::MissingIndices,
        Option<sdm_sampling::driver::output::Warning>,
    ),
    /// What today's method raised, which `next()` raises.
    Raised(Box<sdm_sampling::driver::Raised>),
    /// Plain data.
    Plain(Box<sdm_core::plain::Plain>),
}

/// A run whose value becomes a [`Done`].
pub(crate) struct Mapped<T, R> {
    run: R,
    map: fn(T) -> Done,
}

impl<T, R: Run<T>> Mapped<T, R> {
    pub(crate) fn new(run: R, map: fn(T) -> Done) -> Self {
        Self { run, map }
    }
}

impl<T, R: Run<T>> Run<Done> for Mapped<T, R> {
    fn next(&mut self) -> Result<Step<Done>, Diagnostic> {
        Ok(match self.run.next()? {
            Step::RunQueries(queries) => Step::RunQueries(queries),
            Step::CallModel(call) => Step::CallModel(call),
            Step::Done(value) => Step::Done((self.map)(value)),
        })
    }

    fn submit(&mut self, id: QueryId, outcome: Outcome) -> Result<(), Diagnostic> {
        self.run.submit(id, outcome)
    }

    fn take_warnings(&mut self) -> Vec<Diagnostic> {
        self.run.take_warnings()
    }
}

/// A run of the core, driven by `next()` and `submit(id, outcome)`.
#[pyclass(name = "Run", module = "sdm_core._native", frozen)]
pub(crate) struct PyRun {
    /// `None` once the core panicked in it.
    inner: Mutex<Option<Box<dyn Run<Done> + Send>>>,
}

impl PyRun {
    pub(crate) fn new(run: Box<dyn Run<Done> + Send>) -> Self {
        Self {
            inner: Mutex::new(Some(run)),
        }
    }

    fn with<R: Send>(
        &self,
        py: Python<'_>,
        f: impl FnOnce(&mut dyn Run<Done>) -> Result<R, Diagnostic> + Send,
    ) -> PyResult<R> {
        self.with_warnings(py, f).map(|(value, _)| value)
    }

    /// `f`'s answer and the warnings the run gave meanwhile; a diagnostic
    /// is raised carrying them. The core works without the GIL, so other
    /// Python threads run meanwhile.
    fn with_warnings<R: Send>(
        &self,
        py: Python<'_>,
        f: impl FnOnce(&mut dyn Run<Done>) -> Result<R, Diagnostic> + Send,
    ) -> PyResult<(R, Vec<Diagnostic>)> {
        let stepped = py.detach(|| {
            // A panic is caught before it leaves the guard's scope, so the
            // lock is never poisoned.
            let mut slot = self.inner.lock().unwrap_or_else(PoisonError::into_inner);
            let Some(run) = slot.as_mut() else {
                return Stepped::Finished;
            };
            // A panic would be a bug in the core: it reaches Python as a
            // ValueError, and the run, whose state it may have left half
            // changed, is dropped, so every later call says it has finished.
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let result = f(run.as_mut());
                (result, run.take_warnings())
            })) {
                Ok((Ok(value), warnings)) => Stepped::Answered(value, warnings),
                Ok((Err(diagnostic), warnings)) => Stepped::Refused(diagnostic, warnings),
                Err(_) => {
                    *slot = None;
                    Stepped::Panicked
                }
            }
        });
        match stepped {
            Stepped::Answered(value, warnings) => Ok((value, warnings)),
            Stepped::Refused(diagnostic, warnings) => {
                Err(core_error_after(py, &diagnostic, &warnings))
            }
            Stepped::Finished => Err(core_error(
                py,
                &Diagnostic::new(
                    codes::RUN_FINISHED,
                    "the run stopped when the core panicked in it",
                ),
            )),
            Stepped::Panicked => Err(panicked("the core")),
        }
    }
}

/// How a step of a run went, told once the GIL is held again.
enum Stepped<R> {
    Answered(R, Vec<Diagnostic>),
    Refused(Diagnostic, Vec<Diagnostic>),
    /// The run had stopped, when the core panicked in it earlier.
    Finished,
    Panicked,
}

#[pymethods]
impl PyRun {
    /// The next step: queries to run, a call to make, or the run's result.
    fn next(&self, py: Python<'_>) -> PyResult<PyStep> {
        let (step, warnings) = self.with_warnings(py, |run| run.next())?;
        let given = warning_list(py, &warnings)?.unbind();
        match step {
            Step::RunQueries(queries) => {
                let queries = queries
                    .into_iter()
                    .map(|query| Py::new(py, PyQuery::from(query)))
                    .collect::<PyResult<Vec<_>>>()?;
                let mut step = PyStep::new("run_queries", given);
                step.queries = Some(PyList::new(py, queries)?.unbind());
                Ok(step)
            }
            Step::CallModel(ModelCall {
                id,
                endpoint,
                session_id,
                body,
            }) => {
                let mut step = PyStep::new("call_model", given);
                step.id = Some(id.0);
                step.endpoint = Some(endpoint.name());
                step.session_id = session_id;
                step.body = body;
                let _: Endpoint = endpoint;
                Ok(step)
            }
            Step::Done(value) => {
                let value = match value {
                    Done::Count(n) => n.into_pyobject(py)?.into_any().unbind(),
                    Done::Texts(a, b) => (a, b).into_pyobject(py)?.into_any().unbind(),
                    Done::Target(done) => driver::target_value(py, &done)?.unbind(),
                    Done::Subgraph(done) => driver::subgraph_value(py, &done)?.unbind(),
                    Done::Lagged(done) => driver::lagged_value(py, &done)?.unbind(),
                    Done::Task(done) => driver::task_value(py, &done)?.unbind(),
                    Done::Context(done) => driver::context_value(py, &done)?.unbind(),
                    Done::MissingIndices(missing, warning) => {
                        driver::missing_indices_value(py, &missing, warning.as_ref())?.unbind()
                    }
                    Done::Raised(raised) => return Err(driver::raised_error(py, &raised)),
                    Done::Plain(value) => plain::to_py(py, &value)?.unbind(),
                };
                let mut step = PyStep::new("done", given);
                step.value = Some(value);
                Ok(step)
            }
        }
    }

    /// Give the run the outcome of the query `id`.
    fn submit(&self, py: Python<'_>, id: u64, outcome: &Bound<'_, PyOutcome>) -> PyResult<()> {
        let outcome = outcome.get().inner.clone();
        self.with(py, |run| run.submit(QueryId(id), outcome))
    }
}

// --- PQL --------------------------------------------------------------------

/// What pandas' reading of a date takes from this Python: the local year,
/// which `dateutil` reads two-digit years against, and whether `strftime`
/// pads `%Y` to four digits, which decides some of pandas' format guesses.
fn date_host(py: Python<'_>) -> PyResult<sdm_pql::dates::Host> {
    let year = py
        .import("time")?
        .call_method0("localtime")?
        .getattr("tm_year")?
        .extract()?;
    let year_one: String = py
        .import("datetime")?
        .getattr("datetime")?
        .call1((1, 1, 1))?
        .call_method1("strftime", ("%Y",))?
        .extract()?;
    Ok(sdm_pql::dates::Host {
        year,
        pads_year: year_one == "0001",
    })
}

/// `query` resolved against `schema_json` under the profile named
/// `profile`, as JSON: `{"tree", "warnings"}` or `{"errors", "warnings"}`,
/// as today's `predictive_query.validate` answers. `mode`, if given,
/// replaces the profile's. `ValueError` for a schema, profile or mode it
/// cannot read.
#[pyfunction]
#[pyo3(signature = (query, schema_json, profile, mode=None))]
fn pql_validate(
    py: Python<'_>,
    query: &str,
    schema_json: &str,
    profile: &str,
    mode: Option<&str>,
) -> PyResult<String> {
    let host = date_host(py)?;
    let (query, schema_json, profile) = (
        query.to_string(),
        schema_json.to_string(),
        profile.to_string(),
    );
    let mode = mode.map(str::to_string);
    py.detach(move || {
        sdm_pql::validate_json_in(&query, &schema_json, &profile, mode.as_deref(), host)
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// How a date literal reads, as `pd.to_datetime` reads it: JSON
/// `{"value", "warnings"}`, `value` null when refused.
#[pyfunction]
fn pql_parse_date(py: Python<'_>, literal: &str) -> PyResult<String> {
    Ok(sdm_pql::parse_date_json_in(literal, date_host(py)?))
}

// --- Dialects ------------------------------------------------------------

/// One call of the dialect-call capture (`tests/rust_core/dialects/CONTRACT.md`):
/// `function` of `backend` with `args_json`, its arguments by name, as the
/// capture encodes them. A statement function gives `(sql, params_json)`, any
/// other `(None, result_json)`. `ValueError` for an unknown backend or
/// function, or arguments it cannot read. Test-only: Python keeps its own
/// SQL until the switch.
#[pyfunction]
fn dialect_render(
    py: Python<'_>,
    backend: &str,
    function: &str,
    args_json: &str,
) -> PyResult<(Option<String>, String)> {
    let (backend, function, args_json) = (
        backend.to_string(),
        function.to_string(),
        args_json.to_string(),
    );
    // A panic would be a bug in the port; it reaches Python as a
    // `CorePanic`, a `ValueError` as every other refusal is, never as an
    // abort.
    let rendered = py
        .detach(move || {
            std::panic::catch_unwind(|| sdm_dialects::render(&backend, &function, &args_json))
        })
        .map_err(|_| panicked("the dialect port"))?;
    match rendered.map_err(|e| PyValueError::new_err(e.to_string()))? {
        sdm_dialects::Rendered::Statement { sql, params_json } => Ok((Some(sql), params_json)),
        sdm_dialects::Rendered::Value { result_json } => Ok((None, result_json)),
    }
}

#[pyfunction]
fn prepare_snapshot(py: Python<'_>, files_json: String) -> PyResult<PyPreparedSnapshot> {
    py.detach(move || {
        std::panic::catch_unwind(|| sdm_core::snapshot::PreparedSnapshot::from_json(&files_json))
    })
    .map_err(|_| panicked("snapshot preparation"))?
    .map(|snapshot| PyPreparedSnapshot {
        snapshot: Arc::new(snapshot),
    })
    .map_err(|error| PyValueError::new_err(error.to_string()))
}

#[pyclass(name = "PreparedSnapshot", module = "sdm_core._native", frozen)]
struct PyPreparedSnapshot {
    snapshot: Arc<sdm_core::snapshot::PreparedSnapshot>,
}

#[pyfunction]
fn replay_snapshot(
    py: Python<'_>,
    snapshot: PyRef<'_, PyPreparedSnapshot>,
    case_id: String,
) -> PyResult<String> {
    let snapshot = Arc::clone(&snapshot.snapshot);
    py.detach(move || {
        std::panic::catch_unwind(|| {
            snapshot
                .replay_case_json(&case_id)
                .map(|output| format!("{{{}:{output}}}", serde_json::to_string(&case_id).unwrap()))
        })
    })
    .map_err(|_| panicked("snapshot replay"))?
    .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// The toy run, "count the rows of table T".
#[pyfunction]
fn count_rows(table: String) -> PyRun {
    PyRun::new(Box::new(Mapped::new(
        sdm_core::count_rows(table),
        Done::Count,
    )))
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("CoreError", m.py().get_type::<CoreError>())?;
    m.add("CorePanic", m.py().get_type::<CorePanic>())?;
    m.add_class::<PyOutcome>()?;
    m.add_class::<PyQuery>()?;
    m.add_class::<PyStep>()?;
    m.add_class::<PyRun>()?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(roundtrip_arrow, m)?)?;
    m.add_function(wrap_pyfunction!(roundtrip_payload, m)?)?;
    m.add_function(wrap_pyfunction!(count_rows, m)?)?;
    m.add_function(wrap_pyfunction!(pql_validate, m)?)?;
    m.add_function(wrap_pyfunction!(pql_parse_date, m)?)?;
    m.add_function(wrap_pyfunction!(dialect_render, m)?)?;
    m.add_class::<PyPreparedSnapshot>()?;
    m.add_function(wrap_pyfunction!(prepare_snapshot, m)?)?;
    m.add_function(wrap_pyfunction!(replay_snapshot, m)?)?;
    m.add_function(wrap_pyfunction!(schema::dates_op, m)?)?;
    m.add_function(wrap_pyfunction!(schema::composite_op, m)?)?;
    m.add_function(wrap_pyfunction!(schema::metric_view_parse, m)?)?;
    m.add_function(wrap_pyfunction!(schema::schema_from_data, m)?)?;
    m.add_function(wrap_pyfunction!(schema::schema_from_source, m)?)?;
    m.add_function(wrap_pyfunction!(schema::semantic_view_parse, m)?)?;
    m.add_class::<random::PyRandomState>()?;
    m.add_class::<random::PyMt19937>()?;
    m.add_function(wrap_pyfunction!(random::random_state, m)?)?;
    m.add_function(wrap_pyfunction!(sampling::date_offset_apply, m)?)?;
    m.add_function(wrap_pyfunction!(sampling::date_offset_to_seconds, m)?)?;
    m.add_function(wrap_pyfunction!(sampling::date_offset_to_ns, m)?)?;
    m.add_function(wrap_pyfunction!(sampling::date_offset_pool, m)?)?;
    m.add_function(wrap_pyfunction!(sampling::normalize_text, m)?)?;
    m.add_class::<sampling::PyNeighborSampler>()?;
    m.add_function(wrap_pyfunction!(sampling::graph_store, m)?)?;
    m.add_function(wrap_pyfunction!(labels::labels_op, m)?)?;
    m.add_function(wrap_pyfunction!(sampling::sampling_op, m)?)?;
    m.add_function(wrap_pyfunction!(labels::labels_execute, m)?)?;
    m.add_function(wrap_pyfunction!(driver::sample_target, m)?)?;
    m.add_function(wrap_pyfunction!(driver::sample_subgraph, m)?)?;
    m.add_function(wrap_pyfunction!(driver::sample_lagged_target, m)?)?;
    m.add_function(wrap_pyfunction!(driver::task_table, m)?)?;
    m.add_function(wrap_pyfunction!(driver::context, m)?)?;
    m.add_function(wrap_pyfunction!(driver::missing_indices, m)?)?;
    m.add_function(wrap_pyfunction!(request::payload_size_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(request::request_fixture_op, m)?)?;
    m.add_function(wrap_pyfunction!(request::schema_edit, m)?)?;
    m.add_function(wrap_pyfunction!(request::model_spec, m)?)?;
    m.add_function(wrap_pyfunction!(request::request_op, m)?)?;
    m.add_function(wrap_pyfunction!(request::gather_graph_run, m)?)?;
    m.add_function(wrap_pyfunction!(request::flatten, m)?)?;
    m.add_function(wrap_pyfunction!(request::merge, m)?)?;
    m.add_function(wrap_pyfunction!(request::absent, m)?)?;
    m.add_function(wrap_pyfunction!(request::schema_source_run, m)?)?;
    m.add_function(wrap_pyfunction!(request::data_schema, m)?)?;
    m.add_function(wrap_pyfunction!(request::materialize_run, m)?)?;
    m.add_function(wrap_pyfunction!(request::predict_run, m)?)?;
    m.add_function(wrap_pyfunction!(request::fit_run, m)?)?;
    m.add_function(wrap_pyfunction!(request::session_predict_run, m)?)?;
    m.add_function(wrap_pyfunction!(request::request_build, m)?)?;
    m.add_function(wrap_pyfunction!(request::request_materialized, m)?)?;
    m.add_function(wrap_pyfunction!(request::query_text, m)?)?;
    m.add(
        "OutOfBoundsDatetime",
        m.py().get_type::<sampling::OutOfBoundsDatetime>(),
    )?;
    m.add(
        "OutOfBoundsTimedelta",
        m.py().get_type::<sampling::OutOfBoundsTimedelta>(),
    )?;
    m.add(
        "DateParseError",
        m.py().get_type::<sampling::DateParseError>(),
    )?;
    Ok(())
}
