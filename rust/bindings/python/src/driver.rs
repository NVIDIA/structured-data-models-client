// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The sampling driver through Python
//! (`tests/rust_core/sampling/driver/CONTRACT.md`, "The Python binding"):
//! `sample_target` and `sample_subgraph` as runs of the core, their `done`
//! values as Python dicts of Arrow arrays. Test-only until Phase 9 switches
//! Python over to the core.
//!
//! Where today's method raises, `next()` raises the exception it names, and
//! the exception's `warnings` attribute holds the warnings the method gave
//! before it raised, in the form a `done` value's `warnings` takes, for the
//! host to give before it re-raises.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{
    ArrayRef, Int64Array, RecordBatch, RecordBatchOptions, TimestampNanosecondArray,
    new_empty_array,
};
use arrow_pyarrow::ToPyArrow;
use arrow_schema::{Field, Schema};
use pyo3::exceptions::{PyKeyError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use sdm_sampling::driver::output::{SampledLink, SampledTable, TargetPart, Warning};
use sdm_sampling::driver::{Ended, Raised, SubgraphDone, TargetDone};
use sdm_sampling::frame::{Frame, Series};

use crate::{Done, Mapped, PyRun, caught};

/// The field metadata key the host reads a column's pandas dtype from.
const DTYPE_KEY: &str = "sdm.pandas_dtype";

/// The tables a run is given: each `pyarrow.Table` as one record batch.
fn tables(tables: &Bound<'_, PyDict>) -> PyResult<Vec<(String, RecordBatch)>> {
    let mut out = Vec::new();
    for (name, table) in tables.iter() {
        let name: String = name.extract()?;
        crate::sampling::require_table(&table)?;
        let combined = table.call_method0("combine_chunks")?;
        let (schema, batches) = crate::import_table(&combined)?;
        let batch = match batches.len() {
            0 => RecordBatch::new_empty(schema),
            1 => batches.into_iter().next().unwrap_or_else(|| unreachable!()),
            _ => arrow_select_concat(&schema, &batches)?,
        };
        out.push((name, batch));
    }
    Ok(out)
}

fn arrow_select_concat(
    schema: &arrow_schema::SchemaRef,
    batches: &[RecordBatch],
) -> PyResult<RecordBatch> {
    let columns: Vec<ArrayRef> = (0..schema.fields().len())
        .map(|i| {
            let arrays: Vec<&dyn arrow_array::Array> =
                batches.iter().map(|b| b.column(i).as_ref()).collect();
            arrow_select::concat::concat(&arrays).map_err(|e| PyValueError::new_err(e.to_string()))
        })
        .collect::<PyResult<_>>()?;
    let rows = batches.iter().map(RecordBatch::num_rows).sum();
    RecordBatch::try_new_with_options(
        schema.clone(),
        columns,
        &RecordBatchOptions::new().with_row_count(Some(rows)),
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// `sample_target(inputs_json, tables)`: the run of `Sampler.sample_target`
/// the call's description starts.
#[pyfunction]
pub(crate) fn sample_target(
    py: Python<'_>,
    inputs_json: &str,
    tables: &Bound<'_, PyDict>,
) -> PyResult<PyRun> {
    caught(|| {
        let host = crate::schema::host(py)?;
        let given = self::tables(tables)?;
        let run = sdm_sampling::driver::sample_target_run(inputs_json, given, host);
        Ok(PyRun::new(Box::new(Mapped::new(run, |ended| {
            or_raised(ended, |done| Done::Target(Box::new(done)))
        }))))
    })
}

/// `sample_subgraph(inputs_json, tables)`: the run of
/// `Sampler.sample_subgraph` the call's description starts.
#[pyfunction]
pub(crate) fn sample_subgraph(
    py: Python<'_>,
    inputs_json: &str,
    tables: &Bound<'_, PyDict>,
) -> PyResult<PyRun> {
    caught(|| {
        let host = crate::schema::host(py)?;
        let given = self::tables(tables)?;
        let run = sdm_sampling::driver::sample_subgraph_run(inputs_json, given, host);
        Ok(PyRun::new(Box::new(Mapped::new(run, |ended| {
            or_raised(ended, |done| Done::Subgraph(Box::new(done)))
        }))))
    })
}

/// `sample_lagged_target(inputs_json, tables)`: the run of
/// `Sampler.sample_lagged_target` the call's description starts. Its `done`
/// value holds the frame of lagged labels and the warnings.
#[pyfunction]
pub(crate) fn sample_lagged_target(
    py: Python<'_>,
    inputs_json: &str,
    tables: &Bound<'_, PyDict>,
) -> PyResult<PyRun> {
    caught(|| {
        let host = crate::schema::host(py)?;
        let given = self::tables(tables)?;
        let run = sdm_sampling::driver::sample_lagged_target_run(inputs_json, given, host);
        Ok(PyRun::new(Box::new(Mapped::new(run, |ended| {
            or_raised(ended, |done| Done::Lagged(Box::new(done)))
        }))))
    })
}

/// `task_table(inputs_json, tables)`: the run of
/// `KumoRelational._get_task_table` the call's description starts.
#[pyfunction]
pub(crate) fn task_table(
    py: Python<'_>,
    inputs_json: &str,
    tables: &Bound<'_, PyDict>,
) -> PyResult<PyRun> {
    caught(|| {
        let host = crate::schema::host(py)?;
        let given = self::tables(tables)?;
        let run = sdm_sampling::driver::task_table_run(inputs_json, given, host);
        Ok(PyRun::new(Box::new(Mapped::new(run, |ended| {
            or_raised(ended, |done| Done::Task(Box::new(done)))
        }))))
    })
}

/// `context(inputs_json, tables)`: the run of `KumoRelational._get_context`
/// the call's description starts.
#[pyfunction]
pub(crate) fn context(
    py: Python<'_>,
    inputs_json: &str,
    tables: &Bound<'_, PyDict>,
) -> PyResult<PyRun> {
    caught(|| {
        let host = crate::schema::host(py)?;
        let given = self::tables(tables)?;
        let run = sdm_sampling::driver::context_run(inputs_json, given, host);
        Ok(PyRun::new(Box::new(Mapped::new(run, |ended| {
            or_raised(ended, |done| Done::Context(Box::new(done)))
        }))))
    })
}

/// `_get_task_table`'s `done` value: the task's two frames and what the
/// `TaskTable` is made with.
pub(crate) fn task_value<'py>(
    py: Python<'py>,
    done: &sdm_sampling::driver::task::TaskDone,
) -> PyResult<Bound<'py, PyAny>> {
    let (context, prediction) = done
        .frames()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let out = PyDict::new(py);
    out.set_item("task_type", &done.task_type)?;
    out.set_item("entity_table_names", done.entity_table_names.clone())?;
    out.set_item("context", frame(py, &context)?)?;
    out.set_item("prediction", frame(py, &prediction)?)?;
    out.set_item("time_column", &done.time_column)?;
    out.set_item("num_forecasts", done.num_forecasts)?;
    out.set_item("step_size", done.step_size)?;
    out.set_item("warnings", warnings(py, &done.warnings)?)?;
    Ok(out.into_any())
}

/// `_get_context`'s `done` value: the sampled subgraph and the task's
/// feature table.
pub(crate) fn context_value<'py>(
    py: Python<'py>,
    done: &sdm_sampling::driver::task::ContextDone,
) -> PyResult<Bound<'py, PyAny>> {
    let out = PyDict::new(py);
    out.set_item("subgraph", subgraph_value(py, &done.subgraph)?)?;
    match &done.task_table {
        None => out.set_item("task_table", py.None())?,
        Some(table) => out.set_item("task_table", frame(py, table)?)?,
    }
    Ok(out.into_any())
}

/// `missing_indices(inputs_json, tables)`: the run of the checks building
/// a DuckDB or SQLite sampler makes.
#[pyfunction]
pub(crate) fn missing_indices(
    py: Python<'_>,
    inputs_json: &str,
    tables: &Bound<'_, PyDict>,
) -> PyResult<PyRun> {
    caught(|| {
        let host = crate::schema::host(py)?;
        let given = self::tables(tables)?;
        let run = sdm_sampling::driver::missing_indices_run(inputs_json, given, host);
        Ok(PyRun::new(Box::new(Mapped::new(run, |ended| {
            or_raised(ended, |(missing, warning)| {
                Done::MissingIndices(missing, warning)
            })
        }))))
    })
}

/// The checks' `done` value: the keys by table, and the warning or `None`.
pub(crate) fn missing_indices_value<'py>(
    py: Python<'py>,
    missing: &sdm_sampling::driver::init::MissingIndices,
    warning: Option<&Warning>,
) -> PyResult<Bound<'py, PyAny>> {
    let out = PyDict::new(py);
    let by_table = PyDict::new(py);
    for (table, indices) in missing {
        by_table.set_item(table, indices.clone())?;
    }
    out.set_item("missing", by_table)?;
    out.set_item(
        "warnings",
        warnings(py, &warning.cloned().into_iter().collect::<Vec<_>>())?,
    )?;
    Ok(out.into_any())
}

/// `sample_lagged_target`'s `done` value: `frame`, a `pyarrow.Table`, and
/// the warnings.
pub(crate) fn lagged_value<'py>(
    py: Python<'py>,
    done: &sdm_sampling::driver::lagged::LaggedDone,
) -> PyResult<Bound<'py, PyAny>> {
    let out = PyDict::new(py);
    out.set_item("frame", frame(py, &done.frame)?)?;
    out.set_item("warnings", warnings(py, &done.warnings)?)?;
    Ok(out.into_any())
}

/// How a run ends as Python is handed it: its value, or what it raised.
fn or_raised<T>(ended: Ended<T>, done: impl FnOnce(T) -> Done) -> Done {
    match ended {
        Ok(value) => done(value),
        Err(raised) => Done::Raised(Box::new(raised)),
    }
}

/// The exception today's Python raised, by its class's name and text, with
/// the warnings it gave first as its `warnings` attribute.
pub(crate) fn raised_error(py: Python<'_>, raised: &Raised) -> PyErr {
    let error = python_error(py, &raised.class, raised.message.clone());
    let given =
        warnings(py, &raised.warnings).and_then(|given| error.value(py).setattr("warnings", given));
    match given {
        Ok(()) => error,
        Err(failed) => failed,
    }
}

/// The exception named `class` with the text `message`: a builtin where
/// the name is one, otherwise a class of that name.
fn python_error(py: Python<'_>, class: &str, message: String) -> PyErr {
    match class {
        "KeyError" => PyKeyError::new_err(message),
        "TypeError" => PyTypeError::new_err(message),
        "ValueError" => PyValueError::new_err(message),
        "RuntimeError" => pyo3::exceptions::PyRuntimeError::new_err(message),
        "OverflowError" => pyo3::exceptions::PyOverflowError::new_err(message),
        "AssertionError" => pyo3::exceptions::PyAssertionError::new_err(message),
        "IndexError" => pyo3::exceptions::PyIndexError::new_err(message),
        "ZeroDivisionError" => pyo3::exceptions::PyZeroDivisionError::new_err(message),
        "MemoryError" => pyo3::exceptions::PyMemoryError::new_err(message),
        "OutOfBoundsDatetime" => crate::sampling::OutOfBoundsDatetime::new_err(message),
        "OutOfBoundsTimedelta" => crate::sampling::OutOfBoundsTimedelta::new_err(message),
        name => named_error(py, name, message),
    }
}

/// An exception of a class named `name`, which only its name is known of.
fn named_error(py: Python<'_>, name: &str, message: String) -> PyErr {
    let made = (|| -> PyResult<PyErr> {
        let builtins = py.import("builtins")?;
        let base = builtins.getattr("Exception")?;
        let class =
            builtins
                .getattr("type")?
                .call1((name, PyTuple::new(py, [base])?, PyDict::new(py)))?;
        let error = class.call1((message.clone(),))?;
        Ok(PyErr::from_value(error))
    })();
    made.unwrap_or_else(|_| PyValueError::new_err(message))
}

fn array<'py>(py: Python<'py>, array: &ArrayRef) -> PyResult<Bound<'py, PyAny>> {
    array.to_data().to_pyarrow(py)
}

fn ints<'py>(py: Python<'py>, values: &[i64]) -> PyResult<Bound<'py, PyAny>> {
    let array: ArrayRef = Arc::new(Int64Array::from(values.to_vec()));
    self::array(py, &array)
}

fn warnings<'py>(py: Python<'py>, given: &[Warning]) -> PyResult<Bound<'py, PyList>> {
    let items = given
        .iter()
        .map(|w| {
            let item = PyDict::new(py);
            item.set_item("category", &w.category)?;
            item.set_item("message", &w.message)?;
            if w.masked_float32_cast {
                item.set_item("masked_float32_cast", true)?;
            }
            Ok(item)
        })
        .collect::<PyResult<Vec<_>>>()?;
    PyList::new(py, items)
}

fn part<'py>(py: Python<'py>, given: &TargetPart) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("entity_pkey", array(py, &given.entity_pkey.array)?)?;
    let times: ArrayRef = Arc::new(TimestampNanosecondArray::from(given.anchor_time.clone()));
    out.set_item("anchor_time", array(py, &times)?)?;
    out.set_item("target", array(py, &given.target.array)?)?;
    let dtypes = PyDict::new(py);
    dtypes.set_item("entity_pkey", given.entity_pkey.dtype.tag())?;
    dtypes.set_item("target", given.target.dtype.tag())?;
    out.set_item("dtypes", dtypes)?;
    Ok(out)
}

/// `sample_target`'s `done` value.
pub(crate) fn target_value<'py>(py: Python<'py>, done: &TargetDone) -> PyResult<Bound<'py, PyAny>> {
    let out = PyDict::new(py);
    out.set_item("train", part(py, &done.train)?)?;
    out.set_item("test", part(py, &done.test)?)?;
    out.set_item("warnings", warnings(py, &done.warnings)?)?;
    Ok(out.into_any())
}

/// A frame as a `pyarrow.Table`, each column's pandas dtype in its field's
/// metadata.
fn frame<'py>(py: Python<'py>, given: &Frame) -> PyResult<Bound<'py, PyAny>> {
    let fields: Vec<Field> = given
        .names
        .iter()
        .zip(&given.columns)
        .map(|(name, series): (&String, &Series)| {
            Field::new(name, series.array.data_type().clone(), true)
                .with_metadata(HashMap::from([(DTYPE_KEY.to_string(), series.dtype.tag())]))
        })
        .collect();
    let schema = Arc::new(Schema::new(fields));
    let columns: Vec<ArrayRef> = given.columns.iter().map(|s| s.array.clone()).collect();
    let batch = RecordBatch::try_new_with_options(
        schema.clone(),
        columns,
        &RecordBatchOptions::new().with_row_count(Some(given.rows)),
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let _ = new_empty_array;
    crate::export_table(py, schema, vec![batch])
}

fn optional_ints<'py>(py: Python<'py>, values: &Option<Vec<i64>>) -> PyResult<Bound<'py, PyAny>> {
    match values {
        None => Ok(py.None().into_bound(py)),
        Some(values) => ints(py, values),
    }
}

fn table<'py>(py: Python<'py>, given: &SampledTable) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("df", frame(py, &given.df)?)?;
    out.set_item("row", optional_ints(py, &given.row)?)?;
    out.set_item("batch", ints(py, &given.batch)?)?;
    out.set_item("num_sampled_nodes", given.num_sampled_nodes.clone())?;
    let stypes = PyDict::new(py);
    for (column, stype) in &given.stype_dict {
        stypes.set_item(column, stype)?;
    }
    out.set_item("stype_dict", stypes)?;
    out.set_item("primary_key", given.primary_key.clone())?;
    Ok(out)
}

fn link<'py>(py: Python<'py>, given: &SampledLink) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("layout", given.layout)?;
    out.set_item("row", optional_ints(py, &given.row)?)?;
    out.set_item("col", optional_ints(py, &given.col)?)?;
    out.set_item("num_sampled_edges", given.num_sampled_edges.clone())?;
    Ok(out)
}

/// `sample_subgraph`'s `done` value.
pub(crate) fn subgraph_value<'py>(
    py: Python<'py>,
    done: &SubgraphDone,
) -> PyResult<Bound<'py, PyAny>> {
    let out = PyDict::new(py);
    out.set_item("anchor_time", ints(py, &done.anchor_time)?)?;
    let tables = PyDict::new(py);
    for given in &done.tables {
        tables.set_item(&given.name, table(py, given)?)?;
    }
    out.set_item("tables", tables)?;
    let links = PyDict::new(py);
    for given in &done.links {
        let (src, rel, dst) = &given.edge_type;
        links.set_item(
            PyTuple::new(py, [src.as_str(), rel.as_str(), dst.as_str()])?,
            link(py, given)?,
        )?;
    }
    out.set_item("links", links)?;
    out.set_item("warnings", warnings(py, &done.warnings)?)?;
    Ok(out.into_any())
}
