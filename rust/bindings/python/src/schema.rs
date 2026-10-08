// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The schema core through Python (`tests/rust_core/schema/CONTRACT.md`):
//! test-only until Phase 9 switches Python over to it. Every function takes
//! and gives JSON text, refuses what it cannot read with `ValueError`, and
//! turns a panic, which would be a bug in the port, into that `ValueError`.

use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use sdm_schema::dt::Host;
use sdm_schema::dt::pandas::TimestampSeriesUnits;

/// What reading dates takes from this Python process: the local year and
/// date, the local zone's names, and whether `strftime` pads `%Y`.
pub fn host(py: Python<'_>) -> PyResult<Host> {
    let time = py.import("time")?;
    let local = time.call_method0("localtime")?;
    let year: i64 = local.getattr("tm_year")?.extract()?;
    let month: u32 = local.getattr("tm_mon")?.extract()?;
    let day: u32 = local.getattr("tm_mday")?.extract()?;
    let tznames: Vec<String> = time.getattr("tzname")?.extract()?;
    let year_one: String = py
        .import("datetime")?
        .getattr("datetime")?
        .call1((1, 1, 1))?
        .call_method1("strftime", ("%Y",))?
        .extract()?;
    Ok(Host {
        year,
        pads_year: year_one == "0001",
        today: (year, month, day),
        tznames,
        timestamp_series_units: TimestampSeriesUnits::default(),
        numpy_text_prefix: false,
    })
}

fn guarded<T>(f: impl FnOnce() -> Result<T, String> + std::panic::UnwindSafe) -> PyResult<T> {
    match std::panic::catch_unwind(f) {
        Ok(Ok(value)) => Ok(value),
        Ok(Err(message)) => Err(PyValueError::new_err(message)),
        Err(_) => Err(crate::panicked("the schema core")),
    }
}

/// The `dates.*`, `time_column.read` and `infer.timestamp` fixture
/// operations: `{"output"}` or `{"error"}` as JSON text.
#[pyfunction]
pub fn dates_op(
    py: Python<'_>,
    op: &str,
    values_json: &str,
    options_json: &str,
) -> PyResult<String> {
    let host = host(py)?;
    let (op, values_json, options_json) = (
        op.to_string(),
        values_json.to_string(),
        options_json.to_string(),
    );
    py.detach(move || {
        guarded(move || {
            let values: serde_json::Value = serde_json::from_str(&values_json)
                .map_err(|e| format!("the values are not JSON: {e}"))?;
            let options: serde_json::Value = serde_json::from_str(&options_json)
                .map_err(|e| format!("the options are not JSON: {e}"))?;
            let serde_json::Value::Object(mut input) = options else {
                return Err("the options are a JSON object".to_string());
            };
            if !values.is_null() {
                input.insert("values".to_string(), values);
            }
            if !(op.starts_with("dates.")
                || op.starts_with("time_column.")
                || op == "infer.timestamp")
            {
                return Err(format!("no dates operation {op:?}"));
            }
            sdm_schema::fixture_op_in(&op, &serde_json::Value::Object(input).to_string(), &host)
                .map_err(|e| e.to_string())
        })
    })
}

/// The `composite.*` fixture operations, on a case's whole input.
#[pyfunction]
pub fn composite_op(py: Python<'_>, op: &str, input_json: &str) -> PyResult<String> {
    let host = host(py)?;
    let (op, input_json) = (op.to_string(), input_json.to_string());
    py.detach(move || {
        guarded(move || {
            if !op.starts_with("composite.") {
                return Err(format!("no composite operation {op:?}"));
            }
            sdm_schema::fixture_op_in(&op, &input_json, &host).map_err(|e| e.to_string())
        })
    })
}

/// `parse_metric_view(yaml, trust_expressions=trust)`: `{"spec"}` or
/// `{"error"}` as JSON text.
#[pyfunction]
pub fn metric_view_parse(py: Python<'_>, yaml: &str, trust: bool) -> PyResult<String> {
    let yaml = yaml.to_string();
    py.detach(move || guarded(move || Ok(sdm_schema::metric_view::parse_json(&yaml, trust))))
}

/// `Schema.from_data`, in the core: the description or refusal, and the
/// warnings, as JSON text.
#[pyfunction]
pub fn schema_from_data(
    py: Python<'_>,
    tables: &Bound<'_, PyAny>,
    column_tags_json: &str,
    options_json: &str,
) -> PyResult<(String, String)> {
    let host = host(py)?;
    let tags: serde_json::Value = serde_json::from_str(column_tags_json)
        .map_err(|e| PyValueError::new_err(format!("the tags are not JSON: {e}")))?;
    let options: serde_json::Value = serde_json::from_str(options_json)
        .map_err(|e| PyValueError::new_err(format!("the options are not JSON: {e}")))?;
    let args = sdm_schema::binding::data_args(&options).map_err(PyValueError::new_err)?;
    let tables = tables
        .cast::<PyDict>()
        .map_err(|_| PyValueError::new_err("the tables are a dict of pyarrow.Table"))?;
    let mut imported = Vec::new();
    for (name, table) in tables.iter() {
        let name: String = name
            .extract()
            .map_err(|_| PyValueError::new_err("a table's name is text"))?;
        let (schema, batches) = crate::import_table(&table)?;
        imported.push((name, schema, batches));
    }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(
        move || -> Result<(String, String), String> {
            let mut frames = Vec::new();
            for (name, schema, batches) in &imported {
                let frame = sdm_schema::frame::from_arrow(batches, schema, tags.get(name))?;
                frames.push((name.clone(), Arc::new(frame)));
            }
            Ok(sdm_schema::binding::schema_from_data(frames, &args, &host))
        },
    ));
    match result {
        Ok(Ok(pair)) => Ok(pair),
        Ok(Err(message)) => Err(PyValueError::new_err(message)),
        Err(_) => Err(crate::panicked("the schema core")),
    }
}

/// `Schema.from_source`, as a run: its `done` value is `(schema_json,
/// warnings_json)`.
#[pyfunction]
pub fn schema_from_source(
    py: Python<'_>,
    backend: &str,
    source_json: &str,
    args_json: &str,
) -> PyResult<crate::PyRun> {
    let host = host(py)?;
    let source: serde_json::Value = serde_json::from_str(source_json)
        .map_err(|e| PyValueError::new_err(format!("the source is not JSON: {e}")))?;
    let args: serde_json::Value = serde_json::from_str(args_json)
        .map_err(|e| PyValueError::new_err(format!("the arguments are not JSON: {e}")))?;
    let (info, args) =
        sdm_schema::binding::source_args(backend, &source, &args).map_err(PyValueError::new_err)?;
    let run = sdm_schema::run::source_run(info, args, host);
    Ok(crate::PyRun::new(Box::new(crate::Mapped::new(
        run,
        |(a, b)| crate::Done::Texts(a, b),
    ))))
}

/// A Snowflake semantic view's YAML, read on its own under the name `name`:
/// JSON text of its tables, keys, columns and relationships, every name as
/// it spells. `ValueError` for a document that is not such a view.
#[pyfunction]
pub fn semantic_view_parse(py: Python<'_>, yaml: &str, name: &str) -> PyResult<String> {
    let (yaml, name) = (yaml.to_string(), name.to_string());
    py.detach(move || {
        guarded(move || {
            sdm_schema::semantic_view::parse(&yaml, &name)
                .map(|v| v.to_string())
                .map_err(|e| format!("{}: {}", e.class, e.message))
        })
    })
}
