// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Requests and answers through Python (`tests/rust_core/request/CONTRACT.md`,
//! "The binding"): the runs that gather, send and read a request, the pure
//! functions a `PredictionResult` is built on, and the request fixture
//! operations. Test-only until Phase 9 switches Python over to the core.
//!
//! Request arguments and values are plain data. Public result merging can
//! supply call-local Python protocol callbacks for held host values.
//! Result callbacks run synchronously with the GIL held and cannot escape the call.
//! Materialization sequence callbacks belong to the run and acquire the GIL
//! only when the core reaches the corresponding Python protocol operation.
//! A refusal is raised as a
//! `CoreError` whose attributes name the class a host raises it as
//! (`error_class`), its `code`, `message`, `details` and the `warnings`
//! given before it.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use sdm_core::plain::Plain;

use crate::{caught, core_error_after};

/// `payload_size_bytes(json.loads(payload_json))` of
/// `sdm_relational_engine.rfm.inference.payload`: the bytes Python's compact
/// `json.dumps` writes, however the text spelled it. `ValueError` for text
/// that is not a JSON object, or that holds `NaN` or an infinity.
#[pyfunction]
pub(crate) fn payload_size_bytes(py: Python<'_>, payload_json: &str) -> PyResult<usize> {
    let text = payload_json.to_string();
    py.detach(move || {
        caught(|| {
            sdm_core::request::size::payload_size_bytes(&text)
                .map_err(|e| PyValueError::new_err(e.0))
        })
    })
}

/// The JSON text `sdm_core::fixture_op` gives for a request fixture case.
#[pyfunction]
pub(crate) fn request_fixture_op(
    py: Python<'_>,
    operation: &str,
    input_json: &str,
) -> PyResult<String> {
    let (operation, input) = (operation.to_string(), input_json.to_string());
    py.detach(move || {
        caught(|| sdm_core::fixture_op(&operation, &input).map_err(|e| PyValueError::new_err(e.0)))
    })
}

/// The value `function` of the host API gives for `args`, raised as a core
/// error where it refuses.
fn host_value(py: Python<'_>, function: &str, args: Vec<Plain>) -> PyResult<Py<PyAny>> {
    let host = host(py)?;
    let called = caught(|| Ok(sdm_core::host_api::call(function, &args, &host)))?;
    match called {
        Ok(sdm_core::host_api::Called::Value(value)) => {
            Ok(crate::plain::to_py(py, &value)?.unbind())
        }
        Ok(sdm_core::host_api::Called::Run(_)) => Err(PyValueError::new_err(format!(
            "{function} gave a run where a value was expected"
        ))),
        Err(refused) => Err(core_error_after(py, &refused.error, &refused.warnings)),
    }
}

/// The layout `PredictionResult.to_pandas()` is built from, of the result
/// whose plain JSON is `result_json`: a `pyarrow.Table` of the frame's
/// columns, each field's metadata saying how to build the frame back.
#[pyfunction]
#[pyo3(signature = (result_json, infer_column=None, column_order=None, protocol=None))]
pub(crate) fn flatten(
    py: Python<'_>,
    result_json: &str,
    infer_column: Option<Bound<'_, PyAny>>,
    column_order: Option<Bound<'_, PyAny>>,
    protocol: Option<Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    use sdm_core::request::layout;

    if infer_column.is_none() && column_order.is_none() && protocol.is_none() {
        return host_value(py, "flatten", vec![Plain::Str(result_json.to_string())]);
    }
    caught(|| {
        let infer =
            infer_column.ok_or_else(|| PyValueError::new_err("missing host column inference"))?;
        let order =
            column_order.ok_or_else(|| PyValueError::new_err("missing host column ordering"))?;
        let value = serde_json::from_str(result_json)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let result = layout::decode_result(&value).map_err(PyValueError::new_err)?;
        let layout = layout::host_layout_with_protocol(
            &result,
            |cells| {
                let encoded: Vec<_> = cells.iter().map(layout::encode_deep).collect();
                infer
                    .call1((serde_json::Value::Array(encoded).to_string(),))?
                    .extract()
            },
            |reference, stable, ascending, missing_last| {
                order
                    .call1((reference, stable, ascending, missing_last))?
                    .extract()
            },
            |operation, operands| result_protocol(&protocol, operation, operands),
            |message| PyValueError::new_err(message.to_string()),
        )?;
        Ok(crate::plain::to_py(py, &Plain::Str(layout.to_string()))?.unbind())
    })
}

/// `PredictionResult.merge(parts)` of the results whose plain JSON are
/// `parts_json`: the merged result's plain JSON, or today's refusal.
#[pyfunction]
#[pyo3(signature = (parts_json, equal=None, truth=None, metadata_entry=None, protocol=None))]
pub(crate) fn merge(
    py: Python<'_>,
    parts_json: Vec<String>,
    equal: Option<Bound<'_, PyAny>>,
    truth: Option<Bound<'_, PyAny>>,
    metadata_entry: Option<Bound<'_, PyAny>>,
    protocol: Option<Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    use sdm_core::request::{layout, result};
    use sdm_schema::pyval::PyVal;

    fn reference(value: &PyVal) -> Option<&str> {
        match value {
            PyVal::Opaque {
                kind, type_name, ..
            } if kind == "host_ref" => Some(type_name),
            _ => None,
        }
    }

    if equal.is_none() && truth.is_none() && metadata_entry.is_none() && protocol.is_none() {
        return host_value(
            py,
            "merge",
            vec![Plain::List(
                parts_json.into_iter().map(Plain::Str).collect(),
            )],
        );
    }
    caught(|| {
        let parts = parts_json
            .iter()
            .map(|text| {
                let value = serde_json::from_str(text)
                    .map_err(|error| PyValueError::new_err(error.to_string()))?;
                layout::decode_result(&value).map_err(PyValueError::new_err)
            })
            .collect::<PyResult<Vec<_>>>()?;
        let merged = result::Result_::merge_with_protocol(
            parts,
            |left, right| match (reference(left), reference(right), &equal) {
                (Some(left), Some(right), Some(callback)) => {
                    callback.call1((left, right))?.extract()
                }
                _ => result::merge_equal(left, right)
                    .map_err(|error| core_error_after(py, &error, &[])),
            },
            |value| match (reference(value), &truth) {
                (Some(value), Some(callback)) => callback.call1((value,))?.extract(),
                _ => result::merge_truth(value).map_err(|error| core_error_after(py, &error, &[])),
            },
            |mapping, key, value| match (reference(mapping), &metadata_entry) {
                (Some(mapping), Some(callback)) => {
                    let encoded: String = callback
                        .call1((mapping, key, layout::encode_deep(&value).to_string()))?
                        .extract()?;
                    let value = serde_json::from_str(&encoded)
                        .map_err(|error| PyValueError::new_err(error.to_string()))?;
                    PyVal::decode(&value).map_err(PyValueError::new_err)
                }
                _ => result::with_metadata_entry(mapping, key, value)
                    .map_err(|error| core_error_after(py, &error, &[])),
            },
            |operation, operands| result_protocol(&protocol, operation, operands),
            |error| core_error_after(py, &error, &[]),
        )?;
        Ok(
            crate::plain::to_py(py, &Plain::Str(layout::encode_result(&merged).to_string()))?
                .unbind(),
        )
    })
}

/// `result.absent(requested)`: which of `requested` the result whose plain
/// JSON is `result_json` does not carry.
#[pyfunction]
#[pyo3(signature = (result_json, requested, protocol=None))]
pub(crate) fn absent(
    py: Python<'_>,
    result_json: &str,
    requested: Vec<String>,
    protocol: Option<Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let value: serde_json::Value = serde_json::from_str(result_json)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    if protocol.is_some() || value.get("requested").is_some() {
        return caught(|| {
            use sdm_core::request::{layout, result};
            use sdm_schema::pyval::PyVal;
            let result = layout::decode_result(&value).map_err(PyValueError::new_err)?;
            let requested =
                PyVal::decode(value.get("requested").unwrap_or(&serde_json::Value::Null))
                    .map_err(PyValueError::new_err)?;
            let owner = PyVal::decode(value.get("owner").unwrap_or(&serde_json::Value::Null))
                .map_err(PyValueError::new_err)?;
            let absent = result::absent_with(
                &result,
                &owner,
                &requested,
                |operation, operands| result_protocol(&protocol, operation, operands),
                |message| PyValueError::new_err(message.to_string()),
            )?;
            Ok(crate::plain::to_py(
                py,
                &Plain::Str(layout::encode_deep(&PyVal::List(absent)).to_string()),
            )?
            .unbind())
        });
    }
    host_value(
        py,
        "absent",
        vec![
            Plain::Str(result_json.to_string()),
            Plain::List(requested.into_iter().map(Plain::Str).collect()),
        ],
    )
}

fn result_protocol(
    callback: &Option<Bound<'_, PyAny>>,
    operation: &str,
    operands: &[sdm_schema::pyval::PyVal],
) -> PyResult<sdm_schema::pyval::PyVal> {
    use sdm_core::request::layout;
    let callback = callback
        .as_ref()
        .ok_or_else(|| PyValueError::new_err("missing result host protocol"))?;
    let encoded = serde_json::Value::Array(operands.iter().map(layout::encode_deep).collect());
    let returned: String = callback
        .call1((operation, encoded.to_string()))?
        .extract()?;
    let value = serde_json::from_str(&returned)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    sdm_schema::pyval::PyVal::decode(&value).map_err(PyValueError::new_err)
}

/// The Python process's host, as the schema core reads dates.
fn host(py: Python<'_>) -> PyResult<sdm_core::host_api::HostInfo> {
    crate::schema::host(py)
}

/// `function` of the host API on `args`: a value, or a run.
fn host_call(
    py: Python<'_>,
    function: &str,
    args: &Bound<'_, pyo3::types::PyTuple>,
) -> PyResult<Py<PyAny>> {
    let protocol_index = if function == "request_op" && args.len() == 4 {
        Some(3)
    } else if matches!(
        function,
        "predict_run" | "fit_run" | "session_predict_run" | "request_materialized"
    ) && args.len() >= 3
    {
        Some(2)
    } else {
        None
    };
    let supplied_protocol = protocol_index
        .map(|index| args.get_item(index).map(Bound::unbind))
        .transpose()?;
    let holdout = if function == "fit_run" && args.len() >= 3 {
        let callback = if args.len() == 4 {
            Some(args.get_item(3)?)
        } else {
            let protocol = args.get_item(2)?;
            if protocol.hasattr("sequence")? {
                Some(protocol.getattr("sequence")?)
            } else {
                None
            }
        };
        callback
            .filter(|callback| !callback.is_none())
            .map(|callback| sequence_protocol(callback.unbind()))
    } else {
        None
    };
    let plain = args
        .iter()
        .take(protocol_index.unwrap_or(args.len()))
        .map(|a| crate::plain::from_py(&a))
        .collect::<PyResult<Vec<_>>>()?;
    let host = host(py)?;
    let protocols = sdm_core::request::pyops::HostProtocols::new(move |operation, operands| {
        Python::attach(|py| {
            let answer = (|| -> PyResult<String> {
                let callback = match &supplied_protocol {
                    Some(callback) => callback.bind(py).clone(),
                    None => py.import("sdm_core.answers")?.getattr("_host_protocol")?,
                };
                callback.call1((operation, operands.to_string()))?.extract()
            })();
            match answer {
                Ok(text) => serde_json::from_str(&text)
                    .map_err(|error| sdm_core::request::error::value_error(error.to_string())),
                Err(error) => {
                    let class = error
                        .get_type(py)
                        .name()
                        .map(|name| name.to_string())
                        .unwrap_or_else(|_| "RuntimeError".into());
                    let message = error
                        .value(py)
                        .str()
                        .map(|value| value.to_string())
                        .unwrap_or_else(|_| error.to_string());
                    Err(sdm_core::request::error::builtin(&class, message))
                }
            }
        })
    });
    let called = caught(|| {
        if function == "fit_run" {
            return Ok(sdm_core::host_api::fit_run_with_protocols(
                &plain,
                &host,
                Some(protocols),
                holdout,
            )
            .map_err(Into::into));
        }
        Ok(sdm_core::host_api::call_with_protocols(
            function,
            &plain,
            &host,
            Some(protocols),
        ))
    })?;
    match called {
        Ok(sdm_core::host_api::Called::Value(value)) => {
            Ok(crate::plain::to_py(py, &value)?.unbind())
        }
        Ok(sdm_core::host_api::Called::Run(run)) => {
            let mapped = crate::Mapped::new(run, |value| crate::Done::Plain(Box::new(value)));
            Ok(Py::new(py, crate::PyRun::new(Box::new(mapped)))?.into_any())
        }
        Err(refused) => Err(core_error_after(py, &refused.error, &refused.warnings)),
    }
}

/// `schema_source_run(backend, source_json, args_json)`:
/// `Schema.from_source`, as a run of the core; its `done` value is the
/// schema handle's JSON, or a refusal, and the warnings' JSON.
#[pyfunction]
#[pyo3(signature = (*args))]
pub(crate) fn schema_source_run(
    py: Python<'_>,
    args: &Bound<'_, pyo3::types::PyTuple>,
) -> PyResult<Py<PyAny>> {
    host_call(py, "schema_source_run", args)
}

/// `data_schema(tables, column_tags_json, options_json)`:
/// `Schema.from_data`, as the request core holds it; `(handle_json,
/// warnings_json)`, the handle a refusal where it was refused.
#[pyfunction]
#[pyo3(signature = (*args))]
pub(crate) fn data_schema(
    py: Python<'_>,
    args: &Bound<'_, pyo3::types::PyTuple>,
) -> PyResult<Py<PyAny>> {
    host_call(py, "data_schema", args)
}

/// `materialize_run(state_json, tables)`: `PredictiveQuery.materialize`, as
/// a run of the core; its `done` value is `(request_json, tables)`.
#[pyfunction]
#[pyo3(signature = (*args))]
pub(crate) fn materialize_run(
    py: Python<'_>,
    args: &Bound<'_, pyo3::types::PyTuple>,
) -> PyResult<Py<PyAny>> {
    sequence_run(
        py,
        "materialize_run",
        args,
        sdm_core::host_api::materialize_run_with_sequences,
    )
}

type SequenceRun = fn(
    &[Plain],
    &sdm_core::host_api::HostInfo,
    Option<sdm_core::request::materialize::SequenceProtocol>,
    Option<sdm_core::request::materialize::SequenceProtocol>,
) -> Result<sdm_core::host_api::Called, sdm_core::Diagnostic>;

fn sequence_protocol(callback: Py<PyAny>) -> sdm_core::request::materialize::SequenceProtocol {
    sdm_core::request::materialize::SequenceProtocol(std::sync::Arc::new(move |operation| {
        Python::attach(|py| {
            let result = callback
                .call1(py, (operation,))
                .and_then(|value| value.extract::<String>(py));
            let encoded = result.map_err(|_| {
                sdm_core::request::error::value_error("host sequence operation failed")
            })?;
            let value = serde_json::from_str(&encoded)
                .map_err(|error| sdm_core::request::error::value_error(error.to_string()))?;
            sdm_schema::pyval::PyVal::decode(&value).map_err(sdm_core::request::error::value_error)
        })
    }))
}

fn sequence_run(
    py: Python<'_>,
    function: &str,
    args: &Bound<'_, pyo3::types::PyTuple>,
    factory: SequenceRun,
) -> PyResult<Py<PyAny>> {
    if args.len() == 2 {
        return host_call(py, function, args);
    }
    let protocol = args
        .get_item(2)
        .ok()
        .filter(|callback| !callback.is_none())
        .map(|callback| sequence_protocol(callback.unbind()));
    let candidates = args
        .get_item(3)
        .ok()
        .filter(|callback| !callback.is_none())
        .map(|callback| sequence_protocol(callback.unbind()));
    let plain = args
        .iter()
        .take(2)
        .map(|argument| crate::plain::from_py(&argument))
        .collect::<PyResult<Vec<_>>>()?;
    let host = host(py)?;
    let called = caught(|| Ok(factory(&plain, &host, protocol, candidates)))?;
    match called {
        Ok(sdm_core::host_api::Called::Run(run)) => {
            let mapped = crate::Mapped::new(run, |value| crate::Done::Plain(Box::new(value)));
            Ok(Py::new(py, crate::PyRun::new(Box::new(mapped)))?.into_any())
        }
        Ok(sdm_core::host_api::Called::Value(_)) => Err(PyValueError::new_err(
            "sequence operation did not return a run",
        )),
        Err(refused) => Err(core_error_after(py, &refused, &[])),
    }
}

/// `predict_run(state_json, tables)`: `SDMClient.predict`, as a run of the
/// core; its `done` value is the result's plain JSON.
#[pyfunction]
#[pyo3(signature = (*args))]
pub(crate) fn predict_run(
    py: Python<'_>,
    args: &Bound<'_, pyo3::types::PyTuple>,
) -> PyResult<Py<PyAny>> {
    host_call(py, "predict_run", args)
}

/// `fit_run(state_json, tables)`: `SDMClient.fit`, as a run of the core;
/// its `done` value is the fitted model's plain JSON.
#[pyfunction]
#[pyo3(signature = (*args))]
pub(crate) fn fit_run(
    py: Python<'_>,
    args: &Bound<'_, pyo3::types::PyTuple>,
) -> PyResult<Py<PyAny>> {
    host_call(py, "fit_run", args)
}

/// `session_predict_run(state_json, tables)`: `FittedModel.predict`, as a
/// run of the core; its `done` value is the result's plain JSON.
#[pyfunction]
#[pyo3(signature = (*args))]
pub(crate) fn session_predict_run(
    py: Python<'_>,
    args: &Bound<'_, pyo3::types::PyTuple>,
) -> PyResult<Py<PyAny>> {
    host_call(py, "session_predict_run", args)
}

/// `request_build(state_json, tables)`: `Request(...)` built from frames;
/// `(answer_json, tables)`.
#[pyfunction]
#[pyo3(signature = (*args))]
pub(crate) fn request_build(
    py: Python<'_>,
    args: &Bound<'_, pyo3::types::PyTuple>,
) -> PyResult<Py<PyAny>> {
    host_call(py, "request_build", args)
}

/// `request_materialized(state_json, tables)`: `Request.materialized(...)`,
/// a request of the envelopes a caller holds; `(answer_json, tables)`.
#[pyfunction]
#[pyo3(signature = (*args))]
pub(crate) fn request_materialized(
    py: Python<'_>,
    args: &Bound<'_, pyo3::types::PyTuple>,
) -> PyResult<Py<PyAny>> {
    host_call(py, "request_materialized", args)
}

/// `query_text(query)`: the text a `PredictiveQuery` asks, or today's
/// refusal of what is not one.
#[pyfunction]
#[pyo3(signature = (*args))]
pub(crate) fn query_text(
    py: Python<'_>,
    args: &Bound<'_, pyo3::types::PyTuple>,
) -> PyResult<Py<PyAny>> {
    host_call(py, "query_text", args)
}

/// `schema_edit(state_json, tables)`: an edit a host makes on its own
/// objects of a schema; `(schema_json, warnings_json)`.
#[pyfunction]
#[pyo3(signature = (*args))]
pub(crate) fn schema_edit(
    py: Python<'_>,
    args: &Bound<'_, pyo3::types::PyTuple>,
) -> PyResult<Py<PyAny>> {
    host_call(py, "schema_edit", args)
}

/// `model_spec(model_json)`: which registered model a model id or handle
/// names, and what its profile allows; today's refusal otherwise.
#[pyfunction]
#[pyo3(signature = (*args))]
pub(crate) fn model_spec(
    py: Python<'_>,
    args: &Bound<'_, pyo3::types::PyTuple>,
) -> PyResult<Py<PyAny>> {
    host_call(py, "model_spec", args)
}

/// `request_op(op, state_json, tables)`: what a request's wrapper asks of
/// the core about a request it holds.
#[pyfunction]
#[pyo3(signature = (*args))]
pub(crate) fn request_op(
    py: Python<'_>,
    args: &Bound<'_, pyo3::types::PyTuple>,
) -> PyResult<Py<PyAny>> {
    host_call(py, "request_op", args)
}

/// `gather_graph_run(state_json, tables)`: the graph a gather of a schema
/// over a source samples, as a run.
#[pyfunction]
#[pyo3(signature = (*args))]
pub(crate) fn gather_graph_run(
    py: Python<'_>,
    args: &Bound<'_, pyo3::types::PyTuple>,
) -> PyResult<Py<PyAny>> {
    host_call(py, "gather_graph_run", args)
}
