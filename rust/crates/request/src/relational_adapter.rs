// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `sdm_client/adapters/relational.py`: sending gathered envelopes to
//! `kumo-relational`, the checks made before sending, and reading the
//! answer back onto the entities asked about.

use std::collections::HashSet;

use crate::pyops::TaggedValue;
use sdm_protocol::Diagnostic;
use sdm_schema::pyval::PyVal;
use serde_json::{Map, Value, json};

use crate::error::{NIM_REQUEST, R, Warnings, details, structured, unsupported};
use crate::models::{
    DEFAULT_PREDICTION_STATISTIC, KUMO_RELATIONAL, RELATIONAL, SUMMARISED_TASK_KINDS,
    relational_limits,
};
use crate::request::Request;
use crate::result::Item;

/// Which halves a call puts on the wire.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sends {
    Everything,
    Context,
    RowsToScore,
}

impl Sends {
    fn context(self) -> bool {
        matches!(self, Sends::Everything | Sends::Context)
    }

    fn predict(self) -> bool {
        matches!(self, Sends::Everything | Sends::RowsToScore)
    }
}

const WIDEN_THE_CONTEXT: &str = "Gather a wider context with num_context_samples, or ask about \
     entities whose answers differ";

/// `_payloads(request)`.
fn payloads(request: &Request) -> R<Vec<Value>> {
    request.payloads().ok_or_else(|| {
        unsupported(
            "a relational request has to be gathered first, and this one was not. Build it with \
             PredictiveQuery(...).materialize(indices=[...], model='kumo-relational'). A schema \
             of one table is gathered for the tabular model unless the relational one is named, \
             rather than with Request(context=..., predict=...), which is the single-table form",
            details([("model", json!(KUMO_RELATIONAL))]),
        )
    })
}

/// `_output_for(payload, outputs)`.
fn output_for(
    payload: &Value,
    outputs: &[String],
    protocols: Option<&crate::pyops::HostProtocols>,
) -> R<Value> {
    let mut output = Map::new();
    output.insert("fields".into(), json!(outputs));
    let kind = kind_of(payload, protocols)?;
    if SUMMARISED_TASK_KINDS.contains(&kind.as_str()) {
        let output_section = crate::pyops::mapping_get(payload, "output", &Value::Null, protocols)?;
        let output_section = if crate::pyops::truth_with(
            &PyVal::decode(&output_section).map_err(crate::error::value_error)?,
            protocols,
        )? {
            output_section
        } else {
            json!({})
        };
        let candidate = crate::pyops::mapping_get(
            &output_section,
            "prediction_statistic",
            &Value::Null,
            protocols,
        )?;
        let candidate = Some(&candidate);
        let truthy = candidate
            .map(|value| {
                let decoded = PyVal::decode(value).map_err(crate::error::value_error)?;
                crate::pyops::truth_with(&decoded, protocols)
            })
            .transpose()?
            .unwrap_or(false);
        let chosen = candidate
            .filter(|_| truthy)
            .cloned()
            .unwrap_or_else(|| json!(DEFAULT_PREDICTION_STATISTIC));
        output.insert("prediction_statistic".into(), chosen);
    }
    Ok(Value::Object(output))
}

/// `_keys_the_caller_set(requested)`: the inference-config keys asked for.
fn keys_the_caller_set(
    requested: Option<&Value>,
    protocols: Option<&crate::pyops::HostProtocols>,
) -> R<Vec<PyVal>> {
    match requested {
        Some(map) => {
            if let Some(Value::Object(dataclass)) = map.py_get("$inference_config") {
                return Ok(crate::inference::set_fields(dataclass)
                    .into_iter()
                    .map(PyVal::Str)
                    .collect());
            }
            let decoded = PyVal::decode(map).map_err(crate::error::value_error)?;
            if let PyVal::Dict(pairs) = decoded.semantic() {
                return Ok(pairs.iter().map(|(key, _)| key.clone()).collect());
            }
            if matches!(decoded, PyVal::Held { .. }) {
                let fields = protocols
                    .ok_or_else(|| {
                        crate::error::value_error("dataclass fields require host protocol facts")
                    })?
                    .call("dataclass_nondefault_fields", map)?;
                if fields.is_null() {
                    return Ok(Vec::new());
                }
                return fields
                    .as_array()
                    .ok_or_else(|| crate::error::value_error("dataclass fields must be a list"))?
                    .iter()
                    .map(|field| {
                        field
                            .as_str()
                            .map(|name| PyVal::Str(name.to_string()))
                            .ok_or_else(|| {
                                crate::error::value_error("dataclass field names must be strings")
                            })
                    })
                    .collect();
            }
            Ok(Vec::new())
        }
        _ => Ok(Vec::new()),
    }
}

fn sorted_set_operation(
    protocols: Option<&crate::pyops::HostProtocols>,
    operation: &str,
    left: Vec<PyVal>,
    right: Vec<PyVal>,
) -> R<Vec<PyVal>> {
    if left
        .iter()
        .chain(&right)
        .all(|key| matches!(key, PyVal::Str(_)))
    {
        let mut keys: Vec<PyVal> = left
            .into_iter()
            .filter(|key| {
                let contained = right.iter().any(|other| crate::result::py_eq(key, other));
                if operation == "sorted_set_difference" {
                    !contained
                } else {
                    contained
                }
            })
            .collect();
        keys.sort_by(|left, right| left.as_str().cmp(&right.as_str()));
        keys.dedup();
        return Ok(keys);
    }
    let protocols = protocols.ok_or_else(|| {
        crate::error::value_error("non-string set ordering requires host protocol facts")
    })?;
    let operands =
        crate::layout::encode_deep(&PyVal::List(vec![PyVal::List(left), PyVal::List(right)]));
    let answer = protocols.call(operation, &operands)?;
    match PyVal::decode(&answer).map_err(crate::error::value_error)? {
        PyVal::List(keys) => Ok(keys),
        _ => Err(crate::error::value_error(
            "a sorted-set host fact must be a list",
        )),
    }
}

/// `_inference(inference, request)`: only the options this model reads.
fn inference(given: Option<&Value>, request: &Request) -> R<Option<Value>> {
    let Some(inference) = given else {
        return Ok(None);
    };
    if !crate::pyops::is_instance(
        inference,
        crate::pyops::InstanceFamily::Dict,
        request.protocols.as_ref(),
    )? {
        return Ok(None);
    }
    let config = crate::pyops::mapping_get(
        inference,
        "inference_config",
        &Value::Null,
        request.protocols.as_ref(),
    )?;
    let decoded_config = PyVal::decode(&config).map_err(crate::error::value_error)?;
    let PyVal::Dict(config) = decoded_config.semantic() else {
        return Ok(Some(crate::pyops::mapping_copy(
            inference,
            request.protocols.as_ref(),
        )?));
    };
    let forwarded = RELATIONAL.inference_options;
    let unsupported_keys = sorted_set_operation(
        request.protocols.as_ref(),
        "sorted_set_difference",
        config.iter().map(|(key, _)| key.clone()).collect(),
        forwarded
            .iter()
            .map(|key| PyVal::Str((*key).into()))
            .collect(),
    )?;
    let asked_for = keys_the_caller_set(
        request.requested_inference.py_get("inference_config"),
        request.protocols.as_ref(),
    )?;
    let refused = sorted_set_operation(
        request.protocols.as_ref(),
        "sorted_set_intersection",
        unsupported_keys,
        asked_for,
    )?;
    if !refused.is_empty() {
        let mut forwarded_sorted: Vec<String> = forwarded.iter().map(|s| s.to_string()).collect();
        forwarded_sorted.sort();
        let statistic = if refused
            .iter()
            .any(|key| key.as_str() == Some("output_type"))
        {
            format!(
                ". A regression's statistic is chosen with materialize(..., \
                 prediction_statistic=...) instead: '{DEFAULT_PREDICTION_STATISTIC}', the \
                 default, or 'mean'"
            )
        } else {
            String::new()
        };
        return Err(unsupported(
            format!(
                "{KUMO_RELATIONAL} does not read inference option(s) {}; drop them from \
                 inference_config to send the request without them. The keys this client \
                 forwards are {}, and the endpoint decides for itself whether it serves the \
                 value given for one{statistic}",
                PyVal::List(refused.clone()).repr(),
                sdm_schema::py::str_list_repr(&forwarded_sorted)
            ),
            details([
                ("model", json!(KUMO_RELATIONAL)),
                (
                    "unsupported",
                    crate::layout::encode_deep(&PyVal::List(refused)),
                ),
            ]),
        ));
    }
    let mut trimmed = crate::pyops::mapping_copy(inference, request.protocols.as_ref())?;
    crate::pyops::mapping_update(
        &mut trimmed,
        "inference_config",
        Some(crate::layout::encode_deep(&PyVal::Dict(
            config
                .iter()
                .filter(|(key, _)| key.as_str().is_some_and(|key| forwarded.contains(&key)))
                .cloned()
                .collect(),
        ))),
        request.protocols.as_ref(),
    )?;
    Ok(Some(trimmed))
}

/// `prediction_plan(request, outputs, sends)`.
pub fn prediction_plan(
    request: &Request,
    outputs: &[String],
    sends: Sends,
    said: &mut Warnings,
) -> R<Vec<Value>> {
    let payloads = payloads(request)?;
    preflight(
        &payloads,
        outputs,
        request.query.as_ref().and_then(PyVal::as_str),
        sends,
        said,
        request.protocols.as_ref(),
    )?;
    let mut out = Vec::new();
    for payload in &payloads {
        let mut envelope = crate::pyops::mapping_copy(payload, request.protocols.as_ref())?;
        crate::pyops::mapping_update(
            &mut envelope,
            "model",
            Some(json!(KUMO_RELATIONAL)),
            request.protocols.as_ref(),
        )?;
        crate::pyops::mapping_update(
            &mut envelope,
            "output",
            Some(output_for(payload, outputs, request.protocols.as_ref())?),
            request.protocols.as_ref(),
        )?;
        let mut metadata = crate::pyops::mapping_get(
            payload,
            "metadata",
            &Value::Null,
            request.protocols.as_ref(),
        )?;
        let decoded = PyVal::decode(&metadata).map_err(crate::error::value_error)?;
        if !crate::pyops::truth_with(&decoded, request.protocols.as_ref())? {
            metadata = json!({});
        } else if !matches!(decoded.semantic(), PyVal::Dict(_)) {
            return Err(crate::error::builtin(
                "TypeError",
                format!(
                    "'{}' object is not a mapping",
                    crate::models::type_name(&decoded)
                ),
            ));
        }
        if matches!(decoded, PyVal::Held { .. }) && matches!(decoded.semantic(), PyVal::Dict(_)) {
            let protocols = request.protocols.as_ref().ok_or_else(|| {
                crate::error::value_error("metadata assignment requires host protocol facts")
            })?;
            let copied = protocols.call("mapping_copy", &metadata)?;
            metadata = protocols.call(
                "mapping_set",
                &json!([copied, "request_id", crate::tabular_adapter::request_id()]),
            )?;
        } else {
            crate::pyops::set_field(
                &mut metadata,
                "request_id",
                Some(json!(crate::tabular_adapter::request_id())),
            );
        }
        crate::pyops::mapping_update(
            &mut envelope,
            "metadata",
            Some(metadata),
            request.protocols.as_ref(),
        )?;
        let given = crate::pyops::mapping_get(
            &envelope,
            "inference",
            &Value::Null,
            request.protocols.as_ref(),
        )?;
        let inference = inference(Some(&given), request)?;
        crate::pyops::mapping_update(
            &mut envelope,
            "inference",
            inference,
            request.protocols.as_ref(),
        )?;
        out.push(envelope);
    }
    Ok(out)
}

/// `session_plan(request, outputs)`: the envelope that pins the context.
pub fn session_plan(request: &Request, outputs: &[String], said: &mut Warnings) -> R<Value> {
    let gathered = payloads(request)?;
    if gathered.len() > 1 {
        return Err(unsupported(
            format!(
                "this request gathered {} batches, and a session pins one context; use \
                 predict() instead",
                gathered.len()
            ),
            details([("batches", json!(gathered.len()))]),
        ));
    }
    let plan = prediction_plan(request, outputs, Sends::Context, said)?;
    let envelope = plan.into_iter().next().unwrap_or(Value::Null);
    Ok(crate::tabular_adapter::pinned_half(&envelope))
}

/// `session_prediction_plan(request, outputs, pinned)`.
pub fn session_prediction_plan(
    request: &Request,
    outputs: &[String],
    said: &mut Warnings,
) -> R<Vec<Value>> {
    Ok(prediction_plan(request, outputs, Sends::RowsToScore, said)?
        .iter()
        .map(crate::tabular_adapter::scoring_half)
        .collect())
}

fn context_table(
    payload: &Value,
    protocols: Option<&crate::pyops::HostProtocols>,
) -> R<(Vec<String>, Vec<Vec<Value>>)> {
    let context = crate::pyops::mapping_get(payload, "context", &json!({}), protocols)?;
    let table = context.py_get("instance_table");
    let columns = table
        .and_then(|t| t.py_get("columns"))
        .and_then(crate::pyops::sequence)
        .map(|c| {
            c.iter()
                .map(|v| v.as_str().unwrap_or_default().to_string())
                .collect()
        })
        .unwrap_or_default();
    let rows = table
        .and_then(|t| t.py_get("rows"))
        .and_then(crate::pyops::sequence)
        .map(|r| {
            r.iter()
                .map(|row| crate::pyops::sequence(row).cloned().unwrap_or_default())
                .collect()
        })
        .unwrap_or_default();
    Ok((columns, rows))
}

/// The name of the Python type `json.loads` reads `value` as.
fn json_type_name(value: &Value) -> String {
    crate::models::type_name(&PyVal::decode(value).unwrap_or(PyVal::None))
}

/// A JSON cell as a key a Python set holds.
fn cell_key(value: &Value, protocols: Option<&crate::pyops::HostProtocols>) -> R<String> {
    let value = PyVal::decode(value).map_err(crate::error::value_error)?;
    crate::pyops::hashable_with(&value, protocols)?;
    Ok(value.eq_key())
}

fn distinct_count(
    values: Vec<&Value>,
    protocols: Option<&crate::pyops::HostProtocols>,
) -> R<usize> {
    let decoded = values
        .iter()
        .map(|value| PyVal::decode(value).map_err(crate::error::value_error))
        .collect::<R<Vec<_>>>()?;
    if decoded.iter().any(crate::pyops::has_host_value) {
        let protocols = protocols.ok_or_else(|| {
            crate::error::value_error("set cardinality requires host protocol facts")
        })?;
        let answer = protocols.call(
            "set_cardinality",
            &crate::layout::encode_deep(&PyVal::List(decoded)),
        )?;
        return answer
            .as_u64()
            .and_then(|count| usize::try_from(count).ok())
            .filter(|count| *count <= values.len())
            .ok_or_else(|| {
                crate::error::value_error(
                    "a set cardinality host fact must be within the supplied length",
                )
            });
    }
    Ok(values
        .into_iter()
        .map(|value| cell_key(value, protocols))
        .collect::<R<HashSet<_>>>()?
        .len())
}

fn target_column(
    payload: &Value,
    protocols: Option<&crate::pyops::HostProtocols>,
) -> R<Option<String>> {
    let task = crate::pyops::mapping_get(payload, "task", &json!({}), protocols)?;
    let target = crate::pyops::mapping_get(&task, "target", &json!({}), protocols)?;
    let column = crate::pyops::mapping_get(&target, "column_name", &Value::Null, protocols)?;
    Ok(crate::pyops::string(&column).map(str::to_string))
}

fn kind_of(payload: &Value, protocols: Option<&crate::pyops::HostProtocols>) -> R<String> {
    let task = crate::pyops::mapping_get(payload, "task", &json!({}), protocols)?;
    let kind = crate::pyops::mapping_get(&task, "kind", &json!(""), protocols)?;
    Ok(crate::pyops::string(&kind).unwrap_or("").to_string())
}

/// `_preflight(payloads, outputs, asked, sends)`.
fn preflight(
    payloads: &[Value],
    outputs: &[String],
    asked: Option<&str>,
    sends: Sends,
    said: &mut Warnings,
    protocols: Option<&crate::pyops::HostProtocols>,
) -> R<()> {
    if sends.context() {
        for payload in payloads {
            // `payload.get('context', {}).get(...)`: a context that is there
            // and not a mapping has no `get`.
            let context = crate::pyops::mapping_get(payload, "context", &json!({}), protocols)?;
            if !matches!(PyVal::decode(&context), Ok(PyVal::Dict(_))) {
                return Err(crate::error::builtin(
                    "AttributeError",
                    format!(
                        "'{}' object has no attribute 'get'",
                        json_type_name(&context)
                    ),
                ));
            }
            let table = context.py_get("instance_table");
            let empty =
                match table.filter(|table| matches!(PyVal::decode(table), Ok(PyVal::Dict(_)))) {
                    Some(map) => match map.py_get("rows") {
                        Some(rows) => !crate::pyops::truth_with(
                            &PyVal::decode(rows).map_err(crate::error::value_error)?,
                            protocols,
                        )?,
                        None => true,
                    },
                    None => false,
                };
            if empty {
                return Err(unsupported(
                    "no context rows are left: every row that could be context is also being \
                     scored, and a row carrying the answer being asked for is not sent as \
                     context. Score fewer rows, so that the rest are there to learn from",
                    details([
                        ("model", json!(KUMO_RELATIONAL)),
                        ("context_rows", json!(0)),
                    ]),
                ));
            }
        }
        for payload in payloads {
            if !kind_of(payload, protocols)?.contains("classification") {
                continue;
            }
            let Some(column) = target_column(payload, protocols)? else {
                continue;
            };
            let (columns, rows) = context_table(payload, protocols)?;
            if columns.is_empty() || rows.is_empty() {
                continue;
            }
            let Some(at) = columns.iter().position(|c| *c == column) else {
                continue;
            };
            let labels = distinct_count(
                rows.iter()
                    .filter(|r| r.len() > at)
                    .map(|r| &r[at])
                    .collect(),
                protocols,
            )?;
            if labels as i64 > relational_limits::CLASSES_MAX {
                return Err(unsupported(
                    format!(
                        "{KUMO_RELATIONAL} classifies at most {} classes, and the gathered \
                         context carries {} distinct values of the target; predict a coarser \
                         target, or narrow the context to the classes you care about",
                        relational_limits::CLASSES_MAX,
                        labels
                    ),
                    details([
                        ("model", json!(KUMO_RELATIONAL)),
                        ("classes", json!(labels)),
                        ("limit", json!(relational_limits::CLASSES_MAX)),
                    ]),
                ));
            }
        }
        for payload in payloads {
            let kind = kind_of(payload, protocols)?;
            if !kind.contains("classification") && kind != "regression" {
                continue;
            }
            let column = target_column(payload, protocols)?;
            let (columns, rows) = context_table(payload, protocols)?;
            let Some(name) = &column else {
                continue;
            };
            if columns.is_empty() || rows.is_empty() {
                continue;
            }
            let Some(at) = columns.iter().position(|c| c == name) else {
                continue;
            };
            let answered = distinct_count(
                rows.iter()
                    .filter(|r| r.len() > at && !r[at].is_null())
                    .map(|r| &r[at])
                    .collect(),
                protocols,
            )?;
            if answered != 1 {
                continue;
            }
            let what = how_to_name_the_target(column.as_deref(), asked);
            if kind.contains("classification") {
                return Err(unsupported(
                    format!(
                        "every context row has the same {what}, and a classification needs at \
                         least two classes to learn from. {WIDEN_THE_CONTEXT}"
                    ),
                    details([
                        ("model", json!(KUMO_RELATIONAL)),
                        ("target", json!(column)),
                        ("classes_in_context", json!(1)),
                    ]),
                ));
            }
            said.sdm(format!(
                "every context row has the same {what}, so nothing in the context distinguishes \
                 one answer from another. The endpoint will answer, but the answer cannot say \
                 much. Gather a wider context with num_context_samples, or ask about entities \
                 whose answers differ."
            ));
            break;
        }
    }
    if !outputs.iter().any(|o| o == "explanation") {
        return Ok(());
    }
    let scored: usize = payloads
        .iter()
        .map(|payload| {
            let predict = crate::pyops::mapping_get(payload, "predict", &json!({}), protocols)?;
            Ok(predict
                .py_get("instance_table")
                .and_then(|t| t.py_get("rows"))
                .and_then(crate::pyops::sequence)
                .map_or(0, Vec::len))
        })
        .collect::<R<Vec<usize>>>()?
        .into_iter()
        .sum();
    let allowed = relational_limits::EXPLANATION_PREDICT_ROWS;
    if sends.predict() && scored as i64 != allowed {
        return Err(unsupported(
            format!(
                "an explanation covers exactly {allowed} entity, and this request scores \
                 {scored}; ask for one at a time"
            ),
            details([("model", json!(KUMO_RELATIONAL)), ("scored", json!(scored))]),
        ));
    }
    let cap = relational_limits::CONTEXT_ROWS_FAST;
    if sends.context() && asks_for_the_larger_context(payloads, cap, protocols)? {
        return Err(unsupported(
            format!(
                "an explanation is only produced in the mode the smaller context budget selects, \
                 and this request was gathered with num_context_samples above {}. \
                 Re-materialize with num_context_samples={cap} or fewer",
                sdm_schema::py::thousands(cap)
            ),
            details([
                ("model", json!(KUMO_RELATIONAL)),
                ("context_rows_fast", json!(cap)),
            ]),
        ));
    }
    Ok(())
}

/// `_asks_for_the_larger_context(payloads, cap)`.
fn asks_for_the_larger_context(
    payloads: &[Value],
    cap: i64,
    protocols: Option<&crate::pyops::HostProtocols>,
) -> R<bool> {
    let mut modes: Vec<String> = Vec::new();
    for payload in payloads {
        let inference = crate::pyops::mapping_get(payload, "inference", &Value::Null, protocols)?;
        if crate::pyops::is_instance(&inference, crate::pyops::InstanceFamily::Mapping, protocols)?
        {
            let inference =
                crate::pyops::getitem(payload, &PyVal::Str("inference".into()), protocols)?;
            let mode = crate::pyops::mapping_get(&inference, "run_mode", &Value::Null, protocols)?;
            if mode.is_null() {
                continue;
            }
            let inference =
                crate::pyops::getitem(payload, &PyVal::Str("inference".into()), protocols)?;
            let mode =
                crate::pyops::getitem(&inference, &PyVal::Str("run_mode".into()), protocols)?;
            let text = PyVal::decode(&mode)
                .map(|value| value.str_())
                .unwrap_or_default();
            if !modes.contains(&text) {
                modes.push(text);
            }
        }
    }
    if !modes.is_empty() {
        return Ok(modes != ["fast"]);
    }
    let rows: usize = payloads
        .iter()
        .map(|payload| context_table(payload, protocols).map(|table| table.1.len()))
        .collect::<R<Vec<_>>>()?
        .into_iter()
        .sum();
    Ok(rows as i64 > cap)
}

/// `_how_to_name_the_target(column, asked)`.
fn how_to_name_the_target(column: Option<&str>, asked: Option<&str>) -> String {
    if column != Some("TARGET") {
        return column.map_or_else(|| "None".to_string(), str::to_string);
    }
    if let Some(asked) = asked {
        let trimmed = asked.trim_start();
        if trimmed.len() > 7 && trimmed[..7].eq_ignore_ascii_case("PREDICT") {
            let rest = trimmed[7..].trim_start();
            if rest.len() < trimmed.len() - 7 {
                let named: String = rest
                    .chars()
                    .take_while(|c| c.is_alphanumeric() || *c == '_' || *c == '.')
                    .collect();
                let parts: Vec<&str> = named.split('.').collect();
                let word = |s: &str| {
                    s.chars()
                        .next()
                        .is_some_and(|c| c.is_alphabetic() || c == '_')
                        && s.chars().all(|c| c.is_alphanumeric() || c == '_')
                };
                let after = rest[named.len()..].chars().next();
                if parts.len() == 2
                    && parts.iter().all(|p| word(p))
                    && after.is_some_and(char::is_whitespace)
                {
                    return named;
                }
            }
        }
        return format!("answer to {}", sdm_pql::pyrepr::str_repr(asked));
    }
    "answer".to_string()
}

/// `explain_error(error, payload)`: the columns behind a semantic-type or
/// cardinality refusal, named.
pub fn explain_error(error: Diagnostic, payload: &Value) -> Diagnostic {
    let said = error_text(&error);
    let (columns, complaint) = columns_the_endpoint_means(&said, payload);
    if columns.is_empty() {
        return error;
    }
    let message = if complaint == "cardinality" {
        format!(
            "{}. In this request that is {}. A column with that many distinct values carries no \
             signal the model can use; leave it out with materialize(..., \
             exclude_cols_dict=...), or replace it with something coarser.",
            error.message,
            columns.join(", ")
        )
    } else {
        let named = columns.iter().find(|c| !c.starts_with("instance_table."));
        let suggestion = match named {
            Some(n) => {
                let (table, column) = n.split_once('.').unwrap_or((n, ""));
                format!(
                    "schema[{}][{}].stype = 'categorical'",
                    sdm_pql::pyrepr::str_repr(table),
                    sdm_pql::pyrepr::str_repr(column)
                )
            }
            None => {
                let column = columns[0].split_once('.').map_or("", |(_, c)| c);
                format!(
                    "schema[<the entity table>][{}].stype = 'categorical'",
                    sdm_pql::pyrepr::str_repr(column)
                )
            }
        };
        format!(
            "{}. In this request that is {}. Set a supported type on each, e.g. {suggestion}",
            error.message,
            columns.join(", ")
        )
    };
    let mut reworded = error.clone();
    reworded.message = message;
    if reworded.class.as_deref() != Some(NIM_REQUEST) {
        reworded.class = Some(crate::error::STRUCTURED.to_string());
    }
    reworded
}

/// `str(error)` as today's classes write it.
pub fn error_text(error: &Diagnostic) -> String {
    if error.class.as_deref() == Some(NIM_REQUEST) {
        let status = error.status.unwrap_or(0);
        return match &error.code {
            Some(code) => format!("[{status} {code}] {}", error.message),
            None => format!("[{status}] {}", error.message),
        };
    }
    match &error.code {
        Some(code) if !code.is_empty() => format!("[{code}] {}", error.message),
        _ => error.message.clone(),
    }
}

fn columns_the_endpoint_means(said: &str, payload: &Value) -> (Vec<String>, &'static str) {
    if let Some(start) = said.find("semantic type '") {
        let rest = &said[start + "semantic type '".len()..];
        if let Some(end) = rest.find('\'') {
            let stype = &rest[..end];
            if rest[end..].starts_with("' is unsupported") {
                return (columns_with_stype(payload, stype), "stype");
            }
        }
    }
    let lowered = said.to_lowercase();
    if let Some(start) = lowered.find("cardinality ") {
        let rest = &lowered[start + "cardinality ".len()..];
        let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
        if !digits.is_empty() {
            let after = &rest[digits.len()..];
            if let Some(limit_text) = after.strip_prefix(" exceeds limit ") {
                let limit: String = limit_text
                    .chars()
                    .take_while(char::is_ascii_digit)
                    .collect();
                if let Ok(limit) = limit.parse::<usize>() {
                    return (columns_over(payload, limit), "cardinality");
                }
            }
        }
    }
    (Vec::new(), "")
}

fn columns_with_stype(payload: &Value, stype: &str) -> Vec<String> {
    let mut found = Vec::new();
    let schema = payload.py_get("schema");
    let mut tables: Vec<(String, &Value)> = Vec::new();
    if let Some(instance) = schema.and_then(|s| s.py_get("instance_table")) {
        tables.push(("instance_table".to_string(), instance));
    }
    if let Some(related) = schema
        .and_then(|s| s.py_get("related_tables"))
        .and_then(crate::pyops::mapping_items)
    {
        for (name, table) in related {
            let name = name.str_();
            tables.retain(|(existing, _)| existing != &name);
            tables.push((name, table));
        }
    }
    for (name, table) in tables {
        if let Some(columns) = table
            .py_get("columns")
            .and_then(crate::pyops::mapping_items)
        {
            for (column, spec) in columns {
                if spec.py_get("stype").and_then(Value::as_str) == Some(stype) {
                    found.push(format!("{name}.{}", column.str_()));
                }
            }
        }
    }
    found.sort();
    found
}

/// `high_cardinality_columns(payload, limit)`, named as the adapter names
/// them.
fn columns_over(payload: &Value, limit: usize) -> Vec<String> {
    crate::payload::high_cardinality_columns(payload, limit)
        .into_iter()
        .map(|(table, column, count)| {
            format!(
                "{table}.{column} ({} values)",
                sdm_schema::py::thousands(count as i64)
            )
        })
        .collect()
}

/// `_key_dtype(payload, reference)`.
fn key_dtype(payload: &Value, reference: &str) -> Option<String> {
    let schema = payload.py_get("schema")?;
    for link in crate::pyops::sequence(schema.py_get("relationships")?)? {
        let sources: Vec<&str> = link
            .py_get("source_columns")
            .and_then(Value::as_array)
            .map(|s| s.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default();
        if sources != [reference] {
            continue;
        }
        let target = link.py_get("target_table").and_then(Value::as_str)?;
        let columns = link
            .py_get("target_columns")
            .and_then(crate::pyops::sequence)?;
        if columns.len() != 1 {
            return None;
        }
        return schema
            .py_get("related_tables")?
            .py_get(target)?
            .py_get("columns")?
            .py_get(columns[0].as_str()?)?
            .py_get("dtype")?
            .as_str()
            .map(str::to_string);
    }
    None
}

/// `_named_entities(items, payload)` of the relational adapter.
pub fn named_entities(
    items: Vec<Item>,
    payload: &Value,
    protocols: Option<&crate::pyops::HostProtocols>,
) -> R<Vec<Item>> {
    let table = payload
        .py_get("predict")
        .and_then(|p| p.py_get("instance_table"));
    let columns: Vec<String> = table
        .and_then(|t| t.py_get("columns"))
        .and_then(crate::pyops::sequence)
        .map(|c| {
            c.iter()
                .map(|v| v.as_str().unwrap_or_default().to_string())
                .collect()
        })
        .unwrap_or_default();
    let rows: Vec<Vec<Value>> = table
        .and_then(|t| t.py_get("rows"))
        .and_then(crate::pyops::sequence)
        .map(|r| {
            r.iter()
                .map(|row| crate::pyops::sequence(row).cloned().unwrap_or_default())
                .collect()
        })
        .unwrap_or_default();
    let Some(reference) = columns
        .iter()
        .find(|c| c.starts_with("__kumo_entity_ref"))
        .cloned()
    else {
        return Ok(items);
    };
    let at = columns.iter().position(|c| *c == reference).unwrap_or(0);
    let dtype = key_dtype(payload, &reference);
    let entities: Vec<PyVal> = rows
        .iter()
        .map(|row| {
            let cell = row.get(at).map_or(PyVal::None, crate::keys::json_cell);
            crate::wire::as_declared(&cell, dtype.as_deref())
        })
        .collect();
    let instances: Option<Vec<Value>> = columns.iter().position(|c| c == "instance_id").map(|i| {
        rows.iter()
            .map(|row| row.get(i).cloned().unwrap_or(Value::Null))
            .collect()
    });
    let task = crate::pyops::mapping_get(payload, "task", &json!({}), protocols)?;
    let forecasting = crate::pyops::mapping_get(&task, "kind", &Value::Null, protocols)?.as_str()
        == Some("forecasting");
    let mut named = Vec::new();
    let mut seen: Vec<(i64, Option<i64>)> = Vec::new();
    for item in items {
        let Some(index) = item.row_index_i64() else {
            return Err(structured(
                "INVALID_RESPONSE",
                "the response is missing a row_index, so its answer cannot be attributed to an \
                 entity",
            ));
        };
        if index < 0 || index as usize >= entities.len() {
            return Err(structured(
                "INVALID_RESPONSE",
                format!(
                    "the response names row_index {index}, and this batch asked about {} rows",
                    entities.len()
                ),
            ));
        }
        let step = item.forecast_step_i64();
        if forecasting {
            if step.is_none() {
                return Err(structured(
                    "INVALID_RESPONSE",
                    format!(
                        "a forecast answer for row {index} names no step, so its place in the \
                         horizon is unknown"
                    ),
                ));
            }
        } else if step.is_some() {
            let kind = payload
                .py_get("task")
                .and_then(|t| t.py_get("kind"))
                .map_or("None".to_string(), |k| {
                    crate::pyops::py_repr(&crate::keys::json_cell(k))
                });
            return Err(structured(
                "INVALID_RESPONSE",
                format!("a {kind} answer carries a forecast step, which only a forecast has"),
            ));
        }
        if seen.contains(&(index, step)) {
            let place = if forecasting {
                format!("row {index} step {}", step.unwrap_or(0))
            } else {
                format!("row {index}")
            };
            return Err(structured(
                "INVALID_RESPONSE",
                format!("the response answers {place} more than once"),
            ));
        }
        seen.push((index, step));
        if let Some(instances) = &instances {
            let asked = crate::keys::json_cell(&instances[index as usize]);
            let Some(id) = &item.id else {
                return Err(structured(
                    "INVALID_RESPONSE",
                    format!(
                        "the answer for row {index} carries no id, and this batch asked under {}, \
                         so the two cannot be matched",
                        crate::pyops::py_repr(&asked)
                    ),
                ));
            };
            if crate::pyops::py_str(id) != crate::pyops::py_str(&asked) {
                return Err(structured(
                    "INVALID_RESPONSE",
                    format!(
                        "the response pairs row_index {index} with id {}, and this batch asked \
                         about {} there",
                        crate::pyops::py_repr(id),
                        crate::pyops::py_repr(&asked)
                    ),
                ));
            }
        }
        let mut item = item;
        item.id = Some(entities[index as usize].clone());
        named.push(item);
    }
    let answered: HashSet<i64> = seen.iter().map(|(i, _)| *i).collect();
    let mut missing: Vec<i64> = (0..entities.len() as i64)
        .filter(|i| !answered.contains(i))
        .collect();
    if !missing.is_empty() {
        missing.sort_unstable();
        let shown: Vec<String> = missing.iter().map(i64::to_string).collect();
        return Err(structured(
            "INVALID_RESPONSE",
            format!("the response left row(s) [{}] unanswered", shown.join(", ")),
        ));
    }
    Ok(named)
}

/// What a session needs of a payload's halves besides the context.
pub fn pinned_without_context(payload: &Value) -> Value {
    let mut out = payload.clone();
    crate::pyops::set_field(&mut out, "context", None);
    out
}

#[cfg(test)]
mod protocol_tests {
    use super::sorted_set_operation;
    use crate::pyops::HostProtocols;
    use sdm_schema::pyval::PyVal;
    use serde_json::json;

    #[test]
    fn held_set_cardinality_requires_valid_scoped_facts() {
        let held = json!({"$held": {"reference": {"$host_ref": "0"}, "value": {"$object": {"type": "object", "repr": ""}}, "owner": "scope"}});
        assert!(super::distinct_count(vec![&held], None).is_err());
        for invalid in [json!(-1), json!(2), json!(true), json!(0.5), json!(null)] {
            let protocols = HostProtocols::new(move |operation, _| {
                assert_eq!(operation, "set_cardinality");
                Ok(invalid.clone())
            });
            assert!(super::distinct_count(vec![&held], Some(&protocols)).is_err());
        }
        let protocols = HostProtocols::new(|_, _| Ok(json!(1)));
        assert_eq!(
            super::distinct_count(vec![&held], Some(&protocols)).unwrap(),
            1
        );
        assert_eq!(
            super::distinct_count(vec![&json!(1), &json!(true)], None).unwrap(),
            1
        );
    }

    #[test]
    fn ordinary_set_operations_need_no_host() {
        let given = vec![PyVal::Str("z".into()), PyVal::Str("a".into())];
        let keys = sorted_set_operation(None, "sorted_set_difference", given, Vec::new()).unwrap();
        assert_eq!(keys, vec![PyVal::Str("a".into()), PyVal::Str("z".into())]);
    }

    #[test]
    fn non_string_set_operations_require_explicit_host_facts() {
        let error = sorted_set_operation(
            None,
            "sorted_set_difference",
            vec![PyVal::Int(1)],
            Vec::new(),
        )
        .unwrap_err();
        assert_eq!(
            error.message,
            "non-string set ordering requires host protocol facts"
        );
    }

    #[test]
    fn operation_scoped_host_supplies_ordered_values() {
        let protocols = HostProtocols::new(|operation, operands| {
            assert_eq!(operation, "sorted_set_difference");
            assert_eq!(operands, &json!([[3, 1, 2], [3]]));
            Ok(json!([1, 2]))
        });
        let keys = sorted_set_operation(
            Some(&protocols),
            "sorted_set_difference",
            vec![PyVal::Int(3), PyVal::Int(1), PyVal::Int(2)],
            vec![PyVal::Int(3)],
        )
        .unwrap();
        assert_eq!(keys, vec![PyVal::Int(1), PyVal::Int(2)]);
    }
}
