// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `sdm_client/client.py` and `base.py`: preparing a request for a model,
//! sending it through the host, and reading the answer back; sessions; and
//! a question asked in parts, put back together.
//!
//! Every call to the model is a step the host makes
//! ([`sdm_protocol::Step::CallModel`]); everything else happens here.

use sdm_protocol::{Asker, Diagnostic, Endpoint, Outcome};
use sdm_schema::client::Schema;
use sdm_schema::dt::Host;
use sdm_schema::pyval::PyVal;
use serde_json::{Map, Value, json};

use crate::error::{R, Warnings, details, invalid, structured, unsupported};
use crate::models::{Profile, Spec};
use crate::request::{Request, schema_table};
use crate::result::Result_;
use crate::schemas::Tables;

/// `SHAPE_CAPABILITIES`.
const SHAPE_CAPABILITIES: [&str; 6] = [
    "relationships",
    "multiple_tables",
    "link_prediction",
    "forecasting",
    "static",
    "temporal",
];

/// What a model call is answered with: the answer, or the error the
/// transport raised, as the host described it.
pub fn model_error(text: &str) -> Diagnostic {
    let found: Value = serde_json::from_str(text).unwrap_or(Value::Null);
    let class = found
        .get("class")
        .and_then(Value::as_str)
        .unwrap_or("sdm_client.errors.StructuredDataError");
    let code = found.get("code").and_then(Value::as_str);
    let message = found
        .get("message")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let mut error = Diagnostic::refusal(class, code, message);
    if let Some(Value::Object(details)) = found.get("details") {
        error.details = Box::new(details.clone());
    }
    error.status = found
        .get("status")
        .and_then(Value::as_i64)
        .and_then(|s| i32::try_from(s).ok());
    error
}

/// One call to the model through the host.
pub async fn call(
    asker: &Asker,
    endpoint: Endpoint,
    session_id: Option<String>,
    body: Option<&Value>,
) -> R<Value> {
    let outcome = asker
        .call_model(endpoint, session_id, body.map(Value::to_string))
        .await;
    match outcome {
        Outcome::Response(text) => serde_json::from_str(&text)
            .map_err(|e| crate::error::value_error(format!("the model's answer is not JSON: {e}"))),
        Outcome::ModelError(text) => Err(model_error(&text)),
        _ => Err(crate::error::value_error(
            "a query's outcome was submitted for a model call",
        )),
    }
}

/// The capabilities a request needs (`required_capabilities`).
fn required(request: &Request, schema: &Schema) -> R<Vec<(&'static str, bool)>> {
    let kind = request.task.kind.as_str();
    let temporal = request
        .task
        .options
        .iter()
        .find(|(key, _)| matches!(key, PyVal::Str(name) if name == "anchor_time_column"))
        .map(|(_, value)| crate::pyops::truth_with(value, request.protocols.as_ref()))
        .transpose()?
        .unwrap_or(false);
    let temporal = temporal || {
        hashable_kind(request)?;
        matches!(kind, Some("forecasting" | "temporal_link_prediction"))
    };
    Ok(vec![
        ("relationships", !schema.relationships.is_empty()),
        ("multiple_tables", schema.tables.len() > 1),
        ("link_prediction", kind == Some("temporal_link_prediction")),
        ("forecasting", kind == Some("forecasting")),
        ("temporal", temporal),
        ("static", !temporal),
    ])
}

fn hashable_kind(request: &Request) -> R<()> {
    crate::pyops::hashable_with(&request.task.kind, request.protocols.as_ref())
}

fn sorted_list(names: &[&str]) -> Vec<String> {
    let mut out: Vec<String> = names.iter().map(|s| s.to_string()).collect();
    out.sort();
    out
}

/// `check_request_against_model(request, spec, outputs)`.
pub fn check_request_against_model(
    request: &Request,
    schema: &Schema,
    spec: &Spec,
    outputs: &[String],
) -> R<()> {
    let needed = required(request, schema)?;
    let mut missing: Vec<&str> = SHAPE_CAPABILITIES
        .iter()
        .copied()
        .filter(|name| {
            needed.iter().any(|(n, on)| n == name && *on) && !spec.profile.supports(name)
        })
        .collect();
    if !missing.is_empty() {
        missing.sort_unstable();
        return Err(unsupported(
            format!(
                "{} cannot answer a request needing {}",
                spec.model_id,
                missing.join(", ")
            ),
            details([
                ("model", json!(spec.model_id)),
                ("unsupported", json!(missing)),
            ]),
        ));
    }
    hashable_kind(request)?;
    if !request
        .task
        .kind
        .as_str()
        .is_some_and(|kind| spec.tasks.contains(&kind))
    {
        let tasks = sorted_list(spec.tasks);
        return Err(unsupported(
            format!(
                "{} does not serve the task kind {}; it serves {}",
                spec.model_id,
                crate::pyops::repr_with(&request.task.kind, request.protocols.as_ref())?,
                sdm_schema::py::str_list_repr(&tasks)
            ),
            details([
                ("model", json!(spec.model_id)),
                ("task", crate::layout::encode_deep(&request.task.kind)),
                ("supported", json!(tasks)),
            ]),
        ));
    }
    let mut unknown: Vec<String> = outputs
        .iter()
        .filter(|o| !spec.outputs.contains(&o.as_str()))
        .cloned()
        .collect();
    if !unknown.is_empty() {
        unknown.sort();
        let supported = sorted_list(spec.outputs);
        return Err(unsupported(
            format!(
                "{} does not produce {}; it produces {}",
                spec.model_id,
                sdm_schema::py::str_list_repr(&unknown),
                sdm_schema::py::str_list_repr(&supported)
            ),
            details([
                ("model", json!(spec.model_id)),
                ("unsupported_outputs", json!(unknown)),
                ("supported_outputs", json!(supported)),
            ]),
        ));
    }
    if let Some(cap) = spec.context_rows_max {
        let rows = request.context_rows()? as i64;
        if rows > cap {
            return Err(unsupported(
                format!(
                    "this request carries {} context rows and {} reads at most {}. A request \
                     built directly is sent as it was given, so choose the rows you want as \
                     context, or gather it with materialize(), which keeps the context under \
                     the cap and says how",
                    sdm_schema::py::thousands(rows),
                    spec.model_id,
                    sdm_schema::py::thousands(cap)
                ),
                details([
                    ("model", json!(spec.model_id)),
                    ("context_rows", json!(rows)),
                    ("context_rows_max", json!(cap)),
                ]),
            ));
        }
    }
    Ok(())
}

/// The fields to ask for (`SDMClient._prepare`): the caller's, or the
/// model's default for the task, with the request checked against it.
pub fn prepare(
    request: &Request,
    schema: &Schema,
    spec: &Spec,
    outputs: Option<&[String]>,
) -> R<Vec<String>> {
    let fields: Vec<String> = match outputs {
        Some(o) => o.to_vec(),
        None => {
            hashable_kind(request)?;
            default_output(spec, request.task.kind.as_str().unwrap_or_default())
        }
    };
    if fields.is_empty() {
        return Err(invalid("outputs must name at least one field"));
    }
    check_request_against_model(request, schema, spec, &fields)?;
    Ok(fields)
}

/// The model's default output fields for a task kind.
pub fn default_output(spec: &Spec, kind: &str) -> Vec<String> {
    match spec.profile {
        Profile::Tabular => crate::tabular_adapter::default_output(kind),
        Profile::Relational => match kind {
            "classification" | "binary_classification" | "multiclass_classification" => {
                vec!["prediction".into(), "probabilities".into()]
            }
            "temporal_link_prediction" => vec!["rankings".into()],
            _ => vec!["prediction".into()],
        },
    }
}

/// `_report_absent(result, requested, model)`.
pub fn report_absent(
    result: &Result_,
    requested: Option<&[String]>,
    model: &str,
    said: &mut Warnings,
) {
    let Some(requested) = requested.filter(|r| !r.is_empty()) else {
        return;
    };
    let absent = result.absent(requested);
    if absent.is_empty() {
        return;
    }
    said.sdm(format!(
        "{model} answered without {}, which this request asked for, so no column of to_pandas() \
         will carry it. The answer to the rest of the request is unaffected.",
        sdm_schema::py::str_list_repr(&absent)
    ));
}

/// `_names_its_rows(request)`.
pub fn names_its_rows(request: &Request, schema: &Schema) -> bool {
    if request.payloads().is_some() {
        return true;
    }
    let Some(PyVal::Str(table)) = &request.entity_table else {
        return false;
    };
    let Ok(described) = schema_table(schema, table) else {
        return false;
    };
    let key = sdm_schema::client::key_names_of(&described.primary_key);
    if key.is_empty() {
        return false;
    }
    match &request.halves {
        Some(h) => key.iter().all(|k| h.predict.has(k)),
        None => false,
    }
}

/// `_empty_answer(model, outputs, request)`.
pub fn empty_answer(
    model: &str,
    outputs: &[String],
    request: Option<(&Request, &Schema)>,
) -> Result_ {
    let mut built = Result_::new(model, "", Vec::new(), outputs.to_vec());
    built.names_its_rows = request.is_none_or(|(r, s)| names_its_rows(r, s));
    built
}

/// `warn_of_rows_scored_from_their_own_answers(request)`.
pub fn warn_of_rows_scored_from_their_own_answers(
    request: &Request,
    schema: &Schema,
    said: &mut Warnings,
) {
    if request.payloads().is_some() {
        return;
    }
    let (Some(PyVal::Str(entity)), Some(halves)) = (&request.entity_table, &request.halves) else {
        return;
    };
    let PyVal::Str(target) = &request.task.target.column_name else {
        return;
    };
    if target.is_empty() || !halves.context.has(target) {
        return;
    }
    let Ok(table) = schema_table(schema, entity) else {
        return;
    };
    let key = sdm_schema::client::key_names_of(&table.primary_key);
    if key.is_empty()
        || key
            .iter()
            .any(|k| !halves.context.has(k) || !halves.predict.has(k))
    {
        return;
    }
    let answered_rows: Vec<usize> = halves
        .context
        .series(target)
        .map(|s| {
            s.cells
                .iter()
                .enumerate()
                .filter(|(_, c)| !crate::series::is_na(c))
                .map(|(i, _)| i)
                .collect()
        })
        .unwrap_or_default();
    if answered_rows.is_empty() {
        return;
    }
    let answered = halves.context.take(&answered_rows);
    let dtypes = crate::keys::key_dtypes(table, &key, Some(&halves.predict));
    let answers = crate::keys::keys_from_a_frame(&answered, &key, dtypes.as_deref());
    if answers.is_empty() {
        return;
    }
    let shown = crate::keys::keys_from_a_frame(&halves.predict, &key, dtypes.as_deref())
        .iter()
        .filter(|named| answers.contains(named))
        .count();
    if shown == 0 {
        return;
    }
    said.sdm(format!(
        "{shown} of the {} rows being scored are also in context, carrying the {} being asked \
         for, so the model is being shown the answers to rows it is being asked about. Drop them \
         from context, or gather the request with materialize(), which leaves the rows it scores \
         out of the context it sends.",
        halves.predict.len(),
        sdm_pql::pyrepr::str_repr(target)
    ));
}

/// `adapter.normalize(response, outputs, payload)`.
pub fn normalize(
    spec: &Spec,
    response: &Value,
    outputs: &[String],
    payload: &Value,
    protocols: Option<&crate::pyops::HostProtocols>,
) -> R<Result_> {
    let parsed = crate::response::parse(response).map_err(|e| {
        structured(
            "INVALID_RESPONSE",
            format!(
                "{} returned a response that does not match the contract: {e}",
                spec.model_id
            ),
        )
        .caused_by([sdm_schema::engine::link(
            "ResponseContractError",
            e.to_string(),
        )])
    })?;
    let (items, explanation) = match spec.profile {
        Profile::Tabular => (
            crate::tabular_adapter::named_entities(parsed.predictions, payload)?,
            None,
        ),
        Profile::Relational => {
            let explanation = parsed
                .predictions
                .iter()
                .find_map(|i| i.explanation.clone().filter(crate::pyops::py_bool));
            (
                crate::relational_adapter::named_entities(parsed.predictions, payload, protocols)?,
                explanation,
            )
        }
    };
    let mut result = Result_::new(
        &parsed.model,
        &parsed.id,
        crate::response::as_declared_target(items, payload, protocols)?,
        outputs.to_vec(),
    );
    result.metadata = PyVal::Dict(
        parsed
            .metadata
            .into_iter()
            .map(|(key, value)| (PyVal::Str(key), value))
            .collect(),
    );
    result.explanation = explanation;
    Ok(result)
}

/// The arguments of a prediction: the model, as named, and the fields.
#[derive(Clone, Debug)]
pub struct Asked {
    pub model: PyVal,
    pub outputs: Option<Vec<String>>,
}

/// `SDMClient.predict(request, model=..., outputs=...)`.
pub async fn predict(
    asker: &Asker,
    request: &Request,
    tables: &Tables,
    asked: &Asked,
    host: &Host,
    said: &mut Warnings,
) -> R<Result_> {
    if request.parts.is_some() {
        return crate::parts::answer_in_parts(asker, request, tables, asked, host, said).await;
    }
    let spec = crate::models::spec_for(&asked.model)?;
    let mut schema = request.schema.build(tables, host)?;
    let fields = prepare(request, &schema, spec, asked.outputs.as_deref())?;
    if request.nothing_to_score()? {
        return Ok(empty_answer(
            spec.model_id,
            &fields,
            Some((request, &schema)),
        ));
    }
    warn_of_rows_scored_from_their_own_answers(request, &schema, said);
    let plan = match spec.profile {
        Profile::Tabular => vec![crate::tabular_adapter::envelope(
            request,
            &mut schema,
            &fields,
            None,
            said,
        )?],
        Profile::Relational => crate::relational_adapter::prediction_plan(
            request,
            &fields,
            crate::relational_adapter::Sends::Everything,
            said,
        )?,
    };
    let mut parts = Vec::new();
    for payload in &plan {
        let response = match call(asker, Endpoint::Predict, None, Some(payload)).await {
            Ok(response) => response,
            Err(error) => return Err(explained(spec, error, payload)),
        };
        parts.push(normalize(
            spec,
            &response,
            &fields,
            payload,
            request.protocols.as_ref(),
        )?);
    }
    if parts.is_empty() {
        return Err(invalid("the adapter produced no call for this request"));
    }
    let merged = Result_::merge(parts)?;
    report_absent(&merged, asked.outputs.as_deref(), spec.model_id, said);
    Ok(merged)
}

/// [`predict`], boxed, so a question asked in parts can ask its parts.
pub fn predict_boxed<'a>(
    asker: &'a Asker,
    request: &'a Request,
    tables: &'a Tables,
    asked: &'a Asked,
    host: &'a Host,
    said: &'a mut Warnings,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = R<Result_>> + Send + 'a>> {
    Box::pin(predict(asker, request, tables, asked, host, said))
}

/// The link that stands for the error the host met making the last model
/// call itself.
pub fn from_model() -> Value {
    json!({"from": "model"})
}

/// What a model call refused with, as `_send` raises it: the error itself
/// where the adapter had nothing to add, or what it added, raised from the
/// error the host met (`raise explained from error`).
pub fn explained(spec: &Spec, error: Diagnostic, payload: &Value) -> Diagnostic {
    let original = error.clone();
    let made = explain_error(spec, error, payload);
    if made == original {
        made
    } else {
        made.caused_by([from_model()])
    }
}

/// `adapter.explain_error(error, payload)`, for an error of the
/// `StructuredDataError` family; others pass as they are.
pub fn explain_error(spec: &Spec, error: Diagnostic, payload: &Value) -> Diagnostic {
    let family = matches!(
        error.class.as_deref(),
        Some(
            "sdm_client.errors.StructuredDataError"
                | "sdm_client.errors.NimRequestError"
                | "sdm_client.errors.UnsupportedRequestError"
                | "sdm_client.errors.UnknownModelError"
                | "sdm_client.errors.MissingExtraError"
        )
    );
    if !family {
        return error;
    }
    match spec.profile {
        Profile::Tabular => error,
        Profile::Relational => crate::relational_adapter::explain_error(error, payload),
    }
}

/// Keys of a mapping in JSON form.
pub fn object_keys(value: &Value) -> Vec<String> {
    value
        .as_object()
        .map(|m| m.keys().cloned().collect())
        .unwrap_or_default()
}

/// An empty mapping.
pub fn empty() -> Map<String, Value> {
    Map::new()
}

/// `_require_output_names(outputs)`: the output fields a call names, or
/// `None`; anything but a sequence of names refused.
pub fn require_output_names(outputs: &PyVal) -> R<Option<Vec<String>>> {
    match outputs {
        PyVal::None => Ok(None),
        PyVal::Str(_) => Err(invalid(format!(
            "outputs must be a sequence of field names, not a single string; pass [{}]",
            crate::pyops::py_repr(outputs)
        ))),
        PyVal::List(items) | PyVal::Tuple(items) => {
            let names: Option<Vec<String>> = items
                .iter()
                .map(|v| match v {
                    PyVal::Str(s) => Some(s.clone()),
                    _ => None,
                })
                .collect();
            names.map(Some).ok_or_else(|| {
                invalid(format!(
                    "outputs must be a sequence of field names, got {}",
                    crate::models::type_name(outputs)
                ))
            })
        }
        other => Err(invalid(format!(
            "outputs must be a sequence of field names, got {}",
            crate::models::type_name(other)
        ))),
    }
}
