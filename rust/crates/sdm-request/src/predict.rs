// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A materialized request sent to the model, and its answer laid out.
//!
//! The client's predict-side rules, in the reference's order (T1.28b, the
//! design's "Host-held semantics" rows it moved):
//!
//! - [`prepare`] is `SDMClient._prepare`: the model the request is sent to
//!   (`UNKNOWN_MODEL` for one the contract does not carry), the outputs
//!   asked for (the task's default ones when the caller named none, an
//!   empty list refused `INVALID_REQUEST`) and `check_request_against_model`
//!   (the request's shape, its task's kind, each output and the context's
//!   size against what the model serves);
//! - [`nothing_to_score`] is `Request.nothing_to_score`: a request with no
//!   row to score is answered without a call (`_empty_answer`, [`empty`]);
//! - [`preflight`] is the relational adapter's `_preflight`: an empty
//!   context, more classes than the model takes and a classification whose
//!   context has one answer refused, a regression's constant context given
//!   as a notice, and an explanation's row count and run mode;
//! - [`body`] is `prediction_plan`'s call of one payload: the payload with
//!   the model named, the outputs asked for (a regression's with its
//!   statistic), the request's id in its metadata, and only the inference
//!   options the model reads (`_inference`);
//! - [`layout`] is `normalize` and `_named_entities`, with the answer's
//!   classes given back as the target's own type (`as_declared_target`):
//!   each answer checked against the row it claims and given back under the
//!   entity it is about, read back as the key's dtype (`as_declared`), or,
//!   where the payload names no entity, given back as written, as a
//!   [`Part`];
//! - [`merge`] is `PredictionResult.merge`, the parts of one request put
//!   together, and [`absent`] is `_report_absent`'s finding, the outputs the
//!   caller named that no answer carries;
//! - [`table`] lays a part out as the result: one row per answer, ordered
//!   by `row_index` as `PredictionResult.to_pandas` orders them.
//!
//! What the result does not carry yet is refused by name, never dropped
//! (`UNSUPPORTED_REQUEST`, register P1B-4): an item's `scores`, `rankings`,
//! `embeddings`, a classification answer that is not of its target's type,
//! and the tabular model. From T2.13 an item's `explanation` and
//! `metadata` are carried (each item's, and the result's explanation,
//! [`Part::explanation`]), and from T2.9 a regression's `quantiles`.

use std::collections::BTreeSet;
use std::sync::Arc;

use arrow_array::builder::{Float64Builder, ListBuilder, MapBuilder, MapFieldNames, StringBuilder};
use arrow_array::{
    Array, ArrayRef, BooleanArray, Float64Array, Int64Array, RecordBatch, StringArray,
};
use arrow_schema::{DataType, Field, Schema};
use sdm_contract::{CoreError, Detail, Scalar, Sentence};
use sdm_wire::json::{Value, is_integer_token};
use sdm_wire::response::{
    PredictionResponse, Refused, integer_text, integer_text_limited, numeric_text,
};

/// The model this build sends a relational request to.
pub const RELATIONAL: &str = "kumo-relational";

/// The other model the contract carries, which takes no relational request.
pub const TABULAR: &str = "kumo-tabular";

/// The output a regression asks for when the caller names none.
pub const REGRESSION_FIELDS: [&str; 1] = ["prediction"];

/// The outputs a classification asks for when the caller names none.
pub const CLASSIFICATION_FIELDS: [&str; 2] = ["prediction", "probabilities"];

/// The task kinds whose call this build writes.
const CALLED: [&str; 4] = [
    "temporal_link_prediction",
    "regression",
    "binary_classification",
    "multiclass_classification",
];

/// The statistic a regression's point prediction reports unless one was
/// chosen.
pub const DEFAULT_STATISTIC: &str = "median";

/// The inference-config keys the client forwards to `kumo-relational`
/// (`KUMO_RELATIONAL.runtime.inference_options`); the engine's others are
/// left out of the call.
pub const INFERENCE_OPTIONS: [&str; 6] = [
    "kind",
    "num_estimators",
    "column_shuffle",
    "category_shuffle",
    "hop_shuffle",
    "class_shuffle",
];

/// The most classes `kumo-relational` classifies (`classes_max`).
pub const CLASSES_MAX: usize = 10;

/// The most context rows `kumo-relational` reads (`context_rows_max`).
pub const CONTEXT_ROWS_MAX: usize = 10_000;

/// The context the mode that explains takes at most (`context_rows_fast`).
pub const CONTEXT_ROWS_FAST: usize = 1_000;

/// The rows an explanation covers (`explanation_predict_rows`).
pub const EXPLANATION_PREDICT_ROWS: usize = 1;

/// What the wire calls the target column whatever the question was.
const WIRE_TARGET: &str = "TARGET";

/// What the wire calls the column holding the entity a row is about, by
/// its prefix.
const ENTITY_REF: &str = "__kumo_entity_ref";

/// What the wire calls the column a row is answered under.
const INSTANCE_ID: &str = "instance_id";

/// The shapes a request may need of a model (`SHAPE_CAPABILITIES`), in the
/// reference's order.
const SHAPES: [&str; 6] = [
    "relationships",
    "multiple_tables",
    "link_prediction",
    "forecasting",
    "static",
    "temporal",
];

/// What the client knows a model serves (`ModelSpec`): its id, the shapes
/// its profile takes, its task kinds and its outputs, the explanation
/// shapes it names, and whether its endpoint pins sessions
/// (`runtime.explanation_formats`, `runtime.sessions`).
struct Spec {
    id: &'static str,
    shapes: &'static [&'static str],
    tasks: &'static [&'static str],
    outputs: &'static [&'static str],
    explanation_formats: &'static [&'static str],
    sessions: bool,
    context_rows_max: usize,
}

/// The models the contract carries, in this build's order.
const MODELS: [&Spec; 2] = [&KUMO_RELATIONAL, &KUMO_TABULAR];

/// What one model supports, as the client describes it to a caller
/// (`ModelCapabilities`, `capabilities()`): the client side, never the
/// endpoint's own answer, so its `source` is `sdk`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Capabilities {
    pub model: &'static str,
    /// Its task kinds, sorted.
    pub tasks: Vec<&'static str>,
    /// Its outputs, sorted.
    pub outputs: Vec<&'static str>,
    /// Every model takes one table.
    pub single_table: bool,
    /// Whether its profile takes several tables.
    pub multiple_tables: bool,
    /// Whether its profile takes relationships.
    pub relationships: bool,
    pub explanation_formats: Vec<&'static str>,
    pub sessions: bool,
}

/// Every model's [`Capabilities`], in this build's order of them.
pub fn capabilities() -> Vec<Capabilities> {
    MODELS
        .iter()
        .map(|spec| Capabilities {
            model: spec.id,
            tasks: sorted(spec.tasks),
            outputs: sorted(spec.outputs),
            single_table: true,
            multiple_tables: spec.shapes.contains(&"multiple_tables"),
            relationships: spec.shapes.contains(&"relationships"),
            explanation_formats: spec.explanation_formats.to_vec(),
            sessions: spec.sessions,
        })
        .collect()
}

/// `KUMO_RELATIONAL`: every shape (the relational profile), its six task
/// kinds and seven outputs, each list sorted.
const KUMO_RELATIONAL: Spec = Spec {
    id: RELATIONAL,
    shapes: &SHAPES,
    tasks: &[
        "binary_classification",
        "classification",
        "forecasting",
        "multiclass_classification",
        "regression",
        "temporal_link_prediction",
    ],
    outputs: &[
        "embeddings",
        "explanation",
        "prediction",
        "probabilities",
        "quantiles",
        "rankings",
        "scores",
    ],
    explanation_formats: &[],
    sessions: true,
    context_rows_max: CONTEXT_ROWS_MAX,
};

/// `KUMO_TABULAR`: one table, static, no link or forecast.
const KUMO_TABULAR: Spec = Spec {
    id: TABULAR,
    shapes: &["static"],
    tasks: &["classification", "regression"],
    outputs: &["prediction", "probabilities", "quantiles"],
    explanation_formats: &[],
    sessions: true,
    context_rows_max: CONTEXT_ROWS_MAX,
};

/// The refusal of a `predict` input that takes a rule this build does not
/// port yet, naming the rule, never a value (decision H17).
pub fn not_yet(rule: &'static str) -> CoreError {
    CoreError::new(
        Sentence::NotYetPredict,
        "predicting this request takes a rule this core does not hold yet",
    )
    .with_field("operation", Detail::text("predict"))
    .with_field("rule", Detail::text(rule))
}

/// The model cannot answer the request, as the client refuses it before
/// anything is sent (`UnsupportedRequestError`): the model named, never a
/// value of the request.
fn unsupported(sentence: Sentence, model: &str, message: &str) -> CoreError {
    CoreError::new(sentence, message).with_field("model", Detail::text(model))
}

/// `values` in order, as the reference's refusals list them (`sorted`).
fn sorted<'a>(values: &[&'a str]) -> Vec<&'a str> {
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    sorted
}

fn texts<'a>(values: impl IntoIterator<Item = &'a str>) -> Detail {
    Detail::List(
        values
            .into_iter()
            .map(|v| Scalar::Utf8(v.to_owned()))
            .collect(),
    )
}

fn count(n: usize) -> Detail {
    Detail::Scalar(Scalar::UInt64(u64::try_from(n).unwrap_or(u64::MAX)))
}

/// A request state's payload this build cannot read as one: the field and
/// why, never a value.
fn bad_payload(reason: &'static str) -> CoreError {
    CoreError::new(
        Sentence::StateUnread,
        "the request state's payload is not one materialize writes",
    )
    .with_field("field", Detail::text("state.payload"))
    .with_field("reason", Detail::text(reason))
}

/// The endpoint's answer, not one the contract describes or not the
/// answer to this request: the field and why, never a value.
pub fn bad_response(field: &'static str, reason: &'static str) -> CoreError {
    let sentence = match (field, reason) {
        ("prediction.explanation", "not_a_mapping") => Sentence::ExplanationNotAMapping,
        ("prediction.metadata", "not_a_mapping") => Sentence::MetadataNotAMapping,
        _ => Sentence::ResponseUnread,
    };
    CoreError::new(
        sentence,
        "the model endpoint's answer is not one this request can read",
    )
    .with_field("field", Detail::text(field))
    .with_field("reason", Detail::text(reason))
}

/// A response the wire's reader refused, as the error it ends the run with.
pub fn refused(refused: Refused) -> CoreError {
    bad_response(refused.field, refused.reason)
}

/// The task's kind as the payload writes it.
fn kind_of(payload: &Value) -> Result<&str, CoreError> {
    let Value::Object(_) = payload else {
        return Err(bad_payload("not_an_object"));
    };
    match payload.get("task").and_then(|task| task.get("kind")) {
        Some(Value::Text(kind)) => Ok(kind),
        _ => Err(bad_payload("task")),
    }
}

/// The task's kind, when it is one whose call this build writes.
fn called(payload: &Value) -> Result<&str, CoreError> {
    let kind = kind_of(payload)?;
    if CALLED.contains(&kind) {
        Ok(kind)
    } else {
        Err(not_yet("problem_type"))
    }
}

fn classifies(kind: &str) -> bool {
    kind.contains("classification")
}

/// One half's instance table: its column names and its rows.
fn instance_table<'a>(payload: &'a Value, half: &str) -> Option<(Vec<&'a str>, &'a [Value])> {
    let table = payload.get(half)?.get("instance_table")?;
    let Some(Value::Array(columns)) = table.get("columns") else {
        return None;
    };
    let Some(Value::Array(rows)) = table.get("rows") else {
        return None;
    };
    let names: Vec<&str> = columns.iter().filter_map(Value::as_text).collect();
    (names.len() == columns.len()).then_some((names, rows.as_slice()))
}

/// The rows of one half, counted as `Request._rows` counts them: none
/// where the half cannot be read.
fn rows_of(payload: &Value, half: &str) -> usize {
    instance_table(payload, half).map_or(0, |(_, rows)| rows.len())
}

/// The rows one call scores (`_envelope_rows(payload, 'predict')`): its
/// predict half's rows, none where the half cannot be read.
pub fn predict_rows(payload: &Value) -> Option<usize> {
    instance_table(payload, "predict").map(|(_, rows)| rows.len())
}

/// Where each call's rows are numbered from, for a request sent in several
/// calls (`_numbered_across_calls`, `main`'s 1fc5a686): the endpoint numbers
/// every call's rows from zero, so each call's are numbered on from where
/// the previous call's ended, each call counted before the first is sent.
/// A call before the last whose rows cannot be counted leaves the calls
/// after it unnumberable, `INTERNAL_ERROR`, as the client refuses it.
pub fn call_starts(payloads: &[Value]) -> Result<Vec<usize>, CoreError> {
    let counted: Vec<Option<usize>> = payloads.iter().map(predict_rows).collect();
    if counted
        .iter()
        .take(counted.len().saturating_sub(1))
        .any(Option::is_none)
    {
        return Err(CoreError::new(
            Sentence::Internal,
            "a call of a request sent in several calls has rows that cannot be counted",
        ));
    }
    let mut starts = Vec::with_capacity(counted.len());
    let mut start = 0usize;
    for rows in counted {
        starts.push(start);
        start = start.saturating_add(rows.unwrap_or(0));
    }
    Ok(starts)
}

/// `part` with each row numbered `start` places further on
/// (`_numbered_from`).
pub fn numbered_from(mut part: Part, start: usize) -> Result<Part, CoreError> {
    let start =
        i64::try_from(start).map_err(|_| bad_response("prediction.row_index", "out_of_range"))?;
    if start != 0 {
        for row in &mut part.rows {
            row.row_index = row
                .row_index
                .checked_add(start)
                .ok_or_else(|| bad_response("prediction.row_index", "out_of_range"))?;
        }
    }
    Ok(part)
}

/// The outputs asked of `payload` sent to `model`, as the client's
/// `_prepare` resolves and checks them: the model the contract carries
/// (`UNKNOWN_MODEL` otherwise), the outputs the caller named or, when it
/// named none (`defaulted`), the task's default ones, an empty list refused
/// (`INVALID_REQUEST`, field `outputs`), then `check_request_against_model`
/// in its order: the shapes the model's profile does not take, a task kind
/// it does not serve, an output it does not produce and a context past what
/// it reads, each `UNSUPPORTED_REQUEST` naming the model.
///
/// A request sent in several calls (`main`'s multi-call `predict`) is
/// checked as one: its kind and shapes are its first call's, as the
/// client's request takes its task from its first payload, and its context
/// is held to the most rows any one call sends (`_context_rows_per_call`),
/// since the endpoint's cap is per call.
pub fn prepare(
    payloads: &[Value],
    model: &str,
    asked: &[String],
    defaulted: bool,
) -> Result<Vec<String>, CoreError> {
    let Some(payload) = payloads.first() else {
        return Err(CoreError::new(Sentence::Internal, "a request of no call"));
    };
    let call = resolve_call(payload, model, asked, defaulted)?;
    validate_call(payloads, payload, &call)?;
    Ok(call.outputs)
}

/// Call-local model and output policy for the gathered payload, used only
/// by `prepare`; no native-input observation or suspended work is held.
struct ResolvedCall<'a> {
    spec: &'static Spec,
    kind: &'a str,
    outputs: Vec<String>,
}

/// Resolve the model, read the task and choose explicit or default
/// outputs, retaining the existing payload-path diagnostic order.
fn resolve_call<'a>(
    payload: &'a Value,
    model: &str,
    asked: &[String],
    defaulted: bool,
) -> Result<ResolvedCall<'a>, CoreError> {
    let spec = model_spec(model)?;
    let kind = kind_of(payload)?;
    let outputs = resolve_outputs(model, kind, (!defaulted).then_some(asked))?;

    Ok(ResolvedCall {
        spec,
        kind,
        outputs,
    })
}

/// Check the resolved call in order: shape, task, output names, context
/// cap, then this build's tabular refusal. The caller applies emptiness
/// and adapter preflight only after all these checks succeed.
fn validate_call(
    payloads: &[Value],
    payload: &Value,
    call: &ResolvedCall<'_>,
) -> Result<(), CoreError> {
    let spec = call.spec;
    let kind = call.kind;
    let outputs = &call.outputs;

    // The shapes the request needs (`required_capabilities`), from the
    // payload: its links, its tables, its kind and its anchor.
    let schema = payload.get("schema");
    let linked = matches!(
        schema.and_then(|s| s.get("relationships")),
        Some(Value::Array(links)) if !links.is_empty()
    );
    let tables = 1 + match schema.and_then(|s| s.get("related_tables")) {
        Some(Value::Object(related)) => related.len(),
        _ => 0,
    };
    // Temporal as the reference's `required_capabilities` reads it: the
    // task's options naming an anchor time column, or a forecast or a link
    // prediction (T3.17, where the shapes a refusal names became visible).
    let anchored = payload
        .get("task")
        .and_then(|task| task.get("options"))
        .and_then(|options| options.get("anchor_time_column"))
        .is_some_and(|column| match column {
            Value::Null | Value::Bool(false) => false,
            Value::Text(text) => !text.is_empty(),
            Value::Number(token) => token.parse::<f64>().is_ok_and(|n| n != 0.0),
            Value::Array(items) => !items.is_empty(),
            Value::Object(members) => !members.is_empty(),
            Value::Bool(true) => true,
        });
    validate_shape(spec.id, kind, anchored, linked, tables)?;
    // Preserve the gathered route's existing diagnostic normalization.
    // Direct calls supply the native ordered outputs to this same policy.
    let checked_outputs: Vec<String> = outputs
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    validate_task_outputs(spec.id, kind, &checked_outputs)?;
    let rows = payloads
        .iter()
        .map(|p| rows_of(p, "context"))
        .max()
        .unwrap_or(0);
    if spec.id == RELATIONAL {
        validate_capacity(spec.id, rows)?;
    }
    if spec.id == TABULAR {
        // The tabular model takes no relational request this build sends.
        return Err(not_yet("model"));
    }
    Ok(())
}

fn model_spec(model: &str) -> Result<&'static Spec, CoreError> {
    MODELS
        .into_iter()
        .find(|spec| spec.id == model)
        .ok_or_else(|| {
            CoreError::new(
                Sentence::UnknownModel,
                "the model named is not one the contract carries",
            )
            .with_field("field", Detail::text("model"))
        })
}

/// Resolve a model once, before any later native call observations.
pub fn resolve_model(model: &str) -> Result<&'static str, CoreError> {
    Ok(model_spec(model)?.id)
}

/// Ordered explicit outputs or model-specific defaults, with empty checked now.
pub fn resolve_outputs(
    model: &str,
    kind: &str,
    asked: Option<&[String]>,
) -> Result<Vec<String>, CoreError> {
    let spec = model_spec(model)?;
    let defaults = if spec.id == TABULAR && !matches!(kind, "classification" | "regression") {
        &REGRESSION_FIELDS[..]
    } else {
        default_outputs(kind)
    };
    let outputs = asked.map_or_else(
        || defaults.iter().map(|s| (*s).to_owned()).collect(),
        <[String]>::to_vec,
    );
    if outputs.is_empty() {
        return Err(CoreError::new(
            Sentence::OutputsNone,
            "outputs must name at least one field",
        )
        .with_field("field", Detail::text("outputs")));
    }
    Ok(outputs)
}

/// Shared shape policy after all native capability observations have completed.
pub fn validate_shape(
    model: &str,
    kind: &str,
    anchored: bool,
    linked: bool,
    tables: usize,
) -> Result<(), CoreError> {
    let spec = model_spec(model)?;
    let temporal = anchored || matches!(kind, "forecasting" | "temporal_link_prediction");
    let needed = |shape: &str| match shape {
        "relationships" => linked,
        "multiple_tables" => tables > 1,
        "link_prediction" => kind == "temporal_link_prediction",
        "forecasting" => kind == "forecasting",
        "temporal" => temporal,
        "static" => !temporal,
        _ => false,
    };
    let mut lacking: Vec<&str> = SHAPES
        .into_iter()
        .filter(|shape| needed(shape) && !spec.shapes.contains(shape))
        .collect();
    lacking.sort_unstable();
    if !lacking.is_empty() {
        return Err(unsupported(
            Sentence::ModelLacksShape,
            spec.id,
            "the model cannot answer a request of this shape",
        )
        .with_field("unsupported", texts(lacking)));
    }
    Ok(())
}

/// Shared task then supported-output policy, without reading native row counts.
pub fn validate_task_outputs(model: &str, kind: &str, outputs: &[String]) -> Result<(), CoreError> {
    let spec = model_spec(model)?;
    if !spec.tasks.contains(&kind) {
        return Err(unsupported(
            Sentence::ModelLacksTask,
            spec.id,
            "the model does not serve the task's kind",
        )
        .with_field("task", Detail::text(kind))
        .with_field("supported", texts(sorted(spec.tasks))));
    }
    let mut unknown: Vec<&str> = outputs
        .iter()
        .map(String::as_str)
        .filter(|name| !spec.outputs.contains(name))
        .collect();
    if !unknown.is_empty() {
        unknown.sort_unstable();
        return Err(unsupported(
            Sentence::ModelLacksOutputs,
            spec.id,
            "the model does not produce an output asked for",
        )
        .with_field("unsupported_outputs", texts(unknown))
        .with_field("supported_outputs", texts(sorted(spec.outputs))));
    }
    Ok(())
}

/// The single model table supplies the direct and gathered per-call context cap.
pub fn validate_capacity(model: &str, rows: usize) -> Result<(), CoreError> {
    let spec = model_spec(model)?;
    if rows > spec.context_rows_max {
        return Err(unsupported(
            Sentence::ContextPastModel,
            spec.id,
            "the request's context is larger than the model reads",
        )
        .with_field("context_rows", count(rows))
        .with_field("context_rows_max", count(spec.context_rows_max)));
    }
    Ok(())
}

/// The outputs a task asks for when the caller names none (the adapter's
/// `_DEFAULT_OUTPUTS`): asking for everything would make a regression ask
/// for probabilities, which the endpoint refuses.
pub fn default_outputs(kind: &str) -> &'static [&'static str] {
    match kind {
        "classification" | "binary_classification" | "multiclass_classification" => {
            &CLASSIFICATION_FIELDS
        }
        "temporal_link_prediction" => &["rankings"],
        _ => &REGRESSION_FIELDS,
    }
}

/// Whether the payload asks about no row at all (`Request.nothing_to_score`):
/// its predict half's rows a list, and empty. One that cannot be read is
/// not empty, and goes on to be refused where it is read.
pub fn nothing_to_score(payload: &Value) -> bool {
    instance_table(payload, "predict").is_some_and(|(_, rows)| rows.is_empty())
}

/// A notice the run gives: its sentence and fields, worded by the host: a
/// regression whose every context row has the same answer (the adapter's
/// `_check_target_varies`), an output the caller named that the answer
/// left out (`_report_absent`), a forecast whose steps were explained but
/// not every one (`_warn_of_steps_that_carry_no_explanation`, T2.13).
#[derive(Debug, Clone, PartialEq)]
pub struct Notice {
    pub sentence: Sentence,
    pub message: &'static str,
    pub fields: Vec<(&'static str, Detail)>,
}

/// A JSON value as a member of a set, as Python's `set` holds the values the
/// wire's rows read as: a number by its value (`1` and `1.0` one member, a
/// NaN its own), text, a boolean, null.
fn member(value: &Value) -> String {
    match value {
        Value::Number(token) => match token.parse::<f64>() {
            // 0.0 and -0.0 are one member, as Python's set holds them.
            Ok(n) => format!("n{:016x}", (n + 0.0).to_bits()),
            Err(_) => format!("t{token}"),
        },
        Value::Text(text) => format!("s{text}"),
        Value::Bool(b) => format!("b{b}"),
        Value::Null => "z".to_owned(),
        other => format!("j{}", String::from_utf8_lossy(&other.to_bytes())),
    }
}

/// The context's target column, its values by row, where the context holds
/// the target and at least one row.
fn target_values<'a>(payload: &'a Value, column: Option<&str>) -> Option<Vec<&'a Value>> {
    let (names, rows) = instance_table(payload, "context")?;
    let column = column?;
    let at = names.iter().position(|n| *n == column)?;
    if rows.is_empty() {
        return None;
    }
    Some(
        rows.iter()
            .filter_map(|row| match row {
                Value::Array(cells) => cells.get(at),
                _ => None,
            })
            .collect(),
    )
}

/// The adapter's checks of what the call sends, before it is sent
/// (`_preflight`), in its order: a context with no rows, a classification
/// target with more classes than the model takes, a context whose every
/// answered row has the same answer (a classification refused, a regression
/// noted, [`CONTEXT_CONSTANT`]), then, when an explanation is asked for,
/// the rows scored (exactly one) and the run mode (the one the smaller
/// context selects). `query` is the question's text, which names the
/// target where the wire calls it `TARGET`.
pub fn preflight(
    payloads: &[Value],
    outputs: &[String],
    query: &str,
    model: &str,
) -> Result<Vec<Notice>, CoreError> {
    let mut notices = Vec::new();
    for payload in payloads {
        if let Some((_, rows)) = instance_table(payload, "context")
            && rows.is_empty()
        {
            return Err(
                unsupported(Sentence::NoContextRows, model, "no context rows are left")
                    .with_field("context_rows", count(0)),
            );
        }
    }
    for payload in payloads {
        let kind = kind_of(payload)?;
        if !classifies(kind) {
            continue;
        }
        let column = payload
            .get("task")
            .and_then(|t| t.get("target"))
            .and_then(|t| t.get("column_name"))
            .and_then(Value::as_text);
        if let Some(values) = target_values(payload, column) {
            let labels: BTreeSet<String> = values.iter().map(|v| member(v)).collect();
            if labels.len() > CLASSES_MAX {
                return Err(unsupported(
                    Sentence::TooManyClasses,
                    model,
                    "the target has more classes than the model takes",
                )
                .with_field("classes", count(labels.len()))
                .with_field("limit", count(CLASSES_MAX)));
            }
        }
    }
    for payload in payloads {
        let kind = kind_of(payload)?;
        if !classifies(kind) && kind != "regression" {
            continue;
        }
        let column = payload
            .get("task")
            .and_then(|t| t.get("target"))
            .and_then(|t| t.get("column_name"))
            .and_then(Value::as_text);
        let Some(values) = target_values(payload, column) else {
            continue;
        };
        let answered: BTreeSet<String> = values
            .iter()
            .filter(|v| !matches!(v, Value::Null))
            .map(|v| member(v))
            .collect();
        if answered.len() != 1 {
            continue;
        }
        let column = column.unwrap_or_default();
        let constant = constant_context(column, query);
        if classifies(kind) {
            // Named as the notice names the target, for the host's words
            // (the adapter's `_how_to_name_the_target`).
            let sentence = match constant.sentence {
                Sentence::ContextConstantColumn => Sentence::OneClassColumn,
                Sentence::ContextConstantPredicted => Sentence::OneClassPredicted,
                Sentence::ContextConstantQuestion => Sentence::OneClassQuestion,
                _ => Sentence::OneClassAnswer,
            };
            let mut error = unsupported(sentence, model, "every context row has the same answer")
                .with_field("field", Detail::text("context"))
                .with_field("reason", Detail::text("one_class"));
            for (name, value) in constant.fields {
                error = error.with_field(name, value);
            }
            return Err(error.with_field("classes_in_context", count(1)));
        }
        notices.push(constant);
        break;
    }
    if !outputs.iter().any(|o| o == "explanation") {
        return Ok(notices);
    }
    let scored: usize = payloads.iter().map(|p| rows_of(p, "predict")).sum();
    if scored != EXPLANATION_PREDICT_ROWS {
        // The output and what to change, for the host's words (T2.13).
        return Err(unsupported(
            Sentence::ExplanationOneEntity,
            model,
            "an explanation covers one entity",
        )
        .with_field("field", Detail::text("outputs"))
        .with_field("remedy", Detail::text("one_entity"))
        .with_field("scored", count(scored)));
    }
    if asks_for_the_larger_context(payloads) {
        return Err(unsupported(
            Sentence::ExplanationFastMode,
            model,
            "an explanation is given only in the fast run mode",
        )
        .with_field("field", Detail::text("outputs"))
        .with_field("remedy", Detail::text("smaller_context"))
        .with_field("context_rows_fast", count(CONTEXT_ROWS_FAST)));
    }
    Ok(notices)
}

/// Whether the request selected the mode that does not explain: a run mode
/// other than `fast` on any payload that names one, else more context rows
/// than the fast mode takes (`_asks_for_the_larger_context`).
fn asks_for_the_larger_context(payloads: &[Value]) -> bool {
    let modes: Vec<String> = payloads
        .iter()
        .filter_map(
            |p| match p.get("inference").and_then(|i| i.get("run_mode")) {
                None | Some(Value::Null) => None,
                Some(Value::Text(mode)) => Some(mode.clone()),
                Some(other) => Some(String::from_utf8_lossy(&other.to_bytes()).into_owned()),
            },
        )
        .collect();
    if !modes.is_empty() {
        return modes.iter().any(|m| m != "fast");
    }
    payloads
        .iter()
        .map(|p| rows_of(p, "context"))
        .sum::<usize>()
        > CONTEXT_ROWS_FAST
}

/// The constant-context notice, with how the host names the target
/// (`_how_to_name_the_target`): the target column when the wire names it
/// so (`variant` `column`), else the column the question predicts when its
/// text names one as `PREDICT table.column` (`predicted`), else the
/// question itself (`question`).
fn constant_context(column: &str, query: &str) -> Notice {
    let mut fields = vec![("target", Detail::text(column))];
    let (variant, sentence) = if column != WIRE_TARGET {
        ("column", Sentence::ContextConstantColumn)
    } else if let Some(predicted) = predicted_column(query) {
        fields.push(("predicted", Detail::text(predicted)));
        ("predicted", Sentence::ContextConstantPredicted)
    } else if query.is_empty() {
        ("answer", Sentence::ContextConstantAnswer)
    } else {
        fields.push(("question", Detail::text(query)));
        ("question", Sentence::ContextConstantQuestion)
    };
    fields.push(("variant", Detail::text(variant)));
    Notice {
        sentence,
        message: "every context row has the same answer",
        fields,
    }
}

/// `table.column` when the question opens `PREDICT table.column` followed
/// by white space (the adapter's `_A_PREDICTED_COLUMN`: leading white
/// space, the keyword in any case, then two identifiers, each an ASCII
/// letter or an underscore followed by letters, digits and underscores).
fn predicted_column(query: &str) -> Option<&str> {
    let rest = query.trim_start();
    let keyword = rest.get(..7)?;
    if !keyword.eq_ignore_ascii_case("predict") {
        return None;
    }
    let after = &rest[7..];
    let spaced = after.trim_start();
    if spaced.len() == after.len() {
        return None;
    }
    let ident = |s: &str| -> usize {
        let mut chars = s.char_indices();
        match chars.next() {
            Some((_, c)) if c == '_' || c.is_ascii_alphabetic() => {}
            _ => return 0,
        }
        s.char_indices()
            .find(|(_, c)| !(*c == '_' || c.is_alphanumeric()))
            .map_or(s.len(), |(at, _)| at)
    };
    let table = ident(spaced);
    if table == 0 || !spaced[table..].starts_with('.') {
        return None;
    }
    let column = ident(&spaced[table + 1..]);
    if column == 0 {
        return None;
    }
    let end = table + 1 + column;
    spaced[end..]
        .chars()
        .next()
        .filter(|c| c.is_whitespace())
        .map(|_| &spaced[..end])
}

/// The call one gathered payload makes of `model`, with `request_id`,
/// asking `outputs`; `asked` names the inference options the caller set,
/// the only ones sent, of which one the model does not read is refused
/// before the call, as the adapter refuses it (`_inference`):
/// `UNSUPPORTED_REQUEST` naming the model and those options.
pub fn body(
    payload: &Value,
    model: &str,
    request_id: &str,
    outputs: &[String],
    asked: &[String],
) -> Result<Value, CoreError> {
    let kind = called(payload)?;
    let mut envelope = payload.clone();
    envelope.set("model", Value::text(model));

    // The outputs, and a regression's statistic: the one the payload
    // carries, the median unless one was chosen (`_output_for`).
    let fields = Value::Array(outputs.iter().map(|f| Value::text(f)).collect());
    if kind != "regression" {
        envelope.set(
            "output",
            Value::Object(vec![("fields".to_owned(), fields.clone())]),
        );
    }
    let chosen = match payload.get("output") {
        None | Some(Value::Null) => None,
        // As the adapter's `_output_for` reads it, `chosen or 'median'`:
        // a statistic the payload holds is sent as it is, whatever its
        // type, where Python reads it as true (T2.8: from T1.28b refused
        // by name when not text, `prediction_statistic`).
        Some(output @ Value::Object(_)) => output
            .get("prediction_statistic")
            .filter(|chosen| is_chosen(chosen))
            .cloned(),
        Some(_) => return Err(bad_payload("output")),
    };
    if kind == "regression" {
        envelope.set(
            "output",
            Value::Object(vec![
                ("fields".to_owned(), fields),
                (
                    "prediction_statistic".to_owned(),
                    chosen.unwrap_or_else(|| Value::text(DEFAULT_STATISTIC)),
                ),
            ]),
        );
    }

    // The payload's own metadata, copied, with the call's id.
    let mut metadata = match payload.get("metadata") {
        None | Some(Value::Null) => Value::Object(vec![]),
        Some(object @ Value::Object(_)) => object.clone(),
        Some(_) => return Err(bad_payload("metadata")),
    };
    metadata.set("request_id", Value::text(request_id));
    envelope.set("metadata", metadata);

    // Only the inference options the caller set, which the model reads: one
    // the caller set and the model does not read is refused, and the
    // engine's defaults are left out, the config itself when the caller set
    // none (the reference's 26ecdd28: a deployment applies its own defaults
    // to a request that carries none, and the SDM backends refuse any
    // non-empty `inference_config`).
    match envelope.get("inference") {
        Some(Value::Object(_)) => {
            if let Some(Value::Object(config)) = envelope
                .get("inference")
                .and_then(|i| i.get("inference_config"))
            {
                let mut refused: Vec<&String> = config
                    .iter()
                    .map(|(name, _)| name)
                    .filter(|name| !INFERENCE_OPTIONS.contains(&name.as_str()))
                    .filter(|name| asked.contains(name))
                    .collect();
                refused.sort();
                if !refused.is_empty() {
                    // The option field and what to change, for the host's
                    // words (T2.13): drop the options, and choose a
                    // regression's statistic where `output_type` is one.
                    let mut forwarded = INFERENCE_OPTIONS;
                    forwarded.sort_unstable();
                    let remedy = if refused.iter().any(|n| *n == "output_type") {
                        "drop_or_statistic"
                    } else {
                        "drop"
                    };
                    return Err(CoreError::new(
                        if remedy == "drop" {
                            Sentence::OptionsUnread
                        } else {
                            Sentence::OptionsUnreadStatistic
                        },
                        "the model does not read an inference option the call set",
                    )
                    .with_field("field", Detail::text("inference_config"))
                    .with_field("remedy", Detail::text(remedy))
                    .with_field("model", Detail::text(model))
                    .with_field("unsupported", texts(refused.iter().map(|n| n.as_str())))
                    .with_field("forwarded", texts(forwarded)));
                }
                let sent: Vec<(String, Value)> = config
                    .iter()
                    .filter(|(name, _)| asked.contains(name))
                    .filter(|(name, _)| INFERENCE_OPTIONS.contains(&name.as_str()))
                    .cloned()
                    .collect();
                let mut inference = envelope.get("inference").cloned().unwrap_or(Value::Null);
                if sent.is_empty() {
                    inference.remove("inference_config");
                } else {
                    inference.set("inference_config", Value::Object(sent));
                }
                envelope.set("inference", inference);
            }
        }
        _ => envelope.remove("inference"),
    }
    Ok(envelope)
}

/// Whether a statistic the payload holds is one `_output_for` takes as
/// chosen (`chosen or 'median'`): not null, `false`, a zero, empty text, an
/// empty list or an empty object.
fn is_chosen(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(token) => !token.parse::<f64>().is_ok_and(|n| n == 0.0),
        Value::Text(text) => !text.is_empty(),
        Value::Array(items) => !items.is_empty(),
        Value::Object(members) => !members.is_empty(),
    }
}

/// The bytes of the call `body` writes, as the host sends them, unchanged:
/// refused past the endpoint's limit (`PAYLOAD_BYTES`), counted on these
/// bytes, the envelope included, which are the bytes sent (register PD-2,
/// T1.28a's fix round 1), before the model is called.
pub fn call_bytes(call: &Value) -> Result<Vec<u8>, CoreError> {
    let bytes = call.to_bytes();
    if bytes.len() > crate::payload::PAYLOAD_BYTES {
        return Err(CoreError::new(
            Sentence::CallPastEndpoint,
            "the call is larger than the endpoint takes",
        )
        .with_field("field", Detail::text("call"))
        .with_field("bytes", Detail::Scalar(Scalar::UInt64(bytes.len() as u64)))
        .with_field(
            "limit",
            Detail::Scalar(Scalar::UInt64(crate::payload::PAYLOAD_BYTES as u64)),
        ));
    }
    Ok(bytes)
}

/// The entity a row is about, as the result gives it: as the reference's
/// `as_declared` reads its value back as the key's dtype, or as written
/// where it does not read or no dtype is known.
#[derive(Debug, Clone, PartialEq)]
pub enum Entity {
    Integer(i64),
    Text(String),
    /// A null entity, or an answer that names no id.
    None,
    /// A float key's entity, read as a double.
    Float(f64),
    /// Any other value, as written: a boolean, an integer past int64, a
    /// float where no float is declared, an array or an object.
    Written(Value),
}

/// An answer's prediction as the result gives it: a regression's as an
/// integer or a double, as written (a JSON `null` as `None`), a class as
/// its target's type.
#[derive(Debug, Clone, PartialEq)]
pub enum Answer {
    None,
    Float(f64),
    Bool(bool),
    Int(i64),
    Text(String),
    /// A class's prediction written as a non-finite number, which the
    /// reference's `as_declared` leaves as it is: no value of the target's
    /// type, the item's own the number.
    NonFinite(f64),
    /// A direct regression value or a class that does not read as the target's
    /// type: the decoded item's own value, as written.
    Written(Value),
}

/// The type a part's predictions are given back as: a regression's
/// number (`Float`), or a class as its target's type, a float target's as
/// a double (`FloatClass`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnswerType {
    Float,
    FloatClass,
    Bool,
    Int,
    Text,
}

/// One answer laid out.
#[derive(Debug, Clone, PartialEq)]
pub struct Row {
    pub row_index: i64,
    /// The answer's forecast step, which only an answer given as written
    /// (a payload with no entity-reference column) may carry.
    pub forecast_step: Option<i64>,
    /// Exact direct-only step where the native item exceeds the Arrow width.
    pub forecast_step_as_answered: Option<String>,
    /// The answer's place in the endpoint's order (across a merge's parts,
    /// the parts in order), which the result's items keep.
    pub answer_index: i64,
    pub id: Entity,
    pub prediction: Answer,
    /// Each class's probability, keyed by the class as the target column
    /// spells it, in the order the answer wrote them.
    pub probabilities: Option<Vec<(String, f64)>>,
    /// The item's scores and rankings, each a mapping as a JSON object, in
    /// the order written (T2.8).
    pub scores: Option<Vec<f64>>,
    pub rankings: Option<Vec<Value>>,
    /// The item's explanation, a mapping as `sdm_wire::response::mapping`
    /// lays one out (T2.13).
    pub explanation: Option<Value>,
    /// The item's own metadata, a mapping as the explanation (T2.13).
    pub metadata: Option<Value>,
    /// Each quantile's value, keyed as the answer wrote it, in the order
    /// written, a mapping as the explanation (T2.9).
    pub quantiles: Option<Value>,
    /// The answer's embedding, in the order written (T2.11).
    pub embeddings: Option<Vec<f64>>,
}

/// The answer to one call, or the parts of one request put together.
#[derive(Debug, Clone, PartialEq)]
pub struct Part {
    pub model: String,
    pub request_id: String,
    /// The endpoint's `metadata` object.
    pub metadata: Value,
    /// The answers, in the endpoint's order.
    pub rows: Vec<Row>,
    pub answers: AnswerType,
    /// Whether an entity's key is text (else an integer), for the result's
    /// `id` column where no answer's entity gives its type.
    pub text_ids: bool,
    /// The result's explanation (`PredictionResult.explanation`, T2.13):
    /// the first answer's in the endpoint's order that holds a mapping not
    /// empty (`normalize`), the first part's that has one across a merge's
    /// parts, and for a forecast `{"by_forecast_step": {"<step>":
    /// explanation, ...}}`, each step that has one in step order, keyed by
    /// the step's number in decimal digits, as JSON keys a mapping
    /// (`_by_step`); none where no answer explains.
    pub explanation: Option<Value>,
}

/// Whether a mapping as `sdm_wire::response::mapping` lays one out is one
/// Python takes as true: not empty.
fn holds(mapping: &Value) -> bool {
    match mapping {
        Value::Object(members) => !members.is_empty(),
        Value::Array(pairs) => !pairs.is_empty(),
        _ => true,
    }
}

/// The answer to a request with no rows, made without a call
/// (`_empty_answer`): no rows, no request id, empty metadata.
pub fn empty(model: &str, payload: &Value) -> Part {
    Part {
        model: model.to_owned(),
        request_id: String::new(),
        metadata: Value::Object(vec![]),
        rows: vec![],
        answers: answer_type(payload).unwrap_or(AnswerType::Float),
        text_ids: entity_dtype(payload).as_deref() == Some("string"),
        explanation: None,
    }
}

/// The target's dtype as the payload's task names it.
fn target_dtype(payload: &Value) -> Option<&str> {
    payload.get("task")?.get("target")?.get("dtype")?.as_text()
}

/// How a payload's answers are given back (`as_declared_target`): a
/// regression's as doubles, a classification's as its target's dtype, a
/// boolean, an integer, a double or, for any other dtype, the class's
/// text.
fn answer_type(payload: &Value) -> Result<AnswerType, CoreError> {
    let kind = kind_of(payload)?;
    if kind != "classification" {
        called(payload)?;
    }
    Ok(declared_answers(payload, kind))
}

fn declared_answers(payload: &Value, kind: &str) -> AnswerType {
    if !classifies(kind) {
        return AnswerType::Float;
    }
    match target_dtype(payload) {
        Some(d) if d.starts_with("bool") => AnswerType::Bool,
        Some(d) if d.starts_with("int") => AnswerType::Int,
        Some(d) if d.starts_with("float") => AnswerType::FloatClass,
        _ => AnswerType::Text,
    }
}

/// The spellings a declared boolean is read back from (`_as_bool`).
const TRUE: [&str; 3] = ["true", "t", "1"];
const FALSE: [&str; 3] = ["false", "f", "0"];

/// A class read back as the target's boolean (`_as_bool`): a boolean, text
/// that spells one (trimmed, any case), or the integer 0 or 1.
fn as_bool(value: &Value) -> Option<bool> {
    match value {
        Value::Bool(b) => Some(*b),
        Value::Text(text) => {
            let text = text.trim().to_lowercase();
            if TRUE.contains(&text.as_str()) {
                Some(true)
            } else if FALSE.contains(&text.as_str()) {
                Some(false)
            } else {
                None
            }
        }
        Value::Number(token) if is_integer_token(token) => match token.parse::<i64>() {
            Ok(0) => Some(false),
            Ok(1) => Some(true),
            _ => None,
        },
        _ => None,
    }
}

/// Direct Boolean classes use the native strip set, including U+001C..U+001F.
/// Only the six ASCII spellings can match, so ASCII-insensitive comparison
/// preserves native lowercasing without a Unicode casing profile.
fn direct_bool_class(value: &Value) -> Option<bool> {
    let Value::Text(text) = value else {
        return as_bool(value);
    };
    let text = text.trim_matches(sdm_engine::words::space);
    if TRUE.iter().any(|token| text.eq_ignore_ascii_case(token)) {
        Some(true)
    } else if FALSE.iter().any(|token| text.eq_ignore_ascii_case(token)) {
        Some(false)
    } else {
        None
    }
}

/// A class read back as the target's integer (`as_declared`'s `int`): text
/// of an integer (trimmed), an integer, or a float that is whole.
fn as_int(value: &Value) -> Option<i64> {
    match value {
        Value::Text(text) => text.trim().parse::<i64>().ok(),
        Value::Number(token) if is_integer_token(token) => token.parse::<i64>().ok(),
        Value::Number(token) => {
            let float: f64 = token.parse().ok()?;
            (float.is_finite() && float.fract() == 0.0 && float.abs() < 9.2e18)
                .then_some(float as i64)
        }
        _ => None,
    }
}

/// Direct integer classes retain the existing exact JSON carrier past i64.
/// The shared entity conversion already implements finite whole-float casts;
/// text first observes its reached native process limit before zero stripping.
fn direct_integer_class(value: &Value, decimal_limit: u32) -> Option<String> {
    if let Value::Text(text) = value {
        return integer_text_limited(text, decimal_limit);
    }
    match entity(value, Some("int64")) {
        Entity::Integer(value) => Some(value.to_string()),
        Entity::Written(Value::Number(token)) if is_integer_token(&token) => Some(token),
        _ => None,
    }
}

/// A class read back as the target's double (`as_declared`'s `float`):
/// text of a number (trimmed; an infinity or a NaN by name), or a number,
/// an integer past a double's range not one.
fn as_float(value: &Value) -> Option<f64> {
    match value {
        Value::Text(text) => text.trim().parse::<f64>().ok(),
        Value::Number(token) => token
            .parse::<f64>()
            .ok()
            .filter(|v| v.is_finite() || !is_integer_token(token)),
        _ => None,
    }
}

/// Direct float classes use the native text grammar. Numeric carriers share
/// the established conversion; gathered text retains its approved Rust parser.
fn direct_float_class(value: &Value) -> Option<f64> {
    match value {
        Value::Text(text) => numeric_text(text),
        _ => as_float(value),
    }
}

/// One answer's prediction as the result gives it.
fn answer_of(
    prediction: Option<&Value>,
    answers: AnswerType,
    policy: AnswerPolicy,
) -> Result<Answer, CoreError> {
    let Some(value) = prediction else {
        return Ok(Answer::None);
    };
    // The direct generated decoder accepts prediction as data, and the
    // native adapter returns a nonclassification answer unchanged. Gathered
    // regression retains its separately established numeric admission below.
    if matches!(policy, AnswerPolicy::Direct { .. }) && answers == AnswerType::Float {
        return Ok(written(value));
    }
    match answers {
        // A gathered regression's prediction is a number (owner decision of
        // 2026-10-12, relayed): a token past a double's range, a `NaN` and
        // an infinity read as the reference reads them; an integer past
        // int64, text, a boolean, an array or an object refused.
        AnswerType::Float => match value {
            // An integer is kept as the integer it is (the reference's
            // reader gives a Python int); one past int64 is refused.
            Value::Number(token) if is_integer_token(token) => token
                .parse::<i64>()
                .map(Answer::Int)
                .map_err(|_| bad_response("prediction.prediction", "not_a_number")),
            Value::Number(token) => token
                .parse::<f64>()
                .map(Answer::Float)
                .map_err(|_| bad_response("prediction.prediction", "not_a_number")),
            _ => Err(bad_response("prediction.prediction", "not_a_number")),
        },
        // A float target's class is read back as a double, a non-finite one
        // too, which the target's type holds.
        AnswerType::FloatClass => {
            let read = match policy {
                AnswerPolicy::Gathered => as_float(value),
                AnswerPolicy::Direct { .. } => direct_float_class(value),
            };
            Ok(read.map_or_else(|| written(value), Answer::Float))
        }
        // A class written as a non-finite number is passed through as the
        // reference passes it (owner decision of 2026-10-12): no target's
        // type holds it, so `as_declared` leaves it as it is.
        _ if non_finite(value).is_some() => {
            Ok(Answer::NonFinite(non_finite(value).unwrap_or(f64::NAN)))
        }
        AnswerType::Bool => {
            let read = match policy {
                AnswerPolicy::Gathered => as_bool(value),
                AnswerPolicy::Direct { .. } => direct_bool_class(value),
            };
            Ok(read.map_or_else(|| written(value), Answer::Bool))
        }
        AnswerType::Int => Ok(match policy.decimal_limit() {
            Some(limit) => direct_integer_class(value, limit).map_or_else(
                || written(value),
                |digits| match digits.parse::<i64>() {
                    Ok(value) => Answer::Int(value),
                    Err(_) => Answer::Written(Value::Number(digits)),
                },
            ),
            None => as_int(value).map_or_else(|| written(value), Answer::Int),
        }),
        AnswerType::Text => Ok(match value {
            Value::Text(text) => Answer::Text(text.clone()),
            other => written(other),
        }),
    }
}

/// An unchanged direct regression or an unconverted class prediction:
/// a null, which an item's reader takes as none, none.
fn written(value: &Value) -> Answer {
    match value {
        Value::Null => Answer::None,
        other => Answer::Written(other.clone()),
    }
}

/// The number a token stands for when it is not finite: `NaN`,
/// `Infinity`, `-Infinity`, or a float token past a double's range.
fn non_finite(value: &Value) -> Option<f64> {
    match value {
        Value::Number(token) if !is_integer_token(token) => {
            token.parse::<f64>().ok().filter(|v| !v.is_finite())
        }
        _ => None,
    }
}

/// The probabilities keyed by each class as the target column spells it
/// (`_as_the_column_spells_them`), left as written where reading them back
/// would make two classes one; a float target's class as `float_text`
/// spells the double it reads as.
fn spelled(
    probabilities: &[(String, f64)],
    answers: AnswerType,
    float_text: fn(f64) -> String,
) -> Vec<(String, f64)> {
    spelled_with_limit(probabilities, answers, float_text, None)
}

fn spelled_with_limit(
    probabilities: &[(String, f64)],
    answers: AnswerType,
    float_text: fn(f64) -> String,
    decimal_limit: Option<u32>,
) -> Vec<(String, f64)> {
    let respelled: Vec<(String, f64)> = probabilities
        .iter()
        .map(|(class, p)| {
            let as_written = Value::Text(class.clone());
            let key = match answers {
                AnswerType::Bool => {
                    let read = if decimal_limit.is_some() {
                        direct_bool_class(&as_written)
                    } else {
                        as_bool(&as_written)
                    };
                    read.map(|b| {
                        if b {
                            "True".to_owned()
                        } else {
                            "False".to_owned()
                        }
                    })
                }
                AnswerType::Int => match decimal_limit {
                    Some(limit) => direct_integer_class(&as_written, limit),
                    None => as_int(&as_written).map(|n| n.to_string()),
                },
                AnswerType::FloatClass => {
                    let read = if decimal_limit.is_some() {
                        direct_float_class(&as_written)
                    } else {
                        as_float(&as_written)
                    };
                    read.map(float_text)
                }
                _ => None,
            };
            (key.unwrap_or_else(|| class.clone()), *p)
        })
        .collect();
    let distinct: BTreeSet<&str> = respelled.iter().map(|(k, _)| k.as_str()).collect();
    if distinct.len() == probabilities.len() {
        respelled
    } else {
        probabilities.to_vec()
    }
}

/// The answer to one payload laid out (`normalize`, `_named_entities`,
/// `as_declared_target`), its prediction given back as [`answer_of`] reads
/// it. Where the payload's predicted instance table has a column named with
/// the entity-reference prefix (the first such, by position), each answer
/// is checked against the row it claims before it is given the entity of
/// that row, read as [`entity`] reads it (in range, not a forecast's, once,
/// under the instance id the row was asked under, every row answered).
/// Where it has none, each answer is given back as written, as the
/// reference gives it: its id the instance id it was answered under, as
/// text, and no row checked (a `row_index` or forecast step past int64,
/// which the result's columns do not hold, refused as `INVALID_RESPONSE`,
/// register P1B-86). On both paths its probabilities are keyed as
/// [`spelled`] keys them, a float target's class by `float_text`.
pub fn layout(
    payload: &Value,
    response: &PredictionResponse,
    float_text: fn(f64) -> String,
) -> Result<Part, CoreError> {
    layout_as(
        payload,
        response,
        float_text,
        answer_type(payload)?,
        Attribution::GatheredReference,
        AnswerPolicy::Gathered,
    )
}

/// Direct tabular answers follow the later wire task, without rerunning
/// gathered admission after a source warning changed that task.
pub(crate) fn layout_tabular(
    payload: &Value,
    response: &PredictionResponse,
    decimal_limit: u32,
    float_text: fn(f64) -> String,
) -> Result<Part, CoreError> {
    layout_as(
        payload,
        response,
        float_text,
        declared_answers(payload, kind_of(payload)?),
        Attribution::AsWritten,
        AnswerPolicy::Direct { decimal_limit },
    )
}

/// Attribution is chosen by the owning adapter, independently of answer coercion.
/// Ordinary direct feature names do not acquire gathered entity-reference semantics.
#[derive(Clone, Copy)]
enum Attribution {
    GatheredReference,
    AsWritten,
}

/// The owning adapter's prediction admission, separate from row attribution.
/// Only direct final text coercion observes the fresh native decimal limit.
#[derive(Clone, Copy)]
enum AnswerPolicy {
    Gathered,
    Direct { decimal_limit: u32 },
}

impl AnswerPolicy {
    fn decimal_limit(self) -> Option<u32> {
        match self {
            Self::Gathered => None,
            Self::Direct { decimal_limit } => Some(decimal_limit),
        }
    }
}

fn layout_as(
    payload: &Value,
    response: &PredictionResponse,
    float_text: fn(f64) -> String,
    answers: AnswerType,
    attribution: Attribution,
    policy: AnswerPolicy,
) -> Result<Part, CoreError> {
    let (names, rows) = instance_table(payload, "predict").ok_or_else(|| bad_payload("predict"))?;
    let reference = match attribution {
        Attribution::GatheredReference => names.iter().position(|n| n.starts_with(ENTITY_REF)),
        Attribution::AsWritten => None,
    };
    let (placed, text_ids) = match reference {
        Some(reference) => {
            let dtype = key_dtype(payload, names[reference]);
            let text_ids = dtype.as_deref() == Some("string");
            (
                named(
                    response,
                    &names,
                    rows,
                    reference,
                    dtype.as_deref(),
                    float_text,
                )?,
                text_ids,
            )
        }
        None => (
            as_written(response, matches!(attribution, Attribution::AsWritten))?,
            true,
        ),
    };

    let mut laid = Vec::with_capacity(placed.len());
    for (item, (row_index, id, forecast_step)) in response.predictions.iter().zip(placed) {
        if let Some(field) = item.carried.first() {
            return Err(not_yet("response_field").with_field("field", Detail::text(*field)));
        }
        laid.push(Row {
            row_index,
            forecast_step,
            forecast_step_as_answered: matches!(attribution, Attribution::AsWritten)
                .then(|| item.forecast_step_written.clone())
                .flatten()
                .filter(|_| forecast_step.is_none()),
            answer_index: i64::try_from(laid.len()).unwrap_or(i64::MAX),
            id,
            prediction: answer_of(item.prediction.as_ref(), answers, policy)?,
            probabilities: item
                .probabilities
                .as_ref()
                .map(|given| match policy.decimal_limit() {
                    Some(limit) => spelled_with_limit(given, answers, float_text, Some(limit)),
                    None => spelled(given, answers, float_text),
                }),
            scores: item.scores.clone(),
            rankings: item.rankings.clone(),
            explanation: item.explanation.clone(),
            metadata: item.metadata.clone(),
            quantiles: item.quantiles.clone(),
            embeddings: item.embeddings.clone(),
        });
    }
    let explanation = response
        .predictions
        .iter()
        .filter_map(|item| item.explanation.as_ref())
        .find(|e| holds(e))
        .cloned();
    Ok(Part {
        model: response.model.clone(),
        request_id: response.id.clone(),
        metadata: response.metadata.clone(),
        rows: laid,
        answers,
        text_ids,
        explanation,
    })
}

/// Each answer's row, entity and forecast step, where the payload names
/// its entities in the column at `reference` (`_named_entities`).
fn named(
    response: &PredictionResponse,
    names: &[&str],
    rows: &[Value],
    reference: usize,
    dtype: Option<&str>,
    float_text: fn(f64) -> String,
) -> Result<Vec<(i64, Entity, Option<i64>)>, CoreError> {
    let instance = names.iter().position(|n| *n == INSTANCE_ID);
    let cell = |row: &Value, at: usize| -> Result<Value, CoreError> {
        match row {
            Value::Array(cells) => cells.get(at).cloned().ok_or_else(|| bad_payload("predict")),
            _ => Err(bad_payload("predict")),
        }
    };
    let entities = rows
        .iter()
        .map(|row| Ok(entity(&cell(row, reference)?, dtype)))
        .collect::<Result<Vec<_>, CoreError>>()?;
    let instances = match instance {
        Some(at) => Some(
            rows.iter()
                .map(|row| Ok(instance_text(&cell(row, at)?, float_text)))
                .collect::<Result<Vec<_>, CoreError>>()?,
        ),
        None => None,
    };

    let mut answered = vec![false; entities.len()];
    let mut named = Vec::with_capacity(response.predictions.len());
    for item in &response.predictions {
        let index = item
            .row_index
            .as_usize()
            .filter(|i| *i < entities.len())
            .ok_or_else(|| bad_response("prediction.row_index", "out_of_range"))?;
        if item.forecast_step.is_some() {
            return Err(bad_response("prediction.forecast_step", "not_a_forecast"));
        }
        if answered[index] {
            return Err(bad_response("prediction.row_index", "repeated"));
        }
        answered[index] = true;
        if let Some(instances) = &instances {
            let Some(id) = &item.id else {
                return Err(bad_response("prediction.id", "missing"));
            };
            if instances[index].as_deref() != Some(id.as_str()) {
                return Err(bad_response("prediction.id", "not_the_row_s"));
            }
        }
        named.push((
            i64::try_from(index).unwrap_or(i64::MAX),
            entities[index].clone(),
            None,
        ));
    }
    if answered.iter().any(|a| !a) {
        return Err(bad_response("prediction.row_index", "unanswered"));
    }
    Ok(named)
}

/// Each answer's row, id and forecast step as the endpoint wrote them,
/// where the payload names no entity (`_named_entities` with no reference
/// column): no row is checked, and the id is the instance id the answer
/// names, as text, or none.
fn as_written(
    response: &PredictionResponse,
    direct: bool,
) -> Result<Vec<(i64, Entity, Option<i64>)>, CoreError> {
    response
        .predictions
        .iter()
        .map(|item| {
            let row_index = item
                .row_index
                .as_i64()
                .ok_or_else(|| bad_response("prediction.row_index", "out_of_range"))?;
            let forecast_step = if direct {
                item.forecast_step
                    .and_then(|value| i64::try_from(value).ok())
            } else {
                item.forecast_step
                    .map(i64::try_from)
                    .transpose()
                    .map_err(|_| bad_response("prediction.forecast_step", "out_of_range"))?
            };
            let id = item.id.clone().map_or(Entity::None, Entity::Text);
            Ok((row_index, id, forecast_step))
        })
        .collect()
}

/// The answers of one request sent in several calls, put together
/// (`PredictionResult.merge`): none refused (`INTERNAL_ERROR`, as the
/// reference's merge refuses nothing to merge), one as it is; several, each
/// answering its own rows, are one: every part's answers in order, the
/// first part's model and request id, its metadata with `batches` (how many
/// parts) and `batch_request_ids` (each part's id), parts of different
/// models refused (`INTERNAL_ERROR`).
pub fn merge(parts: Vec<Part>) -> Result<Part, CoreError> {
    let internal = |message: &str| CoreError::new(Sentence::Internal, message);
    let mut parts = parts.into_iter();
    let Some(first) = parts.next() else {
        return Err(internal("no parts to merge"));
    };
    let rest: Vec<Part> = parts.collect();
    if rest.is_empty() {
        return Ok(first);
    }
    if rest.iter().any(|p| p.model != first.model) {
        return Err(internal("the parts were answered by different models"));
    }
    let mut metadata = first.metadata.clone();
    metadata.set(
        "batches",
        Value::int(i64::try_from(rest.len() + 1).unwrap_or(i64::MAX)),
    );
    let ids: Vec<&str> = std::iter::once(first.request_id.as_str())
        .chain(rest.iter().map(|p| p.request_id.as_str()))
        .collect();
    metadata.set("batch_request_ids", Value::texts(ids));
    let mut explanation = first.explanation;
    let mut rows = first.rows;
    for part in rest {
        rows.extend(part.rows);
        if explanation.is_none() {
            explanation = part.explanation;
        }
    }
    for (at, row) in rows.iter_mut().enumerate() {
        row.answer_index = i64::try_from(at).unwrap_or(i64::MAX);
    }
    Ok(Part {
        model: first.model,
        request_id: first.request_id,
        metadata,
        rows,
        answers: first.answers,
        text_ids: first.text_ids,
        explanation,
    })
}

/// A forecast's answers put together, each step's answer to its own
/// regression (`_by_step`, T1.36 fix 21): every step's answers in step
/// order, each in the endpoint's order and carrying its step, the first
/// step's model, request id and answer type, and no metadata (the
/// client's merged result names none); no step refused (`INTERNAL_ERROR`).
pub fn by_step(steps: Vec<(u32, Part)>) -> Result<Part, CoreError> {
    let mut steps = steps.into_iter();
    let Some((step, first)) = steps.next() else {
        return Err(CoreError::new(
            Sentence::Internal,
            "a forecast with no step to put together",
        ));
    };
    let mut rows = Vec::new();
    let mut explained = Vec::new();
    let (model, request_id, answers, text_ids) =
        (first.model, first.request_id, first.answers, first.text_ids);
    for (step, part, explanation) in std::iter::once((step, first.rows, first.explanation))
        .chain(steps.map(|(s, p)| (s, p.rows, p.explanation)))
    {
        rows.extend(part.into_iter().map(|row| Row {
            forecast_step: Some(i64::from(step)),
            ..row
        }));
        if let Some(explanation) = explanation {
            explained.push((step.to_string(), explanation));
        }
    }
    for (at, row) in rows.iter_mut().enumerate() {
        row.answer_index = i64::try_from(at).unwrap_or(i64::MAX);
    }
    Ok(Part {
        model,
        request_id,
        metadata: Value::Object(vec![]),
        rows,
        answers,
        text_ids,
        explanation: (!explained.is_empty()).then(|| {
            Value::Object(vec![(
                "by_forecast_step".to_owned(),
                Value::Object(explained),
            )])
        }),
    })
}

/// The outputs the caller named that the answer does not carry
/// (`PredictionResult.absent`), each once, in the order named: none for an
/// answer with no rows; `prediction` when no answer holds one,
/// `probabilities`, `quantiles` or `embeddings` when none holds any,
/// `explanation` when the result has none ([`Part::explanation`]), and the
/// fields this build refuses an answer for carrying always; a name the
/// result does not know is taken as carried.
pub fn absent(part: &Part, named: &[String]) -> Vec<String> {
    if part.rows.is_empty() {
        return vec![];
    }
    let mut seen = BTreeSet::new();
    named
        .iter()
        .filter(|name| seen.insert(name.as_str()))
        .filter(|name| match name.as_str() {
            "prediction" => part.rows.iter().all(|r| r.prediction == Answer::None),
            "probabilities" => part
                .rows
                .iter()
                .all(|r| r.probabilities.as_ref().is_none_or(Vec::is_empty)),
            "scores" => part
                .rows
                .iter()
                .all(|r| r.scores.as_ref().is_none_or(Vec::is_empty)),
            "rankings" => part
                .rows
                .iter()
                .all(|r| r.rankings.as_ref().is_none_or(Vec::is_empty)),
            "quantiles" => part.rows.iter().all(|r| {
                r.quantiles
                    .as_ref()
                    .is_none_or(|q| matches!(q, Value::Object(m) if m.is_empty()))
            }),
            "explanation" => part.explanation.is_none(),
            "embeddings" => part
                .rows
                .iter()
                .all(|r| r.embeddings.as_ref().is_none_or(Vec::is_empty)),
            _ => false,
        })
        .cloned()
        .collect()
}

/// The outputs [`absent`] looks for, in the order [`left_out`] names them.
pub const LOOKED_FOR: [&str; 7] = [
    "prediction",
    "probabilities",
    "scores",
    "rankings",
    "quantiles",
    "explanation",
    "embeddings",
];

/// Every output of [`LOOKED_FOR`] the answer does not carry, as [`absent`]
/// reads each: what the result says it left out, for a host to read a
/// caller's `absent(...)` off rather than keep a copy of the rule (T3.12
/// fix 1).
pub fn left_out(part: &Part) -> Vec<String> {
    absent(part, &LOOKED_FOR.map(String::from))
}

/// The notice of the outputs [`absent`] found, naming the model asked.
pub fn absent_notice(model: &str, absent: &[String]) -> Notice {
    Notice {
        sentence: Sentence::OutputsAbsent,
        message: "the answer left out outputs the request asked for",
        fields: vec![
            ("model", Detail::text(model)),
            ("absent", texts(absent.iter().map(String::as_str))),
        ],
    }
}

/// A forecast's steps the caller asked to be explained (an `explanation`
/// among the outputs it named) that came back with none, while another
/// step came back with one (`_warn_of_steps_that_carry_no_explanation`):
/// the notice naming the model, how many steps were explained of how many,
/// and the steps with none, in step order; `None` where every step or none
/// was explained, or none was asked for.
pub fn steps_unexplained(model: &str, named: &[String], steps: &[(u32, &Part)]) -> Option<Notice> {
    if !named.iter().any(|n| n == "explanation") {
        return None;
    }
    let missing: Vec<u32> = steps
        .iter()
        .filter(|(_, part)| part.explanation.is_none())
        .map(|(step, _)| *step)
        .collect();
    if missing.is_empty() || missing.len() == steps.len() {
        return None;
    }
    Some(Notice {
        sentence: Sentence::StepsUnexplained,
        message: "a forecast's steps were explained, but not every one",
        fields: vec![
            ("model", Detail::text(model)),
            ("explained", count(steps.len() - missing.len())),
            ("steps", count(steps.len())),
            (
                "missing",
                Detail::List(missing.into_iter().map(Scalar::UInt32).collect()),
            ),
        ],
    })
}

/// The result of a part: one row per answer, ordered by `row_index` (as
/// `PredictionResult.to_pandas` orders them; the order is stable among
/// equal ones, which a merge of parts holds), with `row_index` (int64),
/// `id` (int64 where every id is an integer, utf8 where every one is text,
/// the part's key type deciding where none is either; a missing one null;
/// where they are mixed, or one is of another type, utf8 with
/// `id_as_written`, each id as the JSON text of the reference's item value,
/// or null), `forecast_step` when any answer given as written carries one
/// (int64, null where it has none), `prediction` when any answer holds one (a regression's as pandas infers it: int64 where
/// every answer is an integer, else float64, an integer as its double; a
/// class as bool, int64 or utf8; a missing one as null, a NaN as NaN; with
/// `prediction_as_answered`, each answer's prediction as the JSON text of
/// the reference's item value, or null, where the column changed one: an
/// integer widened, a class written as a non-finite number) and
/// `probabilities` when any answer holds them (a map of each class, as its
/// column spells it, to its probability, in the order written),
/// `quantiles`, `explanation` and `metadata` when any answer carries one
/// (utf8, the JSON text of each answer's mapping, null where it had none;
/// T2.9 and T2.13), and last
/// `answer_index` (int64, the answer's place in the endpoint's order, which
/// the result's items keep, as the reference's do). The result's own
/// explanation ([`Part::explanation`]), where it has one, is the JSON text
/// on the schema's metadata under `explanation`.
pub fn table(part: &Part) -> Result<RecordBatch, CoreError> {
    let mut rows: Vec<&Row> = part.rows.iter().collect();
    rows.sort_by_key(|r| r.row_index);
    let row_index: ArrayRef = Arc::new(Int64Array::from_iter_values(
        rows.iter().map(|r| r.row_index),
    ));
    let ids: Vec<&Entity> = rows.iter().map(|r| &r.id).collect();
    let (id_fields, id_columns) = id_columns(&ids, part.text_ids);
    let mut fields = vec![Field::new("row_index", DataType::Int64, false)];
    fields.extend(id_fields);
    let mut columns = vec![row_index];
    columns.extend(id_columns);
    if rows.iter().any(|r| r.forecast_step.is_some()) {
        fields.push(Field::new("forecast_step", DataType::Int64, true));
        columns.push(Arc::new(Int64Array::from_iter(
            rows.iter().map(|r| r.forecast_step),
        )));
    }
    if rows.iter().any(|r| r.forecast_step_as_answered.is_some()) {
        fields.push(Field::new(
            "forecast_step_as_answered",
            DataType::Utf8,
            true,
        ));
        columns.push(Arc::new(StringArray::from_iter(
            rows.iter().map(|r| r.forecast_step_as_answered.as_deref()),
        )));
    }
    if rows.iter().any(|r| r.prediction != Answer::None) {
        // Classes given back as written, not of the target's type, all of
        // one kind with the rest (T2.8): the column as pandas infers it
        // from the values, as the reference's frame holds them.
        let as_written = as_written_column(&rows);
        let (data_type, column): (DataType, ArrayRef) = match (&as_written, part.answers) {
            (Some((data_type, column)), _) => (data_type.clone(), Arc::clone(column)),
            (None, answers) => match answers {
                // A regression's column as pandas infers it from the answers:
                // int64 where every answer is an integer, else float64, an
                // integer as the double it reads as; a missing one is null.
                AnswerType::Float if regression_integers(&rows) => (
                    DataType::Int64,
                    Arc::new(Int64Array::from_iter(rows.iter().map(
                        |r| match r.prediction {
                            Answer::Int(n) => Some(n),
                            _ => None,
                        },
                    ))),
                ),
                AnswerType::Float => (
                    DataType::Float64,
                    Arc::new(Float64Array::from_iter(rows.iter().map(
                        |r| match r.prediction {
                            Answer::Float(v) => Some(v),
                            #[allow(
                                clippy::cast_precision_loss,
                                reason = "as the reference's frame reads an integer among floats"
                            )]
                            Answer::Int(n) => Some(n as f64),
                            _ => None,
                        },
                    ))),
                ),
                // A float target's classes as doubles.
                AnswerType::FloatClass => (
                    DataType::Float64,
                    Arc::new(Float64Array::from_iter(rows.iter().map(
                        |r| match r.prediction {
                            Answer::Float(v) => Some(v),
                            _ => None,
                        },
                    ))),
                ),
                AnswerType::Bool => (
                    DataType::Boolean,
                    Arc::new(BooleanArray::from_iter(rows.iter().map(
                        |r| match r.prediction {
                            Answer::Bool(b) => Some(b),
                            _ => None,
                        },
                    ))),
                ),
                AnswerType::Int => (
                    DataType::Int64,
                    Arc::new(Int64Array::from_iter(rows.iter().map(
                        |r| match r.prediction {
                            Answer::Int(n) => Some(n),
                            _ => None,
                        },
                    ))),
                ),
                AnswerType::Text => (
                    DataType::Utf8,
                    Arc::new(StringArray::from_iter(rows.iter().map(
                        |r| match &r.prediction {
                            Answer::Text(t) => Some(t.clone()),
                            _ => None,
                        },
                    ))),
                ),
            },
        };
        let integer = |answer: &Answer| match answer {
            Answer::Int(_) => true,
            Answer::Written(Value::Number(token)) => is_integer_token(token),
            _ => false,
        };
        let widened = (data_type == DataType::Float64
            && rows.iter().any(|r| integer(&r.prediction)))
            || rows
                .iter()
                .any(|r| matches!(r.prediction, Answer::NonFinite(_)))
            || (as_written.is_none()
                && rows
                    .iter()
                    .any(|r| matches!(r.prediction, Answer::Written(_))));
        fields.push(Field::new("prediction", data_type, true));
        columns.push(column);
        // Where the column does not hold each answer as the reference's
        // items do (a regression's integer widened to a double; a class
        // written as a non-finite number, which no target's type holds),
        // each answer's prediction as the JSON text of the item's value,
        // null where it had none.
        if widened {
            fields.push(Field::new("prediction_as_answered", DataType::Utf8, true));
            columns.push(Arc::new(StringArray::from_iter(
                rows.iter().map(|r| as_answered(&r.prediction)),
            )));
        }
    }
    if rows.iter().any(|r| r.probabilities.is_some()) {
        let names = MapFieldNames {
            entry: "entries".to_owned(),
            key: "class".to_owned(),
            value: "probability".to_owned(),
        };
        let mut builder = MapBuilder::new(Some(names), StringBuilder::new(), Float64Builder::new());
        for row in &rows {
            // An empty mapping is one, as the reference's item holds `{}`.
            match &row.probabilities {
                Some(given) => {
                    for (class, p) in given {
                        builder.keys().append_value(class);
                        builder.values().append_value(*p);
                    }
                    builder.append(true).map_err(|_| internal_layout())?;
                }
                _ => builder.append(false).map_err(|_| internal_layout())?,
            }
        }
        let column = builder.finish();
        fields.push(Field::new(
            "probabilities",
            column.data_type().clone(),
            true,
        ));
        columns.push(Arc::new(column));
    }
    // An item's scores and rankings (T2.8), each where an answer carries
    // them: a list of doubles, and a list of each mapping's JSON text.
    if rows.iter().any(|r| r.scores.is_some()) {
        let mut builder = ListBuilder::new(Float64Builder::new());
        for row in &rows {
            match &row.scores {
                Some(scores) => {
                    builder.values().append_slice(scores);
                    builder.append(true);
                }
                None => builder.append(false),
            }
        }
        let column = builder.finish();
        fields.push(Field::new("scores", column.data_type().clone(), true));
        columns.push(Arc::new(column));
    }
    if rows.iter().any(|r| r.rankings.is_some()) {
        let mut builder = ListBuilder::new(StringBuilder::new());
        for row in &rows {
            match &row.rankings {
                Some(rankings) => {
                    for mapping in rankings {
                        builder
                            .values()
                            .append_value(String::from_utf8_lossy(&mapping.to_bytes()));
                    }
                    builder.append(true);
                }
                None => builder.append(false),
            }
        }
        let column = builder.finish();
        fields.push(Field::new("rankings", column.data_type().clone(), true));
        columns.push(Arc::new(column));
    }
    // Each answer's quantiles (T2.9), explanation and own metadata (T2.13),
    // where any answer carries one: the JSON text of the mapping, null where
    // it had none.
    for (name, of) in [
        (
            "quantiles",
            (|r: &Row| r.quantiles.as_ref()) as fn(&Row) -> Option<&Value>,
        ),
        ("explanation", |r: &Row| r.explanation.as_ref()),
        ("metadata", |r: &Row| r.metadata.as_ref()),
    ] {
        if rows.iter().any(|r| of(r).is_some()) {
            fields.push(Field::new(name, DataType::Utf8, true));
            columns.push(Arc::new(StringArray::from_iter(rows.iter().map(|r| {
                of(r).map(|v| String::from_utf8_lossy(&v.to_bytes()).into_owned())
            }))));
        }
    }
    // Each answer's embedding, a list of float64, null where it has none
    // (`_SEQUENCE_FIELDS`, T2.11).
    if rows.iter().any(|r| r.embeddings.is_some()) {
        let mut builder = ListBuilder::new(Float64Builder::new());
        for row in &rows {
            match &row.embeddings {
                Some(given) => {
                    builder.values().append_slice(given);
                    builder.append(true);
                }
                None => builder.append(false),
            }
        }
        let column = builder.finish();
        fields.push(Field::new("embeddings", column.data_type().clone(), true));
        columns.push(Arc::new(column));
    }
    fields.push(Field::new("answer_index", DataType::Int64, false));
    columns.push(Arc::new(Int64Array::from_iter_values(
        rows.iter().map(|r| r.answer_index),
    )));
    // The result's explanation (T2.13), as the JSON text of
    // [`Part::explanation`], on the schema's metadata under `explanation`.
    let mut schema = Schema::new(fields);
    if let Some(explanation) = &part.explanation {
        schema = schema.with_metadata(
            [(
                "explanation".to_owned(),
                String::from_utf8_lossy(&explanation.to_bytes()).into_owned(),
            )]
            .into(),
        );
    }
    RecordBatch::try_new(Arc::new(schema), columns).map_err(|_| internal_layout())
}

/// The result's `id` column, and `id_as_written` where it is needed, of
/// `ids` in the result's order: int64 where every id is an integer, utf8
/// where every one is text, `text_ids` (the key's type) deciding where none
/// is either; a missing one null; where they are mixed, or one is of
/// another type, utf8 with `id_as_written`, each id as the JSON text of the
/// reference's item value, or null.
pub(crate) fn id_columns(ids: &[&Entity], text_ids: bool) -> (Vec<Field>, Vec<ArrayRef>) {
    let integers = ids
        .iter()
        .all(|r| matches!(**r, Entity::Integer(_) | Entity::None));
    let texts = ids
        .iter()
        .all(|r| matches!(**r, Entity::Text(_) | Entity::None));
    let (id_type, column): (DataType, ArrayRef) = if integers && !(texts && text_ids) {
        (
            DataType::Int64,
            Arc::new(Int64Array::from_iter(ids.iter().map(|r| match &**r {
                Entity::Integer(n) => Some(*n),
                _ => None,
            }))),
        )
    } else {
        (
            DataType::Utf8,
            Arc::new(StringArray::from_iter(ids.iter().map(|r| match &**r {
                Entity::Integer(n) => Some(n.to_string()),
                Entity::Text(text) => Some(text.clone()),
                other => id_as_written(other),
            }))),
        )
    };
    let mut fields = vec![Field::new("id", id_type, true)];
    let mut columns = vec![column];
    // Where the ids are neither all integers nor all text (a null aside),
    // or one is of another type, each id as the JSON text of the
    // reference's item value, null where it had none.
    if !integers && !texts {
        fields.push(Field::new("id_as_written", DataType::Utf8, true));
        columns.push(Arc::new(StringArray::from_iter(
            ids.iter().map(|r| id_as_written(r)),
        )));
    }
    (fields, columns)
}

/// The prediction column where an answer was given back as written, not
/// of the target's type (T2.8), and every answer is of one kind with it, as
/// pandas infers the reference's frame's column from the items' values: a
/// boolean column, an int64 column of integers, a float64 column of
/// numbers (an integer as its double), or a text column; a missing one
/// null. None where no answer was given back so, or the answers are of
/// several kinds, the column then the target's type's, with each answer's
/// JSON text beside it.
fn as_written_column(rows: &[&Row]) -> Option<(DataType, ArrayRef)> {
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Kind {
        Bool,
        Int,
        Float,
        Text,
    }
    if !rows
        .iter()
        .any(|r| matches!(r.prediction, Answer::Written(_)))
    {
        return None;
    }
    let kind_of = |answer: &Answer| -> Option<Option<Kind>> {
        Some(match answer {
            Answer::None => None,
            Answer::Bool(_) | Answer::Written(Value::Bool(_)) => Some(Kind::Bool),
            Answer::Int(_) => Some(Kind::Int),
            Answer::Written(Value::Number(token)) if is_integer_token(token) => {
                token.parse::<i64>().ok()?;
                Some(Kind::Int)
            }
            Answer::Float(_) | Answer::NonFinite(_) | Answer::Written(Value::Number(_)) => {
                Some(Kind::Float)
            }
            Answer::Text(_) | Answer::Written(Value::Text(_)) => Some(Kind::Text),
            Answer::Written(_) => return None,
        })
    };
    let mut kinds = BTreeSet::new();
    for row in rows {
        if let Some(kind) = kind_of(&row.prediction)? {
            kinds.insert(kind as u8);
        }
    }
    let kind = match kinds.into_iter().collect::<Vec<_>>().as_slice() {
        [one] => *one,
        [a, b] if *a == Kind::Int as u8 && *b == Kind::Float as u8 => Kind::Float as u8,
        _ => return None,
    };
    let float = |answer: &Answer| -> Option<f64> {
        match answer {
            Answer::Float(v) | Answer::NonFinite(v) => Some(*v),
            #[allow(
                clippy::cast_precision_loss,
                reason = "an integer widened as the frame widens it"
            )]
            Answer::Int(n) => Some(*n as f64),
            Answer::Written(Value::Number(token)) => token.parse().ok(),
            _ => None,
        }
    };
    Some(if kind == Kind::Bool as u8 {
        (
            DataType::Boolean,
            Arc::new(BooleanArray::from_iter(rows.iter().map(
                |r| match &r.prediction {
                    Answer::Bool(b) | Answer::Written(Value::Bool(b)) => Some(*b),
                    _ => None,
                },
            ))),
        )
    } else if kind == Kind::Int as u8 {
        (
            DataType::Int64,
            Arc::new(Int64Array::from_iter(rows.iter().map(
                |r| match &r.prediction {
                    Answer::Int(n) => Some(*n),
                    Answer::Written(Value::Number(token)) => token.parse().ok(),
                    _ => None,
                },
            ))),
        )
    } else if kind == Kind::Float as u8 {
        (
            DataType::Float64,
            Arc::new(Float64Array::from_iter(
                rows.iter().map(|r| float(&r.prediction)),
            )),
        )
    } else {
        (
            DataType::Utf8,
            Arc::new(StringArray::from_iter(rows.iter().map(
                |r| match &r.prediction {
                    Answer::Text(t) | Answer::Written(Value::Text(t)) => Some(t.clone()),
                    _ => None,
                },
            ))),
        )
    })
}

/// An answer's prediction as the JSON text of the reference's item value,
/// a non-finite number as the token Python's reader takes; `None` where it
/// had none.
fn as_answered(answer: &Answer) -> Option<String> {
    match answer {
        Answer::None => None,
        Answer::Float(v) | Answer::NonFinite(v) => float_text(*v),
        Answer::Int(n) => Some(n.to_string()),
        Answer::Bool(b) => Some(b.to_string()),
        Answer::Text(text) => {
            Some(String::from_utf8_lossy(&Value::text(text).to_bytes()).into_owned())
        }
        Answer::Written(value) => Some(String::from_utf8_lossy(&value.to_bytes()).into_owned()),
    }
}

/// A double as JSON text, a non-finite one as the token Python's reader
/// takes.
fn float_text(v: f64) -> Option<String> {
    if v.is_nan() {
        Some("NaN".to_owned())
    } else if v.is_infinite() {
        Some(if v > 0.0 { "Infinity" } else { "-Infinity" }.to_owned())
    } else {
        Value::float(v).map(|f| String::from_utf8_lossy(&f.to_bytes()).into_owned())
    }
}

/// An entity as the JSON text of the reference's item value, a non-finite
/// float as the token Python's reader takes; `None` where it had none.
fn id_as_written(id: &Entity) -> Option<String> {
    match id {
        Entity::None => None,
        Entity::Integer(n) => Some(n.to_string()),
        Entity::Float(v) => float_text(*v),
        Entity::Text(text) => {
            Some(String::from_utf8_lossy(&Value::text(text).to_bytes()).into_owned())
        }
        Entity::Written(value) => Some(String::from_utf8_lossy(&value.to_bytes()).into_owned()),
    }
}

/// Whether every answer of a regression holds an integer, none missing:
/// the column pandas infers as int64.
fn regression_integers(rows: &[&Row]) -> bool {
    rows.iter().all(|r| matches!(r.prediction, Answer::Int(_)))
}

fn internal_layout() -> CoreError {
    CoreError::new(Sentence::Internal, "the result does not lay out")
}

/// The dtype of the column the entity reference stands for: the column
/// of the table the reference's relationship points at, when it points at
/// one column.
fn key_dtype(payload: &Value, reference: &str) -> Option<String> {
    let schema = payload.get("schema")?;
    let Some(Value::Array(links)) = schema.get("relationships") else {
        return None;
    };
    let link = links.iter().find(|link| {
        matches!(link.get("source_columns"), Some(Value::Array(c))
            if c.len() == 1 && c[0].as_text() == Some(reference))
    })?;
    let target = link.get("target_table")?.as_text()?;
    let Some(Value::Array(columns)) = link.get("target_columns") else {
        return None;
    };
    let [column] = columns.as_slice() else {
        return None;
    };
    schema
        .get("related_tables")?
        .get(target)?
        .get("columns")?
        .get(column.as_text()?)?
        .get("dtype")?
        .as_text()
        .map(str::to_owned)
}

/// The dtype of the payload's entity key, as [`key_dtype`] finds it.
fn entity_dtype(payload: &Value) -> Option<String> {
    let (names, _) = instance_table(payload, "predict")?;
    let reference = names.iter().find(|n| n.starts_with(ENTITY_REF))?;
    key_dtype(payload, reference)
}

/// The entity of a row: its reference's value read back as the key's
/// dtype, as the reference's `as_declared` reads it, or as written where it
/// does not read or no dtype is known: a null as none; a declared boolean
/// from a boolean, the text spelling one or the integer 0 or 1
/// ([`as_bool`]); a declared integer (any other boolean as written) from an
/// integer, a whole float, or text Python's `int` reads ([`integer_text`]);
/// a declared float from a number not past a double's range, or text
/// Python's `float` reads ([`numeric_text`]); any other dtype as written.
pub(crate) fn entity(value: &Value, dtype: Option<&str>) -> Entity {
    let read = match (dtype, value) {
        (None, _) | (_, Value::Null) => None,
        (Some(d), _) if d.starts_with("bool") => {
            as_bool(value).map(|b| Entity::Written(Value::Bool(b)))
        }
        (_, Value::Bool(_)) => None,
        (Some(d), _) if d.starts_with("int") => match value {
            Value::Text(text) => integer_text(text).map(|digits| integer(&digits)),
            Value::Number(token) if is_integer_token(token) => Some(integer(token)),
            Value::Number(token) => token
                .parse::<f64>()
                .ok()
                .filter(|v| v.is_finite() && v.fract() == 0.0)
                .map(|v| integer(&format!("{v:.0}"))),
            _ => None,
        },
        (Some(d), _) if d.starts_with("float") => match value {
            Value::Text(text) => numeric_text(text).map(Entity::Float),
            Value::Number(token) => token
                .parse::<f64>()
                .ok()
                .filter(|v| v.is_finite())
                .map(Entity::Float),
            _ => None,
        },
        _ => None,
    };
    read.unwrap_or_else(|| match value {
        Value::Null => Entity::None,
        Value::Text(text) => Entity::Text(text.clone()),
        Value::Number(token) if is_integer_token(token) => integer(token),
        other => Entity::Written(other.clone()),
    })
}

/// An integer's digits as an entity: an int64, or past it as written.
fn integer(digits: &str) -> Entity {
    digits.parse::<i64>().map_or_else(
        |_| Entity::Written(Value::Number(digits.to_owned())),
        Entity::Integer,
    )
}

/// A row's instance id as the reference compares it with the id an answer
/// names (`str(instances[index])`, T2.8): an integer's digits (`-0` as
/// `0`), the text, a float as the parser's `float_text` writes it, a
/// boolean as `True` or `False` and a null as `None`; none for a list or an
/// object, which no id an answer names matches.
fn instance_text(value: &Value, float_text: fn(f64) -> String) -> Option<String> {
    Some(match value {
        Value::Text(text) => text.clone(),
        Value::Null => "None".to_owned(),
        Value::Bool(true) => "True".to_owned(),
        Value::Bool(false) => "False".to_owned(),
        Value::Number(token) if is_integer_token(token) => {
            let digits = token.strip_prefix('-').unwrap_or(token);
            if digits.bytes().all(|b| b == b'0') {
                "0".to_owned()
            } else {
                token.clone()
            }
        }
        Value::Number(token) => float_text(token.parse::<f64>().unwrap_or(f64::NAN)),
        Value::Array(_) | Value::Object(_) => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use sdm_contract::ErrorCode;

    #[test]
    fn every_model_is_described_as_the_client_describes_it() {
        let all = capabilities();
        let models: Vec<_> = all.iter().map(|c| c.model).collect();
        assert_eq!(models, [RELATIONAL, TABULAR]);
        let relational = &all[0];
        assert_eq!(
            relational.tasks,
            [
                "binary_classification",
                "classification",
                "forecasting",
                "multiclass_classification",
                "regression",
                "temporal_link_prediction",
            ]
        );
        assert!(relational.single_table && relational.multiple_tables && relational.relationships);
        assert!(relational.sessions && relational.explanation_formats.is_empty());
        let tabular = &all[1];
        assert_eq!(tabular.tasks, ["classification", "regression"]);
        assert_eq!(
            tabular.outputs,
            ["prediction", "probabilities", "quantiles"]
        );
        assert!(tabular.single_table && !tabular.multiple_tables && !tabular.relationships);
        for each in &all {
            let mut sorted = each.outputs.clone();
            sorted.sort_unstable();
            assert_eq!(each.outputs, sorted);
        }
    }

    #[test]
    fn the_predicted_column_is_read_as_the_adapter_reads_it() {
        assert_eq!(
            predicted_column("PREDICT users.age FOR EACH users.id"),
            Some("users.age")
        );
        assert_eq!(predicted_column("  predict u.a\tFOR"), Some("u.a"));
        assert_eq!(
            predicted_column("PREDICT COUNT(orders.*, 0, 1, days) > 0 FOR"),
            None
        );
        assert_eq!(predicted_column("PREDICT users.age"), None);
        assert_eq!(predicted_column("PREDICTusers.age FOR"), None);
        assert_eq!(predicted_column("PREDICT 1a.b FOR"), None);
    }

    #[test]
    fn an_entity_is_read_back_as_as_declared_reads_it() {
        let number = |t: &str| Value::Number(t.to_owned());
        let text = |t: &str| Value::Text(t.to_owned());
        let int = Some("int64");
        assert_eq!(entity(&number("7"), int), Entity::Integer(7));
        assert_eq!(entity(&text(" +0_07 "), int), Entity::Integer(7));
        assert_eq!(entity(&text("\u{665}"), int), Entity::Integer(5));
        assert_eq!(entity(&number("5.0"), int), Entity::Integer(5));
        assert_eq!(entity(&number("5.5"), int), Entity::Written(number("5.5")));
        assert_eq!(entity(&text("abc"), int), Entity::Text("abc".into()));
        assert_eq!(
            entity(&Value::Bool(true), int),
            Entity::Written(Value::Bool(true))
        );
        assert_eq!(entity(&Value::Null, int), Entity::None);
        assert_eq!(
            entity(&text("1180591620717411303424"), int),
            Entity::Written(number("1180591620717411303424"))
        );
        assert_eq!(entity(&number("7"), Some("float64")), Entity::Float(7.0));
        assert_eq!(
            entity(&text("1"), Some("bool")),
            Entity::Written(Value::Bool(true))
        );
        // No dtype known, or a text key: as written.
        assert_eq!(entity(&text("7"), None), Entity::Text("7".into()));
        assert_eq!(entity(&number("7"), Some("string")), Entity::Integer(7));
    }

    #[test]
    fn a_float_target_s_class_reads_back_as_a_double() {
        let number = |t: &str| Value::Number(t.to_owned());
        let text = |t: &str| Value::Text(t.to_owned());
        assert_eq!(as_float(&text(" 3.0 ")), Some(3.0));
        assert_eq!(
            as_float(&text("0.10000000149011612")),
            Some(f64::from(0.1_f32))
        );
        assert_eq!(as_float(&text("-inf")), Some(f64::NEG_INFINITY));
        assert!(as_float(&text("NaN")).is_some_and(f64::is_nan));
        assert_eq!(as_float(&number("3")), Some(3.0));
        assert_eq!(as_float(&number("1e400")), Some(f64::INFINITY));
        // Text that is no number, a boolean, and an integer past a double's
        // range are not a double.
        assert_eq!(as_float(&text("three")), None);
        assert_eq!(as_float(&Value::Bool(true)), None);
        assert_eq!(as_float(&number(&"9".repeat(400))), None);
        assert_eq!(
            answer_of(
                Some(&text("3.0")),
                AnswerType::FloatClass,
                AnswerPolicy::Gathered
            ),
            Ok(Answer::Float(3.0))
        );
        // One that does not read is given back as written (T2.8).
        assert_eq!(
            answer_of(
                Some(&text("three")),
                AnswerType::FloatClass,
                AnswerPolicy::Gathered
            ),
            Ok(Answer::Written(text("three")))
        );
    }

    #[test]
    fn direct_boolean_classes_strip_native_spaces_without_changing_gathered_policy() {
        let spell = |value: f64| value.to_string();
        let whitespace = "\t\n\u{b}\u{c}\r\u{1c}\u{1d}\u{1e}\u{1f} \u{85}\u{a0}\u{1680}\u{2000}\u{2001}\u{2002}\u{2003}\u{2004}\u{2005}\u{2006}\u{2007}\u{2008}\u{2009}\u{200a}\u{2028}\u{2029}\u{202f}\u{205f}\u{3000}";
        for separator in whitespace.chars() {
            for (label, expected) in [
                ("True", true),
                ("t", true),
                ("1", true),
                ("FALSE", false),
                ("F", false),
                ("0", false),
            ] {
                let padded = format!("{separator}{label}{separator}");
                let value = Value::Text(padded.clone());
                assert_eq!(
                    answer_of(
                        Some(&value),
                        AnswerType::Bool,
                        AnswerPolicy::Direct {
                            decimal_limit: 4300
                        }
                    ),
                    Ok(Answer::Bool(expected))
                );
                let gathered = if ('\u{1c}'..='\u{1f}').contains(&separator) {
                    written(&value)
                } else {
                    Answer::Bool(expected)
                };
                assert_eq!(
                    answer_of(Some(&value), AnswerType::Bool, AnswerPolicy::Gathered),
                    Ok(gathered)
                );
                let canonical = if expected { "True" } else { "False" };
                let given = vec![(padded.clone(), 0.75)];
                assert_eq!(
                    spelled_with_limit(&given, AnswerType::Bool, spell, Some(4300)),
                    vec![(canonical.into(), 0.75)]
                );
                let gathered = if ('\u{1c}'..='\u{1f}').contains(&separator) {
                    given.clone()
                } else {
                    vec![(canonical.into(), 0.75)]
                };
                assert_eq!(spelled(&given, AnswerType::Bool, spell), gathered);
                let colliding = vec![(padded, 0.75), (canonical.into(), 0.25)];
                assert_eq!(
                    spelled_with_limit(&colliding, AnswerType::Bool, spell, Some(4300)),
                    colliding
                );
            }
        }
    }

    #[test]
    fn a_float_target_s_probabilities_are_keyed_as_its_classes_are_spelled() {
        let spell = |x: f64| format!("<{x}>");
        let given = vec![("3".to_owned(), 0.5), ("x".to_owned(), 0.5)];
        assert_eq!(
            spelled(&given, AnswerType::FloatClass, spell),
            vec![("<3>".to_owned(), 0.5), ("x".to_owned(), 0.5)]
        );
        // Two keys that read as one double are left as written.
        let same = vec![("3".to_owned(), 0.5), ("3.0".to_owned(), 0.5)];
        assert_eq!(spelled(&same, AnswerType::FloatClass, spell), same);
    }

    /// A temporal link prediction's answer, as the endpoint writes one: each
    /// item ranks candidates under `rankings`, a list of objects, which the
    /// reference's bindings read (`_mapping_tuple`) and its result keeps as
    /// a list in one row (`PredictionResult.to_pandas`, `row['rankings']`).
    /// T4.5 consumes the existing ranking decoder for the actual link
    /// task, preserving each item's mapping order. An unknown task kind
    /// still refuses by name rather than borrowing another task's rules.
    #[test]
    fn a_link_prediction_s_answer_keeps_rankings_and_unknown_tasks_stay_refused() {
        let read = |text: &str| sdm_wire::json::read(text.as_bytes()).unwrap();
        let answer = read(
            r#"{"id": "r1", "model": "kumo-relational", "metadata": {},
                "predictions": [
                  {"row_index": 0, "id": "0", "prediction": null,
                   "rankings": [{"item": 3, "score": 0.9}, {"item": 1, "score": 0.4}]},
                  {"row_index": 1, "id": "1",
                   "rankings": [{"item": 2, "score": 0.7}]}]}"#,
        );
        let response = PredictionResponse::read(&answer).unwrap();
        assert_eq!(response.predictions.len(), 2);
        let written: Vec<usize> = response
            .predictions
            .iter()
            .map(|item| item.rankings.as_ref().map_or(0, Vec::len))
            .collect();
        assert_eq!(written, [2, 1]);
        let payload = |kind: &str| {
            read(&format!(
                r#"{{"task": {{"kind": "{kind}", "target": {{"dtype": "float64"}}}},
                    "predict": {{"instance_table": {{"columns": ["instance_id"],
                                                    "rows": [[0], [1]]}}}}}}"#
            ))
        };
        let spell = |x: f64| x.to_string();
        let linked = layout(&payload("temporal_link_prediction"), &response, spell).unwrap();
        assert_eq!(linked.rows[0].rankings, response.predictions[0].rankings);
        assert_eq!(linked.rows[1].rankings, response.predictions[1].rankings);
        let unknown = layout(&payload("static_link_prediction"), &response, spell).unwrap_err();
        assert_eq!(unknown.code, ErrorCode::UnsupportedRequest);
        assert_eq!(unknown.fields["rule"], Detail::text("problem_type"));
        let asked = layout(&payload("regression"), &response, spell).unwrap();
        assert_eq!(asked.rows.len(), 2);
        assert_eq!(asked.rows[0].rankings, response.predictions[0].rankings);
        assert_eq!(asked.rows[1].rankings, response.predictions[1].rankings);
        assert_eq!(asked.rows[0].prediction, Answer::None);
    }

    /// T3.12 fix 1 (Codex P0-3): every output the result looks for that
    /// the answer left out, in the order [`LOOKED_FOR`] names them, as
    /// [`absent`] reads each, for a host to read off the result rather
    /// than keep a copy of the rule: an empty quantiles mapping on every
    /// row is left out; an answer of no rows leaves out nothing.
    #[test]
    fn every_output_an_answer_leaves_out_is_read_once_for_the_hosts() {
        let read = |text: &str| sdm_wire::json::read(text.as_bytes()).unwrap();
        let payload = read(
            r#"{"task": {"kind": "regression", "target": {"dtype": "float64"}},
                "predict": {"instance_table": {"columns": ["instance_id"],
                                              "rows": [[0], [1]]}}}"#,
        );
        let spell = |x: f64| x.to_string();
        let answer = read(
            r#"{"id": "r1", "model": "kumo-relational", "metadata": {},
                "predictions": [
                  {"row_index": 0, "id": "0", "prediction": 1.0, "quantiles": {}},
                  {"row_index": 1, "id": "1", "prediction": 2.0, "quantiles": {}}]}"#,
        );
        let part = layout(&payload, &PredictionResponse::read(&answer).unwrap(), spell).unwrap();
        assert_eq!(
            left_out(&part),
            [
                "probabilities",
                "scores",
                "rankings",
                "quantiles",
                "explanation",
                "embeddings"
            ]
        );
        let named: Vec<String> = ["quantiles", "prediction", "quantiles", "no_such_output"]
            .map(String::from)
            .to_vec();
        assert_eq!(absent(&part, &named), ["quantiles"]);
        let none =
            read(r#"{"id": "r1", "model": "kumo-relational", "metadata": {}, "predictions": []}"#);
        let empty = read(
            r#"{"task": {"kind": "regression", "target": {"dtype": "float64"}},
                "predict": {"instance_table": {"columns": ["instance_id"], "rows": []}}}"#,
        );
        let part = layout(&empty, &PredictionResponse::read(&none).unwrap(), spell).unwrap();
        assert!(left_out(&part).is_empty());
    }
}
