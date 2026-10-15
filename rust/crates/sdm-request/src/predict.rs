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
//! `embeddings`, `quantiles`, `explanation` or `metadata`, a classification
//! answer that is not of its target's type, and the tabular model.

use std::collections::BTreeSet;
use std::sync::Arc;

use arrow_array::builder::{Float64Builder, MapBuilder, MapFieldNames, StringBuilder};
use arrow_array::{
    Array, ArrayRef, BooleanArray, Float64Array, Int64Array, RecordBatch, StringArray,
};
use arrow_schema::{DataType, Field, Schema};
use sdm_contract::{CoreError, Detail, ErrorCode, ErrorKind, Scalar};
use sdm_wire::json::{Value, is_integer_token};
use sdm_wire::response::{PredictionResponse, Refused, integer_text, numeric_text};

/// The model this build sends a relational request to.
pub const RELATIONAL: &str = "kumo-relational";

/// The other model the contract carries, which takes no relational request.
pub const TABULAR: &str = "kumo-tabular";

/// The output a regression asks for when the caller names none.
pub const REGRESSION_FIELDS: [&str; 1] = ["prediction"];

/// The outputs a classification asks for when the caller names none.
pub const CLASSIFICATION_FIELDS: [&str; 2] = ["prediction", "probabilities"];

/// The task kinds whose call this build writes.
const CALLED: [&str; 3] = [
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

/// The notice of a regression whose every context row has the same answer
/// (the adapter's `_check_target_varies`), worded by the host.
pub const CONTEXT_CONSTANT: &str = "CONTEXT_CONSTANT";

/// The notice of an output the caller named that the answer left out
/// (`_report_absent`), worded by the host.
pub const OUTPUTS_ABSENT: &str = "OUTPUTS_ABSENT";

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
/// its profile takes, its task kinds and its outputs.
struct Spec {
    id: &'static str,
    shapes: &'static [&'static str],
    tasks: &'static [&'static str],
    outputs: &'static [&'static str],
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
};

/// `KUMO_TABULAR`: one table, static, no link or forecast.
const KUMO_TABULAR: Spec = Spec {
    id: TABULAR,
    shapes: &["static"],
    tasks: &["classification", "regression"],
    outputs: &["prediction", "probabilities", "quantiles"],
};

/// The refusal of a `predict` input that takes a rule this build does not
/// port yet, naming the rule, never a value (decision H17).
pub fn not_yet(rule: &'static str) -> CoreError {
    CoreError::new(
        ErrorKind::Unsupported,
        ErrorCode::UnsupportedRequest,
        "predicting this request takes a rule this core does not hold yet",
    )
    .with_field("operation", Detail::text("predict"))
    .with_field("rule", Detail::text(rule))
}

/// The model cannot answer the request, as the client refuses it before
/// anything is sent (`UnsupportedRequestError`): the model named, never a
/// value of the request.
fn unsupported(model: &str, message: &str) -> CoreError {
    CoreError::new(
        ErrorKind::Unsupported,
        ErrorCode::UnsupportedRequest,
        message,
    )
    .with_field("model", Detail::text(model))
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
        ErrorKind::InvalidRequest,
        ErrorCode::InvalidRequest,
        "the request state's payload is not one materialize writes",
    )
    .with_field("field", Detail::text("state.payload"))
    .with_field("reason", Detail::text(reason))
}

/// The endpoint's answer, not one the contract describes or not the
/// answer to this request: the field and why, never a value.
pub fn bad_response(field: &'static str, reason: &'static str) -> CoreError {
    CoreError::new(
        ErrorKind::ModelEndpoint,
        ErrorCode::InvalidResponse,
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

/// The outputs asked of `payload` sent to `model`, as the client's
/// `_prepare` resolves and checks them: the model the contract carries
/// (`UNKNOWN_MODEL` otherwise), the outputs the caller named or, when it
/// named none (`defaulted`), the task's default ones, an empty list refused
/// (`INVALID_REQUEST`, field `outputs`), then `check_request_against_model`
/// in its order: the shapes the model's profile does not take, a task kind
/// it does not serve, an output it does not produce and a context past what
/// it reads, each `UNSUPPORTED_REQUEST` naming the model.
pub fn prepare(
    payload: &Value,
    model: &str,
    asked: &[String],
    defaulted: bool,
) -> Result<Vec<String>, CoreError> {
    let spec = match model {
        RELATIONAL => &KUMO_RELATIONAL,
        TABULAR => &KUMO_TABULAR,
        _ => {
            return Err(CoreError::new(
                ErrorKind::NotFound,
                ErrorCode::UnknownModel,
                "the model named is not one the contract carries",
            )
            .with_field("field", Detail::text("model")));
        }
    };
    let kind = kind_of(payload)?;
    let outputs: Vec<String> = if defaulted {
        default_outputs(kind)
            .iter()
            .map(|f| (*f).to_owned())
            .collect()
    } else {
        asked.to_vec()
    };
    if outputs.is_empty() {
        return Err(CoreError::new(
            ErrorKind::InvalidRequest,
            ErrorCode::InvalidRequest,
            "outputs must name at least one field",
        )
        .with_field("field", Detail::text("outputs")));
    }

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
    let anchored = instance_table(payload, "predict")
        .is_some_and(|(names, _)| names.contains(&"__kumo_anchor_time"));
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
        return Err(
            unsupported(spec.id, "the model cannot answer a request of this shape")
                .with_field("unsupported", texts(lacking)),
        );
    }
    if !spec.tasks.contains(&kind) {
        return Err(
            unsupported(spec.id, "the model does not serve the task's kind")
                .with_field("task", Detail::text(kind))
                .with_field("supported", texts(spec.tasks.iter().copied())),
        );
    }
    let mut unknown: Vec<&str> = outputs
        .iter()
        .map(String::as_str)
        .filter(|name| !spec.outputs.contains(name))
        .collect();
    if !unknown.is_empty() {
        unknown.sort_unstable();
        unknown.dedup();
        return Err(
            unsupported(spec.id, "the model does not produce an output asked for")
                .with_field("unsupported_outputs", texts(unknown))
                .with_field("supported_outputs", texts(spec.outputs.iter().copied())),
        );
    }
    let rows = rows_of(payload, "context");
    if spec.id == RELATIONAL && rows > CONTEXT_ROWS_MAX {
        return Err(unsupported(
            spec.id,
            "the request's context is larger than the model reads",
        )
        .with_field("context_rows", count(rows))
        .with_field("context_rows_max", count(CONTEXT_ROWS_MAX)));
    }
    if spec.id == TABULAR {
        // The tabular model takes no relational request this build sends.
        return Err(not_yet("model"));
    }
    Ok(outputs)
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

/// A notice the run gives: its code and fields, worded by the host.
#[derive(Debug, Clone, PartialEq)]
pub struct Notice {
    pub code: &'static str,
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
                unsupported(model, "no context rows are left").with_field("context_rows", count(0))
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
                return Err(
                    unsupported(model, "the target has more classes than the model takes")
                        .with_field("classes", count(labels.len()))
                        .with_field("limit", count(CLASSES_MAX)),
                );
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
        if classifies(kind) {
            return Err(unsupported(model, "every context row has the same answer")
                .with_field("target", Detail::text(column))
                .with_field("classes_in_context", count(1)));
        }
        notices.push(constant_context(column, query));
        break;
    }
    if !outputs.iter().any(|o| o == "explanation") {
        return Ok(notices);
    }
    let scored: usize = payloads.iter().map(|p| rows_of(p, "predict")).sum();
    if scored != EXPLANATION_PREDICT_ROWS {
        return Err(unsupported(model, "an explanation covers one entity")
            .with_field("scored", count(scored)));
    }
    if asks_for_the_larger_context(payloads) {
        return Err(
            unsupported(model, "an explanation is given only in the fast run mode")
                .with_field("context_rows_fast", count(CONTEXT_ROWS_FAST)),
        );
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
    let variant = if column != WIRE_TARGET {
        "column"
    } else if let Some(predicted) = predicted_column(query) {
        fields.push(("predicted", Detail::text(predicted)));
        "predicted"
    } else if query.is_empty() {
        "answer"
    } else {
        fields.push(("question", Detail::text(query)));
        "question"
    };
    fields.push(("variant", Detail::text(variant)));
    Notice {
        code: CONTEXT_CONSTANT,
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
/// of which one the model does not read is refused before the call, as
/// the adapter refuses it (`_inference`): `UNSUPPORTED_REQUEST` naming the
/// model and those options.
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
        Some(output @ Value::Object(_)) => match output.get("prediction_statistic") {
            None | Some(Value::Null) => None,
            Some(Value::Text(text)) if text.is_empty() => None,
            Some(Value::Text(text)) => Some(text.clone()),
            Some(_) => return Err(not_yet("prediction_statistic")),
        },
        Some(_) => return Err(bad_payload("output")),
    };
    if kind == "regression" {
        envelope.set(
            "output",
            Value::Object(vec![
                ("fields".to_owned(), fields),
                (
                    "prediction_statistic".to_owned(),
                    Value::Text(chosen.unwrap_or_else(|| DEFAULT_STATISTIC.to_owned())),
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

    // Only the inference options the model reads: one the caller set and
    // the model does not read is refused; the engine's defaults of the
    // others are left out.
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
                    return Err(CoreError::new(
                        ErrorKind::Unsupported,
                        ErrorCode::UnsupportedRequest,
                        "the model does not read an inference option the call set",
                    )
                    .with_field("model", Detail::text(model))
                    .with_field("unsupported", texts(refused.iter().map(|n| n.as_str()))));
                }
                let trimmed = Value::Object(
                    config
                        .iter()
                        .filter(|(name, _)| INFERENCE_OPTIONS.contains(&name.as_str()))
                        .cloned()
                        .collect(),
                );
                let mut inference = envelope.get("inference").cloned().unwrap_or(Value::Null);
                inference.set("inference_config", trimmed);
                envelope.set("inference", inference);
            }
        }
        _ => envelope.remove("inference"),
    }
    Ok(envelope)
}

/// The bytes of the call `body` writes, as the host sends them, unchanged:
/// refused past the endpoint's limit (`PAYLOAD_BYTES`), counted on these
/// bytes, the envelope included, which are the bytes sent (register PD-2,
/// T1.28a's fix round 1), before the model is called.
pub fn call_bytes(call: &Value) -> Result<Vec<u8>, CoreError> {
    let bytes = call.to_bytes();
    if bytes.len() > crate::payload::PAYLOAD_BYTES {
        return Err(CoreError::new(
            ErrorKind::InvalidRequest,
            ErrorCode::InvalidRequest,
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
}

/// The type a part's predictions are given back as.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnswerType {
    Float,
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
    /// The answer's place in the endpoint's order (across a merge's parts,
    /// the parts in order), which the result's items keep.
    pub answer_index: i64,
    pub id: Entity,
    pub prediction: Answer,
    /// Each class's probability, keyed by the class as the target column
    /// spells it, in the order the answer wrote them.
    pub probabilities: Option<Vec<(String, f64)>>,
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
    }
}

/// The target's dtype as the payload's task names it.
fn target_dtype(payload: &Value) -> Option<&str> {
    payload.get("task")?.get("target")?.get("dtype")?.as_text()
}

/// How a payload's answers are given back (`as_declared_target`): a
/// regression's as doubles, a classification's as its target's dtype, a
/// boolean, an integer or, for any other dtype, the class's text. A
/// classification of a float target is not one `materialize` writes
/// (`target_type`), and is refused by name.
fn answer_type(payload: &Value) -> Result<AnswerType, CoreError> {
    let kind = called(payload)?;
    if !classifies(kind) {
        return Ok(AnswerType::Float);
    }
    Ok(match target_dtype(payload) {
        Some(d) if d.starts_with("bool") => AnswerType::Bool,
        Some(d) if d.starts_with("int") => AnswerType::Int,
        Some(d) if d.starts_with("float") => return Err(not_yet("target_type")),
        _ => AnswerType::Text,
    })
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

/// One answer's prediction as the result gives it.
fn answer_of(prediction: Option<&Value>, answers: AnswerType) -> Result<Answer, CoreError> {
    let Some(value) = prediction else {
        return Ok(Answer::None);
    };
    match answers {
        // A regression's prediction is a number (owner decision of
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
        // A class written as a non-finite number is passed through as the
        // reference passes it (owner decision of 2026-10-12): no target's
        // type holds it, so `as_declared` leaves it as it is.
        _ if non_finite(value).is_some() => {
            Ok(Answer::NonFinite(non_finite(value).unwrap_or(f64::NAN)))
        }
        AnswerType::Bool => as_bool(value)
            .map(Answer::Bool)
            .ok_or_else(|| not_yet("prediction_type")),
        AnswerType::Int => as_int(value)
            .map(Answer::Int)
            .ok_or_else(|| not_yet("prediction_type")),
        AnswerType::Text => match value {
            Value::Text(text) => Ok(Answer::Text(text.clone())),
            _ => Err(not_yet("prediction_type")),
        },
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
/// would make two classes one.
fn spelled(probabilities: &[(String, f64)], answers: AnswerType) -> Vec<(String, f64)> {
    let respelled: Vec<(String, f64)> = probabilities
        .iter()
        .map(|(class, p)| {
            let as_written = Value::Text(class.clone());
            let key = match answers {
                AnswerType::Bool => as_bool(&as_written).map(|b| {
                    if b {
                        "True".to_owned()
                    } else {
                        "False".to_owned()
                    }
                }),
                AnswerType::Int => as_int(&as_written).map(|n| n.to_string()),
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
/// register P1B-86).
pub fn layout(payload: &Value, response: &PredictionResponse) -> Result<Part, CoreError> {
    let answers = answer_type(payload)?;
    let (names, rows) = instance_table(payload, "predict").ok_or_else(|| bad_payload("predict"))?;
    let (placed, text_ids) = match names.iter().position(|n| n.starts_with(ENTITY_REF)) {
        Some(reference) => {
            let dtype = key_dtype(payload, names[reference]);
            let text_ids = dtype.as_deref() == Some("string");
            (
                named(response, &names, rows, reference, dtype.as_deref())?,
                text_ids,
            )
        }
        None => (as_written(response)?, true),
    };

    let mut laid = Vec::with_capacity(placed.len());
    for (item, (row_index, id, forecast_step)) in response.predictions.iter().zip(placed) {
        if let Some(field) = item.carried.first() {
            return Err(not_yet("response_field").with_field("field", Detail::text(*field)));
        }
        laid.push(Row {
            row_index,
            forecast_step,
            answer_index: i64::try_from(laid.len()).unwrap_or(i64::MAX),
            id,
            prediction: answer_of(item.prediction.as_ref(), answers)?,
            probabilities: item
                .probabilities
                .as_ref()
                .map(|given| spelled(given, answers)),
        });
    }
    Ok(Part {
        model: response.model.clone(),
        request_id: response.id.clone(),
        metadata: response.metadata.clone(),
        rows: laid,
        answers,
        text_ids,
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
                .map(|row| instance_text(&cell(row, at)?))
                .collect::<Result<Vec<_>, _>>()?,
        ),
        None => None,
    };

    let mut answered = vec![false; entities.len()];
    let mut named = Vec::with_capacity(response.predictions.len());
    for item in &response.predictions {
        let index = usize::try_from(item.row_index)
            .ok()
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
            if *id != instances[index] {
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
fn as_written(response: &PredictionResponse) -> Result<Vec<(i64, Entity, Option<i64>)>, CoreError> {
    response
        .predictions
        .iter()
        .map(|item| {
            let row_index = i64::try_from(item.row_index)
                .map_err(|_| bad_response("prediction.row_index", "out_of_range"))?;
            let forecast_step = item
                .forecast_step
                .map(i64::try_from)
                .transpose()
                .map_err(|_| bad_response("prediction.forecast_step", "out_of_range"))?;
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
    let internal =
        |message: &str| CoreError::new(ErrorKind::Internal, ErrorCode::InternalError, message);
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
    let mut rows = first.rows;
    for part in rest {
        rows.extend(part.rows);
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
    })
}

/// The outputs the caller named that the answer does not carry
/// (`PredictionResult.absent`), each once, in the order named: none for an
/// answer with no rows; `prediction` when no answer holds one,
/// `probabilities` when none holds any, `explanation` and the fields this
/// build refuses an answer for carrying always; a name the result does not
/// know is taken as carried.
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
            "explanation" | "quantiles" | "scores" | "rankings" | "embeddings" => true,
            _ => false,
        })
        .cloned()
        .collect()
}

/// The notice of the outputs [`absent`] found, naming the model asked.
pub fn absent_notice(model: &str, absent: &[String]) -> Notice {
    Notice {
        code: OUTPUTS_ABSENT,
        message: "the answer left out outputs the request asked for",
        fields: vec![
            ("model", Detail::text(model)),
            ("absent", texts(absent.iter().map(String::as_str))),
        ],
    }
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
/// column spells it, to its probability, in the order written), and last
/// `answer_index` (int64, the answer's place in the endpoint's order, which
/// the result's items keep, as the reference's do).
pub fn table(part: &Part) -> Result<RecordBatch, CoreError> {
    let mut rows: Vec<&Row> = part.rows.iter().collect();
    rows.sort_by_key(|r| r.row_index);
    let row_index: ArrayRef = Arc::new(Int64Array::from_iter_values(
        rows.iter().map(|r| r.row_index),
    ));
    let integers = rows
        .iter()
        .all(|r| matches!(r.id, Entity::Integer(_) | Entity::None));
    let texts = rows
        .iter()
        .all(|r| matches!(r.id, Entity::Text(_) | Entity::None));
    let (id_type, ids): (DataType, ArrayRef) = if integers && !(texts && part.text_ids) {
        (
            DataType::Int64,
            Arc::new(Int64Array::from_iter(rows.iter().map(|r| match &r.id {
                Entity::Integer(n) => Some(*n),
                _ => None,
            }))),
        )
    } else {
        (
            DataType::Utf8,
            Arc::new(StringArray::from_iter(rows.iter().map(|r| match &r.id {
                Entity::Integer(n) => Some(n.to_string()),
                Entity::Text(text) => Some(text.clone()),
                other => id_as_written(other),
            }))),
        )
    };
    let mut fields = vec![
        Field::new("row_index", DataType::Int64, false),
        Field::new("id", id_type, true),
    ];
    let mut columns = vec![row_index, ids];
    // Where the ids are neither all integers nor all text (a null aside),
    // or one is of another type, each id as the JSON text of the
    // reference's item value, null where it had none.
    if !integers && !texts {
        fields.push(Field::new("id_as_written", DataType::Utf8, true));
        columns.push(Arc::new(StringArray::from_iter(
            rows.iter().map(|r| id_as_written(&r.id)),
        )));
    }
    if rows.iter().any(|r| r.forecast_step.is_some()) {
        fields.push(Field::new("forecast_step", DataType::Int64, true));
        columns.push(Arc::new(Int64Array::from_iter(
            rows.iter().map(|r| r.forecast_step),
        )));
    }
    if rows.iter().any(|r| r.prediction != Answer::None) {
        let (data_type, column): (DataType, ArrayRef) = match part.answers {
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
        };
        let widened = (data_type == DataType::Float64
            && rows.iter().any(|r| matches!(r.prediction, Answer::Int(_))))
            || rows
                .iter()
                .any(|r| matches!(r.prediction, Answer::NonFinite(_)));
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
    fields.push(Field::new("answer_index", DataType::Int64, false));
    columns.push(Arc::new(Int64Array::from_iter_values(
        rows.iter().map(|r| r.answer_index),
    )));
    RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).map_err(|_| internal_layout())
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
    CoreError::new(
        ErrorKind::Internal,
        ErrorCode::InternalError,
        "the result does not lay out",
    )
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
fn entity(value: &Value, dtype: Option<&str>) -> Entity {
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

/// A row's instance id as an answer names it: an integer's digits, or the
/// text.
fn instance_text(value: &Value) -> Result<String, CoreError> {
    match value {
        Value::Number(token) if is_integer_token(token) => Ok(token.clone()),
        Value::Text(text) => Ok(text.clone()),
        _ => Err(not_yet("instance_id_type")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
