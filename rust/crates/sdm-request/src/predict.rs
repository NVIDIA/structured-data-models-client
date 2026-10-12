// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A materialized request sent to the model, and its answer laid out.
//!
//! [`body`] is the call the reference's relational adapter makes of one
//! gathered payload (`RelationalAdapter.prediction_plan`): the payload with
//! the model named, the outputs asked for, the request's id in its
//! metadata, and only the inference options the model reads. [`layout`] is
//! its reading of the answer (`RelationalAdapter.normalize`): each answer
//! checked against the row it claims, given back under the entity it is
//! about, in the endpoint's order.
//!
//! As built at T1.11d it is thin: a regression, asked for its default
//! output, answered by `kumo-relational`; everything else is refused with
//! `UNSUPPORTED_REQUEST` naming the rule ([`not_yet`], register entry
//! P1B-4), never guessed. From T1.28a the call is written for a
//! classification too, asked for its default outputs, with the inference
//! options the model reads, and an option the caller set that it does not
//! read refused before the call (the adapter's `_inference`); the answer to
//! a classification is still refused (`problem_type`, T1.28b). The client's
//! checks before the call (an empty context, a constant target) and its
//! presentation of the answer stay in the host (decision E45).

use std::sync::Arc;

use arrow_array::{ArrayRef, Float64Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use sdm_contract::{CoreError, Detail, ErrorCode, ErrorKind};
use sdm_wire::json::{Value, is_integer_token};
use sdm_wire::response::{PredictionResponse, Refused};

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

/// What the wire calls the column holding the entity a row is about, by
/// its prefix.
const ENTITY_REF: &str = "__kumo_entity_ref";

/// What the wire calls the column a row is answered under.
const INSTANCE_ID: &str = "instance_id";

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

/// The task's kind, which must be a regression in this build.
fn regression(payload: &Value) -> Result<(), CoreError> {
    let Value::Object(_) = payload else {
        return Err(bad_payload("not_an_object"));
    };
    match payload.get("task").and_then(|task| task.get("kind")) {
        Some(Value::Text(kind)) if kind == "regression" => Ok(()),
        _ => Err(not_yet("problem_type")),
    }
}

/// The model a request is sent to: `kumo-relational`; the tabular model
/// takes no relational request yet, and any other is no model.
pub fn check_model(model: &str) -> Result<(), CoreError> {
    match model {
        RELATIONAL => Ok(()),
        TABULAR => Err(not_yet("model")),
        _ => Err(CoreError::new(
            ErrorKind::NotFound,
            ErrorCode::UnknownModel,
            "the model named is not one the contract carries",
        )
        .with_field("field", Detail::text("model"))),
    }
}

/// The task's kind, when it is one whose call this build writes.
fn called(payload: &Value) -> Result<&str, CoreError> {
    let Value::Object(_) = payload else {
        return Err(bad_payload("not_an_object"));
    };
    match payload.get("task").and_then(|task| task.get("kind")) {
        Some(Value::Text(kind)) if CALLED.contains(&kind.as_str()) => Ok(kind),
        _ => Err(not_yet("problem_type")),
    }
}

/// The outputs this build asks for of a payload: its task's default, and
/// any other list refused by name (`outputs`, decision E61).
pub fn check_outputs(payload: &Value, outputs: &[String]) -> Result<(), CoreError> {
    let default: &[&str] = if called(payload)? == "regression" {
        &REGRESSION_FIELDS
    } else {
        &CLASSIFICATION_FIELDS
    };
    if outputs != default {
        return Err(not_yet("outputs"));
    }
    Ok(())
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
    check_model(model)?;
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
                    .with_field(
                        "unsupported",
                        Detail::List(
                            refused
                                .iter()
                                .map(|n| sdm_contract::Scalar::Utf8((*n).clone()))
                                .collect(),
                        ),
                    ));
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

/// The entity a row is about, as the result gives it.
enum Entity {
    Integer(i64),
    Text(String),
}

/// The answer laid out: one row per answer, in the endpoint's order, with
/// `row_index` (int64), `id` (the entity the row is about, int64 for an
/// integer key and utf8 for a text one) and `prediction` (float64).
pub fn layout(payload: &Value, response: &PredictionResponse) -> Result<RecordBatch, CoreError> {
    regression(payload)?;
    let table = payload
        .get("predict")
        .and_then(|p| p.get("instance_table"))
        .ok_or_else(|| bad_payload("predict"))?;
    let Some(Value::Array(columns)) = table.get("columns") else {
        return Err(bad_payload("predict"));
    };
    let Some(Value::Array(rows)) = table.get("rows") else {
        return Err(bad_payload("predict"));
    };
    let names: Vec<&str> = columns.iter().filter_map(Value::as_text).collect();
    if names.len() != columns.len() {
        return Err(bad_payload("predict"));
    }
    let reference = names
        .iter()
        .position(|n| n.starts_with(ENTITY_REF))
        .ok_or_else(|| not_yet("entity_ref"))?;
    let dtype = key_dtype(payload, names[reference]);
    let instance = names.iter().position(|n| *n == INSTANCE_ID);
    let cell = |row: &Value, at: usize| -> Result<Value, CoreError> {
        match row {
            Value::Array(cells) => cells.get(at).cloned().ok_or_else(|| bad_payload("predict")),
            _ => Err(bad_payload("predict")),
        }
    };
    let entities = rows
        .iter()
        .map(|row| entity(&cell(row, reference)?, dtype.as_deref()))
        .collect::<Result<Vec<_>, _>>()?;
    let instances = match instance {
        Some(at) => Some(
            rows.iter()
                .map(|row| instance_text(&cell(row, at)?))
                .collect::<Result<Vec<_>, _>>()?,
        ),
        None => None,
    };

    // Each answer is checked against the row it claims before it is given
    // the entity of that row: in range, not a forecast's, once, and under
    // the instance id that row was asked under.
    let mut answered = vec![false; entities.len()];
    let mut placed = Vec::with_capacity(response.predictions.len());
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
        placed.push(index);
    }
    if answered.iter().any(|a| !a) {
        return Err(bad_response("prediction.row_index", "unanswered"));
    }

    // What the result does not carry yet is refused by name, never
    // dropped: a field of an answer beyond its prediction, and a
    // prediction that is not written as a float.
    let mut predictions = Vec::with_capacity(placed.len());
    for item in &response.predictions {
        if let Some(field) = item.carried.first() {
            return Err(not_yet("response_field").with_field("field", Detail::text(*field)));
        }
        match &item.prediction {
            Some(Value::Number(token)) if !is_integer_token(token) => {
                let value: f64 = token
                    .parse()
                    .map_err(|_| bad_response("prediction.prediction", "not_a_number"))?;
                // A finite token past a double's range reads as infinity,
                // which the reference passes on and this build refuses, as
                // it refuses NaN and Infinity (register P1B-4).
                if !value.is_finite() {
                    return Err(bad_response("prediction.prediction", "not_finite"));
                }
                predictions.push(value);
            }
            _ => return Err(not_yet("prediction_type")),
        }
    }

    let row_index: ArrayRef = Arc::new(Int64Array::from_iter_values(
        placed.iter().map(|i| i64::try_from(*i).unwrap_or(i64::MAX)),
    ));
    let (id_type, ids): (DataType, ArrayRef) = match entities.first() {
        Some(Entity::Text(_)) => (
            DataType::Utf8,
            Arc::new(StringArray::from_iter_values(placed.iter().map(
                |i| match &entities[*i] {
                    Entity::Text(text) => text.as_str(),
                    Entity::Integer(_) => "",
                },
            ))),
        ),
        _ => (
            DataType::Int64,
            Arc::new(Int64Array::from_iter_values(placed.iter().map(
                |i| match &entities[*i] {
                    Entity::Integer(value) => *value,
                    Entity::Text(_) => 0,
                },
            ))),
        ),
    };
    let prediction: ArrayRef = Arc::new(Float64Array::from(predictions));
    let schema = Schema::new(vec![
        Field::new("row_index", DataType::Int64, false),
        Field::new("id", id_type, false),
        Field::new("prediction", DataType::Float64, false),
    ]);
    RecordBatch::try_new(Arc::new(schema), vec![row_index, ids, prediction]).map_err(|_| {
        CoreError::new(
            ErrorKind::Internal,
            ErrorCode::InternalError,
            "the result does not lay out",
        )
    })
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

/// The entity of a row: its reference's value read back as the key's
/// dtype, as the reference's client reads it. This build takes an integer
/// key, written as an integer or, past the doubles' exact range, as its
/// digits, and a text key; any other is refused by name.
fn entity(value: &Value, dtype: Option<&str>) -> Result<Entity, CoreError> {
    match (dtype, value) {
        (Some("int8" | "int16" | "int32" | "int64"), Value::Number(token) | Value::Text(token))
            if is_integer_token(token) =>
        {
            token
                .parse::<i64>()
                .map(Entity::Integer)
                .map_err(|_| not_yet("id_type"))
        }
        (Some("string"), Value::Text(text)) => Ok(Entity::Text(text.clone())),
        _ => Err(not_yet("id_type")),
    }
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
