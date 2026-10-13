// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The model API's prediction response, read as the client's generated
//! bindings read it (`PredictionResponse.from_dict` and
//! `PredictionItem.from_dict` of `sdm_api.generated.tfm_api`), for the
//! fields it reads: the response's `id`, `model`, `predictions` and
//! `metadata`, and each item's `id`, `row_index` and `forecast_step`, each
//! required and typed as the bindings require it, checked in their order.
//! For those fields a response the bindings refuse is refused here, and
//! one they read is read (`contract/response-vectors.json` holds the two
//! together). A refusal names the field and why, never a value.
//!
//! From T1.28b an item's `probabilities` is read as the bindings'
//! `_float_dict` reads it: absent or `null` is none, any other value not an
//! object is refused (`not_an_object`), and each member's value is a
//! number, or text a float reads, not a boolean (`not_a_number`) and
//! finite (`not_finite`), kept in the order written.
//!
//! As built at T1.11d the item's `prediction` is taken as written, as the
//! bindings take it, and its other fields (`scores`,
//! `rankings`, `embeddings`, `quantiles`, `explanation`, `metadata`) are
//! not read or type-checked: when one is there and not null, it is named in
//! [`PredictionItem::carried`], for the caller to refuse by name as a field
//! the result does not carry yet (`UNSUPPORTED_REQUEST`, register P1B-4),
//! where the bindings would refuse a malformed one as a response not the
//! contract's (`INVALID_RESPONSE`, decision E45(f)); T1.28 reads them.
//! Members the bindings do not read are ignored, as there.

use crate::json::{Value, whole};

/// Why a response is not one the contract describes: the field, and the
/// fault, never the value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Refused {
    /// The field, as `prediction.row_index` or `metadata`.
    pub field: &'static str,
    /// `missing`, `null`, `not_text`, `not_whole`, `not_an_object` or
    /// `not_a_sequence`.
    pub reason: &'static str,
}

fn refused(field: &'static str, reason: &'static str) -> Refused {
    Refused { field, reason }
}

/// The fields of an item this build does not read, in the bindings' order.
pub const UNREAD_ITEM_FIELDS: [&str; 6] = [
    "scores",
    "rankings",
    "embeddings",
    "quantiles",
    "explanation",
    "metadata",
];

/// One answer.
#[derive(Debug, Clone, PartialEq)]
pub struct PredictionItem {
    /// Which row of the request this answers; a whole number, of any size
    /// a response may write.
    pub row_index: i128,
    pub id: Option<String>,
    pub forecast_step: Option<i128>,
    /// The answer as the response wrote it; absent and `null` alike are
    /// `None`, as the bindings read both.
    pub prediction: Option<Value>,
    /// Each class's probability, in the order written; absent and `null`
    /// alike are `None`.
    pub probabilities: Option<Vec<(String, f64)>>,
    /// The fields of [`UNREAD_ITEM_FIELDS`] this item carries, not null.
    pub carried: Vec<&'static str>,
}

/// The response to one prediction request.
#[derive(Debug, Clone, PartialEq)]
pub struct PredictionResponse {
    pub id: String,
    pub model: String,
    pub predictions: Vec<PredictionItem>,
    /// The endpoint's `metadata` object, as written.
    pub metadata: Value,
}

impl PredictionResponse {
    /// Reads `value` as the bindings read a response.
    pub fn read(value: &Value) -> Result<PredictionResponse, Refused> {
        let Value::Object(_) = value else {
            return Err(refused("response", "not_an_object"));
        };
        let id = required_text(value, "id", "id")?;
        let model = required_text(value, "model", "model")?;
        let predictions = match value.get("predictions") {
            None => return Err(refused("predictions", "missing")),
            Some(Value::Array(items)) => items
                .iter()
                .map(PredictionItem::read)
                .collect::<Result<_, _>>()?,
            // An object iterates over its names, each of which is text and
            // so not an item; with none, there are no items.
            Some(Value::Object(members)) if members.is_empty() => vec![],
            Some(Value::Object(_)) => return Err(refused("prediction", "not_an_object")),
            Some(_) => return Err(refused("predictions", "not_a_sequence")),
        };
        let metadata = match value.get("metadata") {
            None => return Err(refused("metadata", "missing")),
            Some(object @ Value::Object(_)) => object.clone(),
            Some(_) => return Err(refused("metadata", "not_an_object")),
        };
        Ok(PredictionResponse {
            id,
            model,
            predictions,
            metadata,
        })
    }
}

impl PredictionItem {
    fn read(value: &Value) -> Result<PredictionItem, Refused> {
        let Value::Object(_) = value else {
            return Err(refused("prediction", "not_an_object"));
        };
        let id = match value.get("id") {
            None | Some(Value::Null) => None,
            Some(Value::Text(text)) => Some(text.clone()),
            Some(_) => return Err(refused("prediction.id", "not_text")),
        };
        let row_index = match value.get("row_index") {
            None => return Err(refused("prediction.row_index", "missing")),
            Some(Value::Null) => return Err(refused("prediction.row_index", "null")),
            Some(number) => whole_of(number, "prediction.row_index")?,
        };
        let forecast_step = match value.get("forecast_step") {
            None | Some(Value::Null) => None,
            Some(number) => Some(whole_of(number, "prediction.forecast_step")?),
        };
        let prediction = match value.get("prediction") {
            None | Some(Value::Null) => None,
            Some(answer) => Some(answer.clone()),
        };
        let probabilities = match value.get("probabilities") {
            None | Some(Value::Null) => None,
            Some(Value::Object(members)) => Some(
                members
                    .iter()
                    .map(|(class, p)| Ok((class.clone(), probability(p)?)))
                    .collect::<Result<_, Refused>>()?,
            ),
            Some(_) => return Err(refused("prediction.probabilities", "not_an_object")),
        };
        let carried = UNREAD_ITEM_FIELDS
            .into_iter()
            .filter(|name| !matches!(value.get(name), None | Some(Value::Null)))
            .collect();
        Ok(PredictionItem {
            row_index,
            id,
            forecast_step,
            prediction,
            probabilities,
            carried,
        })
    }
}

fn required_text(value: &Value, name: &str, field: &'static str) -> Result<String, Refused> {
    match value.get(name) {
        None => Err(refused(field, "missing")),
        Some(Value::Text(text)) => Ok(text.clone()),
        Some(_) => Err(refused(field, "not_text")),
    }
}

/// A whole number, as the bindings' `_whole` takes one: an integer, or a
/// float with no fraction; never a boolean or text.
fn whole_of(value: &Value, field: &'static str) -> Result<i128, Refused> {
    match value {
        Value::Number(token) => whole(token).ok_or(refused(field, "not_whole")),
        _ => Err(refused(field, "not_whole")),
    }
}

/// A probability as the bindings' `_finite` reads one: a number, or text
/// `float` reads (surrounding whitespace aside), never a boolean, and
/// finite.
fn probability(value: &Value) -> Result<f64, Refused> {
    const FIELD: &str = "prediction.probabilities";
    let read = match value {
        Value::Number(token) => token.parse::<f64>().ok(),
        Value::Text(text) => text.trim().parse::<f64>().ok(),
        _ => None,
    };
    match read {
        None => Err(refused(FIELD, "not_a_number")),
        Some(p) if !p.is_finite() => Err(refused(FIELD, "not_finite")),
        Some(p) => Ok(p),
    }
}
