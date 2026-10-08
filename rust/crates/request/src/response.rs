// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! An endpoint's answer read as the contract's `PredictionResponse`
//! (`sdm_api.generated.tfm_api`), with today's refusal of one that does not
//! match it, and `as_declared_target` (`wire/answers.py`).

use sdm_schema::pyval::PyVal;
use serde_json::{Map, Value};

use crate::keys::json_cell;
use crate::pyops::py_repr;
use crate::result::Item;

/// What a response that does not match the contract is refused with.
pub type Broken = String;

/// A parsed `PredictionResponse`.
#[derive(Clone, Debug)]
pub struct Response {
    pub id: String,
    pub model: String,
    pub predictions: Vec<Item>,
    pub metadata: Vec<(String, PyVal)>,
}

fn repr(value: &Value) -> String {
    py_repr(&json_cell(value))
}

fn require<'a>(
    data: &'a Map<String, Value>,
    field: &str,
    model: &str,
) -> Result<&'a Value, Broken> {
    data.get(field).ok_or_else(|| {
        format!(
            "{model} response is missing required field {}",
            sdm_pql::pyrepr::str_repr(field)
        )
    })
}

fn require_str(data: &Map<String, Value>, field: &str, model: &str) -> Result<String, Broken> {
    let value = require(data, field, model)?;
    value.as_str().map(str::to_string).ok_or_else(|| {
        format!(
            "{model} response has {} = {}, where the contract requires a string",
            sdm_pql::pyrepr::str_repr(field),
            repr(value)
        )
    })
}

fn whole(value: &Value, field: &str, model: &str) -> Result<i64, Broken> {
    let field_repr = sdm_pql::pyrepr::str_repr(field);
    match value {
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                return Ok(i);
            }
            let x = n.as_f64().unwrap_or(f64::NAN);
            if !x.is_finite() || x.fract() != 0.0 {
                return Err(format!(
                    "{model} response has {field_repr} = {}, which is not a whole number",
                    repr(value)
                ));
            }
            Ok(x as i64)
        }
        _ => Err(format!(
            "{model} response has {field_repr} = {}, where the contract requires a whole number",
            repr(value)
        )),
    }
}

fn finite(value: &Value) -> Result<PyVal, Broken> {
    let number = match value {
        Value::Bool(_) => None,
        Value::Number(n) => n.as_f64(),
        Value::String(s) => crate::pyops::parse_float(s),
        _ => None,
    };
    let Some(number) = number else {
        return Err(format!(
            "response carries {} where the contract requires a number",
            repr(value)
        ));
    };
    if !number.is_finite() {
        return Err(format!(
            "response carries a non-finite number {}, which the contract does not permit",
            repr(value)
        ));
    }
    Ok(PyVal::Float(number))
}

fn float_dict(value: Option<&Value>) -> Result<Option<Vec<(PyVal, PyVal)>>, Broken> {
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Object(map)) => map
            .iter()
            .map(|(k, v)| Ok((PyVal::Str(k.clone()), finite(v)?)))
            .collect::<Result<Vec<_>, _>>()
            .map(Some),
        Some(other) => Err(format!(
            "response carries {} where the contract requires an object",
            repr(other)
        )),
    }
}

fn float_tuple(value: Option<&Value>) -> Result<Option<Vec<PyVal>>, Broken> {
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Array(items)) => items
            .iter()
            .map(finite)
            .collect::<Result<Vec<_>, _>>()
            .map(Some),
        Some(other) => Err(format!(
            "response carries {} where the contract requires an array of numbers",
            repr(other)
        )),
    }
}

/// A mapping, as its pairs in order.
type Pairs = Vec<(PyVal, PyVal)>;

fn mapping_tuple(value: Option<&Value>) -> Result<Option<Vec<Pairs>>, Broken> {
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Array(items)) => items
            .iter()
            .map(|item| match item {
                Value::Object(map) => Ok(map
                    .iter()
                    .map(|(k, v)| (PyVal::Str(k.clone()), json_cell(v)))
                    .collect()),
                _ => Err(
                    "cannot convert dictionary update sequence element #0 to a sequence"
                        .to_string(),
                ),
            })
            .collect::<Result<Vec<_>, _>>()
            .map(Some),
        Some(Value::Object(map)) => Ok(Some(map.keys().map(|_| Vec::new()).collect())),
        Some(_) => Err("object is not iterable".to_string()),
    }
}

fn dict_or_none(value: Option<&Value>) -> Result<Option<PyVal>, Broken> {
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Object(_)) => Ok(value.map(json_cell)),
        Some(_) => {
            Err("dictionary update sequence element #0 has length 1; 2 is required".to_string())
        }
    }
}

fn item(value: &Value) -> Result<Item, Broken> {
    let Value::Object(data) = value else {
        return Err(format!(
            "response carries {} where the contract requires a PredictionItem object",
            repr(value)
        ));
    };
    let id = match data.get("id") {
        None | Some(Value::Null) => None,
        Some(Value::String(s)) => Some(PyVal::Str(s.clone())),
        Some(other) => {
            return Err(format!(
                "prediction response has 'id' = {}, where the contract requires a string",
                repr(other)
            ));
        }
    };
    let row_index = {
        let value = require(data, "row_index", "PredictionItem")?;
        if value.is_null() {
            return Err(
                "PredictionItem response has a null 'row_index', which the contract requires"
                    .to_string(),
            );
        }
        whole(value, "row_index", "PredictionItem")?
    };
    let forecast_step = match data.get("forecast_step") {
        None | Some(Value::Null) => None,
        Some(v) => Some(whole(v, "forecast_step", "prediction")?),
    };
    Ok(Item {
        id,
        row_index: Some(PyVal::Int(i128::from(row_index))),
        row_index_sort_key: None,
        forecast_step: forecast_step.map(|value| PyVal::Int(i128::from(value))),
        prediction: data
            .get("prediction")
            .map(json_cell)
            .filter(|v| !matches!(v, PyVal::None)),
        probabilities: float_dict(data.get("probabilities"))?,
        scores: float_tuple(data.get("scores"))?,
        rankings: mapping_tuple(data.get("rankings"))?,
        embeddings: float_tuple(data.get("embeddings"))?,
        quantiles: float_dict(data.get("quantiles"))?,
        explanation: dict_or_none(data.get("explanation"))?,
        metadata: dict_or_none(data.get("metadata"))?,
    })
}

/// `PredictionResponse.from_dict(data)`.
pub fn parse(data: &Value) -> Result<Response, Broken> {
    let Value::Object(map) = data else {
        return Err(format!(
            "response carries {} where the contract requires a PredictionResponse object",
            repr(data)
        ));
    };
    let model_name = "PredictionResponse";
    let id = require_str(map, "id", model_name)?;
    let model = require_str(map, "model", model_name)?;
    let predictions = require(map, "predictions", model_name)?;
    let items = match predictions {
        Value::Array(items) => items.iter().map(item).collect::<Result<Vec<_>, _>>()?,
        Value::Object(_) => Vec::new(),
        _ => {
            return Err(format!(
                "{model_name} response field 'predictions' is not a sequence"
            ));
        }
    };
    let metadata = match require(map, "metadata", model_name)? {
        Value::Object(m) => m.iter().map(|(k, v)| (k.clone(), json_cell(v))).collect(),
        _ => {
            return Err(format!(
                "{model_name} response field 'metadata' is not an object"
            ));
        }
    };
    Ok(Response {
        id,
        model,
        predictions: items,
        metadata,
    })
}

/// `as_declared_target(items, payload)`: a classification's answer read
/// back as the target column's type.
pub fn as_declared_target(
    items: Vec<Item>,
    payload: &Value,
    protocols: Option<&crate::pyops::HostProtocols>,
) -> crate::error::R<Vec<Item>> {
    let task = crate::pyops::mapping_get(payload, "task", &serde_json::json!({}), protocols)?;
    let kind = crate::pyops::mapping_get(&task, "kind", &serde_json::json!(""), protocols)?;
    let kind = crate::pyops::string(&kind).unwrap_or("");
    if !kind.contains("classification") {
        return Ok(items);
    }
    let target = crate::pyops::mapping_get(&task, "target", &serde_json::json!({}), protocols)?;
    let dtype = crate::pyops::mapping_get(&target, "dtype", &Value::Null, protocols)?;
    let Some(dtype) = crate::pyops::string(&dtype).filter(|d| !d.is_empty()) else {
        return Ok(items);
    };
    if !(dtype.starts_with("int") || dtype.starts_with("float") || dtype.starts_with("bool")) {
        return Ok(items);
    }
    Ok(items
        .into_iter()
        .map(|mut item| {
            if let Some(value) = &item.prediction {
                item.prediction = Some(crate::wire::as_declared(value, Some(dtype)));
            }
            if let Some(given) = &item.probabilities
                && !given.is_empty()
            {
                let mut renamed: Vec<(PyVal, PyVal)> = Vec::new();
                for (key, value) in given {
                    let restored = crate::wire::as_declared(key, Some(dtype));
                    let spelled = if restored == *key {
                        key.clone()
                    } else {
                        PyVal::Str(crate::pyops::py_str(&restored))
                    };
                    match renamed.iter_mut().find(|(k, _)| *k == spelled) {
                        Some(entry) => entry.1 = value.clone(),
                        None => renamed.push((spelled, value.clone())),
                    }
                }
                if renamed.len() == given.len() {
                    item.probabilities = Some(renamed);
                }
            }
            item
        })
        .collect())
}
