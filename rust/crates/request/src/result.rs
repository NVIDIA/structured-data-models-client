// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `sdm_client/result.py`'s `PredictionResult` and the contract's
//! `PredictionItem`, with `merge` and `absent`; and the merges of a
//! question asked in parts (`client.py`'s `_ranked` and `_by_step`).
//!
//! An item's values are Python values ([`PyVal`]): an `id` is the key the
//! caller asked about, in whatever type their schema gave it.

use sdm_schema::pyval::PyVal;
use serde_json::{Map, Value};

use crate::error::{R, structured};
use crate::pyops::py_repr;

/// The fields of an item, in the contract's order.
pub const ITEM_FIELDS: [&str; 6] = [
    "prediction",
    "probabilities",
    "quantiles",
    "scores",
    "rankings",
    "embeddings",
];

/// A `PredictionItem`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Item {
    pub row_index: Option<PyVal>,
    pub row_index_sort_key: Option<PyVal>,
    pub id: Option<PyVal>,
    pub forecast_step: Option<PyVal>,
    pub prediction: Option<PyVal>,
    /// A mapping of text to floats, in order.
    pub probabilities: Option<Vec<(PyVal, PyVal)>>,
    pub scores: Option<Vec<PyVal>>,
    /// Each a mapping, in order.
    pub rankings: Option<Vec<Vec<(PyVal, PyVal)>>>,
    pub embeddings: Option<Vec<PyVal>>,
    pub quantiles: Option<Vec<(PyVal, PyVal)>>,
    pub explanation: Option<PyVal>,
    pub metadata: Option<PyVal>,
}

impl Item {
    pub fn row_index_i64(&self) -> Option<i64> {
        match self.row_index.as_ref() {
            Some(PyVal::Int(value)) => i64::try_from(*value).ok(),
            _ => None,
        }
    }

    pub fn forecast_step_i64(&self) -> Option<i64> {
        match self.forecast_step.as_ref() {
            Some(PyVal::Int(value)) => i64::try_from(*value).ok(),
            _ => None,
        }
    }

    /// Whether the item carries `name`: not `None`, nor an empty mapping or
    /// sequence (`_carries`'s test of one item).
    pub fn carries(&self, name: &str) -> bool {
        fn filled<T>(v: &Option<Vec<T>>) -> bool {
            v.as_ref().is_some_and(|v| !v.is_empty())
        }
        match name {
            "prediction" => match &self.prediction {
                None | Some(PyVal::None) => false,
                Some(PyVal::List(v) | PyVal::Tuple(v)) => !v.is_empty(),
                Some(PyVal::Dict(v)) => !v.is_empty(),
                Some(_) => true,
            },
            "probabilities" => filled(&self.probabilities),
            "quantiles" => filled(&self.quantiles),
            "scores" => filled(&self.scores),
            "rankings" => filled(&self.rankings),
            "embeddings" => filled(&self.embeddings),
            _ => true,
        }
    }
}

/// A `PredictionResult`.
#[derive(Clone, Debug, PartialEq)]
pub struct Result_ {
    pub host_values: Option<HostValues>,
    pub model: String,
    pub request_id: String,
    pub items: Vec<Item>,
    /// A mapping, in order.
    pub metadata: PyVal,
    pub explanation: Option<PyVal>,
    pub outputs: Vec<String>,
    pub unscored_candidates: Vec<PyVal>,
    pub names_its_rows: bool,
    pub names_its_step: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct HostValues {
    pub model: PyVal,
    pub request_id: PyVal,
    pub outputs: PyVal,
}

/// The result type, named as Python names it.
pub type PredictionResult = Result_;

impl Result_ {
    pub fn new(model: &str, request_id: &str, items: Vec<Item>, outputs: Vec<String>) -> Self {
        Self {
            host_values: None,
            model: model.to_string(),
            request_id: request_id.to_string(),
            items,
            metadata: PyVal::Dict(Vec::new()),
            explanation: None,
            outputs,
            unscored_candidates: Vec::new(),
            names_its_rows: true,
            names_its_step: false,
        }
    }

    /// `_carries(name)`.
    pub fn carries(&self, name: &str) -> bool {
        if name == "explanation" {
            return self.explanation.is_some();
        }
        if !ITEM_FIELDS.contains(&name) {
            return true;
        }
        self.items.iter().any(|item| item.carries(name))
    }

    /// `absent(requested)`.
    pub fn absent(&self, requested: &[String]) -> Vec<String> {
        if self.items.is_empty() {
            return Vec::new();
        }
        let mut seen: Vec<&String> = Vec::new();
        for name in requested {
            if !seen.contains(&name) {
                seen.push(name);
            }
        }
        seen.into_iter()
            .filter(|name| !self.carries(name))
            .cloned()
            .collect()
    }

    /// `PredictionResult.merge(parts)`.
    pub fn merge(parts: Vec<Result_>) -> R<Result_> {
        Self::merge_with(
            parts,
            merge_equal,
            merge_truth,
            with_metadata_entry,
            |error| error,
        )
    }

    /// Merge with call-local equality, truthiness and mapping facts for opaque values.
    /// The core chooses when to ask, preserving validation and short circuiting.
    pub fn merge_with<E>(
        parts: Vec<Result_>,
        equal: impl FnMut(&PyVal, &PyVal) -> Result<bool, E>,
        truth: impl FnMut(&PyVal) -> Result<bool, E>,
        metadata_entry: impl FnMut(&PyVal, &str, PyVal) -> Result<PyVal, E>,
        refuse: impl Fn(sdm_protocol::Diagnostic) -> E,
    ) -> Result<Result_, E> {
        Self::merge_with_protocol(
            parts,
            equal,
            truth,
            metadata_entry,
            |_, _| {
                Err(refuse(structured(
                    "INTERNAL_ERROR",
                    "result requires host protocol facts",
                )))
            },
            &refuse,
        )
    }

    pub fn merge_with_protocol<E>(
        parts: Vec<Result_>,
        mut equal: impl FnMut(&PyVal, &PyVal) -> Result<bool, E>,
        mut truth: impl FnMut(&PyVal) -> Result<bool, E>,
        mut metadata_entry: impl FnMut(&PyVal, &str, PyVal) -> Result<PyVal, E>,
        mut protocol: impl FnMut(&str, &[PyVal]) -> Result<PyVal, E>,
        refuse: impl Fn(sdm_protocol::Diagnostic) -> E,
    ) -> Result<Result_, E> {
        if parts.is_empty() {
            return Err(refuse(structured("INTERNAL_ERROR", "no parts to merge")));
        }
        if parts.len() == 1 {
            return Ok(parts.into_iter().next().unwrap_or_else(|| unreachable!()));
        }
        let held = parts.iter().any(|part| part.host_values.is_some());
        if held {
            let models = parts
                .iter()
                .map(|part| {
                    part.host_values.as_ref().map_or_else(
                        || PyVal::Str(part.model.clone()),
                        |values| values.model.clone(),
                    )
                })
                .collect();
            if let Some(shown) = distinct_protocol(models, &mut protocol, &refuse)? {
                return Err(refuse(structured(
                    "INTERNAL_ERROR",
                    format!("cannot merge results from different models: {shown}"),
                )));
            }
        }
        let mut models: Vec<&str> = parts.iter().map(|p| p.model.as_str()).collect();
        models.sort_unstable();
        models.dedup();
        if !held && models.len() > 1 {
            let shown: Vec<String> = models
                .iter()
                .map(|m| sdm_pql::pyrepr::str_repr(m))
                .collect();
            return Err(refuse(structured(
                "INTERNAL_ERROR",
                format!(
                    "cannot merge results from different models: [{}]",
                    shown.join(", ")
                ),
            )));
        }
        let first = &parts[0];
        let mut metadata = first.metadata.clone();
        metadata = metadata_entry(&metadata, "batches", PyVal::Int(parts.len() as i128))?;
        metadata = metadata_entry(
            &metadata,
            "batch_request_ids",
            PyVal::List(
                parts
                    .iter()
                    .map(|part| {
                        part.host_values.as_ref().map_or_else(
                            || PyVal::Str(part.request_id.clone()),
                            |values| values.request_id.clone(),
                        )
                    })
                    .collect(),
            ),
        )?;
        let mut explanation = None;
        for part in &parts {
            if let Some(value) = &part.explanation
                && truth(value)?
            {
                explanation = Some(value.clone());
                break;
            }
        }
        if held {
            let shapes = parts
                .iter()
                .map(|part| {
                    part.host_values.as_ref().map_or_else(
                        || PyVal::Tuple(part.outputs.iter().cloned().map(PyVal::Str).collect()),
                        |values| values.outputs.clone(),
                    )
                })
                .collect();
            if let Some(shown) = distinct_protocol(shapes, &mut protocol, &refuse)? {
                return Err(refuse(structured(
                    "INTERNAL_ERROR",
                    format!(
                        "these parts were asked for different outputs ({shown}), so they are not slices of one request"
                    ),
                )));
            }
        }
        let mut shapes: Vec<Vec<String>> = Vec::new();
        for part in &parts {
            if !shapes.contains(&part.outputs) {
                shapes.push(part.outputs.clone());
            }
        }
        if !held && shapes.len() > 1 {
            shapes.sort();
            let shown: Vec<String> = shapes.iter().map(|s| tuple_text(s)).collect();
            return Err(refuse(structured(
                "INTERNAL_ERROR",
                format!(
                    "these parts were asked for different outputs ([{}]), so they are not \
                     slices of one request",
                    shown.join(", ")
                ),
            )));
        }
        let mut unscored: Vec<PyVal> = Vec::new();
        for part in &parts {
            let previous = unscored.len();
            for candidate in &part.unscored_candidates {
                let mut present = false;
                for existing in &unscored[..previous] {
                    if equal(existing, candidate)? {
                        present = true;
                        break;
                    }
                }
                if !present {
                    unscored.push(candidate.clone());
                }
            }
        }
        let names_its_rows = parts.iter().all(|p| p.names_its_rows);
        let names_its_step = parts.iter().any(|p| p.names_its_step);
        let first = &parts[0];
        Ok(Result_ {
            host_values: first.host_values.clone(),
            model: first.model.clone(),
            request_id: first.request_id.clone(),
            items: parts.iter().flat_map(|p| p.items.clone()).collect(),
            metadata,
            explanation,
            outputs: first.outputs.clone(),
            unscored_candidates: unscored,
            names_its_rows,
            names_its_step,
        })
    }
}

fn distinct_protocol<E>(
    values: Vec<PyVal>,
    protocol: &mut impl FnMut(&str, &[PyVal]) -> Result<PyVal, E>,
    refuse: &impl Fn(sdm_protocol::Diagnostic) -> E,
) -> Result<Option<String>, E> {
    let length = values.len();
    let found = protocol("set", &[PyVal::List(values)])?;
    let invalid = || refuse(structured("INTERNAL_ERROR", "invalid host set fact"));
    let PyVal::List(facts) = found else {
        return Err(invalid());
    };
    let [reference, PyVal::Int(cardinality)] = facts.as_slice() else {
        return Err(invalid());
    };
    if !matches!(reference, PyVal::Opaque { kind, .. } if kind == "host_ref")
        || *cardinality < i128::from(length != 0)
        || *cardinality > length as i128
    {
        return Err(invalid());
    }
    if *cardinality <= 1 {
        return Ok(None);
    }
    match protocol("sorted_repr", std::slice::from_ref(reference))? {
        PyVal::Str(shown) => Ok(Some(shown)),
        _ => Err(refuse(structured(
            "INTERNAL_ERROR",
            "invalid host sorted repr fact",
        ))),
    }
}

pub fn protocol_bool<E>(
    protocol: &mut impl FnMut(&str, &[PyVal]) -> Result<PyVal, E>,
    operation: &str,
    operands: &[PyVal],
    refuse: &impl Fn(&str) -> E,
) -> Result<bool, E> {
    match protocol(operation, operands)? {
        PyVal::Bool(value) => Ok(value),
        _ => Err(refuse("host protocol did not return a boolean")),
    }
}

fn protocol_attribute<E>(
    protocol: &mut impl FnMut(&str, &[PyVal]) -> Result<PyVal, E>,
    owner: &PyVal,
    name: &PyVal,
    refuse: &impl Fn(&str) -> E,
) -> Result<PyVal, E> {
    let value = protocol("getattr_default_none", &[owner.clone(), name.clone()])?;
    if !matches!(&value, PyVal::None)
        && !matches!(&value, PyVal::Opaque { kind, .. } if kind == "host_ref")
    {
        return Err(refuse(
            "host attribute must be None or an original reference",
        ));
    }
    Ok(value)
}

pub fn absent_with<E>(
    result: &Result_,
    owner: &PyVal,
    requested: &PyVal,
    mut protocol: impl FnMut(&str, &[PyVal]) -> Result<PyVal, E>,
    refuse: impl Fn(&str) -> E,
) -> Result<Vec<PyVal>, E> {
    if result.items.is_empty() {
        return Ok(Vec::new());
    }
    let PyVal::List(names) = protocol("dict_keys", std::slice::from_ref(requested))? else {
        return Err(refuse("host dictionary keys must be a list"));
    };
    if names
        .iter()
        .any(|name| !matches!(name, PyVal::Opaque { kind, .. } if kind == "host_ref"))
    {
        return Err(refuse("host dictionary keys must retain references"));
    }
    let fields = PyVal::Tuple(
        ITEM_FIELDS
            .iter()
            .map(|name| PyVal::Str((*name).to_string()))
            .collect(),
    );
    let empty = PyVal::Tuple(vec![
        PyVal::None,
        PyVal::Dict(Vec::new()),
        PyVal::List(Vec::new()),
        PyVal::Tuple(Vec::new()),
    ]);
    let mut absent = Vec::new();
    for name in names {
        let carried = if protocol_bool(
            &mut protocol,
            "equal",
            &[name.clone(), PyVal::Str("explanation".to_string())],
            &refuse,
        )? {
            !matches!(
                protocol_attribute(
                    &mut protocol,
                    owner,
                    &PyVal::Str("_explanation".to_string()),
                    &refuse
                )?,
                PyVal::None
            )
        } else if !protocol_bool(
            &mut protocol,
            "contains",
            &[fields.clone(), name.clone()],
            &refuse,
        )? {
            true
        } else {
            let mut present = false;
            for item in &result.items {
                let reference = item
                    .metadata
                    .as_ref()
                    .ok_or_else(|| refuse("absence requires original item references"))?;
                let value = protocol_attribute(&mut protocol, reference, &name, &refuse)?;
                if !protocol_bool(&mut protocol, "contains", &[empty.clone(), value], &refuse)? {
                    present = true;
                    break;
                }
            }
            present
        };
        if !carried {
            absent.push(name);
        }
    }
    Ok(absent)
}

/// `repr(tuple_of_text)`.
pub fn tuple_text(names: &[String]) -> String {
    let parts: Vec<String> = names.iter().map(|n| sdm_pql::pyrepr::str_repr(n)).collect();
    if parts.len() == 1 {
        format!("({},)", parts[0])
    } else {
        format!("({})", parts.join(", "))
    }
}

pub fn with_metadata_entry(metadata: &PyVal, key: &str, value: PyVal) -> R<PyVal> {
    let PyVal::Dict(mut map) = metadata.clone() else {
        return Err(structured(
            "INTERNAL_ERROR",
            "metadata requires host mapping facts",
        ));
    };
    let key = PyVal::Str(key.to_string());
    if let Some(entry) = map.iter_mut().find(|(existing, _)| existing == &key) {
        entry.1 = value;
    } else {
        map.push((key, value));
    }
    Ok(PyVal::Dict(map))
}

/// Python truthiness of an explanation mapping.
fn truthy(value: &PyVal) -> bool {
    crate::pyops::py_bool(value)
}

pub fn merge_equal(left: &PyVal, right: &PyVal) -> R<bool> {
    for value in [left, right] {
        if matches!(value, PyVal::Opaque { kind, .. } if kind == "host_ref") {
            return Err(structured(
                "INTERNAL_ERROR",
                "candidate equality requires host facts",
            ));
        }
    }
    Ok(py_eq(left, right))
}

pub fn merge_truth(value: &PyVal) -> R<bool> {
    if matches!(value, PyVal::Opaque { kind, .. } if kind == "host_ref") {
        return Err(structured(
            "INTERNAL_ERROR",
            "explanation truthiness requires host facts",
        ));
    }
    Ok(truthy(value))
}

/// `a == b` as Python compares two values: `1 == 1.0 == True`.
pub fn py_eq(a: &PyVal, b: &PyVal) -> bool {
    match (a.semantic(), b.semantic()) {
        (PyVal::Float(x), PyVal::Float(y)) => x == y,
        (PyVal::List(left), PyVal::List(right)) | (PyVal::Tuple(left), PyVal::Tuple(right)) => {
            left.len() == right.len()
                && left
                    .iter()
                    .zip(right)
                    .all(|(left, right)| py_eq(left, right))
        }
        (PyVal::Dict(left), PyVal::Dict(right)) => {
            left.len() == right.len()
                && left.iter().all(|(key, value)| {
                    right
                        .iter()
                        .find(|(other_key, _)| py_eq(key, other_key))
                        .is_some_and(|(_, other_value)| py_eq(value, other_value))
                })
        }
        _ => {
            let (ka, kb) = (a.eq_key(), b.eq_key());
            ka == kb && !ka.starts_with("f:nan")
        }
    }
}

pub fn json_py_eq(left: &Value, right: &Value) -> bool {
    py_eq(
        &crate::keys::json_cell(left),
        &crate::keys::json_cell(right),
    )
}

/// `_how_sure(item, candidate)`: the probability of the positive class.
pub fn how_sure(item: &Item, candidate: &PyVal) -> R<f64> {
    let given = item.probabilities.clone().unwrap_or_default();
    for spelling in ["true", "True", "TRUE", "1"] {
        if let Some((_, value)) = given
            .iter()
            .find(|(k, _)| matches!(k, PyVal::Str(s) if s == spelling))
        {
            return crate::pyops::py_float(value);
        }
    }
    let mut keys: Vec<String> = given.iter().map(|(k, _)| crate::pyops::py_str(k)).collect();
    keys.sort();
    let said = if keys.is_empty() {
        "no probabilities".to_string()
    } else {
        let shown: Vec<String> = keys.iter().map(|k| sdm_pql::pyrepr::str_repr(k)).collect();
        format!("[{}]", shown.join(", "))
    };
    Err(structured(
        "INTERNAL_ERROR",
        format!(
            "the endpoint answered the question about candidate {} without saying how sure it \
             was, and a ranking is only the order of those. It gave back {said}, and this \
             client knows a positive class spelled \"true\", \"True\", \"TRUE\" or \"1\".",
            py_repr(candidate)
        ),
    ))
}

/// `_RANKING_OUTPUTS`.
pub const RANKING_OUTPUTS: [&str; 2] = ["prediction", "rankings"];

/// `_ranked(answers, rewritten, outputs, unscored)`: the candidates each
/// entity reached, likeliest first.
pub fn ranked(
    answers: &[Result_],
    candidates: &[PyVal],
    top: Option<i64>,
    outputs: Option<&[String]>,
    unscored: Vec<PyVal>,
) -> R<Result_> {
    // Keyed by the row and the entity on it, in the order first seen.
    let mut order: Vec<(Option<PyVal>, Option<PyVal>)> = Vec::new();
    let mut scores: Vec<Vec<(f64, PyVal)>> = Vec::new();
    for (answer, candidate) in answers.iter().zip(candidates) {
        for item in &answer.items {
            let sure = how_sure(item, candidate)?;
            let found = order.iter().position(|(row, id)| {
                *row == item.row_index
                    && match (id, &item.id) {
                        (None, None) => true,
                        (Some(a), Some(b)) => py_eq(a, b),
                        _ => false,
                    }
            });
            let at = match found {
                Some(at) => at,
                None => {
                    order.push((item.row_index.clone(), item.id.clone()));
                    scores.push(Vec::new());
                    order.len() - 1
                }
            };
            scores[at].push((sure, candidate.clone()));
        }
    }
    let wanted: Vec<String> = match outputs {
        Some(o) if !o.is_empty() => o.to_vec(),
        _ => RANKING_OUTPUTS.iter().map(|s| s.to_string()).collect(),
    };
    let mut items = Vec::new();
    for (position, ((row, entity), scored)) in order.into_iter().zip(scores).enumerate() {
        let keys: Vec<f64> = scored.iter().map(|(s, _)| *s).collect();
        let sorted: Vec<(f64, PyVal)> = crate::pysort::sort_desc(&keys)
            .into_iter()
            .map(|i| scored[i].clone())
            .collect();
        let chosen: Vec<(f64, PyVal)> = match top {
            Some(k) if k != 0 => {
                if k > 0 {
                    sorted.into_iter().take(k as usize).collect()
                } else {
                    let keep = (sorted.len() as i64 + k).max(0) as usize;
                    sorted.into_iter().take(keep).collect()
                }
            }
            _ => sorted,
        };
        items.push(Item {
            row_index: Some(row.unwrap_or(PyVal::Int(position as i128))),
            id: entity,
            prediction: wanted
                .iter()
                .any(|w| w == "prediction")
                .then(|| PyVal::List(chosen.iter().map(|(_, c)| c.clone()).collect())),
            rankings: wanted.iter().any(|w| w == "rankings").then(|| {
                chosen
                    .iter()
                    .map(|(s, c)| {
                        vec![
                            (PyVal::Str("candidate".to_string()), c.clone()),
                            (PyVal::Str("score".to_string()), PyVal::Float(*s)),
                        ]
                    })
                    .collect()
            }),
            ..Item::default()
        });
    }
    let (model, request_id) = answers
        .first()
        .map(|a| (a.model.clone(), a.request_id.clone()))
        .unwrap_or_default();
    let mut result = Result_::new(&model, &request_id, items, wanted);
    result.unscored_candidates = unscored;
    Ok(result)
}

/// `_by_step(answers, rewritten)`: each step's answers, in step order.
pub fn by_step(answers: &[Result_], steps: &[i64]) -> Result_ {
    let mut items = Vec::new();
    let mut by_step: Vec<(PyVal, PyVal)> = Vec::new();
    for (answer, step) in answers.iter().zip(steps) {
        for item in &answer.items {
            let mut item = item.clone();
            item.forecast_step = Some(PyVal::Int(i128::from(*step)));
            items.push(item);
        }
        if let Some(explanation) = &answer.explanation {
            by_step.push((PyVal::Int(i128::from(*step)), explanation.clone()));
        }
    }
    let (model, request_id, outputs) = answers
        .first()
        .map(|a| (a.model.clone(), a.request_id.clone(), a.outputs.clone()))
        .unwrap_or_default();
    let mut built = Result_::new(&model, &request_id, items, outputs);
    if !by_step.is_empty() {
        built.explanation = Some(PyVal::Dict(vec![(
            PyVal::Str("by_forecast_step".to_string()),
            PyVal::Dict(by_step),
        )]));
    }
    built.names_its_step = true;
    built.names_its_rows = answers.iter().all(|a| a.names_its_rows);
    built
}

/// A mapping of text keys as JSON-ready pairs.
pub fn json_map(items: &[(String, Value)]) -> Map<String, Value> {
    items.iter().cloned().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn python_equality_compares_nested_containers_and_unordered_dictionaries() {
        let left = PyVal::Dict(vec![
            (PyVal::Bool(true), PyVal::List(vec![PyVal::Bool(false)])),
            (
                PyVal::Str("nested".into()),
                PyVal::Tuple(vec![PyVal::Dict(vec![(PyVal::Int(2), PyVal::Int(1))])]),
            ),
        ]);
        let right = PyVal::Dict(vec![
            (
                PyVal::Str("nested".into()),
                PyVal::Tuple(vec![PyVal::Dict(vec![(
                    PyVal::Float(2.0),
                    PyVal::Float(1.0),
                )])]),
            ),
            (PyVal::Int(1), PyVal::List(vec![PyVal::Int(0)])),
        ]);
        assert!(py_eq(&left, &right));
        assert!(py_eq(&right, &left));
        assert!(json_py_eq(
            &json!({"first": [false, {"nested": 1}], "second": null}),
            &json!({"second": null, "first": [0, {"nested": 1.0}]}),
        ));
    }

    #[test]
    fn python_equality_preserves_container_types_and_real_inequality() {
        let list = PyVal::List(vec![PyVal::Bool(false)]);
        let tuple = PyVal::Tuple(vec![PyVal::Int(0)]);
        assert!(!py_eq(&list, &tuple));
        assert!(!py_eq(&tuple, &list));
        for (left, right) in [
            (json!([false]), json!([1])),
            (json!([false]), json!([0, 0])),
            (json!([0, 1]), json!([1, 0])),
            (json!({"x": [false]}), json!({"x": [1]})),
            (json!({"x": false}), json!({"y": 0})),
            (json!({"x": false}), json!({"x": 0, "y": 0})),
            (json!("1"), json!(1)),
            (json!(null), json!(false)),
            (json!([]), json!({})),
        ] {
            assert!(!json_py_eq(&left, &right));
            assert!(!json_py_eq(&right, &left));
        }
    }

    #[test]
    fn python_equality_keeps_integer_float_comparisons_exact() {
        for (integer, float, equal) in [
            (0, -0.0, true),
            (0, f64::NAN, false),
            (0, f64::INFINITY, false),
            (0, f64::NEG_INFINITY, false),
            (1_i128 << 53, 2_f64.powi(53), true),
            ((1_i128 << 53) + 1, 2_f64.powi(53), false),
            (i128::from(i64::MAX), 2_f64.powi(63), false),
            (i128::from(i64::MIN), -2_f64.powi(63), true),
            (i128::from(u64::MAX), 2_f64.powi(64), false),
            ((1_i128 << 64) - 2048, 2_f64.powi(64) - 2048.0, true),
        ] {
            assert_eq!(py_eq(&PyVal::Int(integer), &PyVal::Float(float)), equal);
            assert_eq!(py_eq(&PyVal::Float(float), &PyVal::Int(integer)), equal);
        }
        for (digits, float, equal) in [
            (
                "1361129467683753853853498429727072845824".to_string(),
                2_f64.powi(130),
                true,
            ),
            (
                "1361129467683753853853498429727072845825".to_string(),
                2_f64.powi(130),
                false,
            ),
            (
                "-1361129467683753853853498429727072845824".to_string(),
                -2_f64.powi(130),
                true,
            ),
            (format!("{:.0}", f64::MAX), f64::MAX, true),
            (format!("{:.0}", f64::MAX), f64::INFINITY, false),
            (format!("{:.0}", f64::MAX), f64::NAN, false),
        ] {
            let integer = PyVal::BigInt(digits);
            let float = PyVal::Float(float);
            assert_eq!(py_eq(&integer, &float), equal);
            assert_eq!(py_eq(&float, &integer), equal);
            assert_eq!(integer.eq_key() == float.eq_key(), equal);
        }
        assert!(py_eq(&PyVal::Bool(false), &PyVal::Float(-0.0)));
        assert!(!py_eq(&PyVal::Float(f64::NAN), &PyVal::Float(f64::NAN)));
    }

    fn part(candidates: Vec<PyVal>) -> Result_ {
        let mut result = Result_::new("model", "request", Vec::new(), Vec::new());
        result.unscored_candidates = candidates;
        result
    }

    #[test]
    fn merge_compares_candidates_with_the_part_start_snapshot() {
        for candidates in [
            vec![PyVal::Int(7), PyVal::Int(7)],
            vec![PyVal::Int(1), PyVal::Bool(true), PyVal::Float(1.0)],
        ] {
            let merged = Result_::merge(vec![part(candidates.clone()), part(candidates.clone())])
                .expect("ordinary native values merge");
            assert_eq!(merged.unscored_candidates, candidates);
        }
        let merged = Result_::merge(vec![
            part(vec![PyVal::Int(3), PyVal::Int(1)]),
            part(vec![PyVal::Int(1), PyVal::Int(2), PyVal::Int(2)]),
        ])
        .expect("overlap merges");
        assert_eq!(
            merged.unscored_candidates,
            vec![PyVal::Int(3), PyVal::Int(1), PyVal::Int(2), PyVal::Int(2)]
        );
    }

    #[test]
    fn merge_host_protocol_is_short_circuited_after_validation() {
        let first = part(vec![PyVal::Int(1), PyVal::Int(1)]);
        let mut second = part(vec![PyVal::Int(1), PyVal::Int(2)]);
        second.model = "other-model".to_string();
        let error = Result_::merge_with(
            vec![first.clone(), second.clone()],
            |_, _| panic!("equality precedes validation"),
            |_| panic!("truthiness precedes model validation"),
            with_metadata_entry,
            |error| error,
        )
        .expect_err("models differ");
        assert!(error.message.contains("different models"));
        second.model = first.model.clone();
        let mut comparisons = Vec::new();
        let merged = Result_::merge_with(
            vec![first, second],
            |left, right| {
                comparisons.push((left.clone(), right.clone()));
                Ok(py_eq(left, right))
            },
            |value| Ok(truthy(value)),
            with_metadata_entry,
            |error| error,
        )
        .expect("equal host facts merge");
        assert_eq!(comparisons.len(), 3);
        assert_eq!(merged.unscored_candidates.len(), 3);
    }

    #[test]
    fn typed_metadata_keys_survive_native_merge_and_encoding() {
        let mut first = part(Vec::new());
        first.metadata = PyVal::Dict(vec![
            (PyVal::Int(1), PyVal::Str("integer".to_string())),
            (PyVal::Str("1".to_string()), PyVal::Str("text".to_string())),
        ]);
        let merged = Result_::merge(vec![first.clone(), first]).expect("metadata merges");
        let restored = crate::layout::decode_result(&crate::layout::encode_result(&merged))
            .expect("typed metadata decodes");
        assert_eq!(restored, merged);
        assert!(matches!(restored.metadata, PyVal::Dict(entries) if entries.len() == 4));
    }
}
