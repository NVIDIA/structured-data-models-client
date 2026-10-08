// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The request fixture operations (`tests/parity/_fixture_ops.py`) the
//! request crate answers by itself: the wire's and the envelope's cell
//! rules, a candidate's literal, a two-number target, and a ranking's
//! order. The core answers the rest, which need a whole flow.
//!
//! An answer is `{"output": X, "warnings": W}` or `{"error": {"class",
//! "code", "message", "details_keys"}, "warnings": W}`, as fixture JSON
//! (`tests/parity/FORMAT.md`).

use sdm_protocol::Diagnostic;
use sdm_schema::pyval::PyVal;
use serde_json::{Map, Value, json};

use crate::error::short_class;
use crate::result::{Item, Result_};
use crate::series::Series;

/// An operation that cannot be answered: unknown, or its input unreadable.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InputError(pub String);

impl std::fmt::Display for InputError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for InputError {}

fn bad(message: impl Into<String>) -> InputError {
    InputError(message.into())
}

/// `{"output": output, "warnings": warnings}`.
pub fn answer(output: Value, warnings: &[Diagnostic]) -> String {
    let mut value = json!({
        "output": output,
        "warnings": warnings.iter().map(sdm_schema::diagnostic_compat::fixture_warning).collect::<Vec<_>>(),
    });
    sdm_schema::diagnostic_compat::historical_fixture(&mut value);
    value.to_string()
}

/// `error` as a fixture's `expected_error` writes it.
pub fn error_json(error: &Diagnostic) -> Value {
    let code = match &error.code {
        Some(code) if !code.is_empty() => code.clone(),
        _ => short_class(error),
    };
    let mut keys: Vec<&String> = error.details.keys().collect();
    keys.sort();
    json!({
        "class": error.class.as_deref().unwrap_or("sdm_core._native.CoreError"),
        "code": code,
        "message": error.message,
        "details_keys": keys,
    })
}

/// `{"error": ..., "warnings": warnings}`.
pub fn refusal(error: &Diagnostic, warnings: &[Diagnostic]) -> String {
    let mut value = json!({
        "error": error_json(error),
        "warnings": warnings.iter().map(sdm_schema::diagnostic_compat::fixture_warning).collect::<Vec<_>>(),
    });
    sdm_schema::diagnostic_compat::historical_fixture(&mut value);
    value.to_string()
}

fn outcome(result: Result<Value, Diagnostic>) -> String {
    match result {
        Ok(value) => answer(value, &[]),
        Err(error) => refusal(&error, &[]),
    }
}

pub use crate::values::{encode, timestamp_isoformat};

fn decode(value: Option<&Value>, what: &str) -> Result<PyVal, InputError> {
    PyVal::decode(value.ok_or_else(|| bad(format!("the input has no {what}")))?).map_err(bad)
}

fn series(input: &Map<String, Value>) -> Result<Series, InputError> {
    Series::from_spec(
        input
            .get("series")
            .ok_or_else(|| bad("the input has no series"))?,
    )
    .map_err(bad)
}

fn text<'a>(input: &'a Map<String, Value>, key: &str) -> Result<&'a str, InputError> {
    input
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| bad(format!("{key} is text")))
}

/// The operations this crate answers, on a case's input; `None` for one it
/// does not.
pub fn fixture_op(
    operation: &str,
    input: &Map<String, Value>,
) -> Option<Result<String, InputError>> {
    let answered = match operation {
        "wire.infer_dtype" => {
            series(input).map(|s| answer(json!(crate::wire::infer_tfm_dtype(&s)), &[]))
        }
        "wire.serialize_column" => series(input).map(|s| {
            let dtype = match input.get("dtype").and_then(Value::as_str) {
                Some(d) if !d.is_empty() => d.to_string(),
                _ => crate::wire::infer_tfm_dtype(&s),
            };
            outcome(
                crate::wire::serialize_column(&s, &dtype)
                    .map(|cells| json!({"dtype": dtype, "cells": cells})),
            )
        }),
        "wire.serialize_cell" => (|| {
            let value = decode(input.get("value"), "value")?;
            let dtype = text(input, "dtype")?;
            Ok(outcome(crate::wire::serialize_cell(&value, dtype)))
        })(),
        "wire.as_declared" => (|| {
            let value = decode(input.get("value"), "value")?;
            let dtype = input.get("dtype").and_then(Value::as_str);
            Ok(answer(
                encode(&crate::wire::as_declared(&value, dtype)),
                &[],
            ))
        })(),
        "wire.json_ready" => (|| {
            let value = decode(input.get("value"), "value")?;
            Ok(outcome(crate::wire::json_ready(&value).map(|v| encode(&v))))
        })(),
        "payload.cell_json_value" => (|| {
            let value = decode(input.get("value"), "value")?;
            let dtype = text(input, "dtype")?;
            Ok(outcome(
                crate::cells::cell_json_value(&value, dtype).map(|v| encode(&v)),
            ))
        })(),
        "payload.json_value" => (|| {
            let value = decode(input.get("value"), "value")?;
            Ok(outcome(
                crate::cells::json_value(&value).map(|v| encode(&v)),
            ))
        })(),
        "payload.class_label" => (|| {
            let value = decode(input.get("value"), "value")?;
            let dtype = text(input, "dtype")?;
            Ok(outcome(
                crate::cells::class_label(&value, dtype).map(Value::String),
            ))
        })(),
        "payload.decimal_dtype_name" => series(input).map(|s| {
            answer(
                crate::cells::decimal_dtype_name(&s).map_or(Value::Null, Value::String),
                &[],
            )
        }),
        "payload.dtype_name" => {
            series(input).map(|s| outcome(crate::cells::dtype_name(&s).map(Value::String)))
        }
        "payload.target_json_value" => (|| {
            let value = decode(input.get("value"), "value")?;
            let task_type = text(input, "task_type")?;
            Ok(outcome(
                crate::cells::target_json_value(task_type, &value).map(|v| encode(&v)),
            ))
        })(),
        "decompose.as_literal" => (|| {
            let value = decode(input.get("value"), "value")?;
            Ok(outcome(
                crate::literal::as_literal(&value, None).map(Value::String),
            ))
        })(),
        "tabular.two_whole_numbers" => series(input).map(|series| {
            outcome(crate::tabular::holds_two_whole_numbers(&series).map(Value::Bool))
        }),
        "client.ranked" => client_ranked(input),
        _ => return None,
    };
    Some(answered)
}

/// `client.ranked`: how a ranking orders its candidates, from each
/// candidate's probability of the positive class per row.
fn client_ranked(input: &Map<String, Value>) -> Result<String, InputError> {
    let PyVal::List(candidates) = decode(input.get("candidates"), "candidates")? else {
        return Err(bad("candidates is a list"));
    };
    let scores = input
        .get("scores")
        .and_then(Value::as_array)
        .ok_or_else(|| bad("scores is a list"))?;
    let mut answers = Vec::new();
    for per_candidate in scores {
        let rows = per_candidate
            .as_array()
            .ok_or_else(|| bad("a candidate's scores are a list"))?;
        let mut items = Vec::new();
        for (row, score) in rows.iter().enumerate() {
            items.push(Item {
                row_index: Some(PyVal::Int(row as i128)),
                id: Some(PyVal::Str(row.to_string())),
                probabilities: Some(vec![(
                    PyVal::Str("True".to_string()),
                    PyVal::decode(score).map_err(bad)?,
                )]),
                ..Item::default()
            });
        }
        answers.push(Result_::new(
            "kumo-relational",
            "",
            items,
            vec!["prediction".to_string(), "probabilities".to_string()],
        ));
    }
    let top = input.get("top").and_then(Value::as_i64);
    let outputs: Option<Vec<String>> = input.get("outputs").and_then(Value::as_array).map(|o| {
        o.iter()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect()
    });
    let merged = crate::result::ranked(&answers, &candidates, top, outputs.as_deref(), Vec::new());
    Ok(outcome(merged.map(|result| {
        Value::Array(
            result
                .items
                .iter()
                .map(|item| {
                    json!({
                        "row_index": item.row_index.as_ref().map_or(Value::Null, encode),
                        "prediction": item.prediction.as_ref().map_or(Value::Null, encode),
                        "rankings": Value::Array(
                            item.rankings
                                .as_ref()
                                .map(|r| {
                                    r.iter()
                                        .map(|entry| encode(&PyVal::Dict(entry.clone())))
                                        .collect()
                                })
                                .unwrap_or_default(),
                        ),
                    })
                })
                .collect(),
        )
    })))
}

#[cfg(test)]
mod diagnostic_presentation_tests {
    use super::*;
    use sdm_schema::diagnostics;

    #[test]
    fn fixture_and_public_warning_projections_are_separate() {
        let warning = diagnostics::warning(diagnostics::VIEW_CONVERSION_WARNING, "view".into());
        let public = crate::error::warning_json(&warning);
        assert_eq!(
            public,
            json!({"category":"sdm_connectors._warnings.ViewConversionWarning","message":"view"})
        );
        let fixture: Value =
            serde_json::from_str(&answer(json!(7), std::slice::from_ref(&warning))).unwrap();
        assert_eq!(fixture["output"], 7);
        assert_eq!(
            fixture["warnings"],
            json!([sdm_schema::diagnostic_compat::fixture_warning(&warning)])
        );
        assert_ne!(fixture["warnings"][0], public);
        assert!(!fixture.to_string().contains(diagnostics::INTERNAL_WARNING));
        let refused = Diagnostic::refusal("builtins.ValueError", None, "refused");
        let fixture: Value = serde_json::from_str(&refusal(&refused, &[warning])).unwrap();
        assert_eq!(
            fixture["error"],
            json!({"class":"builtins.ValueError","code":"ValueError","message":"refused","details_keys":[]})
        );
    }
}
