// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The scripted model endpoint, in Rust: the one written rule of
//! `contract/runners/scripted_endpoint.py`, which
//! `contract/scripted-endpoint.json` holds both to (`tests/scripted_endpoint.rs`
//! here, `contract/tests/test_scripted_endpoint.py` there).
//!
//! For `POST /v1/predictions` with a JSON object body, the answer is `200`
//! with one prediction per row of `predict.instance_table`, in row order:
//! `{"id": <the row's instance_id>, "row_index": <its position>,
//! "prediction": <p>}`, where an integer id is answered as the digits the
//! request wrote and text as itself, and `p` is `1.0` for a regression and
//! the first of `task.target.classes` (text) for any other kind. A body
//! that is not JSON (`NaN` and `Infinity` included) is `400 not_json`; one
//! the rule cannot read otherwise is `400` naming the reason; any other
//! method or path is `404 not_found`.
//!
//! The pure-Rust host calls the rule in process ([`answer`]): it has no
//! transport of its own to exercise, so no HTTP server or client is built
//! into the test kit (decision E46).

use serde_json::value::RawValue;
use serde_json::{Value, json};

/// The path the client posts a prediction to.
pub const PREDICTIONS: &str = "/v1/predictions";

/// Why a request the rule cannot read is refused, as the 400 body names it.
pub const REASONS: [&str; 7] = [
    "not_json",
    "not_an_object",
    "no_instance_table",
    "not_arrays",
    "no_instance_id",
    "bad_instance_id",
    "no_classes",
];

/// The rule: the status and JSON body answering one request.
pub fn answer(method: &str, path: &str, body: &[u8]) -> (u16, Value) {
    if method != "POST" || path != PREDICTIONS {
        return (404, json!({"error": "not_found"}));
    }
    match predictions(body) {
        Ok(predictions) => (
            200,
            json!({
                "object": "prediction.response",
                "id": "r1",
                "model": "kumo-relational",
                "metadata": {},
                "predictions": predictions,
            }),
        ),
        Err(reason) => (400, json!({"error": reason})),
    }
}

fn predictions(body: &[u8]) -> Result<Vec<Value>, &'static str> {
    // serde_json, as JSON itself, refuses NaN and Infinity.
    let payload: Value = serde_json::from_slice(body).map_err(|_| "not_json")?;
    let Value::Object(payload) = payload else {
        return Err("not_an_object");
    };
    let table = match payload.get("predict") {
        Some(Value::Object(predict)) => match predict.get("instance_table") {
            Some(Value::Object(table)) => table,
            _ => return Err("no_instance_table"),
        },
        _ => return Err("no_instance_table"),
    };
    let (Some(Value::Array(columns)), Some(Value::Array(_))) =
        (table.get("columns"), table.get("rows"))
    else {
        return Err("not_arrays");
    };
    if table.get("format") != Some(&Value::String("arrays".into())) {
        return Err("not_arrays");
    }
    let Some(at) = columns
        .iter()
        .position(|c| c == &Value::String("instance_id".into()))
    else {
        return Err("no_instance_id");
    };
    let p = prediction(payload.get("task"))?;
    // The rows again, each cell its text as written, so an id is answered
    // as its token; the body is JSON, so this reading cannot fail.
    let rows = raw_rows(body).ok_or("not_json")?;
    let mut answered = Vec::with_capacity(rows.len());
    for (index, row) in rows.iter().enumerate() {
        let Some(row) = row else {
            return Err("not_arrays");
        };
        let Some(cell) = row.get(at) else {
            return Err("not_arrays");
        };
        answered.push(json!({
            "id": instance_id(cell.get())?,
            "row_index": index,
            "prediction": p,
        }));
    }
    Ok(answered)
}

fn prediction(task: Option<&Value>) -> Result<Value, &'static str> {
    let task = match task {
        Some(Value::Object(task)) => task,
        _ => return Err("no_classes"),
    };
    if task.get("kind") == Some(&Value::String("regression".into())) {
        return Ok(json!(1.0));
    }
    match task.get("target") {
        Some(Value::Object(target)) => match target.get("classes") {
            Some(Value::Array(classes)) => match classes.first() {
                Some(Value::String(first)) => Ok(Value::String(first.clone())),
                _ => Err("no_classes"),
            },
            _ => Err("no_classes"),
        },
        _ => Err("no_classes"),
    }
}

/// An instance id as the answer names it: an integer token's digits as
/// written, or the text.
fn instance_id(raw: &str) -> Result<Value, &'static str> {
    if raw.starts_with('"') {
        return serde_json::from_str::<String>(raw)
            .map(Value::String)
            .map_err(|_| "bad_instance_id");
    }
    let digits = raw.strip_prefix('-').unwrap_or(raw);
    if !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit()) {
        return Ok(Value::String(raw.to_owned()));
    }
    Err("bad_instance_id")
}

/// `predict.instance_table.rows`, each row's cells as their raw text, or
/// `None` for a row that is not an array.
fn raw_rows(body: &[u8]) -> Option<Vec<Option<Vec<&RawValue>>>> {
    #[derive(serde::Deserialize)]
    struct Body<'a> {
        #[serde(borrow)]
        predict: Predict<'a>,
    }
    #[derive(serde::Deserialize)]
    struct Predict<'a> {
        #[serde(borrow)]
        instance_table: Table<'a>,
    }
    #[derive(serde::Deserialize)]
    struct Table<'a> {
        #[serde(borrow)]
        rows: Vec<&'a RawValue>,
    }
    let parsed: Body<'_> = serde_json::from_slice(body).ok()?;
    Some(
        parsed
            .predict
            .instance_table
            .rows
            .into_iter()
            .map(|row| serde_json::from_str::<Vec<&RawValue>>(row.get()).ok())
            .collect(),
    )
}
