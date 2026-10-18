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
//! Told to answer probabilities ([`answer_with_probabilities`]; T2.8's
//! ranking cases, as a model answers how sure it is), a classification
//! call whose `output.fields` names `probabilities` is answered with them:
//! for row `i` of call `k`, `d = (3k + 5i) mod 9 + 1`, the first class
//! `(10 - d) / 10`, the last `d / 10` and any other `0.0`, the prediction
//! the last class where `d > 5`, else the first.
//!
//! Told to answer quantiles ([`answer_as_asked`]; T2.9's forecast cases, as
//! a model answers the spread of a regression), a regression call whose
//! `output.fields` names `quantiles` is answered with them beside the
//! rule's prediction: for row `i` of call `k`, with `d` as above,
//! `{"0.1": d - 1, "0.5": d, "0.9": d + 1}`, each a float.
//!
//! Told to answer explanations ([`answer_told`]; T2.13's cases, as a model
//! explains the entity it is asked about), a call whose `output.fields`
//! names `explanation` is answered with one on each item: for row `i` of
//! call `k`, with `d` as above, `{"format": "scripted", "summary": "row <i>
//! of call <k>", "score": d / 10, "rank": d, "path": [k, i], "detail":
//! {"call": k, "row_index": i, "note": null}}`. Told to answer item
//! metadata, each item of every call answered by the rule carries
//! `"metadata": {"call": k, "row_index": i, "source": "scripted"}`.
//!
//! Calls are numbered from zero in the order received, and an endpoint
//! told to fail a call answers that one `500` with `{"error":
//! "scripted_failure", "call": <its number>}` and every other call by the
//! rule ([`Endpoint`], [`answer_call`]; T2.1a, for T2.8c's multi-call
//! cases).
//!
//! The pure-Rust host calls the rule in process ([`answer`]): it has no
//! transport of its own to exercise, so no HTTP server or client is built
//! into the test kit (decision E46).
//!
//! A test may stand a script of its own in for the rule on its thread
//! ([`answered_by`]): each model call the host makes there is then
//! answered by the script, given the call's number (from zero) and body,
//! with a status and the answer's bytes as they travel, so an answer the
//! rule never gives (probabilities, a `NaN` token, a failed call) reaches
//! the core as a recorded answer does (T2.8's ranking vectors).

use std::cell::RefCell;

use serde_json::value::RawValue;
use serde_json::{Value, json};

/// A script answering a model call: its number, from zero, and body, to a
/// status and the answer's bytes.
type Script = Box<dyn FnMut(u64, &[u8]) -> (u16, Vec<u8>)>;

thread_local! {
    /// The script answering this thread's model calls, and how many it has
    /// answered, while [`answered_by`] runs.
    static SCRIPT: RefCell<Option<(Script, u64)>> = const { RefCell::new(None) };
}

/// Takes the script off the thread when [`answered_by`]'s run ends,
/// whether it returns or panics.
struct Unset;

impl Drop for Unset {
    fn drop(&mut self) {
        SCRIPT.with(|script| script.borrow_mut().take());
    }
}

/// Runs `run` with each model call the host makes on this thread answered
/// by `script` (the call's number, from zero, and body, to a status and
/// the answer's bytes) in place of the rule; afterwards the rule answers
/// again.
pub fn answered_by<T>(
    script: impl FnMut(u64, &[u8]) -> (u16, Vec<u8>) + 'static,
    run: impl FnOnce() -> T,
) -> T {
    SCRIPT.with(|slot| *slot.borrow_mut() = Some((Box::new(script), 0)));
    let _unset = Unset;
    run()
}

/// The answer to a model call with `body` from the script this thread
/// runs under ([`answered_by`]), numbering the call; none where it runs
/// under none, and the rule answers.
pub fn scripted(body: &[u8]) -> Option<(u16, Vec<u8>)> {
    SCRIPT.with(|slot| {
        let mut slot = slot.borrow_mut();
        let (script, calls) = slot.as_mut()?;
        let call = *calls;
        *calls += 1;
        Some(script(call, body))
    })
}

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

/// The status of the call an endpoint is told to fail.
pub const FAILED: u16 = 500;

/// The endpoint over several calls: each request it answers is a call,
/// numbered from zero, and the call numbered `fail_call` fails.
#[derive(Debug, Clone, Default)]
pub struct Endpoint {
    fail_call: Option<u64>,
    calls: u64,
}

impl Endpoint {
    /// An endpoint failing the call numbered `fail_call`, or none.
    pub fn new(fail_call: Option<u64>) -> Endpoint {
        Endpoint {
            fail_call,
            calls: 0,
        }
    }

    /// The answer to the next call, numbering it.
    pub fn answer(&mut self, method: &str, path: &str, body: &[u8]) -> (u16, Value) {
        let call = self.calls;
        self.calls += 1;
        answer_call(method, path, body, call, self.fail_call)
    }

    /// How many calls it has answered.
    pub fn calls(&self) -> u64 {
        self.calls
    }
}

/// The rule for the request that is call number `call`, the call numbered
/// `fail_call` failed.
pub fn answer_call(
    method: &str,
    path: &str,
    body: &[u8],
    call: u64,
    fail_call: Option<u64>,
) -> (u16, Value) {
    if fail_call == Some(call) {
        return (FAILED, json!({"error": "scripted_failure", "call": call}));
    }
    answer(method, path, body)
}

/// What an endpoint is told to answer beside the rule's prediction (a
/// case's `input.probabilities`, `input.quantiles`, `input.explanation`
/// and `input.item_metadata`); the default is none of them.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Told {
    /// A classification asking for probabilities is given them.
    pub probabilities: bool,
    /// A regression asking for quantiles is given them.
    pub quantiles: bool,
    /// A call asking for an explanation is given one on each item.
    pub explanation: bool,
    /// Each item answered carries its metadata.
    pub item_metadata: bool,
}

/// The rule: the status and JSON body answering one request.
pub fn answer(method: &str, path: &str, body: &[u8]) -> (u16, Value) {
    answered(method, path, body, 0, Told::default())
}

/// The rule answering call number `call` of an endpoint told to answer
/// probabilities: a classification asking for them is given them.
pub fn answer_with_probabilities(method: &str, path: &str, body: &[u8], call: u64) -> (u16, Value) {
    answer_as_asked(method, path, body, call, true, false)
}

/// The rule answering call number `call` of an endpoint told whether to
/// answer probabilities and whether to answer quantiles (a case's
/// `input.probabilities` and `input.quantiles`): a classification asking
/// for probabilities is given them where `probabilities` says so, and a
/// regression asking for quantiles is given them where `quantiles` does.
pub fn answer_as_asked(
    method: &str,
    path: &str,
    body: &[u8],
    call: u64,
    probabilities: bool,
    quantiles: bool,
) -> (u16, Value) {
    answer_told(
        method,
        path,
        body,
        call,
        Told {
            probabilities,
            quantiles,
            ..Told::default()
        },
    )
}

/// The rule answering call number `call` of an endpoint told what
/// [`Told`] says to answer beside the rule's prediction.
pub fn answer_told(method: &str, path: &str, body: &[u8], call: u64, told: Told) -> (u16, Value) {
    answered(method, path, body, call, told)
}

fn answered(method: &str, path: &str, body: &[u8], call: u64, told: Told) -> (u16, Value) {
    if method != "POST" || path != PREDICTIONS {
        return (404, json!({"error": "not_found"}));
    }
    match predictions(body, call, told) {
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

fn predictions(body: &[u8], call: u64, told: Told) -> Result<Vec<Value>, &'static str> {
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
    // The call's number where the endpoint answers a classification's
    // probabilities and this call asks for them.
    let asks = |field: &str| {
        payload
            .get("output")
            .and_then(|o| o.get("fields"))
            .and_then(Value::as_array)
            .is_some_and(|fields| fields.iter().any(|f| f == field))
    };
    let classes: Vec<String> = payload
        .get("task")
        .and_then(|t| t.get("target"))
        .and_then(|t| t.get("classes"))
        .and_then(Value::as_array)
        .map(|c| {
            c.iter()
                .filter_map(Value::as_str)
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default();
    let sure =
        (told.probabilities && asks("probabilities") && p.is_string() && !classes.is_empty())
            .then_some(call);
    // The call's number where the endpoint answers a regression's quantiles
    // and this call asks for them.
    let spread = (told.quantiles && asks("quantiles") && !p.is_string()).then_some(call);
    // Whether the endpoint answers an explanation and this call asks for
    // one.
    let explains = told.explanation && asks("explanation");
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
        let mut item = json!({
            "id": instance_id(cell.get())?,
            "row_index": index,
            "prediction": p,
        });
        if let Some(call) = sure {
            let (prediction, given) = how_sure(&classes, call, index);
            item["prediction"] = prediction;
            item["probabilities"] = given;
        }
        if let Some(call) = spread {
            item["quantiles"] = quantiles(call, index);
        }
        if explains {
            item["explanation"] = explained(call, index);
        }
        if told.item_metadata {
            item["metadata"] = json!({"call": call, "row_index": index, "source": "scripted"});
        }
        answered.push(item);
    }
    Ok(answered)
}

/// Row `index` of call `call`'s prediction and probabilities, as an
/// endpoint told to answer probabilities gives them.
fn how_sure(classes: &[String], call: u64, index: usize) -> (Value, Value) {
    let index = u64::try_from(index).unwrap_or(u64::MAX);
    let d = (3 * u128::from(call) + 5 * u128::from(index)) % 9 + 1;
    let d = u32::try_from(d).unwrap_or(1);
    let mut given = serde_json::Map::new();
    for class in classes {
        given.insert(class.clone(), json!(0.0));
    }
    let (first, last) = (&classes[0], &classes[classes.len() - 1]);
    given.insert(first.clone(), json!(f64::from(10 - d) / 10.0));
    given.insert(last.clone(), json!(f64::from(d) / 10.0));
    let prediction = if d > 5 { last } else { first };
    (Value::String(prediction.clone()), Value::Object(given))
}

/// Row `index` of call `call`'s quantiles, as an endpoint told to answer
/// quantiles gives them.
fn quantiles(call: u64, index: usize) -> Value {
    let index = u64::try_from(index).unwrap_or(u64::MAX);
    let d = (3 * u128::from(call) + 5 * u128::from(index)) % 9 + 1;
    let d = f64::from(u32::try_from(d).unwrap_or(1));
    json!({"0.1": d - 1.0, "0.5": d, "0.9": d + 1.0})
}

/// Row `index` of call `call`'s explanation, as an endpoint told to answer
/// explanations gives it.
fn explained(call: u64, index: usize) -> Value {
    let index = u64::try_from(index).unwrap_or(u64::MAX);
    let d = (3 * u128::from(call) + 5 * u128::from(index)) % 9 + 1;
    let d = u32::try_from(d).unwrap_or(1);
    json!({
        "format": "scripted",
        "summary": format!("row {index} of call {call}"),
        "score": f64::from(d) / 10.0,
        "rank": d,
        "path": [call, index],
        "detail": {"call": call, "row_index": index, "note": null},
    })
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
