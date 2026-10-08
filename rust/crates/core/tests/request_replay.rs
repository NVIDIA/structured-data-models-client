// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! P85.payloads, in Rust: every recording of `tests/parity/requests/` (or of
//! `$SDM_REQUEST_RECORDINGS`) replayed without Python
//! (`tests/rust_core/request/CONTRACT.md`, "Recordings", "In Rust").
//!
//! Each call is made with the Rust API the binding wraps
//! ([`sdm_core::host_api::call`]), from its recorded arguments. A run is
//! driven step by step: each step must be the recorded one (the queries and
//! their parameters, the call to the model and its body parsed, the warnings
//! before it), each recorded outcome is submitted, and the value it is done
//! with, or the error it raised, must be the recorded one. Every request id
//! the core makes is written alike before anything is compared.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_schema::Schema;
use sdm_core::host_api::{Called, Refused, call};
use sdm_core::plain::Plain;
use sdm_core::{Diagnostic, Outcome, QueryResult, Run, Step};
use serde_json::{Map, Value, json};

const CALLS_FILE: &str = "calls.json";
const TABLES_DIR: &str = "tables";

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

/// Every recording directory under `root`, sorted.
fn recordings(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                if path.file_name().is_some_and(|n| n == TABLES_DIR) && dir == root {
                    continue;
                }
                stack.push(path);
            } else if path.file_name().is_some_and(|n| n == CALLS_FILE) {
                out.push(dir.clone());
            }
        }
    }
    out.sort();
    out
}

/// `text` with each request id the core makes written alike.
fn normalize_text(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < bytes.len() {
        if text[i..].starts_with("sdm-")
            && bytes.len() >= i + 16
            && bytes[i + 4..i + 16]
                .iter()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(b))
        {
            out.push_str("sdm-<request-id>");
            i += 16;
            continue;
        }
        let c = text[i..].chars().next().expect("a character");
        out.push(c);
        i += c.len_utf8();
    }
    out
}

fn normalize_json(value: &Value) -> Value {
    match value {
        Value::String(s) => Value::String(normalize_text(s)),
        Value::Array(items) => Value::Array(items.iter().map(normalize_json).collect()),
        Value::Object(map) => Value::Object(
            map.iter()
                .map(|(k, v)| (normalize_text(k), normalize_json(v)))
                .collect(),
        ),
        other => other.clone(),
    }
}

/// A recorded table, by its file's name.
fn read_table(root: &Path, sha: &str) -> Result<(Arc<Schema>, Vec<RecordBatch>), String> {
    let path = root.join(TABLES_DIR).join(format!("{sha}.arrow"));
    let file = fs::File::open(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let reader = arrow_ipc::reader::FileReader::try_new(file, None)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    let schema = reader.schema();
    let batches = reader
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| format!("{}: {e}", path.display()))?;
    Ok((schema, batches))
}

fn base64_decode(text: &str) -> Result<Vec<u8>, String> {
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = Vec::new();
    let mut buffer = 0u32;
    let mut bits = 0;
    for c in text.bytes() {
        if c == b'=' {
            break;
        }
        let value = ALPHABET.iter().position(|&a| a == c).ok_or("not base64")? as u32;
        buffer = (buffer << 6) | value;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((buffer >> bits) as u8);
            buffer &= (1 << bits) - 1;
        }
    }
    Ok(out)
}

/// A recorded value `V` as plain data.
fn decode(value: &Value, root: &Path) -> Result<Plain, String> {
    Ok(match value {
        Value::Null => Plain::None,
        Value::Bool(b) => Plain::Bool(*b),
        Value::Number(n) => match n.as_i64() {
            Some(i) => Plain::Int(i),
            None => Plain::Float(n.as_f64().unwrap_or(f64::NAN)),
        },
        Value::String(s) => Plain::Str(s.clone()),
        Value::Array(items) => Plain::List(
            items
                .iter()
                .map(|v| decode(v, root))
                .collect::<Result<_, _>>()?,
        ),
        Value::Object(map) => {
            if map.len() == 1 {
                let (tag, body) = map.iter().next().expect("one entry");
                let sha = || body.as_str().ok_or_else(|| format!("{tag} names a file"));
                match tag.as_str() {
                    "$table" => {
                        let (schema, batches) = read_table(root, sha()?)?;
                        return Ok(Plain::Table(schema, batches));
                    }
                    "$batch" => {
                        let (schema, batches) = read_table(root, sha()?)?;
                        let batch = arrow_select::concat::concat_batches(&schema, &batches)
                            .map_err(|e| e.to_string())?;
                        return Ok(Plain::Batch(batch));
                    }
                    "$array" | "$chunked" => {
                        let (schema, batches) = read_table(root, sha()?)?;
                        let column = schema.index_of("values").map_err(|e| e.to_string())?;
                        let chunks: Vec<_> =
                            batches.iter().map(|b| b.column(column).clone()).collect();
                        if tag == "$array" {
                            let refs: Vec<&dyn arrow_array::Array> =
                                chunks.iter().map(|c| c.as_ref()).collect();
                            let array =
                                arrow_select::concat::concat(&refs).map_err(|e| e.to_string())?;
                            return Ok(Plain::Array(array));
                        }
                        return Ok(Plain::Chunked(
                            schema.field(column).data_type().clone(),
                            chunks,
                        ));
                    }
                    "$tuple" => {
                        let items = body.as_array().ok_or("a tuple is a list")?;
                        return Ok(Plain::Tuple(
                            items
                                .iter()
                                .map(|v| decode(v, root))
                                .collect::<Result<_, _>>()?,
                        ));
                    }
                    "$bytes" => {
                        return Ok(Plain::Bytes(base64_decode(
                            body.as_str().ok_or("bytes are base64")?,
                        )?));
                    }
                    "$int" => {
                        return Ok(Plain::BigInt(
                            body.as_str().ok_or("an integer's digits")?.to_string(),
                        ));
                    }
                    "$float" => {
                        return Ok(Plain::Float(match body.as_str() {
                            Some("nan") => f64::NAN,
                            Some("inf") => f64::INFINITY,
                            Some("-inf") => f64::NEG_INFINITY,
                            _ => return Err("a float that is not one".to_string()),
                        }));
                    }
                    "$dict" => {
                        let inner = body.as_object().ok_or("a $dict is an object")?;
                        return Ok(Plain::Dict(
                            inner
                                .iter()
                                .map(|(k, v)| Ok((k.clone(), decode(v, root)?)))
                                .collect::<Result<_, String>>()?,
                        ));
                    }
                    _ => {}
                }
            }
            Plain::Dict(
                map.iter()
                    .map(|(k, v)| Ok((k.clone(), decode(v, root)?)))
                    .collect::<Result<_, String>>()?,
            )
        }
    })
}

/// One table, concatenated.
fn one_batch(schema: &Arc<Schema>, batches: &[RecordBatch]) -> Result<RecordBatch, String> {
    arrow_select::concat::concat_batches(schema, batches).map_err(|e| e.to_string())
}

/// Whether two tables hold the same columns, of the same types and field
/// metadata, with equal values.
fn same_table(want: &RecordBatch, got: &RecordBatch, at: &str) -> Result<(), String> {
    if want.num_rows() != got.num_rows() {
        return Err(format!(
            "{at}: {} rows, where the recording has {}",
            got.num_rows(),
            want.num_rows()
        ));
    }
    let (ws, gs) = (want.schema(), got.schema());
    let names = |s: &Schema| {
        s.fields()
            .iter()
            .map(|f| f.name().clone())
            .collect::<Vec<_>>()
    };
    if names(&ws) != names(&gs) {
        return Err(format!(
            "{at}: the columns are {:?}, where the recording has {:?}",
            names(&gs),
            names(&ws)
        ));
    }
    for (i, (wf, gf)) in ws.fields().iter().zip(gs.fields()).enumerate() {
        if wf.data_type() != gf.data_type() {
            return Err(format!(
                "{at}: column {} is {}, where the recording has {}",
                wf.name(),
                gf.data_type(),
                wf.data_type()
            ));
        }
        if wf.metadata() != gf.metadata() {
            return Err(format!("{at}: column {}'s metadata differs", wf.name()));
        }
        if want.column(i).to_data() != got.column(i).to_data() {
            return Err(format!("{at}: column {}'s values differ", wf.name()));
        }
    }
    Ok(())
}

/// Whether `got` is the recorded `want`: by value and JSON type, request ids
/// written alike, tables by their content.
fn same(want: &Plain, got: &Plain, at: &str) -> Result<(), String> {
    let differ = || {
        Err(format!(
            "{at}: got {}, where the recording has {}",
            describe(got),
            describe(want)
        ))
    };
    match (want, got) {
        (Plain::None, Plain::None) => Ok(()),
        (Plain::Bool(a), Plain::Bool(b)) if a == b => Ok(()),
        (Plain::Int(a), Plain::Int(b)) if a == b => Ok(()),
        (Plain::BigInt(a), Plain::BigInt(b)) if a == b => Ok(()),
        (Plain::Float(a), Plain::Float(b)) if a.to_bits() == b.to_bits() || a == b => Ok(()),
        (Plain::Str(a), Plain::Str(b)) => {
            let (a, b) = (normalize_text(a), normalize_text(b));
            if a == b {
                return Ok(());
            }
            // JSON text compared as what it says, where both are JSON.
            if let (Ok(x), Ok(y)) = (
                serde_json::from_str::<Value>(&a),
                serde_json::from_str::<Value>(&b),
            ) && (x.is_object() || x.is_array())
            {
                return if x == y {
                    Ok(())
                } else {
                    Err(format!("{at}: {}", first_difference(&x, &y, "")))
                };
            }
            differ()
        }
        (Plain::Bytes(a), Plain::Bytes(b)) if a == b => Ok(()),
        (Plain::List(a), Plain::List(b)) | (Plain::Tuple(a), Plain::Tuple(b)) => {
            if a.len() != b.len() {
                return differ();
            }
            for (i, (x, y)) in a.iter().zip(b).enumerate() {
                same(x, y, &format!("{at}[{i}]"))?;
            }
            Ok(())
        }
        (Plain::Dict(a), Plain::Dict(b)) => {
            let keys = |d: &Vec<(String, Plain)>| {
                d.iter().map(|(k, _)| normalize_text(k)).collect::<Vec<_>>()
            };
            if keys(a) != keys(b) {
                return differ();
            }
            for ((k, x), (_, y)) in a.iter().zip(b) {
                same(x, y, &format!("{at}.{k}"))?;
            }
            Ok(())
        }
        (Plain::Table(ws, wb), Plain::Table(gs, gb)) => {
            same_table(&one_batch(ws, wb)?, &one_batch(gs, gb)?, at)
        }
        (Plain::Batch(a), Plain::Batch(b)) => same_table(a, b, at),
        (Plain::Array(a), Plain::Array(b)) if a.to_data() == b.to_data() => Ok(()),
        (Plain::Chunked(dt, a), Plain::Chunked(dt2, b))
            if dt == dt2
                && a.len() == b.len()
                && a.iter().zip(b).all(|(x, y)| x.to_data() == y.to_data()) =>
        {
            Ok(())
        }
        _ => differ(),
    }
}

/// A short account of a value, for a message.
fn describe(value: &Plain) -> String {
    let text = match value.to_json() {
        Some(json) => json.to_string(),
        None => format!("{value:?}"),
    };
    text.chars().take(200).collect()
}

fn warning_json(warning: &Diagnostic) -> Value {
    json!({
        "category": warning.category.clone().unwrap_or_default(),
        "message": normalize_text(&warning.message),
    })
}

fn warnings_json(warnings: &[Diagnostic]) -> Value {
    Value::Array(warnings.iter().map(warning_json).collect())
}

/// A refusal as the binding raises it (`CoreError`), as recorded: compared
/// on what the record says of it.
fn error_matches(want: &Value, error: &Diagnostic, warnings: &[Diagnostic]) -> Result<(), String> {
    let shown = match &error.code {
        Some(code) => format!("{code}: {}", error.message),
        None => error.message.clone(),
    };
    let mut attributes = Map::new();
    attributes.insert("code".into(), json!(error.code));
    attributes.insert("message".into(), json!(normalize_text(&error.message)));
    attributes.insert("error_class".into(), json!(error.class));
    attributes.insert(
        "details".into(),
        normalize_json(&Value::Object((*error.details).clone())),
    );
    attributes.insert("status_code".into(), json!(error.status));
    attributes.insert("warnings".into(), warnings_json(warnings));
    let want_message = want.get("message").and_then(Value::as_str).unwrap_or("");
    if normalize_text(want_message) != normalize_text(&shown) {
        return Err(format!(
            "the error says {:?}, and the recording {:?}",
            normalize_text(&shown),
            normalize_text(want_message)
        ));
    }
    if let Some(Value::Object(recorded)) = want.get("attributes") {
        for (key, got) in &attributes {
            let Some(want_value) = recorded.get(key) else {
                continue;
            };
            let want_value = match key.as_str() {
                "warnings" => Value::Array(
                    want_value
                        .as_array()
                        .map(|ws| {
                            ws.iter()
                                .map(|w| {
                                    json!({
                                        "category": w.get("category").cloned().unwrap_or(Value::Null),
                                        "message": normalize_text(w.get("message").and_then(Value::as_str).unwrap_or("")),
                                    })
                                })
                                .collect()
                        })
                        .unwrap_or_default(),
                ),
                _ => normalize_json(want_value),
            };
            if want_value != *got {
                return Err(format!(
                    "the error's {key} is {got}, and the recording's {want_value}"
                ));
            }
        }
    }
    Ok(())
}

/// What a host submits for a recorded outcome.
fn outcome_of(want: &Value, root: &Path) -> Result<Outcome, String> {
    if let Some(data) = want.get("data") {
        return match decode(data, root)? {
            Plain::Table(schema, batches) => QueryResult::try_new(schema, batches)
                .map(Outcome::Data)
                .map_err(|e| e.to_string()),
            Plain::Batch(batch) => Ok(Outcome::Data(QueryResult::from_batch(batch))),
            _ => Err("a query's data that is not a table".to_string()),
        };
    }
    if let Some(error) = want.get("error") {
        return Ok(Outcome::Error {
            sqlstate: error
                .get("sqlstate")
                .and_then(Value::as_str)
                .map(str::to_string),
            vendor_code: error.get("vendor_code").and_then(Value::as_i64),
            message: error
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
        });
    }
    if let Some(text) = want.get("response").and_then(Value::as_str) {
        return Ok(Outcome::Response(text.to_string()));
    }
    if let Some(text) = want.get("model_error").and_then(Value::as_str) {
        return Ok(Outcome::ModelError(text.to_string()));
    }
    Err(format!("an outcome the replay cannot read: {want}"))
}

/// A statement's parameters as a transcript step encodes them.
fn params_json(params: &[sdm_core::Param]) -> Value {
    if params.is_empty() {
        return Value::Null;
    }
    Value::Array(
        params
            .iter()
            .map(|p| match p {
                sdm_core::Param::Null => Value::Null,
                sdm_core::Param::Bool(b) => json!(b),
                sdm_core::Param::Int(i) => json!(i),
                sdm_core::Param::Float(x) if x.is_nan() => json!({"$float": "nan"}),
                sdm_core::Param::Float(x) if x.is_infinite() => {
                    json!({"$float": if *x > 0.0 { "inf" } else { "-inf" }})
                }
                sdm_core::Param::Float(x) => json!(x),
                sdm_core::Param::Text(s) => json!(s),
            })
            .collect(),
    )
}

fn historical_done(function: &str, mut value: Plain) -> Result<Plain, String> {
    if function == "materialize_run"
        && let Plain::Tuple(items) = &mut value
        && let Some(Plain::Str(state)) = items.first_mut()
    {
        let mut decoded: Value = serde_json::from_str(state).map_err(|error| error.to_string())?;
        sdm_schema::diagnostic_compat::historical_materialize(&mut decoded);
        *state = serde_json::to_string(&decoded).map_err(|error| error.to_string())?;
    }
    Ok(value)
}

/// Drive `run` through the recorded `steps`.
fn drive(
    function: &str,
    mut run: Box<dyn Run<Plain> + Send>,
    steps: &[Value],
    root: &Path,
) -> Result<(), String> {
    for (i, want) in steps.iter().enumerate() {
        let at = format!("step {i}");
        let kind = want.get("kind").and_then(Value::as_str).unwrap_or("");
        let step = run.next();
        let warnings = run.take_warnings();
        let step = match step {
            Err(error) => {
                if kind != "raised" {
                    return Err(format!("{at} raised {error}, where it gave a {kind} step"));
                }
                return error_matches(want.get("error").unwrap_or(&Value::Null), &error, &warnings)
                    .map_err(|e| format!("{at}: {e}"));
            }
            Ok(step) => step,
        };
        let recorded_warnings = normalize_json(want.get("warnings").unwrap_or(&json!([])));
        let given = warnings_json(&warnings);
        if recorded_warnings != given {
            return Err(format!(
                "{at}: the warnings are {given}, and the recording's {recorded_warnings}"
            ));
        }
        match (step, kind) {
            (Step::RunQueries(queries), "run_queries") => {
                let got: Vec<Value> = queries
                    .iter()
                    .map(|q| json!({"sql": normalize_text(&q.sql), "params": params_json(&q.params)}))
                    .collect();
                let recorded = normalize_json(want.get("queries").unwrap_or(&json!([])));
                if recorded != Value::Array(got.clone()) {
                    return Err(format!(
                        "{at}: the queries are {}, and the recording's {}",
                        Value::Array(got),
                        recorded
                    ));
                }
                let outcomes = want
                    .get("outcomes")
                    .and_then(Value::as_array)
                    .ok_or_else(|| format!("{at}: no outcomes recorded"))?;
                if outcomes.len() != queries.len() {
                    return Err(format!(
                        "{at}: the recording answers another number of queries"
                    ));
                }
                for (query, outcome) in queries.iter().zip(outcomes) {
                    run.submit(query.id, outcome_of(outcome, root)?)
                        .map_err(|e| format!("{at}: submitting: {e}"))?;
                }
            }
            (Step::CallModel(call), "call_model") => {
                let body = match &call.body {
                    None => Value::Null,
                    Some(text) => normalize_json(
                        &serde_json::from_str::<Value>(text)
                            .map_err(|e| format!("{at}: the body is not JSON: {e}"))?,
                    ),
                };
                let got = json!({
                    "endpoint": call.endpoint.name(),
                    "session_id": call.session_id,
                    "body": body,
                });
                let recorded = normalize_json(&json!({
                    "endpoint": want.get("endpoint").cloned().unwrap_or(Value::Null),
                    "session_id": want.get("session_id").cloned().unwrap_or(Value::Null),
                    "body": want.get("body").cloned().unwrap_or(Value::Null),
                }));
                if got != recorded {
                    return Err(format!(
                        "{at}: the call differs from the recording's: {}",
                        first_difference(&recorded, &got, "")
                    ));
                }
                let outcome = want
                    .get("outcome")
                    .ok_or_else(|| format!("{at}: no outcome recorded"))?;
                run.submit(call.id, outcome_of(outcome, root)?)
                    .map_err(|e| format!("{at}: submitting: {e}"))?;
            }
            (Step::Done(value), "done") => {
                let recorded = decode(want.get("value").unwrap_or(&Value::Null), root)?;
                let value = historical_done(function, value)?;
                return same(&recorded, &value, &format!("{at}: the done value"));
            }
            (step, kind) => {
                let found = match step {
                    Step::RunQueries(_) => "run_queries",
                    Step::CallModel(_) => "call_model",
                    Step::Done(_) => "done",
                };
                return Err(format!("{at} is a {found} step, not a {kind} one"));
            }
        }
    }
    Err("the run went on past its recorded steps".to_string())
}

/// Where two values first differ, for a message.
fn first_difference(want: &Value, got: &Value, at: &str) -> String {
    match (want, got) {
        (Value::Object(a), Value::Object(b)) => {
            for (k, v) in a {
                match b.get(k) {
                    None => return format!("{at}.{k} is missing"),
                    Some(w) if w != v => return first_difference(v, w, &format!("{at}.{k}")),
                    _ => {}
                }
            }
            for k in b.keys() {
                if !a.contains_key(k) {
                    return format!("{at}.{k} is extra");
                }
            }
            format!("{at} differs")
        }
        (Value::Array(a), Value::Array(b)) if a.len() == b.len() => {
            for (i, (v, w)) in a.iter().zip(b).enumerate() {
                if v != w {
                    return first_difference(v, w, &format!("{at}[{i}]"));
                }
            }
            format!("{at} differs")
        }
        _ => {
            let show = |v: &Value| {
                let text = v.to_string();
                text.chars().take(200).collect::<String>()
            };
            format!("{at}: recorded {}, got {}", show(want), show(got))
        }
    }
}

/// Replay one recording: its calls, and every way they differ.
fn replay(directory: &Path, root: &Path) -> Result<(usize, Vec<String>), String> {
    let text = fs::read_to_string(directory.join(CALLS_FILE)).map_err(|e| e.to_string())?;
    let document: Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
    let name = document
        .get("scenario")
        .and_then(Value::as_str)
        .unwrap_or("?")
        .to_string();
    let calls = document
        .get("calls")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let host = sdm_core::host_api::HostInfo::assumed();
    let mut problems = Vec::new();
    for (at, recorded) in calls.iter().enumerate() {
        let function = recorded
            .get("function")
            .and_then(Value::as_str)
            .unwrap_or("");
        let where_ = format!("{name}: call {at} ({function})");
        let args = recorded
            .get("args")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .map(|v| decode(v, root))
                    .collect::<Result<Vec<_>, _>>()
            })
            .transpose()
            .map_err(|e| format!("{where_}: {e}"))?
            .unwrap_or_default();
        match call(function, &args, &host) {
            Err(Refused { error, warnings }) => match recorded.get("raised") {
                None => {
                    problems.push(format!("{where_} raised {error}"));
                    break;
                }
                Some(want) => {
                    if let Err(e) = error_matches(want, &error, &warnings) {
                        problems.push(format!("{where_}: {e}"));
                    }
                }
            },
            Ok(_) if recorded.get("raised").is_some() => {
                problems.push(format!("{where_} did not raise"));
            }
            Ok(Called::Value(value)) => {
                let want = decode(recorded.get("returned").unwrap_or(&Value::Null), root)?;
                if let Err(e) = same(&want, &value, "the value returned") {
                    problems.push(format!("{where_}: {e}"));
                }
            }
            Ok(Called::Run(run)) => {
                let steps = recorded
                    .get("steps")
                    .and_then(Value::as_array)
                    .cloned()
                    .unwrap_or_default();
                if let Err(e) = drive(function, run, &steps, root) {
                    problems.push(format!("{where_}: {e}"));
                    break;
                }
            }
        }
    }
    Ok((calls.len(), problems))
}

#[test]
fn every_recording_replays_through_the_core() {
    let root = std::env::var("SDM_REQUEST_RECORDINGS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| repo().join("tests/parity/requests"));
    let dirs = recordings(&root);
    assert!(!dirs.is_empty(), "no recordings under {}", root.display());
    let mut calls = 0;
    let mut problems = Vec::new();
    for directory in &dirs {
        match replay(directory, &root) {
            Ok((count, found)) => {
                calls += count;
                problems.extend(found);
            }
            Err(error) => problems.push(format!("{}: {error}", directory.display())),
        }
    }
    println!("request_replay: {} recordings, {calls} calls", dirs.len());
    assert!(
        problems.is_empty(),
        "{} problem(s):\n{}",
        problems.len(),
        problems
            .iter()
            .take(30)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
}

#[test]
fn historical_done_is_confined_to_materialize_state() {
    let cause = sdm_schema::diagnostics::cause_link("TaskReferenceError", "missing");
    let state = json!({"refused":{"chain":[cause]},"payload":cause}).to_string();
    let value = Plain::Tuple(vec![Plain::Str(state.clone()), Plain::Str(state.clone())]);
    let unchanged = historical_done("query_run", value.clone()).unwrap();
    same(&value, &unchanged, "other operation").unwrap();
    let Plain::Tuple(converted) = historical_done("materialize_run", value).unwrap() else {
        panic!("materialize tuple");
    };
    let decoded: Value = serde_json::from_str(converted[0].as_str().unwrap()).unwrap();
    assert_eq!(
        decoded["refused"]["chain"][0]["class"],
        "sdm_relational_engine.rfm.diagnostics.TaskReferenceError"
    );
    assert_eq!(decoded["payload"], cause);
    assert_eq!(converted[1].as_str(), Some(state.as_str()));
}
