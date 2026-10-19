// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! T2.13's exits on `rust-host` (the package's test author's), beside
//! `contract/tests/test_t213_options.py`, which holds `python-core` to the
//! same ones and the cases to what they name: the query and option long
//! tail, "`entity_filter`" (a fact table's column, which the validator
//! refuses; the filters the core answers, and ASSUMING's, are T2.13b's),
//! "`entity_ids`", "`assuming`", "`evaluate`", "`explain`",
//! "`target_transform_null`", "`inference_config_text`", "`response_field`
//! for explanations and item metadata" and U1, "cases per rule", both
//! hosts.
//!
//! - `rust_host_answers_each_case_as_the_reference_does_on_duckdb` and
//!   `_on_sqlite`: this host runs each T2.13 case in process (`host::run`,
//!   `host::run_sqlite` with the contract loader), and its answer is
//!   python-ref's: the outcome (result, and the code wherever the
//!   reference's refusal has one), the call it ended in (predict where the
//!   reference ended there; materialize, or the query's validation before
//!   it, and no predict, where it refused at materialize); for a success
//!   through predict, every call the core wrote, in order, the request
//!   python-ref's endpoint received (the `payload` rule: each but its
//!   request id: the context an entity filter or an assumption gathered,
//!   the query as sent, the inference options as read, the output fields
//!   asked), the result's scalar columns row by row as
//!   `contract/t213-option-vectors.json` holds them from the reference's
//!   `to_pandas()`, and, where the reference's result holds an explanation
//!   or its items metadata, each of them somewhere in what this host's
//!   `predict` gives back (as JSON, or as the text of JSON; a forecast's
//!   explanation step by step, in each row's explanation column), which the
//!   scripted endpoint answers where the case says (`input.explanation`,
//!   `input.item_metadata`; `endpoint::answer_told`). rust-host's Postgres
//!   pass replays python-core's fetches (KO2-20), which `cross-host
//!   (postgres)` holds to python-core.
//!
//! Gating: `contract/tests/t213-pending.json`'s `pending` names, as
//! `rust:<case key>`, the keys that failed when written; the cases are its
//! `authored` `rust:` keys. Each pending key must still fail (strict);
//! once one passes it fails, naming itself, until it is taken out.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use sdm_testkit::bases;
use sdm_testkit::case::{self, Case};
use sdm_testkit::host::{self, Answer};
use serde_json::{Value, json};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

fn read_json(path: &Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

/// The pending file's `list` names of this host, as case keys.
fn named(list: &str) -> BTreeSet<String> {
    read_json(&root().join("contract/tests/t213-pending.json"))[list]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|n| n.as_str().unwrap().strip_prefix("rust:"))
        .map(str::to_owned)
        .collect()
}

/// The case key of `id` on `database`, as the runners key it.
fn key(id: &str, database: &str) -> String {
    if database == "duckdb" {
        id.to_owned()
    } else {
        format!("{id}@{database}")
    }
}

/// The T2.13 cases authored for `database`, by id.
fn case_ids(database: &str) -> Vec<String> {
    named("authored")
        .into_iter()
        .filter_map(|k| match k.split_once('@') {
            Some((id, on)) if on == database => Some(id.to_owned()),
            None if database == "duckdb" => Some(k),
            _ => None,
        })
        .collect()
}

/// python-ref's committed lines, by key.
fn reference() -> BTreeMap<String, Value> {
    let env = std::fs::read_to_string(root().join("contract/reference-env.txt")).unwrap();
    let commit = env
        .lines()
        .find_map(|l| l.strip_prefix("reference: "))
        .unwrap()
        .trim()
        .to_owned();
    let text =
        std::fs::read_to_string(root().join(format!("contract/reference/{commit}.jsonl"))).unwrap();
    text.lines()
        .map(|line| {
            let o: Value = serde_json::from_str(line).unwrap();
            (o["case"].as_str().unwrap().to_owned(), o)
        })
        .collect()
}

/// The reference's vectors (`contract/tools/option_record.py`), by key.
fn vectors() -> Value {
    let all = read_json(&root().join("contract/t213-option-vectors.json"));
    assert_eq!(all["format"], "sdm-t213-option-vectors/1");
    assert_eq!(all["recorded_by"], "contract/tools/option_record.py");
    all["vectors"].clone()
}

/// A request as the `payload` rule compares it: its request id left out.
fn without_request_id(request: &Value) -> Value {
    let mut request = request.clone();
    if let Some(metadata) = request.get_mut("metadata").and_then(Value::as_object_mut) {
        metadata.remove("request_id");
    }
    request
}

/// Two JSON numbers equal as numbers (an integer exactly, else as the
/// double each reads as), as the Python comparison holds them.
fn same_number(a: &serde_json::Number, b: &serde_json::Number) -> bool {
    if let (Some(x), Some(y)) = (a.as_i64(), b.as_i64()) {
        return x == y;
    }
    if let (Some(x), Some(y)) = (a.as_u64(), b.as_u64()) {
        return x == y;
    }
    a.as_f64() == b.as_f64()
}

/// Where `got` differs from `want`, a few paths.
fn differences(want: &Value, got: &Value, at: &str, out: &mut Vec<String>) {
    if out.len() >= 8 {
        return;
    }
    match (want, got) {
        (Value::Object(w), Value::Object(g)) => {
            let names: BTreeSet<&String> = w.keys().chain(g.keys()).collect();
            for name in names {
                differences(
                    w.get(name).unwrap_or(&Value::Null),
                    g.get(name).unwrap_or(&Value::Null),
                    &format!("{at}/{name}"),
                    out,
                );
            }
        }
        (Value::Array(w), Value::Array(g)) if w.len() == g.len() => {
            for (i, (a, b)) in w.iter().zip(g).enumerate() {
                differences(a, b, &format!("{at}[{i}]"), out);
            }
        }
        (Value::Array(w), Value::Array(g)) => out.push(format!(
            "{at}: {} items where the reference has {}",
            g.len(),
            w.len()
        )),
        (Value::Number(a), Value::Number(b)) if same_number(a, b) => {}
        _ if want == got => {}
        _ => out.push(format!("{at}: {got} where the reference has {want}")),
    }
}

/// Whether `value` holds `wanted` anywhere: itself, a member or item at
/// any depth, or a text that reads as JSON holding it.
fn holds(value: &Value, wanted: &Value) -> bool {
    let mut found = Vec::new();
    differences(wanted, value, "", &mut found);
    if found.is_empty() {
        return true;
    }
    match value {
        Value::Object(members) => members.values().any(|v| holds(v, wanted)),
        Value::Array(items) => items.iter().any(|v| holds(v, wanted)),
        Value::String(text) => serde_json::from_str::<Value>(text)
            .ok()
            .is_some_and(|read| !read.is_string() && holds(&read, wanted)),
        _ => false,
    }
}

/// The calls the core made, from `predict`'s in-process answer: its
/// `calls` when it carries several, else its `body`.
fn calls(predicted: &Value) -> Vec<Value> {
    if let Value::Array(bodies) = &predicted["calls"] {
        return bodies.clone();
    }
    match &predicted["body"] {
        Value::Null => vec![],
        body => vec![body.clone()],
    }
}

/// A cell of the reference's `to_pandas()` row as this host writes a
/// result's: an integer's digits, a float's IEEE-754 bits, a boolean as
/// itself.
fn tagged(cell: &Value) -> Value {
    match cell {
        Value::Bool(b) => json!({"bool": b}),
        Value::Number(n) if n.is_i64() || n.is_u64() => json!({"i64": n.to_string()}),
        Value::Number(n) => {
            json!({"f64_bits": format!("{:016x}", n.as_f64().unwrap().to_bits())})
        }
        Value::String(text) => json!({"utf8": text}),
        other => other.clone(),
    }
}

/// The core's result rows laid out as the reference's columns, by name.
fn rows_as(predicted: &Value, names: &[String]) -> Result<Vec<Value>, String> {
    let columns: Vec<String> = predicted["result"]["columns"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|c| c["name"].as_str().unwrap_or_default().to_owned())
        .collect();
    let mut at = Vec::new();
    for name in names {
        match columns.iter().position(|c| c == name) {
            Some(i) => at.push(i),
            None => return Err(format!("no column {name} in {columns:?}")),
        }
    }
    Ok(predicted["result"]["rows"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|row| Value::Array(at.iter().map(|i| row[*i].clone()).collect()))
        .collect())
}

/// Why this host's `result` does not hold the reference's vector rows, in
/// the columns whose every value is a scalar (a ranking's lists are
/// python-core's to hold).
fn result_problem(predicted: &Value, vector: &Value) -> Option<String> {
    let result = &vector["result"];
    let names: Vec<String> = result["columns"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c.as_str().unwrap().to_owned())
        .collect();
    let rows = result["rows"].as_array().unwrap();
    let scalar: Vec<usize> = (0..names.len())
        .filter(|i| {
            rows.iter()
                .all(|row| !row[*i].is_array() && !row[*i].is_object())
        })
        .collect();
    let kept: Vec<String> = scalar.iter().map(|i| names[*i].clone()).collect();
    let got = match rows_as(predicted, &kept) {
        Ok(got) => got,
        Err(why) => return Some(why),
    };
    if got.len() != rows.len() {
        return Some(format!(
            "{} result rows where the reference has {}",
            got.len(),
            rows.len()
        ));
    }
    let want: Vec<Value> = rows
        .iter()
        .map(|row| Value::Array(scalar.iter().map(|i| tagged(&row[*i])).collect()))
        .collect();
    let mut found = Vec::new();
    differences(
        &Value::Array(want),
        &Value::Array(got),
        "/result",
        &mut found,
    );
    (!found.is_empty()).then(|| found.join("; "))
}

/// Why the rows of this host's forecast `result` do not each carry, in
/// their explanation column, the reference's explanation for their step
/// (`steps`, the reference's `by_forecast_step`, keyed by the step's
/// digits), or nothing where the reference has none for it. The rows are
/// in the reference's order ([`result_problem`] holds them so).
fn step_explanation_problem(predicted: &Value, vector: &Value, steps: &Value) -> Option<String> {
    let result = &vector["result"];
    let step_at = result["columns"]
        .as_array()
        .unwrap()
        .iter()
        .position(|c| c == "forecast_step")
        .expect("a forecast's result has forecast_step");
    let got = match rows_as(predicted, &["explanation".to_owned()]) {
        Ok(got) => got,
        Err(why) => return Some(why),
    };
    for (i, row) in result["rows"].as_array().unwrap().iter().enumerate() {
        let step = row[step_at].to_string();
        let cell = &got[i][0];
        match &steps[step.as_str()] {
            Value::Null if cell.is_null() => {}
            Value::Null => {
                return Some(format!(
                    "step {step}'s row has explanation {cell} where the reference has none"
                ));
            }
            wanted if !holds(cell, wanted) => {
                return Some(format!(
                    "step {step}'s row has explanation {cell} where the reference has {wanted}"
                ));
            }
            _ => {}
        }
    }
    None
}

/// Why this host's `observed` answer does not answer as python-ref's
/// `line` and `vector`, if it does not.
fn problem(observed: &host::Observed, line: &Value, vector: &Value) -> Option<String> {
    let (result, code) = observed.answer.outcome();
    let theirs = &line["outcome"];
    if result != theirs["result"] {
        return Some(format!(
            "{result} ({code:?}) where python-ref has {} ({})",
            theirs["result"], theirs["code"]
        ));
    }
    if !theirs["code"].is_null() && code.as_deref() != theirs["code"].as_str() {
        return Some(format!("{code:?} where python-ref has {}", theirs["code"]));
    }
    // The call it ended in: this host observes no stage, but the core
    // operations it ran; a query the validator refuses ends in
    // validate_query, before materialize.
    let ran = |op: &str| observed.operations.iter().any(|o| o == op);
    match theirs["stage"].as_str() {
        Some("predict") if !ran("predict") => {
            return Some(format!(
                "ended before predict ({code:?}) where python-ref ends in predict"
            ));
        }
        Some("materialize") if ran("predict") || !(ran("materialize") || ran("validate_query")) => {
            return Some("did not end in materialize, as python-ref does".to_owned());
        }
        _ => {}
    }
    if result != "success" || theirs["stage"] != "predict" {
        return None;
    }
    let Answer::Predicted(predicted) = &observed.answer else {
        return Some(format!("no answer of predict: {:?}", observed.answer));
    };
    let sent: Vec<Value> = line["public"]["payloads"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| without_request_id(&p["parsed"]))
        .collect();
    let written: Vec<Value> = calls(predicted).iter().map(without_request_id).collect();
    let mut found = Vec::new();
    differences(&Value::Array(sent), &Value::Array(written), "", &mut found);
    if !found.is_empty() {
        return Some(format!("the calls differ: {}", found.join("; ")));
    }
    if let Some(why) = result_problem(predicted, vector) {
        return Some(why);
    }
    if let Some(steps) = vector["explanation"].get("by_forecast_step") {
        // A forecast's whole explanation (`{"by_forecast_step": ...}`) is
        // on the result's Arrow schema metadata, which this host's runner
        // does not record (python-core's test holds the whole one to the
        // reference); here each step's explanation is held, in the rows'
        // explanation column, to the reference's for that step.
        if let Some(why) = step_explanation_problem(predicted, vector, steps) {
            return Some(why);
        }
    } else if !vector["explanation"].is_null() && !holds(predicted, &vector["explanation"]) {
        return Some(format!(
            "predict gives back no explanation {}",
            vector["explanation"]
        ));
    }
    for pair in vector["item_metadata"].as_array().into_iter().flatten() {
        if !pair[1].is_null() && !holds(predicted, &pair[1]) {
            return Some(format!("predict gives back no item metadata {}", pair[1]));
        }
    }
    None
}

/// Each T2.13 case on `database` through rust-host, held to python-ref's
/// line and vector: a key not pending must answer as the reference does, a
/// pending key must still differ.
fn answers_as_the_reference(database: &str) {
    let ids = case_ids(database);
    assert!(!ids.is_empty(), "no T2.13 case authored on {database}");
    let contract = root().join("contract");
    let bases = bases::load(&contract.join("bases.yaml")).unwrap();
    let reference = reference();
    let vectors = vectors();
    let pending = named("pending");
    let mut problems = Vec::new();
    for id in ids {
        let key = key(&id, database);
        let loaded = case::load(&[contract.join(format!("cases/s1/{id}.json"))]).unwrap();
        let case: &Case = &loaded[0].case;
        let observed = match database {
            "duckdb" => host::run(case, &bases),
            _ => host::run_sqlite(case, &bases, &root()),
        };
        let found = match (observed, reference.get(&key), vectors.get(&key)) {
            (Err(failure), ..) => Some(format!("host failure: {failure:#}")),
            (_, None, _) => Some("no python-ref line".to_owned()),
            (_, _, None) => Some("no python-ref vector".to_owned()),
            (Ok(observed), Some(line), Some(vector)) => problem(&observed, line, vector),
        };
        match (found, pending.contains(&key)) {
            (Some(why), false) => problems.push(format!("{key}: {why}")),
            (None, true) => problems.push(format!(
                "{key} passes now: take rust:{key} out of \
                 contract/tests/t213-pending.json's pending"
            )),
            _ => {}
        }
    }
    assert!(problems.is_empty(), "{}", problems.join("\n"));
}

#[test]
fn rust_host_answers_each_case_as_the_reference_does_on_duckdb() {
    answers_as_the_reference("duckdb");
}

#[test]
fn rust_host_answers_each_case_as_the_reference_does_on_sqlite() {
    answers_as_the_reference("sqlite");
}

#[test]
fn the_case_s_endpoint_fields_and_outputs_reach_this_host() {
    // Control, passing before the implementation: the case format's
    // explanation, item_metadata and outputs are read as written.
    let contract = root().join("contract/cases/s1");
    let read = |id: &str| case::load(&[contract.join(format!("{id}.json"))]).unwrap();
    let explained = read("metadata.items-explained");
    let input = &explained[0].case.input;
    assert_eq!(input.explanation, Some(true));
    assert_eq!(input.item_metadata, Some(true));
    assert_eq!(
        input.outputs.as_deref(),
        Some(&["prediction".to_owned(), "explanation".to_owned()][..])
    );
    let untold = read("explain.not-answered");
    assert_eq!(untold[0].case.input.explanation, None);
    assert_eq!(untold[0].case.input.item_metadata, None);
}

#[test]
fn holds_finds_a_value_at_any_depth_or_in_json_text() {
    // Control of the explanation and item metadata check: a value is
    // found as itself, nested, or inside a text of JSON, and not where it
    // differs.
    let wanted = json!({"call": 0, "row_index": 1, "source": "scripted"});
    assert!(holds(&wanted, &wanted));
    assert!(holds(
        &json!({"items": [{"metadata": wanted.clone()}]}),
        &wanted
    ));
    assert!(holds(&json!({"text": wanted.to_string()}), &wanted));
    assert!(!holds(
        &json!({"items": [{"metadata": {"call": 0, "row_index": 2, "source": "scripted"}}]}),
        &wanted
    ));
    assert!(!holds(&json!({"result": {"rows": []}}), &wanted));
}
