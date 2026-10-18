// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! T2.8c's exits on `rust-host` (the package's test author's), beside
//! `contract/tests/test_t28c_multicall.py`, which holds `python-core` to
//! the same ones and the cases to what they name: multi-call `predict`
//! (`main`'s 1fc5a686), "larger requests split with a per-batch walk and
//! context", "`rows_per_call`", "rows numbered across calls", "the
//! 100,000-entity and 100-call ceilings", "the per-batch 30 MiB body
//! refusal", "the pre-send context check per call" and "the
//! batched-forecast refusal", both hosts.
//!
//! - `rust_host_answers_each_case_as_the_reference_does_on_duckdb` and
//!   `_on_sqlite`: this host runs each T2.8c case (`multicall.*`) in
//!   process (`host::run`, `host::run_sqlite` with the contract loader),
//!   the case's `rows_per_call` in the call's values and the scripted
//!   endpoint failing its `fail_call`, and its answer is python-ref's: the
//!   outcome (result, and the code wherever the reference's refusal has
//!   one: an endpoint's failure is the transport's `NimRequestError`
//!   there, with none, and the core's own code here, E45), the call it
//!   ended in (this host runs predict where the reference ended there, and
//!   no predict where it refused at materialize), an invalid
//!   `rows_per_call` refused with a field naming it; for a success
//!   through predict, every call the core wrote, in order, the request
//!   python-ref's endpoint received (the `payload` rule: each but its
//!   request id), and the result's rows (`row_index`, `id`, a forecast's
//!   `forecast_step` and `prediction`) those of
//!   `contract/t28c-multicall-vectors.json`, recorded from the reference's
//!   `to_pandas()`, so the rows are numbered across the calls. rust-host's
//!   Postgres pass replays python-core's fetches (KO2-20), which
//!   `cross-host (postgres)` holds to python-core.
//!
//! Gating: `contract/tests/t28c-pending.json`'s `pending` names, as
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
    read_json(&root().join("contract/tests/t28c-pending.json"))[list]
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

/// The T2.8c cases authored for `database`, by id.
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

/// The reference's vectors (`contract/tools/multicall_record.py`), by key.
fn vectors() -> Value {
    let all = read_json(&root().join("contract/t28c-multicall-vectors.json"));
    assert_eq!(all["format"], "sdm-t28c-multicall-vectors/1");
    assert_eq!(all["recorded_by"], "contract/tools/multicall_record.py");
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
/// result's: an integer's digits, a float's IEEE-754 bits.
fn tagged(cell: &Value) -> Value {
    match cell {
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

/// Why this host's `result` does not hold the reference's vector rows.
fn result_problem(predicted: &Value, vector: &Value) -> Option<String> {
    let result = &vector["result"];
    let names: Vec<String> = result["columns"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c.as_str().unwrap().to_owned())
        .collect();
    let rows = match rows_as(predicted, &names) {
        Ok(rows) => rows,
        Err(why) => return Some(why),
    };
    let count = result["count"].as_u64().unwrap() as usize;
    if rows.len() != count {
        return Some(format!(
            "{} result rows where the reference has {count}",
            rows.len()
        ));
    }
    let tag = |list: &Value| -> Vec<Value> {
        list.as_array()
            .unwrap()
            .iter()
            .map(|row| Value::Array(row.as_array().unwrap().iter().map(tagged).collect()))
            .collect()
    };
    let (want, got) = if result.get("rows").is_some() {
        (tag(&result["rows"]), rows)
    } else {
        let first = tag(&result["first"]);
        let last = tag(&result["last"]);
        let mut want = first.clone();
        want.extend(last.clone());
        let mut got = rows[..first.len()].to_vec();
        got.extend_from_slice(&rows[rows.len() - last.len()..]);
        (want, got)
    };
    let mut found = Vec::new();
    differences(
        &Value::Array(want),
        &Value::Array(got),
        "/result",
        &mut found,
    );
    (!found.is_empty()).then(|| found.join("; "))
}

/// Why this host's `observed` answer does not answer as python-ref's
/// `line` and `vector`, if it does not.
fn problem(id: &str, observed: &host::Observed, line: &Value, vector: &Value) -> Option<String> {
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
    // operations it ran, so a call failed by the endpoint is reached
    // through predict, and a refusal at materialize runs no predict.
    let ran = |op: &str| observed.operations.iter().any(|o| o == op);
    match theirs["stage"].as_str() {
        Some("predict") if !ran("predict") => {
            return Some(format!(
                "ended before predict ({code:?}) where python-ref ends in predict"
            ));
        }
        Some("materialize") if ran("predict") || !ran("materialize") => {
            return Some("did not end in materialize, as python-ref does".to_owned());
        }
        _ => {}
    }
    // An invalid rows_per_call is refused naming it, as the reference's
    // refusal does ("rows_per_call is 0, and a call scores ..."), never as
    // an option the core does not take.
    if id.starts_with("multicall.rows-per-call.") && result == "refusal" {
        let Answer::Error { fields, .. } = &observed.answer else {
            return Some(format!("no error: {:?}", observed.answer));
        };
        if !fields.values().any(|v| v.contains("rows_per_call"))
            || fields.values().any(|v| v == "not_an_option")
        {
            return Some(format!("the refusal names no rows_per_call: {fields:?}"));
        }
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
    result_problem(predicted, vector)
}

/// Each T2.8c case on `database` through rust-host, held to python-ref's
/// line and vector: a key not pending must answer as the reference does, a
/// pending key must still differ.
fn answers_as_the_reference(database: &str) {
    let ids = case_ids(database);
    assert!(!ids.is_empty(), "no T2.8c case authored on {database}");
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
            (Ok(observed), Some(line), Some(vector)) => problem(&id, &observed, line, vector),
        };
        match (found, pending.contains(&key)) {
            (Some(why), false) => problems.push(format!("{key}: {why}")),
            (None, true) => problems.push(format!(
                "{key} passes now: take rust:{key} out of \
                 contract/tests/t28c-pending.json's pending"
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
fn the_case_s_rows_per_call_and_failed_call_reach_this_host() {
    // Control, passing before the implementation: the case format's two
    // fields are read as written, `rows_per_call` of any scalar and size.
    let contract = root().join("contract/cases/s1");
    let read = |id: &str| case::load(&[contract.join(format!("{id}.json"))]).unwrap();
    let failing = read("multicall.fail-call-2");
    assert_eq!(failing[0].case.input.fail_call, Some(2));
    for (id, written) in [
        ("multicall.rows-per-call-3", "3"),
        ("multicall.rows-per-call.whole-float", "3.0"),
        ("multicall.rows-per-call.text", "\"3\""),
        ("multicall.rows-per-call.bool", "true"),
        ("multicall.rows-per-call.past-int64", "18446744073709551616"),
    ] {
        let loaded = read(id);
        let given = loaded[0].case.input.materialize.rows_per_call.as_ref();
        assert_eq!(given.map(|w| w.0.as_str()), Some(written), "{id}");
        assert_eq!(loaded[0].case.input.fail_call, None, "{id}");
    }
}
