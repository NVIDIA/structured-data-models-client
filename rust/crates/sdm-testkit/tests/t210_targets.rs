// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! T2.10's exits on `rust-host` (the package's test author's): the PQL
//! target rules `aggregation`, `open_window`, `condition`,
//! `logical_target`, `target_filter` and `static_target`, beside
//! `contract/tests/test_t210_targets.py`, which holds `python-core` to the
//! same cases, to the reference's label vectors
//! (`contract/t210-target-vectors.json`) and the cases to what they name.
//!
//! - `rust_host_answers_each_case_as_the_reference_does_on_duckdb` and
//!   `_on_sqlite`: this host runs each of T2.10's cases (`target.*`, one
//!   or two per rule; `rank.decimal-key` and `rank.odd-column`, the
//!   rankings carried from T2.8's and T2.18's reviews; and T2.18's
//!   filter role, `name.*-filter`) in process (`host::run`,
//!   `host::run_sqlite` with the contract loader), and its answer is
//!   python-ref's: the outcome (result, and the code wherever the
//!   reference's refusal has one: an endpoint's refusal is the transport's
//!   `NimRequestError` there, with none), the call it ended in (this host
//!   runs predict where the reference ended there, and no predict where it
//!   refused at materialize), and for a success through predict every
//!   call the core wrote, in order, the request python-ref's endpoint
//!   received (the `payload` rule: each but its request id), so each
//!   context row's label is the reference's. A case the reference refuses
//!   only for a column name it cannot say (`rank.odd-column`,
//!   `name.*-filter`) is answered through `predict`, as its twin is on the
//!   reference (KO2-36), and an approved register entry citing no
//!   temporary approval names its key, as T2.18's cases are held.
//!   rust-host's Postgres pass replays python-core's fetches (KO2-20),
//!   which `cross-host (postgres)` holds to python-core.
//!
//! Gating: `contract/tests/t210-pending.json`'s `pending` names, as
//! `rust:<case key>`, the keys that failed when written; the cases are its
//! `authored` `rust:` keys. Each pending key must still fail (strict); once
//! one passes it fails, naming itself, until it is taken out.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use sdm_testkit::bases;
use sdm_testkit::case::{self, Case};
use sdm_testkit::host::{self, Answer};
use serde_json::Value;

/// The cases the reference refuses only for a column name it cannot say:
/// the core answers them as their twins (KO2-36).
const NAMED: [&str; 4] = [
    "rank.odd-column",
    "name.backtick-filter",
    "name.dot-filter",
    "name.line-break-filter",
];

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

fn read_json(path: &Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

/// The pending file's `list` names of this host, as case keys.
fn named(list: &str) -> BTreeSet<String> {
    read_json(&root().join("contract/tests/t210-pending.json"))[list]
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

/// T2.10's cases authored for `database`, by id.
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

/// The decision rows that approve only until a package lands, as
/// `scripts/rust_core/register.py`'s `TEMPORARY` names them.
fn temporary() -> Vec<String> {
    let text = std::fs::read_to_string(root().join("scripts/rust_core/register.py")).unwrap();
    let line = text
        .lines()
        .find_map(|l| l.strip_prefix("TEMPORARY = ("))
        .expect("register.py names TEMPORARY");
    line.trim_end_matches(')')
        .split(',')
        .map(|r| r.trim().trim_matches('\'').to_owned())
        .filter(|r| !r.is_empty())
        .collect()
}

/// The register's umbrellas: approved as E92's Phase 2 class, they excuse
/// no case (T1.33).
const UMBRELLAS: [&str; 4] = ["P1B-1", "P1B-2", "P1B-3", "P1B-4"];

/// The approved entry, no umbrella and citing no temporary approval, whose
/// text names `wanted` (a line key in its `cases` or `case_outcomes`):
/// each entry read from the register's text, block by block.
fn registered(wanted: &str) -> Option<String> {
    let text = std::fs::read_to_string(root().join("docs/design/rust-core/register.yaml")).unwrap();
    let temporary = temporary();
    for block in text.split("\n- id: ").skip(1) {
        let id = block.lines().next().unwrap_or_default().trim().to_owned();
        let approved = block.lines().any(|l| l.trim_end() == "  status: approved");
        let row = block
            .lines()
            .find(|l| l.starts_with("  decision_ref:"))
            .and_then(|l| l.split("row: ").nth(1))
            .map(|r| r.trim_end_matches('}').trim().to_owned());
        let names = block.split(|c: char| {
            c.is_whitespace() || matches!(c, ',' | '[' | ']' | '{' | '}' | '\'' | '"')
        });
        if approved
            && !UMBRELLAS.contains(&id.as_str())
            && !row.is_some_and(|r| temporary.contains(&r))
            && names.into_iter().any(|n| n == wanted)
        {
            return Some(id);
        }
    }
    None
}

/// A request as the `payload` rule compares it: its request id left out.
fn without_request_id(request: &Value) -> Value {
    let mut request = request.clone();
    if let Some(metadata) = request.get_mut("metadata").and_then(Value::as_object_mut) {
        metadata.remove("request_id");
    }
    request
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

/// Why this host's `observed` answer does not answer case `id` as
/// python-ref's `line` does, if it does not.
fn problem(id: &str, key: &str, observed: &host::Observed, line: &Value) -> Option<String> {
    let (result, code) = observed.answer.outcome();
    let theirs = &line["outcome"];
    if NAMED.contains(&id) {
        if !matches!(observed.answer, Answer::Predicted(_)) {
            return Some(format!(
                "{:?} where the twin answers through predict",
                observed.answer
            ));
        }
        if theirs["result"] != "success" && registered(key).is_none() {
            return Some(format!(
                "answered where python-ref is {theirs}, and no approved, permanent entry names it"
            ));
        }
        return None;
    }
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
    // operations it ran, so a call the endpoint refused is reached through
    // predict, and a refusal in the materialize call runs no predict (the
    // query's validation, `validate_query`, is that call's too).
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
    (!found.is_empty()).then(|| format!("the calls differ: {}", found.join("; ")))
}

/// Each of T2.10's cases on `database` through rust-host, held to
/// python-ref's line: a key not pending must answer as the reference does,
/// a pending key must still differ.
fn answers_as_the_reference(database: &str) {
    let ids = case_ids(database);
    assert!(!ids.is_empty(), "no T2.10 case authored on {database}");
    let contract = root().join("contract");
    let bases = bases::load(&contract.join("bases.yaml")).unwrap();
    let reference = reference();
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
        let found = match (observed, reference.get(&key)) {
            (Err(failure), _) => Some(format!("host failure: {failure:#}")),
            (_, None) => Some("no python-ref line".to_owned()),
            (Ok(observed), Some(line)) => problem(&id, &key, &observed, line),
        };
        match (found, pending.contains(&key)) {
            (Some(why), false) => problems.push(format!("{key}: {why}")),
            (None, true) => problems.push(format!(
                "{key} passes now: take rust:{key} out of \
                 contract/tests/t210-pending.json's pending"
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
fn the_rankings_candidates_reach_this_host() {
    // Control, passing before the implementation: the two rankings' case
    // fields are read as written: candidates 0, 1 and 2 and probabilities
    // asked of the endpoint.
    let contract = root().join("contract/cases/s1");
    for id in ["rank.decimal-key", "rank.odd-column"] {
        let loaded = case::load(&[contract.join(format!("{id}.json"))]).unwrap();
        let given = &loaded[0].case.input;
        let candidates = given.materialize.candidates.as_ref().expect(id);
        assert_eq!(candidates.len(), 3, "{id}");
        assert!(given.pql.contains(" RANK TOP 2 "), "{id}");
        assert_eq!(given.probabilities, Some(true), "{id}");
    }
}
