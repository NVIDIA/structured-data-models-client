// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! T2.8's exits on `rust-host` (the package's test author's): ranking, end
//! to end (the design's T2.8a and T2.8b rows), beside
//! `contract/tests/test_t28_ranking.py`, which holds `python-core` to the
//! same cases and vectors and the cases to the rows. On this host no
//! Python client decomposes anything, so a ranking answered here is one
//! the core splits, asks and puts back together (`sdm-request`, design
//! choice 6).
//!
//! - `rust_host_answers_each_case_as_the_reference_does_on_duckdb` and
//!   `_on_sqlite` (T2.8a, "S1 ranking cases at materialize, both hosts"):
//!   each T2.8 case (`rank.*`, `forecast.*`) run in process through its
//!   `until` (SQLite's data loaded by the contract loader), each model call
//!   answered by the scripted endpoint's rule and kept: the outcome
//!   (result and code) python-ref's committed line's, ended in the core
//!   operation of the reference's stage, never by a rule this package
//!   lifts (and `rank.batched` by the rule `batched_part`), the core's own
//!   `materialize` run, and every request the core wrote python-ref's
//!   (the `payload` rule: each but its request id, in order). rust-host's
//!   Postgres pass replays python-core's fetches (KO2-20), which
//!   `cross-host (postgres)` holds to python-core.
//! - `rust_host_answers_each_vector_as_the_reference_does` (T2.8b, "S1
//!   ranking cases at predict, both hosts"): each vector of
//!   `contract/ranking-vectors.json` a case can make (a case's
//!   `materialize` arguments, no edit of the payload, the default outputs),
//!   run on DuckDB over the data of `rank.top-k`, each model call answered
//!   with the bytes the reference's endpoint answered that call: as many
//!   calls as the reference made, and its outcome: the reference's code,
//!   never by a rule this package lifts (an endpoint's refusal refused, its
//!   HTTP status held on python-core, since this host keeps text fields
//!   only), or its result, row by
//!   row: each row's index and id, a ranking's candidates best first with
//!   their scores (contract v26's `rankings`, each the JSON text of the
//!   candidate, and `scores`, in that order), and an answer's own
//!   `scores` and `rankings` and its prediction as the reference gives
//!   them back.
//!
//! Gating: `contract/tests/t28-pending.json`'s `pending` names, as
//! `rust:<case key>` and `rust:vector:<name>`, the keys that failed when
//! written; its `authored` `rust:` keys are this file's. Each pending key
//! must still fail (strict); once one passes it fails, naming itself, until
//! it is taken out.

#![allow(
    clippy::disallowed_methods,
    reason = "test code: reads the recorded cases and vectors"
)]

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::rc::Rc;

use sdm_testkit::bases::Bases;
use sdm_testkit::case::{self, Case};
use sdm_testkit::host::{self, Answer, Observed};
use sdm_testkit::{bases, endpoint};
use serde_json::{Value, json};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

fn read_json(path: &Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

/// The pending file's `list` names of this host.
fn named(list: &str) -> BTreeSet<String> {
    read_json(&root().join("contract/tests/t28-pending.json"))[list]
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

/// The T2.8 cases authored for `database`, by id.
fn case_ids(database: &str) -> Vec<String> {
    named("authored")
        .into_iter()
        .filter(|k| !k.starts_with("vector:"))
        .filter_map(|k| match k.split_once('@') {
            Some((id, on)) if on == database => Some(id.to_owned()),
            None if database == "duckdb" => Some(k),
            _ => None,
        })
        .collect()
}

fn the_bases() -> Bases {
    bases::load(&root().join("contract/bases.yaml")).unwrap()
}

fn load_case(id: &str) -> Case {
    let mut loaded = case::load(&[root().join(format!("contract/cases/s1/{id}.json"))]).unwrap();
    loaded.remove(0).case
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

/// Runs `run` with each model call answered by the scripted endpoint's
/// rule (a classification's probabilities given where the case asks for
/// them, `input.probabilities`), keeping each call's body as the runners record a payload (each
/// number as its text): the run's result and the bodies, in order.
fn with_the_rule<T>(probabilities: bool, run: impl FnOnce() -> T) -> (T, Vec<Value>) {
    let bodies: Rc<RefCell<Vec<Value>>> = Rc::default();
    let kept = Rc::clone(&bodies);
    let out = endpoint::answered_by(
        move |call, body| {
            kept.borrow_mut().push(host::numbers_as_text(
                serde_json::from_slice(body).unwrap_or(Value::Null),
            ));
            let (status, answer) = if probabilities {
                endpoint::answer_with_probabilities("POST", endpoint::PREDICTIONS, body, call)
            } else {
                endpoint::answer("POST", endpoint::PREDICTIONS, body)
            };
            (status, serde_json::to_vec(&answer).unwrap())
        },
        run,
    );
    let bodies = bodies.borrow().clone();
    (out, bodies)
}

/// Why rust-host's run of a case does not answer as python-ref's line
/// `wanted`, if it does not.
fn case_problem(observed: &Observed, bodies: &[Value], wanted: &Value) -> Option<String> {
    if !observed.operations.iter().any(|o| o == "materialize") {
        return Some(format!(
            "the core's materialize did not run: {:?}",
            observed.operations
        ));
    }
    let (result, code) = observed.answer.outcome();
    // The call ended in the core operation of the reference's stage.
    let ended = observed.operations.last().map_or("", String::as_str);
    let stage = match wanted["outcome"]["stage"].as_str() {
        Some("from_source") => "describe_source",
        Some(other) => other,
        None => "",
    };
    let got = (json!(result), json!(code), json!(ended));
    let want = (
        wanted["outcome"]["result"].clone(),
        wanted["outcome"]["code"].clone(),
        json!(stage),
    );
    if got != want {
        return Some(format!(
            "{got:?} where python-ref has {want:?} ({:?})",
            observed.answer
        ));
    }
    if let Some(rule) = not_yet(&observed.answer) {
        return Some(format!("refused by the rule {rule}, which T2.8 lifts"));
    }
    if let Some((id, rule)) = NAMED_RULES
        .iter()
        .find(|(id, _)| Some(*id) == case_id(wanted))
    {
        let named = match &observed.answer {
            Answer::Error { fields, .. } => fields.get("rule").cloned(),
            _ => None,
        };
        if named.as_deref() != Some(rule) {
            return Some(format!(
                "{id} is refused by {named:?}, not by the rule {rule} that refuses it"
            ));
        }
    }
    let sent: Vec<Value> = wanted["public"]["payloads"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| without_request_id(&p["parsed"]))
        .collect();
    let written: Vec<Value> = bodies.iter().map(without_request_id).collect();
    let mut found = Vec::new();
    differences(&Value::Array(sent), &Value::Array(written), "", &mut found);
    (!found.is_empty()).then(|| format!("the requests differ: {}", found.join("; ")))
}

/// The rules of E92's class this package lifts (the design's T2.8a and
/// T2.8b rows), and `target_filter`, which a ranking's part needs.
const LIFTED: [&str; 7] = [
    "problem_type",
    "top_k",
    "prediction_statistic",
    "prediction_type",
    "response_field",
    "instance_id_type",
    "target_filter",
];

/// The rule a case's refusal names where the reference's code alone would
/// be met by another refusal: the batched-ranking refusal, the rule
/// `batched_part` the batched forecast is refused by (`main`'s 1fc5a686,
/// the multi-call package), and not the core's ceiling on the entities one
/// request predicts for, which also answers `UNSUPPORTED_REQUEST`.
const NAMED_RULES: [(&str, &str); 1] = [("rank.batched", "batched_part")];

/// The case id a python-ref line is keyed by, without its database.
fn case_id(line: &Value) -> Option<&str> {
    line["case"]
        .as_str()
        .map(|k| k.split('@').next().unwrap_or(k))
}

/// The rule a refusal names when it is one of [`LIFTED`]: a refusal that
/// agrees with the reference's code only by that rule's accident.
fn not_yet(answer: &Answer) -> Option<String> {
    match answer {
        Answer::Error { fields, .. } => fields
            .get("rule")
            .filter(|rule| LIFTED.contains(&rule.as_str()))
            .cloned(),
        _ => None,
    }
}

/// Each T2.8 case on `database` through rust-host, held to python-ref's
/// line: a key not pending must answer as the reference does, a pending
/// key must still differ.
fn answers_as_the_reference(database: &str) {
    let ids = case_ids(database);
    assert!(!ids.is_empty(), "no T2.8 case authored on {database}");
    let reference = reference();
    let pending = named("pending");
    let bases = the_bases();
    let mut problems = Vec::new();
    for id in ids {
        let key = key(&id, database);
        let case = load_case(&id);
        let probabilities = case.input.probabilities == Some(true);
        let (ran, bodies) = with_the_rule(probabilities, || match database {
            "sqlite" => host::run_sqlite(&case, &bases, &root()),
            _ => host::run(&case, &bases),
        });
        let found = match (ran, reference.get(&key)) {
            (_, None) => Some("no python-ref line".to_owned()),
            (Err(error), _) => Some(format!("this host failed: {error:#}")),
            (Ok(observed), Some(wanted)) => case_problem(&observed, &bodies, wanted),
        };
        match (found, pending.contains(&key)) {
            (Some(why), false) => problems.push(format!("{key}: {why}")),
            (None, true) => problems.push(format!(
                "{key} passes now: take rust:{key} out of \
                 contract/tests/t28-pending.json's pending"
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

// --- the vectors ------------------------------------------------------------------

/// Whether rust-host replays `vector`, as `test_t28_ranking.py`'s
/// `on_rust_host` says: a call a case can make over the S1 cases' data on
/// DuckDB, whose answers its JSON reader takes as Python's does.
fn on_rust_host(vector: &Value) -> bool {
    let allowed = ["indices", "anchor_time", "candidates", "top_k"];
    vector.get("setup").is_none()
        && vector.get("database").is_none()
        && vector["materialize"]
            .as_object()
            .unwrap()
            .keys()
            .all(|k| allowed.contains(&k.as_str()))
        && vector["edit"].is_null()
        && vector["predict"].as_object().unwrap().is_empty()
        && !vector["name"].as_str().unwrap().contains("nan-score")
}

/// The vector as a case over `rank.top-k`'s data: its query, and its
/// `materialize` arguments, which the core operations carry as the case
/// format writes them.
fn vector_case(vector: &Value) -> Case {
    let mut case = read_json(&root().join("contract/cases/s1/rank.top-k.json"));
    let name = vector["name"].as_str().unwrap();
    case["id"] = json!(format!("vector.{name}"));
    case["status"] = json!("diagnostic");
    case["input"]["pql"] = vector["query"].clone();
    case["input"]["materialize"] = vector["materialize"].clone();
    case["core"]["materialize"]["indices"] = Value::Array(
        vector["materialize"]["indices"]
            .as_array()
            .unwrap()
            .iter()
            .map(|i| json!({"t": "i64", "v": i.as_i64().unwrap().to_string()}))
            .collect(),
    );
    case["expect"] = json!({"outcome": "success", "stage": "predict"});
    serde_json::from_value(case).unwrap()
}

/// A recorded cell (`ranking_record.py`'s `_cell`) as the JSON value it
/// stands for: an integer, a float, text, a boolean, a list or an object.
fn plain(cell: &Value) -> Value {
    if let Some(d) = cell.get("int") {
        return serde_json::from_str(d.as_str().unwrap()).unwrap();
    }
    if let Some(bits) = cell.get("f64_bits") {
        let bits = u64::from_str_radix(bits.as_str().unwrap(), 16).unwrap();
        return json!(f64::from_bits(bits));
    }
    if let Some(text) = cell.get("str") {
        return text.clone();
    }
    if let Some(b) = cell.get("bool") {
        return b.clone();
    }
    if let Some(items) = cell.get("list") {
        return Value::Array(items.as_array().unwrap().iter().map(plain).collect());
    }
    if let Some(members) = cell.get("map") {
        return Value::Object(
            members
                .as_array()
                .unwrap()
                .iter()
                .map(|m| (m[0].as_str().unwrap().to_owned(), plain(&m[1])))
                .collect(),
        );
    }
    Value::Null
}

/// A result cell of this host (`host::result_rows`) as the JSON value it
/// stands for, a list's members each so.
fn plain_result(cell: &Value) -> Value {
    if let Some(d) = cell.get("i64") {
        return serde_json::from_str(d.as_str().unwrap()).unwrap();
    }
    if let Some(text) = cell.get("utf8") {
        return text.clone();
    }
    if let Some(items) = cell.get("list") {
        return Value::Array(items.as_array().unwrap().iter().map(plain_result).collect());
    }
    plain(cell)
}

/// Two values equal as the reference's reading holds them: numbers by
/// value, an integer and the double of the same value alike.
fn same(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => match (x.as_i64(), y.as_i64()) {
            (Some(i), Some(j)) => i == j,
            _ => x.as_f64() == y.as_f64(),
        },
        (Value::Array(x), Value::Array(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(p, q)| same(p, q))
        }
        (Value::Object(x), Value::Object(y)) => {
            x.len() == y.len() && x.iter().all(|(k, v)| y.get(k).is_some_and(|w| same(v, w)))
        }
        _ => a == b,
    }
}

/// This host's result as columns by name, each its plain values in row
/// order.
fn columns(predicted: &Value) -> BTreeMap<String, Vec<Value>> {
    let result = &predicted["result"];
    let names: Vec<String> = result["columns"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c["name"].as_str().unwrap().to_owned())
        .collect();
    let mut out: BTreeMap<String, Vec<Value>> = BTreeMap::new();
    for row in result["rows"].as_array().unwrap() {
        for (name, cell) in names.iter().zip(row.as_array().unwrap()) {
            out.entry(name.clone())
                .or_default()
                .push(plain_result(cell));
        }
    }
    out
}

/// Why this host's result `predicted` is not the reference's `expected`
/// frame, if it is not.
fn result_problem(vector: &Value, predicted: &Value, expected: &Value) -> Option<String> {
    let got = columns(predicted);
    let names: Vec<&str> = expected["columns"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c.as_str().unwrap())
        .collect();
    let rows = expected["rows"].as_array().unwrap();
    let column = |name: &str| -> Option<Vec<Value>> {
        let at = names.iter().position(|n| *n == name)?;
        Some(rows.iter().map(|r| plain(&r[at])).collect())
    };
    let ranking = vector["query"].as_str().unwrap().contains(" RANK TOP ");
    let n = rows.len();
    for (name, want) in [("row_index", column("row_index")), ("id", column("id"))] {
        let want = want.unwrap();
        let have = got.get(name).cloned().unwrap_or_default();
        // An id compared by its digits, whether laid out as an integer or
        // as text.
        let digits = |v: &Value| match v {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        };
        if have.len() != n || have.iter().map(digits).ne(want.iter().map(digits)) {
            return Some(format!("{name}: {have:?} where the reference has {want:?}"));
        }
    }
    if ranking {
        // Each row's candidates best first, with their scores, from the
        // reference's rankings ({candidate, score} each).
        let want = column("rankings").unwrap();
        let candidates = got.get("rankings").cloned().unwrap_or_default();
        let scores = got.get("scores").cloned().unwrap_or_default();
        for (row, ranked) in want.iter().enumerate() {
            let ranked = ranked.as_array().unwrap();
            let want_candidates: Vec<Value> =
                ranked.iter().map(|m| m["candidate"].clone()).collect();
            let want_scores: Vec<Value> = ranked.iter().map(|m| m["score"].clone()).collect();
            let have_candidates: Vec<Value> = candidates
                .get(row)
                .and_then(Value::as_array)
                .map(|c| {
                    c.iter()
                        .map(|t| {
                            serde_json::from_str(t.as_str().unwrap_or("")).unwrap_or(Value::Null)
                        })
                        .collect()
                })
                .unwrap_or_default();
            let have_scores = scores.get(row).cloned().unwrap_or(Value::Null);
            if !same(
                &Value::Array(want_candidates.clone()),
                &Value::Array(have_candidates.clone()),
            ) {
                return Some(format!(
                    "row {row}: candidates {have_candidates:?} where the reference ranks \
                     {want_candidates:?}"
                ));
            }
            if !same(&Value::Array(want_scores.clone()), &have_scores) {
                return Some(format!(
                    "row {row}: scores {have_scores} where the reference has {want_scores:?}"
                ));
            }
        }
        return None;
    }
    if let Some(want) = column("prediction") {
        let have = got.get("prediction").cloned().unwrap_or_default();
        if !same(&Value::Array(want.clone()), &Value::Array(have.clone())) {
            return Some(format!(
                "prediction: {have:?} where the reference has {want:?}"
            ));
        }
    }
    if let Some(want) = column("scores") {
        let have = got.get("scores").cloned().unwrap_or_default();
        if !same(&Value::Array(want.clone()), &Value::Array(have.clone())) {
            return Some(format!("scores: {have:?} where the reference has {want:?}"));
        }
    }
    if let Some(want) = column("rankings") {
        // An answer's own rankings, each member the JSON text of the
        // reference's item value.
        let have: Vec<Value> = got
            .get("rankings")
            .cloned()
            .unwrap_or_default()
            .iter()
            .map(|row| {
                Value::Array(
                    row.as_array()
                        .map(|members| {
                            members
                                .iter()
                                .map(|t| {
                                    serde_json::from_str(t.as_str().unwrap_or(""))
                                        .unwrap_or(Value::Null)
                                })
                                .collect()
                        })
                        .unwrap_or_default(),
                )
            })
            .collect();
        if !same(&Value::Array(want.clone()), &Value::Array(have.clone())) {
            return Some(format!(
                "rankings: {have:?} where the reference has {want:?}"
            ));
        }
    }
    None
}

/// Why rust-host's run of `vector` does not answer as the reference did,
/// if it does not.
fn vector_problem(vector: &Value, bases: &Bases) -> Option<String> {
    let expected = vector["expected"].clone();
    let calls = expected["calls"].as_array().unwrap().clone();
    let made = Rc::new(RefCell::new(0_usize));
    let counted = Rc::clone(&made);
    let case = vector_case(vector);
    let ran = endpoint::answered_by(
        move |number, _| {
            *counted.borrow_mut() += 1;
            usize::try_from(number)
                .ok()
                .and_then(|n| calls.get(n))
                .map_or_else(
                    || {
                        (
                            500,
                            br#"{"error": "a call the reference did not make"}"#.to_vec(),
                        )
                    },
                    |call| {
                        (
                            u16::try_from(call["status"].as_u64().unwrap()).unwrap(),
                            call["answer"].as_str().unwrap().as_bytes().to_vec(),
                        )
                    },
                )
        },
        || host::run(&case, bases),
    );
    let observed = match ran {
        Ok(observed) => observed,
        Err(error) => return Some(format!("this host failed: {error:#}")),
    };
    let want_calls = expected["calls"].as_array().unwrap().len();
    if *made.borrow() != want_calls {
        return Some(format!(
            "{} calls where the reference made {want_calls} ({:?})",
            made.borrow(),
            observed.answer
        ));
    }
    match (&observed.answer, expected["outcome"].as_str().unwrap()) {
        (Answer::Predicted(predicted), "result") => result_problem(vector, predicted, &expected),
        (answer @ Answer::Error { .. }, "refused") if not_yet(answer).is_some() => Some(format!(
            "refused by the rule {}, which T2.8 lifts",
            not_yet(answer).unwrap_or_default()
        )),
        (Answer::Error { code, fields, .. }, "refused") => {
            if expected["class"] == "NimRequestError" {
                // The endpoint's own refusal, after the calls the reference
                // made: its HTTP status is a number, which this host's
                // record (text fields only) does not keep, so the status the
                // user reads is held on python-core.
                None
            } else {
                (expected["code"].as_str() != Some(code.as_str())).then(|| {
                    format!(
                        "refused {code} where the reference refuses {}: {fields:?}",
                        expected["code"]
                    )
                })
            }
        }
        (answer, outcome) => Some(format!(
            "{answer:?} where the reference's outcome is {outcome} {}",
            expected["code"]
        )),
    }
}

#[test]
fn rust_host_answers_each_vector_as_the_reference_does() {
    let vectors = read_json(&root().join("contract/ranking-vectors.json"));
    let replayed: Vec<&Value> = vectors["vectors"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|v| on_rust_host(v))
        .collect();
    let authored: BTreeSet<String> = named("authored")
        .into_iter()
        .filter_map(|k| k.strip_prefix("vector:").map(str::to_owned))
        .collect();
    let names: BTreeSet<String> = replayed
        .iter()
        .map(|v| v["name"].as_str().unwrap().to_owned())
        .collect();
    assert_eq!(
        names, authored,
        "the vectors rust-host replays are not those authored"
    );
    let pending = named("pending");
    let bases = the_bases();
    let mut problems = Vec::new();
    for vector in replayed {
        let name = vector["name"].as_str().unwrap();
        let key = format!("vector:{name}");
        match (vector_problem(vector, &bases), pending.contains(&key)) {
            (Some(why), false) => problems.push(format!("{name}: {why}")),
            (None, true) => problems.push(format!(
                "{name} passes now: take rust:{key} out of \
                 contract/tests/t28-pending.json's pending"
            )),
            _ => {}
        }
    }
    assert!(problems.is_empty(), "{}", problems.join("\n"));
}

#[test]
fn a_script_answers_only_the_run_it_was_given() {
    // Control for the test kit's hook: a script answers its own run's
    // calls, numbered from zero, and the rule answers afterwards.
    let numbers = Rc::new(RefCell::new(Vec::new()));
    let seen = Rc::clone(&numbers);
    let answered = endpoint::answered_by(
        move |number, _| {
            seen.borrow_mut().push(number);
            (200, b"{}".to_vec())
        },
        || (endpoint::scripted(b"a"), endpoint::scripted(b"b")),
    );
    assert_eq!(
        answered,
        (Some((200, b"{}".to_vec())), Some((200, b"{}".to_vec())))
    );
    assert_eq!(*numbers.borrow(), vec![0, 1]);
    assert_eq!(endpoint::scripted(b"c"), None);
}
