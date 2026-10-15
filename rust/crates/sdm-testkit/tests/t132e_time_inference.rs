// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! T1.32e part 1's whole-table time inference on `rust-host`, held from
//! outside the implementation (the package's independent test author) to
//! what the reference gave: `contract/time-inference-vectors.json`,
//! recorded from the reference by `contract/tools/time_inference_record.py`,
//! which `contract/tests/test_t132e_time_inference.py` holds `python-core`
//! to as well, so the two hosts agree through it.
//!
//! - Brief, scope 1 ("infer the time column the way the reference does
//!   over the whole table", "lift the `time_whole_table` refusal"): each
//!   vector (a generated table `events` of 999, 1,000, 1,001 or 5,000 rows
//!   with timestamp, date, time-of-day or zoned columns that are not the
//!   key, nulls, values that differ only after the first 1,000 rows, with
//!   and without `time_columns`) runs through this host's
//!   `describe_source` with the vector's `time_columns`. The answer must
//!   be the reference's, as far as this host observes it: a success
//!   describing `events` with the reference's columns (each name, its
//!   dtype as the wire names it, its stype), key, time column and
//!   duplicate-key count; or a refusal with the reference's code (this
//!   host observes no stage, exception class or warning class; the
//!   Python test holds the warnings).
//! - Brief, scope 1 ("honour a named `time_columns`, no inference, no
//!   refusal"), the package's exit: every vector of 1,000 rows or more
//!   whose `time_columns` names a column the reference takes is a success
//!   whose time column is that column.
//! - A vector may differ from the reference only under an approved entry
//!   of `docs/design/rust-core/register.yaml` whose `refs` name it with the
//!   transition it makes (T1.32a's binding-transition form):
//!   `contract/time-inference-vectors.json#<id or glob> <from> -> <to>`,
//!   each side `success` or `refusal <CODE>`. A ref covers exactly the
//!   transition it declares; a proposed entry excuses nothing. The
//!   recorder's `register.required` excuses nothing by itself.
//! - Every vector ref of a proposed or approved entry names a transition
//!   in that form; a malformed one fails.
//!
//! - The owner's decision of 2026-10-13 (P1B-38 rejected): where the query
//!   over every row fails (a view of 200,000 rows whose time columns raise
//!   when read whole), the core decides from the preview as the reference
//!   does: the `fallback-*` vectors, held to the reference's success and
//!   time column here (the warning's class on the Python host).
//!
//! Gating: `contract/tests/t132e-pending.json`'s `pending["rust-host"]`
//! lists the vectors that failed here when the test was written, and
//! `pending["exits"]` the exit's (`rs:` ids). Each must still fail
//! (strict: one that passes must be taken out of the list), and the lists
//! may only shrink: they stay within `authored`. The implementer removes
//! ids as they pass and changes nothing else of this file, the vectors or
//! the recorder.

use std::collections::BTreeSet;
use std::path::PathBuf;

use sdm_contract::ops::Dtype;
use sdm_testkit::bases::{self, Bases};
use sdm_testkit::case::Case;
use sdm_testkit::host::{self, Answer, Observed};
use serde_json::{Value, json};

const REF: &str = "contract/time-inference-vectors.json#";
const TABLE: &str = "events";
const PREVIEW_ROWS: u64 = 1_000;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

fn read_json(path: &str) -> Value {
    let text = std::fs::read_to_string(root().join(path)).unwrap();
    serde_json::from_str(&text).unwrap()
}

fn vectors() -> Value {
    let all = read_json("contract/time-inference-vectors.json");
    assert_eq!(all["format"], "sdm-time-inference-vectors/1");
    assert_eq!(
        all["recorded_by"],
        "contract/tools/time_inference_record.py"
    );
    assert_eq!(all["register_ref"], REF);
    assert_eq!(all["table"], TABLE);
    assert_eq!(all["preview_rows"], PREVIEW_ROWS);
    all
}

fn ids(value: &Value) -> BTreeSet<String> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_owned())
        .collect()
}

fn pending(group: &str) -> BTreeSet<String> {
    ids(&read_json("contract/tests/t132e-pending.json")["pending"][group])
}

/// `*` matches any run of characters, every other character itself.
fn glob(pattern: &str, text: &str) -> bool {
    let p: Vec<char> = pattern.chars().collect();
    let t: Vec<char> = text.chars().collect();
    let (mut pi, mut ti, mut star, mut mark) = (0, 0, None, 0);
    while ti < t.len() {
        if pi < p.len() && p[pi] != '*' && p[pi] == t[ti] {
            pi += 1;
            ti += 1;
        } else if pi < p.len() && p[pi] == '*' {
            star = Some(pi);
            mark = ti;
            pi += 1;
        } else if let Some(s) = star {
            pi = s + 1;
            mark += 1;
            ti = mark;
        } else {
            return false;
        }
    }
    p[pi..].iter().all(|c| *c == '*')
}

/// One vector ref of a register entry, read: the glob, the transition's
/// start and end, or `None` when it names no transition in the form
/// `<glob> <from> -> <to>`, each side `success` or `refusal <CODE>`.
fn transition(item: &str) -> Option<(String, String, String)> {
    let rest = item.strip_prefix(REF)?;
    let (glob, rest) = rest.split_once(' ')?;
    let (old, new) = rest.split_once(" -> ")?;
    let class = |c: &str| {
        c == "success"
            || c.strip_prefix("refusal ").is_some_and(|code| {
                code.starts_with(|ch: char| ch.is_ascii_uppercase())
                    && code.chars().all(|ch| ch.is_ascii_uppercase() || ch == '_')
            })
    };
    (!glob.is_empty() && class(old) && class(new))
        .then(|| (glob.to_owned(), old.to_owned(), new.to_owned()))
}

/// A vector ref read: entry, status, glob, from, to.
type Transition = (String, String, String, String, String);

/// Each vector ref of the register's proposed or approved entries, and the
/// refs naming vectors with no transition in the form (malformed). Read
/// from the YAML's text, entry by entry (`- id:` at the margin), the `refs`
/// field alone, in its flow (`refs: [a, b]`) or block (`refs:` then
/// `  - a`) form.
fn transitions(text: &str) -> (Vec<Transition>, Vec<String>) {
    let mut out = Vec::new();
    let mut malformed = Vec::new();
    for block in text.split("\n- id: ").skip(1) {
        let id = block.lines().next().unwrap_or_default().trim().to_owned();
        let status = block
            .lines()
            .find_map(|l| l.strip_prefix("  status: "))
            .unwrap_or_default()
            .trim()
            .to_owned();
        if status != "proposed" && status != "approved" {
            continue;
        }
        let lines: Vec<&str> = block.lines().collect();
        let Some(at) = lines.iter().position(|l| l.starts_with("  refs:")) else {
            continue;
        };
        let mut refs: Vec<String> = Vec::new();
        let rest = lines[at].trim_start_matches("  refs:").trim();
        if let Some(flow) = rest.strip_prefix('[') {
            let mut flow = flow.to_owned();
            let mut next = at + 1;
            while !flow.contains(']') && next < lines.len() {
                flow.push(' ');
                flow.push_str(lines[next].trim());
                next += 1;
            }
            let inner = flow.split(']').next().unwrap_or_default();
            refs.extend(inner.split(',').map(|r| r.trim().to_owned()));
        } else {
            for line in &lines[at + 1..] {
                match line.strip_prefix("  - ") {
                    Some(item) => refs.push(item.trim().to_owned()),
                    None => break,
                }
            }
        }
        for item in refs
            .iter()
            .map(|r| r.trim_matches(|c| c == '\'' || c == '"'))
            .filter(|r| r.starts_with(REF))
        {
            match transition(item) {
                Some((glob, old, new)) => {
                    out.push((id.clone(), status.clone(), glob, old, new));
                }
                None => malformed.push(format!("{id}: {item}")),
            }
        }
    }
    (out, malformed)
}

/// The approved entry naming `id` with exactly the transition `old -> new`.
fn covered(id: &str, old: &str, new: &str, entries: &[Transition]) -> Option<String> {
    entries
        .iter()
        .find(|(_, status, pattern, from, to)| {
            status == "approved" && from == old && to == new && glob(pattern, id)
        })
        .map(|(entry, ..)| entry.clone())
}

/// `success` or `refusal <CODE>` of an answer in this host's terms (the
/// reference's unnamed exception, with no code, is `refusal UNNAMED`).
fn outcome_class(answer: &Value) -> String {
    if answer["outcome"] == "success" {
        "success".to_owned()
    } else {
        format!("refusal {}", answer["code"].as_str().unwrap_or("UNNAMED"))
    }
}

/// Whether this host's answer passes the vector: the reference's, or a
/// difference an approved entry declares exactly.
fn passes(id: &str, got: &Value, want: &Value, entries: &[Transition]) -> bool {
    if agrees(got, want) {
        return true;
    }
    let answered = got["outcome"] == "success" || got["outcome"] == "refusal";
    answered && covered(id, &outcome_class(want), &outcome_class(got), entries).is_some()
}

/// Whether this host's answer is the reference's, as far as it observes.
fn agrees(got: &Value, want: &Value) -> bool {
    if want["outcome"] == "refusal" {
        // An unnamed exception has no code: no core answer is it.
        return !want["code"].is_null()
            && got["outcome"] == "refusal"
            && got["code"] == want["code"];
    }
    got == want
}

/// The case this host runs for `vector`: `Schema.from_source` over its
/// table, `describe_source` given its `time_columns`, as the recorder's
/// `case_for` writes it.
fn the_case(vector: &Value) -> Case {
    let named = match &vector["time_columns"] {
        Value::Null => json!({}),
        other => other.clone(),
    };
    // The core's spec writes each time column as the contract's `Given`.
    let written: serde_json::Map<String, Value> = named
        .as_object()
        .into_iter()
        .flatten()
        .map(|(table, column)| (table.clone(), json!({"value": column})))
        .collect();
    let mut input = json!({
        "base": "shop",
        "setup": vector["setup"].clone(),
        "pql": format!("PREDICT COUNT({TABLE}.*, 0, 30, days) FOR EACH {TABLE}.event_id"),
        "materialize": {"indices": [0, 1], "anchor_time": "2024-03-01"},
    });
    // Tables named (a fallback vector's view, which is listed only when
    // named), each by its name and its source's name.
    let tables = vector["tables"].as_object();
    let named_tables: Vec<Value> = tables
        .into_iter()
        .flatten()
        .map(|(name, source)| json!({"name": name, "source": source}))
        .collect();
    let mut given = serde_json::Map::new();
    if let Some(tables) = tables {
        given.insert("tables".into(), Value::Object(tables.clone()));
    }
    if !vector["time_columns"].is_null() {
        given.insert("time_columns".into(), named.clone());
    }
    if !given.is_empty() {
        input["from_source"] = Value::Object(given);
    }
    let case = json!({
        "format": "sdm-case/1",
        "id": format!("t132e.{}", vector["id"].as_str().unwrap()),
        "status": "diagnostic",
        "slice": "S1",
        "paths": [],
        "routes": [],
        "cells": [],
        "input": input,
        "until": "from_source",
        "core": {
            "describe_source": {
                "provider": {"provider": "sql", "dialect": "duckdb"},
                "source": {"tables": named_tables, "primary_keys": {}, "time_columns": written},
                "check_keys": true,
            }
        },
        "expect": {"outcome": "success", "stage": "from_source"},
        "why": format!("T1.32e vector {}", vector["id"].as_str().unwrap()),
        "source": "contract/tools/time_inference_record.py",
    });
    serde_json::from_str(&case.to_string()).expect("the vector's case is a case")
}

/// A described column's dtype as the client's wire names it (the public
/// schema's dtype), or the text given when it is no dtype.
fn wire_dtype(dtype: &Value) -> Value {
    serde_json::from_value::<Dtype>(dtype.clone())
        .map_or_else(|_| dtype.clone(), |d| Value::from(d.wire_name()))
}

/// A key as the public schema gives it: none, one column's name, or the
/// names of several.
fn public_key(key: &Value) -> Value {
    match key.as_array() {
        Some(names) if names.is_empty() => Value::Null,
        Some(names) if names.len() == 1 => names[0].clone(),
        _ => key.clone(),
    }
}

/// A count as a number: a `u64` scalar's digits, or the number given.
fn count(value: &Value) -> Value {
    match value {
        Value::Object(scalar) => scalar
            .get("v")
            .and_then(Value::as_str)
            .and_then(|digits| digits.parse::<u64>().ok())
            .map_or_else(|| value.clone(), Value::from),
        other => other.clone(),
    }
}

/// What this host made of `events`, in the vector's terms: the outcome and
/// code, and the table as the description gives it.
fn project(observed: &Observed) -> Value {
    match &observed.answer {
        Answer::Error { code, .. } => json!({"outcome": "refusal", "code": code}),
        Answer::Described(described) => {
            let table = described["tables"]
                .as_array()
                .into_iter()
                .flatten()
                .find(|t| t["name"] == TABLE);
            let schema = match table {
                None => Value::from("not described"),
                Some(t) => json!({
                    "columns": t["columns"]
                        .as_array()
                        .into_iter()
                        .flatten()
                        .map(|c| json!({
                            "name": c["name"],
                            "dtype": wire_dtype(&c["dtype"]),
                            "stype": c["stype"],
                        }))
                        .collect::<Vec<_>>(),
                    "primary_key": public_key(&t["primary_key"]),
                    "time_column": t["time_column"],
                    "duplicate_keys": count(&t["duplicate_keys"]),
                }),
            };
            json!({"outcome": "success", "code": null, "schema": schema})
        }
        other => json!({"outcome": format!("no answer of describe_source: {other:?}")}),
    }
}

/// The reference's answer in this host's terms.
fn wanted(reference: &Value) -> Value {
    let outcome = &reference["outcome"];
    if outcome["result"] == "refusal" {
        return json!({"outcome": "refusal", "code": outcome["code"]});
    }
    json!({"outcome": "success", "code": null, "schema": reference["schema"]})
}

/// The column `time_columns` names for `events`, if it names one.
fn named_column(vector: &Value) -> Option<&str> {
    vector["time_columns"][TABLE].as_str()
}

fn run(vector: &Value, bases: &Bases) -> Value {
    match host::run(&the_case(vector), bases) {
        Ok(observed) => project(&observed),
        Err(failure) => json!({"outcome": "host failure", "why": failure.to_string()}),
    }
}

fn bases() -> Bases {
    bases::load(&root().join("contract/bases.yaml")).unwrap()
}

#[test]
fn the_vectors_are_the_reference_s_and_not_vacuous() {
    let all = vectors();
    let vectors = all["vectors"].as_array().unwrap();
    assert!(vectors.len() >= 100, "{} vectors", vectors.len());
    let reference = std::fs::read_to_string(root().join("contract/reference-env.txt")).unwrap();
    let commit = all["reference"].as_str().unwrap();
    assert!(
        reference.contains(&format!("reference: {commit}")),
        "recorded at the reference of contract/reference-env.txt"
    );
    let mut seen = BTreeSet::new();
    let (mut success, mut refusal, mut large) = (0, 0, 0);
    for vector in vectors {
        let id = vector["id"].as_str().unwrap();
        assert!(seen.insert(id.to_owned()), "{id} twice");
        assert!(!vector["setup"].as_array().unwrap().is_empty(), "{id}");
        match vector["reference"]["outcome"]["result"].as_str() {
            Some("success") => success += 1,
            Some("refusal") => refusal += 1,
            other => panic!("{id}: an answer the vectors do not hold: {other:?}"),
        }
        let required = vector["register"]["required"].as_bool().unwrap();
        let why = vector["register"]["why"].as_array().unwrap();
        assert_eq!(required, !why.is_empty(), "{id}: a mark says why");
        large += usize::from(vector["rows"].as_u64().unwrap() >= PREVIEW_ROWS);
    }
    assert!(success >= 60 && refusal >= 12, "{success} {refusal}");
    assert!(
        large * 4 >= vectors.len() * 3,
        "most vectors past a preview"
    );
}

#[test]
fn the_pending_list_only_shrinks() {
    let gating = read_json("contract/tests/t132e-pending.json");
    let all: BTreeSet<String> = vectors()["vectors"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v["id"].as_str().unwrap().to_owned())
        .collect();
    for group in ["python-core", "rust-host", "exits"] {
        let now = ids(&gating["pending"][group]);
        let authored = ids(&gating["authored"][group]);
        assert!(
            now.is_subset(&authored),
            "{group}: pending ids added: {:?}",
            now.difference(&authored)
        );
    }
    assert!(
        pending("rust-host").is_subset(&all),
        "pending ids that are no vector"
    );
}

/// Runs every vector of `rows` rows on this host, holding each to the
/// reference and to the pending list.
fn size(rows: u64) {
    let all = vectors();
    let bases = bases();
    let register =
        std::fs::read_to_string(root().join("docs/design/rust-core/register.yaml")).unwrap();
    let (entries, _) = transitions(&register);
    let pending = pending("rust-host");
    let mut failures = Vec::new();
    let mut passing_but_pending = Vec::new();
    let mut ran = 0;
    for vector in all["vectors"].as_array().unwrap() {
        if vector["rows"].as_u64() != Some(rows) {
            continue;
        }
        ran += 1;
        let id = vector["id"].as_str().unwrap();
        let want = wanted(&vector["reference"]);
        let got = run(vector, &bases);
        let ok = passes(id, &got, &want, &entries);
        match (ok, pending.contains(id)) {
            (true, true) => passing_but_pending.push(id.to_owned()),
            (false, false) => failures.push(format!("{id}: got {got}, the reference {want}")),
            _ => {}
        }
    }
    assert!(ran > 0, "{rows}: no vectors");
    assert!(
        passing_but_pending.is_empty(),
        "these pass now: take them out of contract/tests/t132e-pending.json's \
         pending[\"rust-host\"]: {passing_but_pending:?}"
    );
    assert!(
        failures.is_empty(),
        "{} vectors answered otherwise than the reference, with no register \
         entry naming them:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// One test per size, so the sizes run in parallel.
macro_rules! sizes {
    ($($name:ident => $rows:literal,)*) => {
        $(
            #[test]
            fn $name() {
                size($rows);
            }
        )*

        #[test]
        fn every_vector_is_of_a_tested_size() {
            let sizes = [$($rows),*];
            for vector in vectors()["vectors"].as_array().unwrap() {
                let rows = vector["rows"].as_u64().unwrap();
                assert!(sizes.contains(&rows), "{}: {rows} rows", vector["id"]);
            }
        }
    };
}

sizes! {
    rust_host_infers_time_on_999_rows_as_the_reference => 999,
    rust_host_infers_time_on_a_full_preview_as_the_reference => 1000,
    rust_host_infers_time_one_row_past_a_preview_as_the_reference => 1001,
    rust_host_infers_time_on_5000_rows_as_the_reference => 5000,
    rust_host_falls_back_to_the_preview_as_the_reference => 200000,
}

/// The exit: on every vector of 1,000 rows or more whose `time_columns`
/// names a column the reference takes, this host's answer is a success
/// whose time column is that column. Each is `rs:<vector id>` in
/// `pending["exits"]`.
#[test]
fn rust_host_honours_a_named_time_column_on_a_large_table() {
    let all = vectors();
    let bases = bases();
    let pending = pending("exits");
    let mut failures = Vec::new();
    let mut passing_but_pending = Vec::new();
    let mut ran = BTreeSet::new();
    for vector in all["vectors"].as_array().unwrap() {
        let Some(named) = named_column(vector) else {
            continue;
        };
        let reference = &vector["reference"];
        if vector["rows"].as_u64().unwrap() < PREVIEW_ROWS
            || reference["outcome"]["result"] != "success"
            || reference["schema"]["time_column"] != named
        {
            continue;
        }
        let id = vector["id"].as_str().unwrap();
        ran.insert(vector["rows"].as_u64().unwrap());
        let got = run(vector, &bases);
        let ok = got["outcome"] == "success" && got["schema"]["time_column"] == named;
        let name = format!("rs:{id}");
        match (ok, pending.contains(&name)) {
            (true, true) => passing_but_pending.push(name),
            (false, false) => failures.push(format!("{id}: got {got}, named {named}")),
            _ => {}
        }
    }
    assert!(
        ran.is_superset(&BTreeSet::from([1_000, 1_001, 5_000])),
        "the exit's vectors cover every large size: {ran:?}"
    );
    assert!(
        passing_but_pending.is_empty(),
        "these pass now: take them out of contract/tests/t132e-pending.json's \
         pending[\"exits\"]: {passing_but_pending:?}"
    );
    assert!(
        failures.is_empty(),
        "a named time column not honoured:\n{}",
        failures.join("\n")
    );
}

#[test]
fn the_comparison_fails_closed() {
    let reference = json!({
        "outcome": {"result": "success", "stage": "from_source", "error": null, "code": null},
        "schema": {
            "columns": [
                {"name": "event_id", "dtype": "int64", "stype": "ID"},
                {"name": "billed_at", "dtype": "timestamp[us]", "stype": "timestamp"},
            ],
            "primary_key": "event_id",
            "time_column": "billed_at",
            "duplicate_keys": 0,
        },
        "warnings": [],
    });
    let want = wanted(&reference);
    assert!(agrees(&want, &want));
    // Each part of the description counts.
    for (part, value) in [
        ("time_column", json!("shipped_at")),
        ("time_column", json!(null)),
        ("primary_key", json!(null)),
        ("duplicate_keys", json!(1)),
        ("columns", json!([])),
    ] {
        let mut got = want.clone();
        got["schema"][part] = value;
        assert!(!agrees(&got, &want), "{part}");
    }
    // A refusal: the same code, and an unnamed one is never matched.
    let named = wanted(&json!({"outcome": {"result": "refusal", "code": "INVALID_REQUEST"}}));
    assert!(agrees(
        &json!({"outcome": "refusal", "code": "INVALID_REQUEST"}),
        &named
    ));
    assert!(!agrees(
        &json!({"outcome": "refusal", "code": "UNSUPPORTED_REQUEST"}),
        &named
    ));
    let unnamed = wanted(&json!({"outcome": {"result": "refusal", "code": null}}));
    assert!(!agrees(
        &json!({"outcome": "refusal", "code": null}),
        &unnamed
    ));
    // The transitions: today's time_whole_table refusal passes only where
    // an approved entry declares it, for the vectors its glob names.
    let refused = json!({"outcome": "refusal", "code": "UNSUPPORTED_REQUEST"});
    let entry = |status: &str, glob: &str, old: &str, new: &str| {
        (
            "E-1".to_owned(),
            status.to_owned(),
            glob.to_owned(),
            old.to_owned(),
            new.to_owned(),
        )
    };
    let declared = "refusal UNSUPPORTED_REQUEST";
    assert!(!passes("one-timestamp.5000", &refused, &want, &[]));
    let proposed = [entry("proposed", "one-timestamp.*", "success", declared)];
    assert!(!passes("one-timestamp.5000", &refused, &want, &proposed));
    let approved = [entry("approved", "one-timestamp.*", "success", declared)];
    assert!(passes("one-timestamp.5000", &refused, &want, &approved));
    assert!(!passes("nulls.5000", &refused, &want, &approved));
    let other = [entry(
        "approved",
        "one-timestamp.*",
        "success",
        "refusal INVALID_REQUEST",
    )];
    assert!(!passes("one-timestamp.5000", &refused, &want, &other));
    let failed = json!({"outcome": "host failure"});
    assert!(!passes("one-timestamp.5000", &failed, &want, &approved));
    // The projection's conversions.
    assert_eq!(wire_dtype(&json!("date")), json!("timestamp[us]"));
    assert_eq!(wire_dtype(&json!("time")), json!("timestamp[us]"));
    assert_eq!(wire_dtype(&json!("int")), json!("int64"));
    assert_eq!(public_key(&json!([])), json!(null));
    assert_eq!(public_key(&json!(["event_id"])), json!("event_id"));
    assert_eq!(count(&json!({"t": "u64", "v": "3"})), json!(3));
    // The register's text: refs of proposed and approved entries, in both
    // forms; a rejected entry's and another file's ignored; a ref naming
    // vectors with no transition malformed.
    let text = "entries:\n- id: A-1\n  status: approved\n  refs: ['contract/time-inference-vectors.json#late-* success -> refusal UNSUPPORTED_REQUEST']\n\
                - id: A-2\n  status: proposed\n  refs:\n  - contract/time-inference-vectors.json#nulls.* success -> refusal INVALID_REQUEST\n\
                - id: A-3\n  status: rejected\n  refs:\n  - contract/time-inference-vectors.json#constant.* success -> success\n\
                - id: A-4\n  status: approved\n  refs:\n  - contract/other.json#late-* success -> success\n  - contract/time-inference-vectors.json#late-*\n  - contract/time-inference-vectors.json#late-* Success -> success\n";
    let (read, malformed) = transitions(text);
    assert_eq!(
        read.iter()
            .map(|(e, s, g, o, n)| format!("{e} {s} {g} {o} -> {n}"))
            .collect::<Vec<_>>(),
        vec![
            "A-1 approved late-* success -> refusal UNSUPPORTED_REQUEST",
            "A-2 proposed nulls.* success -> refusal INVALID_REQUEST",
        ]
    );
    assert_eq!(
        malformed,
        vec![
            "A-4: contract/time-inference-vectors.json#late-*",
            "A-4: contract/time-inference-vectors.json#late-* Success -> success",
        ]
    );
    assert!(glob("late-*", "late-values.5000"));
    assert!(!glob("nulls.*", "nulls-named.5000"));
}

#[test]
fn every_vector_ref_names_its_transition() {
    let register =
        std::fs::read_to_string(root().join("docs/design/rust-core/register.yaml")).unwrap();
    let (_, malformed) = transitions(&register);
    assert!(
        malformed.is_empty(),
        "vector refs naming no transition `<glob> <from> -> <to>`: {malformed:?}"
    );
}
