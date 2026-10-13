// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! T1.32e part 1's key rules on `rust-host`, held from outside the
//! implementation (the package's independent test author) to what the
//! reference gave: `contract/key-inference-vectors.json`, recorded from the
//! reference by `contract/tools/key_inference_record.py`, which
//! `contract/tests/test_t132e_key_inference.py` holds `python-core` to as
//! well, so the two hosts agree through it.
//!
//! - The owner's ruling of 2026-10-13 (port the reference's key inference:
//!   never refuse a whole table because of one null in a key): each vector
//!   (generated `accounts` and `payments` of 200, 1,000 or 5,000 rows, one
//!   key varied: its distinct share, its nulls, its type, inferred or named
//!   in `primary_keys`, a keyless fact table read by foreign key) runs
//!   through this host's `describe_source` and, where the case goes on,
//!   `materialize`. The answer must be the reference's as far as this host
//!   observes it: the outcome, the stage it ended at (`from_source` or
//!   `materialize`), the code, and both tables as the description gives
//!   them (each column's name, dtype as the wire names it and stype, the
//!   key or none, the time column, the duplicate-key count) with the
//!   links. This host observes no exception class or warning class; the
//!   Python test holds the warnings.
//! - The ruling's exit: every vector whose varied key holds a null and
//!   whose `Schema.from_source` the reference answered is described here
//!   too, with the reference's key for that table.
//! - A vector may differ only under an approved register entry naming it
//!   with its transition
//!   (`contract/key-inference-vectors.json#<id or glob> <from> -> <to>`);
//!   a proposed entry excuses nothing; `register.required` excuses nothing
//!   by itself.
//!
//! Gating: `contract/tests/t132e-pending.json`'s `pending["key-rust-host"]`
//! lists the vectors that failed here when the test was written, and
//! `pending["exits"]` the exit's (`rs-key:` ids). Each must still fail
//! until taken out; the lists only shrink. The implementer changes nothing
//! else of this file, the vectors or the recorder.

use std::collections::BTreeSet;
use std::path::PathBuf;

use sdm_contract::ops::Dtype;
use sdm_testkit::bases::{self, Bases};
use sdm_testkit::case::Case;
use sdm_testkit::host::{self, Answer, Observed};
use serde_json::{Value, json};

const REF: &str = "contract/key-inference-vectors.json#";

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

fn read_json(path: &str) -> Value {
    let text = std::fs::read_to_string(root().join(path)).unwrap();
    serde_json::from_str(&text).unwrap()
}

fn vectors() -> Value {
    let all = read_json("contract/key-inference-vectors.json");
    assert_eq!(all["format"], "sdm-key-inference-vectors/1");
    assert_eq!(all["recorded_by"], "contract/tools/key_inference_record.py");
    assert_eq!(all["register_ref"], REF);
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

/// Whether this host's answer is the reference's, as far as it observes:
/// the outcome, the stage, the code and the description, exactly; an
/// unnamed refusal (no code) is no core answer.
fn agrees(got: &Value, want: &Value) -> bool {
    !(want["outcome"] == "refusal" && want["code"].is_null()) && got == want
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

/// The case this host runs for `vector`, as the recorder's `case_for`
/// writes it: `describe_source` with its `primary_keys`, then, where the
/// vector goes on, `materialize` of the first two accounts.
fn the_case(all: &Value, vector: &Value) -> Case {
    let named = &vector["primary_keys"];
    let mut keys = serde_json::Map::new();
    if let Some(named) = named.as_object() {
        for (table, column) in named {
            keys.insert(table.clone(), json!([column]));
        }
    }
    let text = vector["account_kind"] == "text";
    let (public, core) = if text {
        (
            json!(["k0", "k1"]),
            json!([{"t": "utf8", "v": "k0"}, {"t": "utf8", "v": "k1"}]),
        )
    } else {
        (
            json!([0, 1]),
            json!([{"t": "i64", "v": "0"}, {"t": "i64", "v": "1"}]),
        )
    };
    let until = vector["until"].as_str().unwrap();
    let mut input = json!({
        "base": "shop",
        "setup": vector["setup"].clone(),
        "pql": all["pql"].clone(),
        "materialize": {"indices": public, "anchor_time": all["anchor"].clone()},
    });
    if !named.is_null() {
        input["from_source"] = json!({"primary_keys": named.clone()});
    }
    let mut core_inputs = json!({
        "describe_source": {
            "provider": {"provider": "sql", "dialect": "duckdb"},
            "source": {"tables": [], "primary_keys": keys, "time_columns": {}},
            "check_keys": true,
        }
    });
    if until == "materialize" {
        core_inputs["materialize"] = json!({
            "provider": {"provider": "sql", "dialect": "duckdb"},
            "indices": core,
            "anchor_time": {"t": "ts", "v": "1709251200000000000", "unit": "ns", "zone": null},
            "seed": null,
        });
    }
    let case = json!({
        "format": "sdm-case/1",
        "id": format!("t132e.key.{}", vector["id"].as_str().unwrap()),
        "status": "diagnostic",
        "slice": "S1",
        "paths": [],
        "routes": [],
        "cells": [],
        "input": input,
        "until": until,
        "core": core_inputs,
        "expect": {"outcome": "success", "stage": until},
        "why": format!("T1.32e key vector {}", vector["id"].as_str().unwrap()),
        "source": "contract/tools/key_inference_record.py",
    });
    serde_json::from_str(&case.to_string()).expect("the vector's case is a case")
}

/// A description in the vector's terms: each table (by name) with its
/// columns, key, time column and duplicate-key count, and the links, each
/// `[table, column, table, column]`, sorted.
fn schema_of(described: &Value) -> Value {
    let mut tables: Vec<Value> = described["tables"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|t| {
            json!({
                "name": t["name"],
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
            })
        })
        .collect();
    tables.sort_by_key(|t| t["name"].as_str().unwrap_or_default().to_owned());
    let mut links: Vec<Value> = described["links"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|link| {
            json!([
                link[0]["table"],
                link[0]["column"],
                link[1]["table"],
                link[1]["column"]
            ])
        })
        .collect();
    links.sort_by_key(Value::to_string);
    json!({"tables": tables, "links": links})
}

/// What this host answered, in the vector's terms.
fn project(observed: &Observed) -> Value {
    let schema = observed.described.as_ref().map_or(Value::Null, schema_of);
    let stage = if observed.described.is_some() && observed.operations.len() > 1 {
        "materialize"
    } else {
        "from_source"
    };
    match &observed.answer {
        Answer::Error { code, .. } => {
            json!({"outcome": "refusal", "stage": stage, "code": code, "schema": schema})
        }
        Answer::Described(_) | Answer::Materialized(_) => {
            json!({"outcome": "success", "stage": stage, "code": null, "schema": schema})
        }
        other => json!({"outcome": format!("no answer: {other:?}")}),
    }
}

/// The reference's answer in this host's terms. The reference's schema
/// sorts its links as lists; this host's are sorted as JSON text, so both
/// are sorted the same way here.
fn wanted(reference: &Value) -> Value {
    let outcome = &reference["outcome"];
    let mut schema = reference["schema"].clone();
    if let Some(links) = schema.get_mut("links").and_then(Value::as_array_mut) {
        links.sort_by_key(Value::to_string);
    }
    json!({
        "outcome": outcome["result"],
        "stage": outcome["stage"],
        "code": outcome["code"],
        "schema": schema,
    })
}

fn key_of(schema: &Value, table: &str) -> Value {
    schema["tables"]
        .as_array()
        .into_iter()
        .flatten()
        .find(|t| t["name"] == table)
        .map_or(Value::from("not described"), |t| t["primary_key"].clone())
}

fn run(all: &Value, vector: &Value, bases: &Bases) -> Value {
    match host::run(&the_case(all, vector), bases) {
        Ok(observed) => project(&observed),
        Err(failure) => json!({"outcome": "host failure", "why": failure.to_string()}),
    }
}

fn bases() -> Bases {
    bases::load(&root().join("contract/bases.yaml")).unwrap()
}

#[test]
fn the_key_vectors_are_the_reference_s_and_not_vacuous() {
    let all = vectors();
    let vectors = all["vectors"].as_array().unwrap();
    assert!(vectors.len() >= 40, "{} vectors", vectors.len());
    let reference = std::fs::read_to_string(root().join("contract/reference-env.txt")).unwrap();
    let commit = all["reference"].as_str().unwrap();
    assert!(reference.contains(&format!("reference: {commit}")));
    let mut seen = BTreeSet::new();
    let mut marked = 0;
    for vector in vectors {
        let id = vector["id"].as_str().unwrap();
        assert!(seen.insert(id.to_owned()), "{id} twice");
        let required = vector["register"]["required"].as_bool().unwrap();
        let why = vector["register"]["why"].as_array().unwrap();
        assert_eq!(required, !why.is_empty(), "{id}: a mark says why");
        marked += usize::from(required);
    }
    assert!(marked > 0 && marked < vectors.len());
}

#[test]
fn the_key_pending_lists_only_shrink() {
    let gating = read_json("contract/tests/t132e-pending.json");
    let all: BTreeSet<String> = vectors()["vectors"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v["id"].as_str().unwrap().to_owned())
        .collect();
    for group in ["key-python-core", "key-rust-host", "exits"] {
        let now = ids(&gating["pending"][group]);
        let authored = ids(&gating["authored"][group]);
        assert!(
            now.is_subset(&authored),
            "{group}: pending ids added: {:?}",
            now.difference(&authored)
        );
    }
    assert!(pending("key-rust-host").is_subset(&all));
}

/// Runs every vector whose varied table has `rows` rows, holding each to
/// the reference and to the pending list.
fn size(rows: u64) {
    let all = vectors();
    let bases = bases();
    let register =
        std::fs::read_to_string(root().join("docs/design/rust-core/register.yaml")).unwrap();
    let (entries, _) = transitions(&register);
    let pending = pending("key-rust-host");
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
        let got = run(&all, vector, &bases);
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
         pending[\"key-rust-host\"]: {passing_but_pending:?}"
    );
    assert!(
        failures.is_empty(),
        "{} vectors answered otherwise than the reference, with no register \
         entry naming them:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

macro_rules! sizes {
    ($($name:ident => $rows:literal,)*) => {
        $(
            #[test]
            fn $name() {
                size($rows);
            }
        )*

        #[test]
        fn every_key_vector_is_of_a_tested_size() {
            let sizes = [$($rows),*];
            for vector in vectors()["vectors"].as_array().unwrap() {
                let rows = vector["rows"].as_u64().unwrap();
                assert!(sizes.contains(&rows), "{}: {rows} rows", vector["id"]);
            }
        }
    };
}

sizes! {
    rust_host_keys_200_row_tables_as_the_reference => 200,
    rust_host_keys_a_full_preview_as_the_reference => 1000,
    rust_host_keys_5000_row_tables_as_the_reference => 5000,
}

/// The ruling's exit: a key holding a null refuses no table. Each is
/// `rs-key:<vector id>` in `pending["exits"]`.
#[test]
fn rust_host_refuses_no_table_for_a_null_in_its_key() {
    let all = vectors();
    let bases = bases();
    let pending = pending("exits");
    let mut failures = Vec::new();
    let mut passing_but_pending = Vec::new();
    let mut ran = 0;
    for vector in all["vectors"].as_array().unwrap() {
        let reference = &vector["reference"];
        if vector["key"]["nulls"].as_u64().unwrap() == 0 || reference["schema"].is_null() {
            continue;
        }
        ran += 1;
        let id = vector["id"].as_str().unwrap();
        let table = vector["table"].as_str().unwrap();
        let got = run(&all, vector, &bases);
        let ok = !got["schema"].is_null()
            && key_of(&got["schema"], table) == key_of(&reference["schema"], table);
        let name = format!("rs-key:{id}");
        match (ok, pending.contains(&name)) {
            (true, true) => passing_but_pending.push(name),
            (false, false) => failures.push(format!("{id}: got {got}")),
            _ => {}
        }
    }
    assert!(ran >= 10, "{ran} vectors");
    assert!(
        passing_but_pending.is_empty(),
        "these pass now: take them out of contract/tests/t132e-pending.json's \
         pending[\"exits\"]: {passing_but_pending:?}"
    );
    assert!(
        failures.is_empty(),
        "a table refused for a null in its key:\n{}",
        failures.join("\n")
    );
}

/// P1B-40, option (b) (owner, 2026-10-13, relayed): where sampling reaches
/// a table whose key mixes a null with real text or decimal values, which
/// the reference refuses or fails on, the core leaves the null-key rows
/// out and succeeds, with one warning per table. Each vector and its twin
/// (the same layout with an integer key, which the reference answers).
const OPTION_B: [(&str, &str); 2] = [
    ("nulls-1-decimal.200", "nulls-1.200"),
    ("nulls-1-text.200", "nulls-1.200"),
];

/// Each vector of [`OPTION_B`] succeeds at `materialize`, describes the
/// tables as the reference did before it refused, and gives exactly one
/// warning the twin's run does not, a code the twin's run never gives,
/// whose fields name `payments`. Each is `rs-key-b:<vector id>` in
/// `pending["exits"]`.
#[test]
fn rust_host_leaves_null_key_rows_out_with_one_warning() {
    let all = vectors();
    let bases = bases();
    let pending = pending("exits");
    let find = |id: &str| -> Value {
        all["vectors"]
            .as_array()
            .unwrap()
            .iter()
            .find(|v| v["id"] == id)
            .unwrap_or_else(|| panic!("{id} is a vector"))
            .clone()
    };
    let mut failures = Vec::new();
    let mut passing_but_pending = Vec::new();
    for (id, twin_id) in OPTION_B {
        let vector = find(id);
        let reference = &vector["reference"]["outcome"];
        assert_eq!(
            reference["result"], "refusal",
            "{id}: the reference refuses"
        );
        let observed = |vector: &Value| host::run(&the_case(&all, vector), &bases);
        let (got, baseline) = match (observed(&vector), observed(&find(twin_id))) {
            (Ok(got), Ok(baseline)) => (got, baseline),
            (got, baseline) => {
                failures.push(format!("{id}: host failure {got:?} {baseline:?}"));
                continue;
            }
        };
        let projected = project(&got);
        let known: BTreeSet<String> = baseline
            .warnings
            .iter()
            .map(|w| w["code"].to_string())
            .collect();
        let new: Vec<&Value> = got
            .warnings
            .iter()
            .filter(|w| !known.contains(&w["code"].to_string()))
            .collect();
        let mut why = Vec::new();
        if projected["outcome"] != "success" || projected["stage"] != "materialize" {
            why.push(format!("answered {projected}"));
        }
        if projected["schema"] != wanted(&vector["reference"])["schema"] {
            why.push("described otherwise than the reference".to_owned());
        }
        if new.len() != 1 || !new[0]["fields"].to_string().contains("\"payments\"") {
            why.push(format!("warnings beyond the twin's: {new:?}"));
        }
        let name = format!("rs-key-b:{id}");
        match (why.is_empty(), pending.contains(&name)) {
            (true, true) => passing_but_pending.push(name),
            (false, false) => failures.push(format!("{id}: {}", why.join("; "))),
            _ => {}
        }
    }
    assert!(
        passing_but_pending.is_empty(),
        "these pass now: take them out of contract/tests/t132e-pending.json's \
         pending[\"exits\"]: {passing_but_pending:?}"
    );
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn the_key_comparison_fails_closed() {
    let reference = json!({
        "outcome": {"result": "success", "stage": "materialize", "error": null, "code": null},
        "schema": {
            "tables": [
                {"name": "payments", "columns": [], "primary_key": null,
                 "time_column": "paid_at", "duplicate_keys": 0},
            ],
            "links": [["payments", "account_id", "accounts", "account_id"]],
        },
        "warnings": [],
    });
    let want = wanted(&reference);
    assert!(agrees(&want, &want));
    for (path, value) in [
        ("stage", json!("from_source")),
        ("code", json!("INVALID_REQUEST")),
        ("schema", json!(null)),
    ] {
        let mut got = want.clone();
        got[path] = value;
        assert!(!agrees(&got, &want), "{path}");
    }
    for (part, value) in [
        ("primary_key", json!("payment_id")),
        ("duplicate_keys", json!(3)),
        ("time_column", json!(null)),
    ] {
        let mut got = want.clone();
        got["schema"]["tables"][0][part] = value;
        assert!(!agrees(&got, &want), "{part}");
    }
    let mut got = want.clone();
    got["schema"]["links"] = json!([]);
    assert!(!agrees(&got, &want));
    // An unnamed refusal is never matched; INTERNAL_ERROR only by itself.
    let unnamed = wanted(
        &json!({"outcome": {"result": "refusal", "stage": "materialize", "code": null}, "schema": null}),
    );
    assert!(!agrees(&unnamed, &unnamed));
    let internal = wanted(
        &json!({"outcome": {"result": "refusal", "stage": "materialize", "code": "INTERNAL_ERROR"}, "schema": null}),
    );
    let named = json!({"outcome": "refusal", "stage": "materialize", "code": "UNSUPPORTED_REQUEST", "schema": null});
    let entry = |status: &str, old: &str, new: &str| {
        (
            "E-1".to_owned(),
            status.to_owned(),
            "nulls-1-decimal.*".to_owned(),
            old.to_owned(),
            new.to_owned(),
        )
    };
    assert!(!passes("nulls-1-decimal.200", &named, &internal, &[]));
    let proposed = [entry(
        "proposed",
        "refusal INTERNAL_ERROR",
        "refusal UNSUPPORTED_REQUEST",
    )];
    assert!(!passes("nulls-1-decimal.200", &named, &internal, &proposed));
    let approved = [entry(
        "approved",
        "refusal INTERNAL_ERROR",
        "refusal UNSUPPORTED_REQUEST",
    )];
    assert!(passes("nulls-1-decimal.200", &named, &internal, &approved));
    assert!(!passes("nulls-1-text.200", &named, &internal, &approved));
    // The links' conversion from the core's column references.
    let described = json!({
        "tables": [],
        "links": [[{"table": "payments", "column": "account_id"}, {"table": "accounts", "column": "account_id"}]],
    });
    assert_eq!(
        schema_of(&described)["links"],
        json!([["payments", "account_id", "accounts", "account_id"]])
    );
}

#[test]
fn every_key_vector_ref_names_its_transition() {
    let register =
        std::fs::read_to_string(root().join("docs/design/rust-core/register.yaml")).unwrap();
    let (_, malformed) = transitions(&register);
    assert!(malformed.is_empty(), "{malformed:?}");
}
