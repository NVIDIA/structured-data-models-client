// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! T1.32c's nested column types on `rust-host`, held from outside the
//! implementation (the package's independent test author) to what the
//! reference gave: `contract/nested-type-vectors.json`, recorded from the
//! reference by `contract/tools/nested_type_record.py`, which
//! `contract/tests/test_t132c_nested_types.py` holds `python-core` to as
//! well, so the two hosts agree through it.
//!
//! - Exits "`python-core` and `rust-host` agree on `core` fields" and
//!   "every difference is registered", over the family's types: each
//!   vector (`shop` with a column `f` of one DuckDB type, the vector's
//!   `setup`) runs through this host as `nested.list-int` does, through
//!   `describe_source`, `materialize` and `predict` against the scripted
//!   endpoint. The answer must be the reference's, as far as this host
//!   observes it: a success with `f` left out of its table's description
//!   with the core's `COLUMNS_LEFT_OUT` warning naming it where the
//!   reference left it out with its warning, described where the reference
//!   described it, and in the model call's body exactly where the
//!   reference's request held it; or a refusal with the reference's code
//!   (this host observes no stage or exception class).
//! - A vector the recorder marked `register.required` (the reference fails
//!   unnamed or with INTERNAL_ERROR) may differ only where an entry of
//!   `docs/design/rust-core/register.yaml`, proposed or approved, names it
//!   in its `refs` (`contract/nested-type-vectors.json#<id or glob>`); every
//!   other vector is held to the reference exactly.
//! - A refusal by a rule the core does not take yet (`UNSUPPORTED_REQUEST`)
//!   never passes: the brief lifts `column_type` for this family.
//!
//! Gating: `contract/tests/t132c-pending.json`'s `pending["rust-host"]`
//! lists the vectors that failed here when the test was written. Each must
//! still fail (strict: one that passes must be taken out of the list), and
//! the list may only shrink: it stays within `authored["rust-host"]`. The
//! implementer removes ids as they pass and changes nothing else of this
//! file, the vectors or the recorder.

use std::collections::BTreeSet;
use std::path::PathBuf;

use sdm_testkit::bases;
use sdm_testkit::case::{self, Case};
use sdm_testkit::host::{self, Answer, Observed};
use serde_json::{Value, json};

const REF: &str = "contract/nested-type-vectors.json#";

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

fn read_json(path: &str) -> Value {
    let text = std::fs::read_to_string(root().join(path)).unwrap();
    serde_json::from_str(&text).unwrap()
}

fn vectors() -> Value {
    let all = read_json("contract/nested-type-vectors.json");
    assert_eq!(all["format"], "sdm-nested-type-vectors/1");
    assert_eq!(all["recorded_by"], "contract/tools/nested_type_record.py");
    assert_eq!(all["template"], "contract/cases/s1/nested.list-int.json");
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

/// The register's entries that name a vector in their `refs`, proposed or
/// approved: (id, the patterns). Read from the YAML's text, entry by entry
/// (`- id:` at the margin), the `refs` field alone, in its flow
/// (`refs: [a, b]`) or block (`refs:` then `  - a`) form.
fn register_patterns(text: &str) -> Vec<(String, Vec<String>)> {
    let mut out = Vec::new();
    for block in text.split("\n- id: ").skip(1) {
        let id = block.lines().next().unwrap_or_default().trim().to_owned();
        let status = block
            .lines()
            .find_map(|l| l.strip_prefix("  status: "))
            .unwrap_or_default()
            .trim();
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
        let patterns: Vec<String> = refs
            .iter()
            .map(|r| r.trim_matches(|c| c == '\'' || c == '"'))
            .filter_map(|r| r.strip_prefix(REF))
            .map(str::to_owned)
            .collect();
        if !patterns.is_empty() {
            out.push((id, patterns));
        }
    }
    out
}

fn registered(id: &str, entries: &[(String, Vec<String>)]) -> Option<String> {
    entries
        .iter()
        .find(|(_, patterns)| patterns.iter().any(|p| glob(p, id)))
        .map(|(entry, _)| entry.clone())
}

/// The case this host runs for `vector`: `nested.list-int`'s, its setup the
/// vector's.
fn the_case(template: &Case, vector: &Value) -> Case {
    let mut case = template.clone();
    case.id = format!("t132c.{}", vector["id"].as_str().unwrap());
    case.input.setup = Some(
        vector["setup"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| s.as_str().unwrap().to_owned())
            .collect(),
    );
    case
}

/// Whether a JSON object's `columns` (a list of names, or of objects with a
/// `name`, or an object keyed by name) holds `f`.
fn holds_f(columns: &Value) -> bool {
    match columns {
        Value::Array(items) => items
            .iter()
            .any(|c| c == "f" || c.get("name").is_some_and(|n| n == "f")),
        Value::Object(members) => members.contains_key("f"),
        _ => false,
    }
}

/// What this host made of `f`, in the vector's terms: the outcome, `f` in
/// the description (`described`, `left out`, or null with no description),
/// whether a `COLUMNS_LEFT_OUT` warning names it, and whether the model
/// call's body holds it.
fn project(observed: &Observed, table: &str) -> Value {
    let (result, code) = match &observed.answer {
        Answer::Error { code, .. } => ("refusal", Value::from(code.clone())),
        Answer::Predicted(_) => ("success", Value::Null),
        other => {
            return json!({"outcome": format!("no answer of predict: {other:?}")});
        }
    };
    let described = observed.described.as_ref().map_or(Value::Null, |d| {
        let table = d["tables"]
            .as_array()
            .into_iter()
            .flatten()
            .find(|t| t["name"] == table);
        match table {
            None => Value::Null,
            Some(t) if holds_f(&t["columns"]) => Value::from("described"),
            Some(_) => Value::from("left out"),
        }
    });
    let warned = observed.warnings.iter().any(|w| {
        w["code"] == "COLUMNS_LEFT_OUT"
            && w["fields"].to_string().contains(&format!("\"{table}\""))
            && w["fields"].to_string().contains("\"f\"")
    });
    let in_body = match &observed.answer {
        Answer::Predicted(predicted) => {
            let body = &predicted["body"];
            Value::from(
                holds_f(&body["schema"]["related_tables"][table]["columns"])
                    || holds_f(&body["predict"]["related_tables"][table]["columns"])
                    || holds_f(&body["context"]["related_tables"][table]["columns"]),
            )
        }
        _ => Value::Null,
    };
    json!({
        "outcome": result,
        "code": code,
        "described": described,
        "left_out_warning": warned,
        "in_body": in_body,
    })
}

/// The reference's answer in this host's terms.
fn wanted(reference: &Value) -> Value {
    let outcome = &reference["outcome"];
    if outcome["result"] == "refusal" {
        return json!({"outcome": "refusal", "code": outcome["code"]});
    }
    let described = match &reference["described"] {
        Value::Null => Value::Null,
        Value::String(s) if s == "left out" => Value::from("left out"),
        _ => Value::from("described"),
    };
    let payload = &reference["payload"];
    json!({
        "outcome": "success",
        "code": null,
        "described": described,
        "left_out_warning": described == "left out",
        "in_body": !payload["schema"].is_null() || !payload["cells"].is_null(),
    })
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

#[test]
fn the_vectors_are_the_reference_s_and_not_vacuous() {
    let all = vectors();
    let vectors = all["vectors"].as_array().unwrap();
    assert!(vectors.len() > 60, "{} vectors", vectors.len());
    let reference = std::fs::read_to_string(root().join("contract/reference-env.txt")).unwrap();
    let commit = all["reference"].as_str().unwrap();
    assert!(
        reference.contains(&format!("reference: {commit}")),
        "recorded at the reference of contract/reference-env.txt"
    );
    let mut seen = BTreeSet::new();
    let (mut success, mut refusal, mut marked) = (0, 0, 0);
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
        marked += usize::from(required);
    }
    assert!(success > 20 && refusal > 20, "{success} {refusal}");
    assert!(marked > 0 && marked < vectors.len());
}

#[test]
fn the_pending_list_only_shrinks() {
    let pending = read_json("contract/tests/t132c-pending.json");
    let all: BTreeSet<String> = vectors()["vectors"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v["id"].as_str().unwrap().to_owned())
        .collect();
    let now = ids(&pending["pending"]["rust-host"]);
    let authored = ids(&pending["authored"]["rust-host"]);
    assert!(
        now.is_subset(&authored),
        "pending ids added: {:?}",
        now.difference(&authored)
    );
    assert!(now.is_subset(&all), "pending ids that are no vector");
}

/// Runs every vector whose id starts with `prefix` on this host, holding
/// each to the reference and to the pending list.
fn kind(prefix: &str) {
    let all = vectors();
    let contract = root().join("contract");
    let loaded = case::load(&[contract.join("cases")]).unwrap();
    let template = loaded
        .iter()
        .find(|l| l.case.id == "nested.list-int")
        .expect("nested.list-int is a case")
        .case
        .clone();
    let bases = bases::load(&contract.join("bases.yaml")).unwrap();
    let register =
        std::fs::read_to_string(root().join("docs/design/rust-core/register.yaml")).unwrap();
    let entries = register_patterns(&register);
    let pending = ids(&read_json("contract/tests/t132c-pending.json")["pending"]["rust-host"]);
    let mut failures = Vec::new();
    let mut passing_but_pending = Vec::new();
    let mut ran = 0;
    for vector in all["vectors"].as_array().unwrap() {
        let id = vector["id"].as_str().unwrap();
        if !id.starts_with(prefix) {
            continue;
        }
        ran += 1;
        let table = vector["table"].as_str().unwrap();
        let want = wanted(&vector["reference"]);
        let got = match host::run(&the_case(&template, vector), &bases) {
            Ok(observed) => project(&observed, table),
            Err(failure) => json!({"outcome": "host failure", "why": failure.to_string()}),
        };
        let unported = got["code"] == "UNSUPPORTED_REQUEST";
        let ok = !unported
            && (agrees(&got, &want)
                || (vector["register"]["required"] == true
                    && (got["outcome"] == "success" || got["outcome"] == "refusal")
                    && registered(id, &entries).is_some()));
        match (ok, pending.contains(id)) {
            (true, true) => passing_but_pending.push(id.to_owned()),
            (false, false) => failures.push(format!("{id}: got {got}, the reference {want}")),
            _ => {}
        }
    }
    assert!(ran > 0, "{prefix}: no vectors");
    assert!(
        passing_but_pending.is_empty(),
        "these pass now: take them out of contract/tests/t132c-pending.json's \
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

/// One test per kind of type, so the kinds run in parallel.
macro_rules! kinds {
    ($($name:ident => $prefix:literal,)*) => {
        $(
            #[test]
            fn $name() {
                kind($prefix);
            }
        )*

        #[test]
        fn every_vector_is_of_a_tested_kind() {
            let prefixes = [$($prefix),*];
            for vector in vectors()["vectors"].as_array().unwrap() {
                let id = vector["id"].as_str().unwrap();
                assert!(
                    prefixes.iter().filter(|p| id.starts_with(*p)).count() == 1,
                    "{id} is run by exactly one test"
                );
            }
        }
    };
}

kinds! {
    rust_host_handles_list_columns_as_the_reference => "list.",
    rust_host_handles_fixed_size_list_columns_as_the_reference => "array.",
    rust_host_handles_struct_columns_as_the_reference => "struct.",
    rust_host_handles_map_columns_as_the_reference => "map.",
    rust_host_handles_a_union_column_as_the_reference => "union.",
    rust_host_handles_binary_and_interval_columns_as_the_reference => "scalar.",
    rust_host_handles_the_fact_table_s_nested_columns_as_the_reference => "orders.",
}

#[test]
fn the_comparison_fails_closed() {
    // A success: the parts this host observes each count.
    let reference = json!({
        "outcome": {"result": "success", "stage": "predict", "error": null, "code": null},
        "described": "left out",
        "warnings": ["SDMWarning"],
        "payload": {"schema": null, "cells": null},
    });
    let want = wanted(&reference);
    assert_eq!(want["left_out_warning"], true);
    assert_eq!(want["in_body"], false);
    assert!(agrees(&want, &want));
    for (part, value) in [
        ("outcome", json!("refusal")),
        ("described", json!("described")),
        ("left_out_warning", json!(false)),
        ("in_body", json!(true)),
    ] {
        let mut got = want.clone();
        got[part] = value;
        assert!(!agrees(&got, &want), "{part}");
    }
    // A refusal: the same code, and an unnamed one is never matched.
    let named = wanted(&json!({"outcome": {"result": "refusal", "code": "INVALID_REQUEST"}}));
    assert!(agrees(
        &json!({"outcome": "refusal", "code": "INVALID_REQUEST"}),
        &named
    ));
    assert!(!agrees(
        &json!({"outcome": "refusal", "code": "INTERNAL_ERROR"}),
        &named
    ));
    let unnamed = wanted(&json!({"outcome": {"result": "refusal", "code": null}}));
    assert!(!agrees(
        &json!({"outcome": "refusal", "code": null}),
        &unnamed
    ));
    // The register: by id or glob, proposed or approved, this file's refs.
    let text = "entries:\n- id: A-1\n  status: approved\n  refs: ['contract/nested-type-vectors.json#struct.*']\n\
                - id: A-2\n  status: rejected\n  refs:\n  - contract/nested-type-vectors.json#map.*\n\
                - id: A-3\n  status: proposed\n  refs:\n  - contract/other.json#map.*\n";
    let entries = register_patterns(text);
    assert_eq!(registered("struct.integer", &entries), Some("A-1".into()));
    assert_eq!(registered("map.empty", &entries), None);
    assert_eq!(registered("list.integer", &[]), None);
    assert!(glob("struct.*", "struct.null"));
    assert!(!glob("struct.*", "orders.struct"));
}
