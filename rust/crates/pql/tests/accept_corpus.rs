// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! P5.valid and P5.invalid: every case of the PQL corpus, read straight from
//! `tests/parity/pql/`, gives what today's Python recorded.
//!
//! A valid case gives a tree equal to `expected_tree` by value and JSON type,
//! and the recorded warnings. An invalid case gives the same errors in count,
//! code and span, and the same messages except for `PQL_PARSE_SYNTAX`, whose
//! new text must not carry ANTLR's or pydantic's wording.

mod accept_common;

use std::fs;
use std::path::{Path, PathBuf};

use accept_common::{mismatch, read_json, repo, report, resolve};
use serde_json::{Value, json};

/// What the corpus held when Phase 5 started; it may only grow.
const VALID_AT_BASE: usize = 193;
const INVALID_AT_BASE: usize = 186;

fn json_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let mut entries: Vec<PathBuf> = fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
        .map(|entry| entry.expect("a directory entry").path())
        .collect();
    entries.sort();
    for path in entries {
        if path.is_dir() {
            json_files(&path, out);
        } else if path.extension().is_some_and(|ext| ext == "json") {
            out.push(path);
        }
    }
}

/// Every case of every corpus file, as `test_pql_corpus.py` reads them.
fn cases() -> Vec<Value> {
    let dir = repo().join("tests/parity/pql");
    for name in ["valid.json", "invalid.json"] {
        assert!(dir.join(name).is_file(), "no {name} in {}", dir.display());
    }
    let mut files = Vec::new();
    json_files(&dir, &mut files);
    let mut out = Vec::new();
    for file in files {
        let corpus = read_json(&file);
        let cases = corpus["cases"]
            .as_array()
            .unwrap_or_else(|| panic!("{} has no cases", file.display()));
        out.extend(cases.iter().cloned());
    }
    out
}

fn expected(case: &Value) -> Value {
    let warnings = case.get("expected_warnings").cloned().unwrap_or(json!([]));
    if let Some(tree) = case.get("expected_tree") {
        return json!({"tree": tree, "warnings": warnings});
    }
    let errors = case
        .get("expected_errors")
        .cloned()
        .or_else(|| case.get("expected_error").map(|e| json!([e])))
        .unwrap_or_else(|| panic!("{} expects neither a tree nor errors", case["id"]));
    json!({"errors": errors, "warnings": warnings})
}

fn check(valid: bool) {
    let mut checked = 0;
    let mut failures = Vec::new();
    for case in cases() {
        if case.get("expected_tree").is_some() != valid {
            continue;
        }
        checked += 1;
        let id = case["id"].as_str().unwrap_or("?");
        let query = case["query"].as_str().expect("a query");
        let profile = case
            .get("profile")
            .and_then(Value::as_str)
            .unwrap_or("RELATIONAL");
        let found = match resolve(query, &case["schema"], profile, None) {
            Ok(found) => mismatch(&expected(&case), &found),
            Err(error) => Some(error),
        };
        if let Some(problem) = found {
            failures.push(format!("{id}: {problem}"));
        }
    }
    let (what, floor) = if valid {
        ("valid corpus cases", VALID_AT_BASE)
    } else {
        ("invalid corpus cases", INVALID_AT_BASE)
    };
    assert!(
        checked >= floor,
        "{what}: {checked} found, fewer than the {floor} Phase 5 started with"
    );
    report(what, checked, &failures);
}

#[test]
fn every_valid_case() {
    check(true);
}

#[test]
fn every_invalid_case() {
    check(false);
}
