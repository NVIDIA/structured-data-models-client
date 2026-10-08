// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What the Phase 5 acceptance tests share: reading the repository's files,
//! asking the crate, and the one comparison with today's Python
//! (`tests/rust_core/pql/CONTRACT.md`).

#![allow(dead_code, reason = "each test file uses only some of these")]

use std::fs;
use std::path::{Path, PathBuf};

use serde_json::Value;

pub const PARSE_SYNTAX: &str = "PQL_PARSE_SYNTAX";

/// The repository's root, three levels above this crate.
pub fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

pub fn read_json(path: &Path) -> Value {
    let text = fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// The crate's answer for `query`, as `tests/parity/test_pql_corpus.py`'s
/// `resolve` writes today's: `{"tree" | "errors", "warnings"}`.
pub fn resolve(
    query: &str,
    schema: &Value,
    profile: &str,
    mode: Option<&str>,
) -> Result<Value, String> {
    let schema = serde_json::to_string(schema).map_err(|e| e.to_string())?;
    let text = match sdm_pql::validate_json(query, &schema, profile, mode) {
        Ok(text) => text,
        Err(error) => return Err(format!("validate_json failed: {error}")),
    };
    serde_json::from_str(&text).map_err(|e| format!("validate_json gave invalid JSON: {e}"))
}

/// The crate's reading of a date literal: `{"value", "warnings"}`.
pub fn parse_date(literal: &str) -> Result<Value, String> {
    let text = sdm_pql::parse_date_json(literal);
    serde_json::from_str(&text).map_err(|e| format!("parse_date_json gave invalid JSON: {e}"))
}

/// Wording that belongs to ANTLR or pydantic, not to a message written for
/// PQL's users; `_pql.py` holds the same list.
const FOREIGN: &[&str] = &[
    "mismatched input",
    "extraneous input",
    "input mismatch",
    "no viable alternative",
    "token recognition error",
    "antlr",
    "recognitionexception",
    "<eof>",
    "fqn_column",
    "wildcard_column",
    "rel_op",
    "neg_infty",
    "time_unit",
    "quoted_id",
    "for_each",
    "pydantic",
    "value_error",
    "input_value=",
    "input_type=",
    "argskwargs",
    "validation error for",
    "validation errors for",
];

pub fn foreign_wording(message: &str) -> Option<String> {
    let lower = message.to_lowercase();
    if let Some(word) = FOREIGN.iter().find(|w| lower.contains(*w)) {
        return Some((*word).to_string());
    }
    // ANTLR's "missing INT at 'x'".
    let mut rest = lower.as_str();
    while let Some(at) = rest.find("missing ") {
        let after = &rest[at + "missing ".len()..];
        let word_end = after.find(char::is_whitespace).unwrap_or(after.len());
        if word_end > 0 && after[word_end..].starts_with(" at '") {
            return Some(format!("missing {} at '", &after[..word_end]));
        }
        rest = after;
    }
    None
}

/// How `got` differs from `want`, both `{"tree" | "errors", "warnings"}`.
///
/// Trees compare strictly, by value and JSON type. Errors compare by count,
/// then code and span, then message; a `PQL_PARSE_SYNTAX` message may
/// differ, but not carry ANTLR's or pydantic's wording. Warnings compare in
/// full.
pub fn mismatch(want: &Value, got: &Value) -> Option<String> {
    let Some(object) = got.as_object() else {
        return Some(format!("not an object: {got}"));
    };
    let kinds = ["tree", "errors"]
        .iter()
        .filter(|k| object.contains_key(**k))
        .count();
    if kinds != 1 || !object.contains_key("warnings") || object.len() != 2 {
        return Some(format!(
            "a result has warnings and one of tree or errors: {:?}",
            object.keys().collect::<Vec<_>>()
        ));
    }
    let no_warnings = Value::Array(vec![]);
    let want_warnings = want.get("warnings").unwrap_or(&no_warnings);
    if &got["warnings"] != want_warnings {
        return Some(format!("warnings {} != {want_warnings}", got["warnings"]));
    }
    if let Some(tree) = want.get("tree") {
        return match object.get("tree") {
            None => Some(format!("refused: {}", got["errors"])),
            Some(found) if found != tree => Some(format!(
                "tree differs: {}",
                first_difference(tree, found, "$")
            )),
            Some(_) => None,
        };
    }
    let Some(found) = object.get("errors").and_then(Value::as_array) else {
        return Some("resolved, where today it is refused".to_string());
    };
    let expected = want["errors"].as_array().expect("expected errors");
    if found.len() != expected.len() {
        return Some(format!(
            "{} errors where today gives {}: {}",
            found.len(),
            expected.len(),
            got["errors"]
        ));
    }
    for (i, (x, y)) in expected.iter().zip(found).enumerate() {
        let keys: Vec<&String> = y
            .as_object()
            .map(|o| o.keys().collect())
            .unwrap_or_default();
        let mut sorted = keys.clone();
        sorted.sort();
        if sorted != ["code", "message", "span"] {
            return Some(format!("error {i} is not {{code, span, message}}: {y}"));
        }
        if y["code"] != x["code"] {
            return Some(format!("error {i}: code {} != {}", y["code"], x["code"]));
        }
        if y["span"] != x["span"] {
            return Some(format!(
                "error {i} ({}): span {} != {}",
                x["code"], y["span"], x["span"]
            ));
        }
        if x["code"] == PARSE_SYNTAX {
            let message = y["message"].as_str().unwrap_or("");
            if message.trim().is_empty() {
                return Some(format!("error {i}: an empty syntax message"));
            }
            if let Some(word) = foreign_wording(message) {
                return Some(format!(
                    "error {i}: the syntax message carries {word:?}: {message:?}"
                ));
            }
        } else if y["message"] != x["message"] {
            return Some(format!(
                "error {i} ({}): message {} != {}",
                x["code"], y["message"], x["message"]
            ));
        }
    }
    None
}

/// Where two JSON values first differ, by value or JSON type.
pub fn first_difference(a: &Value, b: &Value, path: &str) -> String {
    match (a, b) {
        (Value::Object(x), Value::Object(y)) => {
            let mut xs: Vec<&String> = x.keys().collect();
            let mut ys: Vec<&String> = y.keys().collect();
            xs.sort();
            ys.sort();
            if xs != ys {
                return format!("{path}: keys {xs:?} != {ys:?}");
            }
            for key in xs {
                if x[key] != y[key] {
                    return first_difference(&x[key], &y[key], &format!("{path}.{key}"));
                }
            }
            format!("{path}: equal")
        }
        (Value::Array(x), Value::Array(y)) if x.len() == y.len() => {
            for (i, (p, q)) in x.iter().zip(y).enumerate() {
                if p != q {
                    return first_difference(p, q, &format!("{path}[{i}]"));
                }
            }
            format!("{path}: equal")
        }
        _ => format!("{path}: {a} != {b}"),
    }
}

/// Fails with every mismatch at once, so one run shows them all.
pub fn report(what: &str, checked: usize, failures: &[String]) {
    if failures.is_empty() {
        println!("{what}: {checked} checked, all equal to today's Python");
        return;
    }
    let shown: Vec<&String> = failures.iter().take(60).collect();
    panic!(
        "{what}: {} of {checked} differ from today's Python:\n{}",
        failures.len(),
        shown
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>()
            .join("\n")
    );
}
