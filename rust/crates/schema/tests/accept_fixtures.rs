// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! P65.fixtures, inside the crate: every case of
//! `tests/parity/fixtures/dates.json` and `composite-key.json`, read straight
//! from the fixtures, gives through `sdm_schema::fixture_op` the expected
//! output, by value and JSON type, or the expected error code, and message
//! where the fixture has one (`tests/rust_core/schema/CONTRACT.md`).

use std::fs;
use std::path::PathBuf;

use serde_json::Value;

fn repo() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .ancestors()
        .nth(3)
        .expect("the crate is rust/crates/schema in the repository")
        .to_path_buf()
}

fn cases(name: &str) -> Vec<Value> {
    let path = repo()
        .join("tests/parity/fixtures")
        .join(format!("{name}.json"));
    let text = fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let data: Value =
        serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    data["cases"]
        .as_array()
        .unwrap_or_else(|| panic!("{name}.json has no cases"))
        .clone()
}

/// Equal by value and JSON type: an integer is not a float, `1` is not `true`.
fn same(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => match (x.is_f64(), y.is_f64()) {
            (true, true) => x.as_f64().map(f64::to_bits) == y.as_f64().map(f64::to_bits),
            (false, false) => x == y,
            _ => false,
        },
        (Value::Array(x), Value::Array(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(p, q)| same(p, q))
        }
        (Value::Object(x), Value::Object(y)) => {
            x.len() == y.len()
                && x.iter()
                    .all(|(key, p)| y.get(key).is_some_and(|q| same(p, q)))
        }
        _ => a == b,
    }
}

fn short(value: &Value) -> String {
    let text = value.to_string();
    if text.chars().count() > 300 {
        let cut: String = text.chars().take(300).collect();
        return format!("{cut}...");
    }
    text
}

/// What is wrong with `got` for `case`, if anything.
fn problem(case: &Value, got: &Value) -> Option<String> {
    let Value::Object(found) = got else {
        return Some(format!("gave {}, not an object", short(got)));
    };
    if let Some(expected) = case.get("expected") {
        if found.len() != 1 {
            return Some(format!("gave {}, expected an output", short(got)));
        }
        return match found.get("output") {
            Some(output) if same(output, expected) => None,
            Some(output) => Some(format!(
                "gave {}, expected {}",
                short(output),
                short(expected)
            )),
            None => Some(format!("gave {}, expected an output", short(got))),
        };
    }
    let want = &case["expected_error"];
    let Some(error) = found.get("error") else {
        return Some(format!("gave {}, expected {}", short(got), short(want)));
    };
    if found.len() != 1 {
        return Some(format!("gave {} beside the error", short(got)));
    }
    let Value::Object(error) = error else {
        return Some(format!("an error that is not an object: {}", short(error)));
    };
    let keys: Vec<&String> = error.keys().collect();
    if keys.len() != 2 || !error.contains_key("code") || !error.contains_key("message") {
        return Some(format!("an error with keys {keys:?}, not code and message"));
    }
    if error["code"] != want["code"] {
        return Some(format!(
            "code {}, expected {}",
            short(&error["code"]),
            short(&want["code"])
        ));
    }
    if let Some(message) = want.get("message")
        && error["message"] != *message
    {
        return Some(format!(
            "message {}, expected {}",
            short(&error["message"]),
            short(message)
        ));
    }
    None
}

fn check(name: &str, at_least: usize) {
    let all = cases(name);
    assert!(
        all.len() >= at_least,
        "{name}.json has {} cases, fewer than {at_least}",
        all.len()
    );
    let mut failures = Vec::new();
    for case in &all {
        let id = case["id"].as_str().unwrap_or("?");
        let operation = case["operation"]
            .as_str()
            .expect("a case names its operation");
        let input = case["input"].to_string();
        match sdm_schema::fixture_op(operation, &input) {
            Err(error) => failures.push(format!("{id}: refused: {error}")),
            Ok(text) => match serde_json::from_str::<Value>(&text) {
                Err(error) => failures.push(format!("{id}: not JSON ({error}): {text}")),
                Ok(got) => {
                    if let Some(found) = problem(case, &got) {
                        failures.push(format!("{id}: {found}"));
                    }
                }
            },
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} {name} cases fail:\n{}",
        failures.len(),
        all.len(),
        failures.join("\n")
    );
}

#[test]
fn every_dates_case() {
    check("dates", 185);
}

#[test]
fn every_composite_key_case() {
    check("composite-key", 40);
}

#[test]
fn an_unknown_operation_is_refused() {
    assert!(sdm_schema::fixture_op("dates.no_such_operation", "{}").is_err());
    assert!(sdm_schema::fixture_op("composite.encode_values", "not json").is_err());
}
