// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! P6.dialect-calls: every call of the dialect-call capture, read straight
//! from `tests/parity/dialects/`, gives from `sdm_dialects::render_json` the
//! SQL Python wrote, byte for byte, and the same parameters or result, by
//! value and JSON type (`tests/rust_core/dialects/CONTRACT.md`).

use std::fs;
use std::path::PathBuf;

use serde_json::{Value, json};

const BACKENDS: [&str; 5] = ["databricks", "duckdb", "postgres", "snow", "sqlite"];

fn repo() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .ancestors()
        .nth(3)
        .expect("the crate is rust/crates/dialects in the repository")
        .to_path_buf()
}

fn capture(backend: &str) -> Value {
    let path = repo()
        .join("tests/parity/dialects")
        .join(format!("{backend}.json"));
    let text = fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let data: Value =
        serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert_eq!(data["format"], json!(1), "{backend}.json: format is not 1");
    assert_eq!(data["backend"], json!(backend), "{backend}.json: backend");
    data
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

fn sql_difference(want: &str, got: &str) -> String {
    let at = want
        .chars()
        .zip(got.chars())
        .take_while(|(a, b)| a == b)
        .count();
    let around =
        |text: &str| -> String { text.chars().skip(at.saturating_sub(40)).take(80).collect() };
    format!(
        "SQL differs at character {at}: captured {:?}, got {:?}",
        around(want),
        around(got)
    )
}

fn check(backend: &str, call: &Value) -> Option<String> {
    let function = call["function"].as_str()?;
    let args = serde_json::to_string(&call["args"]).expect("args serialize");
    let got = match sdm_dialects::render_json(backend, function, &args) {
        Ok(text) => text,
        Err(error) => return Some(format!("refused: {error}")),
    };
    let got: Value = match serde_json::from_str(&got) {
        Ok(value) => value,
        Err(error) => return Some(format!("gave text that is not JSON: {error}")),
    };
    if call.get("result").is_some() {
        let want = &call["result"];
        return match got.get("result") {
            Some(value) if same(want, value) => None,
            Some(value) => Some(format!("result {} != {}", short(value), short(want))),
            None => Some(format!("gave {} where a result was captured", short(&got))),
        };
    }
    let want_sql = call["sql"].as_str().unwrap_or_default();
    match got.get("sql").and_then(Value::as_str) {
        Some(sql) if sql == want_sql => {}
        Some(sql) => return Some(sql_difference(want_sql, sql)),
        None => {
            return Some(format!(
                "gave {} where a statement was captured",
                short(&got)
            ));
        }
    }
    let want = &call["params"];
    match got.get("params") {
        Some(value) if same(want, value) => None,
        Some(value) => Some(format!("params {} != {}", short(value), short(want))),
        None => Some("gave no params".to_string()),
    }
}

#[test]
fn every_captured_call() {
    let mut checked = 0;
    let mut failures = Vec::new();
    for backend in BACKENDS {
        let data = capture(backend);
        let calls = data["calls"]
            .as_array()
            .unwrap_or_else(|| panic!("{backend}.json has no calls"));
        assert!(!calls.is_empty(), "{backend}.json has no calls");
        for (i, call) in calls.iter().enumerate() {
            checked += 1;
            if let Some(problem) = check(backend, call) {
                let function = call["function"].as_str().unwrap_or("?");
                let source = call["sources"][0].as_str().unwrap_or("?");
                failures.push(format!("{backend}#{i} {function} [{source}]: {problem}"));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {checked} captured calls differ:\n{}",
        failures.len(),
        failures
            .iter()
            .take(40)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
}
