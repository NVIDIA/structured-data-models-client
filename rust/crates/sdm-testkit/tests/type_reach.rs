// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Type inference through the real DuckDB provider.
//! Witness SQL constructs each supported fixture type, including lossless Arrow extensions.
//! Every observed description, omission warning or named refusal must match its explicit vector.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;

use sdm_testkit::bases;
use sdm_testkit::case::{self, Case};
use sdm_testkit::host::{self, Answer, Observed};
use serde_json::{Value, json};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

fn read_json(path: &str) -> Value {
    let text = std::fs::read_to_string(root().join(path)).unwrap();
    serde_json::from_str(&text).unwrap()
}

/// Every row with a witness: the type set's vectors and its `reached`
/// rows.
fn witnessed() -> Vec<Value> {
    let all = read_json("contract/type-set-vectors.json");
    assert_eq!(all["format"], "sdm-type-set-vectors/1");
    all["vectors"]
        .as_array()
        .unwrap()
        .iter()
        .chain(all["reached"].as_array().unwrap())
        .filter(|row| row.get("witness").is_some_and(|w| !w.is_null()))
        .cloned()
        .collect()
}

/// A data type as the wire names it.
fn wire(dtype: &str) -> &str {
    match dtype {
        "int" => "int64",
        "float" => "float64",
        "date" | "time" => "timestamp[us]",
        other => other,
    }
}

/// The row's `inference` in this host's terms.
fn wanted(row: &Value) -> Value {
    let inference = &row["inference"];
    if let Some(dtype) = inference["dtype"].as_str() {
        return json!({"outcome": "success", "f": wire(dtype)});
    }
    if inference["left_out"].is_string() {
        return json!({"outcome": "success", "f": "left out"});
    }
    let refusal = &inference["refusal"];
    json!({
        "outcome": "refusal",
        "code": refusal["code"],
        "rule": refusal.get("rule").cloned().unwrap_or(Value::Null),
    })
}

/// What this host's `describe_source` made of `f` in `users`: from its
/// description when it gave one (the case runs on to `predict`, whose
/// answer is not this test's), else its refusal.
fn project(observed: &Observed) -> Value {
    let described = match (&observed.described, &observed.answer) {
        (Some(described), _) | (None, Answer::Described(described)) => Some(described),
        _ => None,
    };
    match (described, &observed.answer) {
        (Some(described), _) => {
            let users = described["tables"]
                .as_array()
                .into_iter()
                .flatten()
                .find(|t| t["name"] == "users");
            let Some(users) = users else {
                return json!({"outcome": "success", "f": "no table users"});
            };
            let column = users["columns"]
                .as_array()
                .into_iter()
                .flatten()
                .find(|c| c["name"] == "f");
            let f = match column {
                Some(column) => Value::from(wire(column["dtype"].as_str().unwrap_or("?"))),
                None => {
                    let warned = observed.warnings.iter().any(|w| {
                        w["code"] == "COLUMNS_LEFT_OUT"
                            && w["fields"].to_string().contains("\"users\"")
                            && w["fields"].to_string().contains("\"f\"")
                    });
                    Value::from(if warned {
                        "left out"
                    } else {
                        "left out with no warning"
                    })
                }
            };
            json!({"outcome": "success", "f": f})
        }
        (None, Answer::Error { code, fields, .. }) => json!({
            "outcome": "refusal",
            "code": code,
            "rule": fields.get("rule").cloned(),
        }),
        (None, other) => json!({"outcome": format!("no answer of describe_source: {other:?}")}),
    }
}

/// What a panic said.
fn panic_text(payload: &(dyn std::any::Any + Send)) -> String {
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_owned()))
        .unwrap_or_default()
}

/// The row as a case: `nested.list-int`'s, its setup the witness.
fn the_case(template: &Case, row: &Value) -> Case {
    let mut case = template.clone();
    case.id = format!("t133.{}", row["id"].as_str().unwrap());
    case.input.setup = Some(
        row["witness"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| s.as_str().unwrap().to_owned())
            .collect(),
    );
    case
}

#[test]
fn each_witnessed_type_through_duckdb_is_described_as_its_row_says() {
    let contract = root().join("contract");
    let loaded = case::load_for(&[contract.join("cases")], case::Database::Duckdb).unwrap();
    let template = loaded
        .iter()
        .find(|l| l.case.id == "nested.list-int")
        .expect("nested.list-int is a case")
        .case
        .clone();
    let bases = bases::load(&contract.join("bases.yaml")).unwrap();
    let rows = witnessed();
    assert!(rows.len() > 50, "{} witnessed rows", rows.len());
    let mut failures = Vec::new();
    for row in &rows {
        let id = row["id"].as_str().unwrap();
        let want = wanted(row);
        let case = the_case(&template, row);
        let got = match catch_unwind(AssertUnwindSafe(|| host::run(&case, &bases))) {
            Ok(Ok(observed)) => project(&observed),
            Ok(Err(failure)) => json!({"outcome": "host failure", "why": failure.to_string()}),
            Err(payload) => json!({"outcome": "host panic", "why": panic_text(payload.as_ref())}),
        };
        if got != want {
            failures.push(format!("{id}: got {got}, want {want}"));
        }
    }
    assert!(
        failures.is_empty(),
        "{} rows differ:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
