// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! T1.33's type set through real DuckDB on `rust-host`, held from outside
//! the implementation (the package's independent test author) to
//! `contract/type-set-vectors.json`.
//!
//! `sdm-host`'s `tests/t133_type_set.rs` holds the core to each vector with
//! a column built as Arrow. This file holds the route a user takes: each
//! vector and `reached` row with a `witness` (the DuckDB statements, after
//! `shop`'s, that add a column `f` of the type to `users`, under the
//! session setting that hands the type over) runs through this host's
//! DuckDB provider as `nested.list-int` does; its `describe_source` is
//! held (a run that goes on is held to nothing more here). The
//! answer must be the row's `inference`: `f` described with its data type
//! (as the wire names it: `int64`, `float64`, `string`, `bool`,
//! `timestamp[us]` for a date or a time), left out with the core's
//! `COLUMNS_LEFT_OUT` warning naming it, or the refusal with its code (and
//! rule, for `UNSUPPORTED_REQUEST`). A session setting changes nothing
//! else: on a connection with `arrow_lossless_conversion`, which hands the
//! catalog's own booleans over as `arrow.bool8`, every table is described
//! (the orchestrator's decision of 2026-10-14: the catalog read is made
//! robust to the setting, not a new refusal), a plain column as it is
//! without the setting, and a boolean read as one (`lossless.boolean`).
//!
//! Gating: `contract/tests/t133p2-pending.json`'s
//! `pending["rust-host-duckdb"]` lists the rows that failed here when the
//! test was written. Each must still fail (strict: one that passes must be
//! taken out of the list), and the list may only shrink: it stays within
//! `authored["rust-host-duckdb"]`. The implementer removes ids as they
//! pass and changes nothing else of this file or the vectors.

use std::collections::BTreeSet;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;

use sdm_testkit::bases;
use sdm_testkit::case::{self, Case};
use sdm_testkit::host::{self, Answer, Observed};
use serde_json::{Value, json};

const PENDING: &str = "contract/tests/t133p2-pending.json";
const GROUP: &str = "rust-host-duckdb";

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

fn ids(value: &Value) -> BTreeSet<String> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_owned())
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
fn the_pending_list_only_shrinks() {
    let pending = read_json(PENDING);
    let now = ids(&pending["pending"][GROUP]);
    let authored = ids(&pending["authored"][GROUP]);
    assert!(
        now.is_subset(&authored),
        "pending ids added: {:?}",
        now.difference(&authored)
    );
    let all: BTreeSet<String> = witnessed()
        .iter()
        .map(|row| row["id"].as_str().unwrap().to_owned())
        .collect();
    assert!(now.is_subset(&all), "pending ids that are no witnessed row");
}

#[test]
fn each_witnessed_type_through_duckdb_is_described_as_its_row_says() {
    let contract = root().join("contract");
    let loaded = case::load(&[contract.join("cases")]).unwrap();
    let template = loaded
        .iter()
        .find(|l| l.case.id == "nested.list-int")
        .expect("nested.list-int is a case")
        .case
        .clone();
    let bases = bases::load(&contract.join("bases.yaml")).unwrap();
    let pending = ids(&read_json(PENDING)["pending"][GROUP]);
    let rows = witnessed();
    assert!(rows.len() > 50, "{} witnessed rows", rows.len());
    let mut failures = Vec::new();
    let mut passing_but_pending = Vec::new();
    for row in &rows {
        let id = row["id"].as_str().unwrap();
        let want = wanted(row);
        let case = the_case(&template, row);
        let got = match catch_unwind(AssertUnwindSafe(|| host::run(&case, &bases))) {
            Ok(Ok(observed)) => project(&observed),
            Ok(Err(failure)) => json!({"outcome": "host failure", "why": failure.to_string()}),
            Err(payload) => json!({"outcome": "host panic", "why": panic_text(payload.as_ref())}),
        };
        match (got == want, pending.contains(id)) {
            (true, true) => passing_but_pending.push(id.to_owned()),
            (false, false) => failures.push(format!("{id}: got {got}, want {want}")),
            _ => {}
        }
    }
    assert!(
        passing_but_pending.is_empty(),
        "these pass now: take them out of {PENDING}'s pending[\"{GROUP}\"]: \
         {passing_but_pending:?}"
    );
    assert!(
        failures.is_empty(),
        "{} rows differ:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
