// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! T1.33 part 3, held from outside the implementation (the package's
//! independent test author) to `contract/t133p3-vectors.json`, on the
//! owner's answers of 2026-10-14:
//!
//! The types' rows are `contract/type-set-vectors.json`'s with `cells`
//! (one source of truth with T1.33 part 2's type set); this file's own
//! rows are controls and the `text_time` rows.
//!
//! - Item 26 (P1B-43): support, not skip, the types a lossless DuckDB
//!   connection hands over. A UUID and a BIT are read as text (the
//!   canonical lower-case 8-4-4-4-12 and the '0'/'1' string), a TIMETZ as a
//!   time of day as a TIME is today (`time.control`), and lossless JSON
//!   stays text. A HUGEINT, a UHUGEINT and a VARINT (with or without the
//!   setting) follow the big-integer rule (the orchestrator's, 2026-10-14):
//!   received as `int64` where every value fits it, else as
//!   `decimal128(38, 0)` where every value fits 38 digits, else as text
//!   (`utf8`), each cell DuckDB's exact text in every case.
//! - Item 24 (text_time): a column whose name says time but whose values
//!   are no times is text, not refused (rel-trial's `time_frame`, "up to
//!   12 weeks"); one whose values read as times stays the time column.
//!
//! Each row's `witness` adds its column to `shop`'s `users`, and
//! `nested.list-int`'s case runs on this host through real DuckDB, to
//! `predict`. The row holds the column's description (its data type as
//! the wire names it, and its semantic type and the table's time column
//! where given) and, in the model call's body, each `users` row's cell of
//! the column: `cast`, equal to DuckDB's own `CAST(column AS VARCHAR)` for
//! that user, read on a connection of the same base and witness (a
//! number's cell as its text, so each value is exact); `null`, every
//! cell null.
//!
//! Gating: `contract/tests/t133p3-pending.json`'s `pending["rust-host"]`
//! lists the rows that failed here when the test was written. Each must
//! still fail (strict: one that passes must be taken out of the list),
//! and the list may only shrink: it stays within `authored["rust-host"]`.
//! The implementer removes ids as they pass and changes nothing else of
//! this file or the vectors.

use std::collections::{BTreeMap, BTreeSet};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;

use duckdb::arrow::array::{Array, AsArray};
use duckdb::arrow::datatypes::DataType;
use sdm_testkit::bases::{self, Bases};
use sdm_testkit::case::{self, Case};
use sdm_testkit::duck;
use sdm_testkit::host::{self, Answer, Observed};
use serde_json::{Value, json};

const PENDING: &str = "contract/tests/t133p3-pending.json";
const VECTORS: &str = "contract/t133p3-vectors.json";
const TYPE_SET: &str = "contract/type-set-vectors.json";

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

fn read_json(path: &str) -> Value {
    let text = std::fs::read_to_string(root().join(path)).unwrap();
    serde_json::from_str(&text).unwrap()
}

/// This test's rows: the type set's rows with `cells` (the types P1B-43
/// reads, by the big-integer rule), as rows of this file's shape, then
/// `contract/t133p3-vectors.json`'s own (controls and `text_time`).
fn rows() -> Vec<Value> {
    let types = read_json(TYPE_SET);
    assert_eq!(types["format"], "sdm-type-set-vectors/1");
    let mut found: Vec<Value> = types["vectors"]
        .as_array()
        .unwrap()
        .iter()
        .chain(types["reached"].as_array().unwrap())
        .filter(|row| row.get("cells").is_some_and(|c| !c.is_null()))
        .map(|row| {
            let mut want = json!({
                "dtype": wire(row["inference"]["dtype"].as_str().expect("a row with cells is read")),
                "cells": row["cells"],
            });
            if let Some(received) = row.get("received") {
                want["received"] = received.clone();
            }
            json!({"id": row["id"], "column": "f", "witness": row["witness"], "want": want})
        })
        .collect();
    let all = read_json(VECTORS);
    assert_eq!(all["format"], "sdm-t133p3-vectors/1");
    found.extend(all["rows"].as_array().unwrap().iter().cloned());
    found
}

fn ids(value: &Value) -> BTreeSet<String> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_owned())
        .collect()
}

fn witness(row: &Value) -> Vec<String> {
    row["witness"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s.as_str().unwrap().to_owned())
        .collect()
}

/// A data type as the wire names it.
fn wire(dtype: &str) -> String {
    match dtype {
        "int" => "int64",
        "float" => "float64",
        "date" | "time" => "timestamp[us]",
        other => other,
    }
    .to_owned()
}

/// Each user's value of `column` as DuckDB itself writes it as text, by
/// the user's id as text: `shop` and the witness on a fresh connection.
fn duckdb_text(bases: &Bases, row: &Value) -> BTreeMap<String, Option<String>> {
    let connection = duck::open().unwrap();
    for statement in bases["shop"].iter().cloned().chain(witness(row)) {
        connection.execute_batch(&statement).unwrap();
    }
    let column = row["column"].as_str().unwrap();
    let fetched = duck::run(
        &connection,
        &format!("SELECT CAST(user_id AS VARCHAR) AS k, CAST({column} AS VARCHAR) AS v FROM users"),
    )
    .unwrap();
    let mut found = BTreeMap::new();
    for batch in &fetched.batches {
        let text = |i: usize| {
            let array = batch.column(i);
            match array.data_type() {
                DataType::Utf8View => array
                    .as_string_view()
                    .clone()
                    .iter()
                    .map(|v| v.map(str::to_owned))
                    .collect::<Vec<_>>(),
                // Under arrow_large_buffer_size, DuckDB's text is large.
                DataType::LargeUtf8 => array
                    .as_string::<i64>()
                    .iter()
                    .map(|v| v.map(str::to_owned))
                    .collect(),
                _ => array
                    .as_string::<i32>()
                    .iter()
                    .map(|v| v.map(str::to_owned))
                    .collect(),
            }
        };
        for (k, v) in text(0).into_iter().zip(text(1)) {
            found.insert(k.unwrap(), v);
        }
    }
    found
}

/// A cell of the body as text: a number's digits, text as itself.
fn cell_text(cell: &Value) -> Option<String> {
    match cell {
        Value::Null => None,
        Value::String(s) => Some(s.clone()),
        // A number as the host records it, its text (`numbers_as_text`).
        Value::Object(o) if o.len() == 1 && o["number"].is_string() => {
            o["number"].as_str().map(str::to_owned)
        }
        other => Some(other.to_string()),
    }
}

/// The body's `users` rows of `column`, by user id, from each part of the
/// body that holds `users` rows.
fn body_cells(body: &Value, column: &str) -> Option<BTreeMap<String, Option<String>>> {
    let mut found = BTreeMap::new();
    let mut held = false;
    for part in ["predict", "context"] {
        let users = &body[part]["related_tables"]["users"];
        let Some(columns) = users["columns"].as_array() else {
            continue;
        };
        let at = |name: &str| columns.iter().position(|c| c == name);
        let (Some(key), Some(value)) = (at("user_id"), at(column)) else {
            continue;
        };
        held = true;
        for r in users["rows"].as_array().into_iter().flatten() {
            if let Some(k) = cell_text(&r[key]) {
                found.insert(k, cell_text(&r[value]));
            }
        }
    }
    held.then_some(found)
}

/// What this host made of the row's column: its description and its
/// cells in the body.
fn project(observed: &Observed, column: &str) -> Value {
    let users = observed.described.as_ref().and_then(|d| {
        d["tables"]
            .as_array()
            .into_iter()
            .flatten()
            .find(|t| t["name"] == "users")
            .cloned()
    });
    let described = users.as_ref().and_then(|u| {
        u["columns"]
            .as_array()
            .into_iter()
            .flatten()
            .find(|c| c["name"] == column)
            .cloned()
    });
    let answer = match &observed.answer {
        Answer::Predicted(_) => json!("success"),
        Answer::Error { code, fields, .. } => json!({"refusal": code, "rule": fields.get("rule")}),
        other => json!(format!("{other:?}")),
    };
    json!({
        "answer": answer,
        "dtype": described.as_ref().map(|c| wire(c["dtype"].as_str().unwrap_or("?"))),
        "stype": described.as_ref().map(|c| c["stype"].clone()),
        "received": described.as_ref().map(|c| c["received"].clone()),
        "time_column": users.as_ref().map(|u| u["time_column"].clone()),
    })
}

/// Why the row fails, or `None`.
fn problem(row: &Value, bases: &Bases, template: &Case) -> Option<String> {
    let id = row["id"].as_str().unwrap();
    let column = row["column"].as_str().unwrap();
    let want = &row["want"];
    let mut case = template.clone();
    case.id = format!("t133p3.{id}");
    case.input.setup = Some(witness(row));
    let observed = match catch_unwind(AssertUnwindSafe(|| host::run(&case, bases))) {
        Ok(Ok(observed)) => observed,
        Ok(Err(failure)) => return Some(format!("host failure: {failure}")),
        Err(_) => return Some("host panic".into()),
    };
    let got = project(&observed, column);
    if got["answer"] != "success" {
        return Some(format!("not a success: {got}"));
    }
    if got["dtype"] != want["dtype"] {
        return Some(format!("described {got}, want dtype {}", want["dtype"]));
    }
    if let Some(stype) = want.get("stype")
        && &got["stype"] != stype
    {
        return Some(format!("described {got}, want stype {stype}"));
    }
    if let Some(received) = want.get("received")
        && &got["received"] != received
    {
        return Some(format!("described {got}, want received {received}"));
    }
    if let Some(time_column) = want.get("time_column")
        && &got["time_column"] != time_column
    {
        return Some(format!("described {got}, want time column {time_column}"));
    }
    let Answer::Predicted(predicted) = &observed.answer else {
        unreachable!("a success is a prediction");
    };
    let cells = match body_cells(&predicted["body"], column) {
        Some(cells) if !cells.is_empty() => cells,
        _ => return Some(format!("{column} is in no users row of the body")),
    };
    match want["cells"].as_str() {
        Some("cast") => {
            let duck = duckdb_text(bases, row);
            for (user, cell) in &cells {
                let expected = duck.get(user).cloned().flatten();
                if cell != &expected {
                    return Some(format!("user {user}: cell {cell:?}, DuckDB {expected:?}"));
                }
            }
        }
        Some("null") if cells.values().any(Option::is_some) => {
            return Some(format!("cells {cells:?}, want every one null"));
        }
        _ => {}
    }
    if want["joins"] == "own" {
        return joined_elsewhere(&predicted["body"]);
    }
    None
}

/// Where the body joins a row to another entity's, or `None`: under
/// `nested.list-int`'s query (users, their orders by `user_id`, and those
/// orders' users), every `users` and `orders` row an instance holds, in
/// `predict` and `context`, has the `user_id` of the instance's own
/// entity. A seed join that matches a key it does not equal (the
/// subagent's round 2 P1: a UHUGEINT key against a HUGEINT seed) puts
/// another user's rows under an instance.
fn joined_elsewhere(body: &Value) -> Option<String> {
    for part in ["predict", "context"] {
        let instances = &body[part]["instance_table"];
        let columns = instances["columns"].as_array()?;
        let entity_at = columns.iter().position(|c| c == "__kumo_entity_ref_0")?;
        let entity: BTreeMap<String, Option<String>> = instances["rows"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|r| Some((cell_text(&r[0])?, cell_text(&r[entity_at]))))
            .collect();
        for table in ["users", "orders"] {
            let held = &body[part]["related_tables"][table];
            let Some(columns) = held["columns"].as_array() else {
                return Some(format!("{part}: no {table} rows"));
            };
            let at = |name: &str| columns.iter().position(|c| c == name);
            let (Some(instance), Some(key)) = (at("instance_id"), at("user_id")) else {
                return Some(format!("{part}: {table} has no instance_id or user_id"));
            };
            for row in held["rows"].as_array().into_iter().flatten() {
                let id = cell_text(&row[instance]);
                let own = id.as_ref().and_then(|i| entity.get(i)).cloned().flatten();
                let user = cell_text(&row[key]);
                if user != own {
                    return Some(format!(
                        "{part}: instance {id:?} of user {own:?} holds a {table} row of user {user:?}"
                    ));
                }
            }
        }
    }
    None
}

#[test]
fn the_pending_list_only_shrinks() {
    let pending = read_json(PENDING);
    let now = ids(&pending["pending"]["rust-host"]);
    let authored = ids(&pending["authored"]["rust-host"]);
    assert!(
        now.is_subset(&authored),
        "pending ids added: {:?}",
        now.difference(&authored)
    );
    let all: BTreeSet<String> = rows()
        .iter()
        .map(|r| r["id"].as_str().unwrap().to_owned())
        .collect();
    assert!(now.is_subset(&all), "pending ids that are no row");
}

#[test]
fn each_row_is_read_as_the_owner_decided() {
    let contract = root().join("contract");
    let loaded = case::load(&[contract.join("cases")]).unwrap();
    let template = loaded
        .iter()
        .find(|l| l.case.id == "nested.list-int")
        .expect("nested.list-int is a case")
        .case
        .clone();
    let bases = bases::load(&contract.join("bases.yaml")).unwrap();
    let pending = ids(&read_json(PENDING)["pending"]["rust-host"]);
    let mut failures = Vec::new();
    let mut passing_but_pending = Vec::new();
    for row in rows() {
        let id = row["id"].as_str().unwrap().to_owned();
        match (problem(&row, &bases, &template), pending.contains(&id)) {
            (None, true) => passing_but_pending.push(id),
            (Some(why), false) => failures.push(format!("{id}: {why}")),
            _ => {}
        }
    }
    assert!(
        passing_but_pending.is_empty(),
        "these pass now: take them out of {PENDING}'s pending[\"rust-host\"]: {passing_but_pending:?}"
    );
    assert!(
        failures.is_empty(),
        "{} rows differ:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
