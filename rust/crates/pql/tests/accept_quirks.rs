// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! P5.quirks: every grammar and date quirk the Phase 5 brief lists, one row
//! per input, each with today's Python answer recorded beside it.
//!
//! The table is `accept_quirks.json`, beside this file, written by
//! `tests/rust_core/pql/quirks.py`, which also checks that each recorded
//! answer is still what today's Python gives. A query row compares as a
//! corpus case does; a date row compares the date and the warnings of
//! `parse_date_json` with what `pd.to_datetime` gave.

mod accept_common;

use std::collections::BTreeSet;

use accept_common::{mismatch, parse_date, read_json, report, resolve};
use serde_json::Value;

fn table() -> Value {
    read_json(&std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/accept_quirks.json"))
}

fn rows<'a>(table: &'a Value, key: &str) -> &'a Vec<Value> {
    table[key]
        .as_array()
        .unwrap_or_else(|| panic!("the quirk table has no {key}"))
}

#[test]
fn every_quirk_has_a_row() {
    let table = table();
    let named: BTreeSet<&str> = table["quirks"]
        .as_object()
        .expect("the quirk table names its quirks")
        .keys()
        .map(String::as_str)
        .collect();
    let covered: BTreeSet<&str> = rows(&table, "queries")
        .iter()
        .chain(rows(&table, "dates"))
        .map(|row| row["quirk"].as_str().expect("a row names its quirk"))
        .collect();
    let missing: Vec<&&str> = named.difference(&covered).collect();
    assert!(missing.is_empty(), "quirks without a row: {missing:?}");
    assert!(named.len() >= 29, "the brief lists at least 29 quirks");
}

#[test]
fn every_query_row() {
    let table = table();
    let mut failures = Vec::new();
    let queries = rows(&table, "queries");
    for row in queries {
        let id = row["id"].as_str().unwrap_or("?");
        let schema = &table["schemas"][row["schema"].as_str().expect("a schema name")];
        let query = row["query"].as_str().expect("a query");
        let profile = row["profile"].as_str().expect("a profile");
        let found = match resolve(query, schema, profile, None) {
            Ok(found) => mismatch(&row["python"], &found),
            Err(error) => Some(error),
        };
        if let Some(problem) = found {
            failures.push(format!("{id} ({query:?}): {problem}"));
        }
    }
    report("quirk queries", queries.len(), &failures);
}

#[test]
fn every_date_row() {
    let table = table();
    let mut failures = Vec::new();
    let dates = rows(&table, "dates");
    for row in dates {
        let id = row["id"].as_str().unwrap_or("?");
        let literal = row["literal"].as_str().expect("a literal");
        match parse_date(literal) {
            Ok(found) if found == row["python"] => {}
            Ok(found) => failures.push(format!("{id}: {found} != {}", row["python"])),
            Err(error) => failures.push(format!("{id}: {error}")),
        }
    }
    report("quirk dates", dates.len(), &failures);
}
