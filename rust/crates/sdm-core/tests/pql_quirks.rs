// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The 29 grammar and date quirks (`tests/pql/quirks.json`): the first
//! attempt's rows, answered again by the reference (`record_quirks.py`),
//! each held through the adapter.
//!
//! A query row compares as a corpus case does (`pql_compare`). A date row is
//! read as a literal of a query's filter: the date and offset the reference
//! reads, or `PQL_VALUE_INVALID_LITERAL` where it refuses, and D4's warning
//! by format.
//!
//! From T1.6c a row whose date literal has a run of five digits or more
//! (D8) is held to the core's refusal by name (`date_literal_packed`)
//! instead, whatever the reference read; its recorded reading stays in the
//! table as evidence of the old behaviour.
//!
//! From T2.18 a query row marked `core_only` is held to the core's own
//! answer there (its `errors`), where the reference refuses a spelling the
//! core reads (KO2-36, register P2E-1); the reference's answer stays in the
//! row, and the core must differ from it, so a mark that no longer differs
//! fails.

#![allow(
    clippy::disallowed_methods,
    reason = "test code: reads the quirk table"
)]

mod pql_compare;

use std::collections::BTreeSet;
use std::path::Path;

use pql_compare::{
    NoZones, differences, facts, holds_packed_date, is_packed_refusal, profile, query_schema,
};
use sdm_contract::Detail;
use sdm_contract::pql::{NodeKind, PqlLiteral, PqlNode};
use sdm_core::pql::resolve;
use serde_json::Value;

fn table() -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/pql/quirks.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn rows<'a>(table: &'a Value, key: &str) -> &'a Vec<Value> {
    table[key].as_array().unwrap()
}

#[test]
fn every_quirk_has_a_row() {
    let table = table();
    let named: BTreeSet<&str> = table["quirks"]
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    let covered: BTreeSet<&str> = rows(&table, "queries")
        .iter()
        .chain(rows(&table, "dates"))
        .map(|row| row["quirk"].as_str().unwrap())
        .collect();
    assert_eq!(named.len(), 29, "the 29 quirks");
    assert_eq!(named, covered, "each quirk has a row, and each row a quirk");
    assert_eq!(rows(&table, "queries").len(), 212);
    assert_eq!(rows(&table, "dates").len(), 55);
}

/// Every query row as the reference answers it, except the one whose date
/// literal is D8 (`date-digit-counts/long-year`, `12345-1-1`, which the
/// reference refuses as `PQL_VALUE_INVALID_LITERAL`): from T1.6c the core
/// refuses it by name (brief: "update the quirk expectations for D8
/// literals from read to refused").
/// The two rows marked `core_only` (T2.18) are held to the core's answer
/// instead, and to differing from the reference's.
#[test]
fn every_query_row_answers_as_the_reference_does() {
    let table = table();
    let mut failures = Vec::new();
    let mut packed = Vec::new();
    let mut core_only = Vec::new();
    for row in rows(&table, "queries") {
        let schema = query_schema(&table["schemas"][row["schema"].as_str().unwrap()]);
        let profile = profile(row["profile"].as_str().unwrap());
        let query = row["query"].as_str().unwrap();
        if holds_packed_date(query) {
            packed.push(row["id"].as_str().unwrap());
            match resolve(query, &schema, &profile, &facts(), &NoZones) {
                Err(error) if is_packed_refusal(&error) => {}
                other => failures.push(format!("{}: not refused as packed: {other:?}", row["id"])),
            }
            continue;
        }
        if let Some(core) = row.get("core_only") {
            core_only.push(row["id"].as_str().unwrap());
            if differences(row, &schema, &profile).is_empty() {
                failures.push(format!(
                    "{}: marked core_only and answers as the reference",
                    row["id"]
                ));
            }
            let mut held = row.clone();
            held["errors"] = core["errors"].clone();
            for (part, how) in differences(&held, &schema, &profile) {
                failures.push(format!("{}: core_only: {part}: {how}", row["id"]));
            }
            continue;
        }
        for (part, how) in differences(row, &schema, &profile) {
            failures.push(format!("{}: {part}: {how}", row["id"]));
        }
    }
    assert_eq!(packed, ["date-digit-counts/long-year"]);
    assert_eq!(
        core_only,
        ["backticks/unknown-with-dot", "backticks/newline-inside"]
    );
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// The first time constant under `node`: its reading and offset.
fn time_of(node: &PqlNode) -> Option<(String, Option<i32>)> {
    match &node.kind {
        NodeKind::Constant {
            literal: PqlLiteral::Time { value, offset, .. },
        } => Some((value.clone(), *offset)),
        NodeKind::Join { target, .. } | NodeKind::Aggregation { target, .. } => time_of(target),
        NodeKind::Filter { target, condition } => time_of(target).or_else(|| time_of(condition)),
        NodeKind::Condition { target, value, .. } => time_of(target).or_else(|| time_of(value)),
        _ => None,
    }
}

/// Every date row as the reference reads it, except the four with a run of
/// five digits or more (`10000-01-01`, `230203-06-09`, `20240101-01-01`,
/// `1-1-20240101`), which from T1.6c the core refuses by name, where the
/// reference read two and refused two: no reading, no diagnostic, no
/// warning (brief: "update the quirk expectations for D8 literals").
#[test]
fn every_date_row_reads_as_the_reference_reads_it() {
    let table = table();
    let schema = query_schema(&table["schemas"]["shop"]);
    let relational = profile("RELATIONAL");
    let mut failures = Vec::new();
    let mut packed = Vec::new();
    for row in rows(&table, "dates") {
        let literal = row["literal"].as_str().unwrap();
        let query = format!(
            "PREDICT COUNT(orders.* WHERE orders.ordered_at > {literal}, 0, 30, days) FOR EACH \
             users.user_id"
        );
        if holds_packed_date(&query) {
            packed.push(literal);
            match resolve(&query, &schema, &relational, &facts(), &NoZones) {
                Err(error) if is_packed_refusal(&error) => {}
                other => failures.push(format!("{literal}: not refused as packed: {other:?}")),
            }
            continue;
        }
        let got = resolve(&query, &schema, &relational, &facts(), &NoZones).unwrap();
        let read = match &got.query {
            Some(query) => time_of(&query.target),
            None => {
                let codes: Vec<&str> = got.diagnostics.iter().map(|d| d.code.as_str()).collect();
                if codes != ["PQL_VALUE_INVALID_LITERAL"] {
                    failures.push(format!("{literal}: refused with {codes:?}"));
                }
                None
            }
        };
        let want = row["value"]
            .as_str()
            .map(|v| (v.to_owned(), row["offset"].as_i64().map(|o| o as i32)));
        if read != want {
            failures.push(format!("{literal}: read {read:?}, the reference {want:?}"));
        }
        let warned: BTreeSet<String> = got
            .warnings
            .iter()
            .map(|w| match &w.fields["format"] {
                Detail::Scalar(sdm_contract::Scalar::Utf8(f)) => f.clone(),
                other => panic!("{other:?}"),
            })
            .collect();
        let said: BTreeSet<String> = row["warnings"]
            .as_array()
            .unwrap()
            .iter()
            .map(|w| pql_compare::reference_format(w["message"].as_str().unwrap()))
            .collect();
        if warned != said {
            failures.push(format!(
                "{literal}: warned {warned:?}, the reference {said:?}"
            ));
        }
    }
    assert_eq!(
        packed,
        [
            "10000-01-01",
            "230203-06-09",
            "20240101-01-01",
            "1-1-20240101"
        ]
    );
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
