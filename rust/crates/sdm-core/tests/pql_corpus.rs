// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The PQL corpus (`tests/pql/corpus.json`): 417 queries, 193 the reference
//! accepts and 224 it refuses, each run through the adapter and compared
//! with the reference's answer.
//!
//! An accepted query gives the reference's validated query: its tree, read
//! into the validated-query format, equals the adapter's, field for field,
//! and each date literal reads as the reference reads it (`reading`,
//! recorded by `record_readings.py`). A refused query gives the same errors
//! in count, code and span, unknown-name errors as a multiset, since the
//! reference orders them from a Python set; messages are not promised. The
//! date warnings the reference gave are compared by format (D4).
//!
//! The cases that differ, and the parts of each that do, are pinned in
//! `tests/pql/known-differing.json`, empty since T1.6b. Any other
//! difference fails, and so does a pinned part that agrees, so the list can
//! only shrink.

#![allow(
    clippy::disallowed_methods,
    reason = "test code: reads the corpus file"
)]

mod pql_compare;

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use pql_compare::{differences, in_comparable_order, profile, query_schema, text};
use sdm_contract::ops::QuerySchema;
use serde_json::{Value, json};

fn read(name: &str) -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/pql")
        .join(name);
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn corpus() -> Value {
    read("corpus.json")
}

/// The pinned differences: each case's id and the parts of it that differ.
fn known_differing() -> BTreeMap<String, BTreeSet<String>> {
    read("known-differing.json")["cases"]
        .as_array()
        .unwrap()
        .iter()
        .map(|case| {
            let parts = case["parts"]
                .as_array()
                .unwrap()
                .iter()
                .map(|p| text(p).unwrap())
                .collect();
            (text(&case["id"]).unwrap(), parts)
        })
        .collect()
}
#[test]
fn unknown_name_errors_compare_as_a_multiset_and_the_rest_in_order() {
    let error = |code: &str, col: u32| (code.to_owned(), json!({"start_col": col}));
    let unknown = "PQL_SCHEMA_UNKNOWN_COLUMN";
    let a = vec![error(unknown, 1), error(unknown, 9), error("PQL_X", 3)];
    let b = vec![error(unknown, 9), error(unknown, 1), error("PQL_X", 3)];
    assert_eq!(in_comparable_order(a.clone()), in_comparable_order(b));
    let c = vec![error("PQL_X", 3), error(unknown, 1), error(unknown, 9)];
    assert_ne!(in_comparable_order(a), in_comparable_order(c));
    let d = vec![error("PQL_X", 3), error("PQL_Y", 1)];
    let e = vec![error("PQL_Y", 1), error("PQL_X", 3)];
    assert_ne!(in_comparable_order(d), in_comparable_order(e));
}

#[test]
fn the_corpus_agrees_with_the_reference() {
    let corpus = corpus();
    let schemas: BTreeMap<&str, QuerySchema> = corpus["schemas"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(name, schema)| (name.as_str(), query_schema(schema)))
        .collect();
    let cases = corpus["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 417, "the corpus holds 417 queries");
    let accepted = cases.iter().filter(|c| c.get("tree").is_some()).count();
    assert_eq!((accepted, cases.len() - accepted), (193, 224));
    let known = known_differing();
    let ids: BTreeSet<&str> = cases.iter().map(|c| c["id"].as_str().unwrap()).collect();
    let mut failures = Vec::new();
    for id in known.keys() {
        if !ids.contains(id.as_str()) {
            failures.push(format!("{id}: pinned as differing, and not in the corpus"));
        }
    }
    let mut differing = 0;
    for case in cases {
        let id = case["id"].as_str().unwrap();
        let schema = &schemas[case["schema"].as_str().unwrap()];
        let found = differences(case, schema, &profile(case["profile"].as_str().unwrap()));
        if !found.is_empty() {
            differing += 1;
        }
        let pinned = known.get(id).cloned().unwrap_or_default();
        for (part, how) in &found {
            let status = if pinned.contains(*part) {
                "pinned"
            } else {
                "NOT PINNED"
            };
            println!("  {id}: {part} ({status}): {how}");
            if !pinned.contains(*part) {
                failures.push(format!("{id}: {part} differs: {how}"));
            }
        }
        for part in &pinned {
            if !found.contains_key(part.as_str()) {
                failures.push(format!(
                    "{id}: {part} is pinned as differing and now agrees; remove it from \
                     known-differing.json"
                ));
            }
        }
    }
    println!(
        "PQL corpus: {} of {} queries agree with the reference; {differing} differ, {} pinned",
        cases.len() - differing,
        cases.len(),
        known.len()
    );
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
