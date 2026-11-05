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
//! preserved in the frozen fixture). A refused query gives the same errors
//! in count, code and span, unknown-name errors as a multiset, since the
//! reference orders them from a Python set; messages are not promised. The
//! date warnings the reference gave are compared by format (D4).
//!
#![allow(
    clippy::disallowed_methods,
    reason = "test code: reads the corpus file"
)]

mod pql_compare;

use std::collections::BTreeMap;
use std::path::Path;

use pql_compare::{differences, in_comparable_order, profile, query_schema};
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
fn the_corpus_matches_frozen_expected_answers() {
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
    let mut failures = Vec::new();
    for case in cases {
        let id = case["id"].as_str().unwrap();
        let schema = &schemas[case["schema"].as_str().unwrap()];
        let found = differences(case, schema, &profile(case["profile"].as_str().unwrap()));
        for (part, how) in found {
            failures.push(format!("{id}: {part} differs: {how}"));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
