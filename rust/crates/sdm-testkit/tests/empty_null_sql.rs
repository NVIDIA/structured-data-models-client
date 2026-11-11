// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Execute type-family cases through the real DuckDB provider.

use std::path::PathBuf;

use sdm_testkit::bases;
use sdm_testkit::case::{self, Case};
use sdm_testkit::host::{self, Observed};

const FAMILY: [&str; 3] = ["empty.", "null.", "refuse."];

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

fn family_cases() -> (Vec<Case>, bases::Bases) {
    let contract = root().join("contract");
    let loaded = case::load_for(&[contract.join("cases")], case::Database::Duckdb).unwrap();
    let mut cases: Vec<Case> = loaded
        .into_iter()
        .map(|l| l.case)
        .filter(|c| FAMILY.iter().any(|p| c.id.starts_with(p)))
        .collect();
    cases.sort_by(|a, b| a.id.cmp(&b.id));
    assert_eq!(
        cases.iter().map(|c| c.id.as_str()).collect::<Vec<_>>(),
        [
            "empty.entity-without-facts",
            "empty.fact-table",
            "empty.indices",
            "null.feature-all-null",
            "null.some-foreign-keys",
            "null.some-times",
            "null.target-all-null",
            "refuse.unknown-column",
        ],
        "the family's eight cases"
    );
    (cases, bases::load(&contract.join("bases.yaml")).unwrap())
}

/// Each case through this host, through `predict` against the scripted
/// endpoint, in process.
fn observed() -> Vec<(Case, Observed)> {
    let (cases, bases) = family_cases();
    cases
        .into_iter()
        .map(|case| {
            let seen = host::run(&case, &bases).unwrap_or_else(|e| panic!("{}: {e}", case.id));
            (case, seen)
        })
        .collect()
}

#[test]
fn family_cases_execute_with_expected_outcomes_and_encodable_results() {
    for (case, seen) in observed() {
        let (outcome, code) = seen.answer.outcome();
        assert_eq!(outcome, case.expect.outcome, "{}", case.id);
        // Historical Python RuntimeError had no code; the engine classifies
        // empty-source refusals as invalid requests.
        let expected_code = if case.id == "empty.fact-table" {
            Some("INVALID_REQUEST".to_owned())
        } else {
            case.expect.code.clone()
        };
        assert_eq!(code, expected_code, "{}", case.id);
        let driver = seen.driver.as_ref().expect("real DuckDB execution");
        let statements = driver["capture"]["statements"].as_array().unwrap();
        assert!(!statements.is_empty(), "{}: no database reads", case.id);
        assert_eq!(
            driver["query_log"]["differing"],
            serde_json::json!([]),
            "{}",
            case.id
        );
        for statement in statements {
            for result in statement["results"].as_array().unwrap() {
                assert_eq!(
                    result["values"]["unencodable"],
                    serde_json::json!([]),
                    "{}",
                    case.id
                );
                assert!(
                    result["schema"].is_array(),
                    "{}: missing Arrow schema",
                    case.id
                );
            }
        }
    }
}
