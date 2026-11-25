// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Execute type-family cases through the real DuckDB provider.

use std::path::PathBuf;

use sdm_testkit::bases;
use sdm_testkit::case::{self, Case};
use sdm_testkit::host::{self, Observed};

const FAMILY: [&str; 1] = ["decimal."];

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
    // three cases, ten (exact decimals) and
    // scaled decimal read as the reference counts it.
    assert_eq!(
        cases.iter().map(|c| c.id.as_str()).collect::<Vec<_>>(),
        [
            "decimal.class-target-38-18",
            "decimal.compare-finer-eq",
            "decimal.compare-finer-gt",
            "decimal.compare-finer-negative",
            "decimal.compare-past-precision",
            "decimal.compare-past-precision-gt",
            "decimal.compare-same-scale",
            "decimal.feature-38-0-wide",
            "decimal.feature-inexact-38-18",
            "decimal.key-wide",
            "decimal.stype-past-float-precision",
            "decimal.sum-10-2",
            "decimal.target-38-18",
            "decimal.target-regression-collide",
        ],
        "the family's fourteen cases"
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
        assert_eq!(code, case.expect.code, "{}", case.id);
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

/// The texts `seen`'s `stage` ran.
fn ran(seen: &Observed, stage: &str) -> Vec<String> {
    let driver = seen.driver.as_ref().expect("a driver record");
    driver["capture"]["statements"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|s| s["stage"] == stage)
        .map(|s| s["sql"].as_str().unwrap().to_owned())
        .collect()
}

/// A summed decimal is the label read's one value column; a wide decimal
/// feature is read by the entity lookup; a key past `BIGINT` joins each
/// keyed read through the seed cast to the key's type.
#[test]
fn decimal_reads_project_their_columns_and_cast_wide_keys() {
    let seen: std::collections::BTreeMap<String, Observed> = observed()
        .into_iter()
        .map(|(case, seen)| (case.id, seen))
        .collect();
    let summed = ran(&seen["decimal.sum-10-2"], "materialize");
    assert!(
        summed.iter().any(|sql| sql.starts_with(
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"ts\", t.\"amount\"\nFROM \"__sdm_seeds__\" s\n"
        )),
        "{summed:#?}"
    );
    let wide = ran(&seen["decimal.feature-38-0-wide"], "materialize");
    assert!(
        wide.iter().any(|sql| sql.contains(
            "t.\"user_id\", t.\"age\", t.\"country\", t.\"f\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t"
        )),
        "{wide:#?}"
    );
    let keyed = ran(&seen["decimal.key-wide"], "materialize");
    let cast = r#"ON t."user_id" = TRY_CAST(s."key" AS DECIMAL(38,0))"#;
    assert_eq!(
        keyed.iter().filter(|sql| sql.contains(cast)).count(),
        3,
        "{keyed:#?}"
    );
}
