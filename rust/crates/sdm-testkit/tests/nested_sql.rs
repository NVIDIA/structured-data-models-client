// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Execute type-family cases through the real DuckDB provider.

use std::path::PathBuf;

use sdm_testkit::bases;
use sdm_testkit::case::{self, Case};
use sdm_testkit::host::{self, Observed};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

fn nested_cases() -> (Vec<Case>, bases::Bases) {
    let contract = root().join("contract");
    let loaded = case::load_for(&[contract.join("cases")], case::Database::Duckdb).unwrap();
    let mut cases: Vec<Case> = loaded
        .into_iter()
        .map(|l| l.case)
        .filter(|c| c.id.starts_with("nested."))
        .collect();
    cases.sort_by(|a, b| a.id.cmp(&b.id));
    assert_eq!(cases.len(), 5, "the nested family's five cases");
    (cases, bases::load(&contract.join("bases.yaml")).unwrap())
}

/// Each case through this host, through `predict` against the scripted
/// endpoint, in process.
fn observed() -> Vec<(Case, Observed)> {
    let (cases, bases) = nested_cases();
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
        // Primitive arrays are recognized as lists; unsupported nested
        // structures remain typed refusals (schema/time_choice tests).
        let accepted = matches!(case.id.as_str(), "nested.list-int" | "nested.fixed-array");
        assert_eq!(
            outcome,
            if accepted { "success" } else { "refusal" },
            "{}",
            case.id
        );
        assert_eq!(
            code.as_deref(),
            if accepted {
                None
            } else if case.id == "nested.union" {
                // Arrow admission rejects union arrays before schema inference.
                Some("UNSUPPORTED_TYPE")
            } else {
                Some("UNSUPPORTED_REQUEST")
            },
            "{}",
            case.id
        );
        if accepted {
            let tables = seen.described.as_ref().unwrap()["tables"]
                .as_array()
                .unwrap();
            let users = tables.iter().find(|t| t["name"] == "users").unwrap();
            let columns: Vec<_> = users["columns"]
                .as_array()
                .unwrap()
                .iter()
                .map(|column| column["name"].as_str().unwrap())
                .collect();
            assert_eq!(
                columns,
                ["user_id", "age", "country"],
                "{}: lists omitted",
                case.id
            );
            let payload = &seen.materialized.as_ref().unwrap()["payload"];
            let columns = payload["context"]["related_tables"]["users"]["columns"]
                .as_array()
                .unwrap();
            assert!(!columns.iter().any(|column| column == "f"), "{}", case.id);
        }
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

/// A list column is read by the preview, which describes it, and left out
/// of the entity lookup, which sends no list.
#[test]
fn a_list_column_is_previewed_and_left_out_of_the_lookup() {
    let mut checked = 0;
    for (case, seen) in observed() {
        if !matches!(case.id.as_str(), "nested.list-int" | "nested.fixed-array") {
            continue;
        }
        assert!(
            ran(&seen, "from_source").contains(
                &r#"SELECT "user_id", "age", "country", "f" FROM "users" LIMIT 1000"#.to_owned()
            ),
            "{}: the preview",
            case.id
        );
        assert!(
            ran(&seen, "materialize").iter().any(|sql| sql.contains(
                "t.\"user_id\", t.\"age\", t.\"country\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t"
            )),
            "{}: the lookup",
            case.id
        );
        checked += 1;
    }
    assert_eq!(checked, 2);
}
