// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Schema edits satisfy the same declared case outcomes on the Rust host.

use sdm_testkit::{bases, case, host};
use std::path::PathBuf;

fn contract() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../contract")
}

#[test]
fn schema_edits_have_their_declared_outcomes() {
    let bases = bases::load(&contract().join("bases.yaml")).unwrap();
    let mut paths: Vec<PathBuf> = std::fs::read_dir(contract().join("cases/s1"))
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("edit.") && name.ends_with(".json"))
        })
        .collect();
    paths.sort();
    let loaded = case::load(&paths).unwrap();
    assert_eq!(loaded.len(), 9);
    for item in loaded {
        let ran = host::run(&item.case, &bases).unwrap();
        let (outcome, code) = ran.answer.outcome();
        assert_eq!(outcome, item.case.expect.outcome, "{}", item.case.id);
        assert_eq!(code, item.case.expect.code, "{}", item.case.id);
        assert_eq!(
            ran.edits.len(),
            item.case.input.edits.as_ref().unwrap().len(),
            "{}",
            item.case.id
        );
        if outcome == "success" {
            let schema = &ran.inputs.last().unwrap()["schema"];
            let tables = schema["tables"].as_array().unwrap();
            let orders = tables
                .iter()
                .find(|table| table["name"] == "orders")
                .unwrap();
            let expected_time = if item.case.id == "edit.time-column-text" {
                "day"
            } else {
                "ts"
            };
            assert_eq!(orders["time_column"], expected_time);
            let rows =
                ran.materialized.as_ref().unwrap()["payload"]["context"]["instance_table"]["rows"]
                    .as_array()
                    .unwrap();
            assert!(!rows.is_empty(), "{}", item.case.id);
            if item.case.id == "edit.accepted-meaning-key-links" {
                let users = tables
                    .iter()
                    .find(|table| table["name"] == "users")
                    .unwrap();
                assert_eq!(users["primary_key"], serde_json::json!(["user_id"]));
                let age = users["columns"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|column| column["name"] == "age")
                    .unwrap();
                assert_eq!(age["stype"], "categorical");
                assert_eq!(
                    schema["links"],
                    serde_json::json!([[{"table": "orders", "column": "user_id"}, {"table": "users", "column": "user_id"}]])
                );
            }
        } else {
            assert!(ran.materialized.is_none(), "{}", item.case.id);
        }
    }
}
