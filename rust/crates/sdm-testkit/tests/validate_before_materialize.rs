// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Carried row G5 (T2.5's implementer's test): on DuckDB, rust-host runs a
//! case's `validate_query` after `describe_source` also when the case names
//! `materialize`, as the Python host validates a query before it
//! materializes, and materializes only once the query is accepted.

#![allow(
    clippy::disallowed_methods,
    reason = "test code: writes a derived case to a temporary file"
)]

use std::path::PathBuf;

use sdm_testkit::{bases, case, host};
use serde_json::Value;

fn contract() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../contract")
}

fn read(name: &str) -> Value {
    let path = contract().join("cases/s1").join(format!("{name}.json"));
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

/// `base.regression`, materialized and not predicted, with
/// `refuse.unknown-column`'s `validate_query` (the same base's schema), and
/// `pql` as its query.
fn operations(id: &str, pql: &str) -> Vec<String> {
    let mut derived = read("base.regression");
    derived["id"] = Value::from(id);
    derived["input"]["pql"] = Value::from(pql);
    derived["until"] = Value::from("materialize");
    let core = derived["core"].as_object_mut().unwrap();
    core.remove("predict");
    core.insert(
        "validate_query".into(),
        read("refuse.unknown-column")["core"]["validate_query"].clone(),
    );
    let directory = std::env::temp_dir().join(format!("g5-{}-{id}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let file = directory.join(format!("{id}.json"));
    std::fs::write(&file, serde_json::to_vec(&derived).unwrap()).unwrap();
    let loaded = case::load(&[file]).unwrap();
    std::fs::remove_dir_all(&directory).ok();
    let bases = bases::load(&contract().join("bases.yaml")).unwrap();
    host::run(&loaded[0].case, &bases).unwrap().operations
}

#[test]
fn an_accepted_query_is_validated_then_materialized() {
    assert_eq!(
        operations(
            "g5.accepted",
            "PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id"
        ),
        ["describe_source", "validate_query", "materialize"]
    );
}

#[test]
fn a_refused_query_is_never_materialized() {
    assert_eq!(
        operations(
            "g5.refused",
            "PREDICT SUM(orders.cost, 0, 30, days) FOR EACH users.user_id"
        ),
        ["describe_source", "validate_query"]
    );
}
