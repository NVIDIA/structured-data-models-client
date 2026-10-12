// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Capabilities none of the five named profiles has
//! (`tests/pql/custom-profiles.json`): each capability turned off on its
//! own, none, and only `temporal` and `static`, in either mode, over probe
//! queries, each answered by the reference (`record_quirks.py`) and held
//! through the adapter as a corpus case is (`pql_compare`). These reach
//! capability codes the corpus cannot, since a corpus case names its
//! profile.

#![allow(
    clippy::disallowed_methods,
    reason = "test code: reads the custom-profile table"
)]

mod pql_compare;

use std::path::Path;

use pql_compare::{differences, query_schema};
use sdm_core::profiles::{Capabilities, Mode, Profile};
use serde_json::Value;

fn table() -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/pql/custom-profiles.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

/// A variant's capabilities: every one, or none, then those it sets.
fn capabilities(table: &Value, id: &str) -> Capabilities {
    let variant = &table["capabilities"][id];
    let base = match variant["base"].as_str().unwrap() {
        "all" => true,
        "nothing" => false,
        other => panic!("a base {other}"),
    };
    let set = |name: &str| variant["set"][name].as_bool().unwrap_or(base);
    for name in variant["set"].as_object().unwrap().keys() {
        assert!(
            [
                "relationships",
                "multiple_tables",
                "temporal",
                "static",
                "entity_filters",
                "counterfactuals",
                "derived_targets",
                "link_prediction",
                "forecasting",
                "key_columns_in_filters",
                "timestamp_columns_in_filters",
            ]
            .contains(&name.as_str()),
            "{name}"
        );
    }
    Capabilities {
        model: table["model"].as_str().unwrap().to_owned(),
        relationships: set("relationships"),
        multiple_tables: set("multiple_tables"),
        temporal: set("temporal"),
        static_: set("static"),
        entity_filters: set("entity_filters"),
        counterfactuals: set("counterfactuals"),
        derived_targets: set("derived_targets"),
        link_prediction: set("link_prediction"),
        forecasting: set("forecasting"),
        key_columns_in_filters: set("key_columns_in_filters"),
        timestamp_columns_in_filters: set("timestamp_columns_in_filters"),
    }
}

#[test]
fn every_custom_profile_row_answers_as_the_reference_does() {
    let table = table();
    let schema = query_schema(&table["schema"]);
    let rows = table["profiles"].as_array().unwrap();
    assert_eq!(rows.len(), 224);
    let mut failures = Vec::new();
    for row in rows {
        let profile = Profile {
            mode: match row["mode"].as_str().unwrap() {
                "INFERENCE" => Mode::Inference,
                "TRAINING" => Mode::Training,
                other => panic!("a mode {other}"),
            },
            capabilities: capabilities(&table, row["capabilities"].as_str().unwrap()),
        };
        for (part, how) in differences(row, &schema, &profile) {
            failures.push(format!("{}: {part}: {how}", row["id"]));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
