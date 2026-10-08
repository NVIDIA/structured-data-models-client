// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Capabilities and validator options that none of the five named profiles
//! has: each row of `custom_profiles.json`, recorded from today's Python by
//! `custom_profiles.py`, answers here as it does there.
//!
//! These reach codes the corpus cannot, since a corpus case names its
//! profile (`UNREACHABLE_CODES.md`).

mod accept_common;

use std::collections::BTreeSet;

use accept_common::{mismatch, read_json, report};
use sdm_pql::dates::Host;
use sdm_pql::validator::predictive_query::PredictiveQueryValidator;
use sdm_pql::visitor::parse_query;
use sdm_pql::{Capabilities, Mode, Profile, SchemaDefinition, Validation, context, validate_in};
use serde_json::Value;

/// No row reads a date, so the host only has to be fixed.
const HOST: Host = Host {
    year: 2026,
    pads_year: false,
};

fn table() -> Value {
    read_json(&std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/custom_profiles.json"))
}

fn rows<'a>(table: &'a Value, key: &str) -> &'a Vec<Value> {
    table[key]
        .as_array()
        .unwrap_or_else(|| panic!("the table has no {key}"))
}

fn text<'a>(row: &'a Value, key: &str) -> &'a str {
    row[key]
        .as_str()
        .unwrap_or_else(|| panic!("a row without {key}: {row}"))
}

fn mode(row: &Value) -> Mode {
    Mode::parse(text(row, "mode")).expect("a mode")
}

/// `Capabilities(model=...)` or `Capabilities.nothing(...)`, then `.also`.
fn capabilities(table: &Value, id: &str) -> Capabilities {
    let model = text(table, "model");
    let variant = &table["capabilities"][id];
    let base = match text(variant, "base") {
        "all" => Capabilities {
            model: model.to_string(),
            ..Capabilities::default()
        },
        "nothing" => Capabilities::nothing(model),
        other => panic!("unknown base {other}"),
    };
    let set: Vec<(&str, bool)> = variant["set"]
        .as_object()
        .expect("what a variant sets")
        .iter()
        .map(|(name, on)| (name.as_str(), on.as_bool().expect("a bool")))
        .collect();
    base.also(&set).expect("known capabilities")
}

fn answer(validation: &Validation) -> Value {
    serde_json::from_str(&validation.to_json().render()).expect("JSON")
}

#[test]
fn every_custom_profile_row() {
    let table = table();
    let schema = SchemaDefinition::from_value(&table["schema"]).expect("a schema");
    let profiles = rows(&table, "profiles");
    let mut failures = Vec::new();
    for row in profiles {
        let profile = Profile {
            mode: mode(row),
            capabilities: capabilities(&table, text(row, "capabilities")),
        };
        let got = answer(&validate_in(text(row, "query"), &schema, &profile, HOST));
        if let Some(problem) = mismatch(&row["python"], &got) {
            failures.push(format!("{}: {problem}", text(row, "id")));
        }
    }
    report("custom profiles", profiles.len(), &failures);
}

#[test]
fn every_validator_option_row() {
    let table = table();
    let schema = SchemaDefinition::from_value(&table["schema"]).expect("a schema");
    let options = rows(&table, "options");
    let mut failures = Vec::new();
    for row in options {
        let validator = PredictiveQueryValidator {
            allow_array_targets: row["allow_array_targets"].as_bool().expect("a bool"),
            allow_timestamp_arrays: row["allow_timestamp_arrays"].as_bool().expect("a bool"),
            ..PredictiveQueryValidator::new(&schema, mode(row), Capabilities::default())
        };
        let query = text(row, "query");
        let ((resolved, response), warnings) = context::run(HOST, || {
            let parsed = parse_query(query, validator.mode).expect("every option row parses");
            validator
                .validate_predictive_query(parsed)
                .expect("every option row is answered")
        });
        let validation = Validation {
            query: resolved.filter(|_| response.ok()),
            response,
            warnings,
        };
        if let Some(problem) = mismatch(&row["python"], &answer(&validation)) {
            failures.push(format!("{}: {problem}", text(row, "id")));
        }
    }
    report("validator options", options.len(), &failures);
}

/// What `UNREACHABLE_CODES.md` says this table covers.
#[test]
fn the_table_reaches_what_no_named_profile_can() {
    let table = table();
    let given: BTreeSet<&str> = rows(&table, "profiles")
        .iter()
        .chain(rows(&table, "options"))
        .filter_map(|row| row["python"]["errors"].as_array())
        .flatten()
        .map(|error| text(error, "code"))
        .collect();
    for code in [
        "PQL_MODEL_NO_COUNTERFACTUAL",
        "PQL_MODEL_NO_FORECASTING",
        "PQL_MODEL_NO_KEY_COLUMN_FILTER",
        "PQL_TYPE_AGGREGATION_ON_TIMESTAMP_UNSUPPORTED",
        "PQL_TYPE_TARGET_DTYPE_NOT_PREDICTABLE",
    ] {
        assert!(given.contains(code), "no row gives {code}");
    }
}
