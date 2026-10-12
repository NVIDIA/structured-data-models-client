// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The validator's two options no profile reaches, `allow_array_targets`
//! and `allow_timestamp_arrays` (`tests/validator-options.json`): each row
//! the reference's `PredictiveQueryValidator` answered with the options
//! set (`sdm-core/tests/pql/record_quirks.py`), held here to the same
//! validity and the same error codes and spans, in order.

#![allow(
    clippy::disallowed_methods,
    reason = "test code: reads the option table"
)]

use std::path::Path;

use sdm_pql::dates::Reader;
use sdm_pql::schema::{Column, ColumnKey, ColumnKeyGroup, SchemaDefinition, TableDefinition};
use sdm_pql::types::{Dtype, Stype};
use sdm_pql::validator::predictive_query::PredictiveQueryValidator;
use sdm_pql::{Filters, Mode, parse};
use serde_json::{Value, json};

fn table() -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/validator-options.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn name(value: &Value) -> Option<String> {
    value.as_str().map(str::to_owned)
}

/// A schema as the reference writes it.
fn schema(value: &Value) -> SchemaDefinition {
    SchemaDefinition {
        tables: value["tables"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(table, definition)| {
                let cols = definition["cols"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|c| Column {
                        name: name(&c["name"]).unwrap(),
                        stype: Stype::parse(c["stype"].as_str().unwrap()).unwrap(),
                        dtype: Dtype::parse(c["dtype"].as_str().unwrap()).unwrap(),
                    })
                    .collect();
                let definition = TableDefinition {
                    cols,
                    pkey: name(&definition["pkey"]),
                    time_col: name(&definition["time_col"]),
                    end_time_col: name(&definition["end_time_col"]),
                };
                (table.clone(), definition)
            })
            .collect(),
        col_groups: value["col_groups"]
            .as_array()
            .unwrap()
            .iter()
            .map(|group| ColumnKeyGroup {
                columns: group["columns"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|c| {
                        ColumnKey::new(
                            c["table_name"].as_str().unwrap(),
                            c["col_name"].as_str().unwrap(),
                        )
                    })
                    .collect(),
            })
            .collect(),
    }
}

#[test]
fn every_validator_option_row_answers_as_the_reference_does() {
    let table = table();
    let schema = schema(&table["schema"]);
    let rows = table["options"].as_array().unwrap();
    assert_eq!(rows.len(), 56);
    let reader = Reader {
        year: 2026,
        pads_year: false,
    };
    let mut failures = Vec::new();
    let mut accepted = 0;
    for row in rows {
        let mode = match row["mode"].as_str().unwrap() {
            "INFERENCE" => Mode::Inference,
            "TRAINING" => Mode::Training,
            other => panic!("a mode {other}"),
        };
        let parsed = parse(row["query"].as_str().unwrap(), mode, reader)
            .outcome
            .expect("every option row parses");
        let validator = PredictiveQueryValidator {
            allow_array_targets: row["allow_array_targets"].as_bool().unwrap(),
            allow_timestamp_arrays: row["allow_timestamp_arrays"].as_bool().unwrap(),
            ..PredictiveQueryValidator::new(&schema, mode, Filters::default())
        };
        let (resolved, response) = validator
            .validate_predictive_query(parsed.query)
            .expect("every option row is answered");
        let ok = resolved.is_some() && response.ok();
        accepted += usize::from(ok);
        let errors: Vec<Value> = response
            .errors
            .iter()
            .map(|e| {
                let span = e.span.map(|s| {
                    json!({"start_row": s.start_row, "start_col": s.start_col,
                           "end_row": s.end_row, "end_col": s.end_col})
                });
                json!({"code": e.code, "span": span})
            })
            .collect();
        if ok != row["accepted"].as_bool().unwrap() || json!(errors) != row["errors"] {
            failures.push(format!("{}: accepted {ok}, errors {errors:?}", row["id"]));
        }
    }
    assert!(accepted > 0 && accepted < rows.len(), "{accepted} accepted");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
