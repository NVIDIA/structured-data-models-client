// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! T1.28w exit 1, held from outside the implementation (the package's
//! independent test author): the `StatePayload` lock has one source, the
//! typed `sdm-wire` structs.
//!
//! - The committed `state-payload.json` is, byte for byte, what the types
//!   generate.
//! - The definition the contract lock covers (`$defs/StatePayload` of the
//!   schema file `contract.lock` names for this version) is that same
//!   schema, with the state's version alone added; so is the one
//!   `sdm_contract::schema_json` generates. Neither holds a hand-kept copy.
//! - The schema names every member the types write: a payload with every
//!   optional member present serializes to keys each of which the schema's
//!   node at that path names, and each required key is one the types
//!   always write.

#![allow(
    clippy::disallowed_methods,
    reason = "test code: reads the committed schema files and the lock"
)]

use std::path::PathBuf;

use sdm_contract::ops::Stype;
use sdm_wire::json::Value as Cell;
use sdm_wire::payload::{
    ColumnSpec, Dtype, Format, Half, Inference, InferenceConfig, Metadata, Named, Output,
    PayloadSchema, PrimaryKey, Relationship, Rows, StatePayload, TableSchema, Target, Task,
    schema_json,
};
use serde_json::Value;

fn contract_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../sdm-contract/schema")
}

fn without_version(mut definition: Value) -> Value {
    definition
        .as_object_mut()
        .expect("the definition is an object")
        .remove("x-request-state-version")
        .expect("the definition records the state's version");
    definition
}

#[test]
fn the_committed_payload_schema_is_what_the_types_generate() {
    let committed = std::fs::read_to_string(contract_dir().join("state-payload.json")).unwrap();
    assert_eq!(committed, schema_json());
    assert_eq!(sdm_contract::ops::STATE_PAYLOAD_SCHEMA, schema_json());
}

#[test]
fn the_locked_definition_is_the_types_schema() {
    let types: Value = serde_json::from_str(&schema_json()).unwrap();
    // The schema file the lock names for this build's version.
    let lock = std::fs::read_to_string(contract_dir().join("contract.lock")).unwrap();
    let version = sdm_contract::CONTRACT_VERSION.to_string();
    let file = lock
        .lines()
        .filter(|l| !l.starts_with('#'))
        .find_map(|l| {
            let mut parts = l.split_whitespace();
            (parts.next() == Some(version.as_str()))
                .then(|| parts.nth(1).expect("a lock line names its file").to_owned())
        })
        .expect("the lock has this version's line");
    let locked: Value =
        serde_json::from_str(&std::fs::read_to_string(contract_dir().join(file)).unwrap()).unwrap();
    assert_eq!(
        without_version(locked["$defs"]["StatePayload"].clone()),
        types,
        "the locked $defs/StatePayload is not the typed payload's schema"
    );
    let generated: Value = serde_json::from_str(&sdm_contract::schema_json()).unwrap();
    assert_eq!(
        without_version(generated["$defs"]["StatePayload"].clone()),
        types
    );
}

fn text(s: &str) -> String {
    s.to_owned()
}

/// A payload with every member the types have, the optional ones present.
fn full() -> StatePayload {
    let column = ColumnSpec {
        dtype: Dtype::Int64,
        stype: Some(Stype::Numerical),
        nullable: Some(false),
    };
    let table = TableSchema {
        columns: Named(vec![(text("id"), column.clone())]),
        primary_key: PrimaryKey::Several(vec![text("id")]),
    };
    let rows = Rows {
        format: Format::Arrays,
        columns: vec![text("id")],
        rows: vec![vec![Cell::Number(text("1")), Cell::Null]],
    };
    let half = Half {
        instance_table: rows.clone(),
        related_tables: Named(vec![(text("t"), rows)]),
    };
    StatePayload {
        model: text("m"),
        task: Task {
            kind: text("regression"),
            target: Target {
                column_name: text("y"),
                dtype: Dtype::Float64,
            },
            entity_table_names: vec![text("e")],
            anchor_time_column: Some(text("at")),
            top_k: Some(3),
        },
        schema: PayloadSchema {
            instance_table: table.clone(),
            related_tables: Named(vec![(text("t"), table)]),
            relationships: vec![Relationship {
                source_table: Some(text("e")),
                source_columns: vec![text("id")],
                target_table: text("t"),
                target_columns: vec![text("id")],
            }],
        },
        context: half.clone(),
        predict: half,
        output: Output {
            fields: vec![text("prediction")],
            prediction_statistic: Some(text("median")),
        },
        inference: Inference {
            run_mode: text("fast"),
            use_prediction_time: false,
            inference_config: InferenceConfig {
                kind: text("regression"),
                num_estimators: 1,
                column_shuffle: false,
                category_shuffle: false,
                hop_shuffle: false,
                class_shuffle: Some(false),
                target_transforms: vec![],
                output_type: text("point"),
            },
        },
        metadata: Metadata {
            source_query: Some(text("PREDICT x")),
            num_context_examples: 1,
            num_prediction_examples: 1,
        },
    }
}

/// Walks `value` with the schema node `schema`, every object of a fixed
/// shape (one with `properties`) required to name each key written, and
/// each key it requires to be written.
fn named(schema: &Value, value: &Value, at: &str, seen: &mut usize) {
    match value {
        Value::Object(members) => {
            if let Some(properties) = schema.get("properties").and_then(Value::as_object) {
                for key in members.keys() {
                    assert!(properties.contains_key(key), "{at}.{key} is not named");
                }
                for key in schema["required"].as_array().into_iter().flatten() {
                    let key = key.as_str().unwrap();
                    assert!(
                        members.contains_key(key),
                        "{at}.{key} is required, not written"
                    );
                }
                for (key, item) in members {
                    *seen += 1;
                    named(&properties[key], item, &format!("{at}.{key}"), seen);
                }
            } else if let Some(entry) = schema.get("additionalProperties") {
                for (key, item) in members {
                    named(entry, item, &format!("{at}.{key}"), seen);
                }
            }
        }
        Value::Array(items) => {
            let item_schema = schema.get("items");
            for (n, item) in items.iter().enumerate() {
                if let Some(s) = item_schema {
                    named(s, item, &format!("{at}[{n}]"), seen);
                }
            }
        }
        _ => {}
    }
}

#[test]
fn the_schema_names_every_member_the_types_write() {
    let schema: Value = serde_json::from_str(&schema_json()).unwrap();
    let written: Value = serde_json::from_slice(&full().to_bytes()).unwrap();
    let mut seen = 0;
    named(&schema, &written, "$", &mut seen);
    assert!(seen > 40, "{seen} members walked");
}

#[test]
fn the_payload_s_key_order_is_the_reference_s() {
    // The reference's engine writes the payload's sections in this order
    // (`_base_payload`), which the bytes sent keep.
    let written = String::from_utf8(full().to_bytes()).unwrap();
    let mut last = 0;
    for key in [
        "\"model\"",
        "\"task\"",
        "\"schema\"",
        "\"context\"",
        "\"predict\"",
        "\"output\"",
        "\"inference\"",
        "\"metadata\"",
    ] {
        let at = written
            .find(key)
            .unwrap_or_else(|| panic!("{key} not written"));
        assert!(at >= last, "{key} out of order in {written}");
        last = at;
    }
}
