// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The request state's payload against the contract's locked definition
//! of it (`$defs/StatePayload`, T1.22): the payload the producer
//! ([`sdm_request::payload::regression`]) builds holds, for a sample
//! shaped as `base.regression`'s and for one shaped as the family cases
//! that materialize (no anchor, a related table without a key, a text key,
//! no source query, nulls); a field added to or retyped in what it builds
//! fails, seeded on the real payload, so a producer change fails here
//! until the definition, and with it the version and the lock, moves; and
//! every path the reader (`sdm_request::predict`) and the Python host read
//! is one the definition names, with the type they read. `sdm-testkit`'s
//! `duckdb_provider.rs` holds the payload of every case `rust-host`
//! materializes to the same definition.

use std::sync::Arc;

use arrow_array::{
    ArrayRef, Float64Array, Int32Array, Int64Array, StringArray, TimestampMicrosecondArray,
    UInt64Array,
};
use sdm_contract::ops::Stype;
use sdm_contract::schema_check::check;
use sdm_request::payload::{Sampled, SampledColumn, SampledLink, SampledTable, regression};
use sdm_request::predict::INFERENCE_OPTIONS;
use serde_json::{Value, json};

const DAY_NS: i64 = 86_400_000_000_000;
/// `pd.Timestamp.min`: no anchor.
const NO_ANCHOR: i64 = -9_223_372_036_854_775_807;

fn array<A: arrow_array::Array + 'static>(a: A) -> SampledColumn {
    SampledColumn::Arrow(Arc::new(a) as ArrayRef)
}

/// A sample shaped as `base.regression`'s: users keyed by an integer, the
/// orders referring to them, anchored examples, a source query.
fn shop() -> Sampled {
    Sampled {
        num_train: 3,
        num_test: 2,
        anchors: (0..5)
            .map(|i| 1_709_251_200_000_000_000 - i * DAY_NS)
            .collect(),
        targets: vec![1.5, 0.0, 42.25],
        tables: vec![
            SampledTable {
                name: "users".into(),
                primary_key: Some("user_id".into()),
                batch: vec![0, 1, 2, 3, 4],
                columns: vec![
                    (
                        "user_id".into(),
                        array(Int64Array::from(vec![0, 1, 2, 0, 1])),
                    ),
                    (
                        "age".into(),
                        array(Int32Array::from(vec![20, 21, 22, 20, 21])),
                    ),
                    (
                        "country".into(),
                        array(StringArray::from(vec!["US", "DE", "IN", "US", "DE"])),
                    ),
                ],
                stypes: vec![
                    ("age".into(), Stype::Numerical),
                    ("country".into(), Stype::Categorical),
                ],
            },
            SampledTable {
                name: "orders".into(),
                primary_key: Some("order_id".into()),
                batch: vec![0, 0, 1, 3],
                columns: vec![
                    (
                        "order_id".into(),
                        array(Int64Array::from(vec![7, 8, 9, 10])),
                    ),
                    (
                        "price".into(),
                        array(Float64Array::from(vec![10.0, 11.5, 13.0, 14.5])),
                    ),
                    (
                        "ts".into(),
                        array(TimestampMicrosecondArray::from(vec![
                            1_704_067_200_000_000,
                            1_704_672_000_000_000,
                            1_705_276_800_000_000,
                            1_705_881_600_000_000,
                        ])),
                    ),
                ],
                stypes: vec![
                    ("price".into(), Stype::Numerical),
                    ("ts".into(), Stype::Timestamp),
                ],
            },
        ],
        links: vec![SampledLink {
            src: "orders".into(),
            fkey: "user_id".into(),
            dst: "users".into(),
            pairs: vec![(0, 0), (1, 0), (2, 1), (3, 3)],
        }],
        entity_table: "users".into(),
        source_query: "PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id".into(),
    }
}

/// A sample shaped as the family cases that materialize: no anchor, a text
/// key, a related table without a key of its own, nulls, a key past the
/// JSON-safe range, no source query.
fn family() -> Sampled {
    Sampled {
        num_train: 2,
        num_test: 1,
        anchors: vec![NO_ANCHOR; 3],
        targets: vec![f32::NAN, 3.0],
        tables: vec![
            SampledTable {
                name: "Users".into(),
                primary_key: Some("k".into()),
                batch: vec![0, 1, 2],
                columns: vec![
                    ("k".into(), array(StringArray::from(vec!["a", "b", "c"]))),
                    (
                        "big".into(),
                        array(UInt64Array::from(vec![Some(1 << 60), None, Some(1)])),
                    ),
                ],
                stypes: vec![("big".into(), Stype::Numerical)],
            },
            SampledTable {
                name: "events".into(),
                primary_key: None,
                batch: vec![1, 2],
                columns: vec![(
                    "note".into(),
                    array(StringArray::from(vec![None, Some("x")])),
                )],
                stypes: vec![("note".into(), Stype::Text)],
            },
        ],
        links: vec![SampledLink {
            src: "events".into(),
            fkey: "k".into(),
            dst: "Users".into(),
            pairs: vec![(0, 1), (1, 2)],
        }],
        entity_table: "Users".into(),
        source_query: String::new(),
    }
}

fn built(sampled: &Sampled) -> Value {
    serde_json::from_slice(&regression(sampled).expect("the producer builds it")).unwrap()
}

#[test]
fn the_payload_the_producer_builds_is_the_locked_definition_s() {
    for (name, sampled) in [("shop", shop()), ("family", family())] {
        let payload = built(&sampled);
        assert_eq!(check("StatePayload", &payload), Ok(()), "{name}");
    }
    // The two samples take the producer's optional branches both ways.
    let shop = built(&shop());
    let family = built(&family());
    assert!(shop["task"].get("anchor_time_column").is_some());
    assert!(family["task"].get("anchor_time_column").is_none());
    assert!(shop["metadata"].get("source_query").is_some());
    assert!(family["metadata"].get("source_query").is_none());
    assert!(
        family["schema"]["related_tables"]["events"]["columns"]
            .get("__node_id")
            .is_some()
    );
    assert!(
        shop["schema"]["relationships"][1]
            .get("source_table")
            .is_some()
    );
}

/// `edit` made to the payload the producer builds for the shop sample: the
/// locked definition refuses it, naming `expected`.
fn refused(edit: impl FnOnce(&mut Value), expected: &str) {
    let mut payload = built(&shop());
    let before = payload.clone();
    edit(&mut payload);
    assert_ne!(payload, before, "the seeded change changed nothing");
    let err = check("StatePayload", &payload).expect_err("a changed payload held");
    assert!(err.contains(expected), "{err}");
}

#[test]
fn a_field_added_to_or_retyped_in_the_payload_fails() {
    // Added: at the top, in the task, the metadata, the inference options,
    // a column's spec, a relationship and a half's table.
    refused(
        |p| p["review_extra_field"] = json!(true),
        "review_extra_field",
    );
    refused(|p| p["task"]["horizon"] = json!(3), "horizon");
    refused(|p| p["metadata"]["num_batches"] = json!(1), "num_batches");
    refused(
        |p| p["inference"]["inference_config"]["seed"] = json!(42),
        "seed",
    );
    refused(
        |p| p["schema"]["instance_table"]["columns"]["TARGET"]["unit"] = json!("s"),
        "unit",
    );
    refused(
        |p| p["schema"]["relationships"][0]["kind"] = json!("fk"),
        "kind",
    );
    refused(
        |p| p["predict"]["instance_table"]["index"] = json!([]),
        "index",
    );
    // Retyped.
    refused(
        |p| p["metadata"]["num_context_examples"] = json!("3"),
        "num_context_examples",
    );
    refused(|p| p["task"]["kind"] = json!(1), "kind");
    refused(
        |p| p["inference"]["use_prediction_time"] = json!("false"),
        "use_prediction_time",
    );
    refused(
        |p| p["context"]["instance_table"]["rows"] = json!({}),
        "rows",
    );
    refused(
        |p| p["schema"]["related_tables"]["orders"]["columns"]["price"]["dtype"] = json!(64),
        "dtype",
    );
    refused(
        |p| p["schema"]["related_tables"]["orders"]["columns"]["price"]["stype"] = json!("float"),
        "stype",
    );
    refused(
        |p| p["context"]["instance_table"]["format"] = json!("records"),
        "format",
    );
    // Removed.
    refused(
        |p| {
            p["task"].as_object_mut().unwrap().remove("target");
        },
        "target is missing",
    );
    refused(
        |p| {
            p.as_object_mut().unwrap().remove("metadata");
        },
        "metadata is missing",
    );
}

/// The definition the locked schema gives at `path` (`.`-separated; `*`
/// is any key of an open map, `[]` an array's items).
fn defined(path: &str) -> Value {
    let root: Value = serde_json::from_str(&sdm_contract::schema_json()).unwrap();
    let mut node = root["$defs"]["StatePayload"].clone();
    for step in path.split('.') {
        node = match step {
            "[]" => node["items"].clone(),
            "*" => node["additionalProperties"].clone(),
            key => node["properties"][key].clone(),
        };
        assert!(node.is_object(), "{path}: the definition names no {step}");
    }
    node
}

#[test]
fn what_the_readers_read_is_defined_with_the_type_they_read() {
    // sdm_request::predict: the task's kind, the output and its statistic,
    // the metadata, the inference options it keeps, the predict half's
    // instance table, and the key's dtype through the relationships.
    for (path, kind) in [
        ("task.kind", "string"),
        ("output", "object"),
        ("output.prediction_statistic", "string"),
        ("metadata", "object"),
        ("inference", "object"),
        ("inference.inference_config", "object"),
        ("predict.instance_table.columns", "array"),
        ("predict.instance_table.rows", "array"),
        ("schema.relationships", "array"),
        ("schema.relationships.[].source_columns", "array"),
        ("schema.relationships.[].target_table", "string"),
        ("schema.relationships.[].target_columns", "array"),
        ("schema.related_tables.*.columns.*.dtype", "string"),
        // The Python host's bridge reads the context's size.
        ("metadata.num_context_examples", "integer"),
    ] {
        assert_eq!(defined(path)["type"], json!(kind), "{path}");
    }
    for option in INFERENCE_OPTIONS {
        defined(&format!("inference.inference_config.{option}"));
    }
}
