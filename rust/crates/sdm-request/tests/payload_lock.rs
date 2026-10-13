// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The request state's payload against the contract's locked definition
//! of it (`$defs/StatePayload`, T1.22; from T1.28w the schema
//! `sdm_wire::payload::StatePayload` derives): the payload the producer
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
use sdm_engine::labels::Labels;
use sdm_request::options::{Kind, Options, RunMode};
use sdm_request::payload::{Sampled, SampledColumn, SampledLink, SampledTable, build};
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
        kind: Kind::Regression,
        labels: Labels::Float(vec![1.5, 0.0, 42.25]),
        windowed: true,
        run_mode: RunMode::Fast,
        options: Options::default(),
        statistic: Some("mean".into()),
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
        kind: Kind::Regression,
        labels: Labels::Float(vec![f32::NAN, 3.0]),
        windowed: false,
        run_mode: RunMode::Best,
        options: Options::default(),
        statistic: None,
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
    serde_json::from_slice(&build(sampled).expect("the producer builds it").bytes).unwrap()
}

/// The shop sample asked as a classification: `kind` over `labels`.
fn classified(kind: Kind, labels: Labels) -> Sampled {
    Sampled {
        kind,
        labels,
        windowed: false,
        statistic: None,
        ..shop()
    }
}

#[test]
fn the_payload_the_producer_builds_is_the_locked_definition_s() {
    for (name, sampled) in [
        ("shop", shop()),
        ("family", family()),
        (
            "binary",
            classified(Kind::Binary, Labels::Bool(vec![true, false, true])),
        ),
        (
            "multiclass",
            classified(
                Kind::Multiclass,
                Labels::Text(vec!["b".into(), "a".into(), "b".into()]),
            ),
        ),
    ] {
        let payload = built(&sampled);
        assert_eq!(check("StatePayload", &payload), Ok(()), "{name}");
    }
    // A classification's target names its classes, a binary task's its
    // positive class, and its options their class shuffle.
    let binary = built(&classified(
        Kind::Binary,
        Labels::Bool(vec![true, false, true]),
    ));
    assert_eq!(
        binary["task"]["target"]["classes"],
        json!(["false", "true"])
    );
    assert_eq!(binary["task"]["target"]["positive_class"], json!("true"));
    assert_eq!(
        binary["inference"]["inference_config"]["class_shuffle"],
        json!(false)
    );
    let multiclass = built(&classified(
        Kind::Multiclass,
        Labels::Bool(vec![true, false, true]),
    ));
    assert_eq!(
        multiclass["task"]["target"]["classes"],
        json!(["True", "False"])
    );
    assert!(multiclass["task"]["target"].get("positive_class").is_none());
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
/// locked definition refuses it, naming `expected` (inside the task or the
/// inference options, which take one shape per kind of task, the place
/// none of the kinds' shapes holds).
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
    refused(|p| p["task"]["horizon"] = json!(3), "$.task: none of");
    refused(|p| p["metadata"]["num_batches"] = json!(1), "num_batches");
    refused(
        |p| p["inference"]["inference_config"]["seed"] = json!(42),
        "$.inference.inference_config: none of",
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
    refused(|p| p["task"]["kind"] = json!(1), "$.task: none of");
    // A kind the task's shape is not: a regression named a classification.
    refused(
        |p| p["task"]["kind"] = json!("binary_classification"),
        "$.task: none of",
    );
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
    // A dtype or stype the wire does not name (the typed payload's, T1.28w).
    refused(
        |p| p["schema"]["related_tables"]["orders"]["columns"]["price"]["dtype"] = json!("int32"),
        "dtype",
    );
    refused(
        |p| p["task"]["target"]["dtype"] = json!("decimal"),
        "$.task: none of",
    );
    // Removed.
    refused(
        |p| {
            p["task"].as_object_mut().unwrap().remove("target");
        },
        "$.task: none of",
    );
    refused(
        |p| {
            p.as_object_mut().unwrap().remove("metadata");
        },
        "metadata is missing",
    );
    // A regression's options losing both members a classification's lack
    // (fix round 1, Codex P2-3): the options are told apart by their kind,
    // so neither shape takes them.
    refused(
        |p| {
            let config = p["inference"]["inference_config"].as_object_mut().unwrap();
            config.remove("target_transforms");
            config.remove("output_type");
        },
        "$.inference.inference_config: none of",
    );
    // A classification's options named a regression's.
    let binary = built(&classified(
        Kind::Binary,
        Labels::Bool(vec![true, false, true]),
    ));
    assert_eq!(check("StatePayload", &binary), Ok(()));
    let mut renamed = binary.clone();
    renamed["inference"]["inference_config"]["kind"] = json!("regression");
    assert!(check("StatePayload", &renamed).is_err());
    // A class shuffle left out is the reader's to take.
    let mut unshuffled = binary.clone();
    unshuffled["inference"]["inference_config"]
        .as_object_mut()
        .unwrap()
        .remove("class_shuffle");
    assert_eq!(check("StatePayload", &unshuffled), Ok(()));
}

/// The definitions the locked schema gives at `path` (`.`-separated; `*`
/// is any key of an open map, `[]` an array's items): one, or one per
/// option where a definition is one of several (`anyOf`, the task's and
/// the inference options' shapes per kind), each option naming the step.
fn defined(path: &str) -> Vec<Value> {
    let root: Value = serde_json::from_str(&sdm_contract::schema_json()).unwrap();
    let mut nodes = vec![root["$defs"]["StatePayload"].clone()];
    for step in path.split('.') {
        nodes = nodes
            .iter()
            .flat_map(|node| match node["anyOf"].as_array() {
                Some(options) => options.clone(),
                None => vec![node.clone()],
            })
            .map(|node| match step {
                "[]" => node["items"].clone(),
                "*" => node["additionalProperties"].clone(),
                key => node["properties"][key].clone(),
            })
            .collect();
        for node in &nodes {
            assert!(node.is_object(), "{path}: a definition names no {step}");
        }
    }
    nodes
}

/// The type each definition at `path` gives, its options' for one of
/// several.
fn types(path: &str) -> Vec<Value> {
    defined(path)
        .into_iter()
        .flat_map(|node| match node["anyOf"].as_array() {
            Some(options) => options.iter().map(|o| o["type"].clone()).collect(),
            None => vec![node["type"].clone()],
        })
        .collect()
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
        let given = types(path);
        assert!(!given.is_empty(), "{path}");
        assert!(given.iter().all(|t| *t == json!(kind)), "{path}: {given:?}");
    }
    // Each option the client forwards is named by the options' shape of
    // a classification, and all but the class shuffle by a regression's.
    for option in INFERENCE_OPTIONS {
        let root: Value = serde_json::from_str(&sdm_contract::schema_json()).unwrap();
        let config = &root["$defs"]["StatePayload"]["properties"]["inference"]["properties"]["inference_config"];
        let named: Vec<bool> = config["anyOf"]
            .as_array()
            .unwrap()
            .iter()
            .map(|o| o["properties"].get(option).is_some())
            .collect();
        assert!(named.iter().any(|n| *n), "{option}");
    }
}
