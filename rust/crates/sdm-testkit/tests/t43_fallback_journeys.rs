// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Public describe/materialize/predict journeys through HostRun, with the
//! existing non-SQL provider declining optional operations for the bounded
//! core fallback rather than a test-only single-fetch operation.

use std::path::PathBuf;
use std::sync::Arc;

use arrow_array::{
    ArrayRef, Float64Array, Int64Array, RecordBatch, StringArray, TimestampMicrosecondArray,
};
use arrow_schema::{DataType, Field, Schema, TimeUnit};
use arrow_select::concat::concat_batches;
use sdm_contract::data::{FallbackLimits, OptionalRequest};
use sdm_testkit::case::{self, Case};
use sdm_testkit::host::{self, Answer, Observed};
use sdm_testkit::layer;
use sdm_testkit::memory::{Column, ForeignKey, Key, Memory, Table};
use serde_json::json;

const DAY: i64 = 86_400_000_000;

fn case() -> Case {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../contract/cases/s1/base.binary.json");
    case::load(&[path]).unwrap().remove(0).case
}

fn table(name: &str, fields: Vec<(&str, ArrayRef)>, key: &str) -> Table {
    let schema = Arc::new(Schema::new(
        fields
            .iter()
            .map(|(name, data)| Field::new(*name, data.data_type().clone(), true))
            .collect::<Vec<_>>(),
    ));
    let columns = fields
        .iter()
        .map(|(name, data)| Column {
            name: (*name).into(),
            declared_type: match data.data_type() {
                DataType::Int64 => "BIGINT",
                DataType::Float64 => "DOUBLE",
                DataType::Utf8 => "VARCHAR",
                DataType::Timestamp(TimeUnit::Microsecond, None) => "TIMESTAMP",
                other => panic!("unexpected fixture type {other:?}"),
            }
            .into(),
            not_null: true,
            primary_key: *name == key,
        })
        .collect();
    Table {
        name: name.into(),
        rows: RecordBatch::try_new(schema, fields.into_iter().map(|(_, a)| a).collect()).unwrap(),
        columns,
        keys: vec![Key {
            kind: "primary".into(),
            columns: vec![key.into()],
        }],
        foreign_keys: vec![],
    }
}

fn source() -> Memory {
    let users = table(
        "users",
        vec![("user_id", Arc::new(Int64Array::from_iter_values(0..8)))],
        "user_id",
    );
    let mut orders = table(
        "orders",
        vec![
            ("order_id", Arc::new(Int64Array::from_iter_values(0..80))),
            (
                "user_id",
                Arc::new(Int64Array::from_iter_values((0..80).map(|i| i % 8))),
            ),
            (
                "price",
                Arc::new(Float64Array::from_iter_values(
                    (0..80).map(|i| 10.0 + i as f64 * 1.5),
                )),
            ),
            (
                "ts",
                Arc::new(TimestampMicrosecondArray::from_iter_values(
                    (0..80).map(|i| 1_672_531_200_000_000 + i * 7 * DAY),
                )),
            ),
        ],
        "order_id",
    );
    orders.foreign_keys.push(ForeignKey {
        key: 1,
        referenced_table: "users".into(),
        columns: vec!["user_id".into()],
        referenced_columns: vec!["user_id".into()],
    });
    Memory {
        tables: vec![users, orders],
        ..Memory::default()
    }
}

fn fallback(memory: &mut Memory, withheld: &[OptionalRequest], limits: Option<FallbackLimits>) {
    let mut declared = Memory::capabilities();
    declared
        .optional
        .retain(|request| !withheld.contains(request));
    declared.fallback = limits;
    memory.declared = Some(declared);
}

fn describe(memory: &Memory) -> Observed {
    let mut given = case();
    let core = given.core.as_mut().unwrap();
    core.materialize = None;
    core.predict = None;
    host::run_memory_over(&given, memory, &layer::root()).unwrap()
}

#[test]
fn describe_recovers_preview_after_a_bounded_column_ranges_fallback_fails() {
    let events = table(
        "events",
        vec![
            ("id", Arc::new(Int64Array::from_iter_values(0..1001))),
            (
                "clock",
                Arc::new(TimestampMicrosecondArray::from_iter_values(
                    (0..1001).map(|i| 1_704_067_200_000_000 + i * DAY),
                )),
            ),
        ],
        "id",
    );
    let mut memory = Memory {
        tables: vec![events],
        ..Memory::default()
    };
    fallback(&mut memory, &[OptionalRequest::ColumnRanges], None);
    let preview = describe(&memory);
    assert!(
        matches!(preview.answer, Answer::Described(_)),
        "{preview:?}"
    );
    for limits in [
        FallbackLimits {
            max_rows: 999,
            max_bytes: 1_000_000,
        },
        FallbackLimits {
            max_rows: 2000,
            max_bytes: 1,
        },
    ] {
        fallback(&mut memory, &[OptionalRequest::ColumnRanges], Some(limits));
        let bounded = describe(&memory);
        assert!(
            matches!(bounded.answer, Answer::Described(_)),
            "{limits:?}: {bounded:?}"
        );
        assert_eq!(bounded.described, preview.described);
        assert_eq!(bounded.warnings, preview.warnings);
        assert!(
            bounded
                .fetches
                .iter()
                .any(|f| f["request"]["request"] == json!("scan"))
        );
    }
}

#[test]
fn materialize_uses_declared_fallback_for_each_undeclared_sampling_operation() {
    let needed = [
        OptionalRequest::TimeBounds,
        OptionalRequest::SampleEntities,
        OptionalRequest::Neighbors,
        OptionalRequest::Lookup,
    ];
    for withheld in needed
        .iter()
        .map(std::slice::from_ref)
        .chain(std::iter::once(needed.as_slice()))
    {
        let mut memory = source();
        fallback(
            &mut memory,
            withheld,
            Some(FallbackLimits {
                max_rows: 10_000,
                max_bytes: 10_000_000,
            }),
        );
        let observed = host::run_memory_over(&case(), &memory, &layer::root()).unwrap();
        assert!(
            matches!(observed.answer, Answer::Predicted(_)),
            "{withheld:?}: {observed:?}"
        );
        assert!(observed.materialized.is_some());
        let asked = memory.asked.lock().unwrap();
        assert!(
            asked
                .iter()
                .all(|r| r.optional().is_none_or(|kind| !withheld.contains(&kind))),
            "{withheld:?}: {asked:?}"
        );
        assert!(asked.iter().any(|r| r.name() == "scan"));
    }
}

#[test]
fn a_materialized_prediction_does_not_charge_an_unrelated_parent_of_a_scan_slice() {
    let mut compact = source();
    let limits = Some(FallbackLimits {
        max_rows: 10_000,
        max_bytes: 100_000,
    });
    fallback(&mut compact, &[OptionalRequest::Lookup], limits);
    let expected = host::run_memory_over(&case(), &compact, &layer::root()).unwrap();
    assert!(
        matches!(expected.answer, Answer::Predicted(_)),
        "{expected:?}"
    );

    let mut sliced = source();
    let original = &sliced.tables[1].rows;
    let parent = concat_batches(original.schema_ref(), &vec![original.clone(); 1000]).unwrap();
    sliced.tables[1].rows = parent.slice(40_000, original.num_rows());
    fallback(&mut sliced, &[OptionalRequest::Lookup], limits);
    let observed = host::run_memory_over(&case(), &sliced, &layer::root()).unwrap();
    assert!(
        matches!(observed.answer, Answer::Predicted(_)),
        "{observed:?}"
    );
    assert_eq!(observed.described, expected.described);
    assert_eq!(observed.answer, expected.answer);
}

#[test]
fn a_public_materialize_uses_the_fallbacks_defined_text_order_before_a_cut() {
    let users = table(
        "users",
        vec![
            ("user_id", Arc::new(Int64Array::from_iter_values(0..8))),
            (
                "target",
                Arc::new(Float64Array::from_iter_values(
                    (0..8).map(|i| 10.0 + i as f64 * 1.5),
                )),
            ),
        ],
        "user_id",
    );
    let mut orders = table(
        "orders",
        vec![
            (
                "order_id",
                Arc::new(StringArray::from_iter_values(
                    (0..8).flat_map(|i| [format!("Z{i}"), format!("a{i}")]),
                )),
            ),
            (
                "user_id",
                Arc::new(Int64Array::from_iter_values((0..8).flat_map(|i| [i, i]))),
            ),
            (
                "amount",
                Arc::new(Float64Array::from_iter_values(
                    (0..8).flat_map(|_| [1.0, 9.0]),
                )),
            ),
        ],
        "order_id",
    );
    orders.foreign_keys.push(ForeignKey {
        key: 1,
        referenced_table: "users".into(),
        columns: vec!["user_id".into()],
        referenced_columns: vec!["user_id".into()],
    });
    let mut given = case();
    given.input.pql = "PREDICT users.target FOR EACH users.user_id".into();
    given.input.materialize.num_hops = Some(case::AsWritten("1".into()));
    given.input.materialize.num_neighbors = Some(case::AsWritten("[1]".into()));
    given.core.as_mut().unwrap().predict = None;
    for reverse in [false, true] {
        let mut memory = Memory {
            tables: vec![users.clone(), orders.clone()],
            ..Memory::default()
        };
        if reverse {
            let rows = &memory.tables[1].rows;
            let positions =
                arrow_array::UInt32Array::from_iter_values((0..rows.num_rows() as u32).rev());
            memory.tables[1].rows =
                arrow_select::take::take_record_batch(rows, &positions).unwrap();
        }
        fallback(
            &mut memory,
            &[OptionalRequest::Neighbors],
            Some(FallbackLimits {
                max_rows: 1000,
                max_bytes: 1_000_000,
            }),
        );
        let observed = host::run_memory_over(&given, &memory, &layer::root()).unwrap();
        assert!(
            matches!(observed.answer, Answer::Materialized(_)),
            "{observed:?}"
        );
        let payload = &observed.materialized.as_ref().unwrap()["payload"];
        for (half, expected) in [("context", 6), ("predict", 2)] {
            let orders = &payload[half]["related_tables"]["orders"];
            let amount = orders["columns"]
                .as_array()
                .unwrap()
                .iter()
                .position(|name| name == "amount")
                .unwrap();
            let rows = orders["rows"].as_array().unwrap();
            assert_eq!(rows.len(), expected);
            // UTF-8 byte order keeps Z before a, independently of scan order.
            // Native SQL collation is a provider operation's own semantics.
            assert!(
                rows.iter()
                    .all(|row| row[amount] == json!({"number": "1.0"})),
                "{orders}"
            );
        }
        assert!(
            memory
                .asked
                .lock()
                .unwrap()
                .iter()
                .all(|request| request.name() != "neighbors")
        );
        assert!(
            memory
                .asked
                .lock()
                .unwrap()
                .iter()
                .any(|request| request.name() == "scan")
        );
    }
}
