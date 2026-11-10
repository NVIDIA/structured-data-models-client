// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Labels match independently calculated source values through the Rust host.

use std::path::PathBuf;

use arrow_array::cast::AsArray;
use arrow_array::types::{Float32Type, Int64Type, TimestampNanosecondType};
use arrow_array::{Array, RecordBatch};
use arrow_schema::DataType;
use sdm_contract::ops::{Operation, OperationResult};
use sdm_contract::test_ops::TestOperation;
use sdm_host::Attachment;
use sdm_testkit::host;
use sdm_testkit::{bases, case};
use serde_json::Value;

fn contract() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../contract")
}

/// One context example as the core gave it: the entity's key as the
/// payload writes an integer or text, the anchor as it writes a time, and
/// the label as JSON (a number's text read back exactly, a boolean, text).
#[derive(Debug, Clone, PartialEq)]
struct Example {
    entity: Value,
    anchor: String,
    target: Value,
}

/// The core's context examples for `case`, on this host.
fn core_examples(name: &str) -> Vec<Example> {
    let bases = bases::load(&contract().join("bases.yaml")).unwrap();
    let path = contract().join(format!("cases/s1/{name}.json"));
    let loaded = case::load(&[path]).unwrap().remove(0);
    let (observed, driven) = host::materialize_as(&loaded.case, &bases, &|materialize| {
        Operation::Test(TestOperation::Labels {
            materialize: Box::new(materialize),
        })
    })
    .unwrap();
    let Some((handle, Ok(OperationResult::Test(result)))) = driven else {
        panic!("{name}: the labels were not given: {:?}", observed.answer);
    };
    let Ok(Some(Attachment::Arrow(data))) = handle.take_attachment(result.data.unwrap()) else {
        panic!("{name}: the labels are not attached");
    };
    let batch = arrow_select::concat::concat_batches(&data.schema, &data.batches).unwrap();
    assert_eq!(batch.num_rows() as u64, result.rows);
    examples(&batch)
}

/// `ns` nanoseconds since the epoch as the payload writes a time
/// (`YYYY-MM-DDTHH:MM:SS.ffffffZ`), by H. Hinnant's `civil_from_days`.
fn iso_microseconds(ns: i64) -> String {
    let micros = ns.div_euclid(1_000);
    let seconds = micros.div_euclid(1_000_000);
    let fraction = micros.rem_euclid(1_000_000);
    let days = seconds.div_euclid(86_400);
    let of_day = seconds.rem_euclid(86_400);
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}.{fraction:06}Z",
        of_day / 3_600,
        of_day / 60 % 60,
        of_day % 60
    )
}

fn examples(batch: &RecordBatch) -> Vec<Example> {
    let entity = batch.column_by_name("entity").unwrap();
    let anchor = batch
        .column_by_name("anchor")
        .unwrap()
        .as_primitive::<TimestampNanosecondType>();
    let target = batch.column_by_name("target").unwrap();
    (0..batch.num_rows())
        .map(|row| Example {
            entity: match entity.data_type() {
                DataType::Int64 => Value::from(entity.as_primitive::<Int64Type>().value(row)),
                DataType::Utf8 => Value::from(entity.as_string::<i32>().value(row)),
                other => panic!("an entity key of {other}"),
            },
            anchor: iso_microseconds(anchor.value(row)),
            target: match target.data_type() {
                DataType::Float32 => {
                    Value::from(f64::from(target.as_primitive::<Float32Type>().value(row)))
                }
                DataType::Int64 => Value::from(target.as_primitive::<Int64Type>().value(row)),
                DataType::Boolean => Value::from(target.as_boolean().value(row)),
                DataType::Utf8 => Value::from(target.as_string::<i32>().value(row)),
                other => panic!("a label of {other}"),
            },
        })
        .collect()
}

#[test]
fn temporal_labels_follow_authored_order_values_and_window_bounds() {
    for name in ["base.regression", "base.binary", "decimal.target-38-18"] {
        let examples = core_examples(name);
        // Context windows are emitted newest first, with every entity per window.
        assert!(
            examples
                .windows(2)
                .all(|pair| pair[0].anchor >= pair[1].anchor)
        );
        for window in examples.chunks_exact(8) {
            assert!(window.iter().all(|row| row.anchor == window[0].anchor));
            let entities: std::collections::BTreeSet<_> = window
                .iter()
                .map(|row| row.entity.as_i64().unwrap())
                .collect();
            assert_eq!(entities, (0..8).collect());
        }
        let mut grid = std::collections::BTreeSet::new();
        let calendar = sdm_testkit::duck::open().unwrap();
        for example in examples {
            let entity = example.entity.as_i64().unwrap();
            let day: i64 = calendar
                .query_row(
                    "SELECT date_diff('day', DATE '2023-01-01', CAST(? AS TIMESTAMP))",
                    [&example.anchor],
                    |row| row.get(0),
                )
                .unwrap();
            assert!(grid.insert((entity, day)), "duplicate {name} entity/anchor");
            let orders: Vec<_> = (0..80_i64)
                .filter(|i| i % 8 == entity && i * 7 > day && i * 7 <= day + 30)
                .collect();
            let expected = if name == "base.binary" {
                Value::Bool(!orders.is_empty())
            } else {
                // The decimal input's 1e-18 increment is below float32 precision.
                let sum: f64 = orders.iter().map(|i| 10.0 + 1.5 * *i as f64).sum();
                Value::from(f64::from(sum as f32))
            };
            assert_eq!(example.target, expected, "{name} entity={entity} day={day}");
        }
        let expected: std::collections::BTreeSet<_> = (0..8_i64)
            .flat_map(|entity| (0..14).map(move |n| (entity, 395 - n * 30)))
            .collect();
        assert_eq!(grid, expected, "{name}");
    }
}

#[test]
fn static_labels_preserve_source_entity_and_country() {
    let examples = core_examples("base.multiclass");
    let mut entities = std::collections::BTreeSet::new();
    for example in examples {
        let entity = example.entity.as_i64().unwrap();
        assert!(entities.insert(entity));
        assert_eq!(
            example.target,
            Value::from(["US", "DE", "IN"][(entity % 3) as usize])
        );
        // The authored orders end at day 79 * 7 from 2023-01-01.
        assert_eq!(example.anchor, "2024-07-07T00:00:00.000000Z");
    }
    assert_eq!(entities, (0..8).collect());
}
