// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Public Rust-host constructor twins for the native Arrow Decimal family.
//! These tests wait for serialized composition with the shared frame reader.

use std::sync::Arc;

use arrow_array::{ArrayRef, Decimal128Array, Decimal256Array, RecordBatch};
use arrow_buffer::i256;
use sdm_contract::data::AttachmentId;
use sdm_contract::facts::PlatformFacts;
use sdm_contract::ops::{Dtype, Operation, OperationResult, SourceDescription, Stype};
use sdm_contract::run::StartRun;
use sdm_contract::{CONTRACT_VERSION, ControlMessage, HostFacts, Scalar, Step, TimestampUnit};
use sdm_host::{ArrowData, Attachment, Attachments, HostRun};
use serde_json::json;

fn start(column: ArrayRef) -> SourceDescription {
    let operation: Operation = serde_json::from_value(json!({
        "op":"describe_source", "provider":{"provider":"frames","manifest":2},
        "source":1,"check_keys":true
    }))
    .unwrap();
    let message = sdm_host::encode(&ControlMessage::Start(Box::new(StartRun {
        contract: CONTRACT_VERSION,
        facts: HostFacts {
            now: Scalar::Timestamp {
                unit: TimestampUnit::Us,
                zone: Some("UTC".into()),
                value: 0,
            },
            local_zone: None,
            zone_roots: vec![],
            platform: PlatformFacts {
                os: "linux".into(),
                strftime_pads_year: false,
            },
            date_literal_range: None,
        },
        operation,
    })))
    .unwrap();
    let batch = RecordBatch::try_from_iter([("value_id", column)]).unwrap();
    let attached = Attachments::from([
        (
            AttachmentId(1),
            Attachment::Bytes(
                serde_json::to_vec(&json!({
                    "tables":[{"name":"rows","source":{"source":["rows"]}}],
                    "primary_keys":{},"time_columns":{}
                }))
                .unwrap(),
            ),
        ),
        (
            AttachmentId(2),
            Attachment::Bytes(
                serde_json::to_vec(&json!({
                    "stage":"schema",
                    "tables":[{"source":{"source":["rows"]},"rows":3}]
                }))
                .unwrap(),
            ),
        ),
        (
            AttachmentId(3),
            Attachment::Arrow(ArrowData::new(batch.schema(), vec![batch])),
        ),
    ]);
    let run = HostRun::start(&sdm_host::default_root(), &message, attached)
        .unwrap_or_else(|out| panic!("{:?}", sdm_host::read(&out)));
    loop {
        match sdm_host::read(&run.next().unwrap()).unwrap() {
            ControlMessage::Warnings { .. } => {}
            ControlMessage::Step(Step::Done {
                result: OperationResult::DescribeSource(result),
                ..
            }) => {
                let Attachment::Bytes(bytes) =
                    run.take_attachment(result.description).unwrap().unwrap()
                else {
                    panic!("description must be bytes")
                };
                return serde_json::from_slice(&bytes).unwrap();
            }
            other => panic!("constructor must not fetch or call a model: {other:?}"),
        }
    }
}

#[test]
fn adjacent_scaled_decimals_nominate_a_key_without_float_collisions() {
    let values = vec![
        Some(1_250_000_000_000_000_001),
        Some(1_250_000_000_000_000_002),
    ];
    let arrays: [ArrayRef; 2] = [
        Arc::new(
            Decimal128Array::from(values.clone())
                .with_precision_and_scale(38, 18)
                .unwrap(),
        ),
        Arc::new(
            Decimal256Array::from(
                values
                    .into_iter()
                    .map(|value| value.map(i256::from_i128))
                    .collect::<Vec<_>>(),
            )
            .with_precision_and_scale(76, 18)
            .unwrap(),
        ),
    ];
    for array in arrays {
        let description = start(array);
        let table = &description.tables[0];
        assert_eq!(table.columns[0].dtype, Dtype::Float64);
        assert_eq!(table.primary_key, ["value_id"]);
        assert_eq!(table.columns[0].stype, Stype::Id);
    }
}

#[test]
fn decimal256_wide_keys_preserve_exact_values() {
    let array = Decimal256Array::from(vec![
        Some(
            "1000000000000000000000000000000000000000000000000000000000000"
                .parse::<i256>()
                .unwrap(),
        ),
        Some(
            "2000000000000000000000000000000000000000000000000000000000000"
                .parse::<i256>()
                .unwrap(),
        ),
    ])
    .with_precision_and_scale(76, 0)
    .unwrap();
    let description = start(Arc::new(array));
    let table = &description.tables[0];
    assert_eq!(table.columns[0].dtype, Dtype::String);
    assert_eq!(table.primary_key, ["value_id"]);
    assert_eq!(table.columns[0].stype, Stype::Id);
}

#[test]
fn known_decimal_type_survives_empty_and_all_null_arrays() {
    for values in [vec![], vec![None, None]] {
        for scale in [-2, 0, 2] {
            let array = Decimal128Array::from(values.clone())
                .with_precision_and_scale(18, scale)
                .unwrap();
            let description = start(Arc::new(array));
            let table = &description.tables[0];
            assert_eq!(
                table.columns[0].dtype,
                if scale > 0 {
                    Dtype::Float64
                } else {
                    Dtype::Int64
                }
            );
            assert!(table.primary_key.is_empty());
            assert!(table.time_column.is_none());
        }
    }
}

#[test]
fn declared_precision_fast_path_does_not_become_a_new_value_refusal() {
    // Frozen _decimal_dtype answers from precision before looking at cells,
    // even when negative scale expands the value beyond int64.
    let array = Decimal128Array::from(vec![999_999_999_999_999_999_i128])
        .with_precision_and_scale(18, -2)
        .unwrap();
    let description = start(Arc::new(array));
    assert_eq!(description.tables[0].columns[0].dtype, Dtype::Int64);
}
