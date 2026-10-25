// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Real public host starts, not private inference helpers or a host provider.

use std::sync::Arc;

use arrow_array::{Int16Array, Int32Array, RecordBatch};
use sdm_contract::data::AttachmentId;
use sdm_contract::facts::PlatformFacts;
use sdm_contract::ops::{Operation, OperationResult};
use sdm_contract::run::StartRun;
use sdm_contract::{CONTRACT_VERSION, ControlMessage, ErrorCode, HostFacts, Scalar, Step, Unit};
use sdm_host::{ArrowData, Attachment, Attachments, HostRun, Outgoing, Unreadable, Unreadables};
use serde_json::{Value, json};

fn message() -> Vec<u8> {
    let operation: Operation = serde_json::from_value(json!({
        "op":"describe_source", "provider":{"provider":"frames","manifest":2},
        "source":1,"check_keys":true
    }))
    .unwrap();
    sdm_host::encode(&ControlMessage::Start(Box::new(StartRun {
        contract: CONTRACT_VERSION,
        facts: HostFacts {
            now: Scalar::Timestamp {
                unit: Unit::Us,
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
    .unwrap()
}

fn bytes(value: Value) -> Attachment {
    Attachment::Bytes(serde_json::to_vec(&value).unwrap())
}

fn inputs() -> Attachments {
    let batch = RecordBatch::try_from_iter([
        (
            "user_id",
            Arc::new(Int16Array::from(vec![1, 2, 3])) as arrow_array::ArrayRef,
        ),
        (
            "age",
            Arc::new(Int32Array::from(vec![20, 30, 40])) as arrow_array::ArrayRef,
        ),
    ])
    .unwrap();
    Attachments::from([
        (
            AttachmentId(1),
            bytes(
                json!({"tables":[{"name":"users","source":{"source":["users"]}}],
            "primary_keys":{},"time_columns":{}}),
            ),
        ),
        (
            AttachmentId(2),
            bytes(json!({"stage":"schema","tables":[{"source":{"source":["users"]},"rows":3}]})),
        ),
        (
            AttachmentId(3),
            Attachment::Arrow(ArrowData::new(batch.schema(), vec![batch])),
        ),
    ])
}

fn error(out: Outgoing) -> sdm_contract::CoreError {
    match sdm_host::read(&out).unwrap() {
        ControlMessage::Error { error } => error,
        other => panic!("expected refusal, got {other:?}"),
    }
}

#[test]
fn primitive_frame_is_described_without_any_fetch() {
    let run = HostRun::start(&sdm_host::default_root(), &message(), inputs())
        .unwrap_or_else(|e| panic!("{:?}", error(e)));
    loop {
        match sdm_host::read(&run.next().unwrap()).unwrap() {
            ControlMessage::Step(Step::Done {
                result: OperationResult::DescribeSource(result),
                ..
            }) => {
                let Attachment::Bytes(bytes) =
                    run.take_attachment(result.description).unwrap().unwrap()
                else {
                    panic!("description bytes")
                };
                let value: Value = serde_json::from_slice(&bytes).unwrap();
                assert_eq!(value["tables"][0]["name"], "users");
                assert_eq!(value["tables"][0]["columns"][0]["dtype"], "int64");
                assert_eq!(value["tables"][0]["columns"][1]["dtype"], "int64");
                assert_eq!(value["tables"][0]["primary_key"], json!(["user_id"]));
                assert!(!String::from_utf8(bytes).unwrap().contains("manifest"));
                break;
            }
            ControlMessage::Warnings { .. } => {}
            other => panic!("frame source must not fetch or call a model: {other:?}"),
        }
    }
}

#[test]
fn frame_arrow_validity_retains_present_nan_and_merges_signed_zero() {
    use arrow_array::Float64Array;

    for variant in ["nan", "nan_duplicate", "zero_duplicate", "null"] {
        let mut values: Vec<_> = (0..100).map(|n| Some(f64::from(n))).collect();
        match variant {
            "nan" => values[99] = Some(f64::NAN),
            "nan_duplicate" => {
                values[98] = Some(f64::NAN);
                values[99] = Some(f64::from_bits(0xfff8_0000_0000_0001));
            }
            "zero_duplicate" => values[99] = Some(-0.0),
            "null" => values[99] = None,
            _ => unreachable!(),
        }
        let batch = RecordBatch::try_from_iter([(
            "user_id",
            Arc::new(Float64Array::from(values)) as arrow_array::ArrayRef,
        )])
        .unwrap();
        let mut rows = Some(ArrowData::new(batch.schema(), vec![batch]));
        let mut table_description = None;
        for (stage, rows_id) in [("table", 3), ("schema", 7)] {
            let mut attached = inputs();
            attached.remove(&AttachmentId(3));
            attached.insert(
                AttachmentId(2),
                bytes(json!({
                    "stage":stage, "tables":[{"source":{"source":["users"]},"rows":rows_id}]
                })),
            );
            attached.insert(
                AttachmentId(rows_id),
                Attachment::Arrow(rows.take().expect("the next stage owns its input")),
            );
            let run = HostRun::start(&sdm_host::default_root(), &message(), attached)
                .unwrap_or_else(|e| panic!("{:?}", error(e)));
            let mut notices = 0;
            loop {
                let next = run.next().unwrap();
                if let Some(out) = run.take_warnings().unwrap() {
                    let ControlMessage::Warnings { warnings } = sdm_host::read(&out).unwrap()
                    else {
                        panic!("the warning channel must return warnings")
                    };
                    notices += warnings.len();
                }
                assert!(
                    run.take_warnings().unwrap().is_none(),
                    "warnings drain once"
                );
                match sdm_host::read(&next).unwrap() {
                    ControlMessage::Step(Step::Done {
                        result: OperationResult::DescribeSource(result),
                        ..
                    }) => {
                        let Attachment::Bytes(bytes) =
                            run.take_attachment(result.description).unwrap().unwrap()
                        else {
                            panic!("description bytes")
                        };
                        let value: sdm_contract::ops::SourceDescription =
                            serde_json::from_slice(&bytes).unwrap();
                        assert_eq!(
                            value.tables[0].primary_key,
                            ["user_id"],
                            "{stage}/{variant}"
                        );
                        assert_eq!(
                            value.tables[0].duplicate_keys,
                            Some(Scalar::UInt64(u64::from(matches!(
                                variant,
                                "nan_duplicate" | "zero_duplicate"
                            )))),
                            "{stage}/{variant}"
                        );
                        if stage == "table" {
                            assert_eq!(notices, usize::from(variant != "nan"), "{variant}");
                            let owned_id = result.frame_rows.expect("Table returns its owned rows");
                            let Attachment::Arrow(owned) =
                                run.take_attachment(owned_id).unwrap().unwrap()
                            else {
                                panic!("owned frame rows must be Arrow")
                            };
                            rows = Some(owned);
                            table_description = Some(value);
                        } else {
                            assert_eq!(notices, 0, "Table notices must not repeat: {variant}");
                            assert!(result.frame_rows.is_none());
                            assert_eq!(Some(value), table_description, "{variant}");
                        }
                        break;
                    }
                    other => panic!("a frame must not call a provider: {other:?}"),
                }
            }
            // The taken stream, not a Run or an expired attachment id,
            // survives and is rebound under a fresh input id for Schema.
            drop(run);
        }
    }
}

#[test]
fn manifest_expands_exact_attachments_and_precedes_unreadable_arrow() {
    for variant in ["missing", "extra", "kind", "collision", "malformed"] {
        let mut attached = inputs();
        match variant {
            "missing" => {
                attached.remove(&AttachmentId(3));
            }
            "extra" => {
                attached.insert(AttachmentId(4), bytes(json!("extra")));
            }
            "kind" => {
                attached.insert(AttachmentId(3), bytes(json!("not Arrow")));
            }
            "collision" => {
                attached.insert(AttachmentId(2), bytes(json!({"stage":"schema","tables":[{"source":{"source":["users"]},"rows":2}]})));
            }
            "malformed" => {
                attached.insert(AttachmentId(2), bytes(json!({"tables":"bad"})));
            }
            _ => unreachable!(),
        }
        let failed = HostRun::start(&sdm_host::default_root(), &message(), attached)
            .err()
            .unwrap();
        assert_eq!(error(failed).code, ErrorCode::InvalidMessage, "{variant}");
    }
    let mut attached = inputs();
    attached.remove(&AttachmentId(3));
    attached.insert(AttachmentId(2), bytes(json!({"tables":"bad"})));
    let unreadable = Unreadables::from([(AttachmentId(3), Unreadable::Producer)]);
    let failed =
        HostRun::start_reading(&sdm_host::default_root(), &message(), attached, &unreadable)
            .err()
            .unwrap();
    assert_eq!(error(failed).code, ErrorCode::InvalidMessage);
}

#[test]
fn table_stage_rejects_constructor_overrides_and_the_manifest_is_locked() {
    let mut attached = inputs();
    attached.insert(
        AttachmentId(2),
        bytes(json!({"stage":"table","tables":[{"source":{"source":["users"]},"rows":3}]})),
    );
    attached.insert(
        AttachmentId(1),
        bytes(
            json!({"tables":[{"name":"users","source":{"source":["users"]}}],
        "primary_keys":{"users":{"value":["age"]}},"time_columns":{}}),
        ),
    );
    let failure = match HostRun::start(&sdm_host::default_root(), &message(), attached) {
        Err(out) => error(out),
        Ok(run) => error(run.next().err().expect("table stage must reject overrides")),
    };
    assert_eq!(failure.code, ErrorCode::InvalidMessage);
    let schema: Value = serde_json::from_str(&sdm_contract::schema_json()).unwrap();
    assert_eq!(
        schema["$defs"]["Attachments"]["properties"]["Frames.manifest"]["$ref"],
        "#/$defs/FrameManifest"
    );
    assert!(
        schema["$defs"]["FrameManifest"]["required"]
            .as_array()
            .unwrap()
            .contains(&json!("stage"))
    );
}

#[test]
fn frame_detachment_owns_nested_sliced_and_dictionary_buffers() {
    use arrow_array::builder::StringDictionaryBuilder;
    use arrow_array::types::{Int8Type, Int32Type};
    use arrow_array::{Array, ListArray, StringArray, StructArray};
    use arrow_schema::Field;
    use std::collections::BTreeSet;

    fn buffers(data: &arrow_data::ArrayData, found: &mut BTreeSet<usize>) {
        for buffer in data.buffers() {
            if !buffer.is_empty() {
                found.insert(buffer.as_ptr() as usize);
            }
        }
        if let Some(nulls) = data.nulls() {
            if !nulls.inner().values().is_empty() {
                found.insert(nulls.inner().values().as_ptr() as usize);
            }
        }
        for child in data.child_data() {
            buffers(child, found);
        }
    }
    let list = ListArray::from_iter_primitive::<Int32Type, _, _>([
        Some(vec![Some(1), None]),
        None,
        Some(vec![Some(3), Some(4)]),
    ]);
    let text = StringArray::from(vec![Some("first"), None, Some("last")]);
    let nested = StructArray::from(vec![
        (
            Arc::new(Field::new("list", list.data_type().clone(), true)),
            Arc::new(list) as arrow_array::ArrayRef,
        ),
        (
            Arc::new(Field::new("text", text.data_type().clone(), true)),
            Arc::new(text) as arrow_array::ArrayRef,
        ),
    ]);
    let mut dictionary = StringDictionaryBuilder::<Int8Type>::new();
    dictionary.append("first").unwrap();
    dictionary.append_null();
    dictionary.append("last").unwrap();
    let batch = RecordBatch::try_from_iter([
        ("nested", Arc::new(nested) as arrow_array::ArrayRef),
        (
            "dictionary",
            Arc::new(dictionary.finish()) as arrow_array::ArrayRef,
        ),
    ])
    .unwrap()
    .slice(1, 2);
    let original = ArrowData::new(batch.schema(), vec![batch]);
    let owned = sdm_host::detach_frame(original.clone()).unwrap();
    assert_eq!(owned.batches, original.batches);
    let mut foreign = BTreeSet::new();
    let mut detached = BTreeSet::new();
    for array in original.batches[0].columns() {
        buffers(&array.to_data(), &mut foreign);
    }
    for array in owned.batches[0].columns() {
        buffers(&array.to_data(), &mut detached);
    }
    assert!(
        foreign.is_disjoint(&detached),
        "a nested/dictionary/null/data buffer is still shared"
    );
}

#[test]
fn text_frame_is_an_explicit_unfinished_input_family() {
    let batch = RecordBatch::try_from_iter([(
        "user_id",
        Arc::new(arrow_array::StringArray::from(vec!["a", "b"])) as arrow_array::ArrayRef,
    )])
    .unwrap();
    let mut attached = inputs();
    attached.insert(
        AttachmentId(3),
        Attachment::Arrow(ArrowData::new(batch.schema(), vec![batch])),
    );
    let failure = match HostRun::start(&sdm_host::default_root(), &message(), attached) {
        Err(out) => error(out),
        Ok(run) => error(
            run.next()
                .err()
                .expect("text frame remains an unfinished development family"),
        ),
    };
    assert_eq!(failure.code, ErrorCode::UnsupportedRequest);
}
