// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native Arrow schemas and graph admission agree without host schema policy.
use arrow_array::{ArrayRef, Date32Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::{Field, Schema};
use sdm_contract::data::AttachmentId;
use sdm_contract::facts::PlatformFacts;
use sdm_contract::frames::{DescribeFrameTables, PrepareFrameGraph};
use sdm_contract::ops::{HeldSchema, Operation, OperationResult};
use sdm_contract::run::StartRun;
use sdm_contract::{CONTRACT_VERSION, HostFacts, Scalar, Step, Unit};
use sdm_core::{ArrowData, Attachment, Attachments, CompositionRoot, Run, UnknownZone, Zones};
use std::sync::Arc;

struct NoZones;
impl Zones for NoZones {
    fn zone(&self, _: &str) -> Result<jiff::tz::TimeZone, UnknownZone> {
        Err(UnknownZone)
    }
}
fn bytes(value: &impl serde::Serialize) -> Attachment {
    Attachment::Bytes(serde_json::to_vec(value).unwrap())
}
fn run(operation: Operation, attached: Attachments) -> Run {
    sdm_core::start(
        &CompositionRoot::new(Arc::new(sdm_data::Data::new())),
        StartRun {
            limits: Default::default(),
            contract: CONTRACT_VERSION,
            facts: HostFacts {
                now: Scalar::Timestamp {
                    unit: Unit::Us,
                    zone: Some("UTC".into()),
                    value: 0,
                },
                local_zone: None,
                zone_roots: vec![],
                date_literal_range: None,
                platform: PlatformFacts {
                    os: "linux".into(),
                    strftime_pads_year: false,
                },
            },
            operation,
        },
        attached,
        Arc::new(NoZones),
    )
    .unwrap()
}
fn table(columns: Vec<(&str, ArrayRef)>) -> ArrowData {
    let schema = Arc::new(Schema::new(
        columns
            .iter()
            .map(|(name, array)| Field::new(*name, array.data_type().clone(), true))
            .collect::<Vec<_>>(),
    ));
    let batch = RecordBatch::try_new(
        schema.clone(),
        columns.into_iter().map(|(_, a)| a).collect(),
    )
    .unwrap();
    ArrowData::new(schema, vec![batch])
}

#[test]
fn dates_and_dictionaries_describe_and_prepare_the_same_graph() {
    use arrow_array::types::Int8Type;
    let users = table(vec![
        ("user_id", Arc::new(Int64Array::from(vec![1, 2]))),
        (
            "group",
            Arc::new(arrow_array::DictionaryArray::<Int8Type>::from_iter([
                "a", "b",
            ])),
        ),
    ]);
    let orders = table(vec![
        ("order_id", Arc::new(Int64Array::from(vec![1, 2, 3]))),
        ("user_id", Arc::new(Int64Array::from(vec![1, 1, 2]))),
        (
            "time",
            Arc::new(Date32Array::from(vec![19000, 19001, 19002])),
        ),
        ("name", Arc::new(StringArray::from(vec!["a", "b", "c"]))),
        (
            "negative_scale",
            Arc::new(
                arrow_array::Decimal128Array::from(vec![1, 2, 3])
                    .with_precision_and_scale(10, -2)
                    .unwrap(),
            ),
        ),
        (
            "decimal256",
            Arc::new(
                arrow_array::Decimal256Array::from(vec![
                    arrow_buffer::i256::from_i128(123),
                    arrow_buffer::i256::from_i128(234),
                    arrow_buffer::i256::from_i128(345),
                ])
                .with_precision_and_scale(38, 2)
                .unwrap(),
            ),
        ),
    ]);
    let names = vec!["users", "orders"];
    let tables = vec![AttachmentId(2), AttachmentId(3)];
    let mut described = run(
        Operation::DescribeFrameTables(DescribeFrameTables {
            names: AttachmentId(1),
            tables,
        }),
        [
            (AttachmentId(1), bytes(&names)),
            (AttachmentId(2), Attachment::Arrow(users.clone())),
            (AttachmentId(3), Attachment::Arrow(orders.clone())),
        ]
        .into(),
    );
    let Step::Done {
        result: OperationResult::DirectValue(value),
    } = described.next().unwrap()
    else {
        panic!("schema operation failed")
    };
    let Some(Attachment::Bytes(schema)) = described.take_attachment(value.value) else {
        panic!("schema absent")
    };
    let held: HeldSchema = serde_json::from_slice(&schema).unwrap();
    assert_eq!(held.tables.len(), 2);
    assert_eq!(held.links.len(), 1);
    assert_eq!(held.tables[1].time_column.as_deref(), Some("time"));
    let mut prepared = run(
        Operation::PrepareFrameGraph(PrepareFrameGraph {
            schema: AttachmentId(1),
            names: AttachmentId(2),
            tables: vec![AttachmentId(3), AttachmentId(4)],
        }),
        [
            (AttachmentId(1), Attachment::Bytes(schema)),
            (AttachmentId(2), bytes(&names)),
            (AttachmentId(3), Attachment::Arrow(users)),
            (AttachmentId(4), Attachment::Arrow(orders)),
        ]
        .into(),
    );
    let Step::Done {
        result: OperationResult::PreparedFrameGraph(graph),
    } = prepared.next().unwrap()
    else {
        panic!("graph operation failed")
    };
    assert!(matches!(
        prepared.take_attachment(graph.graph),
        Some(Attachment::FrameGraph(_))
    ));
}

#[test]
fn nested_columns_are_unsupported_even_when_empty_or_all_null() {
    use arrow_array::{ListArray, types::Int64Type};
    use sdm_contract::{ErrorCode, Sentence};

    let cases = [
        vec![Some(vec![Some(1), Some(2)])],
        vec![None, Some(vec![]), Some(vec![Some(1), None])],
        vec![None, None],
        vec![],
    ];
    for values in cases {
        let array = ListArray::from_iter_primitive::<Int64Type, _, _>(values);
        let mut described = run(
            Operation::DescribeFrameTables(DescribeFrameTables {
                names: AttachmentId(1),
                tables: vec![AttachmentId(2)],
            }),
            [
                (AttachmentId(1), bytes(&vec!["users"])),
                (
                    AttachmentId(2),
                    Attachment::Arrow(table(vec![("sequence", Arc::new(array))])),
                ),
            ]
            .into(),
        );
        let error = described.next().unwrap_err();
        assert_eq!(error.code, ErrorCode::UnsupportedRequest);
        assert_eq!(error.sentence, Sentence::DirectNativeContainer);
        assert_eq!(
            error.fields["column"],
            sdm_contract::Detail::text("sequence")
        );
    }
}

#[test]
fn frame_float32_survives_description_assembly_and_edits_without_changing_source_aliases() {
    use sdm_contract::direct::{
        DirectInput,
        schema::{ObservedColumn, SchemaObservation},
    };
    use sdm_contract::frames::AssembleFrameSchema;
    use sdm_contract::ops::{Dtype, EditSchema, SchemaEdit, SchemaEditContext, Stype};
    use std::collections::BTreeMap;

    fn answer(run: &mut Run) -> HeldSchema {
        let id = match run.next().unwrap() {
            Step::Done {
                result: OperationResult::DirectValue(value),
            } => value.value,
            Step::Done {
                result: OperationResult::EditSchema(value),
            } => value.schema,
            other => panic!("unexpected schema answer: {other:?}"),
        };
        let Some(Attachment::Bytes(value)) = run.take_attachment(id) else {
            panic!("schema absent")
        };
        serde_json::from_slice(&value).unwrap()
    }
    let data = table(vec![
        ("id", Arc::new(Int64Array::from(vec![1, 2]))),
        (
            "single",
            Arc::new(arrow_array::Float32Array::from(vec![Some(1.25), None])),
        ),
    ]);
    let mut described = run(
        Operation::DescribeFrameTables(DescribeFrameTables {
            names: AttachmentId(1),
            tables: vec![AttachmentId(2)],
        }),
        [
            (AttachmentId(1), bytes(&vec!["users"])),
            (AttachmentId(2), Attachment::Arrow(data)),
        ]
        .into(),
    );
    let held = answer(&mut described);
    assert_eq!(held.tables[0].columns[1].dtype.as_deref(), Some("float32"));
    let input = AssembleFrameSchema {
        schema: held,
        times: vec![SchemaObservation::TimeChoice {
            columns: vec![
                ObservedColumn {
                    name: "id".into(),
                    dtype: Dtype::Int64,
                    stype: Stype::Id,
                    untyped: false,
                },
                ObservedColumn {
                    name: "single".into(),
                    dtype: Dtype::Float32,
                    stype: Stype::Numerical,
                    untyped: false,
                },
            ],
            primary_key: Some("id".into()),
            rows: 2,
            populations: vec![],
            starts: None,
        }],
        key_counts: BTreeMap::new(),
        primary_keys: BTreeMap::new(),
        time_columns: BTreeMap::new(),
        relationships: Some(vec![]),
    };
    let mut assembled = run(
        Operation::AssembleFrameSchema(DirectInput {
            input: AttachmentId(1),
        }),
        [(AttachmentId(1), bytes(&input))].into(),
    );
    let mut held = answer(&mut assembled);
    assert_eq!(held.tables[0].columns[1].dtype.as_deref(), Some("float32"));
    // Python frame schemas discard catalog provenance; later edits must still keep width.
    for column in &mut held.tables[0].columns {
        column.source = None;
    }
    for frames in [true, false] {
        let mut attached: Attachments = [
            (AttachmentId(1), bytes(&held)),
            (AttachmentId(2), bytes(&SchemaEdit::Check {})),
        ]
        .into();
        if frames {
            attached.insert(
                AttachmentId(3),
                bytes(&SchemaEditContext::Frames {
                    key_counts: BTreeMap::new(),
                }),
            );
        }
        let mut edited = run(
            Operation::EditSchema(EditSchema {
                schema: AttachmentId(1),
                edit: AttachmentId(2),
                context: frames.then_some(AttachmentId(3)),
                frame: None,
            }),
            attached,
        );
        let result = answer(&mut edited);
        assert_eq!(
            result.tables[0].columns[1].dtype.as_deref(),
            Some(if frames { "float32" } else { "float64" })
        );
    }
}
