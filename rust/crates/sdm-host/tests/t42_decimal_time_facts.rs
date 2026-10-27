// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The public core selects native time reads and consumes completed facts.

use std::sync::Arc;

use arrow_array::{
    ArrayRef, Decimal128Array, Float64Array, RecordBatch, TimestampMicrosecondArray,
};
use sdm_contract::data::AttachmentId;
use sdm_contract::facts::PlatformFacts;
use sdm_contract::frames::{
    FrameDecimalExponent, FrameInference, FrameScalar, FrameTimeFacts, FrameTimeFirst,
    NativeDecimal, NativeDecimalColumn, PassedOver,
};
use sdm_contract::ops::{Operation, OperationResult};
use sdm_contract::run::StartRun;
use sdm_contract::{
    CONTRACT_VERSION, ControlMessage, HostFacts, Scalar, Sentence, Step, TimestampUnit,
};
use sdm_host::{ArrowData, Attachment, Attachments, HostRun};
use serde_json::json;

fn stage(
    stage: &str,
    value: Option<FrameInference>,
    rows: Option<RecordBatch>,
    decimals: Vec<NativeDecimalColumn>,
) -> (FrameInference, Vec<Sentence>) {
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
                zone: None,
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
    let mut attached = Attachments::from([(
        AttachmentId(1),
        Attachment::Bytes(
            serde_json::to_vec(&json!({
                "tables":[{"name":"rows","source":{"source":["rows"]}}],
                "primary_keys":{},"time_columns":{}
            }))
            .unwrap(),
        ),
    )]);
    let mut manifest = json!({"stage":stage,"tables":[]});
    if let Some(rows) = rows {
        manifest["tables"] = json!([{
            "source":{"source":["rows"]},"rows":3,"decimals":decimals
        }]);
        attached.insert(
            AttachmentId(3),
            Attachment::Arrow(ArrowData::new(rows.schema(), vec![rows])),
        );
    }
    if let Some(value) = value {
        manifest["inference"] = json!(4);
        attached.insert(
            AttachmentId(4),
            Attachment::Bytes(serde_json::to_vec(&value).unwrap()),
        );
    }
    attached.insert(
        AttachmentId(2),
        Attachment::Bytes(serde_json::to_vec(&manifest).unwrap()),
    );
    let run = HostRun::start(&sdm_host::default_root(), &message, attached)
        .unwrap_or_else(|out| panic!("{:?}", sdm_host::read(&out)));
    let mut warnings = vec![];
    loop {
        match sdm_host::read(&run.next().unwrap()).unwrap() {
            ControlMessage::Warnings { warnings: notices } => {
                warnings.extend(notices.into_iter().map(|n| n.sentence))
            }
            ControlMessage::Step(Step::Done {
                result: OperationResult::InferredFrames(result),
                ..
            }) => {
                let Attachment::Bytes(bytes) = run.take_attachment(result.value).unwrap().unwrap()
                else {
                    panic!("inference facts must be bytes")
                };
                return (serde_json::from_slice(&bytes).unwrap(), warnings);
            }
            other => panic!("time facts must not fetch or call a model: {other:?}"),
        }
    }
}

fn original() -> RecordBatch {
    let stamp: ArrayRef = Arc::new(TimestampMicrosecondArray::from_iter_values(
        (0..100).map(|n| 1_700_000_000_000_000 + n),
    ));
    RecordBatch::try_from_iter([
        (
            "row_id",
            Arc::new(Float64Array::from(
                (0..100)
                    .map(|n| (n != 0).then_some(f64::from(n)))
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
        ),
        ("observed_at", stamp.clone()),
        ("loaded_at", stamp.clone()),
        ("birth_date", stamp.clone()),
        ("other_at", stamp),
    ])
    .unwrap()
}

#[test]
fn selected_time_plan_keeps_order_and_skips_named_non_candidates() {
    let (nominated, warnings) = stage("nominate", None, Some(original()), vec![]);
    assert_eq!(warnings, [Sentence::KeyNoValueColumn]);
    let (adopted, _) = stage("adopt", Some(nominated), Some(original()), vec![]);
    let (plan, _) = stage("time_facts", Some(adopted), None, vec![]);
    let FrameInference::TimePlan(plan) = plan else {
        panic!("time plan")
    };
    assert_eq!(plan.columns, ["observed_at", "other_at"]);

    let firsts = vec![
        FrameTimeFirst {
            name: "observed_at".into(),
            rows: 100,
            held: 100,
            same: true,
            first: Some(FrameScalar::Decimal {
                negative: false,
                digits: "0".into(),
                exponent: FrameDecimalExponent::Finite { value: -2 },
            }),
        },
        FrameTimeFirst {
            name: "other_at".into(),
            rows: 100,
            held: 100,
            same: false,
            first: Some(FrameScalar::Value {
                value: Scalar::Timestamp {
                    unit: TimestampUnit::Us,
                    zone: None,
                    value: 1_700_000_000_000_000,
                },
            }),
        },
    ];
    let rows = RecordBatch::try_from_iter([
        (
            "observed_at",
            Arc::new(
                Decimal128Array::from(vec![0; 100])
                    .with_precision_and_scale(3, 0)
                    .unwrap(),
            ) as ArrayRef,
        ),
        ("other_at", original().column(4).clone()),
    ])
    .unwrap();
    let facts = FrameInference::TimeFacts(Box::new(FrameTimeFacts {
        plan: *plan,
        firsts,
    }));
    let (finished, _) = stage(
        "time",
        Some(facts),
        Some(rows),
        vec![NativeDecimalColumn {
            column: 0,
            name: "observed_at".into(),
            decimal: NativeDecimal::Decimal128 {
                precision: 3,
                scale: 2,
            },
        }],
    );
    let FrameInference::Inferred(tables) = finished else {
        panic!("inferred time")
    };
    assert_eq!(tables[0].table.time_column.as_deref(), Some("other_at"));
    assert!(tables[0].time.refused.iter().any(|(name, why)| {
        name == "observed_at"
            && matches!(why, PassedOver::FrameConstant {
            first: Some(FrameScalar::Decimal { digits, .. }), ..
        } if digits == "0")
    }));
}
