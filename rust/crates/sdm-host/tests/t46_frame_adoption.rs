// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Fixed shared frame values cross actual public HostRun ownership boundaries.

use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringArray};
use sdm_contract::data::AttachmentId;
use sdm_contract::facts::PlatformFacts;
use sdm_contract::ops::{Operation, OperationResult};
use sdm_contract::run::StartRun;
use sdm_contract::{
    CONTRACT_VERSION, ControlMessage, HostFacts, Scalar, Sentence, Step, TimestampUnit,
};
use sdm_host::{ArrowData, Attachment, Attachments, HostRun};
use serde_json::{Value, json};
use std::sync::Arc;

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
    .unwrap()
}

fn bytes(value: Value) -> Attachment {
    Attachment::Bytes(serde_json::to_vec(&value).unwrap())
}
fn phase(stage: &str, rows: RecordBatch, prior: Option<Value>) -> (Value, Vec<Sentence>) {
    let mut manifest =
        json!({"stage":stage,"tables":[{"source":{"source":["instance"]},"rows":3}]});
    let mut attached = Attachments::from([
        (
            AttachmentId(1),
            bytes(
                json!({"tables":[{"name":"instance","source":{"source":["instance"]}}],"primary_keys":{},"time_columns":{}}),
            ),
        ),
        (
            AttachmentId(3),
            Attachment::Arrow(ArrowData::new(rows.schema(), vec![rows])),
        ),
    ]);
    if let Some(prior) = prior {
        manifest["inference"] = json!(4);
        attached.insert(AttachmentId(4), bytes(prior));
    }
    attached.insert(AttachmentId(2), bytes(manifest));
    let run = HostRun::start(&sdm_host::default_root(), &message(), attached)
        .unwrap_or_else(|out| panic!("{:?}", sdm_host::read(&out)));
    let mut warnings = vec![];
    loop {
        let out = match run.next() {
            Ok(out) | Err(out) => out,
        };
        match sdm_host::read(&out).unwrap() {
            ControlMessage::Warnings { warnings: notices } => {
                warnings.extend(notices.into_iter().map(|w| w.sentence))
            }
            ControlMessage::Step(Step::Done {
                result: OperationResult::InferredFrames(result),
            }) => {
                let Attachment::Bytes(value) = run.take_attachment(result.value).unwrap().unwrap()
                else {
                    panic!("inference bytes")
                };
                return (serde_json::from_slice(&value).unwrap(), warnings);
            }
            other => panic!("fixed frame inference does not fetch or call a model: {other:?}"),
        }
    }
}
fn rows(repeated: bool) -> RecordBatch {
    let mut key: Vec<Option<i64>> = (0..100)
        .map(|n| if n == 0 { None } else { Some(n) })
        .collect();
    if repeated {
        key[2] = Some(1);
    }
    RecordBatch::try_from_iter([
        ("user_id", Arc::new(Int64Array::from(key)) as ArrayRef),
        (
            "feature",
            Arc::new(Int64Array::from_iter_values(0..100)) as ArrayRef,
        ),
    ])
    .unwrap()
}
#[test]
fn fresh_adoption_and_later_time_share_the_fixed_nominated_meaning() {
    let (nomination, notices) = phase("nominate", rows(false), None);
    assert_eq!(notices, vec![Sentence::KeyNoValueColumn]);
    assert_eq!(nomination["value"]["nominated_key"], json!(["user_id"]));
    let (adoption, notices) = phase("adopt", rows(true), Some(nomination));
    assert_eq!(notices, vec![Sentence::FrameKeyRepeatsComposite]);
    let duplicates = adoption["value"]["table"]["duplicate_keys"].clone();
    let (completed, notices) = phase("time", rows(false), Some(adoption));
    assert!(notices.is_empty());
    assert_eq!(completed["value"][0]["table"]["duplicate_keys"], duplicates);
    assert_eq!(completed["value"][0]["table"]["columns"][0]["stype"], "ID");
}

#[test]
fn assembled_sample_columns_keep_metadata_and_accept_later_value_or_presence_changes() {
    let original = RecordBatch::try_from_iter([(
        "label",
        Arc::new(StringArray::from_iter_values(
            (0..40).map(|n| format!("label-{n}")),
        )) as ArrayRef,
    )])
    .unwrap();
    let (nomination, _) = phase("nominate", original.clone(), None);
    let (adoption, _) = phase("adopt", original.clone(), Some(nomination));
    let (complete, _) = phase("time", original.clone(), Some(adoption));
    let (schema, notices) = phase("assemble", original.clone(), Some(complete));
    assert!(notices.is_empty());
    assert_eq!(schema["value"]["samples"].as_array().unwrap().len(), 1);
    let column = schema["value"]["samples"][0].clone();
    let input = json!({"kind":"sample_column", "value":column});
    let (stats, notices) = phase("sample", original.clone(), Some(input.clone()));
    assert_eq!(stats["value"]["distinct"], 40);
    assert_eq!(stats["value"]["present"], 40);
    assert_eq!(notices, vec![Sentence::FrameCategoricalSample]);
    let constant = RecordBatch::try_from_iter([(
        "label",
        Arc::new(StringArray::from(vec!["constant"; 40])) as ArrayRef,
    )])
    .unwrap();
    let (stats, notices) = phase("sample", constant, Some(input.clone()));
    assert_eq!(stats["value"]["distinct"], 1);
    assert!(notices.is_empty());
    let (stats, notices) = phase("sample", original.project(&[]).unwrap(), Some(input));
    assert_eq!(stats["value"]["exists"], false);
    assert_eq!(stats["value"]["present"], 0);
    assert!(notices.is_empty());
}
