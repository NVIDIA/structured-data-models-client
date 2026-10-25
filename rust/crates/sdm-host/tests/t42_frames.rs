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
    })).unwrap();
    sdm_host::encode(&ControlMessage::Start(Box::new(StartRun {
        contract: CONTRACT_VERSION,
        facts: HostFacts {
            now: Scalar::Timestamp {unit: Unit::Us, zone: Some("UTC".into()), value: 0},
            local_zone: None, zone_roots: vec![],
            platform: PlatformFacts {os: "linux".into(), strftime_pads_year: false},
            date_literal_range: None,
        },
        operation,
    }))).unwrap()
}

fn bytes(value: Value) -> Attachment {
    Attachment::Bytes(serde_json::to_vec(&value).unwrap())
}

fn inputs() -> Attachments {
    let batch = RecordBatch::try_from_iter([
        ("user_id", Arc::new(Int16Array::from(vec![1, 2, 3])) as arrow_array::ArrayRef),
        ("age", Arc::new(Int32Array::from(vec![20, 30, 40])) as arrow_array::ArrayRef),
    ]).unwrap();
    Attachments::from([
        (AttachmentId(1), bytes(json!({"tables":[{"name":"users","source":{"source":["users"]}}],
            "primary_keys":{},"time_columns":{}}))),
        (AttachmentId(2), bytes(json!({"stage":"schema","tables":[{"source":{"source":["users"]},"rows":3}]}))),
        (AttachmentId(3), Attachment::Arrow(ArrowData::new(batch.schema(), vec![batch]))),
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
    let run = HostRun::start(&sdm_host::default_root(), &message(), inputs()).unwrap_or_else(|e| panic!("{:?}", error(e)));
    loop {
        match sdm_host::read(&run.next().unwrap()).unwrap() {
            ControlMessage::Step(Step::Done {result: OperationResult::DescribeSource(result), ..}) => {
                let Attachment::Bytes(bytes) = run.take_attachment(result.description).unwrap().unwrap() else {panic!("description bytes")};
                let value: Value = serde_json::from_slice(&bytes).unwrap();
                assert_eq!(value["tables"][0]["name"], "users");
                assert_eq!(value["tables"][0]["columns"][0]["dtype"], "int64");
                assert_eq!(value["tables"][0]["columns"][1]["dtype"], "int64");
                assert_eq!(value["tables"][0]["primary_key"], json!(["user_id"]));
                assert!(!String::from_utf8(bytes).unwrap().contains("manifest"));
                break;
            }
            ControlMessage::Warnings { .. } => {},
            other => panic!("frame source must not fetch or call a model: {other:?}"),
        }
    }
}

#[test]
fn manifest_expands_exact_attachments_and_precedes_unreadable_arrow() {
    for variant in ["missing", "extra", "kind", "collision", "malformed"] {
        let mut attached = inputs();
        match variant {
            "missing" => {attached.remove(&AttachmentId(3));},
            "extra" => {attached.insert(AttachmentId(4), bytes(json!("extra")));},
            "kind" => {attached.insert(AttachmentId(3), bytes(json!("not Arrow")));},
            "collision" => {attached.insert(AttachmentId(2), bytes(json!({"stage":"schema","tables":[{"source":{"source":["users"]},"rows":2}]})));},
            "malformed" => {attached.insert(AttachmentId(2), bytes(json!({"tables":"bad"})));},
            _ => unreachable!(),
        }
        let failed = HostRun::start(&sdm_host::default_root(), &message(), attached).err().unwrap();
        assert_eq!(error(failed).code, ErrorCode::InvalidMessage, "{variant}");
    }
    let mut attached = inputs();
    attached.remove(&AttachmentId(3));
    attached.insert(AttachmentId(2), bytes(json!({"tables":"bad"})));
    let unreadable = Unreadables::from([(AttachmentId(3), Unreadable::Producer)]);
    let failed = HostRun::start_reading(&sdm_host::default_root(), &message(), attached, &unreadable).err().unwrap();
    assert_eq!(error(failed).code, ErrorCode::InvalidMessage);
}

#[test]
fn table_stage_rejects_constructor_overrides_and_the_manifest_is_locked() {
    let mut attached = inputs();
    attached.insert(AttachmentId(2), bytes(json!({"stage":"table","tables":[{"source":{"source":["users"]},"rows":3}]})));
    attached.insert(AttachmentId(1), bytes(json!({"tables":[{"name":"users","source":{"source":["users"]}}],
        "primary_keys":{"users":{"value":["age"]}},"time_columns":{}})));
    let failure = match HostRun::start(&sdm_host::default_root(), &message(), attached) {
        Err(out) => error(out),
        Ok(run) => error(run.next().err().expect("table stage must reject overrides")),
    };
    assert_eq!(failure.code, ErrorCode::InvalidMessage);
    let schema: Value = serde_json::from_str(&sdm_contract::schema_json()).unwrap();
    assert_eq!(schema["$defs"]["Attachments"]["properties"]["Frames.manifest"]["$ref"], "#/$defs/FrameManifest");
    assert!(schema["$defs"]["FrameManifest"]["required"].as_array().unwrap().contains(&json!("stage")));
}
