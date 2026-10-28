// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Metadata-only filter notices precede native graph or sampler ownership.

use sdm_contract::data::AttachmentId;
use sdm_contract::facts::PlatformFacts;
use sdm_contract::ops::{Operation, OperationResult};
use sdm_contract::run::StartRun;
use sdm_contract::{CONTRACT_VERSION, ControlMessage, HostFacts, Scalar, Step, TimestampUnit};
use sdm_host::{Attachment, Attachments, HostRun};
use serde_json::{Value, json};

fn bytes(value: Value) -> Attachment {
    Attachment::Bytes(serde_json::to_vec(&value).unwrap())
}

fn message() -> Vec<u8> {
    let operation: Operation =
        serde_json::from_value(json!({"op":"prepare_frames","input":100})).unwrap();
    sdm_host::encode(&ControlMessage::Start(Box::new(StartRun {
        contract: CONTRACT_VERSION,
        facts: HostFacts {
            now: Scalar::Timestamp {
                unit: TimestampUnit::Ns,
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

fn prepare(input: Value, mut attachments: Attachments) -> (Value, Attachments, Vec<Value>) {
    attachments.insert(AttachmentId(100), bytes(input));
    let run = HostRun::start(&sdm_host::default_root(), &message(), attachments)
        .unwrap_or_else(|out| panic!("{:?}", sdm_host::read(&out)));
    let mut warnings = Vec::new();
    loop {
        let out = run
            .next()
            .unwrap_or_else(|out| panic!("{:?}", sdm_host::read(&out)));
        if let Some(out) = run.take_warnings().unwrap() {
            let value = serde_json::to_value(sdm_host::read(&out).unwrap()).unwrap();
            warnings.extend(value["warnings"].as_array().unwrap().iter().cloned());
        }
        match sdm_host::read(&out).unwrap() {
            ControlMessage::Step(Step::Warnings) => {}
            ControlMessage::Step(Step::Done {
                result: OperationResult::PrepareFrames(result),
                ..
            }) => {
                assert!(run.take_sampler_state().unwrap().is_none());
                let Attachment::Bytes(value) = run.take_attachment(result.value).unwrap().unwrap()
                else {
                    panic!("prepared metadata")
                };
                let value: Value = serde_json::from_slice(&value).unwrap();
                let mut owners = Attachments::new();
                if value["value"] == "question" {
                    let id = AttachmentId(value["data"]["owner"].as_u64().unwrap() as u32);
                    owners.insert(id, run.take_attachment(id).unwrap().unwrap());
                }
                return (value, owners, warnings);
            }
            other => panic!(
                "metadata-only questions cannot build a graph, fetch or call a model: {other:?}"
            ),
        }
    }
}

fn record(child: u64, parent: Option<u64>, intersects: Option<bool>) -> Value {
    json!({"observation":"distinct","relationship":[{"table":"child","column":"fk"},{"table":"parent","column":"id"}],"child":child,"parent":parent,"intersects":intersects})
}

#[test]
fn overlap_is_ordered_aggregate_policy_without_graph_or_sampler() {
    let records = json!([
        {"observation":"unavailable"},
        record(0,None,None), record(1,Some(0),None), record(1,Some(1),Some(true)),
        record(2,Some(3),Some(false)), record(2,Some(3),Some(false))
    ]);
    let (value, owners, warnings) = prepare(
        json!({"stage":"relationship_overlap","observations":1}),
        Attachments::from([(AttachmentId(1), bytes(records))]),
    );
    assert!(owners.is_empty());
    assert_eq!(value["value"], "relationship_overlap");
    assert_eq!(warnings.len(), 1);
    assert_eq!(warnings[0]["sentence"], "relationships_disjoint_several");
    let positions = &warnings[0]["fields"]["positions"];
    assert_eq!(
        positions,
        &serde_json::to_value(sdm_contract::Detail::List(vec![
            Scalar::UInt64(4),
            Scalar::UInt64(5)
        ]))
        .unwrap()
    );
}

#[test]
fn empty_or_intersecting_native_evidence_is_silent() {
    for records in [
        json!([]),
        json!([record(0, None, None)]),
        json!([record(1, Some(0), None)]),
        json!([record(1, Some(2), Some(true))]),
    ] {
        let (_, owners, warnings) = prepare(
            json!({"stage":"relationship_overlap","observations":1}),
            Attachments::from([(AttachmentId(1), bytes(records))]),
        );
        assert!(owners.is_empty());
        assert!(warnings.is_empty());
    }
}
