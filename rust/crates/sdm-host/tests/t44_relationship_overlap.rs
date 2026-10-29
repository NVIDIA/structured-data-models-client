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

#[test]
fn malformed_overlap_alternatives_and_short_circuits_are_protocol_errors() {
    for evidence in [
        json!({"observation":"unavailable","child":0}),
        json!({"observation":"distinct","relationship":[{"table":"c","column":"f"},{"table":"p","column":"k"}],"child":0,"parent":1,"intersects":null}),
        json!({"observation":"distinct","relationship":[{"table":"c","column":"f"}],"child":1,"parent":1,"intersects":false}),
        record(1, None, None),
        record(1, Some(0), Some(false)),
        record(0, None, Some(true)),
    ] {
        let attachments = Attachments::from([
            (
                AttachmentId(100),
                bytes(json!({"stage":"relationship_overlap","observations":1})),
            ),
            (AttachmentId(1), bytes(json!([evidence]))),
        ]);
        let run = HostRun::start(&sdm_host::default_root(), &message(), attachments).unwrap();
        let out = run.next().unwrap_err();
        let ControlMessage::Error { error } = sdm_host::read(&out).unwrap() else {
            panic!("malformed overlap accepted")
        };
        assert_eq!(error.code, sdm_contract::ErrorCode::InvalidMessage);
        assert!(run.take_sampler_state().unwrap().is_none());
        assert!(run.take_warnings().unwrap().is_none());
    }
}

fn caller_schema(dtype: &str) -> Value {
    let column = |name: &str, dtype: &str, stype: &str| json!({"name":name,"dtype":dtype,"stype":stype,"stype_before_key":null,"stype_declared":true});
    json!({"tables":[
        {"name":"parent","source":{"source":["parent"]},"columns":[column("id","uint64","numerical"),column("feature",dtype,"numerical")],"primary_key":["id"],"frame_key_form":"scalar","time_column":null},
        {"name":"child","source":{"source":["child"]},"columns":[column("fk","uint64","numerical")],"primary_key":[],"time_column":null}
    ],"links":[[{"table":"child","column":"fk"},{"table":"parent","column":"id"}]]})
}

#[test]
fn completed_caller_report_preserves_marks_even_before_a_column_refusal() {
    for dtype in ["int16", "string"] {
        let original = caller_schema(dtype);
        let (value, owners, warnings) = prepare(
            json!({"stage":"caller_meaning","schema":1}),
            Attachments::from([(AttachmentId(1), bytes(original.clone()))]),
        );
        assert!(owners.is_empty());
        assert!(warnings.is_empty());
        assert_eq!(value["value"], "caller_meaning");
        let report = &value["data"];
        assert_eq!(report["schema"]["tables"][0]["columns"][0]["stype"], "ID");
        assert_eq!(report["schema"]["tables"][1]["columns"][0]["stype"], "ID");
        assert_eq!(
            report["schema"]["tables"][1]["columns"][0]["stype_before_key"],
            "numerical"
        );
        assert_eq!(
            report["schema"]["tables"][0]["columns"][0]["dtype"],
            "uint64"
        );
        assert_eq!(
            report["schema"]["tables"][0]["columns"][0]["stype_declared"],
            true
        );
        assert_eq!(
            report["schema"]["tables"][0]["columns"][1],
            original["tables"][0]["columns"][1]
        );
        if dtype == "string" {
            assert_eq!(report["diagnostic"]["sentence"], "meaning_not_carried");
        } else {
            assert!(report["diagnostic"].is_null());
        }
    }
}

#[test]
fn caller_meanings_precede_links_and_released_marks_remain_live() {
    let mut original = caller_schema("string");
    original["links"][0][1]["table"] = json!("missing");
    let (value, _, _) = prepare(
        json!({"stage":"caller_meaning","schema":1}),
        Attachments::from([(AttachmentId(1), bytes(original))]),
    );
    assert_eq!(
        value["data"]["diagnostic"]["sentence"],
        "meaning_not_carried"
    );
    let mut released = caller_schema("int16");
    released["links"] = json!([]);
    released["tables"][1]["columns"][0]["stype"] = json!("ID");
    released["tables"][1]["columns"][0]["stype_before_key"] = json!("numerical");
    let (value, _, _) = prepare(
        json!({"stage":"caller_meaning","schema":1}),
        Attachments::from([(AttachmentId(1), bytes(released))]),
    );
    assert!(value["data"]["diagnostic"].is_null());
    assert_eq!(
        value["data"]["schema"]["tables"][1]["columns"][0]["stype"],
        "numerical"
    );
    assert!(value["data"]["schema"]["tables"][1]["columns"][0]["stype_before_key"].is_null());
}
