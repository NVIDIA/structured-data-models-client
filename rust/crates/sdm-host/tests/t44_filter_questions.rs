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

fn question(query: &str) -> (Value, Attachments) {
    let column = |name: &str, dtype: &str, stype: &str| json!({"name":name,"dtype":dtype,"stype":stype,"stype_before_key":null});
    let schema = json!({"tables":[
        {"name":"users","source":{"source":["users"]},"columns":[column("user_id","int64","ID"),column("age","int32","numerical")],"primary_key":["user_id"],"frame_key_form":"scalar","time_column":null},
        {"name":"orders","source":{"source":["orders"]},"columns":[column("order_id","int64","ID"),column("user_id","int64","ID"),column("at","timestamp[ns]","timestamp")],"primary_key":["order_id"],"frame_key_form":"scalar","time_column":"at"}
    ],"links":[[{"table":"orders","column":"user_id"},{"table":"users","column":"user_id"}]]});
    let (value, owners, warnings) = prepare(
        json!({"stage":"question","schema":1,"query":2,"candidates_given":false,"statistic_given":false,"top_k_given":false}),
        Attachments::from([
            (AttachmentId(1), bytes(schema)),
            (AttachmentId(2), bytes(json!(query))),
        ]),
    );
    assert!(warnings.is_empty());
    (value["data"].clone(), owners)
}

fn questions(query: &str) -> (Value, Vec<Value>) {
    let (question, mut owners) = question(query);
    owners.insert(
        AttachmentId(4),
        bytes(json!({"num_context_samples":1,"num_neighbors":[3],"num_hops":1})),
    );
    let (value, _, warnings) = prepare(
        json!({"stage":"questions","input":{
            "question":question,"candidates":null,"values":4,"options":[],"option_facts":[],"prediction_rows":2,"limits":{}
        }}),
        owners,
    );
    (value["data"].clone(), warnings)
}

#[test]
fn ordinary_filter_notice_is_metadata_only_and_has_complete_condition_and_count() {
    let (_, warnings) =
        questions("PREDICT users.age FOR EACH users.user_id WHERE users.user_id = 1");
    assert_eq!(warnings.len(), 1);
    assert_eq!(warnings[0]["code"], "FILTER_CHOSE_EXAMPLES");
    assert_eq!(
        warnings[0]["fields"],
        json!({
            "condition":{"t":"utf8","v":"users.user_id = 1"},
            "entities":{"t":"u64","v":"2"}
        })
    );
    assert!(
        questions("PREDICT users.age FOR EACH users.user_id")
            .1
            .is_empty()
    );
}

#[test]
fn forecast_parent_is_silent_and_each_ordinary_child_emits_one_filter_notice() {
    let (parent, warnings) = questions(
        "PREDICT COUNT(orders.*, 0, 30, days) FORECAST 2 TIMEFRAMES FOR EACH users.user_id WHERE users.user_id = 1",
    );
    assert!(warnings.is_empty());
    let parts = parent["decomposition"]["parts"].as_array().unwrap();
    assert_eq!(parts.len(), 2);
    for child in parts {
        let (ordinary, warnings) = questions(child["query"].as_str().unwrap());
        assert!(ordinary["decomposition"].is_null());
        assert_eq!(warnings.len(), 1);
        assert_eq!(warnings[0]["code"], "FILTER_CHOSE_EXAMPLES");
    }
}
