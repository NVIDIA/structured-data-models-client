// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Plain-data direct numeric preparation and tabular prediction through HostRun.
//! Frozen reference: request.py:372-488, base.py:218-310 and tabular.py:171-333.

use arrow_array::{ArrayRef, Float64Array, RecordBatch};
use sdm_contract::data::AttachmentId;
use sdm_contract::facts::PlatformFacts;
use sdm_contract::ops::{Operation, OperationResult};
use sdm_contract::run::{ModelOutcome, ModelReply, StartRun};
use sdm_contract::{CONTRACT_VERSION, ControlMessage, HostFacts, Scalar, Step, TimestampUnit};
use sdm_host::{ArrowData, Attachment, Attachments, HostRun};
use serde_json::{Value, json};
use std::sync::Arc;

fn start(operation: Value, attached: Attachments) -> HostRun {
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
        operation: serde_json::from_value::<Operation>(operation).unwrap(),
    })))
    .unwrap();
    HostRun::start(&sdm_host::default_root(), &message, attached)
        .unwrap_or_else(|error| panic!("start: {:?}", sdm_host::read(&error)))
}
fn bytes(value: Value) -> Attachment {
    Attachment::Bytes(serde_json::to_vec(&value).unwrap())
}
fn prepare(value: Value, more: Attachments) -> Result<Value, sdm_contract::CoreError> {
    let mut attached = more;
    attached.insert(AttachmentId(1), bytes(value));
    let run = start(json!({"op":"prepare_request","input":1}), attached);
    loop {
        match sdm_host::read(&run.next().unwrap()).unwrap() {
            ControlMessage::Warnings { .. } => {}
            ControlMessage::Error { error } => return Err(error),
            ControlMessage::Step(Step::Done {
                result: OperationResult::PrepareRequest(result),
                ..
            }) => {
                let Attachment::Bytes(value) = run.take_attachment(result.value).unwrap().unwrap()
                else {
                    panic!("preparation bytes")
                };
                return Ok(serde_json::from_slice(&value).unwrap());
            }
            other => panic!("preparation must not fetch or call model: {other:?}"),
        }
    }
}
fn arguments() -> Value {
    prepare(json!({"stage":"arguments", "context":{"kind":"frame"}, "predict":{"kind":"frame"}, "related":[], "task":{"value":"regression"}}), Attachments::new()).unwrap()
}
fn header(names: &[&str], rows: usize) -> Value {
    json!({"rows":rows,"columns":names.iter().enumerate().map(|(ordinal,name)|json!({"ordinal":ordinal,"name":{"kind":"text","value":name},"physical":{"type":"float64"}})).collect::<Vec<_>>(),"duplicates":[]})
}
fn headers() -> Value {
    prepare(json!({"stage":"headers", "arguments":arguments(), "context":header(&["feature","target"],3), "predict":header(&["feature"],2), "target_context":{"positions":[1]}, "target_predict":{"positions":[]}, "predict_context":[[0]]}), Attachments::new()).unwrap()
}
fn rows(names: &[(&str, Vec<f64>)]) -> Attachment {
    let batch = RecordBatch::try_from_iter(names.iter().map(|(name, values)| {
        (
            *name,
            Arc::new(Float64Array::from(values.clone())) as ArrayRef,
        )
    }))
    .unwrap();
    Attachment::Arrow(ArrowData::new(batch.schema(), vec![batch]))
}
fn halves() -> Attachments {
    Attachments::from([
        (
            AttachmentId(2),
            rows(&[("0", vec![1., 2., 3.]), ("1", vec![2., 4., 6.])]),
        ),
        (AttachmentId(3), rows(&[("0", vec![4., 5.])])),
    ])
}

#[test]
fn early_arguments_need_no_headers_or_values_and_reject_unknown_task() {
    let error=prepare(json!({"stage":"arguments","context":{"kind":"frame"},"predict":{"kind":"frame"},"related":[],"task":{"value":"unknown"}}),Attachments::new()).unwrap_err();
    assert_eq!(error.sentence.as_str(), "direct_task");
}

#[test]
fn constructor_header_and_arrow_coherence_checks_each_axis() {
    for axis in ["row_count", "field_count", "physical", "ordinal"] {
        let mut header = headers();
        match axis {
            "row_count" => header["context"]["rows"] = json!(4),
            "field_count" => {
                header["context"]["columns"].as_array_mut().unwrap().pop();
            }
            "physical" => header["context"]["columns"][0]["physical"] = json!({"type":"int64"}),
            _ => header["context"]["columns"][0]["ordinal"] = json!(1),
        }
        let error=prepare(json!({"stage":"schema","headers":header,"context":2,"predict":3,"entity_table":"instance"}),halves()).unwrap_err();
        assert_eq!(
            error.sentence,
            sdm_contract::Sentence::InvalidMessage,
            "{axis}"
        );
    }
}

#[test]
fn direct_numeric_preparation_is_model_neutral_and_binds_one_tabular_call() {
    let prepared=prepare(json!({"stage":"schema","headers":headers(),"context":2,"predict":3,"entity_table":"instance"}),halves()).unwrap();
    assert_eq!(prepared["task"], "regression");
    assert_eq!(prepared["target"], "target");
    assert_eq!(
        prepared["description"]["tables"][0]["columns"][0]["dtype"],
        "float64"
    );
    assert!(prepared.get("model").is_none());
    let mut call = prepare(
        json!({"stage":"model","model":"kumo-tabular"}),
        Attachments::new(),
    )
    .unwrap();
    for input in [json!({"stage":"outputs","call":call,"kind":"regression"})] {
        call = prepare(input, Attachments::new()).unwrap();
    }
    call=prepare(json!({"stage":"capabilities","call":call,"kind":"regression","anchored":false,"relationships":false,"tables":1}),Attachments::new()).unwrap();
    call = prepare(
        json!({"stage":"task","call":call,"kind":"regression"}),
        Attachments::new(),
    )
    .unwrap();
    call = prepare(
        json!({"stage":"capacity","call":call,"rows":3}),
        Attachments::new(),
    )
    .unwrap();
    let mut attached = halves();
    attached.insert(AttachmentId(1),bytes(json!({"bound":call,"context":2,"predict":3,"context_header":header(&["feature","target"],3),"predict_header":header(&["feature"],2),"schema":prepared["schema"],"entity_table":"instance","target":"target","task":"regression","inference":{}})));
    let run = start(
        json!({"op":"predict","direct":1,"model":"kumo-tabular","request_id":"sdm-fixed","outputs":[],"default_outputs":true}),
        attached,
    );
    loop {
        match sdm_host::read(&run.next().unwrap()).unwrap() {
            ControlMessage::Warnings { .. } => {}
            ControlMessage::Step(Step::CallModel(call)) => {
                let Attachment::Bytes(body) = run.take_attachment(call.body).unwrap().unwrap()
                else {
                    panic!("body bytes")
                };
                let value: Value = serde_json::from_slice(&body).unwrap();
                assert_eq!(value["model"], "kumo-tabular");
                assert_eq!(
                    value["task"],
                    json!({"kind":"regression","target":{"column_name":"target","dtype":"float64"}})
                );
                assert_eq!(
                    value["context"]["instance_table"]["rows"],
                    json!([[1., 2.], [2., 4.], [3., 6.]])
                );
                assert_eq!(
                    value["predict"]["instance_table"]["rows"],
                    json!([[4.], [5.]])
                );
                assert_eq!(value["metadata"]["request_id"], "sdm-fixed");
                assert!(value.get("inference").is_none());
                assert!(value["output"].get("prediction_statistic").is_none());
                let answer=br#"{"id":"direct-fixture","model":"kumo-tabular","metadata":{},"predictions":[{"row_index":0,"id":"0","prediction":10},{"row_index":1,"id":"1","prediction":11}]}"#;
                let reply = sdm_host::encode(&ControlMessage::ModelReply(ModelReply {
                    id: call.id,
                    outcome: ModelOutcome::Response {
                        status: 200,
                        body: AttachmentId(100),
                        content_type: "application/json".into(),
                    },
                }))
                .unwrap();
                run.reply(
                    &reply,
                    Attachments::from([(AttachmentId(100), Attachment::Bytes(answer.to_vec()))]),
                )
                .unwrap();
            }
            ControlMessage::Step(Step::Done {
                result: OperationResult::Predict(result),
                ..
            }) => {
                assert_eq!(result.model, "kumo-tabular");
                assert_eq!(result.request_id, "direct-fixture");
                let Attachment::Arrow(rows) = run.take_attachment(result.result).unwrap().unwrap()
                else {
                    panic!("result Arrow")
                };
                assert_eq!(
                    rows.batches
                        .iter()
                        .map(RecordBatch::num_rows)
                        .sum::<usize>(),
                    2
                );
                break;
            }
            other => panic!("expected tabular model call/result, got {other:?}"),
        }
    }
}
