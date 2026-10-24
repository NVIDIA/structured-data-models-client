// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `predict` of a request asked in several model calls (`main`'s
//! multi-call `predict`, 1fc5a686; T2.8c), through `sdm_core::start`: every
//! call is written before the first is sent, each carrying the id the host
//! drew for it; each call's rows are numbered on from where the previous
//! call's ended; the answers are given back as one; and a call that fails
//! ends the run with its own error, carrying the answers received before it
//! (`answered`, with `calls_answered`), as a forecast's failed step carries
//! the steps answered before it (`parts_answered`). The implementer's own
//! tests; the acceptance cases are the test author's.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{Array, RecordBatch};
use sdm_contract::data::AttachmentId;
use sdm_contract::facts::PlatformFacts;
use sdm_contract::ops::{
    ForecastStep, Operation, OperationResult, Predict, REQUEST_STATE_VERSION, RequestBatch,
    RequestState,
};
use sdm_contract::run::StartRun;
use sdm_contract::{
    CONTRACT_VERSION, CoreError, Detail, ErrorCode, ErrorKind, HostFacts, Scalar, Step, Unit,
};
use sdm_core::{
    Attachment, Attachments, CompositionRoot, ModelResult, Run, UnknownZone, Zones, start,
};
use serde_json::{Value, json};

struct NoZones;

impl Zones for NoZones {
    fn zone(&self, _: &str) -> Result<jiff::tz::TimeZone, UnknownZone> {
        Err(UnknownZone)
    }
}

fn facts() -> HostFacts {
    HostFacts {
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
    }
}

const QUERY: &str = "PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id";

/// One call's payload, as materialize writes one, scoring `entities`.
fn payload(entities: &[i64]) -> Vec<u8> {
    let rows: Vec<Value> = entities
        .iter()
        .enumerate()
        .map(|(at, entity)| json!([at + 2, entity]))
        .collect();
    serde_json::to_vec(&json!({
        "model": "kumo-relational",
        "task": {
            "kind": "regression",
            "target": {"column_name": "TARGET", "dtype": "float32"},
            "entity_table_names": ["users"],
        },
        "schema": {
            "instance_table": {"columns": {
                "instance_id": {"dtype": "int64"},
                "__kumo_entity_ref_0": {"dtype": "int64"},
            }},
            "related_tables": {"users": {"columns": {"user_id": {"dtype": "int64"}}}},
            "relationships": [{
                "source_columns": ["__kumo_entity_ref_0"],
                "target_columns": ["user_id"],
                "target_table": "users",
            }],
        },
        "context": {"instance_table": {
            "format": "arrays",
            "columns": ["instance_id", "TARGET"],
            "rows": [[0, 1.5], [1, 2.5]],
        }},
        "predict": {"instance_table": {
            "format": "arrays",
            "columns": ["instance_id", "__kumo_entity_ref_0"],
            "rows": rows,
        }},
        "output": {"fields": ["prediction"]},
        "inference": {"run_mode": "fast"},
        "metadata": {"num_context_examples": 2, "source_query": QUERY},
    }))
    .unwrap()
}

/// The endpoint's answer to a call's body: each row of its predict half
/// answered in order, numbered from zero, as the endpoint numbers every
/// call's.
fn answer(body: &[u8], id: &str) -> Vec<u8> {
    let body: Value = serde_json::from_slice(body).unwrap();
    let rows = body["predict"]["instance_table"]["rows"]
        .as_array()
        .unwrap();
    let predictions: Vec<Value> = rows
        .iter()
        .enumerate()
        .map(|(at, row)| json!({"id": row[0].to_string(), "row_index": at, "prediction": 1.0}))
        .collect();
    serde_json::to_vec(&json!({
        "object": "prediction.response",
        "id": id,
        "model": "kumo-relational",
        "metadata": {},
        "predictions": predictions,
    }))
    .unwrap()
}

/// A request state of `calls`, each a call's entities: the first call's
/// payload the state's, every later call's a batch carrying its own id.
fn batched(calls: &[&[i64]]) -> (RequestState, Attachments) {
    let mut attachments = Attachments::from([
        (AttachmentId(1), Attachment::Bytes(payload(calls[0]))),
        (
            AttachmentId(2),
            Attachment::Bytes(serde_json::to_vec(QUERY).unwrap()),
        ),
    ]);
    let mut batches = Vec::new();
    for (at, entities) in (3u32..).zip(&calls[1..]) {
        attachments.insert(AttachmentId(at), Attachment::Bytes(payload(entities)));
        batches.push(RequestBatch {
            payload: AttachmentId(at),
            rows: entities.len() as u64,
            request_id: Some(format!("sdm-call-{}", at - 1)),
        });
    }
    let state = RequestState {
        version: REQUEST_STATE_VERSION,
        payload: AttachmentId(1),
        query: AttachmentId(2),
        forecast: None,
        batches: (!batches.is_empty()).then_some(batches),
        ranking: None,
    };
    (state, attachments)
}

fn predict(state: RequestState, attachments: Attachments) -> Result<Run, CoreError> {
    start(
        &CompositionRoot::new(Arc::new(sdm_data::Data::new())),
        StartRun {
            contract: CONTRACT_VERSION,
            facts: facts(),
            operation: Operation::Predict(Predict {
                state: Some(state),
                held: None,
                model: "kumo-relational".into(),
                request_id: "sdm-call-1".into(),
                outputs: vec!["prediction".into()],
                default_outputs: false,
                inference_asked: vec![],
            }),
        },
        attachments,
        Arc::new(NoZones),
    )
}

/// Answers each call the run asks for, failing the one numbered `fail`
/// (from zero) with `failure`; the run's end, and each call's body.
fn drive(
    run: &mut Run,
    fail: Option<usize>,
    failure: ModelResult,
) -> (Result<Step, CoreError>, Vec<Value>) {
    let mut bodies = Vec::new();
    loop {
        let step = run.next();
        let Ok(Step::CallModel(call)) = step else {
            return (step, bodies);
        };
        let Some(Attachment::Bytes(body)) = run.take_attachment(call.body) else {
            panic!("the body is attached");
        };
        let at = bodies.len();
        bodies.push(serde_json::from_slice(&body).unwrap());
        let reply = if fail == Some(at) {
            failure.clone()
        } else {
            ModelResult::Response {
                status: 200,
                content_type: "application/json".into(),
                body: answer(&body, &format!("r{at}")),
            }
        };
        run.submit_model(call.id, reply).unwrap();
    }
}

fn refused() -> ModelResult {
    ModelResult::Response {
        status: 500,
        content_type: "application/json".into(),
        body: br#"{"error":"scripted_failure"}"#.to_vec(),
    }
}

fn arrow(run: &mut Run, id: AttachmentId) -> RecordBatch {
    let Some(Attachment::Arrow(data)) = run.take_attachment(id) else {
        panic!("the result is attached as Arrow");
    };
    let [batch] = <[RecordBatch; 1]>::try_from(data.batches).unwrap();
    batch
}

fn int64s(batch: &RecordBatch, name: &str) -> Vec<i64> {
    batch
        .column_by_name(name)
        .unwrap()
        .as_primitive::<Int64Type>()
        .values()
        .to_vec()
}

fn count(error: &CoreError, name: &str) -> Option<u64> {
    match error.fields.get(name) {
        Some(Detail::Scalar(Scalar::UInt64(n))) => Some(*n),
        _ => None,
    }
}

#[test]
fn every_call_is_sent_with_its_own_id_and_the_rows_are_numbered_across_calls() {
    // `rows_per_call=3` over 8 entities: three calls of 3, 3 and 2.
    let (state, attachments) = batched(&[&[10, 11, 12], &[13, 14, 15], &[16, 17]]);
    let mut run = predict(state, attachments).unwrap();
    let (end, bodies) = drive(&mut run, None, refused());
    let ids: Vec<&str> = bodies
        .iter()
        .map(|b| b["metadata"]["request_id"].as_str().unwrap())
        .collect();
    assert_eq!(ids, ["sdm-call-1", "sdm-call-2", "sdm-call-3"]);
    let Ok(Step::Done {
        result: OperationResult::Predict(predicted),
    }) = end
    else {
        panic!("the run predicts: {end:?}");
    };
    let result = arrow(&mut run, predicted.result);
    // Each call's rows from where the last call's ended, in the order the
    // entities were asked for, as the client's `_numbered_across_calls`.
    assert_eq!(int64s(&result, "row_index"), (0..8).collect::<Vec<_>>());
    assert_eq!(int64s(&result, "answer_index"), (0..8).collect::<Vec<_>>());
    // The first call's model and id, the parts counted in the metadata, as
    // `PredictionResult.merge` writes it.
    assert_eq!(predicted.request_id, "r0");
    let Some(Attachment::Bytes(metadata)) = run.take_attachment(predicted.metadata) else {
        panic!("the metadata is attached");
    };
    let metadata: Value = serde_json::from_slice(&metadata).unwrap();
    assert_eq!(
        metadata,
        json!({"batches": 3, "batch_request_ids": ["r0", "r1", "r2"]})
    );
}

#[test]
fn a_failure_on_call_2_carries_the_answers_of_the_calls_before_it() {
    let (state, attachments) = batched(&[&[10, 11, 12], &[13, 14, 15], &[16, 17]]);
    let mut run = predict(state, attachments).unwrap();
    let (end, bodies) = drive(&mut run, Some(2), refused());
    // The third call failed, and none after it was made.
    assert_eq!(bodies.len(), 3);
    let error = end.unwrap_err();
    // The error keeps its own code, as the client's carries it.
    assert_eq!(error.code, ErrorCode::InvalidResponse);
    assert_eq!(count(&error, "calls_answered"), Some(2));
    let answered = error.answered.clone().expect("the answers are carried");
    assert_eq!(answered.rows, 6);
    let result = arrow(&mut run, answered.result);
    assert_eq!(int64s(&result, "row_index"), (0..6).collect::<Vec<_>>());
    // What `Predicted` names beside its result travels on the schema.
    let named = result.schema().metadata().clone();
    assert_eq!(named["model"], "kumo-relational");
    assert_eq!(named["request_id"], "r0");
    assert_eq!(named["outputs"], r#"["prediction"]"#);
    let metadata: Value = serde_json::from_str(&named["metadata"]).unwrap();
    assert_eq!(metadata["batches"], json!(2));
}

#[test]
fn a_failure_on_the_first_call_carries_nothing() {
    let (state, attachments) = batched(&[&[10, 11, 12], &[13, 14, 15]]);
    let mut run = predict(state, attachments).unwrap();
    let failure = ModelResult::TransportError {
        kind: ErrorKind::ModelEndpoint,
        retryable: true,
    };
    let (end, bodies) = drive(&mut run, Some(0), failure);
    assert_eq!(bodies.len(), 1);
    let error = end.unwrap_err();
    assert_eq!(
        (error.code, error.retryable),
        (ErrorCode::TransportError, true)
    );
    assert!(error.answered.is_none());
    assert!(!error.fields.contains_key("calls_answered"));
}

#[test]
fn a_failed_transport_after_a_call_keeps_its_code_and_carries_the_call() {
    let (state, attachments) = batched(&[&[10, 11], &[12]]);
    let mut run = predict(state, attachments).unwrap();
    let failure = ModelResult::TransportError {
        kind: ErrorKind::ModelEndpoint,
        retryable: false,
    };
    let (end, _) = drive(&mut run, Some(1), failure);
    let error = end.unwrap_err();
    assert_eq!(error.code, ErrorCode::TransportError);
    assert_eq!(count(&error, "calls_answered"), Some(1));
    assert_eq!(error.answered.expect("carried").rows, 2);
}

#[test]
fn a_later_call_without_an_id_is_refused_before_anything_is_sent() {
    let (mut state, attachments) = batched(&[&[10], &[11]]);
    state.batches.as_mut().unwrap()[0].request_id = None;
    let error = predict(state, attachments).err().expect("refused");
    assert_eq!(error.code, ErrorCode::InvalidRequest);
    assert_eq!(
        error.fields.get("field"),
        Some(&Detail::text("state.batches.request_id"))
    );
}

#[test]
fn a_context_is_held_to_the_cap_per_call_not_to_the_sum() {
    // Two calls of 6,000 context rows each: 12,000 in all, past the
    // model's 10,000, but no call sends more than 6,000, so both are sent.
    let wide = |entities: &[i64]| -> Vec<u8> {
        let mut payload: Value = serde_json::from_slice(&payload(entities)).unwrap();
        let rows: Vec<Value> = (0..6_000)
            .map(|at| json!([at, f64::from(at % 7)]))
            .collect();
        payload["context"]["instance_table"]["rows"] = Value::Array(rows);
        serde_json::to_vec(&payload).unwrap()
    };
    let (state, mut attachments) = batched(&[&[10], &[11]]);
    attachments.insert(AttachmentId(1), Attachment::Bytes(wide(&[10])));
    attachments.insert(AttachmentId(3), Attachment::Bytes(wide(&[11])));
    let mut run = predict(state, attachments).unwrap();
    let (end, bodies) = drive(&mut run, None, refused());
    assert_eq!(bodies.len(), 2);
    assert!(matches!(end, Ok(Step::Done { .. })), "{end:?}");
}

#[test]
fn a_forecast_failing_at_a_step_carries_the_steps_answered_before_it() {
    let mut attachments = Attachments::from([
        (AttachmentId(1), Attachment::Bytes(payload(&[10, 11]))),
        (
            AttachmentId(2),
            Attachment::Bytes(serde_json::to_vec(QUERY).unwrap()),
        ),
    ]);
    let mut steps = Vec::new();
    for step in 2..=3u32 {
        let at = 2 * step - 1;
        attachments.insert(AttachmentId(at), Attachment::Bytes(payload(&[10, 11])));
        attachments.insert(
            AttachmentId(at + 1),
            Attachment::Bytes(serde_json::to_vec(QUERY).unwrap()),
        );
        steps.push(ForecastStep {
            step,
            payload: AttachmentId(at),
            query: AttachmentId(at + 1),
            request_id: Some(format!("sdm-step-{step}")),
        });
    }
    let state = RequestState {
        version: REQUEST_STATE_VERSION,
        payload: AttachmentId(1),
        query: AttachmentId(2),
        forecast: Some(steps),
        batches: None,
        ranking: None,
    };
    let mut run = predict(state, attachments).unwrap();
    let (end, bodies) = drive(&mut run, Some(2), refused());
    assert_eq!(bodies.len(), 3);
    let error = end.unwrap_err();
    assert_eq!(
        error.fields.get("forecast_step"),
        Some(&Detail::Scalar(Scalar::UInt32(3)))
    );
    assert_eq!(count(&error, "parts_answered"), Some(2));
    assert!(!error.fields.contains_key("calls_answered"));
    let answered = error.answered.clone().expect("the steps are carried");
    let result = arrow(&mut run, answered.result);
    // Each step's rows, numbered from its own zero, and named by step.
    assert_eq!(int64s(&result, "row_index"), [0, 0, 1, 1]);
    let steps = result.column_by_name("forecast_step").unwrap();
    assert_eq!(steps.len(), 4);
    assert_eq!(steps.null_count(), 0);
}

#[test]
fn a_forecast_s_state_with_batches_is_refused() {
    let (mut state, attachments) = batched(&[&[10], &[11]]);
    state.forecast = Some(vec![]);
    let error = predict(state, attachments).err().expect("refused");
    assert_eq!(error.code, ErrorCode::InvalidRequest);
    assert_eq!(
        error.fields.get("field"),
        Some(&Detail::text("state.batches"))
    );
}
