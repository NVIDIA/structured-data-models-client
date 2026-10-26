// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `predict` through a Databricks Model Serving endpoint (T3.13), through
//! `sdm_core::start`: the host names the endpoint and the caller's limit
//! (`Predict::model_endpoint`), and the core writes each call wrapped in
//! the endpoint's envelope, weighs it against that limit before it is sent,
//! and reads the endpoint's every answer, as the old client's serving
//! transport does: the served model's row, a failure it answered with, and
//! the endpoint's HTTP or SDK failure, each a refusal named by its status
//! and codes (`ENDPOINT_REFUSED`). A call that fails after earlier calls
//! were answered carries their answers, as on every endpoint.

use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use sdm_contract::data::AttachmentId;
use sdm_contract::error::{REFUSED_CODE, REFUSED_STATUS};
use sdm_contract::facts::PlatformFacts;
use sdm_contract::ops::{
    ModelEndpoint, Operation, OperationResult, Predict, REQUEST_STATE_VERSION, RequestBatch,
    RequestState,
};
use sdm_contract::run::StartRun;
use sdm_contract::{
    CONTRACT_VERSION, CoreError, Detail, ErrorCode, ErrorKind, HostFacts, Scalar, Sentence, Step,
    TimestampUnit,
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

const ENDPOINT: &str = "kumo-rfm-serving";
const SECRET: &str = "Sup3rSecret-CustomerValue-9f2a";
const QUERY: &str = "PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id";

fn facts() -> HostFacts {
    HostFacts {
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
    }
}

/// One call's payload, as materialize writes one, scoring `entities`, a
/// user's name past ASCII in its context.
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
            "related_tables": {"users": {"columns": {
                "user_id": {"dtype": "int64"}, "name": {"dtype": "string"},
            }}},
            "relationships": [{
                "source_columns": ["__kumo_entity_ref_0"],
                "target_columns": ["user_id"],
                "target_table": "users",
            }],
        },
        "context": {
            "instance_table": {
                "format": "arrays",
                "columns": ["instance_id", "TARGET"],
                "rows": [[0, 1.5], [1, 2.5]],
            },
            "related_tables": {"users": {
                "format": "arrays",
                "columns": ["user_id", "name"],
                "rows": [[7, "Zoë \"Ø\""]],
            }},
        },
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

/// The served model's answer to a call: each row of its predict half
/// answered in order, numbered from zero.
fn answer(call: &Value, id: &str) -> String {
    let rows = call["predict"]["instance_table"]["rows"]
        .as_array()
        .unwrap();
    let predictions: Vec<Value> = rows
        .iter()
        .enumerate()
        .map(|(at, row)| json!({"id": row[0].to_string(), "row_index": at, "prediction": 1.0}))
        .collect();
    json!({
        "object": "prediction.response",
        "id": id,
        "model": "kumo-relational",
        "metadata": {},
        "predictions": predictions,
    })
    .to_string()
}

/// The endpoint's body carrying `inner` as its one `response_json` row.
fn served(inner: &str) -> Vec<u8> {
    serde_json::to_vec(&json!({"predictions": [{"response_json": inner}]})).unwrap()
}

/// A request state of `calls`, each a call's entities.
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

fn serving(limit: u64) -> Option<ModelEndpoint> {
    Some(ModelEndpoint::ServingEndpoint {
        name: ENDPOINT.into(),
        max_request_bytes: limit,
    })
}

fn predict(calls: &[&[i64]], endpoint: Option<ModelEndpoint>) -> Run {
    let (state, attachments) = batched(calls);
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
                model_endpoint: endpoint,
            }),
        },
        attachments,
        Arc::new(NoZones),
    )
    .unwrap()
}

/// The call a serving endpoint is sent: its envelope's one row, read back.
fn unwrapped(body: &[u8]) -> Value {
    let envelope: Value = serde_json::from_slice(body).unwrap();
    let records = envelope["dataframe_records"].as_array().unwrap();
    assert_eq!(records.len(), 1);
    assert_eq!(
        envelope.as_object().unwrap().keys().collect::<Vec<_>>(),
        ["dataframe_records"]
    );
    serde_json::from_str(records[0]["request_json"].as_str().unwrap()).unwrap()
}

/// Answers each call the run asks for, the one numbered `fail` (from zero)
/// with `failure`, the others with the served model's answer in the
/// endpoint's row; the run's end, and the bodies sent.
fn drive(
    run: &mut Run,
    fail: Option<usize>,
    failure: &ModelResult,
) -> (Result<Step, CoreError>, Vec<Vec<u8>>) {
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
        let reply = if fail == Some(at) {
            failure.clone()
        } else {
            ModelResult::Response {
                status: 200,
                content_type: "application/json".into(),
                body: served(&answer(&unwrapped(&body), &format!("r{at}"))),
            }
        };
        bodies.push(body);
        run.submit_model(call.id, reply).unwrap();
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

fn written(error: &CoreError) -> String {
    let mut out = serde_json::to_string(error).unwrap();
    out.push_str(&error.to_string());
    out
}

/// The refusal's status and code, as the host raises it.
fn refused_as(error: &CoreError) -> (u16, Option<&str>) {
    assert_eq!(
        (error.kind, error.code),
        (ErrorKind::ModelEndpoint, ErrorCode::EndpointRefused)
    );
    assert!(!written(error).contains(SECRET), "{}", written(error));
    let Some(Detail::Scalar(Scalar::UInt16(status))) = error.fields.get(REFUSED_STATUS) else {
        panic!("a status: {error:?}");
    };
    let code = match error.fields.get(REFUSED_CODE) {
        Some(Detail::Scalar(Scalar::Utf8(code))) => Some(code.as_str()),
        None => None,
        other => panic!("a code: {other:?}"),
    };
    (*status, code)
}

#[test]
fn each_call_is_wrapped_as_the_endpoint_reads_it_and_answered_from_its_row() {
    let calls: &[&[i64]] = &[&[10, 11, 12], &[13, 14]];
    let mut run = predict(calls, serving(16_000_000));
    let failure = ModelResult::TransportError {
        kind: ErrorKind::ModelEndpoint,
        retryable: false,
    };
    let (end, bodies) = drive(&mut run, None, &failure);
    // The same calls the client's own endpoint is sent, each as the one
    // `request_json` row of its envelope. The shared wire codec retains
    // Unicode as UTF-8 while preserving the complete logical payload.
    let mut plain = predict(calls, None);
    let mut sent = Vec::new();
    while let Ok(Step::CallModel(call)) = plain.next() {
        let Some(Attachment::Bytes(body)) = plain.take_attachment(call.body) else {
            panic!("the body is attached");
        };
        let read: Value = serde_json::from_slice(&body).unwrap();
        let answered = answer(&read, &format!("r{}", sent.len())).into_bytes();
        sent.push(read);
        plain
            .submit_model(
                call.id,
                ModelResult::Response {
                    status: 200,
                    content_type: "application/json".into(),
                    body: answered,
                },
            )
            .unwrap();
    }
    assert_eq!(bodies.len(), 2);
    for (body, plain) in bodies.iter().zip(&sent) {
        assert!(!body.is_ascii());
        let text = String::from_utf8(body.clone()).unwrap();
        assert!(text.starts_with(r#"{"dataframe_records":[{"request_json":"{\"model\":"#));
        assert!(text.contains("Zoë") && text.contains('Ø'), "{text}");
        assert_eq!(&unwrapped(body), plain);
    }
    // The answers read as the client's own endpoint's are.
    let Ok(Step::Done {
        result: OperationResult::Predict(predicted),
    }) = end
    else {
        panic!("the run predicts: {end:?}");
    };
    let result = arrow(&mut run, predicted.result);
    assert_eq!(int64s(&result, "row_index"), (0..5).collect::<Vec<_>>());
    assert_eq!(predicted.request_id, "r0");
}

#[test]
fn a_call_past_the_caller_s_limit_is_refused_before_it_is_sent() {
    // The first call fits; the second, larger, does not: it is refused
    // when it would be sent, carrying the first call's answers, as the old
    // transport weighs each call as it sends it.
    let mut probe = predict(&[&[10]], serving(16_000_000));
    let Ok(Step::CallModel(call)) = probe.next() else {
        panic!("a call");
    };
    let Some(Attachment::Bytes(one)) = probe.take_attachment(call.body) else {
        panic!("the body is attached");
    };
    let limit = one.len() as u64;
    let mut run = predict(&[&[10], &[11, 12, 13]], serving(limit));
    let failure = ModelResult::TransportError {
        kind: ErrorKind::ModelEndpoint,
        retryable: false,
    };
    let (end, bodies) = drive(&mut run, None, &failure);
    assert_eq!(bodies.len(), 1);
    assert_eq!(bodies[0], one);
    let error = end.unwrap_err();
    assert_eq!(error.sentence, Sentence::ServingPastLimit);
    assert_eq!(error.code, ErrorCode::InvalidRequest);
    assert_eq!(error.fields["limit"], Detail::Scalar(Scalar::UInt64(limit)));
    assert_eq!(count(&error, "calls_answered"), Some(1));
    // A limit the first call misses: nothing is sent.
    let mut run = predict(&[&[10]], serving(limit - 1));
    let (end, bodies) = drive(&mut run, None, &failure);
    assert!(bodies.is_empty());
    assert_eq!(end.unwrap_err().sentence, Sentence::ServingPastLimit);
}

#[test]
fn an_sdk_failure_after_an_answered_call_is_the_endpoint_s_refusal_carrying_it() {
    let calls: &[&[i64]] = &[&[10, 11, 12], &[13, 14]];
    let failure = ModelResult::Failed {
        status: 503,
        detail: Some(format!("failed to collect any context examples for {SECRET}").into_bytes()),
        provider_code: Some("INTERNAL_ERROR".into()),
        provider_error: Some("InternalError".into()),
    };
    let mut run = predict(calls, serving(16_000_000));
    let (end, bodies) = drive(&mut run, Some(1), &failure);
    assert_eq!(bodies.len(), 2);
    let error = end.unwrap_err();
    assert_eq!(error.sentence, Sentence::ServingCallersFault);
    assert_eq!(refused_as(&error), (503, Some("INVALID_REQUEST")));
    assert_eq!(error.fields["diagnosis"], Detail::text("EMPTY_CONTEXT"));
    assert_eq!(
        error.fields["endpoint_code"],
        Detail::text("INTERNAL_ERROR")
    );
    assert_eq!(
        error.fields["provider_error"],
        Detail::text("InternalError")
    );
    assert_eq!(count(&error, "calls_answered"), Some(1));
    let answered = error.answered.clone().expect("the answers are carried");
    assert_eq!(answered.rows, 3);
    let result = arrow(&mut run, answered.result);
    assert_eq!(int64s(&result, "row_index"), vec![0, 1, 2]);
}

#[test]
fn an_http_failure_is_read_from_its_body_as_the_sdk_reads_it() {
    // A host without the SDK hands over the status and the REST body.
    for (failure, expected) in [
        (
            ModelResult::Failed {
                status: 400,
                detail: Some(
                    format!(
                        r#"{{"error_code":"BAD_REQUEST","message":"INVALID_SCHEMA: row {SECRET}"}}"#
                    )
                    .into_bytes(),
                ),
                provider_code: None,
                provider_error: None,
            },
            (Sentence::ServingRefused, 400, Some("INVALID_SCHEMA")),
        ),
        (
            ModelResult::Response {
                status: 500,
                content_type: "application/json".into(),
                body: format!(r#"{{"error_code":"INTERNAL_ERROR","message":"{SECRET}"}}"#)
                    .into_bytes(),
            },
            (Sentence::ServingFailedCoded, 503, Some("INTERNAL_ERROR")),
        ),
        (
            ModelResult::Failed {
                status: 404,
                detail: None,
                provider_code: None,
                provider_error: None,
            },
            (Sentence::ServingFailed, 404, None),
        ),
    ] {
        let mut run = predict(&[&[10]], serving(16_000_000));
        let (end, _) = drive(&mut run, Some(0), &failure);
        let error = end.unwrap_err();
        assert_eq!(error.sentence, expected.0, "{failure:?}");
        assert_eq!(refused_as(&error), (expected.1, expected.2), "{failure:?}");
        assert_eq!(count(&error, "calls_answered"), None);
    }
}

#[test]
fn a_failure_the_model_answered_with_or_a_row_missing_ends_the_run_by_name() {
    for (body, sentence) in [
        (
            served(&format!(
                r#"{{"status":422,"code":"UNSUPPORTED_TASK","detail":"{SECRET}"}}"#
            )),
            Sentence::ServingRefused,
        ),
        (served("[1]"), Sentence::ServingNotAnObject),
        (served("{"), Sentence::ServingNotJson),
        (
            br#"{"predictions":[]}"#.to_vec(),
            Sentence::ServingPredictionRowCount,
        ),
        (
            br#"{"predictions":[{}]}"#.to_vec(),
            Sentence::ServingResponseColumnMissing,
        ),
    ] {
        let failure = ModelResult::Response {
            status: 200,
            content_type: "application/json".into(),
            body: body.clone(),
        };
        let mut run = predict(&[&[10], &[11]], serving(16_000_000));
        let (end, bodies) = drive(&mut run, Some(1), &failure);
        assert_eq!(bodies.len(), 2);
        let error = end.unwrap_err();
        assert_eq!(
            error.sentence,
            sentence,
            "{}",
            String::from_utf8_lossy(&body)
        );
        assert_eq!(count(&error, "calls_answered"), Some(1));
        assert!(!written(&error).contains(SECRET));
    }
    // A failed transport is still the transport's.
    let failure = ModelResult::TransportError {
        kind: ErrorKind::ModelEndpoint,
        retryable: true,
    };
    let mut run = predict(&[&[10]], serving(16_000_000));
    let (end, _) = drive(&mut run, Some(0), &failure);
    let error = end.unwrap_err();
    assert_eq!(
        (error.code, error.retryable),
        (ErrorCode::TransportError, true)
    );
}
