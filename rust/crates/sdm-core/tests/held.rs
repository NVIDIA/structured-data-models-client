// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `predict` of a request the core did not materialize (T3.26, owner
//! decision 2026-10-23), through `sdm_core::start`: a request held, its
//! payloads, questions and a ranking's candidates as the host holds them,
//! is predicted exactly as the request state `materialize` would have
//! written for it: the same calls, each with the id drawn for it, and the
//! same answer, for a request of one call, one asked in several, a
//! forecast's steps and a ranking's questions, the ranking's candidates
//! named by their place among those held and by the JSON text of their
//! values. A start naming both a state and a request held, or neither, and
//! a ranking held with other than one candidate per question, end before
//! anything is sent.

use std::sync::Arc;

use arrow_array::RecordBatch;
use sdm_contract::data::AttachmentId;
use sdm_contract::facts::PlatformFacts;
use sdm_contract::ops::{
    ForecastStep, HeldCall, HeldPart, HeldRanking, HeldRequest, Operation, OperationResult,
    Predict, REQUEST_STATE_VERSION, Ranking, RankingPart, RequestBatch, RequestState,
};
use sdm_contract::run::StartRun;
use sdm_contract::{
    CONTRACT_VERSION, CoreError, ErrorKind, HostFacts, Scalar, Step, TimestampUnit,
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

const QUERY: &str = "PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id";

/// A call's payload as materialize writes one, scoring `entities`: a
/// regression, or a binary classification of a ranking's question.
fn payload(entities: &[i64], binary: bool) -> Vec<u8> {
    let rows: Vec<Value> = entities
        .iter()
        .enumerate()
        .map(|(at, entity)| json!([at + 2, entity]))
        .collect();
    let (task, labels) = if binary {
        (
            json!({
                "kind": "binary_classification",
                "target": {
                    "column_name": "TARGET", "dtype": "bool",
                    "classes": ["false", "true"], "positive_class": "true",
                },
                "entity_table_names": ["users"],
            }),
            json!([[0, true], [1, false]]),
        )
    } else {
        (
            json!({
                "kind": "regression",
                "target": {"column_name": "TARGET", "dtype": "float32"},
                "entity_table_names": ["users"],
            }),
            json!([[0, 1.5], [1, 2.5]]),
        )
    };
    serde_json::to_vec(&json!({
        "model": "kumo-relational",
        "task": task,
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
            "rows": labels,
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

/// The endpoint's answer to call `at`'s body: each row of its predict half
/// answered in order, a classification with how sure it is, which differs
/// from call to call, so a ranking orders its candidates by it.
fn answer(body: &[u8], at: usize) -> Vec<u8> {
    let body: Value = serde_json::from_slice(body).unwrap();
    let binary = body["task"]["kind"] == "binary_classification";
    let rows = body["predict"]["instance_table"]["rows"]
        .as_array()
        .unwrap();
    let predictions: Vec<Value> = rows
        .iter()
        .enumerate()
        .map(|(row, entity)| {
            if binary {
                let sure = f64::from(u32::try_from((3 * at + 5 * row) % 9 + 1).unwrap()) / 10.0;
                json!({
                    "id": entity[0].to_string(), "row_index": row, "prediction": "true",
                    "probabilities": {"false": 1.0 - sure, "true": sure},
                })
            } else {
                json!({"id": entity[0].to_string(), "row_index": row, "prediction": 1.0})
            }
        })
        .collect();
    serde_json::to_vec(&json!({
        "object": "prediction.response",
        "id": format!("r{at}"),
        "model": "kumo-relational",
        "metadata": {"call": at},
        "predictions": predictions,
    }))
    .unwrap()
}

fn bytes(value: Vec<u8>) -> Attachment {
    Attachment::Bytes(value)
}

fn text(value: &str) -> Attachment {
    Attachment::Bytes(serde_json::to_vec(value).unwrap())
}

fn run(state: Option<RequestState>, held: Option<HeldRequest>, given: Attachments) -> Run {
    try_run(state, held, given).unwrap()
}

fn try_run(
    state: Option<RequestState>,
    held: Option<HeldRequest>,
    given: Attachments,
) -> Result<Run, CoreError> {
    start(
        &CompositionRoot::new(Arc::new(sdm_data::Data::new())),
        StartRun {
            contract: CONTRACT_VERSION,
            facts: facts(),
            operation: Operation::Predict(Predict {
                state,
                held,
                model: "kumo-relational".into(),
                request_id: "sdm-call-1".into(),
                outputs: vec![],
                default_outputs: true,
                inference_asked: vec![],
                model_endpoint: None,
            }),
        },
        given,
        Arc::new(NoZones),
    )
}

/// What a run sends and gives back: each call's body, the result's fields
/// but its attachments' ids, its rows, its metadata and its notices; or the
/// error it ends with.
type Outcome = Result<(Vec<Value>, Value, RecordBatch, Vec<u8>, String), CoreError>;

fn drive(mut run: Run) -> Outcome {
    let mut bodies = Vec::new();
    let end = loop {
        match run.next() {
            Ok(Step::CallModel(call)) => {
                let Some(Attachment::Bytes(body)) = run.take_attachment(call.body) else {
                    panic!("the body is attached");
                };
                let reply = ModelResult::Response {
                    status: 200,
                    content_type: "application/json".into(),
                    body: answer(&body, bodies.len()),
                };
                bodies.push(serde_json::from_slice(&body).unwrap());
                run.submit_model(call.id, reply).unwrap();
            }
            other => break other,
        }
    };
    let Step::Done {
        result: OperationResult::Predict(predicted),
    } = end?
    else {
        panic!("the run predicts");
    };
    let Some(Attachment::Arrow(result)) = run.take_attachment(predicted.result) else {
        panic!("the result is attached as Arrow");
    };
    let [rows] = <[RecordBatch; 1]>::try_from(result.batches).unwrap();
    let Some(Attachment::Bytes(metadata)) = run.take_attachment(predicted.metadata) else {
        panic!("the metadata is attached");
    };
    let fields = json!({
        "model": predicted.model,
        "request_id": predicted.request_id,
        "outputs": predicted.outputs,
        "unscored": predicted.unscored_candidates,
    });
    let notices = format!("{:?}", run.take_warnings());
    Ok((bodies, fields, rows, metadata, notices))
}

fn regression_attachments(calls: &[&[i64]]) -> Attachments {
    let mut given = Attachments::from([
        (AttachmentId(1), bytes(payload(calls[0], false))),
        (AttachmentId(2), text(QUERY)),
    ]);
    for (at, entities) in (3u32..).zip(&calls[1..]) {
        given.insert(AttachmentId(at), bytes(payload(entities, false)));
    }
    given
}

#[test]
fn a_request_of_one_call_held_is_predicted_as_its_state_would_be() {
    let calls: &[&[i64]] = &[&[10, 11, 12]];
    let state = RequestState {
        version: REQUEST_STATE_VERSION,
        payload: AttachmentId(1),
        query: AttachmentId(2),
        forecast: None,
        batches: None,
        ranking: None,
    };
    let held = HeldRequest {
        payload: AttachmentId(1),
        query: AttachmentId(2),
        batches: None,
        forecast: None,
        ranking: None,
    };
    let from_state = drive(run(Some(state), None, regression_attachments(calls))).unwrap();
    let from_held = drive(run(None, Some(held), regression_attachments(calls))).unwrap();
    assert_eq!(from_held.0.len(), 1);
    assert_eq!(from_held, from_state);
}

#[test]
fn a_request_asked_in_several_calls_held_is_predicted_as_its_state_would_be() {
    let calls: &[&[i64]] = &[&[10, 11, 12], &[13, 14, 15], &[16, 17]];
    let state = RequestState {
        version: REQUEST_STATE_VERSION,
        payload: AttachmentId(1),
        query: AttachmentId(2),
        forecast: None,
        batches: Some(
            (3u32..)
                .zip(&calls[1..])
                .map(|(at, entities)| RequestBatch {
                    payload: AttachmentId(at),
                    rows: entities.len() as u64,
                    request_id: Some(format!("sdm-call-{}", at - 1)),
                })
                .collect(),
        ),
        ranking: None,
    };
    let held = HeldRequest {
        payload: AttachmentId(1),
        query: AttachmentId(2),
        batches: Some(
            (3u32..)
                .take(calls.len() - 1)
                .map(|at| HeldCall {
                    payload: AttachmentId(at),
                    request_id: format!("sdm-call-{}", at - 1),
                })
                .collect(),
        ),
        forecast: None,
        ranking: None,
    };
    let from_state = drive(run(Some(state), None, regression_attachments(calls))).unwrap();
    let from_held = drive(run(None, Some(held), regression_attachments(calls))).unwrap();
    let ids: Vec<&str> = from_held
        .0
        .iter()
        .map(|body| body["metadata"]["request_id"].as_str().unwrap())
        .collect();
    assert_eq!(ids, ["sdm-call-1", "sdm-call-2", "sdm-call-3"]);
    assert_eq!(from_held, from_state);
}

/// Each part's payload and question, the first the request's own, from
/// attachment 1 on, two each.
fn parts_attachments(parts: usize, binary: bool) -> Attachments {
    let mut given = Attachments::new();
    for (part, at) in (0..parts).zip((1u32..).step_by(2)) {
        let entity = i64::try_from(part).unwrap();
        given.insert(AttachmentId(at), bytes(payload(&[10 + entity, 11], binary)));
        given.insert(
            AttachmentId(at + 1),
            text(&format!("PREDICT question {part}")),
        );
    }
    given
}

fn held_parts(parts: usize) -> Vec<HeldPart> {
    (1..parts)
        .zip((3u32..).step_by(2))
        .map(|(part, at)| HeldPart {
            payload: AttachmentId(at),
            query: AttachmentId(at + 1),
            request_id: format!("sdm-call-{}", part + 1),
        })
        .collect()
}

#[test]
fn a_forecast_held_is_predicted_as_its_state_would_be_its_steps_numbered_in_order() {
    let state = RequestState {
        version: REQUEST_STATE_VERSION,
        payload: AttachmentId(1),
        query: AttachmentId(2),
        forecast: Some(
            held_parts(3)
                .into_iter()
                .zip(2u32..)
                .map(|(part, step)| ForecastStep {
                    step,
                    payload: part.payload,
                    query: part.query,
                    request_id: Some(part.request_id),
                })
                .collect(),
        ),
        batches: None,
        ranking: None,
    };
    let held = HeldRequest {
        payload: AttachmentId(1),
        query: AttachmentId(2),
        batches: None,
        forecast: Some(held_parts(3)),
        ranking: None,
    };
    let from_state = drive(run(Some(state), None, parts_attachments(3, false))).unwrap();
    let from_held = drive(run(None, Some(held), parts_attachments(3, false))).unwrap();
    assert_eq!(from_held.0.len(), 3);
    assert_eq!(from_held, from_state);
}

#[test]
fn a_ranking_held_is_predicted_as_its_state_would_be_each_candidate_named_by_its_value() {
    // Candidates of three types: the core names each by the JSON text of
    // its value, as materialize writes it.
    let candidates = json!([
        {"value": {"t": "i64", "v": "2"}},
        {"value": {"t": "utf8", "v": "b"}},
        {"integer": "90000000000000001"},
    ]);
    let state = RequestState {
        version: REQUEST_STATE_VERSION,
        payload: AttachmentId(1),
        query: AttachmentId(2),
        forecast: None,
        batches: None,
        ranking: Some(Ranking {
            top_k: Some(2),
            candidate: 0,
            value: "2".into(),
            parts: held_parts(3)
                .into_iter()
                .zip([(1, "\"b\""), (2, "90000000000000001")])
                .map(|(part, (candidate, value))| RankingPart {
                    candidate,
                    value: value.into(),
                    payload: part.payload,
                    query: part.query,
                    request_id: Some(part.request_id),
                })
                .collect(),
        }),
    };
    let held = HeldRequest {
        payload: AttachmentId(1),
        query: AttachmentId(2),
        batches: None,
        forecast: None,
        ranking: Some(HeldRanking {
            top_k: Some(2),
            candidates: AttachmentId(7),
            parts: held_parts(3),
        }),
    };
    let mut given = parts_attachments(3, true);
    given.insert(
        AttachmentId(7),
        bytes(serde_json::to_vec(&candidates).unwrap()),
    );
    let from_state = drive(run(Some(state), None, parts_attachments(3, true))).unwrap();
    let from_held = drive(run(None, Some(held), given)).unwrap();
    assert_eq!(from_held.0.len(), 3);
    assert_eq!(from_held, from_state);
}

#[test]
fn a_start_naming_both_or_neither_or_candidates_not_one_per_question_ends_before_any_call() {
    let held = || HeldRequest {
        payload: AttachmentId(1),
        query: AttachmentId(2),
        batches: None,
        forecast: None,
        ranking: None,
    };
    let state = RequestState {
        version: REQUEST_STATE_VERSION,
        payload: AttachmentId(1),
        query: AttachmentId(2),
        forecast: None,
        batches: None,
        ranking: None,
    };
    for (state, held, given) in [
        (Some(state), Some(held()), regression_attachments(&[&[10]])),
        (None, None, Attachments::new()),
    ] {
        let Err(error) = try_run(state, held, given) else {
            panic!("a start naming both or neither is refused");
        };
        assert_eq!(error.kind, ErrorKind::Protocol);
    }
    let mut ranking = held();
    ranking.ranking = Some(HeldRanking {
        top_k: None,
        candidates: AttachmentId(7),
        parts: held_parts(3),
    });
    let mut given = parts_attachments(3, true);
    given.insert(
        AttachmentId(7),
        bytes(serde_json::to_vec(&json!([{"value": {"t": "i64", "v": "2"}}])).unwrap()),
    );
    let Err(error) = try_run(None, Some(ranking), given) else {
        panic!("a ranking held with too few candidates is refused");
    };
    assert_eq!(error.kind, ErrorKind::Protocol);
}
