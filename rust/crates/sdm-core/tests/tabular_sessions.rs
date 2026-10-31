// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Session policy exercised through the core protocol without a language host.

use sdm_contract::data::AttachmentId;
use sdm_contract::direct::{DirectCallFacts, DirectInput, TabularColumn, TabularColumns};
use sdm_contract::facts::PlatformFacts;
use sdm_contract::ops::{Dtype, Operation, OperationResult, Stype};
use sdm_contract::run::{Endpoint, StartRun};
use sdm_contract::session::*;
use sdm_contract::{CONTRACT_VERSION, HostFacts, Scalar, Step, Unit};
use sdm_core::{Attachment, Attachments, CompositionRoot, ModelResult, Run, UnknownZone, Zones};
use serde::{Serialize, de::DeserializeOwned};
use serde_json::{Value, json};
use std::sync::Arc;

struct NoZones;
impl Zones for NoZones {
    fn zone(&self, _: &str) -> Result<jiff::tz::TimeZone, UnknownZone> {
        Err(UnknownZone)
    }
}
fn bytes(value: &impl Serialize) -> Attachment {
    Attachment::Bytes(serde_json::to_vec(value).unwrap())
}
fn run(
    operation: impl FnOnce(DirectInput) -> Operation,
    input: &impl Serialize,
    mut attached: Attachments,
) -> Run {
    attached.insert(AttachmentId(900), bytes(input));
    sdm_core::start(
        &CompositionRoot::new(Arc::new(sdm_data::Data::new())),
        StartRun {
            contract: CONTRACT_VERSION,
            facts: HostFacts {
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
            },
            operation: operation(DirectInput {
                input: AttachmentId(900),
            }),
        },
        attached,
        Arc::new(NoZones),
    )
    .unwrap()
}
fn take(run: &mut Run, id: AttachmentId) -> Vec<u8> {
    let Some(Attachment::Bytes(value)) = run.take_attachment(id) else {
        panic!("bytes attachment")
    };
    value
}
fn done<T: DeserializeOwned>(run: &mut Run) -> T {
    let Step::Done {
        result: OperationResult::DirectValue(value),
    } = run.next().unwrap()
    else {
        panic!("pure operation must finish without host work")
    };
    serde_json::from_slice(&take(run, value.value)).unwrap()
}
fn reply(run: &mut Run, endpoint: Endpoint, status: u16, response: Value) -> Vec<u8> {
    let Step::CallModel(call) = run.next().unwrap() else {
        panic!("model call")
    };
    assert_eq!(call.endpoint, endpoint);
    assert_eq!(call.content_type, "application/json");
    let payload = take(run, call.body);
    run.submit_model(
        call.id,
        ModelResult::Response {
            status,
            content_type: "application/json".into(),
            body: serde_json::to_vec(&response).unwrap(),
        },
    )
    .unwrap();
    payload
}
fn send(endpoint: Endpoint, payload: Vec<u8>, response: Value) -> Value {
    let mut call = run(
        |control| {
            Operation::SendTabularSession(SessionSendInput {
                input: control.input,
                payload: AttachmentId(71),
            })
        },
        &SendTabularSession {
            endpoint: endpoint.clone(),
            payload: AttachmentId(71),
        },
        Attachments::from([(AttachmentId(71), Attachment::Bytes(payload.clone()))]),
    );
    assert_eq!(reply(&mut call, endpoint, 200, response.clone()), payload);
    let result: SessionTransportReply = done(&mut call);
    assert_eq!(result.status, 200);
    let actual: Value = serde_json::from_slice(&take(&mut call, result.response)).unwrap();
    assert_eq!(actual, response);
    actual
}

#[test]
fn fit_score_twice_and_retry_close_with_remapped_attachments() {
    let bound = sdm_request::direct::bind(DirectCallFacts {
        model: "kumo-tabular".into(),
        outputs: Some(vec!["prediction".into()]),
        kind: "regression".into(),
        anchored: false,
        relationships: false,
        tables: 1,
        context_rows: 3,
    })
    .unwrap();
    let columns = TabularColumns {
        dtypes: vec![("id".into(), Dtype::Int64), ("y".into(), Dtype::Float64)],
        columns: vec![
            TabularColumn {
                name: "id".into(),
                dtype: Dtype::Int64,
                stype: Stype::Id,
            },
            TabularColumn {
                name: "y".into(),
                dtype: Dtype::Float64,
                stype: Stype::Numerical,
            },
        ],
        target: "y".into(),
    };
    let mut fit = run(
        |control| {
            Operation::PrepareTabularFit(SessionFitInput {
                input: control.input,
                context: AttachmentId(41),
                holdout: AttachmentId(42),
            })
        },
        &PrepareTabularFit {
            bound: bound.clone(),
            columns,
            primary_key: vec!["id".into()],
            question: None,
            context: AttachmentId(41),
            holdout: AttachmentId(42),
            request_id: "fit".into(),
            pinned_over_a_window: false,
        },
        Attachments::from([
            (
                AttachmentId(41),
                bytes(
                    &json!({"format":"arrays","columns":["id","y"],"rows":[[1,10.0],[2,20.0],[3,30.0]]}),
                ),
            ),
            (
                AttachmentId(42),
                bytes(&SessionKeys {
                    rows: vec![vec![SessionKeyPart::Numeric {
                        decimal: "2".into(),
                    }]],
                }),
            ),
        ]),
    );
    let prepared: PreparedTabularFit = done(&mut fit);
    assert_eq!((prepared.held_back, prepared.named_holdout_keys), (1, 1));
    let initial_keys: SessionKeys =
        serde_json::from_slice(&take(&mut fit, prepared.pinned_keys)).unwrap();
    assert_eq!(initial_keys.rows.len(), 2);
    let payload = take(&mut fit, prepared.payload);
    let mut sent: Value = serde_json::from_slice(&payload).unwrap();
    assert_eq!(
        sent["context"]["instance_table"]["rows"],
        json!([[1, 10.0], [3, 30.0]])
    );
    assert!(sent.get("predict").is_none());
    let created = send(
        Endpoint::CreateSession,
        payload,
        json!({"session_id":"session-1"}),
    );
    let state = TabularSessionState {
        version: TABULAR_SESSION_VERSION,
        session_id: created["session_id"].as_str().unwrap().into(),
        closed: false,
        bound,
        pinned_model: "kumo-tabular".into(),
        pinned_kind: "regression".into(),
        columns: prepared.columns,
        primary_key: prepared.primary_key,
        question: prepared.question,
        pinned_keys: AttachmentId(111),
        held_back: prepared.held_back,
        pinned_over_a_window: false,
    };
    // A transport may alter the sent document. Acceptance must pin its actual
    // context, rather than blindly retain the earlier holdout key attachment.
    sent["context"]["instance_table"]["rows"] = json!([[7, 70.0]]);
    let mut acceptance = run(
        |control| {
            Operation::AcceptTabularSession(SessionAcceptInput {
                input: control.input,
                observed: AttachmentId(81),
            })
        },
        &AcceptTabularSession {
            state,
            observed: AttachmentId(81),
        },
        Attachments::from([(AttachmentId(81), bytes(&sent))]),
    );
    let mut state: TabularSessionState = done(&mut acceptance);
    let keys = take(&mut acceptance, state.pinned_keys);
    let pinned: SessionKeys = serde_json::from_slice(&keys).unwrap();
    assert_eq!(
        pinned.rows,
        vec![vec![SessionKeyPart::Numeric {
            decimal: "7".into()
        }]]
    );
    state.pinned_keys = AttachmentId(131);
    for id in [7, 9] {
        let table = json!({"format":"arrays","columns":["id"],"rows":[[id]]});
        let mut count = run(
            |control| {
                Operation::CountTabularSessionOverlap(SessionOverlapInput {
                    input: control.input,
                    predict: AttachmentId(132),
                    pinned_keys: AttachmentId(131),
                })
            },
            &CountTabularSessionOverlap {
                state: state.clone(),
                predict: AttachmentId(132),
            },
            Attachments::from([
                (AttachmentId(131), Attachment::Bytes(keys.clone())),
                (AttachmentId(132), bytes(&table)),
            ]),
        );
        let overlap: TabularSessionOverlap = done(&mut count);
        assert_eq!(overlap.scoring_rows, 1);
        assert_eq!(overlap.overlap_rows, usize::from(id == 7));
        let mut score = run(
            |control| {
                Operation::PrepareTabularScore(SessionScoreInput {
                    input: control.input,
                    predict: AttachmentId(151),
                })
            },
            &PrepareTabularScore {
                state: state.clone(),
                predict: AttachmentId(151),
                request_id: format!("score-{id}"),
                inference_present: false,
            },
            Attachments::from([(AttachmentId(151), bytes(&table))]),
        );
        let prepared: PreparedTabularScore = done(&mut score);
        let payload = take(&mut score, prepared.payload);
        let body: Value = serde_json::from_slice(&payload).unwrap();
        assert!(body.get("context").is_none());
        assert!(body.get("schema").is_none());
        assert_eq!(body["predict"]["instance_table"], table);
        let layout: Value = serde_json::from_slice(&take(&mut score, prepared.layout)).unwrap();
        assert_eq!(layout["schema"], sent["schema"]);
        send(
            Endpoint::SessionPredict {
                session_id: state.session_id.clone(),
            },
            payload,
            json!({"predictions":[{"prediction":id}]}),
        );
    }
    let endpoint = Endpoint::DeleteSession {
        session_id: state.session_id.clone(),
    };
    let mut failed = run(Operation::CloseTabularSession, &state, Attachments::new());
    assert!(reply(&mut failed, endpoint.clone(), 503, json!({})).is_empty());
    assert!(failed.next().is_err());
    assert!(!state.closed);
    let mut close = run(Operation::CloseTabularSession, &state, Attachments::new());
    assert!(reply(&mut close, endpoint, 204, Value::Null).is_empty());
    let closed: TabularSessionState = done(&mut close);
    assert!(closed.closed);
    let mut repeated = run(Operation::CloseTabularSession, &closed, Attachments::new());
    assert_eq!(done::<TabularSessionState>(&mut repeated), closed);
}
