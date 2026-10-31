// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Session operations over values retained by the host between calls.

use sdm_contract::direct::DirectInput;
use sdm_contract::ops::OperationResult;
use sdm_contract::run::Endpoint;
use sdm_contract::session::{
    AcceptTabularSession, CheckTabularSessionQuestion, CountTabularSessionOverlap,
    PrepareTabularFit, PrepareTabularScore, PreparedTabularFit, PreparedTabularScore,
    SendTabularSession, SessionIdObservation, SessionKeys, SessionTransportReply,
    TabularSessionOverlap, TabularSessionState,
};
use sdm_contract::{CoreError, Detail, Scalar};
use sdm_request::tabular::session as policy;

use crate::attached::take_json;
use crate::run::{Asker, Attachment, Attachments, ModelResult, Reply, internal};
use sdm_contract::data::AttachmentId;
use sdm_wire::json::Value;

fn take_value(attached: &mut Attachments, id: AttachmentId) -> Result<Value, CoreError> {
    let Some(Attachment::Bytes(bytes)) = attached.remove(&id) else {
        return Err(CoreError::protocol(
            sdm_contract::Sentence::InvalidMessage,
            "session value does not name a bytes attachment",
        ));
    };
    crate::predict::payload(&bytes)
}

fn attach_json<T: serde::Serialize>(asker: &Asker, value: &T) -> Result<AttachmentId, CoreError> {
    asker.attach(Attachment::Bytes(
        serde_json::to_vec(value).map_err(|_| internal("session value does not serialize"))?,
    ))
}

fn object(fields: impl IntoIterator<Item = (&'static str, Value)>) -> Value {
    Value::Object(
        fields
            .into_iter()
            .map(|(key, value)| (key.into(), value))
            .collect(),
    )
}

fn metadata(request_id: String) -> Value {
    object([("request_id", Value::Text(request_id))])
}

pub(crate) fn prepare_fit(
    asker: &Asker,
    input: DirectInput,
    mut attached: Attachments,
) -> Result<OperationResult, CoreError> {
    let request: PrepareTabularFit = take_json(&mut attached, input.input)?;
    let mut payload =
        policy::pinned_metadata(&request.bound, &request.columns, &request.primary_key)?;
    let context = take_value(&mut attached, request.context)?;
    let keys = policy::encoded_keys(&context, &request.primary_key)?;
    let holdout: SessionKeys = take_json(&mut attached, request.holdout)?;
    let selected = policy::filter_holdout(&keys.rows, &holdout.rows)?;
    let surviving = SessionKeys {
        rows: selected
            .retained_rows
            .iter()
            .map(|&row| keys.rows[row].clone())
            .collect(),
    };
    let context = policy::select_encoded_rows(context, &selected.retained_rows)?;
    payload.set("context", object([("instance_table", context)]));
    payload.set("metadata", metadata(request.request_id));
    let prepared = PreparedTabularFit {
        bound: request.bound,
        columns: request.columns,
        primary_key: request.primary_key,
        question: request.question,
        pinned_keys: attach_json(asker, &surviving)?,
        held_back: selected.removed_rows,
        pinned_over_a_window: request.pinned_over_a_window,
        payload: asker.attach(Attachment::Bytes(payload.to_bytes()))?,
        named_holdout_keys: selected.named_keys,
    };
    crate::direct::value(asker, &prepared)
}

pub(crate) fn accept(
    asker: &Asker,
    input: DirectInput,
    mut attached: Attachments,
) -> Result<OperationResult, CoreError> {
    let request: AcceptTabularSession = take_json(&mut attached, input.input)?;
    let mut state = request.state;
    let observed = take_value(&mut attached, request.observed)?;
    let keys = policy::observe_pinned(&mut state, &observed)?;
    state.pinned_keys = attach_json(asker, &keys)?;
    crate::direct::value(asker, &state)
}

pub(crate) fn check_question(
    asker: &Asker,
    input: DirectInput,
    mut attached: Attachments,
) -> Result<OperationResult, CoreError> {
    let request: CheckTabularSessionQuestion = take_json(&mut attached, input.input)?;
    policy::validate_state(&request.state)?;
    crate::direct::value(
        asker,
        &policy::check_question(request.state.question.as_ref(), request.question.as_ref()),
    )
}

pub(crate) fn check_reply(
    asker: &Asker,
    input: DirectInput,
    mut attached: Attachments,
) -> Result<OperationResult, CoreError> {
    let observed: SessionIdObservation = take_json(&mut attached, input.input)?;
    crate::direct::value(asker, &policy::select_session_id(&observed))
}

/// Observe leakage from only the reached scoring key columns, before feature
/// encoding or inference options can fail or perform native work.
pub(crate) fn count_overlap(
    asker: &Asker,
    input: DirectInput,
    mut attached: Attachments,
) -> Result<OperationResult, CoreError> {
    let request: CountTabularSessionOverlap = take_json(&mut attached, input.input)?;
    policy::validate_state(&request.state)?;
    let table = take_value(&mut attached, request.predict)?;
    let scoring = policy::encoded_keys(&table, &request.state.primary_key)?;
    let pinned: SessionKeys = take_json(&mut attached, request.state.pinned_keys)?;
    let overlap = TabularSessionOverlap {
        overlap_rows: policy::overlap_count(&pinned.rows, &scoring.rows)?,
        scoring_rows: scoring.rows.len(),
    };
    crate::direct::value(asker, &overlap)
}

pub(crate) fn prepare_score(
    asker: &Asker,
    input: DirectInput,
    mut attached: Attachments,
) -> Result<OperationResult, CoreError> {
    let request: PrepareTabularScore = take_json(&mut attached, input.input)?;
    policy::validate_state(&request.state)?;
    if request.state.closed {
        return Err(CoreError::protocol(
            sdm_contract::Sentence::InvalidMessage,
            "cannot prepare a scoring call for a closed session",
        ));
    }
    let table = take_value(&mut attached, request.predict)?;
    let mut payload = object([
        ("predict", object([("instance_table", table)])),
        (
            "output",
            object([(
                "fields",
                Value::Array(
                    request
                        .state
                        .bound
                        .outputs
                        .iter()
                        .map(|field| Value::Text(field.clone()))
                        .collect(),
                ),
            )]),
        ),
    ]);
    if request.inference_present {
        payload.set("inference", Value::Null);
    }
    payload.set("metadata", metadata(request.request_id));
    let mut layout = policy::state_metadata(&request.state)?;
    if let Value::Object(fields) = &payload {
        for (name, value) in fields {
            layout.set(name, value.clone());
        }
    }
    let prepared = PreparedTabularScore {
        bound: request.state.bound,
        payload: asker.attach(Attachment::Bytes(payload.to_bytes()))?,
        layout: asker.attach(Attachment::Bytes(layout.to_bytes()))?,
    };
    crate::direct::value(asker, &prepared)
}

pub(crate) async fn send(
    asker: &Asker,
    input: DirectInput,
    mut attached: Attachments,
) -> Result<OperationResult, CoreError> {
    let request: SendTabularSession = take_json(&mut attached, input.input)?;
    match &request.endpoint {
        Endpoint::CreateSession => {}
        Endpoint::SessionPredict { session_id } if !session_id.is_empty() => {}
        _ => {
            return Err(CoreError::protocol(
                sdm_contract::Sentence::InvalidMessage,
                "session send names an unsupported endpoint or empty session id",
            ));
        }
    }
    let Some(Attachment::Bytes(body)) = attached.remove(&request.payload) else {
        return Err(CoreError::protocol(
            sdm_contract::Sentence::InvalidMessage,
            "session send has no body bytes",
        ));
    };
    let reply = asker
        .call_model(request.endpoint, "application/json".into(), body)
        .await;
    let (status, body) = match reply {
        Reply::Model(ModelResult::Response { status, body, .. })
            if (200..300).contains(&status) =>
        {
            (status, body)
        }
        Reply::Model(ModelResult::Response { status, .. } | ModelResult::Failed { status, .. }) => {
            return Err(
                sdm_request::predict::bad_response("status", "not_a_success")
                    .with_field("status", Detail::Scalar(Scalar::UInt16(status))),
            );
        }
        Reply::Model(ModelResult::TransportError { kind, retryable }) => {
            return Err(CoreError::transport(kind, retryable));
        }
        Reply::Fetch(_) => return Err(internal("session call was answered as a fetch")),
    };
    let response = asker.attach(Attachment::Bytes(body))?;
    crate::direct::value(asker, &SessionTransportReply { status, response })
}

/// A deletion has no response document. Only a successful transport outcome
/// commits the replacement state; a failure leaves the host's input untouched.
pub(crate) async fn close(
    asker: &Asker,
    input: DirectInput,
    mut attached: Attachments,
) -> Result<OperationResult, CoreError> {
    let state: TabularSessionState = take_json(&mut attached, input.input)?;
    policy::validate_state(&state)?;
    if state.closed {
        return crate::direct::value(asker, &state);
    }
    let reply = asker
        .call_model(
            Endpoint::DeleteSession {
                session_id: state.session_id.clone(),
            },
            "application/json".to_owned(),
            Vec::new(),
        )
        .await;
    match reply {
        Reply::Model(ModelResult::Response { status, .. }) if (200..300).contains(&status) => {}
        Reply::Model(ModelResult::Response { status, .. } | ModelResult::Failed { status, .. }) => {
            return Err(
                sdm_request::predict::bad_response("status", "not_a_success")
                    .with_field("status", Detail::Scalar(Scalar::UInt16(status))),
            );
        }
        Reply::Model(ModelResult::TransportError { kind, retryable }) => {
            return Err(CoreError::transport(kind, retryable));
        }
        Reply::Fetch(_) => return Err(internal("session deletion was answered as a fetch")),
    }
    crate::direct::value(asker, &policy::close_after_success(&state, true)?)
}
