// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `predict`: a materialized request sent to the model, and its answer.
//!
//! The run reads the request state a `materialize` run answered and the
//! host held since (decision H23), makes the call the reference's client
//! makes of it ([`sdm_request::predict::body`]), asks the host to send it
//! (one `CallModel` step), reads the answer as the client's bindings read
//! it ([`sdm_wire::response`]) and lays it out
//! ([`sdm_request::predict::layout`]): an Arrow table of one row per answer,
//! the endpoint's `metadata` as JSON, and the model and call id the
//! endpoint named.
//!
//! An answer with a status other than 2xx is the endpoint's refusal, which
//! the host's transport reports as the reference's does; a host that hands
//! one over gets `INVALID_RESPONSE` naming the status. A transport that
//! failed before an answer is `TRANSPORT_ERROR` with the kind and
//! retryability the host gave. No error carries a value of the request or
//! the answer (decision H17).

use sdm_contract::ops::{OperationResult, Predicted};
use sdm_contract::run::Endpoint;
use sdm_contract::{CoreError, Detail, ErrorCode, ErrorKind, Scalar};
use sdm_request::predict::{bad_response, body, check_outputs, layout, refused};
use sdm_wire::json::{self, Value};
use sdm_wire::response::PredictionResponse;

use crate::run::{ArrowData, Asker, Attachment, ModelResult, Reply, internal};

/// What a `predict` start gives its body.
pub(crate) struct Inputs {
    /// The request state's payload, as the host attached it.
    pub payload: Vec<u8>,
    pub model: String,
    pub request_id: String,
    /// The outputs asked for, by the wire's names.
    pub outputs: Vec<String>,
    /// The inference options the caller set at `materialize`.
    pub inference_asked: Vec<String>,
}

/// The JSON a request state's payload holds, or a refusal naming where in
/// it and what kind of fault, never the text.
fn payload(bytes: &[u8]) -> Result<Value, CoreError> {
    json::read(bytes).map_err(|e| {
        let count = |n: usize| Detail::Scalar(Scalar::UInt64(u64::try_from(n).unwrap_or(u64::MAX)));
        CoreError::new(
            ErrorKind::InvalidRequest,
            ErrorCode::InvalidRequest,
            "the request state's payload is not JSON",
        )
        .with_field("field", Detail::text("state.payload"))
        .with_field("category", Detail::text(e.category))
        .with_field("line", count(e.line))
        .with_field("column", count(e.column))
    })
}

pub(crate) async fn run(asker: Asker, inputs: Inputs) -> Result<OperationResult, CoreError> {
    // The outputs this core asks for: the task's default alone, any other
    // list refused by name (decision E61), once the state's task is read.
    let payload = payload(&inputs.payload)?;
    check_outputs(&payload, &inputs.outputs)?;
    let call = body(
        &payload,
        &inputs.model,
        &inputs.request_id,
        &inputs.outputs,
        &inputs.inference_asked,
    )?;
    let reply = asker
        .call_model(
            Endpoint::Predict,
            "application/json".to_owned(),
            call.to_bytes(),
        )
        .await;
    let answer = match reply {
        Reply::Model(ModelResult::Response { status, body, .. }) => {
            if !(200..300).contains(&status) {
                return Err(bad_response("status", "not_a_success")
                    .with_field("status", Detail::Scalar(Scalar::UInt16(status))));
            }
            body
        }
        Reply::Model(ModelResult::TransportError { kind, retryable }) => {
            return Err(CoreError::transport(kind, retryable));
        }
        Reply::Fetch(_) => return Err(internal("a model call was answered as a fetch")),
    };
    // The answer is read whatever content type it names, as the client's
    // transport decodes every answer's body as JSON.
    // Text nested past the reader's depth is refused before anything of it
    // is built (`sdm_wire::json::MAX_DEPTH`).
    let read = json::read(&answer).map_err(|e| {
        let reason = match e.category {
            "depth" => "too_deep",
            "not_finite" => "not_finite",
            _ => "not_json",
        };
        bad_response("response", reason).with_field("category", Detail::text(e.category))
    })?;
    // One policy for a number that is not finite, wherever in the answer it
    // is, and however it was written: a `NaN` or `Infinity` token (above), or
    // a token past a double's range, which reads as an infinity, is refused
    // (register P1B-4). The Python host writes back any non-finite number
    // its reader made as such a token, so both hosts refuse alike.
    if !read.all_finite() {
        return Err(bad_response("response", "not_finite"));
    }
    let response = PredictionResponse::read(&read).map_err(refused)?;
    let table = layout(&payload, &response)?;
    let result = asker.attach(Attachment::Arrow(ArrowData::new(
        table.schema(),
        vec![table],
    )))?;
    let metadata = asker.attach(Attachment::Bytes(response.metadata.to_bytes()))?;
    Ok(OperationResult::Predict(Predicted {
        result,
        model: response.model,
        request_id: response.id,
        metadata,
    }))
}
