// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `predict`: a materialized request sent to the model, and its answer.
//!
//! The run reads the request state a `materialize` run answered and the
//! host held since (decision H23), and runs the client's predict-side
//! rules in the reference's order (`sdm_request::predict`, T1.28b): the
//! model and outputs checked and the default outputs chosen (`prepare`); a
//! request with no row to score answered without a call (`empty`); the
//! adapter's checks of what the call sends (`preflight`, its notice of a
//! constant context given as a warning); the call written (`body`) and
//! sent (one `CallModel` step); the answer read as the client's bindings
//! read it ([`sdm_wire::response`]), a non-finite number taken as the
//! reference's reader takes it; laid out (`layout`), the parts merged
//! (`merge`) and the outputs the caller named that no answer carries given
//! as a notice (`absent`); the result an Arrow table of one row per answer
//! ordered by `row_index` (`table`), the endpoint's `metadata` as JSON, and
//! the model and call id the endpoint named.
//!
//! An answer with a status other than 2xx is the endpoint's refusal, which
//! the host's transport reports as the reference's does; a host that hands
//! one over gets `INVALID_RESPONSE` naming the status. A transport that
//! failed before an answer is `TRANSPORT_ERROR` with the kind and
//! retryability the host gave. No error carries a value of the request or
//! the answer (decision H17).

use sdm_contract::ops::{OperationResult, Predicted};
use sdm_contract::run::Endpoint;
use sdm_contract::{CoreError, CoreWarning, Detail, ErrorCode, ErrorKind, Scalar};
use sdm_request::predict::{
    Notice, Part, absent, absent_notice, bad_response, body, by_step, call_bytes, empty, layout,
    merge, nothing_to_score, preflight, prepare, refused, table,
};
use sdm_wire::json::{self, Value};
use sdm_wire::response::PredictionResponse;

use crate::run::{ArrowData, Asker, Attachment, ModelResult, Reply, internal};

/// Predict's notices, each with the variants it gives (none: one
/// sentence), which a host words (decision E59(d)): a regression's constant
/// context, by how the target is named, and an output the caller named
/// that the answer left out.
pub const WARNINGS: &[(&str, &[&str])] = &[
    (
        sdm_request::predict::CONTEXT_CONSTANT,
        &["column", "predicted", "question", "answer"],
    ),
    (sdm_request::predict::OUTPUTS_ABSENT, &[]),
];

/// What a `predict` start gives its body.
#[derive(Clone)]
pub(crate) struct Inputs {
    /// The request state's payload, as the host attached it.
    pub payload: Vec<u8>,
    /// The question's text, as the state holds it.
    pub query: String,
    pub model: String,
    pub request_id: String,
    /// The outputs the caller asked for, by the wire's names.
    pub outputs: Vec<String>,
    /// The caller named no outputs.
    pub default_outputs: bool,
    /// The inference options the caller set at `materialize`.
    pub inference_asked: Vec<String>,
    /// A forecast's later steps, each its number, payload and query, in
    /// order; none for a forecast of one step, and `None` for any other
    /// request (T1.36 fix 21).
    pub forecast: Option<Vec<(u32, Vec<u8>, String)>>,
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

fn warn(asker: &Asker, notice: Notice) {
    asker.warn(CoreWarning {
        code: notice.code.to_owned(),
        message: notice.message.to_owned(),
        fields: notice
            .fields
            .into_iter()
            .map(|(name, value)| (name.to_owned(), value))
            .collect(),
    });
}

pub(crate) async fn run(asker: Asker, mut inputs: Inputs) -> Result<OperationResult, CoreError> {
    let Some(later) = inputs.forecast.take() else {
        let (merged, outputs) = one(&asker, &inputs).await?;
        return finish(&asker, merged, outputs);
    };
    // A forecast: each step's regression predicted as the client predicts
    // each part (`_answer_in_parts`), in order, a refusal naming the step
    // it happened in (`forecast_step`), and the answers put together as
    // its `_by_step` puts them.
    let mut steps = vec![(1, inputs.payload.clone(), inputs.query.clone())];
    steps.extend(later);
    let mut answers = Vec::with_capacity(steps.len());
    let mut outputs = Vec::new();
    for (step, payload, query) in steps {
        let asked = Inputs {
            payload,
            query,
            forecast: None,
            ..inputs.clone()
        };
        let (answer, named) = one(&asker, &asked)
            .await
            .map_err(|e| e.with_field("forecast_step", Detail::Scalar(Scalar::UInt32(step))))?;
        if answers.is_empty() {
            outputs = named;
        }
        answers.push((step, answer));
    }
    finish(&asker, by_step(answers)?, outputs)
}

/// The result of a request's answer, attached.
fn finish(asker: &Asker, merged: Part, outputs: Vec<String>) -> Result<OperationResult, CoreError> {
    let laid = table(&merged)?;
    let result = asker.attach(Attachment::Arrow(ArrowData::new(laid.schema(), vec![laid])))?;
    let metadata = asker.attach(Attachment::Bytes(merged.metadata.to_bytes()))?;
    Ok(OperationResult::Predict(Predicted {
        result,
        model: merged.model,
        request_id: merged.request_id,
        metadata,
        outputs,
    }))
}

/// One request's answer, and the outputs it asked for: the body of
/// `predict` before T1.36 fix 21.
async fn one(asker: &Asker, inputs: &Inputs) -> Result<(Part, Vec<String>), CoreError> {
    let payload = payload(&inputs.payload)?;
    let outputs = prepare(
        &payload,
        &inputs.model,
        &inputs.outputs,
        inputs.default_outputs,
    )?;
    // The parts of the request: the request state holds one payload, which
    // `materialize` writes; the merge holds for several (its own tests).
    let payloads = vec![payload];
    let merged = if payloads.iter().all(nothing_to_score) {
        empty(&inputs.model, &payloads[0])
    } else {
        for notice in preflight(&payloads, &outputs, &inputs.query, &inputs.model)? {
            warn(asker, notice);
        }
        let mut parts = Vec::with_capacity(payloads.len());
        for payload in &payloads {
            parts.push(call(asker, inputs, payload, &outputs).await?);
        }
        merge(parts)?
    };
    if !inputs.default_outputs {
        let left_out = absent(&merged, &outputs);
        if !left_out.is_empty() {
            warn(asker, absent_notice(&inputs.model, &left_out));
        }
    }
    Ok((merged, outputs))
}

/// One payload's call, sent and its answer laid out.
async fn call(
    asker: &Asker,
    inputs: &Inputs,
    payload: &Value,
    outputs: &[String],
) -> Result<Part, CoreError> {
    let call = body(
        payload,
        &inputs.model,
        &inputs.request_id,
        outputs,
        &inputs.inference_asked,
    )?;
    let reply = asker
        .call_model(
            Endpoint::Predict,
            "application/json".to_owned(),
            call_bytes(&call)?,
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
    // transport decodes every answer's body as JSON, a `NaN` or `Infinity`
    // token taken as the number Python's reader takes it for, and a token
    // past a double's range as an infinity (owner decision of 2026-10-12,
    // register P1B-4). Text nested past the reader's depth is refused
    // before anything of it is built (`sdm_wire::json::MAX_DEPTH`, approved
    // by the owner, register P1B-22).
    let read = json::read_answer(&answer).map_err(|e| {
        let reason = match e.category {
            "depth" => "too_deep",
            _ => "not_json",
        };
        bad_response("response", reason).with_field("category", Detail::text(e.category))
    })?;
    let response = PredictionResponse::read(&read).map_err(refused)?;
    layout(payload, &response, crate::pql::float_text)
}
