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
//! one over gets `INVALID_RESPONSE` naming the status. A Databricks Model
//! Serving endpoint the host names (`Predict::model_endpoint`, T3.13) is
//! the core's to speak to: each call is wrapped in its envelope and weighed
//! against the caller's limit when it is sent, and its every answer, its
//! failures included, is read as the old client's serving transport reads
//! it ([`sdm_request::serving`]). A transport that
//! failed before an answer is `TRANSPORT_ERROR` with the kind and
//! retryability the host gave. No error carries a value of the request or
//! the answer (decision H17).

use sdm_contract::error::AnsweredParts;
use sdm_contract::ops::{ModelEndpoint, OperationResult, Predicted};
use sdm_contract::run::Endpoint;
use sdm_contract::{CoreError, CoreWarning, Detail, Scalar, Sentence};
use sdm_request::predict::{
    Notice, Part, absent, absent_notice, bad_response, body, by_step, call_bytes, call_starts,
    empty, layout, left_out, merge, nothing_to_score, numbered_from, predict_rows, preflight,
    prepare, refused, steps_unexplained, table,
};
use sdm_request::serving::{self, Failure};
use sdm_wire::json::{self, Value};
use sdm_wire::response::PredictionResponse;

use crate::run::{ArrowData, Asker, Attachment, ModelResult, Reply, internal};

/// A forecast's later step: its number, payload, query and request id.
pub(crate) type LaterStep = (u32, Vec<u8>, String, String);

/// A later call of a request asked in several (`main`'s multi-call
/// `predict`): its payload, and the request id the host drew for it.
#[derive(Clone)]
pub(crate) struct LaterCall {
    pub payload: Vec<u8>,
    pub request_id: String,
}

/// What a `predict` start asks, read from the request state `materialize`
/// answered or rebuilt from a request the host holds (`crate::held`): the
/// first call's payload and question, and a forecast's later steps, a
/// ranking's questions or a request's later calls, as [`Inputs`] holds
/// them.
pub(crate) struct Asked {
    pub payload: Vec<u8>,
    pub query: String,
    pub forecast: Option<Vec<LaterStep>>,
    pub ranking: Option<(Vec<crate::ranking::Question>, Option<u32>)>,
    pub batches: Option<Vec<LaterCall>>,
}

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
    /// A forecast's later steps, each its number, payload, query and the
    /// request id its call carries (T1.36 fix 24), in order; none for a
    /// forecast of one step, and `None` for any other request (T1.36 fix
    /// 21).
    pub forecast: Option<Vec<LaterStep>>,
    /// A ranking's questions, each its candidate's position, payload, query
    /// and the request id its call carries, the state's own first, and how
    /// many candidates a row keeps (T2.8); `None` for any other request.
    pub ranking: Option<(Vec<crate::ranking::Question>, Option<u32>)>,
    /// The later calls of a request asked in several, in order, each
    /// scoring the rows after the last (`RequestState::batches`); `None`
    /// for a request of one call.
    pub batches: Option<Vec<LaterCall>>,
    /// The serving endpoint the host's transport calls, whose envelope the
    /// core writes and whose answers it reads (T3.13); `None` for the
    /// client's own endpoint.
    pub endpoint: Option<ModelEndpoint>,
}

/// The JSON a request state's payload holds, or a refusal naming where in
/// it and what kind of fault, never the text.
pub(crate) fn payload(bytes: &[u8]) -> Result<Value, CoreError> {
    json::read(bytes).map_err(|e| {
        let count = |n: usize| Detail::Scalar(Scalar::UInt64(u64::try_from(n).unwrap_or(u64::MAX)));
        CoreError::new(
            Sentence::StateUnread,
            "the request state's payload is not JSON",
        )
        .with_field("field", Detail::text("state.payload"))
        .with_field("category", Detail::text(e.category))
        .with_field("line", count(e.line))
        .with_field("column", count(e.column))
    })
}

fn warn(asker: &Asker, notice: Notice) {
    asker.warn(CoreWarning::new(notice.sentence, notice.message).with_fields(notice.fields));
}

pub(crate) async fn run(asker: Asker, mut inputs: Inputs) -> Result<OperationResult, CoreError> {
    if let Some((questions, top_k)) = inputs.ranking.take() {
        return crate::ranking::predict(asker, inputs, questions, top_k).await;
    }
    let Some(later) = inputs.forecast.take() else {
        // The request's calls: the state's payload, then each later call's
        // (`main`'s multi-call `predict`), each with the id drawn for it.
        let mut calls = vec![LaterCall {
            payload: inputs.payload.clone(),
            request_id: inputs.request_id.clone(),
        }];
        calls.extend(inputs.batches.take().unwrap_or_default());
        let (merged, outputs, _) = one(&asker, &inputs, calls).await?;
        return finish(&asker, merged, outputs);
    };
    // A forecast: each step's regression predicted as the client predicts
    // each part (`_answer_in_parts`), in order, a refusal naming the step
    // it happened in (`forecast_step`) and carrying the steps answered
    // before it (`_carry_the_steps_already_paid_for`, `main`'s 1fc5a686),
    // and the answers put together as its `_by_step` puts them.
    let mut steps = vec![(
        1,
        inputs.payload.clone(),
        inputs.query.clone(),
        inputs.request_id.clone(),
    )];
    steps.extend(later);
    let steps_asked = u32::try_from(steps.len()).unwrap_or(u32::MAX);
    let mut answers = Vec::with_capacity(steps.len());
    let mut outputs = Vec::new();
    let mut rows = 0usize;
    for (step, payload, query, request_id) in steps {
        let asked = Inputs {
            query,
            forecast: None,
            ranking: None,
            batches: None,
            ..inputs.clone()
        };
        let done = one(
            &asker,
            &asked,
            vec![LaterCall {
                payload,
                request_id,
            }],
        )
        .await
        .map_err(|e| crate::materialize::in_step(e, step, steps_asked));
        let (answer, named, scored) = match done {
            Ok(done) => done,
            Err(error) => return Err(carry_steps(&asker, error, answers, &outputs, rows)),
        };
        if answers.is_empty() {
            outputs = named;
        }
        rows = rows.saturating_add(scored);
        answers.push((step, answer));
    }
    // The steps asked to be explained that came back with none, while
    // another did (`_warn_of_steps_that_carry_no_explanation`).
    if !inputs.default_outputs {
        let steps: Vec<(u32, &Part)> = answers.iter().map(|(s, p)| (*s, p)).collect();
        if let Some(notice) = steps_unexplained(&inputs.model, &inputs.outputs, &steps) {
            warn(&asker, notice);
        }
    }
    finish(&asker, by_step(answers)?, outputs)
}

/// `error`, carrying the answers a request asked in several calls had
/// received before it (`_carry_the_answers_already_paid_for`, `main`'s
/// 1fc5a686): put together as the request's answer would have been, laid
/// out as `Predicted.result` and attached, with the rows those calls score
/// and how many calls answered (`calls_answered`); the error keeps its
/// code. None answered, the error as it is.
fn carry_calls(
    asker: &Asker,
    error: CoreError,
    answers: Vec<Part>,
    outputs: &[String],
    rows: usize,
) -> CoreError {
    let calls = answers.len();
    if calls == 0 {
        return error;
    }
    let carried = merge(answers).and_then(|merged| carried(asker, &merged, outputs, rows));
    match carried {
        Ok(answered) => error
            .answering(answered)
            .with_field("calls_answered", count(calls)),
        Err(failed) => failed,
    }
}

/// `error`, carrying the steps a forecast had answered before it
/// (`_carry_the_steps_already_paid_for`): put together by step, as the
/// forecast's answer would have been, with how many steps answered
/// (`parts_answered`).
fn carry_steps(
    asker: &Asker,
    error: CoreError,
    answers: Vec<(u32, Part)>,
    outputs: &[String],
    rows: usize,
) -> CoreError {
    let steps = answers.len();
    if steps == 0 {
        return error;
    }
    let carried = by_step(answers).and_then(|merged| carried(asker, &merged, outputs, rows));
    match carried {
        Ok(answered) => error
            .answering(answered)
            .with_field("parts_answered", count(steps)),
        Err(failed) => failed,
    }
}

/// Answers received, laid out as a result and attached, with the rows
/// they score. What `Predicted` names beside its result travels on the
/// result's schema, as text: `model` and `request_id` as the endpoint
/// named them, `metadata` as the JSON of the answers' metadata, and
/// `outputs` as the JSON list of the outputs asked for.
fn carried(
    asker: &Asker,
    merged: &Part,
    outputs: &[String],
    rows: usize,
) -> Result<AnsweredParts, CoreError> {
    let laid = table(merged)?;
    // The result's schema's own map, with these keys; the order of a map
    // of text is no part of what the host reads.
    let mut named = laid.schema().metadata().clone();
    for (key, value) in [
        ("model", merged.model.clone()),
        ("request_id", merged.request_id.clone()),
        (
            "metadata",
            String::from_utf8(merged.metadata.to_bytes())
                .map_err(|_| internal("an answer's metadata is not text"))?,
        ),
        (
            "outputs",
            String::from_utf8(Value::texts(outputs.iter().map(String::as_str)).to_bytes())
                .map_err(|_| internal("the outputs are not text"))?,
        ),
        ("absent", absent_text(merged)?),
    ] {
        named.insert(key.to_owned(), value);
    }
    let schema = std::sync::Arc::new(laid.schema().as_ref().clone().with_metadata(named));
    let laid = laid
        .with_schema(std::sync::Arc::clone(&schema))
        .map_err(|_| internal("a result does not take its own schema"))?;
    Ok(AnsweredParts {
        result: asker.attach(Attachment::Arrow(ArrowData::new(schema, vec![laid])))?,
        rows: u64::try_from(rows).unwrap_or(u64::MAX),
    })
}

/// The JSON list of the outputs the answer left out ([`left_out`]), which
/// the result's schema carries under `absent` for a host to read a caller's
/// `absent(...)` off (T3.12 fix 1): the rule stays here.
fn absent_text(merged: &Part) -> Result<String, CoreError> {
    let named = left_out(merged);
    String::from_utf8(Value::texts(named.iter().map(String::as_str)).to_bytes())
        .map_err(|_| internal("the outputs left out are not text"))
}

fn count(n: usize) -> Detail {
    Detail::Scalar(Scalar::UInt64(u64::try_from(n).unwrap_or(u64::MAX)))
}

/// The result of a request's answer, attached.
fn finish(asker: &Asker, merged: Part, outputs: Vec<String>) -> Result<OperationResult, CoreError> {
    finish_direct(asker, merged, outputs, None)
}

pub(crate) fn finish_direct(
    asker: &Asker,
    merged: Part,
    outputs: Vec<String>,
    names_rows: Option<bool>,
) -> Result<OperationResult, CoreError> {
    let laid = table(&merged)?;
    let mut named = laid.schema().metadata().clone();
    named.insert("absent".to_owned(), absent_text(&merged)?);
    if let Some(names_rows) = names_rows {
        named.insert("names_its_rows".into(), names_rows.to_string());
    }
    let schema = std::sync::Arc::new(laid.schema().as_ref().clone().with_metadata(named));
    let laid = laid
        .with_schema(std::sync::Arc::clone(&schema))
        .map_err(|_| internal("a result does not take its own schema"))?;
    let result = asker.attach(Attachment::Arrow(ArrowData::new(schema, vec![laid])))?;
    let metadata = asker.attach(Attachment::Bytes(merged.metadata.to_bytes()))?;
    Ok(OperationResult::Predict(Predicted {
        result,
        model: merged.model,
        request_id: merged.request_id,
        metadata,
        outputs,
        unscored_candidates: vec![],
    }))
}

/// One request's answer, the outputs it asked for and the rows it scores:
/// the body of `predict` before T1.36 fix 21, of a request sent in one call
/// or, from `main`'s multi-call `predict`, in several (`calls`, in order).
/// As the client's `predict`, every call is checked and written before the
/// first is sent (`prepare`, `preflight`, then each call's body), each
/// call's rows numbered on from where the previous call's ended
/// (`_numbered_across_calls`), the answers merged as one; a call that fails
/// ends the request, carrying the answers received before it.
pub(crate) async fn one(
    asker: &Asker,
    inputs: &Inputs,
    calls: Vec<LaterCall>,
) -> Result<(Part, Vec<String>, usize), CoreError> {
    let mut payloads = Vec::with_capacity(calls.len());
    let mut ids = Vec::with_capacity(calls.len());
    for call in calls {
        payloads.push(payload(&call.payload)?);
        ids.push(call.request_id);
    }
    let outputs = prepare(
        &payloads,
        &inputs.model,
        &inputs.outputs,
        inputs.default_outputs,
    )?;
    let scored = payloads
        .iter()
        .map(|p| predict_rows(p).unwrap_or(0))
        .fold(0usize, usize::saturating_add);
    let merged = if payloads.iter().all(nothing_to_score) {
        empty(&inputs.model, &payloads[0])
    } else {
        for notice in preflight(&payloads, &outputs, &inputs.query, &inputs.model)? {
            warn(asker, notice);
        }
        // Every call written before the first is sent, as the client's
        // plan is listed whole (`_numbered_across_calls`), then each call's
        // rows counted.
        let mut bodies = Vec::with_capacity(payloads.len());
        for (payload, request_id) in payloads.iter().zip(&ids) {
            bodies.push(call_bytes(&body(
                payload,
                &inputs.model,
                request_id,
                &outputs,
                &inputs.inference_asked,
            )?)?);
        }
        let starts = call_starts(&payloads)?;
        let mut parts = Vec::with_capacity(payloads.len());
        for ((payload, bytes), start) in payloads.iter().zip(bodies).zip(starts) {
            let answered = match call(asker, payload, bytes, inputs.endpoint.as_ref()).await {
                Ok(part) => numbered_from(part, start),
                Err(error) => Err(error),
            };
            match answered {
                Ok(part) => parts.push(part),
                Err(error) => return Err(carry_calls(asker, error, parts, &outputs, start)),
            }
        }
        merge(parts)?
    };
    if !inputs.default_outputs {
        let left_out = absent(&merged, &outputs);
        if !left_out.is_empty() {
            warn(asker, absent_notice(&inputs.model, &left_out));
        }
    }
    Ok((merged, outputs, scored))
}

/// One payload's call, its body as written, sent and its answer laid out;
/// to a serving endpoint, in its envelope, its answer read out of its row.
pub(crate) async fn call(
    asker: &Asker,
    payload: &Value,
    bytes: Vec<u8>,
    endpoint: Option<&ModelEndpoint>,
) -> Result<Part, CoreError> {
    let (_, response) = call_response(asker, bytes, endpoint, false).await?;
    layout(payload, &response, crate::pql::float_text)
}

pub(crate) async fn call_response(
    asker: &Asker,
    bytes: Vec<u8>,
    endpoint: Option<&ModelEndpoint>,
    direct: bool,
) -> Result<(Value, PredictionResponse), CoreError> {
    let serving = endpoint.map(
        |ModelEndpoint::ServingEndpoint {
             name,
             max_request_bytes,
         }| { (name.as_str(), *max_request_bytes) },
    );
    let sent = match serving {
        Some((name, limit)) => serving::envelope(name, limit, &bytes)?,
        None => bytes,
    };
    let reply = asker
        .call_model(Endpoint::Predict, "application/json".to_owned(), sent)
        .await;
    let answer = match (reply, serving) {
        (Reply::Model(ModelResult::Response { status, body, .. }), Some((name, _))) => {
            if (200..300).contains(&status) {
                serving::answer(name, &body)?
            } else {
                // The HTTP failure's body, as a host without the SDK has it.
                return Err(serving::failure(
                    name,
                    &Failure {
                        status,
                        detail: Some(&body),
                        provider_code: None,
                        provider_error: None,
                    },
                ));
            }
        }
        (
            Reply::Model(ModelResult::Failed {
                status,
                detail,
                provider_code,
                provider_error,
            }),
            Some((name, _)),
        ) => {
            return Err(serving::failure(
                name,
                &Failure {
                    status,
                    detail: detail.as_deref(),
                    provider_code: provider_code.as_deref(),
                    provider_error: provider_error.as_deref(),
                },
            ));
        }
        (Reply::Model(ModelResult::Response { status, body, .. }), None) => {
            if !(200..300).contains(&status) {
                return Err(bad_response("status", "not_a_success")
                    .with_field("status", Detail::Scalar(Scalar::UInt16(status))));
            }
            // The answer is read whatever content type it names, as the
            // client's transport decodes every answer's body as JSON, a
            // `NaN` or `Infinity` token taken as the number Python's
            // reader takes it for, and a token past a double's range as an
            // infinity (owner decision of 2026-10-12, register P1B-4).
            // Text nested past the reader's depth is refused before
            // anything of it is built (`sdm_wire::json::MAX_DEPTH`,
            // approved by the owner, register P1B-22).
            json::read_answer(&body).map_err(|e| {
                let reason = match e.category {
                    "depth" => "too_deep",
                    _ => "not_json",
                };
                bad_response("response", reason).with_field("category", Detail::text(e.category))
            })?
        }
        // A failure with no body, from the client's own endpoint, is
        // refused as one with a body is, by its status.
        (Reply::Model(ModelResult::Failed { status, .. }), None) => {
            return Err(bad_response("status", "not_a_success")
                .with_field("status", Detail::Scalar(Scalar::UInt16(status))));
        }
        (Reply::Model(ModelResult::TransportError { kind, retryable }), _) => {
            return Err(CoreError::transport(kind, retryable));
        }
        (Reply::Fetch(_), _) => return Err(internal("a model call was answered as a fetch")),
        (Reply::Failed(error), _) => return Err(error),
    };
    let response = PredictionResponse::read(&answer).map_err(|error| {
        if direct && error.field == "prediction.row_index" && error.reason == "missing" {
            CoreError::new(
                Sentence::TabularRowMissing,
                "a tabular response lacks row attribution",
            )
        } else {
            refused(error)
        }
    })?;
    Ok((answer, response))
}
