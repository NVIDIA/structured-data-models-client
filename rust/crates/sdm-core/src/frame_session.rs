// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Frame session composition over host-retained metadata, never a registry.
use crate::{
    attached::take_json,
    run::{Asker, Attachment, Attachments, ModelResult, Reply, internal},
};
use sdm_contract::{
    CoreError, CoreWarning, Detail, Scalar, Sentence,
    data::AttachmentId,
    frame_session::{FrameSessionInput, FrameSessionOperation, FrameSessionState},
    ops::OperationResult,
    run::Endpoint,
};
use sdm_request::{
    frame_session::{holdout, invalid, row_keys, subset},
    predict,
    tabular::session as keys,
};
use sdm_wire::json::Value;

fn take(attached: &mut Attachments, id: AttachmentId) -> Result<Value, CoreError> {
    let Some(Attachment::Bytes(bytes)) = attached.remove(&id) else {
        return Err(internal("frame session attachment absent"));
    };
    crate::predict::payload(&bytes)
}
fn attach(asker: &Asker, value: &Value) -> Result<AttachmentId, CoreError> {
    asker.attach(Attachment::Bytes(value.to_bytes()))
}
fn validate(state: &FrameSessionState) -> Result<(), CoreError> {
    if state.version != 1 || state.model != predict::RELATIONAL || state.session_id.is_empty() {
        return Err(invalid("invalid frame session state"));
    }
    Ok(())
}
async fn call(asker: &Asker, endpoint: Endpoint, body: Value) -> Result<Value, CoreError> {
    match asker
        .call_model(
            endpoint,
            "application/json".into(),
            predict::call_bytes(&body)?,
        )
        .await
    {
        Reply::Model(ModelResult::Response { status, body, .. })
            if (200..300).contains(&status) =>
        {
            if body.is_empty() {
                Ok(Value::Object(vec![]))
            } else {
                crate::predict::payload(&body)
            }
        }
        Reply::Model(ModelResult::TransportError { kind, retryable }) => {
            Err(CoreError::transport(kind, retryable))
        }
        Reply::Model(ModelResult::Response { status, .. } | ModelResult::Failed { status, .. }) => {
            Err(predict::bad_response("status", "not_a_success")
                .with_field("status", Detail::Scalar(Scalar::UInt16(status))))
        }
        _ => Err(internal("session transport returned another reply kind")),
    }
}
pub(crate) async fn run(
    asker: Asker,
    operation: FrameSessionOperation,
    mut attached: Attachments,
) -> Result<OperationResult, CoreError> {
    let (input_id, payload_ids, pinned_id, action) = match operation {
        FrameSessionOperation::Fit { input, payloads } => (input, payloads, None, "fit"),
        FrameSessionOperation::Predict {
            input,
            payloads,
            pinned,
        } => (input, payloads, Some(pinned), "predict"),
        FrameSessionOperation::Close { input } => (input, vec![], None, "close"),
    };
    let input: FrameSessionInput = take_json(&mut attached, input_id)?;
    if !matches!(
        (&input, action),
        (FrameSessionInput::Fit { .. }, "fit")
            | (FrameSessionInput::Predict { .. }, "predict")
            | (FrameSessionInput::Close { .. }, "close")
    ) {
        return Err(internal("frame session action differs from its input"));
    }
    let payloads = payload_ids
        .iter()
        .map(|id| take(&mut attached, *id))
        .collect::<Result<Vec<_>, _>>()?;
    match input {
        FrameSessionInput::Fit {
            model,
            outputs,
            question,
            holdout: held,
            windowed,
            request_id,
            inference_asked,
        } => {
            if payloads.len() != 1 {
                return Err(invalid(
                    "a session pins one context; use predict() for a request gathered as several batches",
                ));
            }
            if model != predict::RELATIONAL {
                return Err(invalid("frame sessions require kumo-relational"));
            }
            let default_outputs = outputs.is_none();
            let fields = predict::prepare(
                &payloads,
                &model,
                outputs.as_deref().unwrap_or(&[]),
                default_outputs,
            )?;
            let query = question
                .as_ref()
                .and_then(|q| q.query.as_deref())
                .unwrap_or("");
            let envelope =
                predict::body(&payloads[0], &model, &request_id, &fields, &inference_asked)?;
            let (envelope, removed, named) = holdout(&envelope, &held)?;
            for notice in predict::preflight_halves(
                std::slice::from_ref(&envelope),
                &fields,
                query,
                &model,
                true,
                false,
            )? {
                asker.warn(
                    CoreWarning::new(notice.sentence, notice.message).with_fields(notice.fields),
                );
            }
            let body = subset(
                &envelope,
                &["model", "task", "schema", "context", "metadata"],
            );
            let pinned = subset(&body, &["model", "task", "schema"]);
            let state_keys = row_keys(&body, "context", windowed)?;
            let answer = call(&asker, Endpoint::CreateSession, body).await?;
            let id = answer
                .get("session_id")
                .and_then(Value::as_text)
                .filter(|s| !s.is_empty())
                .or_else(|| {
                    answer
                        .get("id")
                        .and_then(Value::as_text)
                        .filter(|s| !s.is_empty())
                })
                .ok_or_else(|| predict::bad_response("session_id", "missing"))?;
            let state = FrameSessionState {
                version: 1,
                session_id: id.into(),
                closed: false,
                model,
                outputs: fields,
                default_outputs,
                question,
                keys: state_keys,
                windowed,
            };
            let pinned = attach(&asker, &pinned)?;
            crate::direct::value(
                &asker,
                &serde_json::json!({"state":state,"pinned":pinned,"held_back":removed,"named_holdout_keys":named}),
            )
        }
        FrameSessionInput::Predict {
            state,
            question,
            request_ids,
            inference_asked,
        } => {
            validate(&state)?;
            if state.closed {
                return Err(invalid("this fitted model is closed"));
            }
            if keys::check_question(state.question.as_ref(), question.as_ref()).is_some() {
                return Err(invalid(
                    "this session pinned a different question; fit again to ask something else",
                ));
            }
            if payloads.is_empty() || payloads.len() != request_ids.len() {
                return Err(invalid("session scoring needs one request id per payload"));
            }
            let pinned = take(
                &mut attached,
                pinned_id.ok_or_else(|| internal("pinned metadata absent"))?,
            )?;
            if !payloads.iter().all(predict::nothing_to_score) {
                for notice in predict::preflight_halves(
                    &payloads,
                    &state.outputs,
                    "",
                    &state.model,
                    false,
                    true,
                )? {
                    asker.warn(
                        CoreWarning::new(notice.sentence, notice.message)
                            .with_fields(notice.fields),
                    );
                }
            }
            let mut plans = Vec::new();
            let mut shown = 0;
            let mut total = 0;
            // Prepare every scoring call before the first network effect.
            for (payload, id) in payloads.iter().zip(&request_ids) {
                let mut scoring = pinned.clone();
                for name in ["predict", "output", "inference", "metadata"] {
                    if let Some(value) = payload.get(name) {
                        scoring.set(name, value.clone());
                    }
                }
                let envelope =
                    predict::body(&scoring, &state.model, id, &state.outputs, &inference_asked)?;
                let body = subset(&envelope, &["predict", "output", "inference", "metadata"]);
                let mut layout = pinned.clone();
                if let Value::Object(fields) = &body {
                    for (key, value) in fields {
                        layout.set(key, value.clone());
                    }
                }
                predict::prepare(
                    std::slice::from_ref(&layout),
                    &state.model,
                    &state.outputs,
                    false,
                )?;
                let scored = row_keys(&layout, "predict", state.windowed)?;
                shown += keys::overlap_count(&state.keys.rows, &scored.rows)?;
                total += scored.rows.len();
                plans.push((body, layout));
            }
            if shown != 0 {
                asker.warn(
                    CoreWarning::new(
                        Sentence::DirectOverlap,
                        "scoring rows carry answers in pinned context",
                    )
                    .with_field("shown", Detail::Scalar(Scalar::UInt64(shown as u64)))
                    .with_field("rows", Detail::Scalar(Scalar::UInt64(total as u64)))
                    .with_field("target", Detail::text("the pinned target")),
                );
            }
            let mut parts = Vec::new();
            let mut start = 0;
            for (body, layout) in plans {
                let answered: Result<_, CoreError> = async {
                    let part = if predict::nothing_to_score(&layout) {
                        predict::empty(&state.model, &layout)
                    } else {
                        let answer = call(
                            &asker,
                            Endpoint::SessionPredict {
                                session_id: state.session_id.clone(),
                            },
                            body,
                        )
                        .await?;
                        let response = sdm_wire::response::PredictionResponse::read(&answer)
                            .map_err(predict::refused)?;
                        predict::layout(&layout, &response, crate::pql::float_text)?
                    };
                    predict::numbered_from(part, start)
                }
                .await;
                match answered {
                    Ok(part) => parts.push(part),
                    Err(error) => {
                        return Err(crate::predict::carry_calls(
                            &asker,
                            error,
                            parts,
                            &state.outputs,
                            start,
                        ));
                    }
                }
                start += predict::predict_rows(&layout).unwrap_or(0);
            }
            let merged = predict::merge(parts)?;
            if !state.default_outputs {
                let absent = predict::absent(&merged, &state.outputs);
                if !absent.is_empty() {
                    let n = predict::absent_notice(&state.model, &absent);
                    asker.warn(CoreWarning::new(n.sentence, n.message).with_fields(n.fields));
                }
            }
            crate::predict::finish(&asker, merged, state.outputs)
        }
        FrameSessionInput::Close { mut state } => {
            validate(&state)?;
            if !state.closed {
                call(
                    &asker,
                    Endpoint::DeleteSession {
                        session_id: state.session_id.clone(),
                    },
                    Value::Object(vec![]),
                )
                .await?;
                state.closed = true;
                state.keys.rows.clear();
            }
            crate::direct::value(&asker, &state)
        }
    }
}
