// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A request the core did not materialize, rebuilt for `predict` from what
//! the host holds of it (T3.26, owner decision 2026-10-23).
//!
//! An older client's request, or one this client gathered on its old route,
//! holds no request state a core wrote; what the old client predicts from
//! is the request's own contents, which such a request holds: each call's
//! payload as gathered, each question's text, and, for a ranking, the
//! candidates it asks after, as the caller gave them, once each. So
//! `predict` asks what the old client asks: the request's calls in order;
//! a forecast's steps numbered from 1, in order (`_per_step`); a ranking's
//! questions, each named by its candidate's place among those held and by
//! the JSON text of its value, written as `materialize` writes a candidate
//! it was given (`sdm_request::ranking::json_text`), so the result names
//! each as the core's own state would.

use sdm_contract::data::AttachmentId;
use sdm_contract::ops::{GivenCandidate, HeldPart, HeldRequest};
use sdm_contract::{CoreError, Sentence};
use sdm_request::ranking;

use crate::attached::take_json;
use crate::predict::{Asked, LaterCall};
use crate::run::{Attachment, Attachments, internal};

/// A held request no host holds: a message the core cannot read.
fn malformed(message: &str) -> CoreError {
    CoreError::protocol(Sentence::InvalidMessage, message)
}

/// The bytes of the payload attachment `id` names.
fn bytes(attachments: &mut Attachments, id: AttachmentId) -> Result<Vec<u8>, CoreError> {
    match attachments.remove(&id) {
        Some(Attachment::Bytes(payload)) => Ok(payload),
        _ => Err(internal("the payload is not the bytes named")),
    }
}

/// A later part's payload, question and the id its call carries.
fn part(
    attachments: &mut Attachments,
    held_part: HeldPart,
) -> Result<(Vec<u8>, String, String), CoreError> {
    let payload = bytes(attachments, held_part.payload)?;
    let query: String = take_json(attachments, held_part.query)?;
    Ok((payload, query, held_part.request_id))
}

/// What `held` asks of `predict`, its attachments taken from
/// `attachments`; the first call carries `request_id`.
pub(crate) fn rebuilt(
    held: HeldRequest,
    request_id: &str,
    attachments: &mut Attachments,
) -> Result<Asked, CoreError> {
    let payload = bytes(attachments, held.payload)?;
    let query: String = take_json(attachments, held.query)?;
    let asked_in = [
        held.batches.is_some(),
        held.forecast.is_some(),
        held.ranking.is_some(),
    ];
    if asked_in.iter().filter(|kind| **kind).count() > 1 {
        // A forecast or a ranking is one call per part, never batched, and
        // no question is both: a host that says so holds no request.
        return Err(malformed("a held request is asked in more than one way"));
    }
    let batches = match held.batches {
        None => None,
        Some(calls) => {
            let mut later = Vec::with_capacity(calls.len());
            for call in calls {
                later.push(LaterCall {
                    payload: bytes(attachments, call.payload)?,
                    request_id: call.request_id,
                });
            }
            Some(later)
        }
    };
    let forecast = match held.forecast {
        None => None,
        Some(steps) => {
            let mut later = Vec::with_capacity(steps.len());
            // Step 1 is the request's own; each later one numbered on.
            for (step, held_step) in (2u32..).zip(steps) {
                let (payload, query, request_id) = part(attachments, held_step)?;
                later.push((step, payload, query, request_id));
            }
            Some(later)
        }
    };
    let ranking = match held.ranking {
        None => None,
        Some(held_ranking) => {
            let given: Vec<GivenCandidate> = take_json(attachments, held_ranking.candidates)?;
            if given.len() != held_ranking.parts.len() + 1 {
                return Err(malformed(
                    "a held ranking names other than one candidate per question",
                ));
            }
            let named = given
                .iter()
                .map(|candidate| {
                    ranking::json_text(&ranking::read(candidate), crate::pql::float_text)
                        .ok_or_else(|| internal("a candidate held has no JSON text"))
                })
                .collect::<Result<Vec<String>, CoreError>>()?;
            let mut named = named.into_iter();
            let mut questions = Vec::with_capacity(given.len());
            let first = named
                .next()
                .ok_or_else(|| internal("a ranking held names no candidate"))?;
            questions.push((
                0,
                first,
                payload.clone(),
                query.clone(),
                request_id.to_owned(),
            ));
            for ((position, value), held_part) in (1u32..).zip(named).zip(held_ranking.parts) {
                let (payload, query, request_id) = part(attachments, held_part)?;
                questions.push((position, value, payload, query, request_id));
            }
            Some((questions, held_ranking.top_k))
        }
    };
    Ok(Asked {
        payload,
        query,
        forecast,
        ranking,
        batches,
    })
}
