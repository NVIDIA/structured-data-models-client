// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The control-message codecs and the control cap.
//!
//! A control message is the contract's JSON, at most [`CONTROL_CAP`] bytes
//! in either direction; anything larger travels as an attachment (decision
//! E2). A message the contract cannot read, or one past the cap, is refused
//! with `INVALID_MESSAGE` before anything reads it; a message the core would
//! write past the cap, or one its encoder refuses, is a defect in the core,
//! `INTERNAL_ERROR`.

use sdm_contract::{ControlMessage, CoreError, CoreWarning, Detail, ErrorKind, Scalar, codes};

/// The largest control message, in bytes, in either direction.
pub const CONTROL_CAP: usize = 64 * 1024;

fn size_fields(error: CoreError, bytes: usize) -> CoreError {
    let count = |n: usize| Detail::Scalar(Scalar::UInt64(u64::try_from(n).unwrap_or(u64::MAX)));
    error
        .with_field("bytes", count(bytes))
        .with_field("cap", count(CONTROL_CAP))
}

/// Reads a control message a host sent.
pub fn decode(bytes: &[u8]) -> Result<ControlMessage, CoreError> {
    if bytes.len() > CONTROL_CAP {
        return Err(size_fields(
            CoreError::protocol(
                codes::INVALID_MESSAGE,
                "the control message is larger than the control cap; data travels as attachments",
            ),
            bytes.len(),
        ));
    }
    serde_json::from_slice(bytes).map_err(|e| {
        CoreError::protocol(
            codes::INVALID_MESSAGE,
            format!("not a control message of this contract: {e}"),
        )
    })
}

/// Writes a control message for the host.
pub fn encode(message: &ControlMessage) -> Result<Vec<u8>, CoreError> {
    let bytes = serde_json::to_vec(message).map_err(|e| {
        CoreError::new(
            ErrorKind::Internal,
            codes::INTERNAL_ERROR,
            format!("the core wrote a control message its contract refuses: {e}"),
        )
    })?;
    if bytes.len() > CONTROL_CAP {
        return Err(size_fields(
            CoreError::new(
                ErrorKind::Internal,
                codes::INTERNAL_ERROR,
                "the core wrote a control message larger than the control cap",
            ),
            bytes.len(),
        ));
    }
    Ok(bytes)
}

/// `error` as an `error` control message. An error too large for the cap
/// loses its cause, then its fields and its message, so that one is always
/// written.
pub fn encode_error(error: &CoreError) -> Vec<u8> {
    let attempts = [
        error.clone(),
        CoreError {
            cause: None,
            ..error.clone()
        },
        CoreError::new(
            error.kind,
            &error.code,
            "the error was larger than the control cap",
        ),
    ];
    for attempt in attempts {
        if let Ok(bytes) = encode(&ControlMessage::Error { error: attempt }) {
            return bytes;
        }
    }
    // Only a code past the contract's pattern gets here.
    let fallback = CoreError::new(
        ErrorKind::Internal,
        codes::INTERNAL_ERROR,
        "the error could not be written",
    );
    serde_json::to_vec(&ControlMessage::Error { error: fallback })
        .unwrap_or_else(|_| b"{\"message\":\"error\"}".to_vec())
}

/// `warnings` as `warnings` control messages, in order, each within the
/// cap; none when there are none.
pub fn encode_warnings(warnings: Vec<CoreWarning>) -> Result<Vec<Vec<u8>>, CoreError> {
    let mut out = Vec::new();
    let mut batch: Vec<CoreWarning> = Vec::new();
    for warning in warnings {
        batch.push(warning);
        let Err(error) = encode(&ControlMessage::Warnings {
            warnings: batch.clone(),
        }) else {
            continue;
        };
        let last = batch.pop().expect("one warning was pushed");
        if batch.is_empty() {
            // One warning alone is past the cap.
            return Err(error);
        }
        out.push(encode(&ControlMessage::Warnings {
            warnings: std::mem::take(&mut batch),
        })?);
        batch.push(last);
    }
    if !batch.is_empty() {
        out.push(encode(&ControlMessage::Warnings { warnings: batch })?);
    }
    Ok(out)
}
