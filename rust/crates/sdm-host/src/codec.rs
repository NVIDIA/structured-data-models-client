// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The control-message codecs and the control cap.
//!
//! A control message is the contract's JSON. The cap bounds a message's
//! envelope only: input-sized fields travel as attachments (decision H5).
//!
//! - From the host, a message past [`CONTROL_CAP`] bytes, or one the
//!   contract cannot read, is refused with `INVALID_MESSAGE` before anything
//!   reads it ([`decode`]).
//! - From the core, nothing is truncated, refused or dropped for size: a
//!   message whose encoding would pass the cap travels whole as a bytes
//!   attachment of its JSON, named by a minimal `attached` message
//!   (decision H7; [`write`], [`write_error`]). A message the core's own
//!   encoder refuses is a defect in the core, `INTERNAL_ERROR`.

use sdm_contract::data::AttachmentId;
use sdm_contract::{ControlMessage, CoreError, Detail, Scalar, Sentence};
use sdm_core::{Attachment, Attachments};

/// The largest control message envelope, in bytes, in either direction.
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
                Sentence::InvalidMessage,
                "the control message is larger than the control cap; input-sized fields travel as attachments",
            ),
            bytes.len(),
        ));
    }
    serde_json::from_slice(bytes)
        .map_err(|e| unreadable("not a control message of this contract", &e))
}

/// The refusal of JSON the contract cannot read: where in it and what kind
/// of problem, never the text read there, which may be an input value
/// (decision H17).
fn unreadable(message: &str, e: &serde_json::Error) -> CoreError {
    let category = match e.classify() {
        serde_json::error::Category::Io => "io",
        serde_json::error::Category::Syntax => "syntax",
        serde_json::error::Category::Data => "data",
        serde_json::error::Category::Eof => "eof",
    };
    let count = |n: usize| Detail::Scalar(Scalar::UInt64(u64::try_from(n).unwrap_or(u64::MAX)));
    CoreError::protocol(Sentence::InvalidMessage, message)
        .with_field("category", Detail::text(category))
        .with_field("line", count(e.line()))
        .with_field("column", count(e.column()))
}

/// Writes a control message within the cap, as a host writes the messages
/// it sends; one past the cap is refused.
pub fn encode(message: &ControlMessage) -> Result<Vec<u8>, CoreError> {
    let bytes = to_json(message)?;
    if bytes.len() > CONTROL_CAP {
        return Err(size_fields(
            CoreError::new(
                Sentence::Internal,
                "the control message is larger than the control cap",
            ),
            bytes.len(),
        ));
    }
    Ok(bytes)
}

fn to_json(message: &ControlMessage) -> Result<Vec<u8>, CoreError> {
    serde_json::to_vec(message).map_err(|_| {
        CoreError::new(
            Sentence::Internal,
            "the core wrote a control message its contract refuses",
        )
    })
}

/// A control message the core wrote, with the attachments it names: none,
/// or the one that carries the message whole when its encoding would pass
/// the cap.
#[derive(Debug, Clone)]
pub struct Outgoing {
    /// The control message, at most [`CONTROL_CAP`] bytes.
    pub control: Vec<u8>,
    pub attachments: Attachments,
}

/// Writes a control message from the core: within the cap as it is, or
/// whole as a bytes attachment named by an `attached` message, with an id
/// `reserve` gives.
pub fn write(
    message: &ControlMessage,
    reserve: impl FnOnce() -> Result<AttachmentId, CoreError>,
) -> Result<Outgoing, CoreError> {
    let bytes = to_json(message)?;
    if bytes.len() <= CONTROL_CAP {
        return Ok(Outgoing {
            control: bytes,
            attachments: Attachments::new(),
        });
    }
    let id = reserve()?;
    Ok(Outgoing {
        control: encode(&ControlMessage::Attached { attachment: id })?,
        attachments: Attachments::from([(id, Attachment::Bytes(bytes))]),
    })
}

/// `error` as an `error` control message, whole: within the cap, or as an
/// attachment ([`write`]). Only an error the core's encoder refuses, or a
/// run out of attachment ids, is replaced, by the `INTERNAL_ERROR` that
/// says so; and an error, or a cause in its chain, whose kind its code does
/// not carry is replaced by the `INTERNAL_ERROR` that names the pair
/// (decision E22).
pub fn write_error(
    error: &CoreError,
    reserve: impl FnOnce() -> Result<AttachmentId, CoreError>,
) -> Outgoing {
    let error = if error.holds_to_table() {
        error.clone()
    } else {
        let outside = outside_pair(error);
        CoreError::outside_table(outside.kind, outside.code)
    };
    let replacement = match write(&ControlMessage::Error { error }, reserve) {
        Ok(out) => return out,
        Err(replacement) => replacement,
    };
    let control = encode(&ControlMessage::Error { error: replacement }).unwrap_or_else(|_| {
        // The replacement is the core's own short text; it always encodes.
        encode(&ControlMessage::Error {
            error: CoreError::new(Sentence::Internal, "the error could not be written"),
        })
        .expect("a short internal error is within the cap and of the contract")
    });
    Outgoing {
        control,
        attachments: Attachments::new(),
    }
}

/// The first error in `error`'s cause chain whose kind its code does not
/// carry; `error` itself when none is.
fn outside_pair(error: &CoreError) -> &CoreError {
    let mut at = error;
    while at.code.allows(at.kind) {
        match at.cause.as_deref() {
            Some(cause) => at = cause,
            None => return error,
        }
    }
    at
}

/// Reads a control message the core wrote, as a host does: an `attached`
/// message is replaced by the message its attachment carries.
pub fn read(outgoing: &Outgoing) -> Result<ControlMessage, CoreError> {
    let message = decode(&outgoing.control)?;
    let ControlMessage::Attached { attachment } = message else {
        return Ok(message);
    };
    let Some(Attachment::Bytes(bytes)) = outgoing.attachments.get(&attachment) else {
        return Err(CoreError::protocol(
            Sentence::InvalidMessage,
            "an attached message names an attachment that is not with it",
        ));
    };
    serde_json::from_slice(bytes).map_err(|e| {
        unreadable(
            "an attached message is not a control message of this contract",
            &e,
        )
    })
}
