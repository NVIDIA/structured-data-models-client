// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Attachments a host message names: the check that a message attaches
//! exactly what it names, as the kind it names, and the reading of an
//! input-sized field that travels as a bytes attachment of JSON (decision
//! H5).

use std::collections::BTreeSet;

use sdm_contract::data::AttachmentId;
use sdm_contract::ops::AttachmentKind;
use sdm_contract::{CoreError, Detail, Scalar, Sentence};
use serde::de::DeserializeOwned;

use crate::run::{Attachment, Attachments};

fn invalid(message: &str, id: AttachmentId) -> CoreError {
    CoreError::protocol(Sentence::InvalidMessage, message)
        .with_field("attachment", Detail::Scalar(Scalar::UInt32(id.0)))
}

/// Checks that `attachments` holds exactly the ids `named`, each once and
/// of the kind named, or refuses the message with `INVALID_MESSAGE` naming
/// the attachment.
pub fn check_attachments(
    named: &[(AttachmentId, AttachmentKind)],
    attachments: &Attachments,
) -> Result<(), CoreError> {
    let mut seen = BTreeSet::new();
    for (id, kind) in named {
        if !seen.insert(*id) {
            return Err(invalid("the message names one attachment twice", *id));
        }
        match (attachments.get(id), kind) {
            (None, _) => {
                return Err(invalid(
                    "the message names an attachment it does not attach",
                    *id,
                ));
            }
            (Some(Attachment::Bytes(_)), AttachmentKind::Arrow) => {
                return Err(invalid(
                    "the message names Arrow data and attaches bytes",
                    *id,
                ));
            }
            (Some(Attachment::Arrow(_)), AttachmentKind::Bytes) => {
                return Err(invalid(
                    "the message names bytes and attaches Arrow data",
                    *id,
                ));
            }
            _ => {}
        }
    }
    if let Some(id) = attachments.keys().find(|id| !seen.contains(id)) {
        return Err(invalid("an attachment is not named by its message", *id));
    }
    Ok(())
}

/// Takes the bytes attachment `id` and reads it as the UTF-8 JSON of a `T`.
/// A refusal names the attachment, and where in it and what kind of
/// problem it found, never the text it read.
pub fn take_json<T: DeserializeOwned>(
    attachments: &mut Attachments,
    id: AttachmentId,
) -> Result<T, CoreError> {
    let Some(Attachment::Bytes(bytes)) = attachments.remove(&id) else {
        return Err(invalid(
            "the message names a bytes attachment it does not attach",
            id,
        ));
    };
    serde_json::from_slice(&bytes).map_err(|e| {
        let category = match e.classify() {
            serde_json::error::Category::Io => "io",
            serde_json::error::Category::Syntax => "syntax",
            serde_json::error::Category::Data => "data",
            serde_json::error::Category::Eof => "eof",
        };
        let count = |n: usize| Detail::Scalar(Scalar::UInt64(u64::try_from(n).unwrap_or(u64::MAX)));
        invalid("an attachment is not the JSON its message names", id)
            .with_field("category", Detail::text(category))
            .with_field("line", count(e.line()))
            .with_field("column", count(e.column()))
    })
}

/// Takes the bytes attachment `id`, when a message names one, as
/// [`take_json`] takes it; `None` when it names none.
pub fn take_json_if<T: DeserializeOwned>(
    attachments: &mut Attachments,
    id: Option<AttachmentId>,
) -> Result<Option<T>, CoreError> {
    id.map(|id| take_json(attachments, id)).transpose()
}
