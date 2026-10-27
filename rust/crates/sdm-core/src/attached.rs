// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Attachments a host message names: the check that a message attaches
//! exactly what it names, as the kind it names, and the reading of an
//! input-sized field that travels as a bytes attachment of JSON (decision
//! H5).

use std::collections::BTreeSet;

use sdm_contract::data::{AttachmentId, ProviderChoice};
use sdm_contract::frames::{FrameCheckpointManifest, FrameManifest, MaterializeFrames};
use sdm_contract::ops::{AttachmentKind, Operation};
use sdm_contract::{CoreError, Detail, Scalar, Sentence};
use serde::de::DeserializeOwned;

use crate::run::{Attachment, Attachments};

fn invalid(message: &str, id: AttachmentId) -> CoreError {
    CoreError::protocol(Sentence::InvalidMessage, message)
        .with_field("attachment", Detail::Scalar(Scalar::UInt32(id.0)))
}

/// Resolves the input-sized frame manifest before checking the entire
/// attachment set. HostRun's unreadable path uses this same resolver.
/// Ordinary operations retain their envelope-only attachment rules.
pub fn resolved_attachments(
    operation: &Operation,
    attachments: &Attachments,
) -> Result<Vec<(AttachmentId, AttachmentKind)>, CoreError> {
    let mut named = operation.attachments();
    match operation {
        Operation::PrepareRequest(request) => {
            if let Some(Attachment::Bytes(bytes)) = attachments.get(&request.input) {
                let input: sdm_contract::prepare::Preparation = serde_json::from_slice(bytes)
                    .map_err(|_| invalid("unreadable direct preparation input", request.input))?;
                if let sdm_contract::prepare::Preparation::LayoutResponse {
                    data, attributed, ..
                } = &input
                {
                    named.extend(
                        [data.payload, data.response, data.rows, *attributed]
                            .into_iter()
                            .map(|id| (id, AttachmentKind::Bytes)),
                    );
                }
                let rows = match input {
                    sdm_contract::prepare::Preparation::EncodeBody { input } => {
                        Some((input.context, input.predict))
                    }
                    sdm_contract::prepare::Preparation::Context {
                        context, predict, ..
                    }
                    | sdm_contract::prepare::Preparation::Schema {
                        context, predict, ..
                    }
                    | sdm_contract::prepare::Preparation::FinishConstructor {
                        context,
                        predict,
                        ..
                    } => Some((context, predict)),
                    sdm_contract::prepare::Preparation::Overlap { input } => {
                        Some((input.context, input.predict))
                    }
                    sdm_contract::prepare::Preparation::Envelope { input } => {
                        Some((input.context, input.predict))
                    }
                    _ => None,
                };
                if let Some((context, predict)) = rows {
                    named.extend([
                        (context, AttachmentKind::Arrow),
                        (predict, AttachmentKind::Arrow),
                    ]);
                }
            }
            return Ok(named);
        }
        Operation::Predict(request) if request.direct.is_some() => {
            let id = request.direct.expect("matched direct input");
            if let Some(Attachment::Bytes(bytes)) = attachments.get(&id) {
                let input: sdm_contract::prepare::DirectPredict = serde_json::from_slice(bytes)
                    .map_err(|_| invalid("unreadable direct predict input", id))?;
                if let sdm_contract::prepare::DirectPredict::Encoded { body, .. } = input {
                    named.push((body.payload, AttachmentKind::Bytes));
                }
            }
            return Ok(named);
        }
        _ => {}
    }

    let manifest = match operation {
        Operation::DescribeSource(request) => match request.provider {
            ProviderChoice::Frames { manifest } => manifest,
            _ => return Ok(named),
        },
        Operation::Materialize(request) => match request.provider {
            ProviderChoice::Frames { manifest } => manifest,
            _ => {
                if request
                    .sampler_state
                    .as_ref()
                    .is_some_and(|state| state.frame_checkpoint.is_some())
                {
                    return Err(CoreError::protocol(
                        Sentence::InvalidMessage,
                        "a non-frame materializer cannot consume a frame checkpoint",
                    )
                    .dropping(sdm_contract::HeldState::SamplerState));
                }
                return Ok(named);
            }
        },
        _ => return Ok(named),
    };
    let read = |id| {
        let Some(Attachment::Bytes(bytes)) = attachments.get(&id) else {
            return Err(invalid("a frame manifest lacks bytes", id));
        };
        Ok(bytes.as_slice())
    };
    if !attachments.contains_key(&manifest) {
        return Ok(named);
    }
    // Check the static bytes fields first without discarding extra entries:
    // dynamic Arrow ids are not known until this manifest is read.
    let mut seen = BTreeSet::new();
    for &(id, kind) in &named {
        if !seen.insert(id) {
            return Err(invalid("the message names one attachment twice", id));
        }
        let one: Attachments = attachments
            .get(&id)
            .cloned()
            .map(|value| (id, value))
            .into_iter()
            .collect();
        check_attachments(&[(id, kind)], &one)?;
    }
    let mut tables = match operation {
        Operation::DescribeSource(_) => {
            let frame = serde_json::from_slice::<FrameManifest>(read(manifest)?)
                .map_err(|_| invalid("unreadable constructor frame manifest", manifest))?;
            named.extend(frame.inference.map(|id| (id, AttachmentKind::Bytes)));
            frame.tables
        }
        Operation::Materialize(request) => {
            let mut tables = serde_json::from_slice::<MaterializeFrames>(read(manifest)?)
                .map_err(|_| invalid("unreadable materialize frame manifest", manifest))?
                .tables;
            if let Some(checkpoint) = request
                .sampler_state
                .as_ref()
                .and_then(|state| state.frame_checkpoint.as_ref())
            {
                let checkpoint_manifest: FrameCheckpointManifest =
                    serde_json::from_slice(read(checkpoint.manifest)?).map_err(|_| {
                        invalid("unreadable frame checkpoint manifest", checkpoint.manifest)
                    })?;
                named.push((checkpoint_manifest.metadata, AttachmentKind::Bytes));
                tables.extend(checkpoint_manifest.tables);
            }
            tables
        }
        _ => unreachable!(),
    };
    let mut relations = BTreeSet::new();
    for table in tables.drain(..) {
        if table.source.source.len() != 1
            || table.source.source[0].is_empty()
            || !table.source.columns.is_empty()
            || table.source.filter.is_some()
            || table.source.found
            || !relations.insert(table.source)
        {
            return Err(invalid(
                "frame sources must have unique literal logical relations",
                table.rows,
            ));
        }
        named.push((table.rows, AttachmentKind::Arrow));
    }
    Ok(named)
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
