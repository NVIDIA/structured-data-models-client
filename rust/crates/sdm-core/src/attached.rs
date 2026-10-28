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
use sdm_wire::json::{self, ReadError, Value};
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
        Operation::PrepareFrames(request) => {
            if let Some(Attachment::Bytes(bytes)) = attachments.get(&request.input) {
                use sdm_contract::frames::materialize::FramePreparation;
                let input: FramePreparation = serde_json::from_slice(bytes)
                    .map_err(|_| invalid("unreadable frame preparation input", request.input))?;
                named.extend(input.attachments());
                if let FramePreparation::Graph {
                    frames, checkpoint, ..
                } = input
                {
                    let read = |id| match attachments.get(&id) {
                        Some(Attachment::Bytes(bytes)) => Ok(bytes.as_slice()),
                        _ => Err(invalid("a frame graph manifest lacks bytes", id)),
                    };
                    let manifest: MaterializeFrames = serde_json::from_slice(read(frames)?)
                        .map_err(|_| invalid("unreadable frame graph source manifest", frames))?;
                    named.extend(
                        manifest
                            .row_identities
                            .iter()
                            .map(|identity| (identity.values.values, AttachmentKind::Arrow)),
                    );
                    declare_frame_rows(manifest.tables, &mut named)?;
                    if let Some(checkpoint) = checkpoint {
                        let manifest: FrameCheckpointManifest =
                            serde_json::from_slice(read(checkpoint.manifest)?).map_err(|_| {
                                invalid(
                                    "unreadable frame graph checkpoint manifest",
                                    checkpoint.manifest,
                                )
                            })?;
                        if manifest.retained.is_some() {
                            return Err(invalid(
                                "raw frame fallback cannot also declare a retained graph owner",
                                checkpoint.manifest,
                            ));
                        }
                        named.push((manifest.metadata, AttachmentKind::Bytes));
                        named.extend(
                            manifest
                                .row_identities
                                .iter()
                                .map(|identity| (identity.values.values, AttachmentKind::Arrow)),
                        );
                        declare_frame_rows(manifest.tables, &mut named)?;
                    }
                }
            }
            return Ok(named);
        }
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
                match &input {
                    sdm_contract::prepare::Preparation::CheckRelationalCoverage {
                        data,
                        rows,
                        ..
                    } => {
                        named.extend(
                            [data.response, data.rows, *rows]
                                .into_iter()
                                .map(|id| (id, AttachmentKind::Bytes)),
                        );
                    }
                    sdm_contract::prepare::Preparation::CompleteRelationalResponse { part } => {
                        named.extend(
                            [
                                part.data.response,
                                part.data.rows,
                                part.rows,
                                part.target_rows,
                            ]
                            .into_iter()
                            .map(|id| (id, AttachmentKind::Bytes)),
                        );
                    }
                    sdm_contract::prepare::Preparation::FinishRelationalResponse { input } => {
                        named.extend(
                            input
                                .attachments()
                                .into_iter()
                                .map(|id| (id, AttachmentKind::Bytes)),
                        );
                    }
                    sdm_contract::prepare::Preparation::PrepareRelationalRankPopulation {
                        source,
                    } => {
                        named.extend(
                            source
                                .attachments()
                                .into_iter()
                                .map(|id| (id, AttachmentKind::Bytes)),
                        );
                    }
                    _ => {}
                }
                let rows = match input {
                    sdm_contract::prepare::Preparation::CallRelational { input } => {
                        named.push((input.body, AttachmentKind::Bytes));
                        None
                    }
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
            let input = serde_json::from_slice::<MaterializeFrames>(read(manifest)?)
                .map_err(|_| invalid("unreadable materialize frame manifest", manifest))?;
            named.extend(
                input
                    .row_identities
                    .iter()
                    .map(|identity| (identity.values.values, AttachmentKind::Arrow)),
            );
            let mut tables = input.tables;
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
                named.extend(
                    checkpoint_manifest
                        .row_identities
                        .iter()
                        .map(|identity| (identity.values.values, AttachmentKind::Arrow)),
                );
                tables.extend(checkpoint_manifest.tables);
            }
            tables
        }
        _ => unreachable!(),
    };
    declare_frame_rows(std::mem::take(&mut tables), &mut named)?;
    Ok(named)
}

fn declare_frame_rows(
    tables: Vec<sdm_contract::frames::FrameInput>,
    named: &mut Vec<(AttachmentId, AttachmentKind)>,
) -> Result<(), CoreError> {
    let mut relations = BTreeSet::new();
    for table in tables {
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
    Ok(())
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
            (Some(Attachment::Bytes(_)), AttachmentKind::Bytes)
            | (Some(Attachment::Arrow(_)), AttachmentKind::Arrow)
            | (Some(Attachment::FrameGraph(_)), AttachmentKind::FrameGraph) => {}
            (Some(Attachment::FrameLabelGroups(_)), AttachmentKind::FrameLabelGroups)
            | (Some(Attachment::FrameLabelBatch(_)), AttachmentKind::FrameLabelBatch)
            | (Some(Attachment::FramePopulation(_)), AttachmentKind::FramePopulation)
            | (Some(Attachment::FrameQuestion(_)), AttachmentKind::FrameQuestion)
            | (Some(Attachment::FrameTask(_)), AttachmentKind::FrameTask)
            | (Some(Attachment::FrameReferences(_)), AttachmentKind::FrameReferences)
            | (Some(Attachment::FramePayload(_)), AttachmentKind::FramePayload) => {}
            _ => {
                return Err(invalid(
                    "the attachment does not have its declared kind",
                    *id,
                ));
            }
        }
    }
    if let Some(id) = attachments.keys().find(|id| !seen.contains(id)) {
        return Err(invalid("an attachment is not named by its message", *id));
    }
    Ok(())
}

fn take_bytes(attachments: &mut Attachments, id: AttachmentId) -> Result<Vec<u8>, CoreError> {
    let Some(Attachment::Bytes(bytes)) = attachments.remove(&id) else {
        return Err(invalid(
            "the message names a bytes attachment it does not attach",
            id,
        ));
    };
    Ok(bytes)
}

fn unreadable_json(id: AttachmentId, error: ReadError) -> CoreError {
    let count = |n: usize| Detail::Scalar(Scalar::UInt64(u64::try_from(n).unwrap_or(u64::MAX)));
    invalid("an attachment is not the JSON its message names", id)
        .with_field("category", Detail::text(error.category))
        .with_field("line", count(error.line))
        .with_field("column", count(error.column))
}

/// Takes the bytes attachment `id` and reads it as the UTF-8 JSON of a `T`.
/// A refusal names the attachment, and where in it and what kind of
/// problem it found, never the text it read.
pub fn take_json<T: DeserializeOwned>(
    attachments: &mut Attachments,
    id: AttachmentId,
) -> Result<T, CoreError> {
    let bytes = take_bytes(attachments, id)?;
    serde_json::from_slice(&bytes).map_err(|e| {
        let category = match e.classify() {
            serde_json::error::Category::Io => "io",
            serde_json::error::Category::Syntax => "syntax",
            serde_json::error::Category::Data => "data",
            serde_json::error::Category::Eof => "eof",
        };
        unreadable_json(
            id,
            ReadError {
                category,
                line: e.line(),
                column: e.column(),
            },
        )
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

/// Takes one option-values attachment through the existing option reader.
/// Numeric tokens, including NaN and infinities, remain exact facts for
/// downstream option policy; object order and native text are retained.
pub fn take_option_values(
    attachments: &mut Attachments,
    id: AttachmentId,
) -> Result<Value, CoreError> {
    let bytes = take_bytes(attachments, id)?;
    json::read_options(&bytes).map_err(|error| unreadable_json(id, error))
}

/// The optional form of [`take_option_values`], leaving attachments untouched
/// when no value is named.
pub fn take_option_values_if(
    attachments: &mut Attachments,
    id: Option<AttachmentId>,
) -> Result<Option<Value>, CoreError> {
    id.map(|id| take_option_values(attachments, id)).transpose()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::run::ArrowData;
    use arrow_array::RecordBatch;
    use arrow_schema::Schema;
    use sdm_contract::ErrorCode;
    use std::sync::Arc;

    #[test]
    fn option_attachment_keeps_numeric_tokens_member_order_and_unicode() {
        let id = AttachmentId(1);
        let bytes = br#"{"z":123456789012345678901234567890,"a":1,"negative_zero":-0,"fraction":1.2500e-42,"nan":NaN,"positive":Infinity,"negative":-Infinity,"text":"\u0000\u03a9\ud83d\ude3a","a":2.00}"#;
        let mut attachments = Attachments::from([(id, Attachment::Bytes(bytes.to_vec()))]);
        let decoded = take_option_values(&mut attachments, id).unwrap();
        assert!(attachments.is_empty());
        assert_eq!(
            decoded,
            Value::Object(vec![
                (
                    "z".to_owned(),
                    Value::Number("123456789012345678901234567890".to_owned())
                ),
                ("a".to_owned(), Value::Number("2.00".to_owned())),
                ("negative_zero".to_owned(), Value::Number("-0".to_owned())),
                (
                    "fraction".to_owned(),
                    Value::Number("1.2500e-42".to_owned())
                ),
                ("nan".to_owned(), Value::Number("NaN".to_owned())),
                ("positive".to_owned(), Value::Number("Infinity".to_owned())),
                ("negative".to_owned(), Value::Number("-Infinity".to_owned())),
                ("text".to_owned(), Value::Text("\0Ω😺".into())),
            ])
        );
    }

    #[test]
    fn unreadable_option_attachment_preserves_reader_diagnostics() {
        for bytes in [b"[".as_slice(), br#""\ud800""#.as_slice()] {
            let expected = json::read_options(bytes).unwrap_err();
            let id = AttachmentId(3);
            let mut attachments = Attachments::from([(id, Attachment::Bytes(bytes.to_vec()))]);
            let error = take_option_values(&mut attachments, id).unwrap_err();
            assert_eq!(error.code, ErrorCode::InvalidMessage);
            assert_eq!(error.sentence, Sentence::InvalidMessage);
            assert_eq!(
                error.fields["attachment"],
                Detail::Scalar(Scalar::UInt32(3))
            );
            assert_eq!(error.fields["category"], Detail::text(expected.category));
            assert_eq!(
                error.fields["line"],
                Detail::Scalar(Scalar::UInt64(u64::try_from(expected.line).unwrap()))
            );
            assert_eq!(
                error.fields["column"],
                Detail::Scalar(Scalar::UInt64(u64::try_from(expected.column).unwrap()))
            );
            assert_eq!(
                error.message,
                "an attachment is not the JSON its message names"
            );
            assert!(attachments.is_empty());
        }
    }

    #[test]
    fn optional_option_values_check_kind_and_consume_only_the_named_owner() {
        let id = AttachmentId(4);
        let spare = AttachmentId(5);
        let schema = Arc::new(Schema::empty());
        let arrow = ArrowData::new(schema.clone(), vec![RecordBatch::new_empty(schema)]);
        let mut attachments = Attachments::from([
            (id, Attachment::Arrow(arrow)),
            (spare, Attachment::Bytes(b"null".to_vec())),
        ]);
        assert_eq!(take_option_values_if(&mut attachments, None).unwrap(), None);
        assert_eq!(attachments.len(), 2);
        let error = take_option_values_if(&mut attachments, Some(id)).unwrap_err();
        assert_eq!(error.code, ErrorCode::InvalidMessage);
        assert_eq!(
            error.fields["attachment"],
            Detail::Scalar(Scalar::UInt32(4))
        );
        assert!(!error.fields.contains_key("category"));
        assert!(!attachments.contains_key(&id));
        assert!(attachments.contains_key(&spare));
        assert_eq!(take_option_values(&mut attachments, id).unwrap_err(), error);
        assert_eq!(
            take_option_values_if(&mut attachments, Some(spare)).unwrap(),
            Some(Value::Null)
        );
        assert!(attachments.is_empty());
    }
}
