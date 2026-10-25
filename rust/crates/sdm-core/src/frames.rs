// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A run-local frame source; no DataLayer implementation or host fetch.

use std::collections::BTreeSet;

use sdm_contract::data::AttachmentId;
use sdm_contract::frames::{FrameManifest, FrameStage};
use sdm_contract::ops::{DescribedSource, OperationResult, SourceDescription, SourceSpec};
use sdm_contract::{CoreError, Scalar, Sentence};
use sdm_schema::assemble::{self, Given};

use crate::attached::take_json;
use crate::run::{Asker, Attachment, Attachments, internal};

fn invalid(message: &str) -> CoreError {
    CoreError::protocol(Sentence::InvalidMessage, message)
}

/// The actual describe consumer of the manifest's Arrow inputs.
pub(crate) async fn run(
    asker: Asker,
    spec: SourceSpec,
    manifest: FrameManifest,
    mut attachments: Attachments,
) -> Result<OperationResult, CoreError> {
    let named = spec.tables.clone();
    let names: BTreeSet<_> = named.iter().map(|table| &table.name).collect();
    if names.len() != named.len() {
        return Err(invalid("a frame constructor names one table twice"));
    }
    if named.is_empty() || named.len() != manifest.tables.len() {
        return Err(invalid(
            "a frame describe must name each admitted table once",
        ));
    }
    let exact_order = |order: &[String], expected: BTreeSet<&String>| {
        order.len() == expected.len() && order.iter().collect::<BTreeSet<_>>() == expected
    };
    if !exact_order(
        &manifest.override_order.primary_keys,
        spec.primary_keys.keys().collect(),
    ) || !exact_order(
        &manifest.override_order.time_columns,
        spec.time_columns.keys().collect(),
    ) {
        return Err(invalid(
            "frame override order must name each corresponding SourceSpec key exactly once",
        ));
    }
    if spec.relationships.is_some() {
        return Err(invalid(
            "frame explicit relationships belong only to the ordered manifest slots",
        ));
    }
    if manifest.key_forms.keys().collect::<BTreeSet<_>>()
        != spec.primary_keys.keys().collect::<BTreeSet<_>>()
    {
        return Err(invalid(
            "frame key forms must name each explicit key override exactly once",
        ));
    }
    if manifest.stage == FrameStage::Table
        && (named.len() != 1
            || !spec.primary_keys.is_empty()
            || !spec.time_columns.is_empty()
            || manifest.relationships.is_some())
    {
        return Err(invalid(
            "Table stage requires one complete table and no constructor overrides",
        ));
    }
    let mut reads = Vec::new();
    let mut snapshot = None;
    for table in &named {
        if table.columns.is_some() {
            return Err(invalid(
                "a frame constructor describes the complete admitted columns",
            ));
        }
        let bound = manifest
            .tables
            .iter()
            .find(|input| input.source == table.source)
            .ok_or_else(|| invalid("a named frame has no admitted source binding"))?;
        let Some(Attachment::Arrow(rows)) = attachments.remove(&bound.rows) else {
            return Err(invalid("a frame binding does not name admitted Arrow data"));
        };
        let mut read =
            sdm_schema::frames::read(&table.name, &table.source, &rows.schema, &rows.batches)?;
        let immediate_table_notices = read.hold_inferred()?;
        if manifest.stage == FrameStage::Table {
            for notice in immediate_table_notices {
                asker.warn(notice);
            }
            snapshot = Some(rows);
        }
        // Schema recomputes the same inference over owned snapshots. Only
        // this separately structured table-notice result is discarded;
        // override and assembly warnings still use the normal channel.
        reads.push(read);
    }
    if manifest.stage == FrameStage::Table {
        let read = reads.remove(0);
        let duplicates = if read.table.primary_key.is_empty() {
            0
        } else {
            read.count(&read.table.primary_key).repeats()
        };
        let mut description = SourceDescription {
            tables: vec![assemble::described_table(
                read.table,
                Some(Scalar::UInt64(duplicates)),
            )],
            links: vec![],
        };
        if !description.tables[0].primary_key.is_empty() {
            description.tables[0].frame_key_form = Some(sdm_contract::frames::FrameKeyForm::Scalar);
        }
        assemble::mark_keys(&mut description);
        let rows = snapshot.ok_or_else(|| internal("Table stage lost its admitted snapshot"))?;
        return finish(&asker, description, Some(rows));
    }
    let mut tables: Vec<_> = reads.iter().map(|read| read.table.clone()).collect();
    for name in &manifest.override_order.primary_keys {
        let one =
            std::collections::BTreeMap::from([(name.clone(), spec.primary_keys[name].clone())]);
        assemble::apply_frame_keys_and_times(
            &mut tables,
            &one,
            &std::collections::BTreeMap::new(),
        )?;
        let at = tables
            .iter()
            .position(|table| &table.name == name)
            .ok_or_else(|| internal("a validated override lost its table"))?;
        if !tables[at].primary_key.is_empty() {
            for warning in reads[at].hold_stated(&tables[at].primary_key)? {
                asker.warn(warning);
            }
        }
    }
    for name in &manifest.override_order.time_columns {
        let one =
            std::collections::BTreeMap::from([(name.clone(), spec.time_columns[name].clone())]);
        assemble::apply_frame_keys_and_times(
            &mut tables,
            &std::collections::BTreeMap::new(),
            &one,
        )?;
    }
    let mut duplicates = Vec::new();
    for (table, read) in tables.iter().zip(&reads) {
        let count = read.count(&table.primary_key);
        duplicates.push(if table.primary_key.is_empty() {
            0
        } else {
            count.repeats()
        });
    }
    let timed: BTreeSet<String> = spec.time_columns.keys().cloned().collect();
    let (mut assembled, link_notices) = assemble::assemble_frames(
        tables,
        Given {
            time_columns: Some(&timed),
            relationships: None,
        },
        Some(Scalar::UInt64(0)),
        &manifest.key_forms,
        manifest.relationships.as_deref(),
    )?;
    for (table, duplicates) in assembled.description.tables.iter_mut().zip(duplicates) {
        table.duplicate_keys = Some(Scalar::UInt64(duplicates));
    }
    for warning in link_notices {
        asker.warn(warning);
    }
    for warning in sdm_schema::frames::final_notices(
        &reads,
        &assembled.description,
        manifest.relationships.is_none(),
    )? {
        asker.warn(warning);
    }
    for warning in assembled.warnings {
        asker.warn(warning);
    }
    finish(&asker, assembled.description, None)
}

fn finish(
    asker: &Asker,
    description: SourceDescription,
    rows: Option<crate::ArrowData>,
) -> Result<OperationResult, CoreError> {
    let bytes = serde_json::to_vec(&description)
        .map_err(|_| internal("a frame description does not serialize"))?;
    let description = asker.attach(Attachment::Bytes(bytes))?;
    let frame_rows = rows
        .map(|rows| asker.attach(Attachment::Arrow(rows)))
        .transpose()?;
    Ok(OperationResult::DescribeSource(DescribedSource {
        description,
        frame_rows,
        notes: vec![],
    }))
}

/// Decode the manifest using the same locked type as the attachment resolver.
pub(crate) fn take_manifest(
    attachments: &mut Attachments,
    id: AttachmentId,
) -> Result<FrameManifest, CoreError> {
    take_json(attachments, id)
}
