// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A run-local frame source; no DataLayer implementation or host fetch.

use std::collections::BTreeSet;

use sdm_contract::data::AttachmentId;
use sdm_contract::frames::{
    FrameInference, FrameManifest, FrameSampleColumn, FrameSchema, FrameStage, InferredFrames,
};
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
    if named.is_empty()
        || (manifest.stage != FrameStage::Edit && named.len() != manifest.tables.len())
    {
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
    let inference = manifest
        .inference
        .map(|id| take_json::<FrameInference>(&mut attachments, id))
        .transpose()?;
    if manifest.stage == FrameStage::Edit {
        if !manifest.tables.is_empty()
            || named.iter().any(|table| table.columns.is_some())
            || spec.primary_keys.len() != 1
            || !spec.time_columns.is_empty()
            || manifest.relationships.is_some()
        {
            return Err(invalid(
                "a frame key edit requires only one metadata override",
            ));
        }
        let Some(FrameInference::Inferred(mut values)) = inference else {
            return Err(invalid("a frame key edit requires completed table facts"));
        };
        if values.len() != named.len()
            || named.iter().zip(&values).any(|(table, value)| {
                value.table.name != table.name || value.table.source != table.source
            })
        {
            return Err(invalid(
                "frame key metadata does not match the named schema",
            ));
        }
        let mut tables: Vec<_> = values
            .iter()
            .map(|v| sdm_schema::frames::inferred(&v.table, v.time.clone()))
            .collect();
        assemble::apply_frame_keys_and_times(&mut tables, &spec.primary_keys, &spec.time_columns)?;
        for (value, table) in values.iter_mut().zip(tables) {
            let duplicates = value.table.duplicate_keys.clone();
            let form = manifest
                .key_forms
                .get(&table.name)
                .copied()
                .or(value.table.frame_key_form);
            value.table = assemble::described_table(table, duplicates);
            value.table.frame_key_form = form;
        }
        return finish_inference(&asker, FrameInference::Inferred(values));
    }
    if matches!(
        manifest.stage,
        FrameStage::Nominate
            | FrameStage::Adopt
            | FrameStage::Time
            | FrameStage::StatedKey
            | FrameStage::Sample
    ) {
        if named.len() != 1
            || named[0].columns.is_some()
            || !spec.primary_keys.is_empty()
            || !spec.time_columns.is_empty()
            || manifest.relationships.is_some()
        {
            return Err(invalid(
                "an inference boundary requires exactly one table and no overrides",
            ));
        }
        let table = &named[0];
        let binding = manifest
            .tables
            .iter()
            .find(|t| t.source == table.source)
            .ok_or_else(|| invalid("an inference table has no Arrow binding"))?;
        let Some(Attachment::Arrow(rows)) = attachments.remove(&binding.rows) else {
            return Err(invalid("inference requires current Arrow rows"));
        };
        let (value, notices) = match (manifest.stage, inference) {
            (FrameStage::Nominate, None) => {
                let (value, notices) = sdm_schema::frames::nominate(
                    &table.name,
                    &table.source,
                    &rows.schema,
                    &rows.batches,
                )?;
                (FrameInference::Nomination(Box::new(value)), notices)
            }
            (FrameStage::Adopt, Some(FrameInference::Nomination(value)))
                if value.table.name == table.name && value.table.source == table.source =>
            {
                let (value, notices) =
                    sdm_schema::frames::adopt(*value, &rows.schema, &rows.batches)?;
                (FrameInference::Adoption(Box::new(value)), notices)
            }
            (FrameStage::Time, Some(FrameInference::Adoption(value)))
                if value.table.name == table.name && value.table.source == table.source =>
            {
                let value = sdm_schema::frames::choose_time(*value, &rows.schema, &rows.batches)?;
                (FrameInference::Inferred(vec![value]), vec![])
            }
            (FrameStage::StatedKey, Some(FrameInference::Inferred(mut values)))
                if values.len() == 1
                    && values[0].table.name == table.name
                    && values[0].table.source == table.source =>
            {
                let (value, notices) = sdm_schema::frames::adopt_stated(
                    values.remove(0),
                    &rows.schema,
                    &rows.batches,
                )?;
                (FrameInference::Inferred(vec![value]), notices)
            }
            (FrameStage::Sample, Some(FrameInference::SampleColumn(column)))
                if column.table == table.name =>
            {
                let (value, notices) =
                    sdm_schema::frames::sample(*column, &rows.schema, &rows.batches)?;
                (FrameInference::Sample(Box::new(value)), notices)
            }
            _ => {
                return Err(invalid(
                    "frame inference facts do not match this fixed boundary",
                ));
            }
        };
        for notice in notices {
            asker.warn(notice);
        }
        return finish_inference(&asker, value);
    }
    let completed = match inference {
        Some(FrameInference::Inferred(values))
            if matches!(manifest.stage, FrameStage::Schema | FrameStage::Assemble)
                && values.len() == named.len() =>
        {
            Some(values)
        }
        None if manifest.stage != FrameStage::Assemble => None,
        _ => {
            return Err(invalid(
                "frame schema requires a complete inferred-table set",
            ));
        }
    };
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
        let read = if let Some(values) = &completed {
            let value = values
                .iter()
                .find(|v| v.table.name == table.name && v.table.source == table.source)
                .ok_or_else(|| invalid("completed table facts do not match named frames"))?;
            sdm_schema::frames::observe(
                &value.table,
                value.time.clone(),
                &rows.schema,
                &rows.batches,
            )?
        } else {
            // Compatibility callers own this immutable buffer throughout.
            let (nomination, first) = sdm_schema::frames::nominate(
                &table.name,
                &table.source,
                &rows.schema,
                &rows.batches,
            )?;
            let (adopted, second) =
                sdm_schema::frames::adopt(nomination, &rows.schema, &rows.batches)?;
            let value = sdm_schema::frames::choose_time(adopted, &rows.schema, &rows.batches)?;
            if manifest.stage == FrameStage::Table {
                for notice in first.into_iter().chain(second) {
                    asker.warn(notice);
                }
            }
            sdm_schema::frames::observe(&value.table, value.time, &rows.schema, &rows.batches)?
        };
        if manifest.stage == FrameStage::Table {
            snapshot = Some(rows);
        }
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
        } else if !spec.primary_keys.contains_key(&table.name) {
            completed
                .as_ref()
                .and_then(|values| values.iter().find(|v| v.table.name == table.name))
                .and_then(|v| v.table.duplicate_keys.as_ref())
                .and_then(|v| match v {
                    Scalar::UInt64(n) => Some(*n),
                    _ => None,
                })
                .unwrap_or_else(|| count.repeats())
        } else {
            count.repeats()
        });
    }
    let timed: BTreeSet<String> = spec.time_columns.keys().cloned().collect();
    let mut key_forms = manifest.key_forms.clone();
    if let Some(values) = &completed {
        for value in values {
            if let Some(form) = value.table.frame_key_form {
                key_forms.entry(value.table.name.clone()).or_insert(form);
            }
        }
    }
    let (mut assembled, link_notices) = assemble::assemble_frames(
        tables,
        Given {
            time_columns: Some(&timed),
            relationships: None,
        },
        Some(Scalar::UInt64(0)),
        &key_forms,
        manifest.relationships.as_deref(),
    )?;
    for (table, duplicates) in assembled.description.tables.iter_mut().zip(duplicates) {
        table.duplicate_keys = Some(Scalar::UInt64(duplicates));
    }
    for warning in link_notices {
        asker.warn(warning);
    }
    let notices = if manifest.stage == FrameStage::Assemble {
        sdm_schema::frames::key_notices(
            &reads,
            &assembled.description,
            manifest.relationships.is_none(),
        )?
    } else {
        sdm_schema::frames::final_notices(
            &reads,
            &assembled.description,
            manifest.relationships.is_none(),
        )?
    };
    if manifest.stage == FrameStage::Assemble {
        let samples = assembled
            .description
            .tables
            .iter()
            .flat_map(|table| {
                table
                    .columns
                    .iter()
                    .filter(|column| {
                        column.dtype == sdm_contract::ops::Dtype::String
                            && column.stype == sdm_contract::ops::Stype::Categorical
                    })
                    .map(|column| FrameSampleColumn {
                        table: table.name.clone(),
                        column: column.clone(),
                    })
            })
            .collect();
        return finish_inference(
            &asker,
            FrameInference::Schema(Box::new(FrameSchema {
                description: assembled.description,
                samples,
                key_notices: notices
                    .into_iter()
                    .filter(|notice| notice.sentence != Sentence::FrameCategoricalSample)
                    .collect(),
                notices: assembled.warnings,
            })),
        );
    }
    for warning in notices {
        asker.warn(warning);
    }
    for warning in assembled.warnings {
        asker.warn(warning);
    }
    finish(&asker, assembled.description, None)
}

fn finish_inference(asker: &Asker, value: FrameInference) -> Result<OperationResult, CoreError> {
    let bytes = serde_json::to_vec(&value)
        .map_err(|_| internal("frame inference facts do not serialize"))?;
    let value = asker.attach(Attachment::Bytes(bytes))?;
    Ok(OperationResult::InferredFrames(InferredFrames { value }))
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
