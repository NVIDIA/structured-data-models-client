// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Exact attachment declarations for the closed frame data products.
//! No arbitrary integers, recursive JSON fields or workflow tokens are read.

use super::*;
use crate::ops::AttachmentKind;

type Named = Vec<(AttachmentId, AttachmentKind)>;

fn graph(value: &PreparedFrameGraph, out: &mut Named) {
    out.push((value.owner, AttachmentKind::FrameGraph));
}
fn labels(value: &LabelledFrameTask, out: &mut Named) {
    graph(&value.graph, out);
    out.extend(value.context.iter().map(|id| (*id, AttachmentKind::Arrow)));
}
fn task(value: &CompletedFrameTask, out: &mut Named) {
    labels(&value.labelled, out);
    out.extend(
        value
            .prediction
            .iter()
            .map(|id| (*id, AttachmentKind::Arrow)),
    );
    out.extend(
        value
            .owner
            .iter()
            .map(|id| (*id, AttachmentKind::FrameTask)),
    );
}
fn plan(value: &FrameRequestPlan, out: &mut Named) {
    task(&value.task, out);
    out.push((value.sampling_values, AttachmentKind::Bytes));
}
fn references(value: &ResolvedFrameReferences, out: &mut Named) {
    plan(&value.plan, out);
    out.push((value.owner, AttachmentKind::FrameReferences));
}
fn converted(value: &FrameReferenceValues, out: &mut Named) {
    match value {
        FrameReferenceValues::CoreKeys { values } => {
            out.extend(values.iter().map(|id| (*id, AttachmentKind::Arrow)))
        }
        FrameReferenceValues::NativeString {
            converted: NativeFrameStrings::Value { values },
        } => out.push((values.values, AttachmentKind::Arrow)),
        FrameReferenceValues::NativeString {
            converted: NativeFrameStrings::Failed { .. } | NativeFrameStrings::CastFailed { .. },
        } => {}
    }
}
fn context(value: &SampledFrameContext, out: &mut Named) {
    plan(&value.plan, out);
    out.push((value.anchors, AttachmentKind::Arrow));
    for table in &value.tables {
        out.extend([
            (table.rows, AttachmentKind::Arrow),
            (table.source_rows, AttachmentKind::Arrow),
        ]);
    }
    out.extend(
        value
            .links
            .iter()
            .map(|link| (link.pairs, AttachmentKind::Arrow)),
    );
}

impl FramePreparation {
    /// The retained graph resource this data product already owns. This
    /// identifies no program position and cannot construct a new owner.
    pub fn retained_owner(&self) -> Option<AttachmentId> {
        match self {
            Self::Graph { retained, .. } => *retained,
            Self::Population { graph, .. } => Some(graph.owner),
            Self::LabelBatch { population, .. } | Self::Labels { population, .. } => {
                Some(population.graph.owner)
            }
            Self::Task { labelled, .. } => Some(labelled.graph.owner),
            Self::Plan { task, .. } => Some(task.labelled.graph.owner),
            Self::References { plan, .. } => Some(plan.task.labelled.graph.owner),
            Self::Context { references, .. } => Some(references.plan.task.labelled.graph.owner),
            Self::Payload { context, .. } => Some(context.plan.task.labelled.graph.owner),
            Self::Request { plan, .. } => Some(plan.task.labelled.graph.owner),
            Self::SourceMeaning { .. }
            | Self::TableMeaning { .. }
            | Self::ColumnMeaning { .. }
            | Self::IntegerColumn { .. }
            | Self::TimeColumn { .. }
            | Self::Question { .. }
            | Self::Questions { .. }
            | Self::AssessFailure { .. }
            | Self::ReportFailure { .. } => None,
        }
    }

    /// Raw graph manifests are resolved separately once, before admission.
    pub fn attachments(&self) -> Named {
        use AttachmentKind::{Arrow, Bytes, FrameGraph};
        let mut out = Vec::new();
        match self {
            Self::SourceMeaning { schema, .. } => out.push((*schema, Bytes)),
            Self::TableMeaning { .. } | Self::ColumnMeaning { .. } | Self::IntegerColumn { .. } => {
            }
            Self::TimeColumn { values, .. } => out.push((values.values, Arrow)),
            Self::Question { schema, query, .. } => out.extend([(*schema, Bytes), (*query, Bytes)]),
            Self::Questions { input } => {
                out.push((input.question.owner, AttachmentKind::FrameQuestion));
                out.extend(input.candidates.map(|id| (id, Bytes)));
                out.extend(input.values.map(|id| (id, Bytes)));
            }
            Self::Graph {
                schema,
                frames,
                checkpoint,
                retained,
                values,
                ..
            } => {
                out.extend([(*schema, Bytes), (*frames, Bytes)]);
                out.extend(
                    checkpoint
                        .iter()
                        .map(|checkpoint| (checkpoint.manifest, Bytes)),
                );
                out.extend(retained.iter().map(|owner| (*owner, FrameGraph)));
                out.extend(values.iter().map(|values| (*values, Bytes)));
            }
            Self::Population {
                graph: value,
                query,
                values,
                ..
            } => {
                graph(value, &mut out);
                out.push((*query, Bytes));
                out.extend(values.iter().map(|values| (*values, Bytes)));
            }
            Self::LabelBatch {
                population,
                previous,
                converted: values,
                ..
            } => {
                graph(&population.graph, &mut out);
                out.push((population.owner, AttachmentKind::FramePopulation));
                out.extend(
                    previous
                        .iter()
                        .map(|batch| (batch.owner, AttachmentKind::FrameLabelBatch)),
                );
                converted(values, &mut out);
            }
            Self::Labels {
                population,
                batches,
            } => {
                graph(&population.graph, &mut out);
                out.push((population.owner, AttachmentKind::FramePopulation));
                out.extend(
                    batches
                        .iter()
                        .map(|batch| (batch.owner, AttachmentKind::FrameLabelBatch)),
                );
            }
            Self::Task {
                labelled,
                prediction,
                values,
            } => {
                labels(labelled, &mut out);
                match prediction {
                    FramePredictionValues::CoreKeys { values } => out.push((*values, Arrow)),
                    FramePredictionValues::NativeString {
                        constructed: NativeFrameConstruction::Value { values },
                    } => out.push((values.values, Arrow)),
                    FramePredictionValues::NativeString {
                        constructed: NativeFrameConstruction::Failed { .. },
                    } => {}
                }
                out.extend(values.iter().map(|values| (*values, Bytes)));
            }
            Self::Plan {
                task: value,
                values,
            } => {
                task(value, &mut out);
                out.extend(values.iter().map(|values| (*values, Bytes)));
            }
            Self::References {
                plan: value,
                converted: values,
            } => {
                plan(value, &mut out);
                converted(values, &mut out);
            }
            Self::Context {
                references: value,
                converted: values,
                ..
            } => {
                references(value, &mut out);
                converted(values, &mut out);
            }
            Self::Payload {
                context: value,
                text,
                values,
            } => {
                context(value, &mut out);
                let text = match text {
                    NativeFrameText::Converted { columns } => columns,
                    NativeFrameText::Failed { completed, .. } => completed,
                };
                out.extend(text.iter().map(|column| (column.values.values, Arrow)));
                out.extend(values.iter().map(|values| (*values, Bytes)));
            }
            Self::Request { plan: value, parts } => {
                plan(value, &mut out);
                out.extend(
                    parts
                        .parts
                        .iter()
                        .map(|part| (part.payload, AttachmentKind::FramePayload)),
                );
            }
            Self::AssessFailure { .. } | Self::ReportFailure { .. } => {}
        }
        out
    }
}
