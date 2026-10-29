// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Useful provider-backed source, population and completed label data.
//! These values describe actual source meaning and rows, never callbacks,
//! an operation list, a suspended computation or a program cursor.

use crate::ColumnType;
use crate::data::strings::StringEncoding;
use crate::data::{AttachmentId, ProviderError, Relation};
use crate::frames::materialize::{
    FrameBatchRange, FrameNativeColumnRef, FrameObservedContent, FramePhysicalDeclaration,
    PreparedFrameLabelGroups, RestoredFrameLabels,
};
use crate::ops::{AttachmentKind, Materialize, Materialized, Operation};
use crate::prepare::NativeFailure;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PrepareMaterialize {
    pub input: AttachmentId,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PreparedMaterialize {
    pub value: AttachmentId,
}

/// Actual counted task lookup, bound to the issued logical read.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PreparedLookupRows {
    pub owner: AttachmentId,
    pub read: crate::run::ReplyId,
    pub source: Relation,
    pub key: String,
    pub range: FrameBatchRange,
    pub context_rows: u64,
    pub scored_rows: u64,
}

/// Original captured declaration order, independently of fetched projection order.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct MaterializeSourceTable {
    pub source: Relation,
    /// Exact resolved/pinned read, including its captured projection.
    pub read: Relation,
    pub columns: Vec<String>,
    /// Resolved temporal column meaning, independently of designated time.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub temporal_columns: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub time_column: Option<String>,
}

/// The actual data-read kind reported after an incomplete native source read.
/// This diagnostic fact contains no borrowed or consumed key attachments.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SourceReadKind {
    SampleEntities,
    Neighbors,
    Lookup,
    Scan,
}

impl SourceReadKind {
    pub fn name(self) -> &'static str {
        match self {
            Self::SampleEntities => "sample_entities",
            Self::Neighbors => "neighbors",
            Self::Lookup => "lookup",
            Self::Scan => "scan",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct MaterializeSourceRead {
    pub kind: SourceReadKind,
    pub source: Relation,
    pub columns: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PreparedMaterializeSource {
    pub source: String,
    pub identity: String,
    pub owner: AttachmentId,
    pub target: FrameNativeColumnRef,
    pub tables: Vec<MaterializeSourceTable>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PreparedMaterializePopulation {
    pub source: String,
    pub identity: String,
    pub owner: AttachmentId,
    pub rows: u64,
    pub batch_rows: u32,
    pub target: FrameNativeColumnRef,
}

/// The same neutral ordered group shape used by the local data host.
pub type PreparedMaterializeGroups = PreparedFrameLabelGroups;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PreparedMaterializeBatch {
    pub population: String,
    pub range: FrameBatchRange,
    pub owner: AttachmentId,
    pub complete: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum TargetColumnForm {
    Plain,
    Filtered,
    /// An actual condition column, with full row extent and no primitive
    /// integer cast, target window or aggregation-result restoration.
    Condition,
}

/// Which completed source data supplies this actual column view.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ConditionSourceRole {
    Entity,
    #[default]
    Windowed,
}

impl ConditionSourceRole {
    fn is_windowed(&self) -> bool {
        *self == Self::Windowed
    }
}

/// Completed native column data retained in its actual condition order.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CompletedConditionValue {
    pub column: FrameNativeColumnRef,
    pub owner: AttachmentId,
}

/// Ordered source views and actual completed target-value owners.
/// Child views contain no eligibility history of their own.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PreparedConditionValues {
    pub columns: Vec<PreparedLabelInputs>,
    pub values: Vec<CompletedConditionValue>,
}

/// Complete reached input rows before native target conversion or grouping.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PreparedLabelInputs {
    pub population: String,
    pub range: FrameBatchRange,
    pub owner: AttachmentId,
    /// Actual historical logical read within the producing SQL Run.
    /// Frame sources already bind their original rows through the graph.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_read: Option<crate::run::ReplyId>,
    pub target: FrameNativeColumnRef,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub time: Option<FrameNativeColumnRef>,
    pub form: TargetColumnForm,
    /// Resolved occurrence order, including repeated native reads of one column.
    pub condition_columns: Vec<FrameNativeColumnRef>,
    #[serde(default, skip_serializing_if = "ConditionSourceRole::is_windowed")]
    pub source_role: ConditionSourceRole,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eligibility: Option<PreparedConditionValues>,
}

impl PreparedLabelInputs {
    pub fn attachments(&self) -> Vec<(AttachmentId, AttachmentKind)> {
        let mut out = vec![(self.owner, AttachmentKind::LabelInputs)];
        if let Some(eligibility) = &self.eligibility {
            for column in &eligibility.columns {
                out.extend(column.attachments());
            }
            out.extend(
                eligibility
                    .values
                    .iter()
                    .map(|value| (value.owner, AttachmentKind::TargetColumn)),
            );
        }
        out
    }
}

/// Historical mathematical meaning of the actual completed target column.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum TargetDerivation {
    Column,
    Conditioned,
    Filtered,
    Windowed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PreparedTargetColumn {
    pub input: PreparedLabelInputs,
    pub owner: AttachmentId,
    pub derivation: TargetDerivation,
    pub rows: u64,
    /// Actual completed native columns, not a next-operand cursor.
    pub condition_values: Vec<NativeConditionColumn>,
}

/// Actual target values with ordered fetched-row positions and independent
/// complete native missingness, whose extent is fixed by the predecessor.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct NativeTargetValues {
    pub rows: AttachmentId,
    pub presence: Option<AttachmentId>,
    pub encoding: Option<StringEncoding>,
    /// Full current values of the source's captured time owner at the
    /// reached pre-window boundary, independently of target row selection.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub times: Option<AttachmentId>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct NativeConditionColumn {
    pub column: FrameNativeColumnRef,
    pub converted: NativeTargetValues,
}

/// Actual raw container form, not the provider or native library's identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "storage", rename_all = "snake_case", deny_unknown_fields)]
pub enum MaterializeSourceFacts {
    CellRows {},
    /// Actual content of a reached post-construction native scalar column.
    ColumnValues {
        inferred: FrameObservedContent,
        /// The same native skip-null inference reported no scalar content.
        inferred_empty: bool,
    },
    /// Reached native datetime conversion before timezone/unit finishing.
    TimeConversion {
        rows: u64,
        unread: u64,
        examples: String,
    },
    TypedColumns {
        declaration: Option<FramePhysicalDeclaration>,
    },
}

/// Mathematical scalar column meaning selected for its fixed native converter.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "meaning", rename_all = "snake_case", deny_unknown_fields)]
pub enum MaterializeScalarMeaning {
    Original {},
    Nullable {
        physical: ColumnType,
    },
    Decimal {
        physical: ColumnType,
    },
    /// Execute the native cast while preserving the exact typed mathematical input.
    ExactDecimal {
        physical: ColumnType,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct MaterializeColumnMeaning {
    pub source: String,
    pub column: FrameNativeColumnRef,
    pub input: MaterializeSourceFacts,
    pub scalar: MaterializeScalarMeaning,
    /// Completed conversion notice, presented at the native conversion position.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub notice: Option<crate::error::CoreWarning>,
}

/// Native ancestry observed without message, arguments or display callbacks.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct MaterializeCastFailure {
    pub builtin: NativeFailure,
    pub arithmetic: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "stage", rename_all = "snake_case", deny_unknown_fields)]
pub enum MaterializePreparation {
    Task {
        request: Materialize,
    },
    Population {
        source: PreparedMaterializeSource,
    },
    LookupRows {
        lookup: PreparedLookupRows,
        helpers: AttachmentId,
    },
    SourceColumn {
        source: PreparedMaterializeSource,
        column: FrameNativeColumnRef,
        input: MaterializeSourceFacts,
    },
    SourceFallback {
        source: PreparedMaterializeSource,
        meaning: MaterializeColumnMeaning,
        failure: MaterializeCastFailure,
    },
    SourceFailure {
        source: PreparedMaterializeSource,
        read: MaterializeSourceRead,
        error: ProviderError,
    },
    LabelInputs {
        population: PreparedMaterializePopulation,
        range: FrameBatchRange,
        previous: Option<PreparedMaterializeBatch>,
    },
    TargetColumn {
        inputs: PreparedLabelInputs,
        converted: NativeTargetValues,
    },
    EligibilityColumn {
        inputs: PreparedLabelInputs,
        column: PreparedTargetColumn,
    },
    ConditionColumn {
        column: PreparedTargetColumn,
        operand: NativeConditionColumn,
    },
    FilteredTarget {
        column: PreparedTargetColumn,
        converted: NativeTargetValues,
    },
    WindowedTarget {
        column: PreparedTargetColumn,
        converted: NativeTargetValues,
    },
    NativeLabelGroups {
        column: PreparedTargetColumn,
    },
    RestoreLabelGroups {
        groups: PreparedMaterializeGroups,
        restored: RestoredFrameLabels,
    },
    WhatIfColumn {
        groups: PreparedMaterializeGroups,
        column: PreparedTargetColumn,
    },
    CompleteLabelBatch {
        population: PreparedMaterializePopulation,
        groups: PreparedMaterializeGroups,
        restored: Option<RestoredFrameLabels>,
    },
    Request {
        population: PreparedMaterializePopulation,
        batches: Vec<PreparedMaterializeBatch>,
        /// Mutable native configuration at its reached post-label position.
        values: Option<AttachmentId>,
    },
}

impl MaterializePreparation {
    pub fn attachments(&self) -> Vec<(AttachmentId, AttachmentKind)> {
        use AttachmentKind as Kind;
        match self {
            Self::EligibilityColumn { inputs, column } => {
                let mut out = inputs.attachments();
                out.extend(
                    crate::frames::materialize::FramePreparation::NativeLabelGroups {
                        column: column.clone(),
                    }
                    .attachments(),
                );
                out
            }
            Self::LookupRows { lookup, helpers } => {
                vec![(lookup.owner, Kind::LookupRows), (*helpers, Kind::Arrow)]
            }
            Self::Task { request } => Operation::Materialize(request.clone()).attachments(),
            Self::Population { source }
            | Self::SourceColumn { source, .. }
            | Self::SourceFallback { source, .. }
            | Self::SourceFailure { source, .. } => vec![(source.owner, Kind::MaterializeSource)],
            Self::LabelInputs {
                population,
                previous,
                ..
            } => {
                let mut ids = vec![(population.owner, Kind::MaterializePopulation)];
                ids.extend(
                    previous
                        .iter()
                        .map(|value| (value.owner, Kind::MaterializeBatch)),
                );
                ids
            }
            Self::TargetColumn { inputs, converted } => {
                crate::frames::materialize::FramePreparation::TargetColumn {
                    inputs: inputs.clone(),
                    converted: converted.clone(),
                }
                .attachments()
            }
            Self::ConditionColumn { column, operand } => {
                crate::frames::materialize::FramePreparation::ConditionColumn {
                    column: column.clone(),
                    operand: operand.clone(),
                }
                .attachments()
            }
            Self::FilteredTarget { column, converted }
            | Self::WindowedTarget { column, converted } => {
                crate::frames::materialize::FramePreparation::WindowedTarget {
                    column: column.clone(),
                    converted: converted.clone(),
                }
                .attachments()
            }
            Self::NativeLabelGroups { column } => {
                crate::frames::materialize::FramePreparation::NativeLabelGroups {
                    column: column.clone(),
                }
                .attachments()
            }
            Self::RestoreLabelGroups { groups, restored } => {
                let mut out = groups.attachments(Kind::MaterializeGroups);
                out.push((restored.rows, Kind::Arrow));
                out
            }
            Self::WhatIfColumn { groups, column } => {
                let mut out = groups.attachments(Kind::MaterializeGroups);
                out.extend(
                    crate::frames::materialize::FramePreparation::NativeLabelGroups {
                        column: column.clone(),
                    }
                    .attachments(),
                );
                out
            }
            Self::CompleteLabelBatch {
                population,
                groups,
                restored,
            } => {
                let mut out = vec![(population.owner, Kind::MaterializePopulation)];
                out.extend(groups.attachments(Kind::MaterializeGroups));
                out.extend(restored.iter().map(|value| (value.rows, Kind::Arrow)));
                out
            }
            Self::Request {
                population,
                batches,
                values,
            } => {
                let mut ids = vec![(population.owner, Kind::MaterializePopulation)];
                ids.extend(
                    batches
                        .iter()
                        .map(|value| (value.owner, Kind::MaterializeBatch)),
                );
                ids.extend(values.map(|id| (id, Kind::Bytes)));
                ids
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(
    tag = "value",
    content = "data",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum PreparedMaterializeValue {
    LookupRows { positions: AttachmentId },
    Source(PreparedMaterializeSource),
    Population(PreparedMaterializePopulation),
    ColumnMeaning(MaterializeColumnMeaning),
    LabelGroups(PreparedMaterializeGroups),
    LabelInputs(PreparedLabelInputs),
    TargetColumn(PreparedTargetColumn),
    LabelBatch(PreparedMaterializeBatch),
    Request(Materialized),
}
