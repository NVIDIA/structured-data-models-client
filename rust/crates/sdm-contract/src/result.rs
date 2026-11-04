// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Merging already normalized prediction results, without another model call.
use crate::data::AttachmentId;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct MergeResults {
    pub input: AttachmentId,
    pub tables: Vec<AttachmentId>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct MergeInput {
    pub parts: Vec<ResultPart>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ResultPart {
    pub model: String,
    pub request_id: String,
    /// Raw JSON preserves numbers beyond fixed-width integers.
    pub metadata: String,
    pub outputs: Vec<String>,
    /// Each candidate's JSON; returned positions index the flattened input list.
    pub unscored_candidates: Vec<String>,
}

/// Small observations used to merge owners held by any language, without rows.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PlanResultMerge {
    pub parts: Vec<MergeDescriptor>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct MergeDescriptor {
    pub model: String,
    pub request_id: String,
    pub outputs: Vec<String>,
    pub has_explanation: bool,
    pub names_rows: bool,
    pub names_step: bool,
    pub candidates: Vec<MergeCandidate>,
    #[serde(default)]
    pub gradient_rows: Vec<GradientRecord>,
}

/// Native values, not Python equality callbacks. NaNs compare unequal.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum MergeCandidate {
    /// JSON scalars/containers retain arbitrary precision integer spelling.
    Json {
        value: String,
    },
    Bytes {
        value: Vec<u8>,
    },
    Date {
        ordinal: i32,
    },
    /// Exact ticks since the Unix epoch, with native resolution preserved.
    DateTime {
        ticks: String,
        unit: MergeTimeUnit,
        aware: bool,
    },
    Tuple {
        values: Vec<MergeCandidate>,
    },
    List {
        values: Vec<MergeCandidate>,
    },
    Mapping {
        entries: Vec<(MergeCandidate, MergeCandidate)>,
    },
    Unsupported {
        type_name: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum MergeTimeUnit {
    S,
    Ms,
    Us,
    Ns,
    Ps,
    Fs,
    As,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ResultMergePlan {
    pub single: bool,
    pub batches: usize,
    pub batch_request_ids: Vec<String>,
    pub explanation_part: Option<usize>,
    pub candidate_positions: Vec<u32>,
    pub gradients: Option<GradientCollection>,
    pub names_rows: bool,
    pub names_step: bool,
}

/// Read semantic items from the normalized Arrow result, without another model call.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ReadResultItems {
    pub input: AttachmentId,
    pub table: AttachmentId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ItemEncoding {
    Native,
    Value,
    Mapping,
    Mappings,
    CandidateList,
    CandidateRankings,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ItemField {
    pub name: String,
    pub encoding: ItemEncoding,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ResultItems {
    pub segments: Vec<ResultItemSegment>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ResultItemSegment {
    pub table: AttachmentId,
    pub fields: Vec<ItemField>,
}

/// Score-only input; raw cells and prediction rows remain in the host.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FeatureImportanceInput {
    pub cells: Vec<AttributionCell>,
    pub normalize: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AttributionCell {
    pub table: String,
    pub column: String,
    /// JSON scalar text preserves nonfinite and out-of-range numeric spellings.
    pub score: String,
    pub path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FeatureImportanceOutput {
    pub rows: Vec<FeatureImportanceRow>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FeatureImportanceRow {
    pub table: String,
    pub column: String,
    pub score: f64,
}

/// A language-neutral explanation owner; JSON text preserves exact typed keys/values.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct GradientRecord {
    pub row_index: i64,
    pub id: String,
    pub forecast_step: Option<i64>,
    pub explanation: Option<String>,
}

/// Identity selection is an engine decision; hosts provide presentation adapters.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct GradientCollection {
    pub keyed_by: GradientKeyKind,
    pub rows: Vec<GradientRecord>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all="snake_case")]
pub enum GradientKeyKind { Id, RowIndex }
