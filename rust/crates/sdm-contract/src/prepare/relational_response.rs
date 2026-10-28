// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A decoded relational answer and its native source-row lineage.
//!
//! Source ordinals name rows in this response's retained native envelope.
//! They are neither object identities nor reusable cross-call handles.

use crate::Scalar;
use crate::data::AttachmentId;
use crate::scalar::native_value::NativeValue;
use crate::scalar::unicode::UnicodeText;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

pub const RELATIONAL_RESPONSE_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RelationalResponseScope {
    pub version: u32,
    pub call_request_id: String,
    pub model: String,
    pub request_id: String,
    pub response_rows: usize,
}

/// Complete owned decoded response and native-presentation rows.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RelationalResponseData {
    pub scope: RelationalResponseScope,
    pub response: AttachmentId,
    pub rows: AttachmentId,
    pub entity_reference_prefix: String,
    pub forecast_kind: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RelationalReferenceColumns {
    pub prefix_matches: Vec<bool>,
    pub exhausted: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RelationalReferenceColumn {
    pub ordinal: Option<usize>,
}

/// Generated-decoder values, before native attribution or declared casts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DecodedRelationalRow {
    pub ordinal: usize,
    pub row_index: String,
    pub id: Option<String>,
    pub forecast_step: Option<String>,
    /// Earlier answer to the same (row, step), independent of row validity.
    pub prior_ordinal: Option<usize>,
    pub prediction: NativeValue,
    pub probabilities: Option<Vec<(String, Scalar)>>,
}

/// Native table facts read only after this response was decoded.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RelationalAttribution {
    pub scope: RelationalResponseScope,
    /// None means the native columns contain no entity reference.
    pub source_rows: Option<usize>,
    pub instances: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct BoundRelationalRow {
    pub attribution: RelationalAttribution,
    pub row: DecodedRelationalRow,
    pub source_row: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CheckedRelationalRow {
    pub bound: BoundRelationalRow,
    /// Actual reached truth of the retained native task comparison.
    pub forecasting: bool,
}

/// The actual reached native str(instance), not a sent JSON snapshot.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RelationalInstanceText {
    pub ordinal: usize,
    pub points: Vec<u32>,
    /// Actual reached native str(id) != str(instance) truth.
    pub different: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AttributedRelationalRow {
    pub row: DecodedRelationalRow,
    pub source_row: Option<usize>,
    pub instance: Option<RelationalInstanceText>,
    pub forecasting: Option<bool>,
}

/// A checked prefix and, if present, the first reached native forecast truth.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CheckedRelationalRows {
    pub rows: Vec<AttributedRelationalRow>,
    pub forecast: Option<BoundRelationalRow>,
}

/// Native declared class conversions in decoded response order.
/// Opaque entity owners never enter this value; lineage retains those locally.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RelationalTargetRow {
    pub ordinal: usize,
    pub prediction: NativeValue,
    pub probabilities: Option<Vec<RelationalTargetProbability>>,
}

/// The completed native key conversion and its actual reached display.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RelationalTargetProbability {
    /// Original ordered decoded key, independent of its probability value.
    pub ordinal: usize,
    pub value: NativeValue,
    pub name: String,
    pub probability: Scalar,
}

/// Independently useful, complete data for layout and later core merging.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RelationalResponsePart {
    pub data: RelationalResponseData,
    pub attribution: RelationalAttribution,
    pub rows: AttachmentId,
    pub target: RelationalTargetFacts,
    pub target_rows: AttachmentId,
}

/// The actual native kind display and, only for classifications, reached
/// dtype truth and native startswith(tuple) observations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RelationalTargetFacts {
    pub kind: Vec<u32>,
    /// Actual reached substring comparison, including native overrides.
    pub classification: bool,
    pub dtype_truth: Option<bool>,
    pub numeric_prefix: Option<bool>,
    /// Builtin text of the held declaration, only when that read was reached.
    pub dtype: Option<UnicodeText>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RelationalTargetMeaning {
    pub classification: bool,
    pub numeric_prefixes: Vec<String>,
}

/// Every laid-out item names its actual call and original native source row.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RelationalRowLineage {
    pub call_request_id: String,
    pub answer_ordinal: usize,
    pub source_row: Option<usize>,
}
