// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Reached native response facts. Opaque presentation values stay in the host.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use crate::data::AttachmentId;
use super::BoundDirectCall;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ResponseField {
    Response,
    ResponseId,
    Model,
    Predictions,
    Row,
    RowId,
    RowIndex,
    ForecastStep,
    Probabilities,
    Probability,
    Scores,
    Score,
    Embeddings,
    Embedding,
    Quantiles,
    Quantile,
    Metadata,
    RowMetadata,
    Explanation,
    Rankings,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ResponseObservation {
    OptionalNull { null: bool },
    Presence { present: bool },
    Object { mapping: bool },
    Text { text: bool },
    WholeKind { boolean: bool, number_family: bool, null: bool },
    WholeFloat { finite: bool, integral: Option<bool> },
    Sequence { string: bool, bytes: bool, mapping: bool, iterable: bool },
    NumberKind { boolean: bool },
    /// None means only a native type/value/overflow failure from float coercion.
    /// Other native exceptions propagate without producing this observation.
    NumberConversion { bits: Option<[u8; 8]> },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct NativeResponseField {
    pub field: ResponseField,
    pub observation: ResponseObservation,
}

/// A refusal selected by core policy before the next native field is read.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ResponseFault {
    Missing,
    Null,
    NotObject,
    NotText,
    NotWholeKind,
    NotWholeValue,
    NotSequence,
    NotNumber,
    NotFinite,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CheckedResponseField {
    pub field: ResponseField,
    pub fault: Option<ResponseFault>,
    pub is_null: bool,
}

/// A prepared, unsent call. Native response owners never cross this boundary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PreparedDirectCall { pub value: AttachmentId }

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct NativeDirectCall { pub bound: BoundDirectCall, pub payload: AttachmentId }

/// Reached facts from the mutable native payload and parsed response.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum NativeLayoutInput {
    Rows { is_list: bool },
    KeyCount { count: usize },
    Positions { indices: Vec<String> },
    Position { index: String, rows: usize, prior: bool },
    IdLength { length: usize },
    Parsed { json: bool, value: super::NativeParsedValue },
    Match { index: usize, different: Option<bool> },
    Replacement { key_is_null: bool },
    Complete { indices: Vec<String>, rows: usize },
    TaskKind { points: Vec<u32> },
    TargetDtype { numeric: bool },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum NativeLayoutResult {
    Decision { selected: bool },
    Positions { prior: Vec<bool> },
    Position { index: usize },
    Parsed { literal: bool, stringify: bool },
    Complete,
}

/// Native conversion outcomes; policy selects field replacement.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum RestorationObservation {
    Prediction { same: bool },
    Classes { original_count: usize, renamed_count: usize },
    Probabilities { different: bool },
}

/// Completed native presence observations for explicitly requested outputs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct NativeOutputQuery {
    pub bound: BoundDirectCall,
    pub rows: usize,
    pub present: Option<Vec<(String, bool)>>,
}
