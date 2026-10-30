// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Reached native response facts. Opaque presentation values stay in the host.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use crate::data::AttachmentId;
use super::{AttributedDirectRow, BoundDirectCall, DecodedDirectRow, ResponseScope};
use crate::Scalar;

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

/// Identity facts after all native fields were consumed and validated.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct NativeRowIdentity {
    pub row_index: String,
    pub id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct NativeResponseIdentity {
    pub model: String,
    pub request_id: String,
    pub rows: Vec<NativeRowIdentity>,
}

/// A prepared, unsent call. No successful model-call accounting is implied.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PreparedDirectCall {
    pub value: AttachmentId,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct NativeDirectCall {
    pub bound: BoundDirectCall,
    pub payload: AttachmentId,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct NativeLayoutInput {
    pub call: NativeDirectCall,
    pub response: NativeResponseIdentity,
    /// None requests identity decoding; Some validates completed attribution.
    pub attributed: Option<Vec<AttributedDirectRow>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct NativeLayoutRequest {
    pub input: AttachmentId,
    pub payload: AttachmentId,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum NativeLayoutResult {
    Decoded { scope: ResponseScope, rows: Vec<DecodedDirectRow> },
    Complete { rows: Vec<(usize, Option<Scalar>)>, restore_dtype: Option<String> },
}

/// Native conversion outcomes; policy selects field replacement.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum RestorationObservation {
    Prediction { same: bool },
    Classes { original_count: usize, renamed_count: usize },
    Probabilities { different: bool },
}

pub fn restore(input: sdm_contract::direct::native_response::RestorationObservation) -> bool {
    use sdm_contract::direct::native_response::RestorationObservation;
    match input {
        RestorationObservation::Prediction { same } => !same,
        RestorationObservation::Classes { original_count, renamed_count } => original_count == renamed_count,
        RestorationObservation::Probabilities { different } => different,
    }
}
