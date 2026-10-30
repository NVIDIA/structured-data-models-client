// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Reached native cell observations, not callbacks or resumable instructions.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use crate::{Scalar, ops::Dtype};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum TableHalf { Context, Predict }

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CellIdentity {
    pub table: TableHalf,
    pub column: String,
    pub row: usize,
    pub dtype: Dtype,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum CellObservation {
    ScalarAdmission { shape: CellShape, native_type: String },
    Missingness { missing: bool },
    NumericFiniteness { finite: bool },
    IntegerCoercion { digits: Option<String>, fractional: bool },
    FloatCoercion { bits: Option<[u8; 8]> },
    FloatPrecision { exceeds: bool },
    FloatExactness { bits: [u8; 8], integral_matches: Option<bool>, large_fraction: bool },
    BooleanCoercion { value: bool },
    TextCoercion { value: String },
    TimestampCoercion { value: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum CellShape { Scalar, Bytes, Mapping, Container }

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DirectCellInput {
    pub identity: CellIdentity,
    pub observation: CellObservation,
}

/// A checked native fact or a completed encoded cell. No member directs the
/// host to execute an action; the native dtype branch owns its fixed sequence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CheckedDirectCell {
    pub identity: CellIdentity,
    pub value: Option<Scalar>,
    pub exact_integer_limit: u64,
    pub integer_min: String,
    pub integer_max: String,
}

/// Decode a completed cell carried inside a direct encoded-column attachment.
pub fn encoded_scalar(text: &str) -> Result<Scalar, crate::CoreError> {
    serde_json::from_str(text).map_err(|_| crate::CoreError::protocol(
        crate::Sentence::InvalidMessage, "an encoded direct cell is not a contract scalar"))
}
