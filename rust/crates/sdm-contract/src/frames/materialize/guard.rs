// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Closed native diagnostic facts and independently useful classifications.
//! These contain no Python owner, pending action, callback or run position.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::prepare::NativeFailure;
use crate::{Detail, Sentence};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum FrameErrorOrigin {
    Unguarded,
    Engine,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FrameNativeFailure {
    pub builtin: NativeFailure,
    pub key_error: bool,
    pub structured_data_error: bool,
    /// Exact native class identity, not a qualified-name inference.
    pub relational_error: bool,
    pub deepest_module: Option<String>,
    pub python_unicode_database: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum FrameNativeCategory {
    PassThrough,
    Internal,
    /// A genuine TypeError whose signature meaning is not yet classified.
    TypeError,
    CallSignature,
    CallerType,
    CallerData,
}

/// Diagnostic classification over actual ancestry and an optional actual
/// signature observation. No field denotes a native operation to execute.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FramePartPresentation {
    pub sentence: Sentence,
    pub fields: BTreeMap<String, Detail>,
}

/// Final part wording after the original structured native error truth fact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FrameFailureAssessment {
    pub origin: FrameErrorOrigin,
    pub failure: FrameNativeFailure,
    /// Native Unicode points, preserving lone and adjacent surrogate units.
    /// This is present only at the reached TypeError signature boundary.
    pub signature: Option<Vec<u32>>,
    pub category: FrameNativeCategory,
    pub part: Option<FramePartPresentation>,
}

/// Final native observations in their distinct original positions.
/// Native owner identity/tracebacks stay local to the presentation host.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FrameNativeReport {
    pub assessment: FrameFailureAssessment,
    pub native_type: Option<String>,
    pub refusal: Option<Vec<u32>>,
    pub said: Vec<u32>,
    pub dropped_summary: Option<String>,
    pub rows_per_call: Option<u64>,
}
