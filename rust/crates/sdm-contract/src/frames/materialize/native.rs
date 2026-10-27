// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Actual reached builtin conversions of useful frame data.
//! These are data outcomes, not host actions, casing choices or Run steps.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::data::AttachmentId;
use crate::data::strings::StringData;
use crate::ops::ColumnRef;
use crate::prepare::NativeFailure;

use super::NativeTextFacts;

/// Prediction data after the closed type chosen by the labeled task.
/// Existing numeric casts remain core-owned; only the new native string
/// family carries an actual reached builtin constructor outcome.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum FramePredictionValues {
    CoreKeys {
        values: AttachmentId,
    },
    NativeString {
        constructed: NativeFrameConstruction,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "outcome", rename_all = "snake_case", deny_unknown_fields)]
pub enum NativeFrameConstruction {
    Value {
        values: StringData,
    },
    Failed {
        failure: NativeFailure,
        message: String,
        diagnostic: NativeConstructionDiagnostic,
    },
}

/// The first native element that failed the reached diagnostic constructor.
/// Its repr is native presentation data, not a core inference from content.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct UncastableFrameInput {
    pub ordinal: u64,
    pub native_repr: String,
}

/// The real stop-at-first diagnostic of a failed whole constructor.
/// Native code does not eagerly convert the unvisited suffix. Core decides
/// eligibility, attribution and the final public error from these facts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "outcome", rename_all = "snake_case", deny_unknown_fields)]
pub enum NativeConstructionDiagnostic {
    NotReached,
    Completed {
        first: Option<UncastableFrameInput>,
        dtype_display: String,
    },
    Failed {
        failure: NativeFailure,
        message: String,
    },
}

/// Actual selected raw strings, before character caps, casing or splitting.
/// Values stay in owned Arrow data, and facts describe this column's actual
/// successful conversion rather than an early global dtype observation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ConvertedFrameText {
    pub column: ColumnRef,
    pub values: StringData,
    pub facts: NativeTextFacts,
}

/// Reached per-context conversion in core-selected feature order. A failed
/// conversion retains its successful prefix for exact order/coherence checks.
/// Failure ancestry reuses the existing fixed native-conversion vocabulary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "outcome", rename_all = "snake_case", deny_unknown_fields)]
pub enum NativeFrameText {
    Converted {
        columns: Vec<ConvertedFrameText>,
    },
    Failed {
        completed: Vec<ConvertedFrameText>,
        column: ColumnRef,
        failure: NativeFailure,
        message: String,
    },
}
