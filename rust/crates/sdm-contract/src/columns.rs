// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Context association results. These describe evidence, not causal effects.
use crate::data::AttachmentId;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AnalyzeColumns {
    /// Existing wire payload bytes, one per forecast step (one for ordinary analysis).
    pub payloads: Vec<AttachmentId>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ColumnRow {
    pub table: String,
    pub hop: u32,
    pub column: String,
    pub stype: Option<String>,
    pub cohort: String,
    /// Exact JSON scalar, preserving numbers outside fixed-width integer ranges.
    pub class_json: String,
    pub share: f64,
    pub mean: f64,
    pub score: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
pub enum ColumnAnalysis {
    Ready { rows: Vec<ColumnRow> },
    Unsupported { reason: String },
    Failed { reason: String },
}
