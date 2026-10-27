// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Resolved questions before any frame snapshot or graph construction.

use super::*;

/// Native dictionary membership and exact None identity, without evaluating
/// a value or retaining a native object in the shared contract.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FrameQuestionOption {
    pub name: String,
    pub is_none: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FrameQuestionInputs {
    pub schema: AttachmentId,
    pub query: AttachmentId,
    pub candidates: Option<AttachmentId>,
    pub values: Option<AttachmentId>,
    /// Existing unrecognized option names for ordinary-call admission.
    pub options: Vec<String>,
    pub option_facts: Vec<FrameQuestionOption>,
    pub prediction_rows: u64,
    pub limits: FrameLimits,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum FrameQuestionKind {
    Ranking,
    Forecast,
}

/// One independently gathered query, with its original candidate position
/// or one-based forecast step, never a suspended execution position.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FrameQuestion {
    pub query: String,
    pub position: u32,
    /// Candidate's exact existing JSON representation; absent for forecasts.
    pub name: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FrameDecomposition {
    pub kind: FrameQuestionKind,
    pub parts: Vec<FrameQuestion>,
    pub top_k: Option<u32>,
    /// Names of original native option objects retained by each child.
    /// None values and the whole ranking's top_k are absent, as on the
    /// original fresh-child path. No option value is evaluated again here.
    pub child_options: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PreparedFrameQuestions {
    pub query: QueryFacts,
    pub decomposition: Option<FrameDecomposition>,
}

/// The meaning of one independently gathered question for error presentation.
/// This selects no native action and identifies no suspended computation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FrameQuestionScope {
    pub kind: FrameQuestionKind,
    pub position: u32,
    pub parts: u32,
}
