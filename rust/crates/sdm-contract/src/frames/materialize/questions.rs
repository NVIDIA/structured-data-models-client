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
    pub question: PreparedFrameQuestion,
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
    /// Ordinary query meaning retained across native graph conversion.
    pub question: Option<AttachmentId>,
    /// Reached immutable early call observations, reused without rereading native option objects.
    pub call_values: Option<AttachmentId>,
    pub query: QueryFacts,
    pub decomposition: Option<FrameDecomposition>,
}

/// Useful resolved query data before any native option/candidate observation.
/// Its exact core-created owner contains the same existing PQL validation,
/// not another query schema or a suspended program position.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PreparedFrameQuestion {
    pub query: QueryFacts,
    pub kind: Option<FrameQuestionKind>,
    pub owner: AttachmentId,
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

/// Reached native overlap observations, in the actual live relationship order.
/// Unavailable guards do not observe future endpoint attributes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "observation", rename_all = "snake_case", deny_unknown_fields)]
pub enum FrameRelationshipEvidence {
    Unavailable {},
    Distinct {
        relationship: [ColumnRef; 2],
        child: u64,
        parent: Option<u64>,
        intersects: Option<bool>,
    },
}

/// The ordered disjoint positions selected from complete reached observations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FrameRelationshipOverlap {
    pub disjoint: Vec<u64>,
}

/// Completed caller metadata after reached key marking, including a later
/// selected consistency refusal. Marks remain observable on failure.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FrameCallerMeaning {
    pub schema: crate::ops::HeldSchema,
    pub diagnostic: Option<crate::CoreError>,
}
