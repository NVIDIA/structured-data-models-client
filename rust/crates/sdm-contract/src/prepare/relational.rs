// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Independently useful facts from native gathered request contents.
//! Native owners and hash/equality effects stay in the presentation host.
//! These values contain no native object handle or workflow position.

use super::*;

/// Concrete fields selected by shared request policy. Native values and
/// shallow container copies remain in the host until the transport encodes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RelationalCallFields {
    pub model: String,
    pub outputs: Vec<String>,
    pub statistic: RelationalStatistic,
    pub inference_options: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "source", rename_all = "snake_case", deny_unknown_fields)]
pub enum RelationalStatistic {
    Absent,
    Retained,
    Default { value: String },
}

/// One strict native JSON encoding at its actual transport boundary.
/// The bytes are authoritative, including native Unicode escaping and
/// members not interpreted by the SDK. No native owner crosses this value.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RelationalEncodedCall {
    pub call: CallPreparation,
    pub body: AttachmentId,
    pub request_id: String,
    pub model_endpoint: Option<crate::ops::ModelEndpoint>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RelationalContextFacts {
    /// The actual native instance-table mapping category at this read.
    pub mapping: bool,
    /// Native rows truth, after the mapping guard, without row traversal.
    pub rows_present: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RelationalTargetRole {
    Classification,
    Regression,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RelationalTaskUse {
    ClassLimit,
    Variation,
    Output,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RelationalTaskFacts {
    pub purpose: RelationalTaskUse,
    pub classification: Option<bool>,
    pub regression_not_equal: Option<bool>,
    pub summarized: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RelationalTaskVocabulary {
    pub classification: String,
    pub regression: String,
    pub summarized: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RelationalTaskMeaning {
    /// Actual reached native str(task.kind), retaining all codepoints.
    pub kind: Vec<u32>,
    pub facts: RelationalTaskFacts,
    pub role: RelationalTargetRole,
    pub summarized: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RelationalTargetUse {
    ClassLimit,
    Variation,
}

/// One original native set observation, not a precomputed semantic count.
/// Source ordinals refer to rows reached by this particular traversal.
/// Native object identities used to establish them never cross this boundary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct NativeTargetEquality {
    pub observed_rows: u64,
    pub eligible_rows: Vec<u64>,
    pub distinct_sources: Vec<u64>,
}

/// Actual target population, useful independently of warning presentation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RelationalTargetPopulation {
    pub meaning: RelationalTaskMeaning,
    pub purpose: RelationalTargetUse,
    /// Absent when the native context does not expose a target population.
    pub equality: Option<NativeTargetEquality>,
    pub classes: Option<u64>,
    pub wire_target_name: Vec<u32>,
}

/// The core-selected meaning of a constant target's displayed name.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum RelationalTargetNaming {
    Column,
    Predicted { value: String },
    Question,
    Answer,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RelationalConstantTarget {
    pub population: RelationalTargetPopulation,
    pub wire_target: bool,
    pub query: Vec<u32>,
    pub naming: RelationalTargetNaming,
    pub question_truth: Option<bool>,
}

/// Model-specific preflight questions over actual native data observations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "data", rename_all = "snake_case", deny_unknown_fields)]
pub enum RelationalPreparation {
    TaskVocabulary,
    Payloads {
        call: CallPreparation,
        present: bool,
    },
    CallFields {
        call: CallPreparation,
        meaning: RelationalTaskMeaning,
        statistic_chosen: Option<bool>,
    },
    Inference {
        call: CallPreparation,
        /// Original native sorted-set refusal population, by ordinal.
        refused: Vec<usize>,
        /// Reached native membership of output_type, only for a refusal.
        statistic_option: Option<bool>,
    },
    CallCounts {
        rows: Vec<Option<usize>>,
    },
    ScoringRows {
        rows: Vec<Option<usize>>,
    },
    Context {
        call: CallPreparation,
        facts: RelationalContextFacts,
    },
    TaskMeaning {
        kind: Vec<u32>,
        facts: RelationalTaskFacts,
    },
    Targets {
        call: CallPreparation,
        meaning: RelationalTaskMeaning,
        purpose: RelationalTargetUse,
        equality: Option<NativeTargetEquality>,
    },
    TargetName {
        population: RelationalTargetPopulation,
        wire_target: bool,
        query: Vec<u32>,
    },
    QuestionMeaning {
        target: RelationalConstantTarget,
        truth: bool,
    },
    Constant {
        call: CallPreparation,
        target: RelationalConstantTarget,
        display: Option<Vec<u32>>,
        classification: bool,
    },
    ExplanationSize {
        call: CallPreparation,
        rows: Vec<u64>,
    },
    ExplanationMode {
        call: CallPreparation,
        modes: Vec<Vec<u32>>,
        fast_equal: Option<bool>,
        context_rows: Vec<u64>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(
    tag = "data",
    content = "value",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum RelationalPrepared {
    Vocabulary(RelationalTaskVocabulary),
    Empty(bool),
    Fields(RelationalCallFields),
    Starts(Vec<usize>),
    Call(CallPreparation),
    Task(RelationalTaskMeaning),
    Targets(RelationalTargetPopulation),
    Name(RelationalConstantTarget),
}
