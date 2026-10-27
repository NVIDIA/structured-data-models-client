// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Independently useful facts from native gathered request contents.
//! Native owners and hash/equality effects stay in the presentation host.
//! These values contain no native object handle or workflow position.

use super::*;

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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RelationalTaskMeaning {
    /// Actual reached native str(task.kind), retaining all codepoints.
    pub kind: Vec<u32>,
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
}

/// Model-specific preflight questions over actual native data observations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "data", rename_all = "snake_case", deny_unknown_fields)]
pub enum RelationalPreparation {
    Context {
        call: CallPreparation,
        facts: RelationalContextFacts,
    },
    TaskMeaning {
        kind: Vec<u32>,
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
    Constant {
        call: CallPreparation,
        target: RelationalConstantTarget,
        display: Option<Vec<u32>>,
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
#[serde(tag = "data", content = "value", rename_all = "snake_case", deny_unknown_fields)]
pub enum RelationalPrepared {
    Call(CallPreparation),
    Task(RelationalTaskMeaning),
    Targets(RelationalTargetPopulation),
    Name(RelationalConstantTarget),
}
