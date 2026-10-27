// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Complete relational results and native ranking grouping observations.

use super::{CallPreparation, RelationalResponsePart, RelationalRowLineage};
use crate::data::AttachmentId;
use crate::ops::Predicted;
use crate::{Detail, Scalar, Sentence};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RelationalResultInput {
    pub bound: CallPreparation,
    pub aggregation: RelationalAggregation,
    #[serde(default = "report_absent")]
    pub report_absent: bool,
}

fn report_absent() -> bool {
    true
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RelationalCallResult {
    pub bound: CallPreparation,
    pub parts: Vec<RelationalResponsePart>,
    pub counted: Vec<Option<usize>>,
    #[serde(default)]
    pub known_empty: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum RelationalAggregation {
    Calls {
        parts: Vec<RelationalResponsePart>,
        counted: Vec<Option<usize>>,
        #[serde(default)]
        known_empty: bool,
    },
    Steps {
        children: Vec<RelationalCallResult>,
        steps: Vec<u32>,
    },
    Ranking {
        children: Vec<RelationalCallResult>,
        positions: Vec<u32>,
        /// Internal JSON carriers; native public values are presented by
        /// candidate position. A native-only carrier is JSON null.
        candidates: Vec<String>,
        top_k: Option<u32>,
        outputs: Vec<String>,
        unscored_candidates: Vec<u32>,
        groups: AttachmentId,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RelationalResultData {
    pub predicted: Predicted,
    /// Final answer_index order, retaining each original response ordinal.
    pub lineage: Vec<RelationalRowLineage>,
    pub answer_details: BTreeMap<String, u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RelationalRankingSource {
    pub children: Vec<RelationalCallResult>,
    pub positions: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RelationalRankItem {
    pub ordinal: usize,
    pub part: usize,
    pub candidate: u32,
    pub row_index: i64,
    pub lineage: RelationalRowLineage,
    pub unkeyed_id: Option<String>,
    pub probabilities: Option<Vec<(String, Scalar)>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RelationalRankPopulation {
    pub items: Vec<RelationalRankItem>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RelationalRankConfidence {
    pub score: Scalar,
}

/// Actual returned native dict values, expressed as earliest source ordinals.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RelationalRankMembership {
    pub ordinal: usize,
    pub score_first: usize,
    pub asked_first: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RelationalRankGroup {
    pub members: Vec<usize>,
    /// The value returned by the final actual asked_about[where] lookup.
    pub representative: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RelationalRankGroups {
    pub membership: Vec<RelationalRankMembership>,
    pub groups: Vec<RelationalRankGroup>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ValidatedRelationalRanking {
    pub outputs: Vec<String>,
    pub part_outputs: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum RelationalPartScope {
    Steps {
        step: u32,
        total: u32,
        #[serde(default)]
        window_outran: bool,
    },
    Ranking {
        position: u32,
        total: u32,
        #[serde(default)]
        completed: Vec<u32>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RelationalPartDescription {
    pub sentence: Sentence,
    pub fields: BTreeMap<String, Detail>,
    pub ending: RelationalSentenceEnding,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RelationalSentenceEnding {
    pub terminal_characters: String,
    pub period: String,
}

fn response_attachments(parts: &[RelationalResponsePart]) -> Vec<AttachmentId> {
    parts
        .iter()
        .flat_map(|part| {
            [
                part.data.response,
                part.data.rows,
                part.rows,
                part.target_rows,
            ]
        })
        .collect()
}

impl RelationalResultInput {
    pub fn attachments(&self) -> Vec<AttachmentId> {
        match &self.aggregation {
            RelationalAggregation::Calls { parts, .. } => response_attachments(parts),
            RelationalAggregation::Steps { children, .. } => children
                .iter()
                .flat_map(|child| response_attachments(&child.parts))
                .collect(),
            RelationalAggregation::Ranking {
                children, groups, ..
            } => children
                .iter()
                .flat_map(|child| response_attachments(&child.parts))
                .chain(std::iter::once(*groups))
                .collect(),
        }
    }
}

impl RelationalRankingSource {
    pub fn attachments(&self) -> Vec<AttachmentId> {
        self.children
            .iter()
            .flat_map(|child| response_attachments(&child.parts))
            .collect()
    }
}
