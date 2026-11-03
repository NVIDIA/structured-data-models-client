// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Explicit relational tasks with caller-supplied examples, independent of PQL.
use crate::data::{AttachmentId, ProviderChoice};
use crate::ops::Seed;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "source", rename_all = "snake_case", deny_unknown_fields)]
pub enum ExampleSource {
    Frames {
        graph: AttachmentId,
    },
    Sql {
        provider: ProviderChoice,
        schema: AttachmentId,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct MaterializeExamples {
    pub source: ExampleSource,
    pub input: AttachmentId,
    pub context: AttachmentId,
    pub predict: AttachmentId,
    pub inference: Option<AttachmentId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ExampleKind {
    Regression,
    BinaryClassification,
    MulticlassClassification,
    TemporalLinkPrediction,
    Forecasting,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RelationalTask {
    pub kind: ExampleKind,
    pub entity_table: String,
    pub link_target_table: Option<String>,
    /// Positive fixed step in nanoseconds, never a calendar offset.
    pub step_size_ns: Option<i64>,
    pub num_forecasts: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ExampleCall {
    #[serde(default)]
    pub rows_per_call: Option<u64>,
    pub task: RelationalTask,
    pub num_neighbors: Option<Vec<u32>>,
    pub random_seed: Seed,
}

/// Request metadata from explicit task admission, without a synthetic query.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ExampleFacts {
    pub entity_table: String,
    pub kind: ExampleKind,
    pub windowed: bool,
    pub num_forecasts: u32,
}
