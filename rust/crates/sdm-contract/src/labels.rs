// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Ground-truth extraction without model requests or feature neighborhoods.
use crate::{
    Scalar,
    data::{AttachmentId, ProviderChoice},
    ops::Seed,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct LabelsFrames {
    pub graph: AttachmentId,
    pub input: AttachmentId,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct LabelsSource {
    pub provider: ProviderChoice,
    pub schema: AttachmentId,
    pub input: AttachmentId,
}

/// At most `size` labels, from at most `size * max_iterations` candidate examples.
/// A forecast selects one entity and returns successive scalar-target anchor rows.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct LabelsCall {
    pub query: String,
    pub size: u64,
    pub max_iterations: u64,
    pub anchor_time: Option<Scalar>,
    #[serde(default)]
    pub anchor_entity: bool,
    pub seed: Option<Seed>,
    #[serde(default)]
    pub unseeded: bool,
}

/// Arrow columns ENTITY, ANCHOR_TIMESTAMP (nanoseconds), TARGET (scalar or identifier list).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct LabelRows {
    pub data: AttachmentId,
}
