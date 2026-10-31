// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Merging already normalized prediction results, without another model call.
use crate::data::AttachmentId;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct MergeResults {
    pub input: AttachmentId,
    pub tables: Vec<AttachmentId>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct MergeInput {
    pub parts: Vec<ResultPart>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ResultPart {
    pub model: String,
    pub request_id: String,
    /// Raw JSON preserves numbers beyond fixed-width integers.
    pub metadata: String,
    pub outputs: Vec<String>,
    /// Each candidate's JSON; returned positions index the flattened input list.
    pub unscored_candidates: Vec<String>,
}
