// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Source-less relational inputs. Graph state stays with its native owner;
//! task plans and sampled contexts never round-trip through the host.
use crate::Scalar;
use crate::data::AttachmentId;
use crate::ops::{GivenCandidate, Seed};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PrepareFrameGraph {
    pub schema: AttachmentId,
    /// Ordered table names, as JSON. Exactly one name per Arrow attachment.
    pub names: AttachmentId,
    pub tables: Vec<AttachmentId>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct MaterializeFrames {
    pub graph: AttachmentId,
    /// FrameCall encoded as JSON bytes.
    pub input: AttachmentId,
    pub indices: AttachmentId,
    /// Existing model-option JSON, decoded by the common bounded reader.
    pub values: Option<AttachmentId>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FrameCall {
    pub query: String,
    pub candidates: Option<Vec<GivenCandidate>>,
    pub anchor_time: Option<Scalar>,
    pub seed: Option<Seed>,
    #[serde(default)]
    pub unseeded: bool,
    #[serde(default)]
    pub anchor_entity: bool,
    #[serde(default)]
    pub options: Vec<String>,
    pub most_parts: Option<u32>,
    pub most_entities: Option<u64>,
    pub most_calls: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PreparedFrameGraph {
    pub graph: AttachmentId,
}

/// Inferred frame columns plus observed time populations and caller overrides.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AssembleFrameSchema {
    pub schema: crate::ops::HeldSchema,
    pub times: Vec<crate::direct::schema::SchemaObservation>,
    pub key_counts: std::collections::BTreeMap<String, FrameKeyCount>,
    #[serde(default)]
    pub primary_keys: std::collections::BTreeMap<String, crate::ops::Given<Option<Vec<String>>>>,
    #[serde(default)]
    pub time_columns: std::collections::BTreeMap<String, crate::ops::Given<Option<String>>>,
    pub relationships: Option<Vec<Vec<crate::ops::ColumnRef>>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FrameKeyCount {
    pub rows: u64,
    pub distinct: u64,
    pub missing: u64,
    pub missing_columns: Vec<String>,
    pub example: Option<Vec<Scalar>>,
}
