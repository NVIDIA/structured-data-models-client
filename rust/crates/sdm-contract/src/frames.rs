// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Call-local frame bindings, never persistent table identity or policies.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::data::{AttachmentId, Relation};

/// The ordered Arrow tables admitted for one describe call.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FrameManifest {
    pub stage: FrameStage,
    pub tables: Vec<FrameInput>,
    /// Native mapping order, separate from SourceSpec's value maps.
    /// Empty/default only when both maps are empty.
    #[serde(default)]
    pub override_order: FrameOverrideOrder,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FrameOverrideOrder {
    #[serde(default)]
    pub primary_keys: Vec<String>,
    #[serde(default)]
    pub time_columns: Vec<String>,
}

/// Fixed constructor stages, never configurable inference policies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum FrameStage {
    Table,
    Schema,
}

/// A stable literal source name bound to this message's Arrow data.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FrameInput {
    pub source: Relation,
    pub rows: AttachmentId,
}

/// A frame source is consumed internally by describe, never by DataLayer.
pub fn not_provider() -> crate::CoreError {
    crate::CoreError::protocol(
        crate::Sentence::InvalidMessage,
        "a frame source is only consumed by the core frame describe operation",
    )
}
