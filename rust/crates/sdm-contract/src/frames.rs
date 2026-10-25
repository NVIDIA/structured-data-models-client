// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Call-local frame bindings, never persistent table identity or policies.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use std::collections::BTreeMap;

use crate::data::{AttachmentId, Relation};
use crate::ops::ColumnRef;

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
    /// Native scalar/sequence form of each explicit key override.
    #[serde(default)]
    pub key_forms: BTreeMap<String, FrameKeyForm>,
    /// The single authoritative ordered explicit link list, or inference.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub relationships: Option<Vec<FrameRelationship>>,
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

/// A native scalar name or ordered key sequence, even of one column.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum FrameKeyForm {
    Scalar,
    Sequence,
}

/// One native relationship argument at its original validation position.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum FrameRelationship {
    Link {
        source: ColumnRef,
        target: ColumnRef,
    },
    /// Native presentation facts only; the core decides this slot is invalid.
    Invalid {
        native_type: String,
        native_repr: String,
    },
}
