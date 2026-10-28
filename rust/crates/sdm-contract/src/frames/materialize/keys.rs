// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Immutable native identity declarations and separately owned converted keys.

use super::*;
use crate::data::strings::StringData;
use crate::ops::{Dtype, HeldTable, Stype};

/// Neutral storage/content observed when the physical table is constructed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "storage", rename_all = "snake_case", deny_unknown_fields)]
pub enum FrameConstructorValues {
    Typed {
        physical: ColumnType,
    },
    Enumerated,
    /// Native Unicode points; the later owned column states its encoding.
    Text,
    Temporal,
    Duration,
    Elements {
        content: FrameElementContent,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum FrameElementContent {
    Integer,
    Real,
    Boolean,
    Duration,
    Other,
}

/// A core-derived constructed dtype, independent of later caller declarations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FrameConstructedColumn {
    pub name: String,
    pub dtype: Dtype,
    pub stype: Stype,
}

/// Fixed component type, projected independently of overwritten initial stype.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FrameIdentityComponent {
    pub name: String,
    pub dtype: Dtype,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum FrameIdentityCodec {
    EscapedTextV1,
}

/// A complete ordered compound identity, not a conversion cursor or callback.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FrameCompositeDefinition {
    pub table: String,
    pub source: Relation,
    pub columns: Vec<FrameIdentityComponent>,
    pub derived: String,
    pub codec: FrameIdentityCodec,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FrameKeyMeaning {
    pub table: FrameTableMeaning,
    pub primary_key: Vec<String>,
    pub composite: Option<FrameCompositeDefinition>,
}

/// The reached missingness observation of one declared component.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FrameKeyNulls {
    pub definition: FrameCompositeDefinition,
    pub column: String,
    pub missing: bool,
}

/// Complete aligned native identity values in original source row order.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FrameRowIdentity {
    pub definition: FrameCompositeDefinition,
    pub values: StringData,
}

/// The reached same-table declarations after its selected key was assigned.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FrameTableMetadata {
    pub key: FrameKeyMeaning,
    pub table: HeldTable,
    /// Actual completed new derived column; absent for a pre-existing column.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub derived: Option<FrameConstructedColumn>,
}
