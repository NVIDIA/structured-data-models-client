// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Live declared meaning of one reached native source column.
//! Native owners stay local; no future column declaration is captured here.

use super::*;
use crate::ops::{HeldColumn, HeldSchema, HeldTable};

/// Actual native storage and inference observations, without schema policy.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FrameColumnFacts {
    pub object: bool,
    pub string: bool,
    pub arrow_decimal: bool,
    pub timestamp: bool,
    pub inferred: Option<FrameObservedContent>,
}

/// The observed scalar content, adapted after one reached native inference.
/// None at the containing field means inference was not performed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum FrameObservedContent {
    Text,
    Decimal,
    Temporal,
    Other,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FrameColumnInputs {
    pub table: String,
    pub source: Relation,
    pub column: HeldColumn,
    pub designated_time: bool,
    pub facts: FrameColumnFacts,
}

/// The resolved logical type of one source column. These meanings select
/// no native callback and carry no execution position.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum FrameColumnKind {
    Original,
    DecimalFloat,
    DecimalInteger,
    DecimalString,
    TextTime,
    TextFeatureTime,
    NativeTime,
    TypedTime,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FrameColumnMeaning {
    pub input: FrameColumnInputs,
    pub kind: FrameColumnKind,
}

/// Complete native fractional-mask observations before the whole-value map.
/// Repr is observed only for the first actual fractional element.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FrameIntegerColumn {
    pub meaning: FrameColumnMeaning,
    pub fractional: bool,
    pub example: Option<String>,
}

/// A complete independently useful typed source column, before sanitization.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TypedFrameSourceColumn {
    pub values: AttachmentId,
    pub strings: Option<StringEncoding>,
}

/// Physical native source types read during table construction, after all
/// tables have completed their native declared casts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FrameTableColumnFacts {
    pub name: String,
    pub declared: Option<FramePhysicalDeclaration>,
    pub values: FrameConstructorValues,
    pub meaning: FrameConstructedColumn,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FrameTableInputs {
    pub table: HeldTable,
    pub native: Vec<FrameTableColumnFacts>,
    /// Actual reached native container empty observation, before dtype reads.
    pub empty: bool,
}

/// Cast-time table dtype facts and their ordered physical column identities.
/// Later source composition takes logical metadata from live declarations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FrameTableMeaning {
    pub table: HeldTable,
    pub constructed: Vec<FrameConstructedColumn>,
}

/// Graph declarations composed from cast-time dtypes and live logical metadata
/// after physical table admission, before GraphStore-style native sanitization.
/// These same fixed graph facts identify its retained owner.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FrameSourceMeaning {
    pub schema: HeldSchema,
}
