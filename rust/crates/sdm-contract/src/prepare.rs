// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Closed direct-request preparation stages and useful call-local values.
//! Native equality facts describe ordered headers; the core owns validation.

use crate::data::AttachmentId;
use crate::ops::{Dtype, Given, HeldSchema, SourceDescription, Stype};
use crate::{ColumnType, Scalar};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

pub const DIRECT_PREPARED_VERSION: u32 = 1;
pub const CALL_PREPARATION_VERSION: u32 = 2;
pub const DIRECT_OVERLAP_VERSION: u32 = 1;
pub const TABULAR_ENVELOPE_VERSION: u32 = 2;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PrepareRequest {
    pub input: AttachmentId,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PreparedRequest {
    pub value: AttachmentId,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum FrameForm {
    Frame,
    Dict,
    Series,
    Other { native_type: String },
}

/// Native label forms sufficient for this bounded scalar header family.
/// Unencodable text is identified without losing its native ordinal matches;
/// successful schemas requiring it remain unfinished, after prior guards.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum HeaderName {
    Text { value: String },
    NonText,
    UnencodableText,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct HeaderColumn {
    pub ordinal: usize,
    pub name: HeaderName,
    pub physical: Option<ColumnType>,
    /// Native missingness representation, not a casting decision.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub float_storage: Option<FloatStorage>,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum FloatStorage {
    NanMissing,
    ValidityMasked,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FrameHeader {
    pub rows: usize,
    pub columns: Vec<HeaderColumn>,
    pub duplicates: Vec<usize>,
}
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct NativeLookup {
    pub positions: Vec<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unhashable: Option<String>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ConstructorArguments {
    pub version: u32,
    pub task: String,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ConstructorHeaders {
    pub version: u32,
    pub task: String,
    pub context: FrameHeader,
    pub predict: FrameHeader,
    pub target: usize,
    pub predict_context: Vec<Vec<usize>>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PreparedDirectRequest {
    pub version: u32,
    pub task: String,
    pub target: String,
    pub description: SourceDescription,
    pub schema: HeldSchema,
}

/// Native capability observations retained independently of the later task read.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CallShape {
    pub kind: String,
    pub anchored: bool,
    pub relationships: bool,
    pub tables: usize,
}

/// Each call state has useful resolved facts, never a native action/handle.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "phase", rename_all = "snake_case", deny_unknown_fields)]
pub enum CallPreparation {
    ModelResolved {
        version: u32,
        model: String,
    },
    OutputsResolved {
        version: u32,
        model: String,
        outputs: Vec<String>,
        named: bool,
    },
    ShapeAccepted {
        version: u32,
        model: String,
        outputs: Vec<String>,
        named: bool,
        shape: CallShape,
    },
    TaskAccepted {
        version: u32,
        model: String,
        outputs: Vec<String>,
        named: bool,
        shape: CallShape,
        kind: String,
    },
    Bound {
        version: u32,
        model: String,
        outputs: Vec<String>,
        named: bool,
        shape: CallShape,
        kind: String,
        context_rows: usize,
    },
}

/// Fresh observations at the pre-warning one-shot overlap boundary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DirectPrelude {
    pub bound: CallPreparation,
    pub context: AttachmentId,
    pub predict: AttachmentId,
    pub context_header: FrameHeader,
    pub predict_header: FrameHeader,
    pub schema: HeldSchema,
    pub entity_table: String,
    pub target: String,
    pub task: String,
}
/// Core-checked prelude and overlap counts, before adapter reference capture.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DirectOverlap {
    pub version: u32,
    pub bound: CallPreparation,
    pub shown: usize,
    pub predict_rows: usize,
}
/// The adapter captures these references after the earlier overlap advice.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DirectEnvelope {
    pub overlap: DirectOverlap,
    pub context: AttachmentId,
    pub predict: AttachmentId,
    pub context_header: FrameHeader,
    pub predict_header: FrameHeader,
    pub schema: HeldSchema,
    pub entity_table: String,
    pub target: String,
    pub task: String,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TabularColumn {
    pub name: String,
    pub dtype: Dtype,
    pub stype: Stype,
}
/// Core-resolved schema and wire widths, before text advice and row encoding.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TabularColumns {
    pub dtypes: Vec<(String, Dtype)>,
    pub columns: Vec<TabularColumn>,
    pub target: String,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TabularEnvelope {
    pub version: u32,
    pub bound: CallPreparation,
    pub columns: TabularColumns,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "stage", rename_all = "snake_case", deny_unknown_fields)]
pub enum Preparation {
    Overlap {
        input: DirectPrelude,
    },
    Envelope {
        input: DirectEnvelope,
    },
    Arguments {
        context: FrameForm,
        predict: FrameForm,
        related: Vec<(String, FrameForm)>,
        task: Given<String>,
    },
    Headers {
        arguments: ConstructorArguments,
        context: FrameHeader,
        predict: FrameHeader,
        target_context: NativeLookup,
        target_predict: NativeLookup,
        predict_context: Vec<Vec<usize>>,
    },
    Schema {
        headers: ConstructorHeaders,
        context: AttachmentId,
        predict: AttachmentId,
        entity_table: String,
        #[serde(default)]
        schema: Option<HeldSchema>,
    },
    Model {
        model: String,
    },
    Outputs {
        call: CallPreparation,
        #[serde(default)]
        outputs: Option<Vec<String>>,
        #[serde(default)]
        kind: Option<String>,
    },
    Capabilities {
        call: CallPreparation,
        kind: String,
        anchored: bool,
        relationships: bool,
        tables: usize,
    },
    Task {
        call: CallPreparation,
        kind: String,
    },
    Capacity {
        call: CallPreparation,
        rows: usize,
    },
}

/// Source-position body observations after core-selected text advice.
/// Later values/kind may legitimately differ from the checked prelude.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "stage", rename_all = "snake_case", deny_unknown_fields)]
pub enum DirectPredict {
    Empty {
        bound: CallPreparation,
        schema: HeldSchema,
        entity_table: String,
        predict_header: FrameHeader,
    },
    Body {
        envelope: TabularEnvelope,
        context: AttachmentId,
        predict: AttachmentId,
        context_header: FrameHeader,
        predict_header: FrameHeader,
        task: String,
        /// Read from the same captured native table after text advice.
        primary_key: Vec<String>,
        /// Actual builtin native Unicode database version used by literal decoding.
        unicode_version: String,
        #[serde(default)]
        inference: DirectInference,
    },
}

/// The closed data value answered by one fixed preparation stage.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
pub enum PreparedValue {
    Arguments(ConstructorArguments),
    Headers(ConstructorHeaders),
    Request(PreparedDirectRequest),
    Call(CallPreparation),
    Overlap(DirectOverlap),
    Envelope(TabularEnvelope),
}

/// Ordered flat primitive inference values; exact integer digits do not
/// acquire serde_json's machine-integer limit before final wire policy.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DirectInference {
    #[serde(default)]
    pub entries: Vec<(String, PrimitiveOption)>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum PrimitiveOption {
    Integer(String),
    Value(Scalar),
}
