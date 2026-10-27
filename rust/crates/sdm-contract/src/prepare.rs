// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Closed direct-request preparation stages and useful call-local values.
//! Native equality facts describe ordered headers; the core owns validation.

use crate::data::AttachmentId;
use crate::ops::{Given, HeldSchema, SourceDescription};
use crate::{ColumnType, Scalar};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

pub const DIRECT_PREPARED_VERSION: u32 = 1;
pub const CALL_PREPARATION_VERSION: u32 = 1;

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
    },
    TaskAccepted {
        version: u32,
        model: String,
        outputs: Vec<String>,
        named: bool,
        kind: String,
    },
    Bound {
        version: u32,
        model: String,
        outputs: Vec<String>,
        named: bool,
        kind: String,
        context_rows: usize,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "stage", rename_all = "snake_case", deny_unknown_fields)]
pub enum Preparation {
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

/// Reached predict input, distinct from gathered HeldRequest/state payloads.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DirectPredict {
    pub bound: CallPreparation,
    #[serde(default)]
    pub empty: bool,
    #[serde(default)]
    pub context: Option<AttachmentId>,
    #[serde(default)]
    pub predict: Option<AttachmentId>,
    #[serde(default)]
    pub context_header: Option<FrameHeader>,
    pub predict_header: FrameHeader,
    pub schema: HeldSchema,
    pub entity_table: String,
    #[serde(default)]
    pub target: String,
    #[serde(default)]
    pub task: String,
    #[serde(default)]
    pub inference: DirectInference,
}

/// The closed data value answered by one fixed preparation stage.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
pub enum PreparedValue {
    Arguments(ConstructorArguments),
    Headers(ConstructorHeaders),
    Request(PreparedDirectRequest),
    Call(CallPreparation),
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
