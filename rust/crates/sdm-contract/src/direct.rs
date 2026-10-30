// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Direct request descriptions, ordered columns and decoded native values.

use crate::Scalar;
use crate::data::AttachmentId;
use crate::ops::{Dtype, HeldSchema, Stype};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

pub mod cell;
pub mod dtype;
pub mod native_response;
pub mod schema;
pub mod storage;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum FloatStorage {
    NanMissing,
    ValidityMasked,
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
    Array(Vec<PrimitiveOption>),
    Object(Vec<(String, PrimitiveOption)>),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DirectOptionObservation {
    pub integer: Option<String>,
    pub repeated: bool,
    pub depth: usize,
}

pub const DIRECT_RESPONSE_VERSION: u32 = 2;

/// Owned model response data. The attachments are data, never resume handles.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DecodedDirectResponse {
    pub value: AttachmentId,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ResponseScope {
    pub version: u32,
    pub model: String,
    pub request_id: String,
    pub call_request_id: String,
    pub response_rows: usize,
    pub scoring_rows: usize,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DirectResponseData {
    pub bound: BoundDirectCall,
    pub scope: ResponseScope,
    pub outputs: Vec<String>,
    pub named: bool,
    pub payload: AttachmentId,
    pub response: AttachmentId,
    pub rows: AttachmentId,
}
/// One actual decoded item with pure key/occurrence facts derived once.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DecodedDirectRow {
    pub ordinal: usize,
    pub row_index: String,
    pub id: Option<String>,
    pub key: Option<Scalar>,
    pub raw_in_batch: bool,
    pub prior_ordinal: Option<usize>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CheckedDirectRow {
    pub scope: ResponseScope,
    pub row: DecodedDirectRow,
    pub id: DirectRowId,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "representation", rename_all = "snake_case", deny_unknown_fields)]
pub enum DirectRowId {
    Unkeyed,
    Absent,
    LengthLimitedText,
    NativeJsonText {
        value: String,
    },
    JsonValue {
        value: BuiltinValue,
        members: Option<Vec<BuiltinValue>>,
    },
    NativeLiteralText {
        value: String,
    },
}
/// Native parsed builtin facts. Nonprimitive members do not acquire display text.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum BuiltinValue {
    Null,
    Boolean { value: bool },
    Integer { digits: String, radix: IntegerRadix },
    Float { bits: [u8; 8] },
    Unicode { points: Vec<u32> },
    Bytes,
    Complex,
    List,
    Tuple,
    Dictionary,
    Set,
    Ellipsis,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum IntegerRadix {
    Decimal,
    Hexadecimal,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum BuiltinFailureClass {
    SyntaxError,
    ValueError,
    TypeError,
    MemoryError,
    RecursionError,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct NativeFailure {
    pub ancestry: Vec<BuiltinFailureClass>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "outcome", rename_all = "snake_case", deny_unknown_fields)]
pub enum NativeParsedValue {
    Value {
        value: BuiltinValue,
        members: Option<Vec<BuiltinValue>>,
    },
    Failed {
        failure: NativeFailure,
    },
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct LiteralObservation {
    pub scope: ResponseScope,
    pub ordinal: usize,
    pub text: String,
    pub value: NativeParsedValue,
}
/// An actual native JSON conversion of one reached eligible ID.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct JsonObservation {
    pub scope: ResponseScope,
    pub ordinal: usize,
    pub text: String,
    pub value: NativeParsedValue,
}
/// Native JSON data with the core's checked representation and catch policy.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct InterpretedDirectRow {
    pub checked: CheckedDirectRow,
    pub json: JsonObservation,
}
/// A selected member of a checked row, independently useful builtin data.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SelectedDirectMember {
    pub checked: CheckedDirectRow,
    pub json: JsonObservation,
    pub literal: Option<LiteralObservation>,
    pub member: Option<BuiltinValue>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "outcome", rename_all = "snake_case", deny_unknown_fields)]
pub enum NativeStringValue {
    Value { points: Vec<u32> },
    Failed { failure: NativeFailure },
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct StringObservation {
    pub scope: ResponseScope,
    pub ordinal: usize,
    pub value: NativeStringValue,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AttributedDirectRow {
    pub row: DecodedDirectRow,
    pub json: Option<JsonObservation>,
    pub literal: Option<LiteralObservation>,
    pub string: Option<StringObservation>,
    pub id: Scalar,
}
/// Consecutive pure rows plus at most one reached native representation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CheckedDirectRows {
    pub scope: ResponseScope,
    pub rows: Vec<AttributedDirectRow>,
    pub row: Option<CheckedDirectRow>,
}

/// An input-sized direct value, carried as typed JSON bytes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DirectInput {
    pub input: AttachmentId,
}

/// Two admitted column populations with separate explicit row counts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DirectFrames {
    pub input: AttachmentId,
    pub context: AttachmentId,
    pub predict: AttachmentId,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DirectColumnInput {
    pub input: AttachmentId,
    pub values: AttachmentId,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DirectColumnEncoding {
    pub column: DirectHeader,
    pub dtype: Dtype,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct EncodedDirectColumn {
    pub values: Vec<Scalar>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DirectValue {
    pub value: AttachmentId,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum FrameKind {
    Frame,
    Dict,
    Series,
    Other { native_type: String },
}

/// Headers do not contain or inspect row-index labels.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DirectHeader {
    pub rows: usize,
    pub columns: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DirectShape {
    #[serde(default)]
    pub target_containers: bool,
    pub context_kind: FrameKind,
    pub predict_kind: FrameKind,
    pub related: Vec<(String, FrameKind)>,
    pub task: Option<String>,
    pub target: String,
    pub entity_table: String,
    pub context: DirectHeader,
    pub predict: DirectHeader,
}

/// Native schema observations that the existing held schema does not carry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ColumnFlags {
    pub name: String,
    pub untyped: bool,
    pub stype_declared: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DirectPreparation {
    pub shape: DirectShape,
    pub schema: Option<HeldSchema>,
    pub flags: Vec<ColumnFlags>,
}

/// Model-neutral constructor output; the host retains its original objects.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DirectDescription {
    pub schema: HeldSchema,
    pub task: String,
    pub target: String,
    pub entity_table: String,
    pub target_dtype: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DirectCallFacts {
    pub model: String,
    pub outputs: Option<Vec<String>>,
    pub kind: String,
    pub anchored: bool,
    pub relationships: bool,
    pub tables: usize,
    pub context_rows: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct BoundDirectCall {
    pub facts: DirectCallFacts,
    pub model: String,
    pub outputs: Vec<String>,
    pub named: bool,
    pub kind: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TabularInput {
    pub bound: BoundDirectCall,
    pub schema: HeldSchema,
    pub flags: Vec<ColumnFlags>,
    pub entity_table: String,
    pub target: String,
    pub kind: String,
    pub context: DirectHeader,
    pub predict: DirectHeader,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct EncodedTableShape {
    pub columns: Vec<String>,
    pub rows: usize,
}

/// Fully encoded wire data before the native inference conversion is reached.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct EncodedDirectBody {
    pub bound: BoundDirectCall,
    pub columns: TabularColumns,
    pub kind: String,
    pub primary_key: Vec<String>,
    pub context_shape: EncodedTableShape,
    pub predict_shape: EncodedTableShape,
    pub payload: AttachmentId,
}

/// A direct predict's input is either a known empty call or an encoded body.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum DirectPrediction {
    DeferredOptions {
        body: EncodedDirectBody,
        inference_present: bool,
    },
    Empty {
        bound: BoundDirectCall,
        target_dtype: String,
    },
    Encoded {
        body: EncodedDirectBody,
        inference: DirectInference,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DirectModelInput {
    pub input: AttachmentId,
    /// The encoded body, absent for a known-empty call.
    pub payload: Option<AttachmentId>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DirectRowsInput {
    pub scope: ResponseScope,
    pub rows: Vec<DecodedDirectRow>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DirectJsonInput {
    pub checked: CheckedDirectRow,
    pub json: JsonObservation,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DirectMemberInput {
    pub interpreted: InterpretedDirectRow,
    pub literal: Option<LiteralObservation>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DirectAttributionInput {
    pub selected: SelectedDirectMember,
    pub string: Option<StringObservation>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DirectLayoutInput {
    pub data: DirectResponseData,
    pub attributed: Vec<AttributedDirectRow>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct LayoutDirect {
    pub input: AttachmentId,
    pub payload: AttachmentId,
    pub response: AttachmentId,
}

/// Resolved wire metadata, before native warning delivery and row encoding.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ResolvedTabular {
    pub bound: BoundDirectCall,
    pub columns: TabularColumns,
}

/// Separately resolved wire types and the live schema after native observation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ObservedTabularInput {
    pub input: TabularInput,
    pub dtypes: Vec<(String, Dtype)>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DirectDtypeDeclarations {
    pub dtypes: Vec<(String, Option<Dtype>)>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ObservedKeyPair {
    pub context: DirectHeader,
    pub predict: DirectHeader,
    pub target: String,
    pub scoring_rows: usize,
}

/// Live post-advice observations paired with already-resolved wire metadata.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TabularEncoding {
    pub resolved: ResolvedTabular,
    pub primary_key: Vec<String>,
    pub kind: String,
    pub context: DirectHeader,
    pub predict: DirectHeader,
}
