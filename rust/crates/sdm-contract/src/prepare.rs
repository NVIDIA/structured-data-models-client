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

/// Settled early constructor guards, before schema inference reads/advice.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ConstructorContext {
    pub version: u32,
    pub task: String,
    pub target: String,
    pub entity_table: String,
}

/// Settled inferred or supplied meaning, before later native frame observations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ConstructorSchema {
    pub version: u32,
    pub task: String,
    pub target: String,
    pub entity_table: String,
    pub description: SourceDescription,
    pub schema: HeldSchema,
}

/// Only the selected native table's names and selected target's dtype.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ConstructorTaskTable {
    pub columns: Vec<HeaderName>,
    pub target_dtype: Option<String>,
}

/// Model-neutral task metadata, resolved after constructor advice effects.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PreparedDirectTask {
    pub kind: String,
    pub target: String,
    pub table: String,
    pub dtype: String,
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
    EncodeBody {
        input: DirectBody,
    },
    CheckResponseRows {
        scope: ResponseScope,
        rows: Vec<DecodedDirectRow>,
        decimal_limit: u32,
    },
    InterpretResponseJson {
        checked: CheckedDirectRow,
        json: JsonObservation,
    },
    SelectResponseMember {
        interpreted: InterpretedDirectRow,
        literal: Option<LiteralObservation>,
    },
    AttributeResponseRow {
        selected: SelectedDirectMember,
        string: Option<StringObservation>,
    },
    LayoutResponse {
        data: DirectResponseData,
        attributed: AttachmentId,
        decimal_limit: u32,
    },
    Overlap {
        input: DirectPrelude,
    },
    Envelope {
        input: DirectEnvelope,
    },
    Arguments {
        context: FrameForm,
        predict: FrameForm,
        // Keys are unused by validation and remain native until diagnostic display.
        related: Vec<FrameForm>,
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
    Context {
        headers: ConstructorHeaders,
        context: AttachmentId,
        predict: AttachmentId,
        entity_table: String,
    },
    InferredSchema {
        context: ConstructorContext,
        schema: HeldSchema,
    },
    AdoptSchema {
        schema: HeldSchema,
    },
    ConsistentSchema {
        schema: HeldSchema,
    },
    Schema {
        headers: ConstructorHeaders,
        context: AttachmentId,
        predict: AttachmentId,
        entity_table: String,
        #[serde(default)]
        schema: Option<HeldSchema>,
    },
    FinishConstructor {
        schema: ConstructorSchema,
        context: AttachmentId,
        predict: AttachmentId,
        context_header: FrameHeader,
        predict_header: FrameHeader,
        target_context: NativeLookup,
        context_predict: Vec<NativeLookup>,
    },
    ConstructorTask {
        prepared: PreparedDirectRequest,
        entity_table: String,
        tables: Vec<HeaderName>,
        table: Option<ConstructorTaskTable>,
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
    Encoded {
        body: EncodedDirectBody,
        #[serde(default)]
        inference: DirectInference,
    },
}

/// Later observations of the same captured frames, before native inference.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DirectBody {
    pub envelope: TabularEnvelope,
    pub context: AttachmentId,
    pub predict: AttachmentId,
    pub context_header: FrameHeader,
    pub predict_header: FrameHeader,
    pub task: String,
    pub primary_key: Vec<String>,
}

pub const ENCODED_DIRECT_BODY_VERSION: u32 = 1;
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct EncodedTableShape {
    pub columns: Vec<String>,
    pub rows: usize,
}
/// A complete valid tabular envelope before optional metadata/inference.
/// It is independently inspectable wire data, without deferred commands.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct EncodedDirectBody {
    pub version: u32,
    pub bound: CallPreparation,
    pub columns: TabularColumns,
    pub kind: String,
    pub primary_key: Vec<String>,
    pub context_shape: EncodedTableShape,
    pub predict_shape: EncodedTableShape,
    pub payload: AttachmentId,
}

/// The closed data value answered by one fixed preparation stage.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
pub enum PreparedValue {
    Arguments(ConstructorArguments),
    Headers(ConstructorHeaders),
    Schema(ConstructorSchema),
    Context(ConstructorContext),
    HeldSchema(HeldSchema),
    Request(PreparedDirectRequest),
    Task(PreparedDirectTask),
    Call(CallPreparation),
    Overlap(DirectOverlap),
    Envelope(TabularEnvelope),
    Body(EncodedDirectBody),
    Response(DirectResponseData),
    Rows(CheckedDirectRows),
    Member(SelectedDirectMember),
    Attributed(AttributedDirectRow),
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
    pub decimal_limit: u32,
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
    pub decimal_limit: u32,
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
