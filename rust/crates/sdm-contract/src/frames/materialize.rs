// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Fixed frame values under source-only coordinated composition.
//! Canonical contract publication remains serialized by the root.
//! This adds no Run step or native semantic callback.
//! Input-sized task data, node maps, sampled rows and links remain attachments.

mod attachments;
mod columns;
pub use columns::*;
mod keys;
pub use keys::*;
mod guard;
mod native;
mod questions;
pub use questions::*;

pub use guard::{
    FrameErrorOrigin, FrameFailureAssessment, FrameNativeCategory, FrameNativeFailure,
    FrameNativeReport, FramePartPresentation,
};

pub use native::{
    ConvertedFrameText, FramePredictionValues, FrameReferenceValues, NativeConstructionDiagnostic,
    NativeFrameConstruction, NativeFrameStrings, NativeFrameText, UncastableFrameInput,
};

use crate::data::strings::StringEncoding;
use crate::data::{AttachmentId, Relation};
use crate::ops::{ColumnRef, Dtype, Materialized, QueryFacts, Seed, Stype};
use crate::pql::ValidatedQuery;
use crate::{ColumnType, Scalar};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::FrameCheckpoint;

pub const FRAME_MATERIALIZATION_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PrepareFrames {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub part: Option<FrameQuestionScope>,
    /// FramePreparation as input-sized JSON with explicitly declared owners.
    pub input: AttachmentId,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PreparedFrames {
    /// The complete useful PreparedFrameValue as input-sized JSON.
    pub value: AttachmentId,
}

/// Data only: request-owned Options re-exports this definition.
/// Defaults, validation and model config presentation stay in request policy.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RelationalOptions {
    pub num_estimators: u64,
    pub column_shuffle: bool,
    pub category_shuffle: bool,
    pub hop_shuffle: bool,
    pub class_shuffle: bool,
    pub target_transforms: Vec<Option<String>>,
    pub output_type: String,
}

/// Original reconstruction declaration, independently of exported buffers.
/// Native Decimal widths and full i32 scale are not narrowed to interchange
/// admission's widened scale-zero physical representation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "declaration", rename_all = "snake_case", deny_unknown_fields)]
pub enum FramePhysicalDeclaration {
    /// Canonical non-Decimal scalar declaration. The shared reader rejects
    /// Decimal interchange variants in favor of the original descriptor below.
    Physical {
        physical: ColumnType,
    },
    LargeUtf8,
    Decimal {
        native: super::NativeDecimal,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum FrameTextBacking {
    ElementValues,
    OffsetUtf8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum FrameTextMissing {
    NullScalar,
    IeeeNan,
}

/// Original logical text storage/missing semantics, not its exported offset
/// width or its later reached lowercase operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FrameLogicalText {
    pub backing: FrameTextBacking,
    pub missing: FrameTextMissing,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FrameNativeColumn {
    pub name: String,
    /// Earlier core-completed native type, independent of later export storage.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub constructed: Option<Dtype>,
    /// At most one original declaration or logical-text fact may be present.
    /// Neither is needed for ordinary primitive, object or category sources.
    pub declared: Option<FramePhysicalDeclaration>,
    pub logical_text: Option<FrameLogicalText>,
    /// Present only for logical string source data, explicitly binding an
    /// exceptional Binary buffer to its native Unicode meaning.
    pub strings: Option<StringEncoding>,
}

/// Native source and column iteration order is observable in graph failures.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FrameNativeTable {
    pub source: Relation,
    pub columns: Vec<FrameNativeColumn>,
}

/// The independently useful immutable graph and its native source facts.
/// Source identity includes these facts, retained rows and schema meanings.
/// Rebinding changes attachment ids, never source identity or native owners.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PreparedFrameGraph {
    pub version: u32,
    pub source: String,
    /// Core-created typed graph ownership, independent of a staged workflow.
    pub owner: AttachmentId,
    pub native_tables: Vec<FrameNativeTable>,
    /// Diagnostic data from this exact constructed graph, retained across failures.
    pub dropped_summary: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FrameLimits {
    pub entities: Option<u64>,
    pub calls: Option<u32>,
    pub parts: Option<u32>,
}

/// Closed representation only. Mode choice and capacities remain the
/// existing request options policy, without a second table of defaults.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum FrameRunMode {
    Fast,
    Best,
}

/// Resolved task meaning reuses the validated PQL tree and public facts.
/// Its fingerprint binds actual rows/anchors, never call-local attachment ids.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FrameTaskMeaning {
    pub version: u32,
    pub source: String,
    pub identity: String,
    pub query: ValidatedQuery,
    pub source_query: String,
    pub facts: QueryFacts,
    pub entity: ColumnRef,
    pub mode: FrameRunMode,
    pub seed: Seed,
    pub unseeded: bool,
    pub prediction_anchor: i64,
    pub entity_anchor: bool,
    pub no_time_marker: bool,
    pub requested_context: Option<u64>,
    pub statistic: Option<String>,
    pub limits: FrameLimits,
}

/// Actual prediction input type of the resolved identifying column.
/// Numeric inputs retain the approved shared core cast rule. A genuine
/// native string dtype owner describes the new source family's input type,
/// independently of caller-edited logical dtype metadata.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum FramePredictionType {
    CoreKey {
        physical: ColumnType,
    },
    NativeString {
        column: ColumnRef,
        native: FrameNativeColumnRef,
    },
}

/// Exact immutable source-column ownership, resolved by the core graph.
/// Logical schema aliases are never interpreted by the native converter.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FrameNativeColumnRef {
    pub source: Relation,
    pub column: String,
}

/// One core-selected logical Text feature and its actual native owner.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FrameTextColumn {
    pub column: ColumnRef,
    pub native: FrameNativeColumnRef,
}

/// Complete candidate population and task meaning, before any label lookup.
/// The owner retains actual ordered sampled rows and anchors, not execution state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PreparedFramePopulation {
    pub graph: PreparedFrameGraph,
    pub meaning: FrameTaskMeaning,
    pub prediction_type: FramePredictionType,
    pub owner: AttachmentId,
    /// Complete candidate row count, independent of the number of reached batches.
    pub rows: u64,
    /// Fixed core-selected native lookup width, with a possibly shorter last range.
    pub batch_rows: u32,
}

/// Actual evaluated label rows for one candidate range. Completeness is
/// derived from the complete earlier label evidence, not a resume request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PreparedFrameLabelBatch {
    pub population: String,
    pub range: FrameBatchRange,
    pub complete: bool,
    pub owner: AttachmentId,
    /// Original source dtype for the actual list-label pointwise result.
    /// The native category conversion consumes these already computed rows.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub native_target: Option<FrameNativeColumnRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_encoding: Option<StringEncoding>,
}

/// Actual entity/anchor/target Arrow rows before native entity construction.
/// The local host retains the graph's original native key dtype owner.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct LabelledFrameTask {
    pub graph: PreparedFrameGraph,
    pub meaning: FrameTaskMeaning,
    /// Present for the standalone labelled value; completed tasks retain
    /// these same actual rows through their immutable typed task owner.
    pub context: Option<AttachmentId>,
    /// Logical text meaning of target or list-child buffers. Binary is never inferred as text.
    pub target_encoding: Option<StringEncoding>,
    pub prediction_type: FramePredictionType,
}

/// Complete narrowed context and native-constructed prediction values.
/// The context and prediction halves remain real typed task data.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CompletedFrameTask {
    /// Binds the narrowed rows, actual prediction values and selected layout.
    pub identity: String,
    pub labelled: LabelledFrameTask,
    pub prediction: Option<AttachmentId>,
    /// A typed immutable task owner replaces redundant raw rows on reuse.
    /// The raw alternative remains a fully validated useful value.
    pub owner: Option<AttachmentId>,
    /// Present exactly for the native string prediction family.
    pub prediction_encoding: Option<StringEncoding>,
    pub prediction_rows: u64,
    pub rows_per_call: u64,
    pub calls: u32,
    /// Validated lag input retained by this task's actual immutable owner.
    pub lag_steps: u64,
}

/// Ordered offsets in the task's prediction half, independent of ids.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FrameBatchRange {
    pub start: u64,
    pub stop: u64,
}

/// The useful resolved sampling/request plan after reached option notices.
/// Late native inputs use the existing call-values attachment format.
/// Native neighbor array validation remains at its original post-reset point.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FrameRequestPlan {
    pub version: u32,
    pub identity: String,
    pub task: CompletedFrameTask,
    pub inference: RelationalOptions,
    pub excluded: Vec<ColumnRef>,
    pub top_k: Option<u32>,
    pub sampling_values: AttachmentId,
}

/// Actual ordered node rows for the whole task, before all neighbor effects.
/// The UInt64 Arrow column is context followed by every prediction entity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ResolvedFrameReferences {
    pub identity: String,
    pub plan: FrameRequestPlan,
    /// Genuine whole-task resolution, shared by Python and Rust hosts.
    /// A naked node array cannot prove reached conversion and resolution.
    pub owner: AttachmentId,
}

/// Selected raw feature columns plus the existing batch ordinal column.
/// This is Arrow context data, not partially encoded model JSON.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SampledFrameTable {
    pub name: String,
    pub source: Relation,
    pub primary_key: Option<String>,
    pub rows: AttachmentId,
    /// UInt64 original source positions for immutable native dtype conversion.
    /// Distinct from the example-ownership batch column in rows.
    pub source_rows: AttachmentId,
    pub stypes: Vec<(String, Stype)>,
}

/// Exact sampled row-pair links in a two-column UInt64 Arrow attachment.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SampledFrameLink {
    pub source: ColumnRef,
    pub target: ColumnRef,
    pub pairs: AttachmentId,
}

/// Useful selected context at a reached per-batch normalization boundary.
/// Plan identity and range bind its meaning as well as the graph source.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SampledFrameContext {
    pub version: u32,
    pub plan: FrameRequestPlan,
    pub range: FrameBatchRange,
    pub identity: String,
    pub tables: Vec<SampledFrameTable>,
    pub links: Vec<SampledFrameLink>,
    pub anchors: AttachmentId,
    /// Core-selected logical Text features in reached native conversion order.
    /// Their missing-value input meaning is the empty string, before the
    /// genuine builtin astype(str) conversion. Host code chooses neither it
    /// nor the later character/word/casing policy.
    pub text: Vec<FrameTextColumn>,
    /// Fixed complete Arrow lowercase probe selected by the shared core kernel.
    pub text_identity_input: String,
}

/// Mathematical semantics observed on the actual reached native result.
/// These facts identify its operation, not its library or result dtype, and
/// do not grant the host a choice of model normalization policy.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "mapping", rename_all = "snake_case", deny_unknown_fields)]
pub enum NativeTextFacts {
    /// Full-root lowercase semantics and its actual Unicode table metadata.
    FullRoot { unicode_version: String },
    /// Raw fixed-input output identifying a supported complete simple map.
    /// User values never participate in this bounded observation.
    Simple { identity: String },
}

/// One complete valid encoded call, through the existing request encoder.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct EncodedFramePart {
    pub identity: String,
    pub context_identity: String,
    pub range: FrameBatchRange,
    /// Core-created immutable complete payload ownership, not copied bytes.
    pub payload: AttachmentId,
    pub answering: u64,
    pub tables: Vec<String>,
}

/// Ordered complete request parts, assembled once after the native batch loop.
/// Core validates complete ordered coverage before whole-request notices and
/// returning the existing Materialized/RequestState result to prediction.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FrameRequestParts {
    pub version: u32,
    /// Bind complete encoded parts to the already resolved immutable plan,
    /// without redundant graph/task attachment declarations in every part.
    pub plan_identity: String,
    pub task_identity: String,
    pub parts: Vec<EncodedFramePart>,
}

/// The existing request state and complete native logical companions, in
/// the same core-selected order. Each companion names both its actual
/// immutable logical owner and that owner's encoded RequestState body.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct MaterializedFrames {
    pub request: Materialized,
    pub payloads: Vec<crate::scalar::native_value::NativePayload>,
}

/// Fixed operations over useful values, never host instructions or a program.
/// Each invocation ends normally through the existing Done boundary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "stage", rename_all = "snake_case", deny_unknown_fields)]
pub enum FramePreparation {
    SourceMeaning {
        /// Live logical declarations after physical table admission and before
        /// GraphStore-style native sanitization.
        schema: AttachmentId,
        /// Cast-time dtype facts from those same ordered physical tables.
        typed_tables: Vec<FrameTableMeaning>,
    },
    CallerMeaning {
        schema: AttachmentId,
    },
    TableMeaning {
        input: FrameTableInputs,
    },
    TableKey {
        table: FrameTableMeaning,
        primary_key: Vec<String>,
    },
    KeyNulls {
        input: FrameKeyNulls,
    },
    TableMetadata {
        input: FrameTableMetadata,
    },
    ColumnMeaning {
        input: FrameColumnInputs,
    },
    IntegerColumn {
        input: FrameIntegerColumn,
    },
    TimeColumn {
        meaning: FrameColumnMeaning,
        values: crate::data::strings::StringData,
    },
    Question {
        schema: AttachmentId,
        query: AttachmentId,
        candidates_given: bool,
        statistic_given: bool,
        top_k_given: bool,
    },
    Questions {
        input: FrameQuestionInputs,
    },
    RelationshipOverlap {
        observations: AttachmentId,
    },
    AssessFailure {
        origin: FrameErrorOrigin,
        failure: FrameNativeFailure,
        signature: Option<Vec<u32>>,
        window_outran: Option<bool>,
    },
    ReportFailure {
        report: FrameNativeReport,
    },
    Graph {
        schema: AttachmentId,
        frames: AttachmentId,
        checkpoint: Option<FrameCheckpoint>,
        /// A live typed owner avoids replaying raw checkpoint source data.
        retained: Option<AttachmentId>,
        native_tables: Vec<FrameNativeTable>,
        /// Existing early client call-value observations, before graph notices.
        values: Option<AttachmentId>,
        options: Vec<String>,
        prediction_rows: u64,
        limits: FrameLimits,
    },
    Population {
        graph: PreparedFrameGraph,
        query: AttachmentId,
        /// The same original resolved query, before native metadata effects.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        question: Option<AttachmentId>,
        /// Task-option observations after graph notice/list(indices).
        values: Option<AttachmentId>,
        seed: Option<Seed>,
        /// Actual fresh OS fact for an unseeded draw, independent of the
        /// immutable graph checkpoint and its continuing neighbor stream.
        entropy: Option<u64>,
        anchor_time: Option<Scalar>,
        anchor_entity: bool,
        num_context_samples: Option<u64>,
        limits: FrameLimits,
    },
    LabelBatch {
        population: PreparedFramePopulation,
        range: FrameBatchRange,
        /// The last actual evaluated batch and its immutable accepted-prefix facts.
        previous: Option<PreparedFrameLabelBatch>,
        converted: FrameReferenceValues,
    },
    Labels {
        population: PreparedFramePopulation,
        batches: Vec<PreparedFrameLabelBatch>,
    },
    Task {
        labelled: LabelledFrameTask,
        /// Actual reached native Series constructor result or failure.
        prediction: FramePredictionValues,
        values: Option<AttachmentId>,
    },
    Plan {
        task: CompletedFrameTask,
        /// Fresh late request-option facts, observed after HOPS presentation.
        values: Option<AttachmentId>,
    },
    References {
        plan: FrameRequestPlan,
        /// Actual complete-task native astype(str) outcome, preserving missingness.
        converted: FrameReferenceValues,
    },
    Context {
        references: ResolvedFrameReferences,
        range: FrameBatchRange,
        /// Actual later per-batch native cast outcome, before reset and draws.
        converted: FrameReferenceValues,
    },
    Payload {
        context: SampledFrameContext,
        /// Actual reached raw conversions, not just global dtype metadata.
        text: NativeFrameText,
        /// Native request flags at their reached construction position.
        values: Option<AttachmentId>,
    },
    Request {
        plan: FrameRequestPlan,
        /// Complete independently encoded calls. No earlier payload is
        /// passed through a later native conversion or payload operation.
        parts: FrameRequestParts,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(
    tag = "value",
    content = "data",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum PreparedFrameValue {
    SourceMeaning(FrameSourceMeaning),
    CallerMeaning(FrameCallerMeaning),
    TableMeaning(FrameTableMeaning),
    KeyMeaning(FrameKeyMeaning),
    KeyNulls(FrameKeyNulls),
    ColumnMeaning(FrameColumnMeaning),
    IntegerColumn(FrameIntegerColumn),
    SourceColumn(TypedFrameSourceColumn),
    Question(PreparedFrameQuestion),
    Questions(PreparedFrameQuestions),
    RelationshipOverlap(FrameRelationshipOverlap),
    FailureAssessment(FrameFailureAssessment),
    Graph(PreparedFrameGraph),
    Population(PreparedFramePopulation),
    LabelBatch(PreparedFrameLabelBatch),
    Labels(LabelledFrameTask),
    Task(CompletedFrameTask),
    Plan(FrameRequestPlan),
    References(ResolvedFrameReferences),
    Context(SampledFrameContext),
    Part(EncodedFramePart),
    Request(MaterializedFrames),
}
