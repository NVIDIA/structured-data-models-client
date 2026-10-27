// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Draft fixed frame materialization values, pending serialized contract35.
//! This module is deliberately unlinked while the preceding contract is owned
//! by another package. It adds no Run step or native semantic callback.
//! Input-sized task data, node maps, sampled rows and links remain attachments.

mod native;

pub use native::{
    ConvertedFrameText, FramePredictionValues, FrameReferenceValues, NativeConstructionDiagnostic,
    NativeFrameConstruction, NativeFrameStrings, NativeFrameText, UncastableFrameInput,
};

use crate::data::strings::StringEncoding;
use crate::data::{AttachmentId, Relation};
use crate::ops::{ColumnRef, Materialized, QueryFacts, Seed, Stype};
use crate::pql::ValidatedQuery;
use crate::{ColumnType, Scalar};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::FrameCheckpoint;

pub const FRAME_MATERIALIZATION_VERSION: u32 = 1;

/// Data only: the existing request-owned Options type will re-export this
/// definition. Defaults, validation and model config presentation stay there.
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

/// Native dtype identity, independently of logical schema dtype edits and
/// physical Arrow offset width. No semantic inference policy lives here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum FrameNativeOrigin {
    BuiltinPrimitive,
    PandasString {
        storage: FrameStringStorage,
        missing: FrameStringMissing,
    },
    ArrowDtype,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum FrameStringStorage {
    Python,
    Pyarrow,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum FrameStringMissing {
    PandasNa,
    FloatingNan,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FrameNativeColumn {
    pub name: String,
    pub origin: FrameNativeOrigin,
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
    pub checkpoint: FrameCheckpoint,
    pub native_tables: Vec<FrameNativeTable>,
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
    CoreKey { physical: ColumnType },
    NativeString { column: ColumnRef },
}

/// Actual entity/anchor/target Arrow rows before native entity construction.
/// The local host retains the graph's original native key dtype owner.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct LabelledFrameTask {
    pub graph: PreparedFrameGraph,
    pub meaning: FrameTaskMeaning,
    pub context: AttachmentId,
    pub prediction_type: FramePredictionType,
}

/// Complete narrowed context and native-constructed prediction values.
/// The context and prediction halves remain real typed task data.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CompletedFrameTask {
    pub labelled: LabelledFrameTask,
    pub prediction: AttachmentId,
    pub prediction_rows: u64,
    pub rows_per_call: u64,
    pub calls: u32,
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
    pub plan: FrameRequestPlan,
    pub nodes: AttachmentId,
}

/// Selected raw feature columns plus the existing batch ordinal column.
/// This is Arrow context data, not partially encoded model JSON.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SampledFrameTable {
    pub name: String,
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
    pub text: Vec<ColumnRef>,
}

/// The effective native result dtype at the reached astype(str) position.
/// This is not a caller-selected normalization algorithm or package tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum NativeStringTarget {
    NumpyUnicode,
    PandasStringPython,
    PandasStringPyarrow,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct NativeTextFacts {
    pub target: NativeStringTarget,
    /// Actual builtin database metadata, not an interpreter version guess.
    pub python_unicode_database: String,
    /// Raw fixed-input native kernel output; never a normalized user value.
    pub arrow_simple_identity: Option<String>,
}

/// One complete valid encoded call, through the existing request encoder.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct EncodedFramePart {
    pub range: FrameBatchRange,
    pub payload: AttachmentId,
    pub answering: u64,
    pub tables: Vec<String>,
}

/// Ordered complete request parts, independently inspectable between batches.
/// Core validates ordered coverage before issuing whole-request notices and
/// returning the existing Materialized/RequestState result to prediction.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FrameRequestParts {
    pub version: u32,
    pub plan: FrameRequestPlan,
    pub parts: Vec<EncodedFramePart>,
}

/// Fixed operations over useful values, never host instructions or a program.
/// Each invocation ends normally through the existing Done boundary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "stage", rename_all = "snake_case", deny_unknown_fields)]
pub enum FramePreparation {
    Graph {
        schema: AttachmentId,
        frames: AttachmentId,
        checkpoint: Option<FrameCheckpoint>,
        native_tables: Vec<FrameNativeTable>,
        /// Existing early client call-value observations, before graph notices.
        values: Option<AttachmentId>,
        options: Vec<String>,
        prediction_rows: u64,
        limits: FrameLimits,
    },
    Labels {
        graph: PreparedFrameGraph,
        query: AttachmentId,
        /// Task-option observations after graph notice/list(indices).
        values: Option<AttachmentId>,
        seed: Option<Seed>,
        anchor_time: Option<Scalar>,
        anchor_entity: bool,
        num_context_samples: Option<u64>,
        limits: FrameLimits,
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
        /// Already complete valid encoded earlier calls, not a partial body.
        previous: Option<FrameRequestParts>,
        /// Native request flags at their reached construction position.
        values: Option<AttachmentId>,
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
    Graph(PreparedFrameGraph),
    Labels(LabelledFrameTask),
    Task(CompletedFrameTask),
    Plan(FrameRequestPlan),
    References(ResolvedFrameReferences),
    Context(SampledFrameContext),
    Parts(FrameRequestParts),
    Request(Materialized),
}
