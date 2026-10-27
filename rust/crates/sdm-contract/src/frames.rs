// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Call-local frame bindings, never persistent table identity or policies.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use std::collections::BTreeMap;

use crate::CoreWarning;
use crate::Scalar;
use crate::data::{AttachmentId, Relation};
use crate::ops::{ColumnRef, DescribedColumn, DescribedTable, SourceDescription};

/// The ordered Arrow tables admitted for one describe call.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FrameManifest {
    pub stage: FrameStage,
    /// Prior useful table facts, carried separately from fresh Arrow rows.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inference: Option<AttachmentId>,
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
    Nominate,
    Adopt,
    Time,
    Edit,
    StatedKey,
    Assemble,
    Sample,
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

/// The first frame materializer's ordered, call-local Arrow bindings.
/// Unlike constructor stages, this manifest names no inference policy.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct MaterializeFrames {
    /// Empty only when a frame checkpoint supplies the frozen snapshot.
    pub tables: Vec<FrameInput>,
    /// Fresh OS entropy supplied by the host, consumed only by an unseeded
    /// entity/example sample. Neighbor continuation never reads this fact.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub entropy: Option<u64>,
}

/// Version of the opaque frame checkpoint, independent of the wire version.
pub const FRAME_CHECKPOINT_VERSION: u32 = 1;

/// A fixed-size reference to an input-sized frame checkpoint manifest.
/// Each run uses fresh ids; none identifies a persistent table or snapshot.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FrameCheckpoint {
    pub version: u32,
    pub manifest: AttachmentId,
}

/// Ownership bindings of a materializer's detached frame snapshot.
/// The metadata is JSON; every table remains a separately owned Arrow attachment.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FrameCheckpointManifest {
    pub metadata: AttachmentId,
    pub tables: Vec<FrameInput>,
}

/// The exact continuing state of the legacy MT19937 primitive.
/// The consumer validates the 624 words and the position before reading it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Mt19937State {
    pub words: Vec<u32>,
    pub position: u32,
}

/// Core-owned snapshot semantics, opaque to a host retaining the checkpoint.
/// Arrow data and derived CSC structures never travel inside this JSON.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FrameCheckpointMetadata {
    /// Fingerprint of logical schema/source and detached row values, never ids.
    pub source: String,
    pub description: crate::ops::SourceDescription,
    pub neighbors: Mt19937State,
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

/// Why inference passed a timestamp over for the time column, as the
/// reference's `RejectedTimeColumn` holds it: its values (no row holds
/// one, or every row that holds one holds the same; more than half the
/// rows hold none), or its name (a stamp of a row's load, of its last edit,
/// or a date of birth).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum PassedOver {
    /// No row holds a value (`value` `None`), or the table has more than
    /// one row and every row that holds one holds `value`; `missing` of
    /// its `rows` rows hold none.
    Constant {
        value: Option<Scalar>,
        missing: usize,
        rows: usize,
    },
    /// `missing` of its `rows` rows, more than half, hold no value.
    Sparse {
        missing: usize,
        rows: usize,
    },
    LoadStamp,
    EditStamp,
    BirthDate,
}

impl PassedOver {
    /// The reason's name, as the reference's `RejectedTimeColumn.reason`
    /// gives it.
    pub fn reason(&self) -> &'static str {
        match self {
            PassedOver::Constant { .. } => "constant",
            PassedOver::Sparse { .. } => "sparse",
            PassedOver::LoadStamp => "load_stamp",
            PassedOver::EditStamp => "edit_stamp",
            PassedOver::BirthDate => "birth_date",
        }
    }
}

/// How the time column was taken (`TimeColumnChoice.how`): the one
/// timestamp left, the one creation stamp among several, the one whose
/// values start earliest, or a lone stamp of the last edit in a table no
/// other links to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Taken {
    Only,
    Created,
    Earliest,
    EditStamp,
}

/// The time column inference took and how, and the timestamps it passed
/// over, as the reference's `TimeColumnChoice` holds them.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TimeChoice {
    pub column: Option<String>,
    pub how: Option<Taken>,
    /// The columns it was chosen from, itself included.
    pub among: Vec<String>,
    /// Every timestamp passed over, in table order, the lone edit stamp
    /// taken as the last resort included.
    pub refused: Vec<(String, PassedOver)>,
    /// The lone edit stamp left, when no other timestamp was.
    pub fallback: Option<String>,
    /// Where the whole table was asked about and its answer failed, each
    /// timestamp the preview's rows would pass over for its values, kept
    /// (behind every column they do not) and named, its counts the rows
    /// read (`TimeColumnChoice.unconfirmed`).
    pub unconfirmed: Vec<(String, PassedOver)>,
}

impl TimeChoice {
    /// The timestamps passed over, each with its reason, in table order:
    /// the refused but the one taken.
    pub fn rejected(&self) -> impl Iterator<Item = &(String, PassedOver)> {
        self.refused
            .iter()
            .filter(move |(name, _)| Some(name) != self.column.as_ref())
    }
}

/// Independently inspectable inference facts, before native adoption reads.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FrameNomination {
    pub table: DescribedTable,
    pub nominated_key: Vec<String>,
}

/// Key metadata/count fixed before a possible repeat warning is presented.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FrameAdoption {
    pub table: DescribedTable,
}

/// Completed table inference, including the later time observations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FrameInferred {
    pub table: DescribedTable,
    pub time: TimeChoice,
}

/// Complete assembled meaning and independently usable sample-column plans.
/// Metadata-only advice is retained until its native source position.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FrameSchema {
    pub description: SourceDescription,
    pub samples: Vec<FrameSampleColumn>,
    pub key_notices: Vec<CoreWarning>,
    pub notices: Vec<CoreWarning>,
}

/// The fixed original meaning of one column whose cardinality decided it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FrameSampleColumn {
    pub table: String,
    pub column: DescribedColumn,
}

/// Current native cardinality facts at one actual column-read position.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FrameSample {
    pub column: FrameSampleColumn,
    pub exists: bool,
    pub present: u64,
    pub distinct: u64,
}

/// Fixed data products of shared native-frame constructor boundaries.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(
    tag = "kind",
    content = "value",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum FrameInference {
    Nomination(Box<FrameNomination>),
    Adoption(Box<FrameAdoption>),
    Inferred(Vec<FrameInferred>),
    Schema(Box<FrameSchema>),
    SampleColumn(Box<FrameSampleColumn>),
    Sample(Box<FrameSample>),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct InferredFrames {
    pub value: AttachmentId,
}
