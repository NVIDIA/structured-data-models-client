// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Host-held Tabular session facts, consumed by pure core operations.
//!
//! A value is not a handle into a Rust registry and contains no callbacks or
//! native object owners. The host retains it between calls; the core validates
//! it before use and returns a replacement only after a successful transition.
//! Input-sized key data travels in an attachment, never in the control message.

use crate::data::AttachmentId;
use crate::direct::{BoundDirectCall, TabularColumns};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

pub const TABULAR_SESSION_VERSION: u32 = 1;

/// Control-message attachment inventory for fit preparation.
/// Nested metadata references must equal these declared references.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SessionFitInput {
    pub input: AttachmentId,
    pub context: AttachmentId,
    pub holdout: AttachmentId,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SessionAcceptInput {
    pub input: AttachmentId,
    pub observed: AttachmentId,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SessionScoreInput {
    pub input: AttachmentId,
    pub predict: AttachmentId,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SessionOverlapInput {
    pub input: AttachmentId,
    pub predict: AttachmentId,
    pub pinned_keys: AttachmentId,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SessionSendInput {
    pub input: AttachmentId,
    pub payload: AttachmentId,
}

/// Reached native key facts, before any session equality policy is applied.
/// Numeric spelling is the producer's decimal string, not a rounded binary64
/// conversion. Other native values retain their observed string spelling.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum SessionKeyPart {
    Missing,
    Numeric { decimal: String },
    Boolean { value: bool },
    Spelling { value: String },
}

/// One ordered composite key, including all components of a single-column key.
pub type SessionKey = Vec<SessionKeyPart>;

/// Input-sized attachment contents, in source row order.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SessionKeys {
    pub rows: Vec<SessionKey>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SessionHoldoutSelection {
    pub retained_rows: Vec<usize>,
    pub removed_rows: usize,
    /// Distinct named holdout keys plus each unmatched missing-key occurrence.
    pub named_keys: usize,
}

/// The question pinned at fit, apart from the particular scoring rows.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SessionQuestion {
    pub query: Option<String>,
    pub kind: String,
    pub target: String,
    pub dtype: String,
    pub entity_table: Option<String>,
    /// Canonical decimal digits of the host's context-content fingerprint.
    /// This is an opaque equality fact, not a process-local registry handle.
    /// Decimal text preserves a native integer without a signed-width limit.
    pub context: Option<String>,
}

/// First differing question field, in the public diagnostic's precedence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SessionQuestionField {
    Query,
    Kind,
    Target,
    Dtype,
    EntityTable,
    Context,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SessionQuestionDifference {
    pub field: SessionQuestionField,
    pub pinned: Option<String>,
    pub asked: Option<String>,
}

/// The authoritative metadata kept after a successful context upload.
///
/// `bound` retains the original preflight evidence and selected outputs.
/// `pinned_model` and `pinned_kind` describe the actual post-send metadata;
/// transport mutation does not rewrite the earlier preflight evidence.
/// `columns` and `primary_key` describe the pinned
/// schema; there is no independently mutable duplicate raw schema or context.
/// The key attachment contains [`SessionKeys`] for the surviving context.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TabularSessionState {
    pub version: u32,
    pub session_id: String,
    pub closed: bool,
    pub bound: BoundDirectCall,
    pub pinned_model: String,
    pub pinned_kind: String,
    pub columns: TabularColumns,
    pub primary_key: Vec<String>,
    pub question: Option<SessionQuestion>,
    pub pinned_keys: AttachmentId,
    pub held_back: usize,
    pub pinned_over_a_window: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PrepareTabularFit {
    pub bound: BoundDirectCall,
    pub columns: TabularColumns,
    pub primary_key: Vec<String>,
    pub question: Option<SessionQuestion>,
    pub context: AttachmentId,
    pub holdout: AttachmentId,
    pub request_id: String,
    pub pinned_over_a_window: bool,
}

/// Actual post-send pinned metadata, restricted to model/task/schema/context.
/// The host observes the data; Rust selects the authoritative saved meaning.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AcceptTabularSession {
    pub state: TabularSessionState,
    pub observed: AttachmentId,
}

/// Prepared metadata without a session ID; creation has not succeeded yet.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PreparedTabularFit {
    pub bound: BoundDirectCall,
    pub columns: TabularColumns,
    pub primary_key: Vec<String>,
    pub question: Option<SessionQuestion>,
    pub pinned_keys: AttachmentId,
    pub held_back: usize,
    pub pinned_over_a_window: bool,
    pub payload: AttachmentId,
    pub named_holdout_keys: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PrepareTabularScore {
    pub state: TabularSessionState,
    pub predict: AttachmentId,
    pub request_id: String,
    pub inference_present: bool,
}

/// Key-only scoring observation, consumed before unrelated feature encoding.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CountTabularSessionOverlap {
    pub state: TabularSessionState,
    pub predict: AttachmentId,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TabularSessionOverlap {
    pub overlap_rows: usize,
    pub scoring_rows: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PreparedTabularScore {
    pub bound: BoundDirectCall,
    pub payload: AttachmentId,
    pub layout: AttachmentId,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CheckTabularSessionQuestion {
    pub state: TabularSessionState,
    pub question: Option<SessionQuestion>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SendTabularSession {
    pub endpoint: crate::run::Endpoint,
    pub payload: AttachmentId,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SessionTransportReply {
    pub status: u16,
    pub response: AttachmentId,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SessionIdObservation {
    /// Host-reported eligibility of the primary ID candidate; conversion stays in the host.
    pub session_id_available: bool,
    /// Host-reported eligibility of the fallback ID candidate.
    pub id_available: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SessionIdSelection {
    SessionId,
    Id,
    Missing,
}
