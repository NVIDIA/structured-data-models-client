// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The run protocol's messages.
//!
//! A host starts a run with an operation and its facts, then asks it for
//! steps. A step is one of exactly three things: fetch a batch of data, call
//! the model, or done with the result. No step asks the host for language
//! behavior. The host answers each fetch and model call with a reply naming
//! the step's id. Arrow data and bodies travel as attachments beside the
//! control message, which names them by [`AttachmentId`].
//!
//! # Attachment ids
//!
//! The sender of a message numbers the attachments it sends (decision E2).
//!
//! - Core to host: a run numbers its attachments from 1 in the order it
//!   makes them (key sets, model bodies, result payloads), unique within
//!   that run and never reused. An id stays valid until the host takes the
//!   attachment, the run is cancelled, or the run is dropped; a run that
//!   would pass `u32::MAX` ends with `INTERNAL_ERROR`.
//! - Host to core: the host names the attachments of each message it sends
//!   (`start`, `fetch_reply`, `model_reply`) with ids of its choosing,
//!   scoped to that one message; the core takes them when it reads the
//!   message, and the ids mean nothing afterwards.
//! - A message the core writes whose encoding would pass the control cap
//!   travels whole as an attachment, named by an `attached` message; its id
//!   comes from the run's numbering, and it travels with that message.
//! - Across runs: no id carries over. State that outlives a run, such as
//!   `RequestState.payload` from `materialize`, leaves as bytes the host
//!   takes when that run finishes, and comes back attached to the `start`
//!   of the `predict` run under an id the host picks for that message.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::data::{AttachmentId, DataRequest, FetchPlan, ProviderError};
use crate::error::{CoreError, CoreWarning, ErrorKind};
use crate::facts::HostFacts;
use crate::ops::{Operation, OperationResult};

/// Names one fetch or model call within a run, so its reply can be matched.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, JsonSchema,
)]
#[serde(transparent)]
pub struct ReplyId(pub u32);

/// What a run asks of its host next.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "step", rename_all = "snake_case", deny_unknown_fields)]
pub enum Step {
    /// Answer every one of these before asking for the next step.
    Fetch {
        requests: Vec<FetchStep>,
    },
    CallModel(ModelCall),
    Done {
        result: OperationResult,
    },
}

/// One data request of a fetch step, with how the run's provider answers
/// it: the SQL statements its data layer planned, or the request passed to
/// a host provider, and the schema the answer must have.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FetchStep {
    pub id: ReplyId,
    pub request: DataRequest,
    pub plan: FetchPlan,
    /// The continuation the provider's last page of this request gave
    /// ([`FetchOutcome::Data::continuation`]), asking for the rows after
    /// it; left out on a request's first fetch (design choice 1, paging).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub continuation: Option<String>,
    /// Why the run asks it, where a host or a comparison of the run's
    /// statements with the old client's needs to know: `locate`, a read of
    /// a table as it stands at this `materialize` (its namespace, catalog,
    /// keys and preview), which the owner's plan B adds and the old client
    /// made once, at `from_source` (contract v28, T3.4). Left out, a read
    /// of the operation's own.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub purpose: Option<FetchPurpose>,
}

/// Why a run asks a fetch ([`FetchStep::purpose`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum FetchPurpose {
    /// A read of a table as it stands at this `materialize`.
    Locate,
}

/// Which model endpoint a call goes to.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum Endpoint {
    Predict,
}

/// A call to the model; the host sends the body's bytes as they are.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ModelCall {
    pub id: ReplyId,
    pub endpoint: Endpoint,
    pub content_type: String,
    pub body: AttachmentId,
}

/// The host's answer to one fetch.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FetchReply {
    pub id: ReplyId,
    pub outcome: FetchOutcome,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "outcome", rename_all = "snake_case", deny_unknown_fields)]
pub enum FetchOutcome {
    /// The rows, as an Arrow attachment. With a `continuation`, they are
    /// one page of the answer, and more follow: the core asks again for
    /// the same request with that continuation, opaque to it, until a page
    /// comes without one (design choice 1, paging), and reads the pages as
    /// one answer; nothing is truncated silently. Every page holds the
    /// same schema.
    Data {
        data: AttachmentId,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        continuation: Option<String>,
    },
    Error {
        error: ProviderError,
    },
}

/// The host's answer to one model call.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ModelReply {
    pub id: ReplyId,
    pub outcome: ModelOutcome,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "outcome", rename_all = "snake_case", deny_unknown_fields)]
pub enum ModelOutcome {
    /// The endpoint answered, with any HTTP status; the body is an
    /// attachment of the bytes received.
    Response {
        status: u16,
        content_type: String,
        body: AttachmentId,
    },
    /// The transport failed before an answer: its kind and whether trying
    /// again may succeed; the transport's message stays in the host.
    TransportError { kind: ErrorKind, retryable: bool },
    /// The endpoint failed with no body to hand over, as an SDK's error
    /// reports it (contract v28, T3.27, for T3.13): what the host was told.
    Failed { answer: ModelAnswer },
}

/// A model call's failure as the host's transport reported it
/// ([`ModelOutcome::Failed`]): the HTTP status, and the facts the core
/// reads the endpoint's own codes from. The detail can quote the request,
/// so it travels as an attachment, and the core reads it only for codes,
/// never repeating it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ModelAnswer {
    pub status: u16,
    /// A bytes attachment of the error's message or the REST body, as UTF-8
    /// text, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<AttachmentId>,
    /// The error code the SDK's error carries apart from its message
    /// (`error_code`), if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_code: Option<String>,
    /// The name of the SDK error's class (`BadRequest`), if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_error: Option<String>,
}

/// Starts a run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct StartRun {
    /// The contract version the host speaks; see `CONTRACT_VERSION`.
    pub contract: u32,
    pub facts: HostFacts,
    pub operation: Operation,
}

/// Every control message that crosses between a host and the core.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "message", rename_all = "snake_case", deny_unknown_fields)]
pub enum ControlMessage {
    // Host to core.
    Start(Box<StartRun>),
    FetchReply(FetchReply),
    ModelReply(ModelReply),
    Cancel {},
    // Core to host.
    Step(Step),
    Error {
        error: CoreError,
    },
    Warnings {
        warnings: Vec<CoreWarning>,
    },
    /// The sampler state a `materialize` run left, which the host takes
    /// once the run has ended, done or failed, and hands back with the
    /// materializer's next `materialize` (decision E52).
    SamplerState {
        state: crate::ops::SamplerState,
    },
    /// A message the core wrote whose encoding would pass the control cap:
    /// it travels whole as a bytes attachment of its UTF-8 JSON, named here
    /// (decision H7). Nothing the core writes is truncated, refused or
    /// dropped for size.
    Attached {
        attachment: AttachmentId,
    },
}
