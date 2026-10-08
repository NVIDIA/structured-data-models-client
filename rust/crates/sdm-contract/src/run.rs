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

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::data::{AttachmentId, DataRequest, ExpectedSchema, ProviderError, SqlStatement};
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

/// One data request of a fetch step.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FetchStep {
    pub id: ReplyId,
    pub request: DataRequest,
    /// The compiled statement, for a SQL provider.
    pub sql: Option<SqlStatement>,
    pub expect: ExpectedSchema,
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
    /// The rows, as an Arrow attachment.
    Data {
        data: AttachmentId,
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
    /// The transport failed before an answer.
    TransportError {
        kind: ErrorKind,
        retryable: bool,
        message: String,
    },
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
    Error { error: CoreError },
    Warnings { warnings: Vec<CoreWarning> },
}
