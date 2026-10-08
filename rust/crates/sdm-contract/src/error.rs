// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Errors and warnings.
//!
//! An error is `{kind, code, message, fields, cause}` with a closed list of
//! kinds; a warning travels on its own channel, in order. No string here
//! names a Python or Java class: each host maps kinds and codes to its own
//! classes through one reviewed table. Codes are promised, message text is
//! not.

use std::collections::BTreeMap;
use std::fmt;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::scalar::Scalar;

/// What kind of problem an error is.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum ErrorKind {
    /// The caller's request is not one the core accepts.
    InvalidRequest,
    /// A valid request the core does not support, such as a type outside
    /// the closed set.
    Unsupported,
    /// Something the request names does not exist.
    NotFound,
    /// The data source or its driver failed.
    DataSource,
    /// The model endpoint or its transport failed.
    ModelEndpoint,
    /// The run was cancelled.
    Cancelled,
    /// The host broke the run protocol: a call in the wrong state, a reply
    /// the run did not ask for, a malformed control message.
    Protocol,
    /// A defect in the core, a contained panic included.
    Internal,
}

/// The codes the core gives. Today's public codes keep their names; a new
/// code arrives with a register entry.
pub mod codes {
    pub const INVALID_REQUEST: &str = "INVALID_REQUEST";
    pub const UNSUPPORTED_REQUEST: &str = "UNSUPPORTED_REQUEST";
    pub const INVALID_RESPONSE: &str = "INVALID_RESPONSE";
    pub const INVALID_CONFIGURATION: &str = "INVALID_CONFIGURATION";
    pub const TRANSPORT_ERROR: &str = "TRANSPORT_ERROR";
    pub const INTERNAL_ERROR: &str = "INTERNAL_ERROR";
    pub const UNKNOWN_MODEL: &str = "UNKNOWN_MODEL";
    pub const MISSING_EXTRA: &str = "MISSING_EXTRA";
    pub const INVALID_SCHEMA: &str = "INVALID_SCHEMA";
    pub const SESSIONS_UNSUPPORTED: &str = "SESSIONS_UNSUPPORTED";

    /// An Arrow type outside the closed set.
    pub const UNSUPPORTED_TYPE: &str = "UNSUPPORTED_TYPE";
    /// A provider answered with a schema other than the one asked for.
    pub const PROVIDER_SCHEMA_MISMATCH: &str = "PROVIDER_SCHEMA_MISMATCH";
    /// A provider's query failed.
    pub const PROVIDER_ERROR: &str = "PROVIDER_ERROR";

    // The run protocol's own codes.
    /// A reply for an id the run is not waiting on: unknown, stale or
    /// already answered.
    pub const UNKNOWN_REPLY: &str = "UNKNOWN_REPLY";
    /// `next` was called before every reply the run asked for arrived.
    pub const REPLY_MISSING: &str = "REPLY_MISSING";
    /// A reply of the wrong kind: a model answer for a fetch, or the other
    /// way round.
    pub const WRONG_REPLY: &str = "WRONG_REPLY";
    /// The run was used after it finished.
    pub const RUN_FINISHED: &str = "RUN_FINISHED";
    /// The run was used after it was cancelled.
    pub const RUN_CANCELLED: &str = "RUN_CANCELLED";
    /// The run was used after a panic poisoned it.
    pub const RUN_POISONED: &str = "RUN_POISONED";
    /// A second call reached a run while one was in progress.
    pub const RUN_BUSY: &str = "RUN_BUSY";
    /// A control message the contract cannot read, larger than the control
    /// cap, of the wrong direction, or whose attachments do not match the
    /// ids it names.
    pub const INVALID_MESSAGE: &str = "INVALID_MESSAGE";
}

/// A value under an error's or a warning's `fields`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
pub enum Detail {
    Scalar(Scalar),
    List(Vec<Scalar>),
}

impl Detail {
    pub fn text(value: impl Into<String>) -> Self {
        Detail::Scalar(Scalar::Utf8(value.into()))
    }
}

/// An error the core reports.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CoreError {
    pub kind: ErrorKind,
    /// Stable across releases and languages; see [`codes`].
    #[schemars(regex(pattern = r"^[A-Z][A-Z0-9_]*$"))]
    pub code: String,
    /// For a person to read; not promised.
    pub message: String,
    /// What callers branch on, by name.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub fields: BTreeMap<String, Detail>,
    /// The error this one was raised from.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cause: Option<Box<CoreError>>,
}

impl CoreError {
    pub fn new(kind: ErrorKind, code: &str, message: impl Into<String>) -> Self {
        Self {
            kind,
            code: code.to_owned(),
            message: message.into(),
            fields: BTreeMap::new(),
            cause: None,
        }
    }

    /// A protocol error: the host broke the run protocol.
    pub fn protocol(code: &str, message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Protocol, code, message)
    }

    #[must_use]
    pub fn with_field(mut self, name: &str, value: Detail) -> Self {
        self.fields.insert(name.to_owned(), value);
        self
    }

    #[must_use]
    pub fn caused_by(mut self, cause: CoreError) -> Self {
        self.cause = Some(Box::new(cause));
        self
    }
}

impl fmt::Display for CoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.code, self.message)
    }
}

impl std::error::Error for CoreError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.cause
            .as_deref()
            .map(|c| c as &(dyn std::error::Error + 'static))
    }
}

/// A warning a run gives, kept in the order given.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CoreWarning {
    #[schemars(regex(pattern = r"^[A-Z][A-Z0-9_]*$"))]
    pub code: String,
    pub message: String,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub fields: BTreeMap<String, Detail>,
}
