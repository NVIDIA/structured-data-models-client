// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Errors and warnings.
//!
//! An error is `{kind, code, message, fields, cause, retryable}` with a
//! closed list of kinds and a closed list of codes; a warning travels on its
//! own channel, in order. No string here names a Python or Java class: each
//! host maps every `(kind, code)` to its own classes through one reviewed
//! table, never reading message text. Codes are promised, message text is
//! not.

use std::collections::BTreeMap;
use std::fmt;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::data::ProviderError;
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

impl ErrorKind {
    /// The kind's name, as the contract spells it.
    pub fn as_str(self) -> &'static str {
        match self {
            ErrorKind::InvalidRequest => "invalid_request",
            ErrorKind::Unsupported => "unsupported",
            ErrorKind::NotFound => "not_found",
            ErrorKind::DataSource => "data_source",
            ErrorKind::ModelEndpoint => "model_endpoint",
            ErrorKind::Cancelled => "cancelled",
            ErrorKind::Protocol => "protocol",
            ErrorKind::Internal => "internal",
        }
    }
}

/// The codes the core gives: one closed list, so each host's mapping
/// matches every code with no catch-all arm (decision H6). Host and core
/// ship together, so the list is exhaustive; a new code is a contract
/// version bump. PQL's own diagnostic codes stay strings inside
/// `PqlDiagnostic`.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, JsonSchema,
)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ErrorCode {
    /// The caller's request is not one the core accepts.
    InvalidRequest,
    /// A valid request this core does not support.
    UnsupportedRequest,
    /// The model endpoint answered with something the core cannot read.
    InvalidResponse,
    /// The host's configuration of a run is not one the core accepts.
    InvalidConfiguration,
    /// The model endpoint could not be reached.
    TransportError,
    /// A defect in the core, a contained panic included.
    InternalError,
    /// The model a request names does not exist.
    UnknownModel,
    /// An optional dependency of the host is not installed.
    MissingExtra,
    /// A schema the core cannot work with.
    InvalidSchema,
    /// The endpoint does not support sessions.
    SessionsUnsupported,
    /// An Arrow type outside the closed set.
    UnsupportedType,
    /// A value the admitted type of its column cannot hold, such as a `timestamp[s]` past the microseconds an int64 counts.
    ValueOutOfRange,
    /// A provider answered with a schema other than the one asked for.
    ProviderSchemaMismatch,
    /// A provider's query failed.
    ProviderError,
    /// A reply for an id the run is not waiting on: unknown, stale or already answered.
    UnknownReply,
    /// `next` was called before every reply the run asked for arrived.
    ReplyMissing,
    /// A reply of the wrong kind: a model answer for a fetch, or the other way round.
    WrongReply,
    /// The run was used after it finished.
    RunFinished,
    /// The run was used after it was cancelled.
    RunCancelled,
    /// The run was used after a panic poisoned it.
    RunPoisoned,
    /// A second call reached a run while one was in progress.
    RunBusy,
    /// A control message the contract cannot read, larger than the control cap, of the wrong direction, or whose attachments do not match the ids it names.
    InvalidMessage,
}

impl ErrorCode {
    /// Every code, in the order the contract lists them.
    pub const ALL: [ErrorCode; 22] = [
        ErrorCode::InvalidRequest,
        ErrorCode::UnsupportedRequest,
        ErrorCode::InvalidResponse,
        ErrorCode::InvalidConfiguration,
        ErrorCode::TransportError,
        ErrorCode::InternalError,
        ErrorCode::UnknownModel,
        ErrorCode::MissingExtra,
        ErrorCode::InvalidSchema,
        ErrorCode::SessionsUnsupported,
        ErrorCode::UnsupportedType,
        ErrorCode::ValueOutOfRange,
        ErrorCode::ProviderSchemaMismatch,
        ErrorCode::ProviderError,
        ErrorCode::UnknownReply,
        ErrorCode::ReplyMissing,
        ErrorCode::WrongReply,
        ErrorCode::RunFinished,
        ErrorCode::RunCancelled,
        ErrorCode::RunPoisoned,
        ErrorCode::RunBusy,
        ErrorCode::InvalidMessage,
    ];

    /// The code's name, as the contract spells it.
    pub fn as_str(self) -> &'static str {
        match self {
            ErrorCode::InvalidRequest => "INVALID_REQUEST",
            ErrorCode::UnsupportedRequest => "UNSUPPORTED_REQUEST",
            ErrorCode::InvalidResponse => "INVALID_RESPONSE",
            ErrorCode::InvalidConfiguration => "INVALID_CONFIGURATION",
            ErrorCode::TransportError => "TRANSPORT_ERROR",
            ErrorCode::InternalError => "INTERNAL_ERROR",
            ErrorCode::UnknownModel => "UNKNOWN_MODEL",
            ErrorCode::MissingExtra => "MISSING_EXTRA",
            ErrorCode::InvalidSchema => "INVALID_SCHEMA",
            ErrorCode::SessionsUnsupported => "SESSIONS_UNSUPPORTED",
            ErrorCode::UnsupportedType => "UNSUPPORTED_TYPE",
            ErrorCode::ValueOutOfRange => "VALUE_OUT_OF_RANGE",
            ErrorCode::ProviderSchemaMismatch => "PROVIDER_SCHEMA_MISMATCH",
            ErrorCode::ProviderError => "PROVIDER_ERROR",
            ErrorCode::UnknownReply => "UNKNOWN_REPLY",
            ErrorCode::ReplyMissing => "REPLY_MISSING",
            ErrorCode::WrongReply => "WRONG_REPLY",
            ErrorCode::RunFinished => "RUN_FINISHED",
            ErrorCode::RunCancelled => "RUN_CANCELLED",
            ErrorCode::RunPoisoned => "RUN_POISONED",
            ErrorCode::RunBusy => "RUN_BUSY",
            ErrorCode::InvalidMessage => "INVALID_MESSAGE",
        }
    }
}

impl fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
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
    /// Stable across releases and languages.
    pub code: ErrorCode,
    /// For a person to read; not promised, and never holding an input value.
    pub message: String,
    /// What callers branch on, by name.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub fields: BTreeMap<String, Detail>,
    /// The error this one was raised from.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cause: Option<Box<CoreError>>,
    /// Whether the same call may succeed if tried again; carried from
    /// provider and transport errors, false otherwise.
    #[serde(default)]
    pub retryable: bool,
}

impl CoreError {
    pub fn new(kind: ErrorKind, code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            kind,
            code,
            message: message.into(),
            fields: BTreeMap::new(),
            cause: None,
            retryable: false,
        }
    }

    /// A protocol error: the host broke the run protocol.
    pub fn protocol(code: ErrorCode, message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Protocol, code, message)
    }

    /// The error a provider's failure becomes: its kind and retryability,
    /// with the driver's SQLSTATE and vendor code as fields. The driver's
    /// message is not part of the contract, since drivers quote values in
    /// it.
    pub fn provider(error: &ProviderError) -> Self {
        let mut out = Self::new(
            error.kind,
            ErrorCode::ProviderError,
            "the data provider failed",
        );
        out.retryable = error.retryable;
        if let Some(sqlstate) = &error.sqlstate {
            out = out.with_field("sqlstate", Detail::text(sqlstate.clone()));
        }
        if let Some(vendor_code) = &error.vendor_code {
            out = out.with_field("vendor_code", Detail::text(vendor_code.clone()));
        }
        out
    }

    /// The error a model transport's failure becomes: its kind and
    /// retryability.
    pub fn transport(kind: ErrorKind, retryable: bool) -> Self {
        let mut out = Self::new(
            kind,
            ErrorCode::TransportError,
            "the model endpoint could not be reached",
        );
        out.retryable = retryable;
        out
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
