// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Operations only tests reach, built only with the `test-ops` feature,
//! which a shipped build leaves off. They let a test drive the core through
//! `HostRun::start` into states no Phase 1a operation reaches yet: a fetch
//! with a plan of any size, a result, error or warnings of any size. Their
//! inputs travel as bytes attachments of JSON, as input-sized fields do, so
//! the start message stays small. They are not part of the contract's
//! schema.

use serde::{Deserialize, Serialize};

use crate::data::{AttachmentId, DataRequest, ExpectedSchema, SqlStatement};
use crate::ops::AttachmentKind;

/// A test-only operation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "test", rename_all = "snake_case", deny_unknown_fields)]
pub enum TestOperation {
    /// Asks the one fetch a bytes attachment of [`TestFetch`] JSON holds;
    /// finishes with the number of rows fetched, or fails with the
    /// provider's error.
    Fetch { fetch: AttachmentId },
    /// Finishes with the `OperationResult` a bytes attachment of JSON holds.
    Finish { result: AttachmentId },
    /// Fails with the `CoreError` a bytes attachment of JSON holds.
    Fail { error: AttachmentId },
    /// Gives the `CoreWarning`s a bytes attachment of a JSON list holds, in
    /// order, then finishes with no rows.
    Warn { warnings: AttachmentId },
}

impl TestOperation {
    /// The attachments a start of this operation names.
    pub fn attachments(&self) -> Vec<(AttachmentId, AttachmentKind)> {
        let id = match self {
            TestOperation::Fetch { fetch: id }
            | TestOperation::Finish { result: id }
            | TestOperation::Fail { error: id }
            | TestOperation::Warn { warnings: id } => *id,
        };
        vec![(id, AttachmentKind::Bytes)]
    }
}

/// The fetch a test operation asks.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TestFetch {
    pub request: DataRequest,
    pub sql: Option<SqlStatement>,
    pub expect: ExpectedSchema,
}

/// A test-only operation's result.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TestResult {
    /// The rows fetched.
    pub rows: u64,
}
