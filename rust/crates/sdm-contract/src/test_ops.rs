// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Operations only tests reach, built only with the `test-ops` feature,
//! which a shipped build leaves off. They let a test drive the core through
//! `HostRun::start` into states no Phase 1a operation reaches yet: a fetch
//! with a plan of any size, a result, error or warnings of any size. Their
//! inputs travel as bytes attachments of JSON, as input-sized fields do, so
//! the start message stays small. Two more let a binding's tests reach what
//! only a process can show: a logged event during a long call, and an
//! abort. They are not part of the contract's schema.

use serde::{Deserialize, Serialize};

use crate::data::{AttachmentId, DataRequest, FetchPlan};
use crate::ops::AttachmentKind;
use crate::scalar::Scalar;

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
    /// Panics with `payload` as the panic's message, where `at` says.
    Panic { at: PanicAt, payload: String },
    /// Emits one `tracing` event at its first step, at level `INFO` with
    /// target `sdm_core::test_ops`, then holds the call for `hold_ms`
    /// milliseconds and finishes with no rows (decision H8).
    Log { hold_ms: u64 },
    /// Aborts the process at its first step: what no host can contain.
    Abort,
    /// Resolves `zone` through the run's zones and finishes with its UTC
    /// offset at `at`, a `ts` scalar, in `TestResult::offset_seconds`; an
    /// unknown zone fails the run.
    Zone { zone: String, at: Scalar },
}

/// Where a test-only operation panics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PanicAt {
    /// While the core builds the run, inside `HostRun::start`.
    Start,
    /// In the operation's body, at its first step.
    Body,
    /// In the body, on the grown stack the PQL adapter parses on.
    AdapterStack,
}

impl TestOperation {
    /// The attachments a start of this operation names.
    pub fn attachments(&self) -> Vec<(AttachmentId, AttachmentKind)> {
        match self {
            TestOperation::Fetch { fetch: id }
            | TestOperation::Finish { result: id }
            | TestOperation::Fail { error: id }
            | TestOperation::Warn { warnings: id } => vec![(*id, AttachmentKind::Bytes)],
            TestOperation::Panic { .. }
            | TestOperation::Log { .. }
            | TestOperation::Abort
            | TestOperation::Zone { .. } => vec![],
        }
    }
}

/// The fetch a test operation asks: the request, and the plan it is asked
/// with, in place of the one a data layer would make, so a test can drive
/// a plan of any size or shape.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TestFetch {
    pub request: DataRequest,
    pub plan: FetchPlan,
    /// Whether the admitted data comes back, attached, so a binding's tests
    /// can export what the core holds.
    #[serde(default)]
    pub echo: bool,
}

/// A test-only operation's result.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TestResult {
    /// The rows fetched.
    pub rows: u64,
    /// The attachment holding the admitted data, when the fetch echoes it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<AttachmentId>,
    /// The `zone` operation's answer: the zone's UTC offset in seconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub offset_seconds: Option<i32>,
}
