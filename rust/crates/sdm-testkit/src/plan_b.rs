// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The reads plan B adds to a core run (the owner's decision of 2026-10-24,
//! register P3A-1): each `materialize` over a source's tables reads each
//! table as it stands at that run, its namespace, catalog, keys and
//! preview, which the old client read once, at `from_source`. The core
//! marks each such fetch (`FetchStep::purpose`, `locate`), and a capture
//! records the mark on every statement answering it. Every comparison of a
//! run with the old client's takes exactly the marked statements and
//! fetches out, and compares everything else whole, its parameters
//! included: the one rule, which `scripts/rust_core/compare.py` applies
//! (`LOCATE`) for the Python tools.

use serde_json::Value;

/// The mark ([`FetchPurpose::Locate`](sdm_contract::run::FetchPurpose)).
pub const LOCATE: &str = "locate";

fn marked(record: &Value) -> bool {
    record["purpose"] == LOCATE
}

/// `statements`, a capture's, without those the core marked as plan B's
/// reads: the statements compared with the old client's.
pub fn as_the_old_client_ran(statements: &[Value]) -> Vec<Value> {
    statements.iter().filter(|s| !marked(s)).cloned().collect()
}

/// `fetches`, a run's, each with its `request` and any `purpose`, without
/// those the core marked as plan B's reads.
pub fn requests_as_the_old_client_made(fetches: &[Value]) -> Vec<Value> {
    fetches.iter().filter(|f| !marked(f)).cloned().collect()
}
