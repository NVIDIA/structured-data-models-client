// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The contract between the core and its hosts: versioned types only.
//! Nothing here orchestrates or performs I/O.
//!
//! The JSON Schema of every control message is generated from these types
//! by [`schema_json`], committed as `schema/sdm-contract-<version>.json`,
//! and locked in `schema/contract.lock` by the SHA-256 of a projection that
//! strips only the annotation keywords (`description`, `title`,
//! `examples`), so a doc comment needs no bump while a change of shape does
//! (decision H7). A change of shape, or of behavior (validation, units,
//! ownership, defaults or error meaning), bumps [`CONTRACT_VERSION`]
//! (decision H25). Entries not yet merged into `rust-core` are working
//! versions: the lock holds at most one, rewritten at each bump within a
//! phase, and the entries present at the merge base with `rust-core` are
//! never rewritten (decision H4). The contract version is independent of
//! the package version.

#![forbid(unsafe_code)]

pub mod catalog;
pub mod data;
pub mod error;
pub mod facts;
pub mod ops;
pub mod pql;
pub mod run;
pub mod scalar;
#[cfg(feature = "test-ops")]
pub mod schema_check;
#[cfg(feature = "test-ops")]
pub mod test_ops;
pub mod types;

pub use error::{CoreError, CoreWarning, Detail, ErrorCode, ErrorKind, HeldState};
pub use facts::HostFacts;
pub use pql::{VALIDATED_QUERY_VERSION, ValidatedQuery, validated_query_schema_json};
pub use run::{ControlMessage, Step};
pub use scalar::Scalar;
pub use types::{ColumnType, Field, Unit};

/// The contract version this build speaks.
pub const CONTRACT_VERSION: u32 = 25;

/// The JSON Schema of [`ControlMessage`], as committed: pretty-printed, with
/// a trailing newline. Its `$defs` hold [`ops::Attachments`] too, the JSON
/// of each bytes attachment a control message names, which the root's
/// `x-attachments` refers to, so the lock covers them (T1.22).
pub fn schema_json() -> String {
    let mut generator = schemars::SchemaGenerator::default();
    let attachments = generator.subschema_for::<ops::Attachments>();
    let mut schema = generator.into_root_schema_for::<ControlMessage>();
    schema.insert("x-attachments".into(), attachments.into());
    schema.insert(
        "$id".into(),
        format!("sdm-contract/{CONTRACT_VERSION}").into(),
    );
    let mut text = serde_json::to_string_pretty(&schema).expect("a schema serializes");
    text.push('\n');
    text
}
