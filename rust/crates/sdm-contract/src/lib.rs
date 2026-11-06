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
//! `examples`), so a doc comment needs no bump while a change of shape does.
//! A change of shape, or of behavior (validation, units,
//! ownership, defaults or error meaning), bumps [`CONTRACT_VERSION`]
//! independently of the package version.
//! Only the current schema is retained in the checkout; Git preserves prior versions.
//! The lock check rejects a shape change without a version bump against the target revision.

#![forbid(unsafe_code)]

pub mod catalog;
pub mod columns;
pub mod data;
pub mod direct;
pub mod error;
pub mod examples;
pub mod facts;
pub mod frames;
pub mod labels;
pub mod limits;
pub mod ops;
pub mod pql;
pub mod rebind;
pub mod result;
pub mod run;
pub mod scalar;
#[cfg(feature = "schema-check")]
pub mod schema_check;
pub mod session;
pub mod sql_column;
pub mod tabular;
pub mod tabular_arrow;
#[cfg(feature = "test-ops")]
pub mod test_ops;
pub mod types;
pub mod view;
pub mod wording;

pub use error::{
    AnsweredParts, Code, CoreError, CoreWarning, Detail, ErrorCode, ErrorKind, HeldState, Sentence,
    WarningCode,
};
pub use facts::HostFacts;
pub use pql::{VALIDATED_QUERY_VERSION, ValidatedQuery, validated_query_schema_json};
pub use run::{ControlMessage, Step};
pub use scalar::Scalar;
pub use types::{ColumnType, Field, Unit};

/// The contract version this build speaks.
pub const CONTRACT_VERSION: u32 = 72;

/// The JSON Schema of [`ControlMessage`], as committed: pretty-printed, with
/// a trailing newline. Its `$defs` hold [`ops::Attachments`] too, the JSON
/// of each bytes attachment a control message names, which the root's
/// `x-attachments` refers to, so the lock covers them.
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

pub mod frame_session;

pub mod split;
