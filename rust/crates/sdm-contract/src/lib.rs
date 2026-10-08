// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The contract between the core and its hosts: versioned types only.
//! Nothing here orchestrates or performs I/O.
//!
//! The JSON Schema of every control message is generated from these types
//! by [`schema_json`], committed as `schema/sdm-contract-<version>.json`,
//! and locked by its SHA-256 in `schema/contract.lock`. A change to the
//! types changes the schema, which fails the lock check until
//! [`CONTRACT_VERSION`] is bumped and a new lock entry added; an existing
//! entry is never rewritten. The contract version is independent of the
//! package version.

#![forbid(unsafe_code)]

pub mod data;
pub mod error;
pub mod facts;
pub mod ops;
pub mod run;
pub mod scalar;
pub mod types;

pub use error::{CoreError, CoreWarning, Detail, ErrorKind, codes};
pub use facts::HostFacts;
pub use run::{ControlMessage, Step};
pub use scalar::Scalar;
pub use types::{ColumnType, Field, Unit};

/// The contract version this build speaks.
pub const CONTRACT_VERSION: u32 = 1;

/// The JSON Schema of [`ControlMessage`], as committed: pretty-printed, with
/// a trailing newline.
pub fn schema_json() -> String {
    let mut schema = schemars::schema_for!(ControlMessage);
    schema.insert(
        "$id".into(),
        format!("sdm-contract/{CONTRACT_VERSION}").into(),
    );
    let mut text = serde_json::to_string_pretty(&schema).expect("a schema serializes");
    text.push('\n');
    text
}
