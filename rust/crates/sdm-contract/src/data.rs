// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Data requests and provider capabilities.
//!
//! The core asks for data through a small, fixed vocabulary of typed
//! requests. Every provider answers `describe` and `scan`; the optional
//! requests make common steps faster, and a provider declares which it
//! supports. A key set never travels as JSON: it is an Arrow attachment the
//! request names by id (decision Q1).

use std::collections::BTreeSet;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::error::ErrorKind;
use crate::scalar::Scalar;
use crate::types::Field;

/// Names an attachment that travels beside a control message: Arrow data
/// (a fetch's result, a key set, a predict result) or bytes (a model
/// request or response body).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, JsonSchema,
)]
#[serde(transparent)]
pub struct AttachmentId(pub u32);

/// A set of keys: an Arrow attachment, and the columns it holds.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct KeySet {
    pub attachment: AttachmentId,
    pub columns: Vec<Field>,
}

/// One bound of a time range.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Bound {
    pub value: Scalar,
    pub inclusive: bool,
}

/// A condition on the rows a request reads; a request's filters are a
/// conjunction.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "filter", rename_all = "snake_case", deny_unknown_fields)]
pub enum Filter {
    /// The row's `columns` equal one of the key set's rows.
    KeyIn {
        columns: Vec<String>,
        keys: KeySet,
    },
    TimeRange {
        column: String,
        lower: Option<Bound>,
        upper: Option<Bound>,
    },
    NotNull {
        column: String,
    },
}

/// A typed request for data.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "request", rename_all = "snake_case", deny_unknown_fields)]
pub enum DataRequest {
    /// The catalog: the tables named (all when empty), their columns,
    /// declared keys and foreign keys.
    Describe {
        tables: Vec<String>,
    },
    /// Every matching row of `table`, `columns` in order, paged through to
    /// the end; `limit` asks for any `limit` rows instead.
    Scan {
        table: String,
        columns: Vec<String>,
        filters: Vec<Filter>,
        limit: Option<u64>,
    },
    /// The first rows of `table`, for inference.
    Preview {
        table: String,
        columns: Vec<String>,
        limit: u64,
    },
    CountRows {
        table: String,
        filters: Vec<Filter>,
    },
    /// Rows per key value, for duplicate-key checks.
    KeyCounts {
        table: String,
        key: Vec<String>,
        filters: Vec<Filter>,
    },
    /// The smallest and largest value of a time column.
    TimeBounds {
        table: String,
        column: String,
        filters: Vec<Filter>,
    },
    /// A seeded sample of `n` entities of `table` by `key`.
    SampleEntities {
        table: String,
        key: Vec<String>,
        n: u64,
        seed: Scalar,
    },
}

/// The optional requests a provider may declare.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum OptionalRequest {
    Preview,
    CountRows,
    KeyCounts,
    TimeBounds,
    SampleEntities,
}

/// Whether a provider's seeded sampling repeats.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum Determinism {
    /// The same seed gives the same rows on this provider.
    SeededStable,
    Unseeded,
}

/// What a host provider declares it answers.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ProviderCapabilities {
    pub optional: BTreeSet<OptionalRequest>,
    /// The most keys one request may carry; `None` is no limit.
    pub max_keys_per_request: Option<u64>,
    pub sampling: Determinism,
}

/// A built-in SQL dialect.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum Dialect {
    Duckdb,
    Sqlite,
    Postgres,
    Databricks,
    Snowflake,
}

/// The composition root's choice: a built-in SQL provider by dialect, which
/// compiles each request to SQL the host runs, or a host provider that
/// answers the typed requests itself.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "provider", rename_all = "snake_case", deny_unknown_fields)]
pub enum ProviderChoice {
    Sql { dialect: Dialect },
    Host { capabilities: ProviderCapabilities },
}

/// A statement's parameter: a scalar, or a key set the provider binds (for
/// example by registering the attachment as a relation).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "param", rename_all = "snake_case", deny_unknown_fields)]
pub enum SqlParam {
    Value { value: Scalar },
    KeySet { keys: KeySet },
}

/// A request compiled for a SQL provider.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SqlStatement {
    pub dialect: Dialect,
    pub text: String,
    pub params: Vec<SqlParam>,
}

/// The Arrow schema a request expects back, or that the answer discovers
/// it (a describe or a preview).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "expect", rename_all = "snake_case", deny_unknown_fields)]
pub enum ExpectedSchema {
    Discover {},
    Schema { fields: Vec<Field> },
}

/// A provider's failure: its kind, whether trying again may succeed, and
/// the driver's own SQLSTATE and vendor code. The driver's message and
/// free-text codes stay in the host, since drivers quote values there;
/// nothing is matched on text.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ProviderError {
    pub kind: ErrorKind,
    pub retryable: bool,
    pub sqlstate: Option<String>,
    pub vendor_code: Option<String>,
}
