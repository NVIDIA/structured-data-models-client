// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Data requests, provider capabilities and the data-layer port.
//!
//! The core asks for data through a small, fixed vocabulary of typed
//! requests. Every provider answers `describe` and `scan`; the optional
//! requests make common steps faster, and a provider declares which it
//! supports. A key set never travels as JSON: it is an Arrow attachment the
//! request names by id (decision Q1).
//!
//! The port (decision H14): a [`DataLayer`] plans each request for the
//! run's provider, as SQL statements in a dialect it renders or as the
//! typed request passed to a host provider, with the Arrow schema the plan
//! returns ([`FetchPlan`]). The core holds one data layer, in its
//! composition root, and names no dialect; `sdm-data` implements it.

use std::collections::BTreeSet;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::error::{CoreError, Detail, ErrorCode, ErrorKind};
use crate::scalar::Scalar;
use crate::types::{ColumnType, Field, Unit};

/// The column of a seed key set that holds each seed's key (decision H15).
pub const SEED_KEY: &str = "key";
/// The column of a seed key set that holds each seed's batch position, and
/// of a `neighbors` or `lookup` result, under [`BATCH_COLUMN`], that names
/// the seed a row answers.
pub const SEED_BATCH: &str = "batch";
/// The column of a `neighbors` or `lookup` result naming the seed each row
/// answers, by its batch position: the reference's own name.
pub const BATCH_COLUMN: &str = "__kumo_batch__";
/// The column of a `lookup` result counting the rows that hold the seed's
/// key: the reference's own name.
pub const MATCHES_COLUMN: &str = "__kumo_matches__";

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

/// Which part of the catalog a `describe` reads.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum DescribePart {
    /// The source's table names: one row per table.
    Tables,
    /// Each table's columns, with their declared types, nullability and
    /// primary-key flags.
    Columns,
    /// Each table's declared primary and unique keys.
    Keys,
    /// Each table's declared foreign keys.
    ForeignKeys,
    /// The tables' indexes, which feed the reference's "Missing N indices"
    /// warning.
    Indexes,
}

/// A column of a table.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TableColumn {
    pub table: String,
    pub column: String,
}

/// Columns of a table that together make a key.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TableKey {
    pub table: String,
    pub columns: Vec<String>,
}

/// One bound of a `neighbors` window: a seed time column, compared with the
/// window column in the source's own `CAST(column AS TIMESTAMP)` wall clock.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SeedBound {
    /// A zone-less `timestamp[ns]` column of the seed key set.
    pub seed_column: String,
    pub inclusive: bool,
}

/// The rows of a `neighbors` request in time: the window column between
/// the bounds a seed gives. A row whose window column is NULL is not
/// returned when a window is set.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Window {
    pub column: String,
    pub lower: Option<SeedBound>,
    pub upper: Option<SeedBound>,
}

/// One term of a [`TieBreak`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TieTerm {
    pub column: String,
    /// Order by `CAST(column AS TIMESTAMP)`, the source's wall clock,
    /// rather than the column's own values.
    pub as_timestamp: bool,
    pub descending: bool,
    pub nulls_last: bool,
}

/// The total order a `neighbors` request keeps per seed, built by the core
/// from the reference's rule (design, "Data access").
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(transparent)]
pub struct TieBreak(pub Vec<TieTerm>);

/// A typed request for data.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "request", rename_all = "snake_case", deny_unknown_fields)]
pub enum DataRequest {
    /// One part of the catalog for the tables named (all when empty).
    Describe {
        tables: Vec<String>,
        part: DescribePart,
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
    /// For each key, in order, one row: its position, the table's rows, the
    /// key's distinct values among rows holding all of it, and the rows
    /// missing any part of it, as the reference's single `UNION ALL` reads
    /// them.
    KeyCounts {
        keys: Vec<TableKey>,
    },
    /// For each time column, in order, one row: its position and the
    /// smallest and largest `CAST(column AS TIMESTAMP)`, as the reference's
    /// single `UNION ALL` reads them.
    TimeBounds {
        columns: Vec<TableColumn>,
    },
    /// A seeded sample of `n` entities of `table` by `key`: `columns` in
    /// order, from the rows meeting every filter.
    SampleEntities {
        table: String,
        key: Vec<String>,
        n: u64,
        seed: Scalar,
        columns: Vec<String>,
        filters: Vec<Filter>,
    },
    /// The rows of `table` whose `key` equals a seed's key, `columns` in
    /// order after the seed's batch position (decision H15). `seeds` holds
    /// the columns `key` and `batch` and the seed time columns `window`
    /// names. With `per_seed`, at most that many rows per seed, first in
    /// `order`, and the result is ordered by `(batch, order)`; without it,
    /// there is no `order` and the result is unordered.
    Neighbors {
        table: String,
        key: String,
        columns: Vec<String>,
        seeds: KeySet,
        window: Option<Window>,
        per_seed: Option<u64>,
        order: Option<TieBreak>,
    },
    /// One row per seed whose key `table` holds, `columns` in order after
    /// the seed's batch position and the count of rows holding its key;
    /// which row of a key several rows hold is not defined.
    Lookup {
        table: String,
        key: String,
        columns: Vec<String>,
        seeds: KeySet,
    },
}

impl DataRequest {
    /// The request's name, as the contract spells it.
    pub fn name(&self) -> &'static str {
        match self {
            DataRequest::Describe { .. } => "describe",
            DataRequest::Scan { .. } => "scan",
            DataRequest::Preview { .. } => "preview",
            DataRequest::CountRows { .. } => "count_rows",
            DataRequest::KeyCounts { .. } => "key_counts",
            DataRequest::TimeBounds { .. } => "time_bounds",
            DataRequest::SampleEntities { .. } => "sample_entities",
            DataRequest::Neighbors { .. } => "neighbors",
            DataRequest::Lookup { .. } => "lookup",
        }
    }

    /// The optional request this is, if it is one; `describe` and `scan`
    /// are what every provider answers.
    pub fn optional(&self) -> Option<OptionalRequest> {
        match self {
            DataRequest::Describe { .. } | DataRequest::Scan { .. } => None,
            DataRequest::Preview { .. } => Some(OptionalRequest::Preview),
            DataRequest::CountRows { .. } => Some(OptionalRequest::CountRows),
            DataRequest::KeyCounts { .. } => Some(OptionalRequest::KeyCounts),
            DataRequest::TimeBounds { .. } => Some(OptionalRequest::TimeBounds),
            DataRequest::SampleEntities { .. } => Some(OptionalRequest::SampleEntities),
            DataRequest::Neighbors { .. } => Some(OptionalRequest::Neighbors),
            DataRequest::Lookup { .. } => Some(OptionalRequest::Lookup),
        }
    }

    /// Refuses a request whose fields contradict each other, so every
    /// field is honored or the request refused, never ignored: a
    /// `neighbors` with `order` and no `per_seed` or the reverse, an empty
    /// tie-break, seeds without the `key` and `batch` columns, or a window
    /// bound naming no zone-less `timestamp[ns]` seed column; a request
    /// asking for no columns or no keys where it must name some.
    pub fn check(&self) -> Result<(), CoreError> {
        match self {
            DataRequest::KeyCounts { keys } => {
                if keys.is_empty() || keys.iter().any(|k| k.columns.is_empty()) {
                    return Err(refused(self, "keys", "a key names no columns"));
                }
            }
            DataRequest::TimeBounds { columns } if columns.is_empty() => {
                return Err(refused(self, "columns", "no time column is named"));
            }
            DataRequest::Neighbors {
                seeds,
                window,
                per_seed,
                order,
                ..
            } => {
                check_seeds(self, seeds)?;
                match (per_seed, order) {
                    (Some(_), None) => {
                        return Err(refused(
                            self,
                            "order",
                            "a row limit per seed needs an order",
                        ));
                    }
                    (None, Some(_)) => {
                        return Err(refused(
                            self,
                            "order",
                            "an order without a row limit per seed is not kept",
                        ));
                    }
                    (Some(_), Some(TieBreak(terms))) if terms.is_empty() => {
                        return Err(refused(self, "order", "the order has no terms"));
                    }
                    _ => {}
                }
                if let Some(window) = window {
                    if window.lower.is_none() && window.upper.is_none() {
                        return Err(refused(self, "window", "the window has no bound"));
                    }
                    for bound in window.lower.iter().chain(&window.upper) {
                        let seed_time = seeds.columns.iter().any(|f| {
                            f.name == bound.seed_column
                                && f.column_type
                                    == ColumnType::Timestamp {
                                        unit: Unit::Ns,
                                        zone: None,
                                    }
                        });
                        if !seed_time {
                            return Err(refused(
                                self,
                                "window",
                                "a bound names no zone-less timestamp[ns] seed column",
                            ));
                        }
                    }
                }
            }
            DataRequest::Lookup { seeds, .. } => check_seeds(self, seeds)?,
            _ => {}
        }
        Ok(())
    }
}

/// Seeds hold the role-named `key` and `batch` columns (decision H15).
fn check_seeds(request: &DataRequest, seeds: &KeySet) -> Result<(), CoreError> {
    for role in [SEED_KEY, SEED_BATCH] {
        if !seeds.columns.iter().any(|f| f.name == role) {
            return Err(refused(
                request,
                "seeds",
                "the seeds lack a key or a batch column",
            ));
        }
    }
    Ok(())
}

/// The refusal of a request whose fields contradict each other, naming the
/// request and the field, never a value.
pub fn refused(request: &DataRequest, field: &str, message: &str) -> CoreError {
    CoreError::new(
        ErrorKind::InvalidRequest,
        ErrorCode::InvalidRequest,
        message,
    )
    .with_field("request", Detail::text(request.name()))
    .with_field("field", Detail::text(field))
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
    Neighbors,
    Lookup,
}

impl OptionalRequest {
    /// The request's name, as the contract spells it.
    pub fn name(self) -> &'static str {
        match self {
            OptionalRequest::Preview => "preview",
            OptionalRequest::CountRows => "count_rows",
            OptionalRequest::KeyCounts => "key_counts",
            OptionalRequest::TimeBounds => "time_bounds",
            OptionalRequest::SampleEntities => "sample_entities",
            OptionalRequest::Neighbors => "neighbors",
            OptionalRequest::Lookup => "lookup",
        }
    }
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

/// A statement's parameter: a scalar, or a key set the provider binds
/// under `relation`, the name the statement's text reads it by (for
/// example by registering the attachment as a relation for the statement).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "param", rename_all = "snake_case", deny_unknown_fields)]
pub enum SqlParam {
    Value { value: Scalar },
    KeySet { keys: KeySet, relation: String },
}

/// One statement a SQL provider runs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SqlStatement {
    pub dialect: Dialect,
    pub text: String,
    pub params: Vec<SqlParam>,
}

/// How a provider answers one request (decision H14): SQL statements run
/// in order on one session, the last one's rows the answer, or the typed
/// request itself, passed to a host provider that answers it; either way
/// with the Arrow schema the whole plan returns, and answered with one
/// attachment holding every row.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "plan", rename_all = "snake_case", deny_unknown_fields)]
pub enum FetchPlan {
    Sql {
        statements: Vec<SqlStatement>,
        expect: ExpectedSchema,
    },
    Host {
        expect: ExpectedSchema,
    },
}

impl FetchPlan {
    /// The schema the plan returns.
    pub fn expect(&self) -> &ExpectedSchema {
        match self {
            FetchPlan::Sql { expect, .. } | FetchPlan::Host { expect } => expect,
        }
    }
}

/// The data-layer port (decision H14): what the core plans every data
/// request through. A host provider's choice carries the capabilities it
/// declared; for a SQL dialect the layer fills them itself (decision E33).
pub trait DataLayer: Send + Sync {
    /// What the run's provider answers: the capabilities a host provider
    /// declared, or those this layer fills for its SQL dialect. A dialect
    /// it has no rendering for is refused with `UNSUPPORTED_REQUEST`
    /// naming it.
    fn capabilities(&self, provider: &ProviderChoice) -> Result<ProviderCapabilities, CoreError>;

    /// How the run's provider answers `request`. The core has already
    /// refused an optional request the provider does not declare.
    fn plan(
        &self,
        request: &DataRequest,
        provider: &ProviderChoice,
    ) -> Result<FetchPlan, CoreError>;
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
