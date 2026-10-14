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
//! returns ([`FetchPlan`]), and reads each `describe` answer the provider
//! returned into the catalog's answer, in no dialect's shape
//! ([`crate::catalog`], decision E54). The core holds one data layer, in
//! its composition root, and names no dialect; `sdm-data` implements it.

use std::collections::BTreeSet;

use arrow_array::RecordBatch;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::catalog::CatalogAnswer;
use crate::error::{CoreError, Detail, ErrorCode, ErrorKind};
use crate::scalar::Scalar;
use crate::types::{ColumnType, Field, Unit};

/// The column of a seed key set that holds each seed's key (decision H15).
pub const SEED_KEY: &str = "key";
/// The metadata key, on a seed key set's [`SEED_KEY`] column, naming the
/// joined table's key column's declared type as the source spells it (its
/// catalog's `type`), set by the core for a data layer that reads key
/// types ([`DataLayer::reads_key_types`]): the layer joins the seeds in the
/// key's own type where that is exact (register P1B-43).
pub const SEED_KEY_TYPE: &str = "sdm:key_declared_type";
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
    /// The source's table names: one row per table. With tables named,
    /// the relations, tables or views, the source would read under any of
    /// those names written unqualified, by its own resolution of such a
    /// name (on DuckDB, through its search path): none when it would read
    /// none, which is how the core learns a name the caller gave is not
    /// the source's, never by matching names itself (T1.23, decision
    /// E59).
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
    /// Nulls after every value; the core writes the order the provider
    /// declares (`ProviderCapabilities::null_order`).
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
    /// One part of the catalog for the tables named (all when empty): what
    /// the provider returns is in the shape its plan expects, which the
    /// data layer reads into a [`CatalogAnswer`].
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
    /// One row: the rows of `table`, then, for each of `columns` in order,
    /// its smallest and largest value, each of the column's own type and
    /// null when no row holds one, and the rows holding a value, read over
    /// every row in one statement, as the reference's time column choice
    /// reads a table its preview may not hold whole (`_full_range`).
    ColumnRanges {
        table: String,
        columns: Vec<String>,
    },
    /// At most one row: a value of `key` that more than one row of `table`
    /// holds, among the rows holding every part of it, as the reference
    /// reads one to show a repeated key (`_repeated_key_sql`).
    RepeatedKey {
        table: String,
        key: Vec<String>,
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
            DataRequest::ColumnRanges { .. } => "column_ranges",
            DataRequest::RepeatedKey { .. } => "repeated_key",
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
            DataRequest::ColumnRanges { .. } => Some(OptionalRequest::ColumnRanges),
            DataRequest::RepeatedKey { .. } => Some(OptionalRequest::RepeatedKey),
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
            DataRequest::ColumnRanges { columns, .. } if columns.is_empty() => {
                return Err(refused(self, "columns", "no column is named"));
            }
            DataRequest::RepeatedKey { key, .. } if key.is_empty() => {
                return Err(refused(self, "key", "a key names no columns"));
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
    ColumnRanges,
    RepeatedKey,
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
            OptionalRequest::ColumnRanges => "column_ranges",
            OptionalRequest::RepeatedKey => "repeated_key",
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

/// How a provider's source reads a time it stores as text: which of the
/// reference's two readings the core applies to the text it fetches from
/// the provider (register DT-7). The provider declares it; the core picks
/// its reader by it, never by a dialect.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum TextTimes {
    /// Each value on its own, as the reference's engine reads a SQL
    /// source's text time column (`to_datetime(values, unit='ns')`).
    ValueByValue,
    /// The column by the client's rules (`date-rules.md`, "The client
    /// rules"), as the reference reads a frame's time column.
    ClientRules,
}

/// Where a provider's source puts a null among the values it orders, in
/// either direction: the order the reference's neighbor reads get from the
/// source, which writes no null order of its own (`_neighbor_order_by`).
/// The provider declares it; the core writes it into the tie-break it asks
/// for ([`TieTerm::nulls_last`]), never by a dialect. A source whose order
/// depends on the direction is not declarable yet: the dialect that needs
/// one adds it under a new contract version.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum NullOrder {
    /// Nulls first, ascending or descending.
    First,
    /// Nulls last, ascending or descending.
    Last,
}

/// Which of the source's indexes the reference's sampler counts as serving
/// a read on a key or foreign key, for its warning of the indexes it would
/// use and finds missing. The provider declares it; the core reads the
/// source's indexes, and warns, only when it is declared.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum IndexAdvice {
    /// An index over exactly that one column, its coverage resolved; one
    /// over several columns, or one the reader cannot resolve, serves none.
    SingleColumn,
}

/// What a provider answers: a host provider declares it, and the data
/// layer gives it for each SQL dialect it renders.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ProviderCapabilities {
    pub optional: BTreeSet<OptionalRequest>,
    /// The most keys one request may carry; `None` is no limit.
    pub max_keys_per_request: Option<u64>,
    pub sampling: Determinism,
    /// The largest seed the provider samples with (register entry RS-7);
    /// `None` takes any non-negative integer. The core refuses a sample
    /// with a larger seed, `INVALID_REQUEST` naming the seed, before the
    /// sample is planned; which seeds a provider takes is the provider's
    /// to declare, never a name the core matches.
    pub max_seed: Option<u64>,
    /// How the source reads a time it stores as text (register DT-7);
    /// `None` declares no reading, and the core refuses to read text times
    /// from it (`UNSUPPORTED_REQUEST`, rule `text_times`).
    pub text_times: Option<TextTimes>,
    /// Where the source puts a null when it orders (`NullOrder`); `None`
    /// declares none, and the core refuses to ask it for rows in an order
    /// (`UNSUPPORTED_REQUEST`, rule `null_order`).
    pub null_order: Option<NullOrder>,
    /// Which indexes serve the sampler's reads (`IndexAdvice`); `None`
    /// declares no meaningful index advice: the core neither reads the
    /// source's indexes nor warns of missing ones.
    pub index_advice: Option<IndexAdvice>,
}

/// A built-in SQL dialect, by the name the host gives it. The contract
/// names no dialect and knows none (decision E54): which names a build
/// serves is the data layer's, which refuses one it does not serve before
/// the run exists ([`DataLayer::capabilities`]).
#[derive(
    Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, JsonSchema,
)]
#[serde(transparent)]
pub struct Dialect(pub String);

impl Dialect {
    /// The dialect named `name`.
    pub fn new(name: impl Into<String>) -> Self {
        Dialect(name.into())
    }

    /// The dialect's name, as the host gave it.
    pub fn name(&self) -> &str {
        &self.0
    }
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

/// A statement's parameter: a scalar, or a key set the statement's text
/// reads under `relation`. How the keys come to be read by that name is
/// the dialect's, which `sdm-data` renders and its provider binds: the
/// DuckDB provider registers the attachment as a relation of that name; a
/// PostgreSQL dialect may bind the keys as JSON text that a common table
/// expression of that name unpacks, as the reference's PostgreSQL sampler
/// does. `relation` names what the text reads, not an instruction to
/// register an Arrow relation.
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
/// request through, and reads every catalog answer with. A host provider's
/// choice carries the capabilities it declared; for a SQL dialect the layer
/// fills them itself (decision E33).
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

    /// The catalog answer the run's provider returned to `request`, a
    /// `describe`, read into the contract's dialect-neutral shape (decision
    /// E54). The core calls it after admission: `batches` already hold the
    /// schema the request's plan expects (decision H16, check (b)). A
    /// request that is not a `describe` is refused with `INTERNAL_ERROR`.
    fn read(
        &self,
        request: &DataRequest,
        provider: &ProviderChoice,
        batches: &[RecordBatch],
    ) -> Result<CatalogAnswer, CoreError>;

    /// The vendor whose opaque Arrow extension types (`arrow.opaque`, its
    /// metadata's `vendor_name`) the run's provider hands over, declared by
    /// the layer that knows the source: the host layer reads an opaque
    /// column by that vendor's layout only when the column names this
    /// vendor (register P1B-43). `None`, the default and a host provider's,
    /// declares none, and no opaque column is read as anything but its
    /// storage.
    fn opaque_vendor(&self, provider: &ProviderChoice) -> Option<&'static str> {
        let _ = provider;
        None
    }

    /// Whether the layer joins a `neighbors` or `lookup` request's seeds by
    /// the joined key's declared type, which the core then reads from the
    /// catalog and names on the seed key column ([`SEED_KEY_TYPE`]) for a
    /// `decimal128(38, 0)` seed key, the one type no single comparison
    /// joins exactly with every key it meets (register P1B-43). `false`,
    /// the default and a host provider's, asks nothing.
    fn reads_key_types(&self, provider: &ProviderChoice) -> bool {
        let _ = provider;
        false
    }
}

/// The columns of the answers whose shape the contract fixes (decision
/// E36): every provider answers these requests with exactly these columns,
/// in this order, of these types, and check (b) holds it to them (decision
/// H16). The counts and bounds are aggregates every rendering names by
/// alias, the reference's own aliases, as the result helper columns of
/// `neighbors` and `lookup` are, so each answer is the result of the
/// reference statement it stands for (H9).
///
/// A `describe` answer is a catalog's, whose columns a SQL dialect's own
/// catalog fixes: a SQL plan expects the shape of the statement its dialect
/// runs, which `sdm-data` reads ([`DataLayer::read`]), and a host provider
/// answers in the dialect-neutral shape of [`catalog_columns`] (decision
/// E54).
pub mod results {
    /// `key_counts`: one row per key, its position in the request, the
    /// table's rows, the key's distinct values among rows holding all of
    /// it, and the rows missing part of it.
    pub const KEY_POSITION: &str = "k";
    pub const KEY_ROWS: &str = "n";
    pub const KEY_DISTINCT: &str = "d";
    pub const KEY_MISSING: &str = "z";
    /// `time_bounds`: one row per column, its position in the request and
    /// its smallest and largest `CAST(column AS TIMESTAMP)`, NULL when no
    /// row holds a value.
    pub const BOUNDS_POSITION: &str = "table_index";
    pub const BOUNDS_MIN: &str = "min_date";
    pub const BOUNDS_MAX: &str = "max_date";
}

/// The columns a host provider answers a `describe` with (decision E54):
/// the catalog's answer in no dialect's shape, which the data layer reads
/// into a [`CatalogAnswer`] as it is.
pub mod catalog_columns {
    /// `tables`: one row per table, its name. `indexes`: each index's
    /// table.
    pub const TABLE: &str = "table";
    /// `columns`, for one table: one row per column, in the table's order:
    /// its name, its declared type as the source spells it, whether it is
    /// declared `NOT NULL` and whether it is part of the primary key.
    pub const NAME: &str = "name";
    pub const TYPE: &str = "type";
    pub const NOT_NULL: &str = "not_null";
    pub const PRIMARY_KEY: &str = "primary_key";
    /// `keys`, for one table: one row per key it declares, its kind
    /// ([`KIND_PRIMARY`], [`KIND_UNIQUE`] or another the reader leaves
    /// alone) and its columns. `foreign_keys` and `indexes`: the columns of
    /// each.
    pub const KIND: &str = "kind";
    pub const COLUMNS: &str = "columns";
    pub const KIND_PRIMARY: &str = "primary";
    pub const KIND_UNIQUE: &str = "unique";
    /// `foreign_keys`, for one table: one row per column pair of each
    /// foreign key it declares, the key's number naming it, the table it
    /// refers to, and the columns on both sides.
    pub const KEY: &str = "key";
    pub const REFERENCED_TABLE: &str = "referenced_table";
    pub const REFERENCED_COLUMNS: &str = "referenced_columns";
    /// The name of a list's item field in these answers.
    pub const LIST_ITEM: &str = "item";
}

impl DataRequest {
    /// The schema a host provider's answer to this request must have:
    /// fixed for the requests whose answer the contract shapes ([`results`],
    /// and [`catalog_columns`] for a `describe`), a discovery marker for
    /// those whose columns are the source's own (a preview, a scan, a
    /// sample, neighbors, a lookup, the column ranges, whose smallest and
    /// largest values are of each column's type, which the core checks the
    /// shape of itself, and a repeated key, of the key's types) and for a
    /// count no provider renders yet. A SQL dialect's plan expects the same, but for a `describe`,
    /// whose answer is its own catalog's (decision E54).
    pub fn expected(&self) -> ExpectedSchema {
        use ColumnType::{Bool, Int32, Int64, Utf8};
        use catalog_columns as c;
        let timestamp = || ColumnType::Timestamp {
            unit: Unit::Us,
            zone: None,
        };
        let names = || ColumnType::List {
            item: Box::new(field(c::LIST_ITEM, Utf8, true)),
        };
        let fields = match self {
            DataRequest::Describe { part, .. } => match part {
                DescribePart::Tables => vec![field(c::TABLE, Utf8, false)],
                DescribePart::Columns => vec![
                    field(c::NAME, Utf8, false),
                    field(c::TYPE, Utf8, false),
                    field(c::NOT_NULL, Bool, false),
                    field(c::PRIMARY_KEY, Bool, false),
                ],
                DescribePart::Keys => vec![
                    field(c::KIND, Utf8, false),
                    field(c::COLUMNS, names(), false),
                ],
                DescribePart::ForeignKeys => vec![
                    field(c::KEY, Int64, false),
                    field(c::REFERENCED_TABLE, Utf8, false),
                    field(c::COLUMNS, names(), false),
                    field(c::REFERENCED_COLUMNS, names(), false),
                ],
                DescribePart::Indexes => vec![
                    field(c::TABLE, Utf8, false),
                    field(c::COLUMNS, names(), false),
                ],
            },
            DataRequest::KeyCounts { .. } => vec![
                field(results::KEY_POSITION, Int32, false),
                field(results::KEY_ROWS, Int64, false),
                field(results::KEY_DISTINCT, Int64, false),
                field(results::KEY_MISSING, Int64, false),
            ],
            DataRequest::TimeBounds { .. } => vec![
                field(results::BOUNDS_POSITION, Int32, false),
                field(results::BOUNDS_MIN, timestamp(), true),
                field(results::BOUNDS_MAX, timestamp(), true),
            ],
            DataRequest::Scan { .. }
            | DataRequest::Preview { .. }
            | DataRequest::CountRows { .. }
            | DataRequest::ColumnRanges { .. }
            | DataRequest::RepeatedKey { .. }
            | DataRequest::SampleEntities { .. }
            | DataRequest::Neighbors { .. }
            | DataRequest::Lookup { .. } => return ExpectedSchema::Discover {},
        };
        ExpectedSchema::Schema { fields }
    }
}

/// A field of `name`, of `column_type`, with no metadata.
pub fn field(name: &str, column_type: ColumnType, nullable: bool) -> Field {
    Field {
        name: name.to_owned(),
        column_type,
        nullable,
        metadata: Default::default(),
    }
}

/// The Arrow schema a request expects back, or that the answer discovers
/// it (a preview, a scan).
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
