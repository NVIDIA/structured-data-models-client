// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The one SQL builder (T3.18, design fix E): every statement shape the
//! dialects share, written once over [`SqlDialect`], and the one lookup of
//! a dialect by its name ([`dialect`]).
//!
//! A dialect (`duckdb.rs`, `sqlite.rs`, `postgres.rs`) implements the
//! trait, and holds only what differs: its name and capabilities, its
//! quoting, the name parts a relation takes and whether its source reads
//! under a namespace, how a key set is named, bound and joined, how a time
//! is cast, the seeded sample, the per-seed neighbor read and the key
//! lookup, its catalog statements and the reading of their answers, and
//! the quirks the core asks of a provider. A warehouse dialect (T3.4
//! onward) plugs in the same way: one more `impl SqlDialect` and one more
//! entry in [`DIALECTS`].
//!
//! Everything here renders text the dialects rendered before the trait,
//! byte for byte: the goldens and `tests/dialect_matrix.rs` hold it.

use std::collections::BTreeSet;

use arrow_array::RecordBatch;
use sdm_contract::catalog::CatalogAnswer;
use sdm_contract::data::{
    BATCH_COLUMN, DataRequest, DescribePart, Dialect, ExpectedSchema, FetchPlan, Filter, KeySet,
    OptionalRequest, ProviderCapabilities, Relation, SEED_BATCH, SqlParam, SqlStatement,
    TableColumn, TableKey, TieBreak, TieTerm, Window, refuse_derived, refuse_relation, refused,
    seed_key_columns,
};
use sdm_contract::ops::Dtype;
use sdm_contract::{CoreError, Detail, Scalar, Sentence};

use crate::{databricks, duckdb, postgres, quote, sqlite};

/// The dialects this build renders, in the order `PROVIDERS` names them.
pub(crate) static DIALECTS: &[&dyn SqlDialect] = &[
    &duckdb::Duckdb,
    &sqlite::Sqlite,
    &postgres::Postgres,
    &databricks::Databricks,
];

/// How a dialect's source compares the names of its namespace and
/// relations (design fix C): as written, without regard to case (Unity
/// Catalog's, which stores them in lower case), or upper-cased when
/// unquoted (Snowflake's, T3.6).
#[allow(dead_code, reason = "Snowflake's rule, T3.6")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NameCase {
    AsWritten,
    Insensitive,
    UpperWhenUnquoted,
}

/// The dialect of `name`, if this build renders it.
pub(crate) fn dialect(name: &str) -> Option<&'static dyn SqlDialect> {
    DIALECTS.iter().copied().find(|d| d.name() == name)
}

/// The optional requests the shared builder renders, each a dialect's
/// capability.
const RENDERED: &[OptionalRequest] = &[
    OptionalRequest::CountRows,
    OptionalRequest::Preview,
    OptionalRequest::KeyCounts,
    OptionalRequest::TimeBounds,
    OptionalRequest::ColumnRanges,
    OptionalRequest::RepeatedKey,
    OptionalRequest::SampleEntities,
    OptionalRequest::Neighbors,
    OptionalRequest::Lookup,
];

/// [`RENDERED`], as a dialect declares it.
pub(crate) fn rendered() -> BTreeSet<OptionalRequest> {
    RENDERED.iter().copied().collect()
}

/// What a SQL dialect says for itself: the hooks the shared builder
/// ([`plan`]) renders each statement through. The required ones are what
/// no two dialects write alike; each provided one is what most write
/// alike, overridden where one does not.
pub(crate) trait SqlDialect: Sync {
    // --- name and capabilities -------------------------------------------

    /// The dialect's name on the wire.
    fn name(&self) -> &'static str;

    /// What a source of this dialect answers through this layer.
    fn capabilities(&self) -> ProviderCapabilities;

    // --- quoting, depths and namespace -------------------------------------

    /// The most name parts a relation takes (design choice 12).
    fn most_parts(&self) -> usize;

    /// `name` as an identifier: in double quotes, each one in it doubled.
    fn identifier(&self, name: &str) -> String {
        quote::identifier(name)
    }

    /// How many namespace parts a source of this dialect takes, outermost
    /// first (design fix C), apart from the parts a relation may have
    /// ([`SqlDialect::most_parts`]): DuckDB and SQLite none, Postgres a
    /// schema, Databricks a catalog and a schema. Each relation of one part
    /// is read under the provider's namespace (contract v28), and the
    /// namespace handed to [`SqlDialect::describe`]; a namespace of more
    /// parts is refused by name.
    fn namespace_depth(&self) -> usize {
        0
    }

    /// Whether the source's every namespace part is read before any table
    /// (design fix C): the parts the provider leaves to the connection, a
    /// leading one included, read once as the session's current ones, so
    /// each relation is read under the full namespace, as the reference
    /// reads a Databricks source (`current_catalog()`, `current_schema()`)
    /// and as its catalog names a table a foreign key refers to. `false`,
    /// the default: a relation of fewer parts is read in the connection's
    /// current ones.
    fn settles_namespace(&self) -> bool {
        false
    }

    /// The refusal of `relation` where it names fewer parts than the
    /// dialect reads every relation in: none by default, a relation of
    /// fewer parts read where the connection finds it.
    fn unqualified(&self, _relation: &Relation) -> Option<CoreError> {
        None
    }

    /// How the source compares names (design fix C): as written.
    fn name_case(&self) -> NameCase {
        NameCase::AsWritten
    }

    /// A derived relation's `FROM` (contract v28), `source` the rendering
    /// of the relation it reads; `None` for a dialect that reads none, where
    /// a derived relation is refused by name (`derived_relations` is false
    /// in its capabilities).
    fn derived(&self, _relation: &Relation, _source: &str) -> Option<String> {
        None
    }

    // --- key sets: names, binding and joins --------------------------------

    /// The name a statement answering `request`, which joins `table`,
    /// reads its key set by, never one `table`'s own name would meet in
    /// the dialect's case rules.
    fn seeds_for(&self, request: &DataRequest, table: &Relation) -> String;

    /// The key set's relation as a statement's `FROM` names it.
    fn seeds_relation(&self, name: &str) -> String {
        self.identifier(name)
    }

    /// What the statement opens with to bind its key set, with its trailing
    /// line break; nothing where the provider binds the key set as a
    /// relation of its own.
    fn seeds_prelude(&self, _read: &SeedRead<'_>) -> Result<String, CoreError> {
        Ok(String::new())
    }

    /// The condition of one key column, `key` (rendered), on its seed
    /// column, `seed` (rendered, the column `seed_column` of the key set),
    /// compared by `op`.
    fn key_equals(
        &self,
        _read: &SeedRead<'_>,
        key: &str,
        _seed_column: &str,
        seed: &str,
        op: &str,
    ) -> String {
        format!("{key} {op} {seed}")
    }

    /// A window bound, the key set's column `seed_column`, as the window's
    /// column is compared with it.
    fn window_bound(&self, _seeds: &KeySet, seed_column: &str) -> String {
        format!("s.{}", self.identifier(seed_column))
    }

    // --- casts ----------------------------------------------------------------

    /// `value`, a rendered column, as the dialect compares and orders a
    /// time: the column itself, unless the dialect casts it.
    fn as_time(&self, value: &str) -> String {
        value.to_owned()
    }

    // --- statement shapes that differ -------------------------------------------

    /// The row count of `table` (rendered) where `filtered` holds (the
    /// `WHERE` clause, opened by a space, or nothing): its one column named
    /// `rows`, unless the dialect writes the reference's own statement.
    fn count_rows(&self, table: &str, filtered: &str) -> String {
        format!(
            "SELECT COUNT(*) AS {} FROM {table}{filtered}",
            self.identifier("rows")
        )
    }

    /// What follows a preview's `FROM` before its `LIMIT`: nothing, unless
    /// the dialect reads the first rows in an order of its own.
    fn preview_order(&self) -> &'static str {
        ""
    }

    /// The distinct values of a key of several columns, `quoted`, among the
    /// rows of `table` holding all of it: a row of several values counted
    /// distinct where every part is present.
    fn distinct_keys(&self, quoted: &[String], _table: &str) -> String {
        format!(
            "COUNT(DISTINCT ({})) FILTER (WHERE {})",
            quoted.join(", "),
            present(quoted)
        )
    }

    /// One `SELECT` of the time bounds: the column `time` (rendered and
    /// cast) of `table`, labelled by its `position`.
    fn time_bounds_select(&self, position: usize, time: &str, table: &str) -> String {
        format!(
            "SELECT\n  {position} as table_index,\n  MIN({time}) as min_date,\n  MAX({time}) as max_date\nFROM {table}"
        )
    }

    /// One term of a neighbor order, its column under `alias`; `windowed`
    /// names the window's column, if any. Its null order written out.
    fn order_term(&self, term: &TieTerm, alias: &str, _windowed: Option<&str>) -> String {
        let column = format!("{alias}{}", self.identifier(&term.column));
        let value = if term.as_timestamp {
            self.as_time(&column)
        } else {
            column
        };
        format!(
            "{value} {} {}",
            if term.descending { "DESC" } else { "ASC" },
            if term.nulls_last {
                "NULLS LAST"
            } else {
                "NULLS FIRST"
            }
        )
    }

    /// A sample's seed as the dialect takes it, or its refusal.
    fn sample_seed(&self, request: &DataRequest, seed: &Scalar) -> Result<i128, CoreError>;

    /// The seeded sample's statements, the sample last.
    fn sample(&self, sample: &Sample<'_>) -> Vec<String>;

    /// A neighbor read keeping the first `limit` rows of each seed in the
    /// order of `terms`, the result in `(batch, order)`.
    fn neighbors_per_seed(&self, read: &SeedRead<'_>, limit: u64, terms: &[TieTerm]) -> String;

    /// A neighbor read given a limit per seed without an order, or the
    /// reverse, which the core never asks (`DataRequest::check`): read as
    /// one with neither, unless the dialect refuses it.
    fn neighbors_unpaired(&self, read: &SeedRead<'_>) -> Result<String, CoreError> {
        Ok(read.joined())
    }

    /// The key lookup: one row per seed whose key the table holds.
    fn lookup(&self, read: &SeedRead<'_>) -> String;

    // --- the catalog ---------------------------------------------------------------

    /// One part of the catalog for `tables`, each already read through
    /// the dialect's depth; `namespace` the provider's, where the dialect
    /// takes one. The statement's text and the parameters it binds, in its
    /// placeholders' order: none for a dialect that writes each name in
    /// place, as DuckDB, SQLite and Postgres do; a warehouse's catalog read
    /// binds the names it looks for (Databricks's `lower(?)`, T3.4).
    fn describe(
        &self,
        request: &DataRequest,
        tables: &[Relation],
        part: DescribePart,
        namespace: &[String],
    ) -> Result<(String, Vec<SqlParam>), CoreError>;

    /// The read of the source's namespace, `request` a `describe` of it,
    /// where the provider's `namespace` leaves a part to the connection
    /// inside one it gives (a Databricks source naming its catalog and not
    /// its schema): each part given, or the session's current one, so the
    /// core can read every later relation under the parts it answers. The
    /// default refuses it by name, as no relation could be read under it.
    fn current_namespace(
        &self,
        request: &DataRequest,
        namespace: &[Option<String>],
    ) -> Result<(String, Vec<SqlParam>), CoreError> {
        let _ = namespace;
        Err(not_taken(
            request,
            "namespace",
            "a namespace part left to the connection inside one given cannot qualify a relation",
        ))
    }

    /// The schema the dialect's answer to `request` has.
    fn expected(&self, request: &DataRequest) -> ExpectedSchema;

    /// The dialect's answer to a `describe` of `part`, asked by `request`,
    /// read into the contract's catalog answer.
    fn read(
        &self,
        request: &DataRequest,
        part: DescribePart,
        batches: &[RecordBatch],
    ) -> Result<CatalogAnswer, CoreError>;

    // --- what the core asks of a provider (`DataLayer`) -------------------------

    /// The answer's values conformed to the reference's reading, if the
    /// dialect changes any (`DataLayer::conform`).
    fn conform(
        &self,
        _schema: &arrow_schema::SchemaRef,
        _batches: &[RecordBatch],
        _float_text: fn(f64) -> String,
    ) -> Result<Option<Batches>, CoreError> {
        Ok(None)
    }

    /// A key set as the dialect's provider binds it, if it changes any
    /// (`DataLayer::bound_keys`).
    fn bound_keys(
        &self,
        _schema: &arrow_schema::SchemaRef,
        _batches: &[RecordBatch],
    ) -> Result<Option<Batches>, CoreError> {
        Ok(None)
    }
}

/// A schema and its batches.
pub(crate) type Batches = (arrow_schema::SchemaRef, Vec<RecordBatch>);

/// A request field a rendering does not take, refused by name.
pub(crate) fn not_taken(request: &DataRequest, field: &str, message: &str) -> CoreError {
    CoreError::new(Sentence::RenderingRefused, message)
        .with_field("request", Detail::text(request.name()))
        .with_field("field", Detail::text(field))
}

/// The one rendering of a relation every dialect's `FROM` goes through
/// (design choice 12): each part of `relation` quoted alone by the
/// dialect, never split on a dot, joined by dots. A relation of no part,
/// or of more than the dialect's [`SqlDialect::most_parts`], is refused by
/// name (rule `relation`, [`refuse_relation`]) as `field` of `request`, and
/// one of fewer parts than the dialect reads as the dialect says
/// ([`SqlDialect::unqualified`]).
pub(crate) fn from(
    dialect: &dyn SqlDialect,
    request: &DataRequest,
    field: &str,
    relation: &Relation,
) -> Result<String, CoreError> {
    if let Some(refusal) = refuse_relation(request, field, relation, dialect.most_parts()) {
        return Err(refusal);
    }
    if let Some(refusal) = dialect.unqualified(relation) {
        return Err(refusal);
    }
    let renders = dialect.capabilities().derived_relations;
    if let Some(refusal) = refuse_derived(request, field, relation, renders) {
        return Err(refusal);
    }
    let source = relation
        .source
        .iter()
        .map(|part| dialect.identifier(part))
        .collect::<Vec<_>>()
        .join(".");
    if !relation.is_derived() {
        return Ok(source);
    }
    // A dialect declaring derived relations renders them (contract v28);
    // one that declares them and renders none is this build's fault, never
    // a read of the source with the relation's columns or filter dropped.
    dialect.derived(relation, &source).ok_or_else(|| {
        CoreError::new(
            Sentence::Internal,
            "a dialect declaring derived relations rendered none",
        )
        .with_field("request", Detail::text(request.name()))
        .with_field("field", Detail::text(field))
    })
}

/// The one table a part of the catalog read one table at a time names, or
/// its refusal.
pub(crate) fn one_table<'a>(
    request: &DataRequest,
    tables: &'a [Relation],
) -> Result<&'a Relation, CoreError> {
    match tables {
        [table] => Ok(table),
        _ => Err(not_taken(
            request,
            "tables",
            "this part of the catalog is read for one table at a time",
        )),
    }
}

/// Refuses tables named to a part of the catalog read whole.
pub(crate) fn whole(request: &DataRequest, tables: &[Relation]) -> Result<(), CoreError> {
    if tables.is_empty() {
        Ok(())
    } else {
        Err(not_taken(
            request,
            "tables",
            "this part of the catalog is read whole",
        ))
    }
}

/// `quoted`, each present, joined by `AND`.
fn present(quoted: &[String]) -> String {
    quoted
        .iter()
        .map(|c| format!("{c} IS NOT NULL"))
        .collect::<Vec<_>>()
        .join(" AND ")
}

/// A seeded sample, as the shared builder hands it to its dialect: the
/// request's relation and its rendering, the key, the count, the seed the
/// dialect took, the columns projected and the `WHERE` clause of the
/// filters (opened by a line break, or nothing for none), each rendered.
pub(crate) struct Sample<'a> {
    pub relation: &'a Relation,
    pub table: String,
    pub key: &'a [String],
    pub n: u64,
    pub seed: i128,
    pub projected: String,
    pub filtered: String,
    pub key_dtype: Option<Dtype>,
    /// The table's rows, where the core counted them for a dialect sizing
    /// its draw by them (`ProviderCapabilities::sample_rows`).
    pub rows: Option<u64>,
}

/// A read of a table against a key set (`neighbors`, `lookup`), as the
/// shared builder hands it to its dialect, with the shapes they share.
pub(crate) struct SeedRead<'a> {
    pub dialect: &'a dyn SqlDialect,
    pub request: &'a DataRequest,
    /// The table read, rendered.
    pub table: String,
    /// The name the key set is read by ([`SqlDialect::seeds_for`]).
    pub seeds_name: String,
    pub key: &'a [String],
    pub columns: &'a [String],
    pub seeds: &'a KeySet,
    pub window: Option<&'a Window>,
}

impl SeedRead<'_> {
    /// The key set's relation, as the `FROM` names it.
    pub fn seeds(&self) -> String {
        self.dialect.seeds_relation(&self.seeds_name)
    }

    /// The seed's batch position, under the reference's name.
    pub fn batch(&self) -> String {
        format!(
            "s.{} AS {}",
            self.dialect.identifier(SEED_BATCH),
            self.dialect.identifier(BATCH_COLUMN)
        )
    }

    /// The table's columns asked for, read from its alias.
    pub fn projected(&self) -> String {
        self.columns
            .iter()
            .map(|c| format!("t.{}", self.dialect.identifier(c)))
            .collect::<Vec<_>>()
            .join(", ")
    }

    /// The condition joining the table's key (under `alias`, or unqualified
    /// for `""`) to the seeds' key columns ([`seed_key_columns`]), each by
    /// `op` ([`SqlDialect::key_equals`]), joined by `AND`.
    pub fn seed_join(&self, alias: &str, op: &str) -> String {
        let d = self.dialect;
        self.key
            .iter()
            .zip(seed_key_columns(self.key.len()))
            .map(|(column, seed)| {
                d.key_equals(
                    self,
                    &format!("{alias}{}", d.identifier(column)),
                    &seed,
                    &format!("s.{}", d.identifier(&seed)),
                    op,
                )
            })
            .collect::<Vec<_>>()
            .join(" AND ")
    }

    /// The conditions on a row of the table (under `alias`): its key a
    /// seed's, then its window column, cast as the dialect compares a time,
    /// not after the upper bound and after the lower one.
    pub fn within(&self, alias: &str) -> Vec<String> {
        let d = self.dialect;
        let mut found = vec![self.seed_join(alias, "=")];
        if let Some(window) = self.window {
            let time = d.as_time(&format!("{alias}{}", d.identifier(&window.column)));
            if let Some(upper) = &window.upper {
                let op = if upper.inclusive { "<=" } else { "<" };
                found.push(format!(
                    "{time} {op} {}",
                    d.window_bound(self.seeds, &upper.seed_column)
                ));
            }
            if let Some(lower) = &window.lower {
                let op = if lower.inclusive { ">=" } else { ">" };
                found.push(format!(
                    "{time} {op} {}",
                    d.window_bound(self.seeds, &lower.seed_column)
                ));
            }
        }
        found
    }

    /// The rows of the table each seed's key and window hold, joined to
    /// their seed: the read every dialect writes alike.
    pub fn joined(&self) -> String {
        format!(
            "SELECT {}, {}\nFROM {} s\nJOIN {} t\n  ON {}",
            self.batch(),
            self.projected(),
            self.seeds(),
            self.table,
            self.within("t.").join("\n AND ")
        )
    }
}

/// The statements answering `request` in `dialect`, with the schema its
/// answer has; `namespace` the provider's, for a dialect that takes one.
pub(crate) fn plan(
    dialect: &dyn SqlDialect,
    request: &DataRequest,
    namespace: &[String],
) -> Result<FetchPlan, CoreError> {
    let d = dialect;
    let mut before = Vec::new();
    let mut params = Vec::new();
    let text = match request {
        DataRequest::Describe { tables, part } => {
            for table in tables {
                from(d, request, "tables", table)?;
            }
            let (text, bound) = d.describe(request, tables, *part, namespace)?;
            params = bound;
            text
        }
        DataRequest::Preview {
            table,
            columns,
            limit,
        } => {
            let table = from(d, request, "table", table)?;
            if columns.is_empty() {
                return Err(refused(request, "columns", "a preview names no columns"));
            }
            format!(
                "SELECT {} FROM {table}{} LIMIT {limit}",
                identifiers(d, columns),
                d.preview_order()
            )
        }
        DataRequest::KeyCounts { keys } => key_counts(d, request, keys)?,
        DataRequest::TimeBounds { columns } => time_bounds(d, request, columns)?,
        DataRequest::ColumnRanges { table, columns } => {
            let table = from(d, request, "table", table)?;
            if columns.is_empty() {
                return Err(refused(
                    request,
                    "columns",
                    "the column ranges name no columns",
                ));
            }
            let mut selects = vec!["COUNT(*)".to_owned()];
            for column in columns {
                let quoted = d.identifier(column);
                selects.push(format!("MIN({quoted}), MAX({quoted}), COUNT({quoted})"));
            }
            format!("SELECT {}\nFROM {table}", selects.join(", "))
        }
        DataRequest::RepeatedKey { table, key } => {
            let table = from(d, request, "table", table)?;
            if key.is_empty() {
                return Err(refused(request, "key", "a repeated key names no columns"));
            }
            let quoted = identifiers(d, key);
            let present = present(&key.iter().map(|c| d.identifier(c)).collect::<Vec<_>>());
            format!(
                "SELECT {quoted}\nFROM {table}\nWHERE {present}\nGROUP BY {quoted}\nHAVING COUNT(*) > 1\nLIMIT 1"
            )
        }
        DataRequest::SampleEntities {
            table,
            key,
            n,
            seed,
            columns,
            filters,
            key_dtype,
            rows,
        } => {
            let rendered = from(d, request, "table", table)?;
            if columns.is_empty() {
                return Err(refused(request, "columns", "a sample names no columns"));
            }
            if key.is_empty() || key.iter().any(|k| !columns.contains(k)) {
                return Err(refused(
                    request,
                    "key",
                    "a sample's key is not among the columns it projects",
                ));
            }
            let seed = d.sample_seed(request, seed)?;
            let mut statements = d.sample(&Sample {
                relation: table,
                table: rendered,
                key,
                n: *n,
                seed,
                projected: identifiers(d, columns),
                filtered: conditions(d, request, filters, "\n")?,
                key_dtype: *key_dtype,
                rows: *rows,
            });
            let sample = statements.pop().unwrap_or_default();
            before = statements;
            sample
        }
        DataRequest::Neighbors {
            table,
            key,
            columns,
            seeds,
            window,
            per_seed,
            order,
        } => {
            let read = seed_read(d, request, table, key, columns, seeds, window.as_ref())?;
            let prelude = d.seeds_prelude(&read)?;
            params.push(key_set(&read));
            no_columns(request, columns)?;
            let body = match (per_seed, order) {
                (Some(limit), Some(TieBreak(terms))) => d.neighbors_per_seed(&read, *limit, terms),
                (None, None) => read.joined(),
                _ => d.neighbors_unpaired(&read)?,
            };
            format!("{prelude}{body}")
        }
        DataRequest::Lookup {
            table,
            key,
            columns,
            seeds,
        } => {
            let read = seed_read(d, request, table, key, columns, seeds, None)?;
            let prelude = d.seeds_prelude(&read)?;
            params.push(key_set(&read));
            no_columns(request, columns)?;
            format!("{prelude}{}", d.lookup(&read))
        }
        DataRequest::Scan {
            table,
            columns,
            filters,
            limit,
        } => {
            let table = from(d, request, "table", table)?;
            if columns.is_empty() {
                return Err(refused(request, "columns", "a scan names no columns"));
            }
            let mut text = format!(
                "SELECT {} FROM {table}{}",
                identifiers(d, columns),
                conditions(d, request, filters, " ")?
            );
            if let Some(limit) = limit {
                text.push_str(&format!(" LIMIT {limit}"));
            }
            text
        }
        DataRequest::CountRows { table, filters } => d.count_rows(
            &from(d, request, "table", table)?,
            &conditions(d, request, filters, " ")?,
        ),
    };
    let statement = |text: String, params: Vec<SqlParam>| SqlStatement {
        dialect: Dialect::new(d.name()),
        text,
        params,
    };
    let mut statements: Vec<SqlStatement> = before
        .into_iter()
        .map(|text| statement(text, vec![]))
        .collect();
    statements.push(statement(text, params));
    Ok(FetchPlan::Sql {
        statements,
        expect: d.expected(request),
    })
}

/// `names` as identifiers, joined by commas.
fn identifiers(d: &dyn SqlDialect, names: &[String]) -> String {
    names
        .iter()
        .map(|n| d.identifier(n))
        .collect::<Vec<_>>()
        .join(", ")
}

/// The `WHERE` clause of `filters`, opened by `lead` (a space or a line
/// break), or nothing for none: each a not-null filter, as the reference's
/// sampler writes one, joined by `AND`; another kind of filter is not
/// taken.
pub(crate) fn conditions(
    d: &dyn SqlDialect,
    request: &DataRequest,
    filters: &[Filter],
    lead: &str,
) -> Result<String, CoreError> {
    let mut conditions = Vec::with_capacity(filters.len());
    for filter in filters {
        let Filter::NotNull { column } = filter else {
            return Err(not_taken(
                request,
                "filters",
                "this rendering takes not-null filters only",
            ));
        };
        conditions.push(format!("{} IS NOT NULL", d.identifier(column)));
    }
    if conditions.is_empty() {
        return Ok(String::new());
    }
    Ok(format!("{lead}WHERE {}", conditions.join(" AND ")))
}

/// One `SELECT` per key, in order, labelled by its label or position: the
/// table's rows, the key's distinct values among the rows holding all of
/// it ([`SqlDialect::distinct_keys`] for several columns), and the rows
/// missing part of it.
fn key_counts(
    d: &dyn SqlDialect,
    request: &DataRequest,
    keys: &[TableKey],
) -> Result<String, CoreError> {
    let tables = keys
        .iter()
        .map(|key| from(d, request, "keys", &key.table))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(keys
        .iter()
        .zip(tables)
        .enumerate()
        .map(|(position, (key, table))| {
            let position = key.label_at(position);
            let quoted: Vec<String> = key.columns.iter().map(|c| d.identifier(c)).collect();
            let missing = quoted
                .iter()
                .map(|c| format!("{c} IS NULL"))
                .collect::<Vec<_>>()
                .join(" OR ");
            let nulls = format!("COUNT(CASE WHEN {missing} THEN 1 END)");
            // A key the source keeps unique has one distinct value per row
            // holding all of it, so the reference counts only its nulls
            // (`_key_count_sql`, carried row P2-T2.2-3).
            let distinct = match quoted.as_slice() {
                _ if key.unique => format!("COUNT(*) - {nulls}"),
                [one] => format!("COUNT(DISTINCT {one})"),
                several => d.distinct_keys(several, &table),
            };
            format!(
                "SELECT {position} AS k, COUNT(*) AS n, {distinct} AS d, {nulls} AS z\nFROM {table}"
            )
        })
        .collect::<Vec<_>>()
        .join("\nUNION ALL\n"))
}

/// One `SELECT` per time column, in order, labelled by its position: the
/// smallest and largest values, cast as the dialect compares a time.
fn time_bounds(
    d: &dyn SqlDialect,
    request: &DataRequest,
    columns: &[TableColumn],
) -> Result<String, CoreError> {
    let mut selects = Vec::with_capacity(columns.len());
    for (position, at) in columns.iter().enumerate() {
        let table = from(d, request, "columns", &at.table)?;
        let time = d.as_time(&d.identifier(&at.column));
        selects.push(d.time_bounds_select(position, &time, &table));
    }
    Ok(selects.join("\nUNION ALL\n"))
}

/// A read of `table` against `seeds`, its table rendered.
fn seed_read<'a>(
    d: &'a dyn SqlDialect,
    request: &'a DataRequest,
    table: &'a Relation,
    key: &'a [String],
    columns: &'a [String],
    seeds: &'a KeySet,
    window: Option<&'a Window>,
) -> Result<SeedRead<'a>, CoreError> {
    let seeds_name = d.seeds_for(request, table);
    Ok(SeedRead {
        dialect: d,
        request,
        table: from(d, request, "table", table)?,
        seeds_name,
        key,
        columns,
        seeds,
        window,
    })
}

/// The key set a read binds, under the name it reads it by.
fn key_set(read: &SeedRead<'_>) -> SqlParam {
    SqlParam::KeySet {
        keys: read.seeds.clone(),
        relation: read.seeds_name.clone(),
    }
}

/// Refuses a read against a key set that names no columns.
fn no_columns(request: &DataRequest, columns: &[String]) -> Result<(), CoreError> {
    if columns.is_empty() {
        return Err(refused(request, "columns", "the request names no columns"));
    }
    Ok(())
}
