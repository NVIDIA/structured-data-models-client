// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `SQLSampler` (`rfm/base/sql_sampler.py`) and the five SQL backends'
//! samplers (`rfm/backend/*/sampler.py`): the hop-by-hop traversal pushed
//! down to the database, and each backend's statements for it, written by
//! Phase 6's dialects, with their results read as pandas reads them.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, PoisonError};

use arrow_array::cast::AsArray;
use arrow_array::types::{Decimal128Type, TimestampNanosecondType, UInt64Type};
use arrow_array::{
    Array, ArrayRef, BooleanArray, Float64Array, Int64Array, RecordBatch, StringArray,
    TimestampNanosecondArray, new_empty_array, new_null_array,
};
use arrow_schema::{DataType, TimeUnit};
use sdm_dialects::arrow::{ArrowArray, ArrowType, Unit as ArrowUnit};
use sdm_dialects::fragments::quote_ident;
use sdm_dialects::int::Int;
use sdm_dialects::py::Dtype as PqlDtype;
use sdm_dialects::py::Py;
use sdm_dialects::time::{Civil, Stamp};
use sdm_dialects::{Statement, databricks, duckdb, postgres, snow, sqlite};
use sdm_labels::date_offset::DateOffset;
use sdm_schema::dt::Host;
use sdm_schema::pyval::PyVal;

use super::inputs::{Backend, Inputs, SqlTable, TimeSpelling};
use super::output::Warning;
use super::query::EntityId;
use super::sampler::{Sampler, TIMESTAMP_MAX, TIMESTAMP_MIN, Through};
use super::target::{Columns, TimeOffsets, time_series, times_of};
use super::{Db, Raised, Rows};
use crate::frame::{Dtype, Frame, Series, ints, texts};
use crate::graph_store::rev_edge_type;
use crate::keys::{Key, factorize, first_of_pairs, keys_of, list_repr, np_unique};
use crate::mapper::Mapper;
use crate::neighbor::EdgeType;
use crate::sequences::ListKind;

/// The fields `SnowTable._listing` keeps of a table's listing.
const LISTING_FIELDS: [&str; 7] = [
    "rows",
    "bytes",
    "kind",
    "is_external",
    "is_hybrid",
    "is_iceberg",
    "is_dynamic",
];

/// `Dtype(name)`.
pub fn dtype_of_name(name: &str) -> Option<PqlDtype> {
    PqlDtype::parse(name).ok()
}

/// The column prefix the SQL backends reserve (`RESERVED_COLUMN_PREFIX`).
pub const RESERVED_COLUMN_PREFIX: &str = "__kumo_";

/// What a SQL sampler keeps of its graph (`SQLSampler.__init__` and each
/// backend's).
/// Each table's columns and their dtypes.
type Dtypes = Vec<(String, Vec<(String, Option<PqlDtype>)>)>;

pub struct SqlBackend {
    pub kind: Backend,
    pub source_name_dict: Vec<(String, String)>,
    pub source_table_dict: Vec<(String, SqlTable)>,
    pub table_dtype_dict: Dtypes,
    pub table_column_ref_dict: Vec<(String, Vec<(String, String)>)>,
    pub table_column_proj_dict: Vec<(String, Vec<(String, String)>)>,
    /// The sampler's feature stypes, which `Table._sanitize` is given.
    pub table_stype_dict: Vec<(String, Vec<(String, String)>)>,
    pub enforced_key_tables: Vec<String>,
    pub num_rows: HashMap<String, Option<i64>>,
    /// SQLite's `_time_spelling_dict`, filled as tables are probed.
    pub time_spellings: Mutex<HashMap<String, TimeSpelling>>,
    /// Snowflake's `_listed`, filled as tables are listed.
    pub listed: Mutex<HashMap<String, Vec<sdm_dialects::helpers::Listing>>>,
    /// The warnings `Table._sanitize` gave as results were read, in order,
    /// until the call they belong to takes them.
    pub warned: Mutex<Vec<Warning>>,
    pub host: Host,
}

fn lookup<'a, T>(dict: &'a [(String, T)], key: &str) -> Result<&'a T, Raised> {
    dict.iter()
        .find(|(k, _)| k == key)
        .map(|(_, v)| v)
        .ok_or_else(|| Raised::key_error(key))
}

/// `_reject_reserved_columns(graph)`, over the source's own columns: a
/// column the client derives with an expression (a composite key's
/// `__kumo_key_...`) is named by it, and today's refusing it made every
/// composite key fail on a SQL source.
fn reject_reserved_columns(inputs: &Inputs) -> Result<(), Raised> {
    let mut collisions: Vec<String> = inputs
        .tables
        .iter()
        .flat_map(|t| {
            t.columns
                .iter()
                .filter(|c| c.expr.is_none() && c.name.starts_with(RESERVED_COLUMN_PREFIX))
                .map(move |c| format!("{}.{}", t.name, c.name))
        })
        .collect();
    if collisions.is_empty() {
        return Ok(());
    }
    collisions.sort();
    let names = if collisions.len() > 1 {
        "use names"
    } else {
        "uses a name"
    };
    Err(Raised::value_error(format!(
        "{} {names} reserved by the SQL backends: a column may not start with {}, because the \
         generated joins already select columns of that name and the reference would be \
         ambiguous. Rename the column, or project it under another name.",
        collisions.join(", "),
        sdm_schema::py::str_repr(RESERVED_COLUMN_PREFIX)
    )))
}

impl SqlBackend {
    pub fn new(inputs: &Inputs) -> Result<Self, Raised> {
        reject_reserved_columns(inputs)?;
        let quote = inputs.backend.quote_char();
        let mut source_name_dict = Vec::new();
        let mut source_table_dict = Vec::new();
        let mut table_dtype_dict = Vec::new();
        let mut table_column_ref_dict = Vec::new();
        let mut table_column_proj_dict = Vec::new();
        for table in &inputs.tables {
            let sql = table.sql()?.clone();
            source_name_dict.push((table.name.clone(), sql.source_name.clone()));
            // Only the source columns the graph keeps, as `_source_column_dict`
            // is read for each of its source columns.
            let mut kept = sql.clone();
            kept.source_columns.retain(|c| {
                table
                    .column(&c.name)
                    .is_some_and(|column| column.is_source && column.expr.is_none())
            });
            source_table_dict.push((table.name.clone(), kept));
            table_dtype_dict.push((
                table.name.clone(),
                table
                    .columns
                    .iter()
                    .map(|c| (c.name.clone(), c.dtype.as_deref().and_then(dtype_of_name)))
                    .collect(),
            ));
            let mut refs = Vec::new();
            let mut projections = Vec::new();
            for column in &table.columns {
                match &column.expr {
                    Some(expr) => {
                        refs.push((column.name.clone(), expr.clone()));
                        projections.push((
                            column.name.clone(),
                            format!("{expr} AS {}", quote_ident(&column.name, quote)),
                        ));
                    }
                    None => {
                        let ident = quote_ident(&column.name, quote);
                        refs.push((column.name.clone(), ident.clone()));
                        projections.push((column.name.clone(), ident));
                    }
                }
            }
            table_column_ref_dict.push((table.name.clone(), refs));
            table_column_proj_dict.push((table.name.clone(), projections));
        }
        let mut listed = HashMap::new();
        for table in &inputs.tables {
            if let Some(value) = table.sql.as_ref().and_then(|s| s.listed.as_ref()) {
                listed.insert(table.name.clone(), listing_rows(&Py::decode(value)?)?);
            }
        }
        let foreign_keys: Vec<(&String, &String)> = inputs
            .edges
            .iter()
            .map(|(src, fkey, _)| (src, fkey))
            .collect();
        let table_stype_dict = inputs
            .tables
            .iter()
            .map(|t| {
                let stypes = t
                    .columns
                    .iter()
                    .filter(|c| t.primary_key.as_deref() != Some(c.name.as_str()))
                    .filter(|c| !foreign_keys.contains(&(&t.name, &c.name)))
                    .map(|c| (c.name.clone(), c.stype.clone().unwrap_or_default()))
                    .collect();
                (t.name.clone(), stypes)
            })
            .collect();
        Ok(Self {
            kind: inputs.backend,
            source_name_dict,
            source_table_dict,
            table_dtype_dict,
            table_column_ref_dict,
            table_column_proj_dict,
            table_stype_dict,
            enforced_key_tables: inputs.state.enforced_key_tables.clone(),
            num_rows: inputs.state.num_rows.clone(),
            time_spellings: Mutex::new(inputs.state.time_spellings.clone()),
            listed: Mutex::new(listed),
            warned: Mutex::new(Vec::new()),
            host: Host::assumed(),
        })
    }

    pub fn source_name(&self, table: &str) -> Result<&String, Raised> {
        lookup(&self.source_name_dict, table)
    }

    pub fn source_table(&self, table: &str) -> Result<&SqlTable, Raised> {
        lookup(&self.source_table_dict, table)
    }

    pub fn refs(&self, table: &str) -> Result<&Vec<(String, String)>, Raised> {
        lookup(&self.table_column_ref_dict, table)
    }

    pub fn column_ref(&self, table: &str, column: &str) -> Result<String, Raised> {
        lookup(self.refs(table)?, column).cloned()
    }

    pub fn projections_dict(&self, table: &str) -> Result<&Vec<(String, String)>, Raised> {
        lookup(&self.table_column_proj_dict, table)
    }

    pub fn dtype(&self, table: &str, column: &str) -> Option<PqlDtype> {
        lookup(&self.table_dtype_dict, table)
            .ok()
            .and_then(|d| d.iter().find(|(c, _)| c == column))
            .and_then(|(_, d)| *d)
    }

    /// `table_dtype_dict[table][column]`, or the `KeyError` it raises.
    pub fn dtype_of(&self, table: &str, column: &str) -> Result<Option<PqlDtype>, Raised> {
        lookup(&self.table_dtype_dict, table)?
            .iter()
            .find(|(c, _)| c == column)
            .map(|(_, d)| *d)
            .ok_or_else(|| Raised::key_error(column))
    }

    /// `_ordered_columns(table, columns)`: by their place in the schema,
    /// unknown names last by name.
    pub fn ordered_columns(&self, table: &str, columns: &[String]) -> Result<Vec<String>, Raised> {
        let order = self.projections_dict(table)?;
        let position = |c: &String| {
            order
                .iter()
                .position(|(n, _)| n == c)
                .unwrap_or(order.len())
        };
        let mut out: Vec<String> = Vec::new();
        for c in columns {
            if !out.contains(c) {
                out.push(c.clone());
            }
        }
        out.sort_by(|a, b| {
            position(a)
                .cmp(&position(b))
                .then_with(|| a.chars().cmp(b.chars()))
        });
        Ok(out)
    }

    /// The `SELECT` list of `columns`, in the schema's order.
    pub fn projections(&self, table: &str, columns: &[String]) -> Result<Vec<String>, Raised> {
        match self.kind {
            // `_projections`: driven by the projection dict, unknown names
            // refused.
            Backend::Postgres | Backend::Databricks => {
                let proj = self.projections_dict(table)?;
                let mut missing: Vec<&String> = columns
                    .iter()
                    .filter(|c| !proj.iter().any(|(n, _)| n == *c))
                    .collect();
                if !missing.is_empty() {
                    missing.sort();
                    missing.dedup();
                    return Err(Raised::key_error(&format!(
                        "Unknown columns requested for table {}: {}",
                        sdm_schema::py::str_repr(table),
                        missing
                            .iter()
                            .map(|s| s.as_str())
                            .collect::<Vec<_>>()
                            .join(", ")
                    )));
                }
                Ok(proj
                    .iter()
                    .filter(|(n, _)| columns.contains(n))
                    .map(|(_, p)| p.clone())
                    .collect())
            }
            _ => {
                let proj = self.projections_dict(table)?;
                self.ordered_columns(table, columns)?
                    .iter()
                    .map(|c| lookup(proj, c).cloned())
                    .collect()
            }
        }
    }

    /// `_is_unique(table, column)`.
    pub fn is_unique(&self, sampler: &Sampler, table: &str, column: &str) -> bool {
        self.enforced_key_tables.iter().any(|t| t == table)
            && sampler.primary_key(table).is_ok_and(|k| k == column)
    }

    /// `_tie_break_term(table, column, ref)`.
    fn tie_break_term(&self, table: &str, column: &str, reference: &str) -> String {
        match self.kind {
            Backend::Postgres => postgres::tie_break_term(
                reference,
                self.dtype(table, column) == Some(PqlDtype::String),
            ),
            _ => reference.to_string(),
        }
    }

    /// `_tie_break_terms(table, columns, skip)`.
    pub fn tie_break_terms(
        &self,
        table: &str,
        columns: &[String],
        skip: &[String],
    ) -> Result<Vec<String>, Raised> {
        let refs = self.refs(table)?;
        let mut sorted: Vec<&String> = columns.iter().collect();
        sorted.sort_by(|a, b| a.chars().cmp(b.chars()));
        sorted.dedup();
        let mut terms = Vec::new();
        for column in sorted {
            let Some(reference) = refs.iter().find(|(c, _)| c == column).map(|(_, r)| r) else {
                continue;
            };
            if skip.contains(reference) {
                continue;
            }
            if self.dtype(table, column).is_some_and(PqlDtype::is_list) {
                continue;
            }
            terms.push(self.tie_break_term(table, column, reference));
        }
        Ok(terms)
    }

    /// `_neighbor_order_by(table, time_ref, key_ref, columns, latest_first)`.
    pub fn neighbor_order_by(
        &self,
        sampler: &Sampler,
        table: &str,
        time_ref: Option<&str>,
        key_ref: &str,
        columns: Option<&[String]>,
        latest_first: bool,
    ) -> Result<String, Raised> {
        let mut terms = Vec::new();
        if let Some(time_ref) = time_ref {
            terms.push(if latest_first {
                format!("{time_ref} DESC")
            } else {
                time_ref.to_string()
            });
        }
        if let Ok(primary_key) = sampler.primary_key(table) {
            let primary_ref = self.column_ref(table, primary_key)?;
            terms.push(primary_ref.clone());
            if let Some(columns) = columns
                && !self.is_unique(sampler, table, primary_key)
            {
                let mut skip = vec![primary_ref];
                if let Some(time_ref) = time_ref {
                    skip.push(time_ref.to_string());
                }
                terms.extend(self.tie_break_terms(table, columns, &skip)?);
            }
            return Ok(terms.join(", "));
        }
        let mut refs: Vec<&(String, String)> = self.refs(table)?.iter().collect();
        refs.sort_by(|a, b| {
            a.0.chars()
                .cmp(b.0.chars())
                .then(a.1.chars().cmp(b.1.chars()))
        });
        for (column, reference) in refs {
            if Some(reference.as_str()) == time_ref
                || self.dtype(table, column).is_some_and(PqlDtype::is_list)
            {
                continue;
            }
            terms.push(self.tie_break_term(table, column, reference));
        }
        if terms.is_empty() {
            terms.push(key_ref.to_string());
        }
        Ok(terms.join(", "))
    }

    /// `_rows_order_by(table, columns)`.
    pub fn rows_order_by(
        &self,
        sampler: &Sampler,
        table: &str,
        columns: &[String],
    ) -> Result<String, Raised> {
        let primary_key = sampler.primary_key(table)?;
        let mut terms = vec![self.column_ref(table, primary_key)?];
        if !self.is_unique(sampler, table, primary_key) {
            let skip = terms.clone();
            terms.extend(self.tie_break_terms(table, columns, &skip)?);
        }
        Ok(terms.join(", "))
    }

    /// DuckDB's `_time_ref(table, column)`.
    fn duckdb_time_ref(&self, table: &str, column: &str) -> Result<String, Raised> {
        let zoned = self.source_table(table)?.zoned.iter().any(|c| c == column);
        Ok(duckdb::time_column(&self.column_ref(table, column)?, zoned))
    }

    /// `_get_min_max_time_dict(table_names)`: each table's earliest and
    /// latest time, with the unit of the `Timestamp` each is read as.
    #[allow(clippy::type_complexity)]
    pub async fn min_max_time_dict(
        &self,
        sampler: &Sampler,
        db: &Db,
        table_names: &[String],
    ) -> Result<Vec<(String, ((i64, String), (i64, String)))>, Raised> {
        let mut tables = Vec::new();
        for table in table_names {
            let column = sampler
                .time_column(table)
                .ok_or_else(|| Raised::key_error(table))?;
            let reference = match self.kind {
                Backend::Duckdb => self.duckdb_time_ref(table, column)?,
                _ => self.column_ref(table, column)?,
            };
            tables.push((self.source_name(table)?.clone(), reference));
        }
        let statement = match self.kind {
            Backend::Duckdb => duckdb::time_ranges(&tables),
            Backend::Sqlite => sqlite::time_ranges(&tables),
            Backend::Postgres => postgres::time_ranges(&tables),
            Backend::Snowflake => snow::time_ranges(&tables),
            Backend::Databricks => databricks::time_ranges(&tables),
            Backend::Local => return Err(Raised::internal("a local sampler has no SQL")),
        };
        let rows = db.run(statement).await?;
        let values = row_values(&rows, self.kind == Backend::Postgres);
        let mut out = Vec::new();
        for row in values {
            if row.len() < 3 {
                return Err(Raised::value_error("not enough values to unpack"));
            }
            let index = py_index(&row[0])?;
            let table = table_names
                .get(index)
                .ok_or_else(|| Raised::new("IndexError", "list index out of range"))?;
            let (source, reference) = &tables[index];
            let min = match &row[1] {
                PyVal::None => (TIMESTAMP_MAX, "ns".to_string()),
                value => match timestamp_of(value, &self.host) {
                    Ok(stamp) => stamp,
                    Err(_) => self
                        .first_readable(db, source, reference, "ASC")
                        .await?
                        .unwrap_or((TIMESTAMP_MAX, "ns".to_string())),
                },
            };
            let max = match &row[2] {
                PyVal::None => (TIMESTAMP_MIN, "ns".to_string()),
                value => match timestamp_of(value, &self.host) {
                    Ok(stamp) => stamp,
                    Err(_) => self
                        .first_readable(db, source, reference, "DESC")
                        .await?
                        .unwrap_or((TIMESTAMP_MIN, "ns".to_string())),
                },
            };
            out.push((table.clone(), (min, max)));
        }
        Ok(out)
    }

    /// The earliest (`ASC`) or latest (`DESC`) time of a column that reads
    /// as one, where `MIN` or `MAX` gave text that does not: an unreadable
    /// time is dropped wherever it is, and today's failed the whole request
    /// on it ("Expected pd.Timestamp, got NaT") when it sorted first or last.
    async fn first_readable(
        &self,
        db: &Db,
        source: &str,
        reference: &str,
        order: &str,
    ) -> Result<Option<(i64, String)>, Raised> {
        const PAGE: usize = 1000;
        let mut offset = 0;
        loop {
            let statement = format!(
                "SELECT DISTINCT {reference} AS __kumo_time__ FROM {source} WHERE {reference} IS \
                 NOT NULL ORDER BY 1 {order} LIMIT {PAGE} OFFSET {offset}"
            );
            let rows = db.run((statement, None)).await?;
            let values = row_values(&rows, self.kind == Backend::Postgres);
            for row in &values {
                if let Some(Ok(stamp)) = row.first().map(|v| timestamp_of(v, &self.host)) {
                    return Ok(Some(stamp));
                }
            }
            if values.len() < PAGE {
                return Ok(None);
            }
            offset += PAGE;
        }
    }

    /// `_sample_entity_table(table, columns, num_rows, random_seed,
    /// entity_ids)`.
    #[allow(clippy::too_many_arguments)]
    pub async fn sample_entity_table(
        &self,
        sampler: &Sampler,
        db: &Db,
        table: &str,
        columns: &[String],
        num_rows: i64,
        random_seed: Option<&Int>,
        entity_ids: Option<&[EntityId]>,
    ) -> Result<Frame, Raised> {
        let source_table = self.source_table(table)?;
        let key = sampler.primary_key(table)?.clone();
        let key_ref = self.column_ref(table, &key)?;
        let nullable = |column: &str| {
            source_table
                .source_column(column)
                .is_none_or(|c| c.is_nullable)
        };
        let leading = if self.kind == Backend::Postgres {
            ""
        } else {
            " "
        };
        let mut filters = Vec::new();
        if nullable(&key) {
            filters.push(format!("{leading}{key_ref} IS NOT NULL"));
        }
        if let Some(column) = sampler.time_column(table)
            && nullable(column)
        {
            filters.push(format!(
                "{leading}{} IS NOT NULL",
                self.column_ref(table, column)?
            ));
        }
        let source = self.source_name(table)?.clone();
        let ids_params = entity_ids.map(|ids| ids.iter().map(entity_param).collect::<Vec<_>>());
        let rows = match self.kind {
            Backend::Duckdb => {
                let projections = self.projections(table, columns)?;
                let order_by = self.rows_order_by(sampler, table, columns)?;
                let statement = match &ids_params {
                    None => duckdb::random_rows(
                        &source,
                        &projections,
                        &filters,
                        num_rows,
                        random_seed,
                        &order_by,
                    ),
                    Some(ids) => duckdb::rows_by_ids(
                        &source,
                        &projections,
                        &filters,
                        &key_ref,
                        ids,
                        &order_by,
                    ),
                };
                db.run(statement).await?
            }
            Backend::Sqlite => {
                let projections = self.projections(table, columns)?;
                let order_by = match (&ids_params, random_seed) {
                    (None, None) => None,
                    (None, Some(seed)) => {
                        Some(self.sqlite_seeded_order_by(sampler, table, columns, seed)?)
                    }
                    (Some(_), _) => Some(self.rows_order_by(sampler, table, columns)?),
                };
                db.run(sqlite::entity_rows(
                    &source,
                    &projections,
                    &filters,
                    order_by.as_deref(),
                    Some(num_rows),
                    Some(&key_ref),
                    ids_params.as_deref(),
                ))
                .await?
            }
            Backend::Postgres => {
                let projections = self.projections(table, columns)?;
                match entity_ids {
                    None => {
                        if let Some(seed) = random_seed {
                            db.run(postgres::set_seed(seed)).await?;
                        }
                        let rows = db
                            .run(postgres::random_rows(
                                &source,
                                &projections,
                                &filters,
                                num_rows,
                            ))
                            .await?;
                        return self.postgres_frame(table, &rows);
                    }
                    Some(ids) => {
                        let dtype = self
                            .dtype_of(table, &key)?
                            .ok_or_else(|| Raised::internal("a key without a dtype"))?;
                        let postgres_type = self.postgres_type(table, &key)?;
                        let order_by = self.rows_order_by(sampler, table, columns)?;
                        let coerced = ids
                            .iter()
                            .map(|id| postgres::coerce_id(&entity_py(id), dtype, &postgres_type))
                            .collect::<Result<Vec<_>, _>>()?;
                        let frame = self
                            .postgres_flatten(db, &coerced, |chunk| {
                                postgres::rows_by_ids(
                                    &source,
                                    &projections,
                                    &filters,
                                    &key_ref,
                                    &postgres_type,
                                    &Py::List(chunk.to_vec()),
                                    &order_by,
                                )
                            })
                            .await?;
                        return self.sanitize_postgres(table, frame);
                    }
                }
            }
            Backend::Snowflake => {
                let num_rows = num_rows.min(1_000_000);
                let projections = self.projections(table, columns)?;
                let order_by = self.rows_order_by(sampler, table, columns)?;
                let seeded = match (entity_ids, random_seed) {
                    (None, Some(seed)) => Some(self.snow_seeded_key_hash(table, &key, seed)?),
                    _ => None,
                };
                let prefilter = match (&seeded, random_seed) {
                    (Some(_), Some(seed)) => {
                        self.snow_prefilter(db, table, 100_000.max(10 * num_rows), seed)
                            .await?
                    }
                    _ => None,
                };
                let draw = |prefilter: Option<&str>| {
                    snow::entity_rows(
                        &source,
                        &projections,
                        &filters,
                        &order_by,
                        num_rows,
                        seeded.as_deref(),
                        Some(&key_ref),
                        ids_params.as_deref(),
                        prefilter,
                    )
                };
                let mut rows = db.run(draw(prefilter.as_deref())).await?;
                if prefilter.is_some() && (rows.num_rows() as i64) < num_rows {
                    rows = db.run(draw(None)).await?;
                }
                rows
            }
            Backend::Databricks => {
                let projections = self.projections(table, columns)?;
                let percent = match entity_ids {
                    None => match self.num_rows.get(table).copied().flatten() {
                        Some(total) if total > num_rows => Some(f64::min(
                            100.0,
                            100.0 * num_rows as f64 / total as f64 * 3.0,
                        )),
                        _ => Some(100.0),
                    },
                    Some(_) => None,
                };
                let ids_py: Option<Vec<Py>> =
                    entity_ids.map(|ids| ids.iter().map(entity_py).collect());
                let order_by = self.rows_order_by(sampler, table, columns)?;
                db.run(databricks::entity_rows(
                    &source,
                    &projections,
                    &filters,
                    &key_ref,
                    &order_by,
                    percent,
                    random_seed,
                    Some(num_rows),
                    ids_py.as_deref(),
                    self.dtype(table, &key),
                )?)
                .await?
            }
            Backend::Local => return Err(Raised::internal("a local sampler has no SQL")),
        };
        self.arrow_frame(sampler, table, &rows)
    }

    /// SQLite's `_seeded_order_by(table, columns, random_seed)`.
    fn sqlite_seeded_order_by(
        &self,
        sampler: &Sampler,
        table: &str,
        columns: &[String],
        seed: &Int,
    ) -> Result<String, Raised> {
        let key = sampler.primary_key(table)?;
        let dtype = self
            .source_table(table)?
            .source_column(key)
            .and_then(|c| c.dtype.as_deref().and_then(dtype_of_name))
            .or_else(|| self.dtype(table, key));
        if dtype.is_some_and(PqlDtype::is_int) {
            let key_ref = self.column_ref(table, key)?;
            let hashed = sqlite::seeded_hash(seed, &key_ref)?;
            return Ok(format!(
                "{hashed}, {}",
                self.rows_order_by(sampler, table, columns)?
            ));
        }
        Ok(format!("{}, rowid", sqlite::seeded_hash(seed, "rowid")?))
    }

    /// Snowflake's `_seeded_key_hash(table, key, seed)`.
    fn snow_seeded_key_hash(&self, table: &str, key: &str, seed: &Int) -> Result<String, Raised> {
        let key_ref = self.column_ref(table, key)?;
        let source = self.source_table(table)?;
        let dtype = source
            .source_column(key)
            .and_then(|c| c.dtype.as_deref().and_then(dtype_of_name))
            .or_else(|| self.dtype(table, key));
        let is_date = source.source_column(key).is_some()
            && source
                .snowflake_types
                .iter()
                .any(|(c, t)| c == key && t == "DATE");
        Ok(snow::seeded_key_hash(&key_ref, dtype, is_date, seed))
    }

    /// `SnowTable._prefilter(rows, seed)`, listing the table where it was
    /// not listed yet.
    async fn snow_prefilter(
        &self,
        db: &Db,
        table: &str,
        rows: i64,
        seed: &Int,
    ) -> Result<Option<String>, Raised> {
        let listed = self.snow_listed(db, table).await?;
        let identity = self
            .source_table(table)?
            .identity
            .clone()
            .ok_or_else(|| Raised::internal("a Snowflake table without its identity"))?;
        let listing = match listed {
            None => None,
            Some(rows) => sdm_dialects::helpers::snow_listing(
                &identity.2,
                Some(&rows),
                &LISTING_FIELDS.map(str::to_string),
            )?,
        };
        Ok(sdm_dialects::helpers::snow_prefilter(
            &Int::from(rows),
            seed,
            listing.as_ref(),
            &["TABLE".into(), "TRANSIENT".into(), "TEMPORARY".into()],
            &Int::from(128),
            &Int::from(16 << 20),
        )?)
    }

    /// Snowflake's `_listed`: what `SHOW TABLES` lists of tables named like
    /// this one; `None` where listing fails.
    async fn snow_listed(
        &self,
        db: &Db,
        table: &str,
    ) -> Result<Option<Vec<sdm_dialects::helpers::Listing>>, Raised> {
        let known = self
            .listed
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(table)
            .cloned();
        if let Some(rows) = known {
            return Ok(Some(rows));
        }
        let (database, schema, name) = self
            .source_table(table)?
            .identity
            .clone()
            .ok_or_else(|| Raised::internal("a Snowflake table without its identity"))?;
        let rows = match db.run(snow::tables_like(&database, &schema, &name)).await {
            Ok(rows) => rows,
            // `_listing` reads none where listing raises.
            Err(_) => return Ok(None),
        };
        let fields: Vec<String> = rows
            .schema
            .fields()
            .iter()
            .map(|f| f.name().to_lowercase())
            .collect();
        let listed: Vec<sdm_dialects::helpers::Listing> = row_values(&rows, false)
            .into_iter()
            .map(|row| {
                fields
                    .iter()
                    .cloned()
                    .zip(row.iter().map(pyval_py))
                    .collect()
            })
            .collect();
        self.listed
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(table.to_string(), listed.clone());
        Ok(Some(listed))
    }

    /// PostgreSQL's `_postgres_type(table, column)`.
    fn postgres_type(&self, table: &str, column: &str) -> Result<String, Raised> {
        let source = self.source_table(table)?;
        lookup(&source.postgres_types, column).cloned()
    }

    /// PostgreSQL's `_flatten(statement, rows)`: the statement for each chunk
    /// of `rows`, the frames' `__KUMO_BATCH__` counted over all of `rows`.
    async fn postgres_flatten(
        &self,
        db: &Db,
        rows: &[Py],
        statement: impl Fn(&[Py]) -> Result<Statement, sdm_dialects::Error>,
    ) -> Result<Frame, Raised> {
        let chunks = postgres::chunk_rows(rows, 900_000)?;
        let mut frames: Vec<Frame> = Vec::new();
        let mut offset = 0i64;
        for chunk in chunks {
            let result = db.run(statement(&chunk)?).await?;
            let mut frame = postgres_object_frame(&result)?;
            if let Some(i) = frame.position("__KUMO_BATCH__") {
                let shifted: Vec<PyVal> = object_cells(&frame.columns[i])
                    .into_iter()
                    .map(|v| match v {
                        PyVal::Int(n) => PyVal::Int(n + i128::from(offset)),
                        other => other,
                    })
                    .collect();
                frame.columns[i] = object_series(shifted)?;
            }
            frames.push(frame);
            offset += chunk.len() as i64;
        }
        if frames.is_empty() {
            return Ok(Frame::default());
        }
        Frame::concat(&frames.iter().collect::<Vec<_>>())
    }

    /// A result read with `to_pandas(types_mapper=pd.ArrowDtype)` and
    /// sanitized (`Table._sanitize`).
    pub fn arrow_frame(
        &self,
        sampler: &Sampler,
        table: &str,
        rows: &Rows,
    ) -> Result<Frame, Raised> {
        let frame = arrow_result_frame(rows)?;
        self.sanitized(sampler, table, frame)
    }

    /// `Table._sanitize(frame, ...)` with the sampler's types, its warnings
    /// kept for the call.
    pub fn sanitized(&self, sampler: &Sampler, table: &str, frame: Frame) -> Result<Frame, Raised> {
        let mut warnings = Vec::new();
        let frame = sanitize(sampler, &self.host, table, frame, &mut warnings)?;
        self.keep(warnings);
        Ok(frame)
    }

    fn keep(&self, warnings: Vec<Warning>) {
        self.warned
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .extend(warnings);
    }

    /// The warnings kept since they were last taken, in the order given.
    pub fn take_warnings(&self) -> Vec<Warning> {
        std::mem::take(&mut *self.warned.lock().unwrap_or_else(PoisonError::into_inner))
    }

    /// PostgreSQL's `_sanitize_postgres_frame(_frame(result))`.
    fn postgres_frame(&self, table: &str, rows: &Rows) -> Result<Frame, Raised> {
        let frame = postgres_object_frame(rows)?;
        self.sanitize_postgres(table, frame)
    }

    fn sanitize_postgres(&self, table: &str, mut frame: Frame) -> Result<Frame, Raised> {
        let dtypes = lookup(&self.table_dtype_dict, table)?.clone();
        for (name, dtype) in &dtypes {
            let Some(i) = frame.position(name) else {
                continue;
            };
            let cells = object_cells(&frame.columns[i]);
            let converted = match dtype {
                Some(d) if d.is_int() => Some(nullable_ints(&cells)?),
                Some(d) if d.is_float() => Some(nullable_floats(&cells)?),
                Some(PqlDtype::Bool) => Some(nullable_bools(&cells)?),
                Some(PqlDtype::String) => Some(nullable_strings(&cells)?),
                _ => None,
            };
            if let Some(series) = converted {
                frame.columns[i] = series;
            }
        }
        let stypes = lookup(&self.table_stype_dict, table)?;
        let mut warnings = Vec::new();
        let frame = sanitize_with(&self.host, &dtypes, stypes, frame, &mut warnings)?;
        self.keep(warnings);
        Ok(frame)
    }
}

// --- Reading results -------------------------------------------------------

/// The columns of a result as one array each.
fn result_columns(rows: &Rows) -> Result<Vec<(String, ArrayRef)>, Raised> {
    rows.schema
        .fields()
        .iter()
        .enumerate()
        .map(|(i, field)| {
            let arrays: Vec<&dyn Array> =
                rows.batches.iter().map(|b| b.column(i).as_ref()).collect();
            let array = match arrays.len() {
                0 => new_empty_array(field.data_type()),
                1 => rows.batches[0].column(i).clone(),
                _ => arrow_select::concat::concat(&arrays)
                    .map_err(|e| Raised::internal(format!("cannot read a result: {e}")))?,
            };
            Ok((field.name().clone(), array))
        })
        .collect()
}

/// `table.to_pandas(types_mapper=pd.ArrowDtype)`.
pub fn arrow_result_frame(rows: &Rows) -> Result<Frame, Raised> {
    let columns = result_columns(rows)?;
    let n = rows.num_rows();
    Ok(Frame::new(
        columns.iter().map(|(n, _)| n.clone()).collect(),
        columns
            .into_iter()
            .map(|(_, a)| Series::new(Dtype::Arrow, a))
            .collect(),
        n,
    ))
}

/// Each row of a result as the engine's `rows(table)` reads it, or
/// PostgreSQL's `_rows`.
pub(crate) fn row_values(rows: &Rows, postgres: bool) -> Vec<Vec<PyVal>> {
    let table = sdm_schema::engine::result::ResultTable::from_batches(&rows.schema, &rows.batches);
    if postgres {
        table.pg_rows()
    } else {
        table.rows()
    }
}

fn py_index(value: &PyVal) -> Result<usize, Raised> {
    match value {
        PyVal::Int(i) => {
            usize::try_from(*i).map_err(|_| Raised::new("IndexError", "list index out of range"))
        }
        PyVal::Float(f) => Ok(*f as usize),
        PyVal::Decimal(text) => text
            .split('.')
            .next()
            .and_then(|t| t.parse().ok())
            .ok_or_else(|| Raised::value_error("a table index")),
        other => Err(Raised::new(
            "TypeError",
            format!("int() of {other:?} is not supported by the core"),
        )),
    }
}

/// `base/utils.Timestamp(value)`: naive UTC nanoseconds, and the unit of
/// the `Timestamp`: a pandas `Timestamp` keeps its own, a `datetime` and
/// most text are microseconds, text with more digits nanoseconds.
fn timestamp_of(value: &PyVal, host: &Host) -> Result<(i64, String), Raised> {
    let (text, unit) = match value {
        PyVal::Timestamp(wall) => {
            return i64::try_from(wall.epoch_nanos())
                .map(|ns| (ns, "ns".to_string()))
                .map_err(|_| {
                    Raised::new("OutOfBoundsDatetime", "Out of bounds nanosecond timestamp")
                });
        }
        PyVal::DateTime(wall, offset) => {
            let utc = wall.epoch_nanos() - i128::from(offset.unwrap_or(0)) * 1_000_000_000;
            return i64::try_from(utc)
                .map(|ns| (ns, "us".to_string()))
                .map_err(|_| {
                    Raised::new("OutOfBoundsDatetime", "Out of bounds nanosecond timestamp")
                });
        }
        PyVal::Str(s) => {
            let digits = s.split_once('.').map_or(0, |(_, rest)| {
                rest.chars().take_while(char::is_ascii_digit).count()
            });
            (s.clone(), if digits > 6 { "ns" } else { "us" })
        }
        other => (other.str_(), "us"),
    };
    let array: ArrayRef = Arc::new(StringArray::from(vec![text]));
    let converted = crate::graph_store::to_datetime(&array, host)
        .map_err(|e| Raised::new(&e.code, e.message))?;
    converted
        .into_iter()
        .next()
        .flatten()
        .map(|ns| (ns, unit.to_string()))
        .ok_or_else(|| Raised::new("TypeError", "Expected pd.Timestamp, got NaT"))
}

/// What the Python value of a cell writes as text, for a frame built from
/// `PyVal`s.
fn object_cells(series: &Series) -> Vec<PyVal> {
    let array = &series.array;
    (0..array.len())
        .map(|i| sdm_schema::engine::result::value_at(array, i))
        .collect()
}

/// An `object` column of Python values, as `from_pandas` reads them.
fn object_series(cells: Vec<PyVal>) -> Result<Series, Raised> {
    Ok(Series::new(Dtype::Object, pyvals_to_arrow(&cells)?))
}

/// PostgreSQL's `_frame(result)`: an `object` frame of the values psycopg
/// reads, the helper columns renamed.
fn postgres_object_frame(rows: &Rows) -> Result<Frame, Raised> {
    let table = sdm_schema::engine::result::ResultTable::from_batches(&rows.schema, &rows.batches);
    let values = table.pg_columns(false);
    let mut names = Vec::new();
    let mut columns = Vec::new();
    for ((column, cells), array) in table.columns.iter().zip(values).zip(columns_of(rows)?) {
        // A `time` psycopg reads as a `datetime.time`, which the result's
        // values do not say.
        let cells = match array.data_type() {
            DataType::Time32(_) | DataType::Time64(_) => (0..array.len())
                .map(|i| {
                    if array.is_null(i) {
                        PyVal::None
                    } else {
                        time_of_day(&array, i).unwrap_or(PyVal::None)
                    }
                })
                .collect(),
            _ => cells,
        };
        // A PostgreSQL array psycopg reads as Python lists, which stay lists
        // in the `object` frame; `json` and `jsonb` containers are kept as
        // the text `_sanitize_postgres_frame` writes them back as.
        let json = matches!(column.oid(), Some("114" | "3802"));
        let lists = !json
            && cells.iter().any(|c| matches!(c, PyVal::List(_)))
            && cells
                .iter()
                .all(|c| matches!(c, PyVal::List(_) | PyVal::None | PyVal::Na));
        let name = match column.name() {
            "__kumo_batch__" => "__KUMO_BATCH__",
            "__kumo_row__" => "__KUMO_ROW__",
            "__kumo_matches__" => "__KUMO_MATCHES__",
            other => other,
        };
        names.push(name.to_string());
        columns.push(if lists {
            pylists_to_series(&cells)?
        } else {
            object_series(cells)?
        });
    }
    Ok(Frame::new(names, columns, table.num_rows))
}

/// `pa.array(lists)`: Python lists, or `None`, as one `list<item>` array
/// whose items' type is inferred over every item, as for a column.
fn pylists_to_series(cells: &[PyVal]) -> Result<Series, Raised> {
    let items = cells
        .iter()
        .flat_map(|cell| match cell {
            PyVal::List(items) => items.as_slice(),
            _ => &[],
        })
        .cloned()
        .collect::<Vec<_>>();
    if items.iter().any(|value| matches!(value, PyVal::Decimal(_)))
        && crate::frame::decimal_lists::compatible(cells)
    {
        return crate::frame::decimal_lists::series(cells);
    }
    crate::frame::decimal_lists::list_series(cells, pyvals_to_arrow(&items)?)
}

/// Each column of a result as one array.
fn columns_of(rows: &Rows) -> Result<Vec<ArrayRef>, Raised> {
    Ok(result_columns(rows)?
        .into_iter()
        .map(|(_, array)| array)
        .collect())
}

/// Arrow values for Python objects, as `pa.array(objects)` infers them.
fn pyvals_to_arrow(cells: &[PyVal]) -> Result<ArrayRef, Raised> {
    let present: Vec<&PyVal> = cells
        .iter()
        .filter(|c| !matches!(c, PyVal::None | PyVal::Na))
        .collect();
    if present.is_empty() {
        return Ok(new_null_array(&DataType::Null, cells.len()));
    }
    if present.iter().all(|c| matches!(c, PyVal::Bool(_))) {
        return Ok(Arc::new(BooleanArray::from(
            cells
                .iter()
                .map(|c| match c {
                    PyVal::Bool(b) => Some(*b),
                    _ => None,
                })
                .collect::<Vec<_>>(),
        )));
    }
    if present.iter().all(|c| matches!(c, PyVal::Int(_))) {
        return Ok(Arc::new(Int64Array::from(
            cells
                .iter()
                .map(|c| match c {
                    PyVal::Int(i) => i64::try_from(*i).ok(),
                    _ => None,
                })
                .collect::<Vec<_>>(),
        )));
    }
    if present
        .iter()
        .all(|c| matches!(c, PyVal::Int(_) | PyVal::Float(_)))
    {
        return Ok(Arc::new(Float64Array::from(
            cells
                .iter()
                .map(|c| match c {
                    PyVal::Int(i) => Some(*i as f64),
                    PyVal::Float(f) => Some(*f),
                    _ => None,
                })
                .collect::<Vec<_>>(),
        )));
    }
    if present.iter().all(|c| matches!(c, PyVal::Str(_))) {
        return Ok(Arc::new(StringArray::from(
            cells
                .iter()
                .map(|c| match c {
                    PyVal::Str(s) => Some(s.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>(),
        )));
    }
    if present
        .iter()
        .all(|c| matches!(c, PyVal::DateTime(_, None) | PyVal::Timestamp(_)))
    {
        // Naive `datetime`s are `timestamp[us]`; `Timestamp`s keep their
        // nanoseconds.
        let nanos = present.iter().any(|c| matches!(c, PyVal::Timestamp(_)));
        let mut out = Vec::with_capacity(cells.len());
        for c in cells {
            out.push(match c {
                PyVal::DateTime(wall, _) | PyVal::Timestamp(wall) => {
                    let ns = wall.epoch_nanos();
                    Some(if nanos { ns as i64 } else { (ns / 1000) as i64 })
                }
                _ => None,
            });
        }
        return Ok(if nanos {
            Arc::new(TimestampNanosecondArray::from(out))
        } else {
            Arc::new(arrow_array::TimestampMicrosecondArray::from(out))
        });
    }
    if present.iter().all(|c| matches!(c, PyVal::Decimal(_))) {
        // `numeric`, read as exact decimals: one scale for all, as
        // `pa.array(decimals)` makes one. `NaN` and `Infinity` no decimal
        // array holds; `_sanitize_postgres_frame` casts the column to
        // `Float64` or `Int64` anyway, so they are kept as the floats that
        // cast reads them as.
        if let Some(array) = decimal_array(cells) {
            return Ok(Arc::new(array));
        }
        return Ok(Arc::new(Float64Array::from(
            cells
                .iter()
                .map(|c| match c {
                    PyVal::Decimal(text) => text.parse::<f64>().ok(),
                    _ => None,
                })
                .collect::<Vec<_>>(),
        )));
    }
    if present
        .iter()
        .all(|c| matches!(c, PyVal::Time { offset: None, .. }))
    {
        return Ok(Arc::new(arrow_array::Time64MicrosecondArray::from(
            cells
                .iter()
                .map(|c| match c {
                    PyVal::Time {
                        hour,
                        minute,
                        second,
                        micro,
                        ..
                    } => Some(
                        (i64::from(*hour) * 3600 + i64::from(*minute) * 60 + i64::from(*second))
                            * 1_000_000
                            + i64::from(*micro),
                    ),
                    _ => None,
                })
                .collect::<Vec<_>>(),
        )));
    }
    if present.iter().all(|c| matches!(c, PyVal::Date(..))) {
        let days: Vec<Option<i32>> = cells
            .iter()
            .map(|c| match c {
                PyVal::Date(y, m, d) => {
                    i32::try_from(sdm_dialects::time::days_from_civil(*y, *m, *d)).ok()
                }
                _ => None,
            })
            .collect();
        return Ok(Arc::new(arrow_array::Date32Array::from(days)));
    }
    if present
        .iter()
        .any(|c| matches!(c, PyVal::Dict(_) | PyVal::List(_)))
    {
        // `json` and `jsonb`, which psycopg reads with `json.loads` and
        // `_sanitize_postgres_frame` writes back as text: containers with
        // `json.dumps(value, separators=(',', ':'))`, the rest with `str`.
        return Ok(Arc::new(StringArray::from(
            cells
                .iter()
                .map(|c| match c {
                    PyVal::None | PyVal::Na => None,
                    PyVal::Dict(_) | PyVal::List(_) => Some(json_dumps(c)),
                    PyVal::Str(s) => Some(s.clone()),
                    other => Some(other.str_()),
                })
                .collect::<Vec<_>>(),
        )));
    }
    if present
        .iter()
        .all(|c| matches!(c, PyVal::DateTime(_, Some(_))))
    {
        // Zoned `datetime`s, as the instants they name in UTC.
        let out: Vec<Option<i64>> = cells
            .iter()
            .map(|c| match c {
                PyVal::DateTime(wall, Some(offset)) => {
                    Some(((wall.epoch_nanos() - i128::from(*offset) * 1_000_000_000) / 1000) as i64)
                }
                _ => None,
            })
            .collect();
        return Ok(Arc::new(
            arrow_array::TimestampMicrosecondArray::from(out).with_timezone("UTC"),
        ));
    }
    Err(Raised::new(
        "TypeError",
        format!(
            "an object column of {:?} is not supported by the core",
            present.first()
        ),
    ))
}

/// `Decimal`s as one `decimal128` array of their largest scale, `None`
/// where one is not finite or needs more than 38 digits.
fn decimal_array(cells: &[PyVal]) -> Option<arrow_array::Decimal128Array> {
    let parsed: Vec<Option<(i128, u32)>> = cells
        .iter()
        .map(|c| match c {
            PyVal::Decimal(text) => decimal_parts(text).map(Some),
            _ => Some(None),
        })
        .collect::<Option<_>>()?;
    let scale = parsed.iter().flatten().map(|&(_, s)| s).max().unwrap_or(0);
    if scale > 38 {
        return None;
    }
    let values: Vec<Option<i128>> = parsed
        .iter()
        .map(|p| match p {
            None => Some(None),
            Some((digits, s)) => digits
                .checked_mul(10i128.checked_pow(scale - s)?)
                .filter(|v| v.unsigned_abs() < 10u128.pow(38))
                .map(Some),
        })
        .collect::<Option<_>>()?;
    arrow_array::Decimal128Array::from(values)
        .with_precision_and_scale(38, i8::try_from(scale).ok()?)
        .ok()
}

/// A `Decimal`'s text as its digits and scale: `-14.610` is `(-14610, 3)`,
/// `1.5E+3` is `(1500, 0)`; `None` for `NaN` and `Infinity`.
fn decimal_parts(text: &str) -> Option<(i128, u32)> {
    let (mantissa, exponent) = match text.find(['e', 'E']) {
        Some(i) => (&text[..i], text[i + 1..].parse::<i64>().ok()?),
        None => (text, 0),
    };
    let (negative, mantissa) = match mantissa.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, mantissa.strip_prefix('+').unwrap_or(mantissa)),
    };
    let (whole, fraction) = mantissa.split_once('.').unwrap_or((mantissa, ""));
    if whole.is_empty() && fraction.is_empty()
        || !whole
            .bytes()
            .chain(fraction.bytes())
            .all(|b| b.is_ascii_digit())
    {
        return None;
    }
    let mut digits: i128 = format!("{whole}{fraction}").parse().ok()?;
    let mut scale = fraction.len() as i64 - exponent;
    if scale < 0 {
        digits = digits.checked_mul(10i128.checked_pow(u32::try_from(-scale).ok()?)?)?;
        scale = 0;
    }
    Some((
        if negative { -digits } else { digits },
        u32::try_from(scale).ok()?,
    ))
}

/// `json.dumps(value, separators=(',', ':'))` of what `json.loads` gives:
/// keys in their order, every character past ASCII escaped.
pub fn json_dumps(value: &PyVal) -> String {
    let mut out = String::new();
    json_write(value, &mut out);
    out
}

fn json_write(value: &PyVal, out: &mut String) {
    match value {
        PyVal::None | PyVal::Na => out.push_str("null"),
        PyVal::Bool(true) => out.push_str("true"),
        PyVal::Bool(false) => out.push_str("false"),
        PyVal::Int(i) => out.push_str(&i.to_string()),
        PyVal::BigInt(digits) => out.push_str(digits),
        PyVal::Float(f) if f.is_nan() => out.push_str("NaN"),
        PyVal::Float(f) if f.is_infinite() => {
            out.push_str(if *f > 0.0 { "Infinity" } else { "-Infinity" });
        }
        PyVal::Float(f) => out.push_str(&sdm_schema::py::float_repr(*f)),
        PyVal::Str(s) => json_string(s, out),
        PyVal::List(items) | PyVal::Tuple(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                json_write(item, out);
            }
            out.push(']');
        }
        PyVal::Dict(pairs) => {
            out.push('{');
            for (i, (key, item)) in pairs.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                match key {
                    PyVal::Str(k) => json_string(k, out),
                    other => json_string(&other.str_(), out),
                }
                out.push(':');
                json_write(item, out);
            }
            out.push('}');
        }
        other => json_string(&other.str_(), out),
    }
}

/// `json`'s `ensure_ascii` string: `\"`, `\\`, the short escapes, and
/// `\u` for every other character outside space to `~`, a surrogate pair
/// past the BMP.
fn json_string(text: &str, out: &mut String) {
    out.push('"');
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            ' '..='~' => out.push(c),
            _ => {
                let mut units = [0u16; 2];
                for unit in c.encode_utf16(&mut units) {
                    out.push_str(&format!("\\u{unit:04x}"));
                }
            }
        }
    }
    out.push('"');
}

fn nullable_ints(cells: &[PyVal]) -> Result<Series, Raised> {
    let values = cells
        .iter()
        .map(|v| match v {
            PyVal::None | PyVal::Na => Ok(None),
            PyVal::Int(i) => i64::try_from(*i).map(Some).map_err(|_| {
                Raised::new("OverflowError", "Python int too large to convert to C long")
            }),
            PyVal::Bool(b) => Ok(Some(i64::from(*b))),
            PyVal::Decimal(t) => Ok(Some(
                t.split('.')
                    .next()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0),
            )),
            // `Decimal('NaN')` casts to a missing `Int64`.
            PyVal::Float(f) if f.is_nan() => Ok(None),
            PyVal::Float(f) if f.fract() == 0.0 => Ok(Some(*f as i64)),
            other => Err(Raised::new(
                "TypeError",
                format!("cannot safely cast non-equivalent {other:?} to int64"),
            )),
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Series::new(
        Dtype::Extension("Int64".into()),
        Arc::new(Int64Array::from(values)),
    ))
}

fn nullable_floats(cells: &[PyVal]) -> Result<Series, Raised> {
    let values = cells
        .iter()
        .map(|v| match v {
            PyVal::None | PyVal::Na => Ok(None),
            PyVal::Float(f) if f.is_nan() => Ok(None),
            PyVal::Float(f) => Ok(Some(*f)),
            PyVal::Int(i) => Ok(Some(*i as f64)),
            PyVal::Decimal(t) => t
                .parse()
                .map(Some)
                .map_err(|_| Raised::value_error(format!("could not convert {t} to float"))),
            other => Err(Raised::new(
                "TypeError",
                format!("cannot convert {other:?} to Float64"),
            )),
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Series::new(
        Dtype::Extension("Float64".into()),
        Arc::new(Float64Array::from(values)),
    ))
}

fn nullable_bools(cells: &[PyVal]) -> Result<Series, Raised> {
    let values = cells
        .iter()
        .map(|v| match v {
            PyVal::None | PyVal::Na => Ok(None),
            PyVal::Bool(b) => Ok(Some(*b)),
            other => Err(Raised::new(
                "TypeError",
                format!("Need to pass bool-like values, not {other:?}"),
            )),
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Series::new(
        Dtype::Extension("boolean".into()),
        Arc::new(BooleanArray::from(values)),
    ))
}

fn nullable_strings(cells: &[PyVal]) -> Result<Series, Raised> {
    let values: Vec<Option<String>> = cells
        .iter()
        .map(|v| match v {
            PyVal::None | PyVal::Na => None,
            PyVal::Float(f) if f.is_nan() => None,
            PyVal::Str(s) => Some(s.clone()),
            // `stringify`: a `dict` or `list` as its JSON.
            PyVal::Dict(_) | PyVal::List(_) => Some(json_dumps(v)),
            other => Some(other.str_()),
        })
        .collect();
    Ok(Series::new(
        Dtype::Extension("string".into()),
        Arc::new(StringArray::from(values)),
    ))
}

/// `Table._sanitize(df, dtype_dict, stype_dict)` with the sampler's types.
pub fn sanitize(
    sampler: &Sampler,
    host: &Host,
    table: &str,
    frame: Frame,
    warnings: &mut Vec<Warning>,
) -> Result<Frame, Raised> {
    let info = sampler.table(table)?;
    let dtypes: Vec<(String, Option<PqlDtype>)> = info
        .columns
        .iter()
        .map(|c| (c.name.clone(), c.dtype.as_deref().and_then(dtype_of_name)))
        .collect();
    let stypes = sampler.stypes(table)?.clone();
    sanitize_with(host, &dtypes, &stypes, frame, warnings)
}

/// `Table._sanitize`: each time column read as `datetime64[ns]`.
pub fn sanitize_with(
    host: &Host,
    dtypes: &[(String, Option<PqlDtype>)],
    stypes: &[(String, String)],
    mut frame: Frame,
    warnings: &mut Vec<Warning>,
) -> Result<Frame, Raised> {
    for i in 0..frame.names.len() {
        let name = frame.names[i].clone();
        let dtype = dtypes
            .iter()
            .find(|(n, _)| *n == name)
            .and_then(|(_, d)| *d);
        let stype = stypes
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, s)| s.as_str());
        if dtype == Some(PqlDtype::Time) || stype == Some("timestamp") {
            frame.columns[i] = to_datetime_series(&frame.columns[i], &name, host, warnings)?;
        } else if let Some(kind) = list_kind(dtype, stype) {
            frame.columns[i] = crate::sequences::to_list_with_policy(
                &frame.columns[i],
                kind,
                host.numpy_text_prefix,
                warnings,
            )?;
        }
    }
    Ok(frame)
}

/// What `Table._sanitize` reads a column's text as, if anything: its list
/// dtype's items, or a `sequence` column's as `floatlist`.
pub fn list_kind(dtype: Option<PqlDtype>, stype: Option<&str>) -> Option<ListKind> {
    match dtype {
        Some(PqlDtype::IntList) => Some(ListKind::Int),
        Some(PqlDtype::FloatList) => Some(ListKind::Float),
        Some(PqlDtype::StringList) => Some(ListKind::Text),
        _ => (stype == Some("sequence")).then_some(ListKind::Float),
    }
}

/// `to_datetime(ser, column_name)`: `datetime64[ns]`, an unreadable value
/// `NaT` with a warning.
pub fn to_datetime_series(
    series: &Series,
    column: &str,
    host: &Host,
    warnings: &mut Vec<Warning>,
) -> Result<Series, Raised> {
    if series.dtype.is_datetime()
        && matches!(
            series.array.data_type(),
            DataType::Timestamp(TimeUnit::Nanosecond, None)
        )
    {
        return Ok(series.clone());
    }
    let values: Vec<Option<i64>> = match series.array.data_type() {
        DataType::Timestamp(..) => times_of(series)?,
        _ => {
            let source = &series.array;
            let converted = crate::graph_store::to_datetime(source, host)
                .map_err(|e| Raised::new(&e.code, e.message))?;
            let coerced: Vec<usize> = (0..source.len())
                .filter(|&i| converted[i].is_none() && source.is_valid(i))
                .collect();
            if !coerced.is_empty() {
                let shown = coerced_examples(source, &coerced);
                warnings.push(Warning::user(format!(
                    "Could not parse {} of {} value(s) of time column '{column}' (for example \
                     {shown}); they were dropped as 'NaT'",
                    sdm_schema::py::thousands(coerced.len() as i64),
                    sdm_schema::py::thousands(source.len() as i64),
                )));
            }
            converted
        }
    };
    Ok(time_series(&values))
}

fn coerced_examples(source: &ArrayRef, coerced: &[usize]) -> String {
    let mut seen: Vec<String> = Vec::new();
    for &i in coerced {
        let value = time_of_day(source, i)
            .unwrap_or_else(|| sdm_schema::engine::result::value_at(source, i));
        let shown = match &value {
            PyVal::Str(s) => sdm_schema::py::str_repr(s),
            other => other.repr(),
        };
        if !seen.contains(&shown) {
            seen.push(shown);
        }
        if seen.len() == 3 {
            break;
        }
    }
    format!("[{}]", seen.join(", "))
}

/// A `time` array's value as the `datetime.time` pandas holds; `None` for
/// any other array.
fn time_of_day(array: &ArrayRef, i: usize) -> Option<PyVal> {
    matches!(array.data_type(), DataType::Time32(_) | DataType::Time64(_))
        .then(|| sdm_schema::engine::result::value_at(array, i))
}

// --- Keys as the dialects take them ---------------------------------------

/// `pa.array(series)`: the series' values with their Arrow type, as the
/// dialects read a key or time list.
pub fn arrow_list(series: &Series) -> Result<ArrowArray, Raised> {
    let array = series.array.as_ref();
    let n = array.len();
    let unit_of = |u: &TimeUnit| match u {
        TimeUnit::Second => ArrowUnit::Second,
        TimeUnit::Millisecond => ArrowUnit::Milli,
        TimeUnit::Microsecond => ArrowUnit::Micro,
        TimeUnit::Nanosecond => ArrowUnit::Nano,
    };
    let (data_type, values): (ArrowType, Vec<Py>) = match array.data_type() {
        DataType::Null => (ArrowType::Null, vec![Py::None; n]),
        DataType::Boolean => {
            let values = array.as_boolean();
            (
                ArrowType::Bool,
                (0..n)
                    .map(|i| {
                        if array.is_null(i) {
                            Py::None
                        } else {
                            Py::Bool(values.value(i))
                        }
                    })
                    .collect(),
            )
        }
        t @ (DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32
        | DataType::UInt64) => {
            let (bits, signed) = match t {
                DataType::Int8 => (8, true),
                DataType::Int16 => (16, true),
                DataType::Int32 => (32, true),
                DataType::Int64 => (64, true),
                DataType::UInt8 => (8, false),
                DataType::UInt16 => (16, false),
                DataType::UInt32 => (32, false),
                _ => (64, false),
            };
            let values: Vec<Py> = if matches!(t, DataType::UInt64) {
                let v = array.as_primitive::<UInt64Type>();
                (0..n)
                    .map(|i| {
                        if array.is_null(i) {
                            Py::None
                        } else {
                            Py::Int(Int::from_i128(i128::from(v.value(i))))
                        }
                    })
                    .collect()
            } else {
                ints(array)
                    .unwrap_or_default()
                    .into_iter()
                    .map(|v| v.map_or(Py::None, |x| Py::Int(Int::from(x))))
                    .collect()
            };
            (ArrowType::Int { bits, signed }, values)
        }
        DataType::Float32 | DataType::Float64 => (
            if matches!(array.data_type(), DataType::Float32) {
                ArrowType::Float32
            } else {
                ArrowType::Float64
            },
            crate::frame::floats(array)
                .unwrap_or_default()
                .into_iter()
                .map(|v| v.map_or(Py::None, Py::Float))
                .collect(),
        ),
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => (
            if matches!(array.data_type(), DataType::LargeUtf8) {
                ArrowType::LargeString
            } else {
                ArrowType::String
            },
            texts(array)
                .unwrap_or_default()
                .into_iter()
                .map(|v| v.map_or(Py::None, Py::Str))
                .collect(),
        ),
        DataType::Timestamp(unit, zone) => {
            let storage = match unit {
                TimeUnit::Nanosecond => {
                    let v = array.as_primitive::<TimestampNanosecondType>();
                    (0..n)
                        .map(|i| (!array.is_null(i)).then(|| v.value(i)))
                        .collect::<Vec<_>>()
                }
                _ => {
                    let cast =
                        arrow_array::cast::as_primitive_array::<arrow_array::types::Int64Type>;
                    let _ = cast;
                    let data = array.to_data();
                    let values: &[i64] = data.buffer::<i64>(0);
                    (0..n)
                        .map(|i| (!array.is_null(i)).then(|| values[i]))
                        .collect()
                }
            };
            (
                ArrowType::Timestamp(unit_of(unit), zone.as_ref().map(|z| z.to_string())),
                storage
                    .into_iter()
                    .map(|v| v.map_or(Py::None, |x| Py::Int(Int::from(x))))
                    .collect(),
            )
        }
        DataType::Decimal128(precision, scale) => {
            let v = array.as_primitive::<Decimal128Type>();
            (
                ArrowType::Decimal {
                    wide: false,
                    precision: u32::from(*precision),
                    scale: i32::from(*scale),
                },
                (0..n)
                    .map(|i| {
                        if array.is_null(i) {
                            Py::None
                        } else {
                            Py::Str(sdm_schema::frame::decimal_text(v.value(i), *scale))
                        }
                    })
                    .collect(),
            )
        }
        other => {
            return Err(Raised::new(
                "TypeError",
                format!("keys of Arrow type {other} are not supported by the core"),
            ));
        }
    };
    Ok(ArrowArray::new(data_type, values, None)?)
}

/// What iterating a series yields, as the dialects take Python values.
pub fn series_values(series: &Series) -> Result<Vec<Py>, Raised> {
    let array = series.array.as_ref();
    if series.dtype.is_datetime() || matches!(array.data_type(), DataType::Timestamp(..)) {
        return Ok(times_of(series)?
            .into_iter()
            .map(|t| match t {
                None => Py::NaT,
                Some(ns) => Py::Timestamp(Stamp {
                    civil: Civil::from_epoch_nanos(i128::from(ns)),
                    offset: None,
                }),
            })
            .collect());
    }
    let listed = arrow_list(series)?;
    let missing = match &series.dtype {
        Dtype::Arrow | Dtype::Extension(_) => Py::Na,
        _ => Py::None,
    };
    Ok(listed
        .values
        .into_iter()
        .map(|v| if v.is_none() { missing.clone() } else { v })
        .collect())
}

fn entity_param(id: &EntityId) -> sdm_protocol::Param {
    use sdm_protocol::Param;
    match id {
        EntityId::None => Param::Null,
        EntityId::Bool(b) => Param::Bool(*b),
        EntityId::Int(i) => Param::Int(*i),
        EntityId::Float(f) => Param::Float(*f),
        EntityId::Text(s) => Param::Text(s.clone()),
    }
}

fn entity_py(id: &EntityId) -> Py {
    match id {
        EntityId::None => Py::None,
        EntityId::Bool(b) => Py::Bool(*b),
        EntityId::Int(i) => Py::Int(Int::from(*i)),
        EntityId::Float(f) => Py::Float(*f),
        EntityId::Text(s) => Py::Str(s.clone()),
    }
}

/// A listing's rows, as the capture encodes `_listed`: dicts of text keys.
fn listing_rows(value: &Py) -> Result<Vec<sdm_dialects::helpers::Listing>, Raised> {
    let Py::List(rows) = value else {
        return Err(Raised::internal("a listing that is not a list"));
    };
    rows.iter()
        .map(|row| match row {
            Py::Dict(fields) => Ok(fields.clone()),
            _ => Err(Raised::internal("a listing row that is not a dict")),
        })
        .collect()
}

/// A value of a result as the dialects take Python values.
fn pyval_py(value: &PyVal) -> Py {
    match value {
        PyVal::None => Py::None,
        PyVal::Bool(b) => Py::Bool(*b),
        PyVal::Int(i) => Py::Int(Int::from_i128(*i)),
        PyVal::BigInt(text) => Int::parse(text).map_or(Py::None, Py::Int),
        PyVal::Float(f) => Py::Float(*f),
        PyVal::Str(s) => Py::Str(s.clone()),
        PyVal::Decimal(text) => Py::Decimal(text.clone()),
        PyVal::Na => Py::Na,
        PyVal::NaT => Py::NaT,
        other => Py::Str(other.str_()),
    }
}

// --- The traversal -----------------------------------------------------------

/// `_request_order(table, batch)`: the stable order that puts rows read by
/// key back in request order, refusing a position read twice.
pub fn request_order(table: &str, batch: &[i64]) -> Result<Vec<usize>, Raised> {
    let mut perm: Vec<usize> = (0..batch.len()).collect();
    perm.sort_by_key(|&i| batch[i]);
    for pair in perm.windows(2) {
        if batch[pair[0]] == batch[pair[1]] {
            return Err(Raised::new(
                "RuntimeError",
                format!(
                    "Internal error: table {} returned a duplicate row for request position {}, \
                     which is read once. The order of the two would be arbitrary, so the \
                     request is not built. Please report this.",
                    sdm_schema::py::str_repr(table),
                    batch[pair[0]]
                ),
            ));
        }
    }
    Ok(perm)
}

/// `_refuse_an_id_scored_twice(table, index, batch, matches, scored)`.
pub fn refuse_an_id_scored_twice(
    sampler: &Sampler,
    table: &str,
    index: &Series,
    batch: &[i64],
    matches: &[i64],
    scored: Option<&[i64]>,
) -> Result<(), Raised> {
    let Some(scored) = scored.filter(|s| !s.is_empty()) else {
        return Ok(());
    };
    let mut many = vec![false; index.len()];
    for (&b, &m) in batch.iter().zip(matches) {
        let position = usize::try_from(b)
            .ok()
            .filter(|&p| p < many.len())
            .ok_or_else(|| Raised::new("IndexError", "index out of bounds"))?;
        many[position] = m > 1;
    }
    let mut repeated = Vec::new();
    for &position in scored {
        let p = usize::try_from(position)
            .ok()
            .filter(|&p| p < many.len())
            .ok_or_else(|| Raised::new("IndexError", "index out of bounds"))?;
        if many[p] {
            repeated.push(p);
        }
    }
    if repeated.is_empty() {
        return Ok(());
    }
    let key = sampler.primary_key(table)?;
    let keys = keys_of(&index.take(&repeated)?)?;
    let mut ids: Vec<Key> = Vec::new();
    for k in keys {
        if !ids.iter().any(|i| i.id() == k.id()) {
            ids.push(k);
        }
    }
    let shown = ids
        .iter()
        .take(5)
        .map(Key::repr)
        .collect::<Vec<_>>()
        .join(", ");
    let more = if ids.len() > 5 {
        format!(
            " and {} more",
            sdm_schema::py::thousands(ids.len() as i64 - 5)
        )
    } else {
        String::new()
    };
    let which = if ids.len() > 1 {
        "each of these values"
    } else {
        "this value"
    };
    Err(Raised::value_error(format!(
        "Cannot score {shown}{more}: more than one row of table '{table}' holds {which} of \
         '{key}', so which row is meant would be a guess. The source does not enforce the key; \
         keep one row per value, or key the table on columns that identify a row."
    )))
}

/// An integer column of a result, as `to_numpy()` reads it.
fn int_column(frame: &Frame, name: &str) -> Result<Vec<i64>, Raised> {
    let series = frame.column(name)?;
    let array = series.array.as_ref();
    if let Some(values) = ints(array) {
        return values
            .into_iter()
            .map(|v| v.ok_or_else(|| Raised::value_error("cannot convert NA to integer")))
            .collect();
    }
    if let DataType::Decimal128(_, scale) = array.data_type() {
        let values = array.as_primitive::<Decimal128Type>();
        let divisor = 10i128.pow(u32::try_from(*scale).unwrap_or(0));
        return (0..array.len())
            .map(|i| {
                if array.is_null(i) {
                    Err(Raised::value_error("cannot convert NA to integer"))
                } else {
                    Ok((values.value(i) / divisor) as i64)
                }
            })
            .collect();
    }
    if let Some(values) = crate::frame::floats(array) {
        return Ok(values
            .into_iter()
            .map(|v| v.unwrap_or(0.0) as i64)
            .collect());
    }
    if matches!(array.data_type(), DataType::Null) {
        // An `object` column of no rows, or of nulls only.
        if array.is_empty() {
            return Ok(Vec::new());
        }
        return Err(Raised::value_error("cannot convert NA to integer"));
    }
    Err(Raised::new(
        "TypeError",
        format!(
            "a batch column of Arrow type {} is not supported",
            array.data_type()
        ),
    ))
}

/// Keeps, of each `__KUMO_BATCH__`, its first row, and counts each's rows
/// (`table.group_by('__KUMO_BATCH__')`, the min and count of a row number).
fn first_of_each_batch(frame: &Frame) -> Result<(Frame, Vec<i64>, Vec<i64>), Raised> {
    let batch = int_column(frame, "__KUMO_BATCH__")?;
    let mut order: Vec<i64> = Vec::new();
    let mut first: HashMap<i64, usize> = HashMap::new();
    let mut counts: HashMap<i64, i64> = HashMap::new();
    for (i, b) in batch.iter().enumerate() {
        if !first.contains_key(b) {
            first.insert(*b, i);
            order.push(*b);
        }
        *counts.entry(*b).or_insert(0) += 1;
    }
    let rows: Vec<usize> = order.iter().map(|b| first[b]).collect();
    let kept = frame.take(&rows)?;
    let kept_batch: Vec<i64> = order.clone();
    let matches: Vec<i64> = order.iter().map(|b| counts[b]).collect();
    Ok((kept, kept_batch, matches))
}

/// One backend's `_by_pkey(table, index, columns, scored)`: one row for each
/// id of `index`, and the position each came for.
pub async fn by_pkey(
    sampler: &Sampler,
    db: &Db,
    table: &str,
    index: &Series,
    columns: &[String],
    scored: Option<&[i64]>,
) -> Result<(Frame, Vec<i64>), Raised> {
    let sql = sql_of(sampler)?;
    let key = sampler.primary_key(table)?.clone();
    let key_ref = sql.column_ref(table, &key)?;
    let source = sql.source_name(table)?.clone();
    match sql.kind {
        Backend::Duckdb => {
            let projections = sql.projections(table, columns)?;
            let terms = sql.tie_break_terms(table, columns, &[])?;
            let order_by = if terms.is_empty() {
                key_ref.clone()
            } else {
                terms.join(", ")
            };
            let rows = db
                .run(duckdb::rows_by_key(
                    &source,
                    &key_ref,
                    &arrow_list(index)?,
                    &projections,
                    &order_by,
                )?)
                .await?;
            let frame = arrow_result_frame(&rows)?;
            let batch = int_column(&frame, "__kumo_batch__")?;
            let matches = int_column(&frame, "__kumo_matches__")?;
            refuse_an_id_scored_twice(sampler, table, index, &batch, &matches, scored)?;
            let frame = frame.without("__kumo_batch__").without("__kumo_matches__");
            let frame = sql.sanitized(sampler, table, frame)?;
            Ok((frame, batch))
        }
        Backend::Sqlite => {
            let projections = sql.projections(table, columns)?;
            let unique = sql
                .source_table(table)?
                .source_column(&key)
                .is_some_and(|c| c.is_unique_key);
            let counted = scored.is_some_and(|s| !s.is_empty()) && !unique;
            let rows = db
                .run(sqlite::rows_by_key(
                    &source,
                    &key_ref,
                    &arrow_list(index)?,
                    &projections,
                    unique,
                    &sql.tie_break_terms(table, columns, &[])?,
                    counted,
                )?)
                .await?;
            let frame = arrow_result_frame(&rows)?;
            let batch = int_column(&frame, "__kumo_batch__")?;
            let mut frame = frame.without("__kumo_batch__");
            if counted {
                let matches = int_column(&frame, "__kumo_matches__")?;
                refuse_an_id_scored_twice(sampler, table, index, &batch, &matches, scored)?;
                frame = frame.without("__kumo_matches__");
            }
            let frame = sql.sanitized(sampler, table, frame)?;
            Ok((frame, batch))
        }
        Backend::Postgres => {
            if index.is_empty() {
                return Ok((Frame::default(), Vec::new()));
            }
            let dtype = sql
                .dtype_of(table, &key)?
                .ok_or_else(|| Raised::internal("a key without a dtype"))?;
            let postgres_type = sql.postgres_type(table, &key)?;
            let projections = sql.projections(table, columns)?;
            let terms = sql.tie_break_terms(table, columns, &[])?;
            let order_by = if terms.is_empty() {
                key_ref.clone()
            } else {
                terms.join(", ")
            };
            let rows = series_values(index)?
                .iter()
                .map(|v| postgres::coerce_id(v, dtype, &postgres_type))
                .collect::<Result<Vec<_>, _>>()?;
            let frame = sql
                .postgres_flatten(db, &rows, |chunk| {
                    postgres::rows_by_key(
                        &source,
                        &projections,
                        &key_ref,
                        &postgres_type,
                        &Py::List(chunk.to_vec()),
                        &order_by,
                    )
                })
                .await?;
            let mut frame = frame;
            if frame.position("__KUMO_MATCHES__").is_some() {
                let matches = int_column(&frame, "__KUMO_MATCHES__")?;
                let batch = int_column(&frame, "__KUMO_BATCH__")?;
                frame = frame.without("__KUMO_MATCHES__");
                refuse_an_id_scored_twice(sampler, table, index, &batch, &matches, scored)?;
            }
            sql.postgres_output(table, frame)
        }
        Backend::Snowflake | Backend::Databricks => {
            if index.is_empty() {
                return Ok((Frame::default(), Vec::new()));
            }
            let dtype = sql
                .dtype_of(table, &key)?
                .ok_or_else(|| Raised::internal("a key without a dtype"))?;
            let projections = sql.projections(table, columns)?;
            let order_by = sql.tie_break_terms(table, columns, &[])?;
            let rows = if sql.kind == Backend::Snowflake {
                let ids = Py::List(series_values(index)?);
                db.run(snow::rows_by_key(
                    &source,
                    &projections,
                    &key_ref,
                    dtype,
                    &ids,
                    &order_by,
                )?)
                .await?
            } else {
                let coerced = series_values(index)?
                    .iter()
                    .map(|v| databricks::coerce_id(v, dtype))
                    .collect::<Result<Vec<_>, _>>()?;
                databricks_flatten(db, &coerced, |chunk| {
                    databricks::rows_by_key(
                        &source,
                        &projections,
                        &key_ref,
                        dtype,
                        &Py::List(chunk.to_vec()),
                        &order_by,
                    )
                })
                .await?
            };
            let frame = arrow_result_frame(&rows)?;
            let (kept, batch, matches) = first_of_each_batch(&frame)?;
            refuse_an_id_scored_twice(sampler, table, index, &batch, &matches, scored)?;
            let frame = kept.without("__KUMO_BATCH__");
            let frame = sql.sanitized(sampler, table, frame)?;
            Ok((frame, batch))
        }
        Backend::Local => Err(Raised::internal("a local sampler has no SQL")),
    }
}

impl SqlBackend {
    /// PostgreSQL's `_sanitize_output`: the batch popped, the helpers
    /// dropped, the rest sanitized.
    fn postgres_output(&self, table: &str, frame: Frame) -> Result<(Frame, Vec<i64>), Raised> {
        let batch = int_column(&frame, "__KUMO_BATCH__")?;
        let frame = frame.without("__KUMO_BATCH__").without("__KUMO_ROW__");
        let frame = self.sanitize_postgres(table, frame)?;
        Ok((frame, batch))
    }
}

/// Databricks' `_flatten(statement, rows)`: the statement for each chunk of
/// `rows`, the batches shifted into one space.
async fn databricks_flatten(
    db: &Db,
    rows: &[Py],
    statement: impl Fn(&[Py]) -> Result<Statement, sdm_dialects::Error>,
) -> Result<Rows, Raised> {
    let chunks = databricks::chunk_rows(rows, 900_000)?;
    if chunks.len() == 1 {
        return db.run(statement(&chunks[0])?).await;
    }
    let mut out: Option<Rows> = None;
    let mut offset = 0i64;
    for chunk in &chunks {
        let result = db.run(statement(chunk)?).await?;
        let mut batches = Vec::new();
        for batch in &result.batches {
            let schema = batch.schema();
            let i = schema
                .index_of("__KUMO_BATCH__")
                .map_err(|_| Raised::key_error("__KUMO_BATCH__"))?;
            let shifted: Vec<i64> = ints(batch.column(i).as_ref())
                .unwrap_or_default()
                .into_iter()
                .map(|v| v.unwrap_or(0) + offset)
                .collect();
            let mut columns = batch.columns().to_vec();
            columns[i] = Arc::new(Int64Array::from(shifted));
            let mut fields: Vec<arrow_schema::Field> =
                schema.fields().iter().map(|f| f.as_ref().clone()).collect();
            fields[i] = arrow_schema::Field::new("__KUMO_BATCH__", DataType::Int64, true);
            let schema = Arc::new(arrow_schema::Schema::new(fields));
            batches.push(
                RecordBatch::try_new(schema, columns)
                    .map_err(|e| Raised::internal(e.to_string()))?,
            );
        }
        offset += chunk.len() as i64;
        match &mut out {
            None => {
                let schema = batches
                    .first()
                    .map(RecordBatch::schema)
                    .unwrap_or_else(|| result.schema.clone());
                out = Some(Rows { schema, batches });
            }
            Some(rows) => rows.batches.extend(batches),
        }
    }
    out.ok_or_else(|| Raised::internal("no chunk was read"))
}

/// `strftime(spec)` of naive nanosecond times, for the formats the samplers
/// write bounds in.
fn strftime(ns: i64, spec: &str) -> String {
    let c = Civil::from_epoch_nanos(i128::from(ns));
    let mut out = String::new();
    let mut chars = spec.chars();
    while let Some(ch) = chars.next() {
        if ch != '%' {
            out.push(ch);
            continue;
        }
        match chars.next() {
            Some('Y') => out.push_str(&format!("{:04}", c.year)),
            Some('m') => out.push_str(&format!("{:02}", c.month)),
            Some('d') => out.push_str(&format!("{:02}", c.day)),
            Some('H') => out.push_str(&format!("{:02}", c.hour)),
            Some('M') => out.push_str(&format!("{:02}", c.minute)),
            Some('S') => out.push_str(&format!("{:02}", c.second)),
            Some('f') => out.push_str(&format!("{:06}", c.nanos / 1000)),
            Some(other) => {
                out.push('%');
                out.push(other);
            }
            None => out.push('%'),
        }
    }
    out
}

/// SQLite's `_TimeSpelling.spell(times)`.
fn spell(spelling: &TimeSpelling, times: &[Option<i64>]) -> Vec<Option<String>> {
    times
        .iter()
        .map(|t| {
            t.map(|ns| {
                let mut text = strftime(ns, &format!("%Y-%m-%d{}%H:%M:%S", spelling.separator));
                if spelling.digits > 0 {
                    let fraction = format!("{:09}", ns.rem_euclid(1_000_000_000));
                    let fraction = format!("{fraction:0<width$}", width = spelling.digits as usize);
                    text.push('.');
                    text.push_str(&fraction[..spelling.digits as usize]);
                }
                text.push_str(&spelling.suffix);
                text
            })
        })
        .collect()
}

impl SqlBackend {
    /// SQLite's `_time_spelling(table)`, probing the table once.
    async fn time_spelling(
        &self,
        sampler: &Sampler,
        db: &Db,
        table: &str,
    ) -> Result<TimeSpelling, Raised> {
        let known = self
            .time_spellings
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(table)
            .cloned();
        if let Some(spelling) = known {
            return Ok(spelling);
        }
        let spelling = self.probe_time_spelling(sampler, db, table).await?;
        self.time_spellings
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(table.to_string(), spelling.clone());
        Ok(spelling)
    }

    async fn probe_time_spelling(
        &self,
        sampler: &Sampler,
        db: &Db,
        table: &str,
    ) -> Result<TimeSpelling, Raised> {
        let Some(column) = sampler.time_column(table) else {
            return Ok(TimeSpelling::default());
        };
        let column_ref = self.column_ref(table, column)?;
        let statement = sqlite::first_time(self.source_name(table)?, &column_ref);
        let found = match db.run(statement).await {
            Ok(rows) => row_values(&rows, false),
            Err(_) => return Ok(TimeSpelling::default()),
        };
        let found: Vec<Py> = found
            .iter()
            .map(|row| Py::Tuple(row.iter().map(pyval_py).collect()))
            .collect();
        let found = Some(found);
        let helper =
            sdm_dialects::helpers::sqlite_probe_time_spelling(Some(column), found.as_deref());
        Ok(TimeSpelling {
            separator: helper.separator.clone(),
            digits: helper.digits,
            suffix: helper.suffix.clone(),
        })
    }
}

/// The SQL backend of a sampler that has one.
pub fn sql_of(sampler: &Sampler) -> Result<&SqlBackend, Raised> {
    match &sampler.through {
        Through::Sql(sql) => Ok(sql),
        Through::Local(_) => Err(Raised::internal("a local sampler has no SQL")),
    }
}

/// The window bounds, as each backend writes `anchor + offset` for its
/// statement.
fn offset_times(anchor: &[Option<i64>], offset: &DateOffset) -> Result<Vec<Option<i64>>, Raised> {
    let applied = offset.apply_array(anchor, 1_000_000_000)?;
    Ok(applied)
}

/// One backend's `_by_fkey(table, foreign_key, index, num_neighbors,
/// anchor_time, columns)`: up to `num_neighbors` rows of `table` for each
/// id, and the position each came for.
#[allow(clippy::too_many_arguments)]
pub async fn by_fkey(
    sampler: &mut Sampler,
    db: &Db,
    table: &str,
    foreign_key: &str,
    index: &Series,
    num_neighbors: i64,
    anchor_time: Option<&[Option<i64>]>,
    columns: &[String],
) -> Result<(Frame, Vec<i64>), Raised> {
    let kind = sql_of(sampler)?.kind;
    let time_column = sampler.time_column(table).cloned();
    // Snowflake and Databricks bound the window by the table's time range,
    // which may be read first.
    let mut range = None;
    if matches!(kind, Backend::Snowflake | Backend::Databricks)
        && time_column.is_some()
        && anchor_time.is_some()
    {
        let names = [table.to_string()];
        let min = sampler.get_min_time(db, Some(&names)).await?;
        let max = sampler.get_max_time(db, Some(&names)).await?;
        range = Some((min, max));
    }
    let sampler = &*sampler;
    let sql = sql_of(sampler)?;
    let source = sql.source_name(table)?.clone();
    let key_ref = sql.column_ref(table, foreign_key)?;
    match kind {
        Backend::Duckdb | Backend::Sqlite => {
            let projections = sql.projections(table, columns)?;
            let mut end_times = None;
            if let (Some(column), Some(anchor)) = (&time_column, anchor_time) {
                let _ = column;
                end_times = Some(if kind == Backend::Sqlite {
                    let spelling = sql.time_spelling(sampler, db, table).await?;
                    let spelled = spell(&spelling, anchor);
                    arrow_list(&Series::new(
                        Dtype::Object,
                        Arc::new(StringArray::from(spelled)),
                    ))?
                } else {
                    arrow_list(&time_series(anchor))?
                });
            }
            let time_ref = match &time_column {
                None => None,
                Some(column) if kind == Backend::Duckdb => {
                    Some(sql.duckdb_time_ref(table, column)?)
                }
                Some(column) => Some(sql.column_ref(table, column)?),
            };
            let order_by = sql.neighbor_order_by(
                sampler,
                table,
                time_ref.as_deref(),
                &key_ref,
                Some(columns),
                true,
            )?;
            let ids = arrow_list(index)?;
            let statement = if kind == Backend::Duckdb {
                duckdb::neighbors(
                    &source,
                    &key_ref,
                    &ids,
                    &projections,
                    num_neighbors,
                    &order_by,
                    time_ref.as_deref(),
                    end_times.as_ref(),
                )?
            } else {
                sqlite::neighbors(
                    &source,
                    &key_ref,
                    &ids,
                    &projections,
                    num_neighbors,
                    &order_by,
                    time_ref.as_deref(),
                    end_times.as_ref(),
                )?
            };
            let rows = db.run(statement).await?;
            let frame = arrow_result_frame(&rows)?;
            let batch = int_column(&frame, "__kumo_batch__")?;
            let frame = frame.without("__kumo_batch__");
            let frame = sql.sanitized(sampler, table, frame)?;
            Ok((frame, batch))
        }
        Backend::Postgres => {
            let dtype = sql
                .dtype_of(table, foreign_key)?
                .ok_or_else(|| Raised::internal("a key without a dtype"))?;
            let postgres_type = sql.postgres_type(table, foreign_key)?;
            let ids = series_values(index)?;
            let mut fields: Vec<String> = Vec::new();
            let rows: Vec<Py> = match (&time_column, anchor_time) {
                (Some(column), Some(anchor)) => {
                    let utc = sql.dtype(table, column) != Some(PqlDtype::String);
                    fields.push("e".into());
                    let ends = series_values(&time_series(anchor))?;
                    ids.iter()
                        .zip(ends)
                        .map(|(id, end)| {
                            Ok(Py::Dict(vec![
                                ("id".into(), postgres::coerce_id(id, dtype, &postgres_type)?),
                                (
                                    "e".into(),
                                    postgres::serialize_time(&end, utc)?.map_or(Py::None, Py::Str),
                                ),
                            ]))
                        })
                        .collect::<Result<_, Raised>>()?
                }
                _ => ids
                    .iter()
                    .map(|id| postgres::coerce_id(id, dtype, &postgres_type))
                    .collect::<Result<_, _>>()?,
            };
            let projections = sql.projections(table, columns)?;
            let time_ref = match &time_column {
                None => None,
                Some(column) => Some(sql.column_ref(table, column)?),
            };
            let order_by = sql.neighbor_order_by(
                sampler,
                table,
                time_ref.as_deref(),
                &key_ref,
                Some(columns),
                true,
            )?;
            let time_type = match &time_column {
                None => "timestamp".to_string(),
                Some(column) => sql.postgres_type(table, column)?,
            };
            let frame = sql
                .postgres_flatten(db, &rows, |chunk| {
                    postgres::neighbors(
                        &source,
                        &projections,
                        &key_ref,
                        &postgres_type,
                        &Py::List(chunk.to_vec()),
                        &fields,
                        time_ref.as_deref(),
                        &time_type,
                        &order_by,
                        num_neighbors,
                    )
                })
                .await?;
            sql.postgres_output(table, frame)
        }
        Backend::Snowflake | Backend::Databricks => {
            let dtype = sql
                .dtype_of(table, foreign_key)?
                .ok_or_else(|| Raised::internal("a key without a dtype"))?;
            let ids = series_values(index)?;
            let mut ends: Option<Vec<Option<String>>> = None;
            let mut starts: Option<Vec<Option<String>>> = None;
            if let (Some(column), Some(anchor), Some((min, max))) =
                (&time_column, anchor_time, range)
            {
                let dst = sampler
                    .foreign_keys(table)?
                    .iter()
                    .find(|(k, _)| k == foreign_key)
                    .map(|(_, d)| d.clone())
                    .ok_or_else(|| Raised::new("IndexError", "list index out of range"))?;
                let num_facts = sql
                    .num_rows
                    .get(table)
                    .copied()
                    .flatten()
                    .ok_or_else(|| Raised::key_error(table))?;
                let num_entities = sql
                    .num_rows
                    .get(&dst)
                    .copied()
                    .flatten()
                    .ok_or_else(|| Raised::key_error(&dst))?;
                if num_entities == 0 {
                    return Err(Raised::new("ZeroDivisionError", "division by zero"));
                }
                let freq = num_facts as f64 / num_entities as f64;
                let span = (i128::from(max) - i128::from(min)) as f64 / 1e9;
                let freq = freq / span.max(1.0);
                if freq == 0.0 {
                    return Err(Raised::new("ZeroDivisionError", "float division by zero"));
                }
                let cap: i64 = 5 * 365 * 24 * 60 * 60;
                let wanted = (5.0 * num_neighbors as f64 / freq).ceil();
                let seconds = if wanted < cap as f64 {
                    wanted as i64
                } else {
                    cap
                };
                let spec = sql.bound_format(table, column);
                ends = Some(
                    anchor
                        .iter()
                        .map(|t| t.map(|ns| strftime(ns, &spec)))
                        .collect(),
                );
                starts = Some(
                    anchor
                        .iter()
                        .map(|t| t.map(|ns| strftime(ns - seconds * 1_000_000_000, &spec)))
                        .collect(),
                );
            }
            let windowed = ends.is_some() && starts.is_some();
            let mut bounds: Vec<String> = Vec::new();
            if let (Some(e), Some(s)) = (&ends, &starts) {
                bounds.push(e.iter().flatten().max().cloned().unwrap_or_default());
                bounds.push(s.iter().flatten().min().cloned().unwrap_or_default());
            }
            let projections = sql.projections(table, columns)?;
            let time_ref = match &time_column {
                None => None,
                Some(column) => Some(sql.column_ref(table, column)?),
            };
            let order_by = sql.neighbor_order_by(
                sampler,
                table,
                time_ref.as_deref(),
                &key_ref,
                Some(columns),
                true,
            )?;
            let rows = if kind == Backend::Snowflake {
                let tuples: Vec<Py> = match (&ends, &starts) {
                    (Some(e), Some(s)) => ids
                        .iter()
                        .zip(e)
                        .zip(s)
                        .map(|((id, e), s)| {
                            Py::Tuple(vec![
                                id.clone(),
                                e.clone().map_or(Py::None, Py::Str),
                                s.clone().map_or(Py::None, Py::Str),
                            ])
                        })
                        .collect(),
                    _ => ids.iter().map(|id| Py::Tuple(vec![id.clone()])).collect(),
                };
                let zone = time_column.as_ref().and_then(|c| sql.snow_zone(table, c));
                db.run(snow::neighbors(
                    &source,
                    &projections,
                    &key_ref,
                    dtype,
                    &Py::List(tuples),
                    windowed,
                    time_ref.as_deref(),
                    &bounds,
                    &order_by,
                    num_neighbors,
                    zone,
                )?)
                .await?
            } else {
                let rows: Vec<Py> = match (&ends, &starts) {
                    (Some(e), Some(s)) => ids
                        .iter()
                        .zip(e)
                        .zip(s)
                        .map(|((id, e), s)| {
                            Ok(Py::Dict(vec![
                                ("id".into(), databricks::coerce_id(id, dtype)?),
                                ("e".into(), e.clone().map_or(Py::None, Py::Str)),
                                ("s".into(), s.clone().map_or(Py::None, Py::Str)),
                            ]))
                        })
                        .collect::<Result<_, Raised>>()?,
                    _ => ids
                        .iter()
                        .map(|id| databricks::coerce_id(id, dtype))
                        .collect::<Result<_, _>>()?,
                };
                databricks_flatten(db, &rows, |chunk| {
                    databricks::neighbors(
                        &source,
                        &projections,
                        &key_ref,
                        dtype,
                        &Py::List(chunk.to_vec()),
                        windowed,
                        time_ref.as_deref(),
                        &bounds,
                        &order_by,
                        num_neighbors,
                    )
                })
                .await?
            };
            let frame = arrow_result_frame(&rows)?;
            let batch = int_column(&frame, "__KUMO_BATCH__")?;
            let frame = frame.without("__KUMO_BATCH__");
            let frame = sql.sanitized(sampler, table, frame)?;
            Ok((frame, batch))
        }
        Backend::Local => Err(Raised::internal("a local sampler has no SQL")),
    }
}

impl SqlBackend {
    /// Snowflake's `_zone(table, column)`.
    fn snow_zone(&self, table: &str, column: &str) -> Option<&'static str> {
        let source = self.source_table(table).ok()?;
        if source.source_column(column).is_none() {
            if source.zoned_expr_columns.iter().any(|c| c == column) {
                return Some("TZ");
            }
            return None;
        }
        snow::zone_of(
            source
                .snowflake_types
                .iter()
                .find(|(c, _)| c == column)
                .map(|(_, t)| t.as_str()),
        )
    }

    /// Snowflake's and Databricks' `_bound_format(table, column)`.
    fn bound_format(&self, table: &str, column: &str) -> String {
        match self.kind {
            Backend::Snowflake => {
                if self.snow_zone(table, column).is_none() {
                    snow::NAIVE_BOUND_FORMAT.to_string()
                } else {
                    snow::ZONED_BOUND_FORMAT.to_string()
                }
            }
            _ => {
                if self.dtype(table, column) == Some(PqlDtype::String) {
                    "%Y-%m-%d %H:%M:%S".to_string()
                } else {
                    "%Y-%m-%d %H:%M:%S+00:00".to_string()
                }
            }
        }
    }

    /// `_no_rows(table, columns)`.
    fn no_rows(
        &self,
        sampler: &Sampler,
        table: &str,
        columns: &[String],
    ) -> Result<(Frame, Vec<i64>), Raised> {
        let time_column = sampler.time_column(table);
        let names = self.ordered_columns(table, columns)?;
        let series = names
            .iter()
            .map(|c| {
                if Some(c) == time_column {
                    time_series(&[])
                } else {
                    Series::new(Dtype::Object, new_null_array(&DataType::Null, 0))
                }
            })
            .collect();
        Ok((Frame::new(names, series, 0), Vec::new()))
    }
}

/// One backend's `_by_time(table, foreign_key, index, anchor_time,
/// min_offset, max_offset, columns)`: every row of `table` in each id's
/// window, and the position each came for.
#[allow(clippy::too_many_arguments)]
pub async fn by_time(
    sampler: &Sampler,
    db: &Db,
    table: &str,
    foreign_key: &str,
    index: &Series,
    anchor_time: &[Option<i64>],
    min_offset: Option<&DateOffset>,
    max_offset: &DateOffset,
    columns: &[String],
) -> Result<(Frame, Vec<i64>), Raised> {
    let sql = sql_of(sampler)?;
    let kind = sql.kind;
    let time_column = sampler
        .time_column(table)
        .cloned()
        .ok_or_else(|| Raised::key_error(table))?;
    if matches!(kind, Backend::Snowflake | Backend::Databricks) && index.is_empty() {
        // The window's outer bounds are the latest and earliest of the
        // entities' times, which for none is NaN, and a warehouse refuses.
        return sql.no_rows(sampler, table, columns);
    }
    let source = sql.source_name(table)?.clone();
    let key_ref = sql.column_ref(table, foreign_key)?;
    let ends = offset_times(anchor_time, max_offset)?;
    let starts = match min_offset {
        None => None,
        Some(offset) => Some(offset_times(anchor_time, offset)?),
    };
    match kind {
        Backend::Duckdb | Backend::Sqlite => {
            let (end_times, start_times, time_ref) = if kind == Backend::Sqlite {
                let spelling = sql.time_spelling(sampler, db, table).await?;
                let spelled = |times: &[Option<i64>]| {
                    arrow_list(&Series::new(
                        Dtype::Object,
                        Arc::new(StringArray::from(spell(&spelling, times))),
                    ))
                };
                (
                    spelled(&ends)?,
                    starts.as_deref().map(spelled).transpose()?,
                    sql.column_ref(table, &time_column)?,
                )
            } else {
                (
                    arrow_list(&time_series(&ends))?,
                    starts
                        .as_deref()
                        .map(|s| arrow_list(&time_series(s)))
                        .transpose()?,
                    sql.duckdb_time_ref(table, &time_column)?,
                )
            };
            let projections = sql.projections(table, columns)?;
            let order_by = sql.neighbor_order_by(
                sampler,
                table,
                Some(&time_ref),
                &key_ref,
                Some(columns),
                false,
            )?;
            let ids = arrow_list(index)?;
            let statement = if kind == Backend::Duckdb {
                duckdb::window(
                    &source,
                    &key_ref,
                    &ids,
                    &projections,
                    &time_ref,
                    &end_times,
                    start_times.as_ref(),
                    &order_by,
                )?
            } else {
                sqlite::window(
                    &source,
                    &key_ref,
                    &ids,
                    &projections,
                    &time_ref,
                    &end_times,
                    start_times.as_ref(),
                    &order_by,
                )?
            };
            let rows = db.run(statement).await?;
            let frame = arrow_result_frame(&rows)?;
            let batch = int_column(&frame, "__kumo_batch__")?;
            let frame = frame.without("__kumo_batch__");
            let frame = sql.sanitized(sampler, table, frame)?;
            Ok((frame, batch))
        }
        Backend::Postgres => {
            let dtype = sql
                .dtype_of(table, foreign_key)?
                .ok_or_else(|| Raised::internal("a key without a dtype"))?;
            let postgres_type = sql.postgres_type(table, foreign_key)?;
            let time_type = sql.postgres_type(table, &time_column)?;
            let utc = sql.dtype(table, &time_column) != Some(PqlDtype::String);
            let serialize = |times: &[Option<i64>]| -> Result<Vec<Py>, Raised> {
                series_values(&time_series(times))?
                    .iter()
                    .map(|v| Ok(postgres::serialize_time(v, utc)?.map_or(Py::None, Py::Str)))
                    .collect()
            };
            let end_text = serialize(&ends)?;
            let start_text = starts.as_deref().map(serialize).transpose()?;
            let ids = series_values(index)?;
            let fields: Vec<String> = if start_text.is_some() {
                vec!["e".into(), "s".into()]
            } else {
                vec!["e".into()]
            };
            let rows: Vec<Py> = ids
                .iter()
                .enumerate()
                .map(|(i, id)| {
                    let mut row = vec![
                        (
                            "id".to_string(),
                            postgres::coerce_id(id, dtype, &postgres_type)?,
                        ),
                        ("e".to_string(), end_text[i].clone()),
                    ];
                    if let Some(starts) = &start_text {
                        row.push(("s".to_string(), starts[i].clone()));
                    }
                    Ok(Py::Dict(row))
                })
                .collect::<Result<_, Raised>>()?;
            let time_ref = sql.column_ref(table, &time_column)?;
            let projections = sql.projections(table, columns)?;
            let order_by = sql.neighbor_order_by(
                sampler,
                table,
                Some(&time_ref),
                &key_ref,
                Some(columns),
                false,
            )?;
            let frame = sql
                .postgres_flatten(db, &rows, |chunk| {
                    postgres::window(
                        &source,
                        &projections,
                        &key_ref,
                        &postgres_type,
                        &Py::List(chunk.to_vec()),
                        &fields,
                        &time_ref,
                        &time_type,
                        &order_by,
                    )
                })
                .await?;
            sql.postgres_output(table, frame)
        }
        Backend::Snowflake | Backend::Databricks => {
            let dtype = sql
                .dtype_of(table, foreign_key)?
                .ok_or_else(|| Raised::internal("a key without a dtype"))?;
            let spec = sql.bound_format(table, &time_column);
            let end_text: Vec<Option<String>> = ends
                .iter()
                .map(|t| t.map(|ns| strftime(ns, &spec)))
                .collect();
            let start_text: Option<Vec<Option<String>>> = starts
                .as_ref()
                .map(|s| s.iter().map(|t| t.map(|ns| strftime(ns, &spec))).collect());
            let ids = series_values(index)?;
            let mut bounds = vec![end_text.iter().flatten().max().cloned().unwrap_or_default()];
            if let Some(s) = &start_text {
                bounds.push(s.iter().flatten().min().cloned().unwrap_or_default());
            }
            let time_ref = sql.column_ref(table, &time_column)?;
            let projections = sql.projections(table, columns)?;
            let order_by = sql.neighbor_order_by(
                sampler,
                table,
                Some(&time_ref),
                &key_ref,
                Some(columns),
                false,
            )?;
            let text = |v: &Option<String>| v.clone().map_or(Py::None, Py::Str);
            let rows = if kind == Backend::Snowflake {
                let tuples: Vec<Py> = ids
                    .iter()
                    .enumerate()
                    .map(|(i, id)| {
                        let mut row = vec![id.clone(), text(&end_text[i])];
                        if let Some(s) = &start_text {
                            row.push(text(&s[i]));
                        }
                        Py::Tuple(row)
                    })
                    .collect();
                db.run(snow::window(
                    &source,
                    &projections,
                    &key_ref,
                    dtype,
                    &Py::List(tuples),
                    start_text.is_some(),
                    &time_ref,
                    &bounds,
                    &order_by,
                    sql.snow_zone(table, &time_column),
                )?)
                .await?
            } else {
                let rows: Vec<Py> = ids
                    .iter()
                    .enumerate()
                    .map(|(i, id)| {
                        let mut row = vec![
                            ("id".to_string(), databricks::coerce_id(id, dtype)?),
                            ("e".to_string(), text(&end_text[i])),
                        ];
                        if let Some(s) = &start_text {
                            row.push(("s".to_string(), text(&s[i])));
                        }
                        Ok(Py::Dict(row))
                    })
                    .collect::<Result<_, Raised>>()?;
                databricks_flatten(db, &rows, |chunk| {
                    databricks::window(
                        &source,
                        &projections,
                        &key_ref,
                        dtype,
                        &Py::List(chunk.to_vec()),
                        start_text.is_some(),
                        &time_ref,
                        &bounds,
                        &order_by,
                    )
                })
                .await?
            };
            let frame = arrow_result_frame(&rows)?;
            let batch = int_column(&frame, "__KUMO_BATCH__")?;
            let frame = frame.without("__KUMO_BATCH__");
            let frame = sql.sanitized(sampler, table, frame)?;
            Ok((frame, batch))
        }
        Backend::Local => Err(Raised::internal("a local sampler has no SQL")),
    }
}

/// `SQLSampler._sample_query_data`: the rows a query's labels are computed
/// from, by table, with their times and the examples they are for.
#[allow(clippy::type_complexity)]
pub async fn sample_query_data(
    sampler: &mut Sampler,
    db: &Db,
    entity_table: &str,
    entity_pkey: &Series,
    anchor_time: &[Option<i64>],
    columns_dict: &Columns,
    time_offsets: &TimeOffsets,
) -> Result<
    (
        Vec<(String, Frame)>,
        Vec<(String, Series)>,
        Vec<(String, Vec<i64>)>,
    ),
    Raised,
> {
    let mut feat: Vec<(String, Frame)> = Vec::new();
    let mut time: Vec<(String, Series)> = Vec::new();
    let mut batch: Vec<(String, Vec<i64>)> = Vec::new();
    let entity_columns = columns_dict.get(entity_table).cloned().unwrap_or_default();
    let entity_frame = if entity_columns.len() > 1 {
        let (frame, rows) = by_pkey(
            sampler,
            db,
            entity_table,
            entity_pkey,
            &entity_columns,
            None,
        )
        .await?;
        let perm = request_order(entity_table, &rows)?;
        frame.take(&perm)?
    } else {
        let pkey = sampler.primary_key(entity_table)?.clone();
        Frame::new(vec![pkey], vec![entity_pkey.clone()], entity_pkey.len())
    };
    let n = entity_frame.len() as i64;
    feat.push((entity_table.to_string(), entity_frame));
    batch.push((entity_table.to_string(), (0..n).collect()));
    for ((table, foreign_key, _), (min, max)) in time_offsets {
        let columns = columns_dict
            .get(table)
            .cloned()
            .ok_or_else(|| Raised::key_error(table))?;
        let (frame, rows) = by_time(
            sampler,
            db,
            table,
            foreign_key,
            entity_pkey,
            anchor_time,
            min.as_ref(),
            max,
            &columns,
        )
        .await?;
        if let Some(column) = sampler.time_column(table)
            && columns.contains(column)
        {
            set(&mut time, table, frame.column(column)?.clone());
        }
        set(&mut feat, table, frame);
        set(&mut batch, table, rows);
    }
    Ok((feat, time, batch))
}

fn set<T>(dict: &mut Vec<(String, T)>, key: &str, value: T) {
    match dict.iter_mut().find(|(k, _)| k == key) {
        Some((_, v)) => *v = value,
        None => dict.push((key.to_string(), value)),
    }
}

/// What `_sample_subgraph` gives (`SamplerOutput`).
pub struct SamplerOutput {
    pub anchor_time: Vec<i64>,
    pub df_dict: Vec<(String, Frame)>,
    pub inverse_dict: Vec<(String, Vec<i64>)>,
    pub batch_dict: Vec<(String, Vec<i64>)>,
    pub num_sampled_nodes_dict: Vec<(String, Vec<i64>)>,
    pub row_dict: Vec<(EdgeType, Vec<i64>)>,
    pub col_dict: Vec<(EdgeType, Vec<i64>)>,
    pub num_sampled_edges_dict: Vec<(EdgeType, Vec<i64>)>,
}

fn entry<'a, K: PartialEq + Clone, T>(
    dict: &'a mut Vec<(K, T)>,
    key: &K,
    make: impl FnOnce() -> T,
) -> &'a mut T {
    if let Some(i) = dict.iter().position(|(k, _)| k == key) {
        return &mut dict[i].1;
    }
    dict.push((key.clone(), make()));
    let last = dict.len() - 1;
    &mut dict[last].1
}

/// The anchor times of `positions`, as `anchor_time.iloc[positions]`.
fn anchors_at(anchor: &[Option<i64>], positions: &[i64]) -> Result<Vec<Option<i64>>, Raised> {
    positions
        .iter()
        .map(|&p| {
            usize::try_from(p)
                .ok()
                .and_then(|p| anchor.get(p).copied())
                .ok_or_else(|| Raised::new("IndexError", "positional indexers are out-of-bounds"))
        })
        .collect()
}

/// `SQLSampler._sample_subgraph`: the traversal, hop by hop, each hop's
/// lookups pushed down to the database.
#[allow(clippy::too_many_arguments)]
pub async fn sample_subgraph(
    sampler: &mut Sampler,
    db: &Db,
    entity_table: &str,
    entity_pkey: &Series,
    anchor_time: Option<&[Option<i64>]>,
    columns_dict: &Columns,
    num_neighbors: &[i64],
    scored: Option<&[i64]>,
) -> Result<SamplerOutput, Raised> {
    // Keys, foreign keys and times are read with every table: the
    // traversal walks them.
    let mut sample_columns = Columns::default();
    for (table, columns) in &columns_dict.0 {
        let mut wanted = columns.clone();
        for (fkey, _) in sampler.foreign_keys(table)? {
            if !wanted.contains(fkey) {
                wanted.push(fkey.clone());
            }
        }
        if let Ok(key) = sampler.primary_key(table)
            && !wanted.contains(key)
        {
            wanted.push(key.clone());
        }
        sample_columns.0.push((table.clone(), wanted));
    }
    if anchor_time.is_none() {
        let column = sampler
            .time_column(entity_table)
            .cloned()
            .ok_or_else(|| Raised::key_error(entity_table))?;
        sample_columns.add(entity_table, &column);
    }

    // The entity table.
    let entity_columns = sample_columns
        .get(entity_table)
        .cloned()
        .ok_or_else(|| Raised::key_error(entity_table))?;
    let (df, batch) = by_pkey(
        sampler,
        db,
        entity_table,
        entity_pkey,
        &entity_columns,
        scored,
    )
    .await?;
    if batch.len() != entity_pkey.len() {
        let mut mask = vec![true; entity_pkey.len()];
        for &b in &batch {
            if let Some(m) = usize::try_from(b).ok().and_then(|b| mask.get_mut(b)) {
                *m = false;
            }
        }
        let missing: Vec<usize> = (0..mask.len()).filter(|&i| mask[i]).collect();
        let keys = keys_of(&entity_pkey.take(&missing)?)?;
        return Err(Raised::key_error(&format!(
            "The primary keys {} do not exist in the '{entity_table}' table",
            list_repr(&keys)
        )));
    }
    let perm = request_order(entity_table, &batch)?;
    let batch: Vec<i64> = perm.iter().map(|&p| batch[p]).collect();
    let df = df.take(&perm)?;

    // 'entity' anchor times are the entities' own.
    let anchor: Vec<Option<i64>> = match anchor_time {
        Some(anchor) => anchor.to_vec(),
        None => {
            let column = sampler
                .time_column(entity_table)
                .cloned()
                .ok_or_else(|| Raised::key_error(entity_table))?;
            times_of(df.column(&column)?)?
        }
    };

    let num_examples = entity_pkey.len();
    let hops = num_neighbors.len();
    let mut mappers: Vec<(String, Mapper)> = Vec::new();
    let entity_key = sampler.primary_key(entity_table)?.clone();
    entry(&mut mappers, &entity_table.to_string(), || {
        Mapper::new(num_examples)
    })
    .add(df.column(&entity_key)?, &batch)?;

    let mut dfs: Vec<(String, Vec<Frame>)> = vec![(entity_table.to_string(), vec![df])];
    let mut batches: Vec<(String, Vec<Vec<i64>>)> = vec![(entity_table.to_string(), vec![batch])];
    let mut num_sampled_nodes: Vec<(String, Vec<i64>)> =
        vec![(entity_table.to_string(), vec![0; hops + 1])];
    num_sampled_nodes[0].1[0] = num_examples as i64;
    let mut rows_dict: Vec<(EdgeType, Vec<Vec<i64>>)> = Vec::new();
    let mut cols_dict: Vec<(EdgeType, Vec<Vec<i64>>)> = Vec::new();
    let mut num_sampled_edges: Vec<(EdgeType, Vec<i64>)> = Vec::new();
    let mut offsets: HashMap<String, usize> = HashMap::new();

    let mut visited_hops = 0;
    for (hop, &neighbors) in num_neighbors.iter().enumerate() {
        if neighbors == 0 {
            break;
        }
        let tables: Vec<String> = num_sampled_nodes.iter().map(|(t, _)| t.clone()).collect();
        for table in tables {
            if num_sampled_nodes
                .iter()
                .find(|(t, _)| *t == table)
                .map_or(0, |(_, n)| n[hop])
                == 0
            {
                continue;
            }
            // The previous hop's slices, with only the keys.
            let mut cols: Vec<String> = sampler
                .foreign_keys(&table)?
                .iter()
                .map(|(k, _)| k.clone())
                .collect();
            if let Ok(key) = sampler.primary_key(&table) {
                cols.push(key.clone());
            }
            let offset = *offsets.get(&table).unwrap_or(&0);
            let table_dfs = &dfs
                .iter()
                .find(|(t, _)| *t == table)
                .ok_or_else(|| Raised::key_error(&table))?
                .1;
            let parts: Vec<Frame> = table_dfs[offset..]
                .iter()
                .map(|f| f.select(&cols))
                .collect::<Result<_, _>>()?;
            let df = if parts.len() > 1 {
                Frame::concat(&parts.iter().collect::<Vec<_>>())?
            } else {
                parts
                    .into_iter()
                    .next()
                    .ok_or_else(|| Raised::new("IndexError", "list index out of range"))?
            };
            let table_batches = &batches
                .iter()
                .find(|(t, _)| *t == table)
                .ok_or_else(|| Raised::key_error(&table))?
                .1;
            let batch: Vec<i64> = table_batches[offset..].concat();
            offsets.insert(table.clone(), table_batches.len());

            let mut pkey: Option<Series> = None;
            let mut index: Option<Vec<i64>> = None;
            if let Ok(key) = sampler.primary_key(&table) {
                let key_series = df.column(key)?.clone();
                let mapper = entry(&mut mappers, &table, || Mapper::new(num_examples));
                index = Some(mapper.get(&key_series, &batch)?);
                pkey = Some(key_series);
            }

            // The foreign keys of this table.
            let foreign = sampler.foreign_keys(&table)?.clone();
            for (fkey, dst_table) in &foreign {
                let fkey_series = df.column(fkey)?.clone();
                let mut row = entry(&mut mappers, dst_table, || Mapper::new(num_examples))
                    .get(&fkey_series, &batch)?;
                let missing: Vec<usize> = (0..row.len()).filter(|&i| row[i] == -1).collect();
                if !missing.is_empty() {
                    let key_series = fkey_series.take(&missing)?;
                    let key_batch: Vec<i64> = missing.iter().map(|&i| batch[i]).collect();
                    let key_keys = keys_of(&key_series)?;
                    // Unique keys per example, then across examples.
                    let unique_rows = first_of_pairs(&key_keys, &key_batch);
                    let unique_series = key_series.take(&unique_rows)?;
                    let unique_keys: Vec<Key> =
                        unique_rows.iter().map(|&i| key_keys[i].clone()).collect();
                    let unique_batch: Vec<i64> =
                        unique_rows.iter().map(|&i| key_batch[i]).collect();
                    let (code, first) = factorize(&unique_keys);
                    let fkey_index = unique_series.take(&first)?;
                    let dst_columns = sample_columns
                        .get(dst_table)
                        .cloned()
                        .ok_or_else(|| Raised::key_error(dst_table))?;
                    let (dst_df, dst_batch) =
                        by_pkey(sampler, db, dst_table, &fkey_index, &dst_columns, None).await?;
                    if !dst_df.is_empty() {
                        let order = request_order(dst_table, &dst_batch)?;
                        let dst_df = dst_df.take(&order)?;
                        let mut found = vec![false; fkey_index.len()];
                        for &b in &dst_batch {
                            if let Some(f) = usize::try_from(b).ok().and_then(|b| found.get_mut(b))
                            {
                                *f = true;
                            }
                        }
                        // A null key joins no row; today's read its code,
                        // -1, as the last row looked up, and joined that.
                        let valid: Vec<bool> = code
                            .iter()
                            .map(|&c| {
                                usize::try_from(c)
                                    .ok()
                                    .and_then(|c| found.get(c).copied())
                                    .unwrap_or(false)
                            })
                            .collect();
                        let kept: Vec<usize> = (0..valid.len()).filter(|&i| valid[i]).collect();
                        let kept_keys: Vec<Key> =
                            kept.iter().map(|&i| unique_keys[i].clone()).collect();
                        let (recode, _) = factorize(&kept_keys);
                        // `_df.iloc[code]`, of keys that are none of them null.
                        let positions: Vec<usize> = recode.iter().map(|&c| c as usize).collect();
                        let dst_df = dst_df.take(&positions)?;
                        let new_batch: Vec<i64> = kept.iter().map(|&i| unique_batch[i]).collect();
                        let dst_key = sampler.primary_key(dst_table)?.clone();
                        let mapper = entry(&mut mappers, dst_table, || Mapper::new(num_examples));
                        mapper.add(dst_df.column(&dst_key)?, &new_batch)?;
                        let looked = mapper.get(&key_series, &key_batch)?;
                        for (j, &i) in missing.iter().enumerate() {
                            row[i] = looked[j];
                        }
                        let count = new_batch.len() as i64;
                        entry(&mut dfs, dst_table, Vec::new).push(dst_df);
                        entry(&mut batches, dst_table, Vec::new).push(new_batch);
                        entry(&mut num_sampled_nodes, dst_table, || vec![0; hops + 1])[hop + 1] +=
                            count;
                    }
                }
                let keep: Vec<usize> = (0..row.len()).filter(|&i| row[i] != -1).collect();
                let col: Vec<i64> = match &index {
                    Some(index) => index.clone(),
                    None => {
                        let counts = &num_sampled_nodes
                            .iter()
                            .find(|(t, _)| *t == table)
                            .ok_or_else(|| Raised::key_error(&table))?
                            .1;
                        let start: i64 = counts[..hop].iter().sum();
                        let end: i64 = counts[..hop + 1].iter().sum();
                        (start..end).collect()
                    }
                };
                let row: Vec<i64> = keep.iter().map(|&i| row[i]).collect();
                let col: Vec<i64> = keep
                    .iter()
                    .map(|&i| {
                        col.get(i)
                            .copied()
                            .ok_or_else(|| Raised::new("IndexError", "index out of bounds"))
                    })
                    .collect::<Result<_, _>>()?;
                if !col.is_empty() {
                    let edge_type =
                        rev_edge_type(&(table.clone(), fkey.clone(), dst_table.clone()));
                    entry(&mut rows_dict, &edge_type, Vec::new).push(row);
                    let count = col.len() as i64;
                    entry(&mut cols_dict, &edge_type, Vec::new).push(col);
                    entry(&mut num_sampled_edges, &edge_type, || vec![0; hops])[hop] = count;
                    visited_hops = hop + 1;
                }
            }

            // The foreign keys referencing this table.
            let referencing = sampler.rev_foreign_keys(&table)?.clone();
            for (src_table, fkey) in &referencing {
                let (Some(pkey), Some(index)) = (&pkey, &index) else {
                    return Err(Raised::new("AssertionError", ""));
                };
                let anchors = anchors_at(&anchor, &batch)?;
                let src_columns = sample_columns
                    .get(src_table)
                    .cloned()
                    .ok_or_else(|| Raised::key_error(src_table))?;
                let (src_df, src_batch) = by_fkey(
                    sampler,
                    db,
                    src_table,
                    fkey,
                    pkey,
                    neighbors,
                    Some(&anchors),
                    &src_columns,
                )
                .await?;
                if src_df.is_empty() {
                    continue;
                }
                let edge_type = (src_table.clone(), fkey.clone(), table.clone());
                let col: Vec<i64> = src_batch
                    .iter()
                    .map(|&b| {
                        usize::try_from(b)
                            .ok()
                            .and_then(|b| index.get(b).copied())
                            .ok_or_else(|| Raised::new("IndexError", "index out of bounds"))
                    })
                    .collect::<Result<_, _>>()?;
                let count = src_batch.len() as i64;
                entry(&mut cols_dict, &edge_type, Vec::new).push(col);
                entry(&mut num_sampled_edges, &edge_type, || vec![0; hops])[hop] = count;
                visited_hops = hop + 1;

                let mut new_batch: Vec<i64> = src_batch
                    .iter()
                    .map(|&b| {
                        usize::try_from(b)
                            .ok()
                            .and_then(|b| batch.get(b).copied())
                            .ok_or_else(|| Raised::new("IndexError", "index out of bounds"))
                    })
                    .collect::<Result<_, _>>()?;
                let num_nodes: i64 = num_sampled_nodes
                    .iter()
                    .find(|(t, _)| t == src_table)
                    .map_or(0, |(_, n)| n.iter().sum());
                let mut src_df = src_df;
                let row: Vec<i64>;
                if let Ok(src_key) = sampler.primary_key(src_table) {
                    let src_pkey = src_df.column(src_key)?.clone();
                    let mapper = entry(&mut mappers, src_table, || Mapper::new(num_examples));
                    mapper.add(&src_pkey, &new_batch)?;
                    row = mapper.get(&src_pkey, &new_batch)?;
                    // Only rows not known yet, each (pkey, batch) pair once.
                    let duplicated = crate::keys::duplicated(&row);
                    let keep: Vec<usize> = (0..row.len())
                        .filter(|&i| row[i] >= num_nodes && !duplicated[i])
                        .collect();
                    src_df = src_df.take(&keep)?;
                    new_batch = keep.iter().map(|&i| new_batch[i]).collect();
                } else {
                    row = (num_nodes..num_nodes + new_batch.len() as i64).collect();
                }
                entry(&mut rows_dict, &edge_type, Vec::new).push(row);
                let count = new_batch.len() as i64;
                entry(&mut num_sampled_nodes, src_table, || vec![0; hops + 1])[hop + 1] += count;
                entry(&mut dfs, src_table, Vec::new).push(src_df);
                entry(&mut batches, src_table, Vec::new).push(new_batch);
            }
        }
    }

    // Post-processing.
    let mut df_dict: Vec<(String, Frame)> = Vec::new();
    for (table, frames) in &dfs {
        let frame = if frames.len() > 1 {
            Frame::concat(&frames.iter().collect::<Vec<_>>())?
        } else {
            frames[0].clone()
        };
        df_dict.push((table.clone(), frame));
    }
    // Only unique rows in `df` above a threshold.
    let mut inverse_dict: Vec<(String, Vec<i64>)> = Vec::new();
    for (table, frame) in df_dict.iter_mut() {
        let Ok(key) = sampler.primary_key(table) else {
            continue;
        };
        let keys = keys_of(frame.column(key)?)?;
        let (index, inverse) = np_unique(&keys)?;
        if frame.len() as f64 > 1.05 * index.len() as f64 {
            *frame = frame.take(&index)?;
            inverse_dict.push((table.clone(), inverse));
        }
    }
    let mut out_frames = Vec::new();
    for (table, frame) in df_dict {
        let wanted = columns_dict
            .get(&table)
            .cloned()
            .ok_or_else(|| Raised::key_error(&table))?;
        out_frames.push((table, frame.select_in_order(&wanted)?));
    }
    let batch_dict: Vec<(String, Vec<i64>)> = batches
        .iter()
        .map(|(t, b)| (t.clone(), b.concat()))
        .collect();
    let row_dict: Vec<(EdgeType, Vec<i64>)> = rows_dict
        .iter()
        .map(|(e, r)| (e.clone(), r.concat()))
        .collect();
    let col_dict: Vec<(EdgeType, Vec<i64>)> = cols_dict
        .iter()
        .map(|(e, c)| (e.clone(), c.concat()))
        .collect();
    if visited_hops != hops {
        for (_, counts) in num_sampled_nodes.iter_mut() {
            counts.truncate(visited_hops + 1);
        }
        for (_, counts) in num_sampled_edges.iter_mut() {
            counts.truncate(visited_hops);
        }
    }
    Ok(SamplerOutput {
        anchor_time: anchor.iter().map(|t| t.unwrap_or(i64::MIN)).collect(),
        df_dict: out_frames,
        inverse_dict,
        batch_dict,
        num_sampled_nodes_dict: num_sampled_nodes,
        row_dict,
        col_dict,
        num_sampled_edges_dict: num_sampled_edges,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_is_written_as_python_writes_it() {
        let value = PyVal::Dict(vec![
            (PyVal::Str("w".into()), PyVal::Int(7)),
            (
                PyVal::Str("tags".into()),
                PyVal::List(vec![
                    PyVal::Str("é\"\n\u{7f}😀".into()),
                    PyVal::Float(0.1),
                    PyVal::Bool(true),
                    PyVal::None,
                ]),
            ),
        ]);
        // json.dumps({'w': 7, 'tags': ['é"\n\x7f😀', 0.1, True, None]},
        //            separators=(',', ':'))
        assert_eq!(
            json_dumps(&value),
            r#"{"w":7,"tags":["\u00e9\"\n\u007f\ud83d\ude00",0.1,true,null]}"#
        );
    }

    #[test]
    fn decimals_keep_their_digits_and_scale() {
        assert_eq!(decimal_parts("-14.610"), Some((-14610, 3)));
        assert_eq!(decimal_parts("1.5E+3"), Some((1500, 0)));
        assert_eq!(decimal_parts("1.5E-7"), Some((15, 8)));
        assert_eq!(decimal_parts("NaN"), None);
        assert_eq!(decimal_parts("Infinity"), None);
        let cells = [
            PyVal::Decimal("14.61".into()),
            PyVal::None,
            PyVal::Decimal("2".into()),
        ];
        let array = decimal_array(&cells).unwrap();
        assert_eq!(array.scale(), 2);
        assert_eq!(array.value(2), 200);
    }

    #[test]
    fn decimal_lists_infer_precision_without_rounding() {
        for (text, precision, scale) in [
            ("1.25", 3, 2),
            ("0.001", 3, 3),
            ("1E+3", 4, 0),
            ("-12345678901234567890.1234567890123456789", 39, 19),
        ] {
            let cells = [PyVal::Decimal(text.into()), PyVal::None];
            let array = crate::frame::decimal_lists::decimal_list_items(&cells);
            let expected = if precision <= 38 {
                DataType::Decimal128(precision, scale)
            } else {
                DataType::Decimal256(precision, scale)
            };
            assert_eq!(array.data_type(), &expected);
            assert!(array.is_null(1));
            if precision == 39 {
                assert_eq!(sdm_schema::engine::result::value_at(&array, 0), cells[0]);
            }
        }
        for text in ["9".repeat(77), "NaN".into(), "Infinity".into()] {
            let cells = vec![PyVal::Decimal(text), PyVal::None];
            let array = crate::frame::decimal_lists::decimal_list_items(&cells);
            assert_eq!(crate::frame::object_values(&array), Some(cells));
        }
    }

    #[test]
    fn decimal_list_identity_survives_frame_row_operations() {
        for (fraction, expected_type) in [
            ("0.001", DataType::Decimal128(4, 3)),
            ("1E-39", DataType::Decimal256(40, 39)),
        ] {
            let cells = vec![
                PyVal::List(vec![PyVal::Decimal("2.00".into()), PyVal::None]),
                PyVal::None,
                PyVal::List(vec![PyVal::Decimal(fraction.into())]),
                PyVal::List(vec![]),
            ];
            let series = pylists_to_series(&cells).unwrap();
            assert_eq!(series.object_values(), Some(cells.clone()));
            let DataType::List(field) = series.array.data_type() else {
                panic!("expected an Arrow list");
            };
            assert_eq!(field.data_type(), &expected_type);
            assert_ne!(
                sdm_schema::engine::result::value_at(&series.array, 0),
                cells[0]
            );
            let frame = Frame::new(vec!["tags".into()], vec![series], cells.len());
            let selected = frame.select(&["tags".into()]).unwrap();
            let repeated = selected.take(&[2, 0, 2, 1, 3]).unwrap();
            let filtered = repeated.take(&[1, 3, 4]).unwrap();
            let empty = selected.take(&[]).unwrap();
            let joined = Frame::concat(&[&empty, &filtered, &repeated, &empty]).unwrap();
            let expected = [0, 1, 3, 2, 0, 2, 1, 3]
                .into_iter()
                .map(|position| cells[position].clone())
                .collect::<Vec<_>>();
            let values = joined.column("tags").unwrap();
            assert_eq!(values.object_values(), Some(expected.clone()));
            assert_eq!(values.len(), expected.len());
            assert_eq!(
                crate::driver::sampler::head(values, 3).object_values(),
                Some(expected[..3].to_vec())
            );
            assert_eq!(
                crate::driver::sampler::tail(values, 3).object_values(),
                Some(expected[3..].to_vec())
            );
            assert_eq!(
                crate::driver::sampler::tail(values, values.len()).object_values(),
                Some(vec![])
            );
        }
    }

    fn postgres_rows(oid: &str, column: ArrayRef) -> Rows {
        let field = arrow_schema::Field::new("c", column.data_type().clone(), true).with_metadata(
            [("postgres.type_oid".to_string(), oid.to_string())]
                .into_iter()
                .collect::<std::collections::HashMap<_, _>>(),
        );
        let schema = Arc::new(arrow_schema::Schema::new(vec![field]));
        let batch = RecordBatch::try_new(schema.clone(), vec![column]).unwrap();
        Rows {
            schema,
            batches: vec![batch],
        }
    }

    #[test]
    fn a_postgres_array_stays_lists_and_json_becomes_text() {
        // `text[]`: psycopg reads lists, which `_frame` keeps as lists.
        let mut tags =
            arrow_array::builder::ListBuilder::new(arrow_array::builder::StringBuilder::new());
        tags.values().append_value("a");
        tags.values().append_null();
        tags.append(true);
        tags.append(false);
        tags.append(true);
        let frame = postgres_object_frame(&postgres_rows("1009", Arc::new(tags.finish()))).unwrap();
        let cells = object_cells(&frame.columns[0]);
        assert_eq!(
            cells,
            [
                PyVal::List(vec![PyVal::Str("a".into()), PyVal::None]),
                PyVal::None,
                PyVal::List(vec![]),
            ]
        );
        assert!(texts(frame.columns[0].array.as_ref()).is_none());
        // Typed `string`, `_sanitize_postgres_frame` writes a list as JSON.
        let strings = nullable_strings(&cells).unwrap();
        assert_eq!(
            texts(strings.array.as_ref()).unwrap(),
            [
                Some(r#"["a",null]"#.to_string()),
                None,
                Some("[]".to_string())
            ]
        );
        // `jsonb` holding lists is kept as the text it is written back as.
        let json: ArrayRef = Arc::new(StringArray::from(vec![Some("[1, 2.5]"), None]));
        let frame = postgres_object_frame(&postgres_rows("3802", json)).unwrap();
        assert_eq!(
            texts(frame.columns[0].array.as_ref()).unwrap(),
            [Some("[1,2.5]".to_string()), None]
        );
    }
}
