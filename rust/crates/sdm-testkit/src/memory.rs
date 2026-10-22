// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The minimal non-SQL test provider (T2.7, plan section 3, "Data
//! access"): a host provider that answers the core's typed data requests
//! from in-memory Arrow tables, never through SQL. Dev-only; nothing
//! shipped reaches it.
//!
//! Its source is plain data: each [`Table`] its name, its rows as one
//! Arrow batch, and its catalog (the columns as declared, its keys and
//! its foreign keys). A request's relation names a table by one part, the
//! name exactly, as this source resolves a name: a relation of any other
//! shape, or a name it does not hold, is not a table it reads, and the
//! request fails as the source's (`data_source`, as the kit's
//! `capability-failure.declared` asks), never answered from another
//! table. Two schema tables over one source are two requests naming one
//! table, each answered from the same rows.
//!
//! It answers `describe` and `scan`, which every provider answers, and
//! declares every optional request but `column_ranges`
//! ([`Memory::capabilities`]): the kit's (`preview`, `count_rows`,
//! `key_counts`, `repeated_key`, `neighbors`, `lookup`) and the two the
//! core's sampler requires of a provider before it reads anything
//! (`time_bounds`, `sample_entities`). Without `column_ranges` the core
//! chooses a time column from the preview, as it does for a source that
//! cannot ask. Its only filter is the not-null filter the core writes.
//! A sample keeps the table's order: every row where `n` covers them, else
//! its own seeded draw ([`drawn`]). Values are compared
//! as the SQL pushdowns compare them: a key's parts must all be present to
//! match; integers of any width, and decimals of scale 0, by value; floats
//! with `-0.0` and `0.0` one value and every NaN one; a time as
//! `CAST(column AS TIMESTAMP)` reads it, a zone-less timestamp of any unit
//! (a finer one truncated to microseconds) or a date at its midnight. What
//! it does not read (a time of another type, a key compared with seeds of
//! another type) fails the request, `unsupported`, never answered
//! otherwise.
//!
//! [`Memory::from_duckdb`] copies a DuckDB database's tables and catalog
//! into one, once, to build a fixture (the kit's dataset, a case's base);
//! the provider itself never runs a statement.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use arrow_array::builder::{ListBuilder, StringBuilder};
use arrow_array::cast::AsArray;
use arrow_array::types::{
    Date32Type, Date64Type, Decimal128Type, Float32Type, Float64Type, Int8Type, Int16Type,
    Int32Type, Int64Type, TimestampMicrosecondType, TimestampMillisecondType,
    TimestampNanosecondType, TimestampSecondType, UInt8Type, UInt16Type, UInt32Type, UInt64Type,
};
use arrow_array::{
    Array, ArrayRef, BooleanArray, Decimal128Array, Int32Array, Int64Array, RecordBatch,
    StringArray, TimestampMicrosecondArray, UInt32Array,
};
use arrow_row::{OwnedRow, RowConverter, SortField};
use arrow_schema::{DataType, Field, Schema, SortOptions, TimeUnit};
use arrow_select::concat::concat_batches;
use arrow_select::filter::filter_record_batch;
use arrow_select::take::take;
use sdm_contract::ErrorKind;
use sdm_contract::data::{
    AttachmentId, BATCH_COLUMN, DataRequest, DescribePart, Determinism, EXACT_NUMBER_MARKED,
    EXACT_NUMBER_MARKER, Filter, KeySet, MATCHES_COLUMN, NullOrder, OptionalRequest,
    ProviderCapabilities, ProviderError, Relation, SEED_BATCH, TieTerm, Window,
    catalog_columns as c, results, seed_integer, seed_key_columns,
};

/// A DuckDB constraint as [`Memory::from_duckdb`] reads it: its type, its
/// columns as JSON, its number, and what a foreign key refers to.
type Constraint = (String, String, i64, Option<String>, Option<String>);

/// One column as the source declares it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Column {
    pub name: String,
    /// The declared type, as the source spells it.
    pub declared_type: String,
    pub not_null: bool,
    pub primary_key: bool,
}

/// One key a table declares: its kind as the catalog's answer names it
/// (`catalog_columns::KIND_PRIMARY`, `KIND_UNIQUE`, or another the reader
/// leaves alone) and its columns.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Key {
    pub kind: String,
    pub columns: Vec<String>,
}

/// One foreign key a table declares.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ForeignKey {
    /// The key's number, naming it among the table's.
    pub key: i64,
    pub referenced_table: String,
    pub columns: Vec<String>,
    pub referenced_columns: Vec<String>,
}

/// One table of the source: its name, its rows, and its catalog.
#[derive(Debug, Clone)]
pub struct Table {
    pub name: String,
    pub rows: RecordBatch,
    pub columns: Vec<Column>,
    pub keys: Vec<Key>,
    pub foreign_keys: Vec<ForeignKey>,
}

/// The source: its tables, in the order its catalog lists them.
#[derive(Debug, Clone, Default)]
pub struct Memory {
    pub tables: Vec<Table>,
}

/// The requests this provider answers itself, beside `describe` and
/// `scan`.
pub const DECLARED: &[OptionalRequest] = &[
    OptionalRequest::Preview,
    OptionalRequest::CountRows,
    OptionalRequest::KeyCounts,
    OptionalRequest::TimeBounds,
    OptionalRequest::RepeatedKey,
    OptionalRequest::SampleEntities,
    OptionalRequest::Neighbors,
    OptionalRequest::Lookup,
];

/// A request the source fails: a table or column it does not hold.
fn not_held() -> ProviderError {
    failed(ErrorKind::DataSource)
}

/// A request the provider does not read: a time of a type it does not
/// read, a key compared with seeds of another type.
fn not_read() -> ProviderError {
    failed(ErrorKind::Unsupported)
}

fn failed(kind: ErrorKind) -> ProviderError {
    ProviderError {
        refuses_expression: false,
        kind,
        retryable: false,
        sqlstate: None,
        vendor_code: None,
    }
}

/// A failure of Arrow's own on data this provider built: its defect.
fn arrow_failed<E>(_: E) -> ProviderError {
    failed(ErrorKind::Internal)
}

impl Memory {
    /// What the provider declares: the optional requests it answers
    /// itself ([`DECLARED`]), its sampling stable under a seed, any seed
    /// (its draw takes 64 bits), nulls last in either direction (it orders
    /// as each term asks), no text times (its times are typed), no index
    /// advice (it has no index) and no fallback.
    pub fn capabilities() -> ProviderCapabilities {
        ProviderCapabilities {
            derived_relations: false,
            neighbor_lookback: false,
            sample_rows: false,
            opaque_vendor: None,
            compares_stored_times: false,
            samples_by_key_dtype: false,
            reads_nan_as_missing: false,
            optional: DECLARED.iter().copied().collect(),
            max_keys_per_request: None,
            sampling: Determinism::SeededStable,
            max_seed: u64::MAX,
            text_times: None,
            null_order: Some(NullOrder::Last),
            index_advice: None,
            fallback: None,
        }
    }

    /// The key set `request` names, a `neighbors`' or `lookup`'s seeds,
    /// whose attachment its answer reads.
    pub fn key_sets(request: &DataRequest) -> Vec<&KeySet> {
        match request {
            DataRequest::Neighbors { seeds, .. } | DataRequest::Lookup { seeds, .. } => {
                vec![seeds]
            }
            _ => vec![],
        }
    }

    /// The table `relation` names: one part, the table's name exactly.
    fn table(&self, relation: &Relation) -> Result<&Table, ProviderError> {
        let name = relation.single().ok_or_else(not_held)?;
        self.tables
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(not_held)
    }

    /// The answer to `request`, its key sets read from `keys`, the
    /// attachments its `seeds` and filters name, by id; or the source's
    /// failure.
    pub fn answer(
        &self,
        request: &DataRequest,
        keys: &BTreeMap<AttachmentId, RecordBatch>,
    ) -> Result<RecordBatch, ProviderError> {
        let key_set = |set: &KeySet| keys.get(&set.attachment).ok_or_else(not_held);
        match request {
            DataRequest::Describe { tables, part } => self.describe(tables, *part),
            DataRequest::Scan {
                table,
                columns,
                filters,
                limit,
            } => {
                let rows = project(&filtered(&self.table(table)?.rows, filters)?, columns)?;
                Ok(match limit {
                    Some(limit) => first(&rows, *limit),
                    None => rows,
                })
            }
            DataRequest::Preview {
                table,
                columns,
                limit,
            } => Ok(first(&project(&self.table(table)?.rows, columns)?, *limit)),
            DataRequest::CountRows { table, filters } => {
                let rows = filtered(&self.table(table)?.rows, filters)?;
                counts(&[("rows", vec![count(rows.num_rows())])])
            }
            DataRequest::KeyCounts { keys: asked } => {
                let (mut n, mut d, mut z) = (vec![], vec![], vec![]);
                for key in asked {
                    let rows = &self.table(&key.table)?.rows;
                    let arrays = named(rows, &key.columns)?;
                    let present = present(&arrays);
                    let held: BTreeSet<OwnedRow> = row_values(&arrays)?
                        .into_iter()
                        .zip(&present)
                        .filter_map(|(row, &present)| present.then_some(row))
                        .collect();
                    n.push(count(rows.num_rows()));
                    d.push(count(held.len()));
                    z.push(count(present.iter().filter(|p| !**p).count()));
                }
                // Each row labelled as the dialects label it: the key's
                // label, where it gives one, else its position (T2.12).
                let positions = asked
                    .iter()
                    .enumerate()
                    .map(|(k, key)| i32::try_from(key.label_at(k)).map_err(arrow_failed))
                    .collect::<Result<Vec<_>, _>>()?;
                let schema = Schema::new(vec![
                    Field::new(results::KEY_POSITION, DataType::Int32, false),
                    Field::new(results::KEY_ROWS, DataType::Int64, false),
                    Field::new(results::KEY_DISTINCT, DataType::Int64, false),
                    Field::new(results::KEY_MISSING, DataType::Int64, false),
                ]);
                RecordBatch::try_new(
                    Arc::new(schema),
                    vec![
                        Arc::new(Int32Array::from(positions)),
                        Arc::new(Int64Array::from(n)),
                        Arc::new(Int64Array::from(d)),
                        Arc::new(Int64Array::from(z)),
                    ],
                )
                .map_err(arrow_failed)
            }
            DataRequest::RepeatedKey { table, key } => {
                let rows = project(&self.table(table)?.rows, key)?;
                let present = present(rows.columns());
                let mut seen: BTreeMap<OwnedRow, (u32, usize)> = BTreeMap::new();
                for (at, row) in row_values(rows.columns())?.into_iter().enumerate() {
                    if present[at] {
                        seen.entry(row).or_insert((position(at)?, 0)).1 += 1;
                    }
                }
                // The first value held more than once, in the rows' order.
                let repeated = seen
                    .into_values()
                    .filter(|(_, held)| *held > 1)
                    .map(|(at, _)| at)
                    .min();
                taken(&rows, &repeated.into_iter().collect::<Vec<_>>())
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
                let rows = &self.table(table)?.rows;
                let seeds = key_set(seeds)?;
                let mut pairs = joined(rows, key, seeds)?;
                if let Some(window) = window {
                    pairs = within(rows, seeds, window, pairs)?;
                }
                if let (Some(limit), Some(order)) = (per_seed, order) {
                    pairs = first_per_seed(rows, seeds, &order.0, *limit, pairs)?;
                }
                answered(rows, seeds, columns, &pairs, None)
            }
            DataRequest::Lookup {
                table,
                key,
                columns,
                seeds,
                ..
            } => {
                let rows = &self.table(table)?.rows;
                let seeds = key_set(seeds)?;
                let pairs = joined(rows, key, seeds)?;
                // One row per batch, its first match, with the count of
                // the batch's matches, in the order the batches first match.
                let batch = row_values(&[seed_column(seeds, SEED_BATCH)?])?;
                let mut first: Vec<(u32, u32)> = vec![];
                let mut matches: BTreeMap<&OwnedRow, usize> = BTreeMap::new();
                let mut counted: Vec<i64> = vec![];
                for &(seed, row) in &pairs {
                    let at = *matches.entry(&batch[seed as usize]).or_insert_with(|| {
                        first.push((seed, row));
                        counted.push(0);
                        first.len() - 1
                    });
                    counted[at] += 1;
                }
                answered(rows, seeds, columns, &first, Some(counted))
            }
            DataRequest::TimeBounds { columns } => {
                let (mut lows, mut highs) = (vec![], vec![]);
                for at in columns {
                    let rows = &self.table(&at.table)?.rows;
                    let times = wall_us(&named(rows, std::slice::from_ref(&at.column))?[0])?;
                    lows.push(times.iter().flatten().min().copied());
                    highs.push(times.iter().flatten().max().copied());
                }
                let positions = (0..columns.len())
                    .map(|k| i32::try_from(k).map_err(arrow_failed))
                    .collect::<Result<Vec<_>, _>>()?;
                let time = DataType::Timestamp(TimeUnit::Microsecond, None);
                let schema = Schema::new(vec![
                    Field::new(results::BOUNDS_POSITION, DataType::Int32, false),
                    Field::new(results::BOUNDS_MIN, time.clone(), true),
                    Field::new(results::BOUNDS_MAX, time, true),
                ]);
                RecordBatch::try_new(
                    Arc::new(schema),
                    vec![
                        Arc::new(Int32Array::from(positions)),
                        Arc::new(TimestampMicrosecondArray::from(lows)),
                        Arc::new(TimestampMicrosecondArray::from(highs)),
                    ],
                )
                .map_err(arrow_failed)
            }
            DataRequest::SampleEntities {
                table,
                n,
                seed,
                columns,
                filters,
                ..
            } => {
                let rows = project(&filtered(&self.table(table)?.rows, filters)?, columns)?;
                let seed = seed_integer(seed)
                    .and_then(|seed| u64::try_from(seed).ok())
                    .ok_or_else(not_read)?;
                let n = usize::try_from(*n).unwrap_or(usize::MAX);
                taken(&rows, &drawn(rows.num_rows(), n, seed)?)
            }
            DataRequest::ColumnRanges { .. } => Err(not_read()),
        }
    }

    /// One part of the catalog, in the contract's shape for a host
    /// provider (`catalog_columns`).
    fn describe(
        &self,
        tables: &[Relation],
        part: DescribePart,
    ) -> Result<RecordBatch, ProviderError> {
        let one = || match tables {
            [table] => self.table(table),
            _ => Err(not_read()),
        };
        let text = |name: &str, values: Vec<String>| {
            (
                Field::new(name, DataType::Utf8, false),
                Arc::new(StringArray::from(values)) as ArrayRef,
            )
        };
        let flags = |name: &str, values: Vec<bool>| {
            (
                Field::new(name, DataType::Boolean, false),
                Arc::new(BooleanArray::from(values)) as ArrayRef,
            )
        };
        let columns = match part {
            DescribePart::Tables => {
                // Every table, or each one named that the source holds.
                let held = self
                    .tables
                    .iter()
                    .filter(|t| {
                        tables.is_empty() || tables.iter().any(|r| r.single() == Some(&t.name))
                    })
                    .map(|t| t.name.clone())
                    .collect();
                vec![text(c::TABLE, held)]
            }
            DescribePart::Columns => {
                let declared = &one()?.columns;
                vec![
                    text(c::NAME, declared.iter().map(|c| c.name.clone()).collect()),
                    text(
                        c::TYPE,
                        declared.iter().map(|c| c.declared_type.clone()).collect(),
                    ),
                    flags(c::NOT_NULL, declared.iter().map(|c| c.not_null).collect()),
                    flags(
                        c::PRIMARY_KEY,
                        declared.iter().map(|c| c.primary_key).collect(),
                    ),
                ]
            }
            DescribePart::Keys => {
                let keys = &one()?.keys;
                vec![
                    text(c::KIND, keys.iter().map(|k| k.kind.clone()).collect()),
                    names(c::COLUMNS, keys.iter().map(|k| &k.columns)),
                ]
            }
            DescribePart::ForeignKeys => {
                let keys = &one()?.foreign_keys;
                vec![
                    (
                        Field::new(c::KEY, DataType::Int64, false),
                        Arc::new(Int64Array::from(
                            keys.iter().map(|k| k.key).collect::<Vec<_>>(),
                        )) as ArrayRef,
                    ),
                    text(
                        c::REFERENCED_TABLE,
                        keys.iter().map(|k| k.referenced_table.clone()).collect(),
                    ),
                    names(c::COLUMNS, keys.iter().map(|k| &k.columns)),
                    names(
                        c::REFERENCED_COLUMNS,
                        keys.iter().map(|k| &k.referenced_columns),
                    ),
                ]
            }
            // The source has no index.
            DescribePart::Indexes => vec![
                text(c::TABLE, vec![]),
                names(c::COLUMNS, std::iter::empty()),
            ],
            DescribePart::IndexColumns => vec![(
                Field::new(c::NAME, DataType::Utf8, true),
                Arc::new(StringArray::from(Vec::<Option<String>>::new())) as ArrayRef,
            )],
            // The source has no namespace: one row of none.
            DescribePart::Namespace => {
                let list = DataType::List(Arc::new(Field::new(c::LIST_ITEM, DataType::Utf8, true)));
                vec![(
                    Field::new(c::PARTS, list.clone(), false),
                    arrow_array::new_empty_array(&list),
                )]
            }
        };
        let (fields, arrays): (Vec<Field>, Vec<ArrayRef>) = columns.into_iter().unzip();
        RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays).map_err(arrow_failed)
    }

    /// A copy of the tables of `connection`'s current schema, in the order
    /// DuckDB's catalog lists them, each with its rows as DuckDB's driver
    /// hands them over and its catalog as DuckDB declares it (a key's
    /// kind read as `sdm-data` reads DuckDB's: a primary key, a unique
    /// key, or another kind). Read once, to build a fixture.
    pub fn from_duckdb(connection: &duckdb::Connection) -> Result<Memory> {
        const HERE: &str = "database_name = current_database() AND schema_name = current_schema()";
        let quoted = |name: &str| format!("'{}'", name.replace('\'', "''"));
        let names: Vec<String> = connection
            .prepare(&format!(
                "SELECT table_name FROM duckdb_tables() WHERE NOT temporary AND NOT internal AND {HERE}"
            ))?
            .query_map([], |row| row.get(0))?
            .collect::<Result<_, _>>()?;
        let mut tables = Vec::with_capacity(names.len());
        for name in names {
            let read = crate::duck::run(
                connection,
                &format!("SELECT * FROM \"{}\"", name.replace('"', "\"\"")),
            )
            .with_context(|| format!("reading the rows of {name}"))?;
            let rows = concat_batches(&read.schema, &read.batches)?;
            // Each constraint: its type, its columns, and, for a foreign
            // key, its number and what it refers to (lists as JSON text).
            let constraints: Vec<Constraint> = connection
                .prepare(&format!(
                    "SELECT constraint_type, CAST(to_json(constraint_column_names) AS VARCHAR),
                                CAST(constraint_index AS BIGINT), referenced_table,
                                CAST(to_json(referenced_column_names) AS VARCHAR)
                         FROM duckdb_constraints() WHERE table_name = {} AND {HERE}",
                    quoted(&name)
                ))?
                .query_map([], |row| {
                    Ok((
                        row.get(0)?,
                        row.get(1)?,
                        row.get(2)?,
                        row.get(3)?,
                        row.get(4)?,
                    ))
                })?
                .collect::<Result<_, _>>()?;
            let list = |json: &str| -> Result<Vec<String>> {
                serde_json::from_str(json)
                    .map_err(|e| anyhow!("{name}: a constraint's columns: {e}"))
            };
            let mut keys = Vec::new();
            let mut foreign_keys = Vec::new();
            let mut primary: Vec<String> = Vec::new();
            for (kind, columns, index, referenced, referenced_columns) in constraints {
                let columns = list(&columns)?;
                if kind == "PRIMARY KEY" {
                    primary.extend(columns.iter().cloned());
                }
                if kind == "FOREIGN KEY" {
                    foreign_keys.push(ForeignKey {
                        key: index,
                        referenced_table: referenced.unwrap_or_default(),
                        columns: columns.clone(),
                        referenced_columns: list(referenced_columns.as_deref().unwrap_or("[]"))?,
                    });
                }
                keys.push(Key {
                    kind: match kind.as_str() {
                        "PRIMARY KEY" => c::KIND_PRIMARY,
                        "UNIQUE" => c::KIND_UNIQUE,
                        _ => "other",
                    }
                    .to_owned(),
                    columns,
                });
            }
            let columns: Vec<Column> = connection
                .prepare(&format!(
                    "SELECT column_name, data_type, NOT is_nullable FROM duckdb_columns()
                     WHERE table_name = {} AND {HERE} ORDER BY column_index",
                    quoted(&name)
                ))?
                .query_map([], |row| {
                    Ok((row.get::<_, String>(0)?, row.get(1)?, row.get(2)?))
                })?
                .map(|row| {
                    row.map(|(name, declared_type, not_null)| Column {
                        primary_key: primary.contains(&name),
                        name,
                        declared_type,
                        not_null,
                    })
                })
                .collect::<Result<_, _>>()?;
            tables.push(Table {
                name,
                rows,
                columns,
                keys,
                foreign_keys,
            });
        }
        Ok(Memory { tables })
    }
}

/// A list-of-names column of the catalog's answer, one list per row.
fn names<'a>(name: &str, rows: impl Iterator<Item = &'a Vec<String>>) -> (Field, ArrayRef) {
    let mut lists = ListBuilder::new(StringBuilder::new());
    for row in rows {
        for item in row {
            lists.values().append_value(item);
        }
        lists.append(true);
    }
    let lists = lists.finish();
    (
        Field::new(name, lists.data_type().clone(), false),
        Arc::new(lists),
    )
}

/// A batch of non-null `int64` count columns, one row.
fn counts(columns: &[(&str, Vec<i64>)]) -> Result<RecordBatch, ProviderError> {
    let fields: Vec<Field> = columns
        .iter()
        .map(|(name, _)| Field::new(*name, DataType::Int64, false))
        .collect();
    let arrays: Vec<ArrayRef> = columns
        .iter()
        .map(|(_, values)| Arc::new(Int64Array::from(values.clone())) as ArrayRef)
        .collect();
    RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays).map_err(arrow_failed)
}

fn count(n: usize) -> i64 {
    i64::try_from(n).unwrap_or(i64::MAX)
}

fn position(at: usize) -> Result<u32, ProviderError> {
    u32::try_from(at).map_err(arrow_failed)
}

/// The first `limit` rows of `rows`.
fn first(rows: &RecordBatch, limit: u64) -> RecordBatch {
    let limit = usize::try_from(limit).unwrap_or(usize::MAX);
    rows.slice(0, limit.min(rows.num_rows()))
}

/// The columns `names` of `rows`, in that order; a name it does not hold
/// is the source's failure.
fn project(rows: &RecordBatch, names: &[String]) -> Result<RecordBatch, ProviderError> {
    let schema = rows.schema();
    let mut fields = Vec::with_capacity(names.len());
    let mut arrays = Vec::with_capacity(names.len());
    for name in names {
        let (at, field) = schema.column_with_name(name).ok_or_else(not_held)?;
        fields.push(field.clone());
        arrays.push(rows.column(at).clone());
    }
    RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays).map_err(arrow_failed)
}

/// The columns `names` of `rows`.
fn named(rows: &RecordBatch, names: &[String]) -> Result<Vec<ArrayRef>, ProviderError> {
    Ok(project(rows, names)?.columns().to_vec())
}

/// The rows `at` of `rows`, in that order.
fn taken(rows: &RecordBatch, at: &[u32]) -> Result<RecordBatch, ProviderError> {
    let at = UInt32Array::from(at.to_vec());
    let arrays = rows
        .columns()
        .iter()
        .map(|column| take(column, &at, None).map_err(arrow_failed))
        .collect::<Result<Vec<_>, _>>()?;
    RecordBatch::try_new(rows.schema(), arrays).map_err(arrow_failed)
}

/// The rows of `rows` meeting every filter: each a not-null filter, the
/// one kind the core writes (as the SQL dialects take only it); another
/// kind is not read.
fn filtered(rows: &RecordBatch, filters: &[Filter]) -> Result<RecordBatch, ProviderError> {
    let mut kept = vec![true; rows.num_rows()];
    for filter in filters {
        let Filter::NotNull { column } = filter else {
            return Err(not_read());
        };
        let column = &named(rows, std::slice::from_ref(column))?[0];
        for (at, kept) in kept.iter_mut().enumerate() {
            *kept &= column.is_valid(at);
        }
    }
    filter_record_batch(rows, &BooleanArray::from(kept)).map_err(arrow_failed)
}

/// `n` of `rows` row positions drawn under `seed`, in the rows' order, the
/// same for the same seed and rows: every row where `n` covers them, as a
/// reservoir of `n` keeps them (DuckDB's `reservoir`); otherwise the first
/// `n` of the positions shuffled by a Fisher-Yates pass over SplitMix64's
/// stream, this provider's own draw, stable under a seed as it declares
/// (no other provider's draw of fewer rows than the table holds is
/// matched).
fn drawn(rows: usize, n: usize, seed: u64) -> Result<Vec<u32>, ProviderError> {
    let mut state = seed;
    let mut next = || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let mut at = (0..rows).map(position).collect::<Result<Vec<_>, _>>()?;
    if n >= rows {
        return Ok(at);
    }
    for i in (1..rows).rev() {
        let j = usize::try_from(next() % (i as u64 + 1)).map_err(arrow_failed)?;
        at.swap(i, j);
    }
    at.truncate(n);
    at.sort_unstable();
    Ok(at)
}

/// Whether `time` is within `bound`, a lower or an upper one.
fn holds(time: i128, bound: i128, lower: bool, inclusive: bool) -> bool {
    match (lower, inclusive) {
        (true, true) => time >= bound,
        (true, false) => time > bound,
        (false, true) => time <= bound,
        (false, false) => time < bound,
    }
}

/// `array`'s values in microseconds, as `CAST(column AS TIMESTAMP)` reads
/// them: a zone-less timestamp of any unit (a finer one truncated toward
/// zero) or a date at its midnight; any other type is not read.
fn wall_us(array: &ArrayRef) -> Result<Vec<Option<i64>>, ProviderError> {
    let scaled = |values: Vec<Option<i64>>, per: i64| {
        values
            .into_iter()
            .map(|v| {
                v.map(|v| v.checked_mul(per).ok_or_else(not_read))
                    .transpose()
            })
            .collect::<Result<Vec<_>, _>>()
    };
    match array.data_type() {
        DataType::Timestamp(TimeUnit::Second, None) => scaled(
            array.as_primitive::<TimestampSecondType>().iter().collect(),
            1_000_000,
        ),
        DataType::Timestamp(TimeUnit::Millisecond, None) => scaled(
            array
                .as_primitive::<TimestampMillisecondType>()
                .iter()
                .collect(),
            1_000,
        ),
        DataType::Timestamp(TimeUnit::Microsecond, None) => Ok(array
            .as_primitive::<TimestampMicrosecondType>()
            .iter()
            .collect()),
        DataType::Timestamp(TimeUnit::Nanosecond, None) => Ok(array
            .as_primitive::<TimestampNanosecondType>()
            .iter()
            .map(|v| v.map(|v| v / 1_000))
            .collect()),
        DataType::Date32 => scaled(
            array
                .as_primitive::<Date32Type>()
                .iter()
                .map(|v| v.map(i64::from))
                .collect(),
            86_400_000_000,
        ),
        DataType::Date64 => scaled(array.as_primitive::<Date64Type>().iter().collect(), 1_000),
        _ => Err(not_read()),
    }
}

/// [`wall_us`] in nanoseconds, to compare with a seed's `timestamp[ns]`.
fn wall_ns(array: &ArrayRef) -> Result<Vec<Option<i128>>, ProviderError> {
    Ok(wall_us(array)?
        .into_iter()
        .map(|v| v.map(|v| i128::from(v) * 1_000))
        .collect())
}

/// Whether each row holds every one of `arrays`' values.
fn present(arrays: &[ArrayRef]) -> Vec<bool> {
    let rows = arrays.first().map_or(0, |a| a.len());
    (0..rows)
        .map(|at| arrays.iter().all(|a| a.is_valid(at)))
        .collect()
}

/// `array` with the floats SQL holds equal written alike: `-0.0` as `0.0`,
/// every NaN as one.
fn compared(array: &ArrayRef) -> ArrayRef {
    match array.data_type() {
        DataType::Float64 => Arc::new(
            array
                .as_primitive::<Float64Type>()
                .unary::<_, Float64Type>(|v| if v.is_nan() { f64::NAN } else { v + 0.0 }),
        ),
        DataType::Float32 => Arc::new(
            array
                .as_primitive::<Float32Type>()
                .unary::<_, Float32Type>(|v| if v.is_nan() { f32::NAN } else { v + 0.0 }),
        ),
        _ => array.clone(),
    }
}

/// Each row of `arrays` as one comparable value, the floats SQL holds
/// equal alike.
fn row_values(arrays: &[ArrayRef]) -> Result<Vec<OwnedRow>, ProviderError> {
    let arrays: Vec<ArrayRef> = arrays.iter().map(compared).collect();
    let converter = RowConverter::new(
        arrays
            .iter()
            .map(|a| SortField::new(a.data_type().clone()))
            .collect(),
    )
    .map_err(|_| not_read())?;
    let rows = converter.convert_columns(&arrays).map_err(|_| not_read())?;
    Ok(rows.iter().map(|row| row.owned()).collect())
}

/// `array`'s integers as `decimal128(38, 0)`, so integers of any width and
/// decimals of scale 0 compare by value; `None` for any other type.
fn exact_integers(array: &ArrayRef) -> Option<ArrayRef> {
    fn widened<T: arrow_array::ArrowPrimitiveType>(array: &ArrayRef) -> Vec<Option<i128>>
    where
        T::Native: Into<i128>,
    {
        array
            .as_primitive::<T>()
            .iter()
            .map(|v| v.map(Into::into))
            .collect()
    }
    let values = match array.data_type() {
        DataType::Int8 => widened::<Int8Type>(array),
        DataType::Int16 => widened::<Int16Type>(array),
        DataType::Int32 => widened::<Int32Type>(array),
        DataType::Int64 => widened::<Int64Type>(array),
        DataType::UInt8 => widened::<UInt8Type>(array),
        DataType::UInt16 => widened::<UInt16Type>(array),
        DataType::UInt32 => widened::<UInt32Type>(array),
        DataType::UInt64 => widened::<UInt64Type>(array),
        DataType::Decimal128(_, 0) => array.as_primitive::<Decimal128Type>().iter().collect(),
        _ => return None,
    };
    Decimal128Array::from(values)
        .with_precision_and_scale(38, 0)
        .ok()
        .map(|a| Arc::new(a) as ArrayRef)
}

/// The pairs `(key row, row)` whose values in `columns` equal the key
/// set's `keys`, column by column, every part present: in the key set's
/// order, then the rows'. A pair of columns of types that do not compare
/// by value is not read.
fn pairs(columns: &[ArrayRef], keys: &[ArrayRef]) -> Result<Vec<(u32, u32)>, ProviderError> {
    if columns.len() != keys.len() {
        return Err(not_read());
    }
    let mut left = Vec::with_capacity(columns.len());
    let mut right = Vec::with_capacity(keys.len());
    for (column, key) in columns.iter().zip(keys) {
        match (exact_integers(column), exact_integers(key)) {
            (Some(column), Some(key)) => {
                left.push(column);
                right.push(key);
            }
            _ if column.data_type() == key.data_type() => {
                left.push(column.clone());
                right.push(key.clone());
            }
            _ => return Err(not_read()),
        }
    }
    let mut held: BTreeMap<OwnedRow, Vec<u32>> = BTreeMap::new();
    let present_rows = present(&left);
    for (at, row) in row_values(&left)?.into_iter().enumerate() {
        if present_rows[at] {
            held.entry(row).or_default().push(position(at)?);
        }
    }
    let present_keys = present(&right);
    let mut found = vec![];
    for (at, key) in row_values(&right)?.into_iter().enumerate() {
        if present_keys[at]
            && let Some(rows) = held.get(&key)
        {
            let at = position(at)?;
            found.extend(rows.iter().map(|&row| (at, row)));
        }
    }
    Ok(found)
}

/// The seed column `name`.
fn seed_column(seeds: &RecordBatch, name: &str) -> Result<ArrayRef, ProviderError> {
    seeds.column_by_name(name).cloned().ok_or_else(not_read)
}

/// The pairs `(seed, row)` of the rows of `rows` whose `key` is a seed's.
fn joined(
    rows: &RecordBatch,
    key: &[String],
    seeds: &RecordBatch,
) -> Result<Vec<(u32, u32)>, ProviderError> {
    let seed_keys = seed_key_columns(key.len())
        .iter()
        .map(|name| seed_column(seeds, name))
        .collect::<Result<Vec<_>, _>>()?;
    let mut columns = named(rows, key)?;
    let mut seed_keys = seed_keys;
    // A column of numbers handed over as their exact text compares by the
    // number each text writes, as Postgres compares a `numeric`: `1.0` is
    // `1.00` (`sdm_contract::data::exact_number_text`).
    for (at, name) in key.iter().enumerate() {
        let marked = rows.schema().field_with_name(name).is_ok_and(|f| {
            f.metadata().get(EXACT_NUMBER_MARKER).map(String::as_str) == Some(EXACT_NUMBER_MARKED)
        });
        if marked {
            columns[at] = exact_numbers(&columns[at]);
            seed_keys[at] = exact_numbers(&seed_keys[at]);
        }
    }
    pairs(&columns, &seed_keys)
}

/// Text `array`'s values each as the one text of the number it writes; an
/// array of any other type as it is.
fn exact_numbers(array: &ArrayRef) -> ArrayRef {
    if array.data_type() != &DataType::Utf8 {
        return Arc::clone(array);
    }
    Arc::new(
        array
            .as_string::<i32>()
            .iter()
            .map(|v| v.map(sdm_contract::data::exact_number_text))
            .collect::<StringArray>(),
    )
}

/// The pairs whose row's window column is within the seed's bounds; a row
/// with no time is not.
fn within(
    rows: &RecordBatch,
    seeds: &RecordBatch,
    window: &Window,
    pairs: Vec<(u32, u32)>,
) -> Result<Vec<(u32, u32)>, ProviderError> {
    let times = wall_ns(&named(rows, std::slice::from_ref(&window.column))?[0])?;
    let mut bounds = vec![];
    for (bound, lower) in [(&window.lower, true), (&window.upper, false)] {
        let Some(bound) = bound else { continue };
        let column = seed_column(seeds, &bound.seed_column)?;
        let DataType::Timestamp(TimeUnit::Nanosecond, None) = column.data_type() else {
            return Err(not_read());
        };
        let values: Vec<Option<i128>> = column
            .as_primitive::<TimestampNanosecondType>()
            .iter()
            .map(|v| v.map(i128::from))
            .collect();
        bounds.push((values, lower, bound.inclusive));
    }
    Ok(pairs
        .into_iter()
        .filter(|&(seed, row)| {
            times[row as usize].is_some_and(|time| {
                bounds.iter().all(|(values, lower, inclusive)| {
                    values[seed as usize]
                        .is_some_and(|bound| holds(time, bound, *lower, *inclusive))
                })
            })
        })
        .collect())
}

/// At most `limit` pairs of each batch, the first in `terms`' order, by
/// batch then that order; equal pairs keep the join's order.
fn first_per_seed(
    rows: &RecordBatch,
    seeds: &RecordBatch,
    terms: &[TieTerm],
    limit: u64,
    pairs: Vec<(u32, u32)>,
) -> Result<Vec<(u32, u32)>, ProviderError> {
    let seed_at = UInt32Array::from(pairs.iter().map(|&(seed, _)| seed).collect::<Vec<_>>());
    let row_at = UInt32Array::from(pairs.iter().map(|&(_, row)| row).collect::<Vec<_>>());
    let batch = take(&seed_column(seeds, SEED_BATCH)?, &seed_at, None).map_err(arrow_failed)?;
    let mut fields = vec![SortField::new(batch.data_type().clone())];
    let mut sorted = vec![batch.clone()];
    for term in terms {
        let column = named(rows, std::slice::from_ref(&term.column))?.remove(0);
        let values: ArrayRef = if term.as_timestamp {
            Arc::new(TimestampMicrosecondArray::from(wall_us(&column)?))
        } else {
            compared(&column)
        };
        let values = take(&values, &row_at, None).map_err(arrow_failed)?;
        fields.push(SortField::new_with_options(
            values.data_type().clone(),
            SortOptions {
                descending: term.descending,
                nulls_first: !term.nulls_last,
            },
        ));
        sorted.push(values);
    }
    let order_rows = RowConverter::new(fields)
        .and_then(|converter| converter.convert_columns(&sorted))
        .map_err(|_| not_read())?;
    let batch_rows = row_values(&[batch])?;
    let mut order: Vec<usize> = (0..pairs.len()).collect();
    order.sort_by(|&a, &b| order_rows.row(a).cmp(&order_rows.row(b)));
    let mut kept = vec![];
    let mut run: Option<(usize, u64)> = None;
    for at in order {
        let count = match run {
            Some((last, count)) if batch_rows[last] == batch_rows[at] => count + 1,
            _ => 1,
        };
        run = Some((at, count));
        if count <= limit {
            kept.push(pairs[at]);
        }
    }
    Ok(kept)
}

/// The answer of a `neighbors` or `lookup`: each pair's seed batch
/// position ([`BATCH_COLUMN`]), then, for a lookup, each batch's count of
/// matches ([`MATCHES_COLUMN`]), then the row's `columns`.
fn answered(
    rows: &RecordBatch,
    seeds: &RecordBatch,
    columns: &[String],
    pairs: &[(u32, u32)],
    matches: Option<Vec<i64>>,
) -> Result<RecordBatch, ProviderError> {
    let seed_at = UInt32Array::from(pairs.iter().map(|&(seed, _)| seed).collect::<Vec<_>>());
    let row_at: Vec<u32> = pairs.iter().map(|&(_, row)| row).collect();
    let batch = take(&seed_column(seeds, SEED_BATCH)?, &seed_at, None).map_err(arrow_failed)?;
    let mut fields = vec![Field::new(BATCH_COLUMN, batch.data_type().clone(), false)];
    let mut arrays = vec![batch];
    if let Some(matches) = matches {
        fields.push(Field::new(MATCHES_COLUMN, DataType::Int64, false));
        arrays.push(Arc::new(Int64Array::from(matches)));
    }
    let projected = taken(&project(rows, columns)?, &row_at)?;
    fields.extend(
        projected
            .schema()
            .fields()
            .iter()
            .map(|f| f.as_ref().clone()),
    );
    arrays.extend(projected.columns().iter().cloned());
    RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays).map_err(arrow_failed)
}
