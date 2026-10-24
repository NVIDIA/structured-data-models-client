// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The Postgres dialect (T2.3). Its capabilities are the optional requests
//! it has a rendering for, every one the core asks, so none falls back.
//!
//! Each rendering is the statement the reference's Postgres engine runs for
//! the same step (`rfm/backend/postgres`), so each statement's result is
//! that statement's (decision H9 gates on results, not text): the table
//! list, a table's columns, keys and foreign keys (`information_schema`, as
//! `PostgresTable` reads them), the preview, the key counts, the column
//! ranges, the repeated key, the time bounds (`MIN` and `MAX` in the
//! column's own type, which the core reads after) and the entity sample are
//! the reference's text but for the relation, which is the core's (design
//! choice 12): the reference binds the schema it read with
//! `current_schema()`, a read whose result the core never needs (KO2-58),
//! and qualifies its `FROM` with it, where the core names the schema in
//! place. A key lookup and a neighbor read return the reference's columns
//! (the seed's batch position, the row's columns, its number among its
//! seed's, `__kumo_row__`, and for a lookup the count of rows holding the
//! key), their seeds unpacked as below.
//!
//! The catalog answers are the reference's, read by [`read`] into the
//! contract's catalog answer. A relation is found as the reference finds a
//! table: a one-part name in `current_schema()` alone, two parts by schema
//! and table, a view included; a source given its schema (the provider's
//! namespace, contract v28) has each one-part name read as two parts under
//! it and its tables listed in it, as the reference reads the schema it
//! was given; a name the source does not hold there
//! answers no columns, which the core refuses as the reference refuses a
//! table it cannot find. A one-part `FROM` reads that table unless the
//! search path finds a temporary or `pg_catalog` relation of the name
//! first; such a name also answers no columns ([`regclass`]), so every
//! statement of a run reads the relation the reference reads, or none.
//!
//! A key set is bound as the reference's Postgres sampler binds one: JSON
//! text in the statement's one parameter, `$1`, unpacked by a common table
//! expression named as the plan's relation ([`SEEDS`]; `P1 d:131`). The
//! dialect writes it as the reference writes it (`pg_values::key_rows`,
//! T3.28): each seed its key alone, or an object of `id` and the window's
//! bounds `e` and `s`, its batch its position (`WITH ORDINALITY`), split
//! past the reference's budget as it splits one ([`crate::key_text`]),
//! each later chunk's statement counting its batches from the seeds before
//! it; the provider binds the text it is handed. Postgres's own input
//! functions read each value ([`seed_cte`]): an integer or decimal key as
//! `bigint` or `numeric`, so the join compares the two keys by value,
//! exactly (an integer and a `numeric` meet when their values are equal,
//! whatever their scales, and two `numeric` values float64 would merge
//! stay apart; design choice 5, lesson 19, KO2-35); any other key (a
//! float, a text) in the joined column's own declared type, put in place
//! of that column in one row of the table (`jsonb_populate_record`), so a
//! `real` key compares as a `real` and a `uuid`, an enum or a `varchar` key
//! as itself (the reference casts to the type it read from the catalog);
//! the batch as `bigint`; a seed time in the window column's own type, the
//! column compared in that type and ordered by it, as the reference does
//! (a text time column by its text, the seed's time in the text pandas
//! writes it).
//!
//! The statements are built once, by the shared builder ([`crate::sql`]),
//! through [`Postgres`]'s hooks (T3.18): only what is Postgres's own is
//! here.

use std::collections::BTreeMap;

use arrow_array::cast::AsArray;
use arrow_array::types::Int32Type;
use arrow_array::{Array, Int32Array, RecordBatch, StringArray};
use sdm_contract::catalog::{CatalogAnswer, CatalogColumn, CatalogForeignKey, CatalogKey, KeyKind};
use sdm_contract::data::{
    BATCH_COLUMN, DataRequest, DescribePart, Determinism, ExpectedSchema, MATCHES_COLUMN,
    NullOrder, ProviderCapabilities, Relation, SEED_BATCH, TextTimes, TieTerm,
    catalog_columns as c, field, seed_integer, seed_key_columns, views_unread,
};
use sdm_contract::types::ColumnType;
use sdm_contract::{CoreError, Detail, Scalar, Sentence};

use crate::key_text;
use crate::pg_values;
use crate::quote::identifier;
use crate::sql::{self, Sample, SeedRead, SqlDialect};
use sdm_contract::data::SqlParam;

/// The dialect's name on the wire.
pub(crate) const NAME: &str = "postgres";

/// The name a statement reads its seeds by: the common table expression
/// the key set's JSON is unpacked into. A CTE's name shadows a table of
/// the same name only where the statement names it unqualified, and the
/// joined table is always named by its relation, so no name is lengthened
/// here; [`seeds_for`] still steps past a one-part relation of that name,
/// which would otherwise read the seeds.
const SEEDS: &str = "__sdm_seeds__";

/// The column of a neighbor read with a limit per seed, and of a key
/// lookup, numbering each row among its seed's: the reference's own name
/// (`__KUMO_ROW__`, which Postgres folds to lower case).
const ROW_COLUMN: &str = "__kumo_row__";

/// The CTE a statement joining `table` reads its seeds by: [`SEEDS`],
/// lengthened by underscores until it is not `table`'s one part, so a
/// source table of any name is read, never shadowed by the seeds.
fn seeds_for(table: &Relation) -> String {
    let mut name = SEEDS.to_owned();
    if let Some(single) = table.single() {
        while name == single {
            name.push('_');
        }
    }
    name
}

/// The most name parts a Postgres relation takes: a schema and a table
/// (design choice 12).
const MOST_PARTS: usize = 2;

/// The `FROM` target of `relation`, through the data layer's one rendering
/// of a relation ([`sql::from`]).
fn from(request: &DataRequest, field: &str, relation: &Relation) -> Result<String, CoreError> {
    sql::from(&Postgres, request, field, relation)
}

/// The Postgres dialect.
pub(crate) struct Postgres;

impl SqlDialect for Postgres {
    fn name(&self) -> &'static str {
        NAME
    }

    /// What a Postgres source answers through this layer.
    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            compares_stored_times: false,
            opaque_vendor: None,
            samples_by_key_dtype: false,
            derived_relations: false,
            neighbor_lookback: false,
            keys_optional: false,
            sample_rows: false,
            optional: sql::rendered(),
            max_keys_per_request: None,
            // `setseed` before `ORDER BY random()` repeats the draw on one
            // server (register entry RS-6, KO2-24).
            sampling: Determinism::SeededStable,
            // The reference hashes the seed's text into `setseed`'s range, so
            // it takes any integer seed (`_postgres_seed`).
            max_seed: u64::MAX,
            // The reference's engine reads a SQL source's text times one by
            // one (register DT-7).
            text_times: Some(TextTimes::ValueByValue),
            // Postgres orders a null after every value ascending, the order
            // every term the core asks with nulls among its values has: the
            // reference's neighbor order is the time column descending, then
            // the key (or each other column) ascending, and the time column's
            // nulls never reach the order, since the window that comes with a
            // time column leaves them out (register ORD-2). Each term is still
            // rendered with the null order the core writes into it.
            null_order: Some(NullOrder::Last),
            // The reference's Postgres sampler neither reads the indexes nor
            // warns of missing ones.
            index_advice: None,
            // Every optional request is rendered: none falls back.
            fallback: None,
            // Postgres's driver reads a NaN as missing.
            reads_nan_as_missing: true,
        }
    }

    /// A schema and a table, each quoted alone, never split on a dot.
    fn most_parts(&self) -> usize {
        MOST_PARTS
    }

    /// A Postgres source given its schema is read under it (contract v28):
    /// one part (design fix C).
    fn namespace_depth(&self) -> usize {
        1
    }

    fn seeds_for(&self, _request: &DataRequest, table: &Relation) -> String {
        seeds_for(table)
    }

    /// The common table expression the key set's JSON is unpacked into
    /// ([`seed_cte`]).
    fn seeds_prelude(&self, read: &SeedRead<'_>) -> Result<String, CoreError> {
        seed_cte(read)
    }

    /// `MIN` and `MAX` of the column in its own type, as the reference's
    /// `_get_min_max_time_dict` writes them; the core reads each bound of
    /// whatever time type as the reference's host reads it (a `timestamptz`
    /// at its UTC instant, a date at its midnight, a text by the provider's
    /// reading), so the plan expects whatever schema the source answers
    /// ([`expected`]).
    fn time_bounds_select(&self, position: usize, time: &str, table: &str) -> String {
        format!(
            "SELECT {position} AS table_index, MIN({time}) AS min_date, MAX({time}) AS max_date\nFROM {table}"
        )
    }

    /// Any integer: the reference hashes its text (`_postgres_seed`).
    fn sample_seed(&self, request: &DataRequest, seed: &Scalar) -> Result<i128, CoreError> {
        seed_integer(seed).ok_or_else(|| {
            sdm_contract::data::refused(request, "seed", "a sample's seed is an integer")
        })
    }

    /// The reference's sample of an entity table on Postgres
    /// (`_sample_entity_table`): `setseed` with the seed's place in its
    /// range ([`seed_value`]), then the columns asked for, from the rows that
    /// meet each not-null filter, in `random()` order, the first `n`, in its
    /// text (register entry RS-6, KO2-24).
    fn sample(&self, sample: &Sample<'_>) -> Vec<String> {
        vec![
            format!("SELECT setseed({})", float_text(seed_value(sample.seed))),
            format!(
                "SELECT {}\nFROM {}{}\nORDER BY random()\nLIMIT {}",
                sample.projected, sample.table, sample.filtered, sample.n
            ),
        ]
    }

    /// The reference's neighbor reads (`_by_time` and `_by_fkey` on
    /// Postgres) with a limit per seed: the first rows of each seed in
    /// `order`, read for each seed (`JOIN LATERAL`, as the reference reads
    /// them), the result in `(batch, order)`. The window's column is
    /// compared in its own type with the seed's time read in that type
    /// ([`seed_cte`]), as the reference compares them.
    fn neighbors_per_seed(&self, read: &SeedRead<'_>, limit: u64, terms: &[TieTerm]) -> String {
        let positioned = positioned(read.columns);
        let order = terms
            .iter()
            .map(|t| self.order_term(t, "t.", None))
            .collect::<Vec<_>>()
            .join(", ");
        let inner = positioned
            .iter()
            .map(|(alias, column)| format!("t.{} AS {alias}", identifier(column)))
            .collect::<Vec<_>>()
            .join(", ");
        let outer = positioned
            .iter()
            .map(|(alias, column)| format!("n.{alias} AS {}", identifier(column)))
            .collect::<Vec<_>>()
            .join(", ");
        format!(
            "SELECT {batch}, {outer}, n.r AS {row}\nFROM {seeds} s\nCROSS JOIN LATERAL (\n  SELECT {inner}, ROW_NUMBER() OVER (ORDER BY {order}) AS r\n  FROM {table} t\n  WHERE {within}\n  ORDER BY {order}\n  LIMIT {limit}\n) AS n\nORDER BY s.{batch_column}, n.r",
            batch = read.batch(),
            seeds = read.seeds(),
            table = read.table,
            within = read.within("t.").join("\n    AND "),
            batch_column = identifier(SEED_BATCH),
            row = identifier(ROW_COLUMN),
        )
    }

    /// A limit per seed comes with an order, and an order with a limit.
    fn neighbors_unpaired(&self, read: &SeedRead<'_>) -> Result<String, CoreError> {
        Err(sdm_contract::data::refused(
            read.request,
            "order",
            "a limit per seed comes with an order, and an order with a limit",
        ))
    }

    /// The reference's key lookup (`_by_pkey` on Postgres): one row of the
    /// table per seed whose key it holds, its columns as the reference's
    /// result has them, the seed's batch position, the row's columns, its
    /// number among the rows holding the key (1) and their count, the result
    /// in batch order.
    fn lookup(&self, read: &SeedRead<'_>) -> String {
        let positioned = positioned(read.columns);
        let batch = identifier(SEED_BATCH);
        let inner = positioned
            .iter()
            .map(|(alias, column)| format!("t.{} AS {alias}", identifier(column)))
            .collect::<Vec<_>>()
            .join(", ");
        let outer = positioned
            .iter()
            .map(|(alias, column)| format!("l.{alias} AS {}", identifier(column)))
            .collect::<Vec<_>>()
            .join(", ");
        format!(
            "SELECT l.b AS {helper}, {outer}, l.r AS {row}, l.m AS {matches}\nFROM (\n  SELECT s.{batch} AS b, {inner},\n         ROW_NUMBER() OVER (PARTITION BY s.{batch}\n           ORDER BY {keyed}) AS r,\n         COUNT(*) OVER (PARTITION BY s.{batch}) AS m\n  FROM {seeds} s\n  JOIN {table} t\n    ON {on}\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
            helper = identifier(BATCH_COLUMN),
            matches = identifier(MATCHES_COLUMN),
            row = identifier(ROW_COLUMN),
            keyed = read
                .key
                .iter()
                .map(|k| format!("t.{}", identifier(k)))
                .collect::<Vec<_>>()
                .join(", "),
            seeds = read.seeds(),
            table = read.table,
            on = read.seed_join("t.", "="),
        )
    }

    fn describe(
        &self,
        request: &DataRequest,
        tables: &[Relation],
        part: DescribePart,
        namespace: &[String],
    ) -> Result<(String, Vec<SqlParam>), CoreError> {
        Ok((describe(request, tables, part, namespace)?, vec![]))
    }

    fn expected(&self, request: &DataRequest) -> ExpectedSchema {
        expected(request)
    }

    fn read(
        &self,
        _request: &DataRequest,
        part: DescribePart,
        batches: &[RecordBatch],
    ) -> Result<CatalogAnswer, CoreError> {
        read(part, batches)
    }

    fn conform(
        &self,
        schema: &arrow_schema::SchemaRef,
        batches: &[RecordBatch],
        float_text: fn(f64) -> String,
    ) -> Result<Option<sql::Batches>, CoreError> {
        pg_values::conform(schema, batches, float_text)
    }

    fn key_rows(
        &self,
        request: &DataRequest,
        schema: &arrow_schema::SchemaRef,
        batches: &[RecordBatch],
        float_text: fn(f64) -> String,
    ) -> Result<Option<Vec<String>>, CoreError> {
        pg_values::key_rows(request, schema, batches, float_text).map(Some)
    }

    /// The reference's budget of a key set's JSON text (`_chunk_rows`,
    /// `PG/sampler.py:125-182`); each row carries its batch, so a chunk's
    /// statement is the plan's own.
    fn key_budget(&self) -> Option<usize> {
        Some(key_text::BUDGET)
    }

    /// The statement with its seeds' batches counted from `offset`, the
    /// seeds before its chunk, as the reference's `_flatten` moves a
    /// chunk's batches.
    fn chunk_text(&self, text: &str, offset: usize) -> Result<String, CoreError> {
        if offset == 0 {
            return Ok(text.to_owned());
        }
        let at = batch_select(0);
        if text.matches(&at).count() != 1 {
            return Err(CoreError::new(
                Sentence::Internal,
                "a statement binding a key set does not number its seeds once",
            ));
        }
        Ok(text.replacen(&at, &batch_select(offset), 1))
    }
}

/// The catalog's object id of `relation`, found as the reference finds a
/// table (`PostgresTable`, its catalog read by schema and table name): a
/// one-part name in `current_schema()` alone, two parts the schema and the
/// table so named, each compared as written. Null where the source holds no
/// such relation, and where the relation the same name reads in a `FROM`
/// (`to_regclass` over the quoted parts) is another: a temporary or
/// `pg_catalog` relation of a one-part name, which the search path finds
/// first, so no statement of the run reads a relation the reference does
/// not.
fn regclass(request: &DataRequest, field: &str, relation: &Relation) -> Result<String, CoreError> {
    let read = text(&from(request, field, relation)?);
    let (schema, table) = match relation.source.as_slice() {
        [table] => ("current_schema()".to_owned(), text(table)),
        [schema, table] => (text(schema), text(table)),
        _ => {
            return Err(CoreError::new(
                Sentence::Internal,
                "a relation past the parts the dialect takes reached its catalog read",
            ));
        }
    };
    Ok(format!(
        "(SELECT c.oid FROM pg_catalog.pg_class c\n         JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace\n         WHERE n.nspname = {schema} AND c.relname = {table}\n           AND c.oid = to_regclass({read}))"
    ))
}

/// The schema and table a relation names, each a literal for a catalog
/// read by name, as the reference binds them: a one-part name's schema
/// `current_schema()`, which the reference reads and binds (KO2-58).
fn schema_and_name(
    request: &DataRequest,
    relation: &Relation,
) -> Result<(String, String), CoreError> {
    from(request, "tables", relation)?;
    match relation.source.as_slice() {
        [table] => Ok(("current_schema()".to_owned(), text(table))),
        [schema, table] => Ok((text(schema), text(table))),
        _ => Err(CoreError::new(
            Sentence::Internal,
            "a relation past the parts the dialect takes reached its catalog read",
        )),
    }
}

/// The reference's catalog answers' columns, by name.
mod catalog {
    pub const TABLE_NAME: &str = "table_name";
    pub const COLUMN_NAME: &str = "column_name";
    pub const DATA_TYPE: &str = "data_type";
    pub const UDT_NAME: &str = "udt_name";
    pub const NUMERIC_SCALE: &str = "numeric_scale";
    pub const IS_NULLABLE: &str = "is_nullable";
    pub const TABLE_SCHEMA: &str = "table_schema";
    pub const FORMAT_TYPE: &str = "format_type";
    pub const CONSTRAINT_TYPE: &str = "constraint_type";
    pub const CONSTRAINT_NAME: &str = "constraint_name";
    pub const ORDINAL_POSITION: &str = "ordinal_position";
}

/// The schema a Postgres answer to `request` has: the reference's catalog
/// reads' own for the table list, a table's columns, keys and foreign keys
/// ([`describe`]), what the contract fixes for every other request
/// (`DataRequest::expected`).
fn expected(request: &DataRequest) -> ExpectedSchema {
    use ColumnType::{Int32, Utf8};
    use catalog as k;
    let part = match request {
        DataRequest::Describe { part, .. } => part,
        // The bounds in the column's own type (time_bounds).
        DataRequest::TimeBounds { .. } => return ExpectedSchema::Discover {},
        _ => return request.expected(),
    };
    let fields = match part {
        // No view is read here (T3.27): only a dialect that reads views answers it.
        DescribePart::ViewDefinition => return ExpectedSchema::Discover {},
        DescribePart::Tables => vec![field(k::TABLE_NAME, Utf8, false)],
        DescribePart::Columns => vec![
            field(k::COLUMN_NAME, Utf8, false),
            field(k::DATA_TYPE, Utf8, false),
            field(k::UDT_NAME, Utf8, false),
            field(k::NUMERIC_SCALE, Int32, true),
            field(k::IS_NULLABLE, Utf8, false),
            field(k::TABLE_SCHEMA, Utf8, false),
            field(k::TABLE_NAME, Utf8, false),
            field(k::FORMAT_TYPE, Utf8, false),
        ],
        DescribePart::Keys => vec![
            field(k::CONSTRAINT_TYPE, Utf8, false),
            field(k::CONSTRAINT_NAME, Utf8, false),
            field(k::COLUMN_NAME, Utf8, false),
            field(k::ORDINAL_POSITION, Int32, false),
        ],
        DescribePart::ForeignKeys => vec![
            field(k::CONSTRAINT_NAME, Utf8, false),
            field(k::COLUMN_NAME, Utf8, false),
            field(k::TABLE_SCHEMA, Utf8, false),
            field(k::TABLE_NAME, Utf8, false),
            field(k::COLUMN_NAME, Utf8, false),
            field(k::ORDINAL_POSITION, Int32, false),
        ],
        DescribePart::Indexes | DescribePart::IndexColumns | DescribePart::Namespace => {
            return request.expected();
        }
    };
    ExpectedSchema::Schema { fields }
}

/// A Postgres answer to a `describe` of `part`, which holds [`expected`]'s
/// schema, read into the contract's catalog answer as the reference reads
/// its own (`PostgresTable`): a column's declared type its `format_type`,
/// declared `NOT NULL` where `is_nullable` is `NO`; a key's columns by
/// their position in it, each constraint once, by its name; a foreign key's
/// columns and the columns it refers to the same way, numbered by name.
/// The indexes, which the reference never reads, are in the contract's own
/// shape ([`crate::read::host`]).
fn read(part: DescribePart, batches: &[RecordBatch]) -> Result<CatalogAnswer, CoreError> {
    use catalog as k;
    match part {
        // No view is read here (T3.27): only a dialect that reads views answers it.
        DescribePart::ViewDefinition => Err(views_unread()),
        DescribePart::Tables => crate::read::tables(batches, k::TABLE_NAME),
        DescribePart::Columns => {
            let mut found = Vec::new();
            for batch in batches {
                let names = crate::read::strings(batch, k::COLUMN_NAME)?;
                let nullable = crate::read::strings(batch, k::IS_NULLABLE)?;
                let declared = crate::read::strings(batch, k::FORMAT_TYPE)?;
                for row in 0..batch.num_rows() {
                    found.push(CatalogColumn {
                        name: crate::read::text(names, row).unwrap_or_default(),
                        declared_type: crate::read::text(declared, row).unwrap_or_default(),
                        not_null: crate::read::text(nullable, row).as_deref() == Some("NO"),
                        primary_key: false,
                        key_nullable: false,
                        unique: false,
                        dtype: None,
                    });
                }
            }
            Ok(CatalogAnswer::Columns(found))
        }
        DescribePart::Keys => {
            let mut keys: BTreeMap<(String, String), Vec<(i32, String)>> = BTreeMap::new();
            for batch in batches {
                let kinds = crate::read::strings(batch, k::CONSTRAINT_TYPE)?;
                let names = crate::read::strings(batch, k::CONSTRAINT_NAME)?;
                let columns = crate::read::strings(batch, k::COLUMN_NAME)?;
                let positions = int32s(batch, 3)?;
                for row in 0..batch.num_rows() {
                    keys.entry((
                        crate::read::text(names, row).unwrap_or_default(),
                        crate::read::text(kinds, row).unwrap_or_default(),
                    ))
                    .or_default()
                    .push((
                        positions.value(row),
                        crate::read::text(columns, row).unwrap_or_default(),
                    ));
                }
            }
            let keys: Vec<CatalogKey> = keys
                .into_iter()
                .map(|((_, kind), mut columns)| {
                    columns.sort();
                    CatalogKey {
                        kind: match kind.as_str() {
                            "PRIMARY KEY" => KeyKind::Primary,
                            "UNIQUE" => KeyKind::Unique,
                            "FOREIGN KEY" => KeyKind::Foreign,
                            _ => KeyKind::Other,
                        },
                        columns: columns.into_iter().map(|(_, c)| c).collect(),
                        index: None,
                        enforced: true,
                    }
                })
                .collect();
            // A key of several columns is read as it is declared; the
            // core makes no table's key of one, as DuckDB's and SQLite's
            // references do not (the reference's Postgres graph discovery
            // fails on one, `_get_source_columns`; register P2E-4).
            Ok(CatalogAnswer::Keys(keys))
        }
        DescribePart::ForeignKeys => {
            // By position: the answer names two columns `column_name`.
            let mut keys: BTreeMap<String, (String, Vec<ColumnPair>)> = BTreeMap::new();
            for batch in batches {
                let names = strings_at(batch, 0)?;
                let columns = strings_at(batch, 1)?;
                let tables = strings_at(batch, 3)?;
                let referenced = strings_at(batch, 4)?;
                let positions = int32s(batch, 5)?;
                for row in 0..batch.num_rows() {
                    let entry = keys
                        .entry(crate::read::text(names, row).unwrap_or_default())
                        .or_insert_with(|| {
                            (crate::read::text(tables, row).unwrap_or_default(), vec![])
                        });
                    entry.1.push((
                        positions.value(row),
                        crate::read::text(columns, row).unwrap_or_default(),
                        crate::read::text(referenced, row).unwrap_or_default(),
                    ));
                }
            }
            // A foreign key of several columns is read as it is declared;
            // the core links by none, as DuckDB's and SQLite's references
            // do not (the reference's Postgres graph discovery fails on
            // one, `_get_source_foreign_keys`; register P2E-4).
            Ok(CatalogAnswer::ForeignKeys(
                keys.into_values()
                    .zip(0_i64..)
                    .map(|((table, mut pairs), key)| {
                        pairs.sort();
                        CatalogForeignKey {
                            key,
                            referenced_table: table,
                            columns: pairs.iter().map(|(_, c, _)| c.clone()).collect(),
                            referenced_columns: pairs.into_iter().map(|(_, _, r)| r).collect(),
                        }
                    })
                    .collect(),
            ))
        }
        DescribePart::Indexes | DescribePart::IndexColumns | DescribePart::Namespace => {
            crate::read::host(part, batches)
        }
    }
}

/// One column pair of a foreign key: its position, its column and the
/// column it refers to.
type ColumnPair = (i32, String, String);

fn unread(at: usize, message: &str) -> CoreError {
    CoreError::new(Sentence::Internal, message).with_field("position", Detail::text(at.to_string()))
}

/// The text column at `at` of `batch`.
fn strings_at(batch: &RecordBatch, at: usize) -> Result<&StringArray, CoreError> {
    batch
        .columns()
        .get(at)
        .and_then(|c| c.as_string_opt::<i32>())
        .ok_or_else(|| unread(at, "a catalog answer's column is not of its checked type"))
}

/// The `int32` column at `at` of `batch`, holding no null.
fn int32s(batch: &RecordBatch, at: usize) -> Result<&Int32Array, CoreError> {
    batch
        .columns()
        .get(at)
        .and_then(|c| c.as_primitive_opt::<Int32Type>())
        .filter(|c| c.null_count() == 0)
        .ok_or_else(|| unread(at, "a catalog answer's column is not of its checked type"))
}

/// One part of the catalog, in the contract's shape. The table list and
/// the indexes are read whole, as the reference reads the table list, and
/// a table list naming tables answers each the source holds (asked when a
/// named table's columns could not be read); the columns, keys and foreign
/// keys are read one table at a time, as the reference reads them.
fn describe(
    request: &DataRequest,
    tables: &[Relation],
    part: DescribePart,
    namespace: &[String],
) -> Result<String, CoreError> {
    // The schema the source's tables are listed in: the provider's
    // namespace, the source's schema (contract v28), else the current one,
    // as the reference lists the tables of the schema it was given or read
    // (a namespace deeper than the schema is refused before, by `deeper`).
    let schema = namespace.first().map(|schema| text(schema));
    let one = || regclass(request, "tables", sql::one_table(request, tables)?);
    Ok(match part {
        // No view is read here (T3.27): only a dialect that reads views answers it.
        DescribePart::ViewDefinition => return Err(views_unread()),
        // The reference's listing (`PostgresTable`'s, `information_schema`
        // the base tables of the current schema, by name), as it writes it
        // but for the schema, which it binds after reading
        // `current_schema()` and the core names in place (KO2-58).
        DescribePart::Tables if tables.is_empty() => {
            format!(
                "SELECT table_name\nFROM information_schema.tables\nWHERE table_schema = {}\n  AND table_type = 'BASE TABLE'\nORDER BY table_name",
                schema.as_deref().unwrap_or("current_schema()"),
            )
        }
        DescribePart::Tables => {
            let mut selects = Vec::with_capacity(tables.len());
            for table in tables {
                let found = regclass(request, "tables", table)?;
                selects.push(format!(
                    "SELECT relname AS table_name\nFROM pg_catalog.pg_class\nWHERE oid = {found}"
                ));
            }
            selects.join("\nUNION ALL\n")
        }
        // The reference's three reads of one table (`_get_source_columns`
        // and `_get_source_foreign_keys`), as it writes them but for the
        // schema and table, which it binds and the core writes in place;
        // the columns read only where the name reads the relation so found
        // in a `FROM` ([`regclass`]), so a shadowed name answers none.
        DescribePart::Columns => {
            let (schema, name) = schema_and_name(request, sql::one_table(request, tables)?)?;
            format!(
                "SELECT c.column_name, c.data_type, c.udt_name,\n       c.numeric_scale, c.is_nullable, c.table_schema,\n       c.table_name,\n       pg_catalog.format_type(a.atttypid, a.atttypmod)\nFROM information_schema.columns c\nJOIN pg_catalog.pg_namespace n\n  ON n.nspname = c.table_schema\nJOIN pg_catalog.pg_class rel\n  ON rel.relnamespace = n.oid AND rel.relname = c.table_name\nJOIN pg_catalog.pg_attribute a\n  ON a.attrelid = rel.oid AND a.attname = c.column_name\n AND a.attnum > 0 AND NOT a.attisdropped\nWHERE c.table_schema = {schema} AND c.table_name = {name}\n  AND rel.oid = {found}\nORDER BY c.ordinal_position",
                found = one()?,
            )
        }
        DescribePart::Keys => {
            let (schema, name) = schema_and_name(request, sql::one_table(request, tables)?)?;
            format!(
                "SELECT tc.constraint_type, tc.constraint_name,\n       kcu.column_name, kcu.ordinal_position\nFROM information_schema.table_constraints tc\nJOIN information_schema.key_column_usage kcu\n  ON tc.constraint_catalog = kcu.constraint_catalog\n AND tc.constraint_schema = kcu.constraint_schema\n AND tc.constraint_name = kcu.constraint_name\nWHERE tc.table_schema = {schema} AND tc.table_name = {name}\n  AND tc.constraint_type IN ('PRIMARY KEY', 'UNIQUE', 'FOREIGN KEY')"
            )
        }
        DescribePart::ForeignKeys => {
            let (schema, name) = schema_and_name(request, sql::one_table(request, tables)?)?;
            format!(
                "SELECT tc.constraint_name, kcu.column_name,\n       target_kcu.table_schema, target_kcu.table_name,\n       target_kcu.column_name, kcu.ordinal_position\nFROM information_schema.table_constraints tc\nJOIN information_schema.referential_constraints rc\n  ON tc.constraint_catalog = rc.constraint_catalog\n AND tc.constraint_schema = rc.constraint_schema\n AND tc.constraint_name = rc.constraint_name\nJOIN information_schema.key_column_usage kcu\n  ON tc.constraint_catalog = kcu.constraint_catalog\n AND tc.constraint_schema = kcu.constraint_schema\n AND tc.constraint_name = kcu.constraint_name\nJOIN information_schema.key_column_usage target_kcu\n  ON rc.unique_constraint_catalog = target_kcu.constraint_catalog\n AND rc.unique_constraint_schema = target_kcu.constraint_schema\n AND rc.unique_constraint_name = target_kcu.constraint_name\n AND target_kcu.ordinal_position = kcu.position_in_unique_constraint\nWHERE tc.constraint_type = 'FOREIGN KEY'\n  AND tc.table_schema = {schema} AND tc.table_name = {name}\nORDER BY tc.constraint_name, kcu.ordinal_position"
            )
        }
        // The schema a relation of one part is read in, the provider's
        // namespace or the current one, under which the reference names a
        // table it cannot find (`PostgresTable.source_name`); none where
        // the search path holds no schema.
        DescribePart::Namespace => format!(
            "SELECT s.n AS {name}\nFROM (SELECT {schema} AS n) s\nWHERE s.n IS NOT NULL",
            name = identifier(sdm_contract::data::catalog_columns::NAME),
            schema = schema.as_deref().unwrap_or("current_schema()"),
        ),
        // Never asked (no index advice is declared), and rendered: the
        // index's columns in order, an expression's position a null name.
        DescribePart::IndexColumns => {
            let index = one()?;
            format!(
                "SELECT a.attname::text AS {name}\nFROM pg_catalog.pg_index i\nCROSS JOIN LATERAL unnest(i.indkey::int2[]) WITH ORDINALITY AS n(attnum, position)\nLEFT JOIN pg_catalog.pg_attribute a\n  ON a.attrelid = i.indrelid AND a.attnum = n.attnum\nWHERE i.indexrelid = {index}\nORDER BY n.position",
                name = identifier(c::NAME),
            )
        }
        DescribePart::Indexes => {
            sql::whole(request, tables)?;
            // Each index of a table of the source's schema, its columns in
            // order, an expression's position a null name (unresolved).
            let within = match &schema {
                None => "current_schema()::regnamespace".to_owned(),
                Some(schema) => format!(
                    "(SELECT n.oid FROM pg_catalog.pg_namespace n WHERE n.nspname = {schema})"
                ),
            };
            format!(
                "SELECT t.relname::text AS {table},\n       {columns} AS {names}\nFROM pg_catalog.pg_index i\nJOIN pg_catalog.pg_class t ON t.oid = i.indrelid\nWHERE t.relnamespace = {within}\nORDER BY t.relname, i.indexrelid",
                table = identifier(c::TABLE),
                columns = attribute_names("i.indrelid", "i.indkey::int2[]"),
                names = identifier(c::COLUMNS),
            )
        }
    })
}

/// The names of `relation`'s attributes numbered in `numbers`, in their
/// order, as a `text[]`; a number naming no attribute (an index's
/// expression) a null.
fn attribute_names(relation: &str, numbers: &str) -> String {
    format!(
        "ARRAY(SELECT a.attname::text\n             FROM unnest({numbers}) WITH ORDINALITY AS n(attnum, position)\n             LEFT JOIN pg_catalog.pg_attribute a\n               ON a.attrelid = {relation} AND a.attnum = n.attnum\n             ORDER BY n.position)"
    )
}

/// The reference's map of an integer seed into `setseed`'s `[-1, 1]`
/// (`_postgres_seed`): the first eight bytes of BLAKE2b-64 of the seed's
/// decimal text, read big-endian, scaled as Python's floats scale it
/// (`2.0 * value / (2**64 - 1) - 1.0`, each step rounded to a double).
pub(crate) fn seed_value(seed: i128) -> f64 {
    let digest = crate::blake2b::digest8(seed.to_string().as_bytes());
    let value = u64::from_be_bytes(digest);
    #[allow(
        clippy::cast_precision_loss,
        reason = "an integer to the nearest double, as the reference computes it"
    )]
    let (value, most) = (value as f64, u64::MAX as f64);
    2.0 * value / most - 1.0
}

/// A finite double as the shortest text that reads back as it, with a
/// fraction or an exponent so Postgres reads a `double precision`.
fn float_text(value: f64) -> String {
    let text = format!("{value:?}");
    if text.contains(['.', 'e', 'E']) {
        text
    } else {
        format!("{text}.0")
    }
}

/// The common table expression, named as `read`'s key set, that unpacks
/// the key set bound as `$1` (the module's binding): one row per JSON row,
/// each key column read as [`key_type`] says, else in the type the table
/// declares for the key column it joins, put in place of that column in one
/// row of the table (`jsonb_populate_record` over it); a seed time a bound
/// of the window names in the type the table declares for the window's
/// column, the same way, as the reference casts its seeds to the column's
/// type; the batch as `bigint`; with its trailing line break.
fn seed_cte(read: &SeedRead<'_>) -> Result<String, CoreError> {
    let (request, key, seeds, window) = (read.request, read.key, read.seeds, read.window);
    let row_type = &read.table;
    let seed_keys = seed_key_columns(key.len());
    let carried = seeds
        .columns
        .iter()
        .filter(|c| seed_keys.contains(&c.name))
        .count();
    if carried != key.len() {
        return Err(sdm_contract::data::refused(
            request,
            "seeds",
            "the seeds do not carry every key column",
        ));
    }
    // Each seed as the old client writes it (`pg_values::key_rows`): its
    // key alone, or an object of `id` and the window's bounds.
    let others: Vec<&str> = seeds
        .columns
        .iter()
        .map(|c| c.name.as_str())
        .filter(|name| !seed_keys.iter().any(|k| k == name) && *name != SEED_BATCH)
        .collect();
    let fields: BTreeMap<&str, String> = key_text::fields(others.iter().copied(), request)
        .into_iter()
        .collect();
    let id = if others.is_empty() {
        Path::root()
    } else {
        Path::root().field("id")
    };
    let mut selects = Vec::with_capacity(seeds.columns.len());
    let mut built = Vec::with_capacity(key.len());
    let mut timed = Vec::new();
    for (position, column) in seeds.columns.iter().enumerate() {
        let read = if let Some(at) = seed_keys.iter().position(|k| *k == column.name) {
            let path = if key.len() == 1 {
                id.clone()
            } else {
                id.at(at)
            };
            match key_type(&column.column_type) {
                Some(kind) => format!("({})::{kind}", path.text),
                None => {
                    built.push(format!("{}, {}", text(&key[at]), path.json));
                    format!("k.{}", identifier(&key[at]))
                }
            }
        } else if column.name == SEED_BATCH {
            batch_select(0)
        } else {
            let path = Path::root().field(&fields[column.name.as_str()]);
            if let Some(window) = window.filter(|w| {
                [&w.lower, &w.upper]
                    .into_iter()
                    .flatten()
                    .any(|bound| bound.seed_column == column.name)
            }) {
                let alias = format!("w{position}");
                timed.push(format!(
                    "\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object({}, {})) AS {alias}",
                    text(&window.column),
                    path.json
                ));
                format!("{alias}.{}", identifier(&window.column))
            } else {
                match &column.column_type {
                    sdm_contract::ColumnType::Timestamp { zone: None, .. } => {
                        format!("({})::timestamp", path.text)
                    }
                    _ => {
                        return Err(sql::not_taken(
                            request,
                            "seeds",
                            "a seed column past the key, the batch and zone-less times",
                        ));
                    }
                }
            }
        };
        selects.push(format!("{read} AS {}", identifier(&column.name)));
    }
    // A value is read in its column's own type by putting it in place of
    // that column in one row of the table (`b.base`, the row built from the
    // alias's columns, `ROW(t.*)`, which no column's name can stand for),
    // whose other fields hold the table's own values, so a NOT NULL domain
    // elsewhere in the row is never checked on a null; a table of no row
    // gives no seeds, and so no rows.
    let mut typed = String::new();
    if !built.is_empty() || !timed.is_empty() {
        typed.push_str(&format!(
            "\n  CROSS JOIN (SELECT ROW(t.*)::{row_type} AS base FROM {row_type} AS t LIMIT 1) AS b"
        ));
    }
    if !built.is_empty() {
        typed.push_str(&format!(
            "\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object({})) AS k",
            built.join(", ")
        ));
    }
    typed.push_str(&timed.concat());
    Ok(format!(
        "WITH {name} AS (\n  SELECT {selects}\n  FROM jsonb_array_elements($1::jsonb) WITH ORDINALITY AS r(value, ordinality){typed}\n)\n",
        name = identifier(&read.seeds_name),
        selects = selects.join(", "),
    ))
}

/// The seed's batch, its position in the JSON array (from 1, as `WITH
/// ORDINALITY` counts) moved to count from `offset`, the seeds of the
/// chunks before its own, as the reference's `_json_cte` and `_flatten`
/// number it.
fn batch_select(offset: usize) -> String {
    if offset == 0 {
        "(r.ordinality - 1)::bigint".to_owned()
    } else {
        format!("(r.ordinality - 1 + {offset})::bigint")
    }
}

/// Where a seed's value is in its JSON row, as `jsonb` and as text.
#[derive(Clone)]
struct Path {
    json: String,
    text: String,
}

impl Path {
    /// The row itself, a bare value.
    fn root() -> Path {
        Path {
            json: "r.value".to_owned(),
            text: "r.value #>> '{}'".to_owned(),
        }
    }

    /// The field `name` of an object.
    fn field(&self, name: &str) -> Path {
        Path {
            json: format!("{}->{}", self.json, text(name)),
            text: format!("{}->>{}", self.json, text(name)),
        }
    }

    /// The element `at` of an array.
    fn at(&self, at: usize) -> Path {
        Path {
            json: format!("{}->{at}", self.json),
            text: format!("{}->>{at}", self.json),
        }
    }
}

/// The Postgres type a seed key of `column_type` is read as, for an
/// integer or a decimal: `bigint` or `numeric`, which Postgres compares
/// with the joined key's integers and decimals by value, so equal values
/// of any scale join; `None` for any other (a float, a text, a time), read
/// in the joined column's declared type ([`seed_cte`]), so a `real` key
/// compares as a `real` and a `uuid` as a `uuid`, as the reference casts its
/// seeds to the column's type.
fn key_type(column_type: &sdm_contract::ColumnType) -> Option<&'static str> {
    use sdm_contract::ColumnType as T;
    Some(match column_type {
        T::Int8 | T::Int16 | T::Int32 | T::Int64 | T::Uint8 | T::Uint16 | T::Uint32 => "bigint",
        T::Uint64 | T::Decimal128 { .. } | T::Decimal256 { .. } => "numeric",
        _ => return None,
    })
}

/// The table's columns asked for, each read from its alias and renamed to a
/// position of its own (`c0`, `c1`, ...), so no source column's name meets
/// a helper's inside the statement; the outer select names them back.
fn positioned(columns: &[String]) -> Vec<(String, String)> {
    columns
        .iter()
        .enumerate()
        .map(|(i, c)| (format!("c{i}"), c.clone()))
        .collect()
}

/// `value` as a text literal: in single quotes, each one in it doubled,
/// and, where it holds a backslash, as an escape string with each
/// backslash doubled, so it reads as written whatever
/// `standard_conforming_strings` says.
fn text(value: &str) -> String {
    let quoted = value.replace('\'', "''");
    if value.contains('\\') {
        format!("E'{}'", quoted.replace('\\', "\\\\"))
    } else {
        format!("'{quoted}'")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_seed_maps_as_the_reference_maps_it() {
        // `_postgres_seed(42)` and `_postgres_seed(0)` as the reference's
        // Python computes them.
        assert_eq!(seed_value(42), -0.314_812_071_551_568_76);
        assert_eq!(
            float_text(-0.314_812_071_551_568_76),
            "-0.31481207155156876"
        );
        assert_eq!(float_text(1.0), "1.0");
    }

    #[test]
    fn a_text_with_a_backslash_is_an_escape_string() {
        assert_eq!(text("a'b"), "'a''b'");
        assert_eq!(text("a\\b'"), "E'a\\\\b'''");
    }

    #[test]
    fn a_table_named_as_the_seeds_is_stepped_past() {
        assert_eq!(seeds_for(&Relation::named("t")), SEEDS);
        assert_eq!(seeds_for(&Relation::named(SEEDS)), format!("{SEEDS}_"));
    }
}
