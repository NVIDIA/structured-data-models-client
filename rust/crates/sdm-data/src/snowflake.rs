// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The Snowflake dialect (T3.6), through the shared builder's hooks
//! ([`crate::sql`], T3.18): only what is Snowflake's own is here.
//!
//! Each rendering is the statement the reference's Snowflake engine runs
//! for the same step (`rfm/backend/snow` at the reference commit), text and
//! parameters, so the warehouse answers the core as it answered the old
//! client (T3.20's recordings hold them):
//!
//! - every name in double quotes, a quote in one doubled, its case never
//!   changed (`quote_ident`); a relation of a database, a schema and a
//!   table, each part quoted alone, so `a.b` is one name (`SNOW/table.py:
//!   115-117`); the source's database and schema each given or read once a
//!   run from the session (`SELECT CURRENT_DATABASE(), CURRENT_SCHEMA()`),
//!   and a relation the session leaves without a schema refused as the
//!   reference refuses it;
//! - the catalog: `INFORMATION_SCHEMA.TABLES` for the listing, the schema
//!   bound; `DESCRIBE TABLE` for a table's columns and its keys;
//!   `SHOW IMPORTED KEYS IN TABLE` for its foreign keys; a semantic view's
//!   YAML through `SYSTEM$READ_YAML_FROM_SEMANTIC_VIEW(?)`, its name bound;
//! - the answers of `SHOW` and `DESCRIBE` come as text, which the provider's
//!   row adapter builds (the connector gives them as rows only), each read
//!   by its position in the result, as the reference reads the rows;
//! - an exact aggregate row count, also supporting views (approved P3D-2/3);
//! - an entity sample `SAMPLE ROW (<n> ROWS)`, at most 1,000,000 rows,
//!   which takes no seed (owner option C);
//! - a declared type read by the reference's type map ([`declared_dtype`]);
//! - the connector's Arrow conformed as the reference reads its frames
//!   ([`conform`]);
//! - a failure with errno 2003 or a diagnostic containing "does not exist" is
//!   "not found", which the core's spelling retry reads; a name's
//!   spellings are as given, then all its parts upper-cased, once.
//!
//! - a key set bound as the reference binds it, one JSON text the core
//!   writes ([`crate::key_text`]), unpacked by `FLATTEN(PARSE_JSON(?))`,
//!   its seeds separated by `, ` as `json.dumps` writes them, a window's
//!   bounds over every seed bound after it; a lookup's first row of each
//!   seed and the rows holding its key kept in the statement, where the
//!   reference keeps them after its fetch.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{
    Decimal128Type, Int8Type, Int16Type, Int32Type, TimestampMicrosecondType,
    TimestampMillisecondType, TimestampNanosecondType, TimestampSecondType, UInt8Type, UInt16Type,
    UInt32Type,
};
use arrow_array::{Array, ArrayRef, Float64Array, Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema, SchemaRef, TimeUnit};
use sdm_contract::catalog::{
    CatalogAnswer, CatalogColumn, CatalogForeignKey, CatalogKey, KeyKind, ViewDefinition,
};
use sdm_contract::data::{
    DataRequest, DescribePart, Determinism, ExpectedSchema, FailureFacts, MARKED, MATCHES_COLUMN,
    NullOrder, ProviderCapabilities, ProviderError, Relation, SEED_BATCH, SEED_COUNT_UNREAD,
    SEED_KEY, SEED_KEY_UNIQUE, SqlParam, TextTimes, field,
};
use sdm_contract::ops::Dtype;
use sdm_contract::types::ColumnType;
use sdm_contract::view::ViewKind;
use sdm_contract::{CoreError, Detail, Scalar, Sentence};

use crate::key_text;
use crate::read::{text, text_at};
use crate::sql::{self, Batches, Sample, SeedRead, SqlDialect, not_taken};

/// The dialect's name on the wire.
pub(crate) const NAME: &str = "snowflake";

/// A database, a schema and a table: every relation is read under the
/// source's database and schema, a table's name one part.
const MOST_PARTS: usize = 3;

/// The namespace parts a Snowflake source takes: a database and a schema.
const NAMESPACE_DEPTH: usize = 2;

/// Snowflake's limit on the rows `SAMPLE ROW` draws, which the reference
/// caps a sample at (`SNOW/sampler.py:92`).
const MOST_SAMPLED: u64 = 1_000_000;

/// The connector's errno for an object Snowflake cannot resolve, SQL
/// compilation error `002003` (`SNOW/table.py:31`).
const MISSING_OBJECT: &str = "2003";

/// The reference's read of the session's database and schema
/// (`RE/graph.py:1024-1033`).
const SESSION: &str = "SELECT CURRENT_DATABASE(), CURRENT_SCHEMA()";

/// The name a statement reads its unpacked key set by, the reference's.
const TMP: &str = "TMP";

/// The Snowflake dialect.
pub(crate) struct Snowflake;

impl SqlDialect for Snowflake {
    fn name(&self) -> &'static str {
        NAME
    }

    /// What a Snowflake source answers through this layer.
    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            opaque_vendor: None,
            compares_stored_times: false,
            samples_by_key_dtype: false,
            reads_nan_as_missing: false,
            optional: sql::rendered(),
            max_keys_per_request: None,
            // `SAMPLE ROW (n ROWS)` takes no seed (owner option C).
            // P3D-13 documents this without a routine optional-seed warning.
            sampling: Determinism::Unseeded,
            // Any seed is taken, and none is written.
            max_seed: u64::MAX,
            // The reference's engine reads a SQL source's text times one by
            // one (register DT-7).
            text_times: Some(TextTimes::ValueByValue),
            // Snowflake orders a null last, ascending or descending (the
            // recordings' session, `null_order`).
            null_order: Some(NullOrder::Last),
            // The reference's Snowflake sampler neither reads indexes nor
            // warns of missing ones.
            index_advice: None,
            fallback: None,
            derived_relations: true,
            // The sample's size is its own `n ROWS`.
            sample_rows: false,
            // A neighbor read looks back by the tables' rows and time span
            // (`SNOW/sampler.py:281-305`).
            neighbor_lookback: true,
            keys_optional: false,
            key_enforcement: true,
        }
    }

    fn expr_dialect(&self) -> Option<crate::view_expr::ExprDialect> {
        Some(crate::view_expr::ExprDialect::Snowflake)
    }

    fn view_kind(&self) -> Option<ViewKind> {
        Some(ViewKind::Semantic)
    }

    fn declared_type(&self, data_type: &str) -> sdm_contract::view::DeclaredType {
        use sdm_contract::view::DeclaredType;
        match declared_dtype(data_type) {
            Some(Dtype::Unsupported) => DeclaredType::Unsupported,
            Some(dtype) => DeclaredType::Known(dtype),
            None => DeclaredType::Unknown,
        }
    }

    fn read_view(
        &self,
        request: &DataRequest,
        batches: &[RecordBatch],
        _spaces: &crate::view_expr::Spaces,
    ) -> Result<CatalogAnswer, CoreError> {
        read(request, DescribePart::ViewDefinition, batches)
    }

    fn derived(&self, relation: &Relation, source: &str) -> Option<String> {
        if relation.found {
            return Some(source.to_owned());
        }
        let columns = if relation.columns.is_empty() {
            "*".to_owned()
        } else {
            relation
                .columns
                .iter()
                .map(|column| match (&column.column, &column.expression) {
                    (Some(read), _) if *read == column.name => self.identifier(read),
                    (Some(read), _) => format!(
                        "{} AS {}",
                        self.identifier(read),
                        self.identifier(&column.name)
                    ),
                    (None, Some(expression)) => {
                        format!("({expression}) AS {}", self.identifier(&column.name))
                    }
                    (None, None) => String::new(), // Refused by refuse_derived before rendering.
                })
                .collect::<Vec<_>>()
                .join(", ")
        };
        let filter = relation
            .filter
            .as_ref()
            .map(|f| format!(" WHERE ({f})"))
            .unwrap_or_default();
        Some(format!("(SELECT {columns} FROM {source}{filter})"))
    }

    fn most_parts(&self) -> usize {
        MOST_PARTS
    }

    /// A database and a schema (design fix C).
    fn namespace_depth(&self) -> usize {
        NAMESPACE_DEPTH
    }

    /// The parts a source leaves to the session read once per run, and
    /// every relation named under its database and schema, as the
    /// reference names every table it reads.
    fn settles_namespace(&self) -> bool {
        true
    }

    /// Snowflake returns the raw session database and schema. Supplied
    /// parts align from the database outward, and missing session parts
    /// stop resolution without dropping a database the source names.
    fn settled_namespace(&self, given: &[Option<String>], session: Vec<String>) -> Vec<String> {
        let mut parts = Vec::with_capacity(given.len().max(session.len()));
        for at in 0..given.len().max(session.len()) {
            match given
                .get(at)
                .cloned()
                .flatten()
                .or_else(|| session.get(at).cloned())
            {
                Some(part) => parts.push(part),
                None => break,
            }
        }
        parts
    }

    fn unqualified(&self, relation: &Relation) -> Option<CoreError> {
        unqualified(relation)
    }

    fn seeds_for(&self, _request: &DataRequest, _table: &Relation) -> String {
        TMP.to_owned()
    }

    fn seeds_relation(&self, name: &str) -> String {
        name.to_owned()
    }

    /// The unpacked key set, `TMP`, as the reference unpacks it
    /// (`SNOW/sampler.py:204-244`, `:317-345`, `:383-430`): the JSON
    /// array's index the batch, the key cast by its type, and each of the
    /// window's bounds a `TIMESTAMP_NTZ`.
    fn seeds_prelude(&self, read: &SeedRead<'_>) -> Result<String, CoreError> {
        seeds_cte(read)
    }

    /// A neighbor's order term as the reference writes it
    /// (`_neighbor_order_by`): the column, `DESC` where descending, its null
    /// order written only where it is not Snowflake's own for the
    /// direction (last ascending, first descending); a time term with none,
    /// its column holding no null in the rows it orders (register ORD-2).
    fn order_term(
        &self,
        term: &sdm_contract::data::TieTerm,
        alias: &str,
        _windowed: Option<&str>,
    ) -> String {
        let column = format!("{alias}{}", self.identifier(&term.column));
        let direction = if term.descending { " DESC" } else { "" };
        let nulls = match (term.as_timestamp, term.descending, term.nulls_last) {
            (true, _, _) => "",
            (false, false, false) => " NULLS FIRST",
            (false, true, true) => " NULLS LAST",
            _ => "",
        };
        format!("{column}{direction}{nulls}")
    }

    /// The key set as the reference writes it: `, ` between its seeds.
    fn key_separator(&self) -> &'static str {
        ", "
    }

    fn key_rows(
        &self,
        request: &DataRequest,
        schema: &SchemaRef,
        batches: &[RecordBatch],
        float_text: fn(f64) -> String,
    ) -> Result<Option<Vec<String>>, CoreError> {
        key_rows(request, schema, batches, float_text).map(Some)
    }

    fn key_bounds(
        &self,
        request: &DataRequest,
        schema: &SchemaRef,
        batches: &[RecordBatch],
    ) -> Result<Vec<SqlParam>, CoreError> {
        key_bounds(request, schema, batches)
    }

    /// `COUNT(DISTINCT a, b)`, the reference's: Snowflake leaves out a row
    /// missing any part (`RE/base/key_count.py:40-95`).
    fn distinct_keys(&self, quoted: &[String], _table: &str) -> String {
        format!("COUNT(DISTINCT {})", quoted.join(", "))
    }

    /// `SAMPLE ROW` ignores the optional seed.
    fn sample_seed(&self, _request: &DataRequest, _seed: &Scalar) -> Result<i128, CoreError> {
        Ok(0)
    }

    /// The reference's sample of an entity table (`_sample_entity_table`):
    /// the columns asked for, from `SAMPLE ROW (<n> ROWS)`, `n` at most
    /// Snowflake's limit, then each not-null filter (`SAMPLE` before
    /// `WHERE`, so the filters thin the sample, as they do the reference's).
    fn sample(&self, sample: &Sample<'_>) -> Vec<String> {
        vec![format!(
            "SELECT {}\nFROM {}\nSAMPLE ROW ({} ROWS){}",
            sample.projected,
            sample.table,
            sample.n.min(MOST_SAMPLED),
            sample.filtered
        )]
    }

    /// The reference's `_by_fkey` (`SNOW/sampler.py:317-365`): each seed's
    /// rows within its window, the window's bounds over every seed bound
    /// after the key set, the first `limit` in `terms`' order (`QUALIFY`),
    /// the result in `(batch, order)`.
    fn neighbors_per_seed(
        &self,
        read: &SeedRead<'_>,
        limit: u64,
        terms: &[sdm_contract::data::TieTerm],
    ) -> String {
        let order = terms
            .iter()
            .map(|t| self.order_term(t, "", None))
            .collect::<Vec<_>>()
            .join(", ");
        let bounds = bounds(read);
        let mut text = format!("{}\n", joined(read));
        for (name, op, _) in &bounds {
            text.push_str(&format!(" AND {} {op} TMP.{name}\n", window_column(read)));
        }
        if !bounds.is_empty() {
            text.push_str(&format!("WHERE {}\n", global(read, &bounds, "\n  AND ")));
        }
        text.push_str(&format!(
            "QUALIFY ROW_NUMBER() OVER (\n  PARTITION BY TMP.__KUMO_BATCH__\n  ORDER BY {order}\n) <= {limit}\nORDER BY TMP.__KUMO_BATCH__, {order}"
        ));
        text
    }

    /// A limit per seed comes with an order, and an order with a limit.
    fn neighbors_unpaired(&self, read: &SeedRead<'_>) -> Result<String, CoreError> {
        Err(sdm_contract::data::refused(
            read.request,
            "order",
            "a limit per seed comes with an order, and an order with a limit",
        ))
    }

    /// The reference's `_by_time` (`SNOW/sampler.py:383-450`): every row of
    /// each seed within its window, the window's bounds over every seed
    /// bound after the key set, as the reference writes it.
    fn neighbors_all(&self, read: &SeedRead<'_>) -> String {
        let bounds = bounds(read);
        let mut text = joined(read);
        for (at, (name, op, _)) in bounds.iter().enumerate() {
            let lead = if at == 0 { "\n AND" } else { "\nAND" };
            text.push_str(&format!("{lead} {} {op} TMP.{name}", window_column(read)));
        }
        if !bounds.is_empty() {
            text.push_str(&format!("\nWHERE {}", global(read, &bounds, "\nAND ")));
        }
        text
    }

    /// The reference's `_by_pkey` (`SNOW/sampler.py:204-260`): the rows
    /// holding each seed's key; where the source keeps the key unique, the
    /// plain join the reference sends; otherwise the first row of each
    /// seed, as the reference keeps the first it read, with how many rows
    /// hold the key, which it counts after its fetch, unless the seeds say
    /// that count is not read.
    fn lookup(&self, read: &SeedRead<'_>) -> String {
        let flagged = |column: &str, flag: &str| {
            read.seeds.columns.iter().any(|c| {
                c.name == column && c.metadata.get(flag).map(String::as_str) == Some(MARKED)
            })
        };
        let joined = joined(read);
        if flagged(SEED_KEY, SEED_KEY_UNIQUE) {
            return joined;
        }
        let counted = if flagged(SEED_BATCH, SEED_COUNT_UNREAD) {
            String::new()
        } else {
            format!(
                ", COUNT(*) OVER (PARTITION BY TMP.__KUMO_BATCH__) AS {}",
                self.identifier(MATCHES_COLUMN)
            )
        };
        let (select, rest) = joined.split_once("\nFROM TMP").unwrap_or((&joined, ""));
        format!(
            "{select}{counted}\nFROM TMP{rest}\nQUALIFY ROW_NUMBER() OVER (PARTITION BY TMP.__KUMO_BATCH__ ORDER BY {}) = 1",
            identifier(&read.key[0])
        )
    }

    fn describe(
        &self,
        request: &DataRequest,
        tables: &[Relation],
        part: DescribePart,
        namespace: &[String],
    ) -> Result<(String, Vec<SqlParam>), CoreError> {
        describe(request, tables, part, namespace)
    }

    /// The session's read, whatever the source names: the core keeps the
    /// parts the source gives and takes the others from the answer
    /// (`DataLayer::settled`), as the reference fills only the missing part.
    fn current_namespace(
        &self,
        request: &DataRequest,
        _namespace: &[Option<String>],
    ) -> Result<(String, Vec<SqlParam>), CoreError> {
        describe(request, &[], DescribePart::Namespace, &[])
    }

    /// The catalog's answers and the time bounds as the warehouse gives
    /// them, each read by position; the key counts under the names
    /// Snowflake folds the reference's unquoted aliases to, each an
    /// integer once conformed.
    fn expected(&self, request: &DataRequest) -> ExpectedSchema {
        match request {
            DataRequest::Describe { .. } | DataRequest::TimeBounds { .. } => {
                ExpectedSchema::Discover {}
            }
            DataRequest::KeyCounts { .. } => ExpectedSchema::Schema {
                fields: ["K", "N", "D", "Z"]
                    .iter()
                    .map(|name| field(name, ColumnType::Int64, false))
                    .collect(),
            },
            _ => request.expected(),
        }
    }

    fn read(
        &self,
        request: &DataRequest,
        part: DescribePart,
        batches: &[RecordBatch],
    ) -> Result<CatalogAnswer, CoreError> {
        read(request, part, batches)
    }

    fn conform(
        &self,
        schema: &SchemaRef,
        batches: &[RecordBatch],
        _float_text: fn(f64) -> String,
    ) -> Result<Option<Batches>, CoreError> {
        conform(schema, batches)
    }

    /// The reference retries errno 2003 or a diagnostic containing "does
    /// not exist". Driver text remains internal and is never echoed.
    fn failure(&self, error: &ProviderError) -> FailureFacts {
        FailureFacts {
            not_found: error.vendor_code.as_deref() == Some(MISSING_OBJECT)
                || error
                    .driver_message
                    .as_deref()
                    .is_some_and(|text| text.to_lowercase().contains("does not exist")),
            ..FailureFacts::default()
        }
    }

    /// As given, then every part upper-cased, as an unquoted name resolves,
    /// the second only where it differs (`SNOW/table.py:124-152`).
    fn spellings(&self, relation: &Relation) -> Vec<Relation> {
        let upper: Vec<String> = relation.source.iter().map(|p| p.to_uppercase()).collect();
        if upper == relation.source {
            return vec![relation.clone()];
        }
        vec![relation.clone(), Relation::of(upper)]
    }
}

fn identifier(name: &str) -> String {
    Snowflake.identifier(name)
}

/// The type a key of `column_type` is cast to from its JSON, as the
/// reference casts it by the key's data type: `NUMBER` for an integer,
/// `FLOAT` for a float, `VARCHAR` otherwise.
fn key_cast(column_type: &ColumnType) -> &'static str {
    match key_text::key_type(column_type) {
        key_text::KeyType::Integer => "NUMBER",
        key_text::KeyType::Double => "FLOAT",
        key_text::KeyType::Text => "VARCHAR",
    }
}

/// The window's bounds a read compares with, as the reference names them
/// in `TMP`, the upper first: each its name, its comparison and its seed
/// column.
fn bounds<'a>(read: &SeedRead<'a>) -> Vec<(&'static str, &'static str, &'a str)> {
    let Some(window) = read.window else {
        return Vec::new();
    };
    let mut found = Vec::new();
    if let Some(upper) = &window.upper {
        let op = if upper.inclusive { "<=" } else { "<" };
        found.push(("__KUMO_END_TIME__", op, upper.seed_column.as_str()));
    }
    if let Some(lower) = &window.lower {
        let op = if lower.inclusive { ">=" } else { ">" };
        found.push(("__KUMO_START_TIME__", op, lower.seed_column.as_str()));
    }
    found
}

/// The window's column, quoted.
fn window_column(read: &SeedRead<'_>) -> String {
    read.window.map_or_else(String::new, |w| {
        let column = identifier(&w.column);
        if w.source_type.as_deref().is_some_and(|t| {
            let kind = t.trim().to_uppercase();
            kind.starts_with("TIMESTAMP_LTZ") || kind.starts_with("TIMESTAMP_TZ")
        }) {
            format!("CONVERT_TIMEZONE('UTC', {column})::TIMESTAMP_NTZ")
        } else {
            column
        }
    })
}

/// The window's bounds over every seed, one bound parameter each, joined by
/// `between`.
fn global(read: &SeedRead<'_>, bounds: &[(&str, &str, &str)], between: &str) -> String {
    let column = window_column(read);
    bounds
        .iter()
        .map(|(_, op, _)| format!("{column} {op} ?"))
        .collect::<Vec<_>>()
        .join(between)
}

/// `TMP` as the reference unpacks the key set (`Snowflake::seeds_prelude`).
fn seeds_cte(read: &SeedRead<'_>) -> Result<String, CoreError> {
    let [_] = read.key else {
        return Err(not_taken(
            read.request,
            "key",
            "a key of several columns is read as the reference reads none",
        ));
    };
    let Some(key) = read.seeds.columns.iter().find(|c| c.name == SEED_KEY) else {
        return Err(sdm_contract::data::refused(
            read.request,
            "seeds",
            "the seeds lack a key column",
        ));
    };
    let bounds = bounds(read);
    for column in &read.seeds.columns {
        if column.name != SEED_KEY
            && column.name != SEED_BATCH
            && !bounds.iter().any(|(_, _, seed)| *seed == column.name)
        {
            return Err(not_taken(
                read.request,
                "seeds",
                "a seed column past the key, the batch and the window's bounds",
            ));
        }
    }
    let cast = key_cast(&key.column_type);
    let batch = if read.offset == 0 {
        "f.index".to_owned()
    } else {
        format!("f.index + {}", read.offset)
    };
    let mut lines = vec![format!("{batch} as __KUMO_BATCH__")];
    if matches!(read.request, DataRequest::Lookup { .. }) {
        lines.push(format!("f.value::{cast} as __KUMO_ID__"));
    } else {
        lines.push(format!("f.value[0]::{cast} as __KUMO_ID__"));
        for (at, (name, _, _)) in bounds.iter().enumerate() {
            lines.push(format!("f.value[{}]::TIMESTAMP_NTZ as {name}", at + 1));
        }
    }
    Ok(format!(
        "WITH TMP as (\n  SELECT\n    {}\n  FROM TABLE(FLATTEN(INPUT => PARSE_JSON(?))) f\n)\n",
        lines.join(",\n    ")
    ))
}

/// The rows of the table each seed's key holds, joined to `TMP`, as the
/// reference writes them: the batch, then the columns asked for.
fn joined(read: &SeedRead<'_>) -> String {
    let columns = read
        .columns
        .iter()
        .map(|c| identifier(c))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "SELECT TMP.__KUMO_BATCH__ as __KUMO_BATCH__, {columns}\nFROM TMP\nJOIN {}\n  ON {} = TMP.__KUMO_ID__",
        read.table,
        identifier(&read.key[0])
    )
}

/// Each seed of a key set as the JSON the reference binds it as, in order
/// (`SNOW/sampler.py:216`, `:306-309`, `:398-404`): a lookup's seed its key
/// alone, a neighbor read's an array of its key and its window's bounds;
/// a key coerced as the reference coerces it, a bound its wall clock to the
/// second. A missing bound is JSON null, preserving its entity while its
/// window reaches no temporal rows (P3D-8).
fn key_rows(
    request: &DataRequest,
    schema: &SchemaRef,
    batches: &[RecordBatch],
    float_text: fn(f64) -> String,
) -> Result<Vec<String>, CoreError> {
    key_text::in_batch_order(schema, batches)?;
    let (schema, batches) = key_text::coerced(schema, batches, Sentence::SnowflakeIntegerKeyRange)?;
    let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    if key_text::key_columns(&names).len() != 1 {
        return Err(not_taken(
            request,
            "key",
            "a key of several columns is read as the reference reads none",
        ));
    }
    let order: Vec<usize> = key_text::fields(names.iter().copied(), request)
        .into_iter()
        .filter_map(|(column, _)| schema.index_of(column).ok())
        .collect();
    let lookup = matches!(request, DataRequest::Lookup { .. });
    let mut rows = Vec::new();
    for batch in &batches {
        for row in 0..batch.num_rows() {
            let cells: Vec<String> = order
                .iter()
                .map(|&at| key_text::json_cell(batch.column(at), row, float_text))
                .collect();
            rows.push(if lookup {
                cells.join(", ")
            } else {
                format!("[{}]", cells.join(", "))
            });
        }
    }
    Ok(rows)
}

/// The window's bounds over every seed of a neighbor read, as the
/// reference binds them after the key set (`SNOW/sampler.py:350-360`,
/// `:425-430`): the latest upper bound, then the earliest lower one, each
/// its wall clock to the second, a missing seed's left out.
fn key_bounds(
    request: &DataRequest,
    schema: &SchemaRef,
    batches: &[RecordBatch],
) -> Result<Vec<SqlParam>, CoreError> {
    let DataRequest::Neighbors {
        window: Some(window),
        ..
    } = request
    else {
        return Ok(Vec::new());
    };
    let (schema, batches) = key_text::coerced(schema, batches, Sentence::SnowflakeIntegerKeyRange)?;
    let texts = |column: &str| -> Vec<String> {
        let Ok(at) = schema.index_of(column) else {
            return Vec::new();
        };
        batches
            .iter()
            .flat_map(|batch| {
                let column = batch.column(at).as_string::<i32>().clone();
                (0..column.len())
                    .filter_map(|row| text(&column, row))
                    .collect::<Vec<_>>()
            })
            .collect()
    };
    let bound = |found: Option<String>| SqlParam::Value {
        value: found.map_or(Scalar::Null, Scalar::Utf8),
    };
    let mut params = Vec::new();
    if let Some(upper) = &window.upper {
        params.push(bound(texts(&upper.seed_column).into_iter().max()));
    }
    if let Some(lower) = &window.lower {
        params.push(bound(texts(&lower.seed_column).into_iter().min()));
    }
    Ok(params)
}

/// A text parameter.
fn text_param(value: &str) -> SqlParam {
    SqlParam::Value {
        value: Scalar::Utf8(value.to_owned()),
    }
}

/// The refusal of a relation the session leaves without a schema: named,
/// for a table under the database alone, as the reference's table refuses
/// it (`SNOW/table.py:88-93`); otherwise as its listing does.
fn unqualified(relation: &Relation) -> Option<CoreError> {
    match relation.source.as_slice() {
        [_, _, _] => None,
        [database, table] => Some(
            CoreError::new(
                Sentence::UnspecifiedSchema,
                "a table is read under a database with no schema named or current",
            )
            .with_field("table", Detail::text(table))
            .with_field("in_database", Detail::text(database)),
        ),
        _ => Some(no_current_database()),
    }
}

/// The reference's refusal of a source with no schema, named or current
/// (`RE/graph.py:1035-1040`).
fn no_current_schema() -> CoreError {
    CoreError::new(
        Sentence::NoCurrentSchema,
        "the source names no schema and the session has no current one",
    )
    .with_field("database", Detail::text(NAME))
}

/// A relation as a statement names it, each part quoted alone.
fn rendered(relation: &Relation) -> String {
    relation
        .source
        .iter()
        .map(|part| identifier(part))
        .collect::<Vec<_>>()
        .join(".")
}

/// The reference's catalog reads, as it writes them; `namespace` the
/// source's database and schema, given or read from the session.
fn describe(
    request: &DataRequest,
    tables: &[Relation],
    part: DescribePart,
    namespace: &[String],
) -> Result<(String, Vec<SqlParam>), CoreError> {
    let one = || sql::one_table(request, tables);
    Ok(match part {
        DescribePart::Namespace => {
            sql::whole(request, tables)?;
            (SESSION.to_owned(), vec![])
        }
        // `Graph.from_snowflake`'s listing, as it writes it: the database
        // quoted as given, the schema bound and compared exactly, every
        // kind of table, in the warehouse's order (`RE/graph.py:1042-1051`).
        DescribePart::Tables if tables.is_empty() => {
            let [database, schema] = namespace else {
                return Err(if namespace.is_empty() {
                    no_current_database()
                } else {
                    no_current_schema()
                });
            };
            let pad = " ".repeat(20);
            (
                format!(
                    "\n{pad}SELECT TABLE_NAME\n{pad}FROM {}.INFORMATION_SCHEMA.TABLES\n{pad}WHERE TABLE_SCHEMA = ?\n{pad}",
                    identifier(database)
                ),
                vec![text_param(schema)],
            )
        }
        // A table Snowflake cannot resolve fails its `DESCRIBE` with errno
        // 2003, which says it is missing; no listing is asked.
        DescribePart::Tables => {
            return Err(not_taken(
                request,
                "tables",
                "Snowflake says a table is missing through its DESCRIBE",
            ));
        }
        // `_get_source_columns` (`SNOW/table.py:123-170`): the columns and
        // the keys of one `DESCRIBE`.
        DescribePart::Columns | DescribePart::Keys => {
            (format!("DESCRIBE TABLE {}", rendered(one()?)), vec![])
        }
        // `_get_source_foreign_keys` (`SNOW/table.py:174-189`).
        DescribePart::ForeignKeys => (
            format!("SHOW IMPORTED KEYS IN TABLE {}", rendered(one()?)),
            vec![],
        ),
        DescribePart::KeyEnforcement => {
            let table = one()?;
            let namespace = &table.source[..table.source.len() - 1];
            (
                format!(
                    "SHOW TABLES IN SCHEMA {}",
                    namespace
                        .iter()
                        .map(|p| identifier(p))
                        .collect::<Vec<_>>()
                        .join(".")
                ),
                vec![],
            )
        }
        // A semantic view's YAML (`RE/graph.py:2054-2060`), its name bound:
        // a name with a dot as given, already qualified, otherwise under the
        // source's database and schema (`CL/materializers/sources.py:
        // 310-326`).
        DescribePart::ViewDefinition => {
            let view = one()?;
            let name = match view.source.last() {
                Some(last) if last.contains('.') => last.clone(),
                _ => view.source.join("."),
            };
            (
                "SELECT SYSTEM$READ_YAML_FROM_SEMANTIC_VIEW(?)".to_owned(),
                vec![text_param(&name)],
            )
        }
        DescribePart::Indexes | DescribePart::IndexColumns => {
            return Err(not_taken(
                request,
                "part",
                "Snowflake keeps no index the reference reads",
            ));
        }
    })
}

/// A Snowflake answer to a `describe` of `part`, read as the reference
/// reads it, each column by its position.
fn read(
    request: &DataRequest,
    part: DescribePart,
    batches: &[RecordBatch],
) -> Result<CatalogAnswer, CoreError> {
    match part {
        DescribePart::Namespace => session(batches),
        DescribePart::Tables => {
            let mut names = Vec::new();
            for batch in batches {
                let column = text_at(batch, 0)?;
                names.extend((0..batch.num_rows()).filter_map(|row| text(&column, row)));
            }
            Ok(CatalogAnswer::Tables(names))
        }
        DescribePart::Columns => Ok(CatalogAnswer::Columns(
            described(batches)?
                .into_iter()
                .map(|row| CatalogColumn {
                    dtype: declared_dtype(&row.declared_type),
                    name: row.name,
                    declared_type: row.declared_type,
                    not_null: !row.nullable,
                    primary_key: false,
                    key_nullable: false,
                    unique: false,
                })
                .collect(),
        )),
        DescribePart::Keys => {
            // The columns flagged a primary key, and those flagged unique,
            // each one key: a key of one column is the table's, one of
            // several none's (`RE/base/table.py:1016-1024`, `:778-789`).
            // Only a hybrid table enforces its key, so each is counted.
            let rows = described(batches)?;
            let flagged = |pick: fn(&Described) -> bool| -> Vec<String> {
                rows.iter()
                    .filter(|r| pick(r))
                    .map(|r| r.name.clone())
                    .collect()
            };
            let mut keys = Vec::new();
            for (kind, flagged) in [
                (KeyKind::Primary, flagged(|r| r.primary_key)),
                (KeyKind::Unique, flagged(|r| r.unique)),
            ] {
                if !flagged.is_empty() {
                    keys.push(CatalogKey {
                        kind,
                        columns: flagged,
                        index: None,
                        enforced: false,
                    });
                }
            }
            Ok(CatalogAnswer::Keys(keys))
        }
        DescribePart::ForeignKeys => imported_keys(batches),
        DescribePart::KeyEnforcement => key_enforcement(batches),
        DescribePart::ViewDefinition => {
            let yaml = batches
                .iter()
                .find(|b| b.num_rows() > 0)
                .and_then(|batch| text_at(batch, 0).ok().and_then(|column| text(&column, 0)))
                .filter(|text| !text.is_empty())
                .ok_or_else(|| {
                    let view = match request {
                        DataRequest::Describe { tables, .. } => tables
                            .first()
                            .and_then(|t| t.source.last())
                            .map(String::as_str)
                            .unwrap_or_default(),
                        _ => "",
                    };
                    CoreError::new(
                        Sentence::ViewSemanticDefinitionUnread,
                        "a semantic view's definition is not a text",
                    )
                    .with_field("view", Detail::text(view))
                })?;
            Ok(CatalogAnswer::ViewDefinition(ViewDefinition {
                kind: ViewKind::Semantic,
                text: yaml,
                parameters: Vec::new(),
                located: Vec::new(),
                columns: Vec::new(),
            }))
        }
        DescribePart::Indexes | DescribePart::IndexColumns => Err(not_taken(
            request,
            "part",
            "Snowflake keeps no index the reference reads",
        )),
    }
}

/// The session's database and schema, the schema left out where it has
/// none (no table can then be read under it, and the plan says so); a
/// missing database yields no parts so the source's named database can
/// still be retained before namespace validation.
fn session(batches: &[RecordBatch]) -> Result<CatalogAnswer, CoreError> {
    let batch = batches
        .iter()
        .find(|b| b.num_rows() > 0)
        .ok_or_else(no_current_database)?;
    let Some(database) = text(&text_at(batch, 0)?, 0) else {
        return Ok(CatalogAnswer::Namespace(Vec::new()));
    };
    let schema = text(&text_at(batch, 1)?, 0);
    Ok(CatalogAnswer::Namespace(
        std::iter::once(database).chain(schema).collect(),
    ))
}

/// A session with no current database: the reference's bare `assert`
/// (`RE/graph.py:1030-1032`), replaced by the approved named refusal.
fn no_current_database() -> CoreError {
    CoreError::new(
        Sentence::NoCurrentDatabase,
        "the session has no current database and the source names none",
    )
    .with_field("database", Detail::text(NAME))
}

/// The schema-wide listing is optional evidence. Field names are read
/// without case, as the connector's descriptions are read by the reference.
fn key_enforcement(batches: &[RecordBatch]) -> Result<CatalogAnswer, CoreError> {
    let mut names = Vec::new();
    for batch in batches {
        let at = |name: &str| {
            batch
                .schema()
                .fields()
                .iter()
                .position(|f| f.name().eq_ignore_ascii_case(name))
        };
        let (Some(name), Some(hybrid)) = (at("name"), at("is_hybrid")) else {
            return Ok(CatalogAnswer::KeyEnforcement(Vec::new()));
        };
        let (Ok(name), Ok(hybrid)) = (text_at(batch, name), text_at(batch, hybrid)) else {
            return Ok(CatalogAnswer::KeyEnforcement(Vec::new()));
        };
        for row in 0..batch.num_rows() {
            if text(&hybrid, row).is_some_and(|v| v.trim().eq_ignore_ascii_case("Y"))
                && let Some(name) = text(&name, row)
            {
                names.push(name);
            }
        }
    }
    Ok(CatalogAnswer::KeyEnforcement(names))
}

/// One row of `DESCRIBE TABLE`, as the reference reads it.
struct Described {
    name: String,
    declared_type: String,
    nullable: bool,
    primary_key: bool,
    unique: bool,
}

/// The rows of a `DESCRIBE TABLE` answer: columns 0, 1, 3, 5 and 6, its
/// name, type, `null?`, `primary key` and `unique key`, each flag `Y`
/// after strip and upper-case (`SNOW/table.py:160-170`).
fn described(batches: &[RecordBatch]) -> Result<Vec<Described>, CoreError> {
    let flag = |value: Option<String>| value.is_some_and(|v| v.trim().to_uppercase() == "Y");
    let mut found = Vec::new();
    for batch in batches {
        let [names, types, nulls, primary, unique] = [0, 1, 3, 5, 6].map(|at| text_at(batch, at));
        let (names, types, nulls, primary, unique) = (names?, types?, nulls?, primary?, unique?);
        for row in 0..batch.num_rows() {
            found.push(Described {
                name: text(&names, row).unwrap_or_default(),
                declared_type: text(&types, row).unwrap_or_default(),
                nullable: flag(text(&nulls, row)),
                primary_key: flag(text(&primary, row)),
                unique: flag(text(&unique, row)),
            });
        }
    }
    Ok(found)
}

/// The rows of `SHOW IMPORTED KEYS IN TABLE`, by position: the referenced
/// table (columns 1 to 3, joined with dots, unquoted) and column (4), the
/// referencing column (8), numbered by the referencing key's name (12), so a
/// composite constraint stays grouped, while independent constraints to
/// the same target stay independent (approved P3D-6).
fn imported_keys(batches: &[RecordBatch]) -> Result<CatalogAnswer, CoreError> {
    let mut names: Vec<String> = Vec::new();
    let mut keys = Vec::new();
    for batch in batches {
        let column = |at: usize| text_at(batch, at);
        let (database, schema, table, referenced, referencing, name) = (
            column(1)?,
            column(2)?,
            column(3)?,
            column(4)?,
            column(8)?,
            column(12)?,
        );
        for row in 0..batch.num_rows() {
            let fk_name = text(&name, row).unwrap_or_default();
            let key = match names.iter().position(|n| *n == fk_name) {
                Some(at) => at,
                None => {
                    names.push(fk_name);
                    names.len() - 1
                }
            };
            keys.push(CatalogForeignKey {
                key: i64::try_from(key).unwrap_or(i64::MAX),
                referenced_table: format!(
                    "{}.{}.{}",
                    text(&database, row).unwrap_or_default(),
                    text(&schema, row).unwrap_or_default(),
                    text(&table, row).unwrap_or_default()
                ),
                columns: vec![text(&referencing, row).unwrap_or_default()],
                referenced_columns: vec![text(&referenced, row).unwrap_or_default()],
            });
        }
    }
    Ok(CatalogAnswer::ForeignKeys(keys))
}

/// The data type a column's declared type fixes, as the reference types a
/// Snowflake column (`SnowTable._to_dtype`, `SNOW/table.py:330-383`):
/// `NUMBER` by the scale after its last comma, 0 an integer and any other,
/// or none that reads, a float; `FLOAT` a float; `VARCHAR` text; `BINARY`
/// binary; `BOOLEAN`; `DATE` and every `TIMESTAMP` a date, then `TIME` a
/// time; a `VECTOR` of `INT` or `FLOAT` a list of it; an `ARRAY` of a type
/// the list of its element's integers, floats or text; an untyped `ARRAY`,
/// another list, `DECFLOAT`, `VARIANT`, `OBJECT`, `MAP`, `FILE` and the geo
/// types unsupported; `None` for any other, typed by its values.
pub(crate) fn declared_dtype(declared: &str) -> Option<Dtype> {
    let declared = declared.trim().to_uppercase();
    if declared.starts_with("NUMBER") {
        let scale = declared
            .rsplit(',')
            .next()
            .and_then(|last| last.split(')').next())
            .and_then(|scale| scale.trim().parse::<i64>().ok());
        return Some(if scale == Some(0) {
            Dtype::Int
        } else {
            Dtype::Float
        });
    }
    if declared == "FLOAT" {
        return Some(Dtype::Float);
    }
    if declared.starts_with("VARCHAR") {
        return Some(Dtype::String);
    }
    if declared.starts_with("BINARY") {
        return Some(Dtype::Binary);
    }
    if declared == "BOOLEAN" {
        return Some(Dtype::Bool);
    }
    if declared.starts_with("DATE") || declared.starts_with("TIMESTAMP") {
        return Some(Dtype::Date);
    }
    if declared.starts_with("TIME") {
        return Some(Dtype::Time);
    }
    if declared.starts_with("VECTOR") {
        let element = declared
            .split(',')
            .next()
            .and_then(|head| head.split('(').nth(1))
            .map(str::trim);
        return Some(match element {
            Some("INT") => Dtype::Intlist,
            Some("FLOAT") => Dtype::Floatlist,
            _ => Dtype::Unsupported,
        });
    }
    if declared.starts_with("ARRAY") {
        let element = declared
            .split_once('(')
            .map(|(_, rest)| rest.rsplit_once(')').map_or(rest, |(inner, _)| inner));
        return Some(match element.and_then(declared_dtype) {
            Some(Dtype::Int) => Dtype::Intlist,
            Some(Dtype::Float) => Dtype::Floatlist,
            Some(Dtype::String) => Dtype::Stringlist,
            _ => Dtype::Unsupported,
        });
    }
    if [
        "DECFLOAT",
        "VARIANT",
        "OBJECT",
        "MAP",
        "FILE",
        "GEOGRAPHY",
        "GEOMETRY",
    ]
    .iter()
    .any(|word| declared.contains(word))
    {
        return Some(Dtype::Unsupported);
    }
    None
}

/// The connector's Arrow, read as the reference reads its frames
/// (`RE/base/table.py:1093-1186`, `RE/base/utils.py:97-98`): an integer
/// the connector narrows by its values (`int8` to `int32`) as `int64`; a
/// `NUMBER(p,0)` the connector hands over as a decimal as `int64` where
/// every value fits, left a decimal otherwise; a `NUMBER(p,s)` with a
/// fraction as a double; a zoned time (`TIMESTAMP_LTZ`, `_TZ`) at its UTC
/// instant with no zone. `None` where nothing changes.
fn conform(schema: &SchemaRef, batches: &[RecordBatch]) -> Result<Option<Batches>, CoreError> {
    let read: Vec<Option<DataType>> = schema
        .fields()
        .iter()
        .enumerate()
        .map(|(at, field)| match field.data_type() {
            DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32 => Some(DataType::Int64),
            DataType::Decimal128(_, 0) => batches
                .iter()
                .all(|b| {
                    b.column(at)
                        .as_primitive::<Decimal128Type>()
                        .iter()
                        .flatten()
                        .all(|v| i64::try_from(v).is_ok())
                })
                .then_some(DataType::Int64),
            DataType::Decimal128(_, _) => Some(DataType::Float64),
            DataType::Timestamp(unit, Some(_)) => Some(DataType::Timestamp(*unit, None)),
            _ => None,
        })
        .collect();
    if read.iter().all(Option::is_none) {
        return Ok(None);
    }
    let fields: Vec<Field> = schema
        .fields()
        .iter()
        .zip(&read)
        .map(|(field, to)| match to {
            Some(to) => field.as_ref().clone().with_data_type(to.clone()),
            None => field.as_ref().clone(),
        })
        .collect();
    let out = Arc::new(Schema::new_with_metadata(fields, schema.metadata().clone()));
    let mut conformed = Vec::with_capacity(batches.len());
    for batch in batches {
        let columns: Vec<ArrayRef> = batch
            .columns()
            .iter()
            .zip(&read)
            .map(|(column, to)| match to {
                Some(_) => converted(column),
                None => Arc::clone(column),
            })
            .collect();
        conformed.push(
            RecordBatch::try_new(Arc::clone(&out), columns).map_err(|_| {
                CoreError::new(
                    Sentence::Internal,
                    "a Snowflake answer's conformed columns do not make a batch",
                )
            })?,
        );
    }
    Ok(Some((out, conformed)))
}

/// `column` read as [`conform`] reads its type.
fn converted(column: &ArrayRef) -> ArrayRef {
    fn int64<I: Into<i64> + Copy>(values: impl Iterator<Item = Option<I>>) -> ArrayRef {
        Arc::new(values.map(|v| v.map(Into::into)).collect::<Int64Array>())
    }
    match column.data_type() {
        DataType::Int8 => int64(column.as_primitive::<Int8Type>().iter()),
        DataType::Int16 => int64(column.as_primitive::<Int16Type>().iter()),
        DataType::Int32 => int64(column.as_primitive::<Int32Type>().iter()),
        DataType::UInt8 => int64(column.as_primitive::<UInt8Type>().iter()),
        DataType::UInt16 => int64(column.as_primitive::<UInt16Type>().iter()),
        DataType::UInt32 => int64(column.as_primitive::<UInt32Type>().iter()),
        DataType::Decimal128(_, 0) => Arc::new(
            column
                .as_primitive::<Decimal128Type>()
                .iter()
                .map(|v| v.and_then(|v| i64::try_from(v).ok()))
                .collect::<Int64Array>(),
        ),
        DataType::Decimal128(_, scale) => Arc::new(
            column
                .as_primitive::<Decimal128Type>()
                .iter()
                .map(|v| v.and_then(|v| format!("{v}e-{scale}").parse::<f64>().ok()))
                .collect::<Float64Array>(),
        ),
        DataType::Timestamp(TimeUnit::Second, Some(_)) => Arc::new(
            column
                .as_primitive::<TimestampSecondType>()
                .clone()
                .with_timezone_opt(None::<String>),
        ),
        DataType::Timestamp(TimeUnit::Millisecond, Some(_)) => Arc::new(
            column
                .as_primitive::<TimestampMillisecondType>()
                .clone()
                .with_timezone_opt(None::<String>),
        ),
        DataType::Timestamp(TimeUnit::Microsecond, Some(_)) => Arc::new(
            column
                .as_primitive::<TimestampMicrosecondType>()
                .clone()
                .with_timezone_opt(None::<String>),
        ),
        DataType::Timestamp(TimeUnit::Nanosecond, Some(_)) => Arc::new(
            column
                .as_primitive::<TimestampNanosecondType>()
                .clone()
                .with_timezone_opt(None::<String>),
        ),
        _ => Arc::clone(column),
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::StringArray;

    use super::*;

    /// A text column.
    fn strings(values: &[Option<&str>]) -> ArrayRef {
        Arc::new(values.iter().copied().collect::<StringArray>())
    }

    #[test]
    fn a_name_is_quoted_in_double_quotes_doubled() {
        assert_eq!(identifier("a\"b"), "\"a\"\"b\"");
        assert_eq!(identifier("home.city"), "\"home.city\"");
        assert_eq!(
            rendered(&Relation::of(vec!["d".into(), "s".into(), "t".into()])),
            "\"d\".\"s\".\"t\""
        );
    }

    #[test]
    fn the_session_without_a_schema_is_its_database_alone() {
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("CURRENT_DATABASE()", DataType::Utf8, true),
                Field::new("CURRENT_SCHEMA()", DataType::Utf8, true),
            ])),
            vec![strings(&[Some("D")]), strings(&[None])],
        )
        .unwrap();
        assert_eq!(
            session(&[batch]).unwrap(),
            CatalogAnswer::Namespace(vec!["D".to_owned()])
        );
    }

    /// The binding exercised by the public Snowflake materialize journey.
    fn bound_keys(column: ArrayRef) -> Result<Vec<String>, CoreError> {
        use sdm_contract::data::{AttachmentId, KeySet};

        let column_type = ColumnType::from_arrow(SEED_KEY, column.data_type()).unwrap();
        let request = DataRequest::Lookup {
            table: Relation::of(vec!["d".into(), "s".into(), "users".into()]),
            key: vec!["user_id".into()],
            columns: vec!["user_id".into()],
            seeds: KeySet {
                attachment: AttachmentId(2),
                columns: vec![field(SEED_KEY, column_type, true)],
            },
        };
        let schema = Arc::new(Schema::new(vec![Field::new(
            SEED_KEY,
            column.data_type().clone(),
            true,
        )]));
        let batch = RecordBatch::try_new(Arc::clone(&schema), vec![column]).unwrap();
        key_rows(&request, &schema, &[batch], |value| format!("{value:?}"))
    }

    #[test]
    fn integer_keys_outside_the_snowflake_binding_range_are_invalid_requests() {
        use arrow_array::{Decimal128Array, UInt64Array};
        use sdm_contract::ErrorCode;

        for value in [
            i128::from(i64::MAX) + 1,
            i128::from(i64::MIN) - 1,
            10_i128.pow(25),
            -10_i128.pow(25),
        ] {
            let column = Decimal128Array::from(vec![Some(value), None])
                .with_precision_and_scale(38, 0)
                .unwrap();
            let error = bound_keys(Arc::new(column)).unwrap_err();
            assert_eq!(error.code, ErrorCode::InvalidRequest, "{value}");
            assert_eq!(error.sentence, Sentence::SnowflakeIntegerKeyRange);
        }
        let error = bound_keys(Arc::new(UInt64Array::from(vec![u64::MAX]))).unwrap_err();
        assert_eq!(error.code, ErrorCode::InvalidRequest);
        assert_eq!(error.sentence, Sentence::SnowflakeIntegerKeyRange);
    }

    #[test]
    fn signed_boundaries_in_range_decimals_unsigned_keys_and_nulls_keep_their_json() {
        use arrow_array::{Decimal128Array, UInt64Array};

        let decimal = Decimal128Array::from(vec![
            Some(i128::from(i64::MIN)),
            Some(0),
            Some(1),
            Some(i128::from(i64::MAX)),
            None,
        ])
        .with_precision_and_scale(38, 0)
        .unwrap();
        assert_eq!(
            bound_keys(Arc::new(decimal)).unwrap(),
            [
                i64::MIN.to_string(),
                "0".into(),
                "1".into(),
                i64::MAX.to_string(),
                "null".into(),
            ]
        );
        assert_eq!(
            bound_keys(Arc::new(UInt64Array::from(vec![
                Some(0),
                Some(u64::try_from(i64::MAX).unwrap()),
                None,
            ])))
            .unwrap(),
            ["0".to_owned(), i64::MAX.to_string(), "null".to_owned()]
        );
        let fraction = Decimal128Array::from(vec![Some(10), Some(-25), None])
            .with_precision_and_scale(38, 2)
            .unwrap();
        assert_eq!(
            bound_keys(Arc::new(fraction)).unwrap(),
            ["0.1", "-0.25", "null"]
        );
    }
}
