// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The Databricks dialect (T3.4): Databricks SQL as a SQL warehouse and a
//! notebook's Spark session both run it, through the shared builder's
//! hooks ([`crate::sql`], T3.18): only what is Databricks's own is here.
//!
//! Each rendering is the statement the reference's Databricks engine runs
//! for the same step (`rfm/backend/databricks` at the reference commit),
//! so each statement's result is that statement's (decision H9 gates on
//! results, not text), and the catalog statements are its text and
//! parameters:
//!
//! - names quoted with backticks, a backtick in one doubled (`quote_ident`);
//! - a relation of one to three parts (catalog, schema, table), a source's
//!   relations read under its namespace of a catalog and a schema (design
//!   fix C), Unity Catalog's names compared without regard to case;
//! - the catalog read through `<catalog>.information_schema`, the schema
//!   and table bound as parameters and folded with `lower(?)`, as Unity
//!   Catalog stores them (`DatabricksTable._get_source_columns`); the
//!   table keeps the name its caller gives it, which the host takes back;
//! - a column typed by its declaration first, as the reference types it
//!   (`_to_dtype`, [`declared_dtype`]);
//! - a key set bound as one parameter of JSON text, unpacked by
//!   `posexplode(from_json(?, '<type>'))`, the position the seed's batch
//!   (`_by_pkey`, `_by_fkey`, `_by_time`);
//! - an entity sample as `TABLESAMPLE (<p> PERCENT) REPEATABLE (<seed>)`,
//!   the percentage the reference's arithmetic over the table's rows,
//!   written as the reference's language writes a float
//!   ([`percent_text`], KO3-41);
//! - a neighbor read limited per seed with `QUALIFY ROW_NUMBER()`;
//! - a derived relation read as the reference reads a metric view's table
//!   (`_quoted_source_name`).
//!
//! Where the reference fetches rows and finishes in its own process (a
//! key lookup's duplicates removed after the fetch) the statement does the
//! same in SQL, so the answer is the contract's.
//!
//! A key set's binding (T3.28): [`key_rows`] writes each seed as the
//! old client's JSON writes it (`_dumps`, ASCII only), the batch
//! column left out (the position in the JSON array is the batch, the seeds
//! being in batch order from 0), a key as the reference coerces it
//! (`_coerce_id`: an integer, a double, else text), each seed time as its
//! wall clock's text to the second (`%Y-%m-%d %H:%M:%S`, as the reference
//! writes its bounds): a key set of one column a JSON array of its values,
//! of several a JSON array of objects keyed by column name. Past the
//! reference's budget of the warehouse's parameter size the key set is
//! split as the reference splits it ([`crate::key_text`]), each later
//! chunk's statement adding the seeds before it to `pos`, as the
//! reference's `_flatten` moves each chunk's batches.

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::TimestampNanosecondType;
use arrow_array::{Array, ArrayRef, Float64Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field as ArrowField, Schema, SchemaRef};
use sdm_contract::catalog::{CatalogAnswer, CatalogColumn, CatalogForeignKey, CatalogKey, KeyKind};
use sdm_contract::data::{
    DataRequest, DescribePart, Determinism, ExpectedSchema, MARKED, MATCHES_COLUMN, NullOrder,
    ProviderCapabilities, Relation, SEED_BATCH, SEED_COUNT_UNREAD, SEED_KEY, SEED_KEY_UNIQUE,
    SqlParam, TextTimes, TieTerm, refused, seed_integer, views_unread,
};
use sdm_contract::ops::Dtype;
use sdm_contract::types::ColumnType;
use sdm_contract::{CoreError, Detail, Scalar, Sentence};

use crate::json;
use crate::key_text;
use crate::sql::{self, NameCase, Sample, SeedRead, SqlDialect, not_taken};

/// The dialect's name on the wire.
pub(crate) const NAME: &str = "databricks";

/// The most name parts a Databricks relation takes: a catalog, a schema
/// and a table (Unity Catalog's three-level namespace).
const MOST_PARTS: usize = 3;

/// The namespace parts a Databricks source takes: a catalog and a schema.
const NAMESPACE_DEPTH: usize = 2;

/// The alias a derived relation reads its source under, the reference's
/// (`metric_view.SOURCE_ALIAS`), which a view's expressions may qualify
/// their columns by.
const SOURCE_ALIAS: &str = "source";

/// The reference's oversampling of a sampled percentage
/// (`_SAMPLE_OVERSAMPLE`): `TABLESAMPLE (p PERCENT)` draws each row alone,
/// so the count drawn varies, and the not-null filters after it drop more.
const OVERSAMPLE: f64 = 3.0;

/// The name a statement reads its unpacked key set by, the reference's.
const TMP: &str = "TMP";

/// The Databricks dialect.
pub(crate) struct Databricks;

impl SqlDialect for Databricks {
    fn name(&self) -> &'static str {
        NAME
    }

    /// What a Databricks source answers through this layer.
    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            opaque_vendor: None,
            compares_stored_times: false,
            samples_by_key_dtype: false,
            reads_nan_as_missing: false,
            optional: sql::rendered(),
            // A key set past the warehouse's parameter size is split into
            // several fetches, as the reference's `_chunk_rows` splits it
            // ([`crate::key_text`]), so the core sends any number.
            max_keys_per_request: None,
            // `TABLESAMPLE ... REPEATABLE (seed)` draws the same rows of
            // the same table (register entry RS-6, KO3-41).
            sampling: Determinism::SeededStable,
            // The reference writes the seed into `REPEATABLE` as it is.
            max_seed: u64::MAX,
            // The reference's engine reads a SQL source's text times one by
            // one (register DT-7).
            text_times: Some(TextTimes::ValueByValue),
            // Databricks orders a null first ascending, the direction of
            // every term whose column may hold one: the time term,
            // descending, never meets a null, since the window that comes
            // with it leaves them out (register ORD-2).
            null_order: Some(NullOrder::First),
            // The reference's Databricks sampler neither reads indexes nor
            // warns of missing ones.
            index_advice: None,
            fallback: None,
            derived_relations: true,
            sample_rows: true,
            neighbor_lookback: true,
            // The reference carries on without keys its catalog fails to
            // read (`DatabricksTable`'s key reads).
            keys_optional: true,
        }
    }

    fn most_parts(&self) -> usize {
        MOST_PARTS
    }

    /// A catalog and a schema (design fix C).
    fn namespace_depth(&self) -> usize {
        NAMESPACE_DEPTH
    }

    /// Each part read, as the reference reads `current_catalog()` and
    /// `current_schema()` for those a source leaves out and names every
    /// table in full, as Unity Catalog names a table a foreign key refers
    /// to.
    fn settles_namespace(&self) -> bool {
        true
    }

    /// Unity Catalog matches a name without regard to case and stores its
    /// catalogs, schemas and tables in lower case.
    fn name_case(&self) -> NameCase {
        NameCase::Insensitive
    }

    /// In backticks, each one in it doubled (`quote_ident(name, '`')`).
    fn identifier(&self, name: &str) -> String {
        format!("`{}`", name.replace('`', "``"))
    }

    /// A derived relation as the reference reads a table
    /// (`_quoted_source_name`): its columns, each the source's column under
    /// the relation's name for it or an expression, from the source under
    /// [`SOURCE_ALIAS`], where its filter holds, left unaliased
    /// (Databricks refuses `(...) AS x TABLESAMPLE`). The columns a catalog
    /// read found, each under its own name with no filter, are read so only
    /// where one is named like a session word ([`SESSION_WORDS`]), which
    /// Databricks reads, quoted or not, as the session's value once the
    /// column is dropped or renamed; otherwise the source by its name.
    fn derived(&self, relation: &Relation, source: &str) -> Option<String> {
        if relation.found
            && !relation
                .columns
                .iter()
                .any(|c| SESSION_WORDS.contains(&c.name.to_lowercase().as_str()))
        {
            return Some(source.to_owned());
        }
        let columns = if relation.columns.is_empty() {
            "*".to_owned()
        } else {
            relation
                .columns
                .iter()
                .map(|column| match (&column.column, &column.expression) {
                    (Some(read), _) if *read == column.name => {
                        format!("{SOURCE_ALIAS}.{}", self.identifier(read))
                    }
                    (Some(read), _) => format!(
                        "{SOURCE_ALIAS}.{} AS {}",
                        self.identifier(read),
                        self.identifier(&column.name)
                    ),
                    (None, Some(expression)) => {
                        format!("({expression}) AS {}", self.identifier(&column.name))
                    }
                    // Refused before (`refuse_derived`).
                    (None, None) => String::new(),
                })
                .collect::<Vec<_>>()
                .join(", ")
        };
        let filter = relation
            .filter
            .as_ref()
            .map(|filter| format!(" WHERE ({filter})"))
            .unwrap_or_default();
        Some(format!(
            "(SELECT {columns} FROM {source} AS {SOURCE_ALIAS}{filter})"
        ))
    }

    fn seeds_for(&self, _request: &DataRequest, _table: &Relation) -> String {
        TMP.to_owned()
    }

    /// The unpacked key set, `TMP`: the reference's `posexplode(from_json(?,
    /// ...))`, its position the batch, its key read as the reference reads
    /// a key by its data type (`_elem_type`), each seed time as text.
    fn seeds_prelude(&self, read: &SeedRead<'_>) -> Result<String, CoreError> {
        seed_cte(self, read)
    }

    fn seeds_relation(&self, name: &str) -> String {
        name.to_owned()
    }

    /// One term of a neighbor order, its null order written only where it
    /// is not Databricks' own for the direction (first ascending, last
    /// descending); a time term orders its column as it is, with none, as
    /// the reference's `_neighbor_order_by` does: its column holds no null
    /// in the rows it orders (register ORD-2).
    fn order_term(&self, term: &TieTerm, alias: &str, _windowed: Option<&str>) -> String {
        let column = format!("{alias}{}", self.identifier(&term.column));
        let direction = if term.descending { " DESC" } else { "" };
        let nulls = match (term.as_timestamp, term.descending, term.nulls_last) {
            (true, _, _) => "",
            (false, false, true) => " NULLS LAST",
            (false, true, false) => " NULLS FIRST",
            _ => "",
        };
        format!("{column}{direction}{nulls}")
    }

    /// The reference's `_get_num_rows`, as it writes it.
    fn count_rows(&self, table: &str, filtered: &str) -> String {
        format!("SELECT COUNT(*) FROM {table}{filtered}")
    }

    fn sample_seed(&self, request: &DataRequest, seed: &Scalar) -> Result<i128, CoreError> {
        seed_integer(seed).ok_or_else(|| refused(request, "seed", "a sample's seed is an integer"))
    }

    /// The reference's sample of an entity table (`_sample_entity_table`):
    /// the columns asked for, from a `TABLESAMPLE` of the percentage the
    /// reference computes over the table's rows, repeatable by the seed,
    /// where each not-null filter holds, the first `n`.
    fn sample(&self, sample: &Sample<'_>) -> Vec<String> {
        vec![format!(
            "SELECT {}\nFROM {}\nTABLESAMPLE ({} PERCENT) REPEATABLE ({}){}\nLIMIT {}",
            sample.projected,
            sample.table,
            percent_text(sample_percent(sample.n, sample.rows)),
            sample.seed,
            sample.filtered,
            sample.n
        )]
    }

    /// The reference's `_by_fkey`: each seed's rows within its window, the
    /// first `limit` in `terms`' order (`QUALIFY ROW_NUMBER()`), the result
    /// in `(batch, order)`.
    fn neighbors_per_seed(&self, read: &SeedRead<'_>, limit: u64, terms: &[TieTerm]) -> String {
        let order = terms
            .iter()
            .map(|t| self.order_term(t, "t.", None))
            .collect::<Vec<_>>()
            .join(", ");
        let batch = format!("s.{}", self.identifier(SEED_BATCH));
        format!(
            "{}\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY {batch}\n  ORDER BY {order}\n) <= {limit}\nORDER BY {batch}, {order}",
            read.joined()
        )
    }

    /// A limit per seed comes with an order, and an order with a limit.
    fn neighbors_unpaired(&self, read: &SeedRead<'_>) -> Result<String, CoreError> {
        Err(refused(
            read.request,
            "order",
            "a limit per seed comes with an order, and an order with a limit",
        ))
    }

    /// The reference's key lookup (`_by_pkey`): the rows holding a seed's
    /// key, with the seed's batch position; where the source keeps the key
    /// unique, the plain join the reference sends; otherwise one row per
    /// seed, as the reference keeps the first it read of a key several rows
    /// hold, with how many rows hold it unless the seeds say that count is
    /// not read.
    fn lookup(&self, read: &SeedRead<'_>) -> String {
        let flagged = |column: &str, flag: &str| {
            read.seeds.columns.iter().any(|c| {
                c.name == column && c.metadata.get(flag).map(String::as_str) == Some(MARKED)
            })
        };
        let joined = read.joined();
        if flagged(SEED_KEY, SEED_KEY_UNIQUE) {
            return joined;
        }
        let batch = format!("s.{}", self.identifier(SEED_BATCH));
        let counted = if flagged(SEED_BATCH, SEED_COUNT_UNREAD) {
            String::new()
        } else {
            format!(
                ", COUNT(*) OVER (PARTITION BY {batch}) AS {}",
                self.identifier(MATCHES_COLUMN)
            )
        };
        let (select, rest) = joined.split_once('\n').unwrap_or((&joined, ""));
        let keyed = read
            .key
            .iter()
            .map(|k| format!("t.{}", self.identifier(k)))
            .collect::<Vec<_>>()
            .join(", ");
        format!(
            "{select}{counted}\n{rest}\nQUALIFY ROW_NUMBER() OVER (PARTITION BY {batch} ORDER BY {keyed}) = 1"
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

    fn current_namespace(
        &self,
        request: &DataRequest,
        namespace: &[Option<String>],
    ) -> Result<(String, Vec<SqlParam>), CoreError> {
        current_namespace(request, &[], namespace)
    }

    /// What the contract fixes for the key counts; what the source answers
    /// for the catalog's reads, which [`read`] reads by position, and for
    /// the time bounds, in the column's own type, as the reference reads
    /// them; the namespace's names, as the contract writes them.
    fn expected(&self, request: &DataRequest) -> ExpectedSchema {
        match request {
            DataRequest::Describe {
                part: DescribePart::Namespace,
                ..
            } => request.expected(),
            DataRequest::Describe { .. } | DataRequest::TimeBounds { .. } => {
                ExpectedSchema::Discover {}
            }
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

    fn key_rows(
        &self,
        schema: &SchemaRef,
        batches: &[RecordBatch],
    ) -> Result<Option<Vec<String>>, CoreError> {
        key_rows(schema, batches).map(Some)
    }

    /// The reference's budget of a key set's JSON text, below the
    /// warehouse's native parameter limit of 1 MB (`_chunk_rows`).
    fn key_budget(&self) -> Option<usize> {
        Some(key_text::BUDGET)
    }

    /// The statement with its key set's positions moved by `offset`, the
    /// seeds before its chunk (`pos + <offset>`), as the reference's
    /// `_flatten` moves a chunk's batches.
    fn chunk_text(&self, text: &str, offset: usize) -> Result<String, CoreError> {
        if offset == 0 {
            return Ok(text.to_owned());
        }
        let head = |offset| format!("WITH {TMP} AS (\n  SELECT {}", batch_select(offset));
        text.strip_prefix(&head(0))
            .map(|rest| format!("{}{rest}", head(offset)))
            .ok_or_else(|| {
                CoreError::new(
                    Sentence::Internal,
                    "a statement binding a key set does not open with its key set",
                )
            })
    }
}

/// `name` as a Databricks identifier.
fn identifier(name: &str) -> String {
    Databricks.identifier(name)
}

/// A text parameter.
fn text_param(value: &str) -> SqlParam {
    SqlParam::Value {
        value: Scalar::Utf8(value.to_owned()),
    }
}

/// The catalog a relation's catalog reads go through (`<catalog>.`, the
/// catalog quoted, or nothing for the source's current one) and the schema
/// and table its rows are found by, each bound as a parameter, the schema
/// `None` where only the source's current one names it.
struct CatalogTarget {
    catalog: String,
    schema: Option<String>,
    table: String,
}

impl CatalogTarget {
    /// The relation `relation` names, already read under the source's
    /// namespace (design fix C); a derived one's catalog is its source
    /// table's, which the core never asks of the relation.
    fn of(request: &DataRequest, relation: &Relation) -> Result<CatalogTarget, CoreError> {
        if relation.is_derived() {
            return Err(not_taken(
                request,
                "tables",
                "a derived relation's catalog is read through its source table",
            ));
        }
        Ok(match relation.source.as_slice() {
            [catalog, schema, table] => CatalogTarget {
                catalog: format!("{}.", identifier(catalog)),
                schema: Some(schema.clone()),
                table: table.clone(),
            },
            [schema, table] => CatalogTarget {
                catalog: String::new(),
                schema: Some(schema.clone()),
                table: table.clone(),
            },
            [table] => CatalogTarget {
                catalog: String::new(),
                schema: None,
                table: table.clone(),
            },
            _ => {
                return Err(CoreError::new(
                    Sentence::Internal,
                    "a relation past the parts the dialect takes reached its catalog read",
                ));
            }
        })
    }

    /// The condition on `alias`'s schema and table, and its parameters:
    /// `lower(?)` each, as the reference folds the caller's spelling to
    /// Unity Catalog's; the source's current schema where none is named.
    fn condition(&self, alias: &str) -> (String, Vec<SqlParam>) {
        // Folded as the source compares names (design fix C): Unity
        // Catalog's lower case, so the caller's spelling finds the table.
        let folded = |value: &str| match Databricks.name_case() {
            NameCase::Insensitive => format!("lower({value})"),
            _ => value.to_owned(),
        };
        let mut params = Vec::with_capacity(2);
        let schema = match &self.schema {
            Some(schema) => {
                params.push(text_param(schema));
                folded("?")
            }
            None => folded("current_schema()"),
        };
        params.push(text_param(&self.table));
        (
            format!(
                "WHERE {alias}.table_schema = {schema}\n  AND {alias}.table_name = {}",
                folded("?")
            ),
            params,
        )
    }
}

/// The reference's catalog reads (`DatabricksTable`, `Graph.from_databricks`),
/// as it writes them, with their parameters; `namespace` the parts of the
/// source's namespace given, outermost first.
fn describe(
    request: &DataRequest,
    tables: &[Relation],
    part: DescribePart,
    namespace: &[String],
) -> Result<(String, Vec<SqlParam>), CoreError> {
    let one = || CatalogTarget::of(request, sql::one_table(request, tables)?);
    Ok(match part {
        // No view is read here (T3.27): only a dialect that reads views answers it.
        DescribePart::ViewDefinition => return Err(views_unread()),
        // The source's namespace, one row per part, outermost first, each
        // given or the session's current one.
        DescribePart::Namespace => {
            let given: Vec<Option<String>> = namespace.iter().cloned().map(Some).collect();
            current_namespace(request, tables, &given)?
        }
        // `Graph.from_databricks`'s listing, as it writes it, in the
        // namespace's catalog and schema; the session's current ones where
        // the source names none.
        DescribePart::Tables if tables.is_empty() => {
            let pad = " ".repeat(20);
            let (catalog, schema, params) = match namespace {
                [catalog, schema] => (
                    format!("{}.", identifier(catalog)),
                    "?".to_owned(),
                    vec![text_param(schema)],
                ),
                [schema] => (String::new(), "?".to_owned(), vec![text_param(schema)]),
                _ => (String::new(), "current_schema()".to_owned(), vec![]),
            };
            (
                format!(
                    "\n{pad}SELECT table_name\n{pad}FROM {catalog}information_schema.tables\n{pad}WHERE table_schema = {schema}\n{pad}  AND table_type != 'METRIC_VIEW'\n{pad}"
                ),
                params,
            )
        }
        // Whether the source holds each table named, as it would read the
        // name, a view included: asked only when reading a table failed.
        DescribePart::Tables => {
            let mut selects = Vec::with_capacity(tables.len());
            let mut params = Vec::new();
            for table in tables {
                let target = CatalogTarget::of(request, table)?;
                let (condition, bound) = target.condition("t");
                selects.push(format!(
                    "SELECT t.table_name\nFROM {}information_schema.tables t\n{condition}",
                    target.catalog
                ));
                params.extend(bound);
            }
            (selects.join("\nUNION ALL\n"), params)
        }
        DescribePart::Columns => {
            let target = one()?;
            let catalog = &target.catalog;
            let (condition, params) = target.condition("c");
            (
                format!(
                    "SELECT c.column_name, c.full_data_type, c.is_nullable,\n       c.table_catalog, c.table_schema, c.table_name,\n       t.table_type\nFROM {catalog}information_schema.columns c\nJOIN {catalog}information_schema.tables t\n  ON t.table_catalog = c.table_catalog\n AND t.table_schema = c.table_schema\n AND t.table_name = c.table_name\n{condition}\nORDER BY c.ordinal_position"
                ),
                params,
            )
        }
        DescribePart::Keys => {
            let target = one()?;
            let catalog = &target.catalog;
            let (condition, params) = target.condition("tc");
            (
                format!(
                    "SELECT tc.constraint_type, tc.constraint_name,\n       kcu.column_name\nFROM {catalog}information_schema.table_constraints tc\nJOIN {catalog}information_schema.key_column_usage kcu\n  ON tc.constraint_catalog = kcu.constraint_catalog\n AND tc.constraint_schema = kcu.constraint_schema\n AND tc.constraint_name = kcu.constraint_name\n{condition}\n  AND tc.constraint_type IN ('PRIMARY KEY', 'UNIQUE')"
                ),
                params,
            )
        }
        DescribePart::ForeignKeys => {
            let target = one()?;
            let catalog = &target.catalog;
            let (condition, params) = target.condition("tc");
            let condition = condition.replacen("WHERE ", "  AND ", 1);
            (
                format!(
                    "SELECT tc.constraint_name, kcu.column_name,\n       ccu.table_catalog, ccu.table_schema,\n       ccu.table_name, ccu.column_name\nFROM {catalog}information_schema.table_constraints tc\nJOIN {catalog}information_schema.referential_constraints rc\n  ON tc.constraint_catalog = rc.constraint_catalog\n AND tc.constraint_schema = rc.constraint_schema\n AND tc.constraint_name = rc.constraint_name\nJOIN {catalog}information_schema.key_column_usage kcu\n  ON tc.constraint_catalog = kcu.constraint_catalog\n AND tc.constraint_schema = kcu.constraint_schema\n AND tc.constraint_name = kcu.constraint_name\nJOIN {catalog}information_schema.constraint_column_usage ccu\n  ON rc.unique_constraint_catalog = ccu.constraint_catalog\n AND rc.unique_constraint_schema = ccu.constraint_schema\n AND rc.unique_constraint_name = ccu.constraint_name\nWHERE tc.constraint_type = 'FOREIGN KEY'\n{condition}"
                ),
                params,
            )
        }
        DescribePart::Indexes | DescribePart::IndexColumns => {
            return Err(not_taken(
                request,
                "part",
                "Databricks keeps no index the reference reads",
            ));
        }
    })
}

/// The unpacked key set, `TMP`, as the reference unpacks it: `pos` the
/// batch, the key read as `element_type` reads it, and each seed time a
/// bound of the window names, as text, under the seed column's own name,
/// so the shared join reads them as it reads any dialect's seeds.
fn seed_cte(d: &Databricks, read: &SeedRead<'_>) -> Result<String, CoreError> {
    let [_] = read.key else {
        return Err(not_taken(
            read.request,
            "key",
            "a key of several columns is read as the reference reads none",
        ));
    };
    let Some(key) = read.seeds.columns.iter().find(|c| c.name == SEED_KEY) else {
        return Err(refused(
            read.request,
            "seeds",
            "the seeds lack a key column",
        ));
    };
    let element = element_type(&key.column_type);
    let times: Vec<&str> = read
        .seeds
        .columns
        .iter()
        .filter(|c| c.name != SEED_KEY && c.name != SEED_BATCH)
        .map(|c| c.name.as_str())
        .collect();
    for time in &times {
        let named = read.window.is_some_and(|w| {
            [&w.lower, &w.upper]
                .into_iter()
                .flatten()
                .any(|b| b.seed_column == *time)
        });
        if !named {
            return Err(not_taken(
                read.request,
                "seeds",
                "a seed column past the key, the batch and the window's bounds",
            ));
        }
    }
    let batch = batch_select(0);
    let (schema, selects) = if times.is_empty() {
        (
            format!("array<{element}>"),
            format!("{batch}, col AS {}", d.identifier(SEED_KEY)),
        )
    } else {
        let mut fields = vec![format!("{}:{element}", d.identifier(SEED_KEY))];
        let mut selects = vec![batch, format!("col.{0} AS {0}", d.identifier(SEED_KEY))];
        for time in &times {
            fields.push(format!("{}:string", d.identifier(time)));
            selects.push(format!("col.{0} AS {0}", d.identifier(time)));
        }
        (
            format!("array<struct<{}>>", fields.join(",")),
            selects.join(", "),
        )
    };
    Ok(format!(
        "WITH {TMP} AS (\n  SELECT {selects}\n  FROM posexplode(from_json(?, '{schema}'))\n)\n"
    ))
}

/// The unpacked key set's batch, its position in the JSON array moved by
/// `offset`, the seeds of the chunks before its own.
fn batch_select(offset: usize) -> String {
    let batch = identifier(SEED_BATCH);
    if offset == 0 {
        format!("pos AS {batch}")
    } else {
        format!("pos + {offset} AS {batch}")
    }
}

/// A Databricks answer to a `describe` of `part`, read as the reference
/// reads it, each column by its position in the statement's select list.
/// The source's namespace, one row per part, outermost first: each part
/// `namespace` gives, the parts aligned to the innermost (a catalog and a
/// schema, or a schema alone), or the session's current one where it
/// gives none (`current_catalog()`, `current_schema()`), as the reference
/// reads the parts a source leaves out (`SELECT current_catalog(),
/// current_schema()`).
fn current_namespace(
    request: &DataRequest,
    tables: &[Relation],
    namespace: &[Option<String>],
) -> Result<(String, Vec<SqlParam>), CoreError> {
    sql::whole(request, tables)?;
    let mut params = Vec::new();
    let mut parts = Vec::new();
    for (at, current) in ["current_catalog()", "current_schema()"].iter().enumerate() {
        let given = (namespace.len() + at)
            .checked_sub(NAMESPACE_DEPTH)
            .and_then(|i| namespace.get(i))
            .and_then(Option::as_ref);
        match given {
            Some(part) => {
                params.push(text_param(part));
                parts.push(format!("SELECT {at} AS o, ? AS n"));
            }
            None => parts.push(format!("SELECT {at} AS o, {current} AS n")),
        }
    }
    Ok((
        format!(
            "SELECT p.n AS {name}\nFROM ({parts}) p\nWHERE p.n IS NOT NULL\nORDER BY p.o",
            name = identifier(sdm_contract::data::catalog_columns::NAME),
            parts = parts.join(" UNION ALL "),
        ),
        params,
    ))
}

pub(crate) fn read(
    request: &DataRequest,
    part: DescribePart,
    batches: &[RecordBatch],
) -> Result<CatalogAnswer, CoreError> {
    match part {
        // No view is read here (T3.27): only a dialect that reads views answers it.
        DescribePart::ViewDefinition => Err(views_unread()),
        DescribePart::Namespace => {
            crate::read::namespace(batches, sdm_contract::data::catalog_columns::NAME)
        }
        DescribePart::Tables => {
            let mut names = Vec::new();
            for batch in batches {
                let column = text_at(batch, 0)?;
                names.extend((0..batch.num_rows()).filter_map(|row| value(&column, row)));
            }
            Ok(CatalogAnswer::Tables(names))
        }
        DescribePart::Columns => read_columns(request, batches),
        DescribePart::Keys => {
            // Single-column keys only, each constraint by its name, as the
            // reference keeps them (no composite support): a primary key a
            // primary and unique key, a unique one unique.
            let mut constraints: BTreeMap<String, (String, Vec<String>)> = BTreeMap::new();
            for batch in batches {
                let kinds = text_at(batch, 0)?;
                let names = text_at(batch, 1)?;
                let columns = text_at(batch, 2)?;
                for row in 0..batch.num_rows() {
                    let entry = constraints
                        .entry(value(&names, row).unwrap_or_default())
                        .or_insert_with(|| (value(&kinds, row).unwrap_or_default(), vec![]));
                    entry.1.push(value(&columns, row).unwrap_or_default());
                }
            }
            Ok(CatalogAnswer::Keys(
                constraints
                    .into_values()
                    .filter(|(_, columns)| columns.len() == 1)
                    .map(|(kind, columns)| CatalogKey {
                        kind: match kind.trim().to_uppercase().as_str() {
                            "PRIMARY KEY" => KeyKind::Primary,
                            _ => KeyKind::Unique,
                        },
                        columns,
                        index: None,
                        enforced: false,
                    })
                    .collect(),
            ))
        }
        DescribePart::ForeignKeys => {
            // Single-column foreign keys only, one row per constraint, the
            // table referred to named `catalog.schema.table`.
            let mut keys: Vec<(String, Vec<CatalogForeignKey>)> = Vec::new();
            for batch in batches {
                let names = text_at(batch, 0)?;
                let columns = text_at(batch, 1)?;
                let catalogs = text_at(batch, 2)?;
                let schemas = text_at(batch, 3)?;
                let tables = text_at(batch, 4)?;
                let referenced = text_at(batch, 5)?;
                for row in 0..batch.num_rows() {
                    let name = value(&names, row).unwrap_or_default();
                    let key = CatalogForeignKey {
                        key: 0,
                        referenced_table: format!(
                            "{}.{}.{}",
                            value(&catalogs, row).unwrap_or_default(),
                            value(&schemas, row).unwrap_or_default(),
                            value(&tables, row).unwrap_or_default()
                        ),
                        columns: vec![value(&columns, row).unwrap_or_default()],
                        referenced_columns: vec![value(&referenced, row).unwrap_or_default()],
                    };
                    match keys.iter_mut().find(|(n, _)| *n == name) {
                        Some((_, rows)) => rows.push(key),
                        None => keys.push((name, vec![key])),
                    }
                }
            }
            Ok(CatalogAnswer::ForeignKeys(
                keys.into_iter()
                    .filter(|(_, rows)| rows.len() == 1)
                    .zip(0_i64..)
                    .map(|((_, mut rows), key)| {
                        let mut row = rows.remove(0);
                        row.key = key;
                        row
                    })
                    .collect(),
            ))
        }
        DescribePart::Indexes | DescribePart::IndexColumns => Err(not_taken(
            request,
            "part",
            "Databricks keeps no index the reference reads",
        )),
    }
}

/// The words Databricks reads without parentheses as the session's
/// identity or clock (the reference's `view_expr._SESSION_WORDS`).
const SESSION_WORDS: &[&str] = &[
    "current_catalog",
    "current_database",
    "current_date",
    "current_path",
    "current_role",
    "current_schema",
    "current_time",
    "current_timestamp",
    "current_user",
    "localtime",
    "localtimestamp",
    "session_user",
    "user",
];

/// A table's columns, in order: each its name, its full declared type and
/// whether it is declared not null (`is_nullable` other than `NO`, read as
/// the reference reads it); a metric view refused by name, as the reference refuses one
/// read as a table (`MetricViewAsTableError`).
fn read_columns(
    request: &DataRequest,
    batches: &[RecordBatch],
) -> Result<CatalogAnswer, CoreError> {
    let mut found = Vec::new();
    for batch in batches {
        let names = text_at(batch, 0)?;
        let types = text_at(batch, 1)?;
        let nullable = text_at(batch, 2)?;
        let tables = text_at(batch, 5)?;
        let kinds = text_at(batch, 6)?;
        for row in 0..batch.num_rows() {
            if found.is_empty() && value(&kinds, row).as_deref() == Some("METRIC_VIEW") {
                let mut refusal = CoreError::new(
                    Sentence::MetricViewAsTable,
                    "a metric view is read as a table; its measures have no value per row",
                )
                .with_field("request", Detail::text(request.name()));
                // Named as the caller named it, as the reference names it.
                let named = match request {
                    DataRequest::Describe { tables: asked, .. } => asked
                        .first()
                        .and_then(|relation| relation.source.last().cloned()),
                    _ => None,
                };
                if let Some(table) = named.or_else(|| value(&tables, row)) {
                    refusal = refusal.with_field("metric_view", Detail::text(table));
                }
                return Err(refusal);
            }
            found.push(CatalogColumn {
                name: value(&names, row).unwrap_or_default(),
                declared_type: value(&types, row).unwrap_or_default(),
                not_null: value(&nullable, row).is_some_and(|n| n.trim().to_uppercase() == "NO"),
                primary_key: false,
                key_nullable: false,
                unique: false,
                dtype: value(&types, row).and_then(|declared| declared_dtype(&declared)),
            });
        }
    }
    Ok(CatalogAnswer::Columns(found))
}

/// The data type a column's declared type fixes, as the reference types a
/// Databricks column by its declaration (`DatabricksTable._to_dtype`): the
/// integer types an integer; `float`, `real` and `double` a float; a
/// decimal of scale 0 an integer and of any other scale (or one whose
/// scale does not read) a float; a boolean; text; binary; a date or
/// timestamp of any kind a date; an array of integers, floats or text the
/// list of its element's type, any other array unsupported; an interval,
/// map, struct, variant, object, geography, geometry or void unsupported;
/// `None` for any other declaration, typed by its values.
pub(crate) fn declared_dtype(declared: &str) -> Option<Dtype> {
    let declared = declared.trim().to_lowercase();
    if [
        "tinyint", "byte", "smallint", "short", "int", "integer", "bigint", "long",
    ]
    .contains(&declared.as_str())
    {
        return Some(Dtype::Int);
    }
    if ["float", "real", "double"].contains(&declared.as_str()) {
        return Some(Dtype::Float);
    }
    if declared.starts_with("decimal")
        || declared.starts_with("numeric")
        || declared.starts_with("dec")
    {
        let scale = declared
            .find('(')
            .zip(declared.find(')'))
            .filter(|(open, close)| open < close)
            .and_then(|(open, close)| {
                let parts: Vec<&str> = declared[open + 1..close].split(',').collect();
                match parts.as_slice() {
                    [_] => Some(0),
                    [_, scale, ..] => scale.trim().parse::<i64>().ok(),
                    [] => None,
                }
            });
        return Some(match scale {
            Some(0) => Dtype::Int,
            _ => Dtype::Float,
        });
    }
    if declared == "boolean" || declared == "bool" {
        return Some(Dtype::Bool);
    }
    if declared.starts_with("string")
        || declared.starts_with("varchar")
        || declared.starts_with("char")
    {
        return Some(Dtype::String);
    }
    if declared.starts_with("binary") {
        return Some(Dtype::Binary);
    }
    if declared == "date" || declared.starts_with("timestamp") {
        return Some(Dtype::Date);
    }
    if declared.starts_with("array") {
        let inner = declared
            .find('<')
            .zip(declared.rfind('>'))
            .filter(|(open, close)| open < close)
            .map(|(open, close)| &declared[open + 1..close]);
        return Some(match inner.and_then(declared_dtype) {
            Some(Dtype::Int) => Dtype::Intlist,
            Some(Dtype::Float) => Dtype::Floatlist,
            Some(Dtype::String) => Dtype::Stringlist,
            _ => Dtype::Unsupported,
        });
    }
    if [
        "interval",
        "map",
        "struct",
        "variant",
        "object",
        "geography",
        "geometry",
        "void",
    ]
    .iter()
    .any(|word| declared.contains(word))
    {
        return Some(Dtype::Unsupported);
    }
    None
}

fn unread(at: usize) -> CoreError {
    CoreError::new(
        Sentence::Internal,
        "a catalog answer's column is missing or not text",
    )
    .with_field("position", Detail::text(at.to_string()))
}

/// The column at `at` of `batch` as text: a string column, or a column of
/// nulls (a current schema a session has none of).
fn text_at(batch: &RecordBatch, at: usize) -> Result<StringArray, CoreError> {
    let column = batch.columns().get(at).ok_or_else(|| unread(at))?;
    if let Some(strings) = column.as_string_opt::<i32>() {
        return Ok(strings.clone());
    }
    if let Some(strings) = column.as_string_opt::<i64>() {
        return Ok(strings.iter().collect());
    }
    if let Some(strings) = column.as_string_view_opt() {
        return Ok(strings.iter().collect());
    }
    if *column.data_type() == DataType::Null {
        return Ok(StringArray::new_null(column.len()));
    }
    Err(unread(at))
}

fn value(strings: &StringArray, row: usize) -> Option<String> {
    (!strings.is_null(row)).then(|| strings.value(row).to_owned())
}

/// The percentage of a table's rows the reference samples to draw `n`:
/// `min(100.0, 100.0 * n / rows * 3.0)` where the table holds more than
/// `n` rows, each step a double as the reference's arithmetic rounds it,
/// and `100.0` otherwise, or where the rows are not known.
#[allow(
    clippy::cast_precision_loss,
    reason = "an integer to the nearest double, as the reference computes it"
)]
pub(crate) fn sample_percent(n: u64, rows: Option<u64>) -> f64 {
    match rows {
        Some(rows) if rows > n => (100.0 * n as f64 / rows as f64 * OVERSAMPLE).min(100.0),
        _ => 100.0,
    }
}

/// The text the reference's f-string writes a finite double as (its `str`,
/// KO3-41): the shortest decimal that reads back as the value, the nearest
/// of that many digits, a tie to the even; positional with at least one
/// fraction digit (`100.0`, `0.30000000000000004`) from `1e-4` up to below
/// `1e16`, otherwise one digit, the rest as a fraction, and a signed
/// exponent of at least two digits (`1e-05`, `1.5e+16`).
pub(crate) fn percent_text(x: f64) -> String {
    if x.is_nan() {
        return "nan".to_owned();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf" } else { "-inf" }.to_owned();
    }
    let sign = if x.is_sign_negative() { "-" } else { "" };
    if x == 0.0 {
        return format!("{sign}0.0");
    }
    // As many digits as the shortest that round-trip; of the decimals of
    // that many digits that do, the nearest (a tie to the even), which the
    // exact formatting gives where it round-trips; next to a power of two
    // the nearest may not, and the shortest is the one that does.
    let shortest = format!("{:e}", x.abs());
    let precision = shortest
        .split_once('e')
        .map_or(0, |(m, _)| m.chars().filter(char::is_ascii_digit).count())
        .saturating_sub(1);
    let nearest = format!("{:.*e}", precision, x.abs());
    let sci = if nearest.parse::<f64>() == Ok(x.abs()) {
        nearest
    } else {
        shortest
    };
    let (mantissa, exponent) = sci.split_once('e').unwrap_or((sci.as_str(), "0"));
    let exponent: i32 = exponent.parse().unwrap_or(0);
    let digits: String = mantissa.chars().filter(char::is_ascii_digit).collect();
    let digits = match digits.trim_end_matches('0') {
        "" => "0".to_owned(),
        kept => kept.to_owned(),
    };
    // The value is 0.DIGITS times ten to `point`.
    let point = exponent + 1;
    let count = i32::try_from(digits.len()).unwrap_or(i32::MAX);
    if point <= -4 || point > 16 {
        let mut text = digits[..1].to_owned();
        if count > 1 {
            text.push('.');
            text.push_str(&digits[1..]);
        }
        let e = point - 1;
        let esign = if e < 0 { '-' } else { '+' };
        return format!("{sign}{text}e{esign}{:02}", e.abs());
    }
    let at = usize::try_from(point.max(0)).unwrap_or(0);
    let text = if point <= 0 {
        format!(
            "0.{}{}",
            "0".repeat(usize::try_from(-point).unwrap_or(0)),
            digits
        )
    } else if point >= count {
        format!("{}{}.0", digits, "0".repeat(at - digits.len()))
    } else {
        format!("{}.{}", &digits[..at], &digits[at..])
    };
    format!("{sign}{text}")
}

/// The Databricks type a seed key of `column_type` is read from JSON as,
/// as the reference reads a key by its data type (`_elem_type`): an
/// integer (or a decimal of scale 0) `bigint`, a float (or a decimal with
/// a fraction) `double`, any other `string`.
fn element_type(column_type: &ColumnType) -> &'static str {
    use ColumnType as T;
    match column_type {
        T::Int8 | T::Int16 | T::Int32 | T::Int64 | T::Uint8 | T::Uint16 | T::Uint32 | T::Uint64 => {
            "bigint"
        }
        T::Decimal128 { scale: 0, .. } | T::Decimal256 { scale: 0, .. } => "bigint",
        T::Float32 | T::Float64 | T::Decimal128 { .. } | T::Decimal256 { .. } => "double",
        _ => "string",
    }
}

/// Each seed of a key set as the JSON this dialect binds it as (the
/// module's documentation), in order.
pub(crate) fn key_rows(
    schema: &SchemaRef,
    batches: &[RecordBatch],
) -> Result<Vec<String>, CoreError> {
    let (schema, batches) = coerced(schema, batches)?;
    let names: Vec<String> = schema
        .fields()
        .iter()
        .map(|field| json::string(field.name(), true))
        .collect();
    let mut rows = Vec::new();
    for batch in &batches {
        for row in 0..batch.num_rows() {
            let cells: Vec<String> = batch
                .columns()
                .iter()
                .map(|column| json_cell(column, row))
                .collect();
            rows.push(match &cells[..] {
                [one] => one.clone(),
                several => format!(
                    "{{{}}}",
                    names
                        .iter()
                        .zip(several)
                        .map(|(name, cell)| format!("{name}:{cell}"))
                        .collect::<Vec<_>>()
                        .join(",")
                ),
            });
        }
    }
    Ok(rows)
}

/// One cell of a key set [`coerced`] for its JSON, as JSON: an integer's
/// digits, a double's number, a text's string, ASCII only, `null`.
fn json_cell(column: &ArrayRef, row: usize) -> String {
    if column.is_null(row) {
        return "null".to_owned();
    }
    if let Some(integers) = column.as_primitive_opt::<arrow_array::types::Int64Type>() {
        return integers.value(row).to_string();
    }
    if let Some(doubles) = column.as_primitive_opt::<arrow_array::types::Float64Type>() {
        return json::number(doubles.value(row));
    }
    json::string(column.as_string::<i32>().value(row), true)
}

/// A key set as this dialect writes it (the module's documentation): the
/// batch left out, the key coerced as the reference coerces it for its
/// JSON (`_coerce_id`), each seed time its wall clock's text to the
/// second.
fn coerced(
    schema: &SchemaRef,
    batches: &[RecordBatch],
) -> Result<(SchemaRef, Vec<RecordBatch>), CoreError> {
    let unbound = || {
        CoreError::new(
            Sentence::Internal,
            "a key set holds a column this dialect does not bind",
        )
    };
    let mut fields = Vec::new();
    for field in schema.fields() {
        if field.name() == SEED_BATCH {
            continue;
        }
        let bound = if field.name() == SEED_KEY {
            match key_kind(field.data_type()) {
                "bigint" => DataType::Int64,
                "double" => DataType::Float64,
                _ => DataType::Utf8,
            }
        } else {
            match field.data_type() {
                DataType::Timestamp(_, None) => DataType::Utf8,
                _ => return Err(unbound()),
            }
        };
        fields.push(ArrowField::new(field.name(), bound, true));
    }
    let out = Arc::new(Schema::new(fields));
    let mut rewritten = Vec::with_capacity(batches.len());
    for batch in batches {
        let mut columns: Vec<ArrayRef> = Vec::new();
        for (at, field) in schema.fields().iter().enumerate() {
            if field.name() == SEED_BATCH {
                continue;
            }
            let column = batch.column(at);
            let bound: ArrayRef = if field.name() == SEED_KEY {
                match key_kind(field.data_type()) {
                    "bigint" => arrow_cast(column, &DataType::Int64)?,
                    "double" => arrow_cast(column, &DataType::Float64)?,
                    _ => arrow_cast(column, &DataType::Utf8)?,
                }
            } else {
                let times = arrow_cast(
                    column,
                    &DataType::Timestamp(arrow_schema::TimeUnit::Nanosecond, None),
                )?;
                let times = times.as_primitive::<TimestampNanosecondType>();
                Arc::new(
                    times
                        .iter()
                        .map(|t| t.map(wall_clock_text))
                        .collect::<StringArray>(),
                )
            };
            columns.push(bound);
        }
        rewritten.push(RecordBatch::try_new(Arc::clone(&out), columns).map_err(|_| unbound())?);
    }
    Ok((out, rewritten))
}

/// The element type of a key of Arrow type `data_type` ([`element_type`]).
fn key_kind(data_type: &DataType) -> &'static str {
    match ColumnType::from_arrow("key", data_type) {
        Ok(column_type) => element_type(&column_type),
        Err(_) => "string",
    }
}

fn arrow_cast(column: &ArrayRef, to: &DataType) -> Result<ArrayRef, CoreError> {
    if column.data_type() == to {
        return Ok(Arc::clone(column));
    }
    let converted: ArrayRef = match to {
        DataType::Int64 => Arc::new(
            (0..column.len())
                .map(|row| integer_at(column, row))
                .collect::<Option<Vec<_>>>()
                .map(Int64Array::from)
                .ok_or_else(|| {
                    CoreError::new(
                        Sentence::Internal,
                        "a key is not an integer the binding writes",
                    )
                })?,
        ),
        DataType::Float64 => Arc::new(
            (0..column.len())
                .map(|row| float_at(column, row))
                .collect::<Float64Array>(),
        ),
        _ => Arc::new(
            (0..column.len())
                .map(|row| text_cell(column, row))
                .collect::<Option<Vec<_>>>()
                .map(StringArray::from)
                .ok_or_else(|| {
                    CoreError::new(
                        Sentence::KeySetTypeUnbound,
                        "a key of this type is not one the binding writes as text",
                    )
                })?,
        ),
    };
    Ok(converted)
}

/// Row `row` of an integer or scale-0 decimal column as an `i64`; `Some`
/// of `None` for a null.
fn integer_at(column: &ArrayRef, row: usize) -> Option<Option<i64>> {
    use arrow_array::types as t;
    if column.is_null(row) {
        return Some(None);
    }
    let value: i128 = match column.data_type() {
        DataType::Int8 => column.as_primitive::<t::Int8Type>().value(row).into(),
        DataType::Int16 => column.as_primitive::<t::Int16Type>().value(row).into(),
        DataType::Int32 => column.as_primitive::<t::Int32Type>().value(row).into(),
        DataType::Int64 => column.as_primitive::<t::Int64Type>().value(row).into(),
        DataType::UInt8 => column.as_primitive::<t::UInt8Type>().value(row).into(),
        DataType::UInt16 => column.as_primitive::<t::UInt16Type>().value(row).into(),
        DataType::UInt32 => column.as_primitive::<t::UInt32Type>().value(row).into(),
        DataType::UInt64 => column.as_primitive::<t::UInt64Type>().value(row).into(),
        DataType::Decimal128(_, 0) => column.as_primitive::<t::Decimal128Type>().value(row),
        _ => return None,
    };
    i64::try_from(value).ok().map(Some)
}

/// Row `row` of a numeric column as a double, as the reference's `float`
/// of it.
#[allow(
    clippy::cast_precision_loss,
    reason = "a number to the nearest double, as the reference converts it"
)]
fn float_at(column: &ArrayRef, row: usize) -> Option<f64> {
    use arrow_array::types as t;
    if column.is_null(row) {
        return None;
    }
    Some(match column.data_type() {
        DataType::Float16 => column.as_primitive::<t::Float16Type>().value(row).to_f64(),
        DataType::Float32 => f64::from(column.as_primitive::<t::Float32Type>().value(row)),
        DataType::Float64 => column.as_primitive::<t::Float64Type>().value(row),
        DataType::Decimal128(_, scale) => {
            let value = column.as_primitive::<t::Decimal128Type>().value(row);
            format!("{value}e-{scale}").parse().ok()?
        }
        _ => integer_at(column, row).flatten()? as f64,
    })
}

/// Row `row` of a text column; `Some(None)` for a null, `None` for a
/// column of another type, which no key this dialect binds as text holds.
fn text_cell(column: &ArrayRef, row: usize) -> Option<Option<String>> {
    if column.is_null(row) {
        return Some(None);
    }
    if let Some(strings) = column.as_string_opt::<i32>() {
        return Some(Some(strings.value(row).to_owned()));
    }
    if let Some(strings) = column.as_string_opt::<i64>() {
        return Some(Some(strings.value(row).to_owned()));
    }
    if let Some(strings) = column.as_string_view_opt() {
        return Some(Some(strings.value(row).to_owned()));
    }
    None
}

/// A wall clock in nanoseconds since the epoch as its text to the second,
/// `%Y-%m-%d %H:%M:%S`, as the reference writes a bound.
fn wall_clock_text(nanos: i64) -> String {
    let seconds = nanos.div_euclid(1_000_000_000);
    let days = seconds.div_euclid(86_400);
    let of_day = seconds.rem_euclid(86_400);
    let (year, month, day) = civil_from_days(days);
    format!(
        "{year:04}-{month:02}-{day:02} {:02}:{:02}:{:02}",
        of_day / 3600,
        (of_day % 3600) / 60,
        of_day % 60
    )
}

/// The proleptic Gregorian date `days` after 1970-01-01.
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if m <= 2 { y + 1 } else { y };
    (
        year,
        u32::try_from(m).unwrap_or(1),
        u32::try_from(d).unwrap_or(1),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_percentage_is_written_as_the_reference_writes_it() {
        // `str(x)` of each, as the reference's language gives it.
        for (value, text) in [
            (100.0, "100.0"),
            (0.1 + 0.2, "0.30000000000000004"),
            (1e-05, "1e-05"),
            (0.0001, "0.0001"),
            (1.5e16, "1.5e+16"),
            (1e16, "1e+16"),
            (9_999_999_999_999_998.0, "9999999999999998.0"),
            (3.0, "3.0"),
            (12.5, "12.5"),
            (5e-324, "5e-324"),
            (0.000_123_456, "0.000123456"),
        ] {
            assert_eq!(percent_text(value), text, "{value:e}");
        }
    }

    #[test]
    fn the_percentage_is_the_reference_arithmetic() {
        assert_eq!(sample_percent(10, None), 100.0);
        assert_eq!(sample_percent(10, Some(10)), 100.0);
        assert_eq!(sample_percent(10, Some(5)), 100.0);
        assert_eq!(sample_percent(10, Some(20)), 100.0);
        assert_eq!(sample_percent(1, Some(3)), 100.0);
        assert_eq!(percent_text(sample_percent(10, Some(1_000))), "3.0");
        assert_eq!(
            percent_text(sample_percent(1, Some(7))),
            percent_text(100.0 * 1.0 / 7.0 * 3.0)
        );
    }

    #[test]
    fn a_name_is_quoted_with_backticks_doubled() {
        assert_eq!(identifier("a`b"), "`a``b`");
        assert_eq!(identifier("home.city"), "`home.city`");
    }

    #[test]
    fn a_column_is_typed_by_its_declaration_as_the_reference_types_it() {
        for (declared, dtype) in [
            ("BIGINT", Some(Dtype::Int)),
            ("decimal(38,0)", Some(Dtype::Int)),
            ("decimal(38,6)", Some(Dtype::Float)),
            ("decimal(10)", Some(Dtype::Int)),
            ("decimal", Some(Dtype::Float)),
            ("double", Some(Dtype::Float)),
            ("string", Some(Dtype::String)),
            ("varchar(10)", Some(Dtype::String)),
            ("timestamp_ntz", Some(Dtype::Date)),
            ("date", Some(Dtype::Date)),
            ("array<int>", Some(Dtype::Intlist)),
            ("array<string>", Some(Dtype::Stringlist)),
            ("array<date>", Some(Dtype::Unsupported)),
            ("struct<a:int>", Some(Dtype::Unsupported)),
            ("map<string,int>", Some(Dtype::Unsupported)),
            ("interval day", Some(Dtype::Unsupported)),
            ("boolean", Some(Dtype::Bool)),
            ("binary", Some(Dtype::Binary)),
            ("something_else", None),
        ] {
            assert_eq!(declared_dtype(declared), dtype, "{declared}");
        }
    }

    #[test]
    fn a_key_set_is_bound_as_the_reference_writes_its_json() {
        use arrow_array::{Int32Array, Int64Array, TimestampNanosecondArray};
        let schema = Arc::new(Schema::new(vec![
            ArrowField::new(SEED_KEY, DataType::Int32, true),
            ArrowField::new(SEED_BATCH, DataType::Int64, false),
            ArrowField::new(
                "time",
                DataType::Timestamp(arrow_schema::TimeUnit::Nanosecond, None),
                true,
            ),
        ]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int32Array::from(vec![3, 4])),
                Arc::new(Int64Array::from(vec![0, 1])),
                Arc::new(TimestampNanosecondArray::from(vec![
                    Some(1_709_251_200_500_000_000),
                    None,
                ])),
            ],
        )
        .unwrap();
        assert_eq!(
            key_rows(&schema, &[batch]).unwrap(),
            [
                r#"{"key":3,"time":"2024-03-01 00:00:00"}"#,
                r#"{"key":4,"time":null}"#
            ]
        );
    }

    #[test]
    fn a_later_chunk_s_statement_moves_its_positions_by_the_seeds_before_it() {
        let text = format!("WITH {TMP} AS (\n  SELECT pos AS `batch`, col AS `key`\n)\nSELECT 1");
        assert_eq!(Databricks.chunk_text(&text, 0).unwrap(), text);
        assert_eq!(
            Databricks.chunk_text(&text, 7).unwrap(),
            text.replace("pos AS", "pos + 7 AS")
        );
        assert!(Databricks.chunk_text("SELECT 1", 7).is_err());
    }

    #[test]
    fn a_wall_clock_is_written_to_the_second() {
        assert_eq!(wall_clock_text(0), "1970-01-01 00:00:00");
        assert_eq!(
            wall_clock_text(1_709_251_199_999_999_999),
            "2024-02-29 23:59:59"
        );
        assert_eq!(wall_clock_text(-1), "1969-12-31 23:59:59");
    }
}
