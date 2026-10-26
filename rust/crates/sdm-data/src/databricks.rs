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

use arrow_array::{RecordBatch, StringArray};
use arrow_schema::SchemaRef;
use sdm_contract::catalog::{CatalogAnswer, CatalogColumn, CatalogForeignKey, CatalogKey, KeyKind};
use sdm_contract::data::{
    DataRequest, DescribePart, Determinism, ExpectedSchema, MARKED, MATCHES_COLUMN, NullOrder,
    ProviderCapabilities, Relation, SEED_BATCH, SEED_COUNT_UNREAD, SEED_KEY, SEED_KEY_UNIQUE,
    SqlParam, SqlTimestampKind, TextTimes, TieTerm, refused, seed_integer, views_unread,
};
use sdm_contract::ops::Dtype;
use sdm_contract::types::ColumnType;
use sdm_contract::{CoreError, Detail, Scalar, Sentence};
use std::collections::BTreeMap;

use crate::json;
use crate::key_text;
use crate::read::{text, text_at};
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

    fn timestamp_micros(&self, value: &str, kind: SqlTimestampKind) -> Option<String> {
        Some(match kind {
            SqlTimestampKind::Instant => format!("UNIX_MICROS({value})"),
            // NTZ timestamp subtraction can pass through a session-zone
            // timestamp and normalize nonexistent DST-gap fields. Read
            // the native NTZ calendar components instead, with integer
            // arithmetic widened before multiplying. NULL propagates.
            SqlTimestampKind::WallTime => format!(
                "(CAST(DATEDIFF(CAST({value} AS DATE), DATE '1970-01-01') AS BIGINT) * 86400000000 + CAST(HOUR({value}) AS BIGINT) * 3600000000 + CAST(MINUTE({value}) AS BIGINT) * 60000000 + CAST(EXTRACT(SECOND FROM {value}) * 1000000 AS BIGINT))"
            ),
        })
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
            key_enforcement: false,
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

    /// A metric view's expressions are checked as Databricks reads them
    /// (T3.5).
    fn expr_dialect(&self) -> Option<crate::view_expr::ExprDialect> {
        Some(crate::view_expr::ExprDialect::Databricks)
    }

    fn view_kind(&self) -> Option<sdm_contract::view::ViewKind> {
        Some(sdm_contract::view::ViewKind::Metric)
    }

    /// A metric view's described column type by the table type map
    /// (`DatabricksTable._to_dtype`, old-behaviour row 41).
    fn declared_type(&self, data_type: &str) -> sdm_contract::view::DeclaredType {
        use sdm_contract::view::DeclaredType;
        match declared_dtype(data_type) {
            Some(sdm_contract::ops::Dtype::Unsupported) => DeclaredType::Unsupported,
            Some(dtype) => DeclaredType::Known(dtype),
            None => DeclaredType::Unknown,
        }
    }

    /// What a Databricks failure says (`DB/errors.py:66-99`): an expression
    /// refused for what it says or reads, SQLSTATE class `22` or `42`; a
    /// table missing or hidden, `42P01`; one the caller may not read,
    /// `42501` or a permission error class.
    fn failure(
        &self,
        error: &sdm_contract::data::ProviderError,
    ) -> sdm_contract::data::FailureFacts {
        use sdm_contract::data::{FailureFacts, Unusable};
        let state = error.sqlstate.as_deref();
        let class = error
            .error_class
            .as_deref()
            .map(|name| name.split('.').next().unwrap_or(name));
        let unusable = if state == Some("42P01") {
            Some(Unusable::Missing)
        } else if state == Some("42501") || class.is_some_and(|c| PERMISSION_CLASSES.contains(&c)) {
            Some(Unusable::Forbidden)
        } else {
            None
        };
        FailureFacts {
            not_found: false,
            refuses_expression: state.is_some_and(|s| s.starts_with("22") || s.starts_with("42")),
            unusable,
        }
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
    fn neighbors_per_seed(
        &self,
        read: &SeedRead<'_>,
        limit: &sdm_contract::data::NeighborLimit,
        terms: &[TieTerm],
    ) -> String {
        let limit = crate::sql::neighbor_limit(limit);
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

    fn read_view(
        &self,
        request: &DataRequest,
        batches: &[RecordBatch],
        spaces: &crate::view_expr::Spaces,
    ) -> Result<CatalogAnswer, CoreError> {
        read_view(request, batches, spaces)
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

    /// The reference's budget of a key set's JSON text, below the
    /// warehouse's native parameter limit of 1 MB (`_chunk_rows`).
    fn key_budget(&self) -> Option<usize> {
        Some(key_text::BUDGET)
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
        DescribePart::KeyEnforcement => return Err(views_unread()),
        // A metric view's description, as the reference asks it, each part
        // of its name quoted (`DB/metric_view.py:1440-1450`).
        DescribePart::ViewDefinition => {
            let view = sql::one_table(request, tables)?;
            (
                format!("DESCRIBE TABLE EXTENDED {}", quoted_name(view)),
                vec![],
            )
        }
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
    // The fields as the old client names them (`id`, `e`, `s`), unquoted
    // as it writes them, each read under its seed column's own name.
    let batch = batch_select(read.offset);
    let (schema, selects) = if times.is_empty() {
        (
            format!("array<{element}>"),
            format!("{batch}, col AS {}", d.identifier(SEED_KEY)),
        )
    } else {
        let named = key_text::fields(
            std::iter::once(SEED_KEY).chain(times.iter().copied()),
            read.request,
        );
        let mut fields = Vec::new();
        let mut selects = vec![batch];
        for (column, field) in &named {
            let kind = if *column == SEED_KEY {
                element
            } else {
                "string"
            };
            fields.push(format!("{field}:{kind}"));
            let value = if *column == SEED_KEY {
                format!("col.{field}")
            } else {
                match read
                    .window
                    .and_then(|w| w.source_type.as_deref())
                    .map(|t| t.trim().to_uppercase())
                {
                    Some(kind) if kind.starts_with("TIMESTAMP_NTZ") || kind == "DATE" => {
                        format!("CAST(col.{field} AS TIMESTAMP_NTZ)")
                    }
                    Some(kind) if kind.starts_with("TIMESTAMP") => {
                        format!("CAST(CONCAT(col.{field}, 'Z') AS TIMESTAMP)")
                    }
                    _ => format!("col.{field}"),
                }
            };
            selects.push(format!("{value} AS {}", d.identifier(column)));
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
/// `offset`, the seeds of the chunks before its own. `posexplode` gives an
/// int32 position; widen before addition so every chunk's batch has the
/// contract's int64 transport type, even across the int32 boundary.
fn batch_select(offset: usize) -> String {
    let batch = identifier(SEED_BATCH);
    if offset == 0 {
        format!("CAST(pos AS BIGINT) AS {batch}")
    } else {
        format!("CAST(pos AS BIGINT) + {offset} AS {batch}")
    }
}

#[cfg(test)]
mod batch_transport_tests {
    use super::batch_select;

    #[test]
    fn batch_positions_are_i64_before_any_chunk_offset_arithmetic() {
        assert_eq!(batch_select(0), "CAST(pos AS BIGINT) AS `batch`");
        for offset in [1, 300_000, i32::MAX as usize + 1] {
            assert_eq!(
                batch_select(offset),
                format!("CAST(pos AS BIGINT) + {offset} AS `batch`")
            );
        }
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
        // Read only with the white space its sections are read by
        // ([`read_view`]).
        DescribePart::KeyEnforcement | DescribePart::ViewDefinition => Err(views_unread()),
        DescribePart::Namespace => {
            crate::read::namespace(batches, sdm_contract::data::catalog_columns::NAME)
        }
        DescribePart::Tables => {
            let mut names = Vec::new();
            for batch in batches {
                let column = text_at(batch, 0)?;
                names.extend((0..batch.num_rows()).filter_map(|row| text(&column, row)));
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
                        .entry(text(&names, row).unwrap_or_default())
                        .or_insert_with(|| (text(&kinds, row).unwrap_or_default(), vec![]));
                    entry.1.push(text(&columns, row).unwrap_or_default());
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
                    let name = text(&names, row).unwrap_or_default();
                    let key = CatalogForeignKey {
                        key: 0,
                        referenced_table: format!(
                            "{}.{}.{}",
                            text(&catalogs, row).unwrap_or_default(),
                            text(&schemas, row).unwrap_or_default(),
                            text(&tables, row).unwrap_or_default()
                        ),
                        columns: vec![text(&columns, row).unwrap_or_default()],
                        referenced_columns: vec![text(&referenced, row).unwrap_or_default()],
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
            if found.is_empty() && text(&kinds, row).as_deref() == Some("METRIC_VIEW") {
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
                if let Some(table) = named.or_else(|| text(&tables, row)) {
                    refusal = refusal.with_field("metric_view", Detail::text(table));
                }
                return Err(refusal);
            }
            found.push(CatalogColumn {
                name: text(&names, row).unwrap_or_default(),
                declared_type: text(&types, row).unwrap_or_default(),
                not_null: text(&nullable, row).is_some_and(|n| n.trim().to_uppercase() == "NO"),
                primary_key: false,
                key_nullable: false,
                unique: false,
                dtype: text(&types, row).and_then(|declared| declared_dtype(&declared)),
            });
        }
    }
    Ok(CatalogAnswer::Columns(found))
}

/// The error classes that say the caller may not read a table
/// (`DB/errors.py`, `_PERMISSION_CLASSES`).
const PERMISSION_CLASSES: [&str; 2] = ["INSUFFICIENT_PERMISSIONS", "PERMISSION_DENIED"];

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

/// A relation's name as the reference quotes a metric view's, each part
/// in backticks, joined by dots (`RE/graph.py:1512-1518`).
fn quoted_name(relation: &Relation) -> String {
    relation
        .source
        .iter()
        .map(|part| identifier(part))
        .collect::<Vec<_>>()
        .join(".")
}

/// A metric view's description read (`DB/metric_view.py:1389-1475`): its
/// definition, the dimensions' and parameters' types, and the catalog and
/// schema it is in; a relation that is no metric view, or one with no
/// definition, refused by name, quoting the view's name as asked.
pub(crate) fn read_view(
    request: &DataRequest,
    batches: &[RecordBatch],
    spaces: &crate::view_expr::Spaces,
) -> Result<CatalogAnswer, CoreError> {
    use crate::view_expr::describe::{DescribeRefusal, describe_view};
    let DataRequest::Describe { tables, .. } = request else {
        return Err(views_unread());
    };
    let view = quoted_name(sql::one_table(request, tables)?);
    let mut rows = Vec::new();
    for batch in batches {
        let width = batch.num_columns().min(2);
        let cells: Vec<StringArray> = (0..width)
            .map(|at| text_at(batch, at))
            .collect::<Result<_, _>>()?;
        for row in 0..batch.num_rows() {
            rows.push(cells.iter().map(|cell| text(cell, row)).collect());
        }
    }
    match describe_view(&rows, spaces) {
        Ok(found) => Ok(CatalogAnswer::ViewDefinition(
            sdm_contract::catalog::ViewDefinition {
                kind: sdm_contract::view::ViewKind::Metric,
                text: found.definition,
                parameters: found.parameters,
                located: vec![found.catalog, found.schema],
                columns: found.dtypes,
            },
        )),
        Err(DescribeRefusal::NotAMetricView { got }) => Err(CoreError::new(
            Sentence::ViewNotAMetricView,
            "the relation named is not a metric view",
        )
        .with_field("view", Detail::text(&view))
        .with_field("got", Detail::text(got.unwrap_or_else(|| "UNKNOWN".into())))),
        Err(DescribeRefusal::NoDefinition) => Err(CoreError::new(
            Sentence::ViewNoDefinition,
            "the metric view gives no definition",
        )
        .with_field("view", Detail::text(&view))),
    }
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
    match key_text::key_type(column_type) {
        key_text::KeyType::Integer => "bigint",
        key_text::KeyType::Double => "double",
        key_text::KeyType::Text => "string",
    }
}

/// Each seed of a key set as the JSON this dialect binds it as (the
/// module's documentation), in order: the old client's `_dumps` of its
/// rows, a double as `float_text` writes it.
pub(crate) fn key_rows(
    request: &DataRequest,
    schema: &SchemaRef,
    batches: &[RecordBatch],
    float_text: fn(f64) -> String,
) -> Result<Vec<String>, CoreError> {
    key_text::in_batch_order(schema, batches)?;
    let (schema, batches) = key_text::coerced(schema, batches, Sentence::KeySetIntegerRange)?;
    let named = key_text::fields(
        schema.fields().iter().map(|field| field.name().as_str()),
        request,
    );
    let order: Vec<(usize, String)> = named
        .iter()
        .filter_map(|(column, field)| {
            Some((schema.index_of(column).ok()?, json::string(field, true)))
        })
        .collect();
    let mut rows = Vec::new();
    for batch in &batches {
        for row in 0..batch.num_rows() {
            let cell = |at: usize| key_text::json_cell(batch.column(at), row, float_text);
            rows.push(match &order[..] {
                [(at, _)] => cell(*at),
                several => format!(
                    "{{{}}}",
                    several
                        .iter()
                        .map(|(at, field)| format!("{field}:{}", cell(*at)))
                        .collect::<Vec<_>>()
                        .join(",")
                ),
            });
        }
    }
    Ok(rows)
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

    /// A neighbor read whose window's upper bound is the seeds' `time`.
    fn windowed() -> DataRequest {
        serde_json::from_str(
            r#"{"columns":["ts"],"key":["user_id"],"order":null,"per_seed":null,"request":"neighbors","seeds":{"attachment":2,"columns":[{"name":"key","nullable":true,"type":{"type":"int32"}},{"name":"batch","nullable":false,"type":{"type":"int64"}},{"name":"time","nullable":true,"type":{"type":"timestamp","unit":"ns","zone":null}}]},"table":{"source":["c","s","orders"]},"window":{"column":"ts","lower":null,"upper":{"inclusive":true,"seed_column":"time"}}}"#,
        )
        .unwrap()
    }

    #[test]
    fn a_key_set_is_bound_as_the_reference_writes_its_json() {
        use std::sync::Arc;

        use arrow_array::{Int32Array, Int64Array, TimestampNanosecondArray};
        use arrow_schema::{DataType, Field as ArrowField, Schema};
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
            key_rows(&windowed(), &schema, &[batch], |x| format!("{x:?}")).unwrap(),
            [
                r#"{"id":3,"e":"2024-03-01 00:00:00"}"#,
                r#"{"id":4,"e":null}"#
            ]
        );
    }

    /// The lookup binding exercised by the public materialize journey.
    fn bound_keys(column: arrow_array::ArrayRef) -> Result<Vec<String>, CoreError> {
        use std::sync::Arc;

        use arrow_array::Int64Array;
        use arrow_schema::{DataType, Field as ArrowField, Schema};
        use sdm_contract::data::{AttachmentId, KeySet, field};

        let column_type = ColumnType::from_arrow(SEED_KEY, column.data_type()).unwrap();
        let request = DataRequest::Lookup {
            table: Relation::of(vec!["c".into(), "s".into(), "users".into()]),
            key: vec!["user_id".into()],
            columns: vec!["user_id".into()],
            seeds: KeySet {
                attachment: AttachmentId(2),
                columns: vec![
                    field(SEED_KEY, column_type, true),
                    field(SEED_BATCH, ColumnType::Int64, false),
                ],
            },
        };
        request.check().unwrap();
        let schema = Arc::new(Schema::new(vec![
            ArrowField::new(SEED_KEY, column.data_type().clone(), true),
            ArrowField::new(SEED_BATCH, DataType::Int64, false),
        ]));
        let positions =
            Int64Array::from_iter_values((0..column.len()).map(|row| i64::try_from(row).unwrap()));
        let batch =
            RecordBatch::try_new(Arc::clone(&schema), vec![column, Arc::new(positions)]).unwrap();
        key_rows(&request, &schema, &[batch], |value| format!("{value:?}"))
    }

    #[test]
    fn integer_keys_outside_the_databricks_binding_range_are_invalid_requests() {
        use std::sync::Arc;

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
            assert_eq!(error.sentence, Sentence::KeySetIntegerRange, "{value}");
        }
        let error = bound_keys(Arc::new(UInt64Array::from(vec![u64::MAX]))).unwrap_err();
        assert_eq!(error.code, ErrorCode::InvalidRequest);
        assert_eq!(error.sentence, Sentence::KeySetIntegerRange);
    }

    #[test]
    fn signed_boundaries_decimals_unsigned_keys_and_nulls_keep_their_json() {
        use std::sync::Arc;

        use arrow_array::{Decimal128Array, UInt64Array};

        let column = Decimal128Array::from(vec![
            Some(i128::from(i64::MIN)),
            Some(0),
            Some(1),
            Some(i128::from(i64::MAX)),
            None,
        ])
        .with_precision_and_scale(38, 0)
        .unwrap();
        assert_eq!(
            bound_keys(Arc::new(column)).unwrap(),
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
            ["0".to_owned(), i64::MAX.to_string(), "null".into()]
        );
    }

    #[test]
    fn an_unsupported_integer_key_type_is_not_misreported_as_a_range_error() {
        use std::sync::Arc;

        use arrow_array::Decimal256Array;
        use sdm_contract::ErrorCode;

        // A non-null zero is in i64's range, but integer_at does not read
        // this type. The range classification must not cover that defect.
        let column = Decimal256Array::from_iter_values([Default::default()])
            .with_precision_and_scale(76, 0)
            .unwrap();
        let error = bound_keys(Arc::new(column)).unwrap_err();
        assert_eq!(error.code, ErrorCode::InternalError);
        assert_eq!(error.sentence, Sentence::Internal);
    }
}
