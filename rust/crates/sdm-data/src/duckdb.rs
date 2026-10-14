// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The DuckDB dialect. Its capabilities are the optional requests it has a
//! rendering for, so a request it cannot render is refused by name before
//! it is planned.
//!
//! As built at T1.11b it renders the catalog: `describe` in each part,
//! `preview`, `key_counts` and `time_bounds`. Each rendering is the
//! statement the reference runs for the same step on DuckDB, written the
//! same way, so each answer is that statement's result (decision H9) and
//! the driver capture can hold the two texts equal: the table list, `PRAGMA
//! table_info`, the constraints and foreign keys of one table, the
//! indexes, the first 1,000 rows, the key counts' and the time bounds'
//! single `UNION ALL` (the T1.11a row of the design maps each). Names are
//! quoted as the reference quotes them: an identifier in double quotes and
//! a table name compared as text in single quotes, each with its quote
//! doubled, which DuckDB's standard-conforming strings make sufficient.
//!
//! From T1.11c it renders the sampler's requests too: `sample_entities` as
//! the reference's reservoir sample, text for text, and `neighbors` and
//! `lookup` as the reference's joins against a key set, which the provider
//! binds as the relation the plan names ([`SEEDS`], or longer where the
//! table joined has that name). Their texts are not
//! the reference's: its key table's name is drawn from an object's address,
//! and its columns are its own `__kumo_` names; the seeds' role-named
//! columns (`key`, `batch`) could meet a source column's name, so the
//! table's columns are read from its alias. What each returns is the
//! reference statement's result, its helper columns under the reference's
//! names (decision H9 gates on the results, not the text).
//!
//! From T1.32e it renders `column_ranges`, the reference's one query over
//! every row of a table for its time column's choice, and `repeated_key`,
//! its read of one value a key repeats, each in its text.
//!
//! From T1.21c it reads its catalog's answers too ([`read`], decision E54):
//! the columns of `PRAGMA table_info`, `duckdb_tables()`,
//! `duckdb_constraints()` and `duckdb_indexes()`, which its plans expect
//! ([`expected`]), its constraint types and its index expressions, read
//! into the contract's dialect-neutral catalog answer, so the core reads
//! no DuckDB.

use std::collections::BTreeSet;

use arrow_array::RecordBatch;
use sdm_contract::catalog::{CatalogAnswer, CatalogIndex, IndexColumns, KeyKind};
use sdm_contract::data::{
    BATCH_COLUMN, DataRequest, DescribePart, Determinism, Dialect, ExpectedSchema, FetchPlan,
    Filter, IndexAdvice, KeySet, MATCHES_COLUMN, NullOrder, OptionalRequest, ProviderCapabilities,
    SEED_BATCH, SEED_KEY, SEED_KEY_TYPE, SqlParam, SqlStatement, TableColumn, TableKey, TextTimes,
    TieBreak, TieTerm, Window, field,
};
use sdm_contract::{ColumnType, CoreError, Detail, ErrorCode, ErrorKind, Scalar};

use crate::read::{self, ColumnNames, ForeignKeyNames};

/// The dialect's name on the wire.
pub(crate) const NAME: &str = "duckdb";

/// The columns of DuckDB's catalog answers: what the statements [`plan`]
/// renders for `describe` return, the reference's statements' own.
mod catalog {
    /// `duckdb_tables()` and `duckdb_indexes()`: the table's name.
    pub const TABLE_NAME: &str = "table_name";
    /// `PRAGMA table_info`: each column's position, name, declared type,
    /// whether it is declared `NOT NULL`, its default and whether it is
    /// part of the primary key.
    pub const CID: &str = "cid";
    pub const NAME: &str = "name";
    pub const TYPE: &str = "type";
    pub const NOTNULL: &str = "notnull";
    pub const DFLT_VALUE: &str = "dflt_value";
    pub const PK: &str = "pk";
    /// `duckdb_constraints()`: each constraint's type and columns, and for
    /// a foreign key its index, the table referred to and its columns.
    pub const CONSTRAINT_TYPE: &str = "constraint_type";
    pub const CONSTRAINT_COLUMN_NAMES: &str = "constraint_column_names";
    pub const CONSTRAINT_INDEX: &str = "constraint_index";
    pub const REFERENCED_TABLE: &str = "referenced_table";
    pub const REFERENCED_COLUMN_NAMES: &str = "referenced_column_names";
    /// `duckdb_indexes()`: each index's expressions, as one text.
    pub const EXPRESSIONS: &str = "expressions";
    /// The item field of DuckDB's lists.
    pub const LIST_ITEM: &str = "l";
}

/// The optional requests this dialect renders.
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

/// The name a statement reads its seeds by: the key set the provider binds
/// for it (registered as a relation, on the statement's own cursor), unless
/// the table it joins has that name ([`seeds_for`]).
pub(crate) const SEEDS: &str = "__sdm_seeds__";

/// The relation a statement joining `table` reads its seeds by: [`SEEDS`],
/// lengthened by underscores until it is not `table`'s name in any case
/// (DuckDB matches names without case), so a source table of any name is
/// read, never shadowed by the seeds bound on the cursor. The statement
/// names no other table.
pub(crate) fn seeds_for(table: &str) -> String {
    let mut name = SEEDS.to_owned();
    while name.to_lowercase() == table.to_lowercase() {
        name.push('_');
    }
    name
}

/// The largest seed DuckDB's `REPEATABLE` takes: its parser refuses one
/// from 2**31 (register entry RS-7). The dialect declares it as its
/// capability ([`capabilities`]), and the core refuses a larger seed by
/// reading it, before the sample is planned.
const MAX_SEED: u64 = (1 << 31) - 1;

/// The catalog's own tables of the database and schema an unquoted name
/// resolves in, which is where each table is read from.
const IN_CURRENT_SCHEMA: &str =
    "database_name = current_database() AND schema_name = current_schema()";

/// What a DuckDB source answers through this layer.
pub(crate) fn capabilities() -> ProviderCapabilities {
    ProviderCapabilities {
        optional: RENDERED.iter().copied().collect::<BTreeSet<_>>(),
        max_keys_per_request: None,
        // DuckDB's reservoir sampling repeats under a seed (REPEATABLE).
        sampling: Determinism::SeededStable,
        max_seed: Some(MAX_SEED),
        // The reference's engine reads a SQL source's text times one by
        // one (register DT-7).
        text_times: Some(TextTimes::ValueByValue),
        // DuckDB's default null order, which the reference's neighbor
        // order relies on, writing none: last, in either direction.
        null_order: Some(NullOrder::Last),
        // The reference's DuckDB sampler counts an index over exactly one
        // column (an ART index serves a scan only then) and warns of the
        // rest (`_get_missing_index_dict`).
        index_advice: Some(IndexAdvice::SingleColumn),
    }
}

/// The statements answering `request` on DuckDB, with the schema the
/// contract fixes for its answer (`DataRequest::expected`).
pub(crate) fn plan(request: &DataRequest) -> Result<FetchPlan, CoreError> {
    let mut params = vec![];
    let text = match request {
        DataRequest::Describe { tables, part } => describe(request, tables, *part)?,
        DataRequest::Preview {
            table,
            columns,
            limit,
        } => {
            if columns.is_empty() {
                return Err(sdm_contract::data::refused(
                    request,
                    "columns",
                    "a preview names no columns",
                ));
            }
            format!(
                "SELECT {} FROM {} LIMIT {limit}",
                identifiers(columns),
                identifier(table)
            )
        }
        DataRequest::KeyCounts { keys } => key_counts(keys),
        DataRequest::TimeBounds { columns } => time_bounds(columns),
        DataRequest::ColumnRanges { table, columns } => {
            if columns.is_empty() {
                return Err(sdm_contract::data::refused(
                    request,
                    "columns",
                    "the column ranges name no columns",
                ));
            }
            column_ranges(table, columns)
        }
        DataRequest::RepeatedKey { table, key } => {
            if key.is_empty() {
                return Err(sdm_contract::data::refused(
                    request,
                    "key",
                    "a repeated key names no columns",
                ));
            }
            repeated_key(table, key)
        }
        DataRequest::SampleEntities {
            table,
            key,
            n,
            seed,
            columns,
            filters,
        } => sample_entities(request, table, key, *n, seed, columns, filters)?,
        DataRequest::Neighbors {
            table,
            key,
            columns,
            seeds,
            window,
            per_seed,
            order,
        } => {
            params.push(SqlParam::KeySet {
                keys: seeds.clone(),
                relation: seeds_for(table),
            });
            neighbors(
                request,
                table,
                key,
                columns,
                window.as_ref(),
                *per_seed,
                order.as_ref(),
            )?
        }
        DataRequest::Lookup {
            table,
            key,
            columns,
            seeds,
        } => {
            params.push(SqlParam::KeySet {
                keys: seeds.clone(),
                relation: seeds_for(table),
            });
            lookup(request, table, key, columns)?
        }
        DataRequest::Scan {
            table,
            columns,
            filters,
            limit,
        } => {
            if columns.is_empty() {
                return Err(sdm_contract::data::refused(
                    request,
                    "columns",
                    "a scan names no columns",
                ));
            }
            let mut text = format!(
                "SELECT {} FROM {}{}",
                identifiers(columns),
                identifier(table),
                conditions(request, filters)?
            );
            if let Some(limit) = limit {
                text.push_str(&format!(" LIMIT {limit}"));
            }
            text
        }
        DataRequest::CountRows { table, filters } => format!(
            "SELECT COUNT(*) AS {} FROM {}{}",
            identifier("rows"),
            identifier(table),
            conditions(request, filters)?
        ),
    };
    Ok(FetchPlan::Sql {
        statements: vec![SqlStatement {
            dialect: Dialect::new(NAME),
            text,
            params,
        }],
        expect: expected(request),
    })
}

/// The schema DuckDB's answer to `request` has: its catalog's own for a
/// `describe`, what the contract fixes for every other request
/// (`DataRequest::expected`).
pub(crate) fn expected(request: &DataRequest) -> ExpectedSchema {
    use ColumnType::{Bool, Int32, Int64, Utf8};
    use catalog as c;
    let DataRequest::Describe { part, .. } = request else {
        return request.expected();
    };
    let names = || ColumnType::List {
        item: Box::new(field(c::LIST_ITEM, Utf8, true)),
    };
    let fields = match part {
        DescribePart::Tables => vec![field(c::TABLE_NAME, Utf8, false)],
        DescribePart::Columns => vec![
            field(c::CID, Int32, false),
            field(c::NAME, Utf8, false),
            field(c::TYPE, Utf8, false),
            field(c::NOTNULL, Bool, false),
            field(c::DFLT_VALUE, Utf8, true),
            field(c::PK, Bool, false),
        ],
        DescribePart::Keys => vec![
            field(c::CONSTRAINT_TYPE, Utf8, false),
            field(c::CONSTRAINT_COLUMN_NAMES, names(), false),
        ],
        DescribePart::ForeignKeys => vec![
            field(c::CONSTRAINT_INDEX, Int64, false),
            field(c::REFERENCED_TABLE, Utf8, false),
            field(c::CONSTRAINT_COLUMN_NAMES, names(), false),
            field(c::REFERENCED_COLUMN_NAMES, names(), false),
        ],
        DescribePart::Indexes => vec![
            field(c::TABLE_NAME, Utf8, false),
            field(c::EXPRESSIONS, Utf8, true),
        ],
    };
    ExpectedSchema::Schema { fields }
}

/// DuckDB's answer to a `describe` of `part`, which holds [`expected`]'s
/// schema, read into the contract's catalog answer.
pub(crate) fn read(
    part: DescribePart,
    batches: &[RecordBatch],
) -> Result<CatalogAnswer, CoreError> {
    use catalog as c;
    match part {
        DescribePart::Tables => read::tables(batches, c::TABLE_NAME),
        DescribePart::Columns => read::columns(
            batches,
            &ColumnNames {
                name: c::NAME,
                declared_type: c::TYPE,
                not_null: c::NOTNULL,
                primary_key: c::PK,
            },
        ),
        DescribePart::Keys => read::keys(
            batches,
            c::CONSTRAINT_TYPE,
            c::CONSTRAINT_COLUMN_NAMES,
            key_kind,
        ),
        DescribePart::ForeignKeys => read::foreign_keys(
            batches,
            &ForeignKeyNames {
                key: c::CONSTRAINT_INDEX,
                referenced_table: c::REFERENCED_TABLE,
                columns: c::CONSTRAINT_COLUMN_NAMES,
                referenced_columns: c::REFERENCED_COLUMN_NAMES,
            },
        ),
        DescribePart::Indexes => {
            let mut found = Vec::new();
            for batch in batches {
                let tables = read::strings(batch, c::TABLE_NAME)?;
                let expressions = read::strings(batch, c::EXPRESSIONS)?;
                for row in 0..batch.num_rows() {
                    // An index with no expressions covers no column the
                    // reference reads (it skips such a row).
                    if let (Some(table), Some(expressions)) =
                        (read::text(tables, row), read::text(expressions, row))
                    {
                        found.push(CatalogIndex {
                            table,
                            columns: indexed_columns(&expressions),
                        });
                    }
                }
            }
            Ok(CatalogAnswer::Indexes(found))
        }
    }
}

/// A constraint type as `duckdb_constraints()` spells it.
fn key_kind(constraint_type: &str) -> KeyKind {
    match constraint_type {
        "PRIMARY KEY" => KeyKind::Primary,
        "UNIQUE" => KeyKind::Unique,
        _ => KeyKind::Other,
    }
}

/// The columns an index's `expressions` name, as the reference reads them
/// (`_indexed_column` of its DuckDB sampler): a list in brackets of one
/// expression, quoted or not, is that one column, its quotes taken off.
/// A list holding a comma is unresolved, never split: DuckDB writes a comma
/// between expressions and inside one alike (`[a, b]`,
/// `['(COALESCE(b, \',\'))']`, `['"a,b"']`), so its pieces would be a
/// guess, and the reference reads no column from it either. Text not in
/// brackets is unresolved too, the reference reading none.
pub(crate) fn indexed_columns(expressions: &str) -> IndexColumns {
    let expression = expressions.trim();
    let Some(inner) = expression
        .strip_prefix('[')
        .and_then(|e| e.strip_suffix(']'))
    else {
        return IndexColumns::Unresolved;
    };
    let inner = inner.trim();
    if inner.contains(',') {
        return IndexColumns::Unresolved;
    }
    IndexColumns::Resolved(vec![unquoted(inner)])
}

/// One expression with its single quotes, then its double quotes, taken
/// off, as the reference's `strip("'")` and slice take them.
fn unquoted(expression: &str) -> String {
    let expression = expression.trim_matches('\'');
    if expression.starts_with('"') && expression.ends_with('"') {
        // The reference's slice: a lone quote is the empty name.
        let inner = expression.get(1..expression.len().saturating_sub(1).max(1));
        return inner.unwrap_or_default().replace("\"\"", "\"");
    }
    expression.to_owned()
}

/// A request field this rendering does not take, refused by name.
fn not_taken(request: &DataRequest, field: &str, message: &str) -> CoreError {
    CoreError::new(
        ErrorKind::Unsupported,
        ErrorCode::UnsupportedRequest,
        message,
    )
    .with_field("request", Detail::text(request.name()))
    .with_field("field", Detail::text(field))
}

/// A seed's integer value, if the scalar is an integer.
fn integer(seed: &Scalar) -> Option<i128> {
    Some(match seed {
        Scalar::Int32(v) => i128::from(*v),
        Scalar::Int64(v) => i128::from(*v),
        Scalar::UInt32(v) => i128::from(*v),
        Scalar::UInt64(v) => i128::from(*v),
        Scalar::Decimal128 {
            scale: 0, value, ..
        } => *value,
        _ => return None,
    })
}

/// The reference's reservoir sample of an entity table (`_random_sample_sql`
/// on DuckDB): the columns asked for, from the rows that meet each not-null
/// filter, `n` rows under the seed. The key names the columns the entities
/// are told apart by, which the sample projects, and which the core keeps
/// one row of each of, as the reference does. A seed past the range the
/// dialect declares never reaches here: the core refuses it first (register
/// entry RS-7), so one that does is the core's error, not the caller's.
fn sample_entities(
    request: &DataRequest,
    table: &str,
    key: &[String],
    n: u64,
    seed: &Scalar,
    columns: &[String],
    filters: &[Filter],
) -> Result<String, CoreError> {
    if columns.is_empty() {
        return Err(sdm_contract::data::refused(
            request,
            "columns",
            "a sample names no columns",
        ));
    }
    if key.is_empty() || key.iter().any(|k| !columns.contains(k)) {
        return Err(sdm_contract::data::refused(
            request,
            "key",
            "a sample's key is not among the columns it projects",
        ));
    }
    let seed = match integer(seed) {
        Some(seed) if (0..=i128::from(MAX_SEED)).contains(&seed) => seed,
        Some(seed) if seed >= 0 => {
            return Err(CoreError::new(
                ErrorKind::Internal,
                ErrorCode::InternalError,
                "a seed past the range the dialect declares reached its rendering",
            )
            .with_field("request", Detail::text(request.name()))
            .with_field("field", Detail::text("seed")));
        }
        _ => {
            return Err(sdm_contract::data::refused(
                request,
                "seed",
                "a sample's seed is a non-negative integer",
            ));
        }
    };
    let mut text = format!(
        "SELECT {}\nFROM (SELECT * FROM {}",
        identifiers(columns),
        identifier(table)
    );
    let filtered = conditions(request, filters)?;
    if !filtered.is_empty() {
        text.push('\n');
        text.push_str(filtered.trim_start());
    }
    text.push_str(&format!(
        ")\nUSING SAMPLE reservoir({n} ROWS) REPEATABLE ({seed})"
    ));
    Ok(text)
}

/// The `WHERE` clause of `filters`, with its leading space, or nothing for
/// none: each a not-null filter, as the reference's sampler writes one;
/// another kind of filter is not taken.
fn conditions(request: &DataRequest, filters: &[Filter]) -> Result<String, CoreError> {
    let mut conditions = Vec::with_capacity(filters.len());
    for filter in filters {
        let Filter::NotNull { column } = filter else {
            return Err(not_taken(
                request,
                "filters",
                "this rendering takes not-null filters only",
            ));
        };
        conditions.push(format!(" {} IS NOT NULL", identifier(column)));
    }
    if conditions.is_empty() {
        return Ok(String::new());
    }
    Ok(format!(" WHERE{}", conditions.join(" AND")))
}

/// The table's columns asked for, read from its alias, after the seed's
/// batch position under the reference's name.
fn joined_columns(request: &DataRequest, columns: &[String]) -> Result<String, CoreError> {
    if columns.is_empty() {
        return Err(sdm_contract::data::refused(
            request,
            "columns",
            "the request names no columns",
        ));
    }
    Ok(columns
        .iter()
        .map(|c| format!("t.{}", identifier(c)))
        .collect::<Vec<_>>()
        .join(", "))
}

/// One term of a neighbor order, as DuckDB orders it.
fn order_term(term: &TieTerm) -> String {
    let column = format!("t.{}", identifier(&term.column));
    let value = if term.as_timestamp {
        format!("CAST({column} AS TIMESTAMP)")
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

/// The reference's neighbor reads (`_by_time` and `_by_fkey` on DuckDB):
/// the rows of `table` whose `key` is a seed's key, within the seed's
/// window in the source's own `CAST(column AS TIMESTAMP)` wall clock, and,
/// with a limit per seed, the first rows of each seed in `order`, the
/// result in `(batch, order)`.
fn neighbors(
    request: &DataRequest,
    table: &str,
    key: &str,
    columns: &[String],
    window: Option<&Window>,
    per_seed: Option<u64>,
    order: Option<&TieBreak>,
) -> Result<String, CoreError> {
    let projected = joined_columns(request, columns)?;
    let mut text = format!(
        "SELECT s.{batch} AS {helper}, {projected}\nFROM {seeds} s\nJOIN {table} t\n  ON {on}",
        batch = identifier(SEED_BATCH),
        helper = identifier(BATCH_COLUMN),
        seeds = identifier(&seeds_for(table)),
        table = identifier(table),
        // The key compared exactly whatever its type (seed_join).
        on = seed_join(request, key),
    );
    if let Some(window) = window {
        let time = format!("CAST(t.{} AS TIMESTAMP)", identifier(&window.column));
        if let Some(upper) = &window.upper {
            let op = if upper.inclusive { "<=" } else { "<" };
            text.push_str(&format!(
                "\n AND {time} {op} s.{}",
                identifier(&upper.seed_column)
            ));
        }
        if let Some(lower) = &window.lower {
            let op = if lower.inclusive { ">=" } else { ">" };
            text.push_str(&format!(
                "\n AND {time} {op} s.{}",
                identifier(&lower.seed_column)
            ));
        }
    }
    if let (Some(limit), Some(TieBreak(terms))) = (per_seed, order) {
        let order = terms.iter().map(order_term).collect::<Vec<_>>().join(", ");
        let batch = identifier(SEED_BATCH);
        text.push_str(&format!(
            "\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.{batch}\n  ORDER BY {order}\n) <= {limit}\nORDER BY s.{batch}, {order}"
        ));
    }
    Ok(text)
}

/// The reference's key lookup (`_by_pkey` on DuckDB): one row of `table`
/// per seed whose key it holds, with the count of rows holding that key.
fn lookup(
    request: &DataRequest,
    table: &str,
    key: &str,
    columns: &[String],
) -> Result<String, CoreError> {
    let projected = joined_columns(request, columns)?;
    let batch = identifier(SEED_BATCH);
    Ok(format!(
        "SELECT s.{batch} AS {helper}, COUNT(*) OVER (PARTITION BY s.{batch}) AS {matches}, {projected}\nFROM {seeds} s\nJOIN {table} t\n  ON {on}\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.{batch}\n) = 1",
        helper = identifier(BATCH_COLUMN),
        matches = identifier(MATCHES_COLUMN),
        seeds = identifier(&seeds_for(table)),
        table = identifier(table),
        // The key compared exactly whatever its type (seed_join).
        on = seed_join(request, key),
    ))
}

/// One part of the catalog. The table list and the indexes are read whole,
/// as the reference reads them, so a request naming tables is refused
/// (tables named arrive with the overrides, T1.24); the columns, keys and
/// foreign keys are read one table at a time, one statement each, as the
/// reference reads them, so a request naming none or several is refused.
fn describe(
    request: &DataRequest,
    tables: &[String],
    part: DescribePart,
) -> Result<String, CoreError> {
    let refused = |message: &str| {
        CoreError::new(
            ErrorKind::Unsupported,
            ErrorCode::UnsupportedRequest,
            message,
        )
        .with_field("request", Detail::text(request.name()))
        .with_field("field", Detail::text("tables"))
    };
    let one = || match tables {
        [table] => Ok(table),
        _ => Err(refused(
            "this part of the catalog is read for one table at a time",
        )),
    };
    let whole = || {
        if tables.is_empty() {
            Ok(())
        } else {
            Err(refused("this part of the catalog is read whole"))
        }
    };
    Ok(match part {
        DescribePart::Tables if tables.is_empty() => format!(
            "SELECT table_name FROM duckdb_tables() WHERE NOT temporary AND NOT internal AND {IN_CURRENT_SCHEMA}"
        ),
        DescribePart::Tables => held(tables),
        DescribePart::Columns => columns(one()?),
        DescribePart::Keys => format!(
            "SELECT constraint_type, constraint_column_names\nFROM duckdb_constraints()\nWHERE table_name = {}\n  AND {IN_CURRENT_SCHEMA}",
            text(one()?)
        ),
        DescribePart::ForeignKeys => format!(
            "SELECT constraint_index, referenced_table,\n       constraint_column_names, referenced_column_names\nFROM duckdb_constraints()\nWHERE table_name = {}\n  AND {IN_CURRENT_SCHEMA}\n  AND constraint_type = 'FOREIGN KEY'",
            text(one()?)
        ),
        DescribePart::Indexes => {
            whole()?;
            "SELECT table_name, expressions FROM duckdb_indexes()".to_owned()
        }
    })
}

/// The relations, tables or views, DuckDB would read under any of `names`
/// written unqualified, as it resolves such a name: a temporary relation,
/// or one of the current database in a schema of the search path
/// (`current_schemas(true)`, its implicit schemas included), whose name
/// equals one of `names` once each of their ASCII letters is lowered, as
/// DuckDB folds a name (ASCII letters only, so `"Äpfel"` is not read as
/// `äpfel`). A relation of another schema or another attached database is
/// not one the name reads. Asked only when reading a named table's columns
/// has failed: no row says the source holds no such relation, so the
/// caller named one it lacks; any row, that the failure was another
/// (T1.23 fix 3). A search path naming a schema of another attached
/// database (`SET search_path = 'other.main'`) is not followed: the
/// filter holds the current database. No provider of this build sets
/// such a path on the connection it reads through, so it is unreachable
/// today; recorded here rather than ported (T1.23's review).
fn held(names: &[String]) -> String {
    let fold = |text: &str| {
        format!("translate({text}, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz')")
    };
    let asked = names
        .iter()
        .map(|name| fold(&self::text(name)))
        .collect::<Vec<_>>()
        .join(", ");
    let resolved = "(temporary OR (database_name = current_database() AND list_contains(current_schemas(true), schema_name)))";
    format!(
        "SELECT table_name FROM duckdb_tables() WHERE NOT internal AND {resolved} AND {} IN ({asked})\nUNION ALL\nSELECT view_name AS table_name FROM duckdb_views() WHERE NOT internal AND {resolved} AND {} IN ({asked})",
        fold("table_name"),
        fold("view_name"),
    )
}

/// The columns of `table`, in `PRAGMA table_info`'s columns and order.
///
/// The reference reads them with `PRAGMA table_info`, and so does this
/// rendering, text for text, for every name the pragma reads as itself. The
/// pragma reads its argument as a qualified name, though: a dot splits it,
/// and a double quote opens or closes a quoted part, so a name holding
/// either is read as another name, which DuckDB refuses (a parser error for
/// `double"quote`, no such schema for `a.b`) or, where one exists, reads in
/// its place (`main.x` read as `x`). Quoting cannot help: DuckDB 1.5.6
/// unquotes no doubled quote there. Such a table's columns are read by its
/// own name, as text, from `duckdb_columns()` and its primary key from
/// `duckdb_constraints()`, as the other catalog reads are, the answer's
/// columns, types and rows those the pragma gives for a table of an
/// ordinary name. Where the reference refuses (its `TableNotFoundError`,
/// `INVALID_REQUEST`), the core now describes the table: register entry
/// P1B-5, proposed. A name no table holds (in its exact case) gives no
/// rows where the pragma raises DuckDB's catalog error; no caller asks for
/// one today (`describe_source` reads the names the table list gives), and
/// the core refuses a table with no columns, `INVALID_REQUEST`, the code
/// the reference gives a table its pragma cannot read.
fn columns(table: &str) -> String {
    if !table.contains(['"', '.']) {
        return format!("PRAGMA table_info({})", identifier(table));
    }
    let name = text(table);
    format!(
        "SELECT CAST(c.column_index - 1 AS INTEGER) AS cid, c.column_name AS name,\n       c.data_type AS type, NOT c.is_nullable AS notnull,\n       c.column_default AS dflt_value,\n       COALESCE(list_contains(k.columns, c.column_name), false) AS pk\nFROM duckdb_columns() c\nLEFT JOIN (\n  SELECT constraint_column_names AS columns\n  FROM duckdb_constraints()\n  WHERE table_name = {name}\n    AND {IN_CURRENT_SCHEMA}\n    AND constraint_type = 'PRIMARY KEY'\n) k ON true\nWHERE c.table_name = {name}\n  AND c.database_name = current_database() AND c.schema_name = current_schema()\nORDER BY c.column_index"
    )
}

/// One `SELECT` per key, in order, labelled by its position: the table's
/// rows, the key's distinct values among the rows holding all of it, and
/// the rows missing part of it. DuckDB counts a row of several values as
/// not null when only part of it is, so a composite key leaves those rows
/// out of its distinct count by hand.
fn key_counts(keys: &[TableKey]) -> String {
    keys.iter()
        .enumerate()
        .map(|(position, key)| {
            let quoted: Vec<String> = key.columns.iter().map(|c| identifier(c)).collect();
            let missing = quoted
                .iter()
                .map(|c| format!("{c} IS NULL"))
                .collect::<Vec<_>>()
                .join(" OR ");
            let distinct = match quoted.as_slice() {
                [one] => format!("COUNT(DISTINCT {one})"),
                several => format!(
                    "COUNT(DISTINCT ({})) FILTER (WHERE {})",
                    several.join(", "),
                    several
                        .iter()
                        .map(|c| format!("{c} IS NOT NULL"))
                        .collect::<Vec<_>>()
                        .join(" AND ")
                ),
            };
            format!(
                "SELECT {position} AS k, COUNT(*) AS n, {distinct} AS d, COUNT(CASE WHEN {missing} THEN 1 END) AS z\nFROM {}",
                identifier(&key.table)
            )
        })
        .collect::<Vec<_>>()
        .join("\nUNION ALL\n")
}

/// One `SELECT` per time column, in order, labelled by its position: the
/// smallest and largest values in the source's own wall clock.
fn time_bounds(columns: &[TableColumn]) -> String {
    columns
        .iter()
        .enumerate()
        .map(|(position, at)| {
            let time = format!("CAST({} AS TIMESTAMP)", identifier(&at.column));
            format!(
                "SELECT\n  {position} as table_index,\n  MIN({time}) as min_date,\n  MAX({time}) as max_date\nFROM {}",
                identifier(&at.table)
            )
        })
        .collect::<Vec<_>>()
        .join("\nUNION ALL\n")
}

/// The reference's one query over every row of a table for its time
/// column's choice (`Table._full_range`): `COUNT(*)`, then each column's
/// `MIN`, `MAX` and `COUNT`, in its text.
fn column_ranges(table: &str, columns: &[String]) -> String {
    let mut selects = vec!["COUNT(*)".to_owned()];
    for column in columns {
        let quoted = identifier(column);
        selects.push(format!("MIN({quoted}), MAX({quoted}), COUNT({quoted})"));
    }
    format!("SELECT {}\nFROM {}", selects.join(", "), identifier(table))
}

/// The reference's read of one value of a key that several rows hold, to
/// show it (`Table._repeated_key_sql`), in its text.
fn repeated_key(table: &str, key: &[String]) -> String {
    let quoted = identifiers(key);
    let present = key
        .iter()
        .map(|c| format!("{} IS NOT NULL", identifier(c)))
        .collect::<Vec<_>>()
        .join(" AND ");
    format!(
        "SELECT {quoted}\nFROM {}\nWHERE {present}\nGROUP BY {quoted}\nHAVING COUNT(*) > 1\nLIMIT 1",
        identifier(table)
    )
}

/// `name` as an identifier: in double quotes, each one in it doubled.
fn identifier(name: &str) -> String {
    format!("\"{}\"", name.replace('"', "\"\""))
}

fn identifiers(names: &[String]) -> String {
    names
        .iter()
        .map(|n| identifier(n))
        .collect::<Vec<_>>()
        .join(", ")
}

/// `name` as text: in single quotes, each one in it doubled.
fn text(name: &str) -> String {
    format!("'{}'", name.replace('\'', "''"))
}

/// The vendor DuckDB names in the metadata of its opaque Arrow extension
/// types (`arrow.opaque`'s `vendor_name`), which the host layer reads by
/// DuckDB's layouts only for a column naming it (register P1B-43).
pub(crate) const OPAQUE_VENDOR: &str = "DuckDB";

/// The condition joining the seeds to `table`'s `key`: `t.key = s.key`, as
/// before part 3, but for a `decimal128(38, 0)` seed key whose column names
/// the key's declared type ([`SEED_KEY_TYPE`], which the core reads from
/// the catalog for this layer) and that type is an integer one: then the
/// seed is cast to the key's own type, `t.key = TRY_CAST(s.key AS type)`,
/// so the two compare as one type, exactly, by the key's native equality,
/// and a seed the type cannot hold is null and matches nothing (register
/// P1B-43). Without it a VARINT key cannot be compared with a DECIMAL(38,
/// 0) at all (DuckDB has no such cast), and a HUGEINT seed would meet a
/// UHUGEINT key through DOUBLE. A VARINT (BIGNUM) key takes its seed
/// through HUGEINT, which holds every 38-digit value exactly, since
/// DuckDB's DECIMAL to BIGNUM cast is not implemented (its TRY_CAST gives
/// null). Any other key type, a float or a decimal with a scale, or no
/// declared type, keeps `t.key = s.key`.
fn seed_join(request: &DataRequest, key: &str) -> String {
    let seed = format!("s.{}", identifier(SEED_KEY));
    let key = format!("t.{}", identifier(key));
    let seeds: Option<&KeySet> = match request {
        DataRequest::Neighbors { seeds, .. } | DataRequest::Lookup { seeds, .. } => Some(seeds),
        _ => None,
    };
    let declared = seeds
        .into_iter()
        .flat_map(|k| &k.columns)
        .find(|c| {
            c.name == SEED_KEY
                && c.column_type
                    == ColumnType::Decimal128 {
                        precision: 38,
                        scale: 0,
                    }
        })
        .and_then(|c| c.metadata.get(SEED_KEY_TYPE))
        .and_then(|declared| integer_type(declared));
    match declared {
        Some(Integer::Bignum) => {
            format!("{key} = TRY_CAST(CAST({seed} AS HUGEINT) AS BIGNUM)")
        }
        Some(Integer::Native(name)) => format!("{key} = TRY_CAST({seed} AS {name})"),
        None => format!("{key} = {seed}"),
    }
}

/// An integer key type as `seed_join` casts a seed to it.
enum Integer {
    /// VARINT, which DuckDB's catalog spells BIGNUM.
    Bignum,
    /// Any other, as the SQL spells it: rendered here, never the catalog's
    /// text, so nothing of the catalog reaches the statement.
    Native(String),
}

/// The integer type DuckDB's catalog `declared` spells, if it is one: the
/// signed and unsigned integers, HUGEINT, UHUGEINT, BIGNUM (VARINT) and a
/// DECIMAL of scale 0.
fn integer_type(declared: &str) -> Option<Integer> {
    const NATIVE: [&str; 10] = [
        "TINYINT",
        "SMALLINT",
        "INTEGER",
        "BIGINT",
        "UTINYINT",
        "USMALLINT",
        "UINTEGER",
        "UBIGINT",
        "HUGEINT",
        "UHUGEINT",
    ];
    let declared = declared.trim().to_ascii_uppercase();
    if declared == "BIGNUM" || declared == "VARINT" {
        return Some(Integer::Bignum);
    }
    if let Some(name) = NATIVE.iter().find(|n| **n == declared) {
        return Some(Integer::Native((*name).to_owned()));
    }
    let digits = declared.strip_prefix("DECIMAL(")?.strip_suffix(",0)")?;
    let precision: u8 = digits.parse().ok()?;
    (1..=38)
        .contains(&precision)
        .then(|| Integer::Native(format!("DECIMAL({precision},0)")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_integer_key_type_is_read_and_anything_else_is_not() {
        let native = |declared: &str| match integer_type(declared) {
            Some(Integer::Native(name)) => Some(name),
            _ => None,
        };
        assert_eq!(native("UHUGEINT").as_deref(), Some("UHUGEINT"));
        assert_eq!(native(" bigint ").as_deref(), Some("BIGINT"));
        assert_eq!(native("DECIMAL(38,0)").as_deref(), Some("DECIMAL(38,0)"));
        assert_eq!(native("DECIMAL(20,0)").as_deref(), Some("DECIMAL(20,0)"));
        assert!(matches!(integer_type("BIGNUM"), Some(Integer::Bignum)));
        assert!(matches!(integer_type("VARINT"), Some(Integer::Bignum)));
        for other in [
            "DOUBLE",
            "FLOAT",
            "DECIMAL(10,2)",
            "DECIMAL(39,0)",
            "DECIMAL(0,0)",
            "VARCHAR",
            "BIGINT); DROP TABLE t; --",
            "",
        ] {
            assert!(integer_type(other).is_none(), "{other}");
        }
    }
}
