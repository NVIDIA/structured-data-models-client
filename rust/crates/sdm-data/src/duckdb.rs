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
//! binds as the relation the plan names ([`SEEDS`]). Their texts are not
//! the reference's: its key table's name is drawn from an object's address,
//! and its columns are its own `__kumo_` names; the seeds' role-named
//! columns (`key`, `batch`) could meet a source column's name, so the
//! table's columns are read from its alias. What each returns is the
//! reference statement's result, its helper columns under the reference's
//! names (decision H9 gates on the results, not the text).

use std::collections::BTreeSet;

use sdm_contract::data::{
    BATCH_COLUMN, DataRequest, DescribePart, Determinism, Dialect, FetchPlan, Filter,
    MATCHES_COLUMN, OptionalRequest, ProviderCapabilities, SEED_BATCH, SEED_KEY, SqlParam,
    SqlStatement, TableColumn, TableKey, TieBreak, TieTerm, Window,
};
use sdm_contract::{CoreError, Detail, ErrorCode, ErrorKind, Scalar};

/// The optional requests this dialect renders.
const RENDERED: &[OptionalRequest] = &[
    OptionalRequest::Preview,
    OptionalRequest::KeyCounts,
    OptionalRequest::TimeBounds,
    OptionalRequest::SampleEntities,
    OptionalRequest::Neighbors,
    OptionalRequest::Lookup,
];

/// The name a statement reads its seeds by: the key set the provider binds
/// for it (registered as a relation, on the statement's own cursor).
pub(crate) const SEEDS: &str = "__sdm_seeds__";

/// The largest seed DuckDB's `REPEATABLE` takes: its parser refuses one
/// from 2**31 (register entry RS-7).
const MAX_SEED: i128 = (1 << 31) - 1;

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
                relation: SEEDS.to_owned(),
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
                relation: SEEDS.to_owned(),
            });
            lookup(request, table, key, columns)?
        }
        DataRequest::Scan { .. } | DataRequest::CountRows { .. } => {
            return Err(crate::not_rendered(request, Dialect::Duckdb));
        }
    };
    Ok(FetchPlan::Sql {
        statements: vec![SqlStatement {
            dialect: Dialect::Duckdb,
            text,
            params,
        }],
        expect: request.expected(),
    })
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
/// one row of each of, as the reference does. A seed DuckDB's parser
/// refuses, from 2**31, is refused here by name (register entry RS-7).
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
        Some(seed) if (0..=MAX_SEED).contains(&seed) => seed,
        Some(seed) if seed >= 0 => {
            return Err(CoreError::new(
                ErrorKind::InvalidRequest,
                ErrorCode::InvalidRequest,
                "DuckDB samples repeatably with a seed below 2**31 only",
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
    let mut text = format!(
        "SELECT {}\nFROM (SELECT * FROM {}",
        identifiers(columns),
        identifier(table)
    );
    if !conditions.is_empty() {
        text.push_str(&format!("\nWHERE{}", conditions.join(" AND")));
    }
    text.push_str(&format!(
        ")\nUSING SAMPLE reservoir({n} ROWS) REPEATABLE ({seed})"
    ));
    Ok(text)
}

/// The table's columns asked for, read from its alias, after the seed's
/// batch position under the reference's name.
fn joined_columns(
    request: &DataRequest,
    table: &str,
    columns: &[String],
) -> Result<String, CoreError> {
    if columns.is_empty() {
        return Err(sdm_contract::data::refused(
            request,
            "columns",
            "the request names no columns",
        ));
    }
    if table == SEEDS {
        return Err(not_taken(
            request,
            "table",
            "a table may not have the name its seeds are read by",
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
    let projected = joined_columns(request, table, columns)?;
    let mut text = format!(
        "SELECT s.{batch} AS {helper}, {projected}\nFROM {seeds} s\nJOIN {table} t\n  ON t.{key} = s.{seed_key}",
        batch = identifier(SEED_BATCH),
        helper = identifier(BATCH_COLUMN),
        seeds = identifier(SEEDS),
        table = identifier(table),
        key = identifier(key),
        seed_key = identifier(SEED_KEY),
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
    let projected = joined_columns(request, table, columns)?;
    let batch = identifier(SEED_BATCH);
    Ok(format!(
        "SELECT s.{batch} AS {helper}, COUNT(*) OVER (PARTITION BY s.{batch}) AS {matches}, {projected}\nFROM {seeds} s\nJOIN {table} t\n  ON t.{key} = s.{seed_key}\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.{batch}\n) = 1",
        helper = identifier(BATCH_COLUMN),
        matches = identifier(MATCHES_COLUMN),
        seeds = identifier(SEEDS),
        table = identifier(table),
        key = identifier(key),
        seed_key = identifier(SEED_KEY),
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
        DescribePart::Tables => {
            whole()?;
            format!(
                "SELECT table_name FROM duckdb_tables() WHERE NOT temporary AND NOT internal AND {IN_CURRENT_SCHEMA}"
            )
        }
        DescribePart::Columns => format!("PRAGMA table_info({})", identifier(one()?)),
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
