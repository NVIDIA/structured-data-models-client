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

use std::collections::BTreeSet;

use sdm_contract::data::{
    DataRequest, DescribePart, Determinism, Dialect, FetchPlan, OptionalRequest,
    ProviderCapabilities, SqlStatement, TableColumn, TableKey,
};
use sdm_contract::{CoreError, Detail, ErrorCode, ErrorKind};

/// The optional requests this dialect renders.
const RENDERED: &[OptionalRequest] = &[
    OptionalRequest::Preview,
    OptionalRequest::KeyCounts,
    OptionalRequest::TimeBounds,
];

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
        DataRequest::Scan { .. }
        | DataRequest::CountRows { .. }
        | DataRequest::SampleEntities { .. }
        | DataRequest::Neighbors { .. }
        | DataRequest::Lookup { .. } => {
            return Err(crate::not_rendered(request, Dialect::Duckdb));
        }
    };
    Ok(FetchPlan::Sql {
        statements: vec![SqlStatement {
            dialect: Dialect::Duckdb,
            text,
            params: vec![],
        }],
        expect: request.expected(),
    })
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
