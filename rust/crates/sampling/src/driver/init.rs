// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What the DuckDB and SQLite samplers check when they are built: which
//! keys no index covers, and the warning that names how many
//! (`DuckDBSampler._get_missing_index_dict`, `_indexed_column` and
//! `_warn_missing_indices`; `SQLiteSampler._warn_missing_indices`).

use sdm_dialects::{duckdb, sqlite};
use sdm_schema::pyval::PyVal;

use super::inputs::{Backend, Inputs};
use super::output::Warning;
use super::sampler::Sampler;
use super::sql::{row_values, sql_of};
use super::{Db, Raised};

/// The keys of each table no index covers, in the order Python's dict and
/// sets hold them (the sets' order never shows: only their sizes do).
pub type MissingIndices = Vec<(String, Vec<Vec<String>>)>;

fn entry<'a>(dict: &'a mut MissingIndices, table: &str) -> &'a mut Vec<Vec<String>> {
    if let Some(i) = dict.iter().position(|(t, _)| t == table) {
        return &mut dict[i].1;
    }
    dict.push((table.to_string(), Vec::new()));
    let last = dict.len() - 1;
    &mut dict[last].1
}

fn add(set: &mut Vec<Vec<String>>, index: Vec<String>) {
    if !set.contains(&index) {
        set.push(index);
    }
}

/// `DuckDBSampler._indexed_column(expressions)`: the one column an index
/// covers, or `None`.
pub fn indexed_column(expressions: &str) -> Option<String> {
    let expression = sdm_schema::py::strip(expressions);
    let expression = expression.strip_prefix('[')?.strip_suffix(']')?;
    let expression = sdm_schema::py::strip(expression);
    if expression.contains(',') {
        return None;
    }
    let expression = expression.trim_matches('\'');
    if expression.starts_with('"') && expression.ends_with('"') {
        // `expression[1:-1]`, which of a lone quote is empty.
        let inner = expression
            .get(1..expression.len().saturating_sub(1))
            .unwrap_or("");
        return Some(inner.replace("\"\"", "\""));
    }
    Some(expression.to_string())
}

/// The keys to index: each primary key, and each foreign key (with the time
/// column after it, on SQLite), that no source column keeps unique.
fn wanted(sampler: &Sampler, inputs: &Inputs, with_time: bool) -> Result<MissingIndices, Raised> {
    let sql = sql_of(sampler)?;
    let mut dict: MissingIndices = Vec::new();
    for (table, primary_key) in &sampler.primary_key_dict {
        let source = sql.source_table(table)?;
        let Some(column) = source.source_column(primary_key) else {
            continue;
        };
        if column.is_unique_key {
            continue;
        }
        add(entry(&mut dict, table), vec![primary_key.clone()]);
    }
    for (src, fkey, _) in &inputs.edges {
        let source = sql.source_table(src)?;
        let Some(column) = source.source_column(fkey) else {
            continue;
        };
        if column.is_unique_key {
            continue;
        }
        let time = sampler
            .time_column(src)
            .filter(|t| with_time && source.source_column(t).is_some());
        let index = match time {
            Some(time) => vec![fkey.clone(), time.clone()],
            None => vec![fkey.clone()],
        };
        add(entry(&mut dict, src), index);
    }
    Ok(dict)
}

/// `_warn_missing_indices(index_dict)`: the warning, where any key lacks an
/// index.
pub fn missing_index_warning(dict: &MissingIndices) -> Option<Warning> {
    if dict.is_empty() {
        return None;
    }
    let num: usize = dict.iter().map(|(_, s)| s.len()).sum();
    let index_repr = if num == 1 {
        "1 index".to_string()
    } else {
        format!("{num} indices")
    };
    let tables = dict.len();
    let table_repr = if tables == 1 {
        "1 table".to_string()
    } else {
        format!("{tables} tables")
    };
    Some(Warning::user(format!(
        "Missing {index_repr} in {table_repr} for optimal database querying. For improving \
         runtime, we strongly suggest to create indices for primary and foreign keys, e.g., \
         automatically by instantiating Kumo Relational via `KumoRelational(graph, \
         optimize=True)`."
    )))
}

fn text(value: &PyVal) -> String {
    match value {
        PyVal::Str(s) => s.clone(),
        other => other.str_(),
    }
}

/// `DuckDBSampler._get_missing_index_dict(graph)`.
pub async fn duckdb_missing_index_dict(
    sampler: &Sampler,
    inputs: &Inputs,
    db: &Db,
) -> Result<MissingIndices, Raised> {
    let mut dict = wanted(sampler, inputs, false)?;
    if dict.is_empty() {
        return Ok(dict);
    }
    let rows = db.run(duckdb::indices()).await?;
    for row in row_values(&rows, false) {
        if row.len() < 2 {
            return Err(Raised::value_error("not enough values to unpack"));
        }
        let source_name = text(&row[0]);
        for table in &inputs.tables {
            let name = table
                .sql
                .as_ref()
                .and_then(|s| s.unquoted_name.clone())
                .unwrap_or_else(|| table.name.clone());
            if source_name != name {
                continue;
            }
            if let Some(column) = indexed_column(&text(&row[1]))
                && let Some(i) = dict.iter().position(|(t, _)| *t == table.name)
            {
                dict[i].1.retain(|index| index != &vec![column.clone()]);
                if dict[i].1.is_empty() {
                    dict.remove(i);
                }
            }
            break;
        }
    }
    Ok(dict)
}

/// `SQLiteSampler._warn_missing_indices(graph)`: the keys no index covers,
/// whose first column no index leads with.
pub async fn sqlite_missing_index_dict(
    sampler: &Sampler,
    inputs: &Inputs,
    db: &Db,
) -> Result<MissingIndices, Raised> {
    let sql = sql_of(sampler)?;
    let mut dict = wanted(sampler, inputs, true)?;
    let tables: Vec<String> = dict.iter().map(|(t, _)| t.clone()).collect();
    for table in tables {
        let source_name = sql.source_name(&table)?.clone();
        let listed = db.run(sqlite::index_list(&source_name)).await?;
        for index in row_values(&listed, false) {
            let name = index
                .get(1)
                .map(text)
                .ok_or_else(|| Raised::value_error("not enough values to unpack"))?;
            let info = db.run(sqlite::index_info(&name)).await?;
            let mut columns = row_values(&info, false);
            columns.sort_by_key(|row| match row.first() {
                Some(PyVal::Int(i)) => *i,
                _ => 0,
            });
            let first = columns
                .first()
                .and_then(|row| row.get(2))
                .map(text)
                .ok_or_else(|| Raised::new("IndexError", "tuple index out of range"))?;
            if let Some((_, set)) = dict.iter_mut().find(|(t, _)| *t == table) {
                set.retain(|index| index.first() != Some(&first));
            }
        }
        if dict.iter().any(|(t, set)| *t == table && set.is_empty()) {
            dict.retain(|(t, _)| *t != table);
        }
    }
    Ok(dict)
}

/// What building the sampler checks: on DuckDB and SQLite, unless
/// `optimize`, the keys no index covers and the warning naming them.
pub async fn missing_indices(
    sampler: &Sampler,
    inputs: &Inputs,
    db: &Db,
    optimize: bool,
) -> Result<(MissingIndices, Option<Warning>), Raised> {
    if optimize {
        return Ok((Vec::new(), None));
    }
    let dict = match sampler.backend {
        Backend::Duckdb => duckdb_missing_index_dict(sampler, inputs, db).await?,
        Backend::Sqlite => sqlite_missing_index_dict(sampler, inputs, db).await?,
        _ => Vec::new(),
    };
    let warning = missing_index_warning(&dict);
    Ok((dict, warning))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_index_of_one_column_names_it() {
        assert_eq!(indexed_column("['user_id']"), Some("user_id".into()));
        assert_eq!(indexed_column("[\"a\"\"b\"]"), Some("a\"b".into()));
        assert_eq!(indexed_column("[a, b]"), None);
        assert_eq!(indexed_column("a"), None);
    }

    #[test]
    fn the_warning_counts_indices_and_tables() {
        let dict = vec![
            ("orders".to_string(), vec![vec!["user_id".to_string()]]),
            ("users".to_string(), vec![vec!["user_id".to_string()]]),
        ];
        let warning = missing_index_warning(&dict).unwrap();
        assert!(warning.message.starts_with("Missing 2 indices in 2 tables"));
        assert!(missing_index_warning(&Vec::new()).is_none());
    }
}
