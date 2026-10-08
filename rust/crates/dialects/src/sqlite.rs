// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The SQL the engine runs against SQLite (`backend/sqlite/dialect.py`).
//!
//! Parameters are bound to `?` markers. A key list travels as one parameter
//! holding a JSON array, which `json_each` turns into rows; each value is
//! cast to the type SQLite gave the key tables the engine used to write.

use crate::arrow::{ArrowArray, ArrowType};
use crate::fragments::{count_distinct_values, key_count_select, quote_ident, repeated_key_select};
use crate::int::Int;
use crate::py::Py;
use crate::pyjson::Dumps;
use crate::{Error, KeyCountPart, Param, Statement, opt, text};

/// Every table in the database, by name.
pub fn list_tables() -> Statement {
    (
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name".into(),
        None,
    )
}

/// The columns of `table`, a quoted name.
pub fn table_info(table: &str) -> Statement {
    (format!("PRAGMA table_info({table})"), None)
}

/// The indices of `table`, a quoted name.
pub fn index_list(table: &str) -> Statement {
    (format!("PRAGMA index_list({table})"), None)
}

/// The columns of the index named `index`.
pub fn index_info(index: &str) -> Statement {
    (
        format!("PRAGMA index_info({})", quote_ident(index, "\"")),
        None,
    )
}

/// The foreign keys of `table`, a quoted name.
pub fn foreign_key_list(table: &str) -> Statement {
    (format!("PRAGMA foreign_key_list({table})"), None)
}

/// `projections` of the first `num_rows` rows of `table`, in `rowid` order.
pub fn first_rows(table: &str, projections: &[String], num_rows: i64) -> Statement {
    (
        format!(
            "SELECT {} FROM {table} ORDER BY rowid LIMIT {num_rows}",
            projections.join(", ")
        ),
        None,
    )
}

/// Whether a row of `table` holds one of `values` in `identity`; `values`
/// travel as one JSON array.
pub fn holds_any(table: &str, identity: &str, values: &[String]) -> Statement {
    (
        format!(
            "SELECT 1 FROM {table}\nWHERE ({identity}) IN (SELECT value FROM json_each(?))\nLIMIT 1"
        ),
        Some(vec![Param::Text(text::json_text_list(
            values,
            Dumps::COMPACT,
        ))]),
    )
}

/// The distinct values of a key, leaving out a row missing any part: a
/// subquery counts several columns.
pub fn count_distinct(table: &str, columns: &[String]) -> String {
    if columns.len() == 1 {
        return count_distinct_values(columns);
    }
    let present: Vec<String> = columns.iter().map(|c| format!("{c} IS NOT NULL")).collect();
    format!(
        "(SELECT COUNT(*) FROM (SELECT DISTINCT {} FROM {table} WHERE {}))",
        columns.join(", "),
        present.join(" AND ")
    )
}

/// The rows, distinct keys and rows missing a key of each table, in one
/// `UNION ALL`.
pub fn key_counts(parts: &[KeyCountPart]) -> Statement {
    let selects: Vec<String> = parts
        .iter()
        .map(|part| {
            key_count_select(
                part.label,
                &part.source,
                &part.columns,
                part.unique,
                &count_distinct(&part.source, &part.columns),
            )
        })
        .collect();
    (selects.join("\nUNION ALL\n"), None)
}

/// One value of the key `columns` more than one row holds.
pub fn repeated_key(table: &str, columns: &[String]) -> Statement {
    (repeated_key_select(table, columns), None)
}

/// The modulus of the seeded row order: a prime below 2**31.
const HASH_PRIME: i64 = (1 << 31) - 1;

/// A hash of the integer `reference` that shuffles rows, repeatably for a
/// seed: a cubic polynomial modulo a prime, its coefficients drawn from
/// `random_state(random_seed)`.
pub fn seeded_hash(random_seed: &Int, reference: &str) -> Result<String, Error> {
    if random_seed.is_negative() {
        return Err(Error::new("Seed must be between 0 and 2**32 - 1"));
    }
    let mut state = sdm_random::random_state(&random_seed.magnitude_words())
        .map_err(|e| Error::new(e.to_string()))?;
    let p = HASH_PRIME;
    let c = state
        .randint(0, p, 4)
        .map_err(|e| Error::new(e.to_string()))?;
    let x = format!("(({reference} % {p} + {p}) % {p})");
    Ok(format!(
        "(((({} * {x} + {}) % {p}) * {x} + {}) % {p} * {x} + {}) % {p}",
        c[3], c[2], c[1], c[0]
    ))
}

/// One stored value of the time `column` of `table`, to read how it is
/// written.
pub fn first_time(table: &str, column: &str) -> Statement {
    (
        format!("SELECT {column} FROM {table} WHERE {column} IS NOT NULL ORDER BY rowid LIMIT 1"),
        None,
    )
}

/// The earliest and latest time of each `(table, time column)`.
pub fn time_ranges(tables: &[(String, String)]) -> Statement {
    let selects: Vec<String> = tables
        .iter()
        .enumerate()
        .map(|(index, (table, column))| {
            format!(
                "SELECT\n  {index} as table_index,\n  MIN({column}) as min_date,\n  MAX({column}) as max_date\nFROM {table}"
            )
        })
        .collect();
    (selects.join("\nUNION ALL\n"), None)
}

/// Rows of an entity table: `num_rows` of them in `order_by`, a random draw
/// without one, or the rows holding `ids` in `key`, bound as parameters.
#[allow(clippy::too_many_arguments)]
pub fn entity_rows(
    table: &str,
    projections: &[String],
    filters: &[String],
    order_by: Option<&str>,
    num_rows: Option<i64>,
    key: Option<&str>,
    ids: Option<&[Param]>,
) -> Statement {
    let mut conditions = filters.to_vec();
    let mut params = None;
    if let Some(ids) = ids {
        let markers = vec!["?"; ids.len()].join(", ");
        conditions.push(format!(" {} IN ({markers})", opt(key)));
        params = Some(ids.to_vec());
    }
    let mut sql = format!("SELECT {}\nFROM {table}", projections.join(", "));
    if !conditions.is_empty() {
        sql.push_str(&format!("\nWHERE{}", conditions.join(" AND")));
    }
    match (ids, order_by) {
        (None, None) => sql.push_str(&format!("\nORDER BY RANDOM() LIMIT {}", opt(num_rows))),
        (None, Some(order_by)) => {
            sql.push_str(&format!("\nORDER BY {order_by} LIMIT {}", opt(num_rows)));
        }
        (Some(_), order_by) => sql.push_str(&format!("\nORDER BY {}", opt(order_by))),
    }
    (sql, params)
}

/// The SQLite type of a key value of `arrow_type`: the type the key tables
/// written with `adbc_ingest` gave it.
pub fn sqlite_type(arrow_type: &ArrowType) -> Option<&'static str> {
    match arrow_type {
        ArrowType::Int { .. } | ArrowType::Bool => Some("INTEGER"),
        t if t.is_floating() => Some("REAL"),
        t if t.is_string() || t.is_date() => Some("TEXT"),
        _ => None,
    }
}

/// `values` as JSON holds them, a timestamp or date as the text
/// `adbc_ingest` writes for it.
pub fn json_values(values: &ArrowArray) -> Result<Vec<Py>, Error> {
    match &values.data_type {
        ArrowType::Timestamp(..) => Ok(values
            .strftime(false)?
            .into_iter()
            .map(|v| v.map_or(Py::None, Py::Str))
            .collect()),
        t if t.is_date() => Ok(values
            .to_pylist()?
            .into_iter()
            .map(|v| match v {
                Py::Date(text) => Py::Str(text),
                other => other,
            })
            .collect()),
        t if t.is_floating() => Ok(values
            .values
            .iter()
            .map(|v| match v {
                Py::Float(x) if x.is_nan() => Py::None,
                other => other.clone(),
            })
            .collect()),
        _ => values.to_pylist(),
    }
}

/// The `WITH` clause that reads key columns from a JSON parameter, and that
/// parameter.
#[derive(Clone, Debug, PartialEq)]
pub struct Keys {
    pub sql: String,
    pub param: String,
}

/// The `WITH __kumo_keys__` clause over `columns`: `__kumo_batch__` each
/// element's position, and one column per `(name, values)`, a single column
/// as an array of values, several as an array of arrays.
pub fn keys(columns: &[(String, ArrowArray)]) -> Result<Keys, Error> {
    let values = columns
        .iter()
        .map(|(_, array)| json_values(array))
        .collect::<Result<Vec<_>, _>>()?;
    let mut selects = vec!["key AS __kumo_batch__".to_string()];
    for (index, (name, array)) in columns.iter().enumerate() {
        let reference = if columns.len() == 1 {
            "value".to_string()
        } else {
            format!("json_extract(value, '$[{index}]')")
        };
        selects.push(match sqlite_type(&array.data_type) {
            None => format!("{reference} AS {name}"),
            Some(cast) => format!("CAST({reference} AS {cast}) AS {name}"),
        });
    }
    let payload = if columns.len() == 1 {
        Py::List(values.into_iter().next().unwrap_or_default())
    } else {
        Py::List(text::zip_rows(&values))
    };
    Ok(Keys {
        sql: format!(
            "WITH __kumo_keys__ AS (\n  SELECT {}\n  FROM json_each(?)\n)\n",
            selects.join(", ")
        ),
        param: Dumps::COMPACT.dumps(&payload)?,
    })
}

/// For each id, one row of `table` holding it in `key`, with the id's
/// position as `__kumo_batch__`; of several, the first in `tie_break`, then
/// `rowid`, order.
#[allow(clippy::too_many_arguments)]
pub fn rows_by_key(
    table: &str,
    key: &str,
    ids: &ArrowArray,
    projections: &[String],
    unique: bool,
    tie_break: &[String],
    count_matches: bool,
) -> Result<Statement, Error> {
    let keys = keys(&[("__kumo_id__".to_string(), ids.clone())])?;
    let matches = if count_matches {
        format!(
            "(SELECT COUNT(*) FROM {table} WHERE {key} == tmp.__kumo_id__) as __kumo_matches__, "
        )
    } else {
        String::new()
    };
    let mut sql = format!(
        "{}SELECT tmp.__kumo_batch__ as __kumo_batch__, {matches}{}\nFROM __kumo_keys__ tmp\nJOIN {table} ent\n",
        keys.sql,
        projections.join(", ")
    );
    if unique {
        sql.push_str(&format!("  ON {key} = tmp.__kumo_id__"));
    } else {
        let mut order: Vec<String> = tie_break.to_vec();
        order.push("rowid".into());
        sql.push_str(&format!(
            "  ON ent.rowid = (\n    SELECT rowid\n    FROM {table}\n    WHERE {key} == tmp.__kumo_id__\n    ORDER BY {}\n    LIMIT 1\n)",
            order.join(", ")
        ));
    }
    Ok((sql, Some(vec![Param::Text(keys.param)])))
}

/// For each id, the first `num_neighbors` rows of `table` holding it in
/// `key`, in `order_by` order, no later than its end time if given.
#[allow(clippy::too_many_arguments)]
pub fn neighbors(
    table: &str,
    key: &str,
    ids: &ArrowArray,
    projections: &[String],
    num_neighbors: i64,
    order_by: &str,
    time: Option<&str>,
    end_times: Option<&ArrowArray>,
) -> Result<Statement, Error> {
    let mut columns = vec![("__kumo_id__".to_string(), ids.clone())];
    if let Some(end_times) = end_times {
        columns.push(("__kumo_time__".to_string(), end_times.clone()));
    }
    let keys = keys(&columns)?;
    let mut sql = format!(
        "{}SELECT tmp.__kumo_batch__ as __kumo_batch__, {}\nFROM __kumo_keys__ tmp\nJOIN {table} fact\nON fact.rowid IN (\n  SELECT rowid\n  FROM {table}\n  WHERE {key} = tmp.__kumo_id__\n",
        keys.sql,
        projections.join(", ")
    );
    if end_times.is_some() {
        sql.push_str(&format!("  AND {} <= tmp.__kumo_time__\n", opt(time)));
    }
    sql.push_str(&format!("  ORDER BY {order_by}\n"));
    sql.push_str(&format!("  LIMIT {num_neighbors}\n)\n"));
    sql.push_str(&format!("ORDER BY __kumo_batch__, {order_by}"));
    Ok((sql, Some(vec![Param::Text(keys.param)])))
}

/// For each id, the rows of `table` holding it in `key` whose time is in its
/// window: no later than its end, and later than its start if given.
#[allow(clippy::too_many_arguments)]
pub fn window(
    table: &str,
    key: &str,
    ids: &ArrowArray,
    projections: &[String],
    time: &str,
    end_times: &ArrowArray,
    start_times: Option<&ArrowArray>,
    order_by: &str,
) -> Result<Statement, Error> {
    let mut columns = vec![
        ("__kumo_id__".to_string(), ids.clone()),
        ("__kumo_end__".to_string(), end_times.clone()),
    ];
    if let Some(start_times) = start_times {
        columns.push(("__kumo_start__".to_string(), start_times.clone()));
    }
    let keys = keys(&columns)?;
    let mut sql = format!(
        "{}SELECT tmp.__kumo_batch__ as __kumo_batch__, {}\nFROM __kumo_keys__ tmp\nJOIN {table}\n  ON {key} = tmp.__kumo_id__\n AND {time} <= tmp.__kumo_end__",
        keys.sql,
        projections.join(", ")
    );
    if start_times.is_some() {
        sql.push_str(&format!("\n AND {time} > tmp.__kumo_start__"));
    }
    sql.push_str(&format!("\nORDER BY __kumo_batch__, {order_by}"));
    Ok((sql, Some(vec![Param::Text(keys.param)])))
}
