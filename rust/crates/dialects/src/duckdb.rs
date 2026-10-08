// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The SQL the engine runs against DuckDB (`backend/duckdb/dialect.py`).
//!
//! Parameters are bound to `?` markers. A key list travels as one parameter
//! holding a JSON array, which `from_json` reads into typed rows; each value
//! takes the DuckDB type of the Arrow type it has.

use crate::arrow::{ArrowArray, ArrowType, Unit};
use crate::fragments::{count_distinct_values, key_count_select, quote_ident, repeated_key_select};
use crate::int::Int;
use crate::py::Py;
use crate::pyjson::Dumps;
use crate::{Error, KeyCountPart, Param, Statement, opt, text};

/// A table is read by its bare name, which resolves in the current schema.
const IN_CURRENT_SCHEMA: &str =
    "database_name = current_database() AND schema_name = current_schema()";

/// Every table of the schema a bare name resolves in, by name.
pub fn list_tables() -> Statement {
    (
        "SELECT table_name FROM duckdb_tables() WHERE NOT temporary AND NOT internal \
         AND database_name = current_database() AND schema_name = current_schema() \
         ORDER BY table_name"
            .into(),
        None,
    )
}

/// The columns of `table`, a quoted name.
pub fn table_info(table: &str) -> Statement {
    (format!("PRAGMA table_info({table})"), None)
}

/// The constraints of the table named `table`, unquoted.
pub fn key_constraints(table: &str) -> Statement {
    let name = quote_ident(table, "'");
    (
        format!(
            "SELECT constraint_type, constraint_column_names\nFROM duckdb_constraints()\nWHERE table_name = {name}\n  AND {IN_CURRENT_SCHEMA}"
        ),
        None,
    )
}

/// The foreign keys of the table named `table`, unquoted.
pub fn foreign_keys(table: &str) -> Statement {
    let name = quote_ident(table, "'");
    (
        format!(
            "SELECT constraint_index, referenced_table,\n       constraint_column_names, referenced_column_names\nFROM duckdb_constraints()\nWHERE table_name = {name}\n  AND {IN_CURRENT_SCHEMA}\n  AND constraint_type = 'FOREIGN KEY'\nORDER BY constraint_index"
        ),
        None,
    )
}

/// `projections` of `num_rows` rows of `table`.
pub fn first_rows(table: &str, projections: &[String], num_rows: i64) -> Statement {
    (
        format!(
            "SELECT {} FROM {table} LIMIT {num_rows}",
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
            "SELECT 1 FROM {table}\nWHERE ({identity}) IN (\n  SELECT unnest(from_json(?, '[\"VARCHAR\"]'))\n)\nLIMIT 1"
        ),
        Some(vec![Param::Text(text::json_text_list(
            values,
            Dumps::COMPACT,
        ))]),
    )
}

/// The rows of `table`.
pub fn count_rows(table: &str) -> Statement {
    (format!("SELECT COUNT(*) FROM {table}"), None)
}

/// The distinct values of a key, leaving out a row missing any part.
pub fn count_distinct(columns: &[String]) -> String {
    if columns.len() == 1 {
        return count_distinct_values(columns);
    }
    let present: Vec<String> = columns.iter().map(|c| format!("{c} IS NOT NULL")).collect();
    format!(
        "COUNT(DISTINCT ({})) FILTER (WHERE {})",
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
                &count_distinct(&part.columns),
            )
        })
        .collect();
    (selects.join("\nUNION ALL\n"), None)
}

/// One value of the key `columns` more than one row holds.
pub fn repeated_key(table: &str, columns: &[String]) -> Statement {
    (repeated_key_select(table, columns), None)
}

/// Every index, by table, with the expressions it indexes.
pub fn indices() -> Statement {
    (
        "SELECT table_name, expressions FROM duckdb_indexes()".into(),
        None,
    )
}

/// The time `column` as a `TIMESTAMP`; a zoned one as its UTC wall-clock
/// time.
pub fn time_column(column: &str, zoned: bool) -> String {
    if zoned {
        format!("timezone('UTC', {column})")
    } else {
        format!("CAST({column} AS TIMESTAMP)")
    }
}

/// The earliest and latest time of each `(table, time)`.
pub fn time_ranges(tables: &[(String, String)]) -> Statement {
    let selects: Vec<String> = tables
        .iter()
        .enumerate()
        .map(|(index, (table, time))| {
            format!(
                "SELECT\n  {index} as table_index,\n  MIN({time}) as min_date,\n  MAX({time}) as max_date\nFROM {table}"
            )
        })
        .collect();
    (selects.join("\nUNION ALL\n"), None)
}

/// A reservoir sample of `num_rows` rows of `table` holding `filters`,
/// repeatable for a seed, put in `order_by` order.
pub fn random_rows(
    table: &str,
    projections: &[String],
    filters: &[String],
    num_rows: i64,
    random_seed: Option<&Int>,
    order_by: &str,
) -> Statement {
    let mut sql = format!("SELECT {}\n", projections.join(", "));
    sql.push_str(&format!("FROM (SELECT * FROM {table}"));
    if !filters.is_empty() {
        sql.push_str(&format!("\nWHERE{}", filters.join(" AND")));
    }
    sql.push_str(&format!(")\nUSING SAMPLE reservoir({num_rows} ROWS)"));
    if let Some(seed) = random_seed {
        sql.push_str(&format!(" REPEATABLE ({seed})"));
    }
    sql.push_str(&format!("\nORDER BY {order_by}"));
    (sql, None)
}

/// The rows of `table` holding `ids` in `key` and `filters`, in `order_by`
/// order, the ids bound as parameters.
pub fn rows_by_ids(
    table: &str,
    projections: &[String],
    filters: &[String],
    key: &str,
    ids: &[Param],
    order_by: &str,
) -> Statement {
    let markers = vec!["?"; ids.len()].join(", ");
    let mut conditions = filters.to_vec();
    conditions.push(format!(" {key} IN ({markers})"));
    let sql = format!(
        "SELECT {}\nFROM {table}\nWHERE{}\nORDER BY {order_by}",
        projections.join(", "),
        conditions.join(" AND")
    );
    (sql, Some(ids.to_vec()))
}

/// The DuckDB type of a key value of `arrow_type`, as DuckDB reads an Arrow
/// column of it.
pub fn duckdb_type(arrow_type: &ArrowType) -> String {
    match arrow_type {
        ArrowType::Bool => "BOOLEAN".into(),
        ArrowType::Int { bits, signed } => {
            let name = match bits {
                8 => "TINYINT",
                16 => "SMALLINT",
                32 => "INTEGER",
                _ => "BIGINT",
            };
            if *signed {
                name.into()
            } else {
                format!("U{name}")
            }
        }
        ArrowType::Float32 => "FLOAT".into(),
        t if t.is_floating() => "DOUBLE".into(),
        t if t.is_date() => "DATE".into(),
        ArrowType::Timestamp(_, Some(_)) => "TIMESTAMPTZ".into(),
        ArrowType::Timestamp(unit, None) => match unit {
            Unit::Second => "TIMESTAMP_S",
            Unit::Milli => "TIMESTAMP_MS",
            Unit::Micro => "TIMESTAMP",
            Unit::Nano => "TIMESTAMP_NS",
        }
        .into(),
        ArrowType::Decimal {
            precision, scale, ..
        } => format!("DECIMAL({precision}, {scale})"),
        _ => "VARCHAR".into(),
    }
}

/// `values` as JSON holds them: a timestamp in ISO 8601 to its last digit,
/// with its offset where it has a zone, a date in ISO 8601, a decimal as its
/// exact text.
pub fn json_values(values: &ArrowArray) -> Result<Vec<Py>, Error> {
    let data_type = &values.data_type;
    let texts = |items: Vec<Py>| -> Result<Vec<Py>, Error> {
        items
            .into_iter()
            .map(|v| {
                Ok(match v {
                    Py::None => Py::None,
                    other => Py::Str(other.py_str()?),
                })
            })
            .collect()
    };
    match data_type {
        ArrowType::Timestamp(_, zone) => Ok(values
            .strftime(zone.is_some())?
            .into_iter()
            .map(|v| v.map_or(Py::None, Py::Str))
            .collect()),
        t if t.is_date() => texts(values.to_pylist()?),
        ArrowType::Decimal { .. } => texts(values.to_pylist()?),
        t if t.is_floating() => Ok(values
            .values
            .iter()
            .map(|v| match v {
                Py::Float(x) if x.is_nan() => Py::None,
                other => other.clone(),
            })
            .collect()),
        ArrowType::Int { .. } | ArrowType::Bool | ArrowType::String | ArrowType::LargeString => {
            Ok(values.values.clone())
        }
        _ => texts(values.to_pylist()?),
    }
}

/// The `WITH` clause that reads key columns from a JSON parameter, and that
/// parameter.
#[derive(Clone, Debug, PartialEq)]
pub struct Keys {
    pub sql: String,
    pub param: String,
}

/// The `WITH __kumo_keys__` clause over `columns`: each row one object of
/// the JSON array, `__kumo_batch__` its position, and one column per
/// `(name, values)`, typed as DuckDB reads an Arrow column of the values'
/// type.
pub fn keys(columns: &[(String, ArrowArray)]) -> Result<Keys, Error> {
    let fields: Vec<String> = (0..columns.len()).map(|i| format!("f{i}")).collect();
    let structure = Py::Dict(
        fields
            .iter()
            .zip(columns)
            .map(|(field, (_, array))| (field.clone(), Py::Str(duckdb_type(&array.data_type))))
            .collect(),
    );
    let values = columns
        .iter()
        .map(|(_, array)| json_values(array))
        .collect::<Result<Vec<_>, _>>()?;
    let rows = text::zip_rows(&values);
    let payload = Py::List(
        rows.into_iter()
            .map(|row| match row {
                Py::List(items) => Py::Dict(fields.iter().cloned().zip(items).collect()),
                other => other,
            })
            .collect(),
    );
    let mut selects = vec!["k.i - 1 AS __kumo_batch__".to_string()];
    selects.extend(
        fields
            .iter()
            .zip(columns)
            .map(|(field, (name, _))| format!("k.v.{field} AS {name}")),
    );
    let shape = quote_ident(&Dumps::COMPACT.dumps(&Py::List(vec![structure]))?, "'");
    Ok(Keys {
        sql: format!(
            "WITH __kumo_keys__ AS (\n  SELECT {}\n  FROM unnest(from_json(?, {shape}))\n    WITH ORDINALITY AS k(v, i)\n)\n",
            selects.join(", ")
        ),
        param: Dumps::COMPACT.dumps(&payload)?,
    })
}

/// For each id, one row of `table` holding it in `key`, with the id's
/// position and how many rows hold it; of several, the first in `order_by`
/// order.
pub fn rows_by_key(
    table: &str,
    key: &str,
    ids: &ArrowArray,
    projections: &[String],
    order_by: &str,
) -> Result<Statement, Error> {
    let keys = keys(&[("__kumo_id__".to_string(), ids.clone())])?;
    let sql = format!(
        "{}SELECT tmp.__kumo_batch__, COUNT(*) OVER (PARTITION BY tmp.__kumo_batch__) AS __kumo_matches__, {}\nFROM __kumo_keys__ tmp\nJOIN {table}\n  ON {key} = tmp.__kumo_id__\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY tmp.__kumo_batch__\n  ORDER BY {order_by}\n) = 1",
        keys.sql,
        projections.join(", ")
    );
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
        "{}SELECT tmp.__kumo_batch__, {}\nFROM __kumo_keys__ tmp\nJOIN {table}\n  ON {key} = tmp.__kumo_id__\n",
        keys.sql,
        projections.join(", ")
    );
    if end_times.is_some() {
        sql.push_str(&format!(" AND {} <= tmp.__kumo_time__\n", opt(time)));
    }
    sql.push_str("QUALIFY ROW_NUMBER() OVER (\n  PARTITION BY tmp.__kumo_batch__\n");
    sql.push_str(&format!("  ORDER BY {order_by}\n"));
    sql.push_str(&format!(") <= {num_neighbors}\n"));
    sql.push_str(&format!("ORDER BY tmp.__kumo_batch__, {order_by}"));
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
        "{}SELECT tmp.__kumo_batch__, {}\nFROM __kumo_keys__ tmp\nJOIN {table}\n  ON {key} = tmp.__kumo_id__\n AND {time} <= tmp.__kumo_end__",
        keys.sql,
        projections.join(", ")
    );
    if start_times.is_some() {
        sql.push_str(&format!("\n AND {time} > tmp.__kumo_start__"));
    }
    sql.push_str(&format!("\nORDER BY tmp.__kumo_batch__, {order_by}"));
    Ok((sql, Some(vec![Param::Text(keys.param)])))
}
