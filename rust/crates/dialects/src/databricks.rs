// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The SQL the engine runs against Databricks (`backend/databricks/dialect.py`).
//!
//! Parameters are bound to `?` markers, and identifiers are quoted with
//! backticks. A key list travels as one parameter holding a JSON array,
//! which `posexplode(from_json(...))` turns into rows; a list longer than the
//! 1 MB a parameter may hold is split into chunks.

use crate::fragments::{
    count_distinct_values, key_count_select, ordered_sample_sql, repeated_key_select,
};
use crate::int::Int;
use crate::postgres::item;
use crate::py::{Dtype, Py};
use crate::pyjson::Dumps;
use crate::text::{py_float, py_int};
use crate::{Error, KeyCountPart, Param, Statement, opt};

/// The catalog and schema a bare name resolves in.
pub fn current_namespace() -> Statement {
    ("SELECT current_catalog(), current_schema()".into(), None)
}

/// Every table of `schema` in the quoted `catalog` but its metric views.
pub fn list_tables(catalog: &str, schema: &str) -> Statement {
    let pad = " ".repeat(20);
    (
        format!(
            "\n{pad}SELECT table_name\n{pad}FROM {catalog}.information_schema.tables\n{pad}WHERE table_schema = ?\n{pad}  AND table_type != 'METRIC_VIEW'\n{pad}ORDER BY table_name\n{pad}"
        ),
        Some(vec![Param::Text(schema.into())]),
    )
}

/// The columns of `schema.table` in the quoted `catalog`, in order.
pub fn columns(catalog: &str, schema: &str, table: &str) -> Statement {
    (
        format!(
            "SELECT c.column_name, c.full_data_type, c.is_nullable,\n       c.table_catalog, c.table_schema, c.table_name,\n       t.table_type\nFROM {catalog}.information_schema.columns c\nJOIN {catalog}.information_schema.tables t\n  ON t.table_catalog = c.table_catalog\n AND t.table_schema = c.table_schema\n AND t.table_name = c.table_name\nWHERE c.table_schema = lower(?)\n  AND c.table_name = lower(?)\nORDER BY c.ordinal_position"
        ),
        Some(vec![Param::Text(schema.into()), Param::Text(table.into())]),
    )
}

/// The primary and unique keys of `schema.table` in the quoted `catalog`.
pub fn key_constraints(catalog: &str, schema: &str, table: &str) -> Statement {
    (
        format!(
            "SELECT tc.constraint_type, tc.constraint_name,\n       kcu.column_name\nFROM {catalog}.information_schema.table_constraints tc\nJOIN {catalog}.information_schema.key_column_usage kcu\n  ON tc.constraint_catalog = kcu.constraint_catalog\n AND tc.constraint_schema = kcu.constraint_schema\n AND tc.constraint_name = kcu.constraint_name\nWHERE tc.table_schema = lower(?)\n  AND tc.table_name = lower(?)\n  AND tc.constraint_type IN ('PRIMARY KEY', 'UNIQUE')"
        ),
        Some(vec![Param::Text(schema.into()), Param::Text(table.into())]),
    )
}

/// The foreign keys of `schema.table` in the quoted `catalog`.
pub fn foreign_keys(catalog: &str, schema: &str, table: &str) -> Statement {
    (
        format!(
            "SELECT tc.constraint_name, kcu.column_name,\n       ccu.table_catalog, ccu.table_schema,\n       ccu.table_name, ccu.column_name\nFROM {catalog}.information_schema.table_constraints tc\nJOIN {catalog}.information_schema.referential_constraints rc\n  ON tc.constraint_catalog = rc.constraint_catalog\n AND tc.constraint_schema = rc.constraint_schema\n AND tc.constraint_name = rc.constraint_name\nJOIN {catalog}.information_schema.key_column_usage kcu\n  ON tc.constraint_catalog = kcu.constraint_catalog\n AND tc.constraint_schema = kcu.constraint_schema\n AND tc.constraint_name = kcu.constraint_name\nJOIN {catalog}.information_schema.constraint_column_usage ccu\n  ON rc.unique_constraint_catalog = ccu.constraint_catalog\n AND rc.unique_constraint_schema = ccu.constraint_schema\n AND rc.unique_constraint_name = ccu.constraint_name\nWHERE tc.constraint_type = 'FOREIGN KEY'\n  AND tc.table_schema = lower(?)\n  AND tc.table_name = lower(?)\nORDER BY tc.constraint_name"
        ),
        Some(vec![Param::Text(schema.into()), Param::Text(table.into())]),
    )
}

/// The columns and details of the quoted `view`.
pub fn describe_extended(view: &str) -> Statement {
    (format!("DESCRIBE TABLE EXTENDED {view}"), None)
}

/// The details of the quoted `table`, its `sizeInBytes` among them.
pub fn describe_detail(table: &str) -> Statement {
    (format!("DESCRIBE DETAIL {table}"), None)
}

/// `table` cut to the first `num_rows` rows Databricks reads of it.
pub fn bounded(table: &str, num_rows: i64) -> String {
    format!("(SELECT * FROM {table} LIMIT {num_rows})")
}

/// `projections` of the first `num_rows` rows of `table` in the order
/// `order_terms` give.
pub fn sample(
    table: &str,
    projections: &[String],
    order_terms: &[String],
    num_rows: i64,
) -> Statement {
    (
        ordered_sample_sql(table, projections, order_terms, num_rows),
        None,
    )
}

/// Whether a row of `table` holds one of `values` in `identity`.
pub fn holds_any(table: &str, identity: &str, values: &[String]) -> Statement {
    (
        format!(
            "SELECT 1 FROM {table}\nWHERE ({identity}) IN (\n  SELECT col FROM explode(from_json(?, 'array<string>'))\n)\nLIMIT 1"
        ),
        Some(vec![Param::Text(crate::text::json_text_list(
            values,
            Dumps::WITH_DEFAULT,
        ))]),
    )
}

/// The rows of `table`.
pub fn count_rows(table: &str) -> Statement {
    (format!("SELECT COUNT(*) FROM {table}"), None)
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
                &count_distinct_values(&part.columns),
            )
        })
        .collect();
    (selects.join("\nUNION ALL\n"), None)
}

/// One value of the key `columns` more than one row holds.
pub fn repeated_key(table: &str, columns: &[String]) -> Statement {
    (repeated_key_select(table, columns), None)
}

/// `dumps(obj)`: compact JSON, numpy scalars as Python ones and anything
/// else JSON has no value for as its text.
pub fn dumps(obj: &Py) -> Result<String, Error> {
    Dumps::WITH_DEFAULT.dumps(obj)
}

/// `rows` in chunks whose JSON stays below `max_bytes`.
pub fn chunk_rows(rows: &[Py], max_bytes: i64) -> Result<Vec<Vec<Py>>, Error> {
    crate::postgres::chunk_rows(rows, max_bytes)
}

/// The Databricks SQL type a key of `dtype` is read from JSON as.
pub fn element_type(dtype: Dtype) -> &'static str {
    if dtype.is_int() {
        "bigint"
    } else if dtype.is_float() {
        "double"
    } else {
        "string"
    }
}

/// A key as the Python type matching `dtype`.
pub fn coerce_id(value: &Py, dtype: Dtype) -> Result<Py, Error> {
    if dtype.is_int() {
        return Ok(Py::Int(py_int(value)?));
    }
    if dtype.is_float() {
        return Ok(Py::Float(py_float(value)?));
    }
    Ok(Py::Str(value.py_str()?))
}

/// The earliest and latest time of each `(table, time column)`.
pub fn time_ranges(tables: &[(String, String)]) -> Statement {
    crate::sqlite::time_ranges(tables)
}

/// Rows of an entity table: `num_rows` of a `percent` sample, or the rows
/// holding `ids` in `key`, joined from a JSON parameter in their order.
#[allow(clippy::too_many_arguments)]
pub fn entity_rows(
    table: &str,
    projections: &[String],
    filters: &[String],
    key: &str,
    order_by: &str,
    percent: Option<f64>,
    random_seed: Option<&Int>,
    num_rows: Option<i64>,
    ids: Option<&[Py]>,
    dtype: Option<Dtype>,
) -> Result<Statement, Error> {
    let mut sql = format!("SELECT {}\nFROM {table}", projections.join(", "));
    let mut params = None;
    match ids {
        None => {
            let percent = percent.map(sdm_pql::pyrepr::float_repr);
            sql.push_str(&format!("\nTABLESAMPLE ({} PERCENT)", opt(percent)));
            if let Some(seed) = random_seed {
                sql.push_str(&format!(" REPEATABLE ({seed})"));
            }
        }
        Some(ids) => {
            let dtype = dtype.ok_or_else(|| Error::new("entity_rows by ids needs a dtype"))?;
            sql.push_str(&format!(
                "\nJOIN posexplode(from_json(?, 'array<{}>')) AS __ids__(__pos__, __KUMO_ID__)\n  ON {key} = __ids__.__KUMO_ID__",
                element_type(dtype)
            ));
            let coerced = ids
                .iter()
                .map(|value| coerce_id(value, dtype))
                .collect::<Result<Vec<_>, _>>()?;
            params = Some(vec![Param::Text(dumps(&Py::List(coerced))?)]);
        }
    }
    if !filters.is_empty() {
        sql.push_str(&format!("\nWHERE{}", filters.join(" AND")));
    }
    if ids.is_some() {
        sql.push_str(&format!("\nORDER BY __ids__.__pos__, {order_by}"));
    } else if let Some(seed) = random_seed {
        sql.push_str(&format!("\nORDER BY xxhash64({seed}, {key}), {order_by}"));
    }
    if ids.is_none() {
        sql.push_str(&format!("\nLIMIT {}", opt(num_rows)));
    }
    Ok((sql, params))
}

/// The rows of `table` holding each of `ids` in `key`, with the id's
/// position as `__KUMO_BATCH__`, in that order and then `order_by`.
pub fn rows_by_key(
    table: &str,
    projections: &[String],
    key: &str,
    dtype: Dtype,
    ids: &Py,
    order_by: &[String],
) -> Result<Statement, Error> {
    let mut order = vec!["TMP.__KUMO_BATCH__".to_string()];
    order.extend(order_by.iter().cloned());
    let sql = format!(
        "WITH TMP AS (\n  SELECT pos AS __KUMO_BATCH__, col AS __KUMO_ID__\n  FROM posexplode(from_json(?, 'array<{}>'))\n)\nSELECT TMP.__KUMO_BATCH__ as __KUMO_BATCH__, {}\nFROM TMP\nJOIN {table}\n  ON {key} = TMP.__KUMO_ID__\nORDER BY {}",
        element_type(dtype),
        projections.join(", "),
        order.join(", ")
    );
    Ok((sql, Some(vec![Param::Text(dumps(ids)?)])))
}

fn bound_params(rows: &Py, bounds: &[String]) -> Result<Vec<Param>, Error> {
    let mut params = vec![Param::Text(dumps(rows)?)];
    params.extend(bounds.iter().cloned().map(Param::Text));
    Ok(params)
}

/// For each of `rows`, the first `num_neighbors` rows of `table` holding it
/// in `key` and in its window if `windowed`, in `order_by` order.
#[allow(clippy::too_many_arguments)]
pub fn neighbors(
    table: &str,
    projections: &[String],
    key: &str,
    dtype: Dtype,
    rows: &Py,
    windowed: bool,
    time: Option<&str>,
    bounds: &[String],
    order_by: &str,
    num_neighbors: i64,
) -> Result<Statement, Error> {
    let elem = element_type(dtype);
    let mut select = vec!["pos AS __KUMO_BATCH__".to_string()];
    let schema = if windowed {
        select.push("col.id AS __KUMO_ID__".into());
        select.push("col.e AS __KUMO_END_TIME__".into());
        select.push("col.s AS __KUMO_START_TIME__".into());
        format!("array<struct<id:{elem},e:string,s:string>>")
    } else {
        select.push("col AS __KUMO_ID__".into());
        format!("array<{elem}>")
    };
    let mut sql = format!(
        "WITH TMP AS (\n  SELECT {}\n  FROM posexplode(from_json(?, '{schema}'))\n)\nSELECT TMP.__KUMO_BATCH__ as __KUMO_BATCH__, {}\nFROM TMP\nJOIN {table}\n  ON {key} = TMP.__KUMO_ID__\n",
        select.join(", "),
        projections.join(", ")
    );
    if windowed {
        let time = opt(time);
        sql.push_str(&format!(
            " AND {time} <= TMP.__KUMO_END_TIME__\n AND {time} > TMP.__KUMO_START_TIME__\nWHERE {time} <= ?\n  AND {time} > ?\n"
        ));
    }
    sql.push_str("QUALIFY ROW_NUMBER() OVER (\n  PARTITION BY TMP.__KUMO_BATCH__\n");
    sql.push_str(&format!("  ORDER BY {order_by}\n"));
    sql.push_str(&format!(") <= {num_neighbors}\n"));
    sql.push_str(&format!("ORDER BY TMP.__KUMO_BATCH__, {order_by}"));
    Ok((sql, Some(bound_params(rows, bounds)?)))
}

/// For each of `rows`, the rows of `table` holding it in `key` in its
/// window, in `order_by` order.
#[allow(clippy::too_many_arguments)]
pub fn window(
    table: &str,
    projections: &[String],
    key: &str,
    dtype: Dtype,
    rows: &Py,
    with_start: bool,
    time: &str,
    bounds: &[String],
    order_by: &str,
) -> Result<Statement, Error> {
    let elem = element_type(dtype);
    let schema = if with_start {
        format!("array<struct<id:{elem},e:string,s:string>>")
    } else {
        format!("array<struct<id:{elem},e:string>>")
    };
    let mut select = vec![
        "pos AS __KUMO_BATCH__",
        "col.id AS __KUMO_ID__",
        "col.e AS __KUMO_END_TIME__",
    ];
    if with_start {
        select.push("col.s AS __KUMO_START_TIME__");
    }
    let mut sql = format!(
        "WITH TMP AS (\n  SELECT {}\n  FROM posexplode(from_json(?, '{schema}'))\n)\nSELECT TMP.__KUMO_BATCH__ as __KUMO_BATCH__, {}\nFROM TMP\nJOIN {table}\n  ON {key} = TMP.__KUMO_ID__\n AND {time} <= TMP.__KUMO_END_TIME__\n",
        select.join(", "),
        projections.join(", ")
    );
    if with_start {
        sql.push_str(&format!("AND {time} > TMP.__KUMO_START_TIME__\n"));
    }
    sql.push_str(&format!("WHERE {time} <= ?"));
    if with_start {
        sql.push_str(&format!("\nAND {time} > ?"));
    }
    sql.push_str(&format!("\nORDER BY TMP.__KUMO_BATCH__, {order_by}"));
    Ok((sql, Some(bound_params(rows, bounds)?)))
}

/// `_json_default(obj)`: `obj.item()` where it has one, else `str(obj)`.
pub fn json_default(obj: &Py) -> Result<Py, Error> {
    match obj {
        Py::Np(..) | Py::Ndarray(..) => item(obj),
        other => Ok(Py::Str(other.py_str()?)),
    }
}
