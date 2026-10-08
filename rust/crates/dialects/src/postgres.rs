// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The SQL the engine runs against PostgreSQL (`backend/postgres/dialect.py`).
//!
//! Parameters are bound to `%s` markers. A key list travels as one parameter
//! holding a JSON array, which `jsonb_array_elements` turns into rows; a list
//! longer than one parameter should carry is split into chunks.

use blake2::digest::consts::U8;
use blake2::{Blake2b, Digest};

use crate::fragments::{key_count_select, ordered_sample_sql, repeated_key_select};
use crate::int::Int;
use crate::py::{Dtype, Py};
use crate::pyjson::Dumps;
use crate::text::{py_float, py_int};
use crate::time::Stamp;
use crate::{Error, KeyCountPart, Param, Statement, opt};

/// The schema a bare name resolves in.
pub fn current_schema() -> Statement {
    ("SELECT current_schema()".into(), None)
}

/// Every base table of `schema`, by name.
pub fn list_tables(schema: &str) -> Statement {
    (
        "SELECT table_name\nFROM information_schema.tables\nWHERE table_schema = %s\n  AND table_type = 'BASE TABLE'\nORDER BY table_name".into(),
        Some(vec![Param::Text(schema.into())]),
    )
}

/// The columns of `schema.table`, in order, with their types as PostgreSQL
/// writes them.
pub fn columns(schema: &str, table: &str) -> Statement {
    (
        "SELECT c.column_name, c.data_type, c.udt_name,\n       c.numeric_scale, c.is_nullable, c.table_schema,\n       c.table_name,\n       pg_catalog.format_type(a.atttypid, a.atttypmod)\nFROM information_schema.columns c\nJOIN pg_catalog.pg_namespace n\n  ON n.nspname = c.table_schema\nJOIN pg_catalog.pg_class rel\n  ON rel.relnamespace = n.oid AND rel.relname = c.table_name\nJOIN pg_catalog.pg_attribute a\n  ON a.attrelid = rel.oid AND a.attname = c.column_name\n AND a.attnum > 0 AND NOT a.attisdropped\nWHERE c.table_schema = %s AND c.table_name = %s\nORDER BY c.ordinal_position".into(),
        Some(vec![Param::Text(schema.into()), Param::Text(table.into())]),
    )
}

/// The primary, unique and foreign keys of `schema.table`, by column.
pub fn key_constraints(schema: &str, table: &str) -> Statement {
    (
        "SELECT tc.constraint_type, tc.constraint_name,\n       kcu.column_name, kcu.ordinal_position\nFROM information_schema.table_constraints tc\nJOIN information_schema.key_column_usage kcu\n  ON tc.constraint_catalog = kcu.constraint_catalog\n AND tc.constraint_schema = kcu.constraint_schema\n AND tc.constraint_name = kcu.constraint_name\nWHERE tc.table_schema = %s AND tc.table_name = %s\n  AND tc.constraint_type IN ('PRIMARY KEY', 'UNIQUE', 'FOREIGN KEY')".into(),
        Some(vec![Param::Text(schema.into()), Param::Text(table.into())]),
    )
}

/// The foreign keys of `schema.table`, with what each references.
pub fn foreign_keys(schema: &str, table: &str) -> Statement {
    (
        "SELECT tc.constraint_name, kcu.column_name,\n       target_kcu.table_schema, target_kcu.table_name,\n       target_kcu.column_name, kcu.ordinal_position\nFROM information_schema.table_constraints tc\nJOIN information_schema.referential_constraints rc\n  ON tc.constraint_catalog = rc.constraint_catalog\n AND tc.constraint_schema = rc.constraint_schema\n AND tc.constraint_name = rc.constraint_name\nJOIN information_schema.key_column_usage kcu\n  ON tc.constraint_catalog = kcu.constraint_catalog\n AND tc.constraint_schema = kcu.constraint_schema\n AND tc.constraint_name = kcu.constraint_name\nJOIN information_schema.key_column_usage target_kcu\n  ON rc.unique_constraint_catalog = target_kcu.constraint_catalog\n AND rc.unique_constraint_schema = target_kcu.constraint_schema\n AND rc.unique_constraint_name = target_kcu.constraint_name\n AND target_kcu.ordinal_position = kcu.position_in_unique_constraint\nWHERE tc.constraint_type = 'FOREIGN KEY'\n  AND tc.table_schema = %s AND tc.table_name = %s\nORDER BY tc.constraint_name, kcu.ordinal_position".into(),
        Some(vec![Param::Text(schema.into()), Param::Text(table.into())]),
    )
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

/// The kind of the relation `table` and the blocks it holds now.
pub fn relation_size(table: &str) -> Statement {
    (
        "SELECT c.relkind,\n       CASE WHEN c.relkind = 'p'\n            THEN (SELECT COALESCE(SUM(pg_relation_size(p.relid)), 0)::bigint\n                  FROM pg_partition_tree(c.oid) AS p)\n            ELSE pg_relation_size(c.oid)\n       END / current_setting('block_size')::bigint\nFROM pg_catalog.pg_class AS c\nWHERE c.oid = to_regclass(%s)".into(),
        Some(vec![Param::Text(table.into())]),
    )
}

/// `table` with a repeatable sample of `percent` of its blocks.
pub fn block_sample(table: &str, percent: &str) -> String {
    format!("{table} TABLESAMPLE SYSTEM ({percent}) REPEATABLE (0)")
}

/// Whether a row of `table` holds one of `values` in `identity`; `values`
/// travel as one JSON array.
pub fn holds_any(table: &str, identity: &str, values: &[String]) -> Statement {
    (
        format!(
            "SELECT 1 FROM {table}\nWHERE ({identity}) IN (\n  SELECT jsonb_array_elements_text(%s::jsonb)\n)\nLIMIT 1"
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

/// The distinct values of a key, leaving out a row missing any part.
pub fn count_distinct(columns: &[String]) -> String {
    crate::duckdb::count_distinct(columns)
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

/// `dumps(obj)`: compact JSON, numpy scalars as Python ones and anything
/// else JSON has no value for as its text.
pub fn dumps(obj: &Py) -> Result<String, Error> {
    Dumps::WITH_DEFAULT.dumps(obj)
}

/// `chunk_rows(rows, max_bytes)`: `rows` in chunks whose JSON stays below
/// `max_bytes`; one empty chunk for no rows.
pub fn chunk_rows(rows: &[Py], max_bytes: i64) -> Result<Vec<Vec<Py>>, Error> {
    let mut chunks = Vec::new();
    let mut current: Vec<Py> = Vec::new();
    let mut current_bytes: i64 = 2;
    for row in rows {
        let size = dumps(row)?.chars().count() as i64 + 1;
        if !current.is_empty() && current_bytes + size > max_bytes {
            chunks.push(std::mem::take(&mut current));
            current_bytes = 2;
        }
        current.push(row.clone());
        current_bytes += size;
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    if chunks.is_empty() {
        chunks.push(Vec::new());
    }
    Ok(chunks)
}

/// The type a key of `dtype` with no declared type is read as.
pub fn element_type(dtype: Dtype) -> &'static str {
    if dtype.is_int() {
        "bigint"
    } else if dtype.is_float() {
        "double precision"
    } else {
        "text"
    }
}

/// `value.item()` where it has one.
pub(crate) fn item(value: &Py) -> Result<Py, Error> {
    Ok(match value {
        Py::Np(_, inner) => (**inner).clone(),
        Py::Ndarray(_, values) if values.len() == 1 => values[0].clone(),
        Py::Ndarray(..) => {
            return Err(Error::new(
                "can only convert an array of size 1 to a Python scalar",
            ));
        }
        other => other.clone(),
    })
}

/// `value` as JSON holds a key of `dtype`, declared as `postgres_type`: a
/// `numeric` key as its text, so no digit is lost.
pub fn coerce_id(value: &Py, dtype: Dtype, postgres_type: &str) -> Result<Py, Error> {
    let value = item(value)?;
    if dtype.is_int() {
        return Ok(Py::Int(py_int(&value)?));
    }
    if dtype.is_float() {
        let lowered = postgres_type.to_lowercase();
        if lowered.starts_with("numeric") || lowered.starts_with("decimal") {
            return Ok(Py::Str(value.py_str()?));
        }
        return Ok(Py::Float(py_float(&value)?));
    }
    Ok(Py::Str(value.py_str()?))
}

/// `serialize_time(value, utc)`: a time bound as JSON holds it, with
/// `+00:00` under `utc`, or `None` for a missing one.
pub fn serialize_time(value: &Py, utc: bool) -> Result<Option<String>, Error> {
    let stamp = match value {
        Py::None | Py::NaT | Py::Na => return Ok(None),
        Py::Float(x) if x.is_nan() => return Ok(None),
        Py::Timestamp(stamp) => *stamp,
        Py::Datetime(text) => Stamp::parse(text)?,
        Py::Date(text) => Stamp::parse(&format!("{text}T00:00:00"))?,
        other => {
            return Err(Error::new(format!(
                "serialize_time of {other:?} is not ported"
            )));
        }
    };
    let stamp = if utc { stamp.to_utc() } else { stamp };
    Ok(Some(stamp.isoformat('T')))
}

/// `postgres_seed(random_seed)`: the seed's decimal digits hashed with
/// BLAKE2b to 8 bytes, mapped into `[-1, 1]`.
pub fn postgres_seed(random_seed: &Int) -> f64 {
    let digest = Blake2b::<U8>::digest(random_seed.to_string().as_bytes());
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&digest);
    let value = u64::from_be_bytes(bytes);
    // `2.0 * value / ((1 << 64) - 1) - 1.0`: Python rounds both integers to
    // the nearest float first.
    2.0 * (value as f64) / (u64::MAX as f64) - 1.0
}

/// The `ORDER BY` term of a column in a tie-break: a string column by its
/// text.
pub fn tie_break_term(reference: &str, is_string: bool) -> String {
    if is_string {
        format!("CAST({reference} AS text)")
    } else {
        reference.to_string()
    }
}

/// The `WITH TMP` clause that reads the key list, the statement's first
/// parameter, as rows.
pub fn json_cte(postgres_type: &str, fields: &[String], time_type: &str) -> String {
    let mut select = vec!["(ordinality - 1)::bigint AS __KUMO_BATCH__".to_string()];
    let has = |name: &str| fields.iter().any(|f| f == name);
    if fields.is_empty() {
        select.push(format!(
            "(value #>> '{{}}')::{postgres_type} AS __KUMO_ID__"
        ));
    } else {
        select.push(format!("(value->>'id')::{postgres_type} AS __KUMO_ID__"));
        if has("e") {
            select.push(format!("(value->>'e')::{time_type} AS __KUMO_END_TIME__"));
        }
        if has("s") {
            select.push(format!("(value->>'s')::{time_type} AS __KUMO_START_TIME__"));
        }
    }
    format!(
        "WITH TMP AS (\n  SELECT {}\n  FROM jsonb_array_elements(%s::jsonb) WITH ORDINALITY\n       AS input(value, ordinality)\n)\n",
        select.join(", ")
    )
}

/// The earliest and latest time of each `(table, time column)`.
pub fn time_ranges(tables: &[(String, String)]) -> Statement {
    let selects: Vec<String> = tables
        .iter()
        .enumerate()
        .map(|(index, (table, column))| {
            format!(
                "SELECT {index} AS table_index, MIN({column}) AS min_date, MAX({column}) AS max_date\nFROM {table}"
            )
        })
        .collect();
    (selects.join("\nUNION ALL\n"), None)
}

/// Seed `random()` for the session, from `random_seed`.
pub fn set_seed(random_seed: &Int) -> Statement {
    (
        "SELECT setseed(%s)".into(),
        Some(vec![Param::Float(postgres_seed(random_seed))]),
    )
}

/// `num_rows` rows of `table` holding `filters`, drawn by `random()`.
pub fn random_rows(
    table: &str,
    projections: &[String],
    filters: &[String],
    num_rows: i64,
) -> Statement {
    let mut sql = format!("SELECT {}\nFROM {table}", projections.join(", "));
    if !filters.is_empty() {
        sql.push_str(&format!("\nWHERE {}", filters.join(" AND ")));
    }
    sql.push_str(&format!("\nORDER BY random()\nLIMIT {num_rows}"));
    (sql, None)
}

/// The rows of `table` holding `ids`, a chunk of them, in `key`, and
/// `filters`, in the order of the ids, then `order_by`.
#[allow(clippy::too_many_arguments)]
pub fn rows_by_ids(
    table: &str,
    projections: &[String],
    filters: &[String],
    key: &str,
    postgres_type: &str,
    ids: &Py,
    order_by: &str,
) -> Result<Statement, Error> {
    let mut sql = json_cte(postgres_type, &[], "timestamp");
    sql.push_str(&format!(
        "SELECT {}\nFROM TMP\nJOIN {table}\n  ON {key} = TMP.__KUMO_ID__",
        projections.join(", ")
    ));
    if !filters.is_empty() {
        sql.push_str(&format!("\nWHERE {}", filters.join(" AND ")));
    }
    sql.push_str(&format!("\nORDER BY TMP.__KUMO_BATCH__, {order_by}"));
    Ok((sql, Some(vec![Param::Text(dumps(ids)?)])))
}

/// For each of `ids`, a chunk of them, the first row of `table` holding it
/// in `key` in `order_by` order, with how many do.
pub fn rows_by_key(
    table: &str,
    projections: &[String],
    key: &str,
    postgres_type: &str,
    ids: &Py,
    order_by: &str,
) -> Result<Statement, Error> {
    let mut sql = json_cte(postgres_type, &[], "timestamp");
    sql.push_str(&format!(
        "SELECT * FROM (\n  SELECT TMP.__KUMO_BATCH__, {},\n         ROW_NUMBER() OVER (PARTITION BY TMP.__KUMO_BATCH__\n           ORDER BY {order_by}) AS __KUMO_ROW__,\n         COUNT(*) OVER (PARTITION BY TMP.__KUMO_BATCH__) AS __KUMO_MATCHES__\n  FROM TMP\n  JOIN {table}\n    ON {key} = TMP.__KUMO_ID__\n) AS sampled\nWHERE __KUMO_ROW__ = 1\nORDER BY __KUMO_BATCH__",
        projections.join(", ")
    ));
    Ok((sql, Some(vec![Param::Text(dumps(ids)?)])))
}

/// For each of `rows`, the first `num_neighbors` rows of `table` holding it
/// in `key` and no later than its end, in `order_by` order.
#[allow(clippy::too_many_arguments)]
pub fn neighbors(
    table: &str,
    projections: &[String],
    key: &str,
    postgres_type: &str,
    rows: &Py,
    fields: &[String],
    time: Option<&str>,
    time_type: &str,
    order_by: &str,
    num_neighbors: i64,
) -> Result<Statement, Error> {
    let mut sql = json_cte(postgres_type, fields, time_type);
    sql.push_str(&format!(
        "SELECT TMP.__KUMO_BATCH__, sampled.*\nFROM TMP\nJOIN LATERAL (\n  SELECT {},\n         ROW_NUMBER() OVER (ORDER BY {order_by}) AS __KUMO_ROW__\n  FROM {table}\n  WHERE {key} = TMP.__KUMO_ID__\n",
        projections.join(", ")
    ));
    if !fields.is_empty() {
        sql.push_str(&format!("    AND {} <= TMP.__KUMO_END_TIME__\n", opt(time)));
    }
    sql.push_str(&format!(
        "  ORDER BY {order_by}\n  LIMIT {num_neighbors}\n) AS sampled ON TRUE\nORDER BY __KUMO_BATCH__, __KUMO_ROW__"
    ));
    Ok((sql, Some(vec![Param::Text(dumps(rows)?)])))
}

/// For each of `rows`, the rows of `table` holding it in `key` in its
/// window, in `order_by` order.
#[allow(clippy::too_many_arguments)]
pub fn window(
    table: &str,
    projections: &[String],
    key: &str,
    postgres_type: &str,
    rows: &Py,
    fields: &[String],
    time: &str,
    time_type: &str,
    order_by: &str,
) -> Result<Statement, Error> {
    let mut sql = json_cte(postgres_type, fields, time_type);
    sql.push_str(&format!(
        "SELECT TMP.__KUMO_BATCH__, {}\nFROM TMP\nJOIN {table}\n  ON {key} = TMP.__KUMO_ID__\n AND {time} <= TMP.__KUMO_END_TIME__\n",
        projections.join(", ")
    ));
    if fields.iter().any(|f| f == "s") {
        sql.push_str(&format!(" AND {time} > TMP.__KUMO_START_TIME__\n"));
    }
    sql.push_str(&format!("ORDER BY TMP.__KUMO_BATCH__, {order_by}"));
    Ok((sql, Some(vec![Param::Text(dumps(rows)?)])))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seeds_as_python() {
        // postgres_seed(0) and postgres_seed(7) in today's Python.
        assert_eq!(postgres_seed(&Int::from(0)), 0.032_577_230_252_726_075);
        assert_eq!(postgres_seed(&Int::from(7)), 0.807_131_742_502_808_2);
    }
}
