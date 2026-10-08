// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The SQL the engine runs against Snowflake (`backend/snow/dialect.py`).
//!
//! Parameters are bound to `?` markers by Snowflake itself. A key list
//! travels as one parameter holding a JSON array, which
//! `FLATTEN(PARSE_JSON(...))` turns into rows.

use crate::fragments::{ordered_sample_sql, quote_ident, repeated_key_select};
use crate::int::Int;
use crate::py::{Dtype, Py};
use crate::pyjson::Dumps;
use crate::unicode::strip;
use crate::{Error, KeyCountPart, Param, Statement, opt};

/// The database and schema a bare name resolves in.
pub fn current_namespace() -> Statement {
    ("SELECT CURRENT_DATABASE(), CURRENT_SCHEMA()".into(), None)
}

/// Every table of `schema` in `database`, by name.
pub fn list_tables(database: &str, schema: &str) -> Statement {
    let pad = " ".repeat(20);
    (
        format!(
            "\n{pad}SELECT TABLE_NAME\n{pad}FROM {}.INFORMATION_SCHEMA.TABLES\n{pad}WHERE TABLE_SCHEMA = ?\n{pad}ORDER BY TABLE_NAME\n{pad}",
            quote_ident(database, "\"")
        ),
        Some(vec![Param::Text(schema.into())]),
    )
}

/// The columns of `table`, a quoted name.
pub fn describe_table(table: &str) -> Statement {
    (format!("DESCRIBE TABLE {table}"), None)
}

/// The foreign keys of `table`, a quoted name.
pub fn imported_keys(table: &str) -> Statement {
    (format!("SHOW IMPORTED KEYS IN TABLE {table}"), None)
}

/// Every table of `schema` in `database`, with whether it is a hybrid table.
pub fn tables_in_schema(database: &str, schema: &str) -> Statement {
    (
        format!(
            "SHOW TABLES IN SCHEMA {}.{}",
            quote_ident(database, "\""),
            quote_ident(schema, "\"")
        ),
        None,
    )
}

/// The tables of `schema` in `database` named like `table`, with their row
/// counts.
pub fn tables_like(database: &str, schema: &str, table: &str) -> Statement {
    (
        format!(
            "SHOW TABLES LIKE ? IN SCHEMA {}.{}",
            quote_ident(database, "\""),
            quote_ident(schema, "\"")
        ),
        Some(vec![Param::Text(table.into())]),
    )
}

/// The YAML definition of the semantic view `name`.
pub fn semantic_view(name: &str) -> Statement {
    (
        "SELECT SYSTEM$READ_YAML_FROM_SEMANTIC_VIEW(?)".into(),
        Some(vec![Param::Text(name.into())]),
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

/// The clause after a table's name that keeps `percent` of its
/// micro-partitions, the same ones for the same `seed`.
pub fn block_sample(percent: &str, seed: &Int) -> String {
    format!("SAMPLE SYSTEM ({percent}) SEED ({seed})")
}

/// Whether a row of `table` holds one of `values` in `identity`.
pub fn holds_any(table: &str, identity: &str, values: &[String]) -> Statement {
    (
        format!(
            "SELECT 1 FROM {table}\nWHERE ({identity}) IN (\n  SELECT f.value::VARCHAR\n  FROM TABLE(FLATTEN(INPUT => PARSE_JSON(?))) f\n)\nLIMIT 1"
        ),
        Some(vec![Param::Text(crate::text::json_text_list(
            values,
            Dumps::PLAIN,
        ))]),
    )
}

/// The rows, distinct keys and rows missing a key of each table, in one
/// `UNION ALL`.
pub fn key_counts(parts: &[KeyCountPart]) -> Statement {
    crate::databricks::key_counts(parts)
}

/// One value of the key `columns` more than one row holds.
pub fn repeated_key(table: &str, columns: &[String]) -> Statement {
    (repeated_key_select(table, columns), None)
}

/// The earliest and latest time of each `(table, time column)`.
pub fn time_ranges(tables: &[(String, String)]) -> Statement {
    crate::sqlite::time_ranges(tables)
}

/// A hash of the seed and a row's `key`, the same in every session: each
/// key rendered in a form no session parameter changes.
pub fn seeded_key_hash(
    key: &str,
    dtype: Option<Dtype>,
    is_date: bool,
    random_seed: &Int,
) -> String {
    let text = match dtype {
        Some(Dtype::String) => key.to_string(),
        Some(Dtype::Date) if is_date => format!("TO_VARCHAR({key}, 'YYYY-MM-DD')"),
        Some(Dtype::Date) => format!("TO_VARCHAR(DATE_PART(EPOCH_NANOSECOND, {key}))"),
        Some(Dtype::Time) => format!("TO_VARCHAR({key}, 'HH24:MI:SS.FF9')"),
        Some(Dtype::Binary) => format!("TO_VARCHAR({key}, 'HEX')"),
        _ => format!("TO_VARCHAR({key})"),
    };
    format!("MD5_NUMBER_LOWER64('{random_seed}:' || {text})")
}

/// Rows of an entity table: `num_rows` drawn by `SAMPLE ROW`, the `num_rows`
/// lowest in the `seeded` hash, through `prefilter` where given, or the rows
/// holding `ids` in `key`, bound as parameters, in `order_by` order.
#[allow(clippy::too_many_arguments)]
pub fn entity_rows(
    table: &str,
    projections: &[String],
    filters: &[String],
    order_by: &str,
    num_rows: i64,
    seeded: Option<&str>,
    key: Option<&str>,
    ids: Option<&[Param]>,
    prefilter: Option<&str>,
) -> Statement {
    let mut conditions = filters.to_vec();
    let mut params = None;
    if let Some(ids) = ids {
        let markers = vec!["?"; ids.len()].join(", ");
        conditions.push(format!(" {} IN ({markers})", opt(key)));
        params = Some(ids.to_vec());
    }
    let mut sql = if ids.is_some() || seeded.is_some() {
        let mut sql = format!("SELECT {}\nFROM {table}", projections.join(", "));
        if let (Some(_), Some(prefilter)) = (seeded, prefilter) {
            sql.push_str(&format!("\n{prefilter}"));
        }
        sql
    } else {
        format!(
            "SELECT {}\nFROM {table}\nSAMPLE ROW ({num_rows} ROWS)",
            projections.join(", ")
        )
    };
    if !conditions.is_empty() {
        sql.push_str(&format!("\nWHERE{}", conditions.join(" AND")));
    }
    if ids.is_some() {
        sql.push_str(&format!("\nORDER BY {order_by}"));
    } else if let Some(seeded) = seeded {
        sql.push_str(&format!(
            "\nORDER BY {seeded}, {order_by}\nLIMIT {num_rows}"
        ));
    }
    (sql, params)
}

/// How a bound on a zoned time column is written.
pub const ZONED_BOUND_FORMAT: &str = "%Y-%m-%dT%H:%M:%S+00:00";
const ZONED_BOUND_SNOWFLAKE_FORMAT: &str = "YYYY-MM-DD\"T\"HH24:MI:SSTZH:TZM";
/// How a bound on a naive or `DATE` time column is written.
pub const NAIVE_BOUND_FORMAT: &str = "%Y-%m-%d %H:%M:%S";

/// `LTZ` or `TZ` for a `TIMESTAMP_LTZ` or `TIMESTAMP_TZ` type as `DESCRIBE
/// TABLE` spells it, and `None` for any other.
pub fn zone_of(snowflake_type: Option<&str>) -> Option<&'static str> {
    let spelled = strip(snowflake_type.unwrap_or("")).to_uppercase();
    if spelled.starts_with("TIMESTAMP_LTZ") {
        Some("LTZ")
    } else if spelled.starts_with("TIMESTAMP_TZ") {
        Some("TZ")
    } else {
        None
    }
}

/// A window bound, `value`, as the time column compares with it.
pub fn bound(value: &str, zone: Option<&str>) -> String {
    match zone {
        None => format!("{value}::TIMESTAMP_NTZ"),
        Some(zone) => format!("TO_TIMESTAMP_{zone}({value}, '{ZONED_BOUND_SNOWFLAKE_FORMAT}')"),
    }
}

/// The placeholder of a bound outside the key list.
pub fn outer_bound(zone: Option<&str>) -> String {
    match zone {
        None => "?".into(),
        Some(zone) => format!("TO_TIMESTAMP_{zone}(?, '{ZONED_BOUND_SNOWFLAKE_FORMAT}')"),
    }
}

/// The `index`-th item of a key list entry, a time bound.
pub fn key_list_time(index: i64, zone: Option<&str>) -> String {
    match zone {
        None => bound(&format!("f.value[{index}]"), zone),
        Some(_) => bound(&format!("f.value[{index}]::VARCHAR"), zone),
    }
}

/// The type a key of `dtype` is cast to from the JSON list.
pub fn cast(dtype: Dtype) -> &'static str {
    if dtype.is_int() {
        "NUMBER"
    } else if dtype.is_float() {
        "FLOAT"
    } else {
        "VARCHAR"
    }
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
        "WITH TMP as (\n  SELECT\n    f.index as __KUMO_BATCH__,\n    f.value::{} as __KUMO_ID__\n  FROM TABLE(FLATTEN(INPUT => PARSE_JSON(?))) f\n)\nSELECT TMP.__KUMO_BATCH__ as __KUMO_BATCH__, {}\nFROM TMP\nJOIN {table}\n  ON {key} = TMP.__KUMO_ID__\nORDER BY {}",
        cast(dtype),
        projections.join(", "),
        order.join(", ")
    );
    Ok((sql, Some(vec![Param::Text(Dumps::PLAIN.dumps(ids)?)])))
}

fn bound_params(rows: &Py, bounds: &[String]) -> Result<Vec<Param>, Error> {
    let mut params = vec![Param::Text(Dumps::PLAIN.dumps(rows)?)];
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
    zone: Option<&str>,
) -> Result<Statement, Error> {
    let mut sql = format!(
        "WITH TMP as (\n  SELECT\n    f.index as __KUMO_BATCH__,\n    f.value[0]::{} as __KUMO_ID__",
        cast(dtype)
    );
    if windowed {
        sql.push_str(&format!(
            ",\n    {} as __KUMO_END_TIME__,\n    {} as __KUMO_START_TIME__",
            key_list_time(1, zone),
            key_list_time(2, zone)
        ));
    }
    sql.push_str(&format!(
        "\n  FROM TABLE(FLATTEN(INPUT => PARSE_JSON(?))) f\n)\nSELECT TMP.__KUMO_BATCH__ as __KUMO_BATCH__, {}\nFROM TMP\nJOIN {table}\n  ON {key} = TMP.__KUMO_ID__\n",
        projections.join(", ")
    ));
    if windowed {
        let time = opt(time);
        let outer = outer_bound(zone);
        sql.push_str(&format!(
            " AND {time} <= TMP.__KUMO_END_TIME__\n AND {time} > TMP.__KUMO_START_TIME__\nWHERE {time} <= {outer}\n  AND {time} > {outer}\n"
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
    zone: Option<&str>,
) -> Result<Statement, Error> {
    let mut sql = format!(
        "WITH TMP as (\n  SELECT\n    f.index as __KUMO_BATCH__,\n    f.value[0]::{} as __KUMO_ID__,\n    {} as __KUMO_END_TIME__",
        cast(dtype),
        key_list_time(1, zone)
    );
    if with_start {
        sql.push_str(&format!(
            ",\n    {} as __KUMO_START_TIME__",
            key_list_time(2, zone)
        ));
    }
    sql.push_str(&format!(
        "\n  FROM TABLE(FLATTEN(INPUT => PARSE_JSON(?))) f\n)\nSELECT TMP.__KUMO_BATCH__ as __KUMO_BATCH__, {}\nFROM TMP\nJOIN {table}\n  ON {key} = TMP.__KUMO_ID__\n AND {time} <= TMP.__KUMO_END_TIME__\n",
        projections.join(", ")
    ));
    if with_start {
        sql.push_str(&format!("AND {time} > TMP.__KUMO_START_TIME__\n"));
    }
    sql.push_str(&format!("WHERE {time} <= {}", outer_bound(zone)));
    if with_start {
        sql.push_str(&format!("\nAND {time} > {}", outer_bound(zone)));
    }
    sql.push_str(&format!("\nORDER BY TMP.__KUMO_BATCH__, {order_by}"));
    Ok((sql, Some(bound_params(rows, bounds)?)))
}
