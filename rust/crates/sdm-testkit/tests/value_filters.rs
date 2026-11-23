// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Value filters as the data layer writes them, run on real DuckDB and
//! SQLite: each returns exactly the rows SQL's own rules keep, and a text
//! value holding a quote or a backslash reads as written.

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use sdm_contract::Scalar;
use sdm_contract::data::{
    CompareOp, DataRequest, Dialect, FetchPlan, Filter, ProviderChoice, Relation,
};
use sdm_testkit::duck;

const ROWS: &str =
    "CREATE TABLE users (id BIGINT, age BIGINT, name VARCHAR, score DOUBLE, active BOOLEAN);
INSERT INTO users VALUES
  (1, 80, 'ann', 1.5, TRUE),
  (2, 30, 'O''Brien\\x', 2.5, FALSE),
  (3, NULL, 'cy', NULL, TRUE),
  (4, 77, NULL, 0.5, NULL),
  (5, 76, 'ed', 3.5, TRUE);";

fn provider(dialect: &str) -> ProviderChoice {
    ProviderChoice::Sql {
        timestamp_encoding: None,
        dialect: Dialect::new(dialect),
        namespace: match dialect {
            "postgres" => vec![Some("public".into())],
            "snowflake" => vec![Some("db".into()), Some("public".into())],
            _ => vec![],
        },
        view: None,
    }
}

fn scan(filters: Vec<Filter>) -> DataRequest {
    DataRequest::Scan {
        table: Relation {
            columns: vec![],
            filter: None,
            found: false,
            source: vec!["users".into()],
        },
        columns: vec!["id".into()],
        filters,
        limit: None,
    }
}

fn sql(dialect: &str, request: &DataRequest) -> String {
    let root = sdm_host::default_root();
    let FetchPlan::Sql { statements, .. } =
        root.data_layer().plan(request, &provider(dialect)).unwrap()
    else {
        panic!("{dialect}: expected SQL");
    };
    assert_eq!(statements.len(), 1, "{dialect}: one statement");
    assert!(
        statements[0].params.is_empty(),
        "{dialect}: values are literals"
    );
    statements[0].text.clone()
}

fn duckdb_ids(text: &str) -> Vec<i64> {
    let connection = duck::open().unwrap();
    connection.execute_batch(ROWS).unwrap();
    let fetched = duck::run(&connection, text).unwrap();
    let mut ids: Vec<i64> = fetched
        .batches
        .iter()
        .flat_map(|b| {
            b.column(0)
                .as_primitive::<Int64Type>()
                .iter()
                .flatten()
                .collect::<Vec<_>>()
        })
        .collect();
    ids.sort_unstable();
    ids
}

fn sqlite_ids(text: &str) -> Vec<i64> {
    let connection = rusqlite::Connection::open_in_memory().unwrap();
    // SQLite reads a backslash as itself, as DuckDB does.
    connection
        .execute_batch(&ROWS.replace("\\\\", "\\"))
        .unwrap();
    let mut statement = connection.prepare(text).unwrap();
    let mut ids: Vec<i64> = statement
        .query_map([], |row| row.get(0))
        .unwrap()
        .map(Result::unwrap)
        .collect();
    ids.sort_unstable();
    ids
}

fn compare(column: &str, op: CompareOp, value: Scalar) -> Filter {
    Filter::Compare {
        column: column.into(),
        op,
        value,
    }
}

fn cases() -> Vec<(Vec<Filter>, Vec<i64>)> {
    vec![
        (
            vec![compare("age", CompareOp::Gt, Scalar::Int64(76))],
            vec![1, 4],
        ),
        (
            vec![compare("age", CompareOp::Ge, Scalar::Float64(76.5))],
            vec![1, 4],
        ),
        (
            vec![compare("age", CompareOp::Ne, Scalar::Int64(30))],
            vec![1, 4, 5],
        ),
        (
            vec![compare(
                "name",
                CompareOp::Eq,
                Scalar::Utf8("O'Brien\\x".into()),
            )],
            vec![2],
        ),
        (
            vec![compare("active", CompareOp::Eq, Scalar::Bool(true))],
            vec![1, 3, 5],
        ),
        (
            vec![Filter::In {
                column: "name".into(),
                values: vec![Scalar::Utf8("ann".into()), Scalar::Utf8("ed".into())],
            }],
            vec![1, 5],
        ),
        (
            vec![Filter::In {
                column: "name".into(),
                values: vec![],
            }],
            vec![],
        ),
        (
            vec![Filter::AnyOf {
                filters: vec![
                    compare("score", CompareOp::Lt, Scalar::Float64(1.0)),
                    Filter::IsNull {
                        column: "score".into(),
                    },
                ],
            }],
            vec![3, 4],
        ),
        (
            vec![
                Filter::NotNull {
                    column: "name".into(),
                },
                Filter::AnyOf {
                    filters: vec![
                        Filter::AllOf {
                            filters: vec![
                                compare("age", CompareOp::Le, Scalar::Int64(76)),
                                compare("score", CompareOp::Gt, Scalar::Float64(2.0)),
                            ],
                        },
                        compare("id", CompareOp::Eq, Scalar::Int64(1)),
                    ],
                },
            ],
            vec![1, 2, 5],
        ),
    ]
}

#[test]
fn value_filters_return_the_rows_sql_keeps_on_duckdb() {
    for (filters, expected) in cases() {
        let text = sql("duckdb", &scan(filters.clone()));
        assert_eq!(duckdb_ids(&text), expected, "{filters:?}\n{text}");
    }
}

#[test]
fn value_filters_return_the_rows_sql_keeps_on_sqlite() {
    for (filters, expected) in cases() {
        let text = sql("sqlite", &scan(filters.clone()));
        assert_eq!(sqlite_ids(&text), expected, "{filters:?}\n{text}");
    }
}

#[test]
fn a_sample_draws_only_from_the_rows_its_value_filters_keep() {
    let request = DataRequest::SampleEntities {
        table: Relation {
            columns: vec![],
            filter: None,
            found: false,
            source: vec!["users".into()],
        },
        key: vec!["id".into()],
        n: 100,
        seed: Scalar::UInt64(7),
        columns: vec!["id".into(), "age".into()],
        filters: vec![
            Filter::NotNull {
                column: "id".into(),
            },
            compare("age", CompareOp::Gt, Scalar::Int64(76)),
        ],
        key_dtype: None,
        rows: None,
    };
    let text = sql("duckdb", &request);
    assert_eq!(duckdb_ids(&text), [1, 4], "{text}");
}

#[test]
fn warehouse_text_escapes_backslashes_and_quotes() {
    for dialect in ["databricks", "snowflake"] {
        let text = sql(
            dialect,
            &scan(vec![compare(
                "name",
                CompareOp::Eq,
                Scalar::Utf8("O'Brien\\x".into()),
            )]),
        );
        assert!(text.contains(r"= 'O\'Brien\\x'"), "{dialect}: {text}");
    }
    let text = sql(
        "postgres",
        &scan(vec![compare(
            "name",
            CompareOp::Eq,
            Scalar::Utf8("O'Brien\\x".into()),
        )]),
    );
    assert!(text.contains(r"= E'O''Brien\\x'"), "postgres: {text}");
}

#[test]
fn a_value_no_dialect_writes_widens_the_read_instead_of_failing() {
    let nan = compare("score", CompareOp::Gt, Scalar::Float64(f64::NAN));
    let nul = compare("name", CompareOp::Eq, Scalar::Utf8("a\0b".into()));
    // Alone, each is left out: every row.
    assert_eq!(
        duckdb_ids(&sql("duckdb", &scan(vec![nan.clone()]))),
        [1, 2, 3, 4, 5]
    );
    // In an `AnyOf`, the whole disjunction is left out; in an `AllOf`, the
    // parts written stay.
    let any = Filter::AnyOf {
        filters: vec![nul, compare("id", CompareOp::Eq, Scalar::Int64(1))],
    };
    assert_eq!(
        duckdb_ids(&sql("duckdb", &scan(vec![any]))),
        [1, 2, 3, 4, 5]
    );
    let all = Filter::AllOf {
        filters: vec![nan, compare("id", CompareOp::Eq, Scalar::Int64(1))],
    };
    assert_eq!(duckdb_ids(&sql("duckdb", &scan(vec![all]))), [1]);
}

const ORDERS: &str =
    "CREATE TABLE orders (oid BIGINT, uid BIGINT, placed TIMESTAMP, amount DOUBLE);
INSERT INTO orders VALUES
  (1, 1, TIMESTAMP '2025-01-10 00:00:00', 5.0),
  (2, 1, TIMESTAMP '2025-01-20 00:00:00', NULL),
  (3, 2, TIMESTAMP '2025-01-15 00:00:00', 7.0),
  (4, 3, TIMESTAMP '2024-06-01 00:00:00', 1.0),
  (5, 5, TIMESTAMP '2025-01-31 23:59:59', NULL),
  (6, NULL, TIMESTAMP '2025-01-15 00:00:00', 1.0);";

/// Microseconds since the epoch of a UTC wall clock.
fn micros(day: &str) -> Scalar {
    let days = match day {
        "2025-01-01" => 20_089,
        "2025-02-01" => 20_120,
        _ => unreachable!(),
    };
    Scalar::Timestamp {
        unit: sdm_contract::Unit::Us,
        zone: None,
        value: days * 86_400_000_000,
    }
}

fn has_rows(at_least: u64, filters: Vec<Filter>) -> Filter {
    Filter::HasRows {
        column: "id".into(),
        table: Relation {
            columns: vec![],
            filter: None,
            found: false,
            source: vec!["orders".into()],
        },
        key: "uid".into(),
        filters: [Filter::TimeRange {
            column: "placed".into(),
            lower: Some(sdm_contract::data::Bound {
                value: micros("2025-01-01"),
                inclusive: true,
            }),
            upper: Some(sdm_contract::data::Bound {
                value: micros("2025-02-01"),
                inclusive: false,
            }),
        }]
        .into_iter()
        .chain(filters)
        .collect(),
        at_least,
    }
}

#[test]
fn has_rows_keeps_the_rows_referred_to_within_the_window_on_duckdb() {
    let ids = |filter: Filter| {
        let text = sql("duckdb", &scan(vec![filter]));
        let connection = duck::open().unwrap();
        connection.execute_batch(ROWS).unwrap();
        connection.execute_batch(ORDERS).unwrap();
        let mut ids: Vec<i64> = duck::run(&connection, &text)
            .unwrap()
            .batches
            .iter()
            .flat_map(|b| {
                b.column(0)
                    .as_primitive::<Int64Type>()
                    .iter()
                    .flatten()
                    .collect::<Vec<_>>()
            })
            .collect();
        ids.sort_unstable();
        (ids, text)
    };
    // In January 2025: user 1 twice, users 2 and 5 once; user 3's order is
    // earlier, and an order of no user refers to none.
    let (found, text) = ids(has_rows(1, vec![]));
    assert_eq!(found, [1, 2, 5], "{text}");
    let (found, text) = ids(has_rows(2, vec![]));
    assert_eq!(found, [1], "{text}");
    // Only the orders with an amount: user 5's has none.
    let (found, text) = ids(has_rows(
        1,
        vec![Filter::NotNull {
            column: "amount".into(),
        }],
    ));
    assert_eq!(found, [1, 2], "{text}");
}

#[test]
fn the_timestamp_literal_is_the_wall_clock_and_sqlite_writes_none() {
    let text = sql("duckdb", &scan(vec![has_rows(1, vec![])]));
    assert!(
        text.contains("CAST(\"placed\" AS TIMESTAMP) >= TIMESTAMP '2025-01-01 00:00:00.000000'"),
        "{text}"
    );
    // SQLite writes no time literal: the window is left out, and the
    // count is over every time, never fewer rows.
    let text = sql("sqlite", &scan(vec![has_rows(1, vec![])]));
    assert!(
        text.contains("\"orders\"") && !text.contains("placed"),
        "{text}"
    );
}

/// A source read in a namespace reads a `has_rows` table there too: the
/// subquery names the table as the outer read does, never as the
/// session's current schema would resolve it.
#[test]
fn has_rows_reads_its_table_in_the_source_s_namespace() {
    for dialect in ["postgres", "snowflake"] {
        let text = sql(dialect, &scan(vec![has_rows(1, vec![])]));
        let users = if dialect == "postgres" {
            r#""public"."users""#
        } else {
            r#""db"."public"."users""#
        };
        let orders = users.replace("users", "orders");
        assert!(text.contains(&format!("FROM {users}")), "{dialect}: {text}");
        assert!(
            text.contains(&format!("FROM {orders}")),
            "{dialect}: the subquery reads its table unqualified: {text}"
        );
    }
}
