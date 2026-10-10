// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! DuckDB's catalog renderings (T1.11b): each is the statement the
//! reference runs for the same step, as the driver capture recorded it on
//! `base.regression` (the T1.11a row of the design maps each statement to
//! its request), with the schema the contract fixes for its answer.

use sdm_contract::data::{
    DataLayer, DataRequest, DescribePart, Dialect, ExpectedSchema, FetchPlan, ProviderChoice,
    TableColumn, TableKey,
};
use sdm_contract::{ErrorCode, ErrorKind};
use sdm_data::Data;

/// The statements the reference ran on `base.regression`, in order, through
/// `Schema.from_source` and the first steps of `materialize`, as the
/// capture recorded them at T1.11a.
const REFERENCE: [&str; 12] = [
    "SELECT table_name FROM duckdb_tables() WHERE NOT temporary AND NOT internal AND database_name = current_database() AND schema_name = current_schema()",
    "PRAGMA table_info(\"orders\")",
    "SELECT constraint_type, constraint_column_names\nFROM duckdb_constraints()\nWHERE table_name = 'orders'\n  AND database_name = current_database() AND schema_name = current_schema()",
    "SELECT \"order_id\", \"user_id\", \"price\", \"ts\" FROM \"orders\" LIMIT 1000",
    "PRAGMA table_info(\"users\")",
    "SELECT constraint_type, constraint_column_names\nFROM duckdb_constraints()\nWHERE table_name = 'users'\n  AND database_name = current_database() AND schema_name = current_schema()",
    "SELECT \"user_id\", \"age\", \"country\" FROM \"users\" LIMIT 1000",
    "SELECT constraint_index, referenced_table,\n       constraint_column_names, referenced_column_names\nFROM duckdb_constraints()\nWHERE table_name = 'orders'\n  AND database_name = current_database() AND schema_name = current_schema()\n  AND constraint_type = 'FOREIGN KEY'",
    "SELECT constraint_index, referenced_table,\n       constraint_column_names, referenced_column_names\nFROM duckdb_constraints()\nWHERE table_name = 'users'\n  AND database_name = current_database() AND schema_name = current_schema()\n  AND constraint_type = 'FOREIGN KEY'",
    "SELECT 0 AS k, COUNT(*) AS n, COUNT(DISTINCT \"order_id\") AS d, COUNT(CASE WHEN \"order_id\" IS NULL THEN 1 END) AS z\nFROM \"orders\"\nUNION ALL\nSELECT 1 AS k, COUNT(*) AS n, COUNT(DISTINCT \"user_id\") AS d, COUNT(CASE WHEN \"user_id\" IS NULL THEN 1 END) AS z\nFROM \"users\"",
    "SELECT table_name, expressions FROM duckdb_indexes()",
    "SELECT\n  0 as table_index,\n  MIN(CAST(\"ts\" AS TIMESTAMP)) as min_date,\n  MAX(CAST(\"ts\" AS TIMESTAMP)) as max_date\nFROM \"orders\"",
];

fn duckdb() -> ProviderChoice {
    ProviderChoice::Sql {
        dialect: Dialect::Duckdb,
    }
}

/// The one statement `request` renders to, its expected schema checked.
fn rendered(request: &DataRequest) -> String {
    match Data::new().plan(request, &duckdb()).unwrap() {
        FetchPlan::Sql { statements, expect } => {
            assert_eq!(expect, request.expected());
            assert_eq!(statements.len(), 1);
            assert!(statements[0].params.is_empty());
            assert_eq!(statements[0].dialect, Dialect::Duckdb);
            statements[0].text.clone()
        }
        other => panic!("not SQL: {other:?}"),
    }
}

fn describe(tables: &[&str], part: DescribePart) -> DataRequest {
    DataRequest::Describe {
        tables: tables.iter().map(|t| (*t).to_owned()).collect(),
        part,
    }
}

fn preview(table: &str, columns: &[&str]) -> DataRequest {
    DataRequest::Preview {
        table: table.into(),
        columns: columns.iter().map(|c| (*c).to_owned()).collect(),
        limit: 1000,
    }
}

#[test]
fn each_catalog_request_renders_the_reference_statement() {
    let requests = [
        describe(&[], DescribePart::Tables),
        describe(&["orders"], DescribePart::Columns),
        describe(&["orders"], DescribePart::Keys),
        preview("orders", &["order_id", "user_id", "price", "ts"]),
        describe(&["users"], DescribePart::Columns),
        describe(&["users"], DescribePart::Keys),
        preview("users", &["user_id", "age", "country"]),
        describe(&["orders"], DescribePart::ForeignKeys),
        describe(&["users"], DescribePart::ForeignKeys),
        DataRequest::KeyCounts {
            keys: vec![
                TableKey {
                    table: "orders".into(),
                    columns: vec!["order_id".into()],
                },
                TableKey {
                    table: "users".into(),
                    columns: vec!["user_id".into()],
                },
            ],
        },
        describe(&[], DescribePart::Indexes),
        DataRequest::TimeBounds {
            columns: vec![TableColumn {
                table: "orders".into(),
                column: "ts".into(),
            }],
        },
    ];
    for (position, (request, reference)) in requests.iter().zip(REFERENCE).enumerate() {
        assert_eq!(rendered(request), reference, "statement {position}");
    }
}

#[test]
fn names_are_quoted_with_their_quotes_doubled() {
    assert_eq!(
        rendered(&describe(&["a\"b"], DescribePart::Columns)),
        "PRAGMA table_info(\"a\"\"b\")"
    );
    assert!(rendered(&describe(&["it's"], DescribePart::Keys)).contains("table_name = 'it''s'\n"));
    assert_eq!(
        rendered(&preview("t\"", &["c\"1", "c'2"])),
        "SELECT \"c\"\"1\", \"c'2\" FROM \"t\"\"\" LIMIT 1000"
    );
}

#[test]
fn a_composite_key_counts_only_rows_holding_all_of_it() {
    let text = rendered(&DataRequest::KeyCounts {
        keys: vec![TableKey {
            table: "t".into(),
            columns: vec!["a".into(), "b".into()],
        }],
    });
    assert_eq!(
        text,
        "SELECT 0 AS k, COUNT(*) AS n, COUNT(DISTINCT (\"a\", \"b\")) FILTER (WHERE \"a\" IS NOT NULL AND \"b\" IS NOT NULL) AS d, COUNT(CASE WHEN \"a\" IS NULL OR \"b\" IS NULL THEN 1 END) AS z\nFROM \"t\""
    );
}

#[test]
fn several_time_columns_are_one_union() {
    let text = rendered(&DataRequest::TimeBounds {
        columns: vec![
            TableColumn {
                table: "a".into(),
                column: "t1".into(),
            },
            TableColumn {
                table: "b".into(),
                column: "t2".into(),
            },
        ],
    });
    assert_eq!(text.matches("\nUNION ALL\n").count(), 1);
    assert!(text.contains("  1 as table_index,\n  MIN(CAST(\"t2\" AS TIMESTAMP))"));
}

#[test]
fn a_part_read_per_table_or_whole_is_refused_otherwise() {
    for part in [
        DescribePart::Columns,
        DescribePart::Keys,
        DescribePart::ForeignKeys,
    ] {
        for tables in [&[][..], &["a", "b"][..]] {
            let err = Data::new()
                .plan(&describe(tables, part), &duckdb())
                .unwrap_err();
            assert_eq!(
                (err.kind, err.code),
                (ErrorKind::Unsupported, ErrorCode::UnsupportedRequest)
            );
        }
    }
    // The table list and the indexes are read whole, as the reference
    // reads them; tables named arrive with the overrides (T1.24).
    for part in [DescribePart::Tables, DescribePart::Indexes] {
        let err = Data::new()
            .plan(&describe(&["a"], part), &duckdb())
            .unwrap_err();
        assert_eq!(err.code, ErrorCode::UnsupportedRequest);
    }
    let err = Data::new().plan(&preview("t", &[]), &duckdb()).unwrap_err();
    assert_eq!(err.code, ErrorCode::InvalidRequest);
}

#[test]
fn the_requests_without_a_rendering_are_refused_by_name() {
    let scan = DataRequest::Scan {
        table: "t".into(),
        columns: vec!["c".into()],
        filters: vec![],
        limit: None,
    };
    let err = Data::new().plan(&scan, &duckdb()).unwrap_err();
    assert_eq!(err.code, ErrorCode::UnsupportedRequest);
}

#[test]
fn a_host_provider_is_held_to_the_same_shapes() {
    let host = ProviderChoice::Host {
        capabilities: Data::new().capabilities(&duckdb()).unwrap(),
    };
    for request in [
        describe(&["t"], DescribePart::Columns),
        DataRequest::KeyCounts {
            keys: vec![TableKey {
                table: "t".into(),
                columns: vec!["k".into()],
            }],
        },
        preview("t", &["c"]),
    ] {
        assert_eq!(
            Data::new().plan(&request, &host).unwrap(),
            FetchPlan::Host {
                expect: request.expected()
            }
        );
    }
    assert!(matches!(
        describe(&["t"], DescribePart::Keys).expected(),
        ExpectedSchema::Schema { .. }
    ));
}
