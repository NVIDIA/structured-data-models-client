// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! DuckDB's catalog renderings (T1.11b): each is the statement the
//! reference runs for the same step, as the driver capture recorded it on
//! `base.regression` (the T1.11a row of the design maps each statement to
//! its request), with the schema the contract fixes for its answer, or,
//! for a `describe`, the reference statement's own columns, which
//! `sdm-data` reads into the contract's catalog answer (decision E54).

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
        dialect: Dialect::new("duckdb"),
    }
}

/// The columns DuckDB's catalog statement for `part` returns, in order:
/// the reference statement's own.
fn catalog_columns(part: DescribePart) -> &'static [&'static str] {
    match part {
        DescribePart::Tables => &["table_name"],
        DescribePart::Columns => &["cid", "name", "type", "notnull", "dflt_value", "pk"],
        DescribePart::Keys => &["constraint_type", "constraint_column_names"],
        DescribePart::ForeignKeys => &[
            "constraint_index",
            "referenced_table",
            "constraint_column_names",
            "referenced_column_names",
        ],
        DescribePart::Indexes => &["table_name", "expressions"],
        // DuckDB lists an index's columns with the index; never planned.
        DescribePart::IndexColumns => &[],
    }
}

/// The one statement `request` renders to, its expected schema checked.
fn rendered(request: &DataRequest) -> String {
    match Data::new().plan(request, &duckdb()).unwrap() {
        FetchPlan::Sql { statements, expect } => {
            match request {
                DataRequest::Describe { part, .. } => {
                    let ExpectedSchema::Schema { fields } = &expect else {
                        panic!("a catalog answer has a fixed shape");
                    };
                    let names: Vec<&str> = fields.iter().map(|f| f.name.as_str()).collect();
                    assert_eq!(names, catalog_columns(*part));
                }
                _ => assert_eq!(expect, request.expected()),
            }
            assert_eq!(statements.len(), 1);
            assert!(statements[0].params.is_empty());
            assert_eq!(statements[0].dialect, Dialect::new("duckdb"));
            statements[0].text.clone()
        }
        other => panic!("not SQL: {other:?}"),
    }
}

fn describe(tables: &[&str], part: DescribePart) -> DataRequest {
    DataRequest::Describe {
        tables: tables.iter().map(|t| (*t).into()).collect(),
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
                    label: None,
                },
                TableKey {
                    table: "users".into(),
                    columns: vec!["user_id".into()],
                    label: None,
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
        rendered(&describe(&["it's"], DescribePart::Columns)),
        "PRAGMA table_info(\"it's\")"
    );
    // A name the pragma reads as another (P1B-5): read by itself, as text.
    for (name, quoted) in [("a\"b", "'a\"b'"), ("a.b", "'a.b'"), ("o'd.d", "'o''d.d'")] {
        let text = rendered(&describe(&[name], DescribePart::Columns));
        assert!(text.starts_with("SELECT CAST(c.column_index - 1"), "{text}");
        assert_eq!(
            text.matches(&format!("table_name = {quoted}\n")).count(),
            2,
            "{text}"
        );
    }
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
            label: None,
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
fn the_column_ranges_are_the_reference_s_whole_table_query() {
    // T1.32e: the statement the reference ran at b413e9ad for the time
    // column's choice over a DuckDB table of 3,000 rows whose preview may
    // not hold every row (`Table._full_range`), its candidates a timestamp
    // named with a double quote, a date, a time of day and text read as
    // times; a load stamp is never asked.
    let text = rendered(&DataRequest::ColumnRanges {
        table: "evs".into(),
        columns: ["happ\"ened", "due_on", "at_time", "created_at"]
            .map(String::from)
            .into(),
    });
    assert_eq!(
        text,
        "SELECT COUNT(*), MIN(\"happ\"\"ened\"), MAX(\"happ\"\"ened\"), COUNT(\"happ\"\"ened\"), MIN(\"due_on\"), MAX(\"due_on\"), COUNT(\"due_on\"), MIN(\"at_time\"), MAX(\"at_time\"), COUNT(\"at_time\"), MIN(\"created_at\"), MAX(\"created_at\"), COUNT(\"created_at\")\nFROM \"evs\""
    );
    let err = Data::new()
        .plan(
            &DataRequest::ColumnRanges {
                table: "evs".into(),
                columns: vec![],
            },
            &duckdb(),
        )
        .unwrap_err();
    assert_eq!(err.code, ErrorCode::InvalidRequest);
}

#[test]
fn a_repeated_key_is_read_as_the_reference_reads_one() {
    // T1.32e: the reference's `_repeated_key_sql` at b413e9ad, one value a
    // key several rows hold, to show it.
    let text = rendered(&DataRequest::RepeatedKey {
        table: "payments".into(),
        key: vec!["payment_id".into()],
    });
    assert_eq!(
        text,
        "SELECT \"payment_id\"\nFROM \"payments\"\nWHERE \"payment_id\" IS NOT NULL\nGROUP BY \"payment_id\"\nHAVING COUNT(*) > 1\nLIMIT 1"
    );
    let err = Data::new()
        .plan(
            &DataRequest::RepeatedKey {
                table: "payments".into(),
                key: vec![],
            },
            &duckdb(),
        )
        .unwrap_err();
    assert_eq!(err.code, ErrorCode::InvalidRequest);
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
    // The indexes are read whole, as the reference reads them.
    let err = Data::new()
        .plan(&describe(&["a"], DescribePart::Indexes), &duckdb())
        .unwrap_err();
    assert_eq!(err.code, ErrorCode::UnsupportedRequest);
    // The table list with a name asked: the relations, tables or views,
    // DuckDB would read under it, matched as DuckDB folds a name (T1.23
    // fix 2; sdm-testkit's duckdb_provider.rs runs it).
    let FetchPlan::Sql { statements, .. } = Data::new()
        .plan(&describe(&["O'Brien"], DescribePart::Tables), &duckdb())
        .unwrap()
    else {
        panic!("DuckDB answers with SQL");
    };
    let text = &statements[0].text;
    assert!(
        text.contains("duckdb_tables()") && text.contains("duckdb_views()"),
        "{text}"
    );
    assert!(text.contains("'O''Brien'"), "{text}");
    let err = Data::new().plan(&preview("t", &[]), &duckdb()).unwrap_err();
    assert_eq!(err.code, ErrorCode::InvalidRequest);
}

/// A scan and a count of rows render on DuckDB from T1.23, so every
/// request kind does: what a not-null filter keeps, and a scan's limit;
/// a filter of another kind is not taken, by name.
#[test]
fn a_scan_and_a_count_of_rows_render() {
    let text = |request: &DataRequest| match Data::new().plan(request, &duckdb()).unwrap() {
        FetchPlan::Sql { statements, .. } => statements[0].text.clone(),
        other => panic!("{other:?}"),
    };
    let not_null = || vec![sdm_contract::data::Filter::NotNull { column: "k".into() }];
    let scan = |filters, limit| DataRequest::Scan {
        table: "t".into(),
        columns: vec!["c".into(), "k".into()],
        filters,
        limit,
    };
    assert_eq!(text(&scan(vec![], None)), r#"SELECT "c", "k" FROM "t""#);
    assert_eq!(
        text(&scan(not_null(), Some(5))),
        r#"SELECT "c", "k" FROM "t" WHERE "k" IS NOT NULL LIMIT 5"#
    );
    assert_eq!(
        text(&DataRequest::CountRows {
            table: "t".into(),
            filters: not_null(),
        }),
        r#"SELECT COUNT(*) AS "rows" FROM "t" WHERE "k" IS NOT NULL"#
    );
    let ranged = vec![sdm_contract::data::Filter::TimeRange {
        column: "ts".into(),
        lower: None,
        upper: None,
    }];
    let err = Data::new()
        .plan(&scan(ranged, None), &duckdb())
        .unwrap_err();
    assert_eq!(err.code, ErrorCode::UnsupportedRequest);
}

#[test]
fn a_host_provider_is_held_to_the_contract_s_shapes() {
    let host = ProviderChoice::Host {
        capabilities: Data::new().capabilities(&duckdb()).unwrap(),
    };
    for request in [
        describe(&["t"], DescribePart::Columns),
        DataRequest::KeyCounts {
            keys: vec![TableKey {
                table: "t".into(),
                columns: vec!["k".into()],
                label: None,
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

/// A table whose name, in any case, is the one the seeds are bound by is
/// read, not shadowed: its statement binds the seeds under another name.
#[test]
fn a_table_named_as_the_seeds_are_is_still_the_table_read() {
    use sdm_contract::data::{AttachmentId, KeySet, SqlParam};
    for table in ["__sdm_seeds__", "__SDM_Seeds__", "orders"] {
        let request = DataRequest::Lookup {
            table: table.into(),
            key: vec!["id".into()],
            columns: vec!["id".into()],
            seeds: KeySet {
                attachment: AttachmentId(1),
                columns: ["key", "batch"]
                    .map(|name| sdm_contract::Field {
                        name: name.into(),
                        column_type: sdm_contract::ColumnType::Int64,
                        nullable: false,
                        metadata: Default::default(),
                    })
                    .into(),
            },
        };
        let FetchPlan::Sql { statements, .. } = Data::new().plan(&request, &duckdb()).unwrap()
        else {
            panic!("not SQL");
        };
        let [SqlParam::KeySet { relation, .. }] = statements[0].params.as_slice() else {
            panic!("not one key set");
        };
        assert_ne!(relation.to_lowercase(), table.to_lowercase());
        assert!(
            statements[0]
                .text
                .contains(&format!("FROM \"{relation}\" s\nJOIN \"{table}\" t"))
        );
    }
}
