// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Contract v26's table source on DuckDB (T2.1b, design choice 12): the
//! relation forms DuckDB takes, one to three parts, each rendered through
//! `sdm-data`'s one `FROM` function with every part quoted alone, the
//! statement's text pinned and the statement executed on DuckDB, so each
//! reads the table its parts name and no other: `["a.b"]` the table named
//! `a.b`, not the decoy `b` of schema `a`; a part holding a double quote
//! with the quote doubled. The catalog reads of a relation of several
//! parts (its columns and keys) resolve it as DuckDB resolves the same name
//! in a `FROM`. A relation of no part, or of four, is refused by name
//! (rule `relation`) before anything is run.

use duckdb::Connection;
use duckdb::arrow::array::{Array, AsArray, Int32Array};
use sdm_contract::data::{DataRequest, DescribePart, Dialect, FetchPlan, ProviderChoice, Relation};
use sdm_contract::run::{FetchStep, ReplyId};
use sdm_contract::{Detail, ErrorCode};
use sdm_testkit::duck;

/// A source holding a table of each relation form, each with its own value
/// of `v`, and a decoy `b` of schema `a` beside the table named `a.b`.
fn source() -> Connection {
    let connection = duck::open().unwrap();
    connection
        .execute_batch(
            r#"
            CREATE TABLE t (v INTEGER PRIMARY KEY);
            INSERT INTO t VALUES (1);
            CREATE TABLE "a.b" (v INTEGER);
            INSERT INTO "a.b" VALUES (2);
            CREATE SCHEMA a;
            CREATE TABLE a.b (v INTEGER);
            INSERT INTO a.b VALUES (3);
            CREATE SCHEMA s;
            CREATE TABLE s.t (v INTEGER);
            INSERT INTO s.t VALUES (4);
            CREATE TABLE "say ""hi""" (v INTEGER);
            INSERT INTO "say ""hi""" VALUES (5);
            ATTACH ':memory:' AS other;
            CREATE TABLE other.main.u (v INTEGER PRIMARY KEY);
            INSERT INTO other.main.u VALUES (6);
            "#,
        )
        .unwrap();
    connection
}

fn relation(parts: &[&str]) -> Relation {
    Relation {
        columns: Vec::new(),
        filter: None,
        found: false,
        source: parts.iter().map(|p| (*p).to_owned()).collect(),
    }
}

fn plan(request: &DataRequest) -> Result<FetchPlan, sdm_contract::CoreError> {
    let duckdb = ProviderChoice::Sql {
        dialect: Dialect::new("duckdb"),
        namespace: Vec::new(),
        view: None,
    };
    sdm_host::default_root().data_access(&duckdb)?.plan(request)
}

/// The plan's one statement, and what it reads on `connection`.
fn run(connection: &Connection, request: DataRequest) -> (String, Vec<duck::Fetched>) {
    let plan = plan(&request).unwrap();
    let FetchPlan::Sql { statements, .. } = &plan else {
        panic!("not SQL");
    };
    let text = statements[0].text.clone();
    let step = FetchStep {
        id: ReplyId(1),
        request,
        plan,
        continuation: None,
        purpose: None,
    };
    let fetched = duck::answer(connection, &step).unwrap_or_else(|e| panic!("{text}: {e:?}"));
    (text, vec![fetched])
}

fn scan(parts: &[&str]) -> DataRequest {
    DataRequest::Scan {
        table: relation(parts),
        columns: vec!["v".into()],
        filters: vec![],
        limit: None,
    }
}

fn values(fetched: &[duck::Fetched]) -> Vec<i32> {
    fetched
        .iter()
        .flat_map(|f| &f.batches)
        .flat_map(|b| {
            let column = b.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
            column.values().to_vec()
        })
        .collect()
}

#[test]
fn each_relation_form_duckdb_takes_reads_the_table_its_parts_name() {
    let connection = source();
    for (parts, from, value) in [
        (&["t"][..], r#""t""#, 1),
        // A dotted name is one table, never split on its dot.
        (&["a.b"], r#""a.b""#, 2),
        (&["main", "t"], r#""main"."t""#, 1),
        (&["memory", "main", "t"], r#""memory"."main"."t""#, 1),
        (&["s", "t"], r#""s"."t""#, 4),
        // Two parts naming an attached database, as DuckDB reads them.
        (&["other", "u"], r#""other"."u""#, 6),
        // A part holding a double quote, the quote doubled.
        (&["say \"hi\""], r#""say ""hi""""#, 5),
        (&["main", "say \"hi\""], r#""main"."say ""hi""""#, 5),
    ] {
        let (text, fetched) = run(&connection, scan(parts));
        assert_eq!(text, format!(r#"SELECT "v" FROM {from}"#), "{parts:?}");
        assert_eq!(values(&fetched), [value], "{parts:?}");
    }
}

#[test]
fn a_relation_of_no_part_or_of_four_is_refused_by_name() {
    for (parts, code) in [
        (&[][..], ErrorCode::InvalidRequest),
        (&["memory", "main", "s", "t"], ErrorCode::UnsupportedRequest),
    ] {
        for request in [
            scan(parts),
            DataRequest::Describe {
                tables: vec![relation(parts)],
                part: DescribePart::Columns,
            },
        ] {
            let err = plan(&request).unwrap_err();
            assert_eq!(err.code, code, "{parts:?}");
            assert_eq!(err.fields["rule"], Detail::text("relation"), "{parts:?}");
            let given = u64::try_from(parts.len()).unwrap();
            assert_eq!(
                err.fields["parts"],
                Detail::Scalar(sdm_contract::Scalar::UInt64(given))
            );
        }
    }
}

/// The names of the columns, and the primary key's columns, a relation's
/// catalog reads give, each read as DuckDB resolves the relation.
fn described(connection: &Connection, parts: &[&str]) -> (Vec<String>, Vec<String>) {
    let ask = |part| DataRequest::Describe {
        tables: vec![relation(parts)],
        part,
    };
    let (_, columns) = run(connection, ask(DescribePart::Columns));
    let names = columns
        .iter()
        .flat_map(|f| &f.batches)
        .flat_map(|b| {
            let at = b.schema().index_of("name").unwrap();
            let names = b.column(at).as_string::<i32>();
            (0..names.len())
                .map(|i| names.value(i).to_owned())
                .collect::<Vec<_>>()
        })
        .collect();
    let (_, keys) = run(connection, ask(DescribePart::Keys));
    let keyed = keys
        .iter()
        .flat_map(|f| &f.batches)
        .flat_map(|b| {
            let at = b.schema().index_of("constraint_column_names").unwrap();
            let lists = b.column(at).as_list::<i32>();
            let at = b.schema().index_of("constraint_type").unwrap();
            let kinds = b.column(at).as_string::<i32>();
            (0..lists.len())
                .filter(|&i| kinds.value(i) == "PRIMARY KEY")
                .flat_map(|i| {
                    let list = lists.value(i);
                    let names = list.as_string::<i32>();
                    (0..names.len())
                        .map(|j| names.value(j).to_owned())
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        })
        .collect();
    (names, keyed)
}

#[test]
fn a_relation_s_catalog_is_read_where_its_parts_resolve() {
    let connection = source();
    let v = || vec!["v".to_owned()];
    for (parts, keys) in [
        (&["t"][..], v()),
        (&["main", "t"], v()),
        (&["memory", "main", "t"], v()),
        // Schema `s`'s table has no key, though `main`'s `t` has.
        (&["s", "t"], vec![]),
        (&["other", "u"], v()),
        (&["memory", "s", "t"], vec![]),
        (&["main", "say \"hi\""], vec![]),
    ] {
        assert_eq!(described(&connection, parts), (v(), keys), "{parts:?}");
    }
    // Not vacuous: a value read from the catalog's rows, which an absent
    // table gives none of.
    let (_, none) = run(
        &connection,
        DataRequest::Describe {
            tables: vec![relation(&["memory", "nowhere", "t"])],
            part: DescribePart::Keys,
        },
    );
    assert!(
        none.iter()
            .all(|f| f.batches.iter().all(|b| b.num_rows() == 0))
    );
}

/// The rows the source's own resolution of `parts` gives (the table list
/// read with the relation named), as the core asks once a named table's
/// columns could not be read.
fn held(connection: &Connection, parts: &[&str]) -> usize {
    let (_, fetched) = run(
        connection,
        DataRequest::Describe {
            tables: vec![relation(parts)],
            part: DescribePart::Tables,
        },
    );
    fetched
        .iter()
        .flat_map(|f| &f.batches)
        .map(|b| b.num_rows())
        .sum()
}

#[test]
fn a_relation_of_several_parts_is_held_where_its_parts_resolve() {
    let connection = source();
    for (parts, rows) in [
        (&["s", "t"][..], 1),
        (&["memory", "s", "t"], 1),
        (&["other", "u"], 1),
        (&["s", "u"], 0),
        (&["other", "t"], 0),
        (&["memory", "main", "nowhere"], 0),
    ] {
        assert_eq!(held(&connection, parts), rows, "{parts:?}");
    }
}
