// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! SQLite relation names and invalid qualification through the native provider.

use sdm_contract::data::{DataRequest, DescribePart, Dialect, FetchPlan, ProviderChoice, Relation};
use sdm_contract::{Detail, ErrorCode};

const DATABASE: &str = "sqlite";

fn sqlite() -> ProviderChoice {
    ProviderChoice::Sql {
        timestamp_encoding: None,
        dialect: Dialect::new(DATABASE),
        namespace: Vec::new(),
        view: None,
    }
}

fn relation(parts: &[&str]) -> Relation {
    Relation {
        columns: Vec::new(),
        filter: None,
        found: false,
        source: parts.iter().map(|p| (*p).to_owned()).collect(),
    }
}

fn scan(parts: &[&str]) -> DataRequest {
    DataRequest::Scan {
        table: relation(parts),
        columns: vec!["v".into()],
        filters: vec![],
        limit: None,
    }
}

fn plan(request: &DataRequest) -> Result<FetchPlan, sdm_contract::CoreError> {
    sdm_host::default_root()
        .data_access(&sqlite())?
        .plan(request)
}

fn source() -> rusqlite::Connection {
    let connection = rusqlite::Connection::open_in_memory().unwrap();
    connection
        .execute_batch(
            r#"
            CREATE TABLE t (v INTEGER PRIMARY KEY);
            INSERT INTO t VALUES (1);
            CREATE TABLE "a.b" (v INTEGER);
            INSERT INTO "a.b" VALUES (2);
            CREATE TABLE "say ""hi""" (v INTEGER);
            INSERT INTO "say ""hi""" VALUES (5);
            ATTACH ':memory:' AS other;
            CREATE TABLE other.u (v INTEGER PRIMARY KEY);
            INSERT INTO other.u VALUES (6);
            "#,
        )
        .unwrap();
    connection
}

#[test]
fn each_relation_form_sqlite_takes_reads_the_table_its_parts_name() {
    let connection = source();
    for (parts, from, value) in [
        (&["t"][..], r#""t""#, 1),
        // A dotted name is one table, never split on its dot.
        (&["a.b"], r#""a.b""#, 2),
        (&["main", "t"], r#""main"."t""#, 1),
        // Two parts naming an attached database.
        (&["other", "u"], r#""other"."u""#, 6),
        // A part holding a double quote, the quote doubled.
        (&["say \"hi\""], r#""say ""hi""""#, 5),
        (&["main", "say \"hi\""], r#""main"."say ""hi""""#, 5),
    ] {
        let FetchPlan::Sql { statements, .. } = plan(&scan(parts)).unwrap() else {
            panic!("{parts:?}: not SQL");
        };
        let [statement] = &statements[..] else {
            panic!("{parts:?}: not one statement");
        };
        let text = &statement.text;
        assert!(text.contains(&format!(" FROM {from}")), "{parts:?}: {text}");
        let mut prepared = connection.prepare(text).unwrap();
        let read: Vec<i64> = prepared
            .query_map([], |row| row.get(0))
            .unwrap()
            .map(Result::unwrap)
            .collect();
        assert_eq!(read, [value], "{parts:?}: {text}");
    }
}

#[test]
fn a_relation_of_no_part_or_of_three_is_refused_by_name_on_sqlite() {
    for (parts, code) in [
        (&[][..], ErrorCode::InvalidRequest),
        (&["main", "s", "t"], ErrorCode::UnsupportedRequest),
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
