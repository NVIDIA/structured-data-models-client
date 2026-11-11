// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! SQLite relation names and invalid qualification through the native provider,
//! and one case's journey on SQLite: its read budget and its keyed reads whole.

use std::path::PathBuf;

use sdm_contract::data::{DataRequest, DescribePart, Dialect, FetchPlan, ProviderChoice, Relation};
use sdm_contract::{Detail, ErrorCode};
use sdm_testkit::host::{self, Answer};
use sdm_testkit::{bases, case};
use serde_json::{Value, json};

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

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

/// A key-set parameter: its attachment, its relation, the key and batch
/// columns, then `more`.
fn key_set(attachment: u64, relation: &str, more: &[Value]) -> Value {
    let mut columns = vec![
        json!({"name": "key", "type": {"type": "int64"}, "nullable": true}),
        json!({"name": "batch", "type": {"type": "int64"}, "nullable": false}),
    ];
    columns.extend(more.iter().cloned());
    json!({
        "param": "key_set",
        "keys": {"attachment": attachment, "columns": columns},
        "relation": relation,
    })
}

/// base.regression on SQLite, as the DuckDB journey runs it: a fixed budget
/// of fetches, plan B's refreshes and statements, and the entity lookup and
/// the neighbor read whole, with the key sets they bind.
#[test]
fn base_regression_s_journey_reads_a_fixed_budget_on_sqlite() {
    let contract = root().join("contract");
    let loaded = case::load(&[contract.join("cases/s1/base.regression.json")]).unwrap();
    let bases = bases::load(&contract.join("bases.yaml")).unwrap();
    let observed = host::run_sqlite(&loaded[0].case, &bases, &root()).unwrap();
    assert!(matches!(observed.answer, Answer::Predicted(_)));
    // Each table's catalog, keys and preview are read again at materialize.
    let refreshes = observed
        .fetches
        .iter()
        .filter(|f| f["purpose"] == sdm_testkit::plan_b::LOCATE)
        .count();
    assert_eq!((observed.fetches.len(), refreshes), (24, 6));
    let driver = observed.driver.as_ref().expect("a driver record");
    let statements = driver["capture"]["statements"].as_array().unwrap();
    let per_stage = |stage: &str| statements.iter().filter(|s| s["stage"] == stage).count();
    assert_eq!(
        (
            per_stage("from_source"),
            per_stage("materialize"),
            per_stage("predict")
        ),
        (10, 14, 0)
    );
    assert_eq!(driver["query_log"]["statements"].as_u64(), Some(24));
    assert_eq!(driver["capture"]["bindings"].as_array().unwrap().len(), 3);
    let planned = |wanted: &dyn Fn(&Value) -> bool| {
        let found: Vec<_> = observed
            .fetches
            .iter()
            .filter(|f| wanted(&f["request"]))
            .collect();
        assert_eq!(found.len(), 1, "one such read");
        found[0]["plan"]["statements"].clone()
    };
    assert_eq!(
        planned(&|r| r["request"] == "lookup"),
        json!([{
            "dialect": "sqlite",
            "text": r#"SELECT s."batch" AS "__kumo_batch__", (SELECT COUNT(*) FROM "users" WHERE "user_id" == s."key") AS "__kumo_matches__", t."user_id", t."age", t."country"
FROM temp."__sdm_seeds_19f857a1097d1e84__" s
JOIN "users" t
  ON t.rowid = (
    SELECT rowid
    FROM "users"
    WHERE "user_id" == s."key"
    LIMIT 1
)"#,
            "params": [key_set(2, "__sdm_seeds_19f857a1097d1e84__", &[])],
        }])
    );
    assert_eq!(
        planned(&|r| r["request"] == "neighbors" && !r["per_seed"].is_null()),
        json!([{
            "dialect": "sqlite",
            "text": r#"SELECT s."batch" AS "__kumo_batch__", t."order_id", t."user_id", t."price", t."ts"
FROM temp."__sdm_seeds_0086307c4757992b__" s
JOIN "orders" t
ON t.rowid IN (
  SELECT rowid
  FROM "orders"
  WHERE "user_id" = s."key"
  AND "ts" <= substr(s."time", 1, 10) || ' ' || substr(s."time", 12, 8)
  ORDER BY "ts" DESC, "order_id"
  LIMIT 32
)
ORDER BY "__kumo_batch__", t."ts" DESC, t."order_id""#,
            "params": [key_set(3, "__sdm_seeds_0086307c4757992b__", &[json!({
                "name": "time",
                "type": {"type": "timestamp", "unit": "ns", "zone": null},
                "nullable": false,
                "metadata": {"sdm:stored_time": "2023-01-01 00:00:00"},
            })])],
        }])
    );
}
