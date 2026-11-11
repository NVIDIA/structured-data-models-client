// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Postgres dialect, as the core plans for it (written by
//! independent test author before the implementation).
//!
//! - The staging rule's markers: the data layer renders Postgres
//!   (`sdm_host::providers`, which the binding reports as `PROVIDERS`),
//!   and rust-host has a Postgres provider (`sdm_testkit::PROVIDERS`, the
//!   replay of recorded fetch results), so the core legs run on
//!   Postgres.
//! - Every request kind the core asks, on every base of
//!   `contract/request-kinds.json`, is planned for Postgres through the
//!   root's data access (so each optional request is one the dialect
//!   declares) as SQL statements of the Postgres dialect; a keyed request
//!   binds its key set as a parameter of a statement (`SqlParam::KeySet`,
//!   `P1 d:131`: the binding is the dialect's).
//! - Choice 12's relation forms on Postgres: one part or two (schema and
//!   table) plan, each part quoted alone (a dotted name one identifier, a
//!   double quote doubled); none, or three, is refused by name (rule
//!   `relation`), as DuckDB refuses none or four. The same forms are
//!   executed on a real Postgres by the client's
//!   `tests_test_ops/test_the_postgres_provider.py`.

use std::path::PathBuf;

use sdm_contract::data::{
    DataRequest, DescribePart, Dialect, FetchPlan, ProviderChoice, Relation, SqlParam,
};
use sdm_contract::{Detail, ErrorCode};
use serde_json::{Value, json};

fn contract() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../contract")
}

/// A Postgres source under its schema, as every run reads one: the
/// source's, or its session's current one, which the core reads once per
/// run before any table.
fn postgres() -> ProviderChoice {
    ProviderChoice::Sql {
        timestamp_encoding: None,
        dialect: Dialect::new("postgres"),
        namespace: vec![Some("public".to_owned())],
        view: None,
    }
}

fn plan(request: &DataRequest) -> Result<FetchPlan, sdm_contract::CoreError> {
    sdm_host::default_root()
        .data_access(&postgres())?
        .plan(request)
}

#[test]
fn postgres_is_rendered_by_the_data_layer_and_provided_on_rust_host() {
    assert!(sdm_host::providers().contains(&"postgres"));
    assert!(sdm_testkit::PROVIDERS.contains(&"postgres"));
    sdm_host::default_root()
        .data_access(&postgres())
        .expect("the root's data layer takes the Postgres dialect");
}

#[test]
fn every_request_kind_plans_as_postgres_statements() {
    let kinds: Value =
        serde_json::from_slice(&std::fs::read(contract().join("request-kinds.json")).unwrap())
            .unwrap();
    let fetches = kinds["fetches"].as_array().unwrap();
    let mut keyed = 0;
    for fetch in fetches {
        let place = format!("{}: {}", fetch["base"], fetch["name"]);
        let mut request = fetch["request"].clone();
        let has_keys = fetch.get("keys").is_some();
        if has_keys {
            request["seeds"] = serde_json::json!({
                "attachment": 0,
                "columns": [
                    {"name": "key", "type": {"type": "int64"}, "nullable": true},
                    {"name": "batch", "type": {"type": "int64"}, "nullable": true},
                ],
            });
        }
        let request: DataRequest = serde_json::from_value(request).unwrap();
        let plan = plan(&request).unwrap_or_else(|e| panic!("{place}: {e:?}"));
        let FetchPlan::Sql { statements, .. } = plan else {
            panic!("{place}: not SQL");
        };
        assert!(!statements.is_empty(), "{place}");
        for statement in &statements {
            assert_eq!(statement.dialect, Dialect::new("postgres"), "{place}");
            assert!(!statement.text.trim().is_empty(), "{place}");
        }
        let binds = statements
            .iter()
            .flat_map(|s| &s.params)
            .any(|p| matches!(p, SqlParam::KeySet { .. }));
        assert_eq!(
            binds, has_keys,
            "{place}: the key set is bound by a parameter"
        );
        keyed += usize::from(has_keys);
    }
    assert!(fetches.len() >= 45);
    assert_eq!(keyed, 6);
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

#[test]
fn a_relation_of_one_or_two_parts_plans_each_part_quoted_alone() {
    for (parts, from, not) in [
        (&["t"][..], r#""t""#, None),
        // A dotted name is one table, never split on its dot.
        (&["a.b"], r#""a.b""#, Some(r#""a"."b""#)),
        (&["s", "t"], r#""s"."t""#, None),
        // A part holding a double quote, the quote doubled.
        (&["say \"hi\""], r#""say ""hi""""#, None),
        (&["s", "say \"hi\""], r#""s"."say ""hi""""#, None),
    ] {
        let FetchPlan::Sql { statements, .. } = plan(&scan(parts)).unwrap() else {
            panic!("not SQL");
        };
        let text = &statements.last().unwrap().text;
        assert!(text.contains(from), "{parts:?}: {text}");
        if let Some(not) = not {
            assert!(!text.contains(not), "{parts:?}: {text}");
        }
    }
}

#[test]
fn a_relation_of_no_part_or_of_three_is_refused_by_name() {
    for (parts, code) in [
        (&[][..], ErrorCode::InvalidRequest),
        (&["db", "s", "t"], ErrorCode::UnsupportedRequest),
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

/// A key-set parameter bound as `__sdm_seeds__`: its attachment, the key
/// and batch columns, then `more`.
fn key_set(attachment: u64, more: &[Value]) -> Value {
    let mut columns = vec![
        json!({"name": "key", "type": {"type": "int64"}, "nullable": true}),
        json!({"name": "batch", "type": {"type": "int64"}, "nullable": false}),
    ];
    columns.extend(more.iter().cloned());
    json!({
        "param": "key_set",
        "keys": {"attachment": attachment, "columns": columns},
        "relation": "__sdm_seeds__",
    })
}

/// The statements `request` plans on Postgres, as the contract writes them.
fn planned(request: Value) -> Value {
    let request: DataRequest = serde_json::from_value(request).unwrap();
    let FetchPlan::Sql { statements, .. } = plan(&request).unwrap() else {
        panic!("not SQL");
    };
    serde_json::to_value(statements).unwrap()
}

/// base.regression's entity lookup and its neighbor read, as Postgres
/// renders them whole: the key set bound once as JSON, each column read,
/// the time bound and the per-seed cap, newest first.
#[test]
fn a_lookup_and_a_neighbor_read_render_whole_with_their_key_sets() {
    let time = json!({
        "name": "time",
        "type": {"type": "timestamp", "unit": "ns", "zone": null},
        "nullable": false,
    });
    let column = |name: &str| json!({"name": name, "column": name});
    let lookup = planned(json!({
        "request": "lookup",
        "table": {
            "source": ["users"],
            "columns": [column("user_id"), column("age"), column("country")],
            "found": true,
        },
        "key": ["user_id"],
        "columns": ["user_id", "age", "country"],
        "seeds": {"attachment": 2, "columns": key_set(2, &[])["keys"]["columns"]},
    }));
    assert_eq!(
        lookup,
        json!([{
            "dialect": "postgres",
            "text": r#"WITH "__sdm_seeds__" AS (
  SELECT (r.value #>> '{}')::bigint AS "key", (r.ordinality - 1)::bigint AS "batch"
  FROM jsonb_array_elements($1::jsonb) WITH ORDINALITY AS r(value, ordinality)
)
SELECT l.b AS "__kumo_batch__", l.c0 AS "user_id", l.c1 AS "age", l.c2 AS "country", l.r AS "__kumo_row__", l.m AS "__kumo_matches__"
FROM (
  SELECT s."batch" AS b, t."user_id" AS c0, t."age" AS c1, t."country" AS c2,
         ROW_NUMBER() OVER (PARTITION BY s."batch"
           ORDER BY t."user_id") AS r,
         COUNT(*) OVER (PARTITION BY s."batch") AS m
  FROM "__sdm_seeds__" s
  JOIN "public"."users" t
    ON t."user_id" = s."key"
) AS l
WHERE l.r = 1
ORDER BY l.b"#,
            "params": [key_set(2, &[])],
        }])
    );
    let neighbors = planned(json!({
        "request": "neighbors",
        "table": {
            "source": ["orders"],
            "columns": [column("order_id"), column("user_id"), column("price"), column("ts")],
            "found": true,
        },
        "key": ["user_id"],
        "columns": ["order_id", "user_id", "price", "ts"],
        "seeds": {
            "attachment": 3,
            "columns": key_set(3, std::slice::from_ref(&time))["keys"]["columns"],
        },
        "window": {
            "column": "ts",
            "source_type": "timestamp without time zone",
            "lower": null,
            "upper": {"seed_column": "time", "inclusive": true},
        },
        "per_seed": 32,
        "order": [
            {"column": "ts", "as_timestamp": true, "descending": true, "nulls_last": true},
            {"column": "order_id", "as_timestamp": false, "descending": false, "nulls_last": true},
        ],
    }));
    assert_eq!(
        neighbors,
        json!([{
            "dialect": "postgres",
            "text": r#"WITH "__sdm_seeds__" AS (
  SELECT (r.value->>'id')::bigint AS "key", (r.ordinality - 1)::bigint AS "batch", w2."ts" AS "time"
  FROM jsonb_array_elements($1::jsonb) WITH ORDINALITY AS r(value, ordinality)
  CROSS JOIN (SELECT ROW(t.*)::"public"."orders" AS base FROM "public"."orders" AS t LIMIT 1) AS b
  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('ts', r.value->'e')) AS w2
)
SELECT s."batch" AS "__kumo_batch__", n.c0 AS "order_id", n.c1 AS "user_id", n.c2 AS "price", n.c3 AS "ts", n.r AS "__kumo_row__"
FROM "__sdm_seeds__" s
CROSS JOIN LATERAL (
  SELECT t."order_id" AS c0, t."user_id" AS c1, t."price" AS c2, t."ts" AS c3, ROW_NUMBER() OVER (ORDER BY t."ts" DESC NULLS LAST, t."order_id" ASC NULLS LAST) AS r
  FROM "public"."orders" t
  WHERE t."user_id" = s."key"
    AND t."ts" <= s."time"
  ORDER BY t."ts" DESC NULLS LAST, t."order_id" ASC NULLS LAST
  LIMIT 32
) AS n
ORDER BY s."batch", n.r"#,
            "params": [key_set(3, &[time])],
        }])
    );
}
