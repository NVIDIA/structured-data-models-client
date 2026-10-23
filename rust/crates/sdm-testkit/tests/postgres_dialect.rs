// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! T2.3's Postgres dialect, as the core plans for it (written by T2.3's
//! independent test author before the implementation).
//!
//! - The staging rule's markers (T2.1a): the data layer renders Postgres
//!   (`sdm_host::providers()`, which the binding reports as `PROVIDERS`),
//!   and rust-host has a Postgres provider (`sdm_testkit::PROVIDERS`, the
//!   replay of recorded fetch results, KO2-20), so the core legs run on
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
use serde_json::Value;

fn contract() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../contract")
}

fn postgres() -> ProviderChoice {
    ProviderChoice::Sql {
        dialect: Dialect::new("postgres"),
        namespace: Vec::new(),
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
        columns: Vec::new(), filter: None,
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
