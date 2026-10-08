// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The adapter's own rules: a model that cannot answer a time range refuses
//! before resolution; the two filter capabilities reach resolution; a
//! literal the interim parser cannot read yet is no answer at all; and an
//! accepted query is handed on in the validated-query format.

use std::collections::BTreeMap;

use sdm_contract::facts::PlatformFacts;
use sdm_contract::ops::{Capability, ProfileName, ProfileSpec, QuerySchema};
use sdm_contract::pql::{Aggregation, EntityClause, NodeKind, QueryMode};
use sdm_contract::{ErrorCode, ErrorKind, HostFacts, Scalar, Unit};
use sdm_core::pql::{resolve, validate};
use sdm_core::profiles::Profile;
use serde_json::json;

fn facts() -> HostFacts {
    HostFacts {
        now: Scalar::Timestamp {
            unit: Unit::Us,
            zone: Some("UTC".into()),
            value: 1_767_225_600_000_000,
        },
        local_zone: None,
        zone_roots: vec![],
        platform: PlatformFacts {
            os: "linux".into(),
            strftime_pads_year: false,
        },
    }
}

/// The inventory's `shop` tables, as the reference describes them.
fn shop() -> QuerySchema {
    serde_json::from_value(json!({
        "tables": [
            {"name": "orders", "columns": [
                {"name": "order_id", "dtype": "int", "stype": "ID"},
                {"name": "user_id", "dtype": "int", "stype": "ID"},
                {"name": "price", "dtype": "float", "stype": "numerical"},
                {"name": "ts", "dtype": "time", "stype": "timestamp"}],
             "primary_key": "order_id", "time_column": "ts", "end_time_column": null},
            {"name": "users", "columns": [
                {"name": "user_id", "dtype": "int", "stype": "ID"},
                {"name": "age", "dtype": "int", "stype": "numerical"},
                {"name": "country", "dtype": "string", "stype": "categorical"}],
             "primary_key": "user_id", "time_column": null, "end_time_column": null}],
        "links": [[{"table": "orders", "column": "user_id"}, {"table": "users", "column": "user_id"}]],
    }))
    .unwrap()
}

fn profile(name: ProfileName, capabilities: &[(Capability, bool)]) -> Profile {
    Profile::resolve(&ProfileSpec {
        name,
        model: None,
        capabilities: capabilities.iter().copied().collect::<BTreeMap<_, _>>(),
    })
}

fn codes_of(query: &str, profile: &Profile) -> Vec<String> {
    resolve(query, &shop(), profile, &facts())
        .unwrap()
        .diagnostics
        .into_iter()
        .map(|d| d.code)
        .collect()
}

#[test]
fn a_time_range_is_refused_before_resolution() {
    // A tabular model predicts values as they stand: the window is the
    // reason no schema can fix, and it is the only one given.
    let query = "PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id";
    let validation = resolve(
        query,
        &shop(),
        &profile(ProfileName::Tabular, &[]),
        &facts(),
    )
    .unwrap();
    assert!(validation.query.is_none());
    assert_eq!(validation.diagnostics.len(), 1);
    assert_eq!(
        validation.diagnostics[0].code,
        "PQL_MODEL_NO_TEMPORAL_TARGET"
    );
    assert_eq!(validation.diagnostics[0].span, None);
    // With the capability granted, the same query resolves and meets the
    // checks of the resolved query instead.
    let granted = profile(ProfileName::Tabular, &[(Capability::Temporal, true)]);
    assert_eq!(
        codes_of(query, &granted),
        ["PQL_MODEL_SINGLE_TABLE_ONLY", "PQL_MODEL_NO_DERIVED_TARGET"]
    );
}

#[test]
fn the_filter_capabilities_reach_resolution() {
    let relational = profile(ProfileName::Relational, &[]);
    let on_key =
        "PREDICT COUNT(orders.* WHERE orders.user_id = 3, 0, 30, days) FOR EACH users.user_id";
    let on_time = "PREDICT COUNT(orders.* WHERE orders.ts > 2024-01-01, 0, 30, days) FOR EACH \
                   users.user_id";
    assert_eq!(codes_of(on_key, &relational), Vec::<String>::new());
    assert_eq!(codes_of(on_time, &relational), Vec::<String>::new());

    let no_keys = profile(
        ProfileName::Relational,
        &[(Capability::KeyColumnsInFilters, false)],
    );
    assert!(codes_of(on_key, &no_keys).contains(&"PQL_MODEL_NO_KEY_COLUMN_FILTER".to_owned()));
    let no_times = profile(
        ProfileName::Relational,
        &[(Capability::TimestampColumnsInFilters, false)],
    );
    assert_eq!(
        codes_of(on_time, &no_times),
        ["PQL_MODEL_UNSUPPORTED_COLUMN_OPERATION"]
    );
}

#[test]
fn a_literal_the_interim_parser_cannot_read_is_no_answer() {
    let query = "PREDICT COUNT(orders.* WHERE orders.ts > 20240101/00/123456, 0, 30, days) FOR \
                 EACH users.user_id";
    let err = validate(
        query,
        &shop(),
        &profile(ProfileName::Relational, &[]),
        &facts(),
    )
    .unwrap_err();
    assert_eq!(
        (err.kind, err.code),
        (ErrorKind::Unsupported, ErrorCode::UnsupportedRequest)
    );
}

#[test]
fn an_accepted_query_is_handed_on_in_the_validated_query_format() {
    let validation = resolve(
        "PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id",
        &shop(),
        &profile(ProfileName::Relational, &[]),
        &facts(),
    )
    .unwrap();
    assert!(validation.diagnostics.is_empty());
    assert!(validation.to_result().accepted);
    let query = validation.query.unwrap();
    assert_eq!(query.entity_clause, EntityClause::ForEach);
    assert_eq!(query.mode, QueryMode::Inference);
    let NodeKind::Join {
        target, lhs_key, ..
    } = &query.target.kind
    else {
        panic!("the target is not joined to users: {:?}", query.target.kind);
    };
    assert_eq!(
        (lhs_key.table.as_str(), lhs_key.column.as_str()),
        ("users", "user_id")
    );
    let NodeKind::Aggregation {
        aggregation,
        time_column,
        ..
    } = &target.kind
    else {
        panic!("the joined target is not an aggregation");
    };
    assert_eq!(*aggregation, Aggregation::Sum);
    assert_eq!(time_column.as_ref().unwrap().column, "ts");
}

/// The deepest queries the parser takes, and one level past them.
fn deepest() -> Vec<String> {
    // The interim parser's nesting limit (`parser::MAX_DEPTH`).
    const MAX_DEPTH: usize = 1000;
    let mut out = Vec::new();
    for depth in [MAX_DEPTH - 2, MAX_DEPTH + 1] {
        out.push(format!(
            "PREDICT {}COUNT(orders.*, 0, 30, days) > 1 FOR EACH users.user_id",
            "NOT ".repeat(depth)
        ));
        out.push(format!(
            "PREDICT COUNT(orders.*, 0, 30, days) FOR EACH users.user_id WHERE {}users.age > \
             3{}",
            "(".repeat(depth),
            ")".repeat(depth)
        ));
    }
    out
}

fn on_a_stack_of(bytes: usize) {
    std::thread::Builder::new()
        .stack_size(bytes)
        .spawn(|| {
            let relational = profile(ProfileName::Relational, &[]);
            let answers: Vec<bool> = deepest()
                .iter()
                .map(|query| {
                    resolve(query, &shop(), &relational, &facts())
                        .unwrap()
                        .query
                        .is_some()
                })
                .collect();
            // The deepest are answered, accepted; one level deeper, refused.
            assert_eq!(answers, [true, true, false, false]);
        })
        .unwrap()
        .join()
        .expect("answered");
}

#[test]
fn the_deepest_queries_validate_on_a_spawned_threads_default_stack() {
    on_a_stack_of(2 << 20);
}

#[test]
fn the_deepest_queries_validate_on_a_small_thread_stack() {
    on_a_stack_of(512 << 10);
}
