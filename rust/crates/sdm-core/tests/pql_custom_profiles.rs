// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Model capabilities govern queries, not unused tables in the schema.
use sdm_contract::ops::{Capability, ProfileName, ProfileSpec, QuerySchema};
use sdm_contract::pql::{EntityClause, NodeKind, PqlLiteral, ProblemType, QueryMode};
use sdm_contract::{HostFacts, Scalar, Unit};
use sdm_core::pql::resolve;
use sdm_core::profiles::{Mode, Profile};
use serde_json::json;

struct NoZones;
impl sdm_core::Zones for NoZones {
    fn zone(&self, _: &str) -> Result<jiff::tz::TimeZone, sdm_core::UnknownZone> {
        Err(sdm_core::UnknownZone)
    }
}

fn schema() -> QuerySchema {
    let column = |name: &str, dtype: &str, stype: &str| json!({"name": name, "dtype": dtype, "stype": stype});
    let table = |name: &str, key: &str, time: Option<&str>, columns: Vec<serde_json::Value>| {
        json!({
            "name": name, "primary_key": key, "time_column": time, "end_time_column": null, "columns": columns
        })
    };
    let id = |name| column(name, "int", "ID");
    serde_json::from_value(json!({
        "tables": [
            table("users", "user_id", None, vec![id("user_id"), column("age", "int", "numerical")]),
            table("products", "product_id", None, vec![id("product_id")]),
            table("orders", "order_id", Some("ordered_at"), vec![id("order_id"), id("user_id"), id("product_id"), column("amount", "float", "numerical"), column("status", "string", "categorical"), column("ordered_at", "time", "timestamp")]),
            table("reviews", "review_id", Some("created_at"), vec![id("review_id"), id("user_id"), column("created_at", "time", "timestamp")])
        ],
        "links": [
            [{"table": "orders", "column": "user_id"}, {"table": "users", "column": "user_id"}],
            [{"table": "orders", "column": "product_id"}, {"table": "products", "column": "product_id"}],
            [{"table": "reviews", "column": "user_id"}, {"table": "users", "column": "user_id"}]
        ]
    })).unwrap()
}

const FLAGS: [Capability; 11] = [
    Capability::Relationships,
    Capability::MultipleTables,
    Capability::Temporal,
    Capability::Static,
    Capability::EntityFilters,
    Capability::Counterfactuals,
    Capability::DerivedTargets,
    Capability::LinkPrediction,
    Capability::Forecasting,
    Capability::KeyColumnsInFilters,
    Capability::TimestampColumnsInFilters,
];
const S: &str = "PREDICT users.age FOR EACH users.user_id";
const T: &str = "PREDICT COUNT(orders.*, 0, 30, days) FOR EACH users.user_id";
const D: &str = "PREDICT users.age > 30 FOR EACH users.user_id";
const F: &str =
    "PREDICT SUM(orders.amount, 0, 1, days) FORECAST 5 TIMEFRAMES FOR EACH users.user_id";
const E: &str = "PREDICT COUNT(orders.*, 0, 30, days) FOR EACH users.user_id WHERE users.age > 30";
const I: &str = "PREDICT COUNT(orders.*, 0, 30, days) FOR users.user_id IN (1, 2)";
const KEY: &str =
    "PREDICT COUNT(orders.* WHERE orders.product_id = 3, 0, 30, days) FOR EACH users.user_id";
const TIME: &str = "PREDICT COUNT(orders.* WHERE orders.ordered_at IS NOT NULL, 0, 30, days) FOR EACH users.user_id";
const KEY_CODES: &[&str] = &[
    "PQL_MODEL_NO_KEY_COLUMN_FILTER",
    "PQL_MODEL_NO_KEY_COLUMN_REFERENCE",
];

fn profile(mode: Mode, disabled: &[Capability]) -> Profile {
    let mut profile = Profile::resolve(&ProfileSpec {
        name: ProfileName::Relational,
        model: Some("custom-profile".into()),
        capabilities: disabled.iter().map(|flag| (*flag, false)).collect(),
    });
    profile.mode = mode;
    profile
}

fn check(query: &str, profile: &Profile, expected: &[&str]) {
    let schema = schema();
    let facts = HostFacts {
        now: Scalar::Timestamp {
            unit: Unit::Us,
            zone: Some("UTC".into()),
            value: 1_767_225_600_000_000,
        },
        local_zone: None,
        zone_roots: vec![],
        platform: sdm_contract::facts::PlatformFacts {
            os: "linux".into(),
            strftime_pads_year: false,
        },
        date_literal_range: None,
    };
    let got = resolve(query, &schema, profile, &facts, &NoZones).unwrap();
    let result = got.to_result(&schema);
    if let Some(resolved) = &got.query {
        assert_eq!(
            resolved.mode,
            match profile.mode {
                Mode::Inference => QueryMode::Inference,
                Mode::Training => QueryMode::Training,
            }
        );
        assert_eq!(resolved.num_forecasts, if query == F { 5 } else { 1 });
        if query.contains("RANK TOP") {
            assert_eq!(resolved.top_k.as_deref(), Some("5"));
            assert_eq!(resolved.problem_type, Some(ProblemType::Rank));
        }
        if query == I {
            assert_eq!(resolved.entity_clause, EntityClause::For);
            let NodeKind::Condition { value, .. } = &resolved.entity_ids.as_ref().unwrap().kind
            else {
                panic!("entity ID condition");
            };
            let NodeKind::Constant {
                literal: PqlLiteral::Array { items },
            } = &value.kind
            else {
                panic!("entity IDs array");
            };
            let ids: Vec<_> = items
                .iter()
                .map(|item| match &item.kind {
                    NodeKind::Constant {
                        literal: PqlLiteral::Int { value },
                    } => value.as_str(),
                    _ => panic!("integer entity ID"),
                })
                .collect();
            assert_eq!(ids, ["1", "2"]);
        }
    }

    assert_eq!(result.accepted, expected.is_empty(), "{query}");
    assert_eq!(result.query.is_some(), expected.is_empty(), "{query}");
    if let Some(facts) = &result.query {
        assert_eq!(facts.entity_table, "users", "{query}");
        assert_eq!(
            facts.num_forecasts,
            if query == F { 5 } else { 1 },
            "{query}"
        );
    }
    let codes: Vec<_> = result
        .diagnostics
        .iter()
        .filter(|d| d.severity == sdm_contract::ops::Severity::Error)
        .map(|d| d.code.as_str())
        .collect();
    assert_eq!(codes, expected, "{query}");
}

#[test]
fn each_capability_has_an_enabled_disabled_and_unaffected_control() {
    use Capability::*;
    let rows: &[(Capability, &str, &[&str], &str)] = &[
        (Relationships, T, &["PQL_MODEL_NO_RELATIONSHIPS"], S),
        (MultipleTables, T, &["PQL_MODEL_SINGLE_TABLE_ONLY"], S),
        (Temporal, T, &["PQL_MODEL_NO_TEMPORAL_TARGET"], S),
        (Static, S, &["PQL_MODEL_NO_STATIC_TARGET"], T),
        (EntityFilters, E, &["PQL_MODEL_NO_ENTITY_FILTER"], S),
        (
            Counterfactuals,
            "PREDICT SUM(orders.amount, 0, 30, days) FOR EACH users.user_id ASSUMING COUNT(orders.*, 0, 7, days) > 2",
            &["PQL_MODEL_NO_COUNTERFACTUAL"],
            S,
        ),
        (DerivedTargets, D, &["PQL_MODEL_NO_DERIVED_TARGET"], S),
        (
            LinkPrediction,
            "PREDICT LIST_DISTINCT(orders.product_id, 0, 30, days) RANK TOP 5 FOR EACH users.user_id",
            &["PQL_MODEL_CANNOT_RANK_FOREIGN_KEY"],
            S,
        ),
        (Forecasting, F, &["PQL_MODEL_NO_FORECASTING"], S),
        (KeyColumnsInFilters, KEY, KEY_CODES, S),
        (
            TimestampColumnsInFilters,
            TIME,
            &["PQL_MODEL_UNSUPPORTED_COLUMN_OPERATION"],
            S,
        ),
    ];
    for (flag, required, errors, unaffected) in rows {
        check(required, &profile(Mode::Inference, &[]), &[]);
        let disabled = profile(Mode::Inference, &[*flag]);
        check(required, &disabled, errors);
        check(unaffected, &disabled, &[]);
    }
}

#[test]
fn temporal_gate_and_mode_preserve_error_precedence() {
    for mode in [Mode::Inference, Mode::Training] {
        let all = profile(mode, &[]);
        for query in [S, T, D, F, E] {
            check(query, &all, &[]);
        }
        let none = profile(mode, &FLAGS);
        check(S, &none, &["PQL_MODEL_NO_STATIC_TARGET"]);
        for query in [T, I] {
            check(query, &none, &["PQL_MODEL_NO_TEMPORAL_TARGET"]);
        }
        check(
            D,
            &none,
            &["PQL_MODEL_NO_STATIC_TARGET", "PQL_MODEL_NO_DERIVED_TARGET"],
        );
    }
    check(I, &profile(Mode::Inference, &[]), &[]);
    check(
        I,
        &profile(Mode::Training, &[]),
        &[
            "PQL_MODEL_FOR_REQUIRES_PRETRAINED",
            "PQL_MODEL_ENTITY_IDS_REQUIRE_PRETRAINED",
        ],
    );
    let disabled: Vec<_> = FLAGS
        .into_iter()
        .filter(|flag| !matches!(flag, Capability::Temporal | Capability::Static))
        .collect();
    let temporal_static = profile(Mode::Inference, &disabled);
    check(S, &temporal_static, &[]);
    check(
        T,
        &temporal_static,
        &["PQL_MODEL_SINGLE_TABLE_ONLY", "PQL_MODEL_NO_DERIVED_TARGET"],
    );
    check(
        F,
        &temporal_static,
        &[
            "PQL_MODEL_SINGLE_TABLE_ONLY",
            "PQL_MODEL_NO_DERIVED_TARGET",
            "PQL_MODEL_NO_FORECASTING",
        ],
    );
    check(
        E,
        &temporal_static,
        &[
            "PQL_MODEL_SINGLE_TABLE_ONLY",
            "PQL_MODEL_NO_DERIVED_TARGET",
            "PQL_MODEL_NO_ENTITY_FILTER",
        ],
    );
}

#[test]
fn training_bypasses_only_inference_restrictions() {
    let flags = [
        Capability::KeyColumnsInFilters,
        Capability::TimestampColumnsInFilters,
    ];
    check(KEY, &profile(Mode::Inference, &flags), KEY_CODES);
    check(
        TIME,
        &profile(Mode::Inference, &flags),
        &["PQL_MODEL_UNSUPPORTED_COLUMN_OPERATION"],
    );
    for query in [KEY, TIME] {
        check(query, &profile(Mode::Training, &flags), &[]);
    }
    let query = "PREDICT users.age FOR EACH users.user_id WHERE users.user_id > 3";
    check(query, &profile(Mode::Inference, &FLAGS), KEY_CODES);
    check(
        query,
        &profile(Mode::Training, &FLAGS),
        &["PQL_MODEL_NO_STATIC_TARGET", "PQL_MODEL_NO_ENTITY_FILTER"],
    );
    let first = "PREDICT FIRST(orders.status, 0, 30, days) FOR EACH users.user_id WHERE COUNT(reviews.*, -30, 0, days) > 1";
    check(
        first,
        &profile(Mode::Inference, &[]),
        &[
            "PQL_MODEL_UNSUPPORTED_AGGREGATION",
            "PQL_MODEL_UNSUPPORTED_AGGREGATION_COLUMN",
        ],
    );
    check(first, &profile(Mode::Training, &[]), &[]);
}
