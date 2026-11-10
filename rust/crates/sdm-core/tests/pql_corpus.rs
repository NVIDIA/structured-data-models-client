// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Independent diagnostic rules and selected resolved-query semantics.
mod pql_support;
use pql_support::{NoZones, facts, profile, schema};
use sdm_contract::ops::Severity;
use sdm_core::pql::resolve;

#[test]
fn diagnostic_rules_return_their_named_errors() {
    let many_ids = format!(
        "PREDICT COUNT(orders.*, 0, 30, days) FOR users.user_id IN ({})",
        (0..=1000)
            .map(|n| n.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );
    let cases: &[(&str, &str, &str, &[&str])] = &[
        (
            "schema-1",
            "RELATIONAL",
            r###"PREDICT COUNT(orders.*, 0, 30, days) FOR EACH users.*"###,
            &[
                "PQL_SCHEMA_ENTITY_NOT_PRIMARY_KEY",
                "PQL_SCHEMA_WILDCARD_POSITION",
            ],
        ),
        (
            "schema-1",
            "FINE_TUNING",
            r###"PREDICT COUNT(orders.*, 0, 30, days) FOR users.user_id = 1"###,
            &[
                "PQL_MODEL_FOR_REQUIRES_PRETRAINED",
                "PQL_MODEL_ENTITY_IDS_REQUIRE_PRETRAINED",
            ],
        ),
        (
            "schema-1",
            "RELATIONAL",
            r###"PREDICT LAST(orders.status, 0, 30, days) FOR EACH users.user_id"###,
            &[
                "PQL_MODEL_UNSUPPORTED_AGGREGATION",
                "PQL_MODEL_UNSUPPORTED_AGGREGATION_COLUMN",
            ],
        ),
        (
            "schema-1",
            "FINE_TUNING",
            r###"PREDICT LIST_DISTINCT(orders.product_id) RANK TOP 3 FOR EACH users.user_id"###,
            &[
                "PQL_TARGET_LIST_DISTINCT_NEEDS_TIME_RANGE",
                "PQL_TARGET_LIST_DISTINCT_TIME_COLUMNS",
            ],
        ),
        (
            "schema-3",
            "TABULAR",
            r###"PREDICT people.churn FOR EACH people.person_id ASSUMING people.segment = 'gold'"###,
            &[
                "PQL_TIME_ASSUMING_REQUIRES_TEMPORAL",
                "PQL_TIME_STATIC_CONDITION_NOT_ALLOWED",
            ],
        ),
        (
            "schema-1",
            "TIMESERIES",
            r###"PREDICT LIST_DISTINCT(orders.product_id, 0, 30, days) RANK TOP 3 FOR EACH users.user_id"###,
            &[
                "PQL_MODEL_SINGLE_TABLE_ONLY",
                "PQL_MODEL_CANNOT_RANK_FOREIGN_KEY",
            ],
        ),
        (
            "schema-1",
            "FINE_TUNING",
            r###"PREDICT COUNT(orders.* WHERE orders.quantity > true, 0, 30, days) FOR EACH users.user_id"###,
            &["PQL_TYPE_CONSTANT_MISMATCH", "PQL_TYPE_BOOL_COMPARISON"],
        ),
        (
            "schema-1",
            "RELATIONAL",
            r###"PREDICT COUNT(orders.*, 0, 30, days) > 0 FOR EACH users.user_id WHERE LIST_DISTINCT(orders.product_id, -30, 0, days) = 'x'"###,
            &[
                "PQL_MODEL_UNSUPPORTED_COLUMN_OPERATION",
                "PQL_MODEL_AGGREGATION_NEEDS_FOREIGN_KEY",
            ],
        ),
        ("schema-1", "RELATIONAL", r###" "###, &["PQL_PARSE_SYNTAX"]),
        (
            "schema-1",
            "RELATIONAL",
            r###"PREDICT users.bio FOR EACH users.user_id"###,
            &["PQL_TYPE_TARGET_STYPE_NOT_PREDICTABLE"],
        ),
        (
            "schema-1",
            "RELATIONAL",
            r###"PREDICT users.nope FOR EACH users.user_id"###,
            &["PQL_SCHEMA_UNKNOWN_COLUMN"],
        ),
        (
            "schema-1",
            "RELATIONAL",
            r###"PREDICT users.user_id FOR EACH users.user_id"###,
            &["PQL_MODEL_NO_KEY_COLUMN_REFERENCE"],
        ),
        (
            "schema-3",
            "TIMESERIES",
            r###"PREDICT people.age FOR EACH people.person_id"###,
            &["PQL_MODEL_NO_STATIC_TARGET"],
        ),
        (
            "schema-1",
            "RELATIONAL",
            r###"PREDICT orders.amount FOR EACH users.user_id"###,
            &["PQL_JOIN_AMBIGUOUS_TARGET_VALUE"],
        ),
        (
            "schema-1",
            "RELATIONAL",
            r###"PREDICT COUNT(orders.*) FOR EACH users.user_id"###,
            &["PQL_TIME_AGGREGATION_NEEDS_RANGE"],
        ),
        (
            "schema-1",
            "FINE_TUNING",
            r###"PREDICT users.user_id = 1 FOR EACH users.user_id"###,
            &["PQL_TARGET_IS_ENTITY_KEY"],
        ),
        (
            "schema-3",
            "TABULAR",
            r###"PREDICT people.age > 40 FOR EACH people.person_id"###,
            &["PQL_MODEL_NO_DERIVED_TARGET"],
        ),
        (
            "schema-1",
            "FINE_TUNING",
            r###"PREDICT products.tags FOR EACH products.product_id"###,
            &["PQL_TASK_PROBLEM_TYPE_REQUIRED"],
        ),
        (
            "schema-1",
            "RELATIONAL",
            r###"PREDICT COUNT(orders.*, 0, 30, days) FOR users.user_id"###,
            &["PQL_MODEL_FOR_REQUIRES_ENTITY_IDS"],
        ),
        (
            "schema-3",
            "TABULAR",
            r###"PREDICT people.churn RANK TOP 2 FOR EACH people.person_id"###,
            &["PQL_TASK_PROBLEM_TYPE_NEEDS_MULTICATEGORICAL"],
        ),
        (
            "schema-1",
            "RELATIONAL",
            r###"PREDICT COUNT(order.*, 0, 30, days) FOR EACH users.user_id"###,
            &["PQL_SCHEMA_UNKNOWN_TABLE"],
        ),
        (
            "schema-1",
            "RELATIONAL",
            r###"PREDICT COUNT(orders.*, 0, 30, days) FOR users.user_id > 1"###,
            &["PQL_MODEL_ENTITY_ID_OPERATOR"],
        ),
        (
            "schema-1",
            "TABULAR",
            r###"PREDICT COUNT(orders.*, 0, 30, days) FOR EACH users.user_id"###,
            &["PQL_MODEL_NO_TEMPORAL_TARGET"],
        ),
        (
            "schema-1",
            "RELATIONAL",
            r###"PREDICT products.tags CLASSIFY FOR EACH products.product_id"###,
            &["PQL_MODEL_NO_STATIC_MULTILABEL"],
        ),
        (
            "schema-1",
            "RELATIONAL",
            r###"PREDICT COUNT(orders.*, -30, 0, days) FOR EACH users.user_id"###,
            &["PQL_TIME_PAST_NOT_ALLOWED"],
        ),
        (
            "schema-4",
            "FINE_TUNING",
            r###"PREDICT COUNT(trips.*, 0, 7, days) FOR EACH people.person_id"###,
            &["PQL_JOIN_AMBIGUOUS_FOREIGN_KEY"],
        ),
        (
            "schema-1",
            "RELATIONAL",
            r###"PREDICT COUNT(orders.*, 0, 30, days) FOR EACH orders.order_id"###,
            &["PQL_JOIN_MISSING_FOREIGN_KEY"],
        ),
        (
            "schema-1",
            "RELATIONAL",
            r###"PREDICT SUM(orders.status, 0, 30, days) FOR EACH users.user_id"###,
            &["PQL_TYPE_AGGREGATION_NEEDS_NUMERIC"],
        ),
        (
            "schema-1",
            "RELATIONAL",
            r###"PREDICT users.age FORECAST 3 TIMEFRAMES FOR EACH users.user_id"###,
            &["PQL_TASK_FORECAST_REQUIRES_TEMPORAL_TARGET"],
        ),
        (
            "schema-3",
            "RELATIONAL",
            r###"PREDICT COUNT(people.*, 0, 30, days) FOR EACH people.person_id"###,
            &["PQL_TIME_TABLE_HAS_NO_TIME_COLUMN"],
        ),
        (
            "schema-1",
            "RELATIONAL",
            r###"PREDICT COUNT(orders.*, 0, 30, days) FOR EACH users.user_id = 1"###,
            &["PQL_MODEL_ENTITY_IDS_WITH_FOR_EACH"],
        ),
        (
            "schema-1",
            "RELATIONAL",
            r###"PREDICT SUM(orders.product_id, 0, 30, days) FOR EACH users.user_id"###,
            &["PQL_TYPE_AGGREGATION_ON_KEY_COLUMN"],
        ),
        (
            "schema-1",
            "RELATIONAL",
            r###"PREDICT LAST(orders.product_id, 0, 30, days) FOR EACH users.user_id"###,
            &["PQL_TARGET_AGGREGATION_ON_FOREIGN_KEY"],
        ),
        (
            "schema-3",
            "TABULAR",
            r###"PREDICT people.churn FOR EACH people.person_id WHERE people.age > 30"###,
            &["PQL_MODEL_NO_ENTITY_FILTER"],
        ),
        (
            "schema-6",
            "TIMESERIES",
            r###"PREDICT COUNT(employees.*, 0, 30, days) FOR EACH employees.employee_id"###,
            &["PQL_MODEL_NO_RELATIONSHIPS"],
        ),
        (
            "schema-1",
            "FINE_TUNING",
            r###"PREDICT LIST_DISTINCT(orders.status) RANK TOP 3 FOR EACH users.user_id"###,
            &["PQL_TIME_LIST_DISTINCT_NEEDS_RANGE_IN_TARGET"],
        ),
        (
            "schema-1",
            "FINE_TUNING",
            r###"PREDICT LIST_DISTINCT(orders.amount) RANK TOP 3 FOR EACH users.user_id"###,
            &["PQL_TIME_LIST_DISTINCT_NEEDS_RANGE_OR_FOREIGN_KEY"],
        ),
        (
            "schema-1",
            "RELATIONAL",
            r###"PREDICT LIST_DISTINCT(orders.amount, 0, 30, days) FOR EACH users.user_id"###,
            &["PQL_MODEL_NO_MULTILABEL"],
        ),
        (
            "schema-1",
            "FINE_TUNING",
            r###"PREDICT LIST_DISTINCT(orders.product_id) CLASSIFY FOR EACH users.user_id"###,
            &["PQL_TIME_LIST_DISTINCT_NEEDS_RANGE_FOR_RANK"],
        ),
        (
            "schema-7",
            "RELATIONAL",
            r###"PREDICT LIST_DISTINCT(likes.item_id) RANK TOP 3 FOR EACH members.member_id"###,
            &["PQL_MODEL_NO_STATIC_LINK_PREDICTION"],
        ),
        (
            "schema-1",
            "RELATIONAL",
            r###"PREDICT COUNT(orders.* WHERE users.age > 30, 0, 30, days) FOR EACH users.user_id"###,
            &["PQL_MODEL_NO_IMPLICIT_JOIN"],
        ),
        (
            "schema-1",
            "FINE_TUNING",
            r###"PREDICT users.age FOR EACH users.user_id WHERE COUNT(orders.*, -30, 0, days) > 0"###,
            &["PQL_TIME_TEMPORAL_AGGREGATION_NOT_ALLOWED"],
        ),
        (
            "schema-1",
            "RELATIONAL",
            r###"PREDICT LIST_DISTINCT(orders.product_id, 0, 7, days) FOR EACH products.product_id"###,
            &["PQL_TARGET_EMPTY"],
        ),
        (
            "schema-1",
            "FINE_TUNING",
            r###"PREDICT LIST_DISTINCT(orders.product_id, 0, 30, days) RANK FOR EACH users.user_id"###,
            &["PQL_TASK_RANK_REQUIRES_TOP_K"],
        ),
        (
            "schema-1",
            "FINE_TUNING",
            r###"PREDICT COUNT(orders.*, 0, 30, days) > 0 AND users.age > 3 FOR EACH users.user_id"###,
            &["PQL_TIME_MIXED_STATIC_AND_TEMPORAL"],
        ),
        (
            "schema-1",
            "RELATIONAL",
            r###"PREDICT LIST_DISTINCT(orders.product_id, 0, 30, days) TOP 3 FOR EACH users.user_id"###,
            &["PQL_TASK_TOP_K_REQUIRES_RANK"],
        ),
        (
            "schema-1",
            "RELATIONAL",
            r###"PREDICT COUNT(orders.*, 0, 30, days) FOR EACH users.user_id WHERE orders.amount > 1"###,
            &["PQL_JOIN_EVENT_COLUMN_IN_ENTITY_FILTER"],
        ),
        (
            "schema-1",
            "RELATIONAL",
            r###"PREDICT COUNT(orders.*, 0, 7, days) > 0 FORECAST 3 TIMEFRAMES FOR EACH users.user_id"###,
            &["PQL_TASK_FORECAST_REQUIRES_NUMERICAL_TARGET"],
        ),
        (
            "schema-1",
            "FINE_TUNING",
            r###"PREDICT LIST_DISTINCT(orders.order_id, 0, 30, days) RANK TOP 3 FOR EACH users.user_id"###,
            &["PQL_TARGET_AGGREGATION_ON_PRIMARY_KEY"],
        ),
        (
            "schema-1",
            "RELATIONAL",
            r###"PREDICT SUM(orders.quantity, 0, 30, days) FORECAST 0 TIMEFRAMES FOR EACH users.user_id"###,
            &["PQL_TASK_FORECAST_NOT_POSITIVE"],
        ),
        (
            "schema-1",
            "FINE_TUNING",
            r###"PREDICT COUNT(orders.* WHERE orders.quantity IN 1, 0, 30, days) FOR EACH users.user_id"###,
            &["PQL_TYPE_OPERATOR_NEEDS_ARRAY"],
        ),
        (
            "schema-1",
            "RELATIONAL",
            r###"PREDICT LIST_DISTINCT(orders.product_id, 0, 30, days) RANK TOP 0 FOR EACH users.user_id"###,
            &["PQL_TASK_TOP_K_NOT_POSITIVE"],
        ),
        (
            "schema-1",
            "RELATIONAL",
            r###"PREDICT LIST_DISTINCT(orders.product_id, 0, 30, days) RANK TOP 21 FOR EACH users.user_id"###,
            &["PQL_MODEL_TOP_K_TOO_LARGE"],
        ),
        (
            "schema-1",
            "FINE_TUNING",
            r###"PREDICT COUNT(orders.* WHERE orders.quantity > NULL, 0, 30, days) FOR EACH users.user_id"###,
            &["PQL_TYPE_NULL_COMPARISON"],
        ),
        (
            "schema-1",
            "FINE_TUNING",
            r###"PREDICT SUM(orders.quantity, 0, 30, days) FORECAST 10001 TIMEFRAMES FOR EACH users.user_id"###,
            &["PQL_TASK_FORECAST_TOO_LARGE"],
        ),
        (
            "schema-1",
            "FINE_TUNING",
            r###"PREDICT COUNT(orders.* WHERE orders.quantity = (1, 2), 0, 30, days) FOR EACH users.user_id"###,
            &["PQL_TYPE_OPERATOR_ON_ARRAY"],
        ),
        (
            "schema-1",
            "RELATIONAL",
            r###"PREDICT COUNT(orders.* WHERE orders.status LIKE 'ship%', 0, 30, days) FOR EACH users.user_id"###,
            &["PQL_MODEL_NO_STRING_COMPARISON"],
        ),
        (
            "schema-1",
            "FINE_TUNING",
            r###"PREDICT COUNT(orders.* WHERE orders.amount CONTAINS 'x', 0, 30, days) FOR EACH users.user_id"###,
            &["PQL_TYPE_OPERATOR_NEEDS_STRING"],
        ),
        (
            "schema-1",
            "FINE_TUNING",
            r###"PREDICT COUNT(orders.* WHERE orders.quantity IS IN (1, 2), 0, 30, days) FOR EACH users.user_id"###,
            &["PQL_TYPE_OPERATOR_DEPRECATED"],
        ),
        (
            "schema-1",
            "FINE_TUNING",
            r###"PREDICT COUNT(orders.* WHERE orders.quantity IN ('a', 'b'), 0, 30, days) FOR EACH users.user_id"###,
            &["PQL_TYPE_ARRAY_ELEMENT_MISMATCH"],
        ),
        (
            "schema-1",
            "RELATIONAL",
            r###"PREDICT COUNT(orders.* WHERE orders.ordered_at = 2023-02-29, 0, 30, days) FOR EACH users.user_id"###,
            &["PQL_VALUE_INVALID_LITERAL"],
        ),
        (
            "schema-1",
            "RELATIONAL",
            r###"PREDICT COUNT(orders.*, 0, 30, days) FOR EACH users.user_id WHERE COUNT(orders.*, 0, 30, days) > 0"###,
            &["PQL_TIME_FUTURE_NOT_ALLOWED"],
        ),
        (
            "schema-1",
            "FINE_TUNING",
            r###"PREDICT COUNT(orders.* WHERE orders.quantity > 9223372036854775808, 0, 30, days) FOR EACH users.user_id"###,
            &["PQL_VALUE_INT_OUT_OF_RANGE"],
        ),
        (
            "schema-5",
            "RELATIONAL",
            r###"PREDICT COUNT(orders.* WHERE COUNT(items.*, -7, 0, days) > 0, 0, 30, days) FOR EACH customers.customer_id"###,
            &["PQL_MODEL_NO_NESTED_AGGREGATION"],
        ),
        (
            "schema-1",
            "FINE_TUNING",
            r###"PREDICT LIST_DISTINCT(orders.product_id WHERE users.age > 3, 0, 30, days) RANK TOP 3 FOR EACH users.user_id"###,
            &["PQL_TARGET_FILTER_REFERENCES_ENTITY"],
        ),
        (
            "schema-1",
            "RELATIONAL",
            many_ids.as_str(),
            &["PQL_MODEL_ENTITY_ID_LIST_TOO_LONG"],
        ),
    ];
    for (which, model, text, expected) in cases {
        let got = resolve(text, &schema(which), &profile(model), &facts(), &NoZones).unwrap();
        assert!(got.query.is_none(), "{text}");
        let codes: Vec<_> = got
            .diagnostics
            .iter()
            .filter(|d| d.severity == Severity::Error)
            .map(|d| d.code.as_str())
            .collect();
        assert_eq!(&codes, expected, "{text}");
    }
}

#[test]
fn selected_errors_point_at_what_they_name() {
    use sdm_contract::ops::Span;
    let at = |start_col, end_col| {
        Some(Span {
            start_row: 1,
            start_col,
            end_row: 1,
            end_col,
        })
    };
    type Errors = Vec<(&'static str, Option<Span>)>;
    let cases: Vec<(&str, &str, &str, Errors)> = vec![
        (
            "schema-1",
            "RELATIONAL",
            "PREDICT users.nope FOR EACH users.user_id",
            vec![("PQL_SCHEMA_UNKNOWN_COLUMN", at(8, 8))],
        ),
        (
            "schema-1",
            "RELATIONAL",
            "PREDICT COUNT(order.*, 0, 30, days) FOR EACH users.user_id",
            vec![("PQL_SCHEMA_UNKNOWN_TABLE", at(14, 14))],
        ),
        (
            "schema-1",
            "RELATIONAL",
            "PREDICT COUNT(orders.*, 0, 30, days) FOR EACH users.id",
            vec![("PQL_SCHEMA_UNKNOWN_COLUMN", at(46, 46))],
        ),
        (
            "schema-1",
            "RELATIONAL",
            "PREDICT SUM(orders.status, 0, 30, days) FOR EACH users.user_id",
            vec![("PQL_TYPE_AGGREGATION_NEEDS_NUMERIC", at(8, 38))],
        ),
        (
            "schema-1",
            "RELATIONAL",
            "PREDICT SUM(orders.product_id, 0, 30, days) FOR EACH users.user_id",
            vec![("PQL_TYPE_AGGREGATION_ON_KEY_COLUMN", at(8, 42))],
        ),
        (
            "schema-1",
            "RELATIONAL",
            "PREDICT COUNT(orders.* WHERE orders.amount > 'ten', 0, 30, days) FOR EACH users.user_id",
            vec![("PQL_TYPE_CONSTANT_MISMATCH", at(29, 45))],
        ),
        (
            "schema-1",
            "FINE_TUNING",
            "PREDICT COUNT(orders.* WHERE orders.quantity > true, 0, 30, days) FOR EACH users.user_id",
            vec![
                ("PQL_TYPE_CONSTANT_MISMATCH", at(29, 47)),
                ("PQL_TYPE_BOOL_COMPARISON", at(29, 47)),
            ],
        ),
        (
            "schema-1",
            "RELATIONAL",
            "PREDICT SUM(orders.quantity, 0, 30, days) FORECAST 0 TIMEFRAMES FOR EACH users.user_id",
            vec![("PQL_TASK_FORECAST_NOT_POSITIVE", None)],
        ),
        (
            "schema-1",
            "RELATIONAL",
            "PREDICT LIST_DISTINCT(orders.product_id, 0, 30, days) RANK TOP 0 FOR EACH users.user_id",
            vec![("PQL_TASK_TOP_K_NOT_POSITIVE", None)],
        ),
        (
            "schema-1",
            "FINE_TUNING",
            "PREDICT LIST_DISTINCT(orders.product_id, 0, 30, days) RANK FOR EACH users.user_id",
            vec![("PQL_TASK_RANK_REQUIRES_TOP_K", at(68, 68))],
        ),
        (
            "schema-1",
            "FINE_TUNING",
            "PREDICT products.tags FOR EACH products.product_id",
            vec![("PQL_TASK_PROBLEM_TYPE_REQUIRED", at(31, 31))],
        ),
    ];
    for (which, model, text, expected) in cases {
        let got = resolve(text, &schema(which), &profile(model), &facts(), &NoZones).unwrap();
        let errors: Vec<_> = got
            .diagnostics
            .iter()
            .filter(|d| d.severity == Severity::Error)
            .map(|d| (d.code.as_str(), d.span))
            .collect();
        assert_eq!(errors, expected, "{text}");
    }
}

fn nodes(node: &sdm_contract::pql::PqlNode) -> Vec<&sdm_contract::pql::NodeKind> {
    use sdm_contract::pql::NodeKind::*;
    let mut out = vec![&node.kind];
    match &node.kind {
        Join { target, .. } | Aggregation { target, .. } => out.extend(nodes(target)),
        Filter { target, condition } => {
            out.extend(nodes(target));
            out.extend(nodes(condition));
        }
        Condition { target, value, .. } => {
            out.extend(nodes(target));
            out.extend(nodes(value));
        }
        Logical { left, right, .. } => {
            out.extend(nodes(left));
            if let Some(right) = right {
                out.extend(nodes(right));
            }
        }
        _ => {}
    }
    out
}

#[test]
fn accepted_queries_resolve_columns_windows_and_model_options() {
    use sdm_contract::pql::{
        Aggregation, EntityClause, NodeKind, PqlLiteral, ProblemType, QueryMode, TimeUnit,
    };
    let accepted = |text: &str, which: &str, model: &str| {
        resolve(text, &schema(which), &profile(model), &facts(), &NoZones)
            .unwrap()
            .query
            .expect(text)
    };
    for (unit, expected_unit, end) in [("days", TimeUnit::Days, 30), ("weeks", TimeUnit::Weeks, 2)]
    {
        let query = accepted(
            &format!("PREDICT SUM(orders.quantity, 0, {end}, {unit}) FOR EACH users.user_id"),
            "schema-1",
            "RELATIONAL",
        );
        assert!(
            matches!(&query.entity.kind, NodeKind::Column { column } if column.table == "users" && column.column == "user_id")
        );
        let parts = nodes(&query.target);
        assert!(parts.iter().any(|n| matches!(n, NodeKind::Aggregation { aggregation: Aggregation::Sum, window: Some(w), .. } if w.start == Some(0) && w.end == end && w.unit == expected_unit)));
        assert!(parts.iter().any(|n| matches!(n, NodeKind::Column { column } if column.table == "orders" && column.column == "quantity")));
        assert!(parts.iter().any(|n| matches!(n, NodeKind::Join { lhs_key, rhs_key, .. } if lhs_key.column == "user_id" && rhs_key.column == "user_id")));
    }
    let quoted = accepted(
        "PREDICT SUM(`Order Lines`.`Unit Price`, 0, 30, days) FOR EACH Customers.`Customer ID`",
        "schema-2",
        "RELATIONAL",
    );
    assert!(
        matches!(&quoted.entity.kind, NodeKind::Column { column } if column.table == "Customers" && column.column == "Customer ID")
    );
    assert!(nodes(&quoted.target).iter().any(|n| matches!(n, NodeKind::Column { column } if column.table == "Order Lines" && column.column == "Unit Price")));
    let ranked = accepted(
        "PREDICT LIST_DISTINCT(orders.product_id, 0, 30, days) RANK TOP 7 FOR EACH users.user_id",
        "schema-1",
        "RELATIONAL",
    );
    assert_eq!(ranked.problem_type, Some(ProblemType::Rank));
    assert_eq!(ranked.top_k.as_deref(), Some("7"));
    assert!(nodes(&ranked.target).iter().any(|n| matches!(n, NodeKind::Column { column } if column.table == "orders" && column.column == "product_id")));
    let forecast = accepted(
        "PREDICT SUM(orders.quantity, 0, 1, days) FORECAST 3 TIMEFRAMES FOR EACH users.user_id",
        "schema-1",
        "RELATIONAL",
    );
    assert_eq!(forecast.num_forecasts, 3);
    let assuming = accepted(
        "PREDICT COUNT(orders.*, 0, 30, days) FOR EACH users.user_id ASSUMING COUNT(reviews.*, 0, 30, days) > 0",
        "schema-1",
        "RELATIONAL",
    );
    let parts = nodes(assuming.assuming.as_ref().unwrap());
    assert!(parts.iter().any(|n| matches!(n, NodeKind::Column { column } if column.table == "reviews" && column.column == "*")));
    assert!(parts.iter().any(
        |n| matches!(n, NodeKind::Constant { literal: PqlLiteral::Int { value } } if value == "0")
    ));
    for (which, model, target, entity, mode) in [
        (
            "schema-3",
            "TABULAR",
            "people.churn",
            "people.person_id",
            QueryMode::Inference,
        ),
        (
            "schema-1",
            "ANOMALY",
            "users.country",
            "users.user_id",
            QueryMode::Inference,
        ),
        (
            "schema-1",
            "FINE_TUNING",
            "users.age",
            "users.user_id",
            QueryMode::Training,
        ),
    ] {
        let q = accepted(&format!("PREDICT {target} FOR EACH {entity}"), which, model);
        assert_eq!(q.mode, mode);
        assert_eq!(q.entity_clause, EntityClause::ForEach);
        let (table, name) = target.split_once('.').unwrap();
        assert!(
            matches!(q.target.kind, NodeKind::Column { column } if column.table == table && column.column == name)
        );
    }
    let filtered = accepted(
        "PREDICT COUNT(orders.* WHERE orders.status = 'shipped', 0, 30, days) FOR EACH users.user_id",
        "schema-1",
        "RELATIONAL",
    );
    let NodeKind::Join { target, .. } = &filtered.target.kind else {
        panic!("filtered count must be joined to the entity");
    };
    let NodeKind::Aggregation {
        aggregation: Aggregation::Count,
        target,
        window: Some(window),
        ..
    } = &target.kind
    else {
        panic!("joined target must be a windowed count");
    };
    assert_eq!(window.start, Some(0));
    assert_eq!(window.end, 30);
    assert_eq!(window.unit, TimeUnit::Days);
    let NodeKind::Filter { target, condition } = &target.kind else {
        panic!("filter must apply to the count's input");
    };
    assert!(
        matches!(&target.kind, NodeKind::Column { column } if column.table == "orders" && column.column == "*")
    );
    let NodeKind::Condition {
        target,
        op,
        value,
        from_like,
    } = &condition.kind
    else {
        panic!("count's filter must be a comparison");
    };
    assert_eq!(*op, sdm_contract::pql::ConditionOp::Eq);
    assert!(!from_like);
    assert!(
        matches!(&target.kind, NodeKind::Column { column } if column.table == "orders" && column.column == "status")
    );
    assert!(
        matches!(&value.kind, NodeKind::Constant { literal: PqlLiteral::String { value } } if value == "shipped")
    );
}
