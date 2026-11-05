// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The validated-query format round-trips through JSON, and unknown fields
//! and tags are refused.

use sdm_contract::ValidatedQuery;
use sdm_contract::ops::{ColumnRef, Dtype, Span, Stype};
use sdm_contract::pql::{
    Aggregation, ConditionOp, EntityClause, LogicalOp, NodeKind, NodeType, PqlLiteral, PqlNode,
    ProblemType, QueryMode, TimeUnit, Window,
};

fn span(start: u32, end: u32) -> Option<Span> {
    Some(Span {
        start_row: 1,
        start_col: start,
        end_row: 1,
        end_col: end,
    })
}

fn node(dtype: Option<NodeType>, kind: NodeKind) -> PqlNode {
    PqlNode {
        dtype,
        stype: None,
        range: None,
        span: span(0, 0),
        kind,
    }
}

fn column(table: &str, name: &str) -> PqlNode {
    PqlNode {
        stype: Some(Stype::Numerical),
        ..node(
            Some(NodeType::Value(Dtype::Float)),
            NodeKind::Column {
                column: ColumnRef {
                    table: table.into(),
                    column: name.into(),
                },
            },
        )
    }
}

fn constant(literal: PqlLiteral) -> PqlNode {
    node(None, NodeKind::Constant { literal })
}

/// A query that uses every kind of node and literal.
fn sample() -> ValidatedQuery {
    let window = Window {
        start: None,
        end: 30,
        unit: TimeUnit::Days,
    };
    let condition = |op, literal| {
        Box::new(node(
            Some(NodeType::Value(Dtype::Bool)),
            NodeKind::Condition {
                target: Box::new(column("orders", "price")),
                op,
                value: Box::new(constant(literal)),
                from_like: false,
            },
        ))
    };
    let both = node(
        Some(NodeType::Value(Dtype::Bool)),
        NodeKind::Logical {
            op: LogicalOp::And,
            left: condition(
                ConditionOp::Gt,
                PqlLiteral::Int {
                    value: "99999999999999999999".into(),
                },
            ),
            right: Some(Box::new(node(
                Some(NodeType::Value(Dtype::Bool)),
                NodeKind::Logical {
                    op: LogicalOp::Not,
                    left: condition(ConditionOp::Eq, PqlLiteral::Null),
                    right: None,
                },
            ))),
        },
    );
    let aggregation = node(
        Some(NodeType::Value(Dtype::Float)),
        NodeKind::Aggregation {
            aggregation: Aggregation::Sum,
            target: Box::new(node(
                None,
                NodeKind::Filter {
                    target: Box::new(column("orders", "price")),
                    condition: Box::new(both),
                },
            )),
            window: Some(window),
            group_by: Some(ColumnRef {
                table: "orders".into(),
                column: "user_id".into(),
            }),
            time_column: Some(ColumnRef {
                table: "orders".into(),
                column: "ts".into(),
            }),
        },
    );
    let ids = node(
        Some(NodeType::Value(Dtype::Bool)),
        NodeKind::Condition {
            target: Box::new(column("users", "user_id")),
            op: ConditionOp::In,
            value: Box::new(node(
                Some(NodeType::Array(Dtype::String)),
                NodeKind::Constant {
                    literal: PqlLiteral::Array {
                        items: vec![
                            constant(PqlLiteral::String {
                                value: "é'".into()
                            }),
                            constant(PqlLiteral::Time {
                                text: "1/2/2024".into(),
                                value: "2024-01-02T00:00:00".into(),
                                offset: None,
                            }),
                            constant(PqlLiteral::Float {
                                text: "1.50".into(),
                            }),
                            constant(PqlLiteral::Bool { value: true }),
                        ],
                    },
                },
            )),
            from_like: true,
        },
    );
    ValidatedQuery {
        entity: column("users", "user_id"),
        target: PqlNode {
            range: Some(window),
            ..node(
                Some(NodeType::Value(Dtype::Float)),
                NodeKind::Join {
                    target: Box::new(aggregation),
                    lhs_key: ColumnRef {
                        table: "users".into(),
                        column: "user_id".into(),
                    },
                    rhs_key: ColumnRef {
                        table: "orders".into(),
                        column: "user_id".into(),
                    },
                },
            )
        },
        assuming: None,
        entity_ids: Some(ids),
        problem_type: Some(ProblemType::Rank),
        top_k: Some("10".into()),
        num_forecasts: 1,
        entity_clause: EntityClause::For,
        mode: QueryMode::Inference,
        evaluate: false,
        explain: true,
    }
}

#[test]
fn a_validated_query_round_trips() {
    let query = sample();
    let text = serde_json::to_string(&query).unwrap();
    assert_eq!(
        serde_json::from_str::<ValidatedQuery>(&text).unwrap(),
        query
    );
}

#[test]
fn unknown_fields_and_tags_are_refused() {
    let good = serde_json::to_value(sample()).unwrap();
    let mut extra = good.clone();
    extra["entity"]["kind"]["alias"] = "u".into();
    let mut tag = good.clone();
    tag["entity"]["kind"]["node"] = "python_object".into();
    let mut literal = good.clone();
    literal["entity_ids"]["kind"]["value"]["kind"]["literal"]["type"] = "decimal".into();
    let mut top = good;
    top["rfm_query"] = true.into();
    for bad in [extra, tag, literal, top] {
        assert!(
            serde_json::from_value::<ValidatedQuery>(bad.clone()).is_err(),
            "{bad} was accepted"
        );
    }
}
