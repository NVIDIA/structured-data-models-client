// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Grammar distinctions, diagnostic positions and logical operator binding.
mod pql_support;
use pql_support::{NoZones, facts, profile, schema};
use sdm_contract::ops::{Severity, Span};
use sdm_contract::pql::{LogicalOp, NodeKind, PqlLiteral};
use sdm_core::pql::{Validation, resolve};
use sdm_pql::lexer::{Kind, tokenize};
const B: &str = "PREDICT COUNT(orders.*, 0, 30, days) FOR EACH users.user_id";
fn read(query: &str) -> Validation {
    resolve(
        query,
        &schema("schema-1"),
        &profile("RELATIONAL"),
        &facts(),
        &NoZones,
    )
    .unwrap()
}
fn refuse(query: &str, code: &str) {
    let got = read(query);
    assert!(got.query.is_none(), "{query}");
    let errors: Vec<_> = got
        .diagnostics
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .collect();
    assert_eq!(errors.len(), 1, "{query}: {errors:?}");
    assert_eq!(errors[0].code, code, "{query}");
    if code == "PQL_PARSE_SYNTAX" {
        assert_eq!(errors[0].span, None);
    }
}

#[test]
fn keyword_spelling_and_multiword_tokens_are_deliberate() {
    use Kind::*;
    for (words, kind) in [
        ("predict PREDICT", Predict),
        ("count COUNT", Aggr),
        ("null NULL", Null),
        ("weeks WEEKS", TimeUnit),
        ("true True TRUE false False FALSE", Bool),
        ("IN in", RelOp),
        ("Predict Count Null Days tRUE Like", Id),
    ] {
        for word in words.split(' ') {
            let tokens = tokenize(&word.chars().collect::<Vec<_>>()).unwrap();
            assert_eq!(tokens.len(), 1, "{word}");
            assert_eq!(tokens[0].kind, kind, "{word}");
        }
    }
    for (text, expected) in [
        ("IS NOT", vec![RelOp]),
        ("is not", vec![RelOp]),
        ("IS  NOT", vec![RelOp, Not]),
        ("IS not", vec![RelOp, Not]),
        ("IS\tNOT", vec![RelOp, Not]),
        ("IS\nNOT", vec![RelOp, Not]),
    ] {
        let tokens = tokenize(&text.chars().collect::<Vec<_>>()).unwrap();
        assert_eq!(
            tokens.iter().map(|t| t.kind).collect::<Vec<_>>(),
            expected,
            "{text:?}"
        );
    }
}

#[test]
fn syntax_and_literal_failures_keep_distinct_diagnostics() {
    for query in [
        String::new(),
        B.replace("PREDICT", "Predict"),
        B.replace("COUNT", "Count"),
        B.replace("days", "Days"),
        B.replace("COUNT", "countx"),
        B.replace("0, 30", "30, 0"),
        "PREDICT orders.amount.c FOR EACH users.user_id".into(),
        "PREDICT COUNT(orders.*, 0, 30, days)".into(),
        format!("{B} trailing"),
    ] {
        refuse(&query, "PQL_PARSE_SYNTAX");
    }
    for condition in [
        "users.age IS Null",
        r"users.country = 'a\qb'",
        "users.country NOT LIKE 'x'",
        "users.country Like '%US%'",
        "users.age IS  NOT NULL",
        "users.age IS not NULL",
        "users.age IN (1)",
        "users.age IN (1,)",
        "users.age IN ((1), 2)",
        "users.vip = tRUE",
        "orders.amount > --3",
        "users.country = 'unclosed",
    ] {
        refuse(&format!("{B} WHERE {condition}"), "PQL_PARSE_SYNTAX");
    }
    for (condition, code) in [
        (
            "users.country IS IN ('US', 'FR')",
            "PQL_TYPE_OPERATOR_DEPRECATED",
        ),
        (
            "users.country LIKE '%US%'",
            "PQL_MODEL_NO_STRING_COMPARISON",
        ),
        ("orders.amount > --3.5", "PQL_VALUE_INVALID_LITERAL"),
        ("users.signup > 2024.01", "PQL_TYPE_CONSTANT_MISMATCH"),
    ] {
        refuse(&format!("{B} WHERE {condition}"), code);
    }
    for condition in [
        "users.age IS NULL",
        "users.age IN (1, 2)",
        "users.country IN ('US', 'FR')",
    ] {
        assert!(read(&format!("{B} WHERE {condition}")).query.is_some());
    }
}

#[test]
fn whitespace_comments_and_codepoint_spans() {
    for space in [
        '\t', '\n', '\r', '\u{b}', '\u{c}', '\u{1c}', '\u{1d}', '\u{1e}', '\u{1f}', '\u{85}',
        '\u{a0}', '\u{1680}', '\u{2000}', '\u{2001}', '\u{2002}', '\u{2003}', '\u{2004}',
        '\u{2005}', '\u{2006}', '\u{2007}', '\u{2008}', '\u{2009}', '\u{200a}', '\u{2028}',
        '\u{2029}', '\u{202f}', '\u{205f}', '\u{3000}',
    ] {
        assert!(
            read(&B.replacen(' ', &space.to_string(), 1))
                .query
                .is_some(),
            "{space:?}"
        );
    }
    for space in ['\u{200b}', '\u{180e}', '\u{feff}'] {
        refuse(&B.replacen(' ', &space.to_string(), 1), "PQL_PARSE_SYNTAX");
    }
    for newline in ["\n", "\r"] {
        assert!(
            read(&B.replace(" FOR EACH", &format!(" # comment{newline}FOR EACH")))
                .query
                .is_some()
        );
    }
    refuse(&B.replace(" FOR EACH", " # FOR EACH"), "PQL_PARSE_SYNTAX");
    for (text, col) in [("ab", 91), ("😀😀", 91), ("e\u{301}", 91), ("é", 90)] {
        let got = read(&format!(
            "{B} WHERE users.country = '{text}' AND users.nope = 1"
        ));
        let error = got
            .diagnostics
            .iter()
            .find(|d| d.code == "PQL_SCHEMA_UNKNOWN_COLUMN")
            .unwrap();
        assert_eq!(
            error.span,
            Some(Span {
                start_row: 1,
                start_col: col,
                end_row: 1,
                end_col: col
            })
        );
    }
    for (newline, row, col) in [("\n", 2, 9), ("\r\n", 2, 9), ("\r", 1, 46)] {
        let got = read(&format!(
            "PREDICT COUNT(orders.*, 0, 30, days){newline}FOR EACH users.nope"
        ));
        assert_eq!(
            got.diagnostics[0].span,
            Some(Span {
                start_row: row,
                start_col: col,
                end_row: row,
                end_col: col
            })
        );
    }
    let got = read("PREDICT\nCOUNT(orders.*, 0, 30, days)\nFOR EACH users.user_id")
        .query
        .unwrap();
    assert_eq!(
        got.target.span,
        Some(Span {
            start_row: 2,
            start_col: 0,
            end_row: 2,
            end_col: 27
        })
    );
    let got = read("PREDICT SUM(orders.amount, 0, 30, days) > 100 FOR EACH users.user_id")
        .query
        .unwrap();
    assert_eq!(
        got.target.span,
        Some(Span {
            start_row: 1,
            start_col: 8,
            end_row: 1,
            end_col: 42
        })
    );
}

fn shape(node: &sdm_contract::pql::PqlNode) -> String {
    match &node.kind {
        NodeKind::Logical { op, left, right } => match op {
            LogicalOp::Not => format!("not({})", shape(left)),
            LogicalOp::And => format!("and({},{})", shape(left), shape(right.as_ref().unwrap())),
            LogicalOp::Or => format!("or({},{})", shape(left), shape(right.as_ref().unwrap())),
        },
        NodeKind::Condition { value, .. } => match &value.kind {
            NodeKind::Constant {
                literal: PqlLiteral::Int { value },
            } => value.clone(),
            _ => panic!("integer condition"),
        },
        _ => panic!("logical condition"),
    }
}

#[test]
fn parentheses_and_not_preserve_boolean_precedence() {
    let a = "COUNT(orders.*, 0, 30, days) > 0";
    let b = "COUNT(orders.*, 0, 30, days) > 1";
    let c = "COUNT(orders.*, 0, 7, days) > 2";
    for (condition, expected) in [
        (format!("{a} OR {b} AND NOT {c}"), "or(0,and(1,not(2)))"),
        (format!("({a} OR {b}) AND {c}"), "and(or(0,1),2)"),
        (format!("NOT ({a} OR {b})"), "not(or(0,1))"),
    ] {
        let got = read(&format!("PREDICT {condition} FOR EACH users.user_id"))
            .query
            .unwrap();
        assert_eq!(shape(&got.target), expected);
    }
    let got = read(&format!(
        "{B} WHERE users.age > 3 OR users.age < 1 AND NOT users.age = 2"
    ))
    .query
    .unwrap();
    let NodeKind::Filter { condition, .. } = &got.entity.kind else {
        panic!("filtered entity")
    };
    assert_eq!(shape(condition), "or(3,and(1,not(2)))");
}

#[test]
fn integer_limits_are_not_rounded_through_floats() {
    for literal in ["9223372036854775807", "-9223372036854775808"] {
        let got = read(&format!("{B} WHERE users.age > {literal}"))
            .query
            .unwrap();
        let NodeKind::Filter { condition, .. } = &got.entity.kind else {
            panic!("filtered entity")
        };
        assert_eq!(shape(condition), literal);
    }
    for literal in ["9223372036854775808", "-9223372036854775809"] {
        refuse(
            &format!("{B} WHERE users.age > {literal}"),
            "PQL_VALUE_INT_OUT_OF_RANGE",
        );
    }
    assert!(read(&B.replace("0, 30", "0, 2147483647")).query.is_some());
    for window in ["0, 2147483648", "-2147483649, 0"] {
        refuse(&B.replace("0, 30", window), "PQL_PARSE_SYNTAX");
    }
    refuse(
        &B.replace("0, 30", "-2147483648, 0"),
        "PQL_TIME_PAST_NOT_ALLOWED",
    );
    refuse(
        "PREDICT LIST_DISTINCT(orders.product_id, 0, 30, days) RANK TOP 2147483648 FOR EACH users.user_id",
        "PQL_MODEL_TOP_K_TOO_LARGE",
    );
}

#[test]
fn literals_keep_values_and_types_after_parsing() {
    for (condition, expected) in [
        (
            "users.age IN (1, 2)",
            vec![
                PqlLiteral::Int { value: "1".into() },
                PqlLiteral::Int { value: "2".into() },
            ],
        ),
        (
            "users.country IN ('US', 'FR')",
            vec![
                PqlLiteral::String { value: "US".into() },
                PqlLiteral::String { value: "FR".into() },
            ],
        ),
    ] {
        let got = read(&format!("{B} WHERE {condition}")).query.unwrap();
        let NodeKind::Filter { condition, .. } = got.entity.kind else {
            panic!("entity filter")
        };
        let NodeKind::Condition { value, .. } = condition.kind else {
            panic!("condition")
        };
        let NodeKind::Constant {
            literal: PqlLiteral::Array { items },
        } = value.kind
        else {
            panic!("array")
        };
        let values: Vec<_> = items
            .into_iter()
            .map(|item| match item.kind {
                NodeKind::Constant { literal } => literal,
                _ => panic!("constant"),
            })
            .collect();
        assert_eq!(values, expected);
    }
    for (condition, expected) in [
        ("users.age IS NULL", PqlLiteral::Null),
        (
            "users.country = 'a#b'",
            PqlLiteral::String {
                value: "a#b".into(),
            },
        ),
    ] {
        let got = read(&format!("{B} WHERE {condition}")).query.unwrap();
        let NodeKind::Filter { condition, .. } = got.entity.kind else {
            panic!("entity filter")
        };
        let NodeKind::Condition { value, .. } = condition.kind else {
            panic!("condition")
        };
        assert!(matches!(value.kind,NodeKind::Constant{literal} if literal==expected));
    }
    fn has_negative(node: &sdm_contract::pql::PqlNode) -> bool {
        match &node.kind {
            NodeKind::Constant {
                literal: PqlLiteral::Float { text },
            } => text == "-3.5",
            NodeKind::Join { target, .. } | NodeKind::Aggregation { target, .. } => {
                has_negative(target)
            }
            NodeKind::Filter { target, condition } => {
                has_negative(target) || has_negative(condition)
            }
            NodeKind::Condition { target, value, .. } => {
                has_negative(target) || has_negative(value)
            }
            _ => false,
        }
    }
    let got = read(
        "PREDICT COUNT(orders.* WHERE orders.amount > -3.5, 0, 30, days) FOR EACH users.user_id",
    )
    .query
    .unwrap();
    assert!(has_negative(&got.target));
}
