// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The view expressions' checks (T3.5, old-behaviour rows 61 to 68) against
//! `contract/view-expr-vectors.json`: every call the old client's own tests
//! made of `RE/view_expr.py` and of the Snowflake qualifier helpers of
//! `RE/graph.py`, with its answer, and each word of the old word tables
//! (the dialects' allow-lists, keywords, type names, session words) put to
//! the same checks. Each answer, and each reason's text, is the old
//! client's.

use sdm_data::view_expr::{self, ExprDialect, ViewKind};
use serde_json::Value as Json;

const VECTORS: &str = include_str!("../../../../contract/view-expr-vectors.json");

fn document() -> Json {
    let document: Json = serde_json::from_str(VECTORS).unwrap();
    assert_eq!(document["format"], "sdm-view-expr-vectors/1");
    document
}

fn dialect(value: &Json) -> Option<ExprDialect> {
    match value.as_str() {
        None => None,
        Some("databricks") => Some(ExprDialect::Databricks),
        Some("snowflake") => Some(ExprDialect::Snowflake),
        Some(other) => panic!("{other}"),
    }
}

fn texts(value: &Json) -> Vec<String> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|item| item.as_str().unwrap().to_owned())
        .collect()
}

/// Runs `check` over the vectors of `kind`, and fails listing each that
/// differs.
fn each(kind: &str, least: usize, check: impl Fn(&Json) -> Option<String>) {
    let document = document();
    let vectors = document[kind].as_array().unwrap();
    assert!(vectors.len() >= least, "{} {kind} vectors", vectors.len());
    let wrong: Vec<String> = vectors.iter().filter_map(check).collect();
    assert!(
        wrong.is_empty(),
        "{} of {} {kind} vectors differ:\n{}",
        wrong.len(),
        vectors.len(),
        wrong.join("\n")
    );
}

#[test]
fn unsafe_text_is_refused_as_the_old_client_refused_it() {
    each("unsafe", 400, |vector| {
        let expr = vector["expr"].as_str().unwrap();
        let reason = view_expr::unsafe_reason(expr, dialect(&vector["dialect"]));
        (reason.as_deref() != vector["reason"].as_str()).then(|| {
            format!(
                "{expr:?} ({}): {reason:?} where {}",
                vector["dialect"], vector["reason"]
            )
        })
    });
}

#[test]
fn calls_are_trusted_as_the_old_client_trusted_them() {
    each("calls", 10_000, |vector| {
        let expr = vector["expr"].as_str().unwrap();
        let dialect = dialect(&vector["dialect"]).unwrap();
        let columns = texts(&vector["columns"]);
        let reason = view_expr::untrusted_call_reason(expr, dialect, &columns);
        (reason.as_deref() != vector["reason"].as_str()).then(|| {
            format!(
                "{expr:?} ({dialect:?}, {columns:?}): {reason:?} where {}",
                vector["reason"]
            )
        })
    });
}

#[test]
fn names_are_read_as_the_old_client_read_them() {
    each("names", 4_000, |vector| {
        let expr = vector["expr"].as_str().unwrap();
        let read = view_expr::name_references(expr);
        let got = match &read {
            Ok(references) => Json::Array(
                references
                    .iter()
                    .map(|reference| {
                        serde_json::json!({
                            "parts": reference.parts,
                            "text": &expr[reference.span.clone()],
                            "bare": reference.bare,
                        })
                    })
                    .collect(),
            ),
            Err(error) => Json::String(error.to_string()),
        };
        let expected = vector
            .get("references")
            .cloned()
            .unwrap_or_else(|| vector["error"].clone());
        (got != expected).then(|| format!("{expr:?}: {got} where {expected}"))
    });
}

#[test]
fn comments_are_stripped_as_databricks_drops_them() {
    each("strip_comments", 150, |vector| {
        let expr = vector["expr"].as_str().unwrap();
        let out = view_expr::strip_comments(expr);
        (out != vector["out"].as_str().unwrap())
            .then(|| format!("{expr:?}: {out:?} where {}", vector["out"]))
    });
}

#[test]
fn string_literals_end_where_databricks_ends_them() {
    each("string_literals", 150, |vector| {
        let expr = vector["expr"].as_str().unwrap();
        let literals: Vec<String> = view_expr::string_literals(expr)
            .into_iter()
            .map(|span| expr[span].to_owned())
            .collect();
        (literals != texts(&vector["literals"]))
            .then(|| format!("{expr:?}: {literals:?} where {}", vector["literals"]))
    });
}

#[test]
fn snowflake_qualifiers_are_read_outside_literals() {
    each("mask_literals", 5, |vector| {
        let expr = vector["expr"].as_str().unwrap();
        let out = view_expr::mask_literals(expr);
        (out != vector["out"].as_str().unwrap())
            .then(|| format!("{expr:?}: {out:?} where {}", vector["out"]))
    });
    each("qualifier_found", 5, |vector| {
        let (table, text) = (
            vector["table"].as_str().unwrap(),
            vector["text"].as_str().unwrap(),
        );
        let found = view_expr::qualifier_found(table, text);
        (found != vector["found"].as_bool().unwrap())
            .then(|| format!("{table:?} in {text:?}: {found}"))
    });
    each("qualifier_stripped", 5, |vector| {
        let (table, expr) = (
            vector["table"].as_str().unwrap(),
            vector["expr"].as_str().unwrap(),
        );
        let out = view_expr::strip_qualifier(table, expr);
        (out != vector["out"].as_str().unwrap())
            .then(|| format!("{table:?} from {expr:?}: {out:?} where {}", vector["out"]))
    });
}

#[test]
fn a_dropped_expression_is_worded_as_the_old_client_worded_it() {
    each("expr_message", 30, |vector| {
        let view = match vector["view"].as_str().unwrap() {
            "metric view" => ViewKind::MetricView,
            "semantic view" => ViewKind::SemanticView,
            other => panic!("{other}"),
        };
        let columns = texts(&vector["columns"]);
        let message = view_expr::expression_message(
            vector["column"].as_str().unwrap(),
            vector["expr"].as_str().unwrap(),
            view,
            dialect(&vector["dialect"]).unwrap(),
            vector["trust_expressions"].as_bool().unwrap(),
            &columns,
        );
        (message.as_deref() != vector["message"].as_str()).then(|| format!("{vector}: {message:?}"))
    });
}

/// Rows 64 and 65 by name: what `trust_expressions` never accepts.
#[test]
fn unsafe_text_is_refused_whatever_the_trust() {
    let databricks = Some(ExprDialect::Databricks);
    assert_eq!(
        view_expr::unsafe_reason("x; DROP TABLE t", databricks).as_deref(),
        Some("it contains ';'")
    );
    assert_eq!(
        view_expr::unsafe_reason("trim(FROM users.country)", databricks).as_deref(),
        Some("it contains a sub-query")
    );
    assert_eq!(
        view_expr::unsafe_reason("'a' || $$b$$", databricks).as_deref(),
        Some("it contains '$$', which Databricks SQL does not quote with")
    );
    let message = view_expr::expression_message(
        "column 'x'",
        "trim(FROM users.country)",
        ViewKind::MetricView,
        ExprDialect::Databricks,
        true,
        &["country"],
    );
    assert_eq!(
        message.as_deref(),
        Some(
            "Failed to add column 'x' since it contains a sub-query, and expressions taken \
             from the metric view are executed as SQL under your warehouse credentials"
        )
    );
}

/// Row 66 by name: a call off the allow-list is refused unless trusted.
#[test]
fn a_function_off_the_allow_list_is_refused() {
    assert_eq!(
        view_expr::untrusted_call_reason("secret('s', 'k')", ExprDialect::Databricks, &["x"])
            .as_deref(),
        Some("it calls 'secret', which is not a pure built-in SQL function")
    );
    assert_eq!(
        view_expr::untrusted_call_reason(
            "CASE WHEN price > 100 THEN 'high' ELSE 'low' END",
            ExprDialect::Databricks,
            &["price"]
        ),
        None
    );
    assert_eq!(
        view_expr::untrusted_call_reason("SYSTEM$WAIT(10)", ExprDialect::Snowflake, &["x"])
            .as_deref(),
        Some("it calls 'SYSTEM$WAIT', which is not a pure built-in SQL function")
    );
}
