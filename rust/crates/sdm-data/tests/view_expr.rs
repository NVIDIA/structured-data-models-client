// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The view expressions' checks (T3.5, old-behaviour rows 61 to 68) against
//! `contract/view-expr-vectors.json`: every call the old client's own tests
//! made of `RE/view_expr.py` and of the Snowflake qualifier helpers of
//! `RE/graph.py`, with its answer, and each word of the old word tables
//! (the dialects' allow-lists, keywords, type names, session words) put to
//! the same checks, and from T3.5 fix 1 each separator Python reads as
//! white space (U+001C to U+001F) in both dialects and both trust settings,
//! and tokens of up to a megabyte, recorded by recipe. Here each answer is
//! held to whether the old client
//! refused, and the names, literals and qualifiers it read; each reason's
//! text is the core's to write and is held there
//! (`sdm-core/tests/view_words.rs`).

use sdm_data::view_expr::{self, ExprDialect};
use serde_json::Value as Json;

/// Python's white space (`str.isspace`), which the core hands in from
/// `sdm-pql`: the 29 characters Python 3.12 counts.
const SPACES: Spaces = Spaces(|c| {
    matches!(
        c,
        '\t'..='\r'
            | '\u{1c}'..='\u{20}'
            | '\u{85}'
            | '\u{a0}'
            | '\u{1680}'
            | '\u{2000}'..='\u{200a}'
            | '\u{2028}'
            | '\u{2029}'
            | '\u{202f}'
            | '\u{205f}'
            | '\u{3000}'
    )
});

use sdm_data::view_expr::Spaces;

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
fn unsafe_text_is_refused_where_the_old_client_refused_it() {
    each("unsafe", 400, |vector| {
        let expr = vector["expr"].as_str().unwrap();
        let refused = view_expr::unsafe_reason(expr, dialect(&vector["dialect"])).is_some();
        (refused != vector["reason"].is_string())
            .then(|| format!("{expr:?} ({}): refused {refused}", vector["dialect"]))
    });
}

#[test]
fn calls_are_trusted_where_the_old_client_trusted_them() {
    each("calls", 10_000, |vector| {
        let expr = vector["expr"].as_str().unwrap();
        let dialect = dialect(&vector["dialect"]).unwrap();
        let columns = texts(&vector["columns"]);
        let refused = view_expr::untrusted_call_reason(expr, dialect, &columns, SPACES).is_some();
        (refused != vector["reason"].is_string())
            .then(|| format!("{expr:?} ({dialect:?}, {columns:?}): refused {refused}"))
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
        let out = view_expr::strip_comments(expr, SPACES).unwrap();
        (out != vector["out"].as_str().unwrap())
            .then(|| format!("{expr:?}: {out:?} where {}", vector["out"]))
    });
}

#[test]
fn string_literals_end_where_databricks_ends_them() {
    each("string_literals", 150, |vector| {
        let expr = vector["expr"].as_str().unwrap();
        let literals: Vec<String> = view_expr::string_literals(expr)
            .unwrap()
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
        let out = view_expr::mask_literals(expr).unwrap();
        (out != vector["out"].as_str().unwrap())
            .then(|| format!("{expr:?}: {out:?} where {}", vector["out"]))
    });
    each("qualifier_found", 5, |vector| {
        let (table, text) = (
            vector["table"].as_str().unwrap(),
            vector["text"].as_str().unwrap(),
        );
        let found = view_expr::qualifier_found(table, text).unwrap();
        (found != vector["found"].as_bool().unwrap())
            .then(|| format!("{table:?} in {text:?}: {found}"))
    });
    each("qualifier_stripped", 5, |vector| {
        let (table, expr) = (
            vector["table"].as_str().unwrap(),
            vector["expr"].as_str().unwrap(),
        );
        let out = view_expr::strip_qualifier(table, expr).unwrap();
        (out != vector["out"].as_str().unwrap())
            .then(|| format!("{table:?} from {expr:?}: {out:?} where {}", vector["out"]))
    });
}

/// A long token's expression, from its recipe: a prefix, a text repeated
/// `count` times, and a suffix.
fn recipe(recipe: &Json, count: usize) -> String {
    let part = |i: usize| recipe[i].as_str().unwrap();
    format!("{}{}{}", part(0), part(1).repeat(count), part(3))
}

/// Codex review 1, P1-3, and Claude review 1, P1-1: a long token is read,
/// never a panic, and answered as the old client answered it, accepted
/// where it accepted it.
#[test]
fn a_long_token_is_answered_as_the_old_client_answered_it() {
    each("long_tokens", 15, |vector| {
        let r = &vector["recipe"];
        let expr = recipe(r, r[2].as_u64().unwrap() as usize);
        let called = recipe(r, r[4].as_u64().unwrap() as usize);
        let mut wrong = Vec::new();
        for (key, dialect) in [
            ("None", None),
            ("databricks", Some(ExprDialect::Databricks)),
            ("snowflake", Some(ExprDialect::Snowflake)),
        ] {
            let refused = view_expr::unsafe_reason(&expr, dialect).is_some();
            if refused != vector["unsafe"][key].is_string() {
                wrong.push(format!("unsafe {key}: refused {refused}"));
            }
        }
        for (key, dialect) in [
            ("databricks", ExprDialect::Databricks),
            ("snowflake", ExprDialect::Snowflake),
        ] {
            let refused =
                view_expr::untrusted_call_reason(&called, dialect, &["x", "y", "z"], SPACES)
                    .is_some();
            if refused != vector["calls"][key].is_string() {
                wrong.push(format!("calls {key}: refused {refused}"));
            }
        }
        let names = match view_expr::name_references(&expr) {
            Ok(references) => serde_json::json!({"count": references.len(), "error": null}),
            Err(error) => serde_json::json!({"count": null, "error": error.to_string()}),
        };
        if names != vector["names"] {
            wrong.push(format!("names {names} where {}", vector["names"]));
        }
        (!wrong.is_empty()).then(|| format!("{r}: {}", wrong.join("; ")))
    });
}

/// Every recipe at a megabyte or more, the call check included where the
/// old client takes too long to answer: each check answers, and a failure
/// to read refuses, never accepts.
#[test]
fn a_megabyte_of_any_token_is_answered() {
    let document = document();
    for vector in document["long_tokens"].as_array().unwrap() {
        let r = &vector["recipe"];
        let expr = recipe(r, r[2].as_u64().unwrap() as usize);
        for dialect in [
            None,
            Some(ExprDialect::Databricks),
            Some(ExprDialect::Snowflake),
        ] {
            let _ = view_expr::unsafe_reason(&expr, dialect);
        }
        for dialect in [ExprDialect::Databricks, ExprDialect::Snowflake] {
            let _ = view_expr::untrusted_call_reason(&expr, dialect, &["x"], SPACES);
        }
        let _ = view_expr::name_references(&expr);
        let _ = view_expr::strip_comments(&expr, SPACES);
    }
}
