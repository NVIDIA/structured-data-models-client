// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Expression safety, trust boundaries and lexical meaning, with independent expectations.

use sdm_data::view_expr::{self, ExprDialect};

/// Python's white space (`str.isspace`), which the core hands in from
/// `sdm-pql`: the 29 characters Python 3.12 counts, with the class the
/// patterns are compiled with derived from them.
fn spaces() -> &'static Spaces {
    static SPACES: std::sync::LazyLock<Spaces> = std::sync::LazyLock::new(|| {
        Spaces::new(|c| {
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
        })
    });
    &SPACES
}

use sdm_data::view_expr::Spaces;

const DIALECTS: [ExprDialect; 2] = [ExprDialect::Databricks, ExprDialect::Snowflake];

#[test]
fn unsafe_sql_is_refused_even_when_expressions_are_trusted() {
    for expr in [
        "x; DROP TABLE t",
        "x -- comment",
        "x /* comment */",
        "x */",
        "x |> f()",
        "x ->> 'a'",
        "SELECT x",
        "(SELECT x FROM t)",
        "(FROM t)",
        "(VALUES (1))",
        "UPDATE t SET x = 1",
        "'unclosed",
        "(x",
        "x)",
        "é + 1",
    ] {
        for dialect in DIALECTS {
            for trust in [false, true] {
                let verdict = view_expr::expression_verdict(expr, dialect, trust, &["x"], spaces())
                    .unwrap_or_else(|| panic!("accepted {expr:?}, {dialect:?}, trust={trust}"));
                assert!(
                    !verdict.untrusted,
                    "unsafe SQL must not become legal through trust"
                );
            }
        }
    }
}

#[test]
fn pure_calls_are_allowed_but_user_code_and_session_values_require_trust() {
    for dialect in DIALECTS {
        for expr in [
            "x + 1",
            "abs(x)",
            "coalesce(x, 0)",
            "lower('ABC')",
            "'select'",
        ] {
            assert_eq!(
                view_expr::expression_verdict(expr, dialect, false, &["x"], spaces()),
                None,
                "{expr:?}, {dialect:?}"
            );
        }
        for expr in [
            "secret('scope', 'key')",
            "custom_function(x)",
            "db.custom_function(x)",
            "current_user",
        ] {
            let verdict =
                view_expr::expression_verdict(expr, dialect, false, &["x"], spaces()).unwrap();
            assert!(verdict.untrusted, "{expr:?}, {dialect:?}");
            assert_eq!(
                view_expr::expression_verdict(expr, dialect, true, &["x"], spaces()),
                None
            );
        }
    }
}

#[test]
fn dialect_specific_quotes_and_comments_remain_distinct() {
    assert!(
        view_expr::unsafe_reason("$$text$$", Some(ExprDialect::Databricks), spaces()).is_some()
    );
    assert_eq!(
        view_expr::unsafe_reason("$$text$$", Some(ExprDialect::Snowflake), spaces()),
        None
    );
    assert!(
        view_expr::unsafe_reason("x // comment", Some(ExprDialect::Snowflake), spaces()).is_some()
    );
    assert_eq!(
        view_expr::unsafe_reason("'// not a comment'", Some(ExprDialect::Snowflake), spaces()),
        None
    );
}

#[test]
fn names_preserve_parts_spelling_and_byte_spans() {
    let expr = "coalesce(orders.amount, `tax rate`) + CAST(discount AS DOUBLE)";
    let refs = view_expr::name_references(expr, spaces()).unwrap();
    let parts: Vec<Vec<&str>> = refs
        .iter()
        .map(|r| r.parts.iter().map(String::as_str).collect())
        .collect();
    assert_eq!(
        parts,
        vec![vec!["orders", "amount"], vec!["tax rate"], vec!["discount"]]
    );
    let spelling: Vec<&str> = refs.iter().map(|r| &expr[r.span.clone()]).collect();
    assert_eq!(spelling, ["orders.amount", "`tax rate`", "discount"]);
    assert_eq!(
        refs.iter().map(|r| r.bare).collect::<Vec<_>>(),
        [false, false, true]
    );
    assert_eq!(
        view_expr::name_references("'column' + 42", spaces())
            .unwrap()
            .len(),
        0
    );
    assert!(view_expr::name_references("'unclosed", spaces()).is_err());
}

#[test]
fn comments_and_literals_are_parsed_without_confusing_quoted_text() {
    assert_eq!(
        view_expr::strip_comments("x /* comment */ + y", spaces())
            .unwrap()
            .trim(),
        "x   + y"
    );
    assert_eq!(
        view_expr::strip_comments("'-- literal' + x", spaces()).unwrap(),
        "'-- literal' + x"
    );
    let expr = "concat('a', 'b''c', name)";
    let literals: Vec<&str> = view_expr::string_literals(expr, spaces())
        .unwrap()
        .into_iter()
        .map(|s| &expr[s])
        .collect();
    assert_eq!(literals, ["'a'", "'b''c'"]);
    // Extraction returns complete literals only; unsafe_reason rejects the unmatched quote.
    assert!(
        view_expr::string_literals("'unclosed", spaces())
            .unwrap()
            .is_empty()
    );
}

#[test]
fn qualifiers_are_removed_only_from_references_not_literals() {
    assert_eq!(
        view_expr::qualifier_found("t", "t.x + 1", spaces()),
        Ok(true)
    );
    assert_eq!(
        view_expr::qualifier_found(
            "t",
            &view_expr::mask_literals("other.x + 't.x'", spaces()).unwrap(),
            spaces(),
        ),
        Ok(false)
    );
    assert_eq!(
        view_expr::strip_qualifier("t", "t.x + 't.x'", spaces()),
        Ok("x + 't.x'".into())
    );
    let masked = view_expr::mask_literals("t.x + 't.y'", spaces()).unwrap();
    assert!(masked.contains("t.x"));
    assert!(!masked.contains("t.y"));
}

#[test]
fn control_whitespace_cannot_hide_subqueries_or_untrusted_calls() {
    for separator in ['\u{1c}', '\u{1d}', '\u{1e}', '\u{1f}'] {
        for dialect in DIALECTS {
            assert!(
                view_expr::unsafe_reason(&format!("(FROM{separator}t)"), Some(dialect), spaces())
                    .is_some()
            );
            assert!(
                view_expr::untrusted_call_reason(
                    &format!("secret{separator}('s', 'k')"),
                    dialect,
                    &["x"],
                    spaces()
                )
                .is_some()
            );
        }
    }
}

#[test]
fn long_identifiers_and_literals_are_generated_without_recordings() {
    for length in [1_024, 1_048_576] {
        let name = "x".repeat(length);
        let refs = view_expr::name_references(&name, spaces()).unwrap();
        assert_eq!(refs.len(), 1);
        assert_eq!(refs[0].parts.as_slice(), std::slice::from_ref(&name));
        assert_eq!(refs[0].span, 0..length);
        let literal = format!("'{name}'");
        let literals = view_expr::string_literals(&literal, spaces()).unwrap();
        assert_eq!(literals.len(), 1);
        assert_eq!(literals[0], 0..literal.len());
        for dialect in DIALECTS {
            assert_eq!(
                view_expr::unsafe_reason(&literal, Some(dialect), spaces()),
                None
            );
            assert!(
                view_expr::unsafe_reason(&format!("{name};"), Some(dialect), spaces()).is_some()
            );
        }
    }
}

#[test]
fn generated_token_shapes_do_not_panic_or_bypass_unsafe_checks() {
    let padding = " ".repeat(100_000);
    for dialect in DIALECTS {
        assert!(
            view_expr::unsafe_reason(&format!("SELECT{padding}x"), Some(dialect), spaces())
                .is_some()
        );
        assert!(
            view_expr::untrusted_call_reason(
                &format!("current_user{padding}"),
                dialect,
                &["x"],
                spaces()
            )
            .is_some()
        );
    }
    for token in [
        format!("`{}`", "a".repeat(100_000)),
        format!("\"{}\"", "a".repeat(100_000)),
        format!("{}a", "1".repeat(100_000)),
        format!("{}(", "x".repeat(20_000)),
        "1.".repeat(50_000),
        format!("{}x{}", "(".repeat(1000), ")".repeat(1000)),
    ] {
        // These malformed or unusual tokens may be rejected; each reader must return.
        let _ = view_expr::name_references(&token, spaces());
        let _ = view_expr::strip_comments(&token, spaces());
        let _ = view_expr::string_literals(&token, spaces());
        for dialect in DIALECTS {
            let _ = view_expr::unsafe_reason(&token, Some(dialect), spaces());
            let _ = view_expr::untrusted_call_reason(&token, dialect, &["x"], spaces());
        }
    }
}

#[test]
fn session_named_columns_override_session_values_only_in_databricks() {
    for expr in ["current_user", "`current_user`"] {
        assert_eq!(
            view_expr::untrusted_call_reason(
                expr,
                ExprDialect::Databricks,
                &["CURRENT_USER"],
                spaces()
            ),
            None
        );
        assert!(
            view_expr::untrusted_call_reason(expr, ExprDialect::Databricks, &["other"], spaces())
                .is_some()
        );
        assert!(
            view_expr::untrusted_call_reason(
                expr,
                ExprDialect::Snowflake,
                &["CURRENT_USER"],
                spaces()
            )
            .is_some()
        );
    }
}
