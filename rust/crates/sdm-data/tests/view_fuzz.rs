// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! No expression, name or table panics the view expressions' readers (T3.5
//! fix 2, the owner's stop rule): a few seconds of expressions made of SQL's
//! pieces, of the recorded expressions cut and spliced, and of table names
//! the patterns are made of, a 100,000-character one among them (Codex
//! review 2, P1-7), each through the lexer, the checks, the name reader and
//! every pattern built from a definition's text. Each may refuse; none may
//! panic.

use sdm_data::view_expr::metric;
use sdm_data::view_expr::{
    self, ExprDialect, ViewKind, mask_literals, name_references, qualifier_found, string_literals,
    strip_comments, strip_qualifier, unsafe_reason, untrusted_call_reason,
};
use serde_json::Value as Json;

const EXPR: &str = include_str!("../../../../contract/view-expr-vectors.json");

/// The pieces the expressions are made of.
const PIECES: &[&str] = &[
    "x",
    "t",
    "source",
    ".",
    " ",
    "(",
    ")",
    "'",
    "\"",
    "`",
    "''",
    "\\'",
    "--",
    "/*",
    "*/",
    "\n",
    ",",
    "+",
    "*",
    "=",
    "->",
    "a.b",
    "`a`.`b`",
    "\"T\".",
    "select",
    "SELECT",
    "from",
    "WITH",
    "case",
    "when",
    "then",
    "end",
    "sum",
    "count",
    "distinct",
    "trim",
    "FROM",
    "secret",
    "current_user",
    "lambda",
    "x -> x",
    "1",
    "1e3",
    "0x1F",
    "'a''b'",
    "$",
    "#",
    ";",
    "é",
    "\u{1c}",
    "\u{3000}",
    "\t",
    "\\",
    "[",
    "]",
    "{",
    "}",
    ":",
    "::",
    "IN",
    "NOT",
    "AND",
    "OR",
    "over",
    "partition by",
    "order by",
    "rows",
    "interval",
    "map",
    "using",
    "values",
    "?",
    "@",
    "\0",
    "日本",
];

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// Reads `expr` every way an expression is read, with `table` as the
/// table a qualifier names.
fn read_every_way(expr: &str, table: &str) {
    for dialect in [
        None,
        Some(ExprDialect::Databricks),
        Some(ExprDialect::Snowflake),
    ] {
        let _ = unsafe_reason(expr, dialect);
    }
    for dialect in [ExprDialect::Databricks, ExprDialect::Snowflake] {
        let _ = untrusted_call_reason(expr, dialect, &["x", table]);
        let _ = view_expr::expression_message(
            "column 'x'",
            expr,
            ViewKind::MetricView,
            dialect,
            false,
            &["x"],
            &|text: &str| format!("'{text}'"),
        );
    }
    let _ = name_references(expr);
    let _ = strip_comments(expr);
    let _ = string_literals(expr);
    let _ = mask_literals(expr);
    let _ = qualifier_found(table, expr);
    let _ = strip_qualifier(table, expr);
    let _ = metric::references_alias(expr, &[table, "x"]);
    let _ = metric::strip_alias(expr, &[table]);
    let _ = metric::derived_column(expr);
    let _ = metric::renamed_column(expr);
    let _ = metric::wildcard(expr);
    let _ = metric::table_reference(expr);
    let _ = metric::view_name(expr);
    let _ = metric::join_condition(expr);
    let _ = metric::unquote(expr);
    let _ = metric::reads_star(expr);
    let _ = metric::type_class(expr);
    let _ = metric::check_filter(Some(expr), &[(vec![table.to_owned()], "t".into())]);
}

#[test]
fn no_expression_panics_a_reader() {
    let mut rng = Rng(0x2545_f491_4f6c_dd1d);
    let long_table = "a".repeat(100_000);
    read_every_way("x", &long_table);
    read_every_way(&format!("{long_table}.x"), &long_table);
    for _ in 0..1_500 {
        let length = 1 + rng.below(30);
        let expr: String = (0..length)
            .map(|_| PIECES[rng.below(PIECES.len())])
            .collect();
        let table: String = (0..1 + rng.below(3))
            .map(|_| PIECES[rng.below(PIECES.len())])
            .collect();
        read_every_way(&expr, &table);
    }
    let document: Json = serde_json::from_str(EXPR).unwrap();
    let exprs: Vec<&str> = document["unsafe"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|vector| vector["expr"].as_str())
        .filter(|expr| expr.len() < 2_000)
        .collect();
    for _ in 0..1_000 {
        let a = exprs[rng.below(exprs.len())];
        let b = exprs[rng.below(exprs.len())];
        let cut = |text: &str, at: usize| {
            let at = at % (text.len() + 1);
            (0..=at)
                .rev()
                .find(|&i| text.is_char_boundary(i))
                .unwrap_or(0)
        };
        let (i, j) = (cut(a, rng.below(64)), cut(b, rng.below(64)));
        read_every_way(
            &format!("{}{}{}", &a[..i], PIECES[rng.below(PIECES.len())], &b[j..]),
            "t",
        );
    }
}

/// A table's name too long for the pattern compiler makes the qualifier
/// check unreadable, never a panic (Codex review 2, P1-7).
#[test]
fn a_pattern_past_the_compiler_s_bounds_is_unreadable() {
    let table = "a".repeat(200_000);
    assert!(qualifier_found(&table, "x").is_err());
    assert!(strip_qualifier(&table, "x").is_err());
    assert!(metric::references_alias("x", &[table.as_str()]).is_err());
    assert_eq!(qualifier_found("a", "a.x"), Ok(true));
}

/// No row of `DESCRIBE TABLE EXTENDED` panics its reader.
#[test]
fn no_description_panics_its_reader() {
    let mut rng = Rng(0x1234_5678_9abc_def1);
    let cells = [
        "",
        "# Detailed Table Information",
        "Type",
        "METRIC_VIEW",
        "View Text",
        "#",
        " ",
        "\u{1c}",
        "# col_name",
        "x",
    ];
    for _ in 0..5_000 {
        let rows: Vec<Vec<Option<String>>> = (0..rng.below(8))
            .map(|_| {
                (0..rng.below(4))
                    .map(|_| (rng.below(5) > 0).then(|| cells[rng.below(cells.len())].to_owned()))
                    .collect()
            })
            .collect();
        let _ = view_expr::describe::describe_view(&rows);
    }
}
