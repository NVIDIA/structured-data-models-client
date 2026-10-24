// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A Snowflake semantic view's table qualifiers (`RE/graph.py:107-156`,
//! old-behaviour row 68): `table.` or `"table".`, in any case, not after a
//! word character, dot or quote. A single-quoted literal is data, so it is
//! masked when asking whether an expression reads another table, and left
//! byte for byte when the expression's own qualifier is taken out. A
//! pattern that fails makes the expression unreadable, which the caller
//! refuses.

use std::sync::LazyLock;

use super::names::Unreadable;
use super::pattern::Pattern;

static STRING_LITERAL: LazyLock<Pattern> = LazyLock::new(|| Pattern::new(r"'(?:''|[^'])*'"));

/// The pattern of `table`'s qualifier (`_qualifier_pattern`): the name
/// double-quoted, doubling a quote in it, or as written, then a dot.
fn qualifier(table: &str) -> Pattern {
    let quoted = format!("\"{}\"", table.replace('"', "\"\""));
    Pattern::new(&format!(
        r#"(?i)(?<![A-Za-z0-9_$."])(?:{}|{})\s*\."#,
        fancy_regex::escape(&quoted),
        fancy_regex::escape(table)
    ))
}

/// `expr` with each single-quoted literal blanked out, one space per
/// character (`_mask_literals`).
pub fn mask_literals(expr: &str) -> Result<String, Unreadable> {
    let mut out = String::with_capacity(expr.len());
    let mut last = 0;
    for literal in STRING_LITERAL.find_all(expr)? {
        out.push_str(&expr[last..literal.start]);
        out.push_str(&" ".repeat(expr[literal.clone()].chars().count()));
        last = literal.end;
    }
    out.push_str(&expr[last..]);
    Ok(out)
}

/// Whether `text` holds `table`'s qualifier.
pub fn qualifier_found(table: &str, text: &str) -> Result<bool, Unreadable> {
    qualifier(table).is_match(text)
}

/// `expr` with `table`'s qualifier taken out wherever it stands outside a
/// single-quoted literal (`_sub_outside_literals`).
pub fn strip_qualifier(table: &str, expr: &str) -> Result<String, Unreadable> {
    let pattern = qualifier(table);
    let mut out = String::with_capacity(expr.len());
    let mut last = 0;
    for literal in STRING_LITERAL.find_all(expr)? {
        out.push_str(&pattern.replace_all(&expr[last..literal.start], "")?);
        out.push_str(&expr[literal.clone()]);
        last = literal.end;
    }
    out.push_str(&pattern.replace_all(&expr[last..], "")?);
    Ok(out)
}
