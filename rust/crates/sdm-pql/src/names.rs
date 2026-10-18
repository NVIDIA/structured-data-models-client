// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! How a query spells table and column names (`predictive_query.names`).
//!
//! A name the grammar's bare identifier cannot spell is written in
//! backticks, a backtick inside it doubled: `` `a``b` `` names `` a`b ``. A
//! quoted name holds any other character as it stands, a dot or a line
//! break included. The quotes are only spelling: once parsed, the tree holds
//! the table and the column apart ([`ColumnKey`]), each as the data spells
//! it, so a dot in either is just a character, and writing a query back
//! quotes them again (Phase 2 decision KO2-36). The reference refuses a
//! quoted name holding a dot, and its grammar one holding a backtick or a
//! line break; the core answers them (register, T2.18).

use crate::schema::ColumnKey;

const QUOTE: char = '`';

/// Whether `name` is a bare identifier, `[_a-zA-Z][_a-zA-Z0-9]*`.
fn is_bare(name: &str) -> bool {
    let mut chars = name.chars();
    chars
        .next()
        .is_some_and(|c| c == '_' || c.is_ascii_alphabetic())
        && chars.all(|c| c == '_' || c.is_ascii_alphanumeric())
}

/// The name a bare or quoted spelling stands for, and the rest of `text`
/// after it: a quoted name's doubled backticks read as one.
fn read_name(text: &str) -> (String, &str) {
    let Some(quoted) = text.strip_prefix(QUOTE) else {
        let end = text.find('.').unwrap_or(text.len());
        return (text[..end].to_string(), &text[end..]);
    };
    let mut name = String::new();
    let mut chars = quoted.char_indices().peekable();
    while let Some((at, c)) = chars.next() {
        if c != QUOTE {
            name.push(c);
        } else if chars.peek().is_some_and(|&(_, next)| next == QUOTE) {
            name.push(QUOTE);
            chars.next();
        } else {
            return (name, &quoted[at + 1..]);
        }
    }
    (name, "")
}

/// The table and the column a column token names (`FQN_COLUMN` or
/// `WILDCARD_COLUMN`, `table.column` with either part quoted, or
/// `table.*`), each unquoted. The lexer has matched the token, so its first
/// name ends at the dot that separates the two.
pub fn split_column(text: &str) -> ColumnKey {
    let (table, rest) = read_name(text);
    let (column, _) = read_name(rest.strip_prefix('.').unwrap_or(rest));
    ColumnKey {
        table_name: table,
        col_name: column,
    }
}

/// `quote_name`: the name in backticks, each backtick in it doubled, if a
/// bare identifier cannot spell it.
pub fn quote_name(name: &str) -> String {
    if name == "*" || is_bare(name) {
        return name.to_string();
    }
    format!("{QUOTE}{}{QUOTE}", name.replace(QUOTE, "``"))
}

/// `fqn`: `table.column`, each part quoted if it needs to be.
pub fn fqn(column: &ColumnKey) -> String {
    format!(
        "{}.{}",
        quote_name(&column.table_name),
        quote_name(&column.col_name)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_are_quoted_only_when_needed() {
        let key = |t: &str, c: &str| ColumnKey::new(t, c);
        assert_eq!(split_column("`a b`.c"), key("a b", "c"));
        assert_eq!(split_column("t.`a.b`"), key("t", "a.b"));
        assert_eq!(split_column("`x.y`.`a.b`"), key("x.y", "a.b"));
        assert_eq!(split_column("t.`a``b`"), key("t", "a`b"));
        assert_eq!(split_column("t.````"), key("t", "`"));
        assert_eq!(split_column("t.`a\nb\r`"), key("t", "a\nb\r"));
        assert_eq!(split_column("`t`.*"), key("t", "*"));
        assert_eq!(fqn(&key("a b", "*")), "`a b`.*");
        assert_eq!(fqn(&key("users", "user_id")), "users.user_id");
        assert_eq!(fqn(&key("t", "a.b")), "t.`a.b`");
        assert_eq!(quote_name("a`b"), "`a``b`");
        assert_eq!(quote_name("a\nb"), "`a\nb`");
        for name in ["a.b", "a`b", "`", "``", "a\nb", "a\r\nb", "x.`y`.z"] {
            let column = key("t", name);
            assert_eq!(split_column(&fqn(&column)), column, "{name:?}");
        }
    }
}
