// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Quoting, as the reference quotes for DuckDB and SQLite alike
//! (`quote_ident`): an identifier in double quotes and a name compared as
//! text in single quotes, each with its quote doubled, which both
//! engines' standard-conforming strings make sufficient.

/// `name` as an identifier: in double quotes, each one in it doubled.
pub(crate) fn identifier(name: &str) -> String {
    format!("\"{}\"", name.replace('"', "\"\""))
}

/// `names` as identifiers, joined by commas.
pub(crate) fn identifiers(names: &[String]) -> String {
    names
        .iter()
        .map(|n| identifier(n))
        .collect::<Vec<_>>()
        .join(", ")
}

/// `name` as text: in single quotes, each one in it doubled.
pub(crate) fn text(name: &str) -> String {
    format!("'{}'", name.replace('\'', "''"))
}

/// The reference's fold of a key of several columns into one text
/// (`composite_key.sql_expression`): each of `key`, in order, cast to
/// `text_type`, a backslash in it doubled and the unit separator written as
/// a backslash and `u`, the parts joined by the unit separator; the two
/// characters by their codes (`chr`, the dialect's function), as the
/// reference writes them, since some engines read a backslash in a
/// literal as an escape.
pub(crate) fn folded(key: &[String], text_type: &str, chr: &str) -> String {
    let backslash = format!("{chr}(92)");
    let separator = format!("{chr}(31)");
    key.iter()
        .map(|column| {
            format!(
                "REPLACE(REPLACE(CAST({} AS {text_type}), {backslash}, {backslash} || {backslash}), {separator}, {backslash} || 'u')",
                identifier(column)
            )
        })
        .collect::<Vec<_>>()
        .join(&format!(" || {separator} || "))
}
