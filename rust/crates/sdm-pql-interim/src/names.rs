// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! How a query spells table and column names (`predictive_query.names`).
//!
//! A name the grammar's bare identifier cannot spell is written in
//! backticks. The quotes are only spelling: once parsed, the tree holds the
//! name the data uses, and writing a query back quotes it again.

const QUOTE: char = '`';

/// Whether `name` is a bare identifier, `[_a-zA-Z][_a-zA-Z0-9]*`.
fn is_bare(name: &str) -> bool {
    let mut chars = name.chars();
    chars
        .next()
        .is_some_and(|c| c == '_' || c.is_ascii_alphabetic())
        && chars.all(|c| c == '_' || c.is_ascii_alphanumeric())
}

/// `split_fqn`: the parts of a dotted name, dots inside backticks kept and
/// the backticks dropped.
pub fn split_fqn(full_name: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut current = String::new();
    let mut quoted = false;
    for c in full_name.chars() {
        if c == QUOTE {
            quoted = !quoted;
        } else if c == '.' && !quoted {
            parts.push(std::mem::take(&mut current));
        } else {
            current.push(c);
        }
    }
    parts.push(current);
    parts
}

/// `canonical_fqn`: the name unquoted, refusing a quoted part with a dot.
pub fn canonical_fqn(full_name: &str) -> Result<String, String> {
    let parts = split_fqn(full_name);
    if let Some(part) = parts.iter().find(|p| p.contains('.')) {
        return Err(format!(
            "Column name '{part}' contains a dot, which is not supported in a predictive \
             query. Rename the column to reference it."
        ));
    }
    Ok(parts.join("."))
}

/// `quote_name`: the name in backticks if a bare identifier cannot spell it.
pub fn quote_name(name: &str) -> Result<String, String> {
    if name == "*" || is_bare(name) {
        return Ok(name.to_string());
    }
    for (c, label) in [
        (QUOTE, "a backtick"),
        ('\r', "a carriage return"),
        ('\n', "a newline"),
    ] {
        if name.contains(c) {
            return Err(format!(
                "Name '{name}' contains {label}, which a predictive query cannot express."
            ));
        }
    }
    Ok(format!("{QUOTE}{name}{QUOTE}"))
}

/// `fqn`: `table.column`, each part quoted if it needs to be.
pub fn fqn(table: &str, column: &str) -> Result<String, String> {
    Ok(format!("{}.{}", quote_name(table)?, quote_name(column)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_are_quoted_only_when_needed() {
        assert_eq!(canonical_fqn("`a b`.c").unwrap(), "a b.c");
        assert!(canonical_fqn("`a.b`.c").is_err());
        assert_eq!(fqn("a b", "*").unwrap(), "`a b`.*");
        assert_eq!(fqn("users", "user_id").unwrap(), "users.user_id");
        assert!(quote_name("a`b").is_err());
        assert_eq!(split_fqn("x.`y.z`"), vec!["x", "y.z"]);
    }
}
