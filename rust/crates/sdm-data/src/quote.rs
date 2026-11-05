// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Quoting, as the reference quotes for DuckDB and SQLite alike
//! (`quote_ident`): an identifier in double quotes and a name compared as
//! text in single quotes, each with its quote doubled, which both
//! engines' standard-conforming strings make sufficient. Postgres quotes
//! an identifier the same way (the trait's default, `sql.rs`); its text
//! literal is its own (`postgres.rs`).

/// `name` as an identifier: in double quotes, each one in it doubled.
pub(crate) fn identifier(name: &str) -> String {
    format!("\"{}\"", name.replace('"', "\"\""))
}

/// `name` as text: in single quotes, each one in it doubled.
pub(crate) fn text(name: &str) -> String {
    format!("'{}'", name.replace('\'', "''"))
}
