// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The SQL each supported database is sent, written as today's Python
//! writes it, byte for byte: one module per database (`databricks`,
//! `duckdb`, `postgres`, `snow`, `sqlite`, each `backend/<db>/dialect.py`),
//! the fragments they share (`fragments`), the decisions the tables and
//! samplers make about SQL (`helpers`), and the check on a view's
//! expressions (`view_expr`).
//!
//! Every function takes typed inputs and returns a [`Statement`], the SQL
//! and its parameters, or the text or value the Python function returns. It
//! never touches a database. [`render_json`] answers a call recorded in the
//! dialect-call capture (`tests/rust_core/dialects/CONTRACT.md`).

use std::fmt;

pub mod arrow;
pub mod databricks;
pub mod duckdb;
pub mod fragments;
pub mod helpers;
pub mod int;
pub mod postgres;
pub mod py;
pub mod pyjson;
mod render;
pub mod snow;
pub mod sqlite;
pub mod text;
pub mod time;
mod tzfile;
pub mod unicode;
mod unicode_tables;
pub mod view_expr;

pub use render::{Rendered, render, render_json};
pub use sdm_protocol::Param;

/// One statement and the values its placeholders take, in order, or `None`
/// where it takes none.
pub type Statement = (String, Option<Vec<Param>>);

/// One table's part of a key count (`base/source.py::KeyCountPart`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KeyCountPart {
    /// The table's position in the count.
    pub label: i64,
    /// What its rows are read from.
    pub source: String,
    /// Its key's columns, quoted.
    pub columns: Vec<String>,
    /// Whether the source keeps the key unique.
    pub unique: bool,
}

/// Why a call cannot be answered: arguments it cannot read, or what today's
/// Python raises for them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error(String);

impl Error {
    pub fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

/// A value as a Python f-string writes an optional one: `None` when
/// missing.
pub(crate) fn opt<T: fmt::Display>(value: Option<T>) -> String {
    value.map_or_else(|| "None".to_string(), |v| v.to_string())
}
