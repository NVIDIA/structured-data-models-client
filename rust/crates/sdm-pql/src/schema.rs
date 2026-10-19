// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The schema a query is resolved against (`predictive_query.schema`):
//! tables, their columns and keys, and the groups of columns that link them.

use crate::types::{Dtype, Stype};

/// A column of a table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Column {
    pub name: String,
    pub stype: Stype,
    pub dtype: Dtype,
}

/// The columns of a table, and which of them carry its identity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TableDefinition {
    pub cols: Vec<Column>,
    pub pkey: Option<String>,
    pub time_col: Option<String>,
    pub end_time_col: Option<String>,
}

/// A column within a schema, and a column a query names: its table and its
/// column held apart, so either may hold a dot (Phase 2 decision KO2-36).
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ColumnKey {
    pub table_name: String,
    pub col_name: String,
}

impl ColumnKey {
    pub fn new(table_name: &str, col_name: &str) -> Self {
        ColumnKey {
            table_name: table_name.to_string(),
            col_name: col_name.to_string(),
        }
    }

    /// `table.*`, the rows of a table rather than a column of it.
    pub fn is_wildcard(&self) -> bool {
        self.col_name == "*"
    }
}

/// `table.column`, unquoted, as a message names a column; a query's text
/// quotes it with [`crate::names::fqn`].
impl std::fmt::Display for ColumnKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}", self.table_name, self.col_name)
    }
}

/// Columns linked together, such as a primary key and its foreign keys.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ColumnKeyGroup {
    pub columns: Vec<ColumnKey>,
}

/// An edge between two tables.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Relationship {
    pub src_table: String,
    pub fkey: String,
    pub dst_table: String,
}

/// A schema: its tables, in the order given, and the groups linking them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SchemaDefinition {
    pub tables: Vec<(String, TableDefinition)>,
    pub col_groups: Vec<ColumnKeyGroup>,
}

impl SchemaDefinition {
    pub fn table(&self, name: &str) -> Option<&TableDefinition> {
        self.tables.iter().find(|(n, _)| n == name).map(|(_, t)| t)
    }

    pub fn has_table(&self, name: &str) -> bool {
        self.table(name).is_some()
    }

    /// The table, which the caller has already found in the schema.
    pub fn table_or_empty(&self, name: &str) -> TableDefinition {
        self.table(name).cloned().unwrap_or(TableDefinition {
            cols: Vec::new(),
            pkey: None,
            time_col: None,
            end_time_col: None,
        })
    }

    /// Every column of every group, in order.
    pub fn all_keys(&self) -> Vec<&ColumnKey> {
        self.col_groups
            .iter()
            .flat_map(|g| g.columns.iter())
            .collect()
    }
}
