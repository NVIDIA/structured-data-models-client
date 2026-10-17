// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The catalog's answers as the core reads them, in no dialect's shape
//! (decision E54).
//!
//! A `describe` request is answered in the shape its plan expects: a SQL
//! dialect's own catalog's, or, for a host provider, the contract's
//! (`data::catalog_columns`). The data layer reads either into a
//! [`CatalogAnswer`] after the core has admitted it (decision H16, check
//! (b)), so the core and `sdm-schema` read a column's name, type,
//! nullability and key flag, a key's kind and columns, a foreign key's
//! column pairs and an index's columns, resolved or not, and never a
//! dialect's spelling of them. These types never travel: a host answers in
//! Arrow.

use crate::data::DescribePart;
use crate::ops::Dtype;

/// One part of the catalog, read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CatalogAnswer {
    /// `tables`: each table's name, in the order the catalog lists them.
    Tables(Vec<String>),
    /// `columns`: each column of the one table asked for, in its order.
    Columns(Vec<CatalogColumn>),
    /// `keys`: each key the one table asked for declares.
    Keys(Vec<CatalogKey>),
    /// `foreign_keys`: each column pair of each foreign key the one table
    /// asked for declares.
    ForeignKeys(Vec<CatalogForeignKey>),
    /// `indexes`: each index of the source, or of the one table asked for.
    Indexes(Vec<CatalogIndex>),
    /// `index_columns`: the columns of the one index asked for, in its
    /// order, `None` for an expression.
    IndexColumns(Vec<Option<String>>),
}

impl CatalogAnswer {
    /// The part of the catalog this answers.
    pub fn part(&self) -> DescribePart {
        match self {
            CatalogAnswer::Tables(_) => DescribePart::Tables,
            CatalogAnswer::Columns(_) => DescribePart::Columns,
            CatalogAnswer::Keys(_) => DescribePart::Keys,
            CatalogAnswer::ForeignKeys(_) => DescribePart::ForeignKeys,
            CatalogAnswer::Indexes(_) => DescribePart::Indexes,
            CatalogAnswer::IndexColumns(_) => DescribePart::IndexColumns,
        }
    }
}

/// One column as the catalog declares it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CatalogColumn {
    pub name: String,
    /// The declared type, as the source spells it; what the core infers
    /// from is the data, not this.
    pub declared_type: String,
    /// Declared `NOT NULL`.
    pub not_null: bool,
    /// Part of the declared primary key.
    pub primary_key: bool,
    /// Kept unique by the source itself, beside any key it declares: a
    /// SQLite rowid alias (an `INTEGER PRIMARY KEY`), which no index
    /// lists.
    pub unique: bool,
    /// The data type the declared type fixes, where the dialect's reader
    /// reads one from it and the values cannot say: SQLite's, whose driver
    /// types a column by its values, so a column holding none reads as
    /// integers whatever it declares (the reference's `_to_dtype`: `INT`
    /// an integer, `TEXT`, `CHAR` or `CLOB` text, `REAL`, `FLOA` or `DOUB`
    /// a float). `None` where the values' type is the column's.
    pub dtype: Option<Dtype>,
}

/// What a key the catalog declares is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyKind {
    Primary,
    Unique,
    /// A constraint of another kind, which the reader leaves alone.
    Other,
}

/// One key the catalog declares on a table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CatalogKey {
    pub kind: KeyKind,
    /// The key's columns; none where they are read by `index`.
    pub columns: Vec<String>,
    /// The index the key is, where its columns are read by its name
    /// (`index_columns`), as SQLite's catalog lists a unique index; the
    /// key is read whatever its kind, as the reference reads it.
    pub index: Option<String>,
}

/// One row of a declared foreign key, as the dialect's reader gives it:
/// either one column pair of the key, or the whole key, every pair in
/// `columns` and `referenced_columns` in order. The rows of one key share
/// `key`, so a reader groups them by it; a key is one pair only when it
/// has one row of one pair.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CatalogForeignKey {
    /// The key the row belongs to: the same number for each of its rows.
    pub key: i64,
    pub referenced_table: String,
    pub columns: Vec<String>,
    pub referenced_columns: Vec<String>,
}

/// One index of the source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CatalogIndex {
    pub table: String,
    pub columns: IndexColumns,
}

/// The columns an index covers, as the dialect's reader resolves them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IndexColumns {
    /// The columns, in order, each resolved; one for an index over exactly
    /// one.
    Resolved(Vec<String>),
    /// What the index covers is not resolved: an expression list the
    /// reader cannot split without guessing (DuckDB's, holding a comma), or
    /// a list naming a column it does not give. It covers no column the
    /// core counts, never a guess at its pieces.
    Unresolved,
    /// The index's name only: its columns are read by it
    /// (`index_columns`), as SQLite's catalog lists a table's indexes.
    Named(String),
}
