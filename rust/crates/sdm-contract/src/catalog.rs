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
use crate::view::ViewKind;

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
    /// `namespace`: the parts a relation of one part is read under,
    /// outermost first; none where the source names its tables alone.
    Namespace(Vec<String>),
    /// `view_definition`: the one semantic model asked for.
    ViewDefinition(ViewDefinition),
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
            CatalogAnswer::Namespace(_) => DescribePart::Namespace,
            CatalogAnswer::ViewDefinition(_) => DescribePart::ViewDefinition,
        }
    }
}

/// A semantic model's definition as its warehouse describes it (T3.27,
/// design S5), read as the old client reads it: a Databricks metric view's
/// `DESCRIBE TABLE EXTENDED` (`DB/metric_view.py:1389-1475`), a Snowflake
/// semantic view's YAML alone. A relation that is no semantic model of the
/// dialect's kind, or gives no definition, is refused by the dialect's
/// reader by name, never read as an empty one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ViewDefinition {
    /// The kind of semantic model it is, as the dialect reads one
    /// (Databricks metric views, Snowflake semantic views): the core reads
    /// the definition by it, never by its contents.
    pub kind: ViewKind,
    /// The YAML definition, as written.
    pub text: String,
    /// Each parameter's type, by name, in order, as the warehouse describes
    /// it (Databricks's `# Parameters`), a later one of a name replacing an
    /// earlier in its place; none where it describes none.
    pub parameters: Vec<(String, String)>,
    /// Where the view is: the namespace parts the warehouse names it under
    /// (Databricks's `Catalog` and `Database`), outermost first, each `None`
    /// where it says none, which fill the parts a base table the definition
    /// names leaves out; empty for a dialect whose definition names its
    /// base tables in full (Snowflake's).
    pub located: Vec<Option<String>>,
    /// Each column's type, by name, in order, as the warehouse describes it
    /// (a metric view's dimensions, its measures left out); none where it
    /// describes none.
    pub columns: Vec<(String, String)>,
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
    /// Part of the primary key and still free to hold a null: SQLite's
    /// primary key, unless it is the rowid alias or declared `NOT NULL`
    /// (`not_null` says which), so its nullability is the catalog's; every
    /// other source's primary key holds none.
    pub key_nullable: bool,
    /// Kept unique by the source itself, beside any key it declares: a
    /// SQLite rowid alias (an `INTEGER PRIMARY KEY`), which no index
    /// lists.
    pub unique: bool,
    /// The data type the declared type fixes, where the dialect's reader
    /// reads one from it: SQLite's, which stores any value in any column,
    /// so the reference types a column by its declaration first
    /// (`_to_dtype`: `INT` an integer, `TEXT`, `CHAR` or `CLOB` text,
    /// `REAL`, `FLOA` or `DOUB` a float), whatever its values, and a column
    /// holding none reads as integers by its driver whatever it declares.
    /// `None` where the values' type is the column's (another declaration,
    /// an `INTERVAL` by the owner's KO2-62, every other source).
    pub dtype: Option<Dtype>,
}

/// What a key the catalog declares is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyKind {
    Primary,
    Unique,
    /// A unique index over only the rows its condition covers (SQLite's
    /// partial index): the reference holds a column it keeps unique so
    /// (`unique_keys`), but no constraint of several columns.
    PartialUnique,
    /// A foreign key, by this table's columns: a key only to a column of
    /// numbers handed over as its exact text
    /// ([`crate::data::EXACT_NUMBER_MARKER`]), which it keeps as text.
    Foreign,
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
    /// Whether the source holds its rows to the key: a primary or unique
    /// key DuckDB, SQLite or Postgres declares is enforced; one Databricks
    /// declares is informational, never enforced, so its rows may repeat a
    /// value (contract v28, T3.4). Only an enforced key lets a check of
    /// the key's rows, or a lookup's removal of repeated rows, be skipped,
    /// as the reference's engine reads its `_unique_constraints`.
    pub enforced: bool,
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
    pub columns: IndexCoverage,
}

/// The columns an index covers, as the dialect's reader resolves them
/// (named apart from the `index_columns` part, [`DescribePart::IndexColumns`],
/// which reads a [`IndexCoverage::Named`] index's columns, and its answer,
/// [`CatalogAnswer::IndexColumns`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IndexCoverage {
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
