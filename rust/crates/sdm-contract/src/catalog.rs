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
//! column pairs and an index's columns, and never a dialect's spelling of
//! them. These types never travel: a host answers in Arrow.

use crate::data::DescribePart;

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
    /// `indexes`: each index of the source.
    Indexes(Vec<CatalogIndex>),
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
    pub columns: Vec<String>,
}

/// One column pair of a declared foreign key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CatalogForeignKey {
    /// The key the pair belongs to: the same number for each of its pairs.
    pub key: i64,
    pub referenced_table: String,
    pub columns: Vec<String>,
    pub referenced_columns: Vec<String>,
}

/// One index of the source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CatalogIndex {
    pub table: String,
    /// The columns the index covers, in order, as the dialect's reader
    /// tells them apart; one column for an index over exactly one.
    pub columns: Vec<String>,
}
