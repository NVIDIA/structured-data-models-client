// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The catalog's answers, read in the shapes the contract fixes for them
//! (`sdm_contract::data::results`); the core has checked each answer's
//! schema against that shape before it gets here (decision H16, check (b)).

use arrow_array::cast::AsArray;
use arrow_array::types::{Int32Type, Int64Type};
use arrow_array::{Array, ArrayRef, RecordBatch};
use sdm_contract::data::results;
use sdm_contract::{CoreError, Detail, ErrorCode, ErrorKind};

/// One column as the catalog declares it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CatalogColumn {
    pub name: String,
    /// Declared `NOT NULL`.
    pub not_null: bool,
    /// Part of the declared primary key.
    pub primary_key: bool,
}

/// One constraint the catalog declares on a table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Constraint {
    /// `PRIMARY KEY`, `UNIQUE`, or a kind the reader leaves alone.
    pub kind: String,
    pub columns: Vec<String>,
}

/// One column pair of a declared foreign key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ForeignKeyPart {
    /// The constraint the pair belongs to.
    pub constraint: i64,
    pub referenced_table: String,
    pub columns: Vec<String>,
    pub referenced_columns: Vec<String>,
}

/// One row of a `key_counts` answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KeyCount {
    /// The key's position in the request.
    pub position: i32,
    pub rows: i64,
    pub distinct: i64,
    pub missing: i64,
}

/// The column `name` of a batch, which the checked shape guarantees.
fn named<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a ArrayRef, CoreError> {
    batch.column_by_name(name).ok_or_else(|| {
        CoreError::new(
            ErrorKind::Internal,
            ErrorCode::InternalError,
            "a catalog answer lacks a column its checked shape holds",
        )
        .with_field("column", Detail::text(name))
    })
}

fn text(array: &ArrayRef, row: usize) -> Option<String> {
    let strings = array.as_string::<i32>();
    (!strings.is_null(row)).then(|| strings.value(row).to_owned())
}

fn texts(array: &ArrayRef, row: usize) -> Vec<String> {
    let lists = array.as_list::<i32>();
    if lists.is_null(row) {
        return vec![];
    }
    let items = lists.value(row);
    let strings = items.as_string::<i32>();
    (0..strings.len())
        .filter(|&i| !strings.is_null(i))
        .map(|i| strings.value(i).to_owned())
        .collect()
}

fn flag(array: &ArrayRef, row: usize) -> bool {
    let flags = array.as_boolean();
    !flags.is_null(row) && flags.value(row)
}

/// A `describe` `tables` answer: each table's name, in the order given.
pub fn table_names(batches: &[RecordBatch]) -> Result<Vec<String>, CoreError> {
    let mut names = Vec::new();
    for batch in batches {
        let column = named(batch, results::TABLE_NAME)?;
        names.extend((0..batch.num_rows()).filter_map(|row| text(column, row)));
    }
    Ok(names)
}

/// A `describe` `columns` answer: each column, in the table's order.
pub fn columns(batches: &[RecordBatch]) -> Result<Vec<CatalogColumn>, CoreError> {
    let mut found = Vec::new();
    for batch in batches {
        let name = named(batch, results::COLUMN_NAME)?;
        let not_null = named(batch, results::COLUMN_NOT_NULL)?;
        let primary_key = named(batch, results::COLUMN_PRIMARY_KEY)?;
        for row in 0..batch.num_rows() {
            if let Some(name) = text(name, row) {
                found.push(CatalogColumn {
                    name,
                    not_null: flag(not_null, row),
                    primary_key: flag(primary_key, row),
                });
            }
        }
    }
    Ok(found)
}

/// A `describe` `keys` answer: each constraint.
pub fn constraints(batches: &[RecordBatch]) -> Result<Vec<Constraint>, CoreError> {
    let mut found = Vec::new();
    for batch in batches {
        let kind = named(batch, results::CONSTRAINT_TYPE)?;
        let columns = named(batch, results::CONSTRAINT_COLUMNS)?;
        for row in 0..batch.num_rows() {
            found.push(Constraint {
                kind: text(kind, row).unwrap_or_default(),
                columns: texts(columns, row),
            });
        }
    }
    Ok(found)
}

/// A `describe` `foreign_keys` answer: each column pair.
pub fn foreign_keys(batches: &[RecordBatch]) -> Result<Vec<ForeignKeyPart>, CoreError> {
    let mut found = Vec::new();
    for batch in batches {
        let index = named(batch, results::CONSTRAINT_INDEX)?.as_primitive::<Int64Type>();
        let table = named(batch, results::REFERENCED_TABLE)?;
        let columns = named(batch, results::CONSTRAINT_COLUMNS)?;
        let referenced = named(batch, results::REFERENCED_COLUMNS)?;
        for row in 0..batch.num_rows() {
            found.push(ForeignKeyPart {
                constraint: index.value(row),
                referenced_table: text(table, row).unwrap_or_default(),
                columns: texts(columns, row),
                referenced_columns: texts(referenced, row),
            });
        }
    }
    Ok(found)
}

/// A `key_counts` answer: one row per key.
pub fn key_counts(batches: &[RecordBatch]) -> Result<Vec<KeyCount>, CoreError> {
    let mut found = Vec::new();
    for batch in batches {
        let position = named(batch, results::KEY_POSITION)?.as_primitive::<Int32Type>();
        let rows = named(batch, results::KEY_ROWS)?.as_primitive::<Int64Type>();
        let distinct = named(batch, results::KEY_DISTINCT)?.as_primitive::<Int64Type>();
        let missing = named(batch, results::KEY_MISSING)?.as_primitive::<Int64Type>();
        for row in 0..batch.num_rows() {
            found.push(KeyCount {
                position: position.value(row),
                rows: rows.value(row),
                distinct: distinct.value(row),
                missing: missing.value(row),
            });
        }
    }
    Ok(found)
}
