// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Reading catalog answers into the contract's dialect-neutral
//! [`CatalogAnswer`] (decision E54): the column readers each dialect's
//! reader is built of, and the reader of a host provider's answer, which
//! is in the contract's own shape (`catalog_columns`).
//!
//! The core calls a reader after check (b) held the answer to its plan's
//! schema (decision H16), so each column is there, of its type, and holds
//! no null where the plan takes none. A reader still never panics on an
//! answer that is not so: a column missing or of another type is
//! `INTERNAL_ERROR`, naming the column.

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{Array, ArrayRef, BooleanArray, Int64Array, ListArray, RecordBatch, StringArray};
use sdm_contract::catalog::{
    CatalogAnswer, CatalogColumn, CatalogForeignKey, CatalogIndex, CatalogKey, IndexCoverage,
    KeyKind,
};
use sdm_contract::data::{DescribePart, catalog_columns as c};
use sdm_contract::{CoreError, Detail, ErrorCode, ErrorKind};

fn unread(name: &str, message: &str) -> CoreError {
    CoreError::new(ErrorKind::Internal, ErrorCode::InternalError, message)
        .with_field("column", Detail::text(name))
}

fn column<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a ArrayRef, CoreError> {
    batch.column_by_name(name).ok_or_else(|| {
        unread(
            name,
            "a catalog answer lacks a column its checked shape holds",
        )
    })
}

fn retyped(name: &str) -> CoreError {
    unread(name, "a catalog answer's column is not of its checked type")
}

/// The text column `name` of `batch`.
pub(crate) fn strings<'a>(
    batch: &'a RecordBatch,
    name: &str,
) -> Result<&'a StringArray, CoreError> {
    column(batch, name)?
        .as_string_opt::<i32>()
        .ok_or_else(|| retyped(name))
}

/// The Boolean column `name` of `batch`.
pub(crate) fn flags<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a BooleanArray, CoreError> {
    column(batch, name)?
        .as_boolean_opt()
        .ok_or_else(|| retyped(name))
}

/// The `int64` column `name` of `batch`.
pub(crate) fn int64s<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a Int64Array, CoreError> {
    column(batch, name)?
        .as_primitive_opt::<Int64Type>()
        .ok_or_else(|| retyped(name))
}

/// The list-of-text column `name` of `batch`.
pub(crate) fn lists<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a ListArray, CoreError> {
    let lists = column(batch, name)?
        .as_list_opt::<i32>()
        .ok_or_else(|| retyped(name))?;
    if lists.values().as_string_opt::<i32>().is_none() {
        return Err(retyped(name));
    }
    Ok(lists)
}

/// Row `row`'s text, `None` where it is null.
pub(crate) fn text(strings: &StringArray, row: usize) -> Option<String> {
    (!strings.is_null(row)).then(|| strings.value(row).to_owned())
}

/// Row `row`'s names, in order, leaving out a null one; none where the
/// list is null.
pub(crate) fn names(lists: &ListArray, row: usize) -> Vec<String> {
    if lists.is_null(row) {
        return vec![];
    }
    let items = lists.value(row);
    let Some(strings) = items.as_string_opt::<i32>() else {
        return vec![];
    };
    (0..strings.len())
        .filter(|&i| !strings.is_null(i))
        .map(|i| strings.value(i).to_owned())
        .collect()
}

/// Row `row`'s index columns as a host provider gives them: resolved when
/// the list and each name in it are given, unresolved where the list or a
/// name is null, never the names left over.
fn index_columns(lists: &ListArray, row: usize) -> IndexCoverage {
    if lists.is_null(row) {
        return IndexCoverage::Unresolved;
    }
    let items = lists.value(row);
    match items.as_string_opt::<i32>() {
        Some(strings) if strings.null_count() == 0 => IndexCoverage::Resolved(names(lists, row)),
        _ => IndexCoverage::Unresolved,
    }
}

/// Row `row`'s flag, false where it is null.
pub(crate) fn flag(flags: &BooleanArray, row: usize) -> bool {
    !flags.is_null(row) && flags.value(row)
}

/// Each table's name, in order, from the text column `name`.
pub(crate) fn tables(batches: &[RecordBatch], name: &str) -> Result<CatalogAnswer, CoreError> {
    let mut found = Vec::new();
    for batch in batches {
        let names = strings(batch, name)?;
        found.extend((0..batch.num_rows()).filter_map(|row| text(names, row)));
    }
    Ok(CatalogAnswer::Tables(found))
}

/// The columns of a `columns` answer, by role.
pub(crate) struct ColumnNames<'a> {
    pub name: &'a str,
    pub declared_type: &'a str,
    pub not_null: &'a str,
    pub primary_key: &'a str,
}

/// Each column, in the table's order; a row with no name is left out.
pub(crate) fn columns(
    batches: &[RecordBatch],
    names: &ColumnNames<'_>,
) -> Result<CatalogAnswer, CoreError> {
    let mut found = Vec::new();
    for batch in batches {
        let name = strings(batch, names.name)?;
        let declared_type = strings(batch, names.declared_type)?;
        let not_null = flags(batch, names.not_null)?;
        let primary_key = flags(batch, names.primary_key)?;
        for row in 0..batch.num_rows() {
            if let Some(name) = text(name, row) {
                found.push(CatalogColumn {
                    canonical: None,
                    name,
                    declared_type: text(declared_type, row).unwrap_or_default(),
                    not_null: flag(not_null, row),
                    primary_key: flag(primary_key, row),
                    key_nullable: false,
                    unique: false,
                    dtype: None,
                });
            }
        }
    }
    Ok(CatalogAnswer::Columns(found))
}

/// Each declared key: its kind, from the text column `kind` as `kind_of`
/// reads it (a null kind is another kind), and its columns.
pub(crate) fn keys(
    batches: &[RecordBatch],
    kind: &str,
    columns: &str,
    kind_of: fn(&str) -> KeyKind,
) -> Result<CatalogAnswer, CoreError> {
    let mut found = Vec::new();
    for batch in batches {
        let kinds = strings(batch, kind)?;
        let lists = lists(batch, columns)?;
        for row in 0..batch.num_rows() {
            found.push(CatalogKey {
                kind: text(kinds, row).map_or(KeyKind::Other, |k| kind_of(&k)),
                columns: names(lists, row),
                index: None,
            });
        }
    }
    Ok(CatalogAnswer::Keys(found))
}

/// The columns of a `foreign_keys` answer, by role.
pub(crate) struct ForeignKeyNames<'a> {
    pub key: &'a str,
    pub referenced_table: &'a str,
    pub columns: &'a str,
    pub referenced_columns: &'a str,
}

/// Each column pair of each declared foreign key.
pub(crate) fn foreign_keys(
    batches: &[RecordBatch],
    names: &ForeignKeyNames<'_>,
) -> Result<CatalogAnswer, CoreError> {
    let mut found = Vec::new();
    for batch in batches {
        let key = int64s(batch, names.key)?;
        let table = strings(batch, names.referenced_table)?;
        let columns = lists(batch, names.columns)?;
        let referenced = lists(batch, names.referenced_columns)?;
        for row in 0..batch.num_rows() {
            found.push(CatalogForeignKey {
                key: key.value(row),
                referenced_table: text(table, row).unwrap_or_default(),
                columns: self::names(columns, row),
                referenced_columns: self::names(referenced, row),
            });
        }
    }
    Ok(CatalogAnswer::ForeignKeys(found))
}

/// A host provider's answer, in the contract's own shape
/// (`catalog_columns`), read as it is.
pub(crate) fn host(
    part: DescribePart,
    batches: &[RecordBatch],
) -> Result<CatalogAnswer, CoreError> {
    match part {
        DescribePart::Tables => tables(batches, c::TABLE),
        DescribePart::Columns => columns(
            batches,
            &ColumnNames {
                name: c::NAME,
                declared_type: c::TYPE,
                not_null: c::NOT_NULL,
                primary_key: c::PRIMARY_KEY,
            },
        ),
        DescribePart::Keys => keys(batches, c::KIND, c::COLUMNS, |kind| match kind {
            c::KIND_PRIMARY => KeyKind::Primary,
            c::KIND_UNIQUE => KeyKind::Unique,
            c::KIND_FOREIGN => KeyKind::Foreign,
            _ => KeyKind::Other,
        }),
        DescribePart::ForeignKeys => foreign_keys(
            batches,
            &ForeignKeyNames {
                key: c::KEY,
                referenced_table: c::REFERENCED_TABLE,
                columns: c::COLUMNS,
                referenced_columns: c::REFERENCED_COLUMNS,
            },
        ),
        DescribePart::Indexes => {
            let mut found = Vec::new();
            for batch in batches {
                let table = strings(batch, c::TABLE)?;
                let lists = lists(batch, c::COLUMNS)?;
                for row in 0..batch.num_rows() {
                    if let Some(table) = text(table, row) {
                        found.push(CatalogIndex {
                            table,
                            columns: index_columns(lists, row),
                        });
                    }
                }
            }
            Ok(CatalogAnswer::Indexes(found))
        }
        DescribePart::IndexColumns => {
            let mut found = Vec::new();
            for batch in batches {
                let names = strings(batch, c::NAME)?;
                found.extend((0..batch.num_rows()).map(|row| text(names, row)));
            }
            Ok(CatalogAnswer::IndexColumns(found))
        }
        DescribePart::Namespace => {
            let mut rows = Vec::new();
            for batch in batches {
                let lists = lists(batch, c::PARTS)?;
                for row in 0..batch.num_rows() {
                    let parts = lists.value(row);
                    let parts = parts.as_string::<i32>();
                    rows.push((0..parts.len()).map(|at| text(parts, at)).collect());
                }
            }
            namespace(rows)
        }
    }
}

/// The source's current namespace from the rows of its answer, which must
/// be one: an answer of another count of rows is not the namespace.
pub(crate) fn namespace(mut rows: Vec<Vec<Option<String>>>) -> Result<CatalogAnswer, CoreError> {
    match rows.len() {
        1 => Ok(CatalogAnswer::Namespace(rows.remove(0))),
        _ => Err(unread(c::PARTS, "a namespace answer does not hold one row")),
    }
}
