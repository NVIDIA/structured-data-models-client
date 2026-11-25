// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Reading catalog answers into the contract's dialect-neutral
//! [`CatalogAnswer`]: the column readers each dialect's reader is built of,
//! and the reader of the contract's own shape ([`standard`],
//! `catalog_columns`), which every provider answers a `describe` in.
//!
//! The core reads an answer after check (b) held it to its plan's schema,
//! so each column is there, of its type, and holds no null where the plan
//! takes none. A reader still never panics on an answer that is not so: a
//! column missing or of another type is `INTERNAL_ERROR`, naming the
//! column.

use crate::catalog::{
    CatalogAnswer, CatalogColumn, CatalogForeignKey, CatalogIndex, CatalogKey, IndexCoverage,
    KeyKind,
};
use crate::data::{DescribePart, catalog_columns as c, views_unread};
use crate::error::{CoreError, Detail, Sentence};
use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{Array, ArrayRef, BooleanArray, Int64Array, ListArray, RecordBatch, StringArray};

fn unread(name: &str, message: &str) -> CoreError {
    CoreError::new(Sentence::Internal, message).with_field("column", Detail::text(name))
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
pub fn strings<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a StringArray, CoreError> {
    column(batch, name)?
        .as_string_opt::<i32>()
        .ok_or_else(|| retyped(name))
}

/// The Boolean column `name` of `batch`.
pub fn flags<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a BooleanArray, CoreError> {
    column(batch, name)?
        .as_boolean_opt()
        .ok_or_else(|| retyped(name))
}

/// The `int64` column `name` of `batch`.
pub fn int64s<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a Int64Array, CoreError> {
    column(batch, name)?
        .as_primitive_opt::<Int64Type>()
        .ok_or_else(|| retyped(name))
}

/// The list-of-text column `name` of `batch`.
pub fn lists<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a ListArray, CoreError> {
    let lists = column(batch, name)?
        .as_list_opt::<i32>()
        .ok_or_else(|| retyped(name))?;
    if lists.values().as_string_opt::<i32>().is_none() {
        return Err(retyped(name));
    }
    Ok(lists)
}

/// The column at `at` of `batch` as text, for a reader that reads its
/// answer by position (a warehouse's, whose catalog may name two columns
/// alike): a string column of any width, or a column of nulls (a session
/// with no current schema, a cell the row adapter had none of).
pub fn text_at(batch: &RecordBatch, at: usize) -> Result<StringArray, CoreError> {
    let unread = || {
        CoreError::new(
            Sentence::Internal,
            "a catalog answer's column is missing or not text",
        )
        .with_field("position", Detail::text(at.to_string()))
    };
    let column = batch.columns().get(at).ok_or_else(unread)?;
    if let Some(strings) = column.as_string_opt::<i32>() {
        return Ok(strings.clone());
    }
    if let Some(strings) = column.as_string_opt::<i64>() {
        return Ok(strings.iter().collect());
    }
    if let Some(strings) = column.as_string_view_opt() {
        return Ok(strings.iter().collect());
    }
    if *column.data_type() == arrow_schema::DataType::Null {
        return Ok(StringArray::new_null(column.len()));
    }
    Err(unread())
}

/// Row `row`'s text, `None` where it is null.
pub fn text(strings: &StringArray, row: usize) -> Option<String> {
    (!strings.is_null(row)).then(|| strings.value(row).to_owned())
}

/// Row `row`'s names, in order, leaving out a null one; none where the
/// list is null.
pub fn names(lists: &ListArray, row: usize) -> Vec<String> {
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

/// The optional text column `name` (contract v76), `None` where the
/// answer leaves it out.
fn optional_strings<'a>(
    batch: &'a RecordBatch,
    name: &str,
) -> Result<Option<&'a StringArray>, CoreError> {
    if batch.schema().index_of(name).is_err() {
        return Ok(None);
    }
    strings(batch, name).map(Some)
}

/// The optional flag column `name` (contract v76), `None` where the answer
/// leaves it out.
fn optional_flags<'a>(
    batch: &'a RecordBatch,
    name: &str,
) -> Result<Option<&'a BooleanArray>, CoreError> {
    if batch.schema().index_of(name).is_err() {
        return Ok(None);
    }
    flags(batch, name).map(Some)
}

/// The data type `name` names (`Dtype`'s own name), refused by name where
/// it names none.
fn dtype_named(name: &str) -> Result<crate::ops::Dtype, CoreError> {
    serde_json::from_value(serde_json::Value::String(name.to_owned()))
        .map_err(|_| unread("dtype", "a column's dtype names no data type"))
}

/// Row `row`'s flag, false where it is null.
pub fn flag(flags: &BooleanArray, row: usize) -> bool {
    !flags.is_null(row) && flags.value(row)
}

/// Each table's name, in order, from the text column `name`.
pub fn tables(batches: &[RecordBatch], name: &str) -> Result<CatalogAnswer, CoreError> {
    Ok(CatalogAnswer::Tables(text_names(batches, name)?))
}

/// A `namespace` answer: parts in order, a row with none left out.
pub fn namespace(batches: &[RecordBatch], name: &str) -> Result<CatalogAnswer, CoreError> {
    Ok(CatalogAnswer::Namespace(text_names(batches, name)?))
}

fn text_names(batches: &[RecordBatch], name: &str) -> Result<Vec<String>, CoreError> {
    let mut found = Vec::new();
    for batch in batches {
        let names = strings(batch, name)?;
        found.extend((0..batch.num_rows()).filter_map(|row| text(names, row)));
    }
    Ok(found)
}

/// The columns of a `columns` answer, by role.
pub struct ColumnNames<'a> {
    pub name: &'a str,
    pub declared_type: &'a str,
    pub not_null: &'a str,
    pub primary_key: &'a str,
}

/// Each column, in the table's order; a row with no name is left out.
pub fn columns(
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
pub fn keys(
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
                enforced: true,
            });
        }
    }
    Ok(CatalogAnswer::Keys(found))
}

/// The columns of a `foreign_keys` answer, by role.
pub struct ForeignKeyNames<'a> {
    pub key: &'a str,
    pub referenced_table: &'a str,
    pub columns: &'a str,
    pub referenced_columns: &'a str,
}

/// Each column pair of each declared foreign key.
pub fn foreign_keys(
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

/// A provider's answer, in the contract's own shape (`catalog_columns`),
/// read as it is. A semantic model's definition is not in it: its
/// database's data layer reads it (`DataLayer::read_view`).
pub fn standard(part: DescribePart, batches: &[RecordBatch]) -> Result<CatalogAnswer, CoreError> {
    match part {
        // No view is read here: only a dialect that reads views answers it.
        DescribePart::ViewDefinition => Err(views_unread()),
        DescribePart::Namespace => namespace(batches, c::NAME),
        DescribePart::KeyEnforcement => Ok(CatalogAnswer::KeyEnforcement(text_names(
            batches,
            c::TABLE,
        )?)),
        DescribePart::Tables => tables(batches, c::TABLE),
        DescribePart::Columns => {
            let CatalogAnswer::Columns(mut found) = columns(
                batches,
                &ColumnNames {
                    name: c::NAME,
                    declared_type: c::TYPE,
                    not_null: c::NOT_NULL,
                    primary_key: c::PRIMARY_KEY,
                },
            )?
            else {
                unreachable!("columns answers columns")
            };
            // The optional facts (contract v76), row by row in the order the
            // named rows were read.
            let mut at = found.iter_mut();
            for batch in batches {
                let named = strings(batch, c::NAME)?;
                let key_nullable = optional_flags(batch, c::KEY_NULLABLE)?;
                let unique = optional_flags(batch, c::UNIQUE)?;
                let dtype = optional_strings(batch, c::DTYPE)?;
                for row in 0..batch.num_rows() {
                    if named.is_null(row) {
                        continue;
                    }
                    let Some(column) = at.next() else { break };
                    column.key_nullable = key_nullable.is_some_and(|f| flag(f, row));
                    column.unique = unique.is_some_and(|f| flag(f, row));
                    column.dtype = match dtype.and_then(|d| text(d, row)) {
                        Some(name) => Some(dtype_named(&name)?),
                        None => None,
                    };
                }
            }
            Ok(CatalogAnswer::Columns(found))
        }
        DescribePart::Keys => {
            let CatalogAnswer::Keys(mut found) =
                keys(batches, c::KIND, c::COLUMNS, |kind| match kind {
                    c::KIND_PRIMARY => KeyKind::Primary,
                    c::KIND_UNIQUE => KeyKind::Unique,
                    c::KIND_PARTIAL_UNIQUE => KeyKind::PartialUnique,
                    c::KIND_FOREIGN => KeyKind::Foreign,
                    _ => KeyKind::Other,
                })?
            else {
                unreachable!("keys answers keys")
            };
            let mut at = found.iter_mut();
            for batch in batches {
                let index = optional_strings(batch, c::INDEX)?;
                let enforced = optional_flags(batch, c::ENFORCED)?;
                for row in 0..batch.num_rows() {
                    let Some(key) = at.next() else { break };
                    key.index = index.and_then(|i| text(i, row));
                    if let Some(enforced) = enforced {
                        key.enforced = flag(enforced, row);
                    }
                }
            }
            Ok(CatalogAnswer::Keys(found))
        }
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
                let index = optional_strings(batch, c::INDEX)?;
                let resolved = optional_flags(batch, c::RESOLVED)?;
                for row in 0..batch.num_rows() {
                    if let Some(table) = text(table, row) {
                        let columns = match (index.and_then(|i| text(i, row)), resolved) {
                            (Some(name), _) => IndexCoverage::Named(name),
                            (None, Some(resolved)) if !flag(resolved, row) => {
                                IndexCoverage::Unresolved
                            }
                            _ => index_columns(lists, row),
                        };
                        found.push(CatalogIndex { table, columns });
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
    }
}
