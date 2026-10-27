// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The count of keys, read in the shape the contract fixes for it
//! (`sdm_contract::data::results`); the core has checked the answer's
//! schema against that shape before it gets here (decision H16, check
//! (b)). The catalog's own answers reach this crate already read, in no
//! dialect's shape (`sdm_contract::catalog`, decision E54).

use arrow_array::cast::AsArray;
use arrow_array::types::{Int32Type, Int64Type};
use arrow_array::{ArrayRef, RecordBatch};
use sdm_contract::data::results;
use sdm_contract::{CoreError, Detail, Sentence};

/// One row of a `key_counts` answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KeyCount {
    /// The key's position in the request.
    pub position: i32,
    pub rows: i64,
    pub distinct: i64,
    pub missing: i64,
}

/// The column of a batch the core names by its own alias, `name`: the
/// column of that name, else the one column whose name is it in another
/// case, as a source folds an unquoted alias (Snowflake upper-cases the
/// reference's `AS k`, `as table_index`); `None` for none, or for two.
pub fn result_column<'a>(batch: &'a RecordBatch, name: &str) -> Option<&'a ArrayRef> {
    batch.column_by_name(name).or_else(|| {
        let mut found = batch
            .schema_ref()
            .fields()
            .iter()
            .enumerate()
            .filter(|(_, field)| field.name().eq_ignore_ascii_case(name));
        match (found.next(), found.next()) {
            (Some((at, _)), None) => Some(batch.column(at)),
            _ => None,
        }
    })
}

/// The column `name` of a batch, which the checked shape guarantees, as
/// [`result_column`] finds it.
fn named<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a ArrayRef, CoreError> {
    result_column(batch, name).ok_or_else(|| {
        CoreError::new(
            Sentence::Internal,
            "a catalog answer lacks a column its checked shape holds",
        )
        .with_field("column", Detail::text(name))
    })
}

/// The integers of column `name`: of the type the checked shape holds for
/// it, `int64`, or for a position `int32` too, the width DuckDB writes a
/// literal in (SQLite's driver reads every integer as `int64`); a null, or
/// another type, is the answer's error, never a panic.
fn integers(batch: &RecordBatch, name: &str) -> Result<Vec<i64>, CoreError> {
    let column = named(batch, name)?;
    let retyped = || {
        CoreError::new(
            Sentence::Internal,
            "a key count's column is not of its checked type",
        )
        .with_field("column", Detail::text(name))
    };
    if column.null_count() > 0 {
        return Err(retyped());
    }
    if let Some(values) = column.as_primitive_opt::<Int64Type>() {
        return Ok(values.values().to_vec());
    }
    if let Some(values) = column.as_primitive_opt::<Int32Type>() {
        return Ok(values.values().iter().map(|v| i64::from(*v)).collect());
    }
    Err(retyped())
}

/// A `key_counts` answer: one row per key.
pub fn key_counts(batches: &[RecordBatch]) -> Result<Vec<KeyCount>, CoreError> {
    let mut found = Vec::new();
    for batch in batches {
        let position = integers(batch, results::KEY_POSITION)?;
        let rows = integers(batch, results::KEY_ROWS)?;
        let distinct = integers(batch, results::KEY_DISTINCT)?;
        let missing = integers(batch, results::KEY_MISSING)?;
        for row in 0..batch.num_rows() {
            found.push(KeyCount {
                position: i32::try_from(position[row]).unwrap_or(i32::MAX),
                rows: rows[row],
                distinct: distinct[row],
                missing: missing[row],
            });
        }
    }
    Ok(found)
}
