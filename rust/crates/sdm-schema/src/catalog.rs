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
use sdm_contract::{CoreError, Detail, ErrorCode, ErrorKind};

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
