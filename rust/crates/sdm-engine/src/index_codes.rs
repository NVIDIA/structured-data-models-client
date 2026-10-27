// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Admission of original-row-aligned dictionary codes for native index facts.
//!
//! This module has no sampling, ordering or native runtime policy.
//! Dictionary identity and values remain separately owned Arrow attachments.
//! A caller must retain those attachments with the same admitted frame snapshot.

use arrow_array::{ArrayRef, Int8Array, Int16Array, Int32Array, Int64Array};
use arrow_schema::DataType;

/// The actual signed storage width, retained independently of observed labels.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CodeWidth {
    I8,
    I16,
    I32,
    I64,
}

/// A malformed fact attachment, not an ordinary native-label refusal.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum InvalidCodes {
    RowCount {
        expected: usize,
        actual: usize,
    },
    PhysicalType(DataType),
    NullCode {
        row: usize,
    },
    DictionaryBounds {
        row: usize,
        code: i64,
        dictionary_rows: usize,
    },
    RowBounds {
        row: usize,
        source_rows: usize,
    },
}

/// One categorical dictionary or MultiIndex level's original row codes.
///
/// Every code is exactly -1 (missing), or an index into the entire dictionary.
/// Unused dictionary entries are retained and still constrain valid codes.
/// Arrow nulls are not a second encoding of missing codes.
#[derive(Clone, Debug)]
pub struct IndexCodes {
    values: ArrayRef,
    width: CodeWidth,
    dictionary_rows: usize,
}

impl IndexCodes {
    /// Validate all rows before a core selects a filtered or sampled subset.
    pub fn admit(
        values: ArrayRef,
        dictionary_rows: usize,
        source_rows: usize,
    ) -> Result<Self, InvalidCodes> {
        if values.len() != source_rows {
            return Err(InvalidCodes::RowCount {
                expected: source_rows,
                actual: values.len(),
            });
        }
        let width = match values.data_type() {
            DataType::Int8 => CodeWidth::I8,
            DataType::Int16 => CodeWidth::I16,
            DataType::Int32 => CodeWidth::I32,
            DataType::Int64 => CodeWidth::I64,
            other => return Err(InvalidCodes::PhysicalType(other.clone())),
        };
        let facts = Self {
            values,
            width,
            dictionary_rows,
        };
        for row in 0..source_rows {
            if facts.values.is_null(row) {
                return Err(InvalidCodes::NullCode { row });
            }
            let code = facts.raw_code(row);
            if code != -1
                && (code < 0 || !usize::try_from(code).is_ok_and(|value| value < dictionary_rows))
            {
                return Err(InvalidCodes::DictionaryBounds {
                    row,
                    code,
                    dictionary_rows,
                });
            }
        }
        Ok(facts)
    }

    pub fn width(&self) -> CodeWidth {
        self.width
    }

    pub fn dictionary_rows(&self) -> usize {
        self.dictionary_rows
    }

    pub fn source_rows(&self) -> usize {
        self.values.len()
    }

    /// Keep the admitted physical codes for zero-copy attachment ownership.
    pub fn values(&self) -> &ArrayRef {
        &self.values
    }

    /// Read one original row without defining a sorted rank for missing codes.
    pub fn code(&self, row: usize) -> Result<Option<usize>, InvalidCodes> {
        if row >= self.source_rows() {
            return Err(InvalidCodes::RowBounds {
                row,
                source_rows: self.source_rows(),
            });
        }
        let code = self.raw_code(row);
        Ok(if code == -1 {
            None
        } else {
            // Admission proved the positive code fits usize and the dictionary.
            Some(usize::try_from(code).expect("admitted dictionary code"))
        })
    }

    /// Project precisely the core's chosen rows, preserving their current order.
    ///
    /// The host never computes this subset or supplies a final sort permutation.
    /// Repeated selections remain repeated; uniqueness is a sampler concern.
    pub fn selected(&self, rows: &[usize]) -> Result<Vec<Option<usize>>, InvalidCodes> {
        rows.iter().map(|&row| self.code(row)).collect()
    }

    fn raw_code(&self, row: usize) -> i64 {
        match self.width {
            CodeWidth::I8 => i64::from(
                self.values
                    .as_any()
                    .downcast_ref::<Int8Array>()
                    .expect("admitted int8 codes")
                    .value(row),
            ),
            CodeWidth::I16 => i64::from(
                self.values
                    .as_any()
                    .downcast_ref::<Int16Array>()
                    .expect("admitted int16 codes")
                    .value(row),
            ),
            CodeWidth::I32 => i64::from(
                self.values
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .expect("admitted int32 codes")
                    .value(row),
            ),
            CodeWidth::I64 => self
                .values
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("admitted int64 codes")
                .value(row),
        }
    }
}
