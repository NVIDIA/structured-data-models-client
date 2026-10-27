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

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{UInt8Array, new_null_array};
    use std::sync::Arc;

    #[test]
    fn code_width_and_unused_dictionary_survive_selection() {
        let families: [(ArrayRef, CodeWidth, usize); 7] = [
            (
                Arc::new(Int8Array::from(vec![3, -1, 0, 3])),
                CodeWidth::I8,
                126,
            ),
            (
                Arc::new(Int16Array::from(vec![3, -1, 0, 3])),
                CodeWidth::I16,
                127,
            ),
            (
                Arc::new(Int16Array::from(vec![3, -1, 0, 3])),
                CodeWidth::I16,
                128,
            ),
            (
                Arc::new(Int16Array::from(vec![3, -1, 0, 3])),
                CodeWidth::I16,
                32766,
            ),
            (
                Arc::new(Int32Array::from(vec![3, -1, 0, 3])),
                CodeWidth::I32,
                32767,
            ),
            (
                Arc::new(Int32Array::from(vec![3, -1, 0, 3])),
                CodeWidth::I32,
                32768,
            ),
            (
                Arc::new(Int64Array::from(vec![3, -1, 0, 3])),
                CodeWidth::I64,
                2147483647,
            ),
        ];
        for (input, width, dictionary_rows) in families {
            let facts = IndexCodes::admit(input.clone(), dictionary_rows, 4).unwrap();
            assert_eq!(facts.width(), width);
            assert_eq!(facts.dictionary_rows(), dictionary_rows);
            assert_eq!(facts.source_rows(), 4);
            assert!(Arc::ptr_eq(facts.values(), &input));
            assert_eq!(
                facts.selected(&[3, 1, 0, 3, 2]).unwrap(),
                [Some(3), None, Some(3), Some(3), Some(0)]
            );
            assert_eq!(facts.selected(&[]).unwrap(), []);
        }
    }

    #[test]
    fn admission_checks_unselected_rows_too() {
        let facts = IndexCodes::admit(Arc::new(Int8Array::from(vec![0, 1, 4])), 4, 3);
        assert!(matches!(
            facts,
            Err(InvalidCodes::DictionaryBounds {
                row: 2,
                code: 4,
                dictionary_rows: 4
            })
        ));
        for code in [-2, i64::MIN, i64::MAX] {
            let error = IndexCodes::admit(Arc::new(Int64Array::from(vec![code])), 4, 1);
            assert!(matches!(
                error,
                Err(InvalidCodes::DictionaryBounds { row: 0, .. })
            ));
        }
    }

    #[test]
    fn missing_codes_and_empty_dictionaries_have_one_encoding() {
        let facts = IndexCodes::admit(Arc::new(Int8Array::from(vec![-1, -1])), 0, 2).unwrap();
        assert_eq!(facts.selected(&[1, 0]).unwrap(), [None, None]);
        let empty = IndexCodes::admit(Arc::new(Int16Array::from(Vec::<i16>::new())), 0, 0).unwrap();
        assert_eq!(empty.source_rows(), 0);
        let invalid = IndexCodes::admit(new_null_array(&DataType::Int8, 2), 1, 2);
        assert!(matches!(invalid, Err(InvalidCodes::NullCode { row: 0 })));
        let invalid = IndexCodes::admit(Arc::new(Int8Array::from(vec![0])), 0, 1);
        assert!(matches!(
            invalid,
            Err(InvalidCodes::DictionaryBounds {
                row: 0,
                code: 0,
                dictionary_rows: 0
            })
        ));
    }

    #[test]
    fn mismatched_frames_and_out_of_bounds_rows_are_rejected() {
        let input: ArrayRef = Arc::new(Int8Array::from(vec![0, 1]));
        let invalid = IndexCodes::admit(input.clone(), 2, 3);
        assert!(matches!(
            invalid,
            Err(InvalidCodes::RowCount {
                expected: 3,
                actual: 2
            })
        ));
        let facts = IndexCodes::admit(input, 2, 2).unwrap();
        assert_eq!(
            facts.selected(&[0, 2]),
            Err(InvalidCodes::RowBounds {
                row: 2,
                source_rows: 2
            })
        );
        let invalid = IndexCodes::admit(Arc::new(UInt8Array::from(vec![0, 1])), 2, 2);
        assert!(matches!(
            invalid,
            Err(InvalidCodes::PhysicalType(DataType::UInt8))
        ));
    }
}
