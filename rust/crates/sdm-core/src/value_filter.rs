// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Value filters ([`sdm_contract::data::FilterKind`]) held against rows in
//! memory, where the in-memory fallback answers a sample for a provider
//! that does not apply them itself.
//!
//! A value filter narrows a read and never decides a result: the core
//! checks every row it keeps by the query's own rules. So this reads a
//! filter as "may hold": where a column's type and the value's do not
//! compare plainly here (a decimal, a time, text against a number), the row
//! is kept, never left out.

use std::cmp::Ordering;

use arrow_array::cast::AsArray;
use arrow_array::types::{Float64Type, Int64Type};
use arrow_array::{Array, ArrayRef, RecordBatch};
use arrow_cast::cast;
use arrow_schema::DataType;
use sdm_contract::data::{CompareOp, Filter};
use sdm_contract::{CoreError, Scalar};

use crate::run::internal;

/// For each row of `rows`, whether every one of `filters` may hold for it.
pub(crate) fn may_hold(rows: &RecordBatch, filters: &[Filter]) -> Result<Vec<bool>, CoreError> {
    let mut kept = vec![true; rows.num_rows()];
    for filter in filters {
        for (kept, holds) in kept.iter_mut().zip(holds(rows, filter)?) {
            *kept &= holds;
        }
    }
    Ok(kept)
}

/// For each row, whether `filter` may hold for it.
fn holds(rows: &RecordBatch, filter: &Filter) -> Result<Vec<bool>, CoreError> {
    let n = rows.num_rows();
    Ok(match filter {
        // Not a value filter: the core sends none to the fallback's sample
        // and checks it itself.
        Filter::KeyIn { .. } | Filter::TimeRange { .. } => vec![true; n],
        Filter::NotNull { column } => {
            let array = column_of(rows, column)?;
            (0..n).map(|row| array.is_valid(row)).collect()
        }
        Filter::IsNull { column } => {
            let array = column_of(rows, column)?;
            (0..n).map(|row| array.is_null(row)).collect()
        }
        Filter::Compare { column, op, value } => {
            compared(column_of(rows, column)?, value, |o| meets(*op, o))
        }
        Filter::In { column, values } => {
            let array = column_of(rows, column)?;
            let mut kept = vec![false; n];
            for value in values {
                let equal = compared(array, value, |o| o == Ordering::Equal);
                for (kept, equal) in kept.iter_mut().zip(equal) {
                    *kept |= equal;
                }
            }
            kept
        }
        Filter::AnyOf { filters } => {
            let mut kept = vec![false; n];
            for filter in filters {
                for (kept, holds) in kept.iter_mut().zip(holds(rows, filter)?) {
                    *kept |= holds;
                }
            }
            kept
        }
        Filter::AllOf { filters } => may_hold(rows, filters)?,
    })
}

fn column_of<'a>(rows: &'a RecordBatch, column: &str) -> Result<&'a ArrayRef, CoreError> {
    rows.column_by_name(column)
        .ok_or_else(|| internal("a value filter names a column the scan did not read"))
}

fn meets(op: CompareOp, ordering: Ordering) -> bool {
    match op {
        CompareOp::Eq => ordering == Ordering::Equal,
        CompareOp::Ne => ordering != Ordering::Equal,
        CompareOp::Lt => ordering == Ordering::Less,
        CompareOp::Le => ordering != Ordering::Greater,
        CompareOp::Gt => ordering == Ordering::Greater,
        CompareOp::Ge => ordering != Ordering::Less,
    }
}

/// For each row of `array`, whether its value compared with `value`
/// passes `test`: false for a null, true where the two do not compare
/// plainly here.
fn compared(array: &ArrayRef, value: &Scalar, test: impl Fn(Ordering) -> bool) -> Vec<bool> {
    let n = array.len();
    let unsure = || vec![true; n];
    let each = |ordering: &dyn Fn(usize) -> Option<Ordering>| -> Vec<bool> {
        (0..n)
            .map(|row| array.is_valid(row) && ordering(row).is_none_or(&test))
            .collect()
    };
    let integer_column = array.data_type().is_integer();
    let numeric_column = array.data_type().is_numeric() && !is_decimal(array.data_type());
    match value {
        Scalar::Utf8(text) => match array.data_type() {
            DataType::Utf8 => {
                let strings = array.as_string::<i32>();
                each(&|row| Some(strings.value(row).cmp(text.as_str())))
            }
            DataType::LargeUtf8 => {
                let strings = array.as_string::<i64>();
                each(&|row| Some(strings.value(row).cmp(text.as_str())))
            }
            DataType::Utf8View => {
                let strings = array.as_string_view();
                each(&|row| Some(strings.value(row).cmp(text.as_str())))
            }
            _ => unsure(),
        },
        Scalar::Bool(b) => match array.data_type() {
            DataType::Boolean => {
                let bools = array.as_boolean();
                each(&|row| Some(bools.value(row).cmp(b)))
            }
            _ => unsure(),
        },
        _ => match (integer_constant(value), float_constant(value)) {
            (Some(v), _) if integer_column => match cast(array, &DataType::Int64) {
                // A cast fails on a value past `i64` (a large `u64`): unsure.
                Ok(ints) if ints.null_count() == array.null_count() => {
                    let ints = ints.as_primitive::<Int64Type>().clone();
                    each(&|row| Some(ints.value(row).cmp(&v)))
                }
                _ => unsure(),
            },
            (_, Some(v)) if numeric_column => match cast(array, &DataType::Float64) {
                Ok(floats) => {
                    let floats = floats.as_primitive::<Float64Type>().clone();
                    // A NaN orders with nothing: unsure, kept.
                    each(&|row| floats.value(row).partial_cmp(&v))
                }
                Err(_) => unsure(),
            },
            _ => unsure(),
        },
    }
}

fn is_decimal(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Decimal32(..)
            | DataType::Decimal64(..)
            | DataType::Decimal128(..)
            | DataType::Decimal256(..)
    )
}

fn integer_constant(value: &Scalar) -> Option<i64> {
    match value {
        Scalar::Int8(v) => Some(i64::from(*v)),
        Scalar::Int16(v) => Some(i64::from(*v)),
        Scalar::Int32(v) => Some(i64::from(*v)),
        Scalar::Int64(v) => Some(*v),
        Scalar::UInt8(v) => Some(i64::from(*v)),
        Scalar::UInt16(v) => Some(i64::from(*v)),
        Scalar::UInt32(v) => Some(i64::from(*v)),
        Scalar::UInt64(v) => i64::try_from(*v).ok(),
        _ => None,
    }
}

fn float_constant(value: &Scalar) -> Option<f64> {
    match value {
        Scalar::Float32(v) if v.is_finite() => Some(f64::from(*v)),
        Scalar::Float64(v) if v.is_finite() => Some(*v),
        other => integer_constant(other).map(|v| v as f64),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{Float64Array, Int64Array, StringArray};
    use arrow_schema::{Field, Schema};

    use super::*;

    fn rows() -> RecordBatch {
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("age", DataType::Int64, true),
                Field::new("name", DataType::Utf8, true),
                Field::new("score", DataType::Float64, true),
            ])),
            vec![
                Arc::new(Int64Array::from(vec![Some(80), Some(30), None, Some(77)])),
                Arc::new(StringArray::from(vec![
                    Some("a"),
                    Some("b"),
                    Some("a"),
                    None,
                ])),
                Arc::new(Float64Array::from(vec![
                    Some(1.5),
                    Some(f64::NAN),
                    None,
                    Some(0.0),
                ])),
            ],
        )
        .unwrap()
    }

    fn compare(column: &str, op: CompareOp, value: Scalar) -> Filter {
        Filter::Compare {
            column: column.into(),
            op,
            value,
        }
    }

    #[test]
    fn a_comparison_holds_as_sql_holds_it_and_a_null_meets_none() {
        let kept = may_hold(&rows(), &[compare("age", CompareOp::Gt, Scalar::Int64(76))]).unwrap();
        assert_eq!(kept, [true, false, false, true]);
        let kept = may_hold(&rows(), &[compare("age", CompareOp::Ne, Scalar::Int64(30))]).unwrap();
        assert_eq!(kept, [true, false, false, true]);
    }

    #[test]
    fn an_integer_column_compares_with_a_fraction_as_a_double() {
        let kept = may_hold(
            &rows(),
            &[compare("age", CompareOp::Ge, Scalar::Float64(77.5))],
        )
        .unwrap();
        assert_eq!(kept, [true, false, false, false]);
    }

    #[test]
    fn a_nan_is_kept_since_it_orders_with_nothing() {
        let kept = may_hold(
            &rows(),
            &[compare("score", CompareOp::Gt, Scalar::Float64(1.0))],
        )
        .unwrap();
        assert_eq!(kept, [true, true, false, false]);
    }

    #[test]
    fn values_that_do_not_compare_plainly_keep_every_row() {
        let kept = may_hold(&rows(), &[compare("name", CompareOp::Eq, Scalar::Int64(1))]).unwrap();
        assert_eq!(kept, [true; 4]);
    }

    #[test]
    fn in_any_of_and_all_of_combine_their_parts() {
        let any = Filter::AnyOf {
            filters: vec![
                Filter::In {
                    column: "name".into(),
                    values: vec![Scalar::Utf8("b".into())],
                },
                Filter::IsNull {
                    column: "name".into(),
                },
            ],
        };
        assert_eq!(
            may_hold(&rows(), &[any]).unwrap(),
            [false, true, false, true]
        );
        let all = Filter::AllOf {
            filters: vec![
                compare("name", CompareOp::Eq, Scalar::Utf8("a".into())),
                Filter::NotNull {
                    column: "age".into(),
                },
            ],
        };
        assert_eq!(
            may_hold(&rows(), &[all]).unwrap(),
            [true, false, false, false]
        );
    }
}
