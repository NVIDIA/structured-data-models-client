// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A column's values in a preview, as comparable keys: what inference counts
//! distinct values and tests text over. Nulls are `None`.

use std::collections::BTreeSet;

use arrow_array::cast::AsArray;
use arrow_array::types::{
    Date32Type, Decimal128Type, Float32Type, Float64Type, Int8Type, Int16Type, Int32Type,
    Int64Type, Time64MicrosecondType, Time64NanosecondType, TimestampMicrosecondType,
    TimestampNanosecondType, UInt8Type, UInt16Type, UInt32Type, UInt64Type,
};
use arrow_array::{Array, RecordBatch};
use arrow_schema::{DataType, TimeUnit};
use sdm_contract::{CoreError, Scalar, Unit};

use crate::not_yet;

/// One value, ordered and compared exactly.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Value {
    Bool(bool),
    /// Any integer, a decimal's unscaled value (one column has one scale),
    /// or a temporal count (one column has one unit).
    Int(i128),
    /// A float's bits; a column holding NaN or both zeros is refused before
    /// one is made, since how those count is not ported.
    Float(u64),
    Text(String),
}

/// The values of column `index` of `batches`, in row order.
pub(crate) fn column(
    batches: &[RecordBatch],
    index: usize,
) -> Result<Vec<Option<Value>>, CoreError> {
    let mut out = Vec::new();
    for batch in batches {
        let array = batch.column(index);
        let n = array.len();
        let mut push = |i: usize, value: Value| {
            out.push(if array.is_null(i) { None } else { Some(value) });
        };
        macro_rules! ints {
            ($t:ty) => {{
                let a = array.as_primitive::<$t>();
                for i in 0..n {
                    push(i, Value::Int(i128::from(a.value(i))));
                }
            }};
        }
        match array.data_type() {
            DataType::Boolean => {
                let a = array.as_boolean();
                for i in 0..n {
                    push(i, Value::Bool(a.value(i)));
                }
            }
            DataType::Int8 => ints!(Int8Type),
            DataType::Int16 => ints!(Int16Type),
            DataType::Int32 => ints!(Int32Type),
            DataType::Int64 => ints!(Int64Type),
            DataType::UInt8 => ints!(UInt8Type),
            DataType::UInt16 => ints!(UInt16Type),
            DataType::UInt32 => ints!(UInt32Type),
            DataType::UInt64 => ints!(UInt64Type),
            DataType::Date32 => ints!(Date32Type),
            DataType::Timestamp(TimeUnit::Microsecond, _) => ints!(TimestampMicrosecondType),
            DataType::Timestamp(TimeUnit::Nanosecond, _) => ints!(TimestampNanosecondType),
            DataType::Time64(TimeUnit::Microsecond) => ints!(Time64MicrosecondType),
            DataType::Time64(TimeUnit::Nanosecond) => ints!(Time64NanosecondType),
            DataType::Decimal128(_, _) => {
                let a = array.as_primitive::<Decimal128Type>();
                for i in 0..n {
                    push(i, Value::Int(a.value(i)));
                }
            }
            DataType::Float32 => {
                let a = array.as_primitive::<Float32Type>();
                for i in 0..n {
                    push(i, Value::Float(f64::from(a.value(i)).to_bits()));
                }
            }
            DataType::Float64 => {
                let a = array.as_primitive::<Float64Type>();
                for i in 0..n {
                    push(i, Value::Float(a.value(i).to_bits()));
                }
            }
            DataType::Utf8 => {
                let a = array.as_string::<i32>();
                for i in 0..n {
                    push(i, Value::Text(a.value(i).to_owned()));
                }
            }
            _ => return Err(not_yet("column_type")),
        }
    }
    if out
        .iter()
        .flatten()
        .any(|v| matches!(v, Value::Float(bits) if f64::from_bits(*bits).is_nan()))
    {
        return Err(not_yet("float_nan"));
    }
    let zeros = out
        .iter()
        .flatten()
        .filter_map(|v| match v {
            Value::Float(bits) if f64::from_bits(*bits) == 0.0 => Some(*bits),
            _ => None,
        })
        .collect::<BTreeSet<_>>();
    if zeros.len() > 1 {
        return Err(not_yet("float_signed_zero"));
    }
    Ok(out)
}

/// How many distinct values `values` holds among those not null.
pub(crate) fn distinct(values: &[Option<Value>]) -> usize {
    values.iter().flatten().collect::<BTreeSet<_>>().len()
}

/// The first value of column `index` of `batches` that is not null, as the
/// contract's scalar, for a temporal column: a date, a timestamp (with its
/// zone) or a time of day; `None` when every row is null or the column is
/// of another type.
pub(crate) fn first_temporal(batches: &[RecordBatch], index: usize) -> Option<Scalar> {
    for batch in batches {
        let array = batch.column(index);
        let Some(i) = (0..array.len()).find(|&i| !array.is_null(i)) else {
            continue;
        };
        return match array.data_type() {
            DataType::Date32 => Some(Scalar::Date32(array.as_primitive::<Date32Type>().value(i))),
            DataType::Timestamp(TimeUnit::Microsecond, zone) => Some(Scalar::Timestamp {
                unit: Unit::Us,
                zone: zone.as_ref().map(ToString::to_string),
                value: array.as_primitive::<TimestampMicrosecondType>().value(i),
            }),
            DataType::Timestamp(TimeUnit::Nanosecond, zone) => Some(Scalar::Timestamp {
                unit: Unit::Ns,
                zone: zone.as_ref().map(ToString::to_string),
                value: array.as_primitive::<TimestampNanosecondType>().value(i),
            }),
            _ => None,
        };
    }
    None
}

/// The distinct values of a time-of-day column of `batches`, at most
/// `limit`, in the order they first appear, as the contract's scalars.
pub(crate) fn times_of_day(batches: &[RecordBatch], index: usize, limit: usize) -> Vec<Scalar> {
    let mut seen = Vec::new();
    for batch in batches {
        let array = batch.column(index);
        for i in 0..array.len() {
            if seen.len() == limit {
                return seen;
            }
            if array.is_null(i) {
                continue;
            }
            let value = match array.data_type() {
                DataType::Time64(TimeUnit::Microsecond) => Scalar::Time64 {
                    unit: Unit::Us,
                    value: array.as_primitive::<Time64MicrosecondType>().value(i),
                },
                DataType::Time64(TimeUnit::Nanosecond) => Scalar::Time64 {
                    unit: Unit::Ns,
                    value: array.as_primitive::<Time64NanosecondType>().value(i),
                },
                _ => return seen,
            };
            if !seen.contains(&value) {
                seen.push(value);
            }
        }
    }
    seen
}

/// Each value of a date or timestamp column of `batches` as nanoseconds
/// since the epoch, in UTC (a timestamp without a zone read as UTC, a
/// date at its midnight), in row order; nulls are `None`. A column of any
/// other type is all `None`: no value of it reads as an instant.
pub(crate) fn instants(batches: &[RecordBatch], index: usize) -> Vec<Option<i128>> {
    let mut out = Vec::new();
    for batch in batches {
        let array = batch.column(index);
        for i in 0..array.len() {
            out.push(if array.is_null(i) {
                None
            } else {
                match array.data_type() {
                    DataType::Date32 => Some(
                        i128::from(array.as_primitive::<Date32Type>().value(i))
                            * 86_400_000_000_000,
                    ),
                    DataType::Timestamp(TimeUnit::Microsecond, _) => Some(
                        i128::from(array.as_primitive::<TimestampMicrosecondType>().value(i))
                            * 1_000,
                    ),
                    DataType::Timestamp(TimeUnit::Nanosecond, _) => Some(i128::from(
                        array.as_primitive::<TimestampNanosecondType>().value(i),
                    )),
                    _ => None,
                }
            });
        }
    }
    out
}

/// Value `row` of `array` as the contract's scalar, for a value the source
/// hands over to show (a repeated key): a boolean, an integer, a float, a
/// decimal, text, a date, a timestamp or a time of day; `None` when null or
/// of another type.
pub(crate) fn scalar(array: &dyn Array, row: usize) -> Option<Scalar> {
    use arrow_array::types::{Date32Type as D, Decimal128Type as Dec};
    if row >= array.len() || array.is_null(row) {
        return None;
    }
    Some(match array.data_type() {
        DataType::Boolean => Scalar::Bool(array.as_boolean().value(row)),
        DataType::Int8 => Scalar::Int8(array.as_primitive::<Int8Type>().value(row)),
        DataType::Int16 => Scalar::Int16(array.as_primitive::<Int16Type>().value(row)),
        DataType::Int32 => Scalar::Int32(array.as_primitive::<Int32Type>().value(row)),
        DataType::Int64 => Scalar::Int64(array.as_primitive::<Int64Type>().value(row)),
        DataType::UInt8 => Scalar::UInt8(array.as_primitive::<UInt8Type>().value(row)),
        DataType::UInt16 => Scalar::UInt16(array.as_primitive::<UInt16Type>().value(row)),
        DataType::UInt32 => Scalar::UInt32(array.as_primitive::<UInt32Type>().value(row)),
        DataType::UInt64 => Scalar::UInt64(array.as_primitive::<UInt64Type>().value(row)),
        DataType::Float32 => Scalar::Float32(array.as_primitive::<Float32Type>().value(row)),
        DataType::Float64 => Scalar::Float64(array.as_primitive::<Float64Type>().value(row)),
        DataType::Decimal128(precision, scale) => Scalar::Decimal128 {
            precision: *precision,
            scale: *scale,
            value: array.as_primitive::<Dec>().value(row),
        },
        DataType::Utf8 => Scalar::Utf8(array.as_string::<i32>().value(row).to_owned()),
        DataType::Date32 => Scalar::Date32(array.as_primitive::<D>().value(row)),
        DataType::Timestamp(TimeUnit::Microsecond, zone) => Scalar::Timestamp {
            unit: Unit::Us,
            zone: zone.as_ref().map(ToString::to_string),
            value: array.as_primitive::<TimestampMicrosecondType>().value(row),
        },
        DataType::Timestamp(TimeUnit::Nanosecond, zone) => Scalar::Timestamp {
            unit: Unit::Ns,
            zone: zone.as_ref().map(ToString::to_string),
            value: array.as_primitive::<TimestampNanosecondType>().value(row),
        },
        DataType::Time64(TimeUnit::Microsecond) => Scalar::Time64 {
            unit: Unit::Us,
            value: array.as_primitive::<Time64MicrosecondType>().value(row),
        },
        DataType::Time64(TimeUnit::Nanosecond) => Scalar::Time64 {
            unit: Unit::Ns,
            value: array.as_primitive::<Time64NanosecondType>().value(row),
        },
        _ => return None,
    })
}
