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
use sdm_contract::CoreError;

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
