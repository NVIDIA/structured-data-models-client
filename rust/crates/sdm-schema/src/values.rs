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
    TimestampMillisecondType, TimestampNanosecondType, TimestampSecondType, UInt8Type, UInt16Type,
    UInt32Type, UInt64Type,
};
use arrow_array::{Array, RecordBatch};
use arrow_schema::{DataType, TimeUnit};
use sdm_contract::scalar::decimal_to_f64;
use sdm_contract::{CoreError, Scalar, Unit};

use crate::not_yet;

/// How the reference's frame holds a float of a source's preview: Arrow-
/// backed pandas (DuckDB's and SQLite's), where every float is a value of
/// its own bits, so a NaN is a value, a NaN of each sign (DuckDB's `0.0/0.0`
/// and `'NaN'`) two, and `-0.0` and `0.0` two; or NumPy-backed pandas
/// (Postgres's rows), where a NaN is missing, as a null is (`isna`, and
/// every count drops it), and `-0.0` equals `0.0` (`nunique` counts them
/// once); or constructor frames, where native conversion already supplied
/// missingness through Arrow validity and valid NaNs remain present.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Floats {
    #[default]
    NanValue,
    NanMissing,
    /// Constructor Arrow has already encoded native missingness in its
    /// validity bitmap. Count valid NaNs as one present value and signed
    /// zeros as equal, as native pandas numeric arrays do. This affects
    /// inference equality keys, never the Arrow buffers or model values.
    Frame,
}

/// One value, ordered and compared exactly.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Value {
    Bool(bool),
    /// Any integer, the unscaled value of a decimal of scale 0 or less (one
    /// column has one scale), or a temporal count (one column has one unit).
    Int(i128),
    /// A float's bits, as [`Floats`] reads them; a decimal of scale > 0
    /// as the float64 the reference's cast gives it.
    Float(u64),
    Text(String),
    /// Exact native Decimal comparison key within a fixed-type frame field.
    /// Provider/SQL preview Decimal inference retains its existing branch.
    Decimal(crate::frames::decimal::Unscaled),
}

/// The values of column `index` of `batches`, in row order.
pub(crate) fn column(
    batches: &[RecordBatch],
    index: usize,
    floats: Floats,
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
            DataType::Timestamp(TimeUnit::Second, _) => ints!(TimestampSecondType),
            DataType::Timestamp(TimeUnit::Millisecond, _) => ints!(TimestampMillisecondType),
            DataType::Timestamp(TimeUnit::Microsecond, _) => ints!(TimestampMicrosecondType),
            DataType::Timestamp(TimeUnit::Nanosecond, _) => ints!(TimestampNanosecondType),
            DataType::Time64(TimeUnit::Microsecond) => ints!(Time64MicrosecondType),
            DataType::Time64(TimeUnit::Nanosecond) => ints!(Time64NanosecondType),
            // A decimal of scale > 0 as the float64 the reference's cast
            // gives it (`_sanitize`'s `pa.float64()`, Arrow's
            // `Decimal128::ToDouble`), which inference counts: values that
            // differ only past float64's precision are one value there
            // (T2.14b). Of scale 0 or less, its unscaled integer, as the
            // reference reads it as int64 or keeps it exact. Only what
            // inference counts reads this; materialize and predict keep
            // exact decimals (P2B-1).
            DataType::Decimal128(_, scale) if *scale > 0 => {
                let a = array.as_primitive::<Decimal128Type>();
                for i in 0..n {
                    let value = decimal_to_f64(a.value(i), *scale);
                    out.push(float(
                        array.is_null(i),
                        false,
                        value == 0.0,
                        value.to_bits(),
                        0.0_f64.to_bits(),
                        floats,
                    ));
                }
            }
            DataType::Decimal128(_, _) => {
                let a = array.as_primitive::<Decimal128Type>();
                for i in 0..n {
                    push(i, Value::Int(a.value(i)));
                }
            }
            // Each float by its own bits at its own width: widening a float32
            // would quiet a signaling NaN, making two NaNs the source and
            // the reference hold apart one (T2.12 fix 7).
            DataType::Float32 => {
                let a = array.as_primitive::<Float32Type>();
                for i in 0..n {
                    let value = a.value(i);
                    out.push(float(
                        array.is_null(i),
                        value.is_nan(),
                        value == 0.0,
                        u64::from(value.to_bits()),
                        u64::from(0.0_f32.to_bits()),
                        floats,
                    ));
                }
            }
            DataType::Float64 => {
                let a = array.as_primitive::<Float64Type>();
                for i in 0..n {
                    let value = a.value(i);
                    out.push(float(
                        array.is_null(i),
                        value.is_nan(),
                        value == 0.0,
                        value.to_bits(),
                        0.0_f64.to_bits(),
                        floats,
                    ));
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
    Ok(out)
}

/// A float as the reference's frame holds it (`floats`), given whether it
/// is null, a NaN and a zero, its `bits` at its width and that width's
/// positive zero's (`zero`): a NaN missing, and `-0.0` as `0.0`, in
/// NumPy-backed pandas; its own bits in Arrow-backed pandas, each NaN
/// (its sign and payload) and each zero apart.
fn float(
    null: bool,
    nan: bool,
    is_zero: bool,
    bits: u64,
    zero: u64,
    floats: Floats,
) -> Option<Value> {
    if null {
        return None;
    }
    let bits = match floats {
        Floats::NanMissing if nan => return None,
        Floats::NanMissing | Floats::Frame if is_zero => zero,
        // A width-independent equality key: no Float32 encoding can
        // collide with these Float64 NaN bits; other Float64 NaNs also
        // enter this arm. Native distinct/duplicate counts ignore NaN
        // payload and sign while original Arrow/scalar values stay intact.
        Floats::Frame if nan => f64::NAN.to_bits(),
        _ => bits,
    };
    Some(Value::Float(bits))
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
            DataType::Date32 | DataType::Timestamp(_, _) => scalar(array.as_ref(), i),
            _ => None,
        };
    }
    None
}

/// Native constructor time-choice first present value, even if a warning
/// handler changed an earlier timestamp candidate's physical storage.
pub(crate) fn frame_first(batches: &[RecordBatch], index: usize) -> Option<Scalar> {
    for batch in batches {
        let array = batch.column(index);
        if let Some(row) = (0..array.len()).find(|&row| !array.is_null(row)) {
            return scalar(array.as_ref(), row);
        }
    }
    None
}

/// Current numeric storage of an original timestamp candidate is read as
/// nanoseconds by the native constructor's to_datetime conversion. This
/// does not nominate a new time column or parse textual date content.
pub(crate) fn frame_instants(batches: &[RecordBatch], index: usize) -> Vec<Option<i128>> {
    let temporal = instants(batches, index);
    if batches.iter().all(|batch| {
        matches!(
            batch.column(index).data_type(),
            DataType::Date32 | DataType::Timestamp(_, _)
        )
    }) {
        return temporal;
    }
    let Ok(values) = column(batches, index, Floats::Frame) else {
        return temporal;
    };
    values
        .into_iter()
        .map(|value| match value {
            Some(Value::Int(value))
                if value > i128::from(i64::MIN) && value <= i128::from(i64::MAX) =>
            {
                Some(value)
            }
            Some(Value::Float(bits)) => {
                let value = f64::from_bits(bits);
                (value.is_finite() && value > i64::MIN as f64 && value < -(i64::MIN as f64))
                    .then_some(value as i128)
            }
            _ => None,
        })
        .collect()
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
                    DataType::Timestamp(TimeUnit::Second, _) => Some(
                        i128::from(array.as_primitive::<TimestampSecondType>().value(i))
                            * 1_000_000_000,
                    ),
                    DataType::Timestamp(TimeUnit::Millisecond, _) => Some(
                        i128::from(array.as_primitive::<TimestampMillisecondType>().value(i))
                            * 1_000_000,
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
    use arrow_array::types::{Date32Type as D, Decimal128Type as Dec, Decimal256Type as WideDec};
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
        DataType::Decimal256(precision, scale) => Scalar::Decimal256 {
            precision: *precision,
            scale: *scale,
            value: array.as_primitive::<WideDec>().value(row),
        },
        DataType::Utf8 => Scalar::Utf8(array.as_string::<i32>().value(row).to_owned()),
        DataType::Date32 => Scalar::Date32(array.as_primitive::<D>().value(row)),
        DataType::Timestamp(unit, zone) => Scalar::Timestamp {
            unit: (*unit).into(),
            zone: zone
                .as_deref()
                .filter(|zone| !zone.is_empty())
                .map(str::to_owned),
            value: match unit {
                TimeUnit::Second => array.as_primitive::<TimestampSecondType>().value(row),
                TimeUnit::Millisecond => {
                    array.as_primitive::<TimestampMillisecondType>().value(row)
                }
                TimeUnit::Microsecond => {
                    array.as_primitive::<TimestampMicrosecondType>().value(row)
                }
                TimeUnit::Nanosecond => array.as_primitive::<TimestampNanosecondType>().value(row),
            },
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{ArrayRef, Float64Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};

    use super::*;

    /// A NaN and the two zeros as each reference frame counts them: in
    /// Arrow-backed pandas (DuckDB's and SQLite's) each float is a value of
    /// its own bits, a NaN of each sign and each zero apart; in NumPy-backed
    /// pandas (Postgres's) a NaN is missing and the zeros are one (T2.12
    /// fixes 3 and 5).
    #[test]
    fn floats_are_counted_as_each_reference_frame_counts_them() {
        let values: ArrayRef = Arc::new(Float64Array::from(vec![
            Some(1.0),
            Some(f64::NAN),
            Some(-f64::NAN),
            None,
            Some(0.0),
            Some(-0.0),
        ]));
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("x", DataType::Float64, true)])),
            vec![values],
        )
        .unwrap();
        let arrow = column(std::slice::from_ref(&batch), 0, Floats::NanValue).unwrap();
        assert_eq!(arrow.iter().flatten().count(), 5);
        assert_eq!(distinct(&arrow), 5);
        let numpy = column(std::slice::from_ref(&batch), 0, Floats::NanMissing).unwrap();
        assert_eq!(numpy.iter().flatten().count(), 3);
        assert_eq!(distinct(&numpy), 2);
    }

    /// DuckDB's `0.0/0.0` is a NaN of the sign bit set and its `'NaN'` one
    /// without it; Arrow-backed pandas counts them as two values, as their
    /// bits differ, and each repeat of one as that one (T2.12 fix 5,
    /// probed on the reference: `nunique` 3 over `[-NaN, NaN, NaN, 1.0]`).
    #[test]
    fn a_nan_of_each_sign_is_two_values_where_floats_keep_their_bits() {
        let negative = f64::from_bits(0xfff8_0000_0000_0000);
        let positive = f64::from_bits(0x7ff8_0000_0000_0000);
        let values: ArrayRef = Arc::new(Float64Array::from(vec![
            Some(negative),
            Some(positive),
            Some(positive),
            Some(1.0),
        ]));
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("x", DataType::Float64, true)])),
            vec![values],
        )
        .unwrap();
        let arrow = column(std::slice::from_ref(&batch), 0, Floats::NanValue).unwrap();
        assert_eq!(distinct(&arrow), 3);
        let numpy = column(std::slice::from_ref(&batch), 0, Floats::NanMissing).unwrap();
        assert_eq!(numpy.iter().flatten().count(), 1);
        assert_eq!(distinct(&numpy), 1);
    }

    /// A signaling and a quiet NaN are two values where floats keep their
    /// bits, at each width: DuckDB stores a float32's `0x7f800001` and
    /// `0x7fc00001` apart and the reference counts them apart, and no
    /// widening quiets the first (T2.12 fix 7).
    #[test]
    fn a_signaling_and_a_quiet_nan_are_two_values_at_each_width() {
        use arrow_array::Float32Array;
        let narrow: ArrayRef = Arc::new(Float32Array::from(vec![
            Some(f32::from_bits(0x7f80_0001)),
            Some(f32::from_bits(0x7fc0_0001)),
            Some(f32::from_bits(0x7fc0_0001)),
        ]));
        let wide: ArrayRef = Arc::new(Float64Array::from(vec![
            Some(f64::from_bits(0x7ff0_0000_0000_0001)),
            Some(f64::from_bits(0x7ff8_0000_0000_0001)),
            Some(f64::from_bits(0x7ff8_0000_0000_0001)),
        ]));
        for values in [narrow, wide] {
            let batch = RecordBatch::try_new(
                Arc::new(Schema::new(vec![Field::new(
                    "x",
                    values.data_type().clone(),
                    true,
                )])),
                vec![values],
            )
            .unwrap();
            let arrow = column(std::slice::from_ref(&batch), 0, Floats::NanValue).unwrap();
            assert_eq!(distinct(&arrow), 2, "{:?}", batch.schema());
            let numpy = column(std::slice::from_ref(&batch), 0, Floats::NanMissing).unwrap();
            assert_eq!(numpy.iter().flatten().count(), 0);
        }
    }
}
