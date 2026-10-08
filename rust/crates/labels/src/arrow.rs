// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Arrow in and out of the executor: the sampler's tables, as
//! `pa.Table.from_pandas` writes the frames today's executor reads (their
//! pandas dtypes kept in the schema's `pandas` metadata), and the labels as
//! one Arrow array.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::builder::{
    BinaryBuilder, BooleanBuilder, Decimal128Builder, Float64Builder, Int64Builder, ListBuilder,
    StringBuilder, TimestampNanosecondBuilder,
};
use arrow_array::cast::AsArray;
use arrow_array::types::{
    Decimal128Type, DurationMicrosecondType, DurationMillisecondType, DurationNanosecondType,
    DurationSecondType, Float32Type, Float64Type, Int8Type, Int16Type, Int32Type, Int64Type,
    TimestampMicrosecondType, TimestampMillisecondType, TimestampNanosecondType,
    TimestampSecondType, UInt8Type, UInt16Type, UInt32Type, UInt64Type,
};
use arrow_array::{
    Array, ArrayRef, BooleanArray, DurationMicrosecondArray, DurationMillisecondArray,
    DurationNanosecondArray, DurationSecondArray, Float32Array, Float64Array, Int64Array,
    NullArray, StringArray, TimestampMicrosecondArray, TimestampMillisecondArray,
    TimestampNanosecondArray, TimestampSecondArray,
};
use arrow_schema::{DataType, TimeUnit as ArrowUnit};
use serde_json::Value;

use crate::LabelError;
use crate::column::{Column, Obj, TimeUnit};

fn unsupported(what: &str, data_type: &DataType) -> LabelError {
    LabelError::new(
        "TypeError",
        format!("{what} of Arrow type {data_type} is not supported"),
    )
}

/// The pandas dtype each column of a table had, from the schema's `pandas`
/// metadata: `field_name` to `numpy_type`.
pub fn pandas_dtypes(metadata: &HashMap<String, String>) -> HashMap<String, String> {
    let Some(text) = metadata.get("pandas") else {
        return HashMap::new();
    };
    let Ok(Value::Object(meta)) = serde_json::from_str::<Value>(text) else {
        return HashMap::new();
    };
    meta.get("columns")
        .and_then(Value::as_array)
        .map(|columns| {
            columns
                .iter()
                .filter_map(|c| {
                    let name = c.get("field_name").and_then(Value::as_str)?;
                    let dtype = c.get("numpy_type").and_then(Value::as_str)?;
                    Some((name.to_string(), dtype.to_string()))
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Each value of an integer array, `None` where null.
fn ints(array: &dyn Array) -> Option<Vec<Option<i64>>> {
    macro_rules! read {
        ($t:ty) => {{
            let values = array.as_primitive::<$t>();
            Some(
                (0..array.len())
                    .map(|i| array.is_valid(i).then(|| i64::from(values.value(i))))
                    .collect(),
            )
        }};
    }
    match array.data_type() {
        DataType::Int64 => read!(Int64Type),
        DataType::Int32 => read!(Int32Type),
        DataType::Int16 => read!(Int16Type),
        DataType::Int8 => read!(Int8Type),
        // A `uint64` past `int64` wraps one to one, as the sampler's frames
        // read it; the schema calls a key holding one text.
        DataType::UInt64 => {
            let values = array.as_primitive::<UInt64Type>();
            Some(
                (0..array.len())
                    .map(|i| array.is_valid(i).then(|| values.value(i) as i64))
                    .collect(),
            )
        }
        DataType::UInt32 => read!(UInt32Type),
        DataType::UInt16 => read!(UInt16Type),
        DataType::UInt8 => read!(UInt8Type),
        _ => None,
    }
}

fn floats(array: &dyn Array) -> Option<Vec<Option<f64>>> {
    match array.data_type() {
        DataType::Float64 => {
            let values = array.as_primitive::<Float64Type>();
            Some(
                (0..array.len())
                    .map(|i| array.is_valid(i).then(|| values.value(i)))
                    .collect(),
            )
        }
        DataType::Float32 => {
            let values = array.as_primitive::<Float32Type>();
            Some(
                (0..array.len())
                    .map(|i| array.is_valid(i).then(|| f64::from(values.value(i))))
                    .collect(),
            )
        }
        _ => ints(array).map(|v| v.into_iter().map(|x| x.map(|x| x as f64)).collect()),
    }
}

fn bools(array: &dyn Array) -> Option<Vec<Option<bool>>> {
    match array.data_type() {
        DataType::Boolean => {
            let values = array.as_boolean();
            Some(
                (0..array.len())
                    .map(|i| array.is_valid(i).then(|| values.value(i)))
                    .collect(),
            )
        }
        _ => None,
    }
}

fn texts(array: &dyn Array) -> Option<Vec<Option<String>>> {
    match array.data_type() {
        DataType::Utf8 => {
            let values = array.as_string::<i32>();
            Some(
                (0..array.len())
                    .map(|i| array.is_valid(i).then(|| values.value(i).to_string()))
                    .collect(),
            )
        }
        DataType::LargeUtf8 => {
            let values = array.as_string::<i64>();
            Some(
                (0..array.len())
                    .map(|i| array.is_valid(i).then(|| values.value(i).to_string()))
                    .collect(),
            )
        }
        DataType::Utf8View => {
            let values = array.as_string_view();
            Some(
                (0..array.len())
                    .map(|i| array.is_valid(i).then(|| values.value(i).to_string()))
                    .collect(),
            )
        }
        _ => None,
    }
}

/// The ticks and unit of a timestamp array.
fn stamps(array: &dyn Array) -> Option<(TimeUnit, Vec<Option<i64>>)> {
    macro_rules! read {
        ($t:ty, $unit:expr) => {{
            let values = array.as_primitive::<$t>();
            Some((
                $unit,
                (0..array.len())
                    .map(|i| array.is_valid(i).then(|| values.value(i)))
                    .collect(),
            ))
        }};
    }
    match array.data_type() {
        DataType::Timestamp(ArrowUnit::Nanosecond, _) => {
            read!(TimestampNanosecondType, TimeUnit::Ns)
        }
        DataType::Timestamp(ArrowUnit::Microsecond, _) => {
            read!(TimestampMicrosecondType, TimeUnit::Us)
        }
        DataType::Timestamp(ArrowUnit::Millisecond, _) => {
            read!(TimestampMillisecondType, TimeUnit::Ms)
        }
        DataType::Timestamp(ArrowUnit::Second, _) => read!(TimestampSecondType, TimeUnit::S),
        _ => None,
    }
}

/// Each value of an array as the Python object pandas holds in an
/// `object` column.
fn objects(array: &dyn Array) -> Result<Vec<Obj>, LabelError> {
    let n = array.len();
    if let Some(v) = texts(array) {
        return Ok(v
            .into_iter()
            .map(|x| x.map_or(Obj::None, Obj::Str))
            .collect());
    }
    if let Some(v) = ints(array) {
        return Ok(v
            .into_iter()
            .map(|x| x.map_or(Obj::None, Obj::Int))
            .collect());
    }
    if let Some(v) = floats(array) {
        return Ok(v
            .into_iter()
            .map(|x| x.map_or(Obj::None, Obj::Float))
            .collect());
    }
    if let Some(v) = bools(array) {
        return Ok(v
            .into_iter()
            .map(|x| x.map_or(Obj::None, Obj::Bool))
            .collect());
    }
    if let Some((unit, v)) = stamps(array) {
        return Ok(v
            .into_iter()
            .map(|x| x.map_or(Obj::None, |t| Obj::Time(unit, t)))
            .collect());
    }
    match array.data_type() {
        DataType::Null => Ok(vec![Obj::None; n]),
        DataType::Binary => {
            let values = array.as_binary::<i32>();
            Ok((0..n)
                .map(|index| {
                    if array.is_valid(index) {
                        Obj::Bytes(values.value(index).to_vec())
                    } else {
                        Obj::None
                    }
                })
                .collect())
        }
        DataType::Duration(unit) => {
            let (unit, values): (TimeUnit, Vec<Option<i64>>) = match unit {
                ArrowUnit::Second => (
                    TimeUnit::S,
                    array.as_primitive::<DurationSecondType>().iter().collect(),
                ),
                ArrowUnit::Millisecond => (
                    TimeUnit::Ms,
                    array
                        .as_primitive::<DurationMillisecondType>()
                        .iter()
                        .collect(),
                ),
                ArrowUnit::Microsecond => (
                    TimeUnit::Us,
                    array
                        .as_primitive::<DurationMicrosecondType>()
                        .iter()
                        .collect(),
                ),
                ArrowUnit::Nanosecond => (
                    TimeUnit::Ns,
                    array
                        .as_primitive::<DurationNanosecondType>()
                        .iter()
                        .collect(),
                ),
            };
            Ok(values
                .into_iter()
                .map(|value| value.map_or(Obj::NaT, |ticks| Obj::Duration(unit, ticks)))
                .collect())
        }
        DataType::Decimal128(_, scale) => {
            let values = array.as_primitive::<Decimal128Type>();
            Ok((0..n)
                .map(|i| {
                    if array.is_valid(i) {
                        Obj::Decimal(values.value(i), *scale)
                    } else {
                        Obj::None
                    }
                })
                .collect())
        }
        DataType::List(_) => {
            let lists = array.as_list::<i32>();
            (0..n)
                .map(|i| {
                    if array.is_valid(i) {
                        Ok(Obj::List(objects(lists.value(i).as_ref())?))
                    } else {
                        Ok(Obj::None)
                    }
                })
                .collect()
        }
        DataType::LargeList(_) => {
            let lists = array.as_list::<i64>();
            (0..n)
                .map(|i| {
                    if array.is_valid(i) {
                        Ok(Obj::List(objects(lists.value(i).as_ref())?))
                    } else {
                        Ok(Obj::None)
                    }
                })
                .collect()
        }
        other => Err(unsupported("an object column", other)),
    }
}

/// A column of the pandas dtype `dtype` (its `numpy_type`; inferred from
/// the Arrow type when unknown), as `to_pandas` would give it back.
pub fn column(array: &dyn Array, dtype: Option<&str>) -> Result<Column, LabelError> {
    let data_type = array.data_type();
    let wrong = || unsupported(&format!("a {} column", dtype.unwrap_or("?")), data_type);
    let dtype = match dtype {
        Some(d) => d.to_string(),
        None => match data_type {
            DataType::Float32 => "float32".into(),
            DataType::Float64 => "float64".into(),
            DataType::Boolean if array.null_count() > 0 => "boolean".into(),
            DataType::Boolean => "bool".into(),
            DataType::Timestamp(..) => {
                let (unit, _) = stamps(array).ok_or_else(wrong)?;
                format!("datetime64[{}]", unit.name())
            }
            d if ints(array).is_some() && array.null_count() > 0 => {
                let _ = d;
                "Int64".into()
            }
            _ if ints(array).is_some() => "int64".into(),
            _ => "object".into(),
        },
    };
    if let Some(unit) = dtype
        .strip_prefix("datetime64[")
        .and_then(|rest| rest.strip_suffix(']'))
        .and_then(TimeUnit::parse)
    {
        let (given, values) = stamps(array).ok_or_else(wrong)?;
        return Ok(Column::Time(
            unit,
            values
                .into_iter()
                .map(|v| {
                    v.map(|t| {
                        (i128::from(t) * i128::from(given.nanos()))
                            .div_euclid(i128::from(unit.nanos())) as i64
                    })
                })
                .collect(),
        ));
    }
    Ok(match dtype.as_str() {
        "float16" | "float32" => Column::F32(
            floats(array)
                .ok_or_else(wrong)?
                .into_iter()
                .map(|v| v.map_or(f32::NAN, |x| x as f32))
                .collect(),
        ),
        "float64" => Column::F64(
            floats(array)
                .ok_or_else(wrong)?
                .into_iter()
                .map(|v| v.unwrap_or(f64::NAN))
                .collect(),
        ),
        "Float64" | "Float32" => Column::MF64(floats(array).ok_or_else(wrong)?),
        "int64" | "int32" | "int16" | "int8" | "uint64" | "uint32" | "uint16" | "uint8" => {
            Column::I64(
                ints(array)
                    .ok_or_else(wrong)?
                    .into_iter()
                    .map(|v| v.unwrap_or(0))
                    .collect(),
            )
        }
        "Int64" | "Int32" | "Int16" | "Int8" | "UInt64" | "UInt32" | "UInt16" | "UInt8" => {
            Column::MI64(ints(array).ok_or_else(wrong)?)
        }
        "bool" => Column::Bool(
            bools(array)
                .ok_or_else(wrong)?
                .into_iter()
                .map(|v| v.unwrap_or(false))
                .collect(),
        ),
        "boolean" => Column::MBool(bools(array).ok_or_else(wrong)?),
        "str" | "string" => match data_type {
            DataType::Null => Column::Str(vec![None; array.len()]),
            _ => Column::Str(texts(array).ok_or_else(wrong)?),
        },
        _ => Column::Obj(objects(array)?),
    })
}

/// The `decimal128(precision, scale)` `pa.array` infers for decimals, each
/// digits at a scale: per value, the digits its `as_tuple()` holds and the
/// places its exponent asks for, merged as pyarrow's `DecimalMetadata`
/// merges them (`arrow/python/decimal.cc`).
fn infer_decimal(values: &[(i128, i8)]) -> Result<(u8, i8), LabelError> {
    let mut merged: Option<(i32, i32)> = None;
    for &(digits, scale) in values {
        let num_digits = digits.unsigned_abs().to_string().len() as i32;
        let exponent = -i32::from(scale);
        let (p, s) = if exponent < 0 {
            (num_digits.max(-exponent), -exponent)
        } else {
            (num_digits + exponent, 0)
        };
        merged = Some(match merged {
            None => (p, s),
            Some((precision, current)) => {
                let scale = current.max(s);
                let whole = (precision - current).max(p - s);
                ((whole + scale).max(precision), scale)
            }
        });
    }
    let (precision, scale) = merged.unwrap_or((1, 0));
    match (u8::try_from(precision), i8::try_from(scale)) {
        (Ok(p), Ok(s)) if p <= 38 => Ok((p, s)),
        _ => Err(LabelError::new(
            "TypeError",
            format!("decimals of precision {precision} are not supported by the core"),
        )),
    }
}

/// `digits` at `from` brought to the scale `to`, which is not smaller.
fn rescale(digits: i128, from: i8, to: i8) -> Result<i128, LabelError> {
    u32::try_from(i32::from(to) - i32::from(from))
        .ok()
        .and_then(|places| 10i128.checked_pow(places))
        .and_then(|f| digits.checked_mul(f))
        .ok_or_else(|| LabelError::new("ArithmeticError", "a decimal past 38 digits"))
}

/// An object column's values as one Arrow array: text, decimals, lists of
/// one type, or nulls.
fn object_array(values: &[Obj]) -> Result<ArrayRef, LabelError> {
    if values.iter().any(|value| matches!(value, Obj::Bytes(_)))
        && values
            .iter()
            .all(|value| matches!(value, Obj::Bytes(_) | Obj::None | Obj::Na))
    {
        let mut builder = BinaryBuilder::new();
        for value in values {
            match value {
                Obj::Bytes(bytes) => builder.append_value(bytes),
                _ => builder.append_null(),
            }
        }
        return Ok(Arc::new(builder.finish()));
    }
    if let Some(unit) = values.iter().find_map(|value| match value {
        Obj::Duration(unit, _) => Some(*unit),
        _ => None,
    }) && values
        .iter()
        .all(|value| matches!(value, Obj::Duration(given, _) if *given == unit) || value.is_na())
    {
        let ticks: Vec<Option<i64>> = values
            .iter()
            .map(|value| match value {
                Obj::Duration(_, ticks) => Some(*ticks),
                _ => None,
            })
            .collect();
        return Ok(match unit {
            TimeUnit::S => Arc::new(DurationSecondArray::from(ticks)),
            TimeUnit::Ms => Arc::new(DurationMillisecondArray::from(ticks)),
            TimeUnit::Us => Arc::new(DurationMicrosecondArray::from(ticks)),
            TimeUnit::Ns => Arc::new(DurationNanosecondArray::from(ticks)),
        });
    }
    if values
        .iter()
        .all(|v| matches!(v, Obj::None | Obj::Na | Obj::NaT))
    {
        return Ok(Arc::new(NullArray::new(values.len())));
    }
    if values
        .iter()
        .all(|v| matches!(v, Obj::Str(_) | Obj::None | Obj::Na))
    {
        let mut b = StringBuilder::new();
        for v in values {
            match v {
                Obj::Str(s) => b.append_value(s),
                _ => b.append_null(),
            }
        }
        return Ok(Arc::new(b.finish()));
    }
    if values
        .iter()
        .all(|v| matches!(v, Obj::Decimal(..) | Obj::None | Obj::Na))
    {
        let decimals: Vec<(i128, i8)> = values
            .iter()
            .filter_map(|v| match v {
                Obj::Decimal(digits, scale) => Some((*digits, *scale)),
                _ => None,
            })
            .collect();
        let (precision, scale) = infer_decimal(&decimals)?;
        let mut b = Decimal128Builder::new();
        for v in values {
            match v {
                Obj::Decimal(digits, from) => b.append_value(rescale(*digits, *from, scale)?),
                _ => b.append_null(),
            }
        }
        let array = b
            .finish()
            .with_precision_and_scale(precision, scale)
            .map_err(|e| LabelError::new("ValueError", e.to_string()))?;
        return Ok(Arc::new(array));
    }
    if values
        .iter()
        .all(|v| matches!(v, Obj::List(_) | Obj::None | Obj::Na))
    {
        let items: Vec<&Obj> = values
            .iter()
            .filter_map(|v| {
                if let Obj::List(items) = v {
                    Some(items)
                } else {
                    None
                }
            })
            .flatten()
            .collect();
        macro_rules! lists {
            ($builder:expr, $push:expr) => {{
                let mut b = ListBuilder::new($builder);
                for v in values {
                    match v {
                        Obj::List(items) => {
                            for item in items {
                                $push(b.values(), item);
                            }
                            b.append(true);
                        }
                        _ => b.append(false),
                    }
                }
                return Ok(Arc::new(b.finish()));
            }};
        }
        let all = |f: fn(&Obj) -> bool| items.iter().all(|o| f(o));
        if all(|o| matches!(o, Obj::Int(_) | Obj::NpInt64(_))) {
            lists!(
                Int64Builder::new(),
                |b: &mut Int64Builder, o: &Obj| match o {
                    Obj::Int(i) | Obj::NpInt64(i) => b.append_value(*i),
                    _ => b.append_null(),
                }
            );
        }
        if all(|o| {
            matches!(
                o,
                Obj::Float(_) | Obj::NpFloat64(_) | Obj::NpFloat32(_) | Obj::Int(_)
            )
        }) {
            lists!(
                Float64Builder::new(),
                |b: &mut Float64Builder, o: &Obj| match o {
                    Obj::Float(f) | Obj::NpFloat64(f) => b.append_value(*f),
                    Obj::NpFloat32(f) => b.append_value(f64::from(*f)),
                    Obj::Int(i) => b.append_value(*i as f64),
                    _ => b.append_null(),
                }
            );
        }
        if all(|o| matches!(o, Obj::Bool(_) | Obj::NpBool(_))) {
            lists!(
                BooleanBuilder::new(),
                |b: &mut BooleanBuilder, o: &Obj| match o {
                    Obj::Bool(x) | Obj::NpBool(x) => b.append_value(*x),
                    _ => b.append_null(),
                }
            );
        }
        if all(|o| matches!(o, Obj::Time(..))) {
            lists!(
                TimestampNanosecondBuilder::new(),
                |b: &mut TimestampNanosecondBuilder, o: &Obj| match o {
                    Obj::Time(u, t) => b.append_value(t * u.nanos()),
                    _ => b.append_null(),
                }
            );
        }
        if !items.is_empty() && all(|o| matches!(o, Obj::Decimal(..))) {
            let decimals: Vec<(i128, i8)> = items
                .iter()
                .filter_map(|o| match o {
                    Obj::Decimal(digits, scale) => Some((*digits, *scale)),
                    _ => None,
                })
                .collect();
            let (precision, scale) = infer_decimal(&decimals)?;
            let item = Decimal128Builder::new()
                .with_precision_and_scale(precision, scale)
                .map_err(|e| LabelError::new("ValueError", e.to_string()))?;
            let mut b = ListBuilder::new(item);
            for v in values {
                match v {
                    Obj::List(items) => {
                        for o in items {
                            if let Obj::Decimal(digits, from) = o {
                                b.values().append_value(rescale(*digits, *from, scale)?);
                            }
                        }
                        b.append(true);
                    }
                    _ => b.append(false),
                }
            }
            return Ok(Arc::new(b.finish()));
        }
        if all(|o| matches!(o, Obj::Str(_) | Obj::None)) {
            lists!(
                StringBuilder::new(),
                |b: &mut StringBuilder, o: &Obj| match o {
                    Obj::Str(s) => b.append_value(s),
                    _ => b.append_null(),
                }
            );
        }
    }
    Err(LabelError::new(
        "TypeError",
        "labels of mixed Python types have no one Arrow type",
    ))
}

/// The labels as one Arrow array of their dtype.
pub fn to_arrow(column: &Column) -> Result<ArrayRef, LabelError> {
    Ok(match column {
        Column::Bool(v) => Arc::new(BooleanArray::from(v.clone())),
        Column::MBool(v) => Arc::new(BooleanArray::from(v.clone())),
        Column::I64(v) => Arc::new(Int64Array::from(v.clone())),
        Column::MI64(v) => Arc::new(Int64Array::from(v.clone())),
        Column::F32(v) => Arc::new(Float32Array::from(v.clone())),
        Column::F64(v) => Arc::new(Float64Array::from(v.clone())),
        Column::MF64(v) => Arc::new(Float64Array::from(v.clone())),
        Column::Time(unit, v) => match unit {
            TimeUnit::Ns => Arc::new(TimestampNanosecondArray::from(v.clone())),
            TimeUnit::Us => Arc::new(TimestampMicrosecondArray::from(v.clone())),
            TimeUnit::Ms => Arc::new(TimestampMillisecondArray::from(v.clone())),
            TimeUnit::S => Arc::new(TimestampSecondArray::from(v.clone())),
        },
        Column::Str(v) => Arc::new(StringArray::from(v.clone())),
        Column::Obj(v) => object_array(v)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_labels_round_trip_exact_bytes_and_nulls() {
        let array = arrow_array::BinaryArray::from(vec![
            Some(&b"one"[..]),
            None,
            Some(&b"\xff\0"[..]),
            Some(&b""[..]),
        ]);
        let labels = column(&array, Some("object")).unwrap();
        assert_eq!(labels.notna(), vec![true, false, true, true]);
        assert_eq!(to_arrow(&labels).unwrap().as_ref(), &array);
        let filtered = labels.filter(&labels.notna());
        assert_eq!(
            filtered.objects(),
            vec![
                Obj::Bytes(b"one".to_vec()),
                Obj::Bytes(vec![255, 0]),
                Obj::Bytes(vec![]),
            ]
        );
    }

    #[test]
    fn duration_labels_round_trip_units_ticks_and_nulls() {
        let ticks = vec![Some(1), Some(-2), None, Some(i64::MAX - 1)];
        let arrays: Vec<ArrayRef> = vec![
            Arc::new(DurationSecondArray::from(ticks.clone())),
            Arc::new(DurationMillisecondArray::from(ticks.clone())),
            Arc::new(DurationMicrosecondArray::from(ticks.clone())),
            Arc::new(DurationNanosecondArray::from(ticks)),
        ];
        for array in arrays {
            let labels = column(array.as_ref(), Some("timedelta64[ns]")).unwrap();
            assert_eq!(labels.notna(), vec![true, true, false, true]);
            assert_eq!(to_arrow(&labels).unwrap().as_ref(), array.as_ref());
        }
        assert!(Obj::Duration(TimeUnit::S, -1).py_eq(&Obj::Duration(TimeUnit::Ms, -1000)));
        assert!(!Obj::Duration(TimeUnit::S, 1).py_eq(&Obj::Int(1)));
        assert!(!Obj::Duration(TimeUnit::S, 1).py_eq(&Obj::Time(TimeUnit::S, 1)));
    }

    #[test]
    fn float16_labels_use_the_numeric_float32_path() {
        let array = Float32Array::from(vec![Some(1.25), None, Some(-3.75)]);
        let labels = column(&array, Some("float16")).unwrap();
        let Column::F32(values) = &labels else {
            panic!("float16 labels must stay numeric");
        };
        assert_eq!(values[0], 1.25);
        assert!(values[1].is_nan());
        assert_eq!(values[2], -3.75);
        let output = to_arrow(&labels).unwrap();
        assert_eq!(output.data_type(), &DataType::Float32);
    }

    #[test]
    fn decimals_get_the_type_pyarrow_infers() {
        // `pa.array([Decimal(v), ...]).type` for each list of values.
        for (values, want) in [
            (vec![(12345, 0), (5, 1)], (6, 1)),
            (vec![(5, 1), (12345, 0)], (6, 1)),
            (vec![(1, 3), (100, 0)], (6, 3)),
            (vec![(1, -2), (25, 2)], (5, 2)),
            (vec![(0, 2)], (2, 2)),
            (vec![(9999, 2), (1, 0)], (4, 2)),
            (vec![(-125, 1), (3, 0)], (3, 1)),
        ] {
            assert_eq!(infer_decimal(&values).unwrap(), want, "{values:?}");
        }
    }

    #[test]
    fn lists_of_decimals_share_one_scale() {
        let lists = vec![
            Obj::List(vec![Obj::Decimal(150, 2), Obj::Decimal(12, 0)]),
            Obj::None,
        ];
        let array = object_array(&lists).unwrap();
        assert_eq!(
            array.data_type(),
            &DataType::List(Arc::new(arrow_schema::Field::new(
                "item",
                DataType::Decimal128(4, 2),
                true
            )))
        );
        let items = array.as_list::<i32>().value(0);
        let items = items.as_primitive::<Decimal128Type>();
        assert_eq!((items.value(0), items.value(1)), (150, 1200));
        assert!(array.is_null(1));
    }
}
