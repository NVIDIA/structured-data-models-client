// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! How the relational envelope (`rfm/inference/payload.py`) names a column's
//! wire dtype and writes a cell under it: `_dtype_name`, `_json_value`,
//! `_cell_json_value`, `_target_json_value` and `_class_label`.

use std::cmp::Ordering;

use sdm_schema::pyval::PyVal;
use serde_json::Value;

use crate::error::{R, value_error};
use crate::num::Dec;
use crate::pyops::{decimal_value, integer_value, item, py_float, py_int, py_repr, py_str};
use crate::series::{Series, infer_dtype, is_na};
use crate::wire::{JSON_SAFE_INT_MAX, JSON_SAFE_INT_MIN, exceeds_float64, fits_int64};

/// `_is_null_like(value)`: what `pd.isna` calls missing among scalars.
pub fn is_null_like(value: &PyVal) -> bool {
    match value {
        PyVal::List(_) | PyVal::Tuple(_) | PyVal::Dict(_) | PyVal::NdArray(..) => false,
        other => is_na(other),
    }
}

/// `_decimal_text(value)`: a finite decimal's own digits.
pub fn decimal_text(value: &PyVal) -> Option<String> {
    let dec = decimal_value(value)?;
    dec.is_finite().then(|| dec.format_f())
}

/// `_timestamp_json_value(value)`: at UTC, to the microsecond, with `Z`.
pub fn timestamp_json_value(value: &PyVal) -> R<PyVal> {
    // `pd.Timestamp` reads int64's least value, its NaT sentinel, as NaT,
    // whether it comes as an int or as the float it rounds to.
    let nat = i128::from(i64::MIN);
    match value {
        PyVal::NaT => return Ok(PyVal::None),
        PyVal::Int(n) if *n == nat => return Ok(PyVal::None),
        PyVal::Float(x) if *x == i64::MIN as f64 => return Ok(PyVal::None),
        _ => {}
    }
    crate::wire::timestamp_utc_text(value).map(PyVal::Str)
}

/// `_json_value(value)`: a Python value as the JSON value it travels as.
pub fn json_value(value: &PyVal) -> R<PyVal> {
    if matches!(value, PyVal::None) {
        return Ok(PyVal::None);
    }
    let value = item(value);
    if let Some(dec) = decimal_value(&value) {
        if !dec.is_finite() {
            return Ok(PyVal::None);
        }
        if !dec.is_integral() {
            let number = dec.to_f64();
            let limit = Dec::from_i128(i128::from(JSON_SAFE_INT_MAX));
            let magnitude = Dec {
                negative: false,
                ..dec.clone()
            };
            if !number.is_finite() || magnitude.cmp_finite(&limit) == Ordering::Greater {
                return Err(value_error(format!(
                    "value {} is too large for float64 to represent; send the column as strings \
                     to keep it exact",
                    py_repr(&value)
                )));
            }
            return Ok(PyVal::Float(number));
        }
        if fits_int64(&dec) {
            return Ok(crate::wire::dec_to_pyval(&dec.trunc()));
        }
        return Ok(PyVal::Str(dec.format_f()));
    }
    Ok(match value {
        PyVal::Timestamp(_) | PyVal::TimestampTz(..) => timestamp_json_value(&value)?,
        PyVal::Timedelta(nanos) => PyVal::Int(i128::from(nanos)),
        PyVal::Bytes(bytes) => PyVal::Str(base64(&bytes)),
        PyVal::NdArray(_, items) | PyVal::List(items) | PyVal::Tuple(items) => {
            PyVal::List(items.iter().map(json_value).collect::<R<Vec<_>>>()?)
        }
        PyVal::Dict(items) => PyVal::Dict(
            items
                .iter()
                .map(|(k, v)| Ok((PyVal::Str(py_str(k)), json_value(v)?)))
                .collect::<R<Vec<_>>>()?,
        ),
        PyVal::Float(x) if !x.is_finite() => {
            if x.is_nan() {
                PyVal::None
            } else {
                return Err(value_error(format!(
                    "encountered a non-finite value ({}); NIM requests must contain finite \
                     numbers or nulls",
                    crate::pyops::float_text(x)
                )));
            }
        }
        PyVal::Na | PyVal::NaT => PyVal::None,
        other => other,
    })
}

fn base64(bytes: &[u8]) -> String {
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::new();
    for chunk in bytes.chunks(3) {
        let b = [
            chunk[0],
            *chunk.get(1).unwrap_or(&0),
            *chunk.get(2).unwrap_or(&0),
        ];
        let n = (u32::from(b[0]) << 16) | (u32::from(b[1]) << 8) | u32::from(b[2]);
        for i in 0..4 {
            if i <= chunk.len() {
                out.push(ALPHABET[((n >> (18 - 6 * i)) & 63) as usize] as char);
            } else {
                out.push('=');
            }
        }
    }
    out
}

/// `_wire_string(value)`.
fn wire_string(value: &PyVal) -> String {
    match value {
        PyVal::Str(s) => s.clone(),
        other => py_str(other),
    }
}

/// `_integral_source(value)`: an exact integer, before any conversion.
fn integral_source(value: &PyVal) -> Option<Dec> {
    match value {
        PyVal::Bool(_) => None,
        PyVal::Int(_) | PyVal::BigInt(_) => integer_value(value),
        PyVal::Numpy(name, _) if name.starts_with("int") || name.starts_with("uint") => {
            integer_value(value)
        }
        PyVal::Decimal(_) => {
            let dec = decimal_value(value)?;
            (dec.is_finite() && dec.is_integral()).then(|| dec.trunc())
        }
        _ => None,
    }
}

/// `_survives_float(whole)`.
fn survives_float(whole: &Dec) -> bool {
    let x = whole.to_f64();
    x.is_finite() && Dec::of_whole_f64(x).is_some_and(|w| w.equals(whole))
}

/// `_is_finite_input(value)`: `Decimal(str(value)).is_finite()`.
fn is_finite_input(value: &PyVal) -> bool {
    Dec::parse(&py_str(value)).is_some_and(|d| d.is_finite())
}

/// `_cell_json_value(value, dtype)`: one cell, as the column's declared
/// dtype writes it.
pub fn cell_json_value(value: &PyVal, dtype: &str) -> R<PyVal> {
    if dtype == "string"
        && let Some(text) = decimal_text(value)
    {
        return Ok(PyVal::Str(text));
    }
    if dtype == "stringlist"
        && let PyVal::List(items) | PyVal::Tuple(items) | PyVal::NdArray(_, items) = value
    {
        return items
            .iter()
            .map(|item| {
                if matches!(item, PyVal::None) {
                    return Ok(PyVal::None);
                }
                if let Some(text) = decimal_text(item) {
                    return Ok(PyVal::Str(text));
                }
                Ok(PyVal::Str(wire_string(&json_value(item)?)))
            })
            .collect::<R<Vec<_>>>()
            .map(PyVal::List);
    }
    let original = integral_source(value);
    let value = json_value(value)?;
    if matches!(value, PyVal::None) {
        return Ok(PyVal::None);
    }
    if dtype == "stringlist" {
        let PyVal::List(items) = &value else {
            return Err(value_error(format!(
                "expected an array of values for a 'stringlist' column, but got {}",
                py_repr(&value)
            )));
        };
        return Ok(PyVal::List(
            items
                .iter()
                .map(|item| match item {
                    PyVal::None => PyVal::None,
                    other => PyVal::Str(wire_string(other)),
                })
                .collect(),
        ));
    }
    if dtype == "string" {
        return Ok(PyVal::Str(wire_string(&value)));
    }
    if dtype == "bool" {
        return Ok(PyVal::Bool(crate::pyops::py_bool(&value)));
    }
    if dtype == "int64" || dtype == "int32" {
        let integer = py_int(&value)?;
        if !fits_int64(&integer) {
            return Err(value_error(format!(
                "value {} is outside the range {dtype} represents (-9223372036854775808 to \
                 9223372036854775807); send the column as strings",
                py_repr(&value)
            )));
        }
        let n = integer.to_i64().unwrap_or(0);
        if !(JSON_SAFE_INT_MIN..=JSON_SAFE_INT_MAX).contains(&n) {
            return Ok(PyVal::Str(n.to_string()));
        }
        return Ok(PyVal::Int(i128::from(n)));
    }
    if dtype == "float64" || dtype == "float32" {
        if exceeds_float64(&value) {
            return Err(value_error(format!(
                "value {} cannot be sent as {dtype} without losing precision; send the column as \
                 strings to keep it exact",
                py_repr(&value)
            )));
        }
        if let Some(whole) = &original
            && !survives_float(whole)
        {
            return Err(value_error(format!(
                "value {} cannot be sent as {dtype} without losing precision; give the column \
                 an integer dtype, or convert it to strings",
                py_repr(&value)
            )));
        }
        let number = match py_float(&value) {
            Ok(x) => x,
            Err(e) if e.class.as_deref() == Some("builtins.OverflowError") => {
                return Err(value_error(format!(
                    "value {} is too large for {dtype} to represent; send the column as strings \
                     to keep it exact",
                    py_repr(&value)
                )));
            }
            Err(e) => return Err(e),
        };
        if !number.is_finite() && is_finite_input(&value) {
            return Err(value_error(format!(
                "value {} is too large for {dtype} to represent; send the column as strings to \
                 keep it exact",
                py_repr(&value)
            )));
        }
        return Ok(PyVal::Float(number));
    }
    if dtype == "timestamp[us]" {
        if let PyVal::Str(_) = value {
            return Ok(value);
        }
        return timestamp_json_value(&value);
    }
    match &value {
        PyVal::Str(_)
        | PyVal::Int(_)
        | PyVal::BigInt(_)
        | PyVal::Float(_)
        | PyVal::Bool(_)
        | PyVal::List(_) => Ok(value),
        other => Err(value_error(format!(
            "cannot serialize a value of type {} under wire dtype '{dtype}'; convert the column \
             to a supported dtype",
            type_name(other)
        ))),
    }
}

fn type_name(value: &PyVal) -> String {
    match value {
        PyVal::Dict(_) => "dict".to_string(),
        PyVal::Uuid(_) => "UUID".to_string(),
        PyVal::Date(..) => "date".to_string(),
        PyVal::DateTime(..) => "datetime".to_string(),
        PyVal::Time { .. } => "time".to_string(),
        PyVal::Opaque { type_name, .. } => type_name.clone(),
        _ => "object".to_string(),
    }
}

/// `_list_dtype_name(data)`.
fn list_dtype_name(series: &Series) -> R<Option<String>> {
    if !series.dtype.is_object() {
        return Ok(None);
    }
    let mut found_list = false;
    let mut found_scalar = false;
    for value in &series.cells {
        if is_null_like(value) {
            continue;
        }
        let items = match value {
            PyVal::NdArray(_, items) | PyVal::List(items) | PyVal::Tuple(items) => items,
            _ => {
                found_scalar = true;
                continue;
            }
        };
        found_list = true;
        let shown = match value {
            PyVal::NdArray(_, items) => PyVal::List(items.clone()),
            other => other.clone(),
        };
        for item in items {
            if is_null_like(item) || matches!(json_value(item)?, PyVal::None) {
                return Err(value_error(format!(
                    "stringlist columns must not contain null items, but got {}.",
                    py_repr(&shown)
                )));
            }
        }
    }
    if found_list && found_scalar {
        return Err(value_error(
            "Columns with stringlist values must contain only arrays or nulls.",
        ));
    }
    Ok(found_list.then(|| "stringlist".to_string()))
}

/// `_decimal_dtype_name(data)`.
pub fn decimal_dtype_name(series: &Series) -> Option<String> {
    if let Some((precision, scale)) = series.dtype.arrow_decimal() {
        if scale != 0 {
            return Some("float64".to_string());
        }
        if precision <= 18 {
            return Some("int64".to_string());
        }
        let fits = series.cells.iter().all(|v| {
            is_null_like(v) || decimal_value(v).is_some_and(|d| d.is_finite() && fits_int64(&d))
        });
        return Some(if fits { "int64" } else { "string" }.to_string());
    }
    if series.dtype.is_arrow() || !series.dtype.is_object() {
        return None;
    }
    let (mut found, mut fractional, mut too_wide) = (false, false, false);
    for value in &series.cells {
        if is_null_like(value) {
            continue;
        }
        let dec = decimal_value(value)?;
        found = true;
        if !dec.is_finite() {
            continue;
        }
        if !dec.is_integral() {
            fractional = true;
        } else if !fits_int64(&dec) {
            too_wide = true;
        }
    }
    if !found {
        return None;
    }
    Some(
        if too_wide {
            "string"
        } else if fractional {
            "float64"
        } else {
            "int64"
        }
        .to_string(),
    )
}

/// `_dtype_name(data)`: the wire dtype a column travels under.
pub fn dtype_name(series: &Series) -> R<String> {
    if let Some(list) = list_dtype_name(series)? {
        return Ok(list);
    }
    if let Some(decimal) = decimal_dtype_name(series) {
        return Ok(decimal);
    }
    let dtype = &series.dtype;
    if dtype.is_bool() {
        return Ok("bool".to_string());
    }
    if dtype.is_integer() {
        return Ok("int64".to_string());
    }
    if dtype.is_float() {
        return Ok(if dtype.float_itemsize() <= 4 {
            "float32"
        } else {
            "float64"
        }
        .to_string());
    }
    if dtype.is_datetime() {
        return Ok("timestamp[us]".to_string());
    }
    if dtype.is_timedelta() {
        return Ok("int64".to_string());
    }
    if dtype.is_object() {
        return Ok(match infer_dtype(&series.cells) {
            "integer" | "timedelta" | "timedelta64" => "int64",
            "floating" | "mixed-integer-float" | "decimal" => "float64",
            "boolean" => "bool",
            "date" | "datetime" | "datetime64" => "timestamp[us]",
            _ => "string",
        }
        .to_string());
    }
    Ok("string".to_string())
}

/// `TaskType(...).is_link_pred`.
pub fn is_link_pred(task_type: &str) -> bool {
    matches!(
        task_type,
        "temporal_link_prediction" | "static_link_prediction" | "link_prediction"
    )
}

/// `_target_json_value(task_type, value)`.
pub fn target_json_value(task_type: &str, value: &PyVal) -> R<PyVal> {
    if is_link_pred(task_type) {
        if matches!(value, PyVal::None) {
            return Ok(PyVal::None);
        }
        let items = match value {
            PyVal::NdArray(_, items) | PyVal::List(items) | PyVal::Tuple(items) => items,
            other => {
                return Err(value_error(format!(
                    "Link prediction target values must be stringlist arrays, but got {}.",
                    py_repr(other)
                )));
            }
        };
        let shown = match value {
            PyVal::NdArray(_, items) => PyVal::List(items.clone()),
            other => other.clone(),
        };
        let null_item = || {
            value_error(format!(
                "Link prediction target stringlist values must not contain null items, but got \
                 {}.",
                py_repr(&shown)
            ))
        };
        let mut result = Vec::new();
        for item in items {
            if is_null_like(item) {
                return Err(null_item());
            }
            if let Some(text) = decimal_text(item) {
                result.push(PyVal::Str(text));
                continue;
            }
            let json = json_value(item)?;
            if matches!(json, PyVal::None) {
                return Err(null_item());
            }
            result.push(PyVal::Str(py_str(&json)));
        }
        return Ok(PyVal::List(result));
    }
    let binary = task_type == "binary_classification";
    if !binary && decimal_text(value).is_some() {
        return Ok(value.clone());
    }
    let value = json_value(value)?;
    if !binary || matches!(value, PyVal::None) {
        return Ok(value);
    }
    match &value {
        PyVal::Bool(_) => Ok(value),
        PyVal::Int(0) => Ok(PyVal::Bool(false)),
        PyVal::Int(1) => Ok(PyVal::Bool(true)),
        PyVal::Float(x) if *x == 0.0 => Ok(PyVal::Bool(false)),
        PyVal::Float(x) if *x == 1.0 => Ok(PyVal::Bool(true)),
        other => Err(value_error(format!(
            "Binary classification target values must be booleans or numeric 0/1 values, but \
             got {}.",
            py_repr(other)
        ))),
    }
}

/// `_class_label(value, dtype)`.
pub fn class_label(value: &PyVal, dtype: &str) -> R<String> {
    if dtype == "bool" {
        return Ok(if crate::pyops::py_bool(value) {
            "true"
        } else {
            "false"
        }
        .to_string());
    }
    if let Some(text) = decimal_text(value) {
        return Ok(text);
    }
    Ok(py_str(&json_value(value)?))
}

/// A value of `_json_value` or `_cell_json_value` as JSON.
pub fn to_json(value: &PyVal) -> Value {
    match value {
        PyVal::None | PyVal::Na | PyVal::NaT => Value::Null,
        PyVal::Bool(b) => Value::Bool(*b),
        PyVal::Int(n) => match i64::try_from(*n) {
            Ok(i) => Value::from(i),
            Err(_) => match u64::try_from(*n) {
                Ok(u) => Value::from(u),
                Err(_) => crate::wire::float_json(*n as f64),
            },
        },
        PyVal::BigInt(digits) => {
            crate::wire::float_json(Dec::integer(digits).map_or(f64::NAN, |d| d.to_f64()))
        }
        PyVal::Float(x) => crate::wire::float_json(*x),
        PyVal::Str(s) => Value::String(s.clone()),
        PyVal::List(items) | PyVal::Tuple(items) | PyVal::NdArray(_, items) => {
            Value::Array(items.iter().map(to_json).collect())
        }
        PyVal::Dict(items) => {
            Value::Object(items.iter().map(|(k, v)| (py_str(k), to_json(v))).collect())
        }
        PyVal::Numpy(_, inner) => to_json(inner),
        other => Value::String(py_str(other)),
    }
}
