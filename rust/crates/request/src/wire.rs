// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `sdm_client/wire/dtypes.py`: how a column's wire dtype is named and how a
//! cell is written under it, for the tabular envelope; and `as_declared`,
//! which reads an answer back as its column's type.

use std::cmp::Ordering;

use sdm_schema::pyval::PyVal;
use serde_json::Value;

use crate::error::{R, invalid};
use crate::num::{Dec, Special};
use crate::pyops::{
    decimal_value, integer_value, is_integral_type, item, py_bool, py_float, py_int, py_repr,
    py_str,
};
use crate::series::{Dtype, Series, infer_dtype, is_na};

pub const JSON_SAFE_INT_MAX: i64 = 9_007_199_254_740_991;
pub const JSON_SAFE_INT_MIN: i64 = -9_007_199_254_740_991;
/// `infer_tfm_dtype`'s answer for a column that carries no type.
pub const UNTYPED: &str = "";
const INT64_SAFE_PRECISION: i64 = 18;
const MAX_JSON_DEPTH: usize = 100;

fn int64_min() -> Dec {
    Dec::from_i128(i128::from(i64::MIN))
}

fn int64_max() -> Dec {
    Dec::from_i128(i128::from(i64::MAX))
}

/// `INT64_MIN <= value <= INT64_MAX` of a finite value.
pub fn fits_int64(value: &Dec) -> bool {
    value.cmp_finite(&int64_min()) != Ordering::Less
        && value.cmp_finite(&int64_max()) != Ordering::Greater
}

/// `is_container(value)`: holds other values rather than being one.
pub fn is_container(value: &PyVal) -> bool {
    match value {
        PyVal::Str(_) => false,
        PyVal::List(_)
        | PyVal::Tuple(_)
        | PyVal::Dict(_)
        | PyVal::Bytes(_)
        | PyVal::NdArray(..) => true,
        PyVal::Opaque { kind, .. } => kind == "mapping" || kind == "container",
        _ => false,
    }
}

/// `require_scalar_cells(series, described)`.
pub fn require_scalar_cells(series: &Series, described: &str) -> R<()> {
    if series.dtype.holds_only_scalars() {
        return Ok(());
    }
    for value in &series.cells {
        if !is_container(value) {
            continue;
        }
        let type_name = container_type(value);
        if matches!(value, PyVal::Bytes(_)) {
            return Err(unsupported_request(format!(
                "{described} holds raw bytes ({type_name}), and the contract has no cell that \
                 carries them. It is already one value per row, so what it needs is a reading \
                 rather than a flattening: decode the column to text, or leave it out of the \
                 schema"
            )));
        }
        let shape = if matches!(value, PyVal::Dict(_))
            || matches!(value, PyVal::Opaque { kind, .. } if kind == "mapping")
        {
            "a mapping"
        } else {
            "a container"
        };
        return Err(unsupported_request(format!(
            "{described} holds {shape} ({type_name}), and a nested column is not something this \
             client can send. Multi-label targets are not supported by these models; for a \
             feature, give one value per row"
        )));
    }
    Ok(())
}

fn unsupported_request(message: String) -> sdm_protocol::Diagnostic {
    crate::error::structured("UNSUPPORTED_REQUEST", message)
}

/// `type(value).__name__` of a container.
pub fn container_type(value: &PyVal) -> String {
    match value {
        PyVal::List(_) => "list".to_string(),
        PyVal::Tuple(_) => "tuple".to_string(),
        PyVal::Dict(_) => "dict".to_string(),
        PyVal::Bytes(_) => "bytes".to_string(),
        PyVal::NdArray(..) => "ndarray".to_string(),
        PyVal::Opaque { type_name, .. } => type_name.clone(),
        _ => "object".to_string(),
    }
}

/// `infer_tfm_dtype(series)`.
pub fn infer_tfm_dtype(series: &Series) -> String {
    let dtype = &series.dtype;
    if dtype.is_bool() {
        return "bool".to_string();
    }
    if dtype.is_datetime() {
        return "timestamp[us]".to_string();
    }
    if dtype.is_integer() {
        return whole_numbers_dtype(series);
    }
    if dtype.is_float() {
        return if dtype.float_itemsize() > 4 {
            "float64"
        } else {
            "float32"
        }
        .to_string();
    }
    if let Some((precision, scale)) = dtype.arrow_decimal() {
        return decimal_dtype(series, Some((precision, scale)));
    }
    if dtype.is_object() {
        let content = infer_dtype(&series.cells);
        return match content {
            "empty" => UNTYPED.to_string(),
            "decimal" => decimal_dtype(series, None),
            "integer" => whole_numbers_dtype(series),
            "floating" | "mixed-integer-float" => "float64".to_string(),
            "boolean" => "bool".to_string(),
            "date" | "datetime" | "datetime64" => "timestamp[us]".to_string(),
            _ => "string".to_string(),
        };
    }
    "string".to_string()
}

/// `_whole_numbers_dtype(series, dtype)`.
fn whole_numbers_dtype(series: &Series) -> String {
    let may_exceed =
        series.dtype.is_object() || series.dtype.name().to_lowercase().contains("uint64");
    if !may_exceed {
        return "int64".to_string();
    }
    for value in &series.cells {
        if is_missing(value) {
            continue;
        }
        match py_int(value) {
            Ok(n) if fits_int64(&n) => {}
            _ => return "string".to_string(),
        }
    }
    "int64".to_string()
}

/// `_decimal_dtype(series, arrow_type)`.
pub fn decimal_dtype(series: &Series, arrow: Option<(u32, i32)>) -> String {
    if let Some((precision, scale)) = arrow {
        if scale > 0 {
            return "float64".to_string();
        }
        if i64::from(precision) <= INT64_SAFE_PRECISION {
            return "int64".to_string();
        }
    }
    let mut integral = true;
    for value in &series.cells {
        if is_missing(value) {
            continue;
        }
        let Some(dec) = decimal_value(value) else {
            return "float64".to_string();
        };
        if !dec.is_finite() {
            return "float64".to_string();
        }
        if dec.exponent < 0 && !dec.is_integral() {
            return "float64".to_string();
        }
        if dec.adjusted() >= INT64_SAFE_PRECISION && !fits_int64(&dec) {
            integral = false;
        }
    }
    if integral { "int64" } else { "string" }.to_string()
}

/// `widen_tfm_dtype(dtypes)`.
pub fn widen_tfm_dtype(dtypes: &[String]) -> String {
    let mut candidates: Vec<&str> = Vec::new();
    for dtype in dtypes {
        if !dtype.is_empty() && !candidates.contains(&dtype.as_str()) {
            candidates.push(dtype);
        }
    }
    let width = |d: &str| match d {
        "bool" => Some(0),
        "int32" => Some(1),
        "int64" => Some(2),
        "float32" => Some(3),
        "float64" => Some(4),
        _ => None,
    };
    match candidates.len() {
        0 => return "string".to_string(),
        1 => return candidates[0].to_string(),
        _ => {}
    }
    if candidates.iter().any(|d| width(d).is_none()) {
        return "string".to_string();
    }
    let mut widest = candidates[0];
    for d in &candidates[1..] {
        if width(d) > width(widest) {
            widest = d;
        }
    }
    if widest == "float32" && candidates.contains(&"int64") {
        return "float64".to_string();
    }
    widest.to_string()
}

/// `_is_missing(value)`: `None`, or what `pd.isna` calls missing among
/// scalars; a container is a value.
pub fn is_missing(value: &PyVal) -> bool {
    if is_container(value) {
        return false;
    }
    is_na(value)
}

/// `_require_finite(value)`.
fn require_finite(value: &PyVal) -> R<()> {
    if let Some(dec) = decimal_value(value) {
        if !dec.is_finite() {
            return Err(invalid(format!(
                "value {} is not JSON-representable; replace non-finite numbers with a finite \
                 value or a null",
                py_repr(value)
            )));
        }
        return Ok(());
    }
    let float = match value {
        PyVal::Float(x) => Some(*x),
        PyVal::Numpy(name, inner) if name.starts_with("float") || name == "longdouble" => {
            match inner.as_ref() {
                PyVal::Float(x) => Some(*x),
                _ => None,
            }
        }
        _ => None,
    };
    if let Some(x) = float
        && !x.is_finite()
    {
        return Err(invalid(format!(
            "value {} is not JSON-representable; replace non-finite numbers with a finite value \
             or a null",
            py_repr(value)
        )));
    }
    Ok(())
}

/// `_exceeds_float64(value)`: a numeric string a float64 would round.
pub fn exceeds_float64(value: &PyVal) -> bool {
    if let PyVal::Str(text) = value {
        let Some(exact) = Dec::parse(text) else {
            return false;
        };
        if !exact.is_finite() {
            return false;
        }
        let x = exact.to_f64();
        if !x.is_finite() {
            return true;
        }
        return !Dec::from_f64(x).equals(&exact);
    }
    false
}

/// `_integral_value(value)`.
fn integral_value(value: &PyVal) -> Option<Dec> {
    if matches!(value, PyVal::Bool(_)) {
        return None;
    }
    if is_integral_type(value) {
        return integer_value(value);
    }
    let dec = decimal_value(value)?;
    (dec.is_finite() && dec.is_integral()).then(|| dec.trunc())
}

/// A finite float as JSON.
pub fn float_json(x: f64) -> Value {
    serde_json::Number::from_f64(x).map_or(Value::Null, Value::Number)
}

/// An integer as JSON: past `i64` as its digits would not be JSON, so it
/// is a float there (never reached for a value that fits the wire).
pub fn int_json(value: &Dec) -> Value {
    match value.to_i64() {
        Some(i) => Value::from(i),
        None => float_json(value.to_f64()),
    }
}

/// `serialize_cell(value, dtype)`.
pub fn serialize_cell(value: &PyVal, dtype: &str) -> R<Value> {
    if is_missing(value) {
        return Ok(Value::Null);
    }
    if dtype == "timestamp[us]" {
        return timestamp_utc_text(value).map(Value::String);
    }
    if dtype == "int64" || dtype == "int32" {
        require_finite(value)?;
        let integer = py_int(value)?;
        if !is_integral_type(value) && !equals_number(&integer, value) {
            return Err(invalid(format!(
                "value {} cannot be sent as {dtype} without losing its fractional part",
                py_repr(value)
            )));
        }
        if !fits_int64(&integer) {
            return Err(invalid(format!(
                "value {} is outside the range {dtype} represents (-9223372036854775808 to \
                 9223372036854775807); send the column as strings",
                py_repr(value)
            )));
        }
        let n = integer.to_i64().unwrap_or(0);
        if !(JSON_SAFE_INT_MIN..=JSON_SAFE_INT_MAX).contains(&n) {
            return Ok(Value::String(n.to_string()));
        }
        return Ok(Value::from(n));
    }
    if dtype == "float64" || dtype == "float32" {
        require_finite(value)?;
        let number = match py_float(value) {
            Ok(x) => x,
            Err(e) if e.class.as_deref() == Some("builtins.OverflowError") => {
                return Err(invalid(format!(
                    "value {} is outside the range {dtype} represents; send the column as \
                     strings to keep it exact",
                    py_repr(value)
                )));
            }
            Err(e) => return Err(e),
        };
        if !number.is_finite() {
            return Err(invalid(format!(
                "value {} is outside the range {dtype} represents; send the column as strings \
                 to keep it exact",
                py_repr(value)
            )));
        }
        if exceeds_float64(value) {
            return Err(invalid(format!(
                "value {} cannot be sent as {dtype} without losing precision; send the column \
                 as strings to keep it exact",
                py_repr(value)
            )));
        }
        let whole = integral_value(value);
        if whole.is_none()
            && let Some(dec) = decimal_value(value)
        {
            let limit = Dec::from_i128(i128::from(JSON_SAFE_INT_MAX));
            let magnitude = Dec {
                negative: false,
                ..dec
            };
            if magnitude.cmp_finite(&limit) == Ordering::Greater {
                return Err(invalid(format!(
                    "value {} is too large for {dtype} to represent; send the column as strings \
                     to keep it exact",
                    py_repr(value)
                )));
            }
        }
        if let Some(whole) = whole {
            let back = Dec::of_whole_f64(number.trunc());
            if !back.is_some_and(|b| b.equals(&whole)) {
                return Err(invalid(format!(
                    "value {} cannot be sent as {dtype} without losing precision, because it is \
                     outside the range a {dtype} represents exactly; give the column the same \
                     integer dtype in every frame, or convert it to strings",
                    py_repr(value)
                )));
            }
        }
        return Ok(float_json(number));
    }
    if dtype == "bool" {
        return Ok(Value::Bool(py_bool(value)));
    }
    if let Some(dec) = decimal_value(value) {
        return Ok(Value::String(dec.format_f()));
    }
    Ok(Value::String(py_str(value)))
}

/// `integer == value` as Python compares an `int` to a number.
fn equals_number(integer: &Dec, value: &PyVal) -> bool {
    match item(value) {
        PyVal::Float(x) => Dec::of_whole_f64(x).is_some_and(|w| w.equals(integer)),
        PyVal::Decimal(text) => Dec::parse(&text).is_some_and(|d| d.equals(integer)),
        PyVal::Int(_) | PyVal::BigInt(_) | PyVal::Bool(_) => {
            integer_value(value).is_some_and(|v| v.equals(integer))
        }
        _ => false,
    }
}

/// `pd.Timestamp(value)` written at UTC, to the microsecond, with `Z`.
pub fn timestamp_utc_text(value: &PyVal) -> R<String> {
    let (wall, offset) = match value {
        PyVal::Timestamp(w) | PyVal::DateTime(w, None) => (*w, 0),
        PyVal::TimestampTz(w, o) | PyVal::DateTime(w, Some(o)) => (*w, *o),
        PyVal::Date(y, m, d) => (sdm_schema::dt::civil::Wall::date(*y, *m, *d), 0),
        PyVal::Int(nanos) => (sdm_schema::dt::civil::Wall::from_epoch_nanos(*nanos), 0),
        PyVal::Float(x) if x.is_finite() => {
            (sdm_schema::dt::civil::Wall::from_epoch_nanos(*x as i128), 0)
        }
        PyVal::Numpy(..) => return timestamp_utc_text(&item(value)),
        PyVal::Str(text) => match PyVal::decode(&serde_json::json!({ "$timestamp": text })) {
            Ok(PyVal::Timestamp(w)) => (w, 0),
            Ok(PyVal::TimestampTz(w, o)) => (w, o),
            _ => {
                return Err(crate::error::value_error(format!(
                    "could not convert string to Timestamp: {}",
                    py_repr(value)
                )));
            }
        },
        other => {
            return Err(crate::error::builtin(
                "TypeError",
                format!("Cannot convert input {} to Timestamp", py_repr(other)),
            ));
        }
    };
    let instant = wall.epoch_nanos() - i128::from(offset) * 1_000_000_000;
    let utc = sdm_schema::dt::civil::Wall::from_epoch_nanos(instant);
    Ok(format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:06}Z",
        utc.year,
        utc.month,
        utc.day,
        utc.hour,
        utc.minute,
        utc.second,
        utc.nano / 1000
    ))
}

/// `serialize_column(series, dtype)`.
pub fn serialize_column(series: &Series, dtype: &str) -> R<Vec<Value>> {
    let described = format!(
        "column {}",
        match &series.name {
            Some(name) => sdm_pql::pyrepr::str_repr(name),
            None => "None".to_string(),
        }
    );
    require_scalar_cells(series, &described)?;
    series
        .cells
        .iter()
        .map(|value| serialize_cell(value, dtype))
        .collect()
}

/// `_refuse_inexact_integer(value)`.
fn refuse_inexact_integer(value: &PyVal) -> R<()> {
    if !is_integral_type(value) || matches!(value, PyVal::Bool(_)) {
        return Ok(());
    }
    let Some(number) = integer_value(value) else {
        return Ok(());
    };
    let x = number.to_f64();
    if x.is_finite() && Dec::of_whole_f64(x).is_some_and(|w| w.equals(&number)) {
        return Ok(());
    }
    Err(invalid(format!(
        "value {} cannot be sent as a JSON number without losing precision, since a float64 \
         does not hold it exactly; pass it as a string if the model accepts one",
        number.format_f()
    )))
}

/// `json_ready(value)`.
pub fn json_ready(value: &PyVal) -> R<PyVal> {
    json_ready_at(value, 0)
}

fn json_ready_at(value: &PyVal, depth: usize) -> R<PyVal> {
    if let PyVal::Held { value: held, .. } = value {
        json_ready_at(held, depth)?;
        return Ok(value.clone());
    }
    let value = item(value);
    refuse_inexact_integer(&value)?;
    if matches!(&value, PyVal::Opaque { kind, .. } if kind == sdm_schema::pyval::REFERS_BACK) {
        return Err(invalid(
            "this value refers back to itself, so it cannot be sent as JSON",
        ));
    }
    if depth > MAX_JSON_DEPTH {
        return Err(invalid(format!(
            "this value nests more than {MAX_JSON_DEPTH} levels deep, or refers back to itself, \
             so it cannot be sent as JSON"
        )));
    }
    Ok(match value {
        PyVal::Dict(items) => PyVal::Dict(
            items
                .iter()
                .map(|(k, v)| Ok((k.clone(), json_ready_at(v, depth + 1)?)))
                .collect::<R<_>>()?,
        ),
        PyVal::List(items) | PyVal::Tuple(items) => PyVal::List(
            items
                .iter()
                .map(|v| json_ready_at(v, depth + 1))
                .collect::<R<_>>()?,
        ),
        other => other,
    })
}

/// `_as_bool(value)`.
fn as_bool(value: &PyVal) -> PyVal {
    match value {
        PyVal::Bool(_) => value.clone(),
        PyVal::Str(text) => {
            let t = text.trim().to_lowercase();
            if matches!(t.as_str(), "true" | "t" | "1") {
                PyVal::Bool(true)
            } else if matches!(t.as_str(), "false" | "f" | "0") {
                PyVal::Bool(false)
            } else {
                value.clone()
            }
        }
        v if is_integral_type(v) => match integer_value(v).and_then(|d| d.to_i64()) {
            Some(0) => PyVal::Bool(false),
            Some(1) => PyVal::Bool(true),
            _ => value.clone(),
        },
        _ => value.clone(),
    }
}

/// `as_declared(value, dtype)`: the value read back as the wire dtype
/// declares, or unchanged where it cannot be.
pub fn as_declared(value: &PyVal, dtype: Option<&str>) -> PyVal {
    let Some(dtype) = dtype else {
        return value.clone();
    };
    if matches!(value, PyVal::None) {
        return value.clone();
    }
    if dtype.starts_with("bool") {
        return as_bool(value);
    }
    if matches!(value, PyVal::Bool(_)) {
        return value.clone();
    }
    if dtype.starts_with("int") {
        let Ok(read) = py_int(value) else {
            return value.clone();
        };
        if matches!(value, PyVal::Str(_)) {
            return dec_to_pyval(&read);
        }
        return if equals_number(&read, value) {
            dec_to_pyval(&read)
        } else {
            value.clone()
        };
    }
    if dtype.starts_with("float") {
        return match py_float(value) {
            Ok(x) => PyVal::Float(x),
            Err(_) => value.clone(),
        };
    }
    value.clone()
}

/// An integer as a Python `int`.
pub fn dec_to_pyval(value: &Dec) -> PyVal {
    match value.to_i128() {
        Some(n) => PyVal::Int(n),
        None => PyVal::BigInt(if value.negative {
            format!("-{}", value.trunc_digits())
        } else {
            value.trunc_digits()
        }),
    }
}

/// Whether a `Decimal`'s special value is a NaN.
pub fn is_decimal_nan(value: &PyVal) -> bool {
    decimal_value(value).is_some_and(|d| matches!(d.special, Special::NaN | Special::SNaN))
}

/// The dtype a fixture series names, for the operations that take one.
pub fn series_dtype(series: &Series) -> &Dtype {
    &series.dtype
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_value_referring_back_to_itself_is_refused_as_json_ready_refuses_it() {
        let given = PyVal::decode(&serde_json::json!({
            "options": {"self": {"$refers_back": "dict"}}
        }))
        .unwrap();
        let refused = json_ready(&given).unwrap_err();
        assert_eq!(
            refused.message,
            "this value refers back to itself, so it cannot be sent as JSON"
        );
    }

    #[test]
    fn a_held_cycle_is_refused_without_replacing_valid_held_values() {
        let given = PyVal::decode(&serde_json::json!({
            "options": {"$held": {
                "reference": {"$host_ref": "0"},
                "value": {"$refers_back": "dict"}
            }}
        }))
        .unwrap();
        let refused = json_ready(&given).unwrap_err();
        assert_eq!(
            refused.message,
            "this value refers back to itself, so it cannot be sent as JSON"
        );
        let valid = PyVal::decode(&serde_json::json!({"$held": {
            "reference": {"$host_ref": "1"},
            "value": [1, 2]
        }}))
        .unwrap();
        assert_eq!(json_ready(&valid).unwrap(), valid);
    }
}
