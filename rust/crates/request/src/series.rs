// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A pandas `Series` as the request code reads one: its dtype, as
//! `str(series.dtype)` names it, and its cells as iterating it gives them
//! (`series.tolist()`: Python scalars, `Timestamp`, `pd.NA`, `NaT`).
//!
//! What the wire writes for a column follows from the dtype first and the
//! cells second, so both are kept as pandas has them; `pd.api.types`'
//! predicates and `infer_dtype` are answered here.

use sdm_schema::dt::civil::Wall;
use sdm_schema::pyval::PyVal;
use serde_json::Value;

use crate::num::Dec;

/// A pandas dtype, by what the request code distinguishes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Dtype {
    /// numpy `bool`.
    Bool,
    /// A numpy integer, such as `int64` or `uint8`.
    Int(String),
    /// A numpy float of `bits`.
    Float(u32),
    /// numpy `datetime64[unit]`, or with a zone, `datetime64[unit, tz]`.
    Datetime(String, Option<String>),
    /// numpy `timedelta64[unit]`.
    Timedelta(String),
    /// numpy `object`.
    Object,
    /// pandas 3's default string dtype, `str`.
    Str,
    /// `string`, or `string[pyarrow]`.
    String(String),
    /// `category`.
    Category,
    /// A nullable extension: `Int64`, `UInt64`, `Float64`, `boolean`.
    Nullable(String),
    /// `pd.ArrowDtype`, by its pyarrow type's name: `decimal128(38, 0)`,
    /// `int64`, `double`, `string`, `bool`, ...
    Arrow(String),
    /// Any other dtype, by its name: `complex128`, `period[M]`,
    /// `interval[int64, right]`, which the request path has no reading of.
    Other(String),
}

impl Dtype {
    /// The dtype named `name`, as `str(dtype)` writes it.
    pub fn parse(name: &str) -> Self {
        if let Some(arrow) = name.strip_suffix("[pyarrow]") {
            if arrow == "string" {
                return Dtype::String(name.to_string());
            }
            return Dtype::Arrow(arrow.to_string());
        }
        if let Some(rest) = name.strip_prefix("datetime64[") {
            let inner = rest.trim_end_matches(']');
            return match inner.split_once(", ") {
                Some((unit, tz)) => Dtype::Datetime(unit.to_string(), Some(tz.to_string())),
                None => Dtype::Datetime(inner.to_string(), None),
            };
        }
        if let Some(rest) = name.strip_prefix("timedelta64[") {
            return Dtype::Timedelta(rest.trim_end_matches(']').to_string());
        }
        match name {
            "bool" => Dtype::Bool,
            "float64" => Dtype::Float(64),
            "float32" => Dtype::Float(32),
            "float16" => Dtype::Float(16),
            "object" | "O" => Dtype::Object,
            "str" => Dtype::Str,
            "string" => Dtype::String(name.to_string()),
            "category" => Dtype::Category,
            "Int8" | "Int16" | "Int32" | "Int64" | "UInt8" | "UInt16" | "UInt32" | "UInt64"
            | "Float32" | "Float64" | "boolean" => Dtype::Nullable(name.to_string()),
            // `int64`, `uint8`, ...; not `interval[int64, right]`.
            n if n
                .strip_prefix("uint")
                .or_else(|| n.strip_prefix("int"))
                .is_some_and(|bits| {
                    !bits.is_empty() && bits.chars().all(|c| c.is_ascii_digit())
                }) =>
            {
                Dtype::Int(n.to_string())
            }
            other => Dtype::Other(other.to_string()),
        }
    }

    /// `str(dtype)`.
    pub fn name(&self) -> String {
        match self {
            Dtype::Bool => "bool".to_string(),
            Dtype::Int(name) | Dtype::String(name) | Dtype::Nullable(name) | Dtype::Other(name) => {
                name.clone()
            }
            Dtype::Float(bits) => format!("float{bits}"),
            Dtype::Datetime(unit, None) => format!("datetime64[{unit}]"),
            Dtype::Datetime(unit, Some(tz)) => format!("datetime64[{unit}, {tz}]"),
            Dtype::Timedelta(unit) => format!("timedelta64[{unit}]"),
            Dtype::Object => "object".to_string(),
            Dtype::Str => "str".to_string(),
            Dtype::Category => "category".to_string(),
            Dtype::Arrow(name) => format!("{name}[pyarrow]"),
        }
    }

    /// `pd.api.types.is_bool_dtype`.
    pub fn is_bool(&self) -> bool {
        match self {
            Dtype::Bool => true,
            Dtype::Nullable(n) => n == "boolean",
            Dtype::Arrow(a) => a == "bool",
            _ => false,
        }
    }

    /// `pd.api.types.is_integer_dtype`.
    pub fn is_integer(&self) -> bool {
        match self {
            Dtype::Int(_) => true,
            Dtype::Nullable(n) => n.starts_with("Int") || n.starts_with("UInt"),
            Dtype::Arrow(a) => a.starts_with("int") || a.starts_with("uint"),
            _ => false,
        }
    }

    /// `pd.api.types.is_float_dtype`.
    pub fn is_float(&self) -> bool {
        match self {
            Dtype::Float(_) => true,
            Dtype::Nullable(n) => n.starts_with("Float"),
            Dtype::Arrow(a) => matches!(a.as_str(), "double" | "float" | "halffloat"),
            _ => false,
        }
    }

    /// `dtype.itemsize` of a float dtype.
    pub fn float_itemsize(&self) -> usize {
        match self {
            Dtype::Float(bits) => *bits as usize / 8,
            Dtype::Nullable(n) if n == "Float32" => 4,
            Dtype::Arrow(a) if a == "float" => 4,
            Dtype::Arrow(a) if a == "halffloat" => 2,
            _ => 8,
        }
    }

    /// `pd.api.types.is_datetime64_any_dtype`.
    pub fn is_datetime(&self) -> bool {
        match self {
            Dtype::Datetime(..) => true,
            Dtype::Arrow(a) => a.starts_with("timestamp"),
            _ => false,
        }
    }

    /// `pd.api.types.is_timedelta64_dtype`.
    pub fn is_timedelta(&self) -> bool {
        match self {
            Dtype::Timedelta(_) => true,
            Dtype::Arrow(a) => a.starts_with("duration"),
            _ => false,
        }
    }

    /// `pd.api.types.is_object_dtype`.
    pub fn is_object(&self) -> bool {
        matches!(self, Dtype::Object)
    }

    /// `pd.api.types.is_numeric_dtype`.
    pub fn is_numeric(&self) -> bool {
        self.is_bool()
            || self.is_integer()
            || self.is_float()
            || matches!(self, Dtype::Arrow(a) if a.starts_with("decimal"))
            || self.is_timedelta()
    }

    /// The precision and scale of an Arrow decimal dtype.
    pub fn arrow_decimal(&self) -> Option<(u32, i32)> {
        let Dtype::Arrow(name) = self else {
            return None;
        };
        let body = name
            .strip_prefix("decimal128(")
            .or_else(|| name.strip_prefix("decimal256("))
            .or_else(|| name.strip_prefix("decimal32("))
            .or_else(|| name.strip_prefix("decimal64("))?
            .strip_suffix(')')?;
        let (p, s) = body.split_once(", ")?;
        Some((p.parse().ok()?, s.parse().ok()?))
    }

    /// Whether the dtype is an Arrow type at all (`pyarrow_dtype` exists).
    pub fn is_arrow(&self) -> bool {
        matches!(self, Dtype::Arrow(_))
            || matches!(self, Dtype::String(n) if n.ends_with("[pyarrow]"))
    }

    /// `isinstance(dtype, pd.StringDtype) or dtype == 'str'`.
    pub fn is_string_dtype(&self) -> bool {
        matches!(self, Dtype::Str | Dtype::String(_))
    }

    /// `_holds_only_scalars(dtype)` of `wire/dtypes.py`.
    pub fn holds_only_scalars(&self) -> bool {
        if let Dtype::Arrow(name) = self {
            let base = name.split('[').next().unwrap_or(name);
            return base.starts_with("int")
                || base.starts_with("uint")
                || matches!(base, "double" | "float" | "halffloat" | "bool" | "string")
                || base.starts_with("decimal")
                || base.starts_with("large_string")
                || base.starts_with("string_view")
                || base.starts_with("date")
                || base.starts_with("time")
                || base.starts_with("duration")
                || base == "null";
        }
        if matches!(self, Dtype::String(_)) {
            return true;
        }
        self.is_numeric()
            || self.is_bool()
            || self.is_datetime()
            || self.is_timedelta()
            || matches!(self, Dtype::Str)
    }
}

/// A column.
#[derive(Clone, Debug)]
pub struct Series {
    pub name: Option<String>,
    pub dtype: Dtype,
    pub cells: Vec<PyVal>,
}

impl Series {
    pub fn new(name: Option<String>, dtype: Dtype, cells: Vec<PyVal>) -> Self {
        Self { name, dtype, cells }
    }

    pub fn len(&self) -> usize {
        self.cells.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }

    /// `pd.Series(values, dtype=name)` as `_fixture_ops.build_series`
    /// builds it from a fixture's `{"dtype", "values"}`.
    pub fn from_spec(spec: &Value) -> Result<Series, String> {
        let dtype_name = spec
            .get("dtype")
            .and_then(Value::as_str)
            .ok_or("a series names its dtype")?;
        let values = spec
            .get("values")
            .and_then(Value::as_array)
            .ok_or("a series has values")?;
        let decoded = values
            .iter()
            .map(PyVal::decode)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self::build(None, Dtype::parse(dtype_name), decoded))
    }

    /// The cells pandas holds for `values` under `dtype`, read back.
    pub fn build(name: Option<String>, dtype: Dtype, values: Vec<PyVal>) -> Series {
        let cells = values
            .into_iter()
            .map(|value| coerce(&dtype, value))
            .collect();
        Series { name, dtype, cells }
    }

    /// `series.isna()`, cell by cell.
    pub fn isna(&self) -> Vec<bool> {
        self.cells.iter().map(is_na).collect()
    }
}

/// `pd.isna(value)` of a scalar: `None`, a NaN float, `pd.NA`, `NaT` and a
/// NaN `Decimal`.
pub fn is_na(value: &PyVal) -> bool {
    match value {
        PyVal::Decimal(text) => Dec::parse(text).is_some_and(|d| d.is_nan()),
        PyVal::Numpy(_, inner) => is_na(inner),
        other => other.is_missing(),
    }
}

/// What `pd.Series([value], dtype=dtype).tolist()[0]` gives back.
fn coerce(dtype: &Dtype, value: PyVal) -> PyVal {
    let missing = is_na(&value) && !matches!(value, PyVal::Decimal(_));
    match dtype {
        Dtype::Float(bits) => match value {
            PyVal::Int(n) => float_of(n as f64, *bits),
            PyVal::Float(x) => float_of(x, *bits),
            PyVal::Bool(b) => float_of(if b { 1.0 } else { 0.0 }, *bits),
            _ if missing => PyVal::Float(f64::NAN),
            other => other,
        },
        Dtype::Nullable(name) if name.starts_with("Float") => match value {
            _ if missing => PyVal::Na,
            PyVal::Int(n) => PyVal::Float(n as f64),
            other => other,
        },
        Dtype::Nullable(_) | Dtype::String(_) => {
            if missing {
                PyVal::Na
            } else {
                value
            }
        }
        Dtype::Str => match value {
            _ if missing => PyVal::Float(f64::NAN),
            PyVal::Str(s) => PyVal::Str(s),
            other => PyVal::Str(other.str_()),
        },
        Dtype::Category => {
            if missing {
                PyVal::Float(f64::NAN)
            } else {
                value
            }
        }
        Dtype::Arrow(name) => {
            if missing {
                return PyVal::Na;
            }
            if let Some((_, scale)) = dtype.arrow_decimal() {
                return match value {
                    PyVal::Decimal(text) => PyVal::Decimal(rescale(&text, scale)),
                    PyVal::Int(n) => PyVal::Decimal(rescale(&n.to_string(), scale)),
                    other => other,
                };
            }
            match (name.as_str(), value) {
                ("double" | "float", PyVal::Int(n)) => PyVal::Float(n as f64),
                (_, other) => other,
            }
        }
        Dtype::Datetime(_, tz) => match value {
            _ if missing => PyVal::NaT,
            PyVal::Str(text) => match PyVal::decode(&serde_json::json!({ "$timestamp": text })) {
                Ok(read) => localize(read, tz.as_deref()),
                Err(_) => PyVal::Str(text),
            },
            PyVal::DateTime(wall, None) => localize(PyVal::Timestamp(wall), tz.as_deref()),
            PyVal::DateTime(wall, Some(offset)) => {
                localize(PyVal::TimestampTz(wall, offset), tz.as_deref())
            }
            PyVal::Date(y, m, d) => localize(PyVal::Timestamp(Wall::date(y, m, d)), tz.as_deref()),
            other => localize(other, tz.as_deref()),
        },
        Dtype::Timedelta(unit) => match value {
            _ if missing => PyVal::NaT,
            PyVal::Timedelta(nanos) => PyVal::Timedelta(truncate_nanos(nanos, unit)),
            other => other,
        },
        _ => value,
    }
}

fn float_of(x: f64, bits: u32) -> PyVal {
    if bits == 32 {
        PyVal::Float(f64::from(x as f32))
    } else {
        PyVal::Float(x)
    }
}

/// A timestamp under a zone of a fixed offset, such as `UTC-08:00`.
fn localize(value: PyVal, tz: Option<&str>) -> PyVal {
    let Some(tz) = tz else {
        return value;
    };
    let Some(offset) = fixed_offset(tz) else {
        return value;
    };
    match value {
        PyVal::TimestampTz(wall, from) => {
            let instant = wall.epoch_nanos() - i128::from(from) * 1_000_000_000;
            let local = Wall::from_epoch_nanos(instant + i128::from(offset) * 1_000_000_000);
            PyVal::TimestampTz(local, offset)
        }
        PyVal::Timestamp(wall) => PyVal::TimestampTz(wall, offset),
        other => other,
    }
}

/// The offset in seconds of a zone written `UTC`, `UTC+01:00` or `+01:00`.
pub fn fixed_offset(tz: &str) -> Option<i64> {
    let body = tz.strip_prefix("UTC").unwrap_or(tz);
    if body.is_empty() {
        return Some(0);
    }
    let sign = match body.as_bytes()[0] {
        b'+' => 1,
        b'-' => -1,
        _ => return None,
    };
    let (h, m) = body[1..].split_once(':').unwrap_or((&body[1..], "0"));
    Some(sign * (h.parse::<i64>().ok()? * 3600 + m.parse::<i64>().ok()? * 60))
}

fn truncate_nanos(nanos: i64, unit: &str) -> i64 {
    let per: i64 = match unit {
        "s" => 1_000_000_000,
        "ms" => 1_000_000,
        "us" => 1_000,
        _ => 1,
    };
    nanos - nanos.rem_euclid(per)
}

/// `text` written with exactly `scale` places, as an Arrow decimal holds it.
fn rescale(text: &str, scale: i32) -> String {
    let Some(value) = Dec::parse(text) else {
        return text.to_string();
    };
    if !value.is_finite() {
        return text.to_string();
    }
    let target = -i64::from(scale);
    if value.exponent == target {
        return value.to_sci();
    }
    if value.exponent > target {
        let pad = (value.exponent - target) as usize;
        let digits = if value.digits == "0" {
            "0".to_string()
        } else {
            format!("{}{}", value.digits, "0".repeat(pad))
        };
        let scaled = Dec {
            digits,
            exponent: target,
            ..value
        };
        return scaled.to_sci();
    }
    value.to_sci()
}

/// `pd.api.types.infer_dtype(values, skipna=True)` of an object column.
pub fn infer_dtype(cells: &[PyVal]) -> &'static str {
    let mut first = None;
    let mut seen_nat = false;
    for cell in cells {
        match cell {
            PyVal::None | PyVal::Na => {}
            PyVal::Float(x) if x.is_nan() => {}
            PyVal::Numpy(_, inner) if matches!(inner.as_ref(), PyVal::Float(x) if x.is_nan()) => {}
            PyVal::NaT => seen_nat = true,
            other => {
                first = Some(other);
                break;
            }
        }
    }
    let Some(first) = first else {
        return if seen_nat { "datetime" } else { "empty" };
    };
    let all = |check: &dyn Fn(&PyVal) -> bool, nulls: &dyn Fn(&PyVal) -> bool| {
        cells.iter().all(|c| check(c) || nulls(c))
    };
    let null = |c: &PyVal| {
        matches!(c, PyVal::None | PyVal::Na) || matches!(c, PyVal::Float(x) if x.is_nan())
    };
    let null_dt = |c: &PyVal| null(c) || matches!(c, PyVal::NaT);
    let null_dec =
        |c: &PyVal| null(c) || matches!(c, PyVal::Decimal(t) if is_na(&PyVal::Decimal(t.clone())));
    let found = match first {
        PyVal::Numpy(name, _) if name.starts_with("datetime64") => all(
            &|c| matches!(c, PyVal::Numpy(n, _) if n.starts_with("datetime64")),
            &null_dt,
        )
        .then_some("datetime64"),
        PyVal::Timedelta(_) => {
            all(&|c| matches!(c, PyVal::Timedelta(_)), &null_dt).then_some("timedelta")
        }
        v if is_integer_object(v) => {
            if all(&is_integer_object, &null) {
                Some("integer")
            } else if all(&|c| is_integer_object(c) || is_float_object(c), &null) {
                Some("mixed-integer-float")
            } else {
                Some("mixed-integer")
            }
        }
        PyVal::Timestamp(_) | PyVal::TimestampTz(..) | PyVal::DateTime(..) => {
            if all(&is_datetime, &null_dt) {
                Some("datetime")
            } else if all(&is_date, &null_dt) {
                Some("date")
            } else {
                None
            }
        }
        PyVal::Date(..) => all(&is_date, &null_dt).then_some("date"),
        PyVal::Time { .. } => all(&|c| matches!(c, PyVal::Time { .. }), &null).then_some("time"),
        PyVal::Decimal(_) => {
            all(&|c| matches!(c, PyVal::Decimal(_)), &null_dec).then_some("decimal")
        }
        v if is_float_object(v) => {
            if all(&is_float_object, &null) {
                Some("floating")
            } else if all(&|c| is_integer_object(c) || is_float_object(c), &null) {
                Some("mixed-integer-float")
            } else {
                None
            }
        }
        v if is_bool_object(v) => all(&is_bool_object, &null).then_some("boolean"),
        PyVal::Str(_) => all(&|c| matches!(c, PyVal::Str(_)), &null).then_some("string"),
        PyVal::Bytes(_) => all(&|c| matches!(c, PyVal::Bytes(_)), &null).then_some("bytes"),
        _ => None,
    };
    if let Some(found) = found {
        return found;
    }
    if cells.iter().any(is_integer_object) {
        "mixed-integer"
    } else {
        "mixed"
    }
}

/// `util.is_integer_object`: an `int` or numpy integer, not a `bool`.
pub fn is_integer_object(value: &PyVal) -> bool {
    match value {
        PyVal::Int(_) | PyVal::BigInt(_) => true,
        PyVal::Numpy(name, _) => name.starts_with("int") || name.starts_with("uint"),
        _ => false,
    }
}

/// `util.is_float_object`: a `float` or numpy float.
pub fn is_float_object(value: &PyVal) -> bool {
    match value {
        PyVal::Float(_) => true,
        PyVal::Numpy(name, _) => name.starts_with("float") || name == "longdouble",
        _ => false,
    }
}

/// `util.is_bool_object`: a `bool` or numpy bool.
pub fn is_bool_object(value: &PyVal) -> bool {
    match value {
        PyVal::Bool(_) => true,
        PyVal::Numpy(name, _) => name == "bool",
        _ => false,
    }
}

fn is_datetime(value: &PyVal) -> bool {
    matches!(
        value,
        PyVal::Timestamp(_) | PyVal::TimestampTz(..) | PyVal::DateTime(..)
    )
}

fn is_date(value: &PyVal) -> bool {
    is_datetime(value) || matches!(value, PyVal::Date(..))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cells(json: &str) -> Vec<PyVal> {
        let value: Value = serde_json::from_str(json).unwrap();
        value
            .as_array()
            .unwrap()
            .iter()
            .map(|v| PyVal::decode(v).unwrap())
            .collect()
    }

    #[test]
    fn an_interval_is_no_integer_dtype() {
        assert_eq!(Dtype::parse("int64"), Dtype::Int("int64".into()));
        assert_eq!(Dtype::parse("uint8"), Dtype::Int("uint8".into()));
        assert_eq!(
            Dtype::parse("interval[int64, right]"),
            Dtype::Other("interval[int64, right]".into())
        );
        assert_eq!(Dtype::parse("int"), Dtype::Other("int".into()));
    }

    #[test]
    fn infers_as_pandas_does() {
        for (json, want) in [
            ("[null, null]", "empty"),
            ("[{\"$nat\": null}]", "datetime"),
            ("[1, 2, null]", "integer"),
            ("[1, 2.5]", "mixed-integer-float"),
            ("[1.5]", "floating"),
            ("[true, false]", "boolean"),
            ("[true, 1]", "mixed-integer"),
            ("[\"a\", 1]", "mixed-integer"),
            (
                "[{\"$decimal\": \"1\"}, {\"$decimal\": \"NaN\"}]",
                "decimal",
            ),
            ("[{\"$decimal\": \"1\"}, 1]", "mixed-integer"),
            ("[{\"$decimal\": \"1.5\"}, 2.5]", "mixed"),
            ("[[1, 2]]", "mixed"),
            ("[\"a\", null]", "string"),
        ] {
            assert_eq!(infer_dtype(&cells(json)), want, "{json}");
        }
    }

    #[test]
    fn dtypes_name_themselves() {
        for name in [
            "int64",
            "uint8",
            "float32",
            "datetime64[us]",
            "datetime64[us, UTC-08:00]",
            "timedelta64[s]",
            "decimal128(38, 0)[pyarrow]",
            "string[pyarrow]",
            "UInt64",
            "category",
            "str",
            "object",
        ] {
            assert_eq!(Dtype::parse(name).name(), name);
        }
        assert_eq!(
            Dtype::parse("decimal128(10, 2)[pyarrow]").arrow_decimal(),
            Some((10, 2))
        );
    }
}
