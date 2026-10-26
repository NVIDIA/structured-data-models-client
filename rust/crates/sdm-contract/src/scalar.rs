// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Exact scalar encodings.
//!
//! Every scalar in a control message is a tagged object whose value is a
//! JSON string, never a JSON number, so no language's JSON parser can round
//! it through a double: integers per width, decimals with their precision
//! and scale, floats as their shortest round-trip text (with `NaN`, `inf` and
//! `-inf`), and temporal values as counts in their unit. Rows never travel
//! this way; anything row-shaped is Arrow.

use std::borrow::Cow;
use std::fmt;

use arrow_buffer::i256;
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64;
use schemars::{JsonSchema, Schema, SchemaGenerator};
use serde::{Deserialize, Serialize};

use crate::types::{TimestampUnit, Unit};

/// The largest precision of a `decimal128`.
pub const DECIMAL128_MAX_PRECISION: u8 = 38;
/// The largest precision of a `decimal256`.
pub const DECIMAL256_MAX_PRECISION: u8 = 76;

/// One scalar value, as the contract encodes it.
///
/// Floats compare by their bits, except that every `NaN` is equal, so `-0.0`
/// differs from `0.0`: equality here means "the same encoded value", and the
/// encoding keeps no `NaN` payload.
#[derive(Debug, Clone)]
pub enum Scalar {
    Null,
    Bool(bool),
    Int8(i8),
    Int16(i16),
    Int32(i32),
    Int64(i64),
    UInt8(u8),
    UInt16(u16),
    UInt32(u32),
    UInt64(u64),
    Float32(f32),
    Float64(f64),
    Decimal128 {
        precision: u8,
        scale: i8,
        value: i128,
    },
    Decimal256 {
        precision: u8,
        scale: i8,
        value: i256,
    },
    Utf8(String),
    Binary(Vec<u8>),
    /// Days since the epoch.
    Date32(i32),
    /// A count of `unit` since the epoch, with the zone it is shown in.
    Timestamp {
        unit: TimestampUnit,
        zone: Option<String>,
        value: i64,
    },
    Duration {
        unit: Unit,
        value: i64,
    },
    /// A count of `unit` since midnight.
    Time64 {
        unit: Unit,
        value: i64,
    },
}

impl PartialEq for Scalar {
    fn eq(&self, other: &Self) -> bool {
        use Scalar as S;
        match (self, other) {
            (S::Float32(a), S::Float32(b)) => {
                (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits()
            }
            (S::Float64(a), S::Float64(b)) => {
                (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits()
            }
            _ => ScalarWire::from(self.clone()) == ScalarWire::from(other.clone()),
        }
    }
}

impl Eq for Scalar {}

/// Why an encoded scalar was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScalarError(pub String);

impl fmt::Display for ScalarError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// The encoded form: what crosses the boundary, field for field.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "t", deny_unknown_fields)]
#[schemars(rename = "Scalar")]
enum ScalarWire {
    #[serde(rename = "null")]
    Null {},
    #[serde(rename = "bool")]
    Bool { v: bool },
    #[serde(rename = "i8")]
    I8 { v: Digits },
    #[serde(rename = "i16")]
    I16 { v: Digits },
    #[serde(rename = "i32")]
    I32 { v: Digits },
    #[serde(rename = "i64")]
    I64 { v: Digits },
    #[serde(rename = "u8")]
    U8 { v: Digits },
    #[serde(rename = "u16")]
    U16 { v: Digits },
    #[serde(rename = "u32")]
    U32 { v: Digits },
    #[serde(rename = "u64")]
    U64 { v: Digits },
    #[serde(rename = "f32")]
    F32 { v: FloatText },
    #[serde(rename = "f64")]
    F64 { v: FloatText },
    /// The unscaled integer: the value is `v * 10^-s`.
    #[serde(rename = "d128")]
    D128 { p: u8, s: i8, v: Digits },
    #[serde(rename = "d256")]
    D256 { p: u8, s: i8, v: Digits },
    #[serde(rename = "utf8")]
    Utf8 { v: String },
    /// Standard base64 with padding.
    #[serde(rename = "binary")]
    Binary { v: String },
    #[serde(rename = "date32")]
    Date32 { v: Digits },
    #[serde(rename = "ts")]
    Ts {
        unit: TimestampUnit,
        zone: Option<String>,
        v: Digits,
    },
    #[serde(rename = "duration")]
    Duration { unit: Unit, v: Digits },
    #[serde(rename = "time64")]
    Time64 { unit: Unit, v: Digits },
}

/// An integer as decimal digits, with a minus sign when negative and no
/// leading zeros, plus sign or whitespace.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
struct Digits(String);

impl JsonSchema for Digits {
    fn schema_name() -> Cow<'static, str> {
        "Digits".into()
    }

    fn json_schema(_: &mut SchemaGenerator) -> Schema {
        schemars::json_schema!({
            "type": "string",
            "pattern": "^(0|-?[1-9][0-9]*)$",
        })
    }
}

/// A float as text: any decimal or exponent form that reads back to the
/// same value, or `NaN`, `inf` or `-inf`. Encoders write the shortest
/// round-trip form.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
struct FloatText(String);

impl JsonSchema for FloatText {
    fn schema_name() -> Cow<'static, str> {
        "FloatText".into()
    }

    fn json_schema(_: &mut SchemaGenerator) -> Schema {
        schemars::json_schema!({
            "type": "string",
            "pattern": "^(NaN|inf|-inf|-?[0-9]+(\\.[0-9]+)?([eE][+-]?[0-9]+)?)$",
        })
    }
}

fn digits<T: fmt::Display>(value: T) -> Digits {
    Digits(value.to_string())
}

/// Shortest round-trip text; Rust's `Debug` for floats is exactly that, with
/// `NaN`, `inf` and `-inf` for the non-finite values.
fn float_text<T: fmt::Debug>(value: T) -> FloatText {
    FloatText(format!("{value:?}"))
}

impl From<Scalar> for ScalarWire {
    fn from(value: Scalar) -> Self {
        use Scalar as S;
        use ScalarWire as W;
        match value {
            S::Null => W::Null {},
            S::Bool(v) => W::Bool { v },
            S::Int8(v) => W::I8 { v: digits(v) },
            S::Int16(v) => W::I16 { v: digits(v) },
            S::Int32(v) => W::I32 { v: digits(v) },
            S::Int64(v) => W::I64 { v: digits(v) },
            S::UInt8(v) => W::U8 { v: digits(v) },
            S::UInt16(v) => W::U16 { v: digits(v) },
            S::UInt32(v) => W::U32 { v: digits(v) },
            S::UInt64(v) => W::U64 { v: digits(v) },
            S::Float32(v) => W::F32 { v: float_text(v) },
            S::Float64(v) => W::F64 { v: float_text(v) },
            S::Decimal128 {
                precision,
                scale,
                value,
            } => W::D128 {
                p: precision,
                s: scale,
                v: digits(value),
            },
            S::Decimal256 {
                precision,
                scale,
                value,
            } => W::D256 {
                p: precision,
                s: scale,
                v: digits(value),
            },
            S::Utf8(v) => W::Utf8 { v },
            S::Binary(v) => W::Binary {
                v: BASE64.encode(v),
            },
            S::Date32(v) => W::Date32 { v: digits(v) },
            S::Timestamp { unit, zone, value } => W::Ts {
                unit,
                zone,
                v: digits(value),
            },
            S::Duration { unit, value } => W::Duration {
                unit,
                v: digits(value),
            },
            S::Time64 { unit, value } => W::Time64 {
                unit,
                v: digits(value),
            },
        }
    }
}

/// Checks the canonical form; `str::parse` alone would also take `+1` and
/// `007`.
fn canonical<'a>(tag: &str, text: &'a Digits) -> Result<&'a str, ScalarError> {
    let s = text.0.as_str();
    let body = s.strip_prefix('-').unwrap_or(s);
    let canonical = !body.is_empty()
        && body.bytes().all(|b| b.is_ascii_digit())
        && (body == "0" || !body.starts_with('0'))
        && s != "-0";
    if canonical {
        Ok(s)
    } else {
        Err(ScalarError(format!(
            "{tag}: {s:?} is not an integer in canonical decimal form"
        )))
    }
}

fn int<T: std::str::FromStr>(tag: &str, text: &Digits) -> Result<T, ScalarError> {
    canonical(tag, text)?
        .parse()
        .map_err(|_| ScalarError(format!("{tag}: {} does not fit", text.0)))
}

/// The float widths, for the overflow check.
trait FloatValue: std::str::FromStr {
    fn is_infinite(&self) -> bool;
}

impl FloatValue for f32 {
    fn is_infinite(&self) -> bool {
        f32::is_infinite(*self)
    }
}

impl FloatValue for f64 {
    fn is_infinite(&self) -> bool {
        f64::is_infinite(*self)
    }
}

fn float<T: FloatValue>(tag: &str, text: &FloatText) -> Result<T, ScalarError> {
    let s = text.0.as_str();
    let finite = {
        let body = s.strip_prefix('-').unwrap_or(s);
        let (mantissa, exponent) = match body.find(['e', 'E']) {
            Some(i) => (&body[..i], Some(&body[i + 1..])),
            None => (body, None),
        };
        let (whole, fraction) = match mantissa.split_once('.') {
            Some((w, f)) => (w, Some(f)),
            None => (mantissa, None),
        };
        let all_digits = |t: &str| !t.is_empty() && t.bytes().all(|b| b.is_ascii_digit());
        all_digits(whole)
            && fraction.is_none_or(all_digits)
            && exponent.is_none_or(|e| all_digits(e.strip_prefix(['+', '-']).unwrap_or(e)))
    };
    if !(finite || matches!(s, "NaN" | "inf" | "-inf")) {
        return Err(ScalarError(format!("{tag}: {s:?} is not a float's text")));
    }
    let value: T = s
        .parse()
        .map_err(|_| ScalarError(format!("{tag}: {s:?} is not a float's text")))?;
    // Finite text that rounds to infinity at this width is no float's text:
    // the encoder writes `inf` for an infinite value.
    if finite && value.is_infinite() {
        return Err(ScalarError(format!("{tag}: {s:?} overflows the type")));
    }
    Ok(value)
}

/// The number of decimal digits of `|value|`, for the precision check.
fn digit_count(text: &str) -> usize {
    text.trim_start_matches('-').len()
}

/// The precision and scale rule a decimal type and a decimal scalar share:
/// precision from 1 to `max`, and a scale no larger than the precision. A
/// negative scale is Arrow's, and is kept.
pub(crate) fn decimal_type_rule(tag: &str, p: u8, s: i8, max: u8) -> Result<(), String> {
    if p == 0 || p > max {
        return Err(format!("{tag}: precision {p} is outside 1 to {max}"));
    }
    if i16::from(s) > i16::from(p) {
        return Err(format!("{tag}: scale {s} exceeds precision {p}"));
    }
    Ok(())
}

/// The zone rule a timestamp type and a timestamp scalar share: a zone, if
/// any, is not empty.
pub(crate) fn zone_rule(tag: &str, zone: Option<&str>) -> Result<(), String> {
    match zone {
        Some("") => Err(format!("{tag}: the zone is empty")),
        _ => Ok(()),
    }
}

fn decimal_shape(tag: &str, p: u8, s: i8, max: u8, v: &Digits) -> Result<(), ScalarError> {
    decimal_type_rule(tag, p, s, max).map_err(ScalarError)?;
    if digit_count(&v.0) > usize::from(p) {
        return Err(ScalarError(format!(
            "{tag}: {} has more than {p} digits",
            v.0
        )));
    }
    Ok(())
}

impl TryFrom<ScalarWire> for Scalar {
    type Error = ScalarError;

    fn try_from(wire: ScalarWire) -> Result<Self, ScalarError> {
        use Scalar as S;
        use ScalarWire as W;
        Ok(match wire {
            W::Null {} => S::Null,
            W::Bool { v } => S::Bool(v),
            W::I8 { v } => S::Int8(int("i8", &v)?),
            W::I16 { v } => S::Int16(int("i16", &v)?),
            W::I32 { v } => S::Int32(int("i32", &v)?),
            W::I64 { v } => S::Int64(int("i64", &v)?),
            W::U8 { v } => S::UInt8(int("u8", &v)?),
            W::U16 { v } => S::UInt16(int("u16", &v)?),
            W::U32 { v } => S::UInt32(int("u32", &v)?),
            W::U64 { v } => S::UInt64(int("u64", &v)?),
            W::F32 { v } => S::Float32(float("f32", &v)?),
            W::F64 { v } => S::Float64(float("f64", &v)?),
            W::D128 { p, s, v } => {
                let value = int("d128", &v)?;
                decimal_shape("d128", p, s, DECIMAL128_MAX_PRECISION, &v)?;
                S::Decimal128 {
                    precision: p,
                    scale: s,
                    value,
                }
            }
            W::D256 { p, s, v } => {
                canonical("d256", &v)?;
                decimal_shape("d256", p, s, DECIMAL256_MAX_PRECISION, &v)?;
                let value = i256::from_string(&v.0)
                    .ok_or_else(|| ScalarError(format!("d256: {} does not fit", v.0)))?;
                S::Decimal256 {
                    precision: p,
                    scale: s,
                    value,
                }
            }
            W::Utf8 { v } => S::Utf8(v),
            W::Binary { v } => S::Binary(
                BASE64
                    .decode(v.as_bytes())
                    .map_err(|e| ScalarError(format!("binary: not standard base64: {e}")))?,
            ),
            W::Date32 { v } => S::Date32(int("date32", &v)?),
            W::Ts { unit, zone, v } => {
                zone_rule("ts", zone.as_deref()).map_err(ScalarError)?;
                S::Timestamp {
                    unit,
                    zone,
                    value: int("ts", &v)?,
                }
            }
            W::Duration { unit, v } => S::Duration {
                unit,
                value: int("duration", &v)?,
            },
            W::Time64 { unit, v } => S::Time64 {
                unit,
                value: int("time64", &v)?,
            },
        })
    }
}

impl Scalar {
    /// Whether the decoder would read this value back: a decimal's precision
    /// and scale within its type's limits and its value within the
    /// precision, and a timestamp's zone not empty. Encoding a value that
    /// fails this fails, so the encoder never writes what the decoder
    /// refuses.
    pub fn validate(&self) -> Result<(), ScalarError> {
        Scalar::try_from(ScalarWire::from(self.clone())).map(|_| ())
    }
}

impl Serialize for Scalar {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let wire = ScalarWire::from(self.clone());
        Scalar::try_from(wire.clone()).map_err(serde::ser::Error::custom)?;
        wire.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Scalar {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let wire = ScalarWire::deserialize(deserializer)?;
        Scalar::try_from(wire).map_err(serde::de::Error::custom)
    }
}

impl JsonSchema for Scalar {
    fn schema_name() -> Cow<'static, str> {
        "Scalar".into()
    }

    fn json_schema(generator: &mut SchemaGenerator) -> Schema {
        ScalarWire::json_schema(generator)
    }
}

/// The `f64` Arrow's `Decimal128::ToDouble` gives the decimal of unscaled
/// value `value` at `scale`: the integer's two halves at float64, scaled by
/// the power of ten, split into whole and fractional parts where the value
/// is past what a float64 holds exactly; pandas' `astype('float32')` of a
/// decimal takes this, then narrows it, and the reference's `_sanitize`
/// casts a decimal of scale > 0 to float64 by it before it infers anything.
pub fn decimal_to_f64(value: i128, scale: i8) -> f64 {
    fn no_split(magnitude: u128, scale: i32) -> f64 {
        let high = (magnitude >> 64) as u64;
        let low = magnitude as u64;
        let mut x = (high as f64) * 18_446_744_073_709_551_616.0;
        x += low as f64;
        // Arrow's table of powers of ten holds each as its literal, the
        // closest double, which parsing the literal gives.
        let power: f64 = format!("1e{}", -scale).parse().unwrap_or(f64::NAN);
        x * power
    }
    let scale = i32::from(scale);
    let magnitude = value.unsigned_abs();
    let positive = if scale <= 0 || magnitude <= 1_u128 << 53 {
        no_split(magnitude, scale)
    } else {
        let power = 10_u128.pow(scale as u32);
        no_split(magnitude / power, 0) + no_split(magnitude % power, scale)
    };
    if value < 0 { -positive } else { positive }
}
