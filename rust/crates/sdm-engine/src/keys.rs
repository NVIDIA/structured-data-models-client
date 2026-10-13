// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Key values, compared exactly: an integer of any width as its value, text
//! as its text, a decimal as its value. Nothing passes through a float, so
//! two keys are the same only when their values are (decision H22). The
//! types a sampled key may have are the integers, text and, from T1.32a,
//! the decimals; any other is refused by name (`key_type`).
//!
//! From T1.32a this module also holds the entities' cast to the key's type
//! ([`cast`]): a host hands the entities to predict for as one Arrow column
//! of whatever type its own values have, and the core casts them under one
//! rule table, so every host gives the same answers and the same errors
//! (design, "Host-held semantics").

use arrow_array::cast::AsArray;
use std::sync::Arc;

use arrow_array::builder::{
    Decimal128Builder, Float32Builder, Float64Builder, Int8Builder, Int16Builder, Int32Builder,
    Int64Builder, StringBuilder, UInt8Builder, UInt16Builder, UInt32Builder, UInt64Builder,
};
use arrow_array::types::{
    Decimal128Type, Decimal256Type, Float32Type, Float64Type, Int8Type, Int16Type, Int32Type,
    Int64Type, UInt8Type, UInt16Type, UInt32Type, UInt64Type,
};
use arrow_array::{Array, ArrayRef};
use arrow_schema::DataType;
use sdm_contract::types::ColumnType;
use sdm_contract::{CoreError, Detail, ErrorCode, ErrorKind, Scalar};

use crate::not_yet;

/// One key value. A decimal is held at the smallest scale that holds it
/// exactly, and a whole one as an integer, so the key's value, not the type
/// it was read as, decides which keys are the same.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Key {
    Int(i128),
    Text(String),
    /// A value with a fraction: `value` ten-to-the-`scale`ths, `scale` from
    /// 1 to 38, `value` no multiple of ten.
    Decimal {
        value: i128,
        scale: i8,
    },
    /// A float that is not a whole number within 128 bits, by its bits as a
    /// double (every NaN as one).
    Float(u64),
}

impl Key {
    /// The key a decimal of `value` ten-to-the-`scale`ths is.
    fn decimal(mut value: i128, mut scale: i8) -> Key {
        while scale > 0 && value % 10 == 0 {
            value /= 10;
            scale -= 1;
        }
        if scale > 0 {
            Key::Decimal { value, scale }
        } else {
            Key::Int(value)
        }
    }

    /// The key a float of `value` is: a whole one as the integer it is.
    fn float(value: f64) -> Key {
        if value.fract() == 0.0 && value.abs() < 2f64.powi(127) {
            Key::Int(value as i128)
        } else if value.is_nan() {
            Key::Float(f64::NAN.to_bits())
        } else {
            Key::Float(value.to_bits())
        }
    }
}

/// Whether a key of `data_type` is sampled: the integers, text and, from
/// T1.32a, the floats and a `decimal128` of no negative scale.
pub fn supported(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Utf8
            | DataType::Float32
            | DataType::Float64
    ) || matches!(data_type, DataType::Decimal128(_, scale) if *scale >= 0)
}

/// The keys `array` holds, `None` for a null.
pub fn read(array: &ArrayRef) -> Result<Vec<Option<Key>>, CoreError> {
    macro_rules! ints {
        ($t:ty) => {{
            let values = array.as_primitive::<$t>();
            (0..values.len())
                .map(|i| (!values.is_null(i)).then(|| Key::Int(i128::from(values.value(i)))))
                .collect()
        }};
    }
    Ok(match array.data_type() {
        DataType::Int8 => ints!(Int8Type),
        DataType::Int16 => ints!(Int16Type),
        DataType::Int32 => ints!(Int32Type),
        DataType::Int64 => ints!(Int64Type),
        DataType::UInt8 => ints!(UInt8Type),
        DataType::UInt16 => ints!(UInt16Type),
        DataType::UInt32 => ints!(UInt32Type),
        DataType::UInt64 => ints!(UInt64Type),
        DataType::Utf8 => {
            let values = array.as_string::<i32>();
            (0..values.len())
                .map(|i| (!values.is_null(i)).then(|| Key::Text(values.value(i).to_owned())))
                .collect()
        }
        DataType::Decimal128(_, scale) if *scale >= 0 => {
            let values = array.as_primitive::<Decimal128Type>();
            (0..values.len())
                .map(|i| (!values.is_null(i)).then(|| Key::decimal(values.value(i), *scale)))
                .collect()
        }
        DataType::Float32 => {
            let values = array.as_primitive::<Float32Type>();
            (0..values.len())
                .map(|i| (!values.is_null(i)).then(|| Key::float(f64::from(values.value(i)))))
                .collect()
        }
        DataType::Float64 => {
            let values = array.as_primitive::<Float64Type>();
            (0..values.len())
                .map(|i| (!values.is_null(i)).then(|| Key::float(values.value(i))))
                .collect()
        }
        _ => return Err(not_yet("key_type")),
    })
}

// --- The entities' cast to the key's type (T1.32a) ---------------------------

/// Why an entity does not cast to the key's type: the `reason` field of the
/// refusal, one of the rule table's outcomes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Uncast {
    /// The value is outside what the key's type holds: past an integer's
    /// range, or more digits than a decimal's precision.
    OutOfRange,
    /// The value has digits the key's type cannot hold: a fraction for an
    /// integer key, or more decimal places than a decimal key's scale.
    Inexact,
    /// A float that is infinite or NaN.
    NotFinite,
    /// Text that does not read as a number of the key's kind.
    NotANumber,
    /// No rule casts a value of this type to the key's.
    Type,
}

impl Uncast {
    /// The reason as the refusal names it.
    pub fn name(self) -> &'static str {
        match self {
            Uncast::OutOfRange => "out_of_range",
            Uncast::Inexact => "inexact",
            Uncast::NotFinite => "not_finite",
            Uncast::NotANumber => "not_a_number",
            Uncast::Type => "type",
        }
    }
}

/// The type an entity is cast to: the key's.
#[derive(Debug, Clone, Copy)]
enum Target {
    /// An integer type holding `min..=max`.
    Int {
        min: i128,
        max: i128,
    },
    Text,
    /// A decimal of `precision` digits, `scale` of them after the point.
    Decimal {
        precision: u8,
        scale: i8,
    },
    /// A float, of 32 bits (`single`) or 64.
    Float {
        single: bool,
    },
}

impl Target {
    fn of(key: &DataType) -> Option<Target> {
        let int = |min: i128, max: i128| Some(Target::Int { min, max });
        match key {
            DataType::Int8 => int(i8::MIN.into(), i8::MAX.into()),
            DataType::Int16 => int(i16::MIN.into(), i16::MAX.into()),
            DataType::Int32 => int(i32::MIN.into(), i32::MAX.into()),
            DataType::Int64 => int(i64::MIN.into(), i64::MAX.into()),
            DataType::UInt8 => int(0, u8::MAX.into()),
            DataType::UInt16 => int(0, u16::MAX.into()),
            DataType::UInt32 => int(0, u32::MAX.into()),
            DataType::UInt64 => int(0, u64::MAX.into()),
            DataType::Utf8 => Some(Target::Text),
            DataType::Float32 => Some(Target::Float { single: true }),
            DataType::Float64 => Some(Target::Float { single: false }),
            DataType::Decimal128(precision, scale) if *scale >= 0 && *scale as u8 <= *precision => {
                Some(Target::Decimal {
                    precision: *precision,
                    scale: *scale,
                })
            }
            _ => None,
        }
    }
}

/// One entity as the host handed it over.
enum Entity<'a> {
    Null,
    Int(i128),
    /// A float, with the shortest decimal text that reads back as it at
    /// its own width, with no exponent.
    Float(f64, String),
    Bool(bool),
    Text(&'a str),
    /// `value` ten-to-the-`scale`ths; `None` when it is past what 128 bits
    /// hold (a `decimal256`), kept as its digits for a text key.
    Decimal {
        value: Option<i128>,
        scale: i8,
        digits: String,
    },
    /// A value of a type no rule casts.
    Other,
}

/// The entity at `row` of `entities`.
fn entity(entities: &ArrayRef, row: usize) -> Entity<'_> {
    if entities.is_null(row) {
        return Entity::Null;
    }
    macro_rules! int {
        ($t:ty) => {
            Entity::Int(i128::from(entities.as_primitive::<$t>().value(row)))
        };
    }
    match entities.data_type() {
        DataType::Null => Entity::Null,
        DataType::Int8 => int!(Int8Type),
        DataType::Int16 => int!(Int16Type),
        DataType::Int32 => int!(Int32Type),
        DataType::Int64 => int!(Int64Type),
        DataType::UInt8 => int!(UInt8Type),
        DataType::UInt16 => int!(UInt16Type),
        DataType::UInt32 => int!(UInt32Type),
        DataType::UInt64 => int!(UInt64Type),
        DataType::Float32 => {
            let value = entities.as_primitive::<Float32Type>().value(row);
            Entity::Float(f64::from(value), format!("{value}"))
        }
        DataType::Float64 => {
            let value = entities.as_primitive::<Float64Type>().value(row);
            Entity::Float(value, format!("{value}"))
        }
        DataType::Boolean => Entity::Bool(entities.as_boolean().value(row)),
        DataType::Utf8 => Entity::Text(entities.as_string::<i32>().value(row)),
        DataType::LargeUtf8 => Entity::Text(entities.as_string::<i64>().value(row)),
        DataType::Decimal128(_, scale) => {
            let value = entities.as_primitive::<Decimal128Type>().value(row);
            Entity::Decimal {
                value: Some(value),
                scale: *scale,
                digits: value.to_string(),
            }
        }
        DataType::Decimal256(_, scale) => {
            let value = entities.as_primitive::<Decimal256Type>().value(row);
            Entity::Decimal {
                value: value.to_i128(),
                scale: *scale,
                digits: value.to_string(),
            }
        }
        _ => Entity::Other,
    }
}

/// Ten to the `n`th, where 128 bits hold it (`n` up to 38).
fn ten_to(n: u32) -> Option<i128> {
    10i128.checked_pow(n)
}

/// `value` ten-to-the-`from`ths at the scale `to`, exactly: `Inexact` when
/// that drops a digit other than zero, `OutOfRange` when it overflows.
fn rescale(value: i128, from: i32, to: i32) -> Result<i128, Uncast> {
    if to >= from {
        let factor = u32::try_from(to - from)
            .ok()
            .and_then(ten_to)
            .ok_or(Uncast::OutOfRange)?;
        value.checked_mul(factor).ok_or(Uncast::OutOfRange)
    } else {
        let Some(factor) = u32::try_from(from - to).ok().and_then(ten_to) else {
            // More places than 128 bits count: only zero drops them all.
            return if value == 0 {
                Ok(0)
            } else {
                Err(Uncast::Inexact)
            };
        };
        if value % factor != 0 {
            return Err(Uncast::Inexact);
        }
        Ok(value / factor)
    }
}

/// The number of decimal digits of `value`, its sign not counted.
fn digit_count(value: i128) -> u32 {
    value.unsigned_abs().checked_ilog10().map_or(1, |n| n + 1)
}

/// Text written as a whole number, an optional `-` then ASCII digits, as
/// the integer it is: `NotANumber` for any other text, `OutOfRange` past
/// 128 bits.
fn read_integer(text: &str) -> Result<i128, Uncast> {
    let digits = text.strip_prefix('-').unwrap_or(text);
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return Err(Uncast::NotANumber);
    }
    text.parse::<i128>().map_err(|_| Uncast::OutOfRange)
}

/// Text written as a decimal number: an optional sign, ASCII digits with
/// an optional point and fraction (at least one digit in all), and an
/// optional exponent (`e` or `E`, an optional sign, digits), as its value
/// ten-to-the-`scale`ths, exactly.
fn read_decimal(text: &str, scale: i8) -> Result<i128, Uncast> {
    let (mantissa, exponent) = match text.find(['e', 'E']) {
        Some(at) => (&text[..at], Some(&text[at + 1..])),
        None => (text, None),
    };
    let (negative, unsigned) = match mantissa.as_bytes().first() {
        Some(b'-') => (true, &mantissa[1..]),
        Some(b'+') => (false, &mantissa[1..]),
        _ => (false, mantissa),
    };
    let (whole, fraction) = match unsigned.split_once('.') {
        Some((whole, fraction)) => (whole, fraction),
        None => (unsigned, ""),
    };
    let ascii = |part: &str| part.bytes().all(|b| b.is_ascii_digit());
    if whole.len() + fraction.len() == 0 || !ascii(whole) || !ascii(fraction) {
        return Err(Uncast::NotANumber);
    }
    let exponent: i32 = match exponent {
        None => 0,
        Some(written) => {
            let digits = written.strip_prefix(['+', '-']).unwrap_or(written);
            if digits.is_empty() || !ascii(digits) {
                return Err(Uncast::NotANumber);
            }
            // An exponent past any decimal's reach still reads: it decides
            // only whether the value is zero, out of range or inexact.
            written
                .parse::<i32>()
                .unwrap_or(if written.starts_with('-') {
                    i32::MIN / 2
                } else {
                    i32::MAX / 2
                })
        }
    };
    // The digits as one integer, leading zeros dropped, and the scale they
    // are written at.
    let all = format!("{whole}{fraction}");
    let significant = all.trim_start_matches('0');
    let written_scale = i32::try_from(fraction.len())
        .ok()
        .and_then(|n| n.checked_sub(exponent))
        .ok_or(Uncast::OutOfRange)?;
    if significant.is_empty() {
        return Ok(0);
    }
    // Trailing zeros carry no value: drop them from the digits before they
    // are read, so a long run of them is no overflow. What is left ends in
    // a digit other than zero, so a place past the key's scale is a digit
    // lost, whatever the value's size.
    let trimmed = significant.trim_end_matches('0');
    let dropped =
        i32::try_from(significant.len() - trimmed.len()).map_err(|_| Uncast::OutOfRange)?;
    let at = written_scale - dropped;
    if at > i32::from(scale) {
        return Err(Uncast::Inexact);
    }
    let value: i128 = trimmed.parse().map_err(|_| Uncast::OutOfRange)?;
    let value = rescale(value, at, i32::from(scale))?;
    Ok(if negative { -value } else { value })
}

/// One cast value of the key's type.
enum Cast {
    Null,
    Int(i128),
    Text(String),
    Float(f64),
}

/// The rule table: `entity` as a value of `target`, or why it is not one.
///
/// | Entity \ key | integer | text | decimal (p, s) |
/// |---|---|---|---|
/// | null | null | null | null |
/// | integer | itself, in the key's range | its digits | exact at scale s, in p digits |
/// | float | finite and whole, in range | finite: its shortest text, a whole one without a point | finite: its shortest text, read as a decimal |
/// | boolean | 1 or 0 | `true` or `false` | 1 or 0 |
/// | text | `-?[0-9]+`, in range | itself | `[+-]?` digits, `.`, digits, exponent, exact at scale s, in p digits |
/// | decimal | whole, in range | its digits at its own scale | exact at scale s, in p digits |
/// | any other type | refused (`type`) | refused | refused |
///
/// A float key (32 or 64 bits) takes any number, text read as the decimal
/// syntax above, and a boolean as 1 or 0, each as the nearest float of the
/// key's width, as a float is by its type; one past the width's range is
/// refused (`out_of_range`). Otherwise nothing is rounded or truncated: a
/// value the key's type does not hold exactly is refused, naming why
/// ([`Uncast`]).
fn cast_one(entity: Entity<'_>, target: Target) -> Result<Cast, Uncast> {
    match (entity, target) {
        (Entity::Null, _) => Ok(Cast::Null),
        (Entity::Other, _) => Err(Uncast::Type),
        (Entity::Float(value, _), _) if !value.is_finite() => Err(Uncast::NotFinite),

        (Entity::Int(value), Target::Int { min, max }) => in_range(value, min, max),
        (Entity::Float(value, _), Target::Int { min, max }) => {
            if value.fract() != 0.0 {
                return Err(Uncast::Inexact);
            }
            // A whole double past 2**127 is past every integer key too.
            if value.abs() >= 2f64.powi(127) {
                return Err(Uncast::OutOfRange);
            }
            in_range(value as i128, min, max)
        }
        (Entity::Bool(value), Target::Int { .. }) => Ok(Cast::Int(i128::from(value))),
        (Entity::Text(text), Target::Int { min, max }) => in_range(read_integer(text)?, min, max),
        (Entity::Decimal { value, scale, .. }, Target::Int { min, max }) => {
            let value = value.ok_or(Uncast::OutOfRange)?;
            in_range(rescale(value, i32::from(scale), 0)?, min, max)
        }

        (Entity::Int(value), Target::Text) => Ok(Cast::Text(value.to_string())),
        (Entity::Float(_, text), Target::Text) => Ok(Cast::Text(text)),
        (Entity::Bool(value), Target::Text) => Ok(Cast::Text(value.to_string())),
        (Entity::Text(text), Target::Text) => Ok(Cast::Text(text.to_owned())),
        (Entity::Decimal { scale, digits, .. }, Target::Text) => {
            Ok(Cast::Text(decimal_text(&digits, scale)))
        }

        (Entity::Int(value), Target::Decimal { precision, scale }) => {
            within(rescale(value, 0, i32::from(scale))?, precision)
        }
        (Entity::Float(_, text), Target::Decimal { precision, scale }) => {
            within(read_decimal(&text, scale)?, precision)
        }
        (Entity::Bool(value), Target::Decimal { precision, scale }) => {
            within(rescale(i128::from(value), 0, i32::from(scale))?, precision)
        }
        (Entity::Text(text), Target::Decimal { precision, scale }) => {
            within(read_decimal(text, scale)?, precision)
        }
        (
            Entity::Decimal {
                value, scale: from, ..
            },
            Target::Decimal { precision, scale },
        ) => {
            let value = value.ok_or(Uncast::OutOfRange)?;
            within(
                rescale(value, i32::from(from), i32::from(scale))?,
                precision,
            )
        }

        (Entity::Int(value), Target::Float { single }) => nearest(value as f64, single),
        (Entity::Float(value, _), Target::Float { single }) => nearest(value, single),
        (Entity::Bool(value), Target::Float { single }) => {
            nearest(f64::from(u8::from(value)), single)
        }
        (Entity::Text(text), Target::Float { single }) => nearest(read_float(text)?, single),
        (Entity::Decimal { scale, digits, .. }, Target::Float { single }) => {
            nearest(read_float(&decimal_text(&digits, scale))?, single)
        }
    }
}

/// `value` as the nearest float of the key's width: `OutOfRange` where
/// that is infinite.
fn nearest(value: f64, single: bool) -> Result<Cast, Uncast> {
    let value = if single {
        f64::from(value as f32)
    } else {
        value
    };
    if !value.is_finite() {
        return Err(Uncast::OutOfRange);
    }
    Ok(Cast::Float(value))
}

/// Text in the decimal syntax of [`read_decimal`] as the nearest double;
/// `NotANumber` for any other text (`inf`, `nan` and hexadecimal among
/// it).
fn read_float(text: &str) -> Result<f64, Uncast> {
    // The syntax is the decimal one, checked at a scale no text reaches,
    // so only its form is judged here.
    match read_decimal(text, i8::MAX) {
        Err(Uncast::NotANumber) => Err(Uncast::NotANumber),
        _ => text.parse::<f64>().map_err(|_| Uncast::NotANumber),
    }
}

fn in_range(value: i128, min: i128, max: i128) -> Result<Cast, Uncast> {
    if value < min || value > max {
        return Err(Uncast::OutOfRange);
    }
    Ok(Cast::Int(value))
}

fn within(value: i128, precision: u8) -> Result<Cast, Uncast> {
    if digit_count(value) > u32::from(precision) {
        return Err(Uncast::OutOfRange);
    }
    Ok(Cast::Int(value))
}

/// A decimal's text: `digits` (its unscaled value, signed) at `scale`.
fn decimal_text(digits: &str, scale: i8) -> String {
    let (sign, digits) = match digits.strip_prefix('-') {
        Some(rest) => ("-", rest),
        None => ("", digits),
    };
    if scale <= 0 {
        let zeros = "0".repeat(usize::from(scale.unsigned_abs()));
        return if digits == "0" {
            "0".to_owned()
        } else {
            format!("{sign}{digits}{zeros}")
        };
    }
    let scale = usize::from(scale.unsigned_abs());
    let padded = format!("{digits:0>width$}", width = scale + 1);
    let (whole, fraction) = padded.split_at(padded.len() - scale);
    format!("{sign}{whole}.{fraction}")
}

/// The refusal of the entity at `position`, which does not cast to `key`.
fn uncast(position: usize, key: &DataType, why: Uncast) -> CoreError {
    let key_type = ColumnType::from_arrow("key", key)
        .map(|t| t.cell_type())
        .unwrap_or("unknown");
    CoreError::new(
        ErrorKind::InvalidRequest,
        ErrorCode::InvalidRequest,
        "an entity to predict for does not cast to the type of the entity table's key",
    )
    .with_field("field", Detail::text("indices"))
    .with_field(
        "position",
        Detail::Scalar(Scalar::UInt64(u64::try_from(position).unwrap_or(u64::MAX))),
    )
    .with_field("reason", Detail::text(why.name()))
    .with_field("key_type", Detail::text(key_type))
}

/// The entities to predict for, `entities`, cast to the key's type `key`
/// under the rule table of [`cast_one`]: a column of `key`'s type, a null
/// kept null. The first entity that does not cast is refused,
/// `INVALID_REQUEST`, naming its position and why, never its value
/// (decision H17); every such failure is the caller's, never the core's
/// (E41(l)). A key of a type no rule casts to is not sampled
/// (`key_type`).
pub fn cast(entities: &ArrayRef, key: &DataType) -> Result<ArrayRef, CoreError> {
    let target = Target::of(key).ok_or_else(|| not_yet("key_type"))?;
    let mut cast = Vec::with_capacity(entities.len());
    for row in 0..entities.len() {
        cast.push(cast_one(entity(entities, row), target).map_err(|why| uncast(row, key, why))?);
    }
    macro_rules! ints {
        ($builder:ty, $t:ty) => {{
            let mut built = <$builder>::with_capacity(cast.len());
            for value in &cast {
                match value {
                    Cast::Int(v) => built.append_value(
                        <$t>::try_from(*v).expect("the rule table held it to the key's range"),
                    ),
                    _ => built.append_null(),
                }
            }
            Arc::new(built.finish()) as ArrayRef
        }};
    }
    Ok(match key {
        DataType::Int8 => ints!(Int8Builder, i8),
        DataType::Int16 => ints!(Int16Builder, i16),
        DataType::Int32 => ints!(Int32Builder, i32),
        DataType::Int64 => ints!(Int64Builder, i64),
        DataType::UInt8 => ints!(UInt8Builder, u8),
        DataType::UInt16 => ints!(UInt16Builder, u16),
        DataType::UInt32 => ints!(UInt32Builder, u32),
        DataType::UInt64 => ints!(UInt64Builder, u64),
        DataType::Utf8 => {
            let mut built = StringBuilder::new();
            for value in &cast {
                match value {
                    Cast::Text(v) => built.append_value(v),
                    _ => built.append_null(),
                }
            }
            Arc::new(built.finish())
        }
        DataType::Float32 => {
            let mut built = Float32Builder::with_capacity(cast.len());
            for value in &cast {
                match value {
                    Cast::Float(v) => built.append_value(*v as f32),
                    _ => built.append_null(),
                }
            }
            Arc::new(built.finish())
        }
        DataType::Float64 => {
            let mut built = Float64Builder::with_capacity(cast.len());
            for value in &cast {
                match value {
                    Cast::Float(v) => built.append_value(*v),
                    _ => built.append_null(),
                }
            }
            Arc::new(built.finish())
        }
        DataType::Decimal128(precision, scale) => {
            let mut built = Decimal128Builder::with_capacity(cast.len());
            for value in &cast {
                match value {
                    Cast::Int(v) => built.append_value(*v),
                    _ => built.append_null(),
                }
            }
            Arc::new(
                built
                    .finish()
                    .with_precision_and_scale(*precision, *scale)
                    .map_err(|_| not_yet("key_type"))?,
            )
        }
        _ => return Err(not_yet("key_type")),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{
        BooleanArray, Decimal128Array, Float64Array, Int64Array, NullArray, StringArray,
        TimestampMicrosecondArray, UInt64Array,
    };

    fn cast_to(entities: ArrayRef, key: &DataType) -> Result<ArrayRef, (String, u64)> {
        cast(&entities, key).map_err(|e| {
            let field = |name: &str| e.fields.get(name).cloned();
            let reason = match field("reason") {
                Some(Detail::Scalar(Scalar::Utf8(r))) => r,
                other => panic!("no reason: {other:?}"),
            };
            let position = match field("position") {
                Some(Detail::Scalar(Scalar::UInt64(p))) => p,
                other => panic!("no position: {other:?}"),
            };
            assert_eq!(e.code, ErrorCode::InvalidRequest);
            (reason, position)
        })
    }

    fn texts(values: &[&str]) -> ArrayRef {
        Arc::new(StringArray::from(values.to_vec()))
    }

    #[test]
    fn integers_cast_exactly_within_the_key_s_range() {
        let ints: ArrayRef = Arc::new(Int64Array::from(vec![Some(1), None, Some(255)]));
        let cast = cast_to(ints, &DataType::UInt8).unwrap();
        assert_eq!(cast.data_type(), &DataType::UInt8);
        assert_eq!(cast.null_count(), 1);
        let past: ArrayRef = Arc::new(Int64Array::from(vec![1, 256]));
        assert_eq!(
            cast_to(past, &DataType::UInt8).unwrap_err(),
            ("out_of_range".into(), 1)
        );
        let negative: ArrayRef = Arc::new(Int64Array::from(vec![-1]));
        assert_eq!(
            cast_to(negative, &DataType::UInt64).unwrap_err(),
            ("out_of_range".into(), 0)
        );
        let big: ArrayRef = Arc::new(UInt64Array::from(vec![u64::MAX]));
        assert_eq!(
            cast_to(big, &DataType::Int64).unwrap_err().0,
            "out_of_range"
        );
    }

    #[test]
    fn nothing_is_truncated_or_rounded_into_an_exact_key() {
        let half: ArrayRef = Arc::new(Float64Array::from(vec![1.0, 1.5]));
        assert_eq!(
            cast_to(Arc::clone(&half), &DataType::Int64).unwrap_err(),
            ("inexact".into(), 1)
        );
        assert_eq!(
            cast_to(half, &DataType::Decimal128(18, 0)).unwrap_err().0,
            "inexact"
        );
        let decimal: ArrayRef = Arc::new(
            Decimal128Array::from(vec![15])
                .with_precision_and_scale(2, 1)
                .unwrap(),
        );
        assert_eq!(cast_to(decimal, &DataType::Int32).unwrap_err().0, "inexact");
        let infinite: ArrayRef = Arc::new(Float64Array::from(vec![f64::INFINITY]));
        assert_eq!(
            cast_to(infinite, &DataType::Utf8).unwrap_err().0,
            "not_finite"
        );
    }

    #[test]
    fn text_reads_as_a_whole_number_or_a_decimal_by_one_syntax() {
        let good = texts(&["1", "-2", "007"]);
        assert!(cast_to(good, &DataType::Int16).is_ok());
        for bad in ["+1", " 1", "1.0", "0x1", "1e2", "", "x", "\u{ff11}", "-"] {
            assert_eq!(
                cast_to(texts(&[bad]), &DataType::Int64).unwrap_err().0,
                "not_a_number",
                "{bad:?}"
            );
        }
        let decimals = texts(&["+1", "1.0", "1e0", "0.15e1", "-.5", "5."]);
        let cast = cast_to(decimals, &DataType::Decimal128(38, 10)).unwrap();
        let cast = cast.as_primitive::<Decimal128Type>();
        let unit = 10_i128.pow(10);
        assert_eq!(cast.value(0), unit);
        assert_eq!(cast.value(3), 15 * unit / 10);
        assert_eq!(cast.value(4), -5 * unit / 10);
        assert_eq!(cast.value(5), 5 * unit);
        assert_eq!(
            cast_to(texts(&["1.5"]), &DataType::Decimal128(18, 0))
                .unwrap_err()
                .0,
            "inexact"
        );
        assert_eq!(
            cast_to(texts(&["1e18"]), &DataType::Decimal128(18, 0))
                .unwrap_err()
                .0,
            "out_of_range"
        );
        assert_eq!(
            cast_to(texts(&["1e999999999999"]), &DataType::Decimal128(38, 0))
                .unwrap_err()
                .0,
            "out_of_range"
        );
        assert_eq!(
            cast_to(texts(&["1e-999999999999"]), &DataType::Decimal128(38, 0))
                .unwrap_err()
                .0,
            "inexact"
        );
        assert!(cast_to(texts(&["0e999999999999"]), &DataType::Decimal128(38, 0)).is_ok());
        assert_eq!(
            cast_to(texts(&["1.5e"]), &DataType::Decimal128(38, 2))
                .unwrap_err()
                .0,
            "not_a_number"
        );
    }

    #[test]
    fn a_text_key_takes_each_value_s_own_text() {
        let decimal: ArrayRef = Arc::new(
            Decimal128Array::from(vec![-5, 150, 0])
                .with_precision_and_scale(4, 2)
                .unwrap(),
        );
        let cast = cast_to(decimal, &DataType::Utf8).unwrap();
        let cast = cast.as_string::<i32>();
        assert_eq!(
            (cast.value(0), cast.value(1), cast.value(2)),
            ("-0.05", "1.50", "0.00")
        );
        let floats: ArrayRef = Arc::new(Float64Array::from(vec![1.0, 0.5, -0.0]));
        let cast = cast_to(floats, &DataType::Utf8).unwrap();
        let cast = cast.as_string::<i32>();
        assert_eq!(
            (cast.value(0), cast.value(1), cast.value(2)),
            ("1", "0.5", "-0")
        );
        let booleans: ArrayRef = Arc::new(BooleanArray::from(vec![true, false]));
        let cast = cast_to(booleans, &DataType::Utf8).unwrap();
        assert_eq!(cast.as_string::<i32>().value(0), "true");
    }

    #[test]
    fn a_float_key_takes_the_nearest_float_and_other_types_are_refused() {
        let ints: ArrayRef = Arc::new(Int64Array::from(vec![(1 << 53) + 1]));
        let cast = cast_to(ints, &DataType::Float64).unwrap();
        assert_eq!(
            cast.as_primitive::<Float64Type>().value(0),
            (1u64 << 53) as f64
        );
        assert_eq!(
            cast_to(texts(&["1e39"]), &DataType::Float32).unwrap_err().0,
            "out_of_range"
        );
        assert_eq!(
            cast_to(texts(&["inf"]), &DataType::Float64).unwrap_err().0,
            "not_a_number"
        );
        let times: ArrayRef = Arc::new(TimestampMicrosecondArray::from(vec![1]));
        assert_eq!(cast_to(times, &DataType::Int64).unwrap_err().0, "type");
        let nulls: ArrayRef = Arc::new(NullArray::new(2));
        let cast = cast_to(nulls, &DataType::Int64).unwrap();
        assert_eq!(cast.null_count(), 2);
    }

    #[test]
    fn a_key_of_another_type_is_not_sampled() {
        let ints: ArrayRef = Arc::new(Int64Array::from(vec![1]));
        let refused = cast(&ints, &DataType::Date32).unwrap_err();
        assert_eq!(refused.code, ErrorCode::UnsupportedRequest);
    }

    #[test]
    fn keys_of_one_value_are_one_key_whatever_their_type() {
        let ints: ArrayRef = Arc::new(Int64Array::from(vec![15]));
        let decimal: ArrayRef = Arc::new(
            Decimal128Array::from(vec![1500, 155])
                .with_precision_and_scale(10, 2)
                .unwrap(),
        );
        let floats: ArrayRef = Arc::new(Float64Array::from(vec![15.0, 1.55]));
        let ints = read(&ints).unwrap();
        let decimal = read(&decimal).unwrap();
        let floats = read(&floats).unwrap();
        assert_eq!(ints[0], decimal[0]);
        assert_eq!(ints[0], floats[0]);
        assert_eq!(
            decimal[1],
            Some(Key::Decimal {
                value: 155,
                scale: 2
            })
        );
        assert_ne!(decimal[1], floats[1]);
    }
}
