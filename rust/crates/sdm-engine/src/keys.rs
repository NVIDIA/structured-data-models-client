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
use sdm_contract::data::{WrittenNumber, written_number};
use sdm_contract::types::ColumnType;
use sdm_contract::{CoreError, Detail, Scalar, Sentence};

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

/// Whether a key of `data_type` is read: the integers, either text offset
/// width and, from T1.32a, the floats and a `decimal128` of no negative scale.
/// The separate provider cast rule determines which target types it builds.
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
            | DataType::LargeUtf8
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
    macro_rules! text {
        ($offset:ty) => {{
            let values = array.as_string::<$offset>();
            (0..values.len())
                .map(|i| (!values.is_null(i)).then(|| Key::Text(values.value(i).to_owned())))
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
        DataType::Utf8 => text!(i32),
        DataType::LargeUtf8 => text!(i64),
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

/// The keys `array` holds where its text writes numbers: a column of
/// numbers a host provider hands over as their exact text (a Postgres
/// `numeric` declared without a precision,
/// `sdm_contract::data::EXACT_NUMBER_MARKER`), whose spellings of one
/// value may differ (`1.0` and `1.00`, `1E-7` and `1.0E-7`). Each text is
/// the key of the number it writes ([`Key::number_text`]), so two keys are
/// the same exactly when their values are, as the source compares them,
/// with nothing passed through a float (decision H22, register P2G-2 and
/// P2C-3); an array of any other type is read as [`read`] reads it.
pub fn read_numbers(array: &ArrayRef) -> Result<Vec<Option<Key>>, CoreError> {
    if array.data_type() != &DataType::Utf8 {
        return read(array);
    }
    let values = array.as_string::<i32>();
    Ok((0..values.len())
        .map(|i| (!values.is_null(i)).then(|| Key::number_text(values.value(i))))
        .collect())
}

impl Key {
    /// The key of the number `text` writes ([`written`]), by its value: a
    /// whole one within 128 bits as [`Key::Int`] and one with up to 38
    /// places as [`Key::Decimal`], as a decimal column's key of the same
    /// value is; a wider one as its significant digits and the power of
    /// ten they stand at, one text for every spelling of the value. Text
    /// that writes no number (a `NaN`, an `Infinity`) is its own text.
    pub fn number_text(text: &str) -> Key {
        let Ok(written) = written(text) else {
            return Key::Text(text.to_owned());
        };
        if written.digits.is_empty() {
            return Key::Int(0);
        }
        let negative = written.negative;
        let signed = |value: i128| if negative { -value } else { value };
        if let Ok(value) = written.digits.parse::<i128>() {
            if written.scale <= 0 {
                if let Ok(value) = rescale(value, written.scale, 0) {
                    return Key::Int(signed(value));
                }
            } else if written.scale <= 38 {
                return Key::Decimal {
                    value: signed(value),
                    scale: written.scale as i8,
                };
            }
        }
        Key::Text(sdm_contract::data::exact_number_text(text))
    }
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
    /// A float, with its text as the reference writes it ([`float_text`]):
    /// `1e16` is `1e+16`.
    Float(f64, String),
    Bool(bool),
    Text(&'a str),
    /// Its unscaled `digits` (signed) ten-to-the-`scale`ths, of any width
    /// (a `decimal256`'s too), read by its exact text.
    Decimal {
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
            let value = f64::from(entities.as_primitive::<Float32Type>().value(row));
            Entity::Float(value, float_text(value, true))
        }
        DataType::Float64 => {
            let value = entities.as_primitive::<Float64Type>().value(row);
            Entity::Float(value, float_text(value, false))
        }
        DataType::Boolean => Entity::Bool(entities.as_boolean().value(row)),
        DataType::Utf8 => Entity::Text(entities.as_string::<i32>().value(row)),
        DataType::LargeUtf8 => Entity::Text(entities.as_string::<i64>().value(row)),
        DataType::Decimal128(_, scale) => {
            let value = entities.as_primitive::<Decimal128Type>().value(row);
            Entity::Decimal {
                scale: *scale,
                digits: value.to_string(),
            }
        }
        DataType::Decimal256(_, scale) => {
            let value = entities.as_primitive::<Decimal256Type>().value(row);
            Entity::Decimal {
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
/// that drops a digit other than zero, `OutOfRange` when it overflows. The
/// scales are wide enough that no exponent text can overflow them.
fn rescale(value: i128, from: i64, to: i64) -> Result<i128, Uncast> {
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

/// Text written as a decimal number
/// ([`sdm_contract::data::written_number`]); `NotANumber` for any other.
fn written(text: &str) -> Result<WrittenNumber, Uncast> {
    written_number(text).ok_or(Uncast::NotANumber)
}

/// Text written as a decimal number ([`written`]) as its value
/// ten-to-the-`scale`ths, exactly.
fn read_decimal(text: &str, scale: i8) -> Result<i128, Uncast> {
    let written = written(text)?;
    if written.digits.is_empty() {
        return Ok(0);
    }
    // The digits end in one other than zero, so a place past the key's
    // scale is a digit lost, whatever the value's size.
    if written.scale > i64::from(scale) {
        return Err(Uncast::Inexact);
    }
    let value: i128 = written.digits.parse().map_err(|_| Uncast::OutOfRange)?;
    let value = rescale(value, written.scale, i64::from(scale))?;
    Ok(if written.negative { -value } else { value })
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
/// | float | finite and whole, in range | its text as the reference writes it ([`float_text`]), an infinity `inf` or `-inf` | finite: that text, read as a decimal |
/// | boolean | 1 or 0 | `true` or `false` | 1 or 0 |
/// | text | `-?[0-9]+`, in range | itself | `[+-]?` digits, `.`, digits, exponent, exact at scale s, in p digits |
/// | decimal | whole, in range | its digits at its own scale | exact at scale s, in p digits |
/// | any other type | refused (`type`) | refused | refused |
///
/// A float key (32 or 64 bits) takes an integer or a boolean only within
/// the range its width holds every integer in, 2**24 or 2**53 (`inexact`
/// past it), and a float, text in
/// the decimal syntax above or a decimal as the nearest float of its
/// width, rounded once, as a float is by its type; one past the width's
/// range is refused (`out_of_range`). Otherwise nothing is rounded or
/// truncated: a value the key's type does not hold exactly is refused,
/// naming why ([`Uncast`]). A float NaN is a missing entity, as a null is
/// ([`missing`]), refused before the cast.
fn cast_one(entity: Entity<'_>, target: Target) -> Result<Cast, Uncast> {
    match (entity, target) {
        (Entity::Null, _) => Ok(Cast::Null),
        (Entity::Other, _) => Err(Uncast::Type),
        // An infinity is text (`inf`, `-inf`), as the reference casts it.
        (Entity::Float(_, text), Target::Text) => Ok(Cast::Text(text)),
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
        // A decimal of any width (a decimal256's past 128 bits among them)
        // is read by its exact text, its trailing zeros carrying no value.
        (Entity::Decimal { scale, digits, .. }, Target::Int { min, max }) => {
            in_range(read_decimal(&decimal_text(&digits, scale), 0)?, min, max)
        }

        (Entity::Int(value), Target::Text) => Ok(Cast::Text(value.to_string())),
        (Entity::Bool(value), Target::Text) => Ok(Cast::Text(value.to_string())),
        (Entity::Text(text), Target::Text) => Ok(Cast::Text(text.to_owned())),
        (Entity::Decimal { scale, digits, .. }, Target::Text) => {
            Ok(Cast::Text(decimal_text(&digits, scale)))
        }

        (Entity::Int(value), Target::Decimal { precision, scale }) => {
            within(rescale(value, 0, i64::from(scale))?, precision)
        }
        (Entity::Float(_, text), Target::Decimal { precision, scale }) => {
            within(read_decimal(&text, scale)?, precision)
        }
        (Entity::Bool(value), Target::Decimal { precision, scale }) => {
            within(rescale(i128::from(value), 0, i64::from(scale))?, precision)
        }
        (Entity::Text(text), Target::Decimal { precision, scale }) => {
            within(read_decimal(text, scale)?, precision)
        }
        (
            Entity::Decimal {
                scale: from,
                digits,
                ..
            },
            Target::Decimal { precision, scale },
        ) => within(
            read_decimal(&decimal_text(&digits, from), scale)?,
            precision,
        ),

        (Entity::Int(value), Target::Float { single }) => whole_float(value, single),
        (Entity::Float(value, _), Target::Float { single }) => finite(if single {
            f64::from(value as f32)
        } else {
            value
        }),
        (Entity::Bool(value), Target::Float { single }) => whole_float(i128::from(value), single),
        (Entity::Text(text), Target::Float { single }) => read_float(text, single),
        (Entity::Decimal { scale, digits, .. }, Target::Float { single }) => {
            read_float(&decimal_text(&digits, scale), single)
        }
    }
}

/// An integer as a float of the key's width, only within the range where
/// the width holds every integer exactly: up to 2**24 from zero for 32
/// bits, 2**53 for 64, ends included. Past it, where a float holds only
/// some integers, it is `Inexact` even where one is held exactly, as the
/// reference refuses it (Arrow's checked cast of an integer to a float),
/// so whether an entity is taken never depends on its bits (T1.32a fix
/// round 1, the recorded reference).
fn whole_float(value: i128, single: bool) -> Result<Cast, Uncast> {
    let limit: i128 = if single { 1 << 24 } else { 1 << 53 };
    if value < -limit || value > limit {
        return Err(Uncast::Inexact);
    }
    Ok(Cast::Float(value as f64))
}

/// A float of the key's width, held as a double: `OutOfRange` where the
/// nearest one is infinite.
fn finite(value: f64) -> Result<Cast, Uncast> {
    if !value.is_finite() {
        return Err(Uncast::OutOfRange);
    }
    Ok(Cast::Float(value))
}

/// Text in the decimal syntax of [`written`] as the nearest float of the
/// key's width, rounded once, at that width (an integer, a double or a
/// decimal's text each go to a 32-bit key directly, never through a
/// double); `NotANumber` for any other text (`inf`, `nan` and hexadecimal
/// among it).
fn read_float(text: &str, single: bool) -> Result<Cast, Uncast> {
    written(text)?;
    let value = if single {
        text.parse::<f32>().map(f64::from)
    } else {
        text.parse::<f64>()
    };
    finite(value.map_err(|_| Uncast::NotANumber)?)
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

/// A float's text as the reference writes a float entity of a text key:
/// pandas casts the entities to the key's Arrow string type through
/// pyarrow's cast of a float to text, which is Arrow C++'s
/// `FloatToStringFormatter`, double-conversion's `ToShortest` with the
/// flag `EMIT_POSITIVE_EXPONENT_SIGN`, `inf` and `nan`, the exponent
/// character `e` and the plain range `-6` to `10` (T1.36 fix 4, the final
/// review's P2-2). So: the shortest digits that read back as the float at
/// its own width (`single`: a 32-bit float's as a 32-bit float), the
/// nearer of two such and the even one of two as near; written plainly
/// from `1e-6` to below `1e10` (`0.000001`, `9999999000`, `3`, `-0`),
/// outside that as one digit, a point and the rest, `e`, the exponent's
/// sign and its digits, unpadded (`1e-7`, `1e+10`, `1.5e-7`, `5e-324`);
/// `inf`, `-inf` and `nan`.
fn float_text(value: f64, single: bool) -> String {
    if value.is_nan() {
        return "nan".to_owned();
    }
    let sign = if value.is_sign_negative() { "-" } else { "" };
    if value.is_infinite() {
        return format!("{sign}inf");
    }
    let (digits, exponent) = shortest(value.abs(), single);
    let count = i64::try_from(digits.len()).unwrap_or(i64::MAX);
    let written = if (-6..10).contains(&exponent) {
        // The decimal point's place, counted from the first digit.
        let point = exponent + 1;
        if point <= 0 {
            let zeros = "0".repeat(usize::try_from(-point).unwrap_or(0));
            format!("0.{zeros}{digits}")
        } else if point >= count {
            let zeros = "0".repeat(usize::try_from(point - count).unwrap_or(0));
            format!("{digits}{zeros}")
        } else {
            let (whole, fraction) = digits.split_at(usize::try_from(point).unwrap_or(0));
            format!("{whole}.{fraction}")
        }
    } else {
        let (first, rest) = digits.split_at(1);
        let point = if rest.is_empty() { "" } else { "." };
        let exponent_sign = if exponent < 0 { '-' } else { '+' };
        format!(
            "{first}{point}{rest}e{exponent_sign}{}",
            exponent.unsigned_abs()
        )
    };
    format!("{sign}{written}")
}

/// The shortest digits of a finite `value` (not negative) that read back
/// as it at its own width, and the power of ten of the first: Rust's
/// shortest, which takes the upper of two as near, made the even one, as
/// double-conversion's is (and Python's `repr`).
fn shortest(value: f64, single: bool) -> (String, i64) {
    let (digits, exponent) = exponential(&if single {
        // Exact: `value` is a 32-bit float widened.
        #[allow(clippy::cast_possible_truncation)]
        let narrow = value as f32;
        format!("{narrow:e}")
    } else {
        format!("{value:e}")
    });
    // Rust's shortest is the nearer of two of the same length that read
    // back, and the upper of two as near, where double-conversion takes the
    // even one. So they differ only where Rust's last digit is odd, the
    // digits one less in it read back too, and the float's exact value is
    // their midpoint: those digits and a 5 (every finite double's exact
    // digits are within 800).
    let Some((&last, head)) = digits.as_bytes().split_last() else {
        return (digits, exponent);
    };
    if (last - b'0').is_multiple_of(2) {
        return (digits, exponent);
    }
    let lower = format!("{}{}", String::from_utf8_lossy(head), char::from(last - 1));
    let place = exponent - i64::try_from(head.len()).unwrap_or(i64::MAX);
    let text = format!("{lower}e{place}");
    let reads_back = if single {
        text.parse::<f32>().map(f64::from).ok() == Some(value)
    } else {
        text.parse::<f64>().ok() == Some(value)
    };
    if !reads_back {
        return (digits, exponent);
    }
    let (exact, top) = exponential(&format!("{value:.800e}"));
    if top == exponent && exact.trim_end_matches('0') == format!("{lower}5") {
        return (lower.trim_end_matches('0').to_owned(), exponent);
    }
    (digits, exponent)
}

/// Rust's exponential text (`1.5e-7`) as its digits and the power of ten
/// of the first.
fn exponential(text: &str) -> (String, i64) {
    let (mantissa, exponent) = text.split_once('e').unwrap_or((text, "0"));
    let digits = mantissa.chars().filter(char::is_ascii_digit).collect();
    (digits, exponent.parse().unwrap_or(0))
}

/// Whether `entities` holds a missing entity: a null, or a float NaN,
/// which every host's values may write for a missing one (pandas' NaN, a
/// Python host's float NaN), so the core reads it as one whatever the
/// host's conversion kept.
pub fn missing(entities: &ArrayRef) -> bool {
    if entities.null_count() > 0 || matches!(entities.data_type(), DataType::Null) {
        return true;
    }
    match entities.data_type() {
        DataType::Float32 => entities
            .as_primitive::<Float32Type>()
            .values()
            .iter()
            .any(|v| v.is_nan()),
        DataType::Float64 => entities
            .as_primitive::<Float64Type>()
            .values()
            .iter()
            .any(|v| v.is_nan()),
        _ => false,
    }
}

/// The refusal of the entity at `position`, which does not cast to `key`.
fn uncast(position: usize, key: &DataType, why: Uncast) -> CoreError {
    let key_type = ColumnType::from_arrow("key", key)
        .map(|t| t.cell_type())
        .unwrap_or("unknown");
    CoreError::new(
        Sentence::EntityUncast,
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
        BooleanArray, Decimal128Array, Float32Array, Float64Array, Int64Array, LargeStringArray,
        NullArray, StringArray, TimestampMicrosecondArray, UInt64Array,
    };
    use sdm_contract::ErrorCode;

    #[test]
    fn identifying_text_is_exact_at_both_offset_widths_and_nonzero_slices() {
        // Lexical, Unicode, empty, NUL and missing values are distinct native
        // graph-key facts. No numeric parsing, casing or normalization applies.
        let values = [
            Some("outside slice"),
            Some("007"),
            Some("7"),
            Some("7.0"),
            Some(""),
            Some("ΟΣ"),
            Some("İ"),
            Some("é"),
            Some("e\u{301}"),
            Some("\0"),
            Some("<NA>"),
            Some("nan"),
            None,
            Some("007"),
            Some("outside slice"),
        ];
        let expected = values[1..values.len() - 1]
            .iter()
            .map(|value| value.map(|text| Key::Text(text.to_owned())))
            .collect::<Vec<_>>();
        let narrow: ArrayRef = Arc::new(StringArray::from(values.to_vec()));
        let large: ArrayRef = Arc::new(LargeStringArray::from(values.to_vec()));
        for original in [narrow, large] {
            assert!(supported(original.data_type()));
            let sliced = original.slice(1, values.len() - 2);
            assert_eq!(read(&sliced).unwrap(), expected);
            // Independently read pages preserve the same ordered keys and
            // nulls as a single logical slice, without rewriting offsets.
            let first = sliced.slice(0, 5);
            let rest = sliced.slice(5, sliced.len() - 5);
            let mut paged = read(&first).unwrap();
            paged.extend(read(&rest).unwrap());
            assert_eq!(paged, expected);
            assert_eq!(sliced.data_type(), original.data_type());
            assert_eq!(original.len(), values.len());
        }
    }

    #[test]
    fn empty_and_all_missing_text_keys_preserve_their_native_shape() {
        for values in [Vec::<Option<&str>>::new(), vec![None, None]] {
            for array in [
                Arc::new(StringArray::from(values.clone())) as ArrayRef,
                Arc::new(LargeStringArray::from(values.clone())) as ArrayRef,
            ] {
                assert_eq!(read(&array).unwrap(), vec![None; values.len()]);
            }
        }
    }

    #[test]
    fn reading_large_offsets_does_not_replace_the_provider_cast_rule() {
        let entities: ArrayRef = Arc::new(LargeStringArray::from(vec!["7", "007"]));
        let casted = cast(&entities, &DataType::Int64).unwrap();
        assert_eq!(read(&casted).unwrap(), vec![Some(Key::Int(7)); 2]);
        let refused = cast(&entities, &DataType::LargeUtf8).unwrap_err();
        assert_eq!(refused.code, ErrorCode::UnsupportedRequest);
        assert_eq!(
            read(&entities).unwrap(),
            vec![
                Some(Key::Text("7".to_owned())),
                Some(Key::Text("007".to_owned())),
            ]
        );
    }

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
            cast_to(infinite, &DataType::Decimal128(38, 0))
                .unwrap_err()
                .0,
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
    fn a_float_is_written_as_pyarrow_writes_it() {
        // Each worked by pyarrow 25.0.1's cast of a double to text.
        for (value, text) in [
            (1e16, "1e+16"),
            (1.2e15, "1.2e+15"),
            (1e10, "1e+10"),
            (9_999_999_999.0, "9999999999"),
            (9.999_999e9, "9999999000"),
            (1e9, "1000000000"),
            (1e-6, "0.000001"),
            (1.5e-6, "0.0000015"),
            // The doubles either side of 1e-6 and below 1e10.
            (9.999_999_999_999_997e-7, "9.999999999999997e-7"),
            (1.000_000_000_000_000_2e-6, "0.0000010000000000000002"),
            (9_999_999_999.999_998, "9999999999.999998"),
            (1e-7, "1e-7"),
            (1.5e-7, "1.5e-7"),
            (1.234_567_890_123_456_8e17, "1.2345678901234568e+17"),
            (1e21, "1e+21"),
            (1e22, "1e+22"),
            (5e-324, "5e-324"),
            (f64::MAX, "1.7976931348623157e+308"),
            (f64::MIN_POSITIVE, "2.2250738585072014e-308"),
            // Two shortest texts as near: the even one, as `repr` takes.
            (1_059_438_285_926_254.2, "1.0594382859262542e+15"),
            (0.1, "0.1"),
            (2.5, "2.5"),
            (3.0, "3"),
            (-0.0, "-0"),
            (0.0, "0"),
            (-1.5e-7, "-1.5e-7"),
            (-1e16, "-1e+16"),
            (f64::INFINITY, "inf"),
            (f64::NEG_INFINITY, "-inf"),
        ] {
            assert_eq!(float_text(value, false), text, "{value:e}");
        }
        // A 32-bit float is written at its own width.
        for (value, text) in [
            (1e16_f32, "1e+16"),
            (0.1_f32, "0.1"),
            (1.5e-7_f32, "1.5e-7"),
            (16_777_216_f32, "16777216"),
            (1e10_f32, "1e+10"),
            (9.999_999e9_f32, "9999999000"),
            (1e-6_f32, "0.000001"),
            (9.999_999e-7_f32, "9.999999e-7"),
            (f32::MAX, "3.4028235e+38"),
            (f32::from_bits(1), "1e-45"),
        ] {
            assert_eq!(float_text(f64::from(value), true), text, "{value:e}");
        }
        // Widened, the same 32-bit floats are other doubles.
        assert_eq!(float_text(f64::from(0.1_f32), false), "0.10000000149011612");
        assert_eq!(
            float_text(f64::from(1e16_f32), false),
            "1.0000000272564224e+16"
        );
    }

    #[test]
    fn a_text_key_takes_a_float_as_its_text() {
        let floats: ArrayRef = Arc::new(Float64Array::from(vec![
            1e16,
            1.5e-7,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ]));
        let cast = cast_to(floats, &DataType::Utf8).unwrap();
        let cast = cast.as_string::<i32>();
        let texts: Vec<&str> = (0..cast.len()).map(|row| cast.value(row)).collect();
        assert_eq!(texts, ["1e+16", "1.5e-7", "inf", "-inf"]);
        let single: ArrayRef = Arc::new(Float32Array::from(vec![1e16_f32, 0.1]));
        let cast = cast_to(single, &DataType::Utf8).unwrap();
        let cast = cast.as_string::<i32>();
        assert_eq!((cast.value(0), cast.value(1)), ("1e+16", "0.1"));
        // The same text reads as a decimal's value.
        let floats: ArrayRef = Arc::new(Float64Array::from(vec![1e16, 1.5e-7]));
        let cast = cast_to(floats, &DataType::Decimal128(38, 10)).unwrap();
        let cast = cast.as_primitive::<Decimal128Type>();
        assert_eq!(cast.value(0), 10_i128.pow(26));
        assert_eq!(cast.value(1), 1500);
    }

    #[test]
    fn a_float_key_takes_exact_integers_and_other_types_are_refused() {
        // An integer a float key holds only by rounding is another entity.
        let ints: ArrayRef = Arc::new(Int64Array::from(vec![1 << 53, (1 << 53) + 1]));
        assert_eq!(
            cast_to(ints, &DataType::Float64).unwrap_err(),
            ("inexact".into(), 1)
        );
        // Past 2**24 a float32 key refuses even an integer it holds.
        let ints: ArrayRef = Arc::new(Int64Array::from(vec![1 << 24, (1 << 24) + 2]));
        assert_eq!(
            cast_to(ints, &DataType::Float32).unwrap_err(),
            ("inexact".into(), 1)
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
    fn a_32_bit_float_key_rounds_once_at_its_width() {
        // Just past the midpoint between 1 and the next float32: a double
        // first would round it to 1 (T1.32a fix round 1).
        let text = "1.000000059604644775390625000001";
        let cast = cast_to(texts(&[text, &format!("-{text}")]), &DataType::Float32).unwrap();
        let cast = cast.as_primitive::<Float32Type>();
        assert_eq!(cast.value(0), 1.000_000_1_f32);
        assert_eq!(cast.value(1), -1.000_000_1_f32);
        let decimal: ArrayRef = Arc::new(
            Decimal128Array::from(vec![1_000_000_059_604_644_775_390_625_000_001_i128])
                .with_precision_and_scale(31, 30)
                .unwrap(),
        );
        let cast = cast_to(decimal, &DataType::Float32).unwrap();
        assert_eq!(cast.as_primitive::<Float32Type>().value(0), 1.000_000_1_f32);
        let ints: ArrayRef = Arc::new(Int64Array::from(vec![1 << 24, (1 << 24) + 1]));
        assert_eq!(
            cast_to(ints, &DataType::Float32).unwrap_err(),
            ("inexact".into(), 1)
        );
        let ints: ArrayRef = Arc::new(Int64Array::from(vec![i64::MAX]));
        assert_eq!(cast_to(ints, &DataType::Float32).unwrap_err().0, "inexact");
        let nan: ArrayRef = Arc::new(Float64Array::from(vec![1.0, f64::NAN]));
        assert!(missing(&nan));
        assert!(!missing(&texts(&["NaN"])));
        assert_eq!(
            cast_to(texts(&["3.4028236e38"]), &DataType::Float32)
                .unwrap_err()
                .0,
            "out_of_range"
        );
        assert!(cast_to(texts(&["3.4028235e38"]), &DataType::Float32).is_ok());
    }

    #[test]
    fn exponents_at_any_extreme_are_named_never_a_panic() {
        for (text, key, why) in [
            (
                "100e2147483647",
                DataType::Decimal128(18, 0),
                "out_of_range",
            ),
            ("100e-2147483648", DataType::Decimal128(18, 0), "inexact"),
            (
                "1e99999999999999999999999",
                DataType::Decimal128(38, 10),
                "out_of_range",
            ),
            (
                "1e-99999999999999999999999",
                DataType::Decimal128(38, 10),
                "inexact",
            ),
            (
                "1e99999999999999999999999",
                DataType::Float64,
                "out_of_range",
            ),
        ] {
            assert_eq!(cast_to(texts(&[text]), &key).unwrap_err().0, why, "{text}");
        }
        for text in ["0e99999999999999999999999", "000.000e-2147483648"] {
            assert!(
                cast_to(texts(&[text]), &DataType::Decimal128(18, 0)).is_ok(),
                "{text}"
            );
        }
        let long = format!("1{}e-400", "0".repeat(400));
        let cast = cast_to(texts(&[&long]), &DataType::Decimal128(18, 0)).unwrap();
        assert_eq!(cast.as_primitive::<Decimal128Type>().value(0), 1);
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
