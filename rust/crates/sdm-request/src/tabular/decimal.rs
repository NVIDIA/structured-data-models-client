// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Direct Decimal encoding. The gathered encoder's precision policy differs.

use sdm_contract::data::{WrittenNumber, written_number};
use sdm_contract::ops::Dtype;
use sdm_contract::{CoreError, Detail, Sentence};
use sdm_wire::json::Value;

fn fixed(coefficient: &str, scale: i8) -> String {
    let (sign, digits) = coefficient
        .strip_prefix('-')
        .map_or(("", coefficient), |v| ("-", v));
    if scale <= 0 {
        if digits == "0" {
            return "0".into();
        }
        return format!(
            "{sign}{digits}{}",
            "0".repeat(usize::from(scale.unsigned_abs()))
        );
    }
    let scale = scale as usize;
    if scale >= digits.len() {
        format!("{sign}0.{}{digits}", "0".repeat(scale - digits.len()))
    } else {
        let split = digits.len() - scale;
        format!("{sign}{}.{}", &digits[..split], &digits[split..])
    }
}

fn whole(value: &WrittenNumber) -> Option<String> {
    if value.digits.is_empty() {
        return Some("0".into());
    }
    if value.scale > 0 {
        return None;
    }
    Some(format!(
        "{}{}{}",
        if value.negative { "-" } else { "" },
        value.digits,
        "0".repeat(value.scale.unsigned_abs() as usize)
    ))
}

fn large_fraction(value: &WrittenNumber) -> bool {
    let digits_before_point = value.digits.len() as i64 - value.scale;
    if digits_before_point != 16 {
        return digits_before_point > 16;
    }
    // This branch has a nonzero fractional part. Equality of the integer
    // part to the safe limit therefore also exceeds that limit exactly.
    &value.digits[..16] >= "9007199254740991"
}

pub(super) fn encode(
    coefficient: &str,
    scale: i8,
    dtype: Dtype,
    column: &str,
) -> Result<Value, CoreError> {
    let spelling = fixed(coefficient, scale);
    let parsed = written_number(&spelling).ok_or_else(|| {
        CoreError::protocol(
            Sentence::InvalidMessage,
            "a decimal coefficient is not an integer",
        )
    })?;
    let error = |sentence| {
        CoreError::new(sentence, "a direct decimal cannot be represented")
            .with_field("value", Detail::text(&spelling))
            .with_field("dtype", Detail::text(dtype.wire_name()))
    };
    if dtype == Dtype::String {
        return Ok(Value::Text(spelling));
    }
    if dtype == Dtype::Bool {
        return Ok(Value::Bool(!parsed.digits.is_empty()));
    }
    let integral = whole(&parsed);
    if matches!(dtype, Dtype::Int32 | Dtype::Int64) {
        let digits = integral.ok_or_else(|| error(Sentence::DirectWireFraction))?;
        let integer = digits
            .parse::<i64>()
            .map_err(|_| error(Sentence::DirectWireRange))?;
        return crate::payload::int_cell(i128::from(integer), column);
    }
    if matches!(dtype, Dtype::Float32 | Dtype::Float64) {
        let number = spelling
            .parse::<f64>()
            .ok()
            .filter(|v| v.is_finite())
            .ok_or_else(|| error(Sentence::DirectWireRange))?;
        if integral
            .as_ref()
            .is_some_and(|digits| format!("{number:.0}") != *digits)
            || (integral.is_none() && large_fraction(&parsed))
        {
            return Err(error(Sentence::DirectFloatPrecision));
        }
        return crate::payload::float_cell(number, column);
    }
    Err(crate::predict::not_yet("direct_native_decimal_conversion"))
}
