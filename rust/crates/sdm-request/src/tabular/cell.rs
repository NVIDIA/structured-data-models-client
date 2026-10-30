// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Policy over individually reached native conversions of effectful cells.

use sdm_contract::direct::cell::{CellObservation, CellShape, CheckedDirectCell, DirectCellInput};
use sdm_contract::ops::Dtype;
use sdm_contract::{CoreError, Detail, Scalar, Sentence};

fn canonical_integer(digits: &str) -> bool {
    let body = digits.strip_prefix('-').unwrap_or(digits);
    !body.is_empty() && body.bytes().all(|b| b.is_ascii_digit())
        && (body.len() == 1 || !body.starts_with('0')) && digits != "-0"
}

pub fn check(input: DirectCellInput) -> Result<CheckedDirectCell, CoreError> {
    let dtype = input.identity.dtype;
    let integer = matches!(dtype, Dtype::Int32 | Dtype::Int64);
    let floating = matches!(dtype, Dtype::Float32 | Dtype::Float64);
    let error = |sentence| CoreError::new(sentence, "a native cell cannot be represented")
        .with_field("dtype", Detail::text(dtype.wire_name()));
    let protocol = || CoreError::protocol(Sentence::InvalidMessage,
        "a native cell observation disagrees with its resolved dtype or representation");
    let value = match input.observation {
        CellObservation::ScalarAdmission { shape, native_type } => {
            if shape != CellShape::Scalar {
                return Err(error(if shape == CellShape::Bytes { Sentence::DirectNativeBytes } else { Sentence::DirectNativeContainer })
                    .with_field("column", Detail::text(&input.identity.column))
                    .with_field("native_type", Detail::text(native_type))
                    .with_field("shape", Detail::text(if shape == CellShape::Mapping { "a mapping" } else { "a container" })));
            }
            None
        }
        CellObservation::Missingness { missing } => missing.then_some(Scalar::Null),
        CellObservation::NumericFiniteness { finite } if integer || floating => {
            if !finite { return Err(error(Sentence::DirectNativeNonFinite)); }
            None
        }
        CellObservation::IntegerCoercion { digits, fractional } if integer => {
            if fractional { return Err(error(Sentence::DirectNativeFraction)); }
            let digits = digits.ok_or_else(|| error(Sentence::DirectNativeIntegerRange))?;
            if !canonical_integer(&digits) { return Err(protocol()); }
            let number = digits.parse::<i64>().map_err(|_| error(Sentence::DirectNativeIntegerRange))?;
            Some(if number.unsigned_abs() > 9_007_199_254_740_991 {
                Scalar::Utf8(digits)
            } else { Scalar::Int64(number) })
        }
        CellObservation::FloatCoercion { bits } if floating => {
            if !bits.is_some_and(|bits| f64::from_be_bytes(bits).is_finite()) {
                return Err(error(Sentence::DirectNativeRange));
            }
            None
        }
        CellObservation::FloatPrecision { exceeds } if floating => {
            if exceeds { return Err(error(Sentence::DirectNativePrecision)); }
            None
        }
        CellObservation::FloatExactness { bits, integral_matches, large_fraction } if floating => {
            let number = f64::from_be_bytes(bits);
            if !number.is_finite() {
                return Err(protocol());
            }
            if integral_matches.is_none() && large_fraction {
                return Err(error(Sentence::DirectNativeDecimalMagnitude));
            }
            if integral_matches == Some(false) {
                return Err(error(Sentence::DirectNativeIntegralPrecision));
            }
            Some(Scalar::Float64(number))
        }
        CellObservation::BooleanCoercion { value } if dtype == Dtype::Bool => Some(Scalar::Bool(value)),
        CellObservation::TextCoercion { value } if dtype == Dtype::String => Some(Scalar::Utf8(value)),
        CellObservation::TimestampCoercion { value } if matches!(dtype, Dtype::Date | Dtype::Time) => Some(Scalar::Utf8(value)),
        _ => return Err(protocol()),
    };
    Ok(CheckedDirectCell { identity: input.identity, value, exact_integer_limit: 9_007_199_254_740_991,
        integer_min: i64::MIN.to_string(), integer_max: i64::MAX.to_string() })
}
