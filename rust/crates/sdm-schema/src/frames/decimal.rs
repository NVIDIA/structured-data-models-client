// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native Arrow Decimal constructor facts, before shared-reader composition.
//!
//! Frozen public source: wire/dtypes.py::infer_tfm_dtype/_decimal_dtype and
//! schema.py::_column/_stype_of/_infer_primary_key at 43e55f4f.
//! Constructor distinctness uses native Decimal equality; it does not use
//! the SQL preview's scaled-decimal-to-float inference conversion.
//! This module is deliberately unlinked until the shared reader is composed.

use arrow_array::cast::AsArray;
use arrow_array::types::{Decimal128Type, Decimal256Type};
use arrow_array::{ArrowPrimitiveType, RecordBatch};
use arrow_schema::DataType;
use sdm_contract::ops::Dtype;

type I256 = <Decimal256Type as ArrowPrimitiveType>::Native;

/// Exact equality facts within one admitted, fixed-type Decimal column.
/// The scale is constant within that field, so unscaled equality is exact
/// native Decimal equality, including negative scale and signed zero.
/// These are comparison keys only; original Arrow values remain attached.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Unscaled {
    Decimal128(i128),
    Decimal256(I256),
}

impl Unscaled {
    fn fits_int64(&self, scale: i32) -> bool {
        match self {
            Self::Decimal128(value) => fits_int64(&value.to_string(), scale),
            Self::Decimal256(value) => fits_int64(&value.to_string(), scale),
        }
    }
}

/// One native column's inferred wire type and exact constructor keys.
/// No public schema or behavior is chosen by the native bridge.
pub(crate) struct Column {
    pub(crate) dtype: Dtype,
    pub(crate) values: Vec<Option<Unscaled>>,
}

/// Read a column after shared Arrow admission has checked batch schemas.
/// Empty batches retain the field's declared precision and scale.
/// The caller uses `None` for other physical types and its existing reader.
pub(crate) fn read(batches: &[RecordBatch], index: usize, data_type: &DataType) -> Option<Column> {
    let (precision, scale) = match data_type {
        DataType::Decimal128(precision, scale) | DataType::Decimal256(precision, scale) => {
            (*precision, *scale)
        }
        _ => return None,
    };
    let mut values = Vec::new();
    for batch in batches {
        let array = batch.column(index);
        match data_type {
            DataType::Decimal128(_, _) => values.extend(
                array
                    .as_primitive::<Decimal128Type>()
                    .iter()
                    .map(|value| value.map(Unscaled::Decimal128)),
            ),
            DataType::Decimal256(_, _) => values.extend(
                array
                    .as_primitive::<Decimal256Type>()
                    .iter()
                    .map(|value| value.map(Unscaled::Decimal256)),
            ),
            _ => unreachable!("only a Decimal field reaches native Decimal reading"),
        }
    }
    let dtype = dtype(precision, i32::from(scale), &values);
    Some(Column { dtype, values })
}

/// Infer from complete native Decimal metadata and exact unscaled values.
/// Native scale is an i32 fact even when the closed storage type is narrower.
/// Shared composition must supply that original fact without an i8 cast.
pub(crate) fn dtype(precision: u8, scale: i32, values: &[Option<Unscaled>]) -> Dtype {
    if scale > 0 {
        // Frozen Arrow-backed inference answers from the declared scale,
        // including whole values such as 1.00 and an empty/all-null column.
        Dtype::Float64
    } else if precision <= 18 || values.iter().flatten().all(|value| value.fits_int64(scale)) {
        // Preserve the actual native precision fast path. A negative scale
        // can expand a precision-18 value past int64, but the frozen public
        // constructor still reports int64 before reading that value.
        Dtype::Int64
    } else {
        Dtype::String
    }
}

/// Whether a canonical unscaled integer times 10**(-scale) fits int64.
/// This is used only for nonpositive scale after the precision fast path.
/// Comparing digits avoids overflow throughout the native i32 scale domain.
fn fits_int64(unscaled: &str, scale: i32) -> bool {
    debug_assert!(scale <= 0);
    let negative = unscaled.starts_with('-');
    let digits = unscaled.trim_start_matches('-');
    if digits == "0" {
        return true;
    }
    let zeros = scale.unsigned_abs();
    if zeros > 18 {
        return false;
    }
    let zeros = zeros as usize;
    let length = digits.len() + zeros;
    if length != 19 {
        return length < 19;
    }
    let limit = if negative {
        b"9223372036854775808"
    } else {
        b"9223372036854775807"
    };
    digits
        .bytes()
        .chain(std::iter::repeat_n(b'0', zeros))
        .cmp(limit.iter().copied())
        .is_le()
}
