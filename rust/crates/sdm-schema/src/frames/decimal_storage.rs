// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Typed native Decimal descriptor/storage binding, unlinked until composition.
//! Physical buffers use the existing closed scale-zero Decimal128/256 types.
//! The original logical scale remains a signed-i32 fact, never an Arrow cast.

use std::collections::BTreeMap;

use arrow_array::RecordBatch;
use arrow_schema::{DataType, Schema};
use sdm_contract::frames::{NativeDecimal, NativeDecimalColumn};
use sdm_contract::scalar::{DECIMAL128_MAX_PRECISION, DECIMAL256_MAX_PRECISION};
use sdm_contract::{CoreError, Sentence};

use super::decimal;

fn invalid(message: &str) -> CoreError {
    CoreError::protocol(Sentence::InvalidMessage, message)
}

fn storage_type(native: &NativeDecimal) -> Result<DataType, CoreError> {
    match *native {
        NativeDecimal::Decimal128 { precision, .. }
            if (1..=DECIMAL128_MAX_PRECISION).contains(&precision) =>
        {
            Ok(DataType::Decimal128(precision, 0))
        }
        NativeDecimal::Decimal256 { precision, .. }
            if (1..=DECIMAL256_MAX_PRECISION).contains(&precision) =>
        {
            Ok(DataType::Decimal256(precision, 0))
        }
        _ => Err(invalid(
            "native Decimal precision is outside its physical width",
        )),
    }
}

/// Bind logical declarations to this exact admitted frame schema.
/// Existing shared Arrow admission has already checked every batch against
/// that schema and detached its buffers; no provider type policy is changed.
pub(crate) fn bindings<'a>(
    schema: &Schema,
    columns: &'a [NativeDecimalColumn],
) -> Result<BTreeMap<usize, &'a NativeDecimal>, CoreError> {
    let mut bound = BTreeMap::new();
    for column in columns {
        let field = schema
            .fields()
            .get(column.column)
            .ok_or_else(|| invalid("native Decimal field ordinal is out of bounds"))?;
        if field.name() != &column.name {
            return Err(invalid(
                "native Decimal field name does not match its ordinal",
            ));
        }
        if field.data_type() != &storage_type(&column.decimal)? {
            return Err(invalid(
                "native Decimal requires matching width/precision and scale-zero typed storage",
            ));
        }
        if bound.insert(column.column, &column.decimal).is_some() {
            return Err(invalid("a native Decimal field is declared more than once"));
        }
    }
    Ok(bound)
}

/// Read exact mantissas and infer from the original logical descriptor.
/// The caller first validates field identity through `bindings` for this
/// stage's current frame input; declarations never outlive that snapshot.
pub(crate) fn read(
    batches: &[RecordBatch],
    index: usize,
    native: &NativeDecimal,
) -> Result<decimal::Column, CoreError> {
    let storage = storage_type(native)?;
    for batch in batches {
        let array = batch
            .columns()
            .get(index)
            .ok_or_else(|| invalid("native Decimal batch field is missing"))?;
        if array.data_type() != &storage {
            return Err(invalid(
                "native Decimal batch storage differs from its descriptor",
            ));
        }
    }
    let mut column = decimal::read(batches, index, &storage)
        .ok_or_else(|| invalid("native Decimal storage is not a typed Decimal column"))?;
    let (precision, scale) = match *native {
        NativeDecimal::Decimal128 { precision, scale }
        | NativeDecimal::Decimal256 { precision, scale } => (precision, scale),
    };
    column.dtype = decimal::dtype(precision, scale, &column.values);
    Ok(column)
}
