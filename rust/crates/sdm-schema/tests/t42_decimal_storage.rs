// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native Decimal descriptor/storage tests, pending reviewed execution.

#[path = "../src/frames/decimal.rs"]
mod decimal;
#[path = "../src/frames/decimal_storage.rs"]
mod decimal_storage;

use std::sync::Arc;

use arrow_array::{ArrayRef, Decimal128Array, Decimal256Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use sdm_contract::frames::{NativeDecimal, NativeDecimalColumn};
use sdm_contract::ops::Dtype;

type I256 = <arrow_array::types::Decimal256Type as arrow_array::ArrowPrimitiveType>::Native;

fn declared(decimal: NativeDecimal) -> NativeDecimalColumn {
    NativeDecimalColumn {
        column: 0,
        name: "value_id".into(),
        decimal,
    }
}

#[test]
fn full_scale_domain_keeps_typed_mantissas_and_distinctness() {
    for scale in [i32::MIN, -129, -128, -1, 0, 1, 2, 129, i32::MAX] {
        for wide in [false, true] {
            let (array, native): (ArrayRef, NativeDecimal) = if wide {
                (
                    Arc::new(
                        Decimal256Array::from(vec![
                            Some(I256::from_i128(0)),
                            Some(I256::from_i128(1)),
                            None,
                        ])
                        .with_precision_and_scale(1, 0)
                        .unwrap(),
                    ),
                    NativeDecimal::Decimal256 {
                        precision: 1,
                        scale,
                    },
                )
            } else {
                (
                    Arc::new(
                        Decimal128Array::from(vec![Some(0), Some(1), None])
                            .with_precision_and_scale(1, 0)
                            .unwrap(),
                    ),
                    NativeDecimal::Decimal128 {
                        precision: 1,
                        scale,
                    },
                )
            };
            let batch = RecordBatch::try_from_iter([("value_id", array)]).unwrap();
            let columns = [declared(native.clone())];
            let bound = decimal_storage::bindings(batch.schema().as_ref(), &columns).unwrap();
            assert_eq!(bound.get(&0), Some(&&native));
            let read = decimal_storage::read(&[batch], 0, &native).unwrap();
            assert_eq!(
                read.dtype,
                if scale > 0 {
                    Dtype::Float64
                } else {
                    Dtype::Int64
                }
            );
            assert_ne!(read.values[0], read.values[1]);
            assert!(read.values[2].is_none());
        }
    }
}

#[test]
fn negative_scale_bounds_are_checked_only_after_native_precision_fast_path() {
    for (precision, value, expected) in [
        (18, 999_999_999_999_999_999_i128, Dtype::Int64),
        (19, 922_337_203_685_477_580_i128, Dtype::Int64),
        (19, 922_337_203_685_477_581_i128, Dtype::String),
    ] {
        let array = Decimal128Array::from(vec![value])
            .with_precision_and_scale(precision, 0)
            .unwrap();
        let batch =
            RecordBatch::try_from_iter([("value_id", Arc::new(array) as ArrayRef)]).unwrap();
        let native = NativeDecimal::Decimal128 {
            precision,
            scale: -1,
        };
        let read = decimal_storage::read(&[batch], 0, &native).unwrap();
        assert_eq!(read.dtype, expected);
    }
}

#[test]
fn unmarked_binary_and_mismatched_typed_storage_are_rejected() {
    let native = NativeDecimal::Decimal128 {
        precision: 1,
        scale: 129,
    };
    for physical in [
        DataType::Binary,
        DataType::FixedSizeBinary(16),
        DataType::Decimal256(1, 0),
        DataType::Decimal128(2, 0),
        DataType::Decimal128(1, 1),
    ] {
        let schema = Schema::new(vec![Field::new("value_id", physical, true)]);
        assert!(decimal_storage::bindings(&schema, &[declared(native.clone())]).is_err());
    }
}

#[test]
fn bindings_cannot_name_a_different_field_or_repeat_an_ordinal() {
    let schema = Schema::new(vec![Field::new(
        "value_id",
        DataType::Decimal128(1, 0),
        true,
    )]);
    let column = declared(NativeDecimal::Decimal128 {
        precision: 1,
        scale: -129,
    });
    let mut wrong_name = column.clone();
    wrong_name.name = "other".into();
    let mut wrong_index = column.clone();
    wrong_index.column = 1;
    for columns in [
        vec![wrong_name],
        vec![wrong_index],
        vec![column.clone(), column],
    ] {
        assert!(decimal_storage::bindings(&schema, &columns).is_err());
    }
}
