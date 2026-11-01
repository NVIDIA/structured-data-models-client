// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use arrow_array::{ArrayRef, Decimal128Array, Float64Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use sdm_schema::keys::count_frame;

#[test]
fn compound_keys_count_across_batches_without_rounding_large_ids() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, true),
        Field::new("part", DataType::Utf8, true),
    ]));
    let large = 9_007_199_254_740_993;
    let first = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int64Array::from(vec![Some(large), Some(large + 1), None])) as ArrayRef,
            Arc::new(StringArray::from(vec![Some("a"), Some("a"), Some("a")])) as ArrayRef,
        ],
    )
    .unwrap();
    let second = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int64Array::from(vec![
                Some(large),
                Some(large),
                Some(large),
            ])) as ArrayRef,
            Arc::new(StringArray::from(vec![Some("a"), Some("b"), None])) as ArrayRef,
        ],
    )
    .unwrap();
    let counted = count_frame(&schema, &[first, second], &["id".into(), "part".into()]).unwrap();
    assert_eq!((counted.rows, counted.distinct, counted.missing), (6, 3, 2));
    assert_eq!(counted.missing_columns, ["id", "part"]);
}

#[test]
fn positive_scale_decimals_keep_exact_identity() {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "id",
        DataType::Decimal128(38, 2),
        false,
    )]));
    let large = 9_007_199_254_740_993_i128;
    let values = Decimal128Array::from(vec![large, large + 1, large])
        .with_precision_and_scale(38, 2)
        .unwrap();
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(values)]).unwrap();
    let counted = count_frame(&schema, &[batch], &["id".into()]).unwrap();
    assert_eq!((counted.rows, counted.distinct, counted.missing), (3, 2, 0));
}

#[test]
fn default_frame_float_keys_identify_signed_zeros_and_drop_nan() {
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Float64, true)]));
    let values = Float64Array::from(vec![Some(0.0), Some(-0.0), Some(f64::NAN), None]);
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(values)]).unwrap();
    let counted = count_frame(&schema, &[batch], &["id".into()]).unwrap();
    assert_eq!((counted.rows, counted.distinct, counted.missing), (4, 1, 2));
}
