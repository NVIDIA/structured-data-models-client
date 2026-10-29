// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[test]
fn native_timestamp_conditions_reuse_utc_normalization_and_reject_other_families() {
    let before: ArrayRef = Arc::new(TimestampNanosecondArray::from(vec![Some(0), None, Some(1)]));
    for zone in ["UTC", "America/New_York"] {
        let actual: ArrayRef = Arc::new(
            arrow_array::TimestampMillisecondArray::from(vec![Some(0), None, Some(3_600_000)])
                .with_timezone(zone),
        );
        assert!(times_ns(&actual, "raw").is_err());
        validate_condition_values(&before, None, &actual, None).unwrap();
        let labels::Column::Time(values) =
            native_condition_column(&actual, &[0, 1, 2], None).unwrap()
        else {
            panic!("temporal condition")
        };
        assert_eq!(values, [Some(0), None, Some(3_600_000_000_000)]);
    }
    let wrong: ArrayRef = Arc::new(arrow_array::StringArray::from(vec![
        Some("2023-01-01"),
        None,
        Some("2023-01-02"),
    ]));
    assert!(validate_condition_values(&before, None, &wrong, None).is_err());
}

#[test]
fn native_column_positions_bind_the_complete_actual_missingness_observation() {
    let present = [true, false, true];
    assert_eq!(column_positions(3, &present, true).unwrap(), [0, 2]);
    assert_eq!(column_positions(3, &present, false).unwrap(), [0, 1, 2]);
    assert_eq!(column_positions(0, &[], true).unwrap(), Vec::<usize>::new());
    assert_eq!(
        column_positions(2, &[false, false], true).unwrap(),
        Vec::<usize>::new()
    );
    for present in [vec![], vec![true], vec![true, false, true, true]] {
        assert_eq!(
            column_positions(3, &present, true).unwrap_err().code,
            sdm_contract::ErrorCode::InvalidMessage
        );
    }
}

#[test]
fn native_filter_and_window_selection_keep_source_order_and_complete_alignment() {
    assert_eq!(
        selected_positions(&[0, 2, 7], &[false, true, true]).unwrap(),
        [2, 7]
    );
    assert_eq!(selected_positions(&[], &[]).unwrap(), Vec::<usize>::new());
    for selection in [vec![], vec![true], vec![true, false, true, false]] {
        assert_eq!(
            selected_positions(&[0, 2, 7], &selection).unwrap_err().code,
            sdm_contract::ErrorCode::InvalidMessage
        );
    }
}

#[test]
fn completion_must_cover_the_exact_selected_rows_not_just_an_ordered_subset() {
    assert!(validate_positions(&[0, 2], &[0, 2], 2).is_ok());
    assert!(validate_positions(&[], &[], 0).is_ok());
    for actual in [
        vec![],
        vec![2],
        vec![0],
        vec![2, 0],
        vec![0, 0],
        vec![0, 2, 3],
    ] {
        assert_eq!(
            validate_positions(&[0, 2], &actual, actual.len())
                .unwrap_err()
                .code,
            sdm_contract::ErrorCode::InvalidMessage
        );
    }
    assert_eq!(
        validate_positions(&[0, 2], &[0, 2], 1).unwrap_err().code,
        sdm_contract::ErrorCode::InvalidMessage
    );
}

#[test]
fn empty_native_groups_keep_the_actual_completed_scalar_meaning() {
    for (input, encoding, expected) in [
        (DataType::Utf8, Some(StringEncoding::Utf8), DataType::Utf8),
        (
            DataType::Binary,
            Some(StringEncoding::Utf8Surrogatepass),
            DataType::Binary,
        ),
        (DataType::Boolean, None, DataType::Boolean),
        (DataType::Int8, None, DataType::Int64),
        (DataType::UInt16, None, DataType::Int64),
        (DataType::UInt64, None, DataType::UInt64),
        (DataType::Float32, None, DataType::Float32),
        (
            DataType::Decimal128(19, 3),
            None,
            DataType::Decimal128(38, 3),
        ),
    ] {
        let values = arrow_array::new_empty_array(&input);
        assert_eq!(group_value_type(&values, encoding).unwrap(), expected);
    }
    assert!(
        group_value_type(
            &arrow_array::new_empty_array(&DataType::Int64),
            Some(StringEncoding::Utf8)
        )
        .is_err()
    );
}

#[test]
fn exact_decimal_projection_preserves_scale_nulls_and_existing_label_kernels() {
    let original: ArrayRef = Arc::new(
        arrow_array::Decimal128Array::from(vec![
            Some(9_007_199_254_740_993_i128),
            None,
            Some(9_007_199_254_740_992),
        ])
        .with_precision_and_scale(38, 0)
        .unwrap(),
    );
    let native: ArrayRef = Arc::new(arrow_array::Float64Array::from(vec![Some(1.0), Some(1.0)]));
    validate_values(&original, None, &native, None).unwrap();
    let exact = mathematical_values(&original, &[2, 0], &native).unwrap();
    assert_eq!(exact.data_type(), original.data_type());
    let column = native_condition_column(&exact, &[0, 1], None).unwrap();
    let target = labels::Target::Column(column);
    let classes = labels::label_classes(&target).unwrap();
    assert_ne!(classes.labels, labels::Labels::Float(vec![1.0, 1.0]));
    let distinct = list_distinct_column(&exact, &[0, 1], None).unwrap();
    let listed = labels::distinct_selected(1, &[0, 0], &distinct).unwrap();
    assert_eq!(
        listed.labels,
        labels::Labels::Lists(vec![vec![
            labels::LinkValue::Decimal(9_007_199_254_740_992, 0),
            labels::LinkValue::Decimal(9_007_199_254_740_993, 0)
        ]])
    );
    let nullable_native: ArrayRef = Arc::new(arrow_array::Float64Array::from(vec![None]));
    let missing = mathematical_values(&original, &[1], &nullable_native).unwrap();
    assert_eq!(missing.null_count(), 1);
    assert!(mathematical_values(&original, &[3], &nullable_native).is_err());
    assert!(mathematical_values(&original, &[], &nullable_native).is_err());
    let changed: ArrayRef = Arc::new(arrow_array::Int64Array::from(vec![7, 8]));
    let source: ArrayRef = Arc::new(arrow_array::Int64Array::from(vec![1, 2, 3]));
    assert!(Arc::ptr_eq(
        &mathematical_values(&source, &[2, 0], &changed).unwrap(),
        &changed
    ));
}

#[test]
fn exact_decimal_sum_average_and_conditions_do_not_use_native_rounded_values() {
    let original: ArrayRef = Arc::new(
        arrow_array::Decimal128Array::from(vec![i128::MAX / 10, -(i128::MAX / 10), 3])
            .with_precision_and_scale(38, 0)
            .unwrap(),
    );
    let native: ArrayRef = Arc::new(arrow_array::Float32Array::from(vec![1.0, 1.0, 1.0]));
    let exact = mathematical_values(&original, &[0, 1, 2], &native).unwrap();
    let values = label_values(&exact).unwrap();
    for (aggregation, expected) in [
        (labels::Aggregation::Sum, 3.0),
        (labels::Aggregation::Avg, 1.0),
    ] {
        assert_eq!(
            labels::aggregate_selected(aggregation, 1, &[0, 0, 0], &values)
                .unwrap()
                .values,
            [expected]
        );
    }
    let column = native_condition_column(&exact, &[0, 1, 2], None).unwrap();
    let condition = labels::Target::Condition {
        target: Box::new(labels::Target::Column(column)),
        op: labels::RelOp::Gt,
        constant: Some(labels::Constant::Int(2)),
    };
    assert_eq!(
        labels::truth(&condition).unwrap().values,
        [true, false, true]
    );
}

#[test]
fn decimal256_projection_is_lossless_without_admitting_a_new_label_family() {
    let dtype = DataType::Decimal256(76, 4);
    let original = arrow_array::new_null_array(&dtype, 3);
    let actual: ArrayRef = Arc::new(arrow_array::Float64Array::from(vec![None, None]));
    let exact = mathematical_values(&original, &[2, 0], &actual).unwrap();
    assert_eq!(exact.data_type(), &dtype);
    assert_eq!(exact.len(), 2);
    assert_eq!(exact.null_count(), 2);
    assert!(group_value_type(&exact, None).is_err());
}
