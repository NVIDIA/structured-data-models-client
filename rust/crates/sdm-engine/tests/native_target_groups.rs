// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Group actual completed target rows without manufacturing window inputs.

use sdm_engine::labels::{self, Bounds, Column, Labels, LinkValue};

#[test]
fn selected_grouping_reuses_pair_equality_and_first_occurrence_in_batch_order() {
    let values = Column::UInt(vec![Some(u64::MAX), Some(7), Some(u64::MAX), None, Some(2)]);
    let batch = [1, 0, 1, 0, 1];
    let expected = Labels::Lists(vec![
        vec![LinkValue::UInt(7)],
        vec![LinkValue::UInt(u64::MAX), LinkValue::UInt(2)],
    ]);
    let actual = labels::distinct_selected(3, &batch, &values).unwrap();
    assert_eq!(actual.mask, [true, true, false]);
    assert_eq!(actual.labels, expected);
    let windowed = labels::distinct(
        3,
        &batch,
        &values,
        &[Some(5); 5],
        Bounds {
            starts: Some(&[0; 3]),
            ends: &[10; 3],
        },
        None,
    )
    .unwrap();
    assert_eq!(actual, windowed);
}

#[test]
fn selected_grouping_preserves_float32_zero_equality_and_nonfinite_missingness() {
    let values = Column::Float(vec![
        Some(-0.0),
        Some(0.0),
        Some(f32::NAN),
        Some(f32::INFINITY),
    ]);
    let actual = labels::distinct_selected(1, &[0, 0, 0, 0], &values).unwrap();
    let Labels::Lists(groups) = actual.labels else {
        panic!("expected actual lists")
    };
    assert_eq!(groups[0].len(), 2);
    let LinkValue::Float(first) = &groups[0][0] else {
        panic!("expected float32")
    };
    assert_eq!(first.to_bits(), (-0.0_f32).to_bits());
    assert_eq!(groups[0][1], LinkValue::Float(f32::INFINITY));
    assert_eq!(actual.mask, [true]);
}

#[test]
fn selected_grouping_checks_complete_attribution_and_preserves_empty_population() {
    let empty = labels::distinct_selected(2, &[], &Column::Int(vec![])).unwrap();
    assert_eq!(empty.mask, [false, false]);
    assert_eq!(empty.labels, Labels::Lists(vec![]));
    assert!(labels::distinct_selected(1, &[0], &Column::Int(vec![])).is_err());
    assert!(labels::distinct_selected(1, &[1], &Column::Int(vec![Some(1)])).is_err());
}
