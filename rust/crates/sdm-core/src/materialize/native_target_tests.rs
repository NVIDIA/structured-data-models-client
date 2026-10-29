// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[test]
fn native_column_positions_bind_the_complete_actual_missingness_observation() {
    let present = [true, false, true];
    assert_eq!(column_positions(3, &present, true).unwrap(), [0, 2]);
    assert_eq!(column_positions(3, &present, false).unwrap(), [0, 1, 2]);
    assert_eq!(column_positions(0, &[], true).unwrap(), []);
    assert_eq!(column_positions(2, &[false, false], true).unwrap(), []);
    for present in [vec![], vec![true], vec![true, false, true, true]] {
        assert_eq!(
            column_positions(3, &present, true).unwrap_err().code,
            sdm_contract::ErrorCode::InvalidMessage
        );
    }
}

#[test]
fn native_filter_and_window_selection_keep_source_order_and_complete_alignment() {
    assert_eq!(selected_positions(&[0, 2, 7], &[false, true, true]).unwrap(), [2, 7]);
    assert_eq!(selected_positions(&[], &[]).unwrap(), []);
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
    for actual in [vec![], vec![2], vec![0], vec![2, 0], vec![0, 0], vec![0, 2, 3]] {
        assert_eq!(
            validate_positions(&[0, 2], &actual, actual.len()).unwrap_err().code,
            sdm_contract::ErrorCode::InvalidMessage
        );
    }
    assert_eq!(
        validate_positions(&[0, 2], &[0, 2], 1).unwrap_err().code,
        sdm_contract::ErrorCode::InvalidMessage
    );
}
