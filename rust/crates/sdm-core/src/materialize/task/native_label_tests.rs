// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Consumers of actual restored groups, before later example selection.

use super::*;
use sdm_contract::ErrorCode;
use sdm_engine::labels::{Labelled, Labels, LinkValue};

fn lists(values: &[&[i64]]) -> Labels {
    Labels::Lists(
        values
            .iter()
            .map(|row| row.iter().copied().map(LinkValue::Int).collect())
            .collect(),
    )
}

#[test]
fn restored_group_values_are_authoritative_and_preserve_absent_group_positions() {
    let original = Labelled {
        mask: vec![true, false, true, true],
        labels: lists(&[&[1, 2], &[], &[3]]),
    };
    let restored = restore_group_labels(original, lists(&[&[3], &[2, 1], &[1]])).unwrap();
    assert_eq!(restored.mask, [true, false, true, true]);
    assert_eq!(restored.labels, lists(&[&[3], &[2, 1], &[1]]));
    let selected = select_labelled(restored, &[false, true, true, false]).unwrap();
    assert_eq!(selected.mask, [false, false, true, false]);
    assert_eq!(selected.labels, lists(&[&[2, 1]]));
}

#[test]
fn a_group_dropped_later_still_requires_its_complete_native_result() {
    for values in [lists(&[]), lists(&[&[1]]), lists(&[&[1], &[2], &[3]])] {
        let original = Labelled {
            mask: vec![true, true],
            labels: lists(&[&[1], &[2]]),
        };
        let error = restore_group_labels(original, values).unwrap_err();
        assert_eq!(error.code, ErrorCode::InvalidMessage);
    }
    let original = Labelled {
        mask: vec![true, true],
        labels: lists(&[&[1], &[2]]),
    };
    let restored = restore_group_labels(original, lists(&[&[3], &[4]])).unwrap();
    assert_eq!(
        select_labelled(restored, &[false, false]).unwrap().labels,
        lists(&[])
    );
}

#[test]
fn native_completion_cannot_change_the_list_shape_into_a_scalar_answer() {
    for values in [
        Labels::Int(vec![1]),
        Labels::Text(vec!["one".into()]),
        Labels::Bool(vec![true]),
    ] {
        let original = Labelled {
            mask: vec![true],
            labels: lists(&[&[1]]),
        };
        assert_eq!(
            restore_group_labels(original, values).unwrap_err().code,
            ErrorCode::InvalidMessage
        );
    }
}

#[test]
fn all_empty_list_groups_and_no_groups_are_distinct_valid_completions() {
    for values in [lists(&[]), lists(&[&[], &[]])] {
        let original = Labelled {
            mask: vec![true; values.len()],
            labels: values.clone(),
        };
        let restored = restore_group_labels(original, values.clone()).unwrap();
        assert_eq!(restored.labels, values);
    }
}

#[test]
fn selection_checks_complete_candidate_alignment_instead_of_zip_truncation() {
    for selected in [vec![], vec![true], vec![true, true, true]] {
        let original = Labelled {
            mask: vec![true, false],
            labels: lists(&[&[1]]),
        };
        assert_eq!(
            select_labelled(original, &selected).unwrap_err().code,
            ErrorCode::InvalidMessage
        );
    }
}
