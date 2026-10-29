// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A Rust host supplies source facts to the same selection used by Python.

use sdm_host::{TabularContext, TabularIndexSelection, tabular_labelled_rows};

#[test]
fn typed_host_filters_before_sampling_and_applies_candidate_positions() {
    let keys: Vec<i64> = (0..43).collect();
    let selected = TabularIndexSelection::new(&keys, &[Some(11), Some(2), Some(11), Some(7)]);
    assert!(selected.missing.is_empty());
    assert_eq!(selected.scored, [11, 2, 11, 7]);
    let known: Vec<bool> = (0..43).map(|row| row % 7 != 0).collect();
    let candidates = tabular_labelled_rows(&known, &selected.scored).unwrap();
    let context = TabularContext::new(candidates.clone(), Some(5)).unwrap();
    let actual: Vec<usize> = context
        .positions()
        .unwrap()
        .into_iter()
        .map(|position| candidates[position])
        .collect();
    assert_eq!(actual, context.select());
    assert_eq!(actual.len(), 5);
    assert!(actual.windows(2).all(|pair| pair[0] < pair[1]));
    assert!(
        actual
            .iter()
            .all(|row| row % 7 != 0 && ![2, 7, 11].contains(row))
    );
    assert!(context.notice_rows().is_none());
}

#[test]
fn typed_host_keeps_the_uncapped_population_without_a_second_take() {
    let context = TabularContext::new(vec![1, 4, 8], None).unwrap();
    assert!(context.positions().is_none());
    assert_eq!(context.select(), [1, 4, 8]);
}

#[test]
fn typed_host_preserves_the_existing_context_capacity_refusal() {
    let error = TabularContext::new(vec![1], Some(10_001)).unwrap_err();
    assert_eq!(
        error.sentence,
        sdm_contract::Sentence::NumContextSamplesPastMost
    );
}
