// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A Rust host supplies source facts to the same selection used by Python.

use sdm_host::{TabularContext, TabularIndexSelection, TabularScoringKeys, tabular_labelled_rows};

#[test]
fn supplied_population_binds_native_missingness_and_original_source_positions() {
    let scoring = TabularScoringKeys::new(vec![11, 500]);
    assert!(scoring.needs_source());
    let source: Vec<i64> = (0..43).collect();
    let population = scoring.source(&source).unwrap();
    let retained = population.positions();
    assert_eq!(retained.len(), 42);
    assert!(!retained.contains(&11));
    let known: Vec<bool> = retained.iter().map(|row| row % 7 != 0).collect();
    let candidates = population.labelled_rows(&known).unwrap();
    let context = population.context(&candidates, Some(5)).unwrap();
    let actual: Vec<usize> = context
        .positions()
        .unwrap()
        .iter()
        .map(|&position| retained[candidates[position]])
        .collect();
    assert_eq!(actual, context.select());
    assert!(population.labelled_rows(&[true]).is_err());
    assert!(population.context(&[42], Some(1)).is_err());
    assert!(population.context(&[2, 1], Some(1)).is_err());
    assert!(!TabularScoringKeys::new(vec![]).needs_source());
    assert!(TabularScoringKeys::new(vec![]).source(&source).is_err());
}

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

#[test]
fn actual_candidate_labels_order_the_sample_before_mapping_original_rows() {
    let context = TabularContext::new(vec![2, 5, 9, 12, 20, 31], Some(3)).unwrap();
    let labels = [70, -9, 200, 3, 1, -50];
    assert!(context.reduced());
    // Seed42's six-row sample prefix is [0, 1, 5]; only that subset is sorted.
    assert_eq!(
        context.positions(Some(&labels)).unwrap(),
        Some(vec![5, 1, 0])
    );
    assert_eq!(context.select(Some(&labels)).unwrap(), [31, 5, 2]);
    assert!(context.positions(None).is_err());
    assert!(context.positions(Some(&labels[..5])).is_err());
    assert!(context.positions(Some(&[70, -9, 200, 3, 1, 70])).is_err());
}

#[test]
fn uncapped_context_needs_no_native_index_and_rejects_unsolicited_labels() {
    let context = TabularContext::new(vec![1, 4, 8], None).unwrap();
    assert!(!context.reduced());
    assert!(context.positions(None).unwrap().is_none());
    assert_eq!(context.select(None).unwrap(), [1, 4, 8]);
    assert!(context.positions(Some(&[3, 2, 1])).is_err());
}
