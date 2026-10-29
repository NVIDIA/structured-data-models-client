// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Selection over original native row ordinals, before final frame encoding.

use sdm_engine::tabular::{ContextSelection, IndexSelection, ScoringKeys, labelled_rows};

#[test]
fn supplied_new_entities_do_not_cancel_the_existing_entity_holdout() {
    let keys = ScoringKeys::new(vec![2, 500]);
    assert!(keys.needs_source());
    assert_eq!(keys.retain(&[2, 5, 2, 8]), [1, 3]);
    assert!(!ScoringKeys::new(vec![]).needs_source());
}

#[test]
fn repeated_indices_keep_all_matching_rows_and_missing_request_ordinals() {
    let lookup = IndexSelection::new(&[5, 7, 5, 9], &[Some(5), Some(9), Some(5)]);
    assert_eq!(lookup.scored, [0, 2, 3, 0, 2]);
    assert!(lookup.missing.is_empty());
    let missing = IndexSelection::new(&[5, 7], &[Some(8), Some(5), None, Some(8)]);
    assert_eq!(missing.missing, [0, 2, 3]);
}

#[test]
fn holdout_and_unknown_targets_are_removed_before_selection() {
    let candidates = labelled_rows(&[true, false, true, true, false, true], &[2, 2, 5]).unwrap();
    assert_eq!(candidates, [0, 3]);
    assert!(labelled_rows(&[true], &[1]).is_err());
}

#[test]
fn within_limit_rows_are_unchanged_and_have_no_automatic_notice() {
    let context = ContextSelection::new(vec![1, 4, 7], None, 10_000).unwrap();
    assert_eq!(context.notice_rows(), None);
    assert_eq!(context.select(), [1, 4, 7]);
}

#[test]
fn explicit_sampling_is_repeatable_and_orders_only_the_selected_subset() {
    let rows: Vec<usize> = (0..100).filter(|row| row % 7 != 0).collect();
    let first = ContextSelection::new(rows.clone(), Some(5), 10_000).unwrap();
    let second = ContextSelection::new(rows, Some(5), 10_000).unwrap();
    assert_eq!(first.notice_rows(), None);
    let selected = first.select();
    assert_eq!(selected, second.select());
    assert_eq!(selected.len(), 5);
    assert!(selected.windows(2).all(|pair| pair[0] < pair[1]));
    assert!(selected.iter().all(|row| row % 7 != 0));
}

#[test]
fn automatic_cap_reports_the_filtered_population_before_the_draw() {
    let context = ContextSelection::new((0..10_003).collect(), None, 10_000).unwrap();
    assert_eq!(context.notice_rows(), Some(10_003));
    assert_eq!(context.select().len(), 10_000);
    assert!(ContextSelection::new(vec![0], Some(10_001), 10_000).is_err());
    assert!(ContextSelection::new(vec![0], Some(0), 10_000).is_err());
    assert!(ContextSelection::new(vec![2, 1], Some(1), 10_000).is_err());
}
