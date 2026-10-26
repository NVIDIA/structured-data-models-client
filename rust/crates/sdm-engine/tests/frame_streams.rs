// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Vectors from the actual frozen reference's compiled NeighborSampler,
//! and its legacy pandas entity draw. These streams must remain distinct.

use sdm_engine::random::Mt19937;

#[test]
fn neighbor_stream_matches_the_real_cpp_default_reset_and_continuation() {
    let mut rng = Mt19937::seeded(123);
    assert_eq!(rng.floyd(12, 3), vec![2, 3, 6]);
    let checkpoint = rng.checkpoint();
    assert_eq!(rng.floyd(12, 3), vec![0, 3, 2]);
    assert_eq!(rng.floyd(12, 3), vec![6, 9, 11]);
    let mut restored = Mt19937::from_checkpoint(&checkpoint).unwrap();
    assert_eq!(restored.floyd(12, 3), vec![0, 3, 2]);
    let mut seeded = Mt19937::seeded(7);
    assert_eq!(seeded.floyd(12, 3), vec![5, 3, 1]);
    assert_eq!(seeded.floyd(12, 3), vec![6, 2, 3]);
}

#[test]
fn whole_neighborhood_and_zero_count_consume_no_neighbor_draw() {
    let mut rng = Mt19937::seeded(123);
    let before = rng.checkpoint();
    assert_eq!(rng.floyd(12, 12), (0..12).collect::<Vec<_>>());
    assert!(rng.floyd(12, 0).is_empty());
    assert!(rng.floyd(0, 3).is_empty());
    assert_eq!(rng.checkpoint(), before);
}

#[test]
fn seeded_entity_permutation_is_the_frozen_pandas_draw_order() {
    let selected = Mt19937::seeded(7).permutation(8);
    assert_eq!(&selected[..3], &[2, 5, 0]);
}

#[test]
fn malformed_neighbor_checkpoints_are_rejected_before_use() {
    let state = Mt19937::seeded(123).checkpoint();
    let mut wrong = state.clone();
    wrong.words.pop();
    assert!(Mt19937::from_checkpoint(&wrong).is_none());
    let mut wrong = state.clone();
    wrong.position = 625;
    assert!(Mt19937::from_checkpoint(&wrong).is_none());
    let mut wrong = state;
    wrong.words.fill(0);
    assert!(Mt19937::from_checkpoint(&wrong).is_none());
}
