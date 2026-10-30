// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Legacy RandomState draws, deliberately distinct from the PCG64 fixtures.

use sdm_engine::random::LegacyMt19937;

#[test]
fn context_seed_permutations_preserve_draw_order() {
    // np.random.RandomState(42).permutation(n), not sorted sampled positions.
    assert_eq!(LegacyMt19937::seeded(42).permutation(5), [1, 4, 2, 0, 3]);
    assert_eq!(
        LegacyMt19937::seeded(42).permutation(10),
        [8, 1, 5, 0, 7, 2, 9, 4, 3, 6]
    );
    // An independently recorded frozen pandas entity draw also used seed 7.
    assert_eq!(&LegacyMt19937::seeded(7).permutation(8)[..3], &[2, 5, 0]);
}

#[test]
fn raw_scalar_seed_stream_matches_the_mt19937_reference_vector() {
    // The MT19937 reference implementation's standard seed-5489 vector.
    let mut stream = LegacyMt19937::seeded(5489);
    let actual: Vec<_> = (0..5)
        .map(|_| stream.interval(u64::from(u32::MAX)))
        .collect();
    assert_eq!(
        actual,
        [
            3_499_211_612,
            581_869_302,
            3_890_346_734,
            3_586_334_585,
            545_404_204
        ]
    );
}

#[test]
fn wide_interval_combines_the_first_output_as_high_word() {
    let mut stream = LegacyMt19937::seeded(42);
    assert_eq!(stream.interval(u64::MAX), 0x5fe1_dc66_cbea_3db3);
    assert_eq!(stream.interval(u64::from(u32::MAX)), 4_083_286_876);
}

#[test]
fn empty_and_singleton_permutations_and_zero_interval_do_not_draw() {
    let mut stream = LegacyMt19937::seeded(42);
    assert_eq!(stream.interval(0), 0);
    assert!(stream.permutation(0).is_empty());
    assert_eq!(stream.permutation(1), [0]);
    assert_eq!(stream.permutation(10), [8, 1, 5, 0, 7, 2, 9, 4, 3, 6]);
}

#[test]
fn shuffle_keeps_every_position_across_generator_state_blocks() {
    let mut stream = LegacyMt19937::seeded(42);
    let mut positions = stream.permutation(10_001);
    let first = positions.clone();
    positions.sort_unstable();
    assert_eq!(positions, (0..10_001).collect::<Vec<_>>());
    assert_ne!(first, stream.permutation(10_001));
    assert_eq!(first, LegacyMt19937::seeded(42).permutation(10_001));
}
