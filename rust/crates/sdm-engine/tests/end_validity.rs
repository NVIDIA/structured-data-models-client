// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use sdm_engine::random::{Entropy, Pcg64, SeedSequence};
use sdm_engine::sample::{Validity, static_examples, static_examples_with_validity};

fn rng() -> Pcg64 {
    Pcg64::new(&SeedSequence::new(&Entropy::from_u128(42)))
}

#[test]
fn no_end_keeps_exact_order_and_rng_state() {
    let starts = [Some(1), Some(9), None, Some(2), Some(3)];
    for start in [None, Some(starts.as_slice())] {
        let (mut old, mut new) = (rng(), rng());
        let expected = static_examples(&mut old, 5, start, 4);
        let actual =
            static_examples_with_validity(&mut new, 5, Validity { start, end: None }, 4).unwrap();
        assert_eq!(actual, expected);
        assert_eq!(new.next_u64(), old.next_u64());
    }
}

#[test]
fn end_equal_anchor_expires_null_end_survives_and_bad_lengths_fail() {
    let ends = [Some(3), Some(4), Some(5), None];
    let mut actual = static_examples_with_validity(
        &mut rng(),
        4,
        Validity {
            start: None,
            end: Some(&ends),
        },
        4,
    )
    .unwrap()
    .index;
    actual.sort_unstable();
    assert_eq!(actual, vec![2, 3]);
    assert!(
        static_examples_with_validity(
            &mut rng(),
            5,
            Validity {
                start: None,
                end: Some(&ends)
            },
            4
        )
        .is_err()
    );
}
