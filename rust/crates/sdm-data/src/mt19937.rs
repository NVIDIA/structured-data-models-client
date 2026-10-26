// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! RS-2: the four coefficients of SQLite's seeded entity draw, from numpy's
//! legacy `RandomState` (MT19937), as the reference draws them
//! (`random_state(seed).randint(0, 2**31 - 1, size=4)`, its SQLite
//! sampler's `_seeded_hash`). NEP 19 freezes that stream, so a seed gives
//! the same coefficients in every numpy release.
//!
//! A port of defined algorithms, written from numpy's own sources:
//! `mt19937.c` (`mt19937_seed`, `mt19937_init_by_array`, the generator and
//! its tempering), `_mt19937.pyx` (`_legacy_seeding`: a seed below 2**32
//! seeds by `mt19937_seed`, a larger one, which the reference splits into
//! its 32-bit words least significant first, by `mt19937_init_by_array`)
//! and `distributions.c` (`random_bounded_uint64_fill` for a range below
//! 2**32: a masked rejection draw of 32-bit outputs, which the legacy
//! `randint` takes). `tests/rs2.json` holds it to numpy's answers.

use sdm_engine::random::Mt19937;

pub(crate) const HASH_PRIME: u64 = (1 << 31) - 1;

/// The stream of `seed`, a non-negative integer of any size, as
/// `random_state(seed)` seeds it: below 2**32 the integer itself, above
/// it its 32-bit words, least significant first.
fn stream(seed: u128) -> Mt19937 {
    match u32::try_from(seed) {
        Ok(small) => Mt19937::seeded(small),
        Err(_) => {
            let mut words = Vec::new();
            let mut rest = seed;
            while rest > 0 {
                words.push(rest as u32);
                rest >>= 32;
            }
            Mt19937::by_array(&words)
        }
    }
}

/// The four coefficients `c0` to `c3` of the seeded hash of `seed`
/// (register RS-2), each below [`HASH_PRIME`].
pub(crate) fn coefficients(seed: u128) -> [u64; 4] {
    let mut generator = stream(seed);
    let high = HASH_PRIME as u32;
    [(); 4].map(|_| u64::from(generator.below(high)))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every seed of `tests/rs2.json`, recorded from the reference's
    /// numpy, from 0 past 2**126, both seedings and their edges.
    #[test]
    fn the_coefficients_are_numpy_s() {
        let recorded: serde_json::Value =
            serde_json::from_str(include_str!("../tests/rs2.json")).unwrap();
        let vectors = recorded["vectors"].as_array().unwrap();
        assert!(vectors.len() > 20);
        for vector in vectors {
            let seed: u128 = vector["seed"].as_str().unwrap().parse().unwrap();
            let expected: Vec<u64> = vector["coefficients"]
                .as_array()
                .unwrap()
                .iter()
                .map(|c| c.as_u64().unwrap())
                .collect();
            assert_eq!(coefficients(seed).to_vec(), expected, "seed {seed}");
        }
    }

    #[test]
    fn the_reference_s_coefficients_of_its_default_seed() {
        // `random_state(42).randint(0, 2**31 - 1, size=4)`, the reference's
        // default seed, whose hash every recorded SQLite sample writes.
        assert_eq!(
            coefficients(42),
            [1_608_637_542, 1_273_642_419, 1_935_803_228, 787_846_414]
        );
    }
}
