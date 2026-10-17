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

const N: usize = 624;
const M: usize = 397;
const MATRIX_A: u32 = 0x9908_b0df;
const UPPER_MASK: u32 = 0x8000_0000;
const LOWER_MASK: u32 = 0x7fff_ffff;

/// The modulus of the seeded row order, and the bound of each coefficient:
/// `2**31 - 1`, a prime below 2**31, so a product of two residues stays
/// inside SQLite's 64-bit integers.
pub(crate) const HASH_PRIME: u64 = (1 << 31) - 1;

struct Mt19937 {
    state: [u32; N],
    position: usize,
}

impl Mt19937 {
    /// `mt19937_seed`: the state of one 32-bit seed.
    fn seeded(seed: u32) -> Self {
        let mut state = [0u32; N];
        state[0] = seed;
        for i in 1..N {
            let previous = state[i - 1];
            state[i] = 1_812_433_253u32
                .wrapping_mul(previous ^ (previous >> 30))
                .wrapping_add(i as u32);
        }
        Mt19937 { state, position: N }
    }

    /// `mt19937_init_by_array`: the state of a seed's 32-bit words.
    fn by_array(key: &[u32]) -> Self {
        let mut generator = Mt19937::seeded(19_650_218);
        let state = &mut generator.state;
        let (mut i, mut j) = (1usize, 0usize);
        for _ in 0..N.max(key.len()) {
            let previous = state[i - 1];
            state[i] = (state[i] ^ (previous ^ (previous >> 30)).wrapping_mul(1_664_525))
                .wrapping_add(key[j])
                .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= N {
                state[0] = state[N - 1];
                i = 1;
            }
            if j >= key.len() {
                j = 0;
            }
        }
        for _ in 0..N - 1 {
            let previous = state[i - 1];
            state[i] = (state[i] ^ (previous ^ (previous >> 30)).wrapping_mul(1_566_083_941))
                .wrapping_sub(i as u32);
            i += 1;
            if i >= N {
                state[0] = state[N - 1];
                i = 1;
            }
        }
        state[0] = 0x8000_0000;
        generator.position = N;
        generator
    }

    fn generate(&mut self) {
        let state = &mut self.state;
        for k in 0..N {
            let y = (state[k] & UPPER_MASK) | (state[(k + 1) % N] & LOWER_MASK);
            let mut next = state[(k + M) % N] ^ (y >> 1);
            if y & 1 == 1 {
                next ^= MATRIX_A;
            }
            state[k] = next;
        }
        self.position = 0;
    }

    /// The next 32-bit output, tempered.
    fn next_u32(&mut self) -> u32 {
        if self.position >= N {
            self.generate();
        }
        let mut y = self.state[self.position];
        self.position += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }

    /// `randint(0, high)` of the legacy generator for a range below
    /// 2**32: each draw an output masked to the range's bits, drawn again
    /// while past the range.
    fn below(&mut self, high: u32) -> u32 {
        let range = high - 1;
        let mask = u32::MAX >> range.leading_zeros();
        loop {
            let value = self.next_u32() & mask;
            if value <= range {
                return value;
            }
        }
    }
}

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
