// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Legacy scalar-seeded MT19937 and the unweighted RandomState permutation.
//!
//! The generator/tempering is the audited algorithm in sdm-data's MT19937.
//! It intentionally remains local: the frozen crate layers permit neither an
//! engine-to-data nor a data-to-engine dependency. Warehouse coefficients and
//! their seed-array initialization stay untouched in sdm-data.
//!
//! NumPy v2.3.0 sources: `numpy/random/src/mt19937/mt19937.c` (scalar seed),
//! `mt19937.h` (32/64-bit outputs), `src/distributions/distributions.c`
//! (`random_interval`), and `numpy/random/mtrand.pyx` (`_shuffle_raw`,
//! `permutation`, and unweighted `choice(replace=False)`). The latter draws a
//! full permutation before taking its prefix; it is not reservoir sampling.
//! Source: <https://github.com/numpy/numpy/tree/v2.3.0/numpy/random>.
//! The fixed context seed belongs to the tabular caller, not this primitive.

const N: usize = 624;
const M: usize = 397;
const MATRIX_A: u32 = 0x9908_b0df;
const UPPER_MASK: u32 = 0x8000_0000;
const LOWER_MASK: u32 = 0x7fff_ffff;

/// A legacy scalar-seeded random stream, distinct from PCG64.
#[derive(Debug, Clone)]
pub struct LegacyMt19937 {
    state: [u32; N],
    position: usize,
}

impl LegacyMt19937 {
    /// `mt19937_seed`: the state of one 32-bit seed.
    pub fn seeded(seed: u32) -> Self {
        let mut state = [0u32; N];
        state[0] = seed;
        for i in 1..N {
            let previous = state[i - 1];
            state[i] = 1_812_433_253u32
                .wrapping_mul(previous ^ (previous >> 30))
                .wrapping_add(i as u32);
        }
        LegacyMt19937 { state, position: N }
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

    /// A masked-rejection draw in `0..=maximum`, with NumPy's exact stream
    /// consumption: one 32-bit output below 2**32, two above it, none at zero.
    pub fn interval(&mut self, maximum: u64) -> u64 {
        if maximum == 0 {
            return 0;
        }
        let mask = u64::MAX >> maximum.leading_zeros();
        loop {
            let bits = if maximum <= u64::from(u32::MAX) {
                u64::from(self.next_u32())
            } else {
                (u64::from(self.next_u32()) << 32) | u64::from(self.next_u32())
            };
            let value = bits & mask;
            if value <= maximum {
                return value;
            }
        }
    }

    /// The legacy descending Fisher-Yates shuffle, including its rejection
    /// draws. Empty and singleton slices do not advance the generator.
    pub fn shuffle<T>(&mut self, values: &mut [T]) {
        for index in (1..values.len()).rev() {
            let selected = self.interval(index as u64) as usize;
            values.swap(index, selected);
        }
    }

    /// `RandomState(seed).permutation(length)` on this continuing stream.
    /// A without-replacement sample takes a prefix only after this completes.
    pub fn permutation(&mut self, length: usize) -> Vec<usize> {
        let mut values: Vec<_> = (0..length).collect();
        self.shuffle(&mut values);
        values
    }
}
