// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The shared legacy MT19937 primitive, extracted from the SQLite RS-2
//! coefficient implementation. RS-2 retains that adapter and its vectors;
//! the frame consumer adds checkpointed RS-3 and legacy RS-4 draws.

use sdm_contract::frames::Mt19937State;

const N: usize = 624;
const M: usize = 397;
const MATRIX_A: u32 = 0x9908_b0df;
const UPPER_MASK: u32 = 0x8000_0000;
const LOWER_MASK: u32 = 0x7fff_ffff;

#[derive(Debug, Clone)]
pub struct Mt19937 {
    state: [u32; N],
    position: usize,
}

impl Mt19937 {
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
        Mt19937 { state, position: N }
    }

    /// `mt19937_init_by_array`: the state of a seed's 32-bit words.
    pub fn by_array(key: &[u32]) -> Self {
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
    pub fn next_u32(&mut self) -> u32 {
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
    pub fn below(&mut self, high: u32) -> u32 {
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

impl Mt19937 {
    /// Validates and restores the complete continuing stream.
    pub fn from_checkpoint(state: &Mt19937State) -> Option<Self> {
        if state.words.len() != N
            || state.position as usize > N
            || state.words.iter().all(|&word| word == 0)
        {
            return None;
        }
        let words: [u32; N] = state.words.clone().try_into().ok()?;
        Some(Self {
            state: words,
            position: state.position as usize,
        })
    }

    /// A value-only checkpoint, with no attachment or object identity.
    pub fn checkpoint(&self) -> Mt19937State {
        Mt19937State {
            words: self.state.to_vec(),
            position: self.position as u32,
        }
    }

    /// Legacy RandomState permutation, preserving the drawn order (RS-4).
    pub fn permutation(&mut self, length: usize) -> Vec<usize> {
        let mut rows: Vec<usize> = (0..length).collect();
        for i in (1..length).rev() {
            let at = if let Ok(high) = u32::try_from(i + 1) {
                self.below(high) as usize
            } else {
                let max = i as u64;
                let mask = u64::MAX >> max.leading_zeros();
                loop {
                    let value =
                        ((u64::from(self.next_u32()) << 32) | u64::from(self.next_u32())) & mask;
                    if value <= max {
                        break value as usize;
                    }
                }
            };
            rows.swap(i, at);
        }
        rows
    }

    /// The C++ sampler's modulo-biased Floyd draw (RS-3).
    pub fn floyd(&mut self, population: usize, count: usize) -> Vec<usize> {
        if count >= population {
            return (0..population).collect();
        }
        let mut seen = std::collections::BTreeSet::new();
        let mut rows = Vec::with_capacity(count);
        for i in population - count..population {
            let selected = self.next_u32() as usize % (i + 1);
            let selected = if seen.insert(selected) {
                selected
            } else {
                seen.insert(i);
                i
            };
            rows.push(selected);
        }
        rows
    }
}
