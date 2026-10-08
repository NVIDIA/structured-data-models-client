// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! C++'s `std::mt19937`, as `relationallib`'s neighbor sampler draws from it.
//!
//! The standard seeds it as numpy's `RandomState(seed)` does (Knuth's
//! initialization of the state from one 32-bit word) and draws the same
//! tempered words, so it is a [`RandomState`] read one raw word at a time.
//! The sampler seeds it from any integer as `base/utils.random_state` seeds
//! numpy's, which [`Mt19937::from`] a [`RandomState`] gives.

use std::collections::HashSet;

use crate::{RandomError, RandomState, room};

/// `std::mt19937`.
#[derive(Clone, Debug)]
pub struct Mt19937 {
    state: RandomState,
}

impl From<RandomState> for Mt19937 {
    /// The generator in `state`'s state, as the sampler seeds its own from a
    /// seed's 32-bit words.
    fn from(state: RandomState) -> Self {
        Self { state }
    }
}

impl Mt19937 {
    /// `std::mt19937(seed)`: the standard keeps the seed modulo `2**32`.
    pub fn new(seed: u64) -> Self {
        Self {
            state: RandomState::new(seed as u32),
        }
    }

    /// The generator's state, which [`Mt19937::from`] takes back.
    pub fn state(&self) -> &RandomState {
        &self.state
    }

    /// `rng()`.
    pub fn next_u32(&mut self) -> u32 {
        self.state.next_u32()
    }

    /// The offsets `relationallib`'s `_sample` picks, in order, from a
    /// neighborhood of `population` when it asks for `count` of them
    /// (`0 < count < population`): Floyd's algorithm on `rng() % (i + 1)`,
    /// with `i` taken in place of a pick already made. A `count` no array
    /// can hold is refused as [`RandomState::permutation`] refuses one.
    pub fn floyd(&mut self, population: i64, count: i64) -> Result<Vec<i64>, RandomError> {
        let mut picked = HashSet::new();
        let mut out = room::<i64>(usize::try_from(count).unwrap_or(0))?;
        for i in population.saturating_sub(count).max(0)..population {
            let mut offset = i64::from(self.next_u32()) % (i + 1);
            if !picked.insert(offset) {
                offset = i;
                picked.insert(i);
            }
            out.push(offset);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_standard_tenth_thousandth_draw() {
        // [rand.predef]: the 10000th invocation of a default-constructed
        // std::mt19937 (seed 5489) produces 4123659995.
        let mut rng = Mt19937::new(5489);
        let mut last = 0;
        for _ in 0..10_000 {
            last = rng.next_u32();
        }
        assert_eq!(last, 4_123_659_995);
    }

    #[test]
    fn the_seed_is_kept_modulo_two_to_the_thirty_two() {
        let (mut a, mut b) = (Mt19937::new(7), Mt19937::new(7 + (1 << 32)));
        assert_eq!(a.next_u32(), b.next_u32());
    }

    #[test]
    fn floyd_never_picks_an_offset_twice() {
        let mut rng = Mt19937::new(1);
        let picks = rng.floyd(5, 4).unwrap();
        let distinct: HashSet<_> = picks.iter().collect();
        assert_eq!(distinct.len(), 4);
        assert!(picks.iter().all(|&p| (0..5).contains(&p)));
    }

    #[test]
    fn floyd_refuses_a_count_no_array_can_hold() {
        let mut rng = Mt19937::new(1);
        assert_eq!(rng.floyd(1 << 62, 1 << 61).unwrap_err().code, "ValueError");
        assert_eq!(rng.floyd(1 << 60, 1 << 59).unwrap_err().code, "MemoryError");
    }
}
