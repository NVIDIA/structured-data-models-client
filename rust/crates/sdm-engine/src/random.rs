// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! RS-1: numpy's `default_rng(seed)` (a `SeedSequence` seeding `PCG64`),
//! its bounded draw and its `Generator.shuffle` of a one-dimensional array,
//! which the reference's `sample_target` draws its example order from.
//!
//! These are ports of defined algorithms (decision A4's line; a row in
//! `dependencies.md`), written from numpy's own sources:
//! `bit_generator.pyx` (`SeedSequence`: `mix_entropy`, `generate_state`),
//! `pcg64.h` (the 128-bit XSL RR generator and its seeding), `pcg64.pyx`
//! (`next_uint32` from half of a 64-bit draw, the other half kept for the
//! next one), `distributions.c` (`random_interval`, a masked rejection
//! draw) and `_generator.pyx` (`_shuffle_raw`, Fisher and Yates from the
//! last position down). Vectors recorded from numpy hold them to numpy's
//! answers: `tests/rs1.json` for seeds from 0 to past 2**200, and
//! `tests/rs1-draws.json` (T1.22) for every seed of T0.6's model, 2**70
//! among them, and for numpy's compiled `random_interval` on both of its
//! paths, to a maximum of 2**64 - 1. No other stream is reached: the
//! reference's other draws (RS-2 to RS-5) are on backends and paths the
//! core does not take, and a seed's range is the provider's (RS-7, a
//! capability `sdm-core` reads).

/// A non-negative integer of any size, as numpy's `SeedSequence` reads it:
/// its 32-bit words, least significant first, `[0]` for zero.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entropy(Vec<u32>);

impl Entropy {
    /// The integer whose lowercase hexadecimal digits are `hex` (no sign,
    /// no prefix, no leading zero but for zero itself); `None` for any
    /// other text.
    pub fn from_hex(hex: &str) -> Option<Self> {
        if hex.is_empty() || !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
            return None;
        }
        let digits = hex.trim_start_matches('0');
        let mut words = Vec::new();
        let mut end = digits.len();
        while end > 0 {
            let start = end.saturating_sub(8);
            words.push(u32::from_str_radix(&digits[start..end], 16).ok()?);
            end = start;
        }
        if words.is_empty() {
            words.push(0);
        }
        Some(Entropy(words))
    }

    /// The integer `value`.
    pub fn from_u128(value: u128) -> Self {
        Entropy::from_hex(&format!("{value:x}")).unwrap_or(Entropy(vec![0]))
    }

    /// The words, least significant first.
    pub fn words(&self) -> &[u32] {
        &self.0
    }
}

const POOL_SIZE: usize = 4;
const INIT_A: u32 = 0x43b0_d7e5;
const MULT_A: u32 = 0x931e_8875;
const INIT_B: u32 = 0x8b51_f9dd;
const MULT_B: u32 = 0x58f3_8ded;
const MIX_MULT_L: u32 = 0xca01_f9dd;
const MIX_MULT_R: u32 = 0x4973_f715;
const XSHIFT: u32 = 16;

/// numpy's `SeedSequence` of one entropy value, with no spawn key.
#[derive(Debug, Clone)]
pub struct SeedSequence {
    pool: [u32; POOL_SIZE],
}

impl SeedSequence {
    pub fn new(entropy: &Entropy) -> Self {
        let words = entropy.words();
        let mut hash_const = INIT_A;
        let mut hashmix = |value: u32| {
            let mut value = value ^ hash_const;
            hash_const = hash_const.wrapping_mul(MULT_A);
            value = value.wrapping_mul(hash_const);
            value ^ (value >> XSHIFT)
        };
        let mix = |x: u32, y: u32| {
            let result = MIX_MULT_L
                .wrapping_mul(x)
                .wrapping_sub(MIX_MULT_R.wrapping_mul(y));
            result ^ (result >> XSHIFT)
        };
        let mut pool = [0_u32; POOL_SIZE];
        for (i, slot) in pool.iter_mut().enumerate() {
            *slot = hashmix(words.get(i).copied().unwrap_or(0));
        }
        for i_src in 0..POOL_SIZE {
            for i_dst in 0..POOL_SIZE {
                if i_src != i_dst {
                    pool[i_dst] = mix(pool[i_dst], hashmix(pool[i_src]));
                }
            }
        }
        for &word in words.iter().skip(POOL_SIZE) {
            for slot in pool.iter_mut() {
                *slot = mix(*slot, hashmix(word));
            }
        }
        SeedSequence { pool }
    }

    /// `generate_state(n, np.uint64)`: `2 n` 32-bit words, read as `n`
    /// little-endian 64-bit words.
    pub fn generate_state_u64(&self, n: usize) -> Vec<u64> {
        let mut hash_const = INIT_B;
        let words: Vec<u32> = (0..2 * n)
            .map(|i| {
                let mut value = self.pool[i % POOL_SIZE] ^ hash_const;
                hash_const = hash_const.wrapping_mul(MULT_B);
                value = value.wrapping_mul(hash_const);
                value ^ (value >> XSHIFT)
            })
            .collect();
        words
            .chunks_exact(2)
            .map(|pair| u64::from(pair[0]) | (u64::from(pair[1]) << 32))
            .collect()
    }
}

/// PCG's default 128-bit multiplier.
const MULTIPLIER: u128 = (2_549_297_995_355_413_924_u128 << 64) | 4_865_540_595_714_422_341;

/// numpy's `PCG64` bit generator, with its buffered 32-bit draw.
#[derive(Debug, Clone)]
pub struct Pcg64 {
    state: u128,
    increment: u128,
    /// The high half of the last 64-bit draw, when a 32-bit draw left it.
    kept: Option<u32>,
}

impl Pcg64 {
    /// `PCG64(SeedSequence(entropy))`.
    pub fn new(seeds: &SeedSequence) -> Self {
        let words = seeds.generate_state_u64(4);
        let initial = (u128::from(words[0]) << 64) | u128::from(words[1]);
        let sequence = (u128::from(words[2]) << 64) | u128::from(words[3]);
        let mut rng = Pcg64 {
            state: 0,
            increment: (sequence << 1) | 1,
            kept: None,
        };
        rng.step();
        rng.state = rng.state.wrapping_add(initial);
        rng.step();
        rng
    }

    fn step(&mut self) {
        self.state = self
            .state
            .wrapping_mul(MULTIPLIER)
            .wrapping_add(self.increment);
    }

    /// The next 64-bit draw (`next_uint64`, `random_raw`).
    pub fn next_u64(&mut self) -> u64 {
        self.step();
        let high = (self.state >> 64) as u64;
        let low = self.state as u64;
        (high ^ low).rotate_right((self.state >> 122) as u32)
    }

    /// The next 32-bit draw: the low half of a fresh 64-bit draw, then its
    /// high half.
    pub fn next_u32(&mut self) -> u32 {
        if let Some(kept) = self.kept.take() {
            return kept;
        }
        let next = self.next_u64();
        self.kept = Some((next >> 32) as u32);
        next as u32
    }

    /// A draw from `0..=max`, by masked rejection (`random_interval`).
    pub fn interval(&mut self, max: u64) -> u64 {
        if max == 0 {
            return 0;
        }
        let mask = u64::MAX >> max.leading_zeros();
        if max <= u64::from(u32::MAX) {
            loop {
                let value = u64::from(self.next_u32()) & mask;
                if value <= max {
                    return value;
                }
            }
        }
        loop {
            let value = self.next_u64() & mask;
            if value <= max {
                return value;
            }
        }
    }

    /// `Generator.shuffle` of a one-dimensional array, in place.
    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = self.interval(i as u64) as usize;
            items.swap(i, j);
        }
    }
}

/// `numpy.random.default_rng(seed)` for a non-negative integer seed.
pub fn default_rng(seed: &Entropy) -> Pcg64 {
    Pcg64::new(&SeedSequence::new(seed))
}
