// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! numpy's legacy `RandomState`: the MT19937 generator, seeded and drawn
//! from exactly as numpy does, whose streams NEP 19 freezes.
//!
//! [`random_state`] seeds it as `base/utils.py::random_state` does: a seed
//! below `2**32` directly, a larger one with its 32-bit words, least
//! significant first. Every draw takes from the stream what numpy's takes:
//! [`RandomState::randint`] and [`RandomState::interval`] by masked
//! rejection, 32 bits at a time when the span allows; [`RandomState::shuffle`]
//! by Fisher-Yates from the end, one interval draw per swap; and
//! [`RandomState::permutation`] and [`RandomState::choice`] by shuffling.

use std::fmt;

const N: usize = 624;
const M: usize = 397;
const MATRIX_A: u32 = 0x9908_b0df;
const UPPER_MASK: u32 = 0x8000_0000;
const LOWER_MASK: u32 = 0x7fff_ffff;

/// What numpy refuses: `code` is the class of the exception numpy raises,
/// `ValueError` or, for an array no allocation can hold, `MemoryError`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RandomError {
    pub code: &'static str,
    pub message: String,
}

impl RandomError {
    pub fn value(message: impl Into<String>) -> Self {
        Self {
            code: "ValueError",
            message: message.into(),
        }
    }

    fn memory(message: impl Into<String>) -> Self {
        Self {
            code: "MemoryError",
            message: message.into(),
        }
    }
}

impl fmt::Display for RandomError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for RandomError {}

/// An empty vector with room for `n` values of `T`, refused as numpy
/// refuses the array: a `ValueError` past the largest size an array can
/// have, and a `MemoryError` when the allocation fails. A size comes from
/// the caller, so it must not reach an infallible allocation, which would
/// abort the process.
pub(crate) fn room<T>(n: usize) -> Result<Vec<T>, RandomError> {
    let fits = n
        .checked_mul(size_of::<T>())
        .is_some_and(|bytes| isize::try_from(bytes).is_ok());
    if !fits {
        return Err(RandomError::value(
            "array is too big; `arr.size * arr.dtype.itemsize` is larger than the maximum \
             possible size.",
        ));
    }
    let mut out = Vec::new();
    out.try_reserve_exact(n).map_err(|_| {
        RandomError::memory(format!(
            "Unable to allocate {} bytes for an array of {n} values",
            n * size_of::<T>()
        ))
    })?;
    Ok(out)
}

/// The state of numpy's MT19937.
#[derive(Clone)]
pub struct RandomState {
    key: [u32; N],
    pos: usize,
}

impl fmt::Debug for RandomState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RandomState")
            .field("pos", &self.pos)
            .finish_non_exhaustive()
    }
}

impl RandomState {
    /// The state as `get_state()` gives it: the key and the position in it.
    pub fn key(&self) -> (&[u32], usize) {
        (&self.key, self.pos)
    }

    /// `set_state(('MT19937', key, pos))`.
    pub fn from_key(key: &[u32], pos: usize) -> Result<Self, RandomError> {
        let key: [u32; N] = key
            .try_into()
            .map_err(|_| RandomError::value(format!("state must be {N} words")))?;
        if pos > N {
            return Err(RandomError::value(format!("pos must be at most {N}")));
        }
        Ok(Self { key, pos })
    }

    /// `RandomState(seed)` for a seed below `2**32` (`mt19937_seed`).
    pub fn new(seed: u32) -> Self {
        let mut key = [0u32; N];
        let mut s = seed;
        for (pos, slot) in key.iter_mut().enumerate() {
            *slot = s;
            s = 1_812_433_253u32
                .wrapping_mul(s ^ (s >> 30))
                .wrapping_add(pos as u32 + 1);
        }
        Self { key, pos: N }
    }

    /// `RandomState(words)` for a one-dimensional array of 32-bit words
    /// (`mt19937_init_by_array`).
    pub fn from_words(words: &[u32]) -> Result<Self, RandomError> {
        if words.is_empty() {
            return Err(RandomError::value("Seed must be non-empty"));
        }
        let mut state = Self::new(19_650_218);
        let mt = &mut state.key;
        let length = words.len();
        let mut i = 1usize;
        let mut j = 0usize;
        for _ in 0..N.max(length) {
            mt[i] = (mt[i] ^ (mt[i - 1] ^ (mt[i - 1] >> 30)).wrapping_mul(1_664_525))
                .wrapping_add(words[j])
                .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= N {
                mt[0] = mt[N - 1];
                i = 1;
            }
            if j >= length {
                j = 0;
            }
        }
        for _ in 0..N - 1 {
            mt[i] = (mt[i] ^ (mt[i - 1] ^ (mt[i - 1] >> 30)).wrapping_mul(1_566_083_941))
                .wrapping_sub(i as u32);
            i += 1;
            if i >= N {
                mt[0] = mt[N - 1];
                i = 1;
            }
        }
        mt[0] = 0x8000_0000;
        state.pos = N;
        Ok(state)
    }

    fn generate(&mut self) {
        let mt = &mut self.key;
        for i in 0..N {
            let y = (mt[i] & UPPER_MASK) | (mt[(i + 1) % N] & LOWER_MASK);
            let mut next = mt[(i + M) % N] ^ (y >> 1);
            if y & 1 == 1 {
                next ^= MATRIX_A;
            }
            mt[i] = next;
        }
        self.pos = 0;
    }

    /// The next 32 random bits.
    pub fn next_u32(&mut self) -> u32 {
        if self.pos == N {
            self.generate();
        }
        let mut y = self.key[self.pos];
        self.pos += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }

    /// The next 64 random bits: two draws, the first the high half.
    pub fn next_u64(&mut self) -> u64 {
        let high = u64::from(self.next_u32());
        (high << 32) | u64::from(self.next_u32())
    }

    /// One value in `0..=span`, by masked rejection as numpy's legacy
    /// generator draws it.
    fn bounded(&mut self, span: u64) -> u64 {
        if span == 0 {
            return 0;
        }
        if span <= u64::from(u32::MAX) {
            if span == u64::from(u32::MAX) {
                return u64::from(self.next_u32());
            }
            let mask = mask_of(span) as u32;
            loop {
                let value = self.next_u32() & mask;
                if u64::from(value) <= span {
                    return u64::from(value);
                }
            }
        }
        if span == u64::MAX {
            return self.next_u64();
        }
        let mask = mask_of(span);
        loop {
            let value = self.next_u64() & mask;
            if value <= span {
                return value;
            }
        }
    }

    /// `random_interval(max)`: one value in `0..=max`, the draw `shuffle`
    /// makes for each swap, and `randint(0, max + 1)` for `int64`.
    pub fn interval(&mut self, max: u64) -> u64 {
        self.bounded(max)
    }

    /// `shuffle(x)` of a one-dimensional array, in place: from the last
    /// item down, each swapped with one drawn from those before it.
    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            // An index fits a u64, and the draw is at most `i`.
            let j = self.interval(i as u64) as usize;
            items.swap(i, j);
        }
    }

    /// `permutation(n)`: `0..n`, shuffled.
    pub fn permutation(&mut self, n: usize) -> Result<Vec<i64>, RandomError> {
        let mut items = room::<i64>(n)?;
        // `room` holds an array of `n` int64 values only for `n` below 2**60.
        items.extend(0..n as i64);
        self.shuffle(&mut items);
        Ok(items)
    }

    /// `choice(a, size=size, replace=False)` for an integer `a`, which is
    /// `permutation(a)[:size]`, and what `DataFrame.sample(n=size)` draws.
    pub fn choice(&mut self, a: i64, size: usize) -> Result<Vec<i64>, RandomError> {
        if a <= 0 && size != 0 {
            return Err(RandomError::value(
                "a must be greater than 0 unless no samples are taken",
            ));
        }
        let population = usize::try_from(a.max(0))
            .map_err(|_| RandomError::value("a population too large to draw from"))?;
        if size > population {
            return Err(RandomError::value(
                "Cannot take a larger sample than population when 'replace=False'",
            ));
        }
        let mut drawn = self.permutation(population)?;
        drawn.truncate(size);
        Ok(drawn)
    }

    /// `randint(low, high, size=size)` for `int64`: `size` values in
    /// `low..high`.
    pub fn randint(&mut self, low: i64, high: i64, size: usize) -> Result<Vec<i64>, RandomError> {
        if low >= high {
            return Err(RandomError::value("low >= high"));
        }
        // `high - 1 - low` fits a u64 for any two int64 bounds.
        let span = (high as i128 - 1 - low as i128) as u64;
        let mut drawn = room::<i64>(size)?;
        drawn.extend((0..size).map(|_| (low as i128 + i128::from(self.bounded(span))) as i64));
        Ok(drawn)
    }
}

/// The smallest mask of ones at least `span`.
fn mask_of(span: u64) -> u64 {
    if span == 0 {
        return 0;
    }
    u64::MAX >> span.leading_zeros()
}

/// `base/utils.py::random_state(seed)`: the stream a non-negative seed of any
/// size stands for, given as its 32-bit words, least significant first, with
/// no high zero words; an empty slice is the seed 0.
pub fn random_state(words: &[u32]) -> Result<RandomState, RandomError> {
    match words {
        [] => Ok(RandomState::new(0)),
        [word] => Ok(RandomState::new(*word)),
        _ => RandomState::from_words(words),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_reference_stream_of_seed_5489() {
        // The first outputs of the reference MT19937 for its default seed.
        let mut state = RandomState::new(5489);
        let first: Vec<u32> = (0..3).map(|_| state.next_u32()).collect();
        assert_eq!(first, [3_499_211_612, 581_869_302, 3_890_346_734]);
    }

    #[test]
    fn numpy_draws_of_seed_42() {
        // np.random.RandomState(42).randint(0, 2**31 - 1, size=4).tolist()
        let mut state = random_state(&[42]).unwrap();
        assert_eq!(
            state.randint(0, (1 << 31) - 1, 4).unwrap(),
            [1_608_637_542, 1_273_642_419, 1_935_803_228, 787_846_414]
        );
    }

    #[test]
    fn numpy_draws_from_words_and_small_spans() {
        // RandomState(np.array([0, 1], dtype=np.uint32)), a seed of 2**32.
        let mut state = random_state(&[0, 1]).unwrap();
        assert_eq!(
            state.randint(i64::MIN, i64::MAX, 3).unwrap(),
            [
                -7_138_995_033_315_767_364,
                -1_515_789_998_802_475_822,
                -8_915_747_259_038_587_123
            ]
        );
        let mut state = random_state(&[7]).unwrap();
        assert_eq!(state.randint(-50, 50, 5).unwrap(), [-3, 18, -25, 17, 33]);
    }

    #[test]
    fn refuses_an_empty_span() {
        let mut state = RandomState::new(1);
        assert!(state.randint(3, 3, 1).is_err());
    }

    #[test]
    fn a_size_no_array_can_have_is_an_error_not_an_abort() {
        let mut state = RandomState::new(1);
        // 2**61 int64 values is past isize::MAX bytes; 2**59 of them, 4 EiB,
        // is past any address space, so the allocation fails.
        for (n, code) in [(1usize << 61, "ValueError"), (1 << 59, "MemoryError")] {
            assert_eq!(state.permutation(n).unwrap_err().code, code);
            assert_eq!(state.randint(0, 10, n).unwrap_err().code, code);
            assert_eq!(state.choice(n as i64, 1).unwrap_err().code, code);
        }
        assert_eq!(
            state.permutation(usize::MAX).unwrap_err().code,
            "ValueError"
        );
        // Nothing was drawn for the refusals.
        assert_eq!(state.next_u32(), RandomState::new(1).next_u32());
    }
}
