// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Portable index operations for the source-locked NumPy 2.5.3 SIMD algorithm.
//!
//! These profiles describe algorithms, not detected CPUs or supported runtime
//! bindings. No production caller selects either profile yet. Missing values,
//! including NaNs, must be removed by the caller before entering this kernel.
//! The caller must separately establish its producer's algorithm and fallback
//! profile. See `docs/design/rust-core/phase-4/t46-simd-kernels.md`.

mod gnu;
#[cfg(test)]
mod recorded;

/// Explicit lane/store semantics and depth fallback, without runtime dispatch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SimdArgsortProfile {
    /// Four lanes, reversed right partition groups, GNU 14 sort fallback.
    Avx2Gnu14,
    /// Eight lanes, increasing right partition groups, GNU 14 sort fallback.
    Avx512Gnu14,
}

impl SimdArgsortProfile {
    fn lanes(self) -> usize {
        match self {
            Self::Avx2Gnu14 => 4,
            Self::Avx512Gnu14 => 8,
        }
    }
}

mod sealed {
    pub trait Sealed {}
    impl Sealed for i32 {}
    impl Sealed for u32 {}
    impl Sealed for i64 {}
    impl Sealed for u64 {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

/// The six immutable key types in this source's ascending SIMD wrappers.
///
/// Floating values must be non-NaN. Comparisons preserve the original width;
/// floating zero signs remain in the immutable input and compare equal.
pub trait SimdKey: sealed::Sealed + Copy + PartialOrd {
    #[doc(hidden)]
    const MAX_SENTINEL: Self;
    #[doc(hidden)]
    const MIN_SENTINEL: Self;
}

macro_rules! integer_key {
    ($($key:ty),+) => {$(
        impl SimdKey for $key {
            const MAX_SENTINEL: Self = Self::MAX;
            const MIN_SENTINEL: Self = Self::MIN;
        }
    )+};
}
integer_key!(i32, u32, i64, u64);

impl SimdKey for f32 {
    const MAX_SENTINEL: Self = Self::INFINITY;
    const MIN_SENTINEL: Self = Self::NEG_INFINITY;
}

impl SimdKey for f64 {
    const MAX_SENTINEL: Self = Self::INFINITY;
    const MIN_SENTINEL: Self = Self::NEG_INFINITY;
}

/// Sort an initially identity index permutation using one explicit algorithm.
///
/// This is a source component, not a native-profile classifier. It implements
/// the source's original-array sorted shortcut and the full depth fallback.
/// Callers must exclude NaNs and prove their native producer uses this profile.
/// It does not choose missing placement, selected-subset shortcuts or row ids.
pub fn simd_argsort<T: SimdKey>(values: &[T], profile: SimdArgsortProfile) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..values.len()).collect();
    if values.len() > 1 && values.windows(2).any(|pair| pair[1] < pair[0]) {
        // The source casts log2(n) to size_t before doubling. This conversion
        // concerns only the range length, never a key or an index label.
        let depth = 2 * (values.len() as f64).log2() as u32;
        sort_recursive(values, &mut indices, depth, profile);
    }
    indices
}

fn sort_recursive<T: SimdKey>(
    values: &[T],
    indices: &mut [usize],
    depth: u32,
    profile: SimdArgsortProfile,
) {
    // This check precedes the network check in the producer.
    if depth == 0 {
        gnu::sort(values, indices);
        return;
    }
    if indices.len() <= 256 {
        network(values, indices, profile.lanes());
        return;
    }
    let lanes = profile.lanes();
    let spacing = (indices.len() - 1) / lanes;
    // Intrinsic set() reverses this sequence, but its key-only network's
    // median has the same numeric comparison class for non-NaN values.
    let mut samples: Vec<T> = (1..=lanes).map(|i| values[indices[i * spacing]]).collect();
    for i in 1..samples.len() {
        let key = samples[i];
        let mut at = i;
        while at > 0 && key < samples[at - 1] {
            samples[at] = samples[at - 1];
            at -= 1;
        }
        samples[at] = key;
    }
    let pivot = samples[lanes / 2];
    let (cut, smallest, biggest) = partition(values, indices, pivot, profile);
    let (left, right) = indices.split_at_mut(cut);
    if pivot != smallest {
        sort_recursive(values, left, depth - 1, profile);
    }
    if pivot != biggest {
        sort_recursive(values, right, depth - 1, profile);
    }
}

#[derive(Clone, Copy)]
struct Lane<T> {
    key: T,
    index: usize,
}

fn register_stage<T: SimdKey>(vector: &mut [Lane<T>], block: usize, reverse: bool, maxima: u8) {
    let mut before = [vector[0]; 8];
    before[..vector.len()].copy_from_slice(vector);
    for (i, lane) in vector.iter_mut().enumerate() {
        let other = if reverse {
            i / block * block + block - 1 - i % block
        } else {
            i ^ (block / 2)
        };
        let want_max = maxima & (1 << i) != 0;
        // Equality selects the current lane's index, not its partner's.
        if (want_max && before[i].key < before[other].key)
            || (!want_max && before[other].key < before[i].key)
        {
            *lane = before[other];
        }
    }
}

fn register_sort<T: SimdKey>(vector: &mut [Lane<T>]) {
    let eight = vector.len() == 8;
    register_stage(vector, 2, true, if eight { 0xAA } else { 0xA });
    register_stage(vector, 4, true, if eight { 0xCC } else { 0xC });
    register_stage(vector, 2, false, if eight { 0xAA } else { 0xA });
    if eight {
        register_stage(vector, 8, true, 0xF0);
        register_stage(vector, 4, false, 0xCC);
        register_stage(vector, 2, false, 0xAA);
    }
}

fn register_clean<T: SimdKey>(vector: &mut [Lane<T>]) {
    if vector.len() == 8 {
        register_stage(vector, 8, false, 0xF0);
        register_stage(vector, 4, false, 0xCC);
        register_stage(vector, 2, false, 0xAA);
    } else {
        register_stage(vector, 4, false, 0xC);
        register_stage(vector, 2, false, 0xA);
    }
}

fn exchange<T: SimdKey>(registers: &mut [Vec<Lane<T>>], a: usize, b: usize) {
    for lane in 0..registers[a].len() {
        if registers[b][lane].key < registers[a][lane].key {
            let saved = registers[a][lane];
            registers[a][lane] = registers[b][lane];
            registers[b][lane] = saved;
        }
    }
}

fn network<T: SimdKey>(values: &[T], indices: &mut [usize], lanes: usize) {
    let mut count = 256 / lanes;
    while count > 1 && 2 * indices.len() <= count * lanes {
        count /= 2;
    }
    let padding = Lane {
        key: T::MAX_SENTINEL,
        index: usize::MAX,
    };
    let mut registers = vec![vec![padding; lanes]; count];
    for (i, &index) in indices.iter().enumerate() {
        registers[i / lanes][i % lanes] = Lane {
            key: values[index],
            index,
        };
    }
    for register in &mut registers {
        register_sort(register);
    }
    let mut width = 2;
    while width <= count {
        for block in registers.chunks_exact_mut(width) {
            for i in 0..width / 2 {
                let upper = width - i - 1;
                block[upper].reverse();
                exchange(block, i, upper);
                block[upper].reverse();
            }
            let mut group = width / 2;
            while group >= 2 {
                for base in (0..width).step_by(group) {
                    for i in 0..group / 2 {
                        exchange(block, base + i, base + i + group / 2);
                    }
                }
                group /= 2;
            }
            for register in block {
                register_clean(register);
            }
        }
        width *= 2;
    }
    for (i, index) in indices.iter_mut().enumerate() {
        *index = registers[i / lanes][i % lanes].index;
    }
}

struct Partition<'a, T> {
    values: &'a [T],
    pivot: T,
    smallest: T,
    biggest: T,
    left: usize,
    right: usize,
    profile: SimdArgsortProfile,
}

impl<T: SimdKey> Partition<'_, T> {
    fn observe(&mut self, key: T) {
        if key < self.smallest {
            self.smallest = key;
        }
        if self.biggest < key {
            self.biggest = key;
        }
    }

    fn emit(&mut self, indices: &mut [usize], vector: &[usize]) {
        let mut lower = Vec::with_capacity(vector.len());
        let mut upper = Vec::with_capacity(vector.len());
        for &index in vector {
            let key = self.values[index];
            self.observe(key);
            if key < self.pivot {
                lower.push(index);
            } else {
                upper.push(index);
            }
        }
        if self.profile == SimdArgsortProfile::Avx2Gnu14 {
            upper.reverse();
            let packed: Vec<usize> = lower.iter().chain(&upper).copied().collect();
            // AVX2 stores the entire rearranged vector twice. The overwrite
            // schedule protects unread indices, including in unrolled groups.
            indices[self.left..self.left + vector.len()].copy_from_slice(&packed);
            indices[self.right - vector.len()..self.right].copy_from_slice(&packed);
        } else {
            indices[self.left..self.left + lower.len()].copy_from_slice(&lower);
            indices[self.right - upper.len()..self.right].copy_from_slice(&upper);
        }
        self.left += lower.len();
        self.right -= upper.len();
    }
}

fn partition<T: SimdKey>(
    values: &[T],
    indices: &mut [usize],
    pivot: T,
    profile: SimdArgsortProfile,
) -> (usize, T, T) {
    let lanes = profile.lanes();
    let unroll = if indices.len() <= 32 * lanes { 1 } else { 4 };
    let group_width = unroll * lanes;
    let mut left = 0;
    let mut right = indices.len();
    let mut state = Partition {
        values,
        pivot,
        smallest: T::MAX_SENTINEL,
        biggest: T::MIN_SENTINEL,
        left: 0,
        right,
        profile,
    };
    // The remainder count is fixed before any swaps move the right cursor.
    for _ in 0..indices.len() % group_width {
        let key = values[indices[left]];
        state.observe(key);
        if key < pivot {
            left += 1;
        } else {
            right -= 1;
            indices.swap(left, right);
        }
    }
    state.left = left;
    state.right = right;
    if right - left == lanes {
        let saved = indices[left..right].to_vec();
        state.emit(indices, &saved);
    } else if left != right {
        let saved_left = indices[left..left + group_width].to_vec();
        let saved_right = indices[right - group_width..right].to_vec();
        left += group_width;
        right -= group_width;
        while left != right {
            let saved = if state.right - right < left - state.left {
                right -= group_width;
                indices[right..right + group_width].to_vec()
            } else {
                let saved = indices[left..left + group_width].to_vec();
                left += group_width;
                saved
            };
            for vector in saved.chunks_exact(lanes) {
                state.emit(indices, vector);
            }
        }
        for saved in [&saved_left, &saved_right] {
            for vector in saved.chunks_exact(lanes) {
                state.emit(indices, vector);
            }
        }
    }
    (state.left, state.smallest, state.biggest)
}

#[cfg(test)]
mod tests {
    use super::{SimdArgsortProfile, sort_recursive};

    #[test]
    fn exhausted_depth_uses_gnu_insertion_even_inside_the_network_base_case() {
        let values = [2_i64, 1, 2, 1, 2, 1];
        for profile in [
            SimdArgsortProfile::Avx2Gnu14,
            SimdArgsortProfile::Avx512Gnu14,
        ] {
            let mut indices = [0, 1, 2, 3, 4, 5];
            sort_recursive(&values, &mut indices, 0, profile);
            assert_eq!(indices, [1, 3, 5, 0, 2, 4]);
        }
    }

    #[test]
    fn all_actual_wheel_forced_fallback_permutations_match_both_depth_zero_profiles() {
        let data: serde_json::Value =
            serde_json::from_str(super::recorded::controls::FALLBACK_CONTROLS).unwrap();
        assert_eq!(data["numpy"], "2.5.3");
        assert_eq!(data["naturally_reachable_public_fallback_witness"], false);
        let records = data["records"].as_array().unwrap();
        assert_eq!(records.len(), 123);
        for record in records {
            let name = record["case"].as_str().unwrap();
            assert_eq!(record["forced_depth"], 0);
            let encoded = record["values_hex"].as_str().unwrap();
            let bytes: Vec<u8> = (0..encoded.len())
                .step_by(2)
                .map(|at| u8::from_str_radix(&encoded[at..at + 2], 16).unwrap())
                .collect();
            let expected: Vec<usize> = serde_json::from_value(record["order"].clone()).unwrap();
            macro_rules! check {
                ($dtype:ty, $width:literal) => {{
                    assert_eq!(bytes.len() % $width, 0, "{name}");
                    let values: Vec<$dtype> = bytes
                        .chunks_exact($width)
                        .map(|word| <$dtype>::from_le_bytes(word.try_into().unwrap()))
                        .collect();
                    for profile in [
                        SimdArgsortProfile::Avx2Gnu14,
                        SimdArgsortProfile::Avx512Gnu14,
                    ] {
                        let mut indices: Vec<usize> = (0..values.len()).collect();
                        sort_recursive(&values, &mut indices, 0, profile);
                        assert_eq!(indices, expected, "{name}, {profile:?}");
                    }
                    let after: Vec<u8> = values
                        .iter()
                        .flat_map(|value| value.to_le_bytes())
                        .collect();
                    assert_eq!(after, bytes, "input changed: {name}");
                }};
            }
            match record["dtype"].as_str().unwrap() {
                "<i4" => check!(i32, 4),
                "<u4" => check!(u32, 4),
                "<i8" => check!(i64, 8),
                "<u8" => check!(u64, 8),
                "<f4" => check!(f32, 4),
                "<f8" => check!(f64, 8),
                dtype => panic!("unexpected recorded fallback dtype {dtype}"),
            }
        }
    }
}
