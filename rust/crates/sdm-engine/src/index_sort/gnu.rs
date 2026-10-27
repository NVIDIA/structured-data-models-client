// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Explicit GNU 14 ordering policy for the SIMD depth fallback.
//!
//! This independently expressed index algorithm preserves comparator decisions,
//! median ties, recursion order, insertion threshold and heap-child ties.
//! Its presence does not establish a producer's toolchain/profile binding.

use super::SimdKey;

pub(super) fn sort<T: SimdKey>(keys: &[T], indices: &mut [usize]) {
    if indices.is_empty() {
        return;
    }
    let depth = 2 * indices.len().ilog2();
    introsort(keys, indices, depth);
    // GNU uses a guarded insertion on the first 16 and an unguarded tail.
    // A bound check changes no comparator decision: the first block contains
    // a minimum, so no tail insertion can cross the start of the range.
    insertion(keys, indices);
}

fn insertion<T: SimdKey>(keys: &[T], indices: &mut [usize]) {
    for i in 1..indices.len() {
        let index = indices[i];
        let mut at = i;
        while at != 0 && keys[index] < keys[indices[at - 1]] {
            indices[at] = indices[at - 1];
            at -= 1;
        }
        indices[at] = index;
    }
}

fn introsort<T: SimdKey>(keys: &[T], mut indices: &mut [usize], mut depth: u32) {
    while indices.len() > 16 {
        if depth == 0 {
            heap_sort(keys, indices);
            return;
        }
        depth -= 1;
        let a = 1;
        let b = indices.len() / 2;
        let c = indices.len() - 1;
        let median = if keys[indices[a]] < keys[indices[b]] {
            if keys[indices[b]] < keys[indices[c]] {
                b
            } else if keys[indices[a]] < keys[indices[c]] {
                c
            } else {
                a
            }
        } else if keys[indices[a]] < keys[indices[c]] {
            a
        } else if keys[indices[b]] < keys[indices[c]] {
            c
        } else {
            b
        };
        indices.swap(0, median);
        let pivot = indices[0];
        let mut left = 1;
        let mut right = indices.len();
        let cut = loop {
            while keys[indices[left]] < keys[pivot] {
                left += 1;
            }
            right -= 1;
            while keys[pivot] < keys[indices[right]] {
                right -= 1;
            }
            if left >= right {
                break left;
            }
            indices.swap(left, right);
            left += 1;
        };
        let (lower, upper) = indices.split_at_mut(cut);
        introsort(keys, upper, depth);
        indices = lower;
    }
}

fn adjust_heap<T: SimdKey>(keys: &[T], indices: &mut [usize], top: usize, value: usize) {
    let len = indices.len();
    let mut hole = top;
    let mut child = hole;
    // Descend to a leaf before moving value back up. On equal children the
    // right child wins, unlike the scalar NumPy heap fallback.
    while child < (len - 1) / 2 {
        child = 2 * (child + 1);
        if keys[indices[child]] < keys[indices[child - 1]] {
            child -= 1;
        }
        indices[hole] = indices[child];
        hole = child;
    }
    if len.is_multiple_of(2) && child == (len - 2) / 2 {
        child = 2 * (child + 1);
        indices[hole] = indices[child - 1];
        hole = child - 1;
    }
    while hole > top {
        let parent = (hole - 1) / 2;
        if keys[indices[parent]] < keys[value] {
            indices[hole] = indices[parent];
            hole = parent;
        } else {
            break;
        }
    }
    indices[hole] = value;
}

fn heap_sort<T: SimdKey>(keys: &[T], indices: &mut [usize]) {
    for parent in (0..indices.len() / 2).rev() {
        let value = indices[parent];
        adjust_heap(keys, indices, parent, value);
    }
    for end in (2..=indices.len()).rev() {
        let value = indices[end - 1];
        indices[end - 1] = indices[0];
        adjust_heap(keys, &mut indices[..end - 1], 0, value);
    }
}

#[cfg(test)]
mod tests {
    use super::{heap_sort, introsort, sort};

    #[test]
    fn forced_heap_and_partition_fallbacks_sort_extremes_without_losing_indices() {
        for n in [17, 31, 64, 257, 1025] {
            let keys: Vec<u64> = (0..n)
                .map(|i| u64::MAX - ((i * 17 + 3) % 7) as u64)
                .collect();
            for force_heap in [false, true] {
                let mut indices: Vec<usize> = (0..n).rev().collect();
                if force_heap {
                    introsort(&keys, &mut indices, 0);
                } else {
                    sort(&keys, &mut indices);
                }
                assert!(indices.windows(2).all(|p| keys[p[0]] <= keys[p[1]]));
                indices.sort_unstable();
                assert_eq!(indices, (0..n).collect::<Vec<_>>());
            }
        }
    }

    #[test]
    fn even_and_odd_heap_edges_retain_all_indices() {
        for n in 0..=16 {
            let keys = vec![i64::MAX; n];
            let mut indices: Vec<usize> = (0..n).collect();
            heap_sort(&keys, &mut indices);
            indices.sort_unstable();
            assert_eq!(indices, (0..n).collect::<Vec<_>>());
        }
    }
}
