// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Exact scalar tie permutations, not a selection of the native SIMD route.

use sdm_engine::sample::{introsort_argsort, scalar_argsort};
use serde::Deserialize;

#[derive(Deserialize)]
struct Recording {
    source_pattern: Vec<u8>,
    vectors: Vec<Vector>,
}

#[derive(Deserialize)]
struct Vector {
    length: usize,
    order: Vec<usize>,
}

fn recording() -> Recording {
    serde_json::from_str(include_str!("data/t46_scalar_argsort.json")).unwrap()
}

#[test]
fn scalar_integer_and_floating_widths_keep_recorded_duplicate_permutations() {
    let recording = recording();
    for vector in recording.vectors {
        let input: Vec<u8> = recording
            .source_pattern
            .iter()
            .copied()
            .cycle()
            .take(vector.length)
            .collect();
        macro_rules! check_width {
            ($($dtype:ty),+ $(,)?) => {
                $(
                    let values: Vec<$dtype> = input.iter().map(|&v| v as $dtype).collect();
                    assert_eq!(
                        scalar_argsort(&values),
                        vector.order,
                        "{} scalar values of {}",
                        vector.length,
                        stringify!($dtype)
                    );
                )+
            };
        }
        check_width!(i8, i16, i32, i64, f32, f64);
        let times: Vec<i64> = input.iter().map(|&value| i64::from(value)).collect();
        assert_eq!(introsort_argsort(&times), vector.order);
    }
}

#[test]
fn order_isomorphic_wide_unsigned_values_never_pass_through_float() {
    // The exact same comparisons as the native recording, translated into
    // the top four unsigned values. In particular, all are past 2^63 and
    // collapse to the same f64, so float narrowing changes this permutation.
    let recording = recording();
    for vector in recording.vectors {
        let values: Vec<u64> = recording
            .source_pattern
            .iter()
            .cycle()
            .take(vector.length)
            .map(|&value| u64::MAX - 3 + u64::from(value))
            .collect();
        assert_eq!(scalar_argsort(&values), vector.order);
    }
}

#[test]
fn signed_zero_and_finite_extremes_preserve_value_comparisons_and_input_bits() {
    let recording = recording();
    for vector in recording.vectors {
        let values: Vec<f64> = recording
            .source_pattern
            .iter()
            .cycle()
            .take(vector.length)
            .enumerate()
            .map(|(ordinal, &value)| match value {
                0 => -f64::MAX,
                1 if ordinal % 2 == 0 => -0.0,
                1 => 0.0,
                2 => f64::from_bits(1),
                _ => f64::MAX,
            })
            .collect();
        let original: Vec<u64> = values.iter().map(|value| value.to_bits()).collect();
        assert_eq!(scalar_argsort(&values), vector.order);
        assert_eq!(
            values.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
            original
        );
    }
}

#[test]
fn empty_singleton_and_insertion_only_inputs_keep_native_scalar_order() {
    assert!(scalar_argsort::<u64>(&[]).is_empty());
    assert_eq!(scalar_argsort(&[u64::MAX]), [0]);
    // NumPy's threshold permits insertion sort alone through length 16.
    // Equal values stay in input order only within this defined base case.
    assert_eq!(scalar_argsort(&[true, false, true, false]), [1, 3, 0, 2]);
    assert_eq!(scalar_argsort(&[4_i16; 16]), (0..16).collect::<Vec<_>>());
}
