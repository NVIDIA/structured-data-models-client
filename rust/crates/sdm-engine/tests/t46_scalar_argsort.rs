// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Exact scalar tie permutations, not a selection of the native SIMD route.

use sdm_engine::sample::{introsort_argsort, scalar_argsort};

struct Recording {
    source_pattern: Vec<u8>,
    vectors: Vec<Vector>,
}

struct Vector {
    length: usize,
    order: Vec<usize>,
}

fn recording() -> Recording {
    let data: serde_json::Value =
        serde_json::from_str(include_str!("data/t46_scalar_argsort.json")).unwrap();
    Recording {
        source_pattern: serde_json::from_value(data["source_pattern"].clone()).unwrap(),
        vectors: data["vectors"]
            .as_array()
            .unwrap()
            .iter()
            .map(|vector| Vector {
                length: vector["length"].as_u64().unwrap().try_into().unwrap(),
                order: serde_json::from_value(vector["order"].clone()).unwrap(),
            })
            .collect(),
    }
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
            values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
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

#[test]
fn all_native_scalar_threshold_sentinel_and_width_controls_match_exactly() {
    let data: serde_json::Value =
        serde_json::from_str(include_str!("data/t46_scalar_kernel_controls.json")).unwrap();
    let records = data["records"].as_array().unwrap();
    assert_eq!(records.len(), 184);
    for record in records {
        let name = record["case"].as_str().unwrap();
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
                assert_eq!(scalar_argsort(&values), expected, "{name}");
            }};
        }
        match record["dtype"].as_str().unwrap() {
            "|i1" => check!(i8, 1),
            "|u1" => check!(u8, 1),
            "<i2" => check!(i16, 2),
            "<u2" => check!(u16, 2),
            "<i4" => check!(i32, 4),
            "<u4" => check!(u32, 4),
            "<i8" => check!(i64, 8),
            "<u8" => check!(u64, 8),
            "<f4" => check!(f32, 4),
            "<f8" => check!(f64, 8),
            "|b1" => {
                assert!(bytes.iter().all(|value| *value <= 1), "{name}");
                let values: Vec<bool> = bytes.iter().map(|value| *value != 0).collect();
                assert_eq!(scalar_argsort(&values), expected, "{name}");
            }
            dtype => panic!("unexpected recorded scalar dtype {dtype}"),
        }
    }
}
