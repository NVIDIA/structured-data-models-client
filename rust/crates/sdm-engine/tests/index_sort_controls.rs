// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native AVX2 permutations acquired independently before this implementation.

use sdm_engine::index_sort::{SimdArgsortProfile, simd_argsort};

#[test]
fn all_recorded_native_avx2_threshold_sentinel_and_partition_controls_match() {
    let data: serde_json::Value =
        serde_json::from_str(include_str!("data/t46_simd_kernel_controls.json")).unwrap();
    assert_eq!(data["source_numpy"], "2.5.3");
    let records = data["records"].as_array().unwrap();
    assert_eq!(records.len(), 129);
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
                assert_eq!(
                    simd_argsort(&values, SimdArgsortProfile::Avx2Gnu14),
                    expected,
                    "{name}"
                );
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
            dtype => panic!("unexpected recorded SIMD dtype {dtype}"),
        }
    }
}

#[test]
fn ordered_inputs_keep_the_entrypoint_shortcut_for_both_lane_profiles() {
    for profile in [
        SimdArgsortProfile::Avx2Gnu14,
        SimdArgsortProfile::Avx512Gnu14,
    ] {
        for n in [0, 1, 3, 4, 5, 7, 8, 9, 255, 256, 257, 513] {
            let values: Vec<u64> = (0..n).map(|i| u64::MAX - n as u64 + i as u64).collect();
            assert_eq!(simd_argsort(&values, profile), (0..n).collect::<Vec<_>>());
            assert_eq!(
                simd_argsort(&vec![u64::MAX; n], profile),
                (0..n).collect::<Vec<_>>()
            );
        }
    }
}

#[test]
fn eight_lane_source_path_orders_values_and_retains_every_original_index() {
    // This is an invariant check, not a native AVX512 equivalence receipt.
    for n in [3, 4, 7, 8, 9, 15, 16, 17, 255, 256, 257, 513, 1025] {
        let values: Vec<u64> = (0..n)
            .map(|i| u64::MAX - ((i * 17 + 3) % 7) as u64)
            .collect();
        let order = simd_argsort(&values, SimdArgsortProfile::Avx512Gnu14);
        assert!(
            order
                .windows(2)
                .all(|pair| values[pair[0]] <= values[pair[1]])
        );
        let mut seen = order;
        seen.sort_unstable();
        assert_eq!(seen, (0..n).collect::<Vec<_>>());
    }
}

use sdm_engine::index_sort::scalar_argsort;

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
                assert!(bytes.len().is_multiple_of($width), "{name}");
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

#[test]
fn unconditional_entry_preserves_lanes_but_reorders_partition_sized_sorted_ties() {
    for (modern, unconditional) in [
        (
            SimdArgsortProfile::Avx2Gnu14,
            SimdArgsortProfile::Avx2Gnu14Unconditional,
        ),
        (
            SimdArgsortProfile::Avx512Gnu14,
            SimdArgsortProfile::Avx512Gnu14Unconditional,
        ),
    ] {
        for constant in [false, true] {
            let values: Vec<i64> = (0..319).map(|i| if constant { 7 } else { i / 9 }).collect();
            let identity: Vec<_> = (0..values.len()).collect();
            assert_eq!(simd_argsort(&values, modern), identity);
            let order = simd_argsort(&values, unconditional);
            assert_ne!(order, identity);
            assert!(
                order
                    .windows(2)
                    .all(|pair| values[pair[0]] <= values[pair[1]])
            );
            let mut seen = order;
            seen.sort_unstable();
            assert_eq!(seen, identity);
        }
        // Empty/singleton handling precedes recursion and its logarithm.
        assert_eq!(simd_argsort::<i64>(&[], unconditional), Vec::<usize>::new());
        assert_eq!(simd_argsort(&[7_i64], unconditional), vec![0]);
        for n in [2, 3, 4, 7, 8, 9, 15, 16, 17, 255, 256, 257, 319] {
            let values: Vec<i64> = (0..n).map(|i| ((i * 17 + 3) % 11) as i64).collect();
            // Unsorted entry takes identical lane/store/partition paths.
            assert_eq!(
                simd_argsort(&values, unconditional),
                simd_argsort(&values, modern)
            );
        }
    }
}
