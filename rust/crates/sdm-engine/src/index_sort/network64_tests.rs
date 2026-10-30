// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{SimdArgsortProfile, network, simd_argsort, sort_recursive};

const PROFILE: SimdArgsortProfile = SimdArgsortProfile::Avx512Gnu14Network64;

#[test]
fn independently_transcribed_network_vectors_match() {
    // Source-derived expectations, not native AVX512 execution or a candidate oracle.
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/legacy_network64_controls.json");
    let receipt: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let cases = receipt["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 40);
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let mut indices: Vec<usize> = serde_json::from_value(case["indices"].clone()).unwrap();
        let expected: Vec<usize> = serde_json::from_value(case["order"].clone()).unwrap();
        let keys = case["keys"].as_array().unwrap();
        if name.starts_with("infinity/") {
            let values: Vec<f64> = keys
                .iter()
                .map(|v| v.as_str().unwrap().parse().unwrap())
                .collect();
            network(&values, &mut indices, PROFILE);
            assert_eq!(simd_argsort(&values, PROFILE), expected, "entry: {name}");
        } else {
            let values: Vec<i64> = keys
                .iter()
                .map(|v| v.as_str().unwrap().parse().unwrap())
                .collect();
            network(&values, &mut indices, PROFILE);
            if !name.starts_with("absolute_zero_subrange/") {
                assert_eq!(simd_argsort(&values, PROFILE), expected, "entry: {name}");
            }
        }
        // Sentinel cases deliberately check source output, not permutation invariants.
        assert_eq!(indices, expected, "network: {name}");
    }
}

#[test]
fn recursive_and_partition_boundaries_preserve_ordinary_keys() {
    // Invariants supplement the independent network vectors; not a native receipt.
    for n in [0, 1, 63, 64, 65, 255, 256, 257, 513] {
        let values: Vec<i64> = (0..n).map(|i| ((i * 17 + 3) % 11) as i64).collect();
        let order = simd_argsort(&values, PROFILE);
        assert!(order.windows(2).all(|w| values[w[0]] <= values[w[1]]));
        let mut seen = order;
        seen.sort_unstable();
        assert_eq!(seen, (0..n).collect::<Vec<_>>());
    }
}

#[test]
fn exhausted_depth_reaches_fallback_before_small_network() {
    let values = [7_i64, 1, 4, 0, 3, 6, 2, 5];
    let mut indices: Vec<_> = (0..values.len()).collect();
    sort_recursive(&values, &mut indices, 0, PROFILE);
    assert_eq!(indices, [3, 1, 6, 4, 2, 7, 5, 0]);
}
