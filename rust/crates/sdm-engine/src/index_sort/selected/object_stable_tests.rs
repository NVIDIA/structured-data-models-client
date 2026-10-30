// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{object_stable, temporal};
use sdm_contract::tabular::TabularIndexScalar;

#[test]
#[expect(
    clippy::disallowed_methods,
    reason = "test-only fixture loading; production core performs no I/O"
)]
fn native_stable_vectors_preserve_non_total_temporal_order() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata/temporal-stable.json");
    let fixture: serde_json::Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let atoms: Vec<TabularIndexScalar> = serde_json::from_value(fixture["atoms"].clone()).unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let codes: Vec<usize> = serde_json::from_value(case["codes"].clone()).unwrap();
        let expected: Vec<usize> = serde_json::from_value(case["positions"].clone()).unwrap();
        let mut positions: Vec<_> = (0..codes.len()).collect();
        object_stable::sort(&mut positions, |left, right| {
            temporal::sort_comparison(&atoms[codes[left]], &atoms[codes[right]])
        })
        .unwrap();
        assert_eq!(positions, expected, "{}", case["name"]);
    }
}
