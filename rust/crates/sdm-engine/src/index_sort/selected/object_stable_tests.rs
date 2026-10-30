// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{object_stable, temporal};
use sdm_contract::tabular::TabularIndexScalar;

#[test]
fn native_stable_vectors_preserve_non_total_temporal_order() {
    let fixture: serde_json::Value =
        serde_json::from_str(super::object_stable_fixture::DATA).unwrap();
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
