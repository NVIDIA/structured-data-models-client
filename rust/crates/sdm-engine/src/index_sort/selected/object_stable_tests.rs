// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{object_stable, temporal};
use sdm_contract::tabular::TabularIndexScalar;
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    atoms: Vec<TabularIndexScalar>,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    codes: Vec<usize>,
    positions: Vec<usize>,
}

#[test]
fn native_stable_vectors_preserve_non_total_temporal_order() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata/temporal-stable.json");
    let fixture: Fixture = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    for case in fixture.cases {
        let mut positions: Vec<_> = (0..case.codes.len()).collect();
        object_stable::sort(&mut positions, |left, right| {
            temporal::sort_comparison(
                &fixture.atoms[case.codes[left]],
                &fixture.atoms[case.codes[right]],
            )
        })
        .unwrap();
        assert_eq!(positions, case.positions, "{}", case.name);
    }
}
