// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use sdm_contract::tabular::{
    OrderedSampledTabularRows, TabularIndexFacts, TabularIndexScalar, TabularIndexValues,
    TabularSortProfile,
};
use sdm_engine::index_sort::order_sampled;

fn facts(values: TabularIndexValues, missing: Vec<bool>) -> TabularIndexFacts {
    TabularIndexFacts {
        values,
        missing,
        profile: TabularSortProfile::Scalar,
    }
}
fn check(index: TabularIndexFacts, positions: Vec<usize>, already_ordered: bool) {
    assert_eq!(
        order_sampled(&index).unwrap(),
        OrderedSampledTabularRows::Sorted {
            positions,
            already_ordered
        }
    );
}
#[test]
fn missing_partition_does_not_turn_real_sort_into_monotonic_shortcut() {
    check(
        facts(
            TabularIndexValues::Signed {
                values: vec![1, 1, 0],
                width: 8,
            },
            vec![false, false, true],
        ),
        vec![0, 1, 2],
        false,
    );
    check(
        facts(
            TabularIndexValues::Signed {
                values: vec![1, 1, 2],
                width: 8,
            },
            vec![false; 3],
        ),
        vec![0, 1, 2],
        true,
    );
    check(
        facts(
            TabularIndexValues::Signed {
                values: vec![0, 2, 1, 0],
                width: 8,
            },
            vec![true, false, false, true],
        ),
        vec![2, 1, 0, 3],
        false,
    );
}
#[test]
fn categorical_codes_and_unicode_keep_subset_order() {
    check(
        facts(
            TabularIndexValues::Categorical {
                codes: vec![2, -1, 0, 2, 1],
                width: 1,
            },
            vec![false, true, false, false, false],
        ),
        vec![2, 4, 0, 3, 1],
        false,
    );
    check(
        facts(
            TabularIndexValues::Unicode {
                values: vec![vec![0x10000], vec![0xd800], vec![97]],
            },
            vec![false; 3],
        ),
        vec![2, 1, 0],
        false,
    );
}
#[test]
fn multiindex_normalizes_levels_before_stable_lexicographic_sort() {
    let levels = vec![
        facts(
            TabularIndexValues::Signed {
                values: vec![20, 10],
                width: 8,
            },
            vec![false; 2],
        ),
        facts(
            TabularIndexValues::Unicode {
                values: vec![vec![98], vec![97]],
            },
            vec![false; 2],
        ),
    ];
    check(
        facts(
            TabularIndexValues::Multi {
                levels,
                codes: vec![vec![0, 1, 1, 1, -1], vec![0, 0, 1, 1, 0]],
            },
            vec![false; 5],
        ),
        vec![2, 3, 1, 0, 4],
        false,
    );
}
#[test]
fn built_in_object_numbers_keep_exact_large_integer_comparisons() {
    let values = vec![
        TabularIndexScalar::Integer {
            value: "9007199254740993".into(),
        },
        TabularIndexScalar::Floating {
            bits: 9007199254740992.0_f64.to_bits(),
        },
        TabularIndexScalar::Boolean { value: true },
    ];
    check(
        facts(TabularIndexValues::Object { values }, vec![false; 3]),
        vec![2, 1, 0],
        false,
    );
}
#[test]
fn incomparable_objects_report_first_native_comparison_owners() {
    let values = vec![
        TabularIndexScalar::Integer { value: "1".into() },
        TabularIndexScalar::Text { points: vec![97] },
    ];
    assert!(matches!(
        order_sampled(&facts(
            TabularIndexValues::Object { values },
            vec![false; 2]
        ))
        .unwrap(),
        OrderedSampledTabularRows::Refused {
            left: 1,
            right: 0,
            ..
        }
    ));
}
#[test]
fn present_nan_sorts_before_masked_null() {
    let bits = vec![f64::NAN.to_bits(), 2.0_f64.to_bits(), 0, 1.0_f64.to_bits()];
    for profile in [
        TabularSortProfile::Scalar,
        TabularSortProfile::Avx2Gnu14,
        TabularSortProfile::Stable,
    ] {
        let mut index = facts(
            TabularIndexValues::Floating {
                bits: bits.clone(),
                width: 8,
            },
            vec![false, false, true, false],
        );
        index.profile = profile;
        check(index, vec![3, 1, 0, 2], false);
    }
}
