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
        TabularSortProfile::Avx512Gnu14,
        TabularSortProfile::Avx2Gnu14Unconditional,
        TabularSortProfile::Avx512Gnu14Unconditional,
        TabularSortProfile::Avx512Gnu14Network64,
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

#[test]
fn complex_components_sort_lexicographically_with_missing_labels_last() {
    for width in [4, 8] {
        let bits = |v: f64| {
            if width == 4 {
                u64::from((v as f32).to_bits())
            } else {
                v.to_bits()
            }
        };
        check(
            facts(
                TabularIndexValues::Complex {
                    real_bits: vec![bits(1.0), bits(0.0), bits(1.0), bits(f64::NAN)],
                    imag_bits: vec![bits(2.0), bits(-1.0), bits(-2.0), bits(0.0)],
                    width,
                },
                vec![false, false, false, true],
            ),
            vec![1, 2, 0, 3],
            false,
        );
    }
}

fn decimal(negative: bool, digits: &str, exponent: i64) -> TabularIndexScalar {
    TabularIndexScalar::Decimal {
        negative,
        digits: digits.into(),
        exponent,
    }
}
fn objects(values: Vec<TabularIndexScalar>) -> TabularIndexFacts {
    let mask = vec![false; values.len()];
    facts(TabularIndexValues::Object { values }, mask)
}
#[test]
fn decimal_scale_signed_zero_and_extreme_exponents_remain_exact() {
    check(
        objects(vec![
            decimal(false, "1", i64::MAX),
            decimal(false, "100", -2),
            TabularIndexScalar::Integer { value: "1".into() },
            decimal(true, "0", i64::MIN),
            decimal(false, "0", 0),
            decimal(true, "1", i64::MAX),
            decimal(false, "1", i64::MIN),
        ]),
        vec![5, 3, 4, 6, 1, 2, 0],
        false,
    );
}
#[test]
fn fixed_offsets_compare_adjusted_time_and_naive_values_keep_civil_order() {
    let datetime = |ordinal, micros, offset_micros| TabularIndexScalar::DateTime {
        ordinal,
        micros,
        offset_micros,
    };
    check(
        objects(vec![
            datetime(10, 0, Some(0)),
            datetime(10, 3_600_000_000, Some(7_200_000_000)),
            datetime(9, 82_800_000_000, Some(0)),
        ]),
        vec![1, 2, 0],
        false,
    );
    check(
        objects(vec![datetime(10, 0, None), datetime(9, 0, None)]),
        vec![1, 0],
        false,
    );
    check(
        objects(vec![
            TabularIndexScalar::Date { ordinal: 10 },
            TabularIndexScalar::Date { ordinal: 9 },
        ]),
        vec![1, 0],
        false,
    );
}
#[test]
fn object_failure_reasons_keep_first_reached_comparison_owners() {
    use sdm_contract::tabular::TabularOrderError;
    let datetime = |offset_micros| TabularIndexScalar::DateTime {
        ordinal: 10,
        micros: 0,
        offset_micros,
    };
    for (values, reason) in [
        (
            vec![datetime(None), datetime(Some(0))],
            TabularOrderError::NaiveAwareDatetime,
        ),
        (
            vec![TabularIndexScalar::Date { ordinal: 10 }, datetime(None)],
            TabularOrderError::DateTimeMismatch,
        ),
        (
            vec![
                TabularIndexScalar::Complex {
                    real_bits: 0,
                    imag_bits: 0
                };
                2
            ],
            TabularOrderError::Incomparable,
        ),
    ] {
        assert_eq!(
            order_sampled(&objects(values)).unwrap(),
            OrderedSampledTabularRows::Refused {
                reason,
                left: 1,
                right: 0,
                levels: Vec::new(),
            }
        );
    }
}
#[test]
fn malformed_civil_and_decimal_facts_are_protocol_errors() {
    for value in [
        TabularIndexScalar::Date { ordinal: 0 },
        TabularIndexScalar::DateTime {
            ordinal: 1,
            micros: 86_400_000_000,
            offset_micros: None,
        },
        TabularIndexScalar::DateTime {
            ordinal: 1,
            micros: 0,
            offset_micros: Some(86_400_000_000),
        },
        decimal(false, "01", 0),
    ] {
        assert!(order_sampled(&objects(vec![value])).is_err());
    }
}

fn unbound(mut facts: TabularIndexFacts) -> TabularIndexFacts {
    facts.profile = TabularSortProfile::Unbound;
    facts
}
#[test]
fn unbound_unique_keys_finish_but_unsorted_ties_require_binding() {
    check(
        unbound(facts(
            TabularIndexValues::Signed {
                values: vec![3, 1, 2],
                width: 8,
            },
            vec![false; 3],
        )),
        vec![1, 2, 0],
        false,
    );
    check(
        unbound(facts(
            TabularIndexValues::Signed {
                values: vec![1, 1, 2],
                width: 8,
            },
            vec![false; 3],
        )),
        vec![0, 1, 2],
        true,
    );
    assert_eq!(
        order_sampled(&unbound(facts(
            TabularIndexValues::Signed {
                values: vec![2, 1, 1],
                width: 8
            },
            vec![false; 3]
        )))
        .unwrap(),
        OrderedSampledTabularRows::NeedsProfile
    );
}
#[test]
fn unbound_missing_suffix_needs_no_profile_but_present_nan_ties_do() {
    check(
        unbound(facts(
            TabularIndexValues::Floating {
                bits: vec![f64::NAN.to_bits()],
                width: 8,
            },
            vec![false],
        )),
        vec![0],
        false,
    );

    check(
        unbound(facts(
            TabularIndexValues::Floating {
                bits: vec![0, 2_f64.to_bits(), 0, 1_f64.to_bits()],
                width: 8,
            },
            vec![true, false, true, false],
        )),
        vec![3, 1, 0, 2],
        false,
    );
    check(
        unbound(facts(
            TabularIndexValues::Floating {
                bits: vec![f64::NAN.to_bits(), 1_f64.to_bits()],
                width: 8,
            },
            vec![false; 2],
        )),
        vec![1, 0],
        false,
    );
    assert_eq!(
        order_sampled(&unbound(facts(
            TabularIndexValues::Floating {
                bits: vec![f64::NAN.to_bits(); 2],
                width: 8
            },
            vec![false; 2]
        )))
        .unwrap(),
        OrderedSampledTabularRows::NeedsProfile
    );
}
#[test]
fn unbound_objects_admit_only_structurally_pure_total_families() {
    check(
        unbound(objects(vec![
            decimal(false, "2", 0),
            TabularIndexScalar::Integer { value: "1".into() },
        ])),
        vec![1, 0],
        false,
    );
    for values in [
        vec![
            TabularIndexScalar::Text { points: vec![97] },
            TabularIndexScalar::Integer { value: "1".into() },
        ],
        vec![TabularIndexScalar::Complex {
            real_bits: 0,
            imag_bits: 0,
        }],
        vec![
            TabularIndexScalar::DateTime {
                ordinal: 1,
                micros: 0,
                offset_micros: None,
            },
            TabularIndexScalar::DateTime {
                ordinal: 1,
                micros: 0,
                offset_micros: Some(0),
            },
        ],
    ] {
        assert_eq!(
            order_sampled(&unbound(objects(values))).unwrap(),
            OrderedSampledTabularRows::NeedsProfile
        );
    }
}
#[test]
fn stable_ties_and_multiindex_children_preserve_their_existing_profiles() {
    let mut stable = facts(
        TabularIndexValues::Signed {
            values: vec![2, 1, 1],
            width: 8,
        },
        vec![false; 3],
    );
    stable.profile = TabularSortProfile::Stable;
    check(stable, vec![1, 2, 0], false);
    let level = unbound(facts(
        TabularIndexValues::Signed {
            values: vec![2, 1],
            width: 8,
        },
        vec![false; 2],
    ));
    check(
        facts(
            TabularIndexValues::Multi {
                levels: vec![level],
                codes: vec![vec![0, 1, 1]],
            },
            vec![false; 3],
        ),
        vec![1, 2, 0],
        false,
    );
    let level = unbound(objects(vec![
        TabularIndexScalar::Text { points: vec![97] },
        TabularIndexScalar::Integer { value: "1".into() },
    ]));
    assert_eq!(
        order_sampled(&facts(
            TabularIndexValues::Multi {
                levels: vec![level],
                codes: vec![vec![0, 1]]
            },
            vec![false; 2]
        ))
        .unwrap(),
        OrderedSampledTabularRows::NeedsProfile
    );
}

#[test]
fn unbound_validation_precedes_monotonic_shortcuts() {
    assert!(
        order_sampled(&unbound(facts(
            TabularIndexValues::Complex {
                real_bits: vec![],
                imag_bits: vec![],
                width: 3
            },
            vec![]
        )))
        .is_err()
    );
    assert!(
        order_sampled(&unbound(facts(
            TabularIndexValues::Signed {
                values: vec![1],
                width: 8
            },
            vec![]
        )))
        .is_err()
    );
}

#[test]
fn unconditional_simd_entry_does_not_disable_outer_index_monotonic_shortcut() {
    for profile in [
        TabularSortProfile::Avx2Gnu14Unconditional,
        TabularSortProfile::Avx512Gnu14Unconditional,
        TabularSortProfile::Avx512Gnu14Network64,
    ] {
        let mut index = facts(
            TabularIndexValues::Signed {
                values: vec![7; 320],
                width: 8,
            },
            vec![false; 320],
        );
        index.profile = profile;
        check(index.clone(), (0..320).collect(), true);
        index.missing[0] = true;
        let OrderedSampledTabularRows::Sorted {
            positions,
            already_ordered,
        } = order_sampled(&index).unwrap()
        else {
            panic!("expected ordered positions")
        };
        assert!(!already_ordered);
        assert_eq!(positions.last(), Some(&0));
        assert_ne!(&positions[..319], &(1..320).collect::<Vec<_>>());
    }
}
