// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use sdm_contract::tabular::{
    NativeMissingOwner, NumericComparisonProfile as Semantics,
    OrderedSampledTabularRows as Outcome, TabularIndexFacts, TabularIndexScalar as Atom,
    TabularIndexValues, TabularOrderError as Error, TabularSortProfile,
};
use sdm_engine::index_sort::order_sampled;

fn integer(signed: bool, width: u8, value: impl ToString) -> Atom {
    Atom::FixedInteger {
        signed,
        width,
        value: value.to_string(),
    }
}
fn floating(width: u8, value: f64) -> Atom {
    Atom::FixedFloating {
        width,
        bits: match width {
            2 => u64::from(half::f16::from_f64(value).to_bits()),
            4 => u64::from((value as f32).to_bits()),
            8 => value.to_bits(),
            _ => unreachable!(),
        },
    }
}
fn facts(values: Vec<Atom>, semantics: Semantics) -> TabularIndexFacts {
    let missing = vec![false; values.len()];
    TabularIndexFacts {
        values: TabularIndexValues::ScalarObject {
            values,
            semantics,
            overflow: None,
        },
        missing,
        profile: TabularSortProfile::Scalar,
    }
}
fn ordered(values: Vec<Atom>, semantics: Semantics, positions: Vec<usize>, already_ordered: bool) {
    assert_eq!(
        order_sampled(&facts(values, semantics)).unwrap(),
        Outcome::Sorted {
            positions,
            already_ordered,
            overflow_warnings: Vec::new(),
            overflow_prints: 0,
        }
    );
}
const SEMANTICS: [Semantics; 3] = [
    Semantics::CommonType,
    Semantics::ExactIntegerPairs,
    Semantics::WeakScalars,
];

#[test]
fn all_declared_widths_retain_native_value_and_direction() {
    for semantics in SEMANTICS {
        for width in [1, 2, 4, 8] {
            for signed in [false, true] {
                ordered(
                    vec![integer(signed, width, 3), integer(signed, width, 1)],
                    semantics,
                    vec![1, 0],
                    false,
                );
                ordered(
                    vec![integer(signed, width, 1), integer(signed, width, 3)],
                    semantics,
                    vec![0, 1],
                    true,
                );
            }
        }
        for width in [2, 4, 8] {
            ordered(
                vec![floating(width, 3.0), floating(width, 1.0)],
                semantics,
                vec![1, 0],
                false,
            );
        }
    }
}
#[test]
fn signed_unsigned_boundary_uses_the_selected_comparison_semantics() {
    let pair = vec![integer(false, 8, 1_u64 << 63), integer(true, 8, i64::MAX)];
    ordered(pair.clone(), Semantics::CommonType, vec![0, 1], true);
    for semantics in [Semantics::ExactIntegerPairs, Semantics::WeakScalars] {
        ordered(pair.clone(), semantics, vec![1, 0], false);
    }
}
#[test]
fn weak_python_float_rounding_differs_from_legacy_promotion() {
    let pair = vec![
        Atom::Floating {
            bits: 16_777_217_f64.to_bits(),
        },
        floating(4, 16_777_216.0),
    ];
    ordered(pair.clone(), Semantics::WeakScalars, vec![0, 1], true);
    for semantics in [Semantics::CommonType, Semantics::ExactIntegerPairs] {
        ordered(pair.clone(), semantics, vec![1, 0], false);
    }
}
#[test]
fn huge_integer_overflow_is_reached_during_monotonic_probe_only_in_weak_mode() {
    let huge = Atom::Integer {
        value: format!("1{}", "0".repeat(400)),
    };
    let pair = vec![huge.clone(), floating(8, 1.0)];
    assert_eq!(
        order_sampled(&facts(pair.clone(), Semantics::WeakScalars)).unwrap(),
        Outcome::Refused {
            reason: Error::IntegerToFloatOverflow,
            left: 1,
            right: 0,
            levels: Vec::new(),
            overflow_warnings: Vec::new(),
            overflow_prints: 0,
            integer_destination: None,
        }
    );
    for semantics in [Semantics::CommonType, Semantics::ExactIntegerPairs] {
        ordered(pair.clone(), semantics, vec![1, 0], false);
    }
    for semantics in SEMANTICS {
        ordered(
            vec![huge.clone(), integer(true, 8, 1)],
            semantics,
            vec![1, 0],
            false,
        );
    }
}
#[test]
fn finite_narrowing_overflow_stays_an_explicit_pending_native_effect() {
    for width in [2, 4] {
        let values = vec![
            Atom::Floating {
                bits: 1e100_f64.to_bits(),
            },
            floating(width, 1.0),
        ];
        assert_eq!(
            order_sampled(&facts(values, Semantics::WeakScalars)).unwrap(),
            Outcome::Refused {
                reason: Error::NativeEffectsPending,
                left: 1,
                right: 0,
                levels: Vec::new(),
                overflow_warnings: Vec::new(),
                overflow_prints: 0,
                integer_destination: None,
            }
        );
    }
}
#[test]
fn masked_owners_keep_distinct_probe_behavior_and_original_suffix_order() {
    for owner in [
        NativeMissingOwner::None,
        NativeMissingOwner::Ambiguous,
        NativeMissingOwner::Temporal,
    ] {
        let mut index = facts(vec![Atom::NativeMissing { owner }], Semantics::WeakScalars);
        index.missing[0] = true;
        assert_eq!(
            order_sampled(&index).unwrap(),
            Outcome::Sorted {
                positions: vec![0],
                already_ordered: owner == NativeMissingOwner::None,
                overflow_warnings: Vec::new(),
                overflow_prints: 0,
            }
        );
    }
    let mut index = facts(
        vec![
            integer(true, 8, 2),
            Atom::NativeMissing {
                owner: NativeMissingOwner::None,
            },
            integer(true, 8, 1),
            floating(8, f64::NAN),
        ],
        Semantics::WeakScalars,
    );
    index.missing = vec![false, true, false, true];
    assert_eq!(
        order_sampled(&index).unwrap(),
        Outcome::Sorted {
            positions: vec![2, 0, 1, 3],
            already_ordered: false,
            overflow_warnings: Vec::new(),
            overflow_prints: 0,
        }
    );
    let mut index = facts(
        vec![
            Atom::Integer {
                value: format!("1{}", "0".repeat(400)),
            },
            floating(8, f64::NAN),
        ],
        Semantics::WeakScalars,
    );
    index.missing[1] = true;
    assert_eq!(
        order_sampled(&index).unwrap(),
        Outcome::Refused {
            reason: Error::IntegerToFloatOverflow,
            left: 1,
            right: 0,
            levels: Vec::new(),
            overflow_warnings: Vec::new(),
            overflow_prints: 0,
            integer_destination: None,
        }
    );
}
#[test]
fn owner_ranges_and_unbound_profile_are_checked_before_any_order_claim() {
    for atom in [
        integer(true, 1, 128),
        integer(false, 1, -1),
        Atom::FixedFloating {
            width: 2,
            bits: 65536,
        },
        Atom::Text { points: vec![97] },
    ] {
        assert!(order_sampled(&facts(vec![atom], Semantics::WeakScalars)).is_err());
    }
    let mut index = facts(
        vec![integer(true, 8, 1), integer(true, 8, 2)],
        Semantics::WeakScalars,
    );
    index.profile = TabularSortProfile::Unbound;
    assert_eq!(order_sampled(&index).unwrap(), Outcome::NeedsProfile);
}

fn configured(
    values: Vec<Atom>,
    semantics: Semantics,
    overflow: sdm_contract::tabular::NumericOverflowPolicy,
) -> TabularIndexFacts {
    let mut index = facts(values, semantics);
    let TabularIndexValues::ScalarObject {
        overflow: setting, ..
    } = &mut index.values
    else {
        unreachable!()
    };
    *setting = Some(overflow);
    index
}
#[test]
fn ordinary_overflow_modes_keep_probe_diagnostics_and_terminal_error() {
    use sdm_contract::tabular::{NumericOverflowPolicy as Policy, NumericWarningStage};
    for policy in [Policy::Ignore, Policy::Raise, Policy::Warn, Policy::Print] {
        let index = configured(
            vec![
                floating(4, 1.0),
                Atom::Floating {
                    bits: 1e100_f64.to_bits(),
                },
            ],
            Semantics::WeakScalars,
            policy,
        );
        let result = order_sampled(&index).unwrap();
        if policy == Policy::Raise {
            assert!(matches!(
                result,
                Outcome::Refused {
                    reason: Error::FloatingOverflow,
                    left: 1,
                    right: 0,
                    ..
                }
            ));
        } else {
            let Outcome::Sorted {
                positions,
                already_ordered,
                overflow_warnings,
                overflow_prints,
            } = result
            else {
                panic!("ordinary overflow must finish")
            };
            assert_eq!(positions, vec![0, 1]);
            assert!(already_ordered);
            assert_eq!(overflow_prints, if policy == Policy::Print { 2 } else { 0 });
            if policy == Policy::Warn {
                assert_eq!(
                    overflow_warnings,
                    vec![sdm_contract::tabular::NumericOverflowWarnings {
                        stage: NumericWarningStage::Probe,
                        count: 2
                    }]
                );
            } else {
                assert!(overflow_warnings.is_empty());
            }
        }
    }
}
#[test]
fn weak_legacy_integer_bounds_name_the_actual_unbounded_operand() {
    use sdm_contract::tabular::{IntegerDestination, NumericOverflowPolicy};
    for (signed, value) in [(true, "128"), (false, "-1")] {
        for reversed in [false, true] {
            let mut values = vec![
                integer(signed, 1, 1),
                Atom::Integer {
                    value: value.into(),
                },
            ];
            if reversed {
                values.reverse();
            }
            let index = configured(
                values,
                Semantics::WeakFixedIntegerPairs,
                NumericOverflowPolicy::Ignore,
            );
            let Outcome::Refused {
                reason,
                left,
                integer_destination,
                ..
            } = order_sampled(&index).unwrap()
            else {
                panic!("bounds failure expected")
            };
            assert_eq!(reason, Error::IntegerOutOfBounds);
            assert_eq!(left, usize::from(!reversed));
            assert_eq!(
                integer_destination,
                Some(IntegerDestination { signed, width: 1 })
            );
        }
    }
}
#[test]
fn leading_nan_does_not_reach_a_later_huge_integer_conversion() {
    use sdm_contract::tabular::NumericOverflowPolicy;
    let mut index = configured(
        vec![
            floating(4, f64::NAN),
            Atom::Integer {
                value: format!("1{}", "0".repeat(400)),
            },
        ],
        Semantics::WeakScalars,
        NumericOverflowPolicy::Raise,
    );
    index.missing[0] = true;
    assert!(
        matches!(order_sampled(&index).unwrap(),Outcome::Sorted {positions,already_ordered:false,..} if positions==vec![1,0])
    );
}
#[test]
fn a_terminal_conversion_error_keeps_prior_overflow_warnings() {
    use sdm_contract::tabular::NumericOverflowPolicy;
    let index = configured(
        vec![
            floating(4, 2.0),
            Atom::Floating {
                bits: 1e100_f64.to_bits(),
            },
            floating(4, 1.0),
            Atom::Integer {
                value: format!("1{}", "0".repeat(400)),
            },
        ],
        Semantics::WeakScalars,
        NumericOverflowPolicy::Warn,
    );
    let Outcome::Refused {
        reason,
        overflow_warnings,
        ..
    } = order_sampled(&index).unwrap()
    else {
        panic!("huge integer failure expected")
    };
    assert_eq!(reason, Error::IntegerToFloatOverflow);
    assert!(!overflow_warnings.is_empty());
}

#[test]
fn intermediate_integer_overflow_precedes_destination_bounds() {
    use sdm_contract::tabular::NumericOverflowPolicy;
    for (signed, width) in [(true, 1), (true, 8), (false, 1), (false, 8)] {
        let index = configured(
            vec![
                integer(signed, width, 1),
                Atom::Integer {
                    value: "18446744073709551616".into(),
                },
            ],
            Semantics::WeakFixedIntegerPairs,
            NumericOverflowPolicy::Ignore,
        );
        assert!(matches!(
            order_sampled(&index).unwrap(),
            Outcome::Refused {
                reason: Error::IntegerIntermediateOverflow,
                left: 1,
                ..
            }
        ));
    }
}
#[test]
fn earlier_multi_level_diagnostics_survive_a_later_numeric_failure() {
    use sdm_contract::tabular::{NumericOverflowPolicy, NumericWarningStage};
    let first = configured(
        vec![
            floating(4, 1.0),
            Atom::Floating {
                bits: 1e100_f64.to_bits(),
            },
        ],
        Semantics::WeakScalars,
        NumericOverflowPolicy::Warn,
    );
    let second = configured(
        vec![
            floating(4, 1.0),
            Atom::Integer {
                value: format!("1{}", "0".repeat(400)),
            },
        ],
        Semantics::WeakScalars,
        NumericOverflowPolicy::Warn,
    );
    let index = TabularIndexFacts {
        values: TabularIndexValues::Multi {
            levels: vec![first, second],
            codes: vec![vec![0, 1], vec![0, 1]],
        },
        missing: vec![false, false],
        profile: TabularSortProfile::Scalar,
    };
    let Outcome::Refused {
        reason,
        levels,
        overflow_warnings,
        ..
    } = order_sampled(&index).unwrap()
    else {
        panic!("later numeric failure expected")
    };
    assert_eq!(reason, Error::IntegerToFloatOverflow);
    assert_eq!(levels, vec![1]);
    assert_eq!(
        overflow_warnings,
        vec![sdm_contract::tabular::NumericOverflowWarnings {
            stage: NumericWarningStage::Probe,
            count: 2
        }]
    );
}
