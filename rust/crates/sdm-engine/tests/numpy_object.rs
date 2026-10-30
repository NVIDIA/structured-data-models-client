// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use sdm_contract::tabular::{
    NativeMissingOwner, NumpyScalarSemantics as Semantics, OrderedSampledTabularRows as Outcome,
    TabularIndexFacts, TabularIndexScalar as Atom, TabularIndexValues, TabularOrderError as Error,
    TabularSortProfile,
};
use sdm_engine::index_sort::order_sampled;

fn integer(signed: bool, width: u8, value: impl ToString) -> Atom {
    Atom::NumpyInteger {
        signed,
        width,
        value: value.to_string(),
    }
}
fn floating(width: u8, value: f64) -> Atom {
    Atom::NumpyFloating {
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
        values: TabularIndexValues::NumpyObject { values, semantics },
        missing,
        profile: TabularSortProfile::Scalar,
    }
}
fn ordered(values: Vec<Atom>, semantics: Semantics, positions: Vec<usize>, already_ordered: bool) {
    assert_eq!(
        order_sampled(&facts(values, semantics)).unwrap(),
        Outcome::Sorted {
            positions,
            already_ordered
        }
    );
}
const SEMANTICS: [Semantics; 3] = [
    Semantics::Numpy123Legacy,
    Semantics::Numpy126Legacy,
    Semantics::Numpy25Weak,
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
    ordered(pair.clone(), Semantics::Numpy123Legacy, vec![0, 1], true);
    for semantics in [Semantics::Numpy126Legacy, Semantics::Numpy25Weak] {
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
    ordered(pair.clone(), Semantics::Numpy25Weak, vec![0, 1], true);
    for semantics in [Semantics::Numpy123Legacy, Semantics::Numpy126Legacy] {
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
        order_sampled(&facts(pair.clone(), Semantics::Numpy25Weak)).unwrap(),
        Outcome::Refused {
            reason: Error::IntegerToFloatOverflow,
            left: 1,
            right: 0
        }
    );
    for semantics in [Semantics::Numpy123Legacy, Semantics::Numpy126Legacy] {
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
            order_sampled(&facts(values, Semantics::Numpy25Weak)).unwrap(),
            Outcome::Refused {
                reason: Error::NativeEffectsPending,
                left: 1,
                right: 0
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
        let mut index = facts(vec![Atom::NativeMissing { owner }], Semantics::Numpy25Weak);
        index.missing[0] = true;
        assert_eq!(
            order_sampled(&index).unwrap(),
            Outcome::Sorted {
                positions: vec![0],
                already_ordered: owner == NativeMissingOwner::None
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
        Semantics::Numpy25Weak,
    );
    index.missing = vec![false, true, false, true];
    assert_eq!(
        order_sampled(&index).unwrap(),
        Outcome::Sorted {
            positions: vec![2, 0, 1, 3],
            already_ordered: false
        }
    );
    let mut index = facts(
        vec![
            Atom::Integer {
                value: format!("1{}", "0".repeat(400)),
            },
            floating(8, f64::NAN),
        ],
        Semantics::Numpy25Weak,
    );
    index.missing[1] = true;
    assert_eq!(
        order_sampled(&index).unwrap(),
        Outcome::Refused {
            reason: Error::IntegerToFloatOverflow,
            left: 1,
            right: 0
        }
    );
}
#[test]
fn owner_ranges_and_unbound_profile_are_checked_before_any_order_claim() {
    for atom in [
        integer(true, 1, 128),
        integer(false, 1, -1),
        Atom::NumpyFloating {
            width: 2,
            bits: 65536,
        },
        Atom::Text { points: vec![97] },
    ] {
        assert!(order_sampled(&facts(vec![atom], Semantics::Numpy25Weak)).is_err());
    }
    let mut index = facts(
        vec![integer(true, 8, 1), integer(true, 8, 2)],
        Semantics::Numpy25Weak,
    );
    index.profile = TabularSortProfile::Unbound;
    assert_eq!(order_sampled(&index).unwrap(), Outcome::NeedsProfile);
}
