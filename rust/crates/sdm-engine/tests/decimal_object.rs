// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use sdm_contract::tabular::{
    NumericDiagnostic as Event, OrderedSampledTabularRows as Outcome, TabularIndexFacts as Facts,
    TabularIndexScalar as Atom, TabularIndexValues as Values, TabularOrderError as Error,
    TabularSortProfile as Profile,
};
use sdm_engine::index_sort::order_sampled;
fn decimal(digits: &str, exponent: i64) -> Atom {
    Atom::Decimal {
        negative: false,
        digits: digits.into(),
        exponent,
    }
}
fn floating(value: f64) -> Atom {
    Atom::Floating {
        bits: value.to_bits(),
    }
}
fn facts(values: Vec<Atom>, float_trap: bool, invalid_trap: bool, catches_invalid: bool) -> Facts {
    Facts {
        missing: vec![false; values.len()],
        profile: Profile::Scalar,
        values: Values::DecimalObject {
            values,
            trap_float_operation: float_trap,
            trap_invalid_operation: invalid_trap,
            probe_catches_invalid_operation: catches_invalid,
        },
    }
}
#[test]
fn exact_decimal_and_binary64_do_not_share_a_rounded_value() {
    let outcome = order_sampled(&facts(
        vec![floating(1.1), decimal("11", -1)],
        false,
        false,
        true,
    ))
    .unwrap();
    assert!(
        matches!(outcome,Outcome::Sorted {positions,numeric_diagnostics,..} if positions==vec![1,0] && numeric_diagnostics==vec![Event::DecimalFloatOperation])
    );
}
#[test]
fn a_float_trap_is_caught_in_probe_but_reached_again_by_sort() {
    let outcome = order_sampled(&facts(
        vec![floating(1.1), decimal("11", -1)],
        true,
        false,
        true,
    ))
    .unwrap();
    assert!(
        matches!(outcome,Outcome::Refused {reason:Error::DecimalFloatOperation,numeric_diagnostics,..} if numeric_diagnostics==vec![Event::DecimalFloatOperation])
    );
}
#[test]
fn invalid_probe_suppression_preserves_both_sticky_flags() {
    for catches in [false, true] {
        let mut index = facts(
            vec![decimal("1", 0), floating(f64::NAN)],
            false,
            true,
            catches,
        );
        index.missing[1] = true;
        let outcome = order_sampled(&index).unwrap();
        let expected = vec![Event::DecimalFloatOperation, Event::DecimalInvalidOperation];
        if catches {
            assert!(
                matches!(outcome,Outcome::Sorted {numeric_diagnostics,..} if numeric_diagnostics==expected)
            );
        } else {
            assert!(
                matches!(outcome,Outcome::Refused {reason:Error::DecimalInvalidOperation,numeric_diagnostics,..} if numeric_diagnostics==expected)
            );
        }
    }
}
#[test]
fn leading_float_nan_does_not_reach_decimal_signals() {
    let mut index = facts(vec![floating(f64::NAN), decimal("1", 0)], true, true, false);
    index.missing[0] = true;
    assert!(
        matches!(order_sampled(&index).unwrap(),Outcome::Sorted {positions,numeric_diagnostics,..} if positions==vec![1,0] && numeric_diagnostics.is_empty())
    );
}
#[test]
fn subnormal_conversion_is_exact_and_independent_of_decimal_exponent_size() {
    let outcome = order_sampled(&facts(
        vec![floating(f64::from_bits(1)), decimal("1", -400)],
        false,
        false,
        true,
    ))
    .unwrap();
    assert!(matches!(outcome,Outcome::Sorted {positions,..} if positions==vec![1,0]));
}

#[test]
fn masked_float_nan_against_an_integer_is_unordered_without_decimal_signals() {
    for first in [
        Atom::Integer { value: "1".into() },
        Atom::Boolean { value: true },
    ] {
        let mut index = facts(
            vec![first, floating(f64::NAN), decimal("11", -1)],
            false,
            false,
            true,
        );
        index.missing[1] = true;
        assert!(
            matches!(order_sampled(&index).unwrap(),Outcome::Sorted {positions,numeric_diagnostics,..} if positions==vec![0,2,1] && numeric_diagnostics.is_empty())
        );
    }
}
