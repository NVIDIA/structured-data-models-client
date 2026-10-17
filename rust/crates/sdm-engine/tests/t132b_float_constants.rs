// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A float constant compared with a static decimal column (the owner's
//! KO2-50, which replaces P1B-71's reading): the core compares the column's
//! values exactly with the shortest decimal naming the constant's double,
//! the decimal the query wrote (`0.1` is 0.1, never the double's
//! 0.1000000000000000055...), at any scale, never rounding the constant to
//! the column's scale; a constant past the column's digits orders by its
//! sign, every value under a positive one and over a negative one. No
//! constant is refused.
//!
//! Written by the independent test author, through
//! `sdm_engine::labels::label` only. Each expected value was computed apart
//! from the core, in Python: the constant's shortest decimal is
//! `decimal.Decimal(repr(value))`, and at scale `s` its unscaled value is
//! that times `10**s` (under a context of 2000 digits), exact where it is
//! an integer, otherwise lying strictly between its floor and the next.
//!
//! Gating: a test named in `contract/tests/t132b-pending.json`'s
//! `pending["exits"]` as `rust:<test name>` must still fail; any other
//! must pass. The list is read at compile time: this crate's tests read no
//! file at run time (`clippy.toml`).

use sdm_engine::labels::{Column, Constant, Labels, RelOp, Target, label};
use serde_json::Value;

const PENDING: &str = include_str!("../../../../contract/tests/t132b-pending.json");

fn pending(test: &str) -> bool {
    let gating: Value = serde_json::from_str(PENDING).unwrap();
    gating["pending"]["exits"]
        .as_array()
        .unwrap()
        .iter()
        .any(|t| t == &Value::from(format!("rust:{test}")))
}

/// Runs `check`; a pending test must fail, any other must pass.
fn gated(test: &str, check: impl FnOnce() + std::panic::UnwindSafe) {
    let outcome = std::panic::catch_unwind(check);
    match (outcome, pending(test)) {
        (Ok(()), true) => panic!(
            "{test} passes now: take rust:{test} out of \
             contract/tests/t132b-pending.json's pending[\"exits\"]"
        ),
        (Err(failure), false) => std::panic::resume_unwind(failure),
        _ => {}
    }
}

fn condition(values: Vec<Option<i128>>, precision: u8, scale: i8, op: RelOp, value: f64) -> Target {
    Target::Condition {
        target: Box::new(Target::Column(Column::Decimal {
            values,
            precision,
            scale,
        })),
        op,
        constant: Some(Constant::Float(value)),
    }
}

/// The labels `op` gives the unscaled `values` at DECIMAL(`precision`,
/// `scale`) against `value`, each value labelled.
fn labels(values: &[i128], precision: u8, scale: i8, op: RelOp, value: f64) -> Vec<bool> {
    let at = format!("{value} {op:?} at DECIMAL({precision}, {scale})");
    let column = values.iter().copied().map(Some).collect();
    let out = label(&condition(column, precision, scale, op, value));
    assert!(out.is_ok(), "{at} refused: {out:?}");
    let out = out.unwrap();
    assert_eq!(out.mask, vec![true; values.len()], "{at}");
    let Labels::Bool(got) = out.labels else {
        panic!("{at}: not boolean labels");
    };
    got
}

/// `value`'s shortest decimal is the unscaled `at` at the scale exactly:
/// of the values one under, at and one over it, `=` holds only at it, `>`
/// only over it, `>=` at and over, `<` only under.
fn equals(value: f64, precision: u8, scale: i8, at: i128) {
    let values = [at - 1, at, at + 1];
    for (op, want) in [
        (RelOp::Eq, [false, true, false]),
        (RelOp::Ne, [true, false, true]),
        (RelOp::Gt, [false, false, true]),
        (RelOp::Ge, [false, true, true]),
        (RelOp::Lt, [true, false, false]),
        (RelOp::Le, [true, true, false]),
    ] {
        assert_eq!(
            labels(&values, precision, scale, op, value),
            want,
            "{value} {op:?} at DECIMAL({precision}, {scale}): equal to {at}"
        );
    }
}

/// `value`'s shortest decimal lies strictly between the unscaled `floor`
/// and `floor + 1` at the scale: neither equals it, `floor` is under it
/// and `floor + 1` over it, whatever rounding would have chosen.
fn between(value: f64, precision: u8, scale: i8, floor: i128) {
    let values = [floor, floor + 1];
    for (op, want) in [
        (RelOp::Eq, [false, false]),
        (RelOp::Ne, [true, true]),
        (RelOp::Gt, [false, true]),
        (RelOp::Ge, [false, true]),
        (RelOp::Lt, [true, false]),
        (RelOp::Le, [true, false]),
    ] {
        assert_eq!(
            labels(&values, precision, scale, op, value),
            want,
            "{value} {op:?} at DECIMAL({precision}, {scale}): between {floor} and the next"
        );
    }
}

#[test]
fn a_float_constant_is_the_decimal_it_names_at_scales_23_to_38() {
    gated(
        "a_float_constant_is_the_decimal_it_names_at_scales_23_to_38",
        || {
            for (value, precision, scale, at) in [
                // 0.1 on DECIMAL(38, 23) is 0.1: equal, never the double's
                // 0.10000000000000000555112 (P1B-71's reading).
                (0.1, 38, 23, 10_000_000_000_000_000_000_000_i128),
                (-0.1, 38, 23, -10_000_000_000_000_000_000_000),
                (0.25, 38, 23, 25_000_000_000_000_000_000_000),
                (0.7, 38, 23, 70_000_000_000_000_000_000_000),
                (-0.7, 38, 23, -70_000_000_000_000_000_000_000),
                // 2/3's double is named 0.6666666666666666.
                (2.0 / 3.0, 38, 23, 66_666_666_666_666_660_000_000),
                (123.456, 38, 23, 12_345_600_000_000_000_000_000_000),
                (0.1, 38, 28, 1_000_000_000_000_000_000_000_000_000),
                (0.7, 38, 28, 7_000_000_000_000_000_000_000_000_000),
                (-0.7, 38, 28, -7_000_000_000_000_000_000_000_000_000),
                (2.0 / 3.0, 38, 28, 6_666_666_666_666_666_000_000_000_000),
                (123.456, 38, 28, 1_234_560_000_000_000_000_000_000_000_000),
                // A precision under 38 at a scale past 22.
                (0.7, 30, 28, 7_000_000_000_000_000_000_000_000_000),
                (0.1, 38, 30, 100_000_000_000_000_000_000_000_000_000),
                (0.7, 38, 30, 700_000_000_000_000_000_000_000_000_000),
                (123.456, 38, 30, 123_456_000_000_000_000_000_000_000_000_000),
                (
                    0.1,
                    38,
                    38,
                    10_000_000_000_000_000_000_000_000_000_000_000_000,
                ),
                (
                    -0.1,
                    38,
                    38,
                    -10_000_000_000_000_000_000_000_000_000_000_000_000,
                ),
                (
                    0.7,
                    38,
                    38,
                    70_000_000_000_000_000_000_000_000_000_000_000_000,
                ),
                (
                    2.0 / 3.0,
                    38,
                    38,
                    66_666_666_666_666_660_000_000_000_000_000_000_000,
                ),
                (
                    0.99999,
                    38,
                    38,
                    99_999_000_000_000_000_000_000_000_000_000_000_000,
                ),
                // The largest DECIMAL(38, 37) holds is under 10: 9 is held.
                (
                    9.0,
                    38,
                    37,
                    90_000_000_000_000_000_000_000_000_000_000_000_000,
                ),
                // 2**-24's double is named 5.960464477539063e-08 (exactly
                // 5.9604644775390625e-08): 23 places, equal at scale 23.
                (2.0_f64.powi(-24), 38, 23, 5_960_464_477_539_063),
                (3.0 * 2.0_f64.powi(-24), 38, 23, 17_881_393_432_617_188),
                (2.0_f64.powi(-29), 38, 28, 18_626_451_492_309_570_000),
            ] {
                equals(value, precision, scale, at);
            }
            // Past the scale: between two of the column's values, never
            // rounded onto either (P1B-71 read 6e-39 at scale 38 as 1).
            for (value, precision, scale, floor) in [
                (5e-324, 38, 38, 0_i128),
                (-5e-324, 38, 38, -1),
                (4e-39, 38, 38, 0),
                (6e-39, 38, 38, 0),
                (-6e-39, 38, 38, -1),
            ] {
                between(value, precision, scale, floor);
            }
        },
    );
}

#[test]
fn a_float_constant_half_way_past_the_scale_is_exact_not_rounded_to_even() {
    gated(
        "a_float_constant_half_way_past_the_scale_is_exact_not_rounded_to_even",
        || {
            // 5, 15 and 25 times 10**-(s + 1) lie exactly half-way between
            // two of the column's values at scale s: neither equals the
            // constant, and neither neighbour is taken for it (half to even
            // would read 0, 2 and 2; half up 1, 2 and 3).
            for (value, scale, floor) in [
                (5e-24, 23, 0_i128),
                (1.5e-23, 23, 1),
                (2.5e-23, 23, 2),
                (5e-31, 30, 0),
                (1.5e-30, 30, 1),
                (5e-39, 38, 0),
                (1.5e-38, 38, 1),
                (2.5e-38, 38, 2),
                // At scale 0, KO2-50's own case: d > 1.5 holds for d = 2.
                (1.5, 0, 1),
                (2.5, 0, 2),
            ] {
                between(value, 38, scale, floor);
                between(-value, 38, scale, -floor - 1);
            }
        },
    );
}

#[test]
fn a_float_constant_past_the_decimal_s_digits_orders_by_its_sign() {
    gated(
        "a_float_constant_past_the_decimal_s_digits_orders_by_its_sign",
        || {
            // P1B-71 refused these INVALID_REQUEST; each now answers: every
            // value of the column, its largest and smallest included, is
            // under a positive constant past its digits and over a negative
            // one, and equal to neither.
            for (value, precision, scale) in [
                (1.0, 38, 38),
                (100.0, 30, 28),
                (12_345.678, 38, 34),
                (123.456, 38, 38),
                (1e16, 38, 23),
                (1e300, 38, 30),
                (1e39, 38, 0),
            ] {
                let most = 10_i128.pow(u32::from(precision)) - 1;
                let values = [-most, -1, 0, 1, most];
                for (sign, under) in [(1.0, true), (-1.0, false)] {
                    let constant = sign * value;
                    for (op, want) in [
                        (RelOp::Eq, false),
                        (RelOp::Ne, true),
                        (RelOp::Lt, under),
                        (RelOp::Le, under),
                        (RelOp::Gt, !under),
                        (RelOp::Ge, !under),
                    ] {
                        assert_eq!(
                            labels(&values, precision, scale, op, constant),
                            vec![want; values.len()],
                            "{constant} {op:?} at DECIMAL({precision}, {scale})"
                        );
                    }
                }
            }
        },
    );
}

#[test]
fn a_static_decimal_past_scale_22_is_compared_with_a_float_exactly() {
    gated(
        "a_static_decimal_past_scale_22_is_compared_with_a_float_exactly",
        || {
            // users.f > 0.7 and users.f = 0.7 at DECIMAL(38, 30): the
            // constant is 0.7, so the column's 0.7 equals it and is not
            // greater, and 0.699999999999999955591079014994 (the double's
            // exact value, which P1B-71 read) is under it. A null has no
            // label.
            let seven = 700_000_000_000_000_000_000_000_000_000_i128;
            let double = 699_999_999_999_999_955_591_079_014_994_i128;
            let values = vec![
                Some(seven),
                Some(double),
                Some(seven - 1),
                None,
                Some(seven + 1),
            ];
            let gt = label(&condition(values.clone(), 38, 30, RelOp::Gt, 0.7));
            assert!(gt.is_ok(), "refused: {gt:?}");
            let gt = gt.unwrap();
            assert_eq!(gt.mask, vec![true, true, true, false, true]);
            assert_eq!(gt.labels, Labels::Bool(vec![false, false, false, true]));
            let eq = label(&condition(values, 38, 30, RelOp::Eq, 0.7)).unwrap();
            assert_eq!(eq.labels, Labels::Bool(vec![true, false, false, false]));
            // At DECIMAL(38, 23): 0.5 is held exactly.
            let values = vec![Some(50_000_000_000_000_000_000_000), Some(1), None];
            let ge = label(&condition(values, 38, 23, RelOp::Ge, 0.5));
            assert!(ge.is_ok(), "refused: {ge:?}");
            let ge = ge.unwrap();
            assert_eq!(ge.mask, vec![true, true, false]);
            assert_eq!(ge.labels, Labels::Bool(vec![true, false]));
        },
    );
}
