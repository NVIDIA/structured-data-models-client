// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A float constant compared with a static decimal past scale 22 (T1.32b's
//! bundled admission; the owner's decision on P1B-71, 2026-10-13): a float
//! constant compared with a static DECIMAL(p, s) column at a scale of 23 to
//! 38 is read from the float's exact binary value at the column's scale,
//! rounded half to even, as it is through scale 22; no refusal. A constant
//! the column cannot hold is the caller's error, INVALID_REQUEST, as now.
//!
//! Written by the independent test author before the implementation,
//! through `sdm_engine::labels::label` only: each constant's reading is
//! pinned by labelling the column values one under, at and one over the
//! expected unscaled value with `=` and `>`. Each expected value was
//! computed apart from the core, in Python:
//! `int((decimal.Decimal(value) * 10**s).quantize(1, ROUND_HALF_EVEN))`
//! under a context of 2000 digits. From scale 28 the reference's pyarrow
//! approximates; the core is exact there (P1B-71 rewritten as that
//! difference), so these are held to the exact value, not to pyarrow's.
//!
//! Gating: each test is named in `contract/tests/t132b-pending.json`'s
//! `pending["exits"]` as `rust:<test name>`, failing when written (the core
//! refused every float past scale 22 by name, `condition_value`); a pending
//! test must still fail, any other must pass. The list is read at compile
//! time: this crate's tests read no file at run time (`clippy.toml`).

use sdm_contract::{Detail, ErrorCode};
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

/// Holds `value` read at DECIMAL(`precision`, `scale`) to the unscaled
/// `wanted`: of the column values one under, at and one over it, `=` holds
/// only at it and `>` only over it.
fn reads_as(value: f64, precision: u8, scale: i8, wanted: i128) {
    let column = vec![Some(wanted - 1), Some(wanted), Some(wanted + 1)];
    let at = format!("{value} at DECIMAL({precision}, {scale})");
    for (op, labels) in [
        (RelOp::Eq, vec![false, true, false]),
        (RelOp::Gt, vec![false, false, true]),
    ] {
        let out = label(&condition(column.clone(), precision, scale, op, value));
        assert!(out.is_ok(), "{at} refused: {out:?}");
        let out = out.unwrap();
        assert_eq!(out.mask, vec![true; 3], "{at}");
        assert_eq!(out.labels, Labels::Bool(labels), "{at} {op:?}");
    }
}

#[test]
fn a_float_constant_reads_exactly_at_scales_23_to_38() {
    gated("a_float_constant_reads_exactly_at_scales_23_to_38", || {
        for (value, precision, scale, wanted) in [
            (0.1, 38, 23, 10_000_000_000_000_000_555_112_i128),
            (-0.1, 38, 23, -10_000_000_000_000_000_555_112),
            (0.25, 38, 23, 25_000_000_000_000_000_000_000),
            (0.7, 38, 23, 69_999_999_999_999_995_559_108),
            (-0.7, 38, 23, -69_999_999_999_999_995_559_108),
            (2.0 / 3.0, 38, 23, 66_666_666_666_666_662_965_923),
            (123.456, 38, 23, 12_345_600_000_000_000_306_954_462),
            (0.1, 38, 28, 1_000_000_000_000_000_055_511_151_231),
            (0.25, 38, 28, 2_500_000_000_000_000_000_000_000_000),
            (0.7, 38, 28, 6_999_999_999_999_999_555_910_790_150),
            (-0.7, 38, 28, -6_999_999_999_999_999_555_910_790_150),
            (2.0 / 3.0, 38, 28, 6_666_666_666_666_666_296_592_325_125),
            (123.456, 38, 28, 1_234_560_000_000_000_030_695_446_184_836),
            // A precision under 38 at a scale past 22.
            (0.7, 30, 28, 6_999_999_999_999_999_555_910_790_150),
            (0.1, 38, 30, 100_000_000_000_000_005_551_115_123_126),
            (0.25, 38, 30, 250_000_000_000_000_000_000_000_000_000),
            (0.7, 38, 30, 699_999_999_999_999_955_591_079_014_994),
            (-0.7, 38, 30, -699_999_999_999_999_955_591_079_014_994),
            (2.0 / 3.0, 38, 30, 666_666_666_666_666_629_659_232_512_495),
            (123.456, 38, 30, 123_456_000_000_000_003_069_544_618_483_633),
            (
                0.1,
                38,
                38,
                10_000_000_000_000_000_555_111_512_312_578_270_212,
            ),
            (
                -0.1,
                38,
                38,
                -10_000_000_000_000_000_555_111_512_312_578_270_212,
            ),
            (
                0.25,
                38,
                38,
                25_000_000_000_000_000_000_000_000_000_000_000_000,
            ),
            (
                0.7,
                38,
                38,
                69_999_999_999_999_995_559_107_901_499_373_838_305,
            ),
            (
                -0.7,
                38,
                38,
                -69_999_999_999_999_995_559_107_901_499_373_838_305,
            ),
            (
                2.0 / 3.0,
                38,
                38,
                66_666_666_666_666_662_965_923_251_249_478_198_588,
            ),
            (
                0.99999,
                38,
                38,
                99_999_000_000_000_004_551_026_222_543_441_690_505,
            ),
            // The largest DECIMAL(38, 37) holds is under 10: 9 is read.
            (
                9.0,
                38,
                37,
                90_000_000_000_000_000_000_000_000_000_000_000_000,
            ),
            // Under half the last place, zero; over it, one.
            (5e-324, 38, 38, 0),
            (4e-39, 38, 38, 0),
            (6e-39, 38, 38, 1),
        ] {
            reads_as(value, precision, scale, wanted);
        }
    });
}

#[test]
fn a_float_constant_half_way_past_scale_22_rounds_to_even() {
    gated(
        "a_float_constant_half_way_past_scale_22_rounds_to_even",
        || {
            // 2**-(s + 1) and 3 * 2**-(s + 1) have exactly s + 1 decimal
            // places, the last a 5: each lies exactly half-way at scale s, and
            // of the two neighbours the even one is taken (5**s is odd, so one
            // rounds down and one up).
            for (value, scale, wanted) in [
                (2.0_f64.powi(-24), 23, 5_960_464_477_539_062_i128),
                (3.0 * 2.0_f64.powi(-24), 23, 17_881_393_432_617_188),
                (2.0_f64.powi(-29), 28, 18_626_451_492_309_570_312),
                (3.0 * 2.0_f64.powi(-29), 28, 55_879_354_476_928_710_938),
                (2.0_f64.powi(-31), 30, 465_661_287_307_739_257_812),
                (3.0 * 2.0_f64.powi(-31), 30, 1_396_983_861_923_217_773_438),
                (2.0_f64.powi(-39), 38, 181_898_940_354_585_647_583_007_812),
                (
                    3.0 * 2.0_f64.powi(-39),
                    38,
                    545_696_821_063_756_942_749_023_438,
                ),
            ] {
                reads_as(value, 38, scale, wanted);
                reads_as(-value, 38, scale, -wanted);
            }
        },
    );
}

#[test]
fn a_float_constant_its_decimal_cannot_hold_past_scale_22_is_the_caller_s_error() {
    gated(
        "a_float_constant_its_decimal_cannot_hold_past_scale_22_is_the_caller_s_error",
        || {
            // As at scale 22 and under: a constant past the column's digits
            // is INVALID_REQUEST naming the query, never a refusal by name.
            // Past the precision but inside i128 (12345.678 at
            // DECIMAL(38, 34): 39 digits), and past i128.
            for (value, precision, scale) in [
                (1.0, 38, 38),
                (-1.0, 38, 38),
                (100.0, 30, 28),
                (12_345.678, 38, 34),
                (123.456, 38, 38),
                (1e16, 38, 23),
                (-1e16, 38, 23),
                (1e300, 38, 30),
            ] {
                let at = format!("{value} at DECIMAL({precision}, {scale})");
                let err = label(&condition(
                    vec![Some(1)],
                    precision,
                    scale,
                    RelOp::Gt,
                    value,
                ))
                .unwrap_err();
                assert_eq!(err.code, ErrorCode::InvalidRequest, "{at}");
                assert_eq!(
                    err.fields.get("field"),
                    Some(&Detail::text("query")),
                    "{at}"
                );
                assert!(!err.fields.contains_key("rule"), "{at} refused by name");
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
            // constant is the double nearest 0.7 read exactly,
            // 0.699999999999999955591079014994 at scale 30, so the column's
            // 0.7 is greater, and only the exact reading equals. A null has
            // no label.
            let exact = 699_999_999_999_999_955_591_079_014_994_i128;
            let values = vec![
                Some(700_000_000_000_000_000_000_000_000_000),
                Some(exact),
                Some(exact - 1),
                None,
                Some(exact + 1),
            ];
            let gt = label(&condition(values.clone(), 38, 30, RelOp::Gt, 0.7));
            assert!(gt.is_ok(), "refused: {gt:?}");
            let gt = gt.unwrap();
            assert_eq!(gt.mask, vec![true, true, true, false, true]);
            assert_eq!(gt.labels, Labels::Bool(vec![true, false, false, true]));
            let eq = label(&condition(values, 38, 30, RelOp::Eq, 0.7)).unwrap();
            assert_eq!(eq.labels, Labels::Bool(vec![false, true, false, false]));
            // At DECIMAL(38, 23), the scale P1B-71's diagnostic test
            // refused before the owner's decision: 0.5 is read.
            let values = vec![Some(50_000_000_000_000_000_000_000), Some(1), None];
            let ge = label(&condition(values, 38, 23, RelOp::Ge, 0.5));
            assert!(ge.is_ok(), "refused: {ge:?}");
            let ge = ge.unwrap();
            assert_eq!(ge.mask, vec![true, true, false]);
            assert_eq!(ge.labels, Labels::Bool(vec![true, false]));
        },
    );
}
