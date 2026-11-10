// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Label arithmetic, masks and calendar windows with independent expectations.

use sdm_contract::ErrorCode;
use sdm_contract::pql::TimeUnit;
use sdm_engine::labels::{
    self, Aggregation, Bounds, Column, Constant, Labels, RelOp, Target, Values,
};
use sdm_engine::time::offset::{Offset, timeframe};

fn nanos(text: &str) -> i64 {
    text.parse::<jiff::Timestamp>()
        .unwrap()
        .as_nanosecond()
        .try_into()
        .unwrap()
}

#[test]
fn fixed_windows_and_calendar_months_keep_time_and_clamp_days() {
    for (unit, seconds) in [
        (TimeUnit::Seconds, 1_i64),
        (TimeUnit::Minutes, 60),
        (TimeUnit::Hours, 3600),
        (TimeUnit::Days, 86400),
        (TimeUnit::Weeks, 604800),
    ] {
        for count in [-7, -1, 0, 1, 7] {
            let offset = Offset { count, unit };
            for anchor in [-1_234_567_890, 0, 1_234_567_890] {
                assert_eq!(
                    offset.add_to(anchor, "anchor").unwrap(),
                    anchor + count * seconds * 1_000_000_000
                );
            }
            assert_eq!(
                timeframe(Offset { count: -1, unit }, offset, "anchor").unwrap(),
                i128::from((count + 1) * seconds * 1_000_000_000)
            );
        }
    }
    for (from, count, to) in [
        (
            "2024-01-31T12:34:56.123456789Z",
            1,
            "2024-02-29T12:34:56.123456789Z",
        ),
        (
            "2023-01-31T12:34:56.123456789Z",
            1,
            "2023-02-28T12:34:56.123456789Z",
        ),
        (
            "2024-03-31T12:34:56.123456789Z",
            -1,
            "2024-02-29T12:34:56.123456789Z",
        ),
        (
            "2024-02-29T12:34:56.123456789Z",
            12,
            "2025-02-28T12:34:56.123456789Z",
        ),
        (
            "1969-12-31T23:59:59.999999999Z",
            1,
            "1970-01-31T23:59:59.999999999Z",
        ),
    ] {
        assert_eq!(
            Offset {
                count,
                unit: TimeUnit::Months
            }
            .add_to(nanos(from), "anchor")
            .unwrap(),
            nanos(to)
        );
    }
    let months = |count| Offset {
        count,
        unit: TimeUnit::Months,
    };
    assert_eq!(
        timeframe(months(0), months(2), "anchor").unwrap(),
        59 * 86400 * 1_000_000_000_i128
    );
    assert_eq!(
        timeframe(months(-1), months(2), "anchor").unwrap(),
        90 * 86400 * 1_000_000_000_i128
    );
    for (anchor, count) in [(i64::MAX, 1), (i64::MIN, -1)] {
        let error = Offset {
            count,
            unit: TimeUnit::Seconds,
        }
        .add_to(anchor, "anchor")
        .unwrap_err();
        assert_eq!(error.code, ErrorCode::ValueOutOfRange);
        assert_eq!(
            error.fields.get("field"),
            Some(&sdm_contract::Detail::text("anchor"))
        );
    }
}

#[test]
fn grouped_aggregates_keep_empty_groups_and_missing_values_separate() {
    // Interleaved groups: group0=[1,3], group1=[8], group2 has only
    // missing values, group3 has no rows. All three physical types agree.
    for values in [
        Values::Int(vec![Some(1), Some(8), None, Some(3), None]),
        Values::Float(vec![Some(1.0), Some(8.0), Some(f32::NAN), Some(3.0), None]),
        Values::Decimal {
            values: vec![Some(100), Some(800), None, Some(300), None],
            precision: 10,
            scale: 2,
        },
    ] {
        for (op, wanted, present) in [
            (
                Aggregation::Sum,
                [4.0_f32, 8.0, 0.0, 0.0],
                [true, true, true, true],
            ),
            (
                Aggregation::Count,
                [2.0, 1.0, 0.0, 0.0],
                [true, true, true, true],
            ),
            (
                Aggregation::Avg,
                [2.0, 8.0, 0.0, 0.0],
                [true, true, false, false],
            ),
            (
                Aggregation::Min,
                [1.0, 8.0, 0.0, 0.0],
                [true, true, false, false],
            ),
            (
                Aggregation::Max,
                [3.0, 8.0, 0.0, 0.0],
                [true, true, false, false],
            ),
        ] {
            let got = labels::aggregate(
                op,
                4,
                &[0, 1, 2, 0, 2],
                &values,
                &[Some(0); 5],
                Bounds {
                    starts: None,
                    ends: &[0; 4],
                },
            )
            .unwrap();
            assert_eq!(got.present, present);
            for i in 0..4 {
                if present[i] {
                    assert_eq!(got.values[i].to_bits(), wanted[i].to_bits());
                }
            }
        }
    }
}

#[test]
fn conditions_preserve_masks_and_all_six_comparison_directions() {
    for (op, expected) in [
        (RelOp::Eq, [false, true, false]),
        (RelOp::Ne, [true, false, true]),
        (RelOp::Lt, [true, false, false]),
        (RelOp::Le, [true, true, false]),
        (RelOp::Gt, [false, false, true]),
        (RelOp::Ge, [false, true, true]),
    ] {
        for constant in [Constant::Int(0), Constant::Float(0.0)] {
            for column in [
                Column::Int(vec![Some(-1), Some(0), Some(1), None]),
                Column::Float(vec![Some(-1.0), Some(0.0), Some(1.0), Some(f32::NAN)]),
            ] {
                let got = labels::label(&Target::Condition {
                    target: Box::new(Target::Column(column)),
                    op,
                    constant: Some(constant.clone()),
                })
                .unwrap();
                assert_eq!(got.mask, [true, true, true, false]);
                assert_eq!(got.labels, Labels::Bool(expected.to_vec()));
            }
        }
    }
}

#[test]
fn condition_constants_follow_integer_and_float32_precision() {
    for (column, constant, op, expected) in [
        (
            Column::Int(vec![Some(9_007_199_254_740_993)]),
            Constant::Float(9_007_199_254_740_992.0),
            RelOp::Eq,
            true,
        ),
        (
            Column::Int(vec![Some(9_007_199_254_740_993)]),
            Constant::Int(9_007_199_254_740_992),
            RelOp::Ne,
            true,
        ),
        (
            Column::Float(vec![Some(16_777_216.0)]),
            Constant::Int(16_777_217),
            RelOp::Eq,
            true,
        ),
        (
            Column::Float(vec![Some(16_777_216.0)]),
            Constant::Int(16_777_217),
            RelOp::Gt,
            false,
        ),
        (
            Column::Float(vec![Some(f32::INFINITY)]),
            Constant::Float(1.0),
            RelOp::Gt,
            true,
        ),
        (
            Column::Float(vec![Some(f32::NEG_INFINITY)]),
            Constant::Float(1.0),
            RelOp::Lt,
            true,
        ),
        (
            Column::Int(vec![Some(9_007_199_254_740_993)]),
            Constant::Int(9_007_199_254_740_992),
            RelOp::Gt,
            true,
        ),
        (
            Column::Float(vec![Some(16_777_216.0)]),
            Constant::Float(16_777_217.0),
            RelOp::Eq,
            true,
        ),
        (
            Column::Int(vec![Some(1)]),
            Constant::Float(1.5),
            RelOp::Lt,
            true,
        ),
        (
            Column::Float(vec![Some(-0.0)]),
            Constant::Float(0.0),
            RelOp::Eq,
            true,
        ),
    ] {
        let got = labels::label(&Target::Condition {
            target: Box::new(Target::Column(column)),
            op,
            constant: Some(constant),
        })
        .unwrap();
        assert_eq!(got.mask, [true]);
        assert_eq!(got.labels, Labels::Bool(vec![expected]));
    }
}

#[test]
fn decimal_conversion_keeps_scale_sign_and_small_values() {
    for (unscaled, scale, wanted) in [
        (12345_i128, 2, 123.45_f32),
        (-12345, 2, -123.45),
        (5, 18, 5e-18),
        (97_000_000_000_000_000_002, 18, 97.0),
        (-49_476_625_724_350, 18, -4.947_662_5e-5),
        (25, -2, 2500.0),
        (0, 38, 0.0),
    ] {
        assert_eq!(
            (labels::decimal_to_f64(unscaled, scale) as f32).to_bits(),
            wanted.to_bits()
        );
    }
}

#[test]
fn null_timestamps_and_window_edges_do_not_enter_any_aggregate() {
    for (op, expected) in [
        (Aggregation::Sum, 5.0_f32),
        (Aggregation::Count, 2.0),
        (Aggregation::Avg, 2.5),
        (Aggregation::Min, 2.0),
        (Aggregation::Max, 3.0),
    ] {
        let got = labels::aggregate(
            op,
            1,
            &[0; 5],
            &Values::Int(vec![Some(100), Some(2), Some(3), Some(200), Some(300)]),
            &[Some(0), Some(1), Some(10), Some(11), None],
            Bounds {
                starts: Some(&[0]),
                ends: &[10],
            },
        )
        .unwrap();
        assert_eq!(got.present, [true]);
        assert_eq!(got.values[0].to_bits(), expected.to_bits());
    }
    let got = labels::aggregate(
        Aggregation::Count,
        1,
        &[0; 5],
        &Values::Ones(5),
        &[Some(0), Some(1), Some(10), Some(11), None],
        Bounds {
            starts: Some(&[0]),
            ends: &[10],
        },
    )
    .unwrap();
    assert_eq!(got.values, [2.0]);
}

fn aggregate_one(op: Aggregation, values: Values, n: usize) -> f32 {
    let got = labels::aggregate(
        op,
        1,
        &vec![0; n],
        &values,
        &vec![Some(0); n],
        Bounds {
            starts: None,
            ends: &[0],
        },
    )
    .unwrap();
    assert_eq!(got.present, [true]);
    got.values[0]
}

#[test]
fn float_arithmetic_retains_compensation_infinity_and_zero_contracts() {
    let mut compensated = vec![Some(100_000_000.0)];
    compensated.extend([Some(1.0); 8]);
    compensated.push(Some(-100_000_000.0));
    assert_eq!(
        aggregate_one(Aggregation::Sum, Values::Float(compensated.clone()), 10).to_bits(),
        8.0_f32.to_bits()
    );
    assert_eq!(
        aggregate_one(Aggregation::Avg, Values::Float(compensated), 10).to_bits(),
        0.8_f32.to_bits()
    );
    for op in [
        Aggregation::Sum,
        Aggregation::Avg,
        Aggregation::Min,
        Aggregation::Max,
    ] {
        for value in [f32::INFINITY, f32::NEG_INFINITY] {
            assert_eq!(
                aggregate_one(op, Values::Float(vec![Some(value)]), 1),
                value
            );
        }
    }
    for op in [Aggregation::Sum, Aggregation::Avg] {
        assert!(
            aggregate_one(
                op,
                Values::Float(vec![Some(f32::INFINITY), Some(f32::NEG_INFINITY)]),
                2
            )
            .is_nan()
        );
        assert_eq!(
            aggregate_one(op, Values::Float(vec![Some(-0.0)]), 1).to_bits(),
            0.0_f32.to_bits()
        );
    }
    for op in [Aggregation::Min, Aggregation::Max] {
        assert_eq!(
            aggregate_one(op, Values::Float(vec![Some(-0.0)]), 1).to_bits(),
            (-0.0_f32).to_bits()
        );
    }
}

#[test]
fn integer_sum_wrap_and_mean_precision_are_distinct() {
    // Integer sums retain signed-int64 arithmetic, while means accumulate
    // in float64 and only then project to the float32 label column.
    let values = || Values::Int(vec![Some(i64::MAX), Some(1)]);
    assert_eq!(
        aggregate_one(Aggregation::Sum, values(), 2).to_bits(),
        (-9_223_372_036_854_775_808.0_f32).to_bits()
    );
    assert_eq!(
        aggregate_one(Aggregation::Avg, values(), 2).to_bits(),
        4_611_686_018_427_387_904.0_f32.to_bits()
    );
    assert_eq!(
        aggregate_one(
            Aggregation::Avg,
            Values::Int(vec![Some(16_777_217), Some(16_777_218)]),
            2
        )
        .to_bits(),
        16_777_218.0_f32.to_bits()
    );
}

#[test]
fn independent_arrow_decimal_conversion_sentinels() {
    // Independent PyArrow 25.0.1 -> NumPy 2.5.3 float32 observations.
    // These few bit patterns retain the external conversion boundary.
    for (unscaled, bits) in [
        (5_i128, 0x22b877aa_u32),
        (1, 0x219392ef),
        (1_000_000_000_000_000_000, 0x3f800000),
        (97_000_000_000_000_000_002, 0x42c20000),
    ] {
        assert_eq!(
            (labels::decimal_to_f64(unscaled, 18) as f32).to_bits(),
            bits
        );
    }
}
