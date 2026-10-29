// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Actual task-meaning and lazy native numeric observations, without Python policy.

use sdm_contract::{Detail, ErrorCode, Sentence};
use sdm_host::{TabularTaskMeaning, TabularTaskNumber};

fn numerical() -> TabularTaskMeaning {
    TabularTaskMeaning::new(true)
        .unwrap()
        .stated(false, None, None)
        .unwrap()
        .inferred(false)
        .unwrap()
}

#[test]
fn stype_and_stated_guards_precede_inference_and_keep_exact_public_details() {
    let error = TabularTaskMeaning::new(false).unwrap_err();
    assert_eq!(error.code, ErrorCode::UnsupportedRequest);
    assert_eq!(error.sentence, Sentence::TabularTargetStype);
    assert!(error.public_details.unwrap().is_empty());
    let target = TabularTaskMeaning::new(true).unwrap();
    let explicit = target.stated(true, Some(true), Some(true)).unwrap();
    assert!(explicit.original());
    assert!(explicit.numbers(2).is_err());
    assert!(explicit.inferred(true).is_err());
    let refused = target.stated(true, Some(false), None).unwrap_err();
    assert_eq!(refused.sentence, Sentence::ModelLacksTask);
    let details = refused.public_details.unwrap();
    assert_eq!(details.len(), 2);
    assert!(details.contains_key("task"));
    assert!(details.contains_key("supported"));
    assert!(!details.contains_key("model"));
    assert_eq!(
        refused.fields.get("model"),
        Some(&Detail::text("the tabular model"))
    );
    assert!(target.stated(false, Some(true), None).is_err());
    assert!(target.stated(true, Some(false), Some(true)).is_err());
    let categorical = target
        .stated(false, None, None)
        .unwrap()
        .inferred(true)
        .unwrap();
    assert_eq!(categorical.kind(), Some("classification"));
    assert!(categorical.numbers(2).is_err());
}

#[test]
fn actual_cardinality_gates_iteration_and_exhaustion_is_not_invented() {
    for count in [0, 1, 3, usize::MAX] {
        let numbers = numerical().numbers(count).unwrap();
        assert_eq!(numbers.kind(), Some("regression"));
        assert!(numbers.number(0, TabularTaskNumber::Integral).is_err());
        assert!(numbers.exhausted(0).is_err());
        assert!(numbers.notice().is_none());
    }
    // A native object can report length two and yield zero, one or three values.
    for actual in 0..=3 {
        let mut numbers = numerical().numbers(2).unwrap();
        for ordinal in 0..actual {
            numbers = numbers
                .number(ordinal, TabularTaskNumber::Integral)
                .unwrap();
        }
        assert_eq!(numbers.kind(), None);
        let done = numbers.exhausted(actual).unwrap();
        assert_eq!(done.kind(), Some("classification"));
        assert_eq!(
            done.notice().unwrap().sentence,
            Sentence::TabularBinaryLabel
        );
        assert!(done.number(actual, TabularTaskNumber::Integral).is_err());
        assert!(done.exhausted(actual).is_err());
    }
}

#[test]
fn numeric_prefix_short_circuits_and_rejects_out_of_order_or_repeated_facts() {
    let initial = numerical().numbers(2).unwrap();
    assert!(initial.number(1, TabularTaskNumber::Boolean).is_err());
    assert!(initial.exhausted(1).is_err());
    let first = initial.number(0, TabularTaskNumber::Boolean).unwrap();
    assert!(first.number(0, TabularTaskNumber::Integral).is_err());
    let fractional = first
        .number(
            1,
            TabularTaskNumber::RealValue {
                bits: 0.5f64.to_le_bytes(),
            },
        )
        .unwrap();
    assert_eq!(fractional.kind(), Some("regression"));
    assert!(fractional.number(2, TabularTaskNumber::Integral).is_err());
    assert!(fractional.exhausted(2).is_err());
    assert!(fractional.notice().is_none());
}

#[test]
fn native_decimal_truth_and_real_failure_ancestry_are_completed_input_facts() {
    let initial = numerical().numbers(2).unwrap();
    for number in [
        TabularTaskNumber::Decimal {
            finite: true,
            equal: Some(true),
            returned: None,
        },
        TabularTaskNumber::Decimal {
            finite: false,
            equal: None,
            returned: Some(true),
        },
        TabularTaskNumber::RealValue {
            bits: (-0.0f64).to_le_bytes(),
        },
    ] {
        assert_eq!(initial.number(0, number).unwrap().kind(), None);
    }
    for number in [
        TabularTaskNumber::Decimal {
            finite: true,
            equal: Some(false),
            returned: None,
        },
        TabularTaskNumber::Decimal {
            finite: false,
            equal: None,
            returned: Some(false),
        },
        TabularTaskNumber::RealValue {
            bits: f64::INFINITY.to_le_bytes(),
        },
        TabularTaskNumber::RealValue {
            bits: f64::NAN.to_le_bytes(),
        },
        TabularTaskNumber::Other,
    ] {
        assert_eq!(
            initial.number(0, number).unwrap().kind(),
            Some("regression")
        );
    }
    assert!(
        initial
            .number(
                0,
                TabularTaskNumber::Decimal {
                    finite: false,
                    equal: Some(true),
                    returned: None
                }
            )
            .is_err()
    );
    for flags in [
        (true, false, false),
        (false, true, false),
        (false, false, true),
    ] {
        let caught = initial
            .number(
                0,
                TabularTaskNumber::RealFailure {
                    overflow_error: flags.0,
                    type_error: flags.1,
                    value_error: flags.2,
                },
            )
            .unwrap();
        assert_eq!(caught.kind(), Some("regression"));
        assert!(!caught.propagates());
    }
    let native = initial
        .number(
            0,
            TabularTaskNumber::RealFailure {
                overflow_error: false,
                type_error: false,
                value_error: false,
            },
        )
        .unwrap();
    assert!(native.propagates());
    assert_eq!(native.kind(), None);
    assert!(native.exhausted(1).is_err());
}
