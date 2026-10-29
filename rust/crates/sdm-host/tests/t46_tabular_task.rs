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

fn envelope_widths() -> sdm_contract::prepare::TabularEnvelopeWidths {
    use sdm_contract::prepare::{CALL_PREPARATION_VERSION, TABULAR_ENVELOPE_VERSION};
    serde_json::from_value(serde_json::json!({
        "version": TABULAR_ENVELOPE_VERSION,
        "bound": {
            "phase":"bound", "version":CALL_PREPARATION_VERSION,
            "model":"kumo-tabular", "outputs":["prediction"], "named":true,
            "shape":{"kind":"classification","anchored":false,"relationships":false,"tables":1},
            "kind":"classification", "context_rows":3
        },
        "dtypes":[["answer","int64"],["later","int64"]],
        "names":["answer","later"], "target":"answer"
    }))
    .unwrap()
}

#[test]
fn native_task_vocabulary_is_not_the_pql_two_task_rule_and_keeps_original_text() {
    use sdm_contract::prepare::{NativeConstructorConstant, TaskValue};
    let words = sdm_host::constructor_task_vocabulary();
    assert_eq!(words.tasks.len(), 7);
    assert_eq!(words.classification, words.tasks[..3]);
    assert_eq!(words.regression, "regression");
    assert_eq!(sdm_host::tabular_task_vocabulary().tasks.len(), 2);
    let task = TaskValue::Native {
        value: "native-choice".into(),
    };
    let notice = sdm_host::constructor_constant_notice(
        &task,
        Some(&NativeConstructorConstant {
            classification_first: true,
            regression_not_equal: None,
            classification_second: Some(false),
        }),
        "answer",
    )
    .unwrap()
    .unwrap();
    assert_eq!(notice.sentence, Sentence::DirectConstantRegression);
    assert_eq!(
        serde_json::to_value(&task).unwrap(),
        serde_json::json!({"value":"native-choice".chars().map(u32::from).collect::<Vec<_>>()})
    );
    assert!(sdm_host::constructor_constant_notice(&task, None, "answer").is_err());
    for facts in [
        NativeConstructorConstant {
            classification_first: true,
            regression_not_equal: Some(false),
            classification_second: Some(true),
        },
        NativeConstructorConstant {
            classification_first: false,
            regression_not_equal: None,
            classification_second: Some(true),
        },
        NativeConstructorConstant {
            classification_first: false,
            regression_not_equal: Some(true),
            classification_second: Some(false),
        },
    ] {
        assert_eq!(
            sdm_host::constructor_constant_notice(&task, Some(&facts), "answer")
                .unwrap_err()
                .sentence,
            Sentence::InvalidMessage
        );
    }
}

#[test]
fn complete_native_column_meaning_is_bound_to_actual_order_and_owner() {
    use sdm_contract::ops::Stype;
    let bytes = serde_json::to_vec(&envelope_widths()).unwrap();
    let mut columns = sdm_host::tabular_envelope::columns(&bytes).unwrap();
    let foreign = sdm_host::tabular_envelope::columns(&bytes).unwrap();
    assert!(columns.finish().is_err());
    assert!(columns.declaration("later".into(), Stype::Text).is_err());
    let target = columns
        .declaration("answer".into(), Stype::Numerical)
        .unwrap();
    assert!(!target.text());
    assert!(target.meaning(None, None).is_err());
    let actual = target.meaning(Some(false), Some(true)).unwrap();
    let other = foreign
        .declaration("answer".into(), Stype::Numerical)
        .unwrap()
        .meaning(Some(false), Some(true))
        .unwrap();
    assert!(columns.complete(&other, None).is_err());
    assert!(columns.complete(&actual, Some(false)).is_err());
    columns.complete(&actual, None).unwrap();
    assert!(columns.complete(&actual, None).is_err());
    // A task comparison changed the actual later native declaration after
    // widths were captured. Its reached Text shortcut does not read task.
    let later = columns.declaration("later".into(), Stype::Text).unwrap();
    assert!(later.text());
    assert!(later.meaning(None, Some(true)).is_err());
    columns
        .complete(&later.meaning(None, None).unwrap(), None)
        .unwrap();
    let (finished, notices) = columns.finish().unwrap();
    assert_eq!(finished.columns.columns[0].stype, Stype::Categorical);
    assert_eq!(finished.columns.columns[1].stype, Stype::Categorical);
    assert_eq!(notices.len(), 1);
    assert_eq!(notices[0].sentence, Sentence::DirectTextCategorical);
    let (encoded, held) = sdm_host::tabular_envelope::finish(&columns).unwrap();
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&encoded).unwrap(),
        serde_json::to_value(finished).unwrap()
    );
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&held)
            .unwrap()
            .as_array()
            .unwrap()
            .len(),
        1
    );
}

#[test]
fn text_target_has_no_task_fact_and_incoherent_widths_are_not_accepted() {
    use sdm_contract::ops::Stype;
    let mut data = envelope_widths();
    data.names = vec!["answer".into()];
    let mut columns =
        sdm_host::tabular_envelope::columns(&serde_json::to_vec(&data).unwrap()).unwrap();
    let target = columns.declaration("answer".into(), Stype::Text).unwrap();
    assert!(target.meaning(None, Some(false)).is_err());
    columns
        .complete(&target.meaning(None, None).unwrap(), None)
        .unwrap();
    assert_eq!(
        columns.finish().unwrap().0.columns.columns[0].stype,
        Stype::Categorical
    );
    data.names.push("answer".into());
    assert!(sdm_host::tabular_envelope::columns(&serde_json::to_vec(&data).unwrap()).is_err());
    data.names = vec!["answer".into(), "absent".into()];
    assert!(sdm_host::tabular_envelope::columns(&serde_json::to_vec(&data).unwrap()).is_err());
}

#[test]
fn native_target_identity_can_change_between_ordered_column_meanings() {
    use sdm_contract::ops::Stype;
    let bytes = serde_json::to_vec(&envelope_widths()).unwrap();
    let mut columns = sdm_host::tabular_envelope::columns(&bytes).unwrap();
    for name in ["answer", "later"] {
        let declared = columns.declaration(name.into(), Stype::Numerical).unwrap();
        assert!(declared.meaning(None, Some(true)).is_err());
        assert!(declared.meaning(Some(true), Some(true)).is_err());
        let meaning = declared.meaning(Some(false), Some(true)).unwrap();
        columns.complete(&meaning, None).unwrap();
    }
    let envelope = columns.finish().unwrap().0;
    assert_eq!(envelope.columns.target, "answer");
    assert!(
        envelope
            .columns
            .columns
            .iter()
            .all(|column| column.stype == Stype::Categorical)
    );

    let text = sdm_host::tabular_envelope::columns(&bytes)
        .unwrap()
        .declaration("answer".into(), Stype::Text)
        .unwrap();
    assert!(text.meaning(Some(false), None).is_err());
    assert!(text.meaning(None, Some(true)).is_err());
    assert!(text.meaning(None, None).is_ok());
}
