// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::native_guard::{assess, report};
use super::*;
use sdm_contract::prepare::{BuiltinFailureClass, NativeFailure};

fn facts(classes: Vec<BuiltinFailureClass>) -> FrameNativeFailure {
    FrameNativeFailure {
        builtin: NativeFailure { ancestry: classes },
        key_error: false,
        structured_data_error: false,
        relational_error: false,
        deepest_module: Some("native.library".into()),
        python_unicode_database: "14.0.0".into(),
    }
}

fn text(value: &str) -> Vec<u32> {
    value.chars().map(u32::from).collect()
}

#[test]
fn structured_and_unguarded_failures_are_preserved_without_text() {
    let mut failure = facts(vec![BuiltinFailureClass::TypeError]);
    failure.structured_data_error = true;
    assert_eq!(
        assess(FrameErrorOrigin::Engine, failure.clone(), None)
            .unwrap()
            .category,
        FrameNativeCategory::PassThrough
    );
    failure.structured_data_error = false;
    assert_eq!(
        assess(FrameErrorOrigin::Unguarded, failure, None)
            .unwrap()
            .category,
        FrameNativeCategory::PassThrough
    );
}

#[test]
fn signature_classification_preserves_deepest_module_short_circuit() {
    let mut failure = facts(vec![BuiltinFailureClass::TypeError]);
    failure.deepest_module = Some("sdm_client.materializers.relational".into());
    assert_eq!(
        assess(FrameErrorOrigin::Engine, failure.clone(), None)
            .unwrap()
            .category,
        FrameNativeCategory::CallSignature
    );
    assert!(assess(FrameErrorOrigin::Engine, failure, Some(text("unreached"))).is_err());
    let failure = facts(vec![BuiltinFailureClass::TypeError]);
    assert_eq!(
        assess(FrameErrorOrigin::Engine, failure.clone(), None)
            .unwrap()
            .category,
        FrameNativeCategory::TypeError
    );
    assert_eq!(
        assess(
            FrameErrorOrigin::Engine,
            failure.clone(),
            Some(text("unexpected keyword argument"))
        )
        .unwrap()
        .category,
        FrameNativeCategory::CallSignature
    );
    assert_eq!(
        assess(
            FrameErrorOrigin::Engine,
            failure,
            Some(text("native value failed"))
        )
        .unwrap()
        .category,
        FrameNativeCategory::CallerType
    );
}

#[test]
fn refusal_and_final_said_remain_distinct_and_use_pinned_native_lowering() {
    let assessment = assess(
        FrameErrorOrigin::Engine,
        facts(vec![BuiltinFailureClass::ValueError]),
        None,
    )
    .unwrap();
    let error = report(FrameNativeReport {
        assessment,
        native_type: None,
        // Kelvin-sign lowercasing is a native Unicode operation, while the
        // final said value remains a distinct actual observation.
        refusal: Some(text("we would need data bacK to")),
        said: vec![0xd800, u32::from('!')],
        dropped_summary: Some("[users: 1 of 8 rows dropped (1 with no time)]".into()),
        rows_per_call: Some(2),
    })
    .unwrap();
    assert_eq!(error.sentence, Sentence::FrameNativeCaller);
    assert_eq!(
        error.public_details.unwrap().get("window_outran_the_data"),
        Some(&Detail::Scalar(Scalar::Bool(true)))
    );
    assert_eq!(
        error.fields.get("message_points"),
        Some(&Detail::List(vec![
            Scalar::UInt32(0xd800),
            Scalar::UInt32(u32::from('!'))
        ]))
    );
}

#[test]
fn type_errors_do_not_append_graph_drop_detail() {
    let assessment = assess(
        FrameErrorOrigin::Engine,
        facts(vec![BuiltinFailureClass::TypeError]),
        Some(text("bad native value")),
    )
    .unwrap();
    let error = report(FrameNativeReport {
        assessment,
        native_type: None,
        refusal: Some(text("bad native value")),
        said: text("bad value again"),
        dropped_summary: Some("must not append".into()),
        rows_per_call: None,
    })
    .unwrap();
    assert_eq!(error.fields.get("detail"), Some(&Detail::text("")));
}

#[test]
fn native_part_truth_is_reached_only_after_structured_pass_through() {
    let scope = FrameQuestionScope {
        kind: FrameQuestionKind::Forecast,
        position: 2,
        parts: 3,
    };
    let mut failure = facts(vec![]);
    assert!(
        native_guard::assess_in_part(
            FrameErrorOrigin::Engine,
            failure.clone(),
            None,
            Some(&scope),
            Some(true)
        )
        .is_err()
    );
    failure.structured_data_error = true;
    let first = native_guard::assess_in_part(
        FrameErrorOrigin::Engine,
        failure.clone(),
        None,
        Some(&scope),
        None,
    )
    .unwrap();
    assert_eq!(first.category, FrameNativeCategory::PassThrough);
    assert!(first.part.is_none());
    let reached = native_guard::assess_in_part(
        FrameErrorOrigin::Engine,
        failure,
        None,
        Some(&scope),
        Some(true),
    )
    .unwrap()
    .part
    .unwrap();
    assert_eq!(reached.sentence, Sentence::PartForecastStepOutran);
    assert_eq!(
        reached.fields["forecast_step"],
        Detail::Scalar(Scalar::UInt32(2))
    );
}
