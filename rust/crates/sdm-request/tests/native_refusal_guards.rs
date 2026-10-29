// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Intended-site consumers distinguish observed membership from model policy.

use sdm_contract::Sentence;
use sdm_request::predict::{self, validation};

#[test]
fn native_output_membership_refuses_only_valid_selected_positions() {
    let outputs = vec!["prediction".into(), "unknown".into()];
    predict::validate_native_outputs(predict::TABULAR, &outputs, &[]).unwrap();
    assert_eq!(
        predict::validate_native_outputs(predict::TABULAR, &outputs, &[1])
            .unwrap_err()
            .sentence,
        Sentence::ModelLacksOutputs
    );
    for positions in [vec![2], vec![1, 1], vec![1, 0]] {
        assert_eq!(
            predict::validate_native_outputs(predict::TABULAR, &outputs, &positions)
                .unwrap_err()
                .sentence,
            Sentence::InvalidMessage
        );
    }
}

#[test]
fn shared_task_shape_and_capacity_keep_their_distinct_refusals() {
    let outputs = vec!["prediction".into()];
    predict::validate_task_outputs(predict::TABULAR, "regression", &outputs).unwrap();
    assert_eq!(
        predict::validate_native_task_outputs(
            predict::TABULAR,
            "regression",
            &outputs,
            Some(false),
        )
        .unwrap_err()
        .sentence,
        Sentence::ModelLacksTask
    );
    assert_eq!(
        predict::validate_task_outputs(predict::TABULAR, "regression", &["unknown".into()])
            .unwrap_err()
            .sentence,
        Sentence::ModelLacksOutputs
    );
    predict::validate_shape(predict::TABULAR, "regression", false, false, 1).unwrap();
    assert_eq!(
        predict::validate_shape(predict::TABULAR, "regression", false, false, 2)
            .unwrap_err()
            .sentence,
        Sentence::ModelLacksShape
    );
    predict::validate_capacity(predict::TABULAR, predict::CONTEXT_ROWS_MAX).unwrap();
    assert_eq!(
        predict::validate_capacity(predict::TABULAR, predict::CONTEXT_ROWS_MAX + 1)
            .unwrap_err()
            .sentence,
        Sentence::ContextPastModel
    );
}

#[test]
fn relocated_preflight_policies_keep_their_boundary_twins() {
    validation::context_empty(false, predict::RELATIONAL).unwrap();
    assert_eq!(
        validation::context_empty(true, predict::RELATIONAL)
            .unwrap_err()
            .sentence,
        Sentence::NoContextRows
    );
    validation::class_count(predict::CLASSES_MAX, predict::RELATIONAL).unwrap();
    assert_eq!(
        validation::class_count(predict::CLASSES_MAX + 1, predict::RELATIONAL)
            .unwrap_err()
            .sentence,
        Sentence::TooManyClasses
    );
    validation::explanation_size(1, predict::RELATIONAL).unwrap();
    assert_eq!(
        validation::explanation_size(2, predict::RELATIONAL)
            .unwrap_err()
            .sentence,
        Sentence::ExplanationOneEntity
    );
    validation::explanation_mode(false, predict::RELATIONAL).unwrap();
    assert_eq!(
        validation::explanation_mode(true, predict::RELATIONAL)
            .unwrap_err()
            .sentence,
        Sentence::ExplanationFastMode
    );
}
