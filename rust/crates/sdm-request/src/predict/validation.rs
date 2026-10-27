// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared relational preflight policy over independently observed facts.
//! Native traversal/equality and JSON decoding supply facts at their own
//! reached boundaries; neither source determines limits or warning policy.

use super::*;

pub fn context_empty(empty: bool, model: &str) -> Result<(), CoreError> {
    if empty {
        return Err(
            unsupported(Sentence::NoContextRows, model, "no context rows are left")
                .with_field("context_rows", count(0)),
        );
    }
    Ok(())
}

pub fn class_count(classes: usize, model: &str) -> Result<(), CoreError> {
    if classes > CLASSES_MAX {
        return Err(unsupported(
            Sentence::TooManyClasses,
            model,
            "the target has more classes than the model takes",
        )
        .with_field("classes", count(classes))
        .with_field("limit", count(CLASSES_MAX)));
    }
    Ok(())
}

pub fn variation(
    kind: &str,
    column: &str,
    answered_classes: usize,
    query: &str,
    model: &str,
) -> Result<Option<Notice>, CoreError> {
    if answered_classes != 1 || (!classifies(kind) && kind != "regression") {
        return Ok(None);
    }
    let constant = constant_context(column, query);
    if classifies(kind) {
        let sentence = match constant.sentence {
            Sentence::ContextConstantColumn => Sentence::OneClassColumn,
            Sentence::ContextConstantPredicted => Sentence::OneClassPredicted,
            Sentence::ContextConstantQuestion => Sentence::OneClassQuestion,
            _ => Sentence::OneClassAnswer,
        };
        let mut error = unsupported(sentence, model, "every context row has the same answer")
            .with_field("field", Detail::text("context"))
            .with_field("reason", Detail::text("one_class"));
        for (name, value) in constant.fields {
            error = error.with_field(name, value);
        }
        return Err(error.with_field("classes_in_context", count(1)));
    }
    Ok(Some(constant))
}

pub fn explanation_size(scored: usize, model: &str) -> Result<(), CoreError> {
    if scored != EXPLANATION_PREDICT_ROWS {
        return Err(unsupported(
            Sentence::ExplanationOneEntity,
            model,
            "an explanation covers one entity",
        )
        .with_field("field", Detail::text("outputs"))
        .with_field("remedy", Detail::text("one_entity"))
        .with_field("scored", count(scored)));
    }
    Ok(())
}

pub fn explanation_mode(larger_context: bool, model: &str) -> Result<(), CoreError> {
    if larger_context {
        return Err(unsupported(
            Sentence::ExplanationFastMode,
            model,
            "an explanation is given only in the fast run mode",
        )
        .with_field("field", Detail::text("outputs"))
        .with_field("remedy", Detail::text("smaller_context"))
        .with_field("context_rows_fast", count(CONTEXT_ROWS_FAST)));
    }
    Ok(())
}
