// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared relational preflight policy over independently observed facts.
//! Native traversal/equality and JSON decoding supply facts at their own
//! reached boundaries; neither source determines limits or warning policy.

use super::*;
use sdm_contract::prepare::{RelationalTargetNaming, RelationalTargetRole, RelationalTaskMeaning};
use sdm_contract::scalar::unicode::UnicodeText;

pub fn wire_target_name() -> Vec<u32> {
    WIRE_TARGET.chars().map(u32::from).collect()
}

pub fn task_meaning(kind: Vec<u32>) -> Result<RelationalTaskMeaning, CoreError> {
    let value = UnicodeText::from_points(kind.iter().copied()).map_err(|_| {
        CoreError::protocol(Sentence::InvalidMessage, "native task has invalid Unicode points")
    })?;
    let classification = value
        .as_bytes()
        .windows(b"classification".len())
        .any(|window| window == b"classification");
    let regression = value.utf8() == Some("regression");
    Ok(RelationalTaskMeaning {
        kind,
        role: if classification {
            RelationalTargetRole::Classification
        } else if regression {
            RelationalTargetRole::Regression
        } else {
            RelationalTargetRole::Other
        },
        summarized: regression || value.utf8() == Some("forecasting"),
    })
}

pub fn target_naming(wire_target: bool, query: &UnicodeText) -> RelationalTargetNaming {
    if !wire_target {
        return RelationalTargetNaming::Column;
    }
    // Only an ordinary prefix can match the existing identifier grammar.
    // The shared matcher requires a following whitespace character, so an
    // exceptional point cannot become an invented end-of-input boundary.
    let prefix = match std::str::from_utf8(query.as_bytes()) {
        Ok(text) => text,
        Err(error) => std::str::from_utf8(&query.as_bytes()[..error.valid_up_to()])
            .expect("UTF-8 error identifies a valid prefix"),
    };
    if let Some(value) = predicted_column(prefix) {
        RelationalTargetNaming::Predicted { value: value.to_owned() }
    } else if query.is_empty() {
        RelationalTargetNaming::Answer
    } else {
        RelationalTargetNaming::Question
    }
}

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
    constant(classifies(kind), constant_context(column, query), model)
}

pub fn constant(
    classification: bool,
    constant: Notice,
    model: &str,
) -> Result<Option<Notice>, CoreError> {
    if classification {
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

pub fn native_constant(
    role: RelationalTargetRole,
    naming: &RelationalTargetNaming,
    display: Option<&UnicodeText>,
    model: &str,
) -> Result<Option<Notice>, CoreError> {
    let points = |name, value: &UnicodeText| {
        (name, Detail::List(value.points().map(|point| Scalar::UInt32(point)).collect()))
    };
    let mut fields = vec![("target", Detail::text(WIRE_TARGET))];
    let (variant, sentence) = match naming {
        RelationalTargetNaming::Column => {
            let value = display.ok_or_else(|| CoreError::protocol(Sentence::InvalidMessage, "a native column name has no display observation"))?;
            fields.push(points("target_points", value));
            ("column", Sentence::ContextConstantColumn)
        }
        RelationalTargetNaming::Predicted { value } => {
            fields.push(("predicted", Detail::text(value)));
            ("predicted", Sentence::ContextConstantPredicted)
        }
        RelationalTargetNaming::Question => {
            let value = display.ok_or_else(|| CoreError::protocol(Sentence::InvalidMessage, "a native question has no repr observation"))?;
            fields.push(("question", Detail::text("")));
            fields.push(points("question_repr_points", value));
            ("question", Sentence::ContextConstantQuestion)
        }
        RelationalTargetNaming::Answer => ("answer", Sentence::ContextConstantAnswer),
    };
    if !matches!(naming, RelationalTargetNaming::Column | RelationalTargetNaming::Question) && display.is_some() {
        return Err(CoreError::protocol(Sentence::InvalidMessage, "native display supplied for a name that does not consume one"));
    }
    fields.push(("variant", Detail::text(variant)));
    fields.push(("native_relational_constant", Detail::Scalar(Scalar::Bool(true))));
    constant(role == RelationalTargetRole::Classification, Notice {
        sentence,
        message: "every context row has the same answer",
        fields,
    }, model)
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
