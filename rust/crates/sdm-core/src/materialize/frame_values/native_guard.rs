// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native error classification and the original frame guard's policy.
//! Native objects and observations remain in their presentation host.

use sdm_contract::prepare::BuiltinFailureClass;
use sdm_contract::scalar::unicode::UnicodeText;

use super::*;

fn unicode(points: &[u32]) -> Result<UnicodeText, CoreError> {
    UnicodeText::from_points(points.iter().copied())
        .map_err(|_| protocol("native error observation contains an invalid Unicode point"))
}

fn contains(text: &UnicodeText, needle: &str) -> bool {
    text.as_bytes()
        .windows(needle.len())
        .any(|part| part == needle.as_bytes())
}

pub(super) fn assess(
    origin: FrameErrorOrigin,
    failure: FrameNativeFailure,
    signature: Option<Vec<u32>>,
) -> Result<FrameFailureAssessment, CoreError> {
    use FrameNativeCategory as C;
    let type_error = failure
        .builtin
        .ancestry
        .contains(&BuiltinFailureClass::TypeError);
    let value_error = failure
        .builtin
        .ancestry
        .contains(&BuiltinFailureClass::ValueError);
    let short_circuit =
        failure.deepest_module.as_deref() == Some("sdm_client.materializers.relational");
    let category = if origin == FrameErrorOrigin::Unguarded || failure.structured_data_error {
        C::PassThrough
    } else if type_error && short_circuit {
        C::CallSignature
    } else if type_error {
        match &signature {
            None => C::TypeError,
            Some(points) => {
                let message = unicode(points)?;
                if [
                    "unexpected keyword argument",
                    "positional argument",
                    "required argument",
                    "takes no arguments",
                ]
                .iter()
                .any(|phrase| contains(&message, phrase))
                {
                    C::CallSignature
                } else {
                    C::CallerType
                }
            }
        }
    } else if value_error || failure.key_error || failure.relational_error {
        C::CallerData
    } else {
        C::Internal
    };
    if signature.is_some() && (!type_error || short_circuit || matches!(category, C::PassThrough)) {
        return Err(protocol(
            "native signature text was observed at an unreached error boundary",
        ));
    }
    Ok(FrameFailureAssessment {
        origin,
        failure,
        signature,
        category,
        part: None,
    })
}

/// A structured native exception reaches its part wrapper only after the
/// original guard has selected pass-through. Native truth is observed then,
/// and the same shared error policy selects the existing part sentence.
pub(super) fn assess_in_part(
    origin: FrameErrorOrigin,
    failure: FrameNativeFailure,
    signature: Option<Vec<u32>>,
    scope: Option<&FrameQuestionScope>,
    window_outran: Option<bool>,
) -> Result<FrameFailureAssessment, CoreError> {
    let mut assessment = assess(origin, failure, signature)?;
    let reached = scope.filter(|_| {
        assessment.category == FrameNativeCategory::PassThrough
            && assessment.failure.structured_data_error
    });
    let selection = match (reached, window_outran) {
        (Some(scope), None) if scope.kind == FrameQuestionKind::Ranking => Some((scope, false)),
        (Some(scope), Some(window)) if scope.kind == FrameQuestionKind::Forecast => {
            Some((scope, window))
        }
        (_, Some(_)) => {
            return Err(protocol(
                "native part truth was observed before its reached forecast error boundary",
            ));
        }
        _ => None,
    };
    if let Some((scope, window)) = selection {
        let placeholder = CoreError::new(Sentence::Internal, "native error part meaning")
            .with_field(
                "window_outran_the_data",
                Detail::Scalar(Scalar::Bool(window)),
            );
        let selected = questions::in_part(placeholder, Some(scope));
        assessment.part = Some(FramePartPresentation {
            sentence: selected
                .part
                .ok_or_else(|| internal("native part meaning lacks its sentence"))?,
            fields: selected.fields,
        });
    }
    Ok(assessment)
}

fn measured(said: UnicodeText) -> UnicodeText {
    let needle = b"Please reduce either";
    let Some(at) = said
        .as_bytes()
        .windows(needle.len())
        .position(|part| part == needle)
    else {
        return said;
    };
    let prefix = UnicodeText::from_utf8_surrogatepass(said.as_bytes()[..at].to_vec())
        .expect("an ASCII match begins on a Unicode boundary");
    let mut points = prefix.points().collect::<Vec<_>>();
    while points
        .last()
        .is_some_and(|&point| sdm_engine::words::native_space(point))
    {
        points.pop();
    }
    UnicodeText::from_points(points).expect("a prefix keeps valid native points")
}

fn holds(rows: Option<u64>) -> String {
    match rows.filter(|rows| *rows != 0) {
        None => "number".into(),
        Some(rows) => {
            let digits = rows.to_string();
            digits
                .as_bytes()
                .rchunks(3)
                .rev()
                .map(|digits| std::str::from_utf8(digits).expect("integer digits are ASCII"))
                .collect::<Vec<_>>()
                .join(",")
        }
    }
}

/// Select one ordinary CoreError from the complete actual observations.
/// Returning an error value keeps this classifier independently inspectable;
/// fixed-operation dispatch then ends through the existing error channel.
pub(super) fn report(input: FrameNativeReport) -> Result<CoreError, CoreError> {
    use FrameNativeCategory as C;
    let assessment = assess(
        input.assessment.origin,
        input.assessment.failure.clone(),
        input.assessment.signature.clone(),
    )?;
    if assessment != input.assessment {
        return Err(protocol(
            "native error report changed its diagnostic classification",
        ));
    }
    let mut said = unicode(&input.said)?;
    let mut details = BTreeMap::new();
    let mut detail = String::new();
    let sentence = match assessment.category {
        C::PassThrough | C::TypeError => {
            return Err(protocol(
                "native error report has no final guarded classification",
            ));
        }
        C::Internal | C::CallSignature => {
            if input.native_type.is_none() || input.refusal.is_some() {
                return Err(protocol(
                    "internal native error observations do not match their reached positions",
                ));
            }
            if assessment.category == C::Internal {
                Sentence::FrameNativeInternal
            } else {
                Sentence::FrameNativeSignature
            }
        }
        C::CallerData | C::CallerType => {
            if input.native_type.is_some() {
                return Err(protocol(
                    "caller native error report unexpectedly observed its class name",
                ));
            }
            let refusal = unicode(input.refusal.as_ref().ok_or_else(|| {
                protocol("caller native error report lacks its distinct refusal observation")
            })?)?;
            let profile = casing_profile(&NativeTextFacts {
                target: NativeStringTarget::NumpyUnicode,
                python_unicode_database: assessment.failure.python_unicode_database.clone(),
                arrow_simple_identity: None,
            })?;
            let lowered = sdm_engine::words::lower_native(refusal.as_slice(), profile);
            if assessment.category == C::CallerData {
                if let Some(summary) = &input.dropped_summary {
                    detail = format!(" {summary}");
                }
            }
            if [
                "aggregation time range",
                "we would need data back to",
                "anchor timestamp is too early",
            ]
            .iter()
            .any(|phrase| contains(&lowered, phrase))
            {
                details.insert(
                    "window_outran_the_data".into(),
                    Detail::Scalar(Scalar::Bool(true)),
                );
                Sentence::FrameNativeCaller
            } else if contains(&lowered, "size exceeds the 30mb limit") {
                details.insert(
                    "batch_too_large_to_send".into(),
                    Detail::Scalar(Scalar::Bool(true)),
                );
                let several = contains(&said, "the number of entities predicted at once");
                said = measured(said);
                if several {
                    Sentence::FrameNativeBatchSmaller
                } else {
                    Sentence::FrameNativeBatchSingle
                }
            } else {
                Sentence::FrameNativeCaller
            }
        }
    };
    let mut error = CoreError::new(sentence, "native frame engine failure")
        .with_field(
            "message_points",
            Detail::List(said.points().map(Scalar::UInt32).collect()),
        )
        .with_field("detail", Detail::text(detail))
        .with_field("holds", Detail::text(holds(input.rows_per_call)));
    if let Some(name) = input.native_type {
        error = error.with_field("native_type", Detail::text(name));
    }
    error.public_details = Some(details);
    Ok(error)
}
