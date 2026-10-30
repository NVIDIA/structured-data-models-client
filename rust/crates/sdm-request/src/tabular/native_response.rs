// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Validation of separately reached native response fields.

use sdm_contract::direct::native_response::{CheckedResponseField, NativeResponseField, ResponseFault as F, ResponseField as S, ResponseObservation as O};
use sdm_contract::{CoreError, Sentence};

pub fn check(input: NativeResponseField) -> Result<CheckedResponseField, CoreError> {
    let field = input.field;
    let numeric = matches!(field, S::Probability | S::Score | S::Embedding | S::Quantile);
    let whole = matches!(field, S::RowIndex | S::ForecastStep);
    let is_null = matches!(&input.observation, O::OptionalNull { null: true })
        || field == S::ForecastStep && matches!(&input.observation, O::WholeKind { null: true, .. });
    let fault = match input.observation {
        O::OptionalNull { .. } if matches!(field, S::RowId | S::ForecastStep | S::Probabilities | S::Quantiles | S::Scores | S::Embeddings | S::RowMetadata | S::Explanation | S::Rankings) => None,
        O::Presence { present } if matches!(field, S::ResponseId | S::Model | S::Predictions | S::RowIndex | S::Metadata) => (!present).then_some(F::Missing),
        O::Object { mapping } if matches!(field, S::Response | S::Row | S::Probabilities | S::Quantiles | S::Metadata) => (!mapping).then_some(F::NotObject),
        O::Text { text } if matches!(field, S::ResponseId | S::Model | S::RowId) => (!text).then_some(F::NotText),
        O::WholeKind { boolean, number_family, null } if whole => {
            if null && field == S::RowIndex { Some(F::Null) }
            else if null && field == S::ForecastStep { None }
            else if boolean || !number_family { Some(F::NotWholeKind) }
            else { None }
        }
        O::WholeFloat { finite, integral } if whole => {
            if !finite && integral.is_some() || finite && integral.is_none() {
                return Err(CoreError::protocol(Sentence::InvalidMessage, "native whole-number comparison was observed at the wrong short circuit"));
            }
            (!finite || integral != Some(true)).then_some(F::NotWholeValue)
        }
        O::Sequence { string, bytes, mapping, iterable } if matches!(field, S::Predictions | S::Scores | S::Embeddings) => {
            (string || bytes || !iterable || field != S::Predictions && mapping).then_some(F::NotSequence)
        }
        O::NumberKind { boolean } if numeric => boolean.then_some(F::NotNumber),
        O::NumberConversion { bits } if numeric => match bits {
            None => Some(F::NotNumber),
            Some(bits) => (!f64::from_be_bytes(bits).is_finite()).then_some(F::NotFinite),
        },
        _ => return Err(CoreError::protocol(Sentence::InvalidMessage, "native response observation does not belong to its field")),
    };
    Ok(CheckedResponseField { field, fault, is_null })
}

pub fn restore(input: sdm_contract::direct::native_response::RestorationObservation) -> bool {
    use sdm_contract::direct::native_response::RestorationObservation;
    match input {
        RestorationObservation::Prediction { same } => !same,
        RestorationObservation::Classes { original_count, renamed_count } => original_count == renamed_count,
        RestorationObservation::Probabilities { different } => different,
    }
}


pub fn layout(input: sdm_contract::direct::native_response::NativeLayoutInput) -> Result<sdm_contract::direct::native_response::NativeLayoutResult, CoreError> {
    use std::collections::BTreeSet;
    use sdm_contract::direct::native_response::{NativeLayoutInput as I, NativeLayoutResult as R};
    use sdm_contract::direct::{NativeParsedValue, BuiltinFailureClass};
    use sdm_contract::{Detail, Scalar};
    let selected = |selected| R::Decision { selected };
    Ok(match input {
        I::Rows { is_list } => selected(is_list),
        I::KeyCount { count } => selected(count == 1),
        I::Positions { indices } => {
            let mut seen = BTreeSet::new();
            R::Positions { prior: indices.into_iter().map(|index| !seen.insert(index)).collect() }
        }
        I::Position { index, rows, prior } => R::Position { index: super::response::validate_position(&index, rows, prior)? },
        I::IdLength { length } => selected(length <= 1024),
        I::Parsed { json, value } => match value {
            NativeParsedValue::Failed { failure } => {
                let caught = if json { failure.ancestry.iter().any(|kind| matches!(kind, BuiltinFailureClass::ValueError | BuiltinFailureClass::TypeError | BuiltinFailureClass::RecursionError)) }
                    else { !failure.ancestry.is_empty() };
                if !caught {
                    return Err(CoreError::new(Sentence::DirectNativeFailure, "native conversion failed").with_field("stage", Detail::text(if json { "json" } else { "literal" })));
                }
                R::Parsed { literal: json, stringify: false }
            }
            NativeParsedValue::Value { value, members } => R::Parsed {
                literal: false, stringify: super::response::parsed_member(&value, &members, json)?.is_some(),
            },
        },
        I::Match { index, different } => {
            if different == Some(true) {
                return Err(CoreError::new(Sentence::DirectNativeIdMismatch, "native response identifies another row").with_field("index", Detail::Scalar(Scalar::UInt64(index as u64))));
            }
            R::Complete
        }
        I::Replacement { key_is_null } => selected(!key_is_null),
        I::Complete { indices, rows } => {
            let seen: BTreeSet<usize> = indices.into_iter().filter_map(|index| index.parse().ok()).collect();
            let missing: Vec<_> = (0..rows).filter(|index| !seen.contains(index)).map(|index| Scalar::UInt64(index as u64)).collect();
            if !missing.is_empty() {
                return Err(CoreError::new(Sentence::TabularRowsUnanswered, "native response left rows unanswered").with_field("missing", Detail::List(missing)));
            }
            R::Complete
        }
        I::TaskKind { points } => selected(points.windows(14).any(|text| text.iter().copied().eq("classification".chars().map(u32::from)))),
        I::TargetDtype { numeric } => selected(numeric),
    })
}
