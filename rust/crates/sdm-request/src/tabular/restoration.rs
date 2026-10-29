// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Completed native dtype conversions bound to a complete direct response.
//! Native owners and conversion timing stay in the host; source population,
//! ordering, probability conservation, collision policy and layout stay here.

use crate::native_value::{logical, native};
use crate::predict::{Answer, Part};
use sdm_contract::prepare::{
    DirectTargetRows, RelationalTargetProbability, RelationalTargetRow, ResponseScope,
};
use sdm_contract::{CoreError, Scalar, Sentence};
use sdm_wire::json::Value;
use sdm_wire::response::PredictionResponse;
use std::collections::BTreeSet;

fn protocol(message: &str) -> CoreError {
    CoreError::protocol(Sentence::InvalidMessage, message)
}

/// Decode the actual generated-response values without a native decimal
/// conversion/display or another JSON round trip through a host parser.
pub fn source(
    response: &PredictionResponse,
    scope: &ResponseScope,
) -> Result<DirectTargetRows, CoreError> {
    let rows = response
        .predictions
        .iter()
        .enumerate()
        .map(|(ordinal, item)| {
            let probabilities = item
                .probabilities
                .as_ref()
                .map(|given| {
                    given
                        .iter()
                        .enumerate()
                        .map(|(ordinal, (key, probability))| {
                            Ok(RelationalTargetProbability {
                                ordinal,
                                value: native(&Value::text(key))?,
                                name: key.clone(),
                                probability: Scalar::Float64(*probability),
                            })
                        })
                        .collect::<Result<Vec<_>, CoreError>>()
                })
                .transpose()?;
            Ok(RelationalTargetRow {
                ordinal,
                prediction: native(item.prediction.as_ref().unwrap_or(&Value::Null))?,
                probabilities,
            })
        })
        .collect::<Result<Vec<_>, CoreError>>()?;
    Ok(DirectTargetRows {
        scope: scope.clone(),
        rows,
    })
}

/// Completed casts are actual builtin values, never another dtype alias.
/// Every probability still names its original source ordinal and value.
pub fn apply(
    part: &mut Part,
    response: &PredictionResponse,
    scope: &ResponseScope,
    converted: &DirectTargetRows,
) -> Result<(), CoreError> {
    if converted.scope != *scope
        || converted.rows.len() != response.predictions.len()
        || part.rows.len() != converted.rows.len()
    {
        return Err(protocol(
            "native direct restoration disagrees with its complete response scope",
        ));
    }
    for (ordinal, ((row, original), converted)) in part
        .rows
        .iter_mut()
        .zip(&response.predictions)
        .zip(&converted.rows)
        .enumerate()
    {
        if converted.ordinal != ordinal {
            return Err(protocol(
                "native direct restoration rows are not in response order",
            ));
        }
        let prediction = logical(&converted.prediction)?;
        row.prediction = if prediction == Value::Null {
            Answer::None
        } else {
            Answer::Written(prediction)
        };
        row.probabilities = match (&original.probabilities, &converted.probabilities) {
            (None, None) => None,
            (Some(original), Some(converted)) if original.len() == converted.len() => {
                let mut renamed = Vec::with_capacity(original.len());
                for (ordinal, ((_, probability), converted)) in
                    original.iter().zip(converted).enumerate()
                {
                    if converted.ordinal != ordinal
                        || converted.probability != Scalar::Float64(*probability)
                    {
                        return Err(protocol(
                            "native direct probability conversion changed source order or probability",
                        ));
                    }
                    // Validate canonical builtin storage even for a key that
                    // collision policy later replaces with its original name.
                    logical(&converted.value)?;
                    renamed.push((converted.name.clone(), *probability));
                }
                let keys: BTreeSet<_> = renamed.iter().map(|(key, _)| key).collect();
                if keys.len() != renamed.len() {
                    for ((name, _), (original, _)) in renamed.iter_mut().zip(original) {
                        *name = original.clone();
                    }
                }
                Some(renamed)
            }
            _ => {
                return Err(protocol(
                    "native direct probability conversion changed its source population",
                ));
            }
        };
    }
    Ok(())
}
