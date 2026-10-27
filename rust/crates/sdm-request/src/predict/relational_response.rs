// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Gathered answers attributed to freshly read native source rows.
//!
//! Native owners remain in the host. Every answer carries its bound call and
//! source-row lineage, while bounds, occurrence, coverage and matching are
//! checked here before the next native instance display can be observed.

use std::collections::{BTreeMap, BTreeSet};

use sdm_contract::prepare::relational_response::*;
use sdm_contract::scalar::native_value::{NativeMember, NativeObjectKey, NativeValue};
use sdm_contract::scalar::unicode::UnicodeText;
use sdm_contract::{CoreError, Detail, Scalar, Sentence};
use sdm_wire::json::{Value, is_integer_token};
use sdm_wire::response::PredictionResponse;

use super::{Answer, AnswerPolicy, AnswerType, Attribution, Entity, Part};

fn protocol(message: &str) -> CoreError {
    CoreError::protocol(Sentence::InvalidMessage, message)
}

fn refusal(sentence: Sentence) -> CoreError {
    let mut error = CoreError::new(sentence, "the gathered answer cannot be attributed");
    error.public_details = Some(BTreeMap::new());
    error
}

fn count(value: usize) -> Detail {
    Detail::Scalar(Scalar::UInt64(u64::try_from(value).unwrap_or(u64::MAX)))
}

pub fn reference_prefix() -> &'static str {
    super::ENTITY_REF
}

/// Exact generated-decoder data, with no decimal display-limit observation.
pub fn native(value: &Value) -> Result<NativeValue, CoreError> {
    Ok(match value {
        Value::Null => NativeValue::Null,
        Value::Bool(value) => NativeValue::Boolean { value: *value },
        Value::Number(digits) if is_integer_token(digits) => NativeValue::Integer {
            digits: sdm_wire::response::integer_text(digits)
                .ok_or_else(|| protocol("decoded native integer is unreadable"))?,
        },
        Value::Number(token) => NativeValue::Float {
            bits: token
                .parse::<f64>()
                .map_err(|_| protocol("decoded native number is unreadable"))?
                .to_be_bytes(),
        },
        Value::Text(text) => NativeValue::Unicode {
            points: text.points().collect(),
        },
        Value::Array(items) => NativeValue::Array {
            items: items.iter().map(native).collect::<Result<_, _>>()?,
        },
        Value::Object(members) => NativeValue::Object {
            members: members
                .iter()
                .map(|(key, value)| {
                    Ok(NativeMember {
                        key: NativeObjectKey::Unicode {
                            points: key.chars().map(u32::from).collect(),
                        },
                        value: native(value)?,
                    })
                })
                .collect::<Result<_, CoreError>>()?,
        },
    })
}

fn text(points: &[u32]) -> Result<UnicodeText, CoreError> {
    UnicodeText::from_points(points.to_vec())
        .map_err(|_| protocol("native response text has an invalid Unicode point"))
}

pub fn logical(value: &NativeValue) -> Result<Value, CoreError> {
    Ok(match value {
        NativeValue::Null => Value::Null,
        NativeValue::Boolean { value } => Value::Bool(*value),
        NativeValue::Integer { digits } => {
            if sdm_wire::response::integer_text(digits).as_deref() != Some(digits.as_str()) {
                return Err(protocol("native response integer is not canonical"));
            }
            Value::Number(digits.clone())
        }
        NativeValue::Float { bits } => {
            let value = f64::from_be_bytes(*bits);
            Value::Number(if value.is_nan() {
                "NaN".into()
            } else if value == f64::INFINITY {
                "Infinity".into()
            } else if value == f64::NEG_INFINITY {
                "-Infinity".into()
            } else {
                // A native float must remain a float, including -0.0.
                let token = value.to_string();
                if token.contains('.') || token.contains('e') || token.contains('E') {
                    token
                } else {
                    format!("{token}.0")
                }
            })
        }
        NativeValue::Unicode { points } => Value::Text(text(points)?),
        NativeValue::Array { items } => {
            Value::Array(items.iter().map(logical).collect::<Result<_, _>>()?)
        }
        NativeValue::Object { members } => Value::Object(
            members
                .iter()
                .map(|member| {
                    let NativeObjectKey::Unicode { points } = &member.key;
                    let key = text(points)?;
                    Ok((
                        key.utf8()
                            .ok_or_else(|| {
                                protocol("decoded response member has exceptional text")
                            })?
                            .to_owned(),
                        logical(&member.value)?,
                    ))
                })
                .collect::<Result<_, CoreError>>()?,
        ),
    })
}

pub fn decode(
    response: &PredictionResponse,
    call_request_id: &str,
) -> Result<(RelationalResponseScope, Vec<DecodedRelationalRow>), CoreError> {
    let mut first = BTreeMap::new();
    let rows = response
        .predictions
        .iter()
        .enumerate()
        .map(|(ordinal, item)| {
            let row_index = item.row_index.to_string();
            let forecast_step = item.forecast_step.map(|step| step.to_string());
            let key = (row_index.clone(), forecast_step.clone());
            let prior_ordinal = first.get(&key).copied();
            first.entry(key).or_insert(ordinal);
            Ok(DecodedRelationalRow {
                ordinal,
                row_index,
                id: item.id.clone(),
                forecast_step,
                prior_ordinal,
                prediction: native(item.prediction.as_ref().unwrap_or(&Value::Null))?,
                probabilities: item.probabilities.as_ref().map(|probabilities| {
                    probabilities
                        .iter()
                        .map(|(key, value)| (key.clone(), Scalar::Float64(*value)))
                        .collect()
                }),
            })
        })
        .collect::<Result<_, CoreError>>()?;
    Ok((
        RelationalResponseScope {
            version: RELATIONAL_RESPONSE_VERSION,
            call_request_id: call_request_id.to_owned(),
            model: response.model.clone(),
            request_id: response.id.clone(),
            response_rows: response.predictions.len(),
        },
        rows,
    ))
}

fn scope(attribution: &RelationalAttribution) -> Result<(), CoreError> {
    if attribution.scope.version != RELATIONAL_RESPONSE_VERSION
        || (attribution.source_rows.is_none() && attribution.instances)
    {
        return Err(protocol("native attribution has incoherent source facts"));
    }
    Ok(())
}

fn checked(
    attribution: &RelationalAttribution,
    row: DecodedRelationalRow,
) -> Result<BoundRelationalRow, CoreError> {
    scope(attribution)?;
    if row.ordinal >= attribution.scope.response_rows
        || row.prior_ordinal.is_some_and(|prior| prior >= row.ordinal)
        || sdm_wire::response::integer_text(&row.row_index).as_deref()
            != Some(row.row_index.as_str())
    {
        return Err(protocol("decoded gathered row has incoherent ordinals"));
    }
    let source_row = match attribution.source_rows {
        None => {
            row.row_index
                .parse::<i64>()
                .map_err(|_| super::bad_response("prediction.row_index", "out_of_range"))?;
            if row
                .forecast_step
                .as_ref()
                .is_some_and(|step| step.parse::<i64>().is_err())
            {
                return Err(super::bad_response(
                    "prediction.forecast_step",
                    "out_of_range",
                ));
            }
            None
        }
        Some(rows) => {
            let index = row
                .row_index
                .parse::<usize>()
                .ok()
                .filter(|index| *index < rows)
                .ok_or_else(|| {
                    refusal(Sentence::RelationalRowBounds)
                        .with_field("index", Detail::text(&row.row_index))
                        .with_field("native_field", Detail::text("index"))
                        .with_field("rows", count(rows))
                })?;
            Some(index)
        }
    };
    Ok(BoundRelationalRow {
        attribution: attribution.clone(),
        row,
        source_row,
    })
}

pub fn reference_column(
    facts: RelationalReferenceColumns,
) -> Result<RelationalReferenceColumn, CoreError> {
    let ordinal = facts.prefix_matches.iter().position(|value| *value);
    if ordinal.is_some_and(|ordinal| ordinal + 1 != facts.prefix_matches.len())
        || (ordinal.is_none() && !facts.exhausted)
        || (ordinal.is_some() && facts.exhausted)
    {
        return Err(protocol(
            "native reference facts do not describe the reached prefix",
        ));
    }
    Ok(RelationalReferenceColumn { ordinal })
}

pub fn check_forecast(
    bound: BoundRelationalRow,
    forecasting: bool,
) -> Result<CheckedRelationalRow, CoreError> {
    if checked(&bound.attribution, bound.row.clone())? != bound || bound.source_row.is_none() {
        return Err(protocol("forecast truth has no bound native source row"));
    }
    let index = bound.source_row.unwrap_or(0);
    if forecasting && bound.row.forecast_step.is_none() {
        return Err(
            refusal(Sentence::RelationalForecastStepMissing).with_field("index", count(index))
        );
    }
    if !forecasting && bound.row.forecast_step.is_some() {
        return Err(refusal(Sentence::RelationalForecastStepUnexpected)
            .with_field("native_field", Detail::text("kind")));
    }
    if bound.row.prior_ordinal.is_some() {
        return Err(refusal(Sentence::RelationalRowRepeated)
            .with_field("native_field", Detail::text("where"))
            .with_field("index", count(index))
            .with_field(
                "step",
                Detail::text(bound.row.forecast_step.as_deref().unwrap_or("")),
            )
            .with_field("where_forecast", Detail::text("row {index} step {step}"))
            .with_field("where_static", Detail::text("row {index}")));
    }
    if bound.attribution.instances && bound.row.id.is_none() {
        return Err(refusal(Sentence::RelationalIdMissing)
            .with_field("native_field", Detail::text("expected"))
            .with_field("index", count(index)));
    }
    Ok(CheckedRelationalRow { bound, forecasting })
}

pub fn check_rows(
    attribution: RelationalAttribution,
    rows: Vec<DecodedRelationalRow>,
) -> Result<CheckedRelationalRows, CoreError> {
    scope(&attribution)?;
    let mut accepted = Vec::new();
    for row in rows {
        let checked = checked(&attribution, row)?;
        if checked.source_row.is_some() {
            return Ok(CheckedRelationalRows {
                rows: accepted,
                forecast: Some(checked),
            });
        }
        accepted.push(AttributedRelationalRow {
            row: checked.row,
            source_row: checked.source_row,
            instance: None,
            forecasting: None,
        });
    }
    Ok(CheckedRelationalRows {
        rows: accepted,
        forecast: None,
    })
}

pub fn without_instance(
    checked: CheckedRelationalRow,
) -> Result<AttributedRelationalRow, CoreError> {
    if check_forecast(checked.bound.clone(), checked.forecasting)? != checked
        || checked.bound.attribution.instances
    {
        return Err(protocol(
            "native instance attribution is required for this row",
        ));
    }
    Ok(AttributedRelationalRow {
        row: checked.bound.row,
        source_row: checked.bound.source_row,
        instance: None,
        forecasting: Some(checked.forecasting),
    })
}

pub fn attribute_instance(
    checked: CheckedRelationalRow,
    instance: RelationalInstanceText,
) -> Result<AttributedRelationalRow, CoreError> {
    if check_forecast(checked.bound.clone(), checked.forecasting)? != checked
        || !checked.bound.attribution.instances
        || instance.ordinal != checked.bound.row.ordinal
    {
        return Err(protocol(
            "native instance display names another gathered row",
        ));
    }
    text(&instance.points)?;
    if instance.different {
        return Err(refusal(Sentence::RelationalIdMismatch)
            .with_field("native_field", Detail::text("expected"))
            .with_field("index", count(checked.bound.source_row.unwrap_or(0)))
            .with_field(
                "id",
                Detail::text(checked.bound.row.id.as_deref().unwrap_or("")),
            ));
    }
    Ok(AttributedRelationalRow {
        row: checked.bound.row,
        source_row: checked.bound.source_row,
        instance: Some(instance),
        forecasting: Some(checked.forecasting),
    })
}

pub fn target_meaning(
    kind: &[u32],
    classification: bool,
) -> Result<RelationalTargetMeaning, CoreError> {
    text(kind)?;
    Ok(RelationalTargetMeaning {
        classification,
        numeric_prefixes: ["int", "float", "bool"].map(str::to_owned).to_vec(),
    })
}

pub fn coverage(
    attribution: &RelationalAttribution,
    rows: &[AttributedRelationalRow],
) -> Result<(), CoreError> {
    scope(attribution)?;
    if rows.len() != attribution.scope.response_rows {
        return Err(protocol("gathered attribution has another response length"));
    }
    let mut answered = BTreeSet::new();
    for (ordinal, accepted) in rows.iter().enumerate() {
        if accepted.row.ordinal != ordinal {
            return Err(protocol("gathered attribution is not in response order"));
        }
        let actual = checked(attribution, accepted.row.clone())?;
        if actual.source_row != accepted.source_row {
            return Err(protocol("gathered lineage differs from checked source row"));
        }
        let actual = match accepted.forecasting {
            Some(forecasting) => Some(check_forecast(actual, forecasting)?),
            None if attribution.source_rows.is_some() => {
                return Err(protocol("native forecast truth is absent"));
            }
            None => None,
        };
        match (&accepted.instance, actual) {
            (Some(instance), Some(actual)) => {
                if attribute_instance(actual, instance.clone())? != *accepted {
                    return Err(protocol("gathered instance differs from native display"));
                }
            }
            (None, Some(actual)) => {
                if without_instance(actual)? != *accepted {
                    return Err(protocol("native source attribution differs"));
                }
            }
            (None, None) if !attribution.instances => {}
            _ => return Err(protocol("native instance display is incoherent")),
        }
        if let Some(source) = accepted.source_row {
            answered.insert(source);
        }
    }
    if let Some(rows) = attribution.source_rows {
        let missing: Vec<_> = (0..rows)
            .filter(|index| !answered.contains(index))
            .collect();
        if !missing.is_empty() {
            return Err(refusal(Sentence::RelationalRowsUnanswered).with_field(
                "missing",
                Detail::List(
                    missing
                        .into_iter()
                        .map(|index| Scalar::UInt64(index as u64))
                        .collect(),
                ),
            ));
        }
    }
    Ok(())
}

/// Return the ordinary core Part and complete bound native lineage together.
/// The root preserves this companion while merging/numbering those Parts.
pub fn layout(
    response: &PredictionResponse,
    attribution: &RelationalAttribution,
    rows: &[AttributedRelationalRow],
    target: &RelationalTargetFacts,
    converted: &[RelationalTargetRow],
    float_text: fn(f64) -> String,
) -> Result<(Part, Vec<RelationalRowLineage>), CoreError> {
    let (actual_scope, decoded) = decode(response, &attribution.scope.call_request_id)?;
    if actual_scope != attribution.scope || rows.iter().map(|row| &row.row).ne(decoded.iter()) {
        return Err(protocol(
            "gathered attribution disagrees with decoded response",
        ));
    }
    coverage(attribution, rows)?;
    let meaning = target_meaning(&target.kind, target.classification)?;
    if (meaning.classification && target.dtype_truth.is_none())
        || (!meaning.classification
            && (target.dtype_truth.is_some() || target.numeric_prefix.is_some()))
        || (target.numeric_prefix.is_some() != (target.dtype_truth == Some(true)))
        || converted.len() != rows.len()
    {
        return Err(protocol(
            "native target facts disagree with reached task meaning",
        ));
    }
    let numeric = meaning.classification && target.numeric_prefix == Some(true);
    // The established gathered regression admission is separate from the
    // reached native classification cast. A custom string can skip that cast
    // without turning a classification response into a regression response.
    let regression = text(&target.kind)?.utf8() == Some("regression");
    let placed = rows
        .iter()
        .map(|accepted| {
            let row_index = accepted
                .row
                .row_index
                .parse::<i64>()
                .map_err(|_| super::bad_response("prediction.row_index", "out_of_range"))?;
            let step = accepted
                .row
                .forecast_step
                .as_ref()
                .map(|step| {
                    step.parse::<i64>().map_err(|_| {
                        super::bad_response("prediction.forecast_step", "out_of_range")
                    })
                })
                .transpose()?;
            // Source-row lineage is authoritative for native entities. A stable
            // ordinal placeholder carries no user value through Arrow or JSON.
            let id = accepted.source_row.map_or_else(
                || accepted.row.id.clone().map_or(Entity::None, Entity::Text),
                |source| Entity::Integer(i64::try_from(source).unwrap_or(i64::MAX)),
            );
            Ok((row_index, id, step))
        })
        .collect::<Result<Vec<_>, CoreError>>()?;
    let mut part = super::layout_placed(
        response,
        placed,
        true,
        if regression && !meaning.classification {
            AnswerType::Float
        } else {
            AnswerType::Text
        },
        Attribution::GatheredReference,
        AnswerPolicy::Gathered,
        float_text,
    )?;
    for ((row, decoded), converted) in part.rows.iter_mut().zip(&decoded).zip(converted) {
        if converted.ordinal != decoded.ordinal {
            return Err(protocol(
                "native target conversions are not in response order",
            ));
        }
        if !numeric
            && (converted.prediction != decoded.prediction
                || converted.probabilities != decoded.probabilities)
        {
            return Err(protocol("an unconverted target changed its decoded values"));
        }
        if meaning.classification {
            row.prediction = native_answer(&logical(&converted.prediction)?);
        }
        if numeric {
            row.probabilities = renamed_probabilities(
                decoded.probabilities.as_ref(),
                converted.probabilities.as_ref(),
            )?;
        }
    }
    let lineage = rows
        .iter()
        .map(|row| RelationalRowLineage {
            call_request_id: attribution.scope.call_request_id.clone(),
            answer_ordinal: row.row.ordinal,
            source_row: row.source_row,
        })
        .collect();
    Ok((part, lineage))
}

fn native_answer(value: &Value) -> Answer {
    match value {
        Value::Null => Answer::None,
        // Written retains the actual native scalar type even across mixed
        // classes; existing result table machinery owns its representation.
        other => Answer::Written(other.clone()),
    }
}

fn renamed_probabilities(
    original: Option<&Vec<(String, Scalar)>>,
    renamed: Option<&Vec<(String, Scalar)>>,
) -> Result<Option<Vec<(String, f64)>>, CoreError> {
    match (original, renamed) {
        (None, None) => Ok(None),
        (Some(original), Some(renamed)) if original.len() == renamed.len() => {
            let mut keys = BTreeSet::new();
            for ((_, value), (key, actual)) in original.iter().zip(renamed) {
                if value != actual {
                    return Err(protocol("native probability rename changed a probability"));
                }
                keys.insert(key);
            }
            let selected = if keys.len() == original.len() {
                renamed
            } else {
                original
            };
            Ok(Some(
                selected
                    .iter()
                    .map(|(key, value)| match value {
                        Scalar::Float64(value) if value.is_finite() => Ok((key.clone(), *value)),
                        _ => Err(protocol("decoded probability is not a finite double")),
                    })
                    .collect::<Result<_, _>>()?,
            ))
        }
        _ => Err(protocol("native target rename changed probability shape")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read_response(predictions: &str) -> PredictionResponse {
        let bytes = format!(
            r#"{{"id":"answer","model":"kumo-relational","predictions":{predictions},"metadata":{{}}}}"#
        );
        PredictionResponse::read(&sdm_wire::json::read_answer(bytes.as_bytes()).unwrap()).unwrap()
    }

    fn table_facts(
        response: &PredictionResponse,
        rows: Option<usize>,
        instances: bool,
    ) -> RelationalAttribution {
        RelationalAttribution {
            scope: decode(response, "actual-call").unwrap().0,
            source_rows: rows,
            instances,
        }
    }

    fn accepted(
        attribution: &RelationalAttribution,
        row: DecodedRelationalRow,
        forecasting: bool,
    ) -> AttributedRelationalRow {
        let checked = check_forecast(checked(attribution, row).unwrap(), forecasting).unwrap();
        if attribution.instances {
            let instance = RelationalInstanceText {
                ordinal: checked.bound.row.ordinal,
                different: false,
                points: checked
                    .bound
                    .row
                    .id
                    .as_deref()
                    .unwrap()
                    .chars()
                    .map(u32::from)
                    .collect(),
            };
            attribute_instance(checked, instance).unwrap()
        } else {
            without_instance(checked).unwrap()
        }
    }

    #[test]
    fn the_first_native_boundary_precedes_later_invalid_rows() {
        let response =
            read_response(r#"[{"row_index":0,"id":"first"},{"row_index":999,"id":"later"}]"#);
        let attribution = table_facts(&response, Some(1), true);
        let rows = decode(&response, "actual-call").unwrap().1;
        let frontier = check_rows(attribution.clone(), rows.clone()).unwrap();
        assert!(frontier.rows.is_empty());
        assert_eq!(frontier.forecast.unwrap().row.ordinal, 0);
        let first = accepted(&attribution, rows[0].clone(), false);
        assert_eq!(first.source_row, Some(0));
        let error = check_rows(attribution, vec![rows[1].clone()]).unwrap_err();
        assert_eq!(error.sentence, Sentence::RelationalRowBounds);
        assert_eq!(error.public_details, Some(BTreeMap::new()));
    }

    #[test]
    fn native_instance_text_is_matched_only_after_bounds_and_occurrence() {
        let response = read_response(r#"[{"row_index":0,"id":"server-id"}]"#);
        let attribution = table_facts(&response, Some(1), true);
        let row = decode(&response, "actual-call").unwrap().1.remove(0);
        let checked = check_forecast(checked(&attribution, row).unwrap(), false).unwrap();
        let error = attribute_instance(
            checked,
            RelationalInstanceText {
                ordinal: 0,
                points: "mutated-instance".chars().map(u32::from).collect(),
                different: true,
            },
        )
        .unwrap_err();
        assert_eq!(error.sentence, Sentence::RelationalIdMismatch);
        assert_eq!(
            error.fields.get("native_field"),
            Some(&Detail::text("expected"))
        );
        assert_eq!(error.public_details, Some(BTreeMap::new()));
    }

    #[test]
    fn forecasts_allow_distinct_steps_but_not_a_repeated_row_step() {
        let response = read_response(
            r#"[{"row_index":0,"forecast_step":1},{"row_index":0,"forecast_step":2}]"#,
        );
        let attribution = table_facts(&response, Some(1), false);
        let rows: Vec<_> = decode(&response, "actual-call")
            .unwrap()
            .1
            .into_iter()
            .map(|row| accepted(&attribution, row, true))
            .collect();
        coverage(&attribution, &rows).unwrap();
        let response = read_response(
            r#"[{"row_index":0,"forecast_step":1},{"row_index":0,"forecast_step":1}]"#,
        );
        let attribution = table_facts(&response, Some(1), false);
        let row = decode(&response, "actual-call").unwrap().1.remove(1);
        let error = check_forecast(checked(&attribution, row).unwrap(), true).unwrap_err();
        assert_eq!(error.sentence, Sentence::RelationalRowRepeated);
        // The formatter observes native forecast truth again only after
        // this exact error is selected, as the frozen helper does.
        assert_eq!(
            error.fields.get("native_field"),
            Some(&Detail::text("where"))
        );
    }

    #[test]
    fn coverage_precedes_target_dtype_reads_and_unkeyed_rows_stay_unchecked() {
        let response = read_response(r#"[{"row_index":0}]"#);
        let attribution = table_facts(&response, Some(2), false);
        let rows = vec![accepted(
            &attribution,
            decode(&response, "actual-call").unwrap().1.remove(0),
            false,
        )];
        assert_eq!(
            coverage(&attribution, &rows).unwrap_err().sentence,
            Sentence::RelationalRowsUnanswered
        );
        let response = read_response(
            r#"[{"row_index":900,"forecast_step":1},{"row_index":900,"forecast_step":1}]"#,
        );
        let attribution = table_facts(&response, None, false);
        let rows = check_rows(
            attribution.clone(),
            decode(&response, "actual-call").unwrap().1,
        )
        .unwrap();
        assert!(rows.forecast.is_none());
        assert!(
            rows.rows
                .iter()
                .all(|row| row.source_row.is_none() && row.forecasting.is_none())
        );
        coverage(&attribution, &rows.rows).unwrap();
    }

    #[test]
    fn layout_keeps_call_bound_source_lineage_and_native_target_scalars() {
        let response = read_response(r#"[{"row_index":0,"id":"opaque","prediction":"7"}]"#);
        let attribution = table_facts(&response, Some(1), true);
        let rows = vec![accepted(
            &attribution,
            decode(&response, "actual-call").unwrap().1.remove(0),
            false,
        )];
        let target = RelationalTargetFacts {
            kind: "classification".chars().map(u32::from).collect(),
            classification: true,
            dtype_truth: Some(true),
            numeric_prefix: Some(true),
        };
        let converted = vec![RelationalTargetRow {
            ordinal: 0,
            prediction: NativeValue::Integer { digits: "7".into() },
            probabilities: None,
        }];
        let (part, lineage) = layout(&response, &attribution, &rows, &target, &converted, |n| {
            n.to_string()
        })
        .unwrap();
        assert_eq!(
            lineage,
            vec![RelationalRowLineage {
                call_request_id: "actual-call".into(),
                answer_ordinal: 0,
                source_row: Some(0),
            }]
        );
        assert_eq!(
            part.rows[0].prediction,
            Answer::Written(Value::Number("7".into()))
        );
        // The actual entity (including HTTP-time 0 -> 7 mutation) is held
        // natively; no stale serialized key enters this core representation.
        assert_eq!(part.rows[0].id, Entity::Integer(0));
    }

    #[test]
    fn colliding_restored_probability_names_keep_the_original_classes() {
        let original = vec![
            ("false".into(), Scalar::Float64(0.25)),
            ("False".into(), Scalar::Float64(0.75)),
        ];
        let renamed = vec![
            ("False".into(), Scalar::Float64(0.25)),
            ("False".into(), Scalar::Float64(0.75)),
        ];
        assert_eq!(
            renamed_probabilities(Some(&original), Some(&renamed)).unwrap(),
            Some(vec![("false".into(), 0.25), ("False".into(), 0.75)])
        );
    }

    #[test]
    fn native_builtin_data_keeps_integer_values_and_unicode_points() {
        let value = Value::Array(vec![
            Value::Number("-0".into()),
            Value::Number("10000000000000000000000000000000000000000".into()),
            Value::Number("-0.0".into()),
            Value::Text(UnicodeText::from_points([0xd83d, 0xde3a]).unwrap()),
        ]);
        let native = native(&value).unwrap();
        let roundtrip = logical(&native).unwrap();
        let Value::Array(values) = roundtrip else {
            panic!("not an array")
        };
        assert_eq!(values[0], Value::Number("0".into()));
        assert_eq!(
            values[1],
            Value::Number("10000000000000000000000000000000000000000".into())
        );
        assert_eq!(values[2], Value::Number("-0.0".into()));
        let Value::Array(original) = value else {
            panic!("not an array")
        };
        assert_eq!(values[3], original[3]);
    }
}
