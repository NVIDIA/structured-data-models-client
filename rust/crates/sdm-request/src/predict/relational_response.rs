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
use sdm_contract::scalar::native_value::{
    NativeObjectKey, NativeTreeMember, NativeTreeNode, NativeValue,
};
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
    error.public_details = Some(Box::default());
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
        Value::Array(_) | Value::Object(_) => native_tree(value)?,
    })
}

/// Flatten actual decoded containers without doubling their JSON nesting.
/// This is a postorder value tree, not a parser or a native action protocol.
fn native_tree(value: &Value) -> Result<NativeValue, CoreError> {
    enum Visit<'a> {
        Value(&'a Value),
        Array(usize),
        Object(&'a [(String, Value)]),
    }
    let mut pending = vec![Visit::Value(value)];
    let mut roots = Vec::new();
    let mut nodes = Vec::new();
    while let Some(visit) = pending.pop() {
        let node =
            match visit {
                Visit::Value(Value::Array(items)) => {
                    pending.push(Visit::Array(items.len()));
                    pending.extend(items.iter().rev().map(Visit::Value));
                    continue;
                }
                Visit::Value(Value::Object(members)) => {
                    pending.push(Visit::Object(members));
                    pending.extend(members.iter().rev().map(|(_, value)| Visit::Value(value)));
                    continue;
                }
                Visit::Value(value) => NativeTreeNode::Scalar {
                    value: native(value)?,
                },
                Visit::Array(length) => NativeTreeNode::Array {
                    children: roots.split_off(roots.len().checked_sub(length).ok_or_else(
                        || protocol("decoded native array has inconsistent child data"),
                    )?),
                },
                Visit::Object(members) => {
                    let children =
                        roots.split_off(roots.len().checked_sub(members.len()).ok_or_else(
                            || protocol("decoded native object has inconsistent member data"),
                        )?);
                    NativeTreeNode::Object {
                        members: members
                            .iter()
                            .zip(children)
                            .map(|((key, _), value)| NativeTreeMember {
                                key: NativeObjectKey::Unicode {
                                    points: key.chars().map(u32::from).collect(),
                                },
                                value,
                            })
                            .collect(),
                    }
                }
            };
        let ordinal = u32::try_from(nodes.len())
            .map_err(|_| protocol("decoded native tree exceeds local node ordinals"))?;
        nodes.push(node);
        roots.push(ordinal);
    }
    Ok(NativeValue::Tree { nodes })
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
        NativeValue::Tree { nodes } => logical_tree(nodes)?,
    })
}

fn logical_tree(nodes: &[NativeTreeNode]) -> Result<Value, CoreError> {
    if nodes
        .last()
        .is_none_or(|node| matches!(node, NativeTreeNode::Scalar { .. }))
    {
        return Err(protocol(
            "native container tree lacks its final container root",
        ));
    }
    let mut values: Vec<Option<Value>> = Vec::with_capacity(nodes.len());
    let mut depths = Vec::with_capacity(nodes.len());
    let mut starts = Vec::with_capacity(nodes.len());
    for (ordinal, node) in nodes.iter().enumerate() {
        let children = match node {
            NativeTreeNode::Scalar { value } => {
                if matches!(
                    value,
                    NativeValue::Array { .. }
                        | NativeValue::Object { .. }
                        | NativeValue::Tree { .. }
                ) {
                    return Err(protocol(
                        "native tree scalar contains another container owner",
                    ));
                }
                values.push(Some(logical(value)?));
                depths.push(0usize);
                starts.push(ordinal);
                continue;
            }
            NativeTreeNode::Array { children } => children.clone(),
            NativeTreeNode::Object { members } => {
                members.iter().map(|member| member.value).collect()
            }
        };
        if children.windows(2).any(|pair| pair[0] >= pair[1])
            || children.iter().any(|child| *child as usize >= ordinal)
        {
            return Err(protocol(
                "native tree children are not an ordered earlier population",
            ));
        }
        let start = children
            .first()
            .map_or(ordinal, |child| starts[*child as usize]);
        let mut next = start;
        for child in &children {
            if starts[*child as usize] != next {
                return Err(protocol(
                    "native tree interleaves separate child populations",
                ));
            }
            next = *child as usize + 1;
        }
        if next != ordinal {
            return Err(protocol(
                "native tree omits an intervening child population",
            ));
        }
        let depth = 1 + children
            .iter()
            .map(|child| depths[*child as usize])
            .max()
            .unwrap_or(0);
        if depth > sdm_wire::json::MAX_DEPTH {
            return Err(protocol(
                "native tree exceeds the existing logical value depth",
            ));
        }
        let items = children
            .iter()
            .map(|child| {
                values[*child as usize].take().ok_or_else(|| {
                    protocol("native tree reuses a consumed child or mixes container owners")
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let value = match node {
            NativeTreeNode::Array { .. } => Value::Array(items),
            NativeTreeNode::Object { members } => Value::Object(
                members
                    .iter()
                    .zip(items)
                    .map(|(member, value)| {
                        let NativeObjectKey::Unicode { points } = &member.key;
                        let key = text(points)?;
                        let key = key.utf8().ok_or_else(|| {
                            protocol("decoded native member has exceptional text")
                        })?;
                        Ok((key.to_owned(), value))
                    })
                    .collect::<Result<_, CoreError>>()?,
            ),
            NativeTreeNode::Scalar { .. } => unreachable!("scalar handled before child reads"),
        };
        values.push(Some(value));
        depths.push(depth);
        starts.push(start);
    }
    let root = values
        .pop()
        .flatten()
        .ok_or_else(|| protocol("native tree has no final root"))?;
    if values.iter().any(Option::is_some) {
        return Err(protocol("native tree contains unreachable node owners"));
    }
    Ok(root)
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
            && (target.dtype_truth.is_some()
                || target.numeric_prefix.is_some()
                || target.dtype.is_some()))
        || (target.numeric_prefix.is_some() != (target.dtype_truth == Some(true)))
        || converted.len() != rows.len()
    {
        return Err(protocol(
            "native target facts disagree with reached task meaning",
        ));
    }
    let numeric = meaning.classification && target.numeric_prefix == Some(true);
    let declared = super::declared_class_answers(target.dtype.as_ref().map(UnicodeText::as_bytes));
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
        } else if meaning.classification {
            declared
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
        if !numeric && converted.prediction != decoded.prediction {
            return Err(protocol("an unconverted target changed its decoded values"));
        }
        if meaning.classification {
            row.prediction = gathered_class_answer(
                &decoded.prediction,
                &converted.prediction,
                declared,
                numeric,
            )?;
        }
        row.probabilities = renamed_probabilities(
            decoded.probabilities.as_ref(),
            converted.probabilities.as_ref(),
            declared,
            numeric,
            float_text,
        )?;
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

/// Actual completed numeric products select the corresponding existing
/// Gathered kernel; unchanged strings and containers retain declared meaning.
fn converted_family(value: &NativeValue) -> Option<AnswerType> {
    match value {
        NativeValue::Boolean { .. } => Some(AnswerType::Bool),
        NativeValue::Integer { .. } => Some(AnswerType::Int),
        NativeValue::Float { .. } => Some(AnswerType::FloatClass),
        _ => None,
    }
}

fn gathered_class_answer(
    raw: &NativeValue,
    converted: &NativeValue,
    declared: AnswerType,
    numeric: bool,
) -> Result<Answer, CoreError> {
    let original = logical(raw)?;
    let native = logical(converted)?;
    // A numeric carrier that the native cast left unchanged is still a raw
    // value under its declaration, not evidence of another selected family.
    let family = if numeric && raw != converted {
        converted_family(converted)
    } else {
        None
    };
    let strict = super::answer_of(
        Some(&original),
        family.unwrap_or(declared),
        AnswerPolicy::Gathered,
    )?;
    if let Some(family) = family
        && !matches!(
            strict,
            Answer::Written(_) | Answer::None | Answer::NonFinite(_)
        )
    {
        // Cross-declaration products use the existing actual-value
        // inference and declared heterogeneous fallback. Matching typed
        // successes must remain visible in the primary Arrow column.
        return if family == declared {
            Ok(match converted {
                NativeValue::Boolean { value } => Answer::Bool(*value),
                NativeValue::Integer { digits } => digits
                    .parse::<i64>()
                    .map(Answer::Int)
                    .unwrap_or(Answer::Written(native)),
                NativeValue::Float { bits } => Answer::Float(f64::from_be_bytes(*bits)),
                _ => unreachable!("a selected numeric family has a numeric native product"),
            })
        } else {
            Ok(Answer::Written(native))
        };
    }
    Ok(strict)
}

fn renamed_probabilities(
    original: Option<&Vec<(String, Scalar)>>,
    renamed: Option<&Vec<RelationalTargetProbability>>,
    declared: AnswerType,
    numeric: bool,
    float_text: fn(f64) -> String,
) -> Result<Option<Vec<(String, f64)>>, CoreError> {
    match (original, renamed) {
        (None, None) => Ok(None),
        (Some(original), Some(renamed)) if original.len() == renamed.len() => {
            let mut selected = Vec::with_capacity(original.len());
            for (ordinal, ((key, value), actual)) in original.iter().zip(renamed).enumerate() {
                if actual.ordinal != ordinal {
                    return Err(protocol(
                        "native probability conversions are not in source order",
                    ));
                }
                if value != &actual.probability {
                    return Err(protocol("native probability rename changed a probability"));
                }
                let raw = Value::text(key);
                if !numeric && (actual.value != native(&raw)? || actual.name != *key) {
                    return Err(protocol(
                        "an unconverted probability changed its decoded value",
                    ));
                }
                // Validate the complete native product even when strict
                // admission ultimately selects the original decoded key.
                logical(&actual.value)?;
                let family = if numeric {
                    converted_family(&actual.value)
                } else {
                    None
                };
                let family = family.unwrap_or(declared);
                let strict = super::answer_of(Some(&raw), family, AnswerPolicy::Gathered)?;
                let name = if matches!(strict, Answer::Written(_) | Answer::Text(_)) {
                    key.clone()
                } else if numeric && converted_family(&actual.value).is_some() {
                    actual.name.clone()
                } else {
                    super::spelled_as(
                        &[(key.clone(), 0.0)],
                        family,
                        float_text,
                        AnswerPolicy::Gathered,
                    )[0]
                    .0
                    .clone()
                };
                let Scalar::Float64(probability) = value else {
                    return Err(protocol("decoded probability is not a finite double"));
                };
                if !probability.is_finite() {
                    return Err(protocol("decoded probability is not a finite double"));
                }
                selected.push((name, *probability));
            }
            let keys: BTreeSet<_> = selected.iter().map(|(key, _)| key).collect();
            if keys.len() != selected.len() {
                for ((key, _), (original, _)) in selected.iter_mut().zip(original) {
                    *key = original.clone();
                }
            }
            Ok(Some(selected))
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

    fn class_target(dtype: &str, numeric_prefix: bool) -> RelationalTargetFacts {
        RelationalTargetFacts {
            kind: "classification".chars().map(u32::from).collect(),
            classification: true,
            dtype_truth: Some(true),
            numeric_prefix: Some(numeric_prefix),
            dtype: Some(dtype.into()),
        }
    }

    #[test]
    fn gathered_class_policy_reads_raw_values_before_completed_native_casts() {
        for (dtype, raw, converted, expected) in [
            ("int64", r#""1_0""#, "10", r#""1_0""#),
            ("int64", r#""١٠""#, "10", r#""١٠""#),
            ("int64", "9.21e18", "9210000000000000000", "9.21e18"),
            ("int64", "-9.21e18", "-9210000000000000000", "-9.21e18"),
            ("int64", r#""10""#, "10", "10"),
            (
                "int64",
                "9.19e18",
                "9190000000000000000",
                "9190000000000000000",
            ),
            ("float64", r#""1_0.5""#, "10.5", r#""1_0.5""#),
            ("float64", r#""١٠.5""#, "10.5", r#""١٠.5""#),
            ("float64", r#""10.5""#, "10.5", "10.5"),
            (
                "bool",
                r#""\u001ctrue\u001c""#,
                "true",
                r#""\u001ctrue\u001c""#,
            ),
            ("bool", r#""false""#, "false", "false"),
        ] {
            let response = read_response(&format!(
                r#"[{{"row_index":0,"id":"opaque","prediction":{raw}}}]"#
            ));
            let attribution = table_facts(&response, None, false);
            let rows = vec![accepted(
                &attribution,
                decode(&response, "actual-call").unwrap().1.remove(0),
                false,
            )];
            let converted = vec![RelationalTargetRow {
                ordinal: 0,
                prediction: native(&sdm_wire::json::read_answer(converted.as_bytes()).unwrap())
                    .unwrap(),
                probabilities: None,
            }];
            let (part, _) = layout(
                &response,
                &attribution,
                &rows,
                &class_target(dtype, true),
                &converted,
                |n| n.to_string(),
            )
            .unwrap();
            let expected = sdm_wire::json::read_answer(expected.as_bytes()).unwrap();
            let expected = logical(&native(&expected).unwrap()).unwrap();
            assert_eq!(
                part.rows[0].prediction,
                super::super::answer_of(
                    Some(&expected),
                    super::super::declared_class_answers(Some(dtype.as_bytes())),
                    AnswerPolicy::Gathered
                )
                .unwrap(),
                "{dtype}: {raw}"
            );
        }
    }

    #[test]
    fn reached_declaration_distinguishes_identical_unconverted_bool_and_text_facts() {
        let response = read_response(r#"[{"row_index":0,"id":"opaque","prediction":"true"}]"#);
        let attribution = table_facts(&response, None, false);
        let rows = vec![accepted(
            &attribution,
            decode(&response, "actual-call").unwrap().1.remove(0),
            false,
        )];
        let converted = vec![RelationalTargetRow {
            ordinal: 0,
            prediction: NativeValue::Unicode {
                points: "true".chars().map(u32::from).collect(),
            },
            probabilities: None,
        }];
        for (dtype, expected) in [("bool", Value::Bool(true)), ("string", Value::text("true"))] {
            let (part, _) = layout(
                &response,
                &attribution,
                &rows,
                &class_target(dtype, false),
                &converted,
                |n| n.to_string(),
            )
            .unwrap();
            assert_eq!(
                part.rows[0].prediction,
                if dtype == "bool" {
                    Answer::Bool(true)
                } else {
                    Answer::Text(expected.as_text().unwrap().to_owned())
                }
            );
            assert_eq!(
                part.answers,
                if dtype == "bool" {
                    AnswerType::Bool
                } else {
                    AnswerType::Text
                }
            );
        }
        let mut unreached = class_target("bool", false);
        unreached.classification = false;
        unreached.dtype_truth = None;
        unreached.numeric_prefix = None;
        assert_eq!(
            layout(
                &response,
                &attribution,
                &rows,
                &unreached,
                &converted,
                |n| n.to_string()
            )
            .unwrap_err()
            .sentence,
            Sentence::InvalidMessage
        );
        let mut invalid = serde_json::to_value(class_target("bool", false)).unwrap();
        invalid["dtype"] = serde_json::json!([0x110000]);
        assert!(serde_json::from_value::<RelationalTargetFacts>(invalid).is_err());
    }

    #[test]
    fn strict_probability_selection_precedes_collisions_and_keeps_native_products_bound() {
        let response = read_response(
            r#"[{"row_index":0,"id":"opaque","prediction":"7","probabilities":{"1_0":0.25,"010":0.75}}]"#,
        );
        let attribution = table_facts(&response, None, false);
        let rows = vec![accepted(
            &attribution,
            decode(&response, "actual-call").unwrap().1.remove(0),
            false,
        )];
        let target = class_target("int64", true);
        let entry = |ordinal, probability| RelationalTargetProbability {
            ordinal,
            value: NativeValue::Integer {
                digits: "10".into(),
            },
            name: "10".into(),
            probability: Scalar::Float64(probability),
        };
        let mut converted = vec![RelationalTargetRow {
            ordinal: 0,
            prediction: NativeValue::Integer { digits: "7".into() },
            probabilities: Some(vec![entry(0, 0.25), entry(1, 0.75)]),
        }];
        let (part, _) = layout(&response, &attribution, &rows, &target, &converted, |n| {
            n.to_string()
        })
        .unwrap();
        assert_eq!(
            part.rows[0].probabilities,
            Some(vec![("1_0".into(), 0.25), ("10".into(), 0.75)])
        );
        converted[0].probabilities.as_mut().unwrap().reverse();
        assert_eq!(
            layout(&response, &attribution, &rows, &target, &converted, |n| n
                .to_string())
            .unwrap_err()
            .sentence,
            Sentence::InvalidMessage
        );
        converted[0].probabilities.as_mut().unwrap().pop();
        assert_eq!(
            layout(&response, &attribution, &rows, &target, &converted, |n| n
                .to_string())
            .unwrap_err()
            .sentence,
            Sentence::InvalidMessage
        );
    }

    #[test]
    fn mixed_gathered_classes_keep_primary_arrow_cells_and_exact_raw_fallbacks() {
        use arrow_array::Array;
        use arrow_array::cast::AsArray;
        use arrow_array::types::{Float64Type, Int64Type};
        use arrow_schema::DataType;

        for (dtype, raw, converted, physical, expected) in [
            (
                "bool",
                r#"["maybe",true]"#,
                r#"["maybe",true]"#,
                DataType::Boolean,
                vec![None, Some("true")],
            ),
            (
                "int64",
                r#"["1_0","10"]"#,
                "[10,10]",
                DataType::Int64,
                vec![None, Some("10")],
            ),
            (
                "float64",
                r#"["1_0.5","2.5"]"#,
                "[10.5,2.5]",
                DataType::Float64,
                vec![None, Some("2.5")],
            ),
            (
                "string",
                r#"[1,"two"]"#,
                r#"[1,"two"]"#,
                DataType::Utf8,
                vec![None, Some("two")],
            ),
            (
                "bool",
                "[2,1.0]",
                "[2,1.0]",
                DataType::Float64,
                vec![Some("2"), Some("1")],
            ),
        ] {
            let raw = sdm_wire::json::read_answer(raw.as_bytes()).unwrap();
            let Value::Array(raw) = raw else {
                panic!("raw rows")
            };
            let Value::Array(native_rows) =
                sdm_wire::json::read_answer(converted.as_bytes()).unwrap()
            else {
                panic!("native rows")
            };
            let predictions = Value::Array(
                raw.iter()
                    .enumerate()
                    .map(|(ordinal, value)| {
                        Value::Object(vec![
                            ("row_index".into(), Value::Number(ordinal.to_string())),
                            ("id".into(), Value::text("opaque")),
                            ("prediction".into(), value.clone()),
                        ])
                    })
                    .collect(),
            );
            let response = read_response(std::str::from_utf8(&predictions.to_bytes()).unwrap());
            let attribution = table_facts(&response, None, false);
            let rows = decode(&response, "actual-call")
                .unwrap()
                .1
                .into_iter()
                .map(|row| accepted(&attribution, row, false))
                .collect::<Vec<_>>();
            let converted = native_rows
                .iter()
                .enumerate()
                .map(|(ordinal, value)| RelationalTargetRow {
                    ordinal,
                    prediction: native(value).unwrap(),
                    probabilities: None,
                })
                .collect::<Vec<_>>();
            let (part, _) = layout(
                &response,
                &attribution,
                &rows,
                &class_target(dtype, dtype != "string"),
                &converted,
                |n| n.to_string(),
            )
            .unwrap();
            let table = super::super::table(&part).unwrap();
            assert_eq!(
                table
                    .schema()
                    .fields()
                    .iter()
                    .map(|field| field.name().as_str())
                    .collect::<Vec<_>>(),
                [
                    "row_index",
                    "id",
                    "prediction",
                    "prediction_as_answered",
                    "answer_index"
                ]
            );
            let prediction = table.column_by_name("prediction").unwrap();
            assert_eq!(prediction.data_type(), &physical, "{dtype}");
            let values = (0..prediction.len())
                .map(|row| {
                    if prediction.is_null(row) {
                        return None;
                    }
                    Some(match physical {
                        DataType::Boolean => prediction.as_boolean().value(row).to_string(),
                        DataType::Int64 => prediction
                            .as_primitive::<Int64Type>()
                            .value(row)
                            .to_string(),
                        DataType::Float64 => prediction
                            .as_primitive::<Float64Type>()
                            .value(row)
                            .to_string(),
                        DataType::Utf8 => prediction.as_string::<i32>().value(row).to_owned(),
                        _ => panic!("unexpected physical family"),
                    })
                })
                .collect::<Vec<_>>();
            assert_eq!(
                values,
                expected
                    .into_iter()
                    .map(|value| value.map(str::to_owned))
                    .collect::<Vec<_>>(),
                "{dtype}"
            );
            let fallback = table
                .column_by_name("prediction_as_answered")
                .unwrap()
                .as_string::<i32>();
            for (at, value) in raw.iter().enumerate() {
                let actual = sdm_wire::json::read_answer(fallback.value(at).as_bytes()).unwrap();
                let expected = if prediction.is_null(at) {
                    value.clone()
                } else {
                    logical(&converted[at].prediction).unwrap()
                };
                assert_eq!(actual, expected, "{dtype} fallback row {at}");
            }
        }
    }

    #[test]
    fn custom_numeric_products_keep_declared_fallback_and_negative_zero_bits() {
        use arrow_array::Array;
        use arrow_array::cast::AsArray;
        use arrow_array::types::{Float64Type, Int64Type};
        use arrow_schema::DataType;

        for (raw, converted, homogeneous) in [
            (r#"["-0.0","1.5"]"#, "[-0.0,1.5]", true),
            (r#"["-0.0","1_0.5"]"#, "[-0.0,10.5]", false),
            (r#"["-0.0","2","1_0.5",null]"#, "[-0.0,2,10.5,null]", false),
        ] {
            let Value::Array(raw) = sdm_wire::json::read_answer(raw.as_bytes()).unwrap() else {
                panic!("raw")
            };
            let Value::Array(converted) =
                sdm_wire::json::read_answer(converted.as_bytes()).unwrap()
            else {
                panic!("converted")
            };
            let predictions = raw
                .iter()
                .enumerate()
                .map(|(at, raw)| {
                    format!(
                        r#"{{"row_index":{at},"prediction":{}}}"#,
                        std::str::from_utf8(&raw.to_bytes()).unwrap()
                    )
                })
                .collect::<Vec<_>>()
                .join(",");
            let response = read_response(&format!("[{predictions}]"));
            let attribution = table_facts(&response, None, false);
            let rows = decode(&response, "actual-call")
                .unwrap()
                .1
                .into_iter()
                .map(|row| accepted(&attribution, row, false))
                .collect::<Vec<_>>();
            let converted = converted
                .iter()
                .enumerate()
                .map(|(ordinal, value)| RelationalTargetRow {
                    ordinal,
                    prediction: native(value).unwrap(),
                    probabilities: None,
                })
                .collect::<Vec<_>>();
            let (part, _) = layout(
                &response,
                &attribution,
                &rows,
                &class_target("int64", true),
                &converted,
                |n| n.to_string(),
            )
            .unwrap();
            assert_eq!(part.answers, AnswerType::Int);
            assert_eq!(
                part.rows[0].prediction,
                Answer::Written(Value::Number("-0.0".into()))
            );
            let table = super::super::table(&part).unwrap();
            let primary = table.column_by_name("prediction").unwrap();
            if homogeneous {
                assert_eq!(primary.data_type(), &DataType::Float64);
                let values = primary.as_primitive::<Float64Type>();
                assert_eq!(values.value(0).to_bits(), (-0.0_f64).to_bits());
                assert_eq!(values.value(1), 1.5);
                assert!(table.column_by_name("prediction_as_answered").is_none());
            } else {
                assert_eq!(primary.data_type(), &DataType::Int64);
                let values = primary.as_primitive::<Int64Type>();
                for at in 0..values.len() {
                    if raw.len() == 4 && at == 1 {
                        assert_eq!(values.value(at), 2);
                        assert!(!values.is_null(at));
                    } else {
                        assert!(values.is_null(at));
                    }
                }
                let fallback = table
                    .column_by_name("prediction_as_answered")
                    .unwrap()
                    .as_string::<i32>();
                assert_eq!(fallback.value(0), "-0.0");
                if raw.len() == 4 {
                    assert_eq!(fallback.value(1), "2");
                    assert_eq!(fallback.value(2), r#""1_0.5""#);
                    assert!(fallback.is_null(3));
                } else {
                    assert_eq!(fallback.value(1), r#""1_0.5""#);
                }
            }
        }
    }

    #[test]
    fn equal_probability_products_bind_to_original_entry_ordinals() {
        let response = read_response(
            r#"[{"row_index":0,"id":"opaque","prediction":"7","probabilities":{"1":0.5,"2":0.5}}]"#,
        );
        let attribution = table_facts(&response, None, false);
        let rows = vec![accepted(
            &attribution,
            decode(&response, "actual-call").unwrap().1.remove(0),
            false,
        )];
        let entries = (0..2)
            .map(|ordinal| RelationalTargetProbability {
                ordinal,
                value: NativeValue::Integer {
                    digits: (ordinal + 1).to_string(),
                },
                name: (ordinal + 1).to_string(),
                probability: Scalar::Float64(0.5),
            })
            .collect::<Vec<_>>();
        let original = RelationalTargetRow {
            ordinal: 0,
            prediction: NativeValue::Integer { digits: "7".into() },
            probabilities: Some(entries),
        };
        let target = class_target("int64", true);
        let (part, _) = layout(
            &response,
            &attribution,
            &rows,
            &target,
            std::slice::from_ref(&original),
            |n| n.to_string(),
        )
        .unwrap();
        assert_eq!(
            part.rows[0].probabilities,
            Some(vec![("1".into(), 0.5), ("2".into(), 0.5)])
        );
        for mode in 0..3 {
            let mut changed = original.clone();
            let entries = changed.probabilities.as_mut().unwrap();
            match mode {
                0 => entries.reverse(),
                1 => entries[1] = entries[0].clone(),
                _ => {
                    entries.pop();
                }
            }
            assert_eq!(
                layout(&response, &attribution, &rows, &target, &[changed], |n| n
                    .to_string())
                .unwrap_err()
                .sentence,
                Sentence::InvalidMessage
            );
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
        assert_eq!(error.public_details.as_deref(), Some(&BTreeMap::new()));
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
        assert_eq!(error.public_details.as_deref(), Some(&BTreeMap::new()));
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
            dtype: Some("int64".into()),
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
        assert_eq!(part.rows[0].prediction, Answer::Int(7));
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
            RelationalTargetProbability {
                ordinal: 0,
                value: NativeValue::Boolean { value: false },
                name: "False".into(),
                probability: Scalar::Float64(0.25),
            },
            RelationalTargetProbability {
                ordinal: 1,
                value: NativeValue::Boolean { value: false },
                name: "False".into(),
                probability: Scalar::Float64(0.75),
            },
        ];
        assert_eq!(
            renamed_probabilities(
                Some(&original),
                Some(&renamed),
                AnswerType::Bool,
                true,
                |n| n.to_string()
            )
            .unwrap(),
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

    #[test]
    fn native_response_containers_do_not_multiply_json_transport_depth() {
        for levels in [1, 90, 125] {
            let mut value = Value::text("retained-leaf");
            for _ in 0..levels {
                value = Value::Array(vec![value]);
            }
            let encoded = native(&value).unwrap();
            let bytes = serde_json::to_vec(&encoded).unwrap();
            // Ordinary serde remains sufficient for the intermediate, not
            // an operation-specific escape from its recursion limit.
            let decoded: NativeValue = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(logical(&decoded).unwrap(), value);
        }
        let mixed = Value::Object(vec![
            (
                "second".into(),
                Value::Array(vec![Value::Bool(true), Value::Null]),
            ),
            ("first".into(), Value::Object(vec![])),
        ]);
        assert_eq!(logical(&native(&mixed).unwrap()).unwrap(), mixed);
    }

    #[test]
    fn original_response_depth_boundary_is_unchanged() {
        for (levels, accepted) in [(125, true), (126, false)] {
            let mut prediction = "\"retained-leaf\"".to_owned();
            for _ in 0..levels {
                prediction = format!("[{prediction}]");
            }
            let response = format!(
                "{{\"id\":\"depth\",\"model\":\"kumo-relational\",\"metadata\":{{}},\"predictions\":[{{\"row_index\":0,\"id\":\"3\",\"prediction\":{prediction}}}]}}"
            );
            assert_eq!(
                sdm_wire::json::read_answer(response.as_bytes()).is_ok(),
                accepted
            );
        }
    }

    #[test]
    fn flat_container_children_are_one_local_ordered_tree() {
        let scalar = || NativeTreeNode::Scalar {
            value: NativeValue::Null,
        };
        let array = |children| NativeTreeNode::Array { children };
        for nodes in [
            vec![],
            vec![scalar()],
            vec![array(vec![0])],
            vec![scalar(), array(vec![0, 0])],
            vec![scalar(), scalar(), array(vec![0])],
            vec![scalar(), scalar(), array(vec![0]), array(vec![1, 2])],
            vec![
                NativeTreeNode::Scalar {
                    value: NativeValue::Array { items: vec![] },
                },
                array(vec![0]),
            ],
        ] {
            assert!(logical(&NativeValue::Tree { nodes }).is_err());
        }
        let mut nodes = vec![scalar()];
        for ordinal in 0..sdm_wire::json::MAX_DEPTH {
            nodes.push(array(vec![u32::try_from(ordinal).unwrap()]));
        }
        assert!(
            logical(&NativeValue::Tree {
                nodes: nodes.clone()
            })
            .is_ok()
        );
        nodes.push(array(vec![
            u32::try_from(sdm_wire::json::MAX_DEPTH).unwrap(),
        ]));
        assert!(logical(&NativeValue::Tree { nodes }).is_err());
    }
}
