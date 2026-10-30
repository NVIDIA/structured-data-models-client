// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Fixed decoded-row, builtin-value and attribution data boundaries.

use super::{entity_scalar, entity_text, invalid};
use crate::predict::{self, AnswerType, Entity, Part};
use indexmap::{IndexMap, IndexSet};
use sdm_contract::direct::*;
use sdm_contract::direct::native_response::{NativeResponseIdentity, NativeRowIdentity};
use sdm_contract::{CoreError, Detail, Scalar, Sentence};
use sdm_wire::json::Value;
use sdm_wire::response::PredictionResponse;

/// Tabular task acceptance is independent of gathered prediction tasks.
fn answer_type(payload: &Value) -> Result<AnswerType, CoreError> {
    match payload
        .get("task")
        .and_then(|task| task.get("kind"))
        .and_then(Value::as_text)
    {
        Some("regression") => Ok(AnswerType::Float),
        Some("classification" | "binary_classification" | "multiclass_classification") => {
            Ok(predict::classification_answer_type(payload))
        }
        _ => Err(predict::not_yet("problem_type")),
    }
}

/// Preserve Tabular target typing even when no model call is needed.
pub fn empty(model: &str, payload: &Value) -> Result<Part, CoreError> {
    let answers = answer_type(payload)?;
    let mut part = predict::empty(model, payload);
    part.answers = answers;
    Ok(part)
}

fn protocol(message: &str) -> CoreError {
    CoreError::protocol(Sentence::InvalidMessage, message)
}
fn entity(value: &Scalar) -> Result<Entity, CoreError> {
    Ok(match value {
        Scalar::Null => Entity::None,
        Scalar::Bool(v) => Entity::Written(Value::Bool(*v)),
        Scalar::Int64(v) => Entity::Integer(*v),
        Scalar::UInt64(v) => Entity::Written(Value::Number(v.to_string())),
        Scalar::Float64(v) if v.is_finite() => Entity::Float(*v),
        Scalar::Utf8(v) => Entity::Text(v.clone()),
        _ => return Err(protocol("a direct row has a nonprimitive key")),
    })
}
fn named_scalar(value: &Scalar, float_text: fn(f64) -> String) -> Result<Vec<u32>, CoreError> {
    Ok(entity_text(&entity(value)?, float_text)
        .chars()
        .map(u32::from)
        .collect())
}

pub fn decode(
    payload: &Value,
    response: &PredictionResponse,
    float_text: fn(f64) -> String,
) -> Result<(ResponseScope, Vec<DecodedDirectRow>), CoreError> {
    let identity = NativeResponseIdentity {
        model: response.model.clone(), request_id: response.id.clone(),
        rows: response.predictions.iter().map(|item| NativeRowIdentity {
            row_index: item.row_index.to_string(), id: item.id.clone(),
        }).collect(),
    };
    decode_identities(payload, &identity, float_text)
}

/// Attribute checked native identities without serializing opaque response values.
pub fn decode_identities(
    payload: &Value, response: &NativeResponseIdentity, float_text: fn(f64) -> String,
) -> Result<(ResponseScope, Vec<DecodedDirectRow>), CoreError> {
    let scored = payload
        .get("predict")
        .and_then(|p| p.get("instance_table"))
        .ok_or_else(|| protocol("direct response has no scoring table"))?;
    let Some(Value::Array(rows)) = scored.get("rows") else {
        return Err(protocol("direct scoring rows are not an array"));
    };
    let instance = payload.get("schema").and_then(|s| s.get("instance_table"));
    let key = match instance.and_then(|s| s.get("primary_key")) {
        Some(Value::Array(keys)) if keys.len() == 1 => keys[0].as_text(),
        _ => None,
    };
    let key_at = match (key, scored.get("columns")) {
        (Some(key), Some(Value::Array(cols))) => cols.iter().position(|v| v.as_text() == Some(key)),
        _ => None,
    };
    let dtype = key.and_then(|key| {
        instance
            .and_then(|s| s.get("columns"))
            .and_then(|c| c.get(key))
            .and_then(|c| c.get("dtype"))
            .and_then(Value::as_text)
    });
    let keys: Option<Vec<Entity>> = key_at.map(|at| {
        rows.iter()
            .map(|row| match row {
                Value::Array(cells) => cells
                    .get(at)
                    .map_or(Entity::None, |v| predict::entity(v, dtype)),
                _ => Entity::None,
            })
            .collect()
    });
    let names: IndexSet<String> = keys
        .as_ref()
        .into_iter()
        .flatten()
        .map(|key| entity_text(key, float_text))
        .collect();
    let mut first = IndexMap::new();
    let decoded = response
        .rows
        .iter()
        .enumerate()
        .map(|(ordinal, item)| {
            let prior_ordinal = first.get(&item.row_index).copied();
            first.entry(item.row_index.clone()).or_insert(ordinal);
            let key = keys
                .as_ref()
                .and_then(|keys| {
                    item.row_index.parse::<usize>()
                        .ok()
                        .and_then(|at| keys.get(at))
                })
                .map(entity_scalar);
            DecodedDirectRow {
                ordinal,
                row_index: item.row_index.clone(),
                id: item.id.clone(),
                key,
                raw_in_batch: item.id.as_ref().is_some_and(|id| names.contains(id)),
                prior_ordinal,
            }
        })
        .collect();
    let call_request_id = payload
        .get("metadata")
        .and_then(|m| m.get("request_id"))
        .and_then(Value::as_text)
        .ok_or_else(|| protocol("direct payload has no request id"))?;
    Ok((
        ResponseScope {
            version: DIRECT_RESPONSE_VERSION,
            model: response.model.clone(),
            request_id: response.request_id.clone(),
            call_request_id: call_request_id.into(),
            response_rows: response.rows.len(),
            scoring_rows: rows.len(),
        },
        decoded,
    ))
}
fn scope(scope: &ResponseScope) -> Result<(), CoreError> {
    if scope.version != DIRECT_RESPONSE_VERSION {
        return Err(protocol("unknown direct response data version"));
    }
    Ok(())
}
fn row_index(scope: &ResponseScope, row: &DecodedDirectRow) -> Result<usize, CoreError> {
    self::scope(scope)?;
    if row.row_index == "missing" {
        return Err(invalid(Sentence::TabularMissingRowIndex));
    }
    if row.ordinal >= scope.response_rows
        || row.prior_ordinal.is_some_and(|prior| prior >= row.ordinal)
    {
        return Err(protocol("decoded row ordinal is incoherent"));
    }
    if sdm_wire::response::integer_text(&row.row_index).as_deref() != Some(row.row_index.as_str()) {
        return Err(protocol("decoded row index is not canonical"));
    }
    let at = row
        .row_index
        .parse::<usize>()
        .ok()
        .filter(|at| *at < scope.scoring_rows)
        .ok_or_else(|| {
            invalid(Sentence::TabularRowBounds)
                .with_field("index", Detail::text(&row.row_index))
                .with_field(
                    "rows",
                    Detail::Scalar(Scalar::UInt64(scope.scoring_rows as u64)),
                )
        })?;
    if row.prior_ordinal.is_some() {
        return Err(invalid(Sentence::TabularRowRepeated)
            .with_field("index", Detail::Scalar(Scalar::UInt64(at as u64))));
    }
    if let Some(key) = &row.key {
        entity(key)?;
    }
    Ok(at)
}
fn representation(row: &DecodedDirectRow) -> Result<DirectRowId, CoreError> {
    if row.key.is_none() {
        return Ok(DirectRowId::Unkeyed);
    }
    let Some(text) = &row.id else {
        return Ok(DirectRowId::Absent);
    };
    if text.chars().count() > 1024 {
        return Ok(DirectRowId::LengthLimitedText);
    }
    Ok(DirectRowId::NativeJsonText {
        value: text.clone(),
    })
}
enum Name {
    Text(Vec<u32>),
}
fn attribution(
    checked: &CheckedDirectRow,
    name: Option<Name>,
    json: Option<JsonObservation>,
    literal: Option<LiteralObservation>,
    string: Option<StringObservation>,
    float_text: fn(f64) -> String,
) -> Result<AttributedDirectRow, CoreError> {
    let at = row_index(&checked.scope, &checked.row)?;
    let row = &checked.row;
    let mut id = row.id.clone().map_or(Scalar::Null, Scalar::Utf8);
    if let Some(key) = &row.key {
        let name = name.or_else(|| {
            row.raw_in_batch.then(|| {
                Name::Text(
                    row.id
                        .as_deref()
                        .unwrap_or("")
                        .chars()
                        .map(u32::from)
                        .collect(),
                )
            })
        });
        let wrong = match name {
            Some(Name::Text(text)) => text != named_scalar(key, float_text)?,
            None => false,
        };
        if wrong {
            return Err(invalid(Sentence::TabularIdMismatch)
                .with_field("index", Detail::Scalar(Scalar::UInt64(at as u64)))
                .with_field("id", Detail::text(row.id.as_deref().unwrap_or("")))
                .with_field("expected", Detail::Scalar(key.clone())));
        }
        if *key != Scalar::Null {
            id = key.clone();
        }
    }
    Ok(AttributedDirectRow {
        row: row.clone(),
        json,
        literal,
        string,
        id,
    })
}
pub fn check_rows(
    scope: ResponseScope,
    rows: Vec<DecodedDirectRow>,
    float_text: fn(f64) -> String,
) -> Result<CheckedDirectRows, CoreError> {
    self::scope(&scope)?;
    let mut accepted = Vec::new();
    let mut prior = None;
    for row in rows {
        if prior.is_some_and(|ordinal| row.ordinal != ordinal + 1) {
            return Err(protocol("decoded row slice is not consecutive"));
        }
        prior = Some(row.ordinal);
        row_index(&scope, &row)?;
        let id = representation(&row)?;
        let checked = CheckedDirectRow {
            scope: scope.clone(),
            row,
            id,
        };
        if matches!(&checked.id, DirectRowId::NativeJsonText { .. }) {
            return Ok(CheckedDirectRows {
                scope,
                rows: accepted,
                row: Some(checked),
            });
        }
        accepted.push(attribution(&checked, None, None, None, None, float_text)?);
    }
    Ok(CheckedDirectRows {
        scope,
        rows: accepted,
        row: None,
    })
}
fn checked(row: &CheckedDirectRow) -> Result<(), CoreError> {
    row_index(&row.scope, &row.row)
        .map_err(|_| protocol("a checked row contains a refused source row"))?;
    if representation(&row.row)? != row.id {
        return Err(protocol(
            "checked ID representation differs from its source",
        ));
    }
    Ok(())
}
fn value(value: &BuiltinValue) -> Result<(), CoreError> {
    match value {
        BuiltinValue::Unicode { points } if points.iter().any(|p| *p > 0x10FFFF) => {
            Err(protocol("native Unicode point is out of range"))
        }
        BuiltinValue::Integer { digits, radix } => {
            let valid = match radix {
                IntegerRadix::Decimal => {
                    sdm_wire::response::integer_text(digits).as_deref() == Some(digits.as_str())
                }
                IntegerRadix::Hexadecimal => digits
                    .strip_prefix('-')
                    .unwrap_or(digits)
                    .strip_prefix("0x")
                    .is_some_and(|s| !s.is_empty() && s.bytes().all(|b| b.is_ascii_hexdigit())),
            };
            if valid {
                Ok(())
            } else {
                Err(protocol("native integer digits are malformed"))
            }
        }
        _ => Ok(()),
    }
}
fn native_failure(stage: &str) -> CoreError {
    invalid(Sentence::DirectNativeFailure).with_field("stage", Detail::text(stage))
}
fn parsed_member(
    root: &BuiltinValue,
    members: &Option<Vec<BuiltinValue>>,
    json: bool,
) -> Result<Option<BuiltinValue>, CoreError> {
    let allowed = |item: &BuiltinValue| -> Result<(), CoreError> {
        value(item)?;
        if json
            && !matches!(
                item,
                BuiltinValue::Null
                    | BuiltinValue::Boolean { .. }
                    | BuiltinValue::Integer { .. }
                    | BuiltinValue::Float { .. }
                    | BuiltinValue::Unicode { .. }
                    | BuiltinValue::List
                    | BuiltinValue::Dictionary
            )
        {
            return Err(protocol("JSON conversion contains a non-JSON builtin"));
        }
        Ok(())
    };
    allowed(root)?;
    if matches!(root, BuiltinValue::List) != members.is_some() {
        return Err(protocol("builtin list members disagree with root kind"));
    }
    if let Some(members) = members {
        for member in members {
            allowed(member)?;
        }
        if members.len() == 1 {
            return Ok(Some(members[0].clone()));
        }
    }
    Ok(None)
}
pub fn interpret_json(
    mut checked: CheckedDirectRow,
    json: JsonObservation,
) -> Result<InterpretedDirectRow, CoreError> {
    self::checked(&checked)?;
    let DirectRowId::NativeJsonText { value: text } = &checked.id else {
        return Err(protocol(
            "JSON observation belongs to another ID representation",
        ));
    };
    if json.scope != checked.scope || json.ordinal != checked.row.ordinal || json.text != *text {
        return Err(protocol(
            "native JSON observation belongs to another decoded row",
        ));
    }
    checked.id = match &json.value {
        NativeParsedValue::Failed { failure } => {
            if !failure.ancestry.iter().any(|kind| {
                matches!(
                    kind,
                    BuiltinFailureClass::ValueError
                        | BuiltinFailureClass::TypeError
                        | BuiltinFailureClass::RecursionError
                )
            }) {
                return Err(native_failure("json"));
            }
            DirectRowId::NativeLiteralText {
                value: text.clone(),
            }
        }
        NativeParsedValue::Value { value, members } => {
            parsed_member(value, members, true)?;
            DirectRowId::JsonValue {
                value: value.clone(),
                members: members.clone(),
            }
        }
    };
    Ok(InterpretedDirectRow { checked, json })
}
fn interpreted(row: &InterpretedDirectRow) -> Result<(), CoreError> {
    let mut original = row.checked.clone();
    original.id = representation(&original.row)?;
    if interpret_json(original, row.json.clone())? != *row {
        return Err(protocol(
            "interpreted JSON differs from its source observations",
        ));
    }
    Ok(())
}
pub fn select_member(
    interpreted: InterpretedDirectRow,
    literal: Option<LiteralObservation>,
) -> Result<SelectedDirectMember, CoreError> {
    self::interpreted(&interpreted)?;
    let checked = interpreted.checked;
    let member = match (&checked.id, &literal) {
        (DirectRowId::JsonValue { value, members }, None) => parsed_member(value, members, true)?,
        (DirectRowId::NativeLiteralText { value: text }, Some(observed)) => {
            if observed.scope != checked.scope
                || observed.ordinal != checked.row.ordinal
                || observed.text != *text
            {
                return Err(protocol(
                    "native literal observation belongs to another decoded row",
                ));
            }
            match &observed.value {
                NativeParsedValue::Failed { failure } => {
                    if failure.ancestry.is_empty() {
                        return Err(native_failure("literal"));
                    }
                    None
                }
                NativeParsedValue::Value { value, members } => {
                    parsed_member(value, members, false)?
                }
            }
        }
        _ => {
            return Err(protocol(
                "literal observation disagrees with JSON interpretation",
            ));
        }
    };
    Ok(SelectedDirectMember {
        checked,
        json: interpreted.json,
        literal,
        member,
    })
}
pub fn attribute(
    selected: SelectedDirectMember,
    string: Option<StringObservation>,
    float_text: fn(f64) -> String,
) -> Result<AttributedDirectRow, CoreError> {
    if select_member(
        InterpretedDirectRow {
            checked: selected.checked.clone(),
            json: selected.json.clone(),
        },
        selected.literal.clone(),
    )? != selected
    {
        return Err(protocol(
            "selected builtin member differs from literal value",
        ));
    }
    let name = match (&selected.member, &string) {
        (None, None) => None,
        (Some(_), Some(observed)) => {
            if observed.scope != selected.checked.scope
                || observed.ordinal != selected.checked.row.ordinal
            {
                return Err(protocol("string value belongs to another decoded row"));
            }
            match &observed.value {
                NativeStringValue::Failed { .. } => return Err(native_failure("string")),
                NativeStringValue::Value { points } => {
                    if points.iter().any(|p| *p > 0x10FFFF) {
                        return Err(protocol("native string has an invalid Unicode point"));
                    }
                    Some(Name::Text(points.clone()))
                }
            }
        }
        _ => {
            return Err(protocol(
                "native string conversion disagrees with singleton selection",
            ));
        }
    };
    attribution(
        &selected.checked,
        name,
        Some(selected.json.clone()),
        selected.literal.clone(),
        string,
        float_text,
    )
}
pub fn layout(
    payload: &Value,
    response: &PredictionResponse,
    scope: &ResponseScope,
    attributed: Vec<AttributedDirectRow>,
    _decimal_limit: u32,
    float_text: fn(f64) -> String,
) -> Result<Part, CoreError> {
    let (actual, rows) = decode(payload, response, float_text)?;
    if actual != *scope || attributed.len() != rows.len() {
        return Err(protocol(
            "attributed rows disagree with decoded response scope",
        ));
    }
    validate_attributed(scope, &rows, &attributed, float_text)?;
    let mut part =
        predict::layout_with_answers(payload, response, float_text, answer_type(payload)?)?;
    for (row, accepted) in part.rows.iter_mut().zip(attributed) {
        row.id = entity(&accepted.id)?;
    }
    Ok(part)
}

/// Validate completed attribution without touching retained presentation fields.
pub fn validate_attributed(
    scope: &ResponseScope, rows: &[DecodedDirectRow], attributed: &[AttributedDirectRow],
    float_text: fn(f64) -> String,
) -> Result<(), CoreError> {
    if rows.len() != attributed.len() {
        return Err(protocol("native attribution has a different row count"));
    }
    let mut seen = IndexSet::new();
    for (row, accepted) in rows.iter().zip(attributed) {
        if *row != accepted.row {
            return Err(protocol("attribution source differs from decoded response"));
        }
        let replay = if let Some(json) = &accepted.json {
            let checked = CheckedDirectRow {
                scope: scope.clone(),
                row: row.clone(),
                id: representation(row)?,
            };
            attribute(
                select_member(
                    interpret_json(checked, json.clone())?,
                    accepted.literal.clone(),
                )?,
                accepted.string.clone(),
                float_text,
            )?
        } else {
            if accepted.literal.is_some() || accepted.string.is_some() {
                return Err(protocol("native member data has no JSON observation"));
            }
            let mut checked = check_rows(scope.clone(), vec![row.clone()], float_text)?;
            if checked.row.is_some() {
                return Err(protocol("a reached ID has no JSON observation"));
            }
            checked
                .rows
                .pop()
                .ok_or_else(|| protocol("attribution row is absent"))?
        };
        if replay != *accepted {
            return Err(protocol("attributed ID differs from source facts"));
        }
        seen.insert(
            row.row_index
                .parse::<usize>()
                .map_err(|_| protocol("attributed index is not a row"))?,
        );
    }
    let missing: Vec<_> = (0..scope.scoring_rows)
        .filter(|n| !seen.contains(n))
        .collect();
    if !missing.is_empty() {
        return Err(invalid(Sentence::TabularRowsUnanswered).with_field(
            "missing",
            Detail::List(
                missing
                    .into_iter()
                    .map(|n| Scalar::UInt64(n as u64))
                    .collect(),
            ),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_classification_keeps_the_declared_target_type() {
        for (dtype, expected) in [
            ("bool", AnswerType::Bool),
            ("int64", AnswerType::Int),
            ("float64", AnswerType::FloatClass),
        ] {
            let payload = sdm_wire::json::read(
                format!(r#"{{"task":{{"kind":"classification","target":{{"dtype":"{dtype}"}}}}}}"#)
                    .as_bytes(),
            )
            .unwrap();
            let part = empty(predict::TABULAR, &payload).unwrap();
            assert_eq!(part.answers, expected);
            assert!(part.rows.is_empty());
        }
    }
}

/// Validate position before reaching a native row identifier's text conversion.
pub fn native_row_requires_text(scope: &ResponseScope, row: &DecodedDirectRow) -> Result<bool, CoreError> {
    row_index(scope, row)?;
    Ok(row.key.is_some() && row.id.is_some())
}
