// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Fixed decoded-row, builtin-value and attribution data boundaries.

use super::{entity_scalar, entity_text, invalid};
use crate::predict::{self, Entity, Part};
use indexmap::{IndexMap, IndexSet};
use sdm_contract::prepare::*;
use sdm_contract::{CoreError, Detail, Scalar, Sentence};
use sdm_wire::json::Value;
use sdm_wire::response::PredictionResponse;

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
        .predictions
        .iter()
        .enumerate()
        .map(|(ordinal, item)| {
            let prior_ordinal = first.get(&item.row_index).copied();
            first.entry(item.row_index.clone()).or_insert(ordinal);
            let key = keys
                .as_ref()
                .and_then(|keys| item.row_index.as_usize().and_then(|at| keys.get(at)))
                .map(entity_scalar);
            DecodedDirectRow {
                ordinal,
                row_index: item.row_index.to_string(),
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
            request_id: response.id.clone(),
            call_request_id: call_request_id.into(),
            response_rows: response.predictions.len(),
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
fn builtin(member: sdm_wire::id_json::Member) -> BuiltinValue {
    use sdm_wire::id_json::Member as M;
    match member {
        M::Null => BuiltinValue::Null,
        M::Boolean(value) => BuiltinValue::Boolean { value },
        M::Integer(digits) => BuiltinValue::Integer {
            digits,
            radix: IntegerRadix::Decimal,
        },
        M::Float(v) => BuiltinValue::Float {
            bits: v.to_be_bytes(),
        },
        M::Unicode(points) => BuiltinValue::Unicode { points },
        M::Array => BuiltinValue::List,
        M::Object => BuiltinValue::Dictionary,
    }
}
fn representation(row: &DecodedDirectRow, limit: u32) -> Result<DirectRowId, CoreError> {
    if limit != 0 && limit < 640 {
        return Err(protocol("native decimal limit is incoherent"));
    }
    if row.key.is_none() {
        return Ok(DirectRowId::Unkeyed);
    }
    let Some(text) = &row.id else {
        return Ok(DirectRowId::Absent);
    };
    if text.chars().count() > 1024 {
        return Ok(DirectRowId::LengthLimitedText);
    }
    Ok(match sdm_wire::id_json::read(text, limit) {
        Some(value) => DirectRowId::JsonValue {
            value: builtin(value.value),
            members: value.members.map(|v| v.into_iter().map(builtin).collect()),
        },
        None => DirectRowId::NativeLiteralText {
            value: text.clone(),
        },
    })
}
enum Name {
    Text(Vec<u32>),
    Other,
}
fn json_name(value: &BuiltinValue, float_text: fn(f64) -> String) -> Result<Name, CoreError> {
    let text = match value {
        BuiltinValue::Null => "None".into(),
        BuiltinValue::Boolean { value } => if *value { "True" } else { "False" }.into(),
        BuiltinValue::Integer {
            digits,
            radix: IntegerRadix::Decimal,
        } => digits.clone(),
        BuiltinValue::Float { bits } => float_text(f64::from_be_bytes(*bits)),
        BuiltinValue::Unicode { points } => return Ok(Name::Text(points.clone())),
        BuiltinValue::List | BuiltinValue::Dictionary => return Ok(Name::Other),
        _ => return Err(protocol("JSON row contains a non-JSON builtin")),
    };
    Ok(Name::Text(text.chars().map(u32::from).collect()))
}
fn attribution(
    checked: &CheckedDirectRow,
    name: Option<Name>,
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
            Some(Name::Other) => true,
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
        decimal_limit: checked.decimal_limit,
        row: row.clone(),
        literal,
        string,
        id,
    })
}
pub fn check_rows(
    scope: ResponseScope,
    rows: Vec<DecodedDirectRow>,
    limit: u32,
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
        let id = representation(&row, limit)?;
        let checked = CheckedDirectRow {
            scope: scope.clone(),
            row,
            decimal_limit: limit,
            id,
        };
        if matches!(&checked.id, DirectRowId::NativeLiteralText { .. }) {
            return Ok(CheckedDirectRows {
                scope,
                rows: accepted,
                row: Some(checked),
            });
        }
        let name = match &checked.id {
            DirectRowId::JsonValue {
                value: BuiltinValue::List,
                members: Some(members),
            } if members.len() == 1 => Some(json_name(&members[0], float_text)?),
            _ => None,
        };
        accepted.push(attribution(&checked, name, None, None, float_text)?);
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
    if representation(&row.row, row.decimal_limit)? != row.id {
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
pub fn select_member(
    checked: CheckedDirectRow,
    literal: LiteralObservation,
) -> Result<SelectedDirectMember, CoreError> {
    self::checked(&checked)?;
    let DirectRowId::NativeLiteralText { value: text } = &checked.id else {
        return Err(protocol(
            "literal value belongs to another ID representation",
        ));
    };
    if literal.scope != checked.scope
        || literal.ordinal != checked.row.ordinal
        || literal.text != *text
    {
        return Err(protocol(
            "native literal observation belongs to another decoded row",
        ));
    }
    let member = match &literal.value {
        NativeLiteralValue::Failed { failure } => {
            if failure.ancestry.is_empty() {
                return Err(native_failure("literal"));
            }
            None
        }
        NativeLiteralValue::Value {
            value: root,
            members,
        } => {
            value(root)?;
            if matches!(root, BuiltinValue::List) != members.is_some() {
                return Err(protocol("builtin list members disagree with root kind"));
            }
            if let Some(members) = members {
                for member in members {
                    value(member)?;
                }
                if members.len() == 1 {
                    Some(members[0].clone())
                } else {
                    None
                }
            } else {
                None
            }
        }
    };
    Ok(SelectedDirectMember {
        checked,
        literal,
        member,
    })
}
pub fn attribute(
    selected: SelectedDirectMember,
    string: Option<StringObservation>,
    float_text: fn(f64) -> String,
) -> Result<AttributedDirectRow, CoreError> {
    if select_member(selected.checked.clone(), selected.literal.clone())? != selected {
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
        Some(selected.literal.clone()),
        string,
        float_text,
    )
}
pub fn layout(
    payload: &Value,
    response: &PredictionResponse,
    scope: &ResponseScope,
    attributed: Vec<AttributedDirectRow>,
    decimal_limit: u32,
    float_text: fn(f64) -> String,
) -> Result<Part, CoreError> {
    let (actual, rows) = decode(payload, response, float_text)?;
    if actual != *scope || attributed.len() != rows.len() {
        return Err(protocol(
            "attributed rows disagree with decoded response scope",
        ));
    }
    let mut seen = IndexSet::new();
    for (row, accepted) in rows.iter().zip(&attributed) {
        if *row != accepted.row {
            return Err(protocol("attribution source differs from decoded response"));
        }
        let replay = if let Some(literal) = &accepted.literal {
            let checked = CheckedDirectRow {
                scope: scope.clone(),
                row: row.clone(),
                decimal_limit: accepted.decimal_limit,
                id: DirectRowId::NativeLiteralText {
                    value: literal.text.clone(),
                },
            };
            attribute(
                select_member(checked, literal.clone())?,
                accepted.string.clone(),
                float_text,
            )?
        } else {
            let mut checked = check_rows(
                scope.clone(),
                vec![row.clone()],
                accepted.decimal_limit,
                float_text,
            )?;
            if checked.row.is_some() {
                return Err(protocol("a native row has no literal observation"));
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
    let mut part = predict::layout_tabular(payload, response, decimal_limit, float_text)?;
    for (row, accepted) in part.rows.iter_mut().zip(attributed) {
        row.id = entity(&accepted.id)?;
    }
    Ok(part)
}
