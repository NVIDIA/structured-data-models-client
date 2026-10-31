// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pure policy over host-held Tabular session metadata.
//! There is no session registry, native callback or transport here.

use sdm_contract::direct::{BoundDirectCall, TabularColumns};
use sdm_contract::session::{
    SessionHoldoutSelection, SessionKey, SessionKeyPart, SessionQuestion,
    SessionQuestionDifference, SessionQuestionField, TABULAR_SESSION_VERSION, TabularSessionState,
};
use sdm_contract::{CoreError, Sentence};
use sdm_wire::json::Value;
use std::collections::{BTreeMap, BTreeSet};

fn malformed(message: &str) -> CoreError {
    CoreError::protocol(Sentence::InvalidMessage, message)
}

fn canonical_integer(value: &str) -> bool {
    let digits = value.strip_prefix('-').unwrap_or(value);
    !digits.is_empty()
        && digits.bytes().all(|byte| byte.is_ascii_digit())
        && (digits == "0" || !digits.starts_with('0'))
        && value != "-0"
}

/// Reject incoherent imported state before it can select a model operation.
/// Key attachment contents are validated by the operation consuming them.
pub fn validate_state(state: &TabularSessionState) -> Result<(), CoreError> {
    if state.version != TABULAR_SESSION_VERSION || state.session_id.is_empty() {
        return Err(malformed(
            "session state has an unsupported version or empty id",
        ));
    }
    if state.pinned_model.is_empty() || state.pinned_kind.is_empty() {
        return Err(malformed("session pinned model or task kind is empty"));
    }
    validate_definition(&state.bound, &state.columns, &state.primary_key)?;
    if state
        .question
        .as_ref()
        .and_then(|question| question.context.as_deref())
        .is_some_and(|value| !canonical_integer(value))
    {
        return Err(malformed(
            "session context fingerprint is not a canonical integer",
        ));
    }
    Ok(())
}

/// Check the pinned schema/model meaning before or after a session is created.
pub fn validate_definition(
    bound: &BoundDirectCall,
    schema: &TabularColumns,
    primary_key: &[String],
) -> Result<(), CoreError> {
    crate::direct::bound(bound)
        .map_err(|_| malformed("session state has an invalid bound model call"))?;
    if bound.model != crate::predict::TABULAR {
        return Err(malformed("a Tabular session names another model"));
    }
    let mut dtypes = BTreeMap::new();
    for (name, dtype) in &schema.dtypes {
        if dtypes.insert(name.as_str(), *dtype).is_some() {
            return Err(malformed("session schema repeats a column dtype"));
        }
    }
    let mut columns = BTreeSet::new();
    for column in &schema.columns {
        if !columns.insert(column.name.as_str())
            || dtypes.get(column.name.as_str()) != Some(&column.dtype)
        {
            return Err(malformed(
                "session schema has repeated or inconsistent columns",
            ));
        }
    }
    if !columns.contains(schema.target.as_str()) {
        return Err(malformed("session target is absent from its pinned schema"));
    }
    let mut keys = BTreeSet::new();
    for key in primary_key {
        if !keys.insert(key.as_str()) || !columns.contains(key.as_str()) {
            return Err(malformed(
                "session primary key is repeated or absent from its schema",
            ));
        }
    }
    Ok(())
}

/// Choose the native ID owner before its reached string conversion.
pub fn select_session_id(
    observed: &sdm_contract::session::SessionIdObservation,
) -> sdm_contract::session::SessionIdSelection {
    use sdm_contract::session::SessionIdSelection as Selected;
    if observed.session_id_truthy {
        Selected::SessionId
    } else if observed.id_truthy {
        Selected::Id
    } else {
        Selected::Missing
    }
}

/// Model/task/schema metadata shared by creation and scoring attribution.
pub fn pinned_metadata(
    bound: &BoundDirectCall,
    columns: &TabularColumns,
    primary_key: &[String],
) -> Result<Value, CoreError> {
    validate_definition(bound, columns, primary_key)?;
    let dtype = columns
        .dtypes
        .iter()
        .find(|(name, _)| name == &columns.target)
        .ok_or_else(|| malformed("session target dtype is absent"))?
        .1;
    Ok(super::object([
        ("model", super::text(&bound.model)),
        (
            "task",
            super::object([
                ("kind", super::text(&bound.kind)),
                (
                    "target",
                    super::object([
                        ("column_name", super::text(&columns.target)),
                        ("dtype", super::text(dtype.wire_name())),
                    ]),
                ),
            ]),
        ),
        (
            "schema",
            super::object([(
                "instance_table",
                super::instance_schema(columns, primary_key),
            )]),
        ),
    ]))
}

/// Rebuild layout metadata from the actual pinned meaning, retaining the
/// original preflight decision separately for outputs and the public model.
pub fn state_metadata(state: &TabularSessionState) -> Result<Value, CoreError> {
    validate_state(state)?;
    let mut value = pinned_metadata(&state.bound, &state.columns, &state.primary_key)?;
    value.set("model", super::text(&state.pinned_model));
    let mut task = value
        .get("task")
        .cloned()
        .ok_or_else(|| malformed("session task metadata is absent"))?;
    task.set("kind", super::text(&state.pinned_kind));
    value.set("task", task);
    Ok(value)
}

fn member<'a>(value: &'a Value, name: &str) -> Result<&'a Value, CoreError> {
    value
        .get(name)
        .ok_or_else(|| malformed("post-send session metadata is incomplete"))
}

fn text_member<'a>(value: &'a Value, name: &str) -> Result<&'a str, CoreError> {
    match member(value, name)? {
        Value::Text(text) => Ok(text),
        _ => Err(malformed("post-send session metadata field is not text")),
    }
}

/// Interpret actual post-send fields, including transport mutations, in Rust.
/// No old encoded context or pre-send key list overrides these observations.
pub fn observe_pinned(
    state: &mut TabularSessionState,
    observed: &Value,
) -> Result<sdm_contract::session::SessionKeys, CoreError> {
    use sdm_contract::direct::TabularColumn;
    use sdm_contract::ops::{Dtype, Stype};
    let model = text_member(observed, "model")?.to_owned();
    let task = member(observed, "task")?;
    let kind = text_member(task, "kind")?.to_owned();
    let target = member(task, "target")?;
    let target_name = text_member(target, "column_name")?.to_owned();
    let target_dtype = text_member(target, "dtype")?;
    let schema = member(member(observed, "schema")?, "instance_table")?;
    let Value::Object(observed_columns) = member(schema, "columns")? else {
        return Err(malformed("post-send session columns are not an object"));
    };
    let candidates = [
        Dtype::Bool,
        Dtype::Int32,
        Dtype::Int64,
        Dtype::Float32,
        Dtype::Float64,
        Dtype::String,
        Dtype::Binary,
        Dtype::Time,
        Dtype::Timedelta,
        Dtype::Floatlist,
        Dtype::Intlist,
        Dtype::Stringlist,
    ];
    let meanings = [
        Stype::Numerical,
        Stype::Categorical,
        Stype::Multicategorical,
        Stype::Id,
        Stype::Text,
        Stype::Timestamp,
        Stype::Sequence,
        Stype::Image,
    ];
    let mut dtypes = Vec::with_capacity(observed_columns.len());
    let mut columns = Vec::with_capacity(observed_columns.len());
    for (name, column) in observed_columns {
        let dtype = text_member(column, "dtype")?;
        let prior = state
            .columns
            .dtypes
            .iter()
            .find(|(prior, _)| prior == name)
            .map(|(_, dtype)| *dtype)
            .filter(|prior| prior.wire_name() == dtype);
        let dtype = prior
            .or_else(|| {
                candidates
                    .iter()
                    .copied()
                    .find(|candidate| candidate.wire_name() == dtype)
            })
            .ok_or_else(|| malformed("post-send session dtype is unsupported"))?;
        let meaning = text_member(column, "stype")?;
        let stype = meanings
            .iter()
            .copied()
            .find(|stype| crate::direct::stype_name(*stype) == meaning)
            .ok_or_else(|| malformed("post-send session semantic type is unsupported"))?;
        dtypes.push((name.clone(), dtype));
        columns.push(TabularColumn {
            name: name.clone(),
            dtype,
            stype,
        });
    }
    if !dtypes
        .iter()
        .any(|(name, dtype)| name == &target_name && dtype.wire_name() == target_dtype)
    {
        return Err(malformed(
            "post-send session task dtype differs from its schema",
        ));
    }
    let primary_key = match schema.get("primary_key") {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::Text(name)) => vec![name.clone()],
        Some(Value::Array(names)) => names
            .iter()
            .map(|name| match name {
                Value::Text(name) => Ok(name.clone()),
                _ => Err(malformed("post-send session primary key is not text")),
            })
            .collect::<Result<Vec<_>, _>>()?,
        _ => return Err(malformed("post-send session primary key has invalid shape")),
    };
    let context = member(member(observed, "context")?, "instance_table")?;
    let keys = encoded_keys(context, &primary_key)?;
    for key in &keys.rows {
        canonical_key(key)?;
    }
    state.pinned_model = model;
    state.pinned_kind = kind;
    state.columns = TabularColumns {
        dtypes,
        columns,
        target: target_name,
    };
    state.primary_key = primary_key;
    validate_state(state)?;
    Ok(keys)
}

fn table_rows(table: &Value) -> Result<(&[Value], &[Value]), CoreError> {
    let Some(Value::Array(columns)) = table.get("columns") else {
        return Err(malformed("session table has no columns"));
    };
    let Some(Value::Array(rows)) = table.get("rows") else {
        return Err(malformed("session table has no rows"));
    };
    let mut names = BTreeSet::new();
    for column in columns {
        let Value::Text(name) = column else {
            return Err(malformed("session table column is not text"));
        };
        if !names.insert(name) {
            return Err(malformed("session table has duplicate columns"));
        }
    }
    if rows
        .iter()
        .any(|row| !matches!(row, Value::Array(values) if values.len() == columns.len()))
    {
        return Err(malformed("session table rows disagree with columns"));
    }
    Ok((columns, rows))
}

/// Observe keys from the encoded table rather than rereading native frames.
pub fn encoded_keys(
    table: &Value,
    primary_key: &[String],
) -> Result<sdm_contract::session::SessionKeys, CoreError> {
    let (columns, rows) = table_rows(table)?;
    let positions: Option<Vec<usize>> = primary_key
        .iter()
        .map(|key| {
            columns
                .iter()
                .position(|column| matches!(column, Value::Text(name) if name == key))
        })
        .collect();
    let Some(positions) = positions.filter(|positions| !positions.is_empty()) else {
        return Ok(sdm_contract::session::SessionKeys {
            rows: rows.iter().map(|_| vec![SessionKeyPart::Missing]).collect(),
        });
    };
    let mut keys = Vec::with_capacity(rows.len());
    for row in rows {
        let Value::Array(values) = row else {
            return Err(malformed("session table row is not an array"));
        };
        let mut key = Vec::with_capacity(positions.len());
        for &position in &positions {
            key.push(match &values[position] {
                Value::Null => SessionKeyPart::Missing,
                Value::Bool(value) => SessionKeyPart::Boolean { value: *value },
                Value::Number(value) if matches!(value.as_str(), "NaN" | "nan") => {
                    SessionKeyPart::Missing
                }
                Value::Number(value) => SessionKeyPart::Numeric {
                    decimal: value.clone(),
                },
                Value::Text(value) => SessionKeyPart::Spelling {
                    value: value.clone(),
                },
                _ => return Err(malformed("session entity key is not an encoded scalar")),
            });
        }
        keys.push(key);
    }
    Ok(sdm_contract::session::SessionKeys { rows: keys })
}

/// Retain only selected encoded rows without another native conversion.
pub fn select_encoded_rows(mut table: Value, retained: &[usize]) -> Result<Value, CoreError> {
    let (_, rows) = table_rows(&table)?;
    if retained.windows(2).any(|pair| pair[0] >= pair[1])
        || retained.last().is_some_and(|&last| last >= rows.len())
    {
        return Err(malformed(
            "session row selection is not ordered source positions",
        ));
    }
    if let Value::Object(fields) = &mut table {
        let (_, row_value) = fields
            .iter_mut()
            .find(|(name, _)| name == "rows")
            .ok_or_else(|| malformed("session table rows are absent"))?;
        let Value::Array(rows) = std::mem::replace(row_value, Value::Null) else {
            return Err(malformed("session table rows are not an array"));
        };
        let mut wanted = retained.iter().copied().peekable();
        *row_value = Value::Array(
            rows.into_iter()
                .enumerate()
                .filter_map(|(index, row)| {
                    if wanted.peek() == Some(&index) {
                        wanted.next();
                        Some(row)
                    } else {
                        None
                    }
                })
                .collect(),
        );
    }
    Ok(table)
}

/// Select the same first mismatch as the public fitted-question diagnostic.
/// If either request has no question, the legacy guard does not compare them.
pub fn check_question(
    pinned: Option<&SessionQuestion>,
    asked: Option<&SessionQuestion>,
) -> Option<SessionQuestionDifference> {
    let (Some(pinned), Some(asked)) = (pinned, asked) else {
        return None;
    };
    use SessionQuestionField as Field;
    let fields = [
        (
            Field::Query,
            pinned.query.as_deref(),
            asked.query.as_deref(),
        ),
        (
            Field::Kind,
            Some(pinned.kind.as_str()),
            Some(asked.kind.as_str()),
        ),
        (
            Field::Target,
            Some(pinned.target.as_str()),
            Some(asked.target.as_str()),
        ),
        (
            Field::Dtype,
            Some(pinned.dtype.as_str()),
            Some(asked.dtype.as_str()),
        ),
        (
            Field::EntityTable,
            pinned.entity_table.as_deref(),
            asked.entity_table.as_deref(),
        ),
        (
            Field::Context,
            pinned.context.as_deref(),
            asked.context.as_deref(),
        ),
    ];
    fields.into_iter().find_map(|(field, pinned, asked)| {
        (pinned != asked).then(|| SessionQuestionDifference {
            field,
            pinned: pinned.map(str::to_owned),
            asked: asked.map(str::to_owned),
        })
    })
}

/// Commit closure only after the host confirms successful remote deletion.
/// A failed deletion leaves the returned value retryable; already closed
/// values remain closed. The caller skips sending when `closed` is already set.
pub fn close_after_success(
    state: &TabularSessionState,
    succeeded: bool,
) -> Result<TabularSessionState, CoreError> {
    validate_state(state)?;
    let mut next = state.clone();
    if succeeded {
        next.closed = true;
    }
    Ok(next)
}

/// Exact finite decimal identity, kept compact even for very large exponents.
/// Normalization does not perform arithmetic in the host's Decimal context.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct DecimalKey {
    negative: bool,
    digits: String,
    exponent: i128,
}

impl DecimalKey {
    fn parse(value: &str) -> Option<Self> {
        let (negative, unsigned) = match value.as_bytes().first()? {
            b'-' => (true, &value[1..]),
            b'+' => (false, &value[1..]),
            _ => (false, value),
        };
        let (coefficient, exponent) = match unsigned.find(['e', 'E']) {
            Some(at) => (&unsigned[..at], unsigned[at + 1..].parse::<i128>().ok()?),
            None => (unsigned, 0),
        };
        let mut digits = String::with_capacity(coefficient.len());
        let mut point = false;
        let mut fractional = 0_i128;
        for byte in coefficient.bytes() {
            if byte == b'.' && !point {
                point = true;
            } else if byte.is_ascii_digit() {
                digits.push(char::from(byte));
                if point {
                    fractional += 1;
                }
            } else {
                return None;
            }
        }
        if digits.is_empty() {
            return None;
        }
        let significant = digits.trim_start_matches('0');
        if significant.is_empty() {
            return Some(Self {
                negative: false,
                digits: "0".into(),
                exponent: 0,
            });
        }
        let trimmed = significant.trim_end_matches('0');
        let exponent = exponent
            .checked_sub(fractional)?
            .checked_add((significant.len() - trimmed.len()) as i128)?;
        Some(Self {
            negative,
            digits: trimmed.into(),
            exponent,
        })
    }

    /// Does text already have the legacy canonical spelling for this number?
    /// Check the required length first, so an exponent cannot cause an
    /// allocation proportional to its magnitude merely to compare a short key.
    fn spells(&self, text: &str) -> bool {
        let sign = i128::from(self.negative);
        let digits = self.digits.len() as i128;
        let Some(point) = digits.checked_add(self.exponent) else {
            return false;
        };
        let Some(adjusted) = point.checked_sub(1) else {
            return false;
        };
        let scientific = self.exponent < 0 && adjusted < -6;
        let required = if self.exponent >= 0 {
            point.checked_add(sign)
        } else if scientific {
            let extra = i128::from(self.digits.len() > 1);
            (digits + extra + 1 + adjusted.to_string().len() as i128).checked_add(sign)
        } else if point > 0 {
            (digits + 1).checked_add(sign)
        } else {
            digits
                .checked_sub(point)
                .and_then(|n| n.checked_add(2 + sign))
        };
        if required != Some(text.len() as i128) {
            return false;
        }
        let mut expected = String::with_capacity(text.len());
        if self.negative {
            expected.push('-');
        }
        if self.exponent >= 0 {
            expected.push_str(&self.digits);
            expected.extend(std::iter::repeat_n('0', self.exponent as usize));
        } else if scientific {
            expected.push(char::from(self.digits.as_bytes()[0]));
            if self.digits.len() > 1 {
                expected.push('.');
                expected.push_str(&self.digits[1..]);
            }
            expected.push('E');
            expected.push_str(&adjusted.to_string());
        } else if point > 0 {
            let at = point as usize;
            expected.push_str(&self.digits[..at]);
            expected.push('.');
            expected.push_str(&self.digits[at..]);
        } else {
            expected.push_str("0.");
            expected.extend(std::iter::repeat_n('0', (-point) as usize));
            expected.push_str(&self.digits);
        }
        expected == text
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum KeyPart {
    Decimal(DecimalKey),
    Spelling(String),
}

/// Comparable key produced by Rust policy; missing keys have no value.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct CanonicalSessionKey(Vec<KeyPart>);

fn spelling(value: &str) -> KeyPart {
    match DecimalKey::parse(value) {
        Some(decimal) if decimal.spells(value) => KeyPart::Decimal(decimal),
        _ => KeyPart::Spelling(value.into()),
    }
}

/// Normalize numeric identities exactly; preserve nonnumeric spellings.
/// A missing component makes the entire key unidentifiable and unmatchable.
pub fn canonical_key(key: &SessionKey) -> Result<Option<CanonicalSessionKey>, CoreError> {
    if key
        .iter()
        .any(|part| matches!(part, SessionKeyPart::Missing))
    {
        return Ok(None);
    }
    let mut parts = Vec::with_capacity(key.len());
    for part in key {
        parts.push(match part {
            SessionKeyPart::Missing => return Ok(None),
            SessionKeyPart::Boolean { value } => {
                KeyPart::Spelling(if *value { "True" } else { "False" }.into())
            }
            SessionKeyPart::Spelling { value } => spelling(value),
            SessionKeyPart::Numeric { decimal } => {
                if let Some(number) = DecimalKey::parse(decimal) {
                    KeyPart::Decimal(number)
                } else if matches!(
                    decimal.as_str(),
                    "Infinity" | "+Infinity" | "-Infinity" | "inf" | "+inf" | "-inf"
                ) {
                    KeyPart::Spelling(decimal.clone())
                } else {
                    return Err(malformed(
                        "session numeric key is not a finite decimal or infinity",
                    ));
                }
            }
        });
    }
    Ok(Some(CanonicalSessionKey(parts)))
}

fn key_set(keys: &[SessionKey]) -> Result<(BTreeSet<CanonicalSessionKey>, usize), CoreError> {
    let mut named = BTreeSet::new();
    let mut unidentified = 0;
    for key in keys {
        if let Some(key) = canonical_key(key)? {
            named.insert(key);
        } else {
            unidentified += 1;
        }
    }
    Ok((named, unidentified))
}

/// Select surviving context row positions; source order and duplicates remain.
/// The caller applies these positions to the encoded context and pinned keys.
pub fn filter_holdout(
    context: &[SessionKey],
    holdout: &[SessionKey],
) -> Result<SessionHoldoutSelection, CoreError> {
    let (wanted, unidentified) = key_set(holdout)?;
    let named_keys = wanted.len() + unidentified;
    let mut retained_rows = Vec::with_capacity(context.len());
    for (row, key) in context.iter().enumerate() {
        if canonical_key(key)?.is_none_or(|key| !wanted.contains(&key)) {
            retained_rows.push(row);
        }
    }
    Ok(SessionHoldoutSelection {
        removed_rows: context.len() - retained_rows.len(),
        retained_rows,
        named_keys,
    })
}

/// Count scoring rows whose identities were pinned, including repeated rows.
pub fn overlap_count(pinned: &[SessionKey], scoring: &[SessionKey]) -> Result<usize, CoreError> {
    let (pinned, _) = key_set(pinned)?;
    let mut count = 0;
    for key in scoring {
        if canonical_key(key)?.is_some_and(|key| pinned.contains(&key)) {
            count += 1;
        }
    }
    Ok(count)
}
