// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Declared-type numeric tabular encoding, independent of gathered envelopes.

use crate::predict::{self, Entity, Part};
use arrow_array::cast::AsArray;
use arrow_array::types::*;
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::DataType;
use sdm_contract::ops::{DescribedTable, Dtype, HeldSchema, HeldTable, SourceDescription, Stype};
use sdm_contract::prepare::{
    DirectInference, FloatStorage, PrimitiveOption, TabularColumn, TabularColumns,
};
use sdm_contract::{CoreError, CoreWarning, Detail, Scalar, Sentence};
use sdm_wire::json::Value;
use sdm_wire::response::PredictionResponse;
use std::collections::{BTreeMap, BTreeSet};

fn text(value: impl Into<String>) -> Value {
    Value::Text(value.into())
}
fn object<const N: usize>(pairs: [(&str, Value); N]) -> Value {
    Value::Object(pairs.into_iter().map(|(k, v)| (k.to_owned(), v)).collect())
}
fn invalid(sentence: Sentence) -> CoreError {
    CoreError::new(sentence, "tabular request or answer is invalid")
}
fn texts(values: impl IntoIterator<Item = String>) -> Detail {
    Detail::List(values.into_iter().map(Scalar::Utf8).collect())
}

/// One widening rule for both request halves, matching widen_tfm_dtype.
fn widened(a: Dtype, b: Dtype) -> Dtype {
    if a == b {
        return a;
    }
    let rank = |d| match d {
        Dtype::Int32 => Some(1),
        Dtype::Int64 => Some(2),
        Dtype::Float32 => Some(3),
        Dtype::Float64 => Some(4),
        _ => None,
    };
    match (rank(a), rank(b)) {
        (Some(_), Some(_))
            if (a == Dtype::Int64 && b == Dtype::Float32)
                || (a == Dtype::Float32 && b == Dtype::Int64) =>
        {
            Dtype::Float64
        }
        (Some(left), Some(right)) => {
            if left > right {
                a
            } else {
                b
            }
        }
        _ => Dtype::String,
    }
}

fn raw(array: &ArrayRef, row: usize) -> Result<Scalar, CoreError> {
    if array.is_null(row) {
        return Ok(Scalar::Null);
    }
    macro_rules! signed {
        ($t:ty) => {
            Scalar::Int64(i64::from(array.as_primitive::<$t>().value(row)))
        };
    }
    macro_rules! unsigned {
        ($t:ty) => {
            Scalar::UInt64(u64::from(array.as_primitive::<$t>().value(row)))
        };
    }
    Ok(match array.data_type() {
        DataType::Int8 => signed!(Int8Type),
        DataType::Int16 => signed!(Int16Type),
        DataType::Int32 => signed!(Int32Type),
        DataType::Int64 => signed!(Int64Type),
        DataType::UInt8 => unsigned!(UInt8Type),
        DataType::UInt16 => unsigned!(UInt16Type),
        DataType::UInt32 => unsigned!(UInt32Type),
        DataType::UInt64 => unsigned!(UInt64Type),
        DataType::Float32 => {
            Scalar::Float64(f64::from(array.as_primitive::<Float32Type>().value(row)))
        }
        DataType::Float64 => Scalar::Float64(array.as_primitive::<Float64Type>().value(row)),
        DataType::Boolean => Scalar::Bool(array.as_boolean().value(row)),
        _ => return Err(crate::predict::not_yet("direct_numeric_storage")),
    })
}

fn cell(
    value: Scalar,
    dtype: Dtype,
    column: &str,
    float_text: fn(f64) -> String,
) -> Result<Value, CoreError> {
    if matches!(value, Scalar::Null) || matches!(value,Scalar::Float64(v) if v.is_nan()) {
        return Ok(Value::Null);
    }
    if dtype == Dtype::String {
        return Ok(text(match value {
            Scalar::Int64(v) => v.to_string(),
            Scalar::UInt64(v) => v.to_string(),
            Scalar::Float64(v) => float_text(v),
            Scalar::Bool(v) => {
                if v {
                    "True".into()
                } else {
                    "False".into()
                }
            }
            _ => return Err(crate::predict::not_yet("direct_numeric_storage")),
        }));
    }
    if dtype == Dtype::Bool {
        return Ok(Value::Bool(match value {
            Scalar::Bool(v) => v,
            Scalar::Int64(v) => v != 0,
            Scalar::UInt64(v) => v != 0,
            Scalar::Float64(v) => v != 0.,
            _ => false,
        }));
    }
    if let Scalar::Float64(v) = value {
        if !v.is_finite() {
            return Err(invalid(Sentence::DirectWireNonFinite)
                .with_field("value", Detail::text(float_text(v))));
        }
        if matches!(dtype, Dtype::Int32 | Dtype::Int64) {
            if v.fract() != 0. {
                return Err(invalid(Sentence::DirectWireFraction)
                    .with_field("value", Detail::Scalar(Scalar::Float64(v)))
                    .with_field("dtype", Detail::text(dtype.wire_name())));
            }
            if v < i64::MIN as f64 || v >= i64::MAX as f64 {
                return Err(invalid(Sentence::DirectWireRange)
                    .with_field("value", Detail::Scalar(Scalar::Float64(v)))
                    .with_field("dtype", Detail::text(dtype.wire_name())));
            }
            return crate::payload::int_cell(v as i128, column);
        }
        return crate::payload::float_cell(v, column);
    }
    let number = match value {
        Scalar::Int64(v) => i128::from(v),
        Scalar::UInt64(v) => i128::from(v),
        Scalar::Bool(v) => i128::from(v),
        _ => return Err(crate::predict::not_yet("direct_numeric_storage")),
    };
    if matches!(dtype, Dtype::Float32 | Dtype::Float64) {
        let widened = number as f64;
        if widened as i128 != number {
            return Err(invalid(Sentence::DirectFloatPrecision)
                .with_field("value", Detail::Scalar(value))
                .with_field("dtype", Detail::text(dtype.wire_name())));
        }
        return crate::payload::float_cell(widened, column);
    }
    if number < i128::from(i64::MIN) || number > i128::from(i64::MAX) {
        return Err(invalid(Sentence::DirectWireRange)
            .with_field("value", Detail::Scalar(value))
            .with_field("dtype", Detail::text(dtype.wire_name())));
    }
    crate::payload::int_cell(number, column)
}

fn nullable_whole(
    batches: &[RecordBatch],
    at: usize,
    storage: Option<FloatStorage>,
) -> Result<bool, CoreError> {
    if !batches.iter().any(|b| b.column(at).null_count() > 0) {
        return Ok(false);
    }
    let storage = storage.ok_or_else(|| {
        CoreError::protocol(
            Sentence::InvalidMessage,
            "nullable floating storage has no native missingness fact",
        )
    })?;
    let present = batches
        .iter()
        .flat_map(|batch| (0..batch.num_rows()).map(move |row| raw(batch.column(at), row)))
        .collect::<Result<Vec<_>, _>>()?;
    if storage == FloatStorage::NanMissing
        && present
            .iter()
            .any(|v| matches!(v,Scalar::Float64(v) if !v.is_finite()))
    {
        return Err(invalid(Sentence::DirectNullableCast)
            .with_field("ordinal", Detail::Scalar(Scalar::UInt64(at as u64))));
    }
    Ok(present.into_iter().all(|value| match value {
        Scalar::Null => true,
        Scalar::Float64(v) => {
            v.is_finite() && v.fract() == 0. && v >= i64::MIN as f64 && v < i64::MAX as f64
        }
        _ => false,
    }))
}

fn widths(
    table: &DescribedTable,
    native: Option<&HeldTable>,
    scoring: &[RecordBatch],
    arriving: &[(String, Dtype)],
    storage: &[Option<FloatStorage>],
) -> Result<BTreeMap<String, Dtype>, CoreError> {
    let mut dtypes: BTreeMap<String, Dtype> = table
        .columns
        .iter()
        .filter(|c| {
            !native
                .and_then(|t| t.columns.iter().find(|n| n.name == c.name))
                .is_some_and(|n| n.untyped)
        })
        .map(|c| (c.name.clone(), c.dtype))
        .collect();
    for (name, arrived) in arriving {
        if let Some(before) = dtypes.get(name).copied() {
            let at = scoring.first().and_then(|b| b.schema().index_of(name).ok());
            let retain = if matches!(before, Dtype::Int32 | Dtype::Int64)
                && matches!(arrived, Dtype::Float32 | Dtype::Float64)
            {
                if let Some(at) = at {
                    nullable_whole(scoring, at, storage.get(at).copied().flatten())?
                } else {
                    false
                }
            } else {
                false
            };
            let arrived = if retain { before } else { *arrived };
            dtypes.insert(name.clone(), widened(before, arrived));
        } else if table.columns.iter().any(|c| &c.name == name) {
            dtypes.insert(name.clone(), *arrived);
        }
    }
    for column in &table.columns {
        dtypes.entry(column.name.clone()).or_insert(Dtype::String);
    }
    Ok(dtypes)
}

fn primitive_inference(inference: DirectInference) -> Result<Value, CoreError> {
    let integer = |digits: String| -> Result<Value, CoreError> {
        let body = digits.strip_prefix('-').unwrap_or(&digits);
        if body.is_empty()
            || !body.bytes().all(|b| b.is_ascii_digit())
            || (body.len() > 1 && body.starts_with('0'))
        {
            return Err(CoreError::protocol(
                Sentence::InvalidMessage,
                "a direct inference integer has noncanonical digits",
            ));
        }
        let exact = digits
            .parse::<f64>()
            .ok()
            .filter(|v| v.is_finite())
            .is_some_and(|v| format!("{v:.0}") == digits);
        if !exact {
            return Err(
                invalid(Sentence::DirectOptionInteger).with_field("digits", Detail::text(&digits))
            );
        }
        Ok(Value::Number(digits))
    };
    let mut fields = Vec::new();
    for (name, value) in inference.entries {
        let value = match value {
            PrimitiveOption::Integer(digits) => integer(digits)?,
            PrimitiveOption::Value(Scalar::Int64(v)) => integer(v.to_string())?,
            PrimitiveOption::Value(Scalar::UInt64(v)) => integer(v.to_string())?,
            PrimitiveOption::Value(Scalar::Null) => Value::Null,
            PrimitiveOption::Value(Scalar::Bool(v)) => Value::Bool(v),
            PrimitiveOption::Value(Scalar::Utf8(v)) => Value::Text(v),
            PrimitiveOption::Value(Scalar::Float64(v)) => Value::float(v).ok_or_else(|| {
                CoreError::protocol(
                    Sentence::InvalidMessage,
                    "direct inference floats must be finite",
                )
            })?,
            _ => {
                return Err(CoreError::protocol(
                    Sentence::InvalidMessage,
                    "direct inference holds a value outside the primitive family",
                ));
            }
        };
        fields.push((name, value));
    }
    Ok(Value::Object(fields))
}

/// Resolve adapter schema columns before its warning and fallible body work.
#[allow(clippy::too_many_arguments)]
pub fn envelope(
    description: &SourceDescription,
    native_schema: &HeldSchema,
    entity: &str,
    context_names: &[String],
    scoring: &[RecordBatch],
    arriving: &[(String, Dtype)],
    storage: &[Option<FloatStorage>],
    target: &str,
    kind: &str,
    supports: fn(Stype, Dtype) -> bool,
) -> Result<(TabularColumns, Vec<CoreWarning>), CoreError> {
    let table = description
        .tables
        .iter()
        .find(|t| t.name == entity)
        .ok_or_else(|| {
            invalid(Sentence::DirectSchemaMissing).with_field("name", Detail::text(entity))
        })?;
    let native = native_schema.tables.iter().find(|t| t.name == entity);
    let dtypes = widths(table, native, scoring, arriving, storage)?;
    let missing: Vec<String> = context_names
        .iter()
        .filter(|n| !dtypes.contains_key(*n))
        .cloned()
        .collect();
    if !missing.is_empty() {
        return Err(invalid(Sentence::DirectSchemaColumns).with_field("columns", texts(missing)));
    }
    let mut columns = Vec::new();
    let mut text_columns = Vec::new();
    for name in context_names {
        let column = table
            .columns
            .iter()
            .find(|c| &c.name == name)
            .ok_or_else(|| {
                invalid(Sentence::DirectSchemaMissing).with_field("name", Detail::text(name))
            })?;
        let dtype = dtypes[name];
        let mut stype = column.stype;
        if stype == Stype::Text {
            text_columns.push(name.clone());
            stype = Stype::Categorical;
        }
        if name == target {
            if kind.contains("classification") {
                stype = Stype::Categorical;
            } else if stype == Stype::Categorical {
                stype = Stype::Numerical;
            }
        }
        if !supports(stype, dtype)
            && !native
                .and_then(|t| t.columns.iter().find(|c| &c.name == name))
                .is_some_and(|c| c.stype_declared)
        {
            stype = if matches!(
                dtype,
                Dtype::Int32 | Dtype::Int64 | Dtype::Float32 | Dtype::Float64
            ) {
                Stype::Numerical
            } else {
                Stype::Categorical
            };
        }
        columns.push(TabularColumn {
            name: name.clone(),
            dtype,
            stype,
        });
    }
    let notices = if text_columns.is_empty() {
        vec![]
    } else {
        vec![
            CoreWarning::new(
                Sentence::DirectTextCategorical,
                "tabular text is categorical",
            )
            .with_field("columns", texts(text_columns)),
        ]
    };
    Ok((
        TabularColumns {
            dtypes: dtypes.into_iter().collect(),
            columns,
            primary_key: table.primary_key.clone(),
            target: target.to_owned(),
        },
        notices,
    ))
}

/// Encode current values with the schema already resolved before text advice.
#[allow(clippy::too_many_arguments)]
pub fn body(
    plan: &TabularColumns,
    context: &[RecordBatch],
    scoring: &[RecordBatch],
    kind: &str,
    outputs: &[String],
    request_id: &str,
    inference: DirectInference,
    float_text: fn(f64) -> String,
) -> Result<Value, CoreError> {
    let dtypes: BTreeMap<&str, Dtype> = plan
        .dtypes
        .iter()
        .map(|(name, dtype)| (name.as_str(), *dtype))
        .collect();
    let columns = plan
        .columns
        .iter()
        .map(|column| {
            (
                column.name.clone(),
                object([
                    ("dtype", text(column.dtype.wire_name())),
                    (
                        "stype",
                        text(match column.stype {
                            Stype::Id => "ID",
                            Stype::Numerical => "numerical",
                            Stype::Text => "text",
                            _ => "categorical",
                        }),
                    ),
                ]),
            )
        })
        .collect();
    let mut instance = object([("columns", Value::Object(columns))]);
    if !plan.primary_key.is_empty() {
        instance.set(
            "primary_key",
            Value::Array(plan.primary_key.iter().map(text).collect()),
        );
    }
    let encode = |batches: &[RecordBatch]| -> Result<Value, CoreError> {
        let names: Vec<String> = batches
            .first()
            .map(|b| {
                b.schema()
                    .fields()
                    .iter()
                    .map(|f| f.name().clone())
                    .collect()
            })
            .unwrap_or_default();
        let mut rows = Vec::new();
        for batch in batches {
            for row in 0..batch.num_rows() {
                rows.push(Value::Array(
                    names
                        .iter()
                        .enumerate()
                        .map(|(at, name)| {
                            let dtype = dtypes.get(name.as_str()).copied().ok_or_else(|| {
                                invalid(Sentence::DirectSchemaMissing)
                                    .with_field("name", Detail::text(name))
                            })?;
                            cell(raw(batch.column(at), row)?, dtype, name, float_text)
                        })
                        .collect::<Result<_, _>>()?,
                ));
            }
        }
        Ok(object([
            ("format", text("arrays")),
            ("columns", Value::Array(names.iter().map(text).collect())),
            ("rows", Value::Array(rows)),
        ]))
    };
    let target_dtype = dtypes.get(plan.target.as_str()).ok_or_else(|| {
        invalid(Sentence::DirectSchemaMissing).with_field("name", Detail::text(&plan.target))
    })?;
    let mut result = object([
        ("model", text(predict::TABULAR)),
        (
            "task",
            object([
                ("kind", text(kind)),
                (
                    "target",
                    object([
                        ("column_name", text(&plan.target)),
                        ("dtype", text(target_dtype.wire_name())),
                    ]),
                ),
            ]),
        ),
        ("schema", object([("instance_table", instance)])),
        ("context", object([("instance_table", encode(context)?)])),
        ("predict", object([("instance_table", encode(scoring)?)])),
        (
            "output",
            object([("fields", Value::Array(outputs.iter().map(text).collect()))]),
        ),
        ("metadata", object([("request_id", text(request_id))])),
    ]);
    let inference = primitive_inference(inference)?;
    if !matches!(&inference,Value::Object(fields) if fields.is_empty()) {
        result.set("inference", inference);
    }
    Ok(result)
}

/// Shared response decoding plus direct batch attribution/key restoration.
pub fn layout(
    payload: &Value,
    response: &PredictionResponse,
    float_text: fn(f64) -> String,
) -> Result<Part, CoreError> {
    let mut part = predict::layout_tabular(payload, response, float_text)?;
    let rows = match payload
        .get("predict")
        .and_then(|p| p.get("instance_table"))
        .and_then(|t| t.get("rows"))
    {
        Some(Value::Array(rows)) => rows,
        _ => return Err(predict::bad_response("predict", "not_an_array")),
    };
    let instance = payload.get("schema").and_then(|s| s.get("instance_table"));
    let key = match instance.and_then(|s| s.get("primary_key")) {
        Some(Value::Array(keys)) if keys.len() == 1 => keys[0].as_text(),
        _ => None,
    };
    let columns = payload
        .get("predict")
        .and_then(|p| p.get("instance_table"))
        .and_then(|t| t.get("columns"));
    let key_at = match (key, columns) {
        (Some(key), Some(Value::Array(columns))) => {
            columns.iter().position(|c| c.as_text() == Some(key))
        }
        _ => None,
    };
    let dtype = key.and_then(|key| {
        instance
            .and_then(|s| s.get("columns"))
            .and_then(|c| c.get(key))
            .and_then(|c| c.get("dtype"))
            .and_then(Value::as_text)
    });
    let keys: Option<Vec<Entity>> = key_at.map(|key_at| {
        rows.iter()
            .map(|row| {
                if let Value::Array(cells) = row {
                    cells
                        .get(key_at)
                        .map_or(Entity::None, |v| predict::entity(v, dtype))
                } else {
                    Entity::None
                }
            })
            .collect()
    });
    let key_texts: BTreeSet<String> = keys
        .as_ref()
        .into_iter()
        .flatten()
        .map(|key| entity_text(key, float_text))
        .collect();
    let mut seen = BTreeSet::new();
    for row in &mut part.rows {
        let at = usize::try_from(row.row_index)
            .ok()
            .filter(|&n| n < rows.len())
            .ok_or_else(|| {
                invalid(Sentence::TabularRowBounds)
                    .with_field("index", Detail::Scalar(Scalar::Int64(row.row_index)))
                    .with_field("rows", Detail::Scalar(Scalar::UInt64(rows.len() as u64)))
            })?;
        if !seen.insert(at) {
            return Err(invalid(Sentence::TabularRowRepeated)
                .with_field("index", Detail::Scalar(Scalar::UInt64(at as u64))));
        }
        if let Some(keys) = &keys {
            if let Some(answered) = response
                .predictions
                .get(row.answer_index as usize)
                .and_then(|item| item.id.as_deref())
            {
                let named = key_named(answered, float_text)
                    .or_else(|| key_texts.contains(answered).then(|| answered.to_owned()));
                if named.is_some_and(|named| named != entity_text(&keys[at], float_text)) {
                    return Err(invalid(Sentence::TabularIdMismatch)
                        .with_field("index", Detail::Scalar(Scalar::UInt64(at as u64)))
                        .with_field("id", Detail::text(answered))
                        .with_field("expected", Detail::Scalar(entity_scalar(&keys[at]))));
                }
            }
            if !matches!(&keys[at], Entity::None) {
                row.id = keys[at].clone();
            }
        }
    }
    let missing: Vec<_> = (0..rows.len()).filter(|n| !seen.contains(n)).collect();
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
    Ok(part)
}

fn entity_text(value: &Entity, float_text: fn(f64) -> String) -> String {
    match value {
        Entity::None => "None".into(),
        Entity::Integer(n) => n.to_string(),
        Entity::Float(v) => float_text(*v),
        Entity::Text(s) => s.clone(),
        Entity::Written(Value::Bool(v)) => {
            if *v {
                "True".into()
            } else {
                "False".into()
            }
        }
        Entity::Written(Value::Text(s)) => s.clone(),
        Entity::Written(v) => String::from_utf8_lossy(&v.to_bytes()).into_owned(),
    }
}
fn entity_scalar(value: &Entity) -> Scalar {
    match value {
        Entity::None => Scalar::Null,
        Entity::Integer(n) => Scalar::Int64(*n),
        Entity::Float(v) => Scalar::Float64(*v),
        Entity::Text(s) => Scalar::Utf8(s.clone()),
        Entity::Written(Value::Bool(v)) => Scalar::Bool(*v),
        Entity::Written(v) => Scalar::Utf8(String::from_utf8_lossy(&v.to_bytes()).into_owned()),
    }
}
fn key_named(value: &str, float_text: fn(f64) -> String) -> Option<String> {
    if value.len() > 1024 {
        return None;
    }
    if let Ok(Value::Array(parts)) = sdm_wire::json::read(value.as_bytes()) {
        if let [part] = parts.as_slice() {
            return Some(entity_text(&predict::entity(part, None), float_text));
        }
    }
    let part = value.trim().strip_prefix('[')?.strip_suffix(']')?.trim();
    if matches!(part, "True" | "False" | "None") {
        return Some(part.to_owned());
    }
    let quote = part.chars().next()?;
    if !matches!(quote, '\'' | '"') {
        return None;
    }
    let content = part.strip_prefix(quote)?.strip_suffix(quote)?;
    // Numeric/Boolean keys sent as strings contain no Python escapes.
    // Plain quoted strings are read as text; no expression or nested value is evaluated.
    (!content.contains(['\'', '"', '\\'])).then(|| content.to_owned())
}

/// Direct one-shot leakage advice, before schema consistency/body encoding.
/// Keys are compared as the same declared/widened wire values on both halves.
#[allow(clippy::too_many_arguments)]
pub fn leakage(
    description: &SourceDescription,
    native_schema: &HeldSchema,
    entity: &str,
    context: &[RecordBatch],
    scoring: &[RecordBatch],
    arriving: &[(String, Dtype)],
    storage: &[Option<FloatStorage>],
    target: &str,
    float_text: fn(f64) -> String,
) -> Result<(usize, Option<CoreWarning>), CoreError> {
    let Some(table) = description.tables.iter().find(|t| t.name == entity) else {
        return Ok((0, None));
    };
    if table.primary_key.is_empty() {
        return Ok((0, None));
    }
    let context_schema = context.first().map(RecordBatch::schema);
    let scoring_schema = scoring.first().map(RecordBatch::schema);
    let (Some(context_schema), Some(scoring_schema)) = (context_schema, scoring_schema) else {
        return Ok((0, None));
    };
    let Ok(target_at) = context_schema.index_of(target) else {
        return Ok((0, None));
    };
    let positions = |schema: &arrow_schema::Schema| {
        table
            .primary_key
            .iter()
            .map(|key| schema.index_of(key).ok())
            .collect::<Option<Vec<_>>>()
    };
    let (Some(context_at), Some(scoring_at)) =
        (positions(&context_schema), positions(&scoring_schema))
    else {
        return Ok((0, None));
    };
    let native = native_schema.tables.iter().find(|t| t.name == entity);
    let Ok(dtypes) = widths(table, native, scoring, arriving, storage) else {
        return Ok((0, None));
    };
    let key =
        |batch: &RecordBatch, row: usize, at: &[usize]| -> Result<Option<Vec<u8>>, CoreError> {
            let values = table
                .primary_key
                .iter()
                .zip(at)
                .map(|(name, &at)| {
                    let dtype = dtypes.get(name).copied().unwrap_or(Dtype::String);
                    cell(raw(batch.column(at), row)?, dtype, name, float_text)
                })
                .collect::<Result<Vec<_>, _>>()?;
            if values.iter().any(|v| matches!(v, Value::Null)) {
                return Ok(None);
            }
            Ok(Some(Value::Array(values).to_bytes()))
        };
    let mut answered = BTreeSet::new();
    for batch in context {
        for row in 0..batch.num_rows() {
            if !batch.column(target_at).is_null(row) {
                let Ok(key) = key(batch, row, &context_at) else {
                    return Ok((0, None));
                };
                if let Some(key) = key {
                    answered.insert(key);
                }
            }
        }
    }
    let mut shown = 0;
    let mut rows = 0;
    for batch in scoring {
        rows += batch.num_rows();
        for row in 0..batch.num_rows() {
            let Ok(key) = key(batch, row, &scoring_at) else {
                return Ok((0, None));
            };
            if key.is_some_and(|k| answered.contains(&k)) {
                shown += 1;
            }
        }
    }
    Ok((
        shown as usize,
        (shown > 0).then(|| {
            CoreWarning::new(
                Sentence::DirectAnswersShown,
                "scoring rows carry answers in context",
            )
            .with_field("shown", Detail::Scalar(Scalar::UInt64(shown)))
            .with_field("rows", Detail::Scalar(Scalar::UInt64(rows as u64)))
            .with_field("target", Detail::text(target))
        }),
    ))
}
