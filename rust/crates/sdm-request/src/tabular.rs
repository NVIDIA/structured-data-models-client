// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Declared-type tabular encoding, independent of gathered envelopes.

use crate::predict::{self, Entity};
use arrow_array::cast::AsArray;
use arrow_array::types::*;
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::DataType;
use sdm_contract::direct::{
    ColumnFlags, DirectInference, EncodedDirectBody, EncodedTableShape, FloatStorage,
    PrimitiveOption, TabularColumn, TabularColumns,
};
use sdm_contract::direct::storage;
use sdm_contract::ops::{DescribedTable, Dtype, SourceDescription, Stype};
use sdm_contract::{CoreError, CoreWarning, Detail, Scalar, Sentence};
use sdm_wire::json::Value;
use std::collections::BTreeMap;

pub mod response;
pub mod cell;
mod decimal;

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
        Dtype::Bool => Some(0),
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

pub fn resolve_dtype(input: sdm_contract::direct::dtype::DtypeResolutionObservation) -> Result<sdm_contract::direct::dtype::DtypeResolution, CoreError> {
    use sdm_contract::direct::dtype::{DtypeResolution, DtypeResolutionObservation as O};
    let dtype = match input {
        O::NullableInteger { declared, equal } => {
            if !matches!(declared, Dtype::Int32 | Dtype::Int64) {
                return Err(CoreError::protocol(Sentence::InvalidMessage, "nullable integer retention has a noninteger declaration"));
            }
            equal.then_some(declared)
        }
        O::Widen { before, arriving, untyped } => {
            if untyped { before } else { Some(before.map_or(arriving, |before| widened(before, arriving))) }
        }
    };
    Ok(DtypeResolution { dtype })
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
        DataType::Utf8 => Scalar::Utf8(array.as_string::<i32>().value(row).to_owned()),
        DataType::LargeUtf8 => Scalar::Utf8(array.as_string::<i64>().value(row).to_owned()),
        DataType::Decimal128(precision, scale) => Scalar::Decimal128 {
            precision: *precision, scale: *scale,
            value: array.as_primitive::<Decimal128Type>().value(row),
        },
        DataType::Decimal256(precision, scale) => Scalar::Decimal256 {
            precision: *precision, scale: *scale,
            value: array.as_primitive::<Decimal256Type>().value(row),
        },
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
    match value {
        Scalar::Decimal128 { value, scale, .. } => return decimal::encode(&value.to_string(), scale, dtype, column),
        Scalar::Decimal256 { value, scale, .. } => return decimal::encode(&value.to_string(), scale, dtype, column),
        _ => {}
    }
    if dtype == Dtype::String {
        return Ok(text(match value {
            Scalar::Utf8(v) => v,
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

fn encoded_cell(
    batch: &RecordBatch,
    at: usize,
    row: usize,
    native: Option<&storage::DirectStorage>,
    dtype: Dtype,
    column: &str,
    float_text: fn(f64) -> String,
) -> Result<Value, CoreError> {
    if let Some(storage::DirectStorage::Encoded { dtype: encoded_dtype }) = native {
        if *encoded_dtype != dtype {
            return Err(CoreError::protocol(Sentence::InvalidMessage, "encoded cell dtype disagrees with resolved wire dtype"));
        }
        let Scalar::Utf8(encoded) = raw(batch.column(at), row)? else {
            return Err(CoreError::protocol(Sentence::InvalidMessage, "encoded cell lacks its scalar representation"));
        };
        return Ok(match sdm_contract::direct::cell::encoded_scalar(&encoded)? {
            Scalar::Null => Value::Null,
            Scalar::Bool(v) => Value::Bool(v),
            Scalar::Utf8(v) => text(v),
            Scalar::Int64(v) => Value::Number(v.to_string()),
            Scalar::Float64(v) if v.is_finite() => crate::payload::float_cell(v, column)?,
            _ => return Err(CoreError::protocol(Sentence::InvalidMessage, "encoded cell is not a model JSON scalar")),
        });
    }
    if let Some(storage::DirectStorage::Timestamp { unit, .. }) = native
    {
        let value = raw(batch.column(at), row)?;
        if matches!(value, Scalar::Null) { return Ok(Value::Null); }
        if !matches!(dtype, Dtype::Date | Dtype::Time) {
            return Err(crate::predict::not_yet("direct_native_timestamp_conversion"));
        }
        let Scalar::Int64(count) = value else {
            return Err(CoreError::protocol(Sentence::InvalidMessage, "timestamp count is not int64"));
        };
        return Ok(text(crate::payload::iso_microsecond_count(unit.microseconds(count))));
    }
    cell(raw(batch.column(at), row)?, dtype, column, float_text)
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

/// Finish a proven-passive column before the host observes the next column.
pub fn encode_column(batches: &[RecordBatch], dtype: Dtype, float_text: fn(f64) -> String) -> Result<Vec<Scalar>, CoreError> {
    let mut values = Vec::new();
    for batch in batches {
        if batch.num_columns() != 1 {
            return Err(CoreError::protocol(Sentence::InvalidMessage, "direct column encoding requires exactly one column"));
        }
        let schema = batch.schema();
        let native = storage::read(schema.field(0))?;
        for row in 0..batch.num_rows() {
            let cell = encoded_cell(batch, 0, row, native.as_ref(), dtype, schema.field(0).name(), float_text)?;
            values.push(match cell {
                Value::Null => Scalar::Null,
                Value::Bool(value) => Scalar::Bool(value),
                Value::Text(value) => Scalar::Utf8(value),
                Value::Number(value) if matches!(dtype, Dtype::Int32 | Dtype::Int64) => Scalar::Int64(value.parse().map_err(|_| CoreError::protocol(Sentence::InvalidMessage, "encoded integer is not int64"))?),
                Value::Number(value) => Scalar::Float64(value.parse().map_err(|_| CoreError::protocol(Sentence::InvalidMessage, "encoded number is not a float"))?),
                _ => return Err(CoreError::protocol(Sentence::InvalidMessage, "encoded column contains a nonscalar value")),
            });
        }
    }
    Ok(values)
}

fn widths(
    table: &DescribedTable,
    native: Option<&[ColumnFlags]>,
    scoring: &[RecordBatch],
    arriving: &[(String, Dtype)],
    storage: &[Option<FloatStorage>],
) -> Result<BTreeMap<String, Dtype>, CoreError> {
    let mut dtypes: BTreeMap<String, Dtype> = table
        .columns
        .iter()
        .filter(|c| {
            !native
                .and_then(|t| t.iter().find(|n| n.name == c.name))
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

fn json_integer(digits: String) -> Result<Value, CoreError> {
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
        Ok(Value::Number(digits))
}

fn option_integer(digits: String) -> Result<Value, CoreError> {
        json_integer(digits.clone())?;
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
}

pub fn check_option(input: &sdm_contract::direct::DirectOptionObservation) -> Result<(), CoreError> {
    if let Some(integer) = &input.integer { option_integer(integer.clone())?; }
    if input.repeated { return Err(invalid(Sentence::DirectOptionCycle)); }
    if input.depth > 100 {
        return Err(invalid(Sentence::DirectOptionDepth).with_field("limit", Detail::Scalar(Scalar::UInt64(100))));
    }
    Ok(())
}

fn option_value(value: PrimitiveOption) -> Result<Value, CoreError> {
    Ok(match value {
        // The reached native integer coercion was checked before traversal
        // continued. Its JSON spelling can differ for a native subclass;
        // validating that spelling must not repeat the coercion policy.
        PrimitiveOption::Integer(digits) => json_integer(digits)?,
        PrimitiveOption::Value(Scalar::Int64(v)) => option_integer(v.to_string())?,
        PrimitiveOption::Value(Scalar::UInt64(v)) => option_integer(v.to_string())?,
        PrimitiveOption::Value(Scalar::Null) => Value::Null,
        PrimitiveOption::Value(Scalar::Bool(v)) => Value::Bool(v),
        PrimitiveOption::Value(Scalar::Utf8(v)) => Value::Text(v),
        PrimitiveOption::Value(Scalar::Float64(v)) => Value::float(v).ok_or_else(|| {
            CoreError::protocol(Sentence::InvalidMessage, "direct inference floats must be finite")
        })?,
        PrimitiveOption::Array(values) => Value::Array(values.into_iter().map(option_value).collect::<Result<_, _>>()?),
        PrimitiveOption::Object(fields) => Value::Object(fields.into_iter().map(|(name, value)| {
            Ok((name, option_value(value)?))
        }).collect::<Result<_, CoreError>>()?),
        _ => return Err(CoreError::protocol(Sentence::InvalidMessage, "direct inference holds a value outside the JSON family")),
    })
}

fn primitive_inference(inference: DirectInference) -> Result<Value, CoreError> {
    let mut fields = Vec::new();
    for (name, value) in inference.entries {
        fields.push((name, option_value(value)?));
    }
    Ok(Value::Object(fields))
}

/// Resolve adapter schema columns before its warning and fallible body work.
#[allow(clippy::too_many_arguments)]
pub fn envelope(
    description: &SourceDescription,
    native_schema: &[ColumnFlags],
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
    let dtypes = widths(table, Some(native_schema), scoring, arriving, storage)?;
    schema_columns(table, native_schema, context_names, dtypes, target, kind, supports)
}

/// Adapter meaning with wire dtypes already resolved at their native boundaries.
#[allow(clippy::too_many_arguments)]
pub fn schema_columns(
    table: &DescribedTable,
    native_schema: &[ColumnFlags],
    context_names: &[String],
    dtypes: BTreeMap<String, Dtype>,
    target: &str,
    kind: &str,
    supports: fn(Stype, Dtype) -> bool,
) -> Result<(TabularColumns, Vec<CoreWarning>), CoreError> {
    let native = Some(native_schema);
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
        } else if name == target {
            if kind.contains("classification") {
                stype = Stype::Categorical;
            } else if stype == Stype::Categorical {
                stype = Stype::Numerical;
            }
        }
        if !supports(stype, dtype)
            && !native
                .and_then(|t| t.iter().find(|c| &c.name == name))
                .is_some_and(|c| c.stype_declared)
        {
            stype = if matches!(dtype, Dtype::Date | Dtype::Time) {
                Stype::Timestamp
            } else if matches!(
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
            target: target.to_owned(),
        },
        notices,
    ))
}

fn instance_schema(plan: &TabularColumns, primary_key: &[String]) -> Value {
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
                        text(crate::direct::stype_name(column.stype)),
                    ),
                ]),
            )
        })
        .collect();
    let mut instance = object([("columns", Value::Object(columns))]);
    if !primary_key.is_empty() {
        instance.set(
            "primary_key",
            Value::Array(primary_key.iter().map(text).collect()),
        );
    }
    instance
}

/// Encode current values with the schema already resolved before text advice.
pub fn body(
    plan: &TabularColumns,
    context: &[RecordBatch],
    scoring: &[RecordBatch],
    primary_key: &[String],
    kind: &str,
    outputs: &[String],
    float_text: fn(f64) -> String,
) -> Result<Value, CoreError> {
    let dtypes: BTreeMap<&str, Dtype> = plan
        .dtypes
        .iter()
        .map(|(name, dtype)| (name.as_str(), *dtype))
        .collect();
    let instance = instance_schema(plan, primary_key);
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
        let count = batches.iter().map(RecordBatch::num_rows).sum();
        let mut rows = vec![Vec::with_capacity(names.len()); count];
        // The native encoder consumes a whole column before the next, across
        // every Arrow batch, then transposes. Competing failures keep that order.
        for (at, name) in names.iter().enumerate() {
            let dtype = dtypes.get(name.as_str()).copied().ok_or_else(|| {
                invalid(Sentence::DirectSchemaMissing).with_field("name", Detail::text(name))
            })?;
            let mut position = 0;
            for batch in batches {
                let native = storage::read(batch.schema().field(at))?;
                for row in 0..batch.num_rows() {
                    rows[position].push(encoded_cell(
                        batch, at, row, native.as_ref(),
                        dtype,
                        name,
                        float_text,
                    )?);
                    position += 1;
                }
            }
        }
        let rows = rows.into_iter().map(Value::Array).collect();
        Ok(object([
            ("format", text("arrays")),
            ("columns", Value::Array(names.iter().map(text).collect())),
            ("rows", Value::Array(rows)),
        ]))
    };
    let target_dtype = dtypes.get(plan.target.as_str()).ok_or_else(|| {
        invalid(Sentence::DirectSchemaMissing).with_field("name", Detail::text(&plan.target))
    })?;
    let result = object([
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
    ]);
    Ok(result)
}

/// Add only the observations reached after all row encoding has succeeded.
pub fn complete_body(
    mut payload: Value,
    request_id: &str,
    inference: DirectInference,
) -> Result<Value, CoreError> {
    payload.set("metadata", object([("request_id", text(request_id))]));
    let inference = primitive_inference(inference)?;
    if !matches!(&inference,Value::Object(fields) if fields.is_empty()) {
        payload.set("inference", inference);
    }
    Ok(payload)
}

/// The owned envelope must agree with its useful typed wire facts.
/// This validates serialized coherence, without re-encoding native rows or
/// rerunning pre-advice model, capacity or constructor policy.
pub fn check_body(
    payload: &Value,
    body: &EncodedDirectBody,
    outputs: &[String],
) -> Result<(), CoreError> {
    let protocol = || {
        CoreError::protocol(
            Sentence::InvalidMessage,
            "encoded tabular body disagrees with its typed facts",
        )
    };
    let dtypes: BTreeMap<&str, Dtype> = body
        .columns
        .dtypes
        .iter()
        .map(|(name, dtype)| (name.as_str(), *dtype))
        .collect();
    let target = dtypes
        .get(body.columns.target.as_str())
        .ok_or_else(protocol)?;
    let task = object([
        ("kind", text(&body.kind)),
        (
            "target",
            object([
                ("column_name", text(&body.columns.target)),
                ("dtype", text(target.wire_name())),
            ]),
        ),
    ]);
    if !matches!(payload,Value::Object(fields) if fields.len()==6)
        || payload.get("model") != Some(&text(predict::TABULAR))
        || payload.get("task") != Some(&task)
        || payload.get("schema")
            != Some(&object([(
                "instance_table",
                instance_schema(&body.columns, &body.primary_key),
            )]))
        || payload.get("output")
            != Some(&object([(
                "fields",
                Value::Array(outputs.iter().map(text).collect()),
            )]))
        || payload.get("metadata").is_some()
        || payload.get("inference").is_some()
        || body
            .columns
            .columns
            .iter()
            .any(|column| dtypes.get(column.name.as_str()) != Some(&column.dtype))
    {
        return Err(protocol());
    }
    let table = |half: &str, shape: &EncodedTableShape| -> Result<(), CoreError> {
        let table = payload
            .get(half)
            .and_then(|v| v.get("instance_table"))
            .ok_or_else(protocol)?;
        if !matches!(payload.get(half),Some(Value::Object(fields)) if fields.len()==1)
            || !matches!(table,Value::Object(fields) if fields.len()==3)
            || table.get("format") != Some(&text("arrays"))
            || table.get("columns") != Some(&Value::Array(shape.columns.iter().map(text).collect()))
        {
            return Err(protocol());
        }
        let Some(Value::Array(rows)) = table.get("rows") else {
            return Err(protocol());
        };
        if rows.len() != shape.rows {
            return Err(protocol());
        }
        for row in rows {
            let Value::Array(cells) = row else {
                return Err(protocol());
            };
            if cells.len() != shape.columns.len() {
                return Err(protocol());
            }
            for (name, cell) in shape.columns.iter().zip(cells) {
                let dtype = dtypes.get(name.as_str()).ok_or_else(protocol)?;
                let valid = matches!(cell, Value::Null)
                    || match (dtype, cell) {
                        (Dtype::Bool, Value::Bool(_)) | (Dtype::String | Dtype::Date | Dtype::Time, Value::Text(_)) => true,
                        (
                            Dtype::Int32 | Dtype::Int64,
                            Value::Number(token) | Value::Text(token),
                        ) => {
                            sdm_wire::json::is_integer_token(token)
                                && token.parse::<i64>().is_ok_and(|value| {
                                    // Coherence uses the same canonical codec as EncodeBody:
                                    // a wide int64 is lossless text, not a string column.
                                    crate::payload::int_cell(i128::from(value), name)
                                        .is_ok_and(|encoded| encoded == *cell)
                                })
                        }
                        (Dtype::Float32 | Dtype::Float64, Value::Number(token)) => {
                            token.parse::<f64>().is_ok_and(f64::is_finite)
                        }
                        _ => false,
                    };
                if !valid {
                    return Err(protocol());
                }
            }
        }
        Ok(())
    };
    table("context", &body.context_shape)?;
    table("predict", &body.predict_shape)
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

/// Warn about labelled entities also being scored, using their wire key types.
/// As in the native warning helper, inability to encode a key suppresses advice.
pub fn overlap(
    table: &DescribedTable,
    flags: &[ColumnFlags],
    target: &str,
    context: &[RecordBatch],
    scoring: &[RecordBatch],
    arriving: &[(String, Dtype)],
    float_text: fn(f64) -> String,
) -> Option<CoreWarning> {
    let key = &table.primary_key;
    if key.is_empty() {
        return None;
    }
    let context_schema = context.first()?.schema();
    let scoring_schema = scoring.first()?.schema();
    let target_at = context_schema.index_of(target).ok()?;
    let target_storage = storage::floats(context_schema.field(target_at)).ok()?;
    let context_at: Vec<_> = key
        .iter()
        .map(|name| context_schema.index_of(name).ok())
        .collect::<Option<_>>()?;
    let scoring_at: Vec<_> = key
        .iter()
        .map(|name| scoring_schema.index_of(name).ok())
        .collect::<Option<_>>()?;
    let context_native = context_at.iter().map(|at| storage::read(context_schema.field(*at)))
        .collect::<Result<Vec<_>, _>>().ok()?;
    let scoring_native = scoring_at.iter().map(|at| storage::read(scoring_schema.field(*at)))
        .collect::<Result<Vec<_>, _>>().ok()?;
    let mut key_table = table.clone();
    key_table
        .columns
        .retain(|column| key.contains(&column.name));
    let key_arriving: Vec<_> = arriving
        .iter()
        .filter(|(name, _)| key.contains(name))
        .cloned()
        .collect();
    let storage: Vec<_> = scoring_schema
        .fields()
        .iter()
        .map(|field| storage::floats(field))
        .collect::<Result<Vec<_>, _>>().ok()?;
    let dtypes = widths(&key_table, Some(flags), scoring, &key_arriving, &storage).ok()?;
    let encode_key = |batch: &RecordBatch,
                      row: usize,
                      positions: &[usize],
                      native: &[Option<storage::DirectStorage>]|
     -> Result<Option<Vec<u8>>, CoreError> {
        let mut values = Vec::new();
        for ((name, at), native) in key.iter().zip(positions).zip(native) {
            let value = encoded_cell(batch, *at, row, native.as_ref(), dtypes[name], name, float_text)?;
            if matches!(value, Value::Null) {
                return Ok(None);
            }
            // Both halves use one resolved dtype. Their finite numeric
            // spellings agree except signed zero, which names one entity.
            // Normalize only comparison data, never model wire values.
            let value = match value {
                Value::Number(ref number) if number.parse::<f64>().ok() == Some(0.0) => {
                    Value::Number("0".into())
                }
                other => other,
            };
            values.push(value);
        }
        Ok(Some(Value::Array(values).to_bytes()))
    };
    let mut answered = std::collections::BTreeSet::new();
    for batch in context {
        for row in 0..batch.num_rows() {
            let target = raw(batch.column(target_at), row).ok()?;
            if matches!(target, Scalar::Null)
                || (target_storage != Some(FloatStorage::ValidityMasked)
                    && matches!(target, Scalar::Float64(v) if v.is_nan()))
            {
                continue;
            }
            if let Some(key) = encode_key(batch, row, &context_at, &context_native).ok()? {
                answered.insert(key);
            }
        }
    }
    let mut shown = 0u64;
    for batch in scoring {
        for row in 0..batch.num_rows() {
            if encode_key(batch, row, &scoring_at, &scoring_native)
                .ok()?
                .is_some_and(|key| answered.contains(&key))
            {
                shown += 1;
            }
        }
    }
    (shown > 0).then(|| {
        CoreWarning::new(
            Sentence::DirectOverlap,
            "scoring rows carry their answers in context",
        )
        .with_field("shown", Detail::Scalar(Scalar::UInt64(shown)))
        .with_field(
            "rows",
            Detail::Scalar(Scalar::UInt64(
                scoring.iter().map(|b| b.num_rows() as u64).sum(),
            )),
        )
        .with_field("target", Detail::text(target))
    })
}

/// Compare keys whose native conversions have already been reached in order.
pub fn observed_overlap(
    context: &[RecordBatch], scoring: &[RecordBatch], target: &str, scoring_rows: usize,
) -> Result<Option<CoreWarning>, CoreError> {
    let protocol = || CoreError::protocol(Sentence::InvalidMessage, "observed keys do not share one encoded column layout");
    let (Some(left), Some(right)) = (context.first(), scoring.first()) else { return Ok(None); };
    if left.schema() != right.schema() { return Err(protocol()); }
    if left.num_columns() == 0 { return Ok(None); }
    let native = left.schema().fields().iter().map(|field| storage::read(field)).collect::<Result<Vec<_>, _>>()?;
    let dtypes = native.iter().map(|value| match value {
        Some(storage::DirectStorage::Encoded { dtype }) => Ok(*dtype),
        _ => Err(protocol()),
    }).collect::<Result<Vec<_>, _>>()?;
    let keys = |batches: &[RecordBatch]| -> Result<Vec<Option<Vec<u8>>>, CoreError> {
        let mut keys = Vec::new();
        for batch in batches {
            for row in 0..batch.num_rows() {
                let mut values = Vec::new();
                let mut missing = false;
                for (at, dtype) in dtypes.iter().enumerate() {
                    let value = encoded_cell(batch, at, row, native[at].as_ref(), *dtype,
                        batch.schema().field(at).name(), |v| v.to_string())?;
                    if matches!(value, Value::Null) { missing = true; }
                    let value = match value {
                        Value::Number(ref number) if number.parse::<f64>().ok() == Some(0.0) => Value::Number("0".into()),
                        other => other,
                    };
                    values.push(value);
                }
                keys.push((!missing).then(|| Value::Array(values).to_bytes()));
            }
        }
        Ok(keys)
    };
    let answers: std::collections::BTreeSet<_> = keys(context)?.into_iter().flatten().collect();
    let shown = keys(scoring)?.into_iter().flatten().filter(|key| answers.contains(key)).count();
    Ok((shown > 0).then(|| CoreWarning::new(Sentence::DirectOverlap, "scoring rows carry their answers in context")
        .with_field("shown", Detail::Scalar(Scalar::UInt64(shown as u64)))
        .with_field("rows", Detail::Scalar(Scalar::UInt64(scoring_rows as u64)))
        .with_field("target", Detail::text(target))))
}
