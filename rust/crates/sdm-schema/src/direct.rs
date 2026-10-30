// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Whole-column schema inference for the ordinary direct consumer.
//! This has no provider, catalog, graph or sampling dependency.

pub mod dtype;
pub mod observed;

use std::collections::{BTreeMap, BTreeSet};

use arrow_array::RecordBatch;
use arrow_array::cast::AsArray;
use arrow_array::types::{Decimal128Type, Decimal256Type};
use arrow_schema::{DataType, Schema};
use sdm_contract::data::Relation;
use sdm_contract::direct::storage::{self, DirectStorage, StorageOwner};
use sdm_contract::ops::{Dtype, HeldSchema, Stype};
use sdm_contract::{ColumnType, CoreError, CoreWarning, Scalar, Sentence};

use crate::assemble::{self, Given};
use crate::infer::{InferredColumn, InferredTable};
use crate::keys::KEY_UNIQUENESS;
use crate::values::{self, Floats, Value};

/// Earliest real native instant, retaining precision before wire encoding.
pub fn time_start(
    schema: &Schema,
    batches: &[RecordBatch],
) -> Result<sdm_contract::direct::schema::TimeStart, CoreError> {
    if schema.fields().len() != 1 {
        return Err(CoreError::protocol(
            Sentence::InvalidMessage,
            "native time observation requires one column",
        ));
    }
    let Some(DirectStorage::Timestamp { unit, .. }) = storage::read(schema.field(0))? else {
        return Err(CoreError::protocol(
            Sentence::InvalidMessage,
            "native time observation requires timestamp storage",
        ));
    };
    let factor = match unit {
        storage::TimestampUnit::Seconds => 1_000_000_000,
        storage::TimestampUnit::Milliseconds => 1_000_000,
        storage::TimestampUnit::Microseconds => 1_000,
        storage::TimestampUnit::Nanoseconds => 1,
    };
    let values = values::column(batches, 0, Floats::NanValue)?;
    let nanos = crate::infer::earliest_direct_instant(values.into_iter().flatten().filter_map(
        |v| match v {
            Value::Int(count) => Some(count * factor),
            _ => None,
        },
    ));
    Ok(sdm_contract::direct::schema::TimeStart {
        nanos: nanos.map(|v| v.to_string()),
    })
}

/// Wire storage meaning, distinct from Arrow's physical width.
pub fn storage_dtypes(schema: &Schema, batches: &[RecordBatch]) -> Result<Vec<Dtype>, CoreError> {
    schema
        .fields()
        .iter()
        .enumerate()
        .map(|(at, field)| {
            let native = storage::read(field)?;
            if native == Some(DirectStorage::Categorical) {
                return Ok(Dtype::String);
            }
            if matches!(native, Some(DirectStorage::Timestamp { .. })) {
                return Ok(Dtype::Time);
            }
            if let Some(DirectStorage::Decimal { precision, scale }) = native {
                if scale > 0 {
                    return Ok(Dtype::Float64);
                }
                if precision <= 18 {
                    return Ok(Dtype::Int64);
                }
                let values = decimal_values(batches, at)?;
                let suffix = "0".repeat(usize::from(scale.unsigned_abs()));
                let fits = values.iter().flatten().all(|value| {
                    let Value::Text(coefficient) = value else {
                        return false;
                    };
                    coefficient == "0" || format!("{coefficient}{suffix}").parse::<i64>().is_ok()
                });
                return Ok(if fits { Dtype::Int64 } else { Dtype::String });
            }
            Ok(match field.data_type() {
                DataType::Boolean => Dtype::Bool,
                DataType::Int8
                | DataType::Int16
                | DataType::Int32
                | DataType::Int64
                | DataType::UInt8
                | DataType::UInt16
                | DataType::UInt32 => Dtype::Int64,
                DataType::UInt64 => {
                    let wide = values::column(batches, at, Floats::NanMissing)?
                        .iter()
                        .flatten()
                        .any(|value| matches!(value, Value::Int(n) if *n > i128::from(i64::MAX)));
                    if wide { Dtype::String } else { Dtype::Int64 }
                }
                DataType::Float32 => Dtype::Float32,
                DataType::Float64 => Dtype::Float64,
                DataType::Utf8 if native == Some(DirectStorage::String) => Dtype::String,
                _ => {
                    return Err(CoreError::new(
                        Sentence::DirectUnfinished,
                        "the direct column is not primitive numeric",
                    )
                    .with_field(
                        "family",
                        sdm_contract::Detail::text("primitive numeric or Boolean columns"),
                    ));
                }
            })
        })
        .collect()
}

/// Infer one entire context, with the native direct constructor's fixed key nominee.
pub fn infer(
    name: &str,
    schema: &Schema,
    batches: &[RecordBatch],
) -> Result<(HeldSchema, Vec<CoreWarning>), CoreError> {
    check_name(name)?;
    for field in schema.fields() {
        check_name(field.name())?;
    }
    let rows: usize = batches.iter().map(RecordBatch::num_rows).sum();
    let dtypes = storage_dtypes(schema, batches)?;
    let values = schema
        .fields()
        .iter()
        .enumerate()
        .map(|(at, field)| native_values(batches, at, field))
        .collect::<Result<Vec<_>, _>>()?;
    let columns: Vec<_> = schema
        .fields()
        .iter()
        .zip(&dtypes)
        .enumerate()
        .map(|(at, (field, dtype))| {
            Ok(InferredColumn {
                name: field.name().clone(),
                dtype: *dtype,
                stype: match dtype {
                    Dtype::Int64 | Dtype::Float32 | Dtype::Float64 => Stype::Numerical,
                    Dtype::Bool => Stype::Categorical,
                    Dtype::Date | Dtype::Time => Stype::Timestamp,
                    _ if values::distinct(&values[at]) <= 64 => Stype::Categorical,
                    _ => Stype::Text,
                },
                stype_before_key: None,
                received: ColumnType::from_arrow(field.name(), field.data_type())?,
                nullable: field.is_nullable(),
                unique: false,
                exact_number: None,
            })
        })
        .collect::<Result<_, CoreError>>()?;
    let primary = columns.iter().enumerate().find_map(|(at, column)| {
        let lower = column.name.to_lowercase();
        (rows > 0
            && ["_id", "_key", "id", "key"]
                .iter()
                .any(|suffix| lower.ends_with(suffix))
            && assemble::can_identify(column.dtype)
            && values::distinct(&values[at]) as f64 >= rows as f64 * KEY_UNIQUENESS)
            .then_some(at)
    });
    let repeated = primary.map_or(0, |at| {
        values[at]
            .iter()
            .flatten()
            .count()
            .saturating_sub(values::distinct(&values[at]))
    });
    let primary_key = primary
        .map(|at| vec![schema.field(at).name().clone()])
        .unwrap_or_default();
    let mut instants = Vec::with_capacity(columns.len());
    let mut firsts = Vec::with_capacity(columns.len());
    for (at, field) in schema.fields().iter().enumerate() {
        if let Some(DirectStorage::Timestamp { unit, zone }) = storage::read(field)? {
            let nanos = |count: i128| match unit {
                storage::TimestampUnit::Seconds => count * 1_000_000_000,
                storage::TimestampUnit::Milliseconds => count * 1_000_000,
                storage::TimestampUnit::Microseconds => count * 1_000,
                storage::TimestampUnit::Nanoseconds => count,
            };
            instants.push(Ok(values[at]
                .iter()
                .map(|v| match v {
                    Some(Value::Int(count)) => Some(nanos(*count)),
                    _ => None,
                })
                .collect()));
            firsts.push(values[at].iter().flatten().find_map(|value| {
                let Value::Int(count) = value else {
                    return None;
                };
                let (native_unit, count) = if unit == storage::TimestampUnit::Nanoseconds {
                    (sdm_contract::Unit::Ns, *count)
                } else {
                    (sdm_contract::Unit::Us, nanos(*count).div_euclid(1_000))
                };
                i64::try_from(count).ok().map(|value| Scalar::Timestamp {
                    unit: native_unit,
                    zone: zone.clone(),
                    value,
                })
            }));
        } else {
            instants.push(Ok(vec![]));
            firsts.push(None);
        }
    }
    let time =
        crate::infer::choose_direct(&columns, rows, &primary_key, &instants, &values, &firsts)?;
    let table = InferredTable {
        name: name.into(),
        source: Relation::named(name),
        columns,
        primary_key,
        time_column: time.column.clone(),
        time,
        left_out: vec![],
        unread_times: vec![],
        day_first: vec![],
        tied_keys: vec![],
        unique: vec![],
        declared: BTreeSet::new(),
    };
    let built = assemble::assemble(
        vec![table],
        vec![],
        Given {
            relationships: Some(&[]),
            time_columns: None,
        },
        Some(Scalar::UInt64(repeated as u64)),
    )?;
    let held = crate::held::held(&built.description, &BTreeSet::new(), &BTreeMap::new());
    Ok((held, built.warnings))
}

/// Present target values, using the same equality as ordinary NumPy inference.
pub fn distinct(batches: &[RecordBatch], column: usize) -> Result<usize, CoreError> {
    let Some(batch) = batches.first() else {
        return Ok(0);
    };
    Ok(values::distinct(&native_values(
        batches,
        column,
        batch.schema().field(column),
    )?))
}

fn native_values(
    batches: &[RecordBatch],
    column: usize,
    field: &arrow_schema::Field,
) -> Result<Vec<Option<Value>>, CoreError> {
    let native = storage::read(field)?;
    if matches!(native, Some(DirectStorage::Decimal { .. })) {
        return decimal_values(batches, column);
    }
    let masked = matches!(
        native,
        Some(DirectStorage::Primitive {
            owner: StorageOwner::Nullable | StorageOwner::Arrow,
            ..
        })
    );
    let mut values = values::column(
        batches,
        column,
        if masked {
            Floats::NanValue
        } else {
            Floats::NanMissing
        },
    )?;
    if matches!(
        native,
        Some(DirectStorage::Primitive {
            owner: StorageOwner::Nullable,
            ..
        })
    ) {
        // Masked native arrays retain present NaNs, but their uniqueness
        // operation identifies signed zeros and all NaN payloads alike.
        // This direct-only rule does not change warehouse comparison policy.
        for value in values.iter_mut().flatten() {
            if let Value::Float(bits) = value {
                match field.data_type() {
                    DataType::Float32 => {
                        let number = f32::from_bits(*bits as u32);
                        if number == 0.0 {
                            *bits = 0;
                        } else if number.is_nan() {
                            *bits = u64::from(f32::NAN.to_bits());
                        }
                    }
                    DataType::Float64 => {
                        let number = f64::from_bits(*bits);
                        if number == 0.0 {
                            *bits = 0;
                        } else if number.is_nan() {
                            *bits = f64::NAN.to_bits();
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    Ok(values)
}

fn decimal_values(batches: &[RecordBatch], column: usize) -> Result<Vec<Option<Value>>, CoreError> {
    let mut values = Vec::new();
    for batch in batches {
        let array = batch.column(column);
        match array.data_type() {
            DataType::Decimal128(_, _) => values.extend(
                array
                    .as_primitive::<Decimal128Type>()
                    .iter()
                    .map(|v| v.map(|v| Value::Text(v.to_string()))),
            ),
            DataType::Decimal256(_, _) => values.extend(
                array
                    .as_primitive::<Decimal256Type>()
                    .iter()
                    .map(|v| v.map(|v| Value::Text(v.to_string()))),
            ),
            _ => {
                return Err(CoreError::protocol(
                    Sentence::InvalidMessage,
                    "decimal observation has nondecimal storage",
                ));
            }
        }
    }
    Ok(values)
}

fn check_name(name: &str) -> Result<(), CoreError> {
    if name.is_empty() {
        return Err(
            CoreError::new(Sentence::DirectNameEmpty, "a frame name is empty")
                .with_field("name", sdm_contract::Detail::text(name)),
        );
    }
    let characters: Vec<_> = assemble::NOT_IN_A_NAME
        .iter()
        .filter(|character| name.contains(**character))
        .map(|character| Scalar::Utf8(character.to_string()))
        .collect();
    if !characters.is_empty() {
        return Err(CoreError::new(
            Sentence::DirectNameCharacters,
            "a frame name contains invalid characters",
        )
        .with_field("name", sdm_contract::Detail::text(name))
        .with_field("characters", sdm_contract::Detail::List(characters)));
    }
    Ok(())
}
