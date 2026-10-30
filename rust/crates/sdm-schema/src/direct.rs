// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Whole-column schema inference for the ordinary numeric direct consumer.
//! This has no provider, catalog, graph or sampling dependency.

use std::collections::{BTreeMap, BTreeSet};

use arrow_array::RecordBatch;
use arrow_schema::{DataType, Schema};
use sdm_contract::data::Relation;
use sdm_contract::ops::{Dtype, HeldSchema, Stype};
use sdm_contract::{ColumnType, CoreError, CoreWarning, Scalar, Sentence};

use crate::assemble::{self, Given};
use crate::infer::{InferredColumn, InferredTable, TimeChoice};
use crate::keys::KEY_UNIQUENESS;
use crate::values::{self, Floats, Value};

/// Wire storage meaning, distinct from Arrow's physical width.
pub fn storage_dtypes(schema: &Schema, batches: &[RecordBatch]) -> Result<Vec<Dtype>, CoreError> {
    schema
        .fields()
        .iter()
        .enumerate()
        .map(|(at, field)| {
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
                _ => {
                    return Err(CoreError::new(
                        Sentence::DirectUnfinished,
                        "the direct column is not primitive numeric",
                    )
                    .with_field(
                        "family",
                        sdm_contract::Detail::text("NumPy numeric or Boolean columns"),
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
    let values = (0..schema.fields().len())
        .map(|at| values::column(batches, at, Floats::NanMissing))
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
    let table = InferredTable {
        name: name.into(),
        source: Relation::named(name),
        columns,
        primary_key: primary
            .map(|at| vec![schema.field(at).name().clone()])
            .unwrap_or_default(),
        time_column: None,
        time: TimeChoice::default(),
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
    Ok(values::distinct(&values::column(
        batches,
        column,
        Floats::NanMissing,
    )?))
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
