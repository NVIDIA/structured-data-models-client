// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Arrow-native schema composition using the same table and relationship rules.
use std::collections::{BTreeMap, BTreeSet};

use crate::run::{Asker, Attachment, Attachments, internal};
use sdm_contract::{CoreError, frames::DescribeFrameTables, ops::OperationResult};

/// Normalize physical container encodings without losing timestamp units/zones.
/// Both schema inference and graph admission consume this representation.
pub(crate) fn logical(data: crate::ArrowData) -> Result<crate::ArrowData, CoreError> {
    use arrow_schema::{DataType, Field, Schema, TimeUnit};
    use std::sync::Arc;
    let fields: Vec<Field> = data
        .schema
        .fields()
        .iter()
        .map(|field| {
            let mut dtype = field.data_type().clone();
            if let DataType::Dictionary(_, value) = dtype {
                dtype = *value;
            }
            dtype = match dtype {
                DataType::LargeUtf8 | DataType::Utf8View | DataType::Null => DataType::Utf8,
                DataType::Date32 => DataType::Timestamp(TimeUnit::Second, None),
                DataType::Date64 => DataType::Timestamp(TimeUnit::Millisecond, None),
                other => other,
            };
            Field::new(field.name(), dtype, field.is_nullable())
        })
        .collect();
    let schema = Arc::new(Schema::new(fields));
    let batches = data
        .batches
        .iter()
        .map(|batch| {
            let columns = batch
                .columns()
                .iter()
                .zip(schema.fields())
                .map(|(column, field)| {
                    arrow_cast::cast_with_options(
                        column,
                        field.data_type(),
                        &arrow_cast::CastOptions {
                            safe: false,
                            ..Default::default()
                        },
                    )
                    .map_err(|_| internal("frame Arrow type cannot be normalized"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            arrow_array::RecordBatch::try_new_with_options(
                schema.clone(),
                columns,
                &arrow_array::RecordBatchOptions::new().with_row_count(Some(batch.num_rows())),
            )
            .map_err(|_| internal("normalized frame layout is invalid"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(crate::ArrowData::new(schema, batches))
}

/// Match already chosen string storage (wide integers/decimals included).
pub(crate) fn graph_columns(
    data: crate::ArrowData,
    table: &sdm_engine::graph::Table,
) -> Result<crate::ArrowData, CoreError> {
    use std::sync::Arc;
    let fields: Vec<_> = data
        .schema
        .fields()
        .iter()
        .map(|field| {
            let dtype = if table
                .column(field.name())
                .is_some_and(|c| c.dtype == sdm_contract::ops::Dtype::String)
            {
                arrow_schema::DataType::Utf8
            } else {
                match field.data_type() {
                    arrow_schema::DataType::Decimal128(_, scale)
                    | arrow_schema::DataType::Decimal256(_, scale)
                        if *scale < 0 =>
                    {
                        arrow_schema::DataType::Int64
                    }
                    arrow_schema::DataType::Decimal256(precision, scale) => {
                        arrow_schema::DataType::Decimal128((*precision).min(38), *scale)
                    }
                    other => other.clone(),
                }
            };
            arrow_schema::Field::new(field.name(), dtype, field.is_nullable())
        })
        .collect();
    let schema = Arc::new(arrow_schema::Schema::new(fields));
    let batches = data
        .batches
        .iter()
        .map(|batch| {
            let columns = batch
                .columns()
                .iter()
                .zip(schema.fields())
                .map(|(column, field)| {
                    arrow_cast::cast_with_options(column, field.data_type(), &arrow_cast::CastOptions { safe: false, ..Default::default() })
                        .map_err(|_| CoreError::new(sdm_contract::Sentence::ArrowTabularInput, "frame decimal or integer cannot be represented exactly in its declared storage").with_field("reason", sdm_contract::Detail::text("in-memory Relational decimals must fit 38 digits and integral decimal keys must fit signed 64-bit storage")))
                })
                .collect::<Result<Vec<_>, _>>()?;
            arrow_array::RecordBatch::try_new_with_options(
                schema.clone(),
                columns,
                &arrow_array::RecordBatchOptions::new().with_row_count(Some(batch.num_rows())),
            )
            .map_err(|_| internal("frame declared storage has invalid layout"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(crate::ArrowData::new(schema, batches))
}

pub(crate) fn describe(
    asker: &Asker,
    input: DescribeFrameTables,
    mut attached: Attachments,
) -> Result<OperationResult, CoreError> {
    let names: Vec<String> = crate::attached::take_json(&mut attached, input.names)?;
    if names.is_empty()
        || names.len() != input.tables.len()
        || names.iter().collect::<BTreeSet<_>>().len() != names.len()
    {
        return Err(internal(
            "frame schema needs distinct names aligned with tables",
        ));
    }
    let mut tables = Vec::new();
    let mut choices = Vec::new();
    for (name, id) in names.into_iter().zip(input.tables) {
        let Some(Attachment::Arrow(data)) = attached.remove(&id) else {
            return Err(internal("frame schema table is absent"));
        };
        let data = crate::tabular_arrow::normalize(logical(data)?)?;
        let (held, time, warnings) =
            sdm_schema::direct::infer_with_time(&name, &data.schema, &data.batches)?;
        // Assembly below owns relationship-dependent notices. Per-table inference
        // has already checked the native columns and primary key.
        for warning in warnings {
            asker.warn(warning);
        }
        let read = sdm_schema::held::read_frame(held, false)?;
        tables.extend(read.description.tables);
        choices.push(time);
    }
    let description = sdm_contract::ops::SourceDescription {
        tables,
        links: vec![],
    };
    let built = sdm_schema::assemble::assemble_frames(
        description,
        choices,
        &BTreeMap::new(),
        &BTreeMap::new(),
        None,
    )?;
    for warning in built.warnings {
        asker.warn(warning);
    }
    crate::direct::value(
        asker,
        &sdm_schema::held::held(&built.description, &BTreeSet::new(), &BTreeMap::new()),
    )
}
