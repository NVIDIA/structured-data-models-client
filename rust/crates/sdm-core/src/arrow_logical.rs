// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared physical Arrow normalization for in-memory inputs.
use crate::run::internal;
use sdm_contract::CoreError;

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
