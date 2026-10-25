// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Logical Arrow page buffers, separate from the memory retained by a
//! fallback operation. Accepted pages are copied off shared parent buffers
//! before they are kept. The fallback still bounds its compacted working
//! data and answer with its existing retained-memory check.

use arrow_array::{RecordBatch, RecordBatchOptions, make_array};
use arrow_data::{ArrayData, transform::MutableArrayData};
use arrow_schema::DataType;
use sdm_contract::CoreError;

use crate::run::{ArrowData, internal};

pub(crate) fn logical_bytes(batch: &RecordBatch) -> Result<u64, CoreError> {
    batch.columns().iter().try_fold(0u64, |bytes, array| {
        Ok(bytes.saturating_add(buffers(&array.to_data())?))
    })
}

/// The offsets include their trailing entry; children of lists/maps and
/// structs are measured only over the slice they describe. A dictionary
/// carries its complete dictionary, including values not used on this page.
fn buffers(data: &ArrayData) -> Result<u64, CoreError> {
    let n = data.len() as u64;
    let nulls = if data.nulls().is_some() {
        n.div_ceil(8)
    } else {
        0
    };
    let values = match data.data_type() {
        DataType::Null => 0,
        DataType::Boolean => n.div_ceil(8),
        kind if kind.primitive_width().is_some() => {
            n.saturating_mul(kind.primitive_width().unwrap_or_default() as u64)
        }
        DataType::Utf8 | DataType::Binary | DataType::List(_) | DataType::Map(_, _) => {
            let offsets = data.buffers()[0].typed_data::<i32>();
            if offsets.is_empty() && data.is_empty() {
                return Ok(nulls);
            }
            let start = offsets[data.offset()] as usize;
            let end = offsets[data.offset() + data.len()] as usize;
            let child = match data.data_type() {
                DataType::Utf8 | DataType::Binary => (end - start) as u64,
                _ => buffers(&data.child_data()[0].slice(start, end - start))?,
            };
            (n.saturating_add(1))
                .saturating_mul(4)
                .saturating_add(child)
        }
        DataType::LargeUtf8 | DataType::LargeBinary | DataType::LargeList(_) => {
            let offsets = data.buffers()[0].typed_data::<i64>();
            if offsets.is_empty() && data.is_empty() {
                return Ok(nulls);
            }
            let start = offsets[data.offset()] as usize;
            let end = offsets[data.offset() + data.len()] as usize;
            let child = match data.data_type() {
                DataType::LargeUtf8 | DataType::LargeBinary => (end - start) as u64,
                _ => buffers(&data.child_data()[0].slice(start, end - start))?,
            };
            (n.saturating_add(1))
                .saturating_mul(8)
                .saturating_add(child)
        }
        DataType::Struct(_) => data.child_data().iter().try_fold(0u64, |bytes, child| {
            Ok::<_, CoreError>(
                bytes.saturating_add(buffers(&child.slice(data.offset(), data.len()))?),
            )
        })?,
        DataType::Dictionary(key, _) => {
            let width = key
                .primitive_width()
                .ok_or_else(|| internal("a page's dictionary has no integer key width"))?;
            n.saturating_mul(width as u64)
                .saturating_add(buffers(&data.child_data()[0])?)
        }
        _ => return Err(internal("a bounded page holds a type outside admission")),
    };
    Ok(nulls.saturating_add(values))
}

/// Copy the admitted slice using Arrow's own range-copy kernels, without
/// building a row-index vector. The dictionary's values need their own
/// copy: MutableArrayData normally shares that child unchanged.
fn copy(data: &ArrayData) -> Result<ArrayData, CoreError> {
    let mut builder = MutableArrayData::new(vec![data], false, data.len());
    builder.extend(0, 0, data.len());
    detach_dictionaries(builder.freeze())
}

fn detach_dictionaries(data: ArrayData) -> Result<ArrayData, CoreError> {
    let children = if matches!(data.data_type(), DataType::Dictionary(_, _)) {
        vec![copy(&data.child_data()[0])?]
    } else {
        data.child_data()
            .iter()
            .cloned()
            .map(detach_dictionaries)
            .collect::<Result<Vec<_>, _>>()?
    };
    data.into_builder()
        .child_data(children)
        .build()
        .map_err(|_| internal("a compacted page has invalid Arrow data"))
}

pub(crate) fn compact(data: ArrowData) -> Result<ArrowData, CoreError> {
    let mut batches = Vec::with_capacity(data.batches.len());
    for batch in data.batches {
        // Empty pages contribute no rows and must not accumulate retained
        // batch objects while a provider keeps returning continuations.
        if batch.num_rows() == 0 {
            continue;
        }
        let arrays = batch
            .columns()
            .iter()
            .map(|array| copy(&array.to_data()).map(make_array))
            .collect::<Result<Vec<_>, _>>()?;
        batches.push(
            RecordBatch::try_new_with_options(
                batch.schema(),
                arrays,
                &RecordBatchOptions::new().with_row_count(Some(batch.num_rows())),
            )
            .map_err(|_| internal("a compacted page does not match its schema"))?,
        );
    }
    Ok(ArrowData { batches, ..data })
}

#[cfg(test)]
mod tests;
