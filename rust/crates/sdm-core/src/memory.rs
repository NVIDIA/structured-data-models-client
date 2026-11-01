// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Logical bytes of the rows a provider actually returned, not shared parent capacity.
use crate::run::internal;
use arrow_array::cast::AsArray;
use arrow_array::types::*;
use arrow_array::{Array, RecordBatch};
use arrow_schema::DataType;
use sdm_contract::CoreError;

fn array_bytes(array: &dyn Array) -> Result<u64, CoreError> {
    let validity = if array.nulls().is_some() {
        array.len().div_ceil(8) as u64
    } else {
        0
    };
    if let DataType::Dictionary(key, _) = array.data_type() {
        macro_rules! dictionary {
            ($t:ty) => {{
                let dictionary = array.as_dictionary::<$t>();
                let unique: std::collections::BTreeSet<usize> = dictionary
                    .keys()
                    .iter()
                    .flatten()
                    .map(|v| v as usize)
                    .collect();
                let indices = dictionary
                    .keys()
                    .to_data()
                    .get_slice_memory_size()
                    .map_err(|_| internal("cannot measure dictionary indices"))?
                    as u64;
                // Adjacent logical dictionary values share one terminal offset.
                let shared_offsets = match dictionary.values().data_type() {
                    DataType::Utf8 | DataType::Binary | DataType::List(_) | DataType::Map(_, _) => {
                        4
                    }
                    DataType::LargeUtf8 | DataType::LargeBinary | DataType::LargeList(_) => 8,
                    _ => 0,
                } * unique.len().saturating_sub(1) as u64;
                let bytes =
                    unique.into_iter().try_fold(indices, |bytes, index| {
                        Ok::<_, CoreError>(bytes.saturating_add(array_bytes(
                            dictionary.values().slice(index, 1).as_ref(),
                        )?))
                    })?;
                return Ok(bytes.saturating_sub(shared_offsets));
            }};
        }
        match key.as_ref() {
            DataType::Int8 => dictionary!(Int8Type),
            DataType::Int16 => dictionary!(Int16Type),
            DataType::Int32 => dictionary!(Int32Type),
            DataType::Int64 => dictionary!(Int64Type),
            DataType::UInt8 => dictionary!(UInt8Type),
            DataType::UInt16 => dictionary!(UInt16Type),
            DataType::UInt32 => dictionary!(UInt32Type),
            DataType::UInt64 => dictionary!(UInt64Type),
            _ => return Err(internal("invalid dictionary key storage")),
        }
    }
    let (offsets, child) = match array.data_type() {
        DataType::List(_) => {
            let list = array.as_list::<i32>();
            let offsets = list.value_offsets();
            let start = offsets[0] as usize;
            let len = offsets[array.len()] as usize - start;
            (
                4 * (array.len() as u64 + 1),
                Some(list.values().slice(start, len)),
            )
        }
        DataType::LargeList(_) => {
            let list = array.as_list::<i64>();
            let offsets = list.value_offsets();
            let start = offsets[0] as usize;
            let len = offsets[array.len()] as usize - start;
            (
                8 * (array.len() as u64 + 1),
                Some(list.values().slice(start, len)),
            )
        }
        DataType::Map(_, _) => {
            let map = array.as_map();
            let offsets = map.value_offsets();
            let start = offsets[0] as usize;
            let len = offsets[array.len()] as usize - start;
            (
                4 * (array.len() as u64 + 1),
                Some(std::sync::Arc::new(map.entries().slice(start, len)) as arrow_array::ArrayRef),
            )
        }
        DataType::Struct(_) => {
            return array
                .as_struct()
                .columns()
                .iter()
                .try_fold(validity, |size, a| {
                    Ok(size.saturating_add(array_bytes(a.as_ref())?))
                });
        }
        _ => (0, None),
    };
    if let Some(child) = child {
        return Ok(validity
            .saturating_add(offsets)
            .saturating_add(array_bytes(child.as_ref())?));
    }
    array
        .to_data()
        .get_slice_memory_size()
        .map(|size| size as u64)
        .map_err(|_| internal("cannot measure admitted Arrow slice"))
}

pub(crate) fn batch_bytes(batch: &RecordBatch) -> Result<u64, CoreError> {
    batch.columns().iter().try_fold(0u64, |size, array| {
        Ok(size.saturating_add(array_bytes(array.as_ref())?))
    })
}
