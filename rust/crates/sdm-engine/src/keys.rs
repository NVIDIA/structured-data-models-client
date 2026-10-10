// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Key values, compared exactly: an integer of any width as its value, text
//! as its text. Nothing passes through a float, so two keys are the same
//! only when their values are (decision H22). The types a sampled key may
//! have at T1.11c are the integers and text; any other is refused by name.

use arrow_array::cast::AsArray;
use arrow_array::types::{
    Int8Type, Int16Type, Int32Type, Int64Type, UInt8Type, UInt16Type, UInt32Type, UInt64Type,
};
use arrow_array::{Array, ArrayRef};
use arrow_schema::DataType;
use sdm_contract::CoreError;

use crate::not_yet;

/// One key value.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Key {
    Int(i128),
    Text(String),
}

/// Whether a key of `data_type` is sampled at T1.11c.
pub fn supported(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Utf8
    )
}

/// The keys `array` holds, `None` for a null.
pub fn read(array: &ArrayRef) -> Result<Vec<Option<Key>>, CoreError> {
    macro_rules! ints {
        ($t:ty) => {{
            let values = array.as_primitive::<$t>();
            (0..values.len())
                .map(|i| (!values.is_null(i)).then(|| Key::Int(i128::from(values.value(i)))))
                .collect()
        }};
    }
    Ok(match array.data_type() {
        DataType::Int8 => ints!(Int8Type),
        DataType::Int16 => ints!(Int16Type),
        DataType::Int32 => ints!(Int32Type),
        DataType::Int64 => ints!(Int64Type),
        DataType::UInt8 => ints!(UInt8Type),
        DataType::UInt16 => ints!(UInt16Type),
        DataType::UInt32 => ints!(UInt32Type),
        DataType::UInt64 => ints!(UInt64Type),
        DataType::Utf8 => {
            let values = array.as_string::<i32>();
            (0..values.len())
                .map(|i| (!values.is_null(i)).then(|| Key::Text(values.value(i).to_owned())))
                .collect()
        }
        _ => return Err(not_yet("key_type")),
    })
}
