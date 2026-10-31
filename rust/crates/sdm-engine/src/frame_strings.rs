// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Typed string views for in-memory frames, independent of warehouse key policy.

use arrow_array::{Array, ArrayRef, LargeStringArray, StringArray};
use sdm_contract::{CoreError, Sentence};

use crate::keys::Key;

/// The physical string representation admitted by the frame boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StringEncoding {
    Utf8,
}

pub struct StringColumn<'a> {
    values: Values<'a>,
}

enum Values<'a> {
    Utf8(&'a StringArray),
    LargeUtf8(&'a LargeStringArray),
}

fn invalid() -> CoreError {
    CoreError::protocol(
        Sentence::InvalidMessage,
        "frame string values do not match their typed encoding",
    )
}

impl<'a> StringColumn<'a> {
    pub fn new(array: &'a dyn Array, _encoding: StringEncoding) -> Result<Self, CoreError> {
        let values = array
            .as_any()
            .downcast_ref::<StringArray>()
            .map(Values::Utf8)
            .or_else(|| {
                array
                    .as_any()
                    .downcast_ref::<LargeStringArray>()
                    .map(Values::LargeUtf8)
            })
            .ok_or_else(invalid)?;
        Ok(Self { values })
    }

    pub fn len(&self) -> usize {
        match self.values {
            Values::Utf8(values) => values.len(),
            Values::LargeUtf8(values) => values.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn get(&self, row: usize) -> Result<Option<&'a str>, CoreError> {
        if row >= self.len() {
            return Err(invalid());
        }
        Ok(match self.values {
            Values::Utf8(values) => (!values.is_null(row)).then(|| values.value(row)),
            Values::LargeUtf8(values) => (!values.is_null(row)).then(|| values.value(row)),
        })
    }
}

pub fn read_keys(
    array: &ArrayRef,
    encoding: StringEncoding,
) -> Result<Vec<Option<Key>>, CoreError> {
    let values = StringColumn::new(array.as_ref(), encoding)?;
    (0..values.len())
        .map(|row| {
            values
                .get(row)
                .map(|value| value.map(|s| Key::Text(s.to_owned())))
        })
        .collect()
}
