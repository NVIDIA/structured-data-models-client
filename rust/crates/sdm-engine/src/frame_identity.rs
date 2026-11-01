// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Standard producer of complete native row identities for typed Rust hosts.
//! A native host's actual aligned conversion is another authoritative producer.

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::frame_strings::StringEncoding;
use arrow_array::{Array, ArrayRef, BooleanArray, RecordBatch, StringArray};
use arrow_schema::DataType;
use sdm_contract::{CoreError, Detail, Scalar, Sentence};

use crate::frame_strings::StringColumn;
use crate::keys::Key;

#[derive(Debug, Clone)]
pub struct IdentityColumn {
    pub values: ArrayRef,
    pub encoding: StringEncoding,
}

pub fn refusal(reason: &str, table: &str, column: &str) -> CoreError {
    CoreError::new(
        Sentence::FrameNativeCaller,
        "native row identity admission failed",
    )
    .with_field("native_key_reason", Detail::text(reason))
    .with_field("table", Detail::text(table))
    .with_field("column", Detail::text(column))
    .with_field("detail", Detail::text(""))
}

fn invalid(message: &str) -> CoreError {
    CoreError::protocol(Sentence::InvalidMessage, message)
}

fn escape(text: &[u8], out: &mut Vec<u8>) {
    for &byte in text {
        match byte {
            b'\\' => out.extend_from_slice(b"\\\\"),
            0x1f => out.extend_from_slice(b"\\u"),
            other => out.push(other),
        }
    }
}

/// The fixed native codec for admitted primitive data, without native hooks.
/// Components are observed in declaration order; a null refuses before join.
pub fn encode(
    rows: &RecordBatch,
    key: &[String],
    strings: &BTreeMap<String, StringEncoding>,
) -> Result<IdentityColumn, CoreError> {
    if key.len() < 2 {
        return Err(invalid(
            "a compound identity needs at least two source components",
        ));
    }
    let columns = key
        .iter()
        .map(|name| {
            rows.column_by_name(name)
                .ok_or_else(|| invalid("native identity lacks its source component"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    for (name, array) in key.iter().zip(&columns) {
        if array.data_type().is_floating() {
            return Err(refusal("approximate_component", "", name)
                .with_field("dtype", Detail::text("float")));
        }
    }
    if crate::graph::folded_name(key).len() > crate::graph::FOLDED_NAME_MOST {
        return Err(refusal("name_long", "", "")
            .with_field(
                "limit",
                Detail::Scalar(Scalar::UInt64(crate::graph::FOLDED_NAME_MOST as u64)),
            )
            .with_field(
                "columns",
                Detail::List(key.iter().cloned().map(Scalar::Utf8).collect()),
            ));
    }
    let mut joined = vec![Vec::new(); rows.num_rows()];
    for (ordinal, (name, array)) in key.iter().zip(columns).enumerate() {
        if array.null_count() != 0 {
            return Err(refusal("component_null", "", name));
        }
        let text = strings.get(name).copied().or_else(|| {
            matches!(array.data_type(), DataType::Utf8 | DataType::LargeUtf8)
                .then_some(StringEncoding::Utf8)
        });
        if let Some(encoding) = text {
            let column = StringColumn::new(array.as_ref(), encoding).map_err(|_| {
                invalid("native identity component encoding differs from its values")
            })?;
            for (row, out) in joined.iter_mut().enumerate() {
                if ordinal != 0 {
                    out.push(0x1f);
                }
                let value = column
                    .get(row)
                    .map_err(|_| invalid("native identity component has invalid encoded text"))?
                    .ok_or_else(|| refusal("component_null", "", name))?;
                escape(value.as_bytes(), out);
            }
        } else if let Some(column) = array.as_any().downcast_ref::<BooleanArray>() {
            for (row, out) in joined.iter_mut().enumerate() {
                if ordinal != 0 {
                    out.push(0x1f);
                }
                out.extend_from_slice(if column.value(row) { b"true" } else { b"false" });
            }
        } else {
            let values = crate::keys::read(array)?;
            for (value, out) in values.into_iter().zip(&mut joined) {
                if ordinal != 0 {
                    out.push(0x1f);
                }
                let Some(Key::Int(value)) = value else {
                    return Err(invalid(
                        "standard native identity component is outside its primitive domain",
                    ));
                };
                out.extend_from_slice(value.to_string().as_bytes());
            }
        }
    }
    let values = StringArray::from_iter_values(
        joined
            .iter()
            .map(|value| std::str::from_utf8(value).expect("identity components are UTF-8")),
    );
    Ok(IdentityColumn {
        values: Arc::new(values),
        encoding: StringEncoding::Utf8,
    })
}
