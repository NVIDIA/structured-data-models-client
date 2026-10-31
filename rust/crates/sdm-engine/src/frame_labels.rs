// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! List-valued labels for in-memory link prediction, separate from warehouse labels.

use crate::labels::{Bounds, Column};
use sdm_contract::{CoreError, Sentence};
use std::collections::BTreeSet;

#[derive(Debug, Clone, PartialEq)]
pub enum LinkValue {
    Int(i64),
    UInt(u64),
    Float(f64),
    Bool(bool),
    Text(String),
    Decimal(i128, i8),
}

impl LinkValue {
    fn key(&self) -> Vec<u8> {
        match self {
            Self::Int(value) => value.to_le_bytes().to_vec(),
            Self::UInt(value) => value.to_le_bytes().to_vec(),
            Self::Float(value) => (value + 0.0).to_bits().to_le_bytes().to_vec(),
            Self::Bool(value) => vec![u8::from(*value)],
            Self::Text(value) => value.as_bytes().to_vec(),
            Self::Decimal(value, _) => value.to_le_bytes().to_vec(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LinkLabels {
    /// One bit per input example; only nonempty groups produce labels.
    pub mask: Vec<bool>,
    pub labels: Vec<Vec<LinkValue>>,
}

fn invalid() -> CoreError {
    CoreError::protocol(
        Sentence::InvalidMessage,
        "link label values, groups and windows do not align",
    )
}

/// Group values already selected by the query, retaining first-seen order.
pub fn distinct_selected(
    examples: usize,
    batch: &[usize],
    column: &Column,
) -> Result<LinkLabels, CoreError> {
    distinct_rows(examples, batch, column, |_, _| true)
}

/// Select exact temporal bounds, then group nonmissing values by example.
pub fn distinct(
    examples: usize,
    batch: &[usize],
    column: &Column,
    times: &[Option<i64>],
    bounds: Bounds<'_>,
    kept: Option<&[bool]>,
) -> Result<LinkLabels, CoreError> {
    if times.len() != batch.len()
        || bounds.ends.len() != examples
        || bounds.starts.is_some_and(|values| values.len() != examples)
        || kept.is_some_and(|values| values.len() != batch.len())
    {
        return Err(invalid());
    }
    distinct_rows(examples, batch, column, |row, at| {
        kept.is_none_or(|values| values[row])
            && times[row].is_some_and(|time| {
                time <= bounds.ends[at] && bounds.starts.is_none_or(|values| time > values[at])
            })
    })
}

fn distinct_rows(
    examples: usize,
    batch: &[usize],
    column: &Column,
    included: impl Fn(usize, usize) -> bool,
) -> Result<LinkLabels, CoreError> {
    let values: Vec<Option<LinkValue>> = match column {
        Column::Int(values) => values.iter().map(|v| v.map(LinkValue::Int)).collect(),
        Column::UInt(values) => values.iter().map(|v| v.map(LinkValue::UInt)).collect(),
        Column::Float(values) => values
            .iter()
            .map(|v| {
                v.filter(|v| !v.is_nan())
                    .map(|v| LinkValue::Float(f64::from(v)))
            })
            .collect(),
        Column::Bool(values) => values.iter().map(|v| v.map(LinkValue::Bool)).collect(),
        Column::Text(values) => values
            .iter()
            .map(|v| v.clone().map(LinkValue::Text))
            .collect(),
        Column::Decimal { values, scale, .. } => values
            .iter()
            .map(|v| v.map(|v| LinkValue::Decimal(v, *scale)))
            .collect(),
        Column::Time(_) => return Err(crate::not_yet("frame_link_time_target")),
    };
    group_values(examples, batch, values, included)
}

fn group_values(
    examples: usize,
    batch: &[usize],
    values: Vec<Option<LinkValue>>,
    included: impl Fn(usize, usize) -> bool,
) -> Result<LinkLabels, CoreError> {
    if values.len() != batch.len() || batch.iter().any(|&at| at >= examples) {
        return Err(invalid());
    }
    let mut seen = vec![BTreeSet::new(); examples];
    let mut groups = vec![Vec::new(); examples];
    for (row, (value, &at)) in values.into_iter().zip(batch).enumerate() {
        if let Some(value) = value {
            if included(row, at) && seen[at].insert(value.key()) {
                groups[at].push(value);
            }
        }
    }
    Ok(LinkLabels {
        mask: groups.iter().map(|group| !group.is_empty()).collect(),
        labels: groups
            .into_iter()
            .filter(|group| !group.is_empty())
            .collect(),
    })
}

/// Link targets identify entities rather than numerical model labels.
/// Preserve the source identifier width, especially Float64 precision.
pub fn distinct_identifiers(
    examples: usize,
    batch: &[usize],
    array: &arrow_array::ArrayRef,
    times: &[Option<i64>],
    bounds: Bounds<'_>,
    kept: Option<&[bool]>,
) -> Result<LinkLabels, CoreError> {
    use arrow_array::cast::AsArray;
    use arrow_array::types::*;
    use arrow_schema::DataType;
    if times.len() != batch.len()
        || bounds.ends.len() != examples
        || bounds.starts.is_some_and(|values| values.len() != examples)
        || kept.is_some_and(|values| values.len() != batch.len())
    {
        return Err(invalid());
    }
    macro_rules! values {
        ($ty:ty, $map:expr) => {
            array
                .as_primitive::<$ty>()
                .iter()
                .map(|v| v.map($map))
                .collect()
        };
    }
    let values = match array.data_type() {
        DataType::Int8 => values!(Int8Type, |v| LinkValue::Int(i64::from(v))),
        DataType::Int16 => values!(Int16Type, |v| LinkValue::Int(i64::from(v))),
        DataType::Int32 => values!(Int32Type, |v| LinkValue::Int(i64::from(v))),
        DataType::Int64 => values!(Int64Type, LinkValue::Int),
        DataType::UInt8 => values!(UInt8Type, |v| LinkValue::UInt(u64::from(v))),
        DataType::UInt16 => values!(UInt16Type, |v| LinkValue::UInt(u64::from(v))),
        DataType::UInt32 => values!(UInt32Type, |v| LinkValue::UInt(u64::from(v))),
        DataType::UInt64 => values!(UInt64Type, LinkValue::UInt),
        DataType::Float32 => array
            .as_primitive::<Float32Type>()
            .iter()
            .map(|v| {
                v.filter(|v| !v.is_nan())
                    .map(|v| LinkValue::Float(f64::from(v)))
            })
            .collect(),
        DataType::Float64 => array
            .as_primitive::<Float64Type>()
            .iter()
            .map(|v| v.filter(|v| !v.is_nan()).map(LinkValue::Float))
            .collect(),
        DataType::Boolean => array
            .as_boolean()
            .iter()
            .map(|v| v.map(LinkValue::Bool))
            .collect(),
        DataType::Utf8 => array
            .as_string::<i32>()
            .iter()
            .map(|v| v.map(|v| LinkValue::Text(v.into())))
            .collect(),
        DataType::LargeUtf8 => array
            .as_string::<i64>()
            .iter()
            .map(|v| v.map(|v| LinkValue::Text(v.into())))
            .collect(),
        DataType::Decimal128(_, scale) => {
            values!(Decimal128Type, |v| LinkValue::Decimal(v, *scale))
        }
        _ => return Err(crate::not_yet("frame_link_identifier_type")),
    };
    group_values(examples, batch, values, |row, at| {
        kept.is_none_or(|values| values[row])
            && times[row].is_some_and(|time| {
                time <= bounds.ends[at] && bounds.starts.is_none_or(|values| time > values[at])
            })
    })
}

#[cfg(test)]
mod identity_tests {
    use super::*;
    use arrow_array::{ArrayRef, Float64Array};
    use std::sync::Arc;

    #[test]
    fn float_links_retain_exact_identity_and_window_rules() {
        let array: ArrayRef = Arc::new(Float64Array::from(vec![
            Some(16777216.0),
            Some(16777217.0),
            Some(16777217.0),
            None,
            Some(f64::NAN),
            Some(99.0),
        ]));
        let labels = distinct_identifiers(
            1,
            &[0; 6],
            &array,
            &[Some(1), Some(2), Some(3), Some(3), Some(3), Some(9)],
            Bounds {
                starts: Some(&[0]),
                ends: &[3],
            },
            None,
        )
        .unwrap();
        assert_eq!(
            labels.labels,
            vec![vec![
                LinkValue::Float(16777216.0),
                LinkValue::Float(16777217.0)
            ]]
        );
    }
}
