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
    Float(f32),
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
            .map(|v| v.filter(|v| !v.is_nan()).map(LinkValue::Float))
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
