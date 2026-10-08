// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What a run gives when it is done
//! (`tests/rust_core/sampling/driver/CONTRACT.md`, "What a run gives").

use arrow_array::ArrayRef;

use crate::frame::{Frame, Series};

/// A `UserWarning` the call gave: its class, as `module.QualifiedName`,
/// and its text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Warning {
    pub category: String,
    pub message: String,
    pub masked_float32_cast: bool,
}

impl Warning {
    pub fn user(message: impl Into<String>) -> Self {
        Self {
            category: "builtins.UserWarning".to_string(),
            message: message.into(),
            masked_float32_cast: false,
        }
    }

    pub fn diagnostic(self) -> sdm_protocol::Diagnostic {
        let mut diagnostic = sdm_protocol::Diagnostic::warning(&self.category, self.message);
        if self.masked_float32_cast {
            diagnostic
                .details
                .insert("masked_float32_cast".into(), serde_json::Value::Bool(true));
        }
        diagnostic
    }
}

/// One of `sample_target`'s `TargetOutput`s.
#[derive(Clone, Debug)]
pub struct TargetPart {
    pub entity_pkey: Series,
    /// Nanoseconds, `None` for `NaT`.
    pub anchor_time: Vec<Option<i64>>,
    pub target: Series,
}

/// `sample_target`'s `(train, test)`.
#[derive(Clone, Debug)]
pub struct TargetDone {
    pub train: TargetPart,
    pub test: TargetPart,
    pub warnings: Vec<Warning>,
}

/// A `SampledTable`.
#[derive(Clone, Debug)]
pub struct SampledTable {
    pub name: String,
    pub df: Frame,
    pub row: Option<Vec<i64>>,
    pub batch: Vec<i64>,
    pub num_sampled_nodes: Vec<i64>,
    pub stype_dict: Vec<(String, String)>,
    pub primary_key: Option<String>,
}

/// A `SampledLink`.
#[derive(Clone, Debug)]
pub struct SampledLink {
    pub edge_type: (String, String, String),
    /// `REV`, `COO` or `CSC`.
    pub layout: &'static str,
    pub row: Option<Vec<i64>>,
    pub col: Option<Vec<i64>>,
    pub num_sampled_edges: Vec<i64>,
}

/// `sample_subgraph`'s `SampledSubgraph`.
#[derive(Clone, Debug)]
pub struct SubgraphDone {
    pub anchor_time: Vec<i64>,
    pub tables: Vec<SampledTable>,
    pub links: Vec<SampledLink>,
    pub warnings: Vec<Warning>,
}

/// A label column and its pandas dtype's tag.
pub fn labelled(array: ArrayRef, dtype: crate::frame::Dtype) -> Series {
    Series::new(dtype, array)
}
