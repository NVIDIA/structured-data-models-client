// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Sampling and label computation over Arrow tables.
//! This crate handles graph traversal, exact keys, row selection, index ordering,
//! random streams, text features and time arithmetic.
//! It contains no SQL or connector-specific dispatch.
//!
//! [`sample`] selects relational examples and neighbors; [`tabular`] selects
//! single-table rows.
//! [`frames`] prepares in-memory graphs, and [`time`] handles instants, civil
//! arithmetic, zones and supported text formats.

#![forbid(unsafe_code)]

pub mod frame_identity;
pub mod frame_labels;
pub mod frame_strings;
pub mod frames;
pub mod graph;
pub mod keys;
pub mod labels;
pub mod random;
pub mod sample;
pub mod time;
pub mod words;

use sdm_contract::{CoreError, Detail, Sentence};

/// The refusal of a `materialize` input that takes a rule this build does
/// not port yet, naming the rule, never a value or a name the input holds
///.
pub fn not_yet(rule: &'static str) -> CoreError {
    CoreError::new(
        Sentence::NotYetMaterialize,
        "materializing this request takes a rule this core does not hold yet",
    )
    .with_field("operation", Detail::text("materialize"))
    .with_field("rule", Detail::text(rule))
}

/// In-memory single-table row selection.
pub mod tabular;

/// Sampled-index ordering for verified native producer profiles.
pub mod index_sort;
