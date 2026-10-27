// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Time, random streams, sampling and labels over fetched Arrow (Phase 1b).
//! It holds no SQL and matches on no provider.
//!
//! As built at T1.11c it is thin, holding what `materialize` needs for
//! `base.regression`, by the reference's rules: the zones port
//! ([`time::Zones`]); numpy's `default_rng` and its shuffle ([`random`],
//! RS-1); the graph a request samples ([`graph`]); key values compared
//! exactly ([`keys`]); and the sampler's examples, labels, seeds and
//! neighbor order ([`sample`]). What those rules decide by a path not
//! ported yet is refused by name ([`not_yet`], register entry P1B-3);
//! T1.22, T1.25, T1.26a, T1.26b and T1.27 complete them. From T1.25
//! [`time`] holds the core's instant and span, civil arithmetic, zones to
//! the range ends and the readings of text times (E62).
//! From T1.26b [`sample`] builds the sampler's typed reads (`_by_pkey`,
//! `_by_fkey`, `_by_time`) and [`words`] is a text feature as the
//! reference's sampler sends it.

#![forbid(unsafe_code)]

pub mod frames;
pub mod graph;
pub mod index_sort;
pub mod keys;
pub mod labels;
pub mod random;
pub mod sample;
pub mod time;
pub mod words;

use sdm_contract::{CoreError, Detail, Sentence};

/// The refusal of a `materialize` input that takes a rule this build does
/// not port yet, naming the rule, never a value or a name the input holds
/// (decision H17).
pub fn not_yet(rule: &'static str) -> CoreError {
    CoreError::new(
        Sentence::NotYetMaterialize,
        "materializing this request takes a rule this core does not hold yet",
    )
    .with_field("operation", Detail::text("materialize"))
    .with_field("rule", Detail::text(rule))
}
