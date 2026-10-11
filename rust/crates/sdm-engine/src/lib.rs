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

#![forbid(unsafe_code)]

pub mod graph;
pub mod keys;
pub mod random;
pub mod sample;
pub mod time;

use sdm_contract::{CoreError, Detail, ErrorCode, ErrorKind};

/// The refusal of a `materialize` input that takes a rule this build does
/// not port yet, naming the rule, never a value or a name the input holds
/// (decision H17).
pub fn not_yet(rule: &'static str) -> CoreError {
    CoreError::new(
        ErrorKind::Unsupported,
        ErrorCode::UnsupportedRequest,
        "materializing this request takes a rule this core does not hold yet",
    )
    .with_field("operation", Detail::text("materialize"))
    .with_field("rule", Detail::text(rule))
}
