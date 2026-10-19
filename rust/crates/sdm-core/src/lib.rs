// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The single Run state machine, operation dispatch, the composition root,
//! model capability checks and profiles, and the PQL adapter.

#![forbid(unsafe_code)]

pub mod attached;
pub mod big_integer;
pub mod compose;
mod describe;
pub mod dispatch;
mod edit;
pub mod exact_number;
mod fallback;
pub mod materialize;
pub mod pql;
mod predict;
pub mod profiles;
pub mod run;
mod subgraph;
#[cfg(feature = "test-ops")]
mod test_ops;

pub use compose::{CompositionRoot, DataAccess};
pub use dispatch::start;
pub use run::{
    ArrowData, Ask, Asker, Attachment, AttachmentIds, Attachments, FetchResult, ModelResult, Reply,
    Run, RunState,
};
/// The zones port a host implements and passes to [`start`]; re-exported
/// so a host reaches it through the core, never through an edge to
/// `sdm-engine` (decision E30).
pub use sdm_engine::time::{UnknownZone, Zones};

/// Every warning the core gives, with the variants of each (none: one
/// sentence), for a host to word each (decision E59(d)): the schema's,
/// the key rules', then the sampler's, then the PQL adapter's, then predict's (T1.28b).
pub fn warning_codes() -> Vec<(&'static str, &'static [&'static str])> {
    sdm_schema::assemble::WARNINGS
        .iter()
        .chain(sdm_schema::keys::WARNINGS)
        .chain(materialize::WARNINGS)
        .chain(pql::WARNINGS)
        .chain(predict::WARNINGS)
        .copied()
        .collect()
}
