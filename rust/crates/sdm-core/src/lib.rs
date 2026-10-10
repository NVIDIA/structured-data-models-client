// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The single Run state machine, operation dispatch, the composition root,
//! model capability checks and profiles, and the PQL adapter.

#![forbid(unsafe_code)]

pub mod attached;
pub mod compose;
mod describe;
pub mod dispatch;
pub mod materialize;
pub mod pql;
mod predict;
pub mod profiles;
pub mod run;
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
