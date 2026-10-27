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
mod discover;
pub mod dispatch;
mod edit;
pub mod exact_number;
mod fallback;
mod frame_materialize;
mod frames;
mod held;
mod lag;
pub mod materialize;
mod pages;
pub mod pql;
mod predict;
mod prepare;
mod prepare_response;
pub mod profiles;
mod ranking;
pub mod run;
mod subgraph;
#[cfg(feature = "test-ops")]
mod test_ops;
pub mod view;
pub mod view_steps;
pub mod view_words;

pub use compose::{CompositionRoot, DataAccess};
pub use dispatch::start;
pub use edit::{dtype_names, meanings, wire_names};
/// The text a float is spelled in, as the reference's language writes it
/// (Python's `repr`): for a host that hands the data layer a value to
/// write as the reference does (T3.12 fix 1).
pub use pql::float_text;
pub use run::{
    ArrowData, Ask, Asker, Attachment, AttachmentIds, Attachments, FetchResult, ModelResult, Reply,
    Run, RunState,
};
/// The zones port a host implements and passes to [`start`]; re-exported
/// so a host reaches it through the core, never through an edge to
/// `sdm-engine` (decision E30).
pub use sdm_engine::time::{UnknownZone, Zones};
/// What each model supports, as the client describes it to a caller
/// (`capabilities()`), re-exported so a host reaches the request crate's
/// table through the core (decision E30).
pub use sdm_request::predict::{Capabilities, capabilities};
