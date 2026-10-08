// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The runs every language drives the core through.
//!
//! The run loop's types are `sdm-protocol`'s, re-exported here: the crates
//! this one drives (schema inference, sampling, requests) report through
//! them too, so they live below all of them. The protocol a run keeps is
//! described there.

mod count_rows;
mod fixture;
mod flow;
pub mod host_api;
pub mod plain;
pub mod public_replay;
pub mod replay;
pub mod semantic_replay;
pub mod snapshot;

pub use count_rows::{CountRows, count_rows};
pub use fixture::{InputError, fixture_op};
pub use sdm_protocol::{
    Ask, Asker, AsyncRun, Diagnostic, Endpoint, ModelCall, Outcome, Param, Query, QueryId,
    QueryResult, Run, Step, asking, codes,
};
/// The request crate the core drives, as the bindings reach it.
pub use sdm_request as request;
