// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Schema inference: what each table and column of a request means, ported
//! from `sdm_client/schema.py`, the engine's metadata inference and graph,
//! and the date reading both stand on (Phase 6.5).

pub mod binding;
pub mod client;
pub mod composite;
pub mod dates;
pub mod decimal;
pub mod diagnostic_compat;
pub mod diagnostics;
pub mod dt;
pub mod edits;
pub mod engine;
mod fixture;
pub mod frame;
pub mod metric_view;
pub mod py;
pub mod pyval;
pub mod run;
pub mod semantic_view;
pub mod source;
pub mod time_columns;
pub mod yaml;

pub use fixture::{InputError, fixture_op, fixture_op_in};
