// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Building the prediction request and reading its answer (Phase 8.5):
//! the envelopes `sdm_client` and the relational engine's inference build,
//! the questions a ranking or forecast is asked as and the merges of their
//! answers, the checks and refusals on the way, and the flattened layout a
//! result's frame is built from.

pub mod cells;
pub mod client;
pub mod decompose;
pub mod error;
pub mod fixture;
pub mod frames;
pub mod gather;
pub mod inference;
pub mod keys;
pub mod layout;
pub mod literal;
pub mod materialize;
pub mod models;
pub mod num;
pub mod parts;
pub mod payload;
pub mod pyliteral;
pub mod pyops;
pub mod pysort;
pub mod query;
pub mod relational;
pub mod relational_adapter;
pub mod request;
pub mod response;
pub mod result;
pub mod sampled;
pub mod schemas;
pub mod series;
pub mod sessions;
pub mod size;
pub mod source_gather;
pub mod tabular;
pub mod tabular_adapter;
pub mod values;
pub mod wire;
