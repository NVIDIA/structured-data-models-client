// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Materialization's payload, the response and the result layout
//! (Phase 1b). It holds no SQL and matches on no provider.
//!
//! As built at T1.11c it is thin: it lays out the payload of a regression's
//! request from what the sampler gathered ([`payload`]), as the reference's
//! engine lays it out, in JSON whose objects keep their order and whose
//! numbers are exact ([`json`]); a column type or value the layout does not
//! port yet is refused by name. T1.28 completes it with the wire types,
//! the response and the result layout.

#![forbid(unsafe_code)]

pub mod json;
pub mod payload;
