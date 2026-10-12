// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Materialization's payload, the response and the result layout
//! (Phase 1b). It holds no SQL and matches on no provider.
//!
//! As built at T1.11c it is thin: it lays out the payload of a regression's
//! request from what the sampler gathered ([`payload`]), as the reference's
//! engine lays it out, as `sdm-wire`'s typed payload, whose objects keep
//! their order and whose numbers are exact (T1.28w); a column type or value
//! the layout does not port yet is refused by name. From T1.11d it also
//! sends a regression's payload to the model and lays out the answer
//! ([`predict`]), thin in the same way. T1.28a and T1.28b complete it with
//! the payload's rules, the response and the result layout.

#![forbid(unsafe_code)]

pub mod payload;
pub mod predict;
