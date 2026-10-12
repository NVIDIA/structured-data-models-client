// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Materialization's payload, the response and the result layout
//! (Phase 1b). It holds no SQL and matches on no provider.
//!
//! It lays out the payload of a request from what the sampler gathered
//! ([`payload`]), as the reference's engine lays it out, as `sdm-wire`'s
//! typed payload, whose objects keep their order and whose numbers are
//! exact (T1.28w), with the payload's rules the reference's client holds
//! (T1.28a): the classification task's classes, the context that answers
//! the question taken out, and the call's options read as the engine reads
//! them ([`options`]: the run mode, the statistic, the inference options);
//! a column type or value the layout does not port yet is refused by name.
//! It also writes the call the client makes of a payload and lays out the
//! answer ([`predict`]), thin until T1.28b: a regression's answer alone.

#![forbid(unsafe_code)]

pub mod options;
pub mod payload;
pub mod predict;
