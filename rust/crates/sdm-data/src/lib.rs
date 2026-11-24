// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The engine-neutral data layer: a provider's request asked in parts as
//! its declared key limit takes it, and aligned time series prepared as
//! ordinary relational tables. Every dialect's SQL is `sdm-sql`'s.

#![forbid(unsafe_code)]

pub mod parts;
pub mod timeseries;
