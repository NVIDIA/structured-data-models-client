// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The engine-neutral data layer: preparing aligned time series as ordinary
//! relational tables. Every dialect's SQL is `sdm-sql`'s.

#![forbid(unsafe_code)]

pub mod timeseries;
