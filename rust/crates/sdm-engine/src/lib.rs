// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Time, random streams, sampling and labels over fetched Arrow (Phase 1b).
//! It holds no SQL and matches on no provider. Today it holds the zones
//! port ([`time::Zones`]); the T1.11 and later tasks fill the rest.

#![forbid(unsafe_code)]

pub mod time;
