// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The data layer (Phase 1b): it plans each typed data request as SQL in a
//! dialect, or passes it to a host provider. SQL dialects live here and in
//! no other domain crate. Empty until the T1.11 tasks fill it.

#![forbid(unsafe_code)]
