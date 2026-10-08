// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The two random generators today's Python draws from, reproduced draw for
//! draw: numpy's legacy `RandomState` ([`RandomState`], [`random_state`]),
//! which every client-side random choice uses, and C++'s `std::mt19937`
//! ([`Mt19937`]), which the in-memory neighbor sampler draws its neighbors
//! with. Both are MT19937; they differ in how they are seeded and drawn
//! from.

mod legacy;
mod mt19937;

pub(crate) use legacy::room;
pub use legacy::{RandomError, RandomState, random_state};
pub use mt19937::Mt19937;
