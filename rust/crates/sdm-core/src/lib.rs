// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The single Run state machine, operation dispatch, the composition root,
//! model capability checks and profiles, and the PQL adapter.

#![forbid(unsafe_code)]

pub mod compose;
pub mod dispatch;
pub mod pql;
pub mod profiles;
pub mod run;

pub use compose::CompositionRoot;
pub use dispatch::start;
pub use run::{ArrowData, Ask, Asker, Attachment, FetchResult, ModelResult, Reply, Run, RunState};
