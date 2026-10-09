// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Time (design, "Time"). Today it holds the zones a run resolves names
//! with; `Instant`, `Span` and civil arithmetic arrive with T1.25.

use jiff::tz::TimeZone;

/// A zone name the run's zones do not hold. It carries no name: a zone's
/// name can come from the data or the caller, and no input value reaches a
/// `CoreError` (decision H17).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnknownZone;

/// The zones a run resolves names with: the host's zone lookup roots in
/// order, then a bundled database, with the source of each zone recorded
/// by the host (decision A7). The host passes them to `sdm_core::start`
/// (decision H14), and the core resolves through nothing else.
pub trait Zones: Send + Sync {
    /// The zone `name`, in any case, or [`UnknownZone`].
    fn zone(&self, name: &str) -> Result<TimeZone, UnknownZone>;
}
