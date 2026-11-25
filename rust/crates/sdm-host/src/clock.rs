// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The one clock the host layer reads: a run's monotonic time since it
//! started, which [`crate::runs::HostRun`] tells the core with each reply
//! (`Run::tell_elapsed`). The core reads no clock; its time limits
//! (`context_time_limit`) are measured by this one, read here once for
//! every host that runs the core through this layer.

/// When a run started, by the monotonic clock.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Started(std::time::Instant);

impl Started {
    /// Now.
    pub(crate) fn now() -> Started {
        Started(std::time::Instant::now())
    }

    /// The nanoseconds since the run started, at most `u64::MAX`.
    pub(crate) fn elapsed_ns(self) -> u64 {
        u64::try_from(self.0.elapsed().as_nanos()).unwrap_or(u64::MAX)
    }
}
