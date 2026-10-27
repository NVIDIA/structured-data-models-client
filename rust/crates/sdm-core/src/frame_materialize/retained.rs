// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A useful core-created graph owner, independent of any staged workflow.
//! It owns the existing immutable graph/indexes and continuing neighbor state.
//! No Python object, registry, callback or program position lives here.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use sdm_contract::frames::materialize::FrameNativeTable;
use sdm_contract::{CoreError, Sentence};

use super::{Context, PreparedGraph};

struct Owner {
    prepared: Arc<PreparedGraph>,
    frames: Context,
    native: Arc<Vec<FrameNativeTable>>,
    source: String,
    entered: AtomicBool,
}

/// Host-owned native data, created only by core graph preparation.
/// A binding's exact frozen owner class can hold this value directly.
/// Cloning shares immutable indexes and the existing FrameRun, without
/// exporting source rows through foreign Arrow or rebuilding a graph.
#[derive(Clone)]
pub struct RetainedFrameGraph {
    owner: Arc<Owner>,
}

impl std::fmt::Debug for RetainedFrameGraph {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RetainedFrameGraph")
            .field("source", &self.owner.source)
            .finish_non_exhaustive()
    }
}

impl RetainedFrameGraph {
    /// Only the core can construct this after graph/native-source validation.
    pub(crate) fn new(
        prepared: PreparedGraph,
        frames: Context,
        native: Vec<FrameNativeTable>,
        source: String,
    ) -> Self {
        super::lock(&frames).retain_owner();
        Self {
            owner: Arc::new(Owner {
                prepared: Arc::new(prepared),
                frames,
                native: Arc::new(native),
                source,
                entered: AtomicBool::new(false),
            }),
        }
    }

    pub fn source(&self) -> &str {
        &self.owner.source
    }

    pub(crate) fn prepared(&self) -> Arc<PreparedGraph> {
        Arc::clone(&self.owner.prepared)
    }

    pub(crate) fn frames(&self) -> Context {
        Arc::clone(&self.owner.frames)
    }

    pub(crate) fn native(&self) -> Arc<Vec<FrameNativeTable>> {
        Arc::clone(&self.owner.native)
    }

    /// Like HostRun's existing enter boundary, overlapping calls fail busy
    /// instead of blocking while another operation uses the same state.
    /// The lease covers only the current core call and carries no stage.
    pub(crate) fn try_lease(&self) -> Result<GraphLease, CoreError> {
        if self.owner.frames.is_poisoned() {
            return Err(CoreError::protocol(
                Sentence::RunPoisoned,
                "a panic while using this retained frame graph poisoned its state",
            ));
        }
        self.owner
            .entered
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .map_err(|_| {
                CoreError::protocol(
                    Sentence::RunBusy,
                    "another call is using this retained frame graph",
                )
            })?;
        Ok(GraphLease {
            owner: Arc::clone(&self.owner),
        })
    }
}

/// Core-local resource borrowing; drop, error, cancellation and unwind all
/// release access without rolling back neighbor progression or source data.
pub(crate) struct GraphLease {
    owner: Arc<Owner>,
}

impl Drop for GraphLease {
    fn drop(&mut self) {
        self.owner.entered.store(false, Ordering::Release);
    }
}
