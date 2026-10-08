// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `Schema.from_source` as a [`Run`](sdm_protocol::Run) of the core.
//!
//! The reading of a source is plain `async` code that awaits each
//! statement's result from [`Db`]; [`AsyncRun`] resumes it on each step
//! where it stopped, so every statement is asked, and every sample
//! inferred, once.

use sdm_protocol::AsyncRun;

use crate::client::{self, Said};
use crate::dt::Host;
use crate::engine::{Ctx, Db, Warnings};
use crate::source::{SourceArgs, SourceInfo, from_source};

/// `Schema.from_source`, run: its value is `(schema_json, warnings_json)`.
pub type SourceRun = AsyncRun<(String, String)>;

/// The run of `Schema.from_source` over the source `info` describes.
pub fn source_run(info: SourceInfo, args: SourceArgs, host: Host) -> SourceRun {
    AsyncRun::new(|asker| async move {
        let db = Db(asker);
        let mut ctx = Ctx {
            db: &db,
            host: &host,
            warnings: Warnings::default(),
        };
        let mut said = Said::default();
        let result = from_source(&mut ctx, &info, &args, &mut said).await;
        Ok(client::outcome_json(&result, &said))
    })
}
