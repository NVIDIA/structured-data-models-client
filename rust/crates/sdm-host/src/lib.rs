// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The host layer every binding shares, plain Rust: the control codecs and
//! the 64 KB control cap ([`codec`]), Arrow admission ([`admission`]), run
//! ownership with panic containment and busy handling ([`runs`]), and zone
//! resolution through the host's lookup roots ([`zones`]). It schedules no
//! domain work; that is the core's.

#![forbid(unsafe_code)]

pub mod admission;
pub mod codec;
pub mod runs;
pub mod zones;

pub use admission::{OFFSET_CAPACITY, admit, admit_with_capacity, admitted_type};
pub use codec::{CONTROL_CAP, Outgoing, decode, encode, read, write, write_error};
pub use runs::{Failure, HostRun, Unreadable, Unreadables, contain};
/// The types a binding hands across, so a binding depends on `sdm-host`
/// and on `sdm-contract` for types, never on `sdm-core` (decision E21).
pub use sdm_core::{ArrowData, Attachment, Attachments};
pub use zones::{ZoneError, ZoneResolver, ZoneRoot, ZoneSource};

/// The composition root a binding starts its runs with. Today's root is
/// empty, with no SQL compiler registered; T1.11a builds the data layer in
/// it here, where every binding reaches it (decisions E21, H14).
pub fn default_root() -> sdm_core::CompositionRoot {
    sdm_core::CompositionRoot::default()
}
