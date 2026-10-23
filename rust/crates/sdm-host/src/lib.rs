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
mod extensions;
pub mod runs;
pub mod zones;

pub use admission::{OFFSET_CAPACITY, admit, admit_from, admit_with_capacity, admitted_type};
pub use codec::{CONTROL_CAP, Outgoing, decode, encode, read, write, write_error};
pub use runs::{Failure, HostRun, Unreadable, Unreadables, contain, is_posix_zone};
/// The types a binding hands across, so a binding depends on `sdm-host`
/// and on `sdm-contract` for types, never on `sdm-core` (decision E21);
/// `CompositionRoot` too, so a host composes its own data layer
/// (`CompositionRoot::new`) where it does not take [`default_root`]'s, as
/// the dev-only test host swaps one in (T2.7).
pub use sdm_core::{ArrowData, Attachment, Attachments, CompositionRoot, dtype_names, meanings};
pub use zones::{ZoneError, ZoneResolver, ZoneRoot, ZoneSource};

/// The table of what every sentence says, which a binding exports as the
/// JSON it is (`sdm_contract::wording`, T3.17): a host reads it once and
/// fills each error's and warning's sentence from it, formatting nothing
/// the table does not name.
pub fn templates() -> String {
    sdm_contract::wording::table_json()
}

/// The databases the data layer of [`default_root`] provides for
/// (`sdm_data::PROVIDERS`), which a binding reports as plain data.
pub fn providers() -> &'static [&'static str] {
    sdm_data::PROVIDERS
}

/// `query` as the PQL library writes it back (`sdm_core::pql::normalized`),
/// or `None` when it does not parse: the text a host sends materialize.
pub fn normalized_query(query: &str) -> Option<String> {
    sdm_core::pql::normalized(query)
}

/// The composition root a binding starts its runs with: the one data layer,
/// `sdm-data`'s, built here where every binding reaches it (decisions E21,
/// H14, E30).
pub fn default_root() -> CompositionRoot {
    CompositionRoot::new(std::sync::Arc::new(sdm_data::Data::new()))
}
