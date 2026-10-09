// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dev-only: the case loader, the pure-Rust test host and the scripted
//! endpoint. Never a dependency of a shipped crate.
//!
//! The case loader ([`case`]) reads `contract/cases/` in the format of
//! `contract/case.schema.json`; the pure-Rust host ([`host`]) runs a case's
//! core operations through `sdm_host::HostRun`, with no Python in the
//! process; [`observation`] writes what it saw in the format of
//! `contract/observation.schema.json`, which `sdm-contract-run` puts out for
//! `scripts/rust_core/compare.py`. The scripted endpoint arrives with
//! T1.11d.

pub mod case;
pub mod host;
pub mod observation;
