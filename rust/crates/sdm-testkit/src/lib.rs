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
//! `scripts/rust_core/compare.py`. [`duck`] is its DuckDB: the connection,
//! with ICU loaded, and the provider that answers a fetch; [`bases`] reads
//! `contract/bases.yaml`. [`endpoint`] is the scripted model endpoint's
//! rule (T1.11d), which the host answers each model call with. [`driver`]
//! records what the provider ran, each result in [`values`]' canonical
//! form, for the observation's `driver` section (T1.11e). [`edits`] makes
//! the user's edits to the description the host's own `describe_source`
//! answered, which is the schema its `materialize` samples (T1.21d).
//! [`pre_given`] reads a case written before contract 23, for
//! `sdm-contract-run --pre-given-specs` alone.

/// The databases the pure-Rust host has a provider for (rust-host's marker
/// for the staging rule, `contract/runners/run_python.py`'s `check_legs`;
/// T2.1a): its DuckDB ([`duck`]). T2.2's dev-only SQLite provider and
/// T2.3's Postgres replay add theirs. `sdm-contract-run --providers` prints
/// it.
pub const PROVIDERS: &[&str] = &["duckdb"];

pub mod bases;
pub mod case;
pub mod driver;
pub mod duck;
pub mod edits;
pub mod endpoint;
pub mod host;
pub mod observation;
pub mod pre_given;
pub mod values;
