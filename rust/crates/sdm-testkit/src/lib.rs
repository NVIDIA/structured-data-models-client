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
//! `sdm-contract-run --pre-given-specs` alone. [`sqlite`] is its SQLite
//! (T2.2): a case's tables loaded by the contract loader, and the dev-only
//! `rusqlite` provider. [`kit`] runs the provider conformance kit on its
//! providers (T2.5). [`memory`] is the minimal non-SQL test provider, and
//! [`layer`] the second data layer, for host providers alone, swapped in
//! at the composition root (T2.7). [`standin`] is the stand-in provider
//! of the warehouse legs (T3.3): exact replay of the old client's
//! statements from the early capture, then a DuckDB loaded with its
//! tables, then a reason the step is judged live only; the `sdm-standin`
//! binary serves it over a pipe to the Python and Java harnesses.

/// The databases the pure-Rust host has a provider for (rust-host's marker
/// for the staging rule, `contract/runners/run_python.py`'s `check_legs`;
/// T2.1a): its DuckDB ([`duck`]), SQLite through the dev-only `rusqlite`
/// provider ([`sqlite`], T2.2), and Postgres, by replaying recorded fetch
/// results ([`replay`], T2.3, KO2-20). `sdm-contract-run --providers`
/// prints it.
pub const PROVIDERS: &[&str] = &["duckdb", "sqlite", "postgres"];

pub mod bases;
pub mod case;
pub mod driver;
pub mod duck;
pub mod edits;
pub mod endpoint;
pub mod host;
pub mod kit;
pub mod layer;
pub mod memory;
pub mod observation;
pub mod pre_given;
pub mod replay;
pub mod sqlite;
pub mod standin;
pub mod values;
