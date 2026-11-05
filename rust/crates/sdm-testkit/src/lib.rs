// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Development-only host, providers and fixtures for native correctness tests.
//! Never a dependency of a shipped crate.
//!
//! [`case`] loads typed cases; [`host`] runs them through `sdm_host::HostRun`.
//! [`observation`] records execution for completeness and correctness checks.
//! [`endpoint`] supplies scripted model responses, and [`driver`] records data reads.
//! [`duck`], [`sqlite`], [`memory`] and [`replay`] provide database or in-memory data.
//! [`kit`] checks provider conformance, and [`layer`] exercises a host-supplied data layer.

/// Database providers supported by the native test host.
/// Postgres uses recorded fetch answers; `sdm-contract-run --providers` lists them.
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
pub mod plan_b;
pub mod replay;
pub mod sqlite;
pub mod values;
