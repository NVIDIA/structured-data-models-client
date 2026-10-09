// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The data layer (Phase 1b): it plans each typed data request as SQL in a
//! dialect, or passes it to a host provider (decision H14). SQL dialects
//! live here and in no other domain crate; the wire names a source's
//! backend only for this crate to interpret.
//!
//! [`Data`] is the data layer `sdm_host::default_root` builds (decision
//! E30). A host provider is answered with the typed request itself, and is
//! asked only for what it declared; a built-in SQL provider is answered
//! with statements this crate renders for its dialect, and its
//! capabilities are the requests this crate renders for that dialect. As
//! built at T1.11a no request has a rendering yet: DuckDB's catalog
//! requests come with T1.11b, and the rest with the tasks that reach them.

#![forbid(unsafe_code)]

mod duckdb;

use sdm_contract::data::{
    DataLayer, DataRequest, Dialect, ExpectedSchema, FetchPlan, ProviderCapabilities,
    ProviderChoice,
};
use sdm_contract::{CoreError, Detail, ErrorCode, ErrorKind};

/// The data layer of a shipped build: DuckDB's SQL, and host providers.
#[derive(Debug, Clone, Copy, Default)]
pub struct Data;

impl Data {
    pub fn new() -> Self {
        Data
    }
}

impl DataLayer for Data {
    fn capabilities(&self, provider: &ProviderChoice) -> Result<ProviderCapabilities, CoreError> {
        match provider {
            ProviderChoice::Host { capabilities } => Ok(capabilities.clone()),
            ProviderChoice::Sql {
                dialect: Dialect::Duckdb,
            } => Ok(duckdb::capabilities()),
            ProviderChoice::Sql { dialect } => Err(no_dialect(*dialect)),
        }
    }

    fn plan(
        &self,
        request: &DataRequest,
        provider: &ProviderChoice,
    ) -> Result<FetchPlan, CoreError> {
        match provider {
            ProviderChoice::Host { .. } => Ok(FetchPlan::Host {
                expect: expected(request),
            }),
            ProviderChoice::Sql {
                dialect: Dialect::Duckdb,
            } => duckdb::plan(request),
            ProviderChoice::Sql { dialect } => Err(no_dialect(*dialect)),
        }
    }
}

/// The schema a request's answer must have. As built at T1.11a every
/// answer establishes its own (a discovery request); the requests whose
/// shape the contract fixes get their expected schemas with their
/// renderings, so the data layer and the host provider are held to one.
fn expected(_request: &DataRequest) -> ExpectedSchema {
    ExpectedSchema::Discover {}
}

/// The name a dialect has on the wire.
fn dialect_name(dialect: Dialect) -> &'static str {
    match dialect {
        Dialect::Duckdb => "duckdb",
        Dialect::Sqlite => "sqlite",
        Dialect::Postgres => "postgres",
        Dialect::Databricks => "databricks",
        Dialect::Snowflake => "snowflake",
    }
}

/// The refusal of a SQL dialect this build renders nothing for.
fn no_dialect(dialect: Dialect) -> CoreError {
    CoreError::new(
        ErrorKind::Unsupported,
        ErrorCode::UnsupportedRequest,
        "this build has no built-in SQL provider for the dialect asked for",
    )
    .with_field("dialect", Detail::text(dialect_name(dialect)))
}

/// The refusal of a request the dialect has no rendering for yet.
fn not_rendered(request: &DataRequest, dialect: Dialect) -> CoreError {
    CoreError::new(
        ErrorKind::Unsupported,
        ErrorCode::UnsupportedRequest,
        "this build renders no SQL for this request in the dialect asked for",
    )
    .with_field("request", Detail::text(request.name()))
    .with_field("dialect", Detail::text(dialect_name(dialect)))
}
