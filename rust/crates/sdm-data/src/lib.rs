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
//! built at T1.11b DuckDB renders the catalog (`describe` in each part,
//! `preview`, `key_counts`, `time_bounds`), and from T1.11c the sampler's
//! `sample_entities`, `neighbors` and `lookup`, and from T1.32e
//! `column_ranges` and `repeated_key`; the rest come with the
//! tasks that reach them. A plan expects the schema the contract fixes for
//! the request's answer, or discovers it where the answer's columns are the
//! source's own (`DataRequest::expected`, decision E36), so a host provider
//! is held to the same shape as a rendering; but a `describe`, whose answer
//! is a catalog's: a dialect's plan expects its own catalog's shape, and a
//! host provider answers in the contract's.
//!
//! [`DataLayer::read`] reads each `describe` answer, after the core has
//! admitted it, into the contract's dialect-neutral catalog answer
//! (decision E54): a dialect's reader knows its catalog's columns and
//! spelling (DuckDB's `PRAGMA table_info`, its constraint types, its index
//! expressions), so no other crate does. `l0_scan.py`'s `dialects` check
//! keeps them here.

#![forbid(unsafe_code)]

mod duckdb;
mod read;

use arrow_array::RecordBatch;
use sdm_contract::catalog::CatalogAnswer;
use sdm_contract::data::{
    DataLayer, DataRequest, Dialect, FetchPlan, ProviderCapabilities, ProviderChoice,
};
use sdm_contract::{CoreError, Detail, ErrorCode, ErrorKind};

/// The SQL dialects this build knows by name. It renders DuckDB's; the
/// others are refused by name, as a dialect it has no rendering for, and a
/// name it does not know is refused without being repeated.
const KNOWN: &[&str] = &[
    duckdb::NAME,
    "sqlite",
    "postgres",
    "databricks",
    "snowflake",
];

/// A dialect this build renders.
enum Rendered {
    Duckdb,
}

/// The dialect `dialect` names, if this build renders it.
fn rendered(dialect: &Dialect) -> Result<Rendered, CoreError> {
    match dialect.name() {
        duckdb::NAME => Ok(Rendered::Duckdb),
        _ => Err(no_dialect(dialect)),
    }
}

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
            ProviderChoice::Sql { dialect } => match rendered(dialect)? {
                Rendered::Duckdb => Ok(duckdb::capabilities()),
            },
        }
    }

    fn plan(
        &self,
        request: &DataRequest,
        provider: &ProviderChoice,
    ) -> Result<FetchPlan, CoreError> {
        match provider {
            ProviderChoice::Host { .. } => Ok(FetchPlan::Host {
                expect: request.expected(),
            }),
            ProviderChoice::Sql { dialect } => match rendered(dialect)? {
                Rendered::Duckdb => duckdb::plan(request),
            },
        }
    }

    fn opaque_vendor(&self, provider: &ProviderChoice) -> Option<&'static str> {
        match provider {
            ProviderChoice::Host { .. } => None,
            ProviderChoice::Sql { dialect } => match rendered(dialect).ok()? {
                Rendered::Duckdb => Some(duckdb::OPAQUE_VENDOR),
            },
        }
    }

    fn read(
        &self,
        request: &DataRequest,
        provider: &ProviderChoice,
        batches: &[RecordBatch],
    ) -> Result<CatalogAnswer, CoreError> {
        let DataRequest::Describe { part, .. } = request else {
            return Err(CoreError::new(
                ErrorKind::Internal,
                ErrorCode::InternalError,
                "only a catalog answer is read by the data layer",
            )
            .with_field("request", Detail::text(request.name())));
        };
        let answer = match provider {
            ProviderChoice::Host { .. } => read::host(*part, batches)?,
            ProviderChoice::Sql { dialect } => match rendered(dialect)? {
                Rendered::Duckdb => duckdb::read(*part, batches)?,
            },
        };
        debug_assert_eq!(answer.part(), *part);
        Ok(answer)
    }
}

/// The refusal of a SQL dialect this build renders nothing for, naming it
/// when it is one this build knows by name.
fn no_dialect(dialect: &Dialect) -> CoreError {
    let error = CoreError::new(
        ErrorKind::Unsupported,
        ErrorCode::UnsupportedRequest,
        "this build has no built-in SQL provider for the dialect asked for",
    );
    match KNOWN.iter().find(|known| **known == dialect.name()) {
        Some(known) => error.with_field("dialect", Detail::text(*known)),
        None => error,
    }
}
