// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The composition root: which data provider answers a run's requests.
//!
//! A run takes either a built-in SQL provider, chosen by dialect from the
//! compilers registered here, or a host provider with the capabilities it
//! declares. This is the one place a dialect is looked up; nothing else in
//! the core matches on a backend name.

use std::collections::BTreeMap;
use std::sync::Arc;

use sdm_contract::data::{
    DataRequest, Dialect, OptionalRequest, ProviderCapabilities, ProviderChoice, SqlStatement,
};
use sdm_contract::{CoreError, Detail, ErrorKind, codes};

/// Compiles data requests to one dialect's SQL.
pub trait SqlCompiler: Send + Sync {
    fn compile(&self, request: &DataRequest) -> Result<SqlStatement, CoreError>;
}

/// The built-in SQL compilers, by dialect.
#[derive(Clone, Default)]
pub struct CompositionRoot {
    compilers: BTreeMap<Dialect, Arc<dyn SqlCompiler>>,
}

/// How one run reaches its data.
#[derive(Clone)]
pub enum DataAccess {
    Sql(Arc<dyn SqlCompiler>),
    Host(ProviderCapabilities),
}

/// The optional request a data request is, if it is one; `describe` and
/// `scan` are what every provider answers.
pub fn optional_request(request: &DataRequest) -> Option<OptionalRequest> {
    match request {
        DataRequest::Describe { .. } | DataRequest::Scan { .. } => None,
        DataRequest::Preview { .. } => Some(OptionalRequest::Preview),
        DataRequest::CountRows { .. } => Some(OptionalRequest::CountRows),
        DataRequest::KeyCounts { .. } => Some(OptionalRequest::KeyCounts),
        DataRequest::TimeBounds { .. } => Some(OptionalRequest::TimeBounds),
        DataRequest::SampleEntities { .. } => Some(OptionalRequest::SampleEntities),
    }
}

impl CompositionRoot {
    /// Registers the built-in compiler for `dialect`.
    #[must_use]
    pub fn with_dialect(mut self, dialect: Dialect, compiler: Arc<dyn SqlCompiler>) -> Self {
        self.compilers.insert(dialect, compiler);
        self
    }

    /// The data access a run's provider choice names.
    pub fn data_access(&self, choice: &ProviderChoice) -> Result<DataAccess, CoreError> {
        match choice {
            ProviderChoice::Sql { dialect } => match self.compilers.get(dialect) {
                Some(compiler) => Ok(DataAccess::Sql(Arc::clone(compiler))),
                None => Err(CoreError::new(
                    ErrorKind::Unsupported,
                    codes::UNSUPPORTED_REQUEST,
                    "this build has no built-in SQL provider for the dialect asked for",
                )
                .with_field(
                    "dialect",
                    Detail::text(serde_json::to_value(dialect).map_or_else(
                        |_| String::new(),
                        |v| v.as_str().unwrap_or_default().to_owned(),
                    )),
                )),
            },
            ProviderChoice::Host { capabilities } => Ok(DataAccess::Host(capabilities.clone())),
        }
    }
}

impl DataAccess {
    /// The statement for `request`: compiled for a SQL provider, none for a
    /// host provider, which answers the typed request itself. A host
    /// provider is asked only for what it declared.
    pub fn prepare(&self, request: &DataRequest) -> Result<Option<SqlStatement>, CoreError> {
        match self {
            DataAccess::Sql(compiler) => compiler.compile(request).map(Some),
            DataAccess::Host(capabilities) => match optional_request(request) {
                Some(optional) if !capabilities.optional.contains(&optional) => {
                    Err(CoreError::new(
                        ErrorKind::Unsupported,
                        codes::UNSUPPORTED_REQUEST,
                        "the host provider did not declare this request",
                    )
                    .with_field(
                        "request",
                        Detail::text(serde_json::to_value(optional).map_or_else(
                            |_| String::new(),
                            |v| v.as_str().unwrap_or_default().to_owned(),
                        )),
                    ))
                }
                _ => Ok(None),
            },
        }
    }
}
