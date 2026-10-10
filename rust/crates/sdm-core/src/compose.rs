// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The composition root: the one data layer every run plans its data
//! requests through (decision H14).
//!
//! The root holds one `Arc<dyn DataLayer>`, which `sdm_host::default_root`
//! builds; swapping the data layer replaces that one value. A run takes the
//! provider its operation names, with the capabilities the layer gives for
//! it, asks for data only through [`DataAccess::plan`], and reads a
//! catalog answer only through [`DataAccess::read`], into the contract's
//! dialect-neutral shape (decision E54). Nothing in the core matches on a
//! backend's name.

use std::sync::Arc;

use arrow_array::RecordBatch;
use sdm_contract::catalog::CatalogAnswer;
use sdm_contract::data::{DataLayer, DataRequest, FetchPlan, ProviderCapabilities, ProviderChoice};
use sdm_contract::{CoreError, Detail, ErrorCode, ErrorKind};

/// What a binding starts its runs with.
#[derive(Clone)]
pub struct CompositionRoot {
    data: Arc<dyn DataLayer>,
}

/// How one run reaches its data: the root's data layer, the provider the
/// operation names, and what that provider answers.
#[derive(Clone)]
pub struct DataAccess {
    layer: Arc<dyn DataLayer>,
    provider: ProviderChoice,
    capabilities: ProviderCapabilities,
}

impl CompositionRoot {
    /// The root over `data`, the one data layer every run plans through.
    pub fn new(data: Arc<dyn DataLayer>) -> Self {
        CompositionRoot { data }
    }

    /// The data layer every run plans through.
    pub fn data_layer(&self) -> &Arc<dyn DataLayer> {
        &self.data
    }

    /// The data access of a run whose operation names `provider`; a
    /// provider the data layer cannot serve is refused here, before the
    /// run exists.
    pub fn data_access(&self, provider: &ProviderChoice) -> Result<DataAccess, CoreError> {
        let capabilities = self.data.capabilities(provider)?;
        Ok(DataAccess {
            layer: Arc::clone(&self.data),
            provider: provider.clone(),
            capabilities,
        })
    }
}

impl DataAccess {
    /// What the run's provider answers.
    pub fn capabilities(&self) -> &ProviderCapabilities {
        &self.capabilities
    }

    /// How the run's provider answers `request`. A request whose fields
    /// contradict each other is refused with `INVALID_REQUEST`, and an
    /// optional request the provider did not declare with
    /// `UNSUPPORTED_REQUEST` naming the request; a provider's declared
    /// capability is never replaced by a fallback (design, "Data access").
    pub fn plan(&self, request: &DataRequest) -> Result<FetchPlan, CoreError> {
        request.check()?;
        if let Some(optional) = request.optional()
            && !self.capabilities.optional.contains(&optional)
        {
            return Err(undeclared(optional.name()));
        }
        self.layer.plan(request, &self.provider)
    }

    /// The catalog answer the run's provider returned to `request`, a
    /// `describe` it planned, read by the data layer into the contract's
    /// dialect-neutral shape (decision E54). `batches` are the answer as
    /// the run admitted it (decision H16, check (b)).
    pub fn read(
        &self,
        request: &DataRequest,
        batches: &[RecordBatch],
    ) -> Result<CatalogAnswer, CoreError> {
        self.layer.read(request, &self.provider, batches)
    }
}

/// The refusal of an optional request the provider did not declare.
fn undeclared(request: &str) -> CoreError {
    CoreError::new(
        ErrorKind::Unsupported,
        ErrorCode::UnsupportedRequest,
        "the provider did not declare this request",
    )
    .with_field("request", Detail::text(request))
}
