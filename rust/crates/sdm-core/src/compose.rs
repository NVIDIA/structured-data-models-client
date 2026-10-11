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
//! backend's name: what a provider takes is what its capabilities say, and
//! the core refuses by reading them (the requests it declared; from T1.22,
//! the largest seed it samples with, register entry RS-7).

use std::sync::Arc;

use arrow_array::RecordBatch;
use sdm_contract::catalog::CatalogAnswer;
use sdm_contract::data::{DataLayer, DataRequest, FetchPlan, ProviderCapabilities, ProviderChoice};
use sdm_contract::{CoreError, Detail, ErrorCode, ErrorKind, Scalar};

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
    /// contradict each other is refused with `INVALID_REQUEST`, an optional
    /// request the provider did not declare with `UNSUPPORTED_REQUEST`
    /// naming the request, and a sample seeded past the largest seed the
    /// provider declares with `INVALID_REQUEST` naming the seed (RS-7); a
    /// provider's declared capability is never replaced by a fallback
    /// (design, "Data access").
    pub fn plan(&self, request: &DataRequest) -> Result<FetchPlan, CoreError> {
        request.check()?;
        if let Some(optional) = request.optional()
            && !self.capabilities.optional.contains(&optional)
        {
            return Err(undeclared(optional.name()));
        }
        if let DataRequest::SampleEntities { seed, .. } = request
            && let Some(max) = self.capabilities.max_seed
            && integer(seed).is_some_and(|seed| seed > i128::from(max))
        {
            return Err(past_seed_range(request, max));
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
pub(crate) fn undeclared(request: &str) -> CoreError {
    CoreError::new(
        ErrorKind::Unsupported,
        ErrorCode::UnsupportedRequest,
        "the provider did not declare this request",
    )
    .with_field("request", Detail::text(request))
}

/// A seed's value, when the scalar it travels as is an integer: the
/// narrowest exact scalar `materialize` writes, a `u64` or a decimal of
/// scale 0, or any other integer scalar.
fn integer(seed: &Scalar) -> Option<i128> {
    Some(match seed {
        Scalar::Int32(v) => i128::from(*v),
        Scalar::Int64(v) => i128::from(*v),
        Scalar::UInt32(v) => i128::from(*v),
        Scalar::UInt64(v) => i128::from(*v),
        Scalar::Decimal128 {
            scale: 0, value, ..
        } => *value,
        _ => return None,
    })
}

/// The refusal of a sample whose seed is past the largest the provider
/// declares (register entry RS-7), naming the request, the seed and the
/// provider's largest, never the provider.
fn past_seed_range(request: &DataRequest, max: u64) -> CoreError {
    CoreError::new(
        ErrorKind::InvalidRequest,
        ErrorCode::InvalidRequest,
        "the provider samples with no seed larger than the largest it declares",
    )
    .with_field("request", Detail::text(request.name()))
    .with_field("field", Detail::text("seed"))
    .with_field("max_seed", Detail::Scalar(Scalar::UInt64(max)))
}
