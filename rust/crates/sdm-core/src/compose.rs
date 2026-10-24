// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The composition root: the one data layer every run plans its data
//! requests through (decision H14).
//!
//! The root holds one `Arc<dyn DataLayer>`, which `sdm_host::default_root`
//! builds; swapping the data layer replaces that one value. A run takes the
//! provider its operation names, with the capabilities the layer gives for
//! it, asks for data only through [`DataAccess::plan`] (bound as the
//! provider binds it, [`DataAccess::bind`]), and reads a
//! catalog answer only through [`DataAccess::read`], into the contract's
//! dialect-neutral shape (decision E54). Nothing in the core matches on a
//! backend's name: what a provider takes is what its capabilities say, and
//! the core refuses by reading them (the requests it declared; from T1.22,
//! the largest seed it samples with, register entry RS-7).

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow_array::RecordBatch;
use sdm_contract::catalog::CatalogAnswer;
use sdm_contract::data::{
    AttachmentId, BoundRequest, DataLayer, DataRequest, FallbackLimits, FetchPlan, KeyData,
    ProviderCapabilities, ProviderChoice, Relation, refuse_derived, seed_integer,
};
use sdm_contract::{CoreError, Detail, Scalar, Sentence};

use crate::ArrowData;

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
    /// Whether the run's provider leaves a namespace part to the connection
    /// that must be read before any table (`DataLayer::unsettled`), until
    /// the source says its current parts
    /// ([`DataAccess::settled`]).
    pub fn unsettled(&self) -> bool {
        self.layer.unsettled(&self.provider)
    }

    /// The run's data access with its provider's namespace the parts the
    /// source answered for it (`DataLayer::settled`).
    pub fn settled(&self, parts: Vec<String>) -> DataAccess {
        let mut access = self.clone();
        access.provider = self.layer.settled(&self.provider, parts);
        access
    }

    /// Whether the run's provider reads a source of tables, whose tables
    /// each `materialize` reads as they stand (the owner's plan B, T3.4;
    /// `DataLayer::reads_tables_as_they_stand`), where a host provider's
    /// tables are the host's own.
    pub fn reads_a_source(&self) -> bool {
        self.layer.reads_tables_as_they_stand(&self.provider)
    }

    /// Whether the run's provider's source holds a type per column that a
    /// caller's type must match (`DataLayer::holds_column_types`).
    pub fn holds_column_types(&self) -> bool {
        self.layer.holds_column_types(&self.provider)
    }

    /// What the run's provider answers.
    pub fn capabilities(&self) -> &ProviderCapabilities {
        &self.capabilities
    }

    /// The vendor of the opaque Arrow extension types the run's provider
    /// hands over, as its capabilities declare it
    /// (`ProviderCapabilities::opaque_vendor`).
    pub fn opaque_vendor(&self) -> Option<&str> {
        self.capabilities.opaque_vendor.as_deref()
    }

    /// Whether the run's provider compares a window's bounds with the
    /// stored text of the window's column
    /// (`ProviderCapabilities::compares_stored_times`).
    pub fn compares_stored_times(&self) -> bool {
        self.capabilities.compares_stored_times
    }

    /// Whether the run's provider orders a seeded sample by the key's data
    /// type (`ProviderCapabilities::samples_by_key_dtype`).
    pub fn samples_by_key_dtype(&self) -> bool {
        self.capabilities.samples_by_key_dtype
    }

    /// Whether the reference reads the run's provider's floats with a NaN
    /// missing and `-0.0` as `0.0`
    /// (`ProviderCapabilities::reads_nan_as_missing`).
    pub fn reads_nan_as_missing(&self) -> bool {
        self.capabilities.reads_nan_as_missing
    }

    /// The database the run's provider reads, by name
    /// (`DataLayer::source_database`); `None` for a host provider.
    pub fn source_database(&self) -> Option<&'static str> {
        self.layer.source_database(&self.provider)
    }

    /// `relation` as the run's provider reads it (`DataLayer::read_as`).
    pub fn read_as(&self, relation: &Relation) -> Relation {
        self.layer.read_as(&self.provider, relation)
    }

    /// An admitted answer of the run's provider, its values read as the
    /// reference reads them where the provider hands over the driver's own
    /// (`DataLayer::conform`, T2.14d); the answer as it is where the data
    /// layer holds no rule for it.
    pub fn conform(&self, data: ArrowData) -> Result<ArrowData, CoreError> {
        match self.layer.conform(
            &self.provider,
            &data.schema,
            &data.batches,
            crate::pql::float_text,
        )? {
            Some((schema, batches)) => Ok(ArrowData {
                schema,
                batches,
                received: data.received,
            }),
            None => Ok(data),
        }
    }

    /// `plan`, the run's provider's answer to `request`, bound as the
    /// provider binds it with the key sets `keys` it names
    /// (`DataLayer::bind`, T3.28): one fetch or more, each with the
    /// batches it binds by local reference, a double written as the
    /// reference's language writes one ([`crate::pql::float_text`]).
    pub fn bind(
        &self,
        request: &DataRequest,
        plan: FetchPlan,
        keys: &BTreeMap<AttachmentId, KeyData>,
    ) -> Result<BoundRequest, CoreError> {
        self.layer
            .bind(&self.provider, request, plan, keys, crate::pql::float_text)
    }

    /// How the run's provider answers `request`. A request whose fields
    /// contradict each other is refused with `INVALID_REQUEST`, an optional
    /// request the provider did not declare with `UNSUPPORTED_REQUEST`
    /// naming the request, and a sample seeded past the largest seed the
    /// provider declares with `INVALID_REQUEST` naming the seed (RS-7); a
    /// provider's declared capability is never replaced by a fallback
    /// (design, "Data access").
    pub fn plan(&self, request: &DataRequest) -> Result<FetchPlan, CoreError> {
        match self.route(request)? {
            Route::Plan(plan) => Ok(plan),
            Route::Fallback(_) => Err(undeclared(request.name())),
        }
    }

    /// How `request` is answered: the data layer's plan for the run's
    /// provider, as [`DataAccess::plan`] gives it, or, for an optional
    /// request the provider did not declare while it declares the
    /// in-memory fallback's limits, the fallback within them (design
    /// choice 1, T2.6), which the core answers from the rows `scan`s read.
    /// Refused as [`DataAccess::plan`] refuses, but for the fallback's
    /// request; a request the provider declared is never the fallback's.
    pub fn route(&self, request: &DataRequest) -> Result<Route, CoreError> {
        request.check()?;
        // A derived relation only of a provider declaring it reads one,
        // whichever way the request is answered (contract v28).
        for (field, relation) in request.relations() {
            if let Some(refusal) = refuse_derived(
                request,
                field,
                relation,
                self.capabilities.derived_relations,
            ) {
                return Err(refusal);
            }
        }
        if let Some(optional) = request.optional()
            && !self.capabilities.optional.contains(&optional)
        {
            let Some(limits) = self.capabilities.fallback else {
                return Err(undeclared(optional.name()));
            };
            self.seeded_within(request)?;
            return Ok(Route::Fallback(limits));
        }
        self.seeded_within(request)?;
        Ok(Route::Plan(self.layer.plan(request, &self.provider)?))
    }

    /// Refuses a sample seeded past the largest seed the provider declares
    /// (RS-7).
    fn seeded_within(&self, request: &DataRequest) -> Result<(), CoreError> {
        if let DataRequest::SampleEntities { seed, .. } = request
            && seed_integer(seed).is_some_and(|seed| seed > i128::from(self.capabilities.max_seed))
        {
            return Err(past_seed_range(request.name(), self.capabilities.max_seed));
        }
        Ok(())
    }

    /// This access with `undeclared` taken out of what the provider
    /// declares and, where given, the fallback's `limits` declared in place
    /// of its own: for the test-only fetch, so the conformance kit asks any
    /// provider, a SQL one included, a request it then does not declare
    /// (T2.5, T2.6).
    #[cfg(feature = "test-ops")]
    pub(crate) fn declaring(
        mut self,
        undeclared: &[sdm_contract::data::OptionalRequest],
        limits: Option<FallbackLimits>,
    ) -> Self {
        for request in undeclared {
            self.capabilities.optional.remove(request);
        }
        if limits.is_some() {
            self.capabilities.fallback = limits;
        }
        self
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

/// How one request is answered ([`DataAccess::route`]).
#[derive(Debug, Clone)]
pub enum Route {
    /// As the data layer plans it for the run's provider.
    Plan(FetchPlan),
    /// By the in-memory fallback, within the provider's declared limits.
    Fallback(FallbackLimits),
}

/// The refusal of an optional request the provider did not declare.
pub(crate) fn undeclared(request: &str) -> CoreError {
    CoreError::new(
        Sentence::Undeclared,
        "the provider did not declare this request",
    )
    .with_field("request", Detail::text(request))
}

/// The refusal of a sample whose seed is past the largest the provider
/// declares (register entry RS-7), naming the request, the seed and the
/// provider's largest, never the provider. `materialize` gives it too, for
/// a seed past what its scalar carries (`seed_scalar`).
pub(crate) fn past_seed_range(request: &str, max: u64) -> CoreError {
    CoreError::new(
        Sentence::SeedPastRange,
        "the provider samples with no seed larger than the largest it declares",
    )
    .with_field("request", Detail::text(request))
    .with_field("field", Detail::text("seed"))
    .with_field("max_seed", Detail::Scalar(Scalar::UInt64(max)))
}
