// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The composition root: the one data layer every run reads a source's
//! database through.
//!
//! The root holds one `Arc<dyn DataLayer>`, which `sdm_host::default_root`
//! builds; swapping the data layer replaces that one value. A run takes the
//! provider its operation names, with the capabilities it declared, asks
//! for data only through [`DataAccess::plan`] (asked in parts as the
//! provider's key limit takes it, [`DataAccess::bind`]), and reads a
//! catalog answer only through [`DataAccess::read`], into the contract's
//! dialect-neutral shape. Nothing in the core matches on a
//! backend's name: what a provider takes is what its capabilities say, and
//! the core refuses by reading them (the requests it declared;
//! the largest seed it samples with, register entry RS-7).

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow_array::RecordBatch;
use sdm_contract::catalog::CatalogAnswer;
use sdm_contract::data::{
    AttachmentId, BoundRequest, DataLayer, DataRequest, DescribePart, FailureFacts, FallbackLimits,
    FetchPlan, KeyData, ProviderCapabilities, ProviderChoice, ProviderError, Relation, ViewChoice,
    refuse_derived, seed_integer,
};
use sdm_contract::view::{ViewTextAnswer, ViewTextAsk};
use sdm_contract::{CoreError, Detail, Scalar, Sentence};

/// What a binding starts its runs with.
#[derive(Clone)]
pub struct CompositionRoot {
    data: Arc<dyn DataLayer>,
}

/// A data layer reading no database: every answer in the contract's
/// shape, and no semantic model read.
struct NoDatabases;

impl DataLayer for NoDatabases {}

impl Default for CompositionRoot {
    /// The root over a data layer reading no database: for a host whose
    /// providers are all its own.
    fn default() -> Self {
        CompositionRoot::new(Arc::new(NoDatabases))
    }
}

/// How one run reaches its data: the root's data layer and what the
/// operation's provider declared it answers.
#[derive(Clone)]
pub struct DataAccess {
    layer: Arc<dyn DataLayer>,
    capabilities: ProviderCapabilities,
}

impl CompositionRoot {
    /// The root over `data`, the one data layer every run reads through.
    pub fn new(data: Arc<dyn DataLayer>) -> Self {
        CompositionRoot { data }
    }

    /// The data layer every run reads through.
    pub fn data_layer(&self) -> &Arc<dyn DataLayer> {
        &self.data
    }

    /// The data access of a run whose operation names `provider`.
    pub fn data_access(&self, provider: &ProviderChoice) -> Result<DataAccess, CoreError> {
        Ok(DataAccess {
            layer: Arc::clone(&self.data),
            capabilities: provider.capabilities.clone(),
        })
    }
}

impl DataAccess {
    /// Whether the run's provider must be asked its namespace before any
    /// table ([`ProviderCapabilities::namespace_first`]), until the source
    /// says it ([`DataAccess::settled`]).
    pub fn unsettled(&self) -> bool {
        self.capabilities.namespace_first
    }

    /// The run's data access with its provider's namespace the parts the
    /// source answered for it ([`ProviderChoice::settled`]).
    pub fn settled(&self, parts: Vec<String>) -> DataAccess {
        let mut access = self.clone();
        access.capabilities = ProviderChoice::new(self.capabilities.clone())
            .settled(parts)
            .capabilities;
        access
    }

    /// Whether the run's provider reads a source of tables, whose tables
    /// each `materialize` reads as they stand
    /// ([`ProviderCapabilities::reads_tables_as_they_stand`]), where a host
    /// provider's tables are the host's own.
    pub fn reads_a_source(&self) -> bool {
        self.capabilities.reads_tables_as_they_stand
    }

    /// The semantic model the run's provider reads the source through,
    /// if any ([`ProviderCapabilities::view`]).
    pub fn view(&self) -> Option<ViewChoice> {
        self.capabilities.view.clone()
    }

    /// The kind of model the provider reads, before its definition is fetched.
    pub fn view_kind(&self) -> Option<sdm_contract::view::ViewKind> {
        self.layer.view_kind(&self.capabilities)
    }

    /// The data layer's answer to a question of a semantic model's text, in
    /// the run's provider's database's dialect (`DataLayer::view_text`).
    pub fn view_text(&self, ask: ViewTextAsk) -> Result<ViewTextAnswer, CoreError> {
        self.layer.view_text(&self.capabilities, ask)
    }

    /// Whether the run's provider's source holds a type per column that a
    /// caller's type must match
    /// ([`ProviderCapabilities::column_affinities`]).
    pub fn holds_column_types(&self) -> bool {
        !self.capabilities.column_affinities
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
    /// ([`ProviderCapabilities::database`]); `None` for a caller's source.
    pub fn source_database(&self) -> Option<String> {
        self.capabilities.database.clone()
    }

    /// `relation` as the run's provider reads it
    /// ([`ProviderCapabilities::read_as`]).
    pub fn read_as(&self, relation: &Relation) -> Relation {
        self.capabilities.read_as(relation)
    }

    /// What the run's provider's failure says, as the provider said it
    /// ([`ProviderError::facts`]): the facts the step that asked reads the
    /// one of that fits it; where it says nothing, only what the SQL
    /// standard's codes say ([`FailureFacts::standard`]).
    pub fn failure(&self, error: &ProviderError) -> FailureFacts {
        error
            .facts
            .clone()
            .unwrap_or_else(|| FailureFacts::standard(error))
    }

    /// The name forms the catalog read of `relation`, as the run's
    /// provider reads it, tries in order
    /// ([`ProviderCapabilities::spellings`]).
    pub fn spellings(&self, relation: &Relation) -> Vec<Relation> {
        self.capabilities.spellings(relation)
    }

    /// `plan`, the run's provider's answer to `request`, asked in parts of
    /// at most the keys the provider takes in one request
    /// ([`sdm_data::parts`]): one fetch or more, each with the batches it
    /// binds by local reference.
    pub fn bind(
        &self,
        request: &DataRequest,
        plan: FetchPlan,
        keys: &BTreeMap<AttachmentId, KeyData>,
    ) -> Result<BoundRequest, CoreError> {
        let max = self.capabilities.max_keys_per_request.unwrap_or(u64::MAX);
        sdm_data::parts::bind(request, plan, keys, max)
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

    /// How `request` is answered: the provider's plan, as
    /// [`DataAccess::plan`] gives it, its answer of the shape the
    /// provider's database answers in (`DataLayer::expected`), or, for an
    /// optional request the provider did not declare while it declares the
    /// in-memory fallback's limits, the fallback within them (design
    /// choice 1), which the core answers from the rows `scan`s read.
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
        let expect = self.layer.expected(&self.capabilities, request);
        Ok(Route::Plan(FetchPlan::of(expect, request)))
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
    ///.
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
    /// `describe` it planned, read into the contract's dialect-neutral
    /// shape: as the contract's own shape
    /// ([`sdm_contract::catalog_read::standard`]), but for a semantic
    /// model's definition, which the data layer reads by the provider's
    /// database (`DataLayer::read_view`). `batches` are the answer as the
    /// run admitted it (check (b)). A request that is not a `describe` is
    /// refused with `INTERNAL_ERROR`.
    pub fn read(
        &self,
        request: &DataRequest,
        batches: &[RecordBatch],
    ) -> Result<CatalogAnswer, CoreError> {
        let DataRequest::Describe { part, .. } = request else {
            return Err(CoreError::new(
                Sentence::Internal,
                "only a catalog answer is read by the data layer",
            )
            .with_field("request", Detail::text(request.name())));
        };
        let answer = match part {
            DescribePart::ViewDefinition => {
                self.layer.read_view(&self.capabilities, request, batches)?
            }
            _ => sdm_contract::catalog_read::standard(*part, batches)?,
        };
        debug_assert_eq!(answer.part(), *part);
        Ok(answer)
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
