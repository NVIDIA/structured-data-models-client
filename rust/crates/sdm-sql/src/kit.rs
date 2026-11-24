// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The SQL kit: one built-in source's answer to a typed request, written
//! once for every host.
//!
//! A built-in source (DuckDB, SQLite, Postgres, Databricks, Snowflake)
//! answers each typed request the core sends it through [`SqlKit::answer`].
//! The kit renders the request in the source's dialect, binds its key sets
//! as that dialect binds them (several fetches where the key set is split),
//! asks the host's driver to run each fetch through `run`, joins the answers
//! in order and reads their values as the reference reads them. The host
//! writes no SQL and decides nothing: it runs the statements it is handed,
//! with the batches it is handed, on one session per `answer`.
//!
//! The kit takes each step in the order the core's run takes it for a SQL
//! provider today (`sdm-core`'s `Asker::fetch`): the plan; where the source
//! needs one, a metadata read with empty key sets and the plan projected by
//! it; the plan bound; each fetch run; the answers joined; the joined answer
//! conformed. So a request answered through the kit runs exactly the
//! statements the core's run asks of the host today.

use std::collections::BTreeMap;

use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use sdm_contract::data::{
    AttachmentId, DataLayer, DataRequest, FailureFacts, FetchPlan, KeyData, ProviderCapabilities,
    ProviderChoice, ProviderError, Relation,
};
use sdm_contract::view::{ViewTextAnswer, ViewTextAsk};
use sdm_contract::{CoreError, Detail, Sentence};

use crate::Data;

/// One fetch the kit asks the host's driver to run: the statements of
/// `plan`, in order, on one session, binding `keys` where a statement names
/// a key set, the last statement's result the answer.
#[derive(Debug, Clone, Copy)]
pub struct KitFetch<'a> {
    /// The request this fetch answers, or the part of it the fetch reads.
    pub request: &'a DataRequest,
    /// The statements to run (always a SQL plan).
    pub plan: &'a FetchPlan,
    /// The key sets and text batches the statements bind, by the id each
    /// statement names them by.
    pub keys: &'a BTreeMap<AttachmentId, KeyData>,
}

/// A result: its schema and every batch the driver handed over.
#[derive(Debug, Clone)]
pub struct Answered {
    pub schema: SchemaRef,
    pub batches: Vec<RecordBatch>,
}

/// Why [`SqlKit::answer`] did not answer.
#[derive(Debug, Clone)]
pub enum KitError {
    /// The kit refused the request by name: it cannot be written in this
    /// dialect, or its answer broke a rule the kit holds.
    Refused(CoreError),
    /// The host's driver failed to run a fetch.
    Provider(ProviderError),
}

impl From<CoreError> for KitError {
    fn from(error: CoreError) -> Self {
        KitError::Refused(error)
    }
}

/// A built-in source's SQL: its dialect, the namespace it reads under, the
/// semantic model it reads through, and the timestamp transport its driver
/// needs. One kit answers one run: the namespace it settles is kept for
/// the run, as the core keeps a run's, and never between runs.
#[derive(Debug)]
pub struct SqlKit {
    data: Data,
    provider: std::sync::Mutex<ProviderChoice>,
}

impl SqlKit {
    /// The kit of the SQL source `provider` names, reading semantic models
    /// with `spaces` where given. Refused by name where `provider` is not a
    /// SQL source this build writes, or names a semantic model its dialect
    /// cannot read.
    pub fn new(
        provider: ProviderChoice,
        spaces: Option<&'static crate::view_expr::Spaces>,
    ) -> Result<Self, CoreError> {
        if !matches!(provider, ProviderChoice::Sql { .. }) {
            return Err(CoreError::new(
                Sentence::Internal,
                "the SQL kit takes a SQL source, not a host provider",
            ));
        }
        let data = match spaces {
            Some(spaces) => Data::reading_views(spaces),
            None => Data::new(),
        };
        data.capabilities(&provider)?;
        Ok(SqlKit {
            data,
            provider: std::sync::Mutex::new(provider),
        })
    }

    /// The source this kit answers for, as the core names it, its
    /// namespace settled once the run asked it.
    pub fn provider(&self) -> ProviderChoice {
        self.provider
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    /// What the source answers: every request and filter its dialect
    /// writes.
    pub fn capabilities(&self) -> Result<ProviderCapabilities, CoreError> {
        let provider = self.provider();
        let mut declared = self.data.capabilities(&provider)?;
        // The engine's traits the core reads by them, alike for every
        // source (contract v76).
        let probe = Relation::of(vec!["t".to_owned()]);
        declared.database = self.data.source_database(&provider);
        declared.column_affinities = !self.data.holds_column_types(&provider);
        declared.reads_tables_as_they_stand = self.data.reads_tables_as_they_stand(&provider);
        declared.namespace = self.data.read_as(&provider, &probe).source;
        declared.namespace.pop();
        declared.tries_upper_case_names = self.data.spellings(&provider, &probe).len() > 1;
        declared.namespace_first = self.data.unsettled(&provider);
        declared.view = provider.view().cloned();
        Ok(declared)
    }

    /// The answer to `request`, its key sets `keys` by the id the request
    /// names each by, every fetch run by `run`; doubles written into SQL
    /// text as `float_text` writes them (the reference's language's
    /// spelling, which the core holds).
    pub fn answer(
        &self,
        request: &DataRequest,
        keys: &BTreeMap<AttachmentId, KeyData>,
        float_text: fn(f64) -> String,
        run: &mut dyn FnMut(KitFetch<'_>) -> Result<Answered, ProviderError>,
    ) -> Result<Answered, KitError> {
        let provider = self.provider();
        let mut plan = self.data.plan(request, &provider)?;
        if let Some(metadata) = self.data.result_metadata(&provider, request, &plan)? {
            // A schema-only read binds empty key sets: the source query is
            // never run to learn its result's types.
            let empty: BTreeMap<AttachmentId, KeyData> = keys
                .iter()
                .map(|(id, made)| {
                    (
                        *id,
                        KeyData {
                            schema: made.schema.clone(),
                            batches: Vec::new(),
                        },
                    )
                })
                .collect();
            let described =
                self.run_bound(&provider, request, metadata, &empty, float_text, run)?;
            plan = self
                .data
                .project_result(&provider, request, &plan, &described.batches)?;
        }
        let answered = self.run_bound(&provider, request, plan, keys, float_text, run)?;
        // A catalog answer is read from the dialect's own rows and written
        // in the standard shape every source answers in.
        if let DataRequest::Describe { part, .. } = request {
            let mut read = self.data.read(request, &provider, &answered.batches)?;
            // The namespace a source leaves to its connection, settled with
            // the parts it names, as the core settles a run's: kept for the
            // rest of the run, and answered in full.
            if *part == sdm_contract::data::DescribePart::Namespace
                && self.data.unsettled(&provider)
                && let sdm_contract::catalog::CatalogAnswer::Namespace(parts) = read
            {
                let settled = self.data.settled(&provider, parts);
                let probe = Relation::of(vec!["t".to_owned()]);
                let mut full = self.data.read_as(&settled, &probe).source;
                full.pop();
                *self
                    .provider
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner) = settled;
                read = sdm_contract::catalog::CatalogAnswer::Namespace(full);
            }
            if let Some((schema, batch)) = crate::standard::catalog(&read)? {
                return Ok(Answered {
                    schema,
                    batches: vec![batch],
                });
            }
        }
        Ok(answered)
    }

    /// `plan` bound as the source binds it, each fetch run in order, their
    /// answers joined and read as the reference reads them.
    fn run_bound(
        &self,
        provider: &ProviderChoice,
        request: &DataRequest,
        plan: FetchPlan,
        keys: &BTreeMap<AttachmentId, KeyData>,
        float_text: fn(f64) -> String,
        run: &mut dyn FnMut(KitFetch<'_>) -> Result<Answered, ProviderError>,
    ) -> Result<Answered, KitError> {
        let named = plan.key_sets();
        let given: BTreeMap<AttachmentId, KeyData> = keys
            .iter()
            .filter(|(id, _)| named.contains(id))
            .map(|(id, made)| (*id, made.clone()))
            .collect();
        let bound = self
            .data
            .bind(provider, request, plan, &given, float_text)?;
        let mut joined: Option<Answered> = None;
        for fetch in bound.fetches {
            // Each fetch is handed what its statements bind and nothing
            // more, as the core hands its host only what a fetch names.
            let binds = fetch.plan.key_sets();
            let mut bound_keys: BTreeMap<AttachmentId, KeyData> = given
                .iter()
                .filter(|(id, _)| binds.contains(id))
                .map(|(id, made)| (*id, made.clone()))
                .collect();
            bound_keys.extend(fetch.batches);
            let answered = run(KitFetch {
                request: &fetch.request,
                plan: &fetch.plan,
                keys: &bound_keys,
            })
            .map_err(|mut error| {
                // What the failure means, read by the engine that failed,
                // so the core reads it as it reads every source's.
                error.facts = Some(self.data.failure(provider, &error));
                KitError::Provider(error)
            })?;
            joined = Some(match joined {
                None => answered,
                Some(mut earlier) => {
                    if earlier.schema != answered.schema {
                        let reason =
                            "a chunk of the answer has another schema than the chunks before it";
                        return Err(KitError::Refused(
                            CoreError::new(
                                Sentence::ProviderSchemaMismatch,
                                format!(
                                    "the provider's data does not match the chunks schema: {reason}"
                                ),
                            )
                            .with_field("check", Detail::text("chunks"))
                            .with_field("reason", Detail::text(reason)),
                        ));
                    }
                    earlier.batches.extend(answered.batches);
                    earlier
                }
            });
        }
        let whole = joined.ok_or_else(|| {
            KitError::Refused(CoreError::new(
                Sentence::Internal,
                "a request bound into no fetch",
            ))
        })?;
        Ok(
            match self
                .data
                .conform(provider, &whole.schema, &whole.batches, float_text)?
            {
                Some((schema, batches)) => Answered { schema, batches },
                None => whole,
            },
        )
    }

    /// What the driver's failure `error` tells the core.
    pub fn failure(&self, error: &ProviderError) -> FailureFacts {
        self.data.failure(&self.provider(), error)
    }

    /// The answer to a question about a semantic model's text, in the
    /// source's dialect.
    pub fn view_text(&self, ask: ViewTextAsk) -> Result<ViewTextAnswer, CoreError> {
        self.data.view_text(&self.provider(), ask)
    }
}
