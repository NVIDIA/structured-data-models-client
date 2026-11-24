// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A built-in source as every host reaches it: the core sends it the typed
//! requests it sends any source, declaring what its SQL kit declares
//! ([`SqlSource::provider`]), and each request is answered through the kit
//! ([`SqlSource::answer`]), which writes the statements, binds the key
//! sets and hands each fetch to this host's driver to run: DuckDB's
//! ([`crate::duck`]), SQLite's ([`crate::sqlite`]) or a recording's
//! ([`crate::replay`]). One source answers one run, as the Python and Java
//! hosts make one kit per run: the namespace the kit settles and the
//! catalog reads it keeps are that run's.

use std::collections::BTreeMap;

use anyhow::{Result, anyhow};
use sdm_contract::data::{
    AttachmentId, DataRequest, FailureFacts, KeyData, ProviderChoice, ProviderError,
};
use sdm_contract::run::FetchOutcome;
use sdm_contract::{CoreError, ErrorKind};
use sdm_host::kit::{Answered, HostKit, KitError, SqlChoice, SqlPlan};
use serde_json::Value;

use crate::duck::Fetched;

/// How a request was answered: its rows, the driver's error, or the kit's
/// refusal by name.
#[derive(Debug)]
pub enum Outcome {
    Data(Fetched),
    Error(ProviderError),
    Refused(CoreError),
}

impl Outcome {
    /// The outcome as the fetch reply carries it, its rows the attachment
    /// `data` names.
    pub fn reply(self, data: AttachmentId) -> (FetchOutcome, Option<Fetched>) {
        match self {
            Outcome::Data(fetched) => (
                FetchOutcome::Data {
                    data,
                    continuation: None,
                },
                Some(fetched),
            ),
            Outcome::Error(error) => (FetchOutcome::Error { error }, None),
            Outcome::Refused(error) => (FetchOutcome::Refused { error }, None),
        }
    }

    /// The rows, or the failure as a test reads it.
    pub fn data(self) -> std::result::Result<Fetched, Failed> {
        match self {
            Outcome::Data(fetched) => Ok(fetched),
            Outcome::Error(error) => Err(Failed::Provider(error)),
            Outcome::Refused(error) => Err(Failed::Refused(error)),
        }
    }
}

/// Why a request was not answered.
#[derive(Debug)]
pub enum Failed {
    /// The driver failed to run a fetch.
    Provider(ProviderError),
    /// The kit refused the request by name.
    Refused(CoreError),
}

/// One fetch the kit asks this host's driver to run: the request it
/// answers or the part of it, its statements, and the batches they bind.
pub struct Fetch<'a> {
    pub request: &'a DataRequest,
    pub plan: &'a SqlPlan,
    pub keys: &'a BTreeMap<AttachmentId, Fetched>,
}

/// A driver as the kit calls it: one fetch's answer, the driver's error,
/// or this host's own failure (a fetch a recording does not hold), which
/// fails the run rather than reaching the core.
pub type Run<'a> = dyn FnMut(Fetch<'_>) -> Result<std::result::Result<Fetched, ProviderError>> + 'a;

/// A built-in source of one run, answering through its kit.
#[derive(Debug)]
pub struct SqlSource {
    kit: HostKit,
}

impl SqlSource {
    /// The source `choice` names; refused by name where this build writes
    /// no SQL for it.
    pub fn new(choice: SqlChoice) -> std::result::Result<Self, CoreError> {
        Ok(SqlSource {
            kit: HostKit::new(choice)?,
        })
    }

    /// The source of `dialect`, over the connection's current namespace.
    pub fn of(dialect: &str) -> Self {
        SqlSource::new(SqlChoice::new(dialect))
            .unwrap_or_else(|error| panic!("{dialect} has no kit: {error:?}"))
    }

    /// The source a case's provider choice names, as JSON writes it.
    pub fn from_json(choice: &Value) -> Result<Self> {
        let choice: SqlChoice = serde_json::from_value(choice.clone())
            .map_err(|_| anyhow!("a provider choice that names no built-in source"))?;
        SqlSource::new(choice).map_err(|error| anyhow!("the source has no kit: {error:?}"))
    }

    /// The source the kit answers for, its namespace settled once the run
    /// asked it.
    pub fn choice(&self) -> SqlChoice {
        self.kit.provider()
    }

    /// The provider the core is started with: what the kit declares.
    pub fn provider(&self) -> Result<ProviderChoice> {
        Ok(ProviderChoice::new(self.kit.capabilities().map_err(
            |error| anyhow!("the kit declares nothing: {error:?}"),
        )?))
    }

    /// What the driver's failure `error` tells the core, as the kit reads
    /// it.
    pub fn failure(&self, error: &ProviderError) -> FailureFacts {
        self.kit.failure(error)
    }

    /// The statements of the first fetch the kit asks for `request`, each
    /// key set it names bound empty, as the kit writes them, or its refusal
    /// by name; nothing is run.
    pub fn plan(&self, request: &DataRequest) -> std::result::Result<SqlPlan, CoreError> {
        let empty: BTreeMap<AttachmentId, Fetched> = crate::memory::Memory::key_sets(request)
            .into_iter()
            .map(|set| {
                let fields: Vec<_> = set.columns.iter().map(|c| c.to_arrow()).collect();
                (
                    set.attachment,
                    Fetched {
                        schema: std::sync::Arc::new(arrow_schema::Schema::new(fields)),
                        batches: Vec::new(),
                    },
                )
            })
            .collect();
        let mut planned = None;
        let outcome = self
            .answer(request, &empty, &mut |fetch| {
                planned = Some(fetch.plan.clone());
                Ok(Err(ProviderError {
                    kind: ErrorKind::Internal,
                    retryable: false,
                    sqlstate: None,
                    vendor_code: None,
                    error_class: None,
                    driver_message: None,
                    facts: None,
                }))
            })
            .unwrap_or_else(|error| panic!("planning ran nothing: {error}"));
        match (planned, outcome) {
            (Some(plan), _) => Ok(plan),
            (None, Outcome::Refused(error)) => Err(error),
            (None, other) => panic!("the kit planned no fetch: {other:?}"),
        }
    }

    /// The answer to `request`, its key sets `keys` by the id the request
    /// names each by, every fetch run by `run`.
    pub fn answer(
        &self,
        request: &DataRequest,
        keys: &BTreeMap<AttachmentId, Fetched>,
        run: &mut Run<'_>,
    ) -> Result<Outcome> {
        let made: BTreeMap<AttachmentId, KeyData> = keys
            .iter()
            .map(|(id, fetched)| {
                (
                    *id,
                    KeyData {
                        schema: fetched.schema.clone(),
                        batches: fetched.batches.clone(),
                    },
                )
            })
            .collect();
        let mut failed: Option<anyhow::Error> = None;
        let answered = self.kit.answer(request, &made, &mut |fetch| {
            let handed: BTreeMap<AttachmentId, Fetched> = fetch
                .keys
                .iter()
                .map(|(id, made)| {
                    (
                        *id,
                        Fetched {
                            schema: made.schema.clone(),
                            batches: made.batches.clone(),
                        },
                    )
                })
                .collect();
            match run(Fetch {
                request: fetch.request,
                plan: fetch.plan,
                keys: &handed,
            }) {
                Ok(Ok(fetched)) => Ok(Answered {
                    schema: fetched.schema,
                    batches: fetched.batches,
                }),
                Ok(Err(error)) => Err(error),
                Err(error) => {
                    failed = Some(error);
                    Err(ProviderError {
                        kind: ErrorKind::Internal,
                        retryable: false,
                        sqlstate: None,
                        vendor_code: None,
                        error_class: None,
                        driver_message: None,
                        facts: None,
                    })
                }
            }
        });
        if let Some(error) = failed {
            return Err(error);
        }
        Ok(match answered {
            Ok(answered) => Outcome::Data(Fetched {
                schema: answered.schema,
                batches: answered.batches,
            }),
            Err(KitError::Provider(error)) => Outcome::Error(error),
            Err(KitError::Refused(error)) => Outcome::Refused(error),
        })
    }
}
