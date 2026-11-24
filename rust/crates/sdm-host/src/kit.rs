// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The SQL kit as every binding reaches it: `sdm-sql`'s [`SqlKit`] reading
//! semantic models with the core's white space, writing doubles into SQL
//! as the core writes them, and admitting each answer the host's driver
//! hands over as a run admits a fetch's (check (a), the provider's opaque
//! Arrow extension types read as what they mean).

use std::collections::BTreeMap;

use sdm_contract::CoreError;
use sdm_contract::data::{
    AttachmentId, DataRequest, FailureFacts, KeyData, ProviderCapabilities, ProviderChoice,
    ProviderError,
};
use sdm_contract::view::{ViewTextAnswer, ViewTextAsk};
pub use sdm_sql::kit::{Answered, KitError, KitFetch};

use crate::ArrowData;
use crate::admission::admit_from;

/// A built-in source's SQL kit, as a binding holds it.
#[derive(Debug, Clone)]
pub struct HostKit {
    kit: sdm_sql::kit::SqlKit,
    vendor: Option<String>,
}

impl HostKit {
    /// The kit of the built-in source `provider` names; refused by name
    /// where this build writes no SQL for it.
    pub fn new(provider: ProviderChoice) -> Result<Self, CoreError> {
        let kit = sdm_sql::kit::SqlKit::new(provider, Some(crate::spaces()))?;
        let vendor = kit.capabilities()?.opaque_vendor;
        Ok(HostKit { kit, vendor })
    }

    /// The source the kit answers for, as the core names it.
    pub fn provider(&self) -> &ProviderChoice {
        self.kit.provider()
    }

    /// What the source answers.
    pub fn capabilities(&self) -> Result<ProviderCapabilities, CoreError> {
        self.kit.capabilities()
    }

    /// The answer to `request`, its key sets `keys`, each fetch run by
    /// `run` and its answer admitted before the kit reads it.
    pub fn answer(
        &self,
        request: &DataRequest,
        keys: &BTreeMap<AttachmentId, KeyData>,
        run: &mut dyn FnMut(KitFetch<'_>) -> Result<Answered, ProviderError>,
    ) -> Result<Answered, KitError> {
        let mut refused: Option<CoreError> = None;
        let vendor = self.vendor.as_deref();
        let answered = self
            .kit
            .answer(request, keys, sdm_core::float_text, &mut |fetch| {
                let raw = run(fetch)?;
                match admit_from(&ArrowData::new(raw.schema, raw.batches), vendor) {
                    Ok(admitted) => Ok(Answered {
                        schema: admitted.schema,
                        batches: admitted.batches,
                    }),
                    Err(error) => {
                        // Not the driver's failure: the kit refuses the
                        // answer by the admission's own error.
                        refused = Some(error);
                        Err(ProviderError {
                            kind: sdm_contract::ErrorKind::Internal,
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
        match (answered, refused) {
            (Err(_), Some(error)) => Err(KitError::Refused(error)),
            (answered, _) => answered,
        }
    }

    /// What the driver's failure `error` tells the core.
    pub fn failure(&self, error: &ProviderError) -> FailureFacts {
        self.kit.failure(error)
    }

    /// The answer to a question about a semantic model's text.
    pub fn view_text(&self, ask: ViewTextAsk) -> Result<ViewTextAnswer, CoreError> {
        self.kit.view_text(ask)
    }
}

/// A binding's driver as [`HostKit::answer_bytes`] calls it: one fetch's
/// plan and request as the contract writes them, and the batches the plan
/// binds by id; its answer as Arrow, or the driver's error.
pub type RunBytes<'a> = dyn FnMut(&[u8], &[u8], BTreeMap<AttachmentId, ArrowData>) -> Result<ArrowData, ProviderError>
    + 'a;

/// Why [`HostKit::answer_bytes`] did not answer, as a binding raises it.
#[derive(Debug)]
pub enum KitFailure {
    /// The kit refused the request: the `error` control message to raise.
    Refused(crate::Outgoing),
    /// The host's driver failed: the error its `run` returned.
    Provider(ProviderError),
}

/// `error` as the `error` control message a binding raises.
pub fn refusal(error: &CoreError) -> crate::Outgoing {
    crate::write_error(error, || {
        Err(CoreError::new(
            sdm_contract::Sentence::Internal,
            "a kit refusal carries no attachment",
        ))
    })
}

fn from_json<T: serde::de::DeserializeOwned>(
    bytes: &[u8],
    what: &str,
) -> Result<T, crate::Outgoing> {
    serde_json::from_slice(bytes).map_err(|_| {
        refusal(&CoreError::new(
            sdm_contract::Sentence::Internal,
            format!("the SQL kit was handed a {what} it cannot read"),
        ))
    })
}

fn to_json<T: serde::Serialize>(value: &T) -> Result<Vec<u8>, crate::Outgoing> {
    serde_json::to_vec(value).map_err(|_| {
        refusal(&CoreError::new(
            sdm_contract::Sentence::Internal,
            "the SQL kit's answer cannot be written",
        ))
    })
}

impl HostKit {
    /// [`HostKit::new`] of a provider choice as the contract writes it.
    pub fn from_bytes(provider: &[u8]) -> Result<Self, crate::Outgoing> {
        let provider: ProviderChoice = from_json(provider, "source")?;
        HostKit::new(provider).map_err(|error| refusal(&error))
    }

    /// [`HostKit::capabilities`] as the contract writes them.
    pub fn capabilities_bytes(&self) -> Result<Vec<u8>, crate::Outgoing> {
        to_json(&self.capabilities().map_err(|error| refusal(&error))?)
    }

    /// [`HostKit::answer`] of a request as the contract writes it, its key
    /// sets as Arrow by id; `run` is handed each fetch's plan and request as
    /// the contract writes them and the batches the plan binds.
    pub fn answer_bytes(
        &self,
        request: &[u8],
        keys: BTreeMap<AttachmentId, ArrowData>,
        run: &mut RunBytes<'_>,
    ) -> Result<ArrowData, KitFailure> {
        let request: DataRequest = from_json(request, "request").map_err(KitFailure::Refused)?;
        let keys: BTreeMap<AttachmentId, KeyData> = keys
            .into_iter()
            .map(|(id, data)| {
                (
                    id,
                    KeyData {
                        schema: data.schema,
                        batches: data.batches,
                    },
                )
            })
            .collect();
        let mut unwritten: Option<crate::Outgoing> = None;
        let answered = self.answer(&request, &keys, &mut |fetch| {
            let written = to_json(fetch.plan).and_then(|plan| Ok((plan, to_json(fetch.request)?)));
            let (plan, asked) = match written {
                Ok(written) => written,
                Err(out) => {
                    unwritten = Some(out);
                    return Err(ProviderError {
                        kind: sdm_contract::ErrorKind::Internal,
                        retryable: false,
                        sqlstate: None,
                        vendor_code: None,
                        error_class: None,
                        driver_message: None,
                        facts: None,
                    });
                }
            };
            let handed = fetch
                .keys
                .iter()
                .map(|(id, made)| {
                    (
                        *id,
                        ArrowData::new(made.schema.clone(), made.batches.clone()),
                    )
                })
                .collect();
            let data = run(&plan, &asked, handed)?;
            Ok(Answered {
                schema: data.schema,
                batches: data.batches,
            })
        });
        match (answered, unwritten) {
            (_, Some(out)) => Err(KitFailure::Refused(out)),
            (Ok(answered), None) => Ok(ArrowData::new(answered.schema, answered.batches)),
            (Err(KitError::Refused(error)), None) => Err(KitFailure::Refused(refusal(&error))),
            (Err(KitError::Provider(error)), None) => Err(KitFailure::Provider(error)),
        }
    }
}
