// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The bodies of the test-only operations (`sdm_contract::test_ops`), built
//! only with the `test-ops` feature, which a shipped build leaves off.

use std::sync::Arc;
use std::time::Duration;

use sdm_contract::data::DataRequest;
use sdm_contract::ops::{OperationResult, ProfileName, ProfileSpec, QuerySchema};
use sdm_contract::test_ops::{PanicAt, TestFetch, TestOperation, TestResult};
use sdm_contract::{CoreError, CoreWarning, Detail, HostFacts, Scalar, Sentence, TimestampUnit};
use sdm_engine::time::Zones;

use crate::attached::take_json;
use crate::compose::CompositionRoot;
use crate::profiles::Profile;
use crate::run::{Asker, Attachment, Attachments, FetchResult, Reply, Run, internal};

/// The run of a test-only operation.
pub(crate) fn run(
    root: &CompositionRoot,
    operation: TestOperation,
    attachments: &mut Attachments,
    facts: HostFacts,
    zones: Arc<dyn Zones>,
) -> Result<Run, CoreError> {
    let keyed = match &operation {
        TestOperation::Keyed { keys, .. } => match attachments.remove(keys) {
            Some(Attachment::Arrow(data)) => Some(data),
            _ => return Err(internal("a keyed fetch's keys are not Arrow")),
        },
        _ => None,
    };
    Ok(match operation {
        TestOperation::Fetch { fetch } | TestOperation::Keyed { fetch, .. } => {
            let mut fetch: TestFetch = take_json(attachments, fetch)?;
            // The test's own plan, or the one the root's data layer makes
            // for the provider named, through the run's data access, as an
            // operation's body asks.
            let declaring = !fetch.undeclared.is_empty() || fetch.fallback.is_some();
            let (plan, access) = match (fetch.plan, &fetch.provider) {
                (Some(plan), None) if !declaring => (Some(plan), None),
                (None, Some(provider)) => (
                    None,
                    Some(
                        root.data_access(provider)?
                            .declaring(&fetch.undeclared, fetch.fallback),
                    ),
                ),
                _ => {
                    return Err(CoreError::new(
                        Sentence::TestOperationRefused,
                        "a test fetch names a plan or a provider, and not both, and what a provider declares only with a provider",
                    ));
                }
            };
            let make = move |asker: Asker| async move {
                // The keys, attached as the run's own, become the request's
                // seeds, as an operation's are.
                if let Some(keys) = keyed {
                    let id = asker.attach(Attachment::Arrow(keys))?;
                    match &mut fetch.request {
                        DataRequest::Neighbors { seeds, .. }
                        | DataRequest::Lookup { seeds, .. } => {
                            seeds.attachment = id;
                        }
                        _ => return Err(internal("a keyed fetch's request takes no seeds")),
                    }
                }
                let asked = match plan {
                    Some(plan) => asker.fetch_planned(fetch.request, plan),
                    None => asker.fetch(fetch.request)?,
                };
                match asked.await {
                    Reply::Fetch(FetchResult::Data(data)) => {
                        let rows = data.batches.iter().map(|b| b.num_rows() as u64).sum();
                        let data = if fetch.echo {
                            Some(asker.attach(Attachment::Arrow(data))?)
                        } else {
                            None
                        };
                        Ok(OperationResult::Test(TestResult {
                            rows,
                            data,
                            offset_seconds: None,
                        }))
                    }
                    Reply::Fetch(FetchResult::Error(error)) => Err(CoreError::provider(&error)),
                    Reply::Failed(error) => Err(error),
                    Reply::Model(_) => Err(internal("a fetch was answered by the model")),
                    Reply::Fetch(FetchResult::Page { .. }) => {
                        Err(internal("a page reached the body"))
                    }
                }
            };
            match access {
                Some(access) => Run::with_data(access, make),
                None => Run::new(make),
            }
        }
        TestOperation::Labels { materialize } => crate::dispatch::materialize(
            root,
            sdm_contract::ops::Operation::Materialize(*materialize),
            std::mem::take(attachments),
            facts,
            Arc::clone(&zones),
            true,
        )?,
        TestOperation::Finish { result } => {
            let result: OperationResult = take_json(attachments, result)?;
            Run::new(move |_| async move { Ok(result) })
        }
        TestOperation::Fail { error: id } => {
            let error: CoreError = take_json(attachments, id)?;
            // The one host-to-core message that carries a `CoreError`: one
            // whose kind its code does not carry is refused as the start's
            // own message (decision E22).
            if !error.holds_to_table() {
                return Err(CoreError::protocol(
                    Sentence::InvalidMessage,
                    "the error carries a kind its code does not allow",
                )
                .with_field("attachment", Detail::Scalar(Scalar::UInt32(id.0))));
            }
            Run::new(move |_| async move { Err(error) })
        }
        TestOperation::Warn { warnings: id } => {
            let warnings: Vec<CoreWarning> = take_json(attachments, id)?;
            // A warning whose code is not its sentence's is refused as the
            // start's own message (T3.17).
            if !warnings.iter().all(CoreWarning::holds_to_table) {
                return Err(CoreError::protocol(
                    Sentence::InvalidMessage,
                    "a warning carries a code its sentence does not give",
                )
                .with_field("attachment", Detail::Scalar(Scalar::UInt32(id.0))));
            }
            Run::new(move |asker| async move {
                for warning in warnings {
                    asker.warn(warning);
                }
                Ok(OperationResult::Test(TestResult {
                    rows: 0,
                    data: None,
                    offset_seconds: None,
                }))
            })
        }
        TestOperation::Panic { at, payload } => match at {
            PanicAt::Start => panic!("{payload}"),
            PanicAt::Body => Run::new(move |_| async move { panic!("{payload}") }),
            // Inside the parser, on the adapter's grown stack.
            PanicAt::AdapterStack => Run::new(move |_| async move {
                sdm_pql::parser::panic_in_next_parse(payload);
                let profile = Profile::resolve(&ProfileSpec {
                    name: ProfileName::Relational,
                    model: None,
                    capabilities: Default::default(),
                });
                let schema = QuerySchema {
                    tables: vec![],
                    links: vec![],
                };
                crate::pql::validate(
                    "PREDICT t.c FOR EACH t.id",
                    &schema,
                    &profile,
                    &facts,
                    &*zones,
                )
                .map(|(validation, _)| OperationResult::ValidateQuery(validation))
            }),
        },
        // One known event, then a long call, so a binding's tests can show
        // the event reaches the host's logging once and another thread runs
        // while the core holds the call (decision H8).
        TestOperation::Log { hold_ms } => Run::new(move |_| async move {
            tracing::info!(target: "sdm_core::test_ops", "a test-only event");
            std::thread::sleep(Duration::from_millis(hold_ms));
            Ok(OperationResult::Test(TestResult {
                rows: 0,
                data: None,
                offset_seconds: None,
            }))
        }),
        TestOperation::Abort => Run::new(|_| async { std::process::abort() }),
        // The zones the host passed are the ones the run resolves with: the
        // answer is the offset the run's own zones give (decision H14).
        TestOperation::Zone { zone, at } => Run::new(move |_| async move {
            let instant = match at {
                Scalar::Timestamp {
                    unit: TimestampUnit::S,
                    value,
                    ..
                } => jiff::Timestamp::from_second(value),
                Scalar::Timestamp {
                    unit: TimestampUnit::Ms,
                    value,
                    ..
                } => jiff::Timestamp::from_millisecond(value),
                Scalar::Timestamp {
                    unit: TimestampUnit::Us,
                    value,
                    ..
                } => jiff::Timestamp::from_microsecond(value),
                Scalar::Timestamp {
                    unit: TimestampUnit::Ns,
                    value,
                    ..
                } => jiff::Timestamp::from_nanosecond(i128::from(value)),
                _ => {
                    return Err(CoreError::new(
                        Sentence::TestOperationRefused,
                        "the zone operation reads a ts scalar",
                    ));
                }
            }
            .map_err(|_| internal("the instant is out of the zone database's range"))?;
            let found = zones.zone(&zone).map_err(|_| unknown_zone())?;
            Ok(OperationResult::Test(TestResult {
                rows: 0,
                data: None,
                offset_seconds: Some(found.to_offset(instant).seconds()),
            }))
        }),
    })
}

/// A zone name the run's zones do not hold. The name is left out: it can
/// come from the data or the caller (decision H17).
fn unknown_zone() -> CoreError {
    CoreError::new(
        Sentence::UnknownZone,
        "a time zone is not in the host's zone lookup roots or the bundled database",
    )
}
