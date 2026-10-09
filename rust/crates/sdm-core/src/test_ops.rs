// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The bodies of the test-only operations (`sdm_contract::test_ops`), built
//! only with the `test-ops` feature, which a shipped build leaves off.

use std::time::Duration;

use sdm_contract::ops::{OperationResult, ProfileName, ProfileSpec, QuerySchema};
use sdm_contract::test_ops::{PanicAt, TestFetch, TestOperation, TestResult};
use sdm_contract::{CoreError, CoreWarning, HostFacts};

use crate::attached::take_json;
use crate::profiles::Profile;
use crate::run::{Attachment, Attachments, FetchResult, Reply, Run, internal};

/// The run of a test-only operation.
pub(crate) fn run(
    operation: TestOperation,
    attachments: &mut Attachments,
    facts: HostFacts,
) -> Result<Run, CoreError> {
    Ok(match operation {
        TestOperation::Fetch { fetch } => {
            let fetch: TestFetch = take_json(attachments, fetch)?;
            Run::new(move |asker| async move {
                match asker.fetch(fetch.request, fetch.sql, fetch.expect).await {
                    Reply::Fetch(FetchResult::Data(data)) => {
                        let rows = data.batches.iter().map(|b| b.num_rows() as u64).sum();
                        let data = if fetch.echo {
                            Some(asker.attach(Attachment::Arrow(data))?)
                        } else {
                            None
                        };
                        Ok(OperationResult::Test(TestResult { rows, data }))
                    }
                    Reply::Fetch(FetchResult::Error(error)) => Err(CoreError::provider(&error)),
                    Reply::Model(_) => Err(internal("a fetch was answered by the model")),
                }
            })
        }
        TestOperation::Finish { result } => {
            let result: OperationResult = take_json(attachments, result)?;
            Run::new(move |_| async move { Ok(result) })
        }
        TestOperation::Fail { error } => {
            let error: CoreError = take_json(attachments, error)?;
            Run::new(move |_| async move { Err(error) })
        }
        TestOperation::Warn { warnings } => {
            let warnings: Vec<CoreWarning> = take_json(attachments, warnings)?;
            Run::new(move |asker| async move {
                for warning in warnings {
                    asker.warn(warning);
                }
                Ok(OperationResult::Test(TestResult {
                    rows: 0,
                    data: None,
                }))
            })
        }
        TestOperation::Panic { at, payload } => match at {
            PanicAt::Start => panic!("{payload}"),
            PanicAt::Body => Run::new(move |_| async move { panic!("{payload}") }),
            // Inside the interim parser, on the adapter's grown stack.
            PanicAt::AdapterStack => Run::new(move |_| async move {
                sdm_pql_interim::parser::panic_in_next_parse(payload);
                let profile = Profile::resolve(&ProfileSpec {
                    name: ProfileName::Relational,
                    model: None,
                    capabilities: Default::default(),
                });
                let schema = QuerySchema {
                    tables: vec![],
                    links: vec![],
                };
                crate::pql::validate("PREDICT t.c FOR EACH t.id", &schema, &profile, &facts)
                    .map(OperationResult::ValidateQuery)
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
            }))
        }),
        TestOperation::Abort => Run::new(|_| async { std::process::abort() }),
    })
}
