// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The bodies of the test-only operations (`sdm_contract::test_ops`), built
//! only with the `test-ops` feature, which a shipped build leaves off.

use sdm_contract::ops::OperationResult;
use sdm_contract::test_ops::{TestFetch, TestOperation, TestResult};
use sdm_contract::{CoreError, CoreWarning};

use crate::attached::take_json;
use crate::run::{Attachments, FetchResult, Reply, Run, internal};

/// The run of a test-only operation.
pub(crate) fn run(
    operation: TestOperation,
    attachments: &mut Attachments,
) -> Result<Run, CoreError> {
    Ok(match operation {
        TestOperation::Fetch { fetch } => {
            let fetch: TestFetch = take_json(attachments, fetch)?;
            Run::new(move |asker| async move {
                match asker.fetch(fetch.request, fetch.sql, fetch.expect).await {
                    Reply::Fetch(FetchResult::Data(data)) => {
                        let rows = data.batches.iter().map(|b| b.num_rows() as u64).sum();
                        Ok(OperationResult::Test(TestResult { rows }))
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
                Ok(OperationResult::Test(TestResult { rows: 0 }))
            })
        }
    })
}
