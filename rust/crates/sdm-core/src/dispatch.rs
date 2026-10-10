// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Operation dispatch: a typed operation to the body of its run.

use std::sync::Arc;

use sdm_contract::ops::{Operation, OperationResult, QuerySchema, SourceDescription, SourceSpec};
use sdm_contract::run::StartRun;
use sdm_contract::{CONTRACT_VERSION, CoreError, Detail, ErrorCode, ErrorKind, Scalar};
use sdm_engine::time::Zones;

use crate::attached::{check_attachments, take_json};
use crate::compose::CompositionRoot;
use crate::profiles::Profile;
use crate::run::{Attachments, Run};

/// Refuses a start from a host speaking another contract version, before
/// anything else of the message is read (decision H25): host and core ship
/// together, and no other version's fields mean anything here.
pub fn check_contract(contract: u32) -> Result<(), CoreError> {
    if contract == CONTRACT_VERSION {
        return Ok(());
    }
    Err(CoreError::protocol(
        ErrorCode::InvalidMessage,
        format!(
            "the host speaks contract version {contract}, and this core version {CONTRACT_VERSION}"
        ),
    )
    .with_field("contract", Detail::Scalar(Scalar::UInt32(contract))))
}

/// The run of `start`'s operation, with the attachments the start names
/// and the zones of the host's facts (decision H14): the run resolves zone
/// names through `zones` and nothing else. A contract version other than
/// this build's, host facts the contract does not allow, attachments other
/// than the ones the operation names, or a provider the root's data layer
/// cannot serve, are refused before any run exists. Input-sized fields are
/// read from their attachments here (decision H5).
pub fn start(
    root: &CompositionRoot,
    start: StartRun,
    attachments: Attachments,
    zones: Arc<dyn Zones>,
) -> Result<Run, CoreError> {
    check_contract(start.contract)?;
    // The clock is an instant in UTC, spelled as the contract spells it (E5).
    if !matches!(&start.facts.now, Scalar::Timestamp { zone: Some(zone), .. } if zone == "UTC") {
        return Err(CoreError::new(
            ErrorKind::InvalidRequest,
            ErrorCode::InvalidRequest,
            "the host's clock must be a ts scalar in UTC",
        )
        .with_field("field", Detail::text("facts.now")));
    }
    check_attachments(&start.operation.attachments(), &attachments)?;
    // Only the test-only operations resolve a zone yet; T1.25's time module
    // and T1.6b's local-zone year take them from here.
    #[cfg(not(feature = "test-ops"))]
    let _ = zones;
    let mut attachments = attachments;
    let facts = start.facts;
    Ok(match start.operation {
        Operation::ValidateQuery(request) => {
            let query: String = take_json(&mut attachments, request.query)?;
            let schema: QuerySchema = take_json(&mut attachments, request.schema)?;
            Run::new(move |_asker| async move {
                let profile = Profile::resolve(&request.profile);
                crate::pql::validate(&query, &schema, &profile, &facts)
                    .map(OperationResult::ValidateQuery)
            })
        }
        Operation::DescribeSource(request) => {
            let source: SourceSpec = take_json(&mut attachments, request.source)?;
            let access = root.data_access(&request.provider)?;
            let check_keys = request.check_keys;
            Run::with_data(access, move |asker| {
                crate::describe::run(asker, source, check_keys)
            })
        }
        Operation::Materialize(request) => {
            let _schema: SourceDescription = take_json(&mut attachments, request.schema)?;
            let _query: String = take_json(&mut attachments, request.query)?;
            // The decoder takes a seed of any size and sign; which seeds a
            // run takes is the core's rule (decision H7). Per-backend
            // ranges (RS-7) come with the random streams (T1.22).
            if request.seed.as_ref().is_some_and(|seed| seed.is_negative()) {
                return Err(CoreError::new(
                    ErrorKind::InvalidRequest,
                    ErrorCode::InvalidRequest,
                    "a sampling seed cannot be negative",
                )
                .with_field("field", Detail::text("seed")));
            }
            let access = root.data_access(&request.provider)?;
            Run::with_data(access, |_| async { Err(not_yet("materialize")) })
        }
        Operation::Predict(request) => {
            let _query: String = take_json(&mut attachments, request.state.query)?;
            Run::new(|_| async { Err(not_yet("predict")) })
        }
        #[cfg(feature = "test-ops")]
        Operation::Test(operation) => {
            crate::test_ops::run(root, operation, &mut attachments, facts, zones)?
        }
    })
}

/// The Phase 1b operations, until their routes are built.
fn not_yet(operation: &str) -> CoreError {
    CoreError::new(
        ErrorKind::Unsupported,
        ErrorCode::UnsupportedRequest,
        format!("{operation} is not built into this core yet"),
    )
    .with_field("operation", Detail::text(operation))
}
