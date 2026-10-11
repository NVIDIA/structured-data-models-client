// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Operation dispatch: a typed operation to the body of its run.

use std::sync::Arc;

use sdm_contract::ops::{
    Operation, OperationResult, QuerySchema, REQUEST_STATE_VERSION, SourceDescription, SourceSpec,
};
use sdm_contract::run::StartRun;
use sdm_contract::{CONTRACT_VERSION, CoreError, Detail, ErrorCode, ErrorKind, Scalar};
use sdm_engine::time::Zones;

use crate::attached::{check_attachments, take_json};
use crate::compose::CompositionRoot;
use crate::profiles::Profile;
use crate::run::{Attachment, Attachments, Run, internal};

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
            let description: SourceDescription = take_json(&mut attachments, request.schema)?;
            let query: String = take_json(&mut attachments, request.query)?;
            let indices = match request.indices {
                Some(id) => match attachments.remove(&id) {
                    Some(Attachment::Arrow(data)) => Some(data),
                    _ => return Err(internal("the indices are not the Arrow data named")),
                },
                None => None,
            };
            // The decoder takes a seed of any size and sign; which seeds a
            // run takes is the core's rule (decision H7): not a negative
            // one, and none past the largest the provider declares (RS-7),
            // which the core reads when it plans the sample
            // (`DataAccess::plan`).
            if request.seed.as_ref().is_some_and(|seed| seed.is_negative()) {
                return Err(CoreError::new(
                    ErrorKind::InvalidRequest,
                    ErrorCode::InvalidRequest,
                    "a sampling seed cannot be negative",
                )
                .with_field("field", Detail::text("seed")));
            }
            let access = root.data_access(&request.provider)?;
            let inputs = crate::materialize::Inputs {
                description,
                query,
                indices,
                anchor_time: request.anchor_time,
                seed: request.seed,
                facts,
            };
            Run::with_data(access, move |asker| crate::materialize::run(asker, inputs))
        }
        Operation::Predict(request) => {
            // The state's version first, by name, before anything of the
            // state is read (decision H23).
            if request.state.version != REQUEST_STATE_VERSION {
                return Err(CoreError::new(
                    ErrorKind::InvalidRequest,
                    ErrorCode::InvalidRequest,
                    "the request state is of a version this core does not read",
                )
                .with_field("field", Detail::text("state.version"))
                .with_field(
                    "version",
                    Detail::Scalar(Scalar::UInt32(request.state.version)),
                ));
            }
            let Some(Attachment::Bytes(payload)) = attachments.remove(&request.state.payload)
            else {
                return Err(internal("the payload is not the bytes named"));
            };
            // The query rides with the state; the checks that name it (an
            // empty context, a constant target) stay in the host's client
            // at T1.11d (decision E45), so it is read and not used.
            let _query: String = take_json(&mut attachments, request.state.query)?;
            let inputs = crate::predict::Inputs {
                payload,
                model: request.model,
                request_id: request.request_id,
            };
            Run::new(move |asker| crate::predict::run(asker, inputs))
        }
        #[cfg(feature = "test-ops")]
        Operation::Test(operation) => {
            crate::test_ops::run(root, operation, &mut attachments, facts, zones)?
        }
    })
}
