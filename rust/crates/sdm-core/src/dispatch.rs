// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Operation dispatch: a typed operation to the body of its run.

use sdm_contract::ops::{Operation, OperationResult};
use sdm_contract::run::StartRun;
use sdm_contract::{CONTRACT_VERSION, CoreError, Detail, ErrorKind, Scalar, codes};

use crate::compose::CompositionRoot;
use crate::profiles::Profile;
use crate::run::Run;

/// The run of `start`'s operation. A contract version other than this
/// build's, or host facts the contract does not allow, are refused before
/// any run exists.
pub fn start(root: &CompositionRoot, start: StartRun) -> Result<Run, CoreError> {
    if start.contract != CONTRACT_VERSION {
        return Err(CoreError::protocol(
            codes::INVALID_MESSAGE,
            format!(
                "the host speaks contract version {}, and this core version {CONTRACT_VERSION}",
                start.contract
            ),
        )
        .with_field("contract", Detail::Scalar(Scalar::UInt32(start.contract))));
    }
    // The clock is an instant in UTC, spelled as the contract spells it (E5).
    if !matches!(&start.facts.now, Scalar::Timestamp { zone: Some(zone), .. } if zone == "UTC") {
        return Err(CoreError::new(
            ErrorKind::InvalidRequest,
            codes::INVALID_REQUEST,
            "the host's clock must be a ts scalar in UTC",
        )
        .with_field("field", Detail::text("facts.now")));
    }
    let facts = start.facts;
    Ok(match start.operation {
        Operation::ValidateQuery(request) => Run::new(move |_asker| async move {
            let profile = Profile::resolve(&request.profile);
            crate::pql::validate(&request.query, &request.schema, &profile, &facts)
                .map(OperationResult::ValidateQuery)
        }),
        Operation::DescribeSource(request) => {
            let _access = root.data_access(&request.provider)?;
            Run::new(|_| async { Err(not_yet("describe_source")) })
        }
        Operation::Materialize(request) => {
            let _access = root.data_access(&request.provider)?;
            Run::new(|_| async { Err(not_yet("materialize")) })
        }
        Operation::Predict(_) => Run::new(|_| async { Err(not_yet("predict")) }),
    })
}

/// The Phase 1b operations, until their routes are built.
fn not_yet(operation: &str) -> CoreError {
    CoreError::new(
        ErrorKind::Unsupported,
        codes::UNSUPPORTED_REQUEST,
        format!("{operation} is not built into this core yet"),
    )
    .with_field("operation", Detail::text(operation))
}
