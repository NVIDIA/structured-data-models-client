// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Operation dispatch: a typed operation to the body of its run.

use std::sync::Arc;

use sdm_contract::ops::{
    HeldSchema, Operation, OperationResult, QuerySchema, REQUEST_STATE_VERSION,
    SAMPLER_STATE_VERSION, SamplerMemo, SchemaEdit, SourceDescription, SourceSpec,
};
use sdm_contract::run::StartRun;
use sdm_contract::{
    CONTRACT_VERSION, CoreError, Detail, ErrorCode, ErrorKind, HeldState, HostFacts, Scalar,
};
use sdm_engine::time::Zones;

use crate::attached::{check_attachments, take_json, take_json_if};
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
    let mut attachments = attachments;
    let facts = start.facts;
    Ok(match start.operation {
        Operation::ValidateQuery(request) => {
            let query: String = take_json(&mut attachments, request.query)?;
            let schema: QuerySchema = take_json(&mut attachments, request.schema)?;
            let zones = Arc::clone(&zones);
            Run::new(move |asker| async move {
                let profile = Profile::resolve(&request.profile);
                let (validation, warnings) =
                    crate::pql::validate(&query, &schema, &profile, &facts, &*zones)?;
                // D4's warning, on the run's warnings channel.
                for warning in warnings {
                    asker.warn(warning);
                }
                Ok(OperationResult::ValidateQuery(validation))
            })
        }
        Operation::DescribeSource(request) => {
            let source: SourceSpec = take_json(&mut attachments, request.source)?;
            let access = root.data_access(&request.provider)?;
            let check_keys = request.check_keys;
            let zones = Arc::clone(&zones);
            Run::with_data(access, move |asker| {
                crate::describe::run(asker, source, check_keys, facts, zones)
            })
        }
        Operation::EditSchema(request) => {
            let described: SourceDescription = take_json(&mut attachments, request.schema)?;
            let held: Option<HeldSchema> = take_json_if(&mut attachments, request.held)?;
            let edit: SchemaEdit = take_json(&mut attachments, request.edit)?;
            Run::new(move |asker| crate::edit::run(asker, described, held, edit))
        }
        operation @ Operation::Materialize(_) => materialize(
            root,
            operation,
            attachments,
            facts,
            Arc::clone(&zones),
            #[cfg(feature = "test-ops")]
            false,
        )?,
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
            // A request asked in several model calls (contract v26's
            // `batches`, `main`'s multi-call predict) is T2.8c's: refused by
            // name until then, never predicted as its first call alone.
            if request.state.batches.is_some() {
                return Err(CoreError::new(
                    ErrorKind::Unsupported,
                    ErrorCode::UnsupportedRequest,
                    "a request asked in several model calls is not predicted by this core yet",
                )
                .with_field("field", Detail::text("state.batches")));
            }
            let Some(Attachment::Bytes(payload)) = attachments.remove(&request.state.payload)
            else {
                return Err(internal("the payload is not the bytes named"));
            };
            // The query rides with the state: the constant-context notice
            // names the target from it where the wire calls it TARGET
            // (T1.28b).
            let query: String = take_json(&mut attachments, request.state.query)?;
            // A forecast's later steps, each as the first is (T1.36 fix 21).
            let forecast = match &request.state.forecast {
                None => None,
                Some(steps) => {
                    let mut later = Vec::with_capacity(steps.len());
                    for step in steps {
                        let Some(Attachment::Bytes(payload)) = attachments.remove(&step.payload)
                        else {
                            return Err(internal("the payload is not the bytes named"));
                        };
                        let query: String = take_json(&mut attachments, step.query)?;
                        // Each step's call carries its own id (T1.36 fix 24).
                        let Some(request_id) = step.request_id.clone() else {
                            return Err(CoreError::new(
                                ErrorKind::InvalidRequest,
                                ErrorCode::InvalidRequest,
                                "a forecast's later step names no request id",
                            )
                            .with_field("field", Detail::text("state.forecast.request_id")));
                        };
                        later.push((step.step, payload, query, request_id));
                    }
                    Some(later)
                }
            };
            let inputs = crate::predict::Inputs {
                payload,
                query,
                model: request.model,
                request_id: request.request_id,
                outputs: request.outputs,
                default_outputs: request.default_outputs,
                inference_asked: request.inference_asked,
                forecast,
            };
            Run::new(move |asker| crate::predict::run(asker, inputs))
        }
        #[cfg(feature = "test-ops")]
        Operation::Test(operation) => {
            crate::test_ops::run(root, operation, &mut attachments, facts, zones)?
        }
    })
}

/// A `materialize` run of `request`, with the attachments it names; with
/// `labels_only`, the test-only `labels` operation's, which finishes with
/// the context's labels.
pub(crate) fn materialize(
    root: &CompositionRoot,
    operation: Operation,
    mut attachments: Attachments,
    facts: HostFacts,
    zones: Arc<dyn Zones>,
    #[cfg(feature = "test-ops")] labels_only: bool,
) -> Result<Run, CoreError> {
    match operation {
        Operation::Materialize(request) => {
            let described: SourceDescription = take_json(&mut attachments, request.schema)?;
            let held: Option<HeldSchema> = take_json_if(&mut attachments, request.held)?;
            // The schema as the host holds it, before anything else of the
            // call is read; one changed in a way no edit the core takes says
            // is refused by name (`schema_edited`, decision E67), as the
            // host's bridge refused it before the core ran until T1.24b.
            let description = sdm_schema::edit::read_held(described, held, "materialize")?;
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
            // The sampler state's version first, by name, before anything of
            // the state is read, as a request state's (decision H23); then
            // its memo, which a run reads only if it was made under this
            // provider and description (decision E52). Each refusal of the
            // state tells the host to drop it (decision E74), so its next
            // run starts afresh rather than being refused the same way.
            let memo = match &request.sampler_state {
                None => None,
                Some(state) => {
                    if state.version != SAMPLER_STATE_VERSION {
                        return Err(CoreError::new(
                            ErrorKind::InvalidRequest,
                            ErrorCode::InvalidRequest,
                            "the sampler state is of a version this core does not read",
                        )
                        .with_field("field", Detail::text("sampler_state.version"))
                        .with_field("version", Detail::Scalar(Scalar::UInt32(state.version)))
                        .dropping(HeldState::SamplerState));
                    }
                    let memo: SamplerMemo =
                        take_json(&mut attachments, state.memo).map_err(|e| {
                            e.with_field("field", Detail::text("sampler_state"))
                                .dropping(HeldState::SamplerState)
                        })?;
                    crate::materialize::check_memo(&memo)
                        .map_err(|e| e.dropping(HeldState::SamplerState))?;
                    Some(memo)
                }
            };
            // The call's option values, as the host attached them: read with
            // the wire's bounded reader, so text nested past its depth is
            // refused by name, never read recursively (decision E49).
            let values = match request.values {
                None => None,
                Some(id) => {
                    let Some(Attachment::Bytes(bytes)) = attachments.remove(&id) else {
                        return Err(internal("the call's values are not the bytes named"));
                    };
                    Some(sdm_wire::json::read(&bytes).map_err(|e| {
                        CoreError::new(
                            ErrorKind::InvalidRequest,
                            ErrorCode::InvalidRequest,
                            "the call's option values are not JSON this core reads",
                        )
                        .with_field("field", Detail::text("values"))
                        .with_field("category", Detail::text(e.category))
                    })?)
                }
            };
            let source = crate::materialize::fingerprint(&request.provider, &description)?;
            let access = root.data_access(&request.provider)?;
            let inputs = crate::materialize::Inputs {
                memo: memo.filter(|memo| memo.source == source),
                source,
                description,
                query,
                indices,
                anchor_time: request.anchor_time,
                seed: request.seed,
                unseeded: request.unseeded,
                anchor_entity: request.anchor_entity,
                values,
                options: request.options,
                most_parts: request.most_parts,
                facts,
                zones: Arc::clone(&zones),
                #[cfg(feature = "test-ops")]
                labels_only,
            };
            Ok(Run::with_data(access, move |asker| {
                crate::materialize::run(asker, inputs)
            }))
        }
        _ => Err(internal("a materialize run of another operation")),
    }
}
