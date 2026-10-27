// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Operation dispatch: a typed operation to the body of its run.

use std::sync::Arc;

use sdm_contract::ops::{
    GivenCandidate, HeldSchema, Operation, OperationResult, QuerySchema, REQUEST_STATE_VERSION,
    RequestState, SAMPLER_STATE_VERSION, SamplerMemo, SchemaEdit, SourceSpec,
};
use sdm_contract::prepare::{DirectPredict, Preparation};
use sdm_contract::run::StartRun;
use sdm_contract::{CONTRACT_VERSION, CoreError, Detail, HeldState, HostFacts, Scalar, Sentence};
use sdm_engine::time::Zones;

use crate::attached::{check_attachments, resolved_attachments, take_json, take_json_if};
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
        Sentence::InvalidMessage,
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
            Sentence::HostClockNotUtc,
            "the host's clock must be a ts scalar in UTC",
        )
        .with_field("field", Detail::text("facts.now")));
    }
    check_attachments(
        &resolved_attachments(&start.operation, &attachments)?,
        &attachments,
    )?;
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
                    crate::pql::validate_warned(&query, &schema, &profile, &facts, &*zones);
                // D4's warning, on the run's warnings channel, given also
                // when the run then ends with an error (U1).
                for warning in warnings {
                    asker.warn(warning);
                }
                Ok(OperationResult::ValidateQuery(validation?))
            })
        }
        Operation::DescribeSource(request) => {
            let source: SourceSpec = take_json(&mut attachments, request.source)?;
            if let sdm_contract::data::ProviderChoice::Frames { manifest } = request.provider {
                let manifest = crate::frames::take_manifest(&mut attachments, manifest)?;
                return Ok(Run::new(move |asker| {
                    crate::frames::run(asker, source, manifest, attachments)
                }));
            }
            let access = root.data_access(&request.provider)?;
            let check_keys = request.check_keys;
            let zones = Arc::clone(&zones);
            Run::with_data(access, move |asker| async move {
                // A semantic model is read through its own definition, which
                // names its base tables: no session read settles them (T3.5).
                if asker.data().is_some_and(|data| data.view().is_some()) {
                    let reader = crate::describe::text_reader(&facts)?;
                    return crate::view::describe_source(asker, source, &reader).await;
                }
                let asker = crate::describe::settled(asker).await?;
                crate::describe::run(asker, source, check_keys, facts, zones).await
            })
        }
        Operation::EditSchema(request) => {
            let held: HeldSchema = take_json(&mut attachments, request.schema)?;
            let edit: SchemaEdit = take_json(&mut attachments, request.edit)?;
            Run::new(move |asker| crate::edit::run(asker, held, edit))
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
        Operation::PrepareRequest(request) => {
            let input: Preparation = take_json(&mut attachments, request.input)?;
            Run::new(move |asker| crate::prepare::run(asker, input, attachments))
        }
        Operation::Predict(request) => {
            let direct: Option<DirectPredict> = take_json_if(&mut attachments, request.direct)?;
            if let Some(input) = direct {
                if request.state.is_some() || request.held.is_some() {
                    return Err(crate::prepare::invalid(
                        "a direct prediction cannot name gathered state",
                    ));
                }
                return Ok(Run::new(move |asker| {
                    crate::prepare::predict(asker, input, request, attachments)
                }));
            }

            // What predict asks: the state `materialize` answered, or a
            // request the core did not materialize, rebuilt from what the
            // host holds of it (T3.26).
            let asked = match (request.state, request.held) {
                (Some(state), None) => from_state(state, &request.request_id, &mut attachments)?,
                (None, Some(held)) => {
                    crate::held::rebuilt(held, &request.request_id, &mut attachments)?
                }
                _ => {
                    return Err(CoreError::protocol(
                        Sentence::InvalidMessage,
                        "a predict names other than one of a request state and a held request",
                    ));
                }
            };
            let inputs = crate::predict::Inputs {
                payload: asked.payload,
                query: asked.query,
                model: request.model,
                request_id: request.request_id,
                outputs: request.outputs,
                default_outputs: request.default_outputs,
                inference_asked: request.inference_asked,
                forecast: asked.forecast,
                ranking: asked.ranking,
                batches: asked.batches,
                endpoint: request.model_endpoint,
            };
            Run::new(move |asker| crate::predict::run(asker, inputs))
        }
        #[cfg(feature = "test-ops")]
        Operation::Test(operation) => {
            crate::test_ops::run(root, operation, &mut attachments, facts, zones)?
        }
    })
}

/// What `state`, the request state `materialize` answered, asks of
/// `predict`, its attachments taken from `attachments`; the first call
/// carries `request_id`. A version other than this build's is refused by
/// name before anything of the state is read (decision H23).
fn from_state(
    request_state: RequestState,
    request_id: &str,
    attachments: &mut Attachments,
) -> Result<crate::predict::Asked, CoreError> {
    if request_state.version != REQUEST_STATE_VERSION {
        return Err(CoreError::new(
            Sentence::StateVersion,
            "the request state is of a version this core does not read",
        )
        .with_field("field", Detail::text("state.version"))
        .with_field(
            "version",
            Detail::Scalar(Scalar::UInt32(request_state.version)),
        )
        .with_field(
            "reads",
            Detail::Scalar(Scalar::UInt32(REQUEST_STATE_VERSION)),
        ));
    }
    let Some(Attachment::Bytes(payload)) = attachments.remove(&request_state.payload) else {
        return Err(internal("the payload is not the bytes named"));
    };
    // The query rides with the state: the constant-context notice names the
    // target from it where the wire calls it TARGET (T1.28b).
    let query: String = take_json(attachments, request_state.query)?;
    // A forecast's later steps, each as the first is (T1.36 fix 21).
    let forecast = match &request_state.forecast {
        None => None,
        Some(steps) => {
            let mut later = Vec::with_capacity(steps.len());
            for step in steps {
                let Some(Attachment::Bytes(payload)) = attachments.remove(&step.payload) else {
                    return Err(internal("the payload is not the bytes named"));
                };
                let query: String = take_json(attachments, step.query)?;
                // Each step's call carries its own id (T1.36 fix 24).
                let Some(request_id) = step.request_id.clone() else {
                    return Err(CoreError::new(
                        Sentence::StateUnread,
                        "a forecast's later step names no request id",
                    )
                    .with_field("field", Detail::text("state.forecast.request_id")));
                };
                later.push((step.step, payload, query, request_id));
            }
            Some(later)
        }
    };
    // A ranking's later questions, each as the first is (T2.8): the first
    // asks after the first candidate given, at position 0.
    let ranking = match &request_state.ranking {
        None => None,
        Some(ranking) => {
            // Each candidate named by its position, the first's the state's
            // own, every later one's past the one before: a state that
            // names them otherwise would give the answers back under other
            // candidates' names.
            let positions = std::iter::once(ranking.candidate)
                .chain(ranking.parts.iter().map(|part| part.candidate));
            if !positions.clone().zip(positions.skip(1)).all(|(a, b)| a < b) {
                return Err(CoreError::new(
                    Sentence::StateUnread,
                    "a ranking's questions do not name its candidates in the order given",
                )
                .with_field("field", Detail::text("state.ranking.candidate")));
            }
            let mut questions = vec![(
                ranking.candidate,
                ranking.value.clone(),
                payload.clone(),
                query.clone(),
                request_id.to_owned(),
            )];
            for part in &ranking.parts {
                let Some(Attachment::Bytes(payload)) = attachments.remove(&part.payload) else {
                    return Err(internal("the payload is not the bytes named"));
                };
                let query: String = take_json(attachments, part.query)?;
                let Some(request_id) = part.request_id.clone() else {
                    return Err(CoreError::new(
                        Sentence::StateUnread,
                        "a ranking's later question names no request id",
                    )
                    .with_field("field", Detail::text("state.ranking.request_id")));
                };
                questions.push((
                    part.candidate,
                    part.value.clone(),
                    payload,
                    query,
                    request_id,
                ));
            }
            Some((questions, ranking.top_k))
        }
    };
    // A request asked in several model calls (`main`'s multi-call
    // `predict`): each later call's payload and the id it carries, which
    // the host draws for each call as for a forecast's step.
    let batches = match &request_state.batches {
        None => None,
        Some(batches) => {
            if request_state.forecast.is_some() {
                return Err(CoreError::new(
                    Sentence::StateUnread,
                    "a forecast's request is one call per step, never batched",
                )
                .with_field("field", Detail::text("state.batches")));
            }
            let mut later = Vec::with_capacity(batches.len());
            for batch in batches {
                let Some(Attachment::Bytes(payload)) = attachments.remove(&batch.payload) else {
                    return Err(internal("the payload is not the bytes named"));
                };
                let Some(request_id) = batch.request_id.clone() else {
                    return Err(CoreError::new(
                        Sentence::StateUnread,
                        "a later call of a request names no request id",
                    )
                    .with_field("field", Detail::text("state.batches.request_id")));
                };
                later.push(crate::predict::LaterCall {
                    payload,
                    request_id,
                });
            }
            Some(later)
        }
    };
    Ok(crate::predict::Asked {
        payload,
        query,
        forecast,
        ranking,
        batches,
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
            // The schema as the host holds it, as plain data (T3.16): read
            // at the run's start, where a column with no facts is read from
            // the source (`crate::discover`).
            let held: HeldSchema = take_json(&mut attachments, request.schema)?;
            let query: String = take_json(&mut attachments, request.query)?;
            let indices = match request.indices {
                Some(id) => match attachments.remove(&id) {
                    Some(Attachment::Arrow(data)) => Some(data),
                    _ => return Err(internal("the indices are not the Arrow data named")),
                },
                None => None,
            };
            let candidates: Option<Vec<GivenCandidate>> =
                take_json_if(&mut attachments, request.candidates)?;
            // The decoder takes a seed of any size and sign; which seeds a
            // run takes is the core's rule (decision H7): not a negative
            // one, and none past the largest the provider declares (RS-7),
            // which the core reads when it plans the sample
            // (`DataAccess::plan`).
            if request.seed.as_ref().is_some_and(|seed| seed.is_negative()) {
                return Err(CoreError::new(
                    Sentence::SeedNegative,
                    "a sampling seed cannot be negative",
                )
                .with_field("field", Detail::text("seed"))
                .with_field("reason", Detail::text("negative"))
                .with_field(
                    "given",
                    Detail::text(crate::materialize::seed_text(request.seed.as_ref())),
                ));
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
                            Sentence::SamplerStateVersion,
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
                            Sentence::OptionValuesUnread,
                            "the call's option values are not JSON this core reads",
                        )
                        .with_field("field", Detail::text("values"))
                        .with_field("category", Detail::text(e.category))
                    })?)
                }
            };
            let frames = match request.provider {
                sdm_contract::data::ProviderChoice::Frames { manifest } => {
                    Some(crate::frame_materialize::take(
                        manifest,
                        request
                            .sampler_state
                            .as_ref()
                            .and_then(|state| state.frame_checkpoint.clone()),
                        &mut attachments,
                    )?)
                }
                _ => None,
            };
            let provider = request.provider.clone();
            let frame_inputs = frames.clone();
            // The run's inputs, once the schema is read (below) and its
            // fingerprint taken: the memo is read only if it was made under
            // this provider and schema (decision E52).
            let inputs =
                move |description, source: String, pending, types| crate::materialize::Inputs {
                    frames: frame_inputs.clone(),
                    memo: memo.filter(|memo| memo.source == source),
                    source,
                    description,
                    pending,
                    types,
                    query,
                    indices,
                    candidates,
                    anchor_time: request.anchor_time,
                    seed: request.seed,
                    unseeded: request.unseeded,
                    anchor_entity: request.anchor_entity,
                    values,
                    options: request.options,
                    most_parts: request.most_parts,
                    most_entities: request.most_entities,
                    most_calls: request.most_calls,
                    facts,
                    zones: Arc::clone(&zones),
                    #[cfg(feature = "test-ops")]
                    labels_only,
                };
            if let Some(frames) = frames {
                let (description, types) = crate::frame_materialize::description(&frames, held)?;
                let source = crate::frame_materialize::lock(&frames).source();
                return Ok(Run::new(move |asker| {
                    let inputs = inputs(description, source, None, types);
                    crate::frame_materialize::lock(&frames).carry(&asker, inputs.memo.as_ref());
                    crate::materialize::run(asker, inputs)
                }));
            }
            let access = root.data_access(&provider)?;
            Ok(Run::with_data(access, move |asker| async move {
                // A source's tables (a SQL provider's) are read as they stand
                // at each run, their location, columns and types, once the
                // request's inputs are checked (`describe::initialize`), and
                // nothing of them is remembered between runs (the owner's
                // plan B, T3.4); a host provider's columns with no facts are
                // read from it then. Until then the schema is read as the
                // caller holds it, a column with no facts by its data type,
                // as an edit reads it.
                let source_tables = asker
                    .data()
                    .is_some_and(crate::compose::DataAccess::reads_a_source);
                let lacking = !sdm_schema::held::lacking_facts(&held).is_empty();
                let read = sdm_schema::held::read(
                    held.clone(),
                    &std::collections::BTreeMap::new(),
                    lacking,
                )?;
                // The schema as the caller holds it first, as the reference's
                // `check_consistent` reads it; then as its engine reads the
                // source (T3.16 review round 1), where its facts are held.
                crate::materialize::meanings_carried(&read.description, Some(&read.types))?;
                sdm_schema::assemble::check_links(&read.description, &read.types)?;
                let description = read.by_the_source();
                let source = crate::materialize::fingerprint(&provider, &description)?;
                // Each table read as the columns its description holds,
                // until its catalog read says which it holds.
                for table in &description.tables {
                    asker.guard_held(
                        &table.source,
                        table
                            .columns
                            .iter()
                            .map(|column| column.name.clone())
                            .collect(),
                    );
                }
                crate::materialize::run(
                    asker,
                    inputs(
                        description,
                        source,
                        (source_tables || lacking).then_some(held),
                        read.types,
                    ),
                )
                .await
            }))
        }
        _ => Err(internal("a materialize run of another operation")),
    }
}
