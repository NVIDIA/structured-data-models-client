// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Operation dispatch: a typed operation to the body of its run.

use std::sync::Arc;

use sdm_contract::ops::{
    GivenCandidate, HeldSchema, Operation, OperationResult, QuerySchema, REQUEST_STATE_VERSION,
    SAMPLER_STATE_VERSION, SamplerMemo, SchemaEdit, SourceSpec,
};
use sdm_contract::run::StartRun;
use sdm_contract::{CONTRACT_VERSION, CoreError, Detail, HeldState, HostFacts, Scalar, Sentence};
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
            let access = root.data_access(&request.provider)?;
            let check_keys = request.check_keys;
            let zones = Arc::clone(&zones);
            Run::with_data(access, move |asker| async move {
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
        Operation::Predict(request) => {
            // The state's version first, by name, before anything of the
            // state is read (decision H23).
            if request.state.version != REQUEST_STATE_VERSION {
                return Err(CoreError::new(
                    Sentence::StateVersion,
                    "the request state is of a version this core does not read",
                )
                .with_field("field", Detail::text("state.version"))
                .with_field(
                    "version",
                    Detail::Scalar(Scalar::UInt32(request.state.version)),
                )
                .with_field(
                    "reads",
                    Detail::Scalar(Scalar::UInt32(REQUEST_STATE_VERSION)),
                ));
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
            // A ranking's later questions, each as the first is (T2.8): the
            // first asks after the first candidate given, at position 0.
            let ranking = match &request.state.ranking {
                None => None,
                Some(ranking) => {
                    // Each candidate named by its position, the first's the
                    // state's own, every later one's past the one before:
                    // a state that names them otherwise would give the
                    // answers back under other candidates' names.
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
                        Vec::new(),
                        String::new(),
                        String::new(),
                    )];
                    for part in &ranking.parts {
                        let Some(Attachment::Bytes(payload)) = attachments.remove(&part.payload)
                        else {
                            return Err(internal("the payload is not the bytes named"));
                        };
                        let query: String = take_json(&mut attachments, part.query)?;
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
            let ranking = ranking.map(|(mut questions, top_k)| {
                questions[0].2.clone_from(&payload);
                questions[0].3.clone_from(&query);
                questions[0].4.clone_from(&request.request_id);
                (questions, top_k)
            });
            // A request asked in several model calls (`main`'s multi-call
            // `predict`): each later call's payload, the rows it scores and
            // the id it carries, which the host draws for each call as for
            // a forecast's step.
            let batches = match &request.state.batches {
                None => None,
                Some(batches) => {
                    if request.state.forecast.is_some() {
                        return Err(CoreError::new(
                            Sentence::StateUnread,
                            "a forecast's request is one call per step, never batched",
                        )
                        .with_field("field", Detail::text("state.batches")));
                    }
                    let mut later = Vec::with_capacity(batches.len());
                    for batch in batches {
                        let Some(Attachment::Bytes(payload)) = attachments.remove(&batch.payload)
                        else {
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
            let inputs = crate::predict::Inputs {
                payload,
                query,
                model: request.model,
                request_id: request.request_id,
                outputs: request.outputs,
                default_outputs: request.default_outputs,
                inference_asked: request.inference_asked,
                forecast,
                ranking,
                batches,
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
            // Whether the host takes the run's progress steps.
            let progress = request.progress;
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
            let access = root.data_access(&request.provider)?;
            let provider = request.provider;
            // The run's inputs, once the schema is read (below) and its
            // fingerprint taken: the memo is read only if it was made under
            // this provider and schema (decision E52).
            let inputs = move |description, source: String, pending| crate::materialize::Inputs {
                memo: memo.filter(|memo| memo.source == source),
                source,
                description,
                pending,
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
            Ok(Run::with_data(access, move |asker| async move {
                if progress {
                    asker.says_progress();
                }
                // Each table read where this schema read its relation, as the
                // reference keeps each table it read by its source name: under
                // the namespace the schema read it under, where the source's
                // own parts agree with it (`HeldSchema::read_under`); every
                // other table where the source and its session say now.
                let pinned: Vec<(sdm_contract::data::Relation, Vec<String>)> = held
                    .tables
                    .iter()
                    .filter_map(|table| {
                        table.source.single()?;
                        let read = held
                            .read_under
                            .iter()
                            .find(|r| r.relation == table.source)?;
                        let data = asker.data()?;
                        data.keeps(&read.namespace)
                            .then(|| (table.source.clone(), read.namespace.clone()))
                    })
                    .collect();
                // The tables the schema holds no reading of, read under the
                // parts the session gives once the run builds its graph
                // (`describe::initialize`), where the provider leaves parts
                // to the connection.
                let unsettled = asker
                    .data()
                    .is_some_and(crate::compose::DataAccess::unsettled);
                let anew: Vec<sdm_contract::data::Relation> = held
                    .tables
                    .iter()
                    .filter(|table| unsettled && table.source.single().is_some())
                    .filter(|table| !pinned.iter().any(|(relation, _)| *relation == table.source))
                    .map(|table| table.source.clone())
                    .collect();
                for (relation, parts) in pinned {
                    asker.pin(&relation, parts);
                }
                asker.reads_anew(anew);
                // A column with no facts is read from the source only once the
                // request's inputs are checked (`describe::initialize`), as the
                // reference's engine reads its tables after its materializer's
                // checks; until then the schema is read by its columns' data
                // types, as an edit reads it.
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
                if !lacking {
                    crate::materialize::meanings_carried(&description, None)?;
                }
                let source = crate::materialize::fingerprint(&provider, &description)?;
                // Each table read as the columns its description holds.
                for table in &description.tables {
                    asker.guard(
                        &table.source,
                        table
                            .columns
                            .iter()
                            .map(|column| column.name.clone())
                            .collect(),
                    );
                }
                crate::materialize::run(asker, inputs(description, source, lacking.then_some(held)))
                    .await
            }))
        }
        _ => Err(internal("a materialize run of another operation")),
    }
}
