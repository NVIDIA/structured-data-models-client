// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native gathered preflight over the same request policy as wire inputs.

use super::*;
use sdm_contract::scalar::unicode::UnicodeText;

fn task(kind: Vec<u32>, facts: RelationalTaskFacts) -> Result<RelationalTaskMeaning, CoreError> {
    policy::validation::task_meaning(kind, facts)
}

fn public_fields(mut error: CoreError, names: &[&str]) -> CoreError {
    error.public_details = Some(
        names
            .iter()
            .filter_map(|name| {
                error
                    .fields
                    .get(*name)
                    .cloned()
                    .map(|value| ((*name).to_owned(), value))
            })
            .collect(),
    );
    error
}

fn equality(value: &NativeTargetEquality) -> Result<u64, CoreError> {
    if value
        .eligible_rows
        .iter()
        .any(|row| *row >= value.observed_rows)
        || value
            .eligible_rows
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
    {
        return Err(invalid(
            "native target equality has invalid eligible row ordinals",
        ));
    }
    let mut seen = BTreeSet::new();
    for source in &value.distinct_sources {
        if value.eligible_rows.binary_search(source).is_err() || !seen.insert(*source) {
            return Err(invalid(
                "native target equality has invalid distinct source ordinals",
            ));
        }
    }
    if value.eligible_rows.is_empty() != value.distinct_sources.is_empty() {
        return Err(invalid(
            "native target equality has inconsistent population facts",
        ));
    }
    u64::try_from(value.distinct_sources.len())
        .map_err(|_| invalid("native target equality cardinality exceeds u64"))
}

fn population(value: &RelationalTargetPopulation) -> Result<(), CoreError> {
    if task(value.meaning.kind.clone(), value.meaning.facts.clone())? != value.meaning
        || value.equality.as_ref().map(equality).transpose()? != value.classes
        || value.wire_target_name != policy::validation::wire_target_name()
        || value.meaning.facts.purpose
            != match value.purpose {
                RelationalTargetUse::ClassLimit => RelationalTaskUse::ClassLimit,
                RelationalTargetUse::Variation => RelationalTaskUse::Variation,
            }
    {
        return Err(invalid(
            "native target population disagrees with its observed facts",
        ));
    }
    Ok(())
}

fn target_name(
    observed: RelationalTargetPopulation,
    wire_target: bool,
    query: Vec<u32>,
) -> Result<RelationalConstantTarget, CoreError> {
    population(&observed)?;
    if observed.purpose != RelationalTargetUse::Variation
        || observed.classes != Some(1)
        || observed.meaning.role == RelationalTargetRole::Other
    {
        return Err(invalid(
            "constant target naming requires one reached answered class",
        ));
    }
    let text = UnicodeText::from_points(query.iter().copied())
        .map_err(|_| invalid("native target question contains invalid Unicode points"))?;
    let naming = policy::validation::target_naming(wire_target, &text);
    Ok(RelationalConstantTarget {
        population: observed,
        wire_target,
        query,
        naming,
        question_truth: None,
    })
}

fn question_meaning(
    mut target: RelationalConstantTarget,
    truth: bool,
) -> Result<RelationalConstantTarget, CoreError> {
    if target_name(
        target.population.clone(),
        target.wire_target,
        target.query.clone(),
    )? != target
        || target.naming != RelationalTargetNaming::Question
    {
        return Err(invalid(
            "native question truth requires an unmatched target question",
        ));
    }
    target.question_truth = Some(truth);
    if !truth {
        target.naming = RelationalTargetNaming::Answer;
    }
    Ok(target)
}

pub(super) fn prepare(
    asker: &Asker,
    input: RelationalPreparation,
) -> Result<RelationalPrepared, CoreError> {
    Ok(match input {
        RelationalPreparation::OutputShapes {
            call,
            parts,
            distinct_sources,
        } => {
            checked_bound(&call)?;
            if parts < 2
                || distinct_sources.is_empty()
                || distinct_sources.len() > parts
                || distinct_sources.iter().any(|position| *position >= parts)
            {
                return Err(invalid(
                    "native output shapes do not match their completed call population",
                ));
            }
            if distinct_sources.len() > 1 {
                let mut error = CoreError::new(
                    Sentence::RelationalOutputShapes,
                    "completed calls have different native output shapes",
                );
                error.public_details = Some(BTreeMap::new());
                return Err(error);
            }
            RelationalPrepared::Call(call)
        }
        RelationalPreparation::OutputVocabulary => {
            RelationalPrepared::OutputVocabulary(policy::native_output_vocabulary())
        }
        RelationalPreparation::OutputMeaning { explanation, item } => {
            RelationalPrepared::OutputMeaning(policy::native_output_meaning(explanation, item)?)
        }
        RelationalPreparation::OutputAbsence { columns } => {
            RelationalPrepared::OutputAbsence(policy::native_output_absence(columns)?)
        }
        RelationalPreparation::OutputNotice {
            call,
            population,
            display,
        } => {
            let bound = checked_bound(&call)?;
            let checked = policy::native_output_absence(population.columns.clone())?;
            if checked != population || population.absent.is_empty() {
                return Err(invalid(
                    "native output warning has no matching absent population",
                ));
            }
            UnicodeText::from_points(display.clone())
                .map_err(|_| invalid("native output display contains invalid codepoints"))?;
            asker.warn(
                CoreWarning::new(
                    Sentence::OutputsAbsent,
                    "the answer left out requested outputs",
                )
                .with_field("model", Detail::text(bound.model))
                .with_field(
                    "native_absent_repr_points",
                    Detail::List(
                        display
                            .into_iter()
                            .map(|point| Scalar::UInt64(u64::from(point)))
                            .collect(),
                    ),
                ),
            );
            RelationalPrepared::Call(call)
        }
        RelationalPreparation::Payloads { call, present } => {
            let bound = checked_bound(&call)?;
            policy::validation::payloads(present, bound.model)?;
            RelationalPrepared::Call(call)
        }
        RelationalPreparation::CallFields {
            call,
            meaning,
            statistic_chosen,
        } => {
            let bound = checked_bound(&call)?;
            if task(meaning.kind.clone(), meaning.facts.clone())? != meaning {
                return Err(invalid("native call task meaning disagrees with its value"));
            }
            RelationalPrepared::Fields(policy::validation::call_fields(
                bound.model,
                bound.outputs,
                &meaning,
                statistic_chosen,
            )?)
        }
        RelationalPreparation::Inference {
            call,
            refused,
            statistic_option,
        } => {
            let bound = checked_bound(&call)?;
            if refused.iter().copied().ne(0..refused.len()) {
                return Err(invalid(
                    "native inference refusal has invalid population ordinals",
                ));
            }
            policy::validation::native_inference(refused.len(), statistic_option, bound.model)?;
            RelationalPrepared::Call(call)
        }
        RelationalPreparation::CallCounts { rows } => {
            RelationalPrepared::Starts(policy::call_starts_from_counts(&rows).map_err(|_| {
                let mut error = CoreError::new(
                    Sentence::RelationalRowsUncounted,
                    "a later native call cannot be numbered from unknown prior row counts",
                );
                error.public_details = Some(BTreeMap::new());
                error
            })?)
        }
        RelationalPreparation::ScoringRows { rows } => RelationalPrepared::Empty(
            !rows.is_empty() && rows.iter().all(|count| *count == Some(0)),
        ),
        RelationalPreparation::Context { call, facts } => {
            let bound = checked_bound(&call)?;
            policy::validation::context_empty(facts.mapping && !facts.rows_present, bound.model)
                .map_err(|error| public_fields(error, &["model", "context_rows"]))?;
            RelationalPrepared::Call(call)
        }
        RelationalPreparation::TaskVocabulary => {
            RelationalPrepared::Vocabulary(policy::validation::task_vocabulary())
        }
        RelationalPreparation::TaskMeaning { kind, facts } => {
            RelationalPrepared::Task(task(kind, facts)?)
        }
        RelationalPreparation::Targets {
            call,
            meaning,
            purpose,
            equality: observed,
        } => {
            let bound = checked_bound(&call)?;
            let expected = task(meaning.kind.clone(), meaning.facts.clone())?;
            let task_use = match purpose {
                RelationalTargetUse::ClassLimit => RelationalTaskUse::ClassLimit,
                RelationalTargetUse::Variation => RelationalTaskUse::Variation,
            };
            if expected != meaning || meaning.facts.purpose != task_use {
                return Err(invalid(
                    "native relational task meaning disagrees with its value",
                ));
            }
            let relevant = match purpose {
                RelationalTargetUse::ClassLimit => {
                    meaning.role == RelationalTargetRole::Classification
                }
                RelationalTargetUse::Variation => meaning.role != RelationalTargetRole::Other,
            };
            if !relevant && observed.is_some() {
                return Err(invalid(
                    "native target facts were supplied for an unrelated task meaning",
                ));
            }
            let classes = observed.as_ref().map(equality).transpose()?;
            if purpose == RelationalTargetUse::ClassLimit {
                if let Some(classes) = classes {
                    policy::validation::class_count(
                        usize::try_from(classes)
                            .map_err(|_| invalid("native class count exceeds this host width"))?,
                        bound.model,
                    )
                    .map_err(|error| public_fields(error, &["model", "classes", "limit"]))?;
                }
            }
            RelationalPrepared::Targets(RelationalTargetPopulation {
                meaning,
                purpose,
                equality: observed,
                classes,
                wire_target_name: policy::validation::wire_target_name(),
            })
        }
        RelationalPreparation::TargetName {
            population,
            wire_target,
            query,
        } => RelationalPrepared::Name(target_name(population, wire_target, query)?),
        RelationalPreparation::QuestionMeaning { target, truth } => {
            RelationalPrepared::Name(question_meaning(target, truth)?)
        }
        RelationalPreparation::Constant {
            call,
            target,
            display,
            classification,
        } => {
            let bound = checked_bound(&call)?;
            let mut expected = target_name(
                target.population.clone(),
                target.wire_target,
                target.query.clone(),
            )?;
            if expected.naming == RelationalTargetNaming::Question {
                expected = question_meaning(
                    expected,
                    target.question_truth.ok_or_else(|| {
                        invalid("a constant question lacks its reached native truth")
                    })?,
                )?;
            }
            if expected != target {
                return Err(invalid(
                    "native constant target naming disagrees with its meaning",
                ));
            }
            let display = display
                .map(UnicodeText::from_points)
                .transpose()
                .map_err(|_| invalid("native target display contains invalid Unicode points"))?;
            let notice = policy::validation::native_constant(
                classification,
                &target.naming,
                display.as_ref(),
                bound.model,
            )
            .map_err(|mut error| {
                if error.fields.get("native_relational_constant")
                    == Some(&Detail::Scalar(Scalar::Bool(true)))
                {
                    error.public_details = Some(BTreeMap::from([
                        ("model".to_owned(), Detail::text(bound.model)),
                        ("target".to_owned(), Detail::text("")),
                        (
                            "classes_in_context".to_owned(),
                            Detail::Scalar(Scalar::UInt64(1)),
                        ),
                    ]));
                }
                error
            })?;
            if let Some(notice) = notice {
                asker.warn(
                    CoreWarning::new(notice.sentence, notice.message).with_fields(notice.fields),
                );
            }
            RelationalPrepared::Call(call)
        }
        RelationalPreparation::ExplanationSize { call, rows } => {
            let bound = checked_bound(&call)?;
            if !bound.outputs.iter().any(|output| output == "explanation") {
                return Err(invalid(
                    "native explanation facts supplied for outputs without explanation",
                ));
            }
            let rows = rows.into_iter().try_fold(0_usize, |sum, rows| {
                usize::try_from(rows)
                    .ok()
                    .and_then(|rows| sum.checked_add(rows))
                    .ok_or_else(|| invalid("native explanation row count exceeds this host width"))
            })?;
            policy::validation::explanation_size(rows, bound.model)
                .map_err(|error| public_fields(error, &["model", "scored"]))?;
            RelationalPrepared::Call(call)
        }
        RelationalPreparation::ExplanationMode {
            call,
            modes,
            fast_equal,
            context_rows,
        } => {
            let bound = checked_bound(&call)?;
            if !bound.outputs.iter().any(|output| output == "explanation") {
                return Err(invalid(
                    "native explanation facts supplied for outputs without explanation",
                ));
            }
            for mode in &modes {
                UnicodeText::from_points(mode.iter().copied())
                    .map_err(|_| invalid("native run mode contains invalid Unicode points"))?;
            }
            let larger = if modes.is_empty() {
                if fast_equal.is_some() {
                    return Err(invalid(
                        "empty native run modes cannot have a reached set comparison",
                    ));
                }
                context_rows.into_iter().fold(0_u64, u64::saturating_add)
                    > policy::CONTEXT_ROWS_FAST as u64
            } else {
                if !context_rows.is_empty() {
                    return Err(invalid(
                        "native explanation supplied unused fallback context counts",
                    ));
                }
                !fast_equal
                    .ok_or_else(|| invalid("native run modes lack their reached set comparison"))?
            };
            policy::validation::explanation_mode(larger, bound.model)
                .map_err(|error| public_fields(error, &["model", "context_rows_fast"]))?;
            RelationalPrepared::Call(call)
        }
    })
}

/// A single reached encoding, admitted immediately before its model call.
/// Earlier paid responses remain owned by the caller if a later encoding
/// fails before reaching this operation.
pub(super) async fn call(
    asker: Asker,
    input: RelationalEncodedCall,
    mut attached: Attachments,
) -> Result<OperationResult, CoreError> {
    let bound = checked_bound(&input.call)?;
    if bound.model != policy::RELATIONAL {
        return Err(invalid(
            "gathered native calls require the relational model",
        ));
    }
    let Some(Attachment::Bytes(bytes)) = attached.remove(&input.body) else {
        return Err(invalid("native relational call has no encoded bytes"));
    };
    policy::check_call_bytes(&bytes)?;
    let (value, response) = crate::predict::call_native_response(
        &asker,
        bytes,
        input.model_endpoint.as_ref(),
        bound.model,
    )
    .await?;
    let data = super::relational_response::decode(&asker, value, &response, &input.request_id)?;
    let value = asker
        .attach(Attachment::Bytes(serde_json::to_vec(&data).map_err(
            |_| invalid("native relational response cannot serialize"),
        )?))?;
    Ok(OperationResult::PrepareRequest(PreparedRequest { value }))
}
