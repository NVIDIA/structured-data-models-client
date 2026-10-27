// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native gathered preflight over the same request policy as wire inputs.

use super::*;
use sdm_contract::scalar::unicode::UnicodeText;

fn task(kind: Vec<u32>) -> Result<RelationalTaskMeaning, CoreError> {
    policy::validation::task_meaning(kind)
}

fn equality(value: &NativeTargetEquality) -> Result<u64, CoreError> {
    if value.eligible_rows.iter().any(|row| *row >= value.observed_rows)
        || value.eligible_rows.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err(invalid("native target equality has invalid eligible row ordinals"));
    }
    let mut seen = BTreeSet::new();
    for source in &value.distinct_sources {
        if value.eligible_rows.binary_search(source).is_err() || !seen.insert(*source) {
            return Err(invalid("native target equality has invalid distinct source ordinals"));
        }
    }
    if value.eligible_rows.is_empty() != value.distinct_sources.is_empty() {
        return Err(invalid("native target equality has inconsistent population facts"));
    }
    u64::try_from(value.distinct_sources.len())
        .map_err(|_| invalid("native target equality cardinality exceeds u64"))
}

fn population(value: &RelationalTargetPopulation) -> Result<(), CoreError> {
    if task(value.meaning.kind.clone())? != value.meaning
        || value.equality.as_ref().map(equality).transpose()? != value.classes
        || value.wire_target_name != policy::validation::wire_target_name()
    {
        return Err(invalid("native target population disagrees with its observed facts"));
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
        return Err(invalid("constant target naming requires one reached answered class"));
    }
    let text = UnicodeText::from_points(query.iter().copied())
        .map_err(|_| invalid("native target question contains invalid Unicode points"))?;
    let naming = policy::validation::target_naming(wire_target, &text);
    Ok(RelationalConstantTarget { population: observed, wire_target, query, naming })
}

pub(super) fn prepare(asker: &Asker, input: RelationalPreparation) -> Result<RelationalPrepared, CoreError> {
    Ok(match input {
        RelationalPreparation::Context { call, facts } => {
            let bound = checked_bound(&call)?;
            policy::validation::context_empty(facts.mapping && !facts.rows_present, bound.model)?;
            RelationalPrepared::Call(call)
        }
        RelationalPreparation::TaskMeaning { kind } => RelationalPrepared::Task(task(kind)?),
        RelationalPreparation::Targets { call, meaning, purpose, equality: observed } => {
            let bound = checked_bound(&call)?;
            let expected = task(meaning.kind.clone())?;
            if expected != meaning {
                return Err(invalid("native relational task meaning disagrees with its value"));
            }
            let relevant = match purpose {
                RelationalTargetUse::ClassLimit => meaning.role == RelationalTargetRole::Classification,
                RelationalTargetUse::Variation => meaning.role != RelationalTargetRole::Other,
            };
            if !relevant && observed.is_some() {
                return Err(invalid("native target facts were supplied for an unrelated task meaning"));
            }
            let classes = observed.as_ref().map(equality).transpose()?;
            if purpose == RelationalTargetUse::ClassLimit {
                if let Some(classes) = classes {
                    policy::validation::class_count(
                        usize::try_from(classes).map_err(|_| invalid("native class count exceeds this host width"))?,
                        bound.model,
                    )?;
                }
            }
            RelationalPrepared::Targets(RelationalTargetPopulation {
                meaning, purpose, equality: observed, classes,
                wire_target_name: policy::validation::wire_target_name(),
            })
        }
        RelationalPreparation::TargetName { population, wire_target, query } => {
            RelationalPrepared::Name(target_name(population, wire_target, query)?)
        }
        RelationalPreparation::Constant { call, target, display } => {
            let bound = checked_bound(&call)?;
            let expected = target_name(target.population.clone(), target.wire_target, target.query.clone())?;
            if expected != target {
                return Err(invalid("native constant target naming disagrees with its meaning"));
            }
            let display = display.map(UnicodeText::from_points).transpose()
                .map_err(|_| invalid("native target display contains invalid Unicode points"))?;
            let notice = policy::validation::native_constant(
                target.population.meaning.role, &target.naming, display.as_ref(), bound.model,
            ).map_err(|mut error| {
                if error.fields.get("native_relational_constant") == Some(&Detail::Scalar(Scalar::Bool(true))) {
                    error.public_details = Some(BTreeMap::from([
                        ("model".to_owned(), Detail::text(bound.model)),
                        ("target".to_owned(), Detail::text("")),
                        ("classes_in_context".to_owned(), Detail::Scalar(Scalar::UInt64(1))),
                    ]));
                }
                error
            })?;
            if let Some(notice) = notice {
                asker.warn(CoreWarning::new(notice.sentence, notice.message).with_fields(notice.fields));
            }
            RelationalPrepared::Call(call)
        }
        RelationalPreparation::ExplanationSize { call, rows } => {
            let bound = checked_bound(&call)?;
            if !bound.outputs.iter().any(|output| output == "explanation") {
                return Err(invalid("native explanation facts supplied for outputs without explanation"));
            }
            let rows = rows.into_iter().try_fold(0_usize, |sum, rows| {
                usize::try_from(rows).ok().and_then(|rows| sum.checked_add(rows))
                    .ok_or_else(|| invalid("native explanation row count exceeds this host width"))
            })?;
            policy::validation::explanation_size(rows, bound.model)?;
            RelationalPrepared::Call(call)
        }
        RelationalPreparation::ExplanationMode { call, modes, fast_equal, context_rows } => {
            let bound = checked_bound(&call)?;
            if !bound.outputs.iter().any(|output| output == "explanation") {
                return Err(invalid("native explanation facts supplied for outputs without explanation"));
            }
            for mode in &modes {
                UnicodeText::from_points(mode.iter().copied())
                    .map_err(|_| invalid("native run mode contains invalid Unicode points"))?;
            }
            let larger = if modes.is_empty() {
                if fast_equal.is_some() {
                    return Err(invalid("empty native run modes cannot have a reached set comparison"));
                }
                context_rows.into_iter().fold(0_u64, u64::saturating_add) > policy::CONTEXT_ROWS_FAST as u64
            } else {
                if !context_rows.is_empty() {
                    return Err(invalid("native explanation supplied unused fallback context counts"));
                }
                !fast_equal.ok_or_else(|| invalid("native run modes lack their reached set comparison"))?
            };
            policy::validation::explanation_mode(larger, bound.model)?;
            RelationalPrepared::Call(call)
        }
    })
}
