// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Relational call, step and native-grouped ranking result composition.

use std::collections::{BTreeMap, BTreeSet};

use crate::attached::take_json;
use crate::prepare::{checked_bound, invalid};
use crate::run::{ArrowData, Asker, Attachment, Attachments};
use sdm_contract::ops::{OperationResult, Predicted};
use sdm_contract::prepare::relational_result::*;
use sdm_contract::prepare::{CallPreparation, RelationalRowLineage};
use sdm_contract::{CoreError, CoreWarning, Detail, Scalar, Sentence};
use sdm_request::predict::{self as policy, Entity, Part};
use sdm_request::ranking;
use sdm_wire::json::Value;

type Answer = (Part, Vec<RelationalRowLineage>);

fn call_group(
    input: RelationalCallResult,
    attached: &mut Attachments,
    seen: &mut BTreeSet<String>,
) -> Result<Answer, CoreError> {
    let bound = checked_bound(&input.bound)?;
    if input.known_empty {
        if !input.parts.is_empty() || !input.counted.is_empty() {
            return Err(invalid(
                "known-empty relational result carries model responses",
            ));
        }
        let payload = Value::Object(vec![(
            "task".into(),
            Value::Object(vec![("kind".into(), Value::text(bound.kind))]),
        )]);
        return Ok((policy::empty(bound.model, &payload), Vec::new()));
    }
    if input.parts.is_empty() || input.parts.len() != input.counted.len() {
        return Err(invalid(
            "relational call result has inconsistent counted responses",
        ));
    }
    let starts = policy::call_starts_from_counts(&input.counted)?;
    let mut parts = Vec::with_capacity(input.parts.len());
    let mut lineage = Vec::new();
    for (input, start) in input.parts.iter().zip(starts) {
        if !seen.insert(input.data.scope.call_request_id.clone()) {
            return Err(invalid("relational result repeats a native call dataset"));
        }
        let (part, source) = super::relational_response::layout(input, attached)?;
        parts.push(policy::numbered_from(part, start)?);
        lineage.extend(source);
    }
    Ok((policy::merge(parts)?, lineage))
}

pub(crate) fn validate_ranking(
    asker: &Asker,
    outputs: Option<Vec<String>>,
    native: Option<NativeRankingOutputFacts>,
    selected: Option<ValidatedRelationalRanking>,
    total_parts: usize,
    cold: Option<Vec<u32>>,
    model: Option<String>,
) -> Result<ValidatedRelationalRanking, CoreError> {
    let (outputs, native) = if let Some(selected) = selected {
        if outputs.is_some()
            || native.is_some()
            || selected.part_outputs != crate::ranking::part_outputs()
        {
            return Err(invalid(
                "retained ranking output selection conflicts with fresh names",
            ));
        }
        let checked = match &selected.native {
            Some(facts) => crate::ranking::native_selected_outputs(Some(&selected.outputs), facts)?,
            None => crate::ranking::selected_outputs(Some(&selected.outputs))?,
        };
        if checked != selected.outputs {
            return Err(invalid("retained ranking output selection changed"));
        }
        (selected.outputs, selected.native)
    } else {
        let selected = match &native {
            Some(facts) => crate::ranking::native_selected_outputs(outputs.as_deref(), facts)?,
            None => crate::ranking::selected_outputs(outputs.as_deref())?,
        };
        (selected, native)
    };
    if let Some(cold) = cold {
        if cold.len() > total_parts || cold.iter().collect::<BTreeSet<_>>().len() != cold.len() {
            return Err(invalid(
                "native cold-candidate observations repeat or exceed the question",
            ));
        }
        let model =
            model.ok_or_else(|| invalid("cold-candidate result lacks its resolved model"))?;
        policy::resolve_model(&model)?;
        crate::ranking::selected_candidates(asker, &model, total_parts, &cold)?;
    }
    Ok(ValidatedRelationalRanking {
        outputs,
        part_outputs: crate::ranking::part_outputs(),
        native,
    })
}

pub(crate) fn describe(scope: RelationalPartScope) -> Result<RelationalPartDescription, CoreError> {
    let paid = match &scope {
        RelationalPartScope::Ranking {
            completed, total, ..
        } => {
            if completed.len() > *total as usize
                || completed.iter().collect::<BTreeSet<_>>().len() != completed.len()
            {
                return Err(invalid(
                    "completed ranking parts exceed or repeat the question",
                ));
            }
            (!completed.is_empty()).then_some(completed.len())
        }
        _ => None,
    };
    let (sentence, fields) = match scope {
        RelationalPartScope::Steps {
            step,
            total,
            window_outran,
        } => (
            if window_outran {
                Sentence::PartForecastStepOutran
            } else {
                Sentence::PartForecastStep
            },
            BTreeMap::from([
                ("forecast_step".into(), Detail::Scalar(Scalar::UInt32(step))),
                (
                    "forecast_steps".into(),
                    Detail::Scalar(Scalar::UInt32(total)),
                ),
            ]),
        ),
        RelationalPartScope::Ranking {
            position, total, ..
        } => (
            Sentence::PartRankingCandidate,
            BTreeMap::from([
                (
                    "ranking_candidate".into(),
                    Detail::Scalar(Scalar::UInt32(position)),
                ),
                (
                    "ranking_parts".into(),
                    Detail::Scalar(Scalar::UInt32(total)),
                ),
            ]),
        ),
    };
    let mut fields = fields;
    if let Some(paid) = paid {
        fields.insert(
            "parts_answered".into(),
            Detail::Scalar(Scalar::UInt64(paid as u64)),
        );
    }
    Ok(RelationalPartDescription {
        sentence,
        fields,
        ending: RelationalSentenceEnding {
            terminal_characters: ".!?".to_owned(),
            period: ".".to_owned(),
        },
    })
}

fn probabilities(
    given: &Option<Vec<(String, Scalar)>>,
) -> Result<Option<Vec<(String, f64)>>, CoreError> {
    given
        .as_ref()
        .map(|given| {
            given
                .iter()
                .map(|(key, value)| match value {
                    Scalar::Float64(value) => Ok((key.clone(), *value)),
                    _ => Err(invalid("rank confidence has a non-double probability")),
                })
                .collect()
        })
        .transpose()
}

pub(crate) fn confidence(item: RelationalRankItem) -> Result<RelationalRankConfidence, CoreError> {
    let probabilities = probabilities(&item.probabilities)?;
    let score =
        ranking::confidence(probabilities.as_ref(), item.candidate).map_err(|mut error| {
            error.public_details = Some(Box::default());
            error
        })?;
    Ok(RelationalRankConfidence {
        score: Scalar::Float64(score),
    })
}

fn population(
    answers: &[Answer],
    positions: &[u32],
) -> Result<RelationalRankPopulation, CoreError> {
    if positions.len() != answers.len()
        || positions.iter().collect::<BTreeSet<_>>().len() != positions.len()
    {
        return Err(invalid(
            "ranking candidate positions do not name its child results once",
        ));
    }
    let mut items = Vec::new();
    for (part, ((answer, lineage), position)) in answers.iter().zip(positions).enumerate() {
        if answer.rows.len() != lineage.len() {
            return Err(invalid("rank population lost response lineage"));
        }
        for (row, lineage) in answer.rows.iter().zip(lineage) {
            let unkeyed_id = match (&row.id, lineage.source_row) {
                (Entity::None, None) => None,
                (Entity::Text(value), None) => Some(value.clone()),
                (_, Some(_)) => None,
                _ => return Err(invalid("unkeyed rank answer has a nontext endpoint id")),
            };
            items.push(RelationalRankItem {
                ordinal: items.len(),
                part,
                candidate: *position,
                row_index: row.row_index,
                lineage: lineage.clone(),
                unkeyed_id,
                probabilities: row.probabilities.as_ref().map(|given| {
                    given
                        .iter()
                        .map(|(key, value)| (key.clone(), Scalar::Float64(*value)))
                        .collect()
                }),
            });
        }
    }
    Ok(RelationalRankPopulation { items })
}

pub(crate) fn prepare_ranking(
    source: RelationalRankingSource,
    mut attached: Attachments,
) -> Result<RelationalRankPopulation, CoreError> {
    let mut seen = BTreeSet::new();
    let answers = source
        .children
        .into_iter()
        .map(|child| call_group(child, &mut attached, &mut seen))
        .collect::<Result<Vec<_>, _>>()?;
    population(&answers, &source.positions)
}

fn grouped(
    population: &RelationalRankPopulation,
    facts: RelationalRankGroups,
    names: &[String],
    top_k: Option<u32>,
) -> Result<(Vec<ranking::Row>, Vec<RelationalRowLineage>), CoreError> {
    let items = &population.items;
    if facts.membership.len() != items.len() {
        return Err(invalid(
            "native rank membership has another population length",
        ));
    }
    let mut buckets: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (ordinal, observed) in facts.membership.iter().enumerate() {
        if observed.ordinal != ordinal
            || observed.score_first > ordinal
            || observed.asked_first > ordinal
        {
            return Err(invalid(
                "native rank membership names a future or different source item",
            ));
        }
        if facts.membership[observed.score_first].score_first != observed.score_first
            || facts.membership[observed.asked_first].asked_first != observed.asked_first
        {
            return Err(invalid(
                "native rank membership does not name actual first dictionary values",
            ));
        }
        if items[ordinal].row_index != items[observed.score_first].row_index
            || items[ordinal].row_index != items[observed.asked_first].row_index
        {
            return Err(invalid(
                "native rank dictionary membership changes the builtin row index",
            ));
        }
        buckets
            .entry(observed.score_first)
            .or_default()
            .push(ordinal);
    }
    if buckets.len() != facts.groups.len() {
        return Err(invalid("native rank groups omit a score bucket"));
    }
    let mut rows = Vec::new();
    let mut lineage = Vec::new();
    for ((first, members), group) in buckets.into_iter().zip(facts.groups) {
        if group.members != members
            || members.first() != Some(&first)
            || group.representative >= items.len()
            || facts.membership[group.representative].asked_first != group.representative
            || items[group.representative].row_index != items[first].row_index
        {
            return Err(invalid(
                "native final rank lookups disagree with observed dictionary membership",
            ));
        }
        let representative = &items[group.representative];
        let mut ranked = Vec::new();
        for ordinal in members {
            let item = &items[ordinal];
            let score = confidence(item.clone())?;
            let Scalar::Float64(score) = score.score else {
                return Err(invalid("rank score ceased to be a double"));
            };
            let name = names
                .get(item.part)
                .ok_or_else(|| invalid("rank item names another candidate value"))?;
            ranked.push((item.candidate, name.clone(), score));
        }
        rows.push(ranking::Row {
            row_index: representative.row_index,
            id: representative.lineage.source_row.map_or_else(
                || {
                    representative
                        .unkeyed_id
                        .clone()
                        .map_or(Entity::None, Entity::Text)
                },
                |source| Entity::Integer(i64::try_from(source).unwrap_or(i64::MAX)),
            ),
            ranked,
        });
        lineage.push(representative.lineage.clone());
    }
    ranking::order(&mut rows, top_k);
    Ok((rows, lineage))
}

pub(crate) fn finish(
    asker: &Asker,
    input: RelationalResultInput,
    mut attached: Attachments,
) -> Result<RelationalResultData, CoreError> {
    let bound = checked_bound(&input.bound)?;
    let report = input.report_absent;
    let mut seen = BTreeSet::new();
    let mut answer_details = BTreeMap::new();
    let (predicted, lineage) = match input.aggregation {
        RelationalAggregation::Calls {
            parts,
            counted,
            known_empty,
        } => {
            answer_details.insert("calls_answered".into(), parts.len() as u64);
            answer_details.insert(
                "rows_answered".into(),
                counted
                    .iter()
                    .flatten()
                    .fold(0u64, |n, rows| n.saturating_add(*rows as u64)),
            );
            let group = RelationalCallResult {
                bound: input.bound.clone(),
                parts,
                counted,
                known_empty,
            };
            let (part, lineage) = call_group(group, &mut attached, &mut seen)?;
            if report && bound.named {
                let absent = policy::absent(&part, bound.outputs);
                if !absent.is_empty() {
                    let warning = policy::absent_notice(bound.model, &absent);
                    asker.warn(
                        CoreWarning::new(warning.sentence, warning.message)
                            .with_fields(warning.fields),
                    );
                }
            }
            (
                finish_part(asker, part, bound.outputs.to_vec(), known_empty)?,
                lineage,
            )
        }
        RelationalAggregation::Steps { children, steps } => {
            if children.is_empty()
                || children.len() != steps.len()
                || steps.iter().any(|step| *step == 0)
                || steps.iter().collect::<BTreeSet<_>>().len() != steps.len()
            {
                return Err(invalid("forecast steps do not name its child results once"));
            }
            let outputs = checked_bound(&children[0].bound)?.outputs.to_vec();
            answer_details.insert("parts_answered".into(), children.len() as u64);
            let answers = children
                .into_iter()
                .map(|child| call_group(child, &mut attached, &mut seen))
                .collect::<Result<Vec<_>, _>>()?;
            if report && bound.named {
                let references: Vec<_> = steps
                    .iter()
                    .copied()
                    .zip(answers.iter().map(|answer| &answer.0))
                    .collect();
                if let Some(warning) =
                    policy::steps_unexplained(bound.model, bound.outputs, &references)
                {
                    asker.warn(
                        CoreWarning::new(warning.sentence, warning.message)
                            .with_fields(warning.fields),
                    );
                }
            }
            let mut lineage = Vec::new();
            let parts = steps
                .into_iter()
                .zip(answers)
                .map(|(step, (part, source))| {
                    lineage.extend(source);
                    (step, part)
                })
                .collect();
            (
                finish_part(asker, policy::by_step(parts)?, outputs, false)?,
                lineage,
            )
        }
        RelationalAggregation::Ranking {
            children,
            positions,
            candidates,
            top_k,
            outputs,
            selected,
            unscored_candidates,
            groups,
        } => {
            let outputs = if let Some(selected) = selected {
                if selected.outputs != outputs
                    || selected.part_outputs != crate::ranking::part_outputs()
                {
                    return Err(invalid(
                        "ranking composition changed its retained output selection",
                    ));
                }
                match &selected.native {
                    Some(facts) => crate::ranking::native_selected_outputs(Some(&outputs), facts)?,
                    None => crate::ranking::selected_outputs(Some(&outputs))?,
                }
            } else {
                crate::ranking::selected_outputs(Some(&outputs))?
            };
            if children.is_empty() || candidates.len() != children.len() {
                return Err(invalid(
                    "ranking candidate values do not name its child results",
                ));
            }
            if positions
                .iter()
                .any(|position| unscored_candidates.contains(position))
                || unscored_candidates.iter().collect::<BTreeSet<_>>().len()
                    != unscored_candidates.len()
            {
                return Err(invalid("ranked and unscored candidates overlap or repeat"));
            }
            answer_details.insert("parts_answered".into(), children.len() as u64);
            let answers = children
                .into_iter()
                .map(|child| call_group(child, &mut attached, &mut seen))
                .collect::<Result<Vec<_>, _>>()?;
            let population = population(&answers, &positions)?;
            let (rows, lineage) = grouped(
                &population,
                take_json(&mut attached, groups)?,
                &candidates,
                top_k,
            )?;
            let laid = ranking::table(&rows, answers[0].0.text_ids)?;
            let result =
                asker.attach(Attachment::Arrow(ArrowData::new(laid.schema(), vec![laid])))?;
            let metadata = asker.attach(Attachment::Bytes(b"{}".to_vec()))?;
            (
                Predicted {
                    result,
                    metadata,
                    model: answers[0].0.model.clone(),
                    request_id: answers[0].0.request_id.clone(),
                    outputs,
                    unscored_candidates,
                },
                lineage,
            )
        }
    };
    Ok(RelationalResultData {
        predicted,
        lineage,
        answer_details,
    })
}

fn finish_part(
    asker: &Asker,
    part: Part,
    outputs: Vec<String>,
    known_empty: bool,
) -> Result<Predicted, CoreError> {
    match crate::predict::finish_direct(asker, part, outputs, known_empty.then_some(true))? {
        OperationResult::Predict(predicted) => Ok(predicted),
        _ => Err(invalid(
            "relational result construction returned another operation",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn population() -> RelationalRankPopulation {
        RelationalRankPopulation {
            items: (0..2)
                .map(|ordinal| RelationalRankItem {
                    ordinal,
                    part: ordinal,
                    candidate: ordinal as u32,
                    row_index: 0,
                    lineage: RelationalRowLineage {
                        call_request_id: format!("call-{ordinal}"),
                        answer_ordinal: 0,
                        source_row: Some(0),
                    },
                    unkeyed_id: None,
                    probabilities: Some(vec![("True".into(), Scalar::Float64(0.5))]),
                })
                .collect(),
        }
    }

    #[test]
    fn native_distinct_keys_keep_two_groups_despite_equal_source_ordinals() {
        let facts = RelationalRankGroups {
            membership: vec![
                RelationalRankMembership {
                    ordinal: 0,
                    score_first: 0,
                    asked_first: 0,
                },
                RelationalRankMembership {
                    ordinal: 1,
                    score_first: 1,
                    asked_first: 1,
                },
            ],
            groups: vec![
                RelationalRankGroup {
                    members: vec![0],
                    representative: 0,
                },
                RelationalRankGroup {
                    members: vec![1],
                    representative: 1,
                },
            ],
        };
        let (rows, lineage) =
            grouped(&population(), facts, &["null".into(), "null".into()], None).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(lineage[0].source_row, Some(0));
        assert_eq!(lineage[1].source_row, Some(0));
        assert_eq!(lineage[0].call_request_id, "call-0");
        assert_eq!(lineage[1].call_request_id, "call-1");
    }

    #[test]
    fn native_equal_keys_share_scores_with_stable_ties_and_actual_asked_owner() {
        let facts = RelationalRankGroups {
            membership: vec![
                RelationalRankMembership {
                    ordinal: 0,
                    score_first: 0,
                    asked_first: 0,
                },
                // Independent dict observations may retain distinct owners;
                // the final actual lookup, not score ordinal, names the id.
                RelationalRankMembership {
                    ordinal: 1,
                    score_first: 0,
                    asked_first: 1,
                },
            ],
            groups: vec![RelationalRankGroup {
                members: vec![0, 1],
                representative: 1,
            }],
        };
        let (rows, lineage) = grouped(
            &population(),
            facts,
            &["null".into(), "null".into()],
            Some(1),
        )
        .unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].ranked, vec![(0, "null".into(), 0.5)]);
        assert_eq!(lineage[0].call_request_id, "call-1");
    }

    #[test]
    fn grouping_cannot_change_builtin_row_identity_or_omit_population_members() {
        let mut population = population();
        population.items[1].row_index = 1;
        let facts = RelationalRankGroups {
            membership: vec![
                RelationalRankMembership {
                    ordinal: 0,
                    score_first: 0,
                    asked_first: 0,
                },
                RelationalRankMembership {
                    ordinal: 1,
                    score_first: 0,
                    asked_first: 0,
                },
            ],
            groups: vec![RelationalRankGroup {
                members: vec![0, 1],
                representative: 0,
            }],
        };
        assert!(grouped(&population, facts, &["null".into(), "null".into()], None).is_err());
    }
}
