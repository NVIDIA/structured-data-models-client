// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Metadata-only query preparation before native frame export.
//! Existing ranking/forecast policy produces the complete bounded questions;
//! the host owns fresh native materializers, never decomposition semantics.

use super::*;
use crate::attached::take_json_if;

struct QuestionData {
    raw_query: String,
    description: SourceDescription,
    types: sdm_schema::assemble::HostTypes,
    validation: crate::pql::Validation,
    facts: QueryFacts,
}

/// Independently useful immutable query meaning, without a host object.
#[derive(Clone)]
pub struct RetainedFrameQuestion {
    data: Arc<QuestionData>,
}

impl std::fmt::Debug for RetainedFrameQuestion {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RetainedFrameQuestion")
            .field("facts", &self.data.facts)
            .finish_non_exhaustive()
    }
}

impl RetainedFrameQuestion {
    pub(super) fn target(
        &self,
        raw_query: &str,
    ) -> Result<(ResolvedTarget, QueryFacts), CoreError> {
        if raw_query != self.data.raw_query
            || self.data.validation.ranking.is_some()
            || self.data.validation.steps.is_some()
        {
            return Err(protocol(
                "frame population query differs from its ordinary resolved question owner",
            ));
        }
        let validation = &self.data.validation;
        let validated = validation
            .query
            .clone()
            .ok_or_else(|| protocol("retained frame question has no validated tree"))?;
        let source_query = validation
            .text
            .clone()
            .ok_or_else(|| protocol("retained frame question has no canonical query"))?;
        Ok((
            ResolvedTarget {
                validated,
                source_query,
                entity_filter: validation.entity_filter.clone(),
            },
            self.data.facts.clone(),
        ))
    }
}

pub(super) struct QuestionMetadata {
    pub(super) schema_id: AttachmentId,
    pub(super) query_id: AttachmentId,
    pub(super) candidates_given: bool,
    pub(super) statistic_given: bool,
    pub(super) top_k_given: bool,
}

pub(super) fn resolve(
    asker: &Asker,
    input: QuestionMetadata,
    facts: &HostFacts,
    zones: &dyn sdm_engine::time::Zones,
    attachments: &mut Attachments,
) -> Result<PreparedFrameQuestion, CoreError> {
    let QuestionMetadata {
        schema_id,
        query_id,
        candidates_given,
        statistic_given,
        top_k_given,
    } = input;
    let held = take_json(attachments, schema_id)?;
    let read = sdm_schema::held::read(held, &BTreeMap::new(), true)?;
    let mut description = read.description;
    consistent(&mut description, &read.types)?;
    let query: String = take_json(attachments, query_id)?;
    let schema = query_schema(&description);
    let profile = Profile::resolve(&ProfileSpec {
        name: ProfileName::Relational,
        model: None,
        capabilities: Default::default(),
    });
    let validation = crate::pql::resolve(&query, &schema, &profile, facts, zones)?;
    let query_facts = validation
        .facts(&schema)
        .ok_or_else(|| unresolved(&query, &validation.diagnostics))?;
    let kind = if validation.ranking.is_some() {
        Some(FrameQuestionKind::Ranking)
    } else if validation.steps.is_some() {
        Some(FrameQuestionKind::Forecast)
    } else {
        None
    };
    if let Some(plan) = validation.ranking.as_ref() {
        crate::ranking::admit_question(plan, statistic_given)?;
    } else {
        if candidates_given {
            return Err(crate::ranking::candidates_on_no_ranking());
        }
        if validation.steps.is_some() {
            super::super::forecast_admission(top_k_given)?;
        }
    }
    let owner = RetainedFrameQuestion {
        data: Arc::new(QuestionData {
            raw_query: query,
            description,
            types: read.types,
            validation,
            facts: query_facts.clone(),
        }),
    };
    Ok(PreparedFrameQuestion {
        query: query_facts,
        kind,
        owner: asker.attach(Attachment::FrameQuestion(owner))?,
    })
}

pub(super) fn prepare(
    asker: &Asker,
    input: FrameQuestionInputs,
    attachments: &mut Attachments,
) -> Result<PreparedFrameQuestions, CoreError> {
    let Some(Attachment::FrameQuestion(owner)) = attachments.remove(&input.question.owner) else {
        return Err(protocol(
            "frame questions lack their actual resolved query owner",
        ));
    };
    let data = &owner.data;
    let kind = if data.validation.ranking.is_some() {
        Some(FrameQuestionKind::Ranking)
    } else if data.validation.steps.is_some() {
        Some(FrameQuestionKind::Forecast)
    } else {
        None
    };
    if input.question.query != data.facts || input.question.kind != kind {
        return Err(protocol(
            "frame question facts no longer match their actual resolved query meaning",
        ));
    }
    let description = &data.description;
    let validation = &data.validation;
    let query_facts = data.facts.clone();
    let raw_values = input.values.and_then(|id| match attachments.get(&id) {
        Some(Attachment::Bytes(bytes)) => Some(bytes.clone()),
        _ => None,
    });
    let values: Option<Value> = take_option_values_if(attachments, input.values)?;
    let candidates: Option<Vec<sdm_contract::ops::GivenCandidate>> =
        take_json_if(attachments, input.candidates)?;
    let decomposition = if let Some(plan) = validation.ranking.clone() {
        let prepared = crate::ranking::questions(
            description,
            candidates.as_deref(),
            values.as_ref(),
            plan,
            input.prediction_rows,
            input.limits.parts,
            input.limits.entities,
        )?;
        Some(FrameDecomposition {
            kind: FrameQuestionKind::Ranking,
            parts: prepared
                .parts
                .into_iter()
                .map(|(position, name, query)| FrameQuestion {
                    query,
                    position,
                    name: Some(name),
                })
                .collect(),
            top_k: prepared.top_k,
            child_options: child_options(input.option_facts)?,
        })
    } else {
        if candidates.is_some() {
            return Err(crate::ranking::candidates_on_no_ranking());
        }
        if let Some(plan) = validation.steps.clone() {
            forecast_questions(
                &plan,
                values.as_ref(),
                input.prediction_rows,
                input.limits.parts,
                input.limits.entities,
            )?;
            // The shared guard checked the host's u32 bound before any
            // per-step strings are allocated.
            let count = u32::try_from(plan.count)
                .map_err(|_| internal("a checked forecast count exceeds u32"))?;
            Some(FrameDecomposition {
                kind: FrameQuestionKind::Forecast,
                parts: (1..=count)
                    .map(|position| FrameQuestion {
                        query: plan.step(u64::from(position)),
                        position,
                        name: None,
                    })
                    .collect(),
                top_k: None,
                child_options: child_options(input.option_facts)?,
            })
        } else {
            check_call(
                values.as_ref(),
                &input.options,
                input.prediction_rows,
                &input.limits,
            )?;
            reserved(description)?;
            // Caller consistency uses the original declared names, before
            // any native cast. The later kept checks retain source semantics.
            meanings_carried(description, Some(&data.types))?;
            kept_checks(description)?;
            // This original client notice is outside the native engine guard
            // and before native export. Decomposition parents stay silent;
            // each ordinary child reaches this same metadata-only boundary.
            if let Some(warning) = super::super::filter_chose_examples(
                validation.entity_filter.as_deref(),
                input.prediction_rows,
            ) {
                asker.warn(warning);
            }
            None
        }
    };
    Ok(PreparedFrameQuestions {
        question: if decomposition.is_none() {
            Some(asker.attach(Attachment::FrameQuestion(owner))?)
        } else {
            None
        },
        call_values: if decomposition.is_none() {
            raw_values
                .map(|bytes| asker.attach(Attachment::Bytes(bytes)))
                .transpose()?
        } else {
            None
        },
        query: query_facts,
        decomposition,
    })
}

fn child_options(options: Vec<FrameQuestionOption>) -> Result<Vec<String>, CoreError> {
    let mut names = BTreeSet::new();
    let mut kept = Vec::with_capacity(options.len());
    for option in options {
        if !names.insert(option.name.clone()) {
            return Err(protocol("frame question option facts name an option twice"));
        }
        if !option.is_none && option.name != "top_k" {
            kept.push(option.name);
        }
    }
    Ok(kept)
}

/// Reuse the existing part wording before the host presents any error.
pub(crate) fn in_part(error: CoreError, scope: Option<&FrameQuestionScope>) -> CoreError {
    let Some(scope) = scope else {
        return error;
    };
    match scope.kind {
        FrameQuestionKind::Forecast => super::in_step(error, scope.position, scope.parts),
        FrameQuestionKind::Ranking => error
            .with_field(
                "ranking_candidate",
                Detail::Scalar(Scalar::UInt32(scope.position)),
            )
            .with_field(
                "ranking_parts",
                Detail::Scalar(Scalar::UInt64(u64::from(scope.parts))),
            )
            .in_part(Sentence::PartRankingCandidate),
    }
}

/// Shared warning policy over finite observed native facts, without native values.
pub(super) fn relationship_overlap(
    observations: Vec<FrameRelationshipEvidence>,
) -> Result<(FrameRelationshipOverlap, Option<CoreWarning>), CoreError> {
    let mut disjoint = Vec::new();
    let mut names = Vec::new();
    let mut first = None;
    for (position, observation) in observations.into_iter().enumerate() {
        let FrameRelationshipEvidence::Distinct {
            relationship,
            child,
            parent,
            intersects,
        } = observation
        else {
            continue;
        };
        if relationship
            .iter()
            .any(|column| column.table.is_empty() || column.column.is_empty())
        {
            return Err(protocol("relationship overlap has an empty endpoint"));
        }
        let selected = match (child, parent, intersects) {
            (0, None, None) | (1.., Some(0), None) => false,
            (1.., Some(1..), Some(intersects)) => !intersects,
            _ => {
                return Err(protocol(
                    "relationship overlap contradicts its reached native counts",
                ));
            }
        };
        if selected {
            let [source, target] = relationship;
            names.push(format!(
                "{}.{} -> {}.{}",
                source.table, source.column, target.table, target.column
            ));
            if first.is_none() {
                first = Some((source, target));
            }
            disjoint.push(position as u64);
        }
    }
    let notice = first.map(|(source, target)| {
        let mut notice = CoreWarning::new(
            if disjoint.len() == 1 {
                Sentence::RelationshipsDisjointOne
            } else {
                Sentence::RelationshipsDisjointSeveral
            },
            "declared relationships match no given rows",
        );
        notice.fields = [
            (
                "positions".into(),
                Detail::List(disjoint.iter().copied().map(Scalar::UInt64).collect()),
            ),
            ("relationships".into(), Detail::text(names.join(", "))),
            ("source_table".into(), Detail::text(source.table)),
            ("target_table".into(), Detail::text(target.table)),
            ("target_column".into(), Detail::text(target.column)),
        ]
        .into();
        notice
    });
    Ok((FrameRelationshipOverlap { disjoint }, notice))
}

#[cfg(test)]
#[path = "overlap_tests.rs"]
mod overlap_tests;
