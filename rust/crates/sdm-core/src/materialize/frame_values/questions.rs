// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Metadata-only query preparation before native frame export.
//! Existing ranking/forecast policy produces the complete bounded questions;
//! the host owns fresh native materializers, never decomposition semantics.

use super::*;

pub(super) fn prepare(
    input: FrameQuestionInputs,
    facts: &HostFacts,
    zones: &dyn sdm_engine::time::Zones,
    attachments: &mut Attachments,
) -> Result<PreparedFrameQuestions, CoreError> {
    let held = take_json(attachments, input.schema)?;
    let read = sdm_schema::held::read(held, &BTreeMap::new(), true)?;
    let mut description = read.description;
    consistent(&mut description, &read.types)?;
    let query: String = take_json(attachments, input.query)?;
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
    let values: Option<Value> = take_json_if(attachments, input.values)?;
    let candidates: Option<Vec<sdm_contract::ops::GivenCandidate>> =
        take_json_if(attachments, input.candidates)?;
    let decomposition = if let Some(plan) = validation.ranking {
        let prepared = crate::ranking::questions(
            &description,
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
        if let Some(plan) = validation.steps {
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
            reserved(&description)?;
            kept_checks(&description)?;
            None
        }
    };
    Ok(PreparedFrameQuestions {
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
