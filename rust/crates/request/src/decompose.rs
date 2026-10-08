// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `sdm_client/decompose.py`: a ranking asked as one yes-or-no question per
//! candidate, and a forecast as one question per step.

use sdm_schema::client::Schema;
use sdm_schema::pyval::PyVal;

use crate::error::{R, invalid};
use crate::parts::{FORECAST, RANKING};
use crate::query::Resolved;
use crate::request::Decomposition;

/// `MOST_PARTS`.
pub const MOST_PARTS: usize = 200;

/// `_refuse_too_many(many, what)`.
fn refuse_too_many(many: usize, what: &str) -> R<()> {
    if many <= MOST_PARTS {
        return Ok(());
    }
    Err(invalid(format!(
        "{many} {what} means {many} calls, one for each, because the model has no task that \
         answers them together. That is past the {MOST_PARTS} this will do without being told \
         to; ask about fewer, or raise sdm_client.decompose.MOST_PARTS if the wait is what you \
         want."
    )))
}

/// `_for_each(query)`.
fn for_each(query: &Resolved) -> String {
    format!("{} {}", query.for_each, query.entity_text)
}

/// `_how_the_column_travels(schema, table, column)`.
fn how_the_column_travels(schema: Option<&Schema>, table: &str, column: &str) -> Option<String> {
    let described = schema?.tables.iter().find(|t| t.name == table)?;
    Some(described.col(column)?.dtype.clone())
}

/// `_per_candidate(query, candidates, schema)`.
fn per_candidate(
    query: &Resolved,
    candidates: Option<&[PyVal]>,
    schema: Option<&Schema>,
) -> R<Vec<(String, PyVal)>> {
    let aggregation = query
        .final_aggregation
        .clone()
        .ok_or_else(|| invalid("a ranking is asked one candidate at a time, about what an entity reaches in a window, and this one ranks a column instead. Rank a LIST_DISTINCT over a foreign key."))?;
    let ranked = aggregation.target_text.clone();
    if ranked.contains(" WHERE ") {
        return Err(invalid(format!(
            "the rows to rank are chosen by a filter ({}), and each candidate is asked about \
             with a filter of its own, so the two would have to be combined into one question \
             this cannot write. Rank without the filter, or narrow the rows before building the \
             schema.",
            sdm_pql::pyrepr::str_repr(&ranked)
        )));
    }
    let (table, column) = match ranked.split_once('.') {
        Some((t, c)) => (t.to_string(), c.to_string()),
        None => (ranked.clone(), String::new()),
    };
    let Some(candidates) = candidates else {
        return Err(invalid(format!(
            "a ranking is over a set of candidates, and this model has no ranking task to \
             choose them with, so they have to be named: materialize(..., \
             candidates=frame[{}]). One call is made per candidate.",
            sdm_pql::pyrepr::str_repr(&column)
        )));
    };
    let mut unique: Vec<PyVal> = Vec::new();
    for candidate in candidates {
        if !unique.iter().any(|u| crate::result::py_eq(u, candidate)) {
            unique.push(candidate.clone());
        }
    }
    if unique.is_empty() {
        return Err(invalid("candidates is empty, so there is nothing to rank"));
    }
    refuse_too_many(unique.len(), "candidates")?;
    let dtype = how_the_column_travels(schema, &table, &column);
    let start = aggregation.start.unwrap_or(0);
    let (end, unit) = (aggregation.end, aggregation.unit.clone());
    let entity = for_each(query);
    unique
        .into_iter()
        .map(|value| {
            let literal = crate::literal::as_literal(&value, dtype.as_deref())?;
            Ok((
                format!(
                    "PREDICT COUNT({table}.* WHERE {ranked}={literal}, {start}, {end}, {unit}) > 0 \
                     {entity}"
                ),
                value,
            ))
        })
        .collect()
}

/// `_per_step(query)`.
fn per_step(query: &Resolved) -> R<Vec<(String, PyVal)>> {
    let steps = query.num_forecasts.max(1);
    refuse_too_many(steps as usize, "forecast steps")?;
    let aggregation = query
        .final_aggregation
        .clone()
        .ok_or_else(|| invalid("a forecast forecasts an aggregation"))?;
    let start = aggregation.start.unwrap_or(0);
    let span = aggregation.end - start;
    let entity = for_each(query);
    Ok((0..steps)
        .map(|step| {
            (
                format!(
                    "PREDICT {}({}, {}, {}, {}) {entity}",
                    aggregation.aggr,
                    aggregation.target_text,
                    start + step * span,
                    aggregation.end + step * span,
                    aggregation.unit
                ),
                PyVal::Int(i128::from(step + 1)),
            )
        })
        .collect())
}

/// `parts_of(query, text, candidates, options, schema)`.
pub fn parts_of(
    query: &Resolved,
    text: &str,
    candidates: Option<&[PyVal]>,
    options: &[(String, PyVal)],
    schema: Option<&Schema>,
) -> R<Option<Decomposition>> {
    let kind = query.problem_type.clone().unwrap_or_default();
    let has = |name: &str| options.iter().any(|(k, _)| k == name);
    if kind.ends_with("RANK") {
        if has("prediction_statistic") {
            return Err(invalid(
                "prediction_statistic applies only to a regression, and this question is a \
                 ranking, ordered by how likely each candidate is; ask it without \
                 prediction_statistic",
            ));
        }
        let parts = per_candidate(query, candidates, schema)?;
        let top = crate::materialize::candidates_to_rank(query, options)?;
        return Ok(Some(Decomposition {
            kind: RANKING.to_string(),
            parts,
            top,
        }));
    }
    if candidates.is_some() {
        return Err(invalid(format!(
            "candidates are the rows a ranking is over, and {} does not rank anything. Ask it \
             without candidates, or rank with \"RANK TOP k\", which is the question they are for",
            sdm_pql::pyrepr::str_repr(text)
        )));
    }
    if kind.ends_with("FORECAST") {
        if has("top_k") {
            return Err(invalid(
                "top_k is how many candidates a ranking keeps, and a forecast ranks nothing; it \
                 gives back one value per step",
            ));
        }
        return Ok(Some(Decomposition {
            kind: FORECAST.to_string(),
            parts: per_step(query)?,
            top: None,
        }));
    }
    Ok(None)
}
