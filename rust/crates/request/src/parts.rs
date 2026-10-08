// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A question asked in parts (`SDMClient._answer_in_parts`): a ranking
//! asked per candidate and ordered by how sure each answer was, or a
//! forecast asked per step and put back in step order.

use sdm_protocol::{Asker, Diagnostic};
use sdm_schema::dt::Host;
use sdm_schema::pyval::PyVal;
use serde_json::json;

use crate::client::{Asked, prepare, report_absent};
use crate::error::{R, Warnings, details, invalid, unsupported};
use crate::request::{Decomposition, Request};
use crate::result::{Result_, by_step, ranked};
use crate::schemas::Tables;

pub const RANKING: &str = "ranking";
pub const FORECAST: &str = "forecast";

/// What a ranking's parts are asked for.
const RANKING_PARTS: [&str; 2] = ["prediction", "probabilities"];
const RANKING_OUTPUTS: [&str; 2] = ["prediction", "rankings"];

const SCORE_THE_UNSCORED: &str = "Gather a wider context with num_context_samples if other \
     entities in the data answer differently, or ask over a table of (entity, candidate) pairs, \
     which scores every candidate from its features.";

/// Whether an error is of today's `StructuredDataError` family.
pub fn is_structured(error: &Diagnostic) -> bool {
    error
        .class
        .as_deref()
        .is_some_and(|c| c.starts_with("sdm_client.errors."))
}

/// `_ended(message)`.
fn ended(message: &str) -> String {
    let text = message.trim_end();
    if text.is_empty() || text.ends_with(['.', '!', '?']) {
        text.to_string()
    } else {
        format!("{text}.")
    }
}

/// `about_the_part(error, rewritten, about)`.
pub fn about_the_part(
    mut error: Diagnostic,
    rewritten: &Decomposition,
    about: &PyVal,
) -> Diagnostic {
    let how_many = rewritten.parts.len();
    let first = ended(&error.message);
    let said = if rewritten.kind == RANKING {
        format!(
            "This is the question asked for candidate {}, one of {how_many} this ranking is \
             asked as.",
            crate::pyops::py_repr(about)
        )
    } else {
        let step = crate::pyops::py_str(about);
        let mut said = format!("This is forecast step {step} of {how_many}.");
        // At step 1 the window is the one written, and there are no fewer
        // steps to ask for; today's offered "fewer than 1 steps".
        if !matches!(about, PyVal::Int(1))
            && error
                .details
                .get("window_outran_the_data")
                .is_some_and(|v| crate::pyops::py_bool(&crate::keys::json_cell(v)))
        {
            said.push_str(&format!(
                " The window named above is that step's rather than the one written; ask for \
                 fewer than {step} steps if the data does not reach that far."
            ));
        }
        said
    };
    // A copy of the error, raised from it (`raise about_the_part(error, ...)
    // from error`).
    let links = error.as_links();
    error.message = format!("{first} {said}");
    error.chain = Box::new(links);
    error
}

/// `_refuse_outputs_a_ranking_cannot_give(outputs)`.
fn refuse_outputs_a_ranking_cannot_give(outputs: Option<&[String]>) -> R<()> {
    if let Some(o) = outputs
        && o.is_empty()
    {
        return Err(invalid("outputs must name at least one field"));
    }
    let unknown: Vec<&String> = outputs
        .unwrap_or_default()
        .iter()
        .filter(|n| !RANKING_OUTPUTS.contains(&n.as_str()))
        .collect();
    if unknown.is_empty() {
        return Ok(());
    }
    let named: Vec<String> = unknown
        .iter()
        .map(|n| sdm_pql::pyrepr::str_repr(n))
        .collect();
    Err(invalid(format!(
        "a ranking gives back {}, and {} {} not among them. The per-candidate questions it is \
         asked as are classifications; ask one of those directly for the fields a \
         classification gives.",
        RANKING_OUTPUTS.join(", "),
        named.join(", "),
        if unknown.len() > 1 { "are" } else { "is" }
    )))
}

/// `_candidates_no_context_teaches(request, model, rewritten)`: the parts
/// whose context holds one class only.
fn candidates_no_context_teaches(
    request: &Request,
    tables: &Tables,
    asked: &Asked,
    rewritten: &Decomposition,
    host: &Host,
    said: &mut Warnings,
) -> R<Vec<usize>> {
    let spec = crate::models::spec_for(&asked.model)?;
    let wanted: Vec<String> = RANKING_PARTS.iter().map(|s| s.to_string()).collect();
    let mut positions = Vec::new();
    for (position, (part, (_, about))) in request
        .parts
        .iter()
        .flatten()
        .zip(&rewritten.parts)
        .enumerate()
    {
        let tried = (|| -> R<()> {
            let schema = part.schema.build(tables, host)?;
            let fields = prepare(part, &schema, spec, Some(&wanted))?;
            match spec.profile {
                crate::models::Profile::Relational => {
                    crate::relational_adapter::prediction_plan(
                        part,
                        &fields,
                        crate::relational_adapter::Sends::Everything,
                        said,
                    )?;
                }
                crate::models::Profile::Tabular => {
                    let mut schema = schema;
                    crate::tabular_adapter::envelope(part, &mut schema, &fields, None, said)?;
                }
            }
            Ok(())
        })();
        if let Err(error) = tried {
            if !is_structured(&error) {
                return Err(error);
            }
            let one = error
                .details
                .get("classes_in_context")
                .and_then(serde_json::Value::as_i64)
                == Some(1);
            if !one {
                return Err(about_the_part(error, rewritten, about));
            }
            positions.push(position);
        }
    }
    if positions.is_empty() {
        return Ok(positions);
    }
    let cold: Vec<PyVal> = positions
        .iter()
        .map(|&p| rewritten.parts[p].1.clone())
        .collect();
    let many = cold.len() > 1;
    let named = cold[..cold.len() - 1]
        .iter()
        .map(crate::pyops::py_repr)
        .collect::<Vec<_>>()
        .join(", ");
    let last = crate::pyops::py_repr(&cold[cold.len() - 1]);
    if cold.len() == rewritten.parts.len() {
        let which = if many {
            format!("the contexts gathered for candidates {named} and {last} each hold")
        } else {
            format!(
                "the context gathered for candidate {} holds",
                crate::pyops::py_repr(&cold[0])
            )
        };
        return Err(unsupported(
            format!(
                "{which} one answer only, and a classification needs at least two classes to \
                 learn from, so there is nothing to rank and nothing was sent. \
                 {SCORE_THE_UNSCORED}"
            ),
            details([
                ("model", json!(spec.model_id)),
                (
                    "candidates",
                    serde_json::Value::Array(cold.iter().map(crate::layout::encode_deep).collect()),
                ),
                ("classes_in_context", json!(1)),
            ]),
        ));
    }
    let which = if many {
        format!("candidates {named} and {last} are")
    } else {
        format!("candidate {} is", crate::pyops::py_repr(&cold[0]))
    };
    let them = if many { "them" } else { "it" };
    said.sdm(format!(
        "{which} left out of this ranking, unscored: the context sampled for {} holds one \
         answer only, nearly always because no entity in it linked to the candidate in the \
         window, and a classification needs two to learn from. No call was sent for {them}, so \
         each row ranks only the rest, and the result's unscored_candidates names {them}. \
         {SCORE_THE_UNSCORED}",
        if many { "each" } else { "it" }
    ));
    Ok(positions)
}

/// `_warn_of_steps_that_carry_no_explanation(answers, rewritten, ...)`.
fn warn_of_steps_that_carry_no_explanation(
    answers: &[Result_],
    rewritten: &Decomposition,
    requested: Option<&[String]>,
    model: &str,
    said: &mut Warnings,
) {
    let Some(requested) = requested.filter(|r| !r.is_empty()) else {
        return;
    };
    if !requested.iter().any(|r| r == "explanation") {
        return;
    }
    let missing: Vec<String> = rewritten
        .parts
        .iter()
        .zip(answers)
        .filter(|(_, a)| a.explanation.is_none())
        .map(|((_, step), _)| crate::pyops::py_str(step))
        .collect();
    if missing.is_empty() || missing.len() == answers.len() {
        return;
    }
    said.sdm(format!(
        "{model} explained {} of {} forecast steps, and step {} carries none, so the explanation \
         this request asked for is not there for every step.",
        answers.len() - missing.len(),
        answers.len(),
        missing.join(", ")
    ));
}

/// `SDMClient._answer_in_parts(request, model, outputs)`.
pub async fn answer_in_parts(
    asker: &Asker,
    request: &Request,
    tables: &Tables,
    asked: &Asked,
    host: &Host,
    said: &mut Warnings,
) -> R<Result_> {
    let rewritten = request
        .decomposition
        .clone()
        .ok_or_else(|| invalid("a request asked in parts says nothing of its parts"))?;
    let ranking = rewritten.kind == RANKING;
    let mut unscored: Vec<usize> = Vec::new();
    if ranking {
        refuse_outputs_a_ranking_cannot_give(asked.outputs.as_deref())?;
        unscored = candidates_no_context_teaches(request, tables, asked, &rewritten, host, said)?;
    }
    let wanted = if ranking {
        Some(
            RANKING_PARTS
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>(),
        )
    } else {
        asked.outputs.clone()
    };
    let part_asked = Asked {
        model: asked.model.clone(),
        outputs: wanted,
    };
    let mut answers = Vec::new();
    for (position, (part, (_, about))) in request
        .parts
        .iter()
        .flatten()
        .zip(&rewritten.parts)
        .enumerate()
    {
        if unscored.contains(&position) {
            continue;
        }
        match crate::client::predict_boxed(asker, part, tables, &part_asked, host, said).await {
            Ok(answer) => answers.push(answer),
            Err(error) if is_structured(&error) => {
                return Err(about_the_part(error, &rewritten, about));
            }
            Err(error) => return Err(error),
        }
    }
    let model_id = crate::models::spec_for(&asked.model)?.model_id;
    if ranking {
        let scored: Vec<PyVal> = rewritten
            .parts
            .iter()
            .enumerate()
            .filter(|(i, _)| !unscored.contains(i))
            .map(|(_, (_, c))| c.clone())
            .collect();
        let mut left = unscored.clone();
        left.sort_unstable();
        let left_out: Vec<PyVal> = left.iter().map(|&i| rewritten.parts[i].1.clone()).collect();
        return ranked(
            &answers,
            &scored,
            rewritten.top,
            asked.outputs.as_deref(),
            left_out,
        );
    }
    let steps: Vec<i64> = rewritten
        .parts
        .iter()
        .map(|(_, s)| match s {
            PyVal::Int(n) => *n as i64,
            _ => 0,
        })
        .collect();
    let merged = by_step(&answers, &steps);
    warn_of_steps_that_carry_no_explanation(
        &answers,
        &rewritten,
        asked.outputs.as_deref(),
        model_id,
        said,
    );
    report_absent(&merged, asked.outputs.as_deref(), model_id, said);
    Ok(merged)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn outran(step: i128) -> String {
        let forecast = Decomposition {
            kind: "forecast".to_string(),
            parts: vec![
                ("PREDICT ...".to_string(), PyVal::Int(1)),
                ("PREDICT ...".to_string(), PyVal::Int(2)),
            ],
            top: None,
        };
        let error = crate::error::structured("INVALID_REQUEST", "The window is too early.")
            .with_detail("window_outran_the_data", json!(true));
        about_the_part(error, &forecast, &PyVal::Int(step)).message
    }

    /// Asking for fewer steps answers a window that outran the data from
    /// step 2 on; at step 1 there are no fewer to ask for (today's said
    /// "ask for fewer than 1 steps").
    #[test]
    fn fewer_steps_are_offered_only_where_there_are_fewer() {
        assert!(
            outran(2).ends_with("ask for fewer than 2 steps if the data does not reach that far.")
        );
        assert_eq!(
            outran(1),
            "The window is too early. This is forecast step 1 of 2."
        );
    }
}
