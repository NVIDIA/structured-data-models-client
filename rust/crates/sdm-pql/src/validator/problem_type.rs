// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The problem type, `TOP K` and `FORECAST` (`validator/problem_type_validator.py`).

use crate::ast;
use crate::codes;
use crate::profiles::Mode;
use crate::query::ParsedQuery;
use crate::response::{Span, ValidationError, ValidationResponse};
use crate::types::{ProblemType, Stype};

pub const VALID_PROBLEM_TYPES: [&str; 3] = ["CLASSIFY", "RANK", "FORECAST"];
pub const MAX_TIMEFRAMES: i64 = 10_000;

fn only(
    title: impl Into<String>,
    message: impl Into<String>,
    code: &'static str,
    span: Option<Span>,
) -> ValidationResponse {
    ValidationResponse::error(ValidationError::new(title, message, code, span))
}

/// Checks the problem type against the target, which needs its semantic
/// types resolved.
pub fn validate(query: &ParsedQuery, mode: Mode) -> ValidationResponse {
    let problem_type = query.problem_type;
    let target = &query.target_ast;
    let target_stype = ast::stype(target);
    if let Some(top_k) = &query.top_k {
        if top_k.lt(1) {
            return only(
                "Invalid TOP K value",
                format!("TOP K value must be at least 1, got {top_k}."),
                codes::TASK_TOP_K_NOT_POSITIVE,
                None,
            );
        }
        if problem_type != Some(ProblemType::Rank) {
            return only(
                "TOP K requires RANK.",
                "TOP K can only be specified with the RANK problem type. Please rewrite the query \
                 as 'RANK TOP k'.",
                codes::TASK_TOP_K_REQUIRES_RANK,
                None,
            );
        }
    }
    let entity_location = ast::get_location(&query.entity_ast);
    let target_text = ast::to_string(target);
    if target_stype == Some(Stype::Multicategorical)
        && problem_type.is_none()
        && mode == Mode::Training
    {
        return only(
            "Problem type is required for multicategorical targets.",
            format!(
                "{}: Since your target({target_text}) is multicategorical, you must specify a \
                 problem type (CLASSIFY or RANK). Please rewrite the query as: PREDICT \
                 {target_text} CLASSIFY/RANK TOP k",
                entity_location.message_start()
            ),
            codes::TASK_PROBLEM_TYPE_REQUIRED,
            Span::at(entity_location),
        );
    }
    let Some(problem_type) = problem_type else {
        return ValidationResponse::default();
    };
    if problem_type == ProblemType::Forecast {
        return validate_forecast(query);
    }
    if !VALID_PROBLEM_TYPES.contains(&problem_type.value()) {
        return only(
            format!("Problem type '{problem_type}' is not supported. "),
            "Only ['CLASSIFY', 'RANK', 'FORECAST'] are supported problem types In PREDICT.",
            codes::TASK_PROBLEM_TYPE_UNKNOWN,
            None,
        );
    }
    if target_stype != Some(Stype::Multicategorical) {
        return only(
            format!(
                "Problem type '{problem_type}' is only supported for multicategorical targets. "
            ),
            format!(
                "{}: Target '{target_text}' is not multicategorical. Either drop the problem type \
                 {problem_type} OR change the target to a multicategorical column, 'LAST'/'FIRST' \
                 aggregation of a multicategorical column, or a 'LIST_DISTINCT' aggregation.",
                ast::get_location(target).message_start()
            ),
            codes::TASK_PROBLEM_TYPE_NEEDS_MULTICATEGORICAL,
            None,
        );
    }
    if problem_type == ProblemType::Rank && query.top_k.is_none() {
        return only(
            format!("Problem type '{problem_type}' requires TOP K to be specified. "),
            format!(
                "{}: Please rewrite the query with 'TOP' as 'RANK TOP k'.",
                entity_location.message_start()
            ),
            codes::TASK_RANK_REQUIRES_TOP_K,
            Span::at(entity_location),
        );
    }
    ValidationResponse::default()
}

/// The `FORECAST` clause: a count from 1 to 10,000, and a numerical target
/// with a window.
pub fn validate_forecast(query: &ParsedQuery) -> ValidationResponse {
    let n = &query.num_forecasts;
    if n.lt(1) {
        return only(
            "Invalid FORECAST value",
            format!("FORECAST value must be at least 1, got {n}."),
            codes::TASK_FORECAST_NOT_POSITIVE,
            None,
        );
    }
    if n.gt(MAX_TIMEFRAMES) {
        return only(
            "FORECAST value too large",
            format!("FORECAST {n} TIMEFRAMES exceeds the maximum of {MAX_TIMEFRAMES}."),
            codes::TASK_FORECAST_TOO_LARGE,
            None,
        );
    }
    if query.target_timeframe().is_none() {
        return only(
            "FORECAST requires a temporal target",
            "FORECAST can only be used with targets that have a forward-looking time range \
             (e.g., SUM(table.col, 0, 7, days)). Static column targets are not supported with \
             FORECAST.",
            codes::TASK_FORECAST_REQUIRES_TEMPORAL_TARGET,
            None,
        );
    }
    if ast::stype(&query.target_ast) != Some(Stype::Numerical) {
        return only(
            "FORECAST requires a numerical target",
            "FORECAST is only supported for numerical targets.",
            codes::TASK_FORECAST_REQUIRES_NUMERICAL_TARGET,
            None,
        );
    }
    ValidationResponse::default()
}
