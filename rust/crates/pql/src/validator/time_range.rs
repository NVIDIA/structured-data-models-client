// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Time windows: where they are needed, where allowed, and which way they
//! may look (`validator/time_range_validator.py`).

use crate::ast::{self, Kind, NodeRef};
use crate::codes;
use crate::profiles::Mode;
use crate::query::ParsedQuery;
use crate::response::{Span, ValidationError, ValidationResponse, ValidationWarning};
use crate::schema::{ColumnKey, SchemaDefinition};
use crate::types::{AggregationType, ProblemType, Stype};

use super::type_validator::validate_int64;
use super::{col_name, table_name};

pub const ENTITY: &str = "ENTITY";
pub const TARGET: &str = "TARGET";
pub const ASSUMPTION: &str = "ASSUMPTION";

/// What a part of the query may do with time.
#[derive(Clone, Copy)]
pub struct Allow {
    pub future_looking: bool,
    pub past_looking: bool,
    pub static_: bool,
    pub temporal: bool,
}

impl Default for Allow {
    fn default() -> Self {
        Allow {
            future_looking: true,
            past_looking: true,
            static_: true,
            temporal: true,
        }
    }
}

pub struct TimeRangeValidator<'a> {
    pub schema: &'a SchemaDefinition,
    pub mode: Mode,
}

fn loc_start(node: &NodeRef) -> String {
    ast::get_location(node).message_start()
}

fn span(node: &NodeRef) -> Option<Span> {
    Span::at(ast::get_location(node))
}

impl TimeRangeValidator<'_> {
    /// What to ask instead, when the aggregated table has no time column.
    fn instead_of_aggregating(&self, target_col_fqn: &str) -> String {
        if target_col_fqn.ends_with(".*") {
            return ". There is no static form of this query to fall back on: a wildcard is \
                    only legal inside COUNT, so ask about a column of the entity table instead"
                .to_string();
        }
        if self.mode == Mode::Inference {
            return ", or ask about a column of the entity table instead of an aggregation"
                .to_string();
        }
        format!(
            " or remove the aggregation and turn this into a static prediction problem with \
             \"PREDICT {target_col_fqn}\""
        )
    }

    /// What to do instead of a static aggregation, for this mode.
    fn static_advice(&self) -> &'static str {
        if self.mode == Mode::Inference {
            return "A foundation model has no static aggregation at all: every aggregation it \
                    answers is over a window, so give this one a time range such as (0, 30, \
                    days). That needs a time column on the aggregated table; without one, ask \
                    about a column of the entity instead of an aggregation.";
        }
        "At the moment, static usage aggregations is only supported for LIST_DISTINCT combined \
         with RANK."
    }

    /// `validate_tree`: the windows of a subtree.
    pub fn validate_tree(
        &self,
        node: &NodeRef,
        query_name: &str,
        problem_type: Option<ProblemType>,
        allow: Allow,
        is_label: bool,
    ) -> ValidationResponse {
        let mut response = ValidationResponse::default();
        if let Kind::Filter { target, condition } = &node.borrow().kind {
            response =
                response.merge(self.validate_tree(target, query_name, problem_type, allow, false));
            let condition_allow = Allow {
                past_looking: true,
                static_: true,
                ..allow
            };
            return response.merge(self.validate_tree(
                condition,
                query_name,
                problem_type,
                condition_allow,
                false,
            ));
        }
        let is_join = ast::is_join(node);
        for child in ast::children(node) {
            response = response.merge(self.validate_tree(
                &child,
                query_name,
                problem_type,
                allow,
                is_label && is_join,
            ));
        }
        if ast::is_aggregation(node) {
            response =
                response.merge(self.validate_aggr(node, query_name, problem_type, allow, is_label));
        }
        if ast::is_condition(node) {
            response = response.merge(self.validate_condition(node, query_name, allow));
        }
        response
    }

    fn validate_aggr(
        &self,
        node: &NodeRef,
        query_name: &str,
        problem_type: Option<ProblemType>,
        allow: Allow,
        is_label: bool,
    ) -> ValidationResponse {
        let mut response = ValidationResponse::default();
        let target_col_fqn = ast::get_target_column_name(node);
        let target_table_name = table_name(&target_col_fqn);
        let window = ast::aggr_time_range(node);
        let node_text = ast::to_string(node);
        if !allow.temporal && window.is_some() {
            response.errors.push(ValidationError::new(
                "Temporal aggregation not allowed",
                format!(
                    "{}: {query_name} aggregation {node_text} has a time range but temporal \
                     aggregation is not supported here. This error usually happens when one \
                     tries to use a temporal aggregation in a filter in a query with a static \
                     target that does not contain any temporal aggregations.",
                    loc_start(node)
                ),
                codes::TIME_TEMPORAL_AGGREGATION_NOT_ALLOWED,
                span(node),
            ));
        }
        let table = self.schema.table_or_empty(&target_table_name);
        if table.time_col.is_none() && window.is_some() {
            response.errors.push(ValidationError::new(
                "Table missing a time column",
                format!(
                    "{}: Missing a \"Create date\" column in the table '{target_table_name}'. A \
                     \"Create date\" column is required to use a time range since it defines a \
                     temporal prediction. To fix this, you can update your schema to include a \
                     \"Create date\" column in {target_table_name}{}",
                    loc_start(node),
                    self.instead_of_aggregating(&target_col_fqn)
                ),
                codes::TIME_TABLE_HAS_NO_TIME_COLUMN,
                span(node),
            ));
            return response;
        }
        let Some(window) = window else {
            let aggr = ast::aggr(node);
            let is_target_list_distinct =
                query_name == TARGET && aggr == Some(AggregationType::ListDistinct);
            let column = col_name(&target_col_fqn);
            let (target_is_fkey, target_is_categorical_col, is_ranking, target_col) =
                if column == "*" {
                    (false, false, false, None)
                } else {
                    let mut is_fkey = self
                        .schema
                        .all_keys()
                        .contains(&&ColumnKey::new(&target_table_name, &column));
                    if table.pkey.as_deref() == Some(column.as_str()) {
                        is_fkey = false;
                    }
                    let target_col = table.cols.iter().find(|c| c.name == column).cloned();
                    let categorical = target_col
                        .as_ref()
                        .is_some_and(|c| c.stype == Stype::Categorical);
                    (
                        is_fkey,
                        categorical,
                        problem_type != Some(ProblemType::Classify),
                        target_col,
                    )
                };
            let valid_static_list_distinct =
                is_target_list_distinct && is_label && target_is_fkey && is_ranking;
            if !valid_static_list_distinct {
                if !is_target_list_distinct {
                    response.errors.push(ValidationError::new(
                        "Aggregation missing a time range",
                        format!(
                            "{}: {query_name} aggregation {node_text} is missing a time range. {}",
                            loc_start(node),
                            self.static_advice()
                        ),
                        codes::TIME_AGGREGATION_NEEDS_RANGE,
                        span(node),
                    ));
                } else if !is_ranking {
                    response.errors.push(ValidationError::new(
                        "Aggregation missing a time range",
                        format!(
                            "{}: {query_name} aggregation {node_text} is missing a time range. At \
                             the moment, static usage of LIST_DISTINCT is only possible for RANK \
                             problems.",
                            loc_start(node)
                        ),
                        codes::TIME_LIST_DISTINCT_NEEDS_RANGE_FOR_RANK,
                        span(node),
                    ));
                } else if target_is_categorical_col {
                    response.errors.push(ValidationError::new(
                        "Aggregation missing a time range",
                        format!(
                            "{}: {query_name} aggregation {node_text} is missing a time range. At \
                             the moment, static usage of LIST_DISTINCT is only possible for the \
                             target clause with a foreign key. (Currently using LIST_DISTINCT in \
                             {query_name}). If you want to use LIST_DISTINCT in a filter please \
                             make sure to add a time range.",
                            loc_start(node)
                        ),
                        codes::TIME_LIST_DISTINCT_NEEDS_RANGE_IN_TARGET,
                        span(node),
                    ));
                } else {
                    let stype = target_col.map_or("None", |c| c.stype.value());
                    let fkey = if target_is_fkey { "True" } else { "False" };
                    response.errors.push(ValidationError::new(
                        "Aggregation missing a time range",
                        format!(
                            "{}: {query_name} aggregation {node_text} is missing a time range. At \
                             the moment, static usage of LIST_DISTINCT is only possible for the \
                             target clause with a foreign key or categorical target. (Currently \
                             using LIST_DISTINCT in {query_name}). If you want to use \
                             LIST_DISTINCT in a filter please make sure to add a time range. If \
                             you want to use LIST_DISTINCT with a column, change its semantic type \
                             to cateogrical (currently {stype} (foreign key: {fkey})",
                            loc_start(node)
                        ),
                        codes::TIME_LIST_DISTINCT_NEEDS_RANGE_OR_FOREIGN_KEY,
                        span(node),
                    ));
                }
            }
            return response;
        };
        let (start, end, is_open) = {
            let w = window.borrow();
            (w.start, w.end, w.is_open())
        };
        response = response.merge(validate_int64(&crate::pyint::PyInt::from_i64(end)));
        if let Some(start) = start {
            response = response.merge(validate_int64(&crate::pyint::PyInt::from_i64(start)));
        }
        if end > 0 && !allow.future_looking {
            response.errors.push(ValidationError::new(
                "Future-looking time range not allowed",
                format!(
                    "{}: {node_text} within the {query_name} contains a future-looking time \
                     range. This is not allowed as it would fail during batch prediction due to \
                     unavailable future data. Consider using the ASSUMING clause for future \
                     scenarios instead.",
                    loc_start(node)
                ),
                codes::TIME_FUTURE_NOT_ALLOWED,
                span(node),
            ));
        }
        if (is_open || start.is_some_and(|s| s < 0)) && !allow.past_looking {
            response.errors.push(ValidationError::new(
                "Invalid time range",
                format!(
                    "{}: {query_name} cannot have an aggregation in the past. Both start and end \
                     need to be non negative.",
                    loc_start(node)
                ),
                codes::TIME_PAST_NOT_ALLOWED,
                span(node),
            ));
        }
        if let Some(time_col) = &table.time_col
            && let Kind::Aggregation { time_col: slot, .. } = &mut node.borrow_mut().kind
        {
            *slot = Some(format!("{target_table_name}.{time_col}"));
        }
        response
    }

    fn validate_condition(
        &self,
        node: &NodeRef,
        query_name: &str,
        allow: Allow,
    ) -> ValidationResponse {
        let mut response = ValidationResponse::default();
        if ast::date_offset_range(node).is_none() && !allow.static_ {
            if query_name == TARGET {
                response.errors.push(ValidationError::new(
                    "Mixing static and temporal conditions is not allowed",
                    format!(
                        "{}: Mixing static and temporal targets is not supported.",
                        loc_start(node)
                    ),
                    codes::TIME_MIXED_STATIC_AND_TEMPORAL,
                    span(node),
                ));
            } else {
                response.errors.push(ValidationError::new(
                    "Static condition not allowed",
                    format!(
                        "{}: {query_name} condition {} has no time range but a temporal \
                         aggregation is required.",
                        loc_start(node),
                        ast::to_string(node)
                    ),
                    codes::TIME_STATIC_CONDITION_NOT_ALLOWED,
                    span(node),
                ));
            }
        }
        response
    }

    /// What applies to a query whose target has no window.
    pub fn static_query_validations(&self, query: &ParsedQuery) -> ValidationResponse {
        let mut response = ValidationResponse::default();
        let entity_table = table_name(&query.entity_column());
        if self.schema.table_or_empty(&entity_table).time_col.is_none() {
            let with_time: Vec<String> = query
                .all_query_columns()
                .into_iter()
                .filter(|c| {
                    self.schema
                        .table(&table_name(c))
                        .is_some_and(|t| t.time_col.is_some())
                })
                .collect();
            if !with_time.is_empty() {
                let listed: Vec<String> = with_time
                    .iter()
                    .map(|c| crate::pyrepr::str_repr(c))
                    .collect();
                response.warnings.push(ValidationWarning::new(
                    "Potential data leakage",
                    format!(
                        "{}: Entity table {entity_table} does not have a \"Create date\", but \
                         several other columns in the query do: [{}]. Training/validation/test \
                         data split will be done at random, potentially resulting in data \
                         leakage. To prevent this and remove this message, add a \"Create date\" \
                         column to your entity table, corresponding to the time when the \
                         prediction would have been done.",
                        loc_start(&query.entity_ast),
                        listed.join(", ")
                    ),
                    codes::TIME_POTENTIAL_LEAKAGE,
                    None,
                ));
            }
        }
        if let Some(whatif) = &query.whatif_ast {
            response.errors.push(ValidationError::new(
                "ASSUMING not allowed for static queries",
                format!(
                    "{}: Target {} contains no temporal aggregations. ASSUMING is not allowed for \
                     such queries.",
                    loc_start(whatif),
                    ast::to_string(&query.target_ast)
                ),
                codes::TIME_ASSUMING_REQUIRES_TEMPORAL,
                None,
            ));
        }
        response
    }

    /// `validate_time_ranges`: the entity may not look ahead, and the
    /// target and `ASSUMING` may not look back.
    pub fn validate_time_ranges(&self, query: &ParsedQuery) -> ValidationResponse {
        let mut response = ValidationResponse::default();
        let mut allow_temporal = true;
        let mut allow_static_target = true;
        if query.target_timeframe().is_none() {
            response = response.merge(self.static_query_validations(query));
            allow_temporal = false;
        } else {
            allow_static_target = false;
        }
        response = response.merge(self.validate_tree(
            &query.entity_ast,
            ENTITY,
            query.problem_type,
            Allow {
                future_looking: false,
                temporal: allow_temporal,
                ..Allow::default()
            },
            false,
        ));
        response = response.merge(self.validate_tree(
            &query.target_ast,
            TARGET,
            query.problem_type,
            Allow {
                past_looking: false,
                static_: allow_static_target,
                ..Allow::default()
            },
            true,
        ));
        if let Some(whatif) = &query.whatif_ast {
            response = response.merge(self.validate_tree(
                whatif,
                ASSUMPTION,
                query.problem_type,
                Allow {
                    past_looking: false,
                    static_: false,
                    ..Allow::default()
                },
                false,
            ));
        }
        response
    }
}
