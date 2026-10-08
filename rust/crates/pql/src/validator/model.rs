// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What a query must satisfy to be asked of a pretrained model
//! (`validator/model_validator.py`).

use crate::ast::{self, Kind, Location, NodeRef};
use crate::codes;
use crate::profiles::{Capabilities, Mode};
use crate::query::ParsedQuery;
use crate::response::{Span, ValidationError, ValidationResponse};
use crate::schema::{ColumnKey, SchemaDefinition};
use crate::types::{AggregationType, Dtype, MemberOp, Op, ProblemType, RelOp, Stype};
use crate::value::Typed;

use super::type_validator::typed;
use super::{col_name, table_name};

pub const FOR: &str = "FOR";
pub const FOR_EACH: &str = "FOR EACH";
pub const MAX_ENTITY_LIST_LENGTH: usize = 1000;
pub const MAX_TOP_K: i64 = 20;

pub struct ModelValidator<'a> {
    pub schema: &'a SchemaDefinition,
    pub mode: Mode,
    pub can: Capabilities,
}

fn loc_start(node: &NodeRef) -> String {
    ast::get_location(node).message_start()
}

fn span(node: &NodeRef) -> Option<Span> {
    Span::at(ast::get_location(node))
}

fn structure(message: String, code: &'static str) -> ValidationError {
    ValidationError::new("Invalid query structure", message, code, None)
}

impl ModelValidator<'_> {
    /// Every reason a pretrained model refuses the query.
    pub fn validate(&self, query: &ParsedQuery) -> ValidationResponse {
        let mut response = ValidationResponse::default();
        if self.mode == Mode::Training {
            if query.for_each == FOR {
                response.errors.push(structure(
                    "\"FOR\" clause is only supported for a pretrained model, use FOR EACH when \
                     training your own model."
                        .to_string(),
                    codes::MODEL_FOR_REQUIRES_PRETRAINED,
                ));
            }
            if query.rfm_entity_ids.is_some() {
                response.errors.push(structure(
                    "Specifying entities is only supported for a pretrained model, not when \
                     training your own model."
                        .to_string(),
                    codes::MODEL_ENTITY_IDS_REQUIRE_PRETRAINED,
                ));
            }
            return response;
        }
        let mut target = query.target_ast.clone();
        if let Some(rhs) = ast::is_join(&target)
            .then(|| ast::target(&target))
            .flatten()
        {
            target = rhs;
        }
        if ast::is_column(&target) && query.problem_type == Some(ProblemType::Classify) {
            response.errors.push(ValidationError::new(
                "Unsupported query structure",
                "Foundation Model queries do not support static multilabel classification. To \
                 make predictions with this query, train a new model.",
                codes::MODEL_NO_STATIC_MULTILABEL,
                None,
            ));
        }
        if ast::aggr(&target) == Some(AggregationType::ListDistinct) {
            let aggregated = ast::get_target_column_name(&target);
            let (_, target_is_fkey) = self.is_key_col(&aggregated);
            if query.problem_type == Some(ProblemType::Classify)
                || (ast::stype(&target) == Some(Stype::Multicategorical) && !target_is_fkey)
            {
                response.errors.push(ValidationError::new(
                    "Unsupported query structure",
                    "Foundation Model queries do not support multilabel tasks. Use \
                     `LIST_DISTINCT` on a foreign key column to make a link prediction query. To \
                     make predictions with this query, train a new model.",
                    codes::MODEL_NO_MULTILABEL,
                    None,
                ));
            } else {
                if ast::is_static(&target) {
                    response.errors.push(ValidationError::new(
                        "Static Link Prediction not supported",
                        "LIST_DISTINCT without a time range is not supported by the foundation \
                         model. Give it a time range such as (0, 30, days), which needs a time \
                         column on the aggregated table. To make predictions with this query as \
                         written, train a new model.",
                        codes::MODEL_NO_STATIC_LINK_PREDICTION,
                        None,
                    ));
                }
                if query.problem_type == Some(ProblemType::Rank)
                    && let Some(top_k) = &query.top_k
                    && top_k.gt(MAX_TOP_K)
                {
                    response.errors.push(ValidationError::new(
                        "Top k is too large",
                        format!(
                            "Top k {top_k} exceeds maximum foundation model supported value of \
                             {MAX_TOP_K}. To list more candidates, train a new model."
                        ),
                        codes::MODEL_TOP_K_TOO_LARGE,
                        None,
                    ));
                }
            }
        }
        let parts: Vec<NodeRef> = std::iter::once(query.entity_ast.clone())
            .chain(std::iter::once(query.target_ast.clone()))
            .chain(query.whatif_ast.clone())
            .collect();
        for part in &parts {
            response = response.merge(self.find_disabled_joins(part));
        }
        for part in &parts {
            response = response.merge(self.find_disabled_ops(part));
        }
        response = response.merge(self.find_disabled_aggrs(&query.entity_ast, false, false));
        response = response.merge(self.find_disabled_aggrs(&query.target_ast, true, false));
        if let Some(whatif) = &query.whatif_ast {
            response = response.merge(self.find_disabled_aggrs(whatif, false, false));
        }
        response = response.merge(self.find_key_cols(&query.entity_ast, false, false, true));
        response = response.merge(self.find_key_cols(&query.target_ast, true, false, false));
        if let Some(whatif) = &query.whatif_ast {
            response = response.merge(self.find_key_cols(whatif, false, false, false));
        }
        if query.for_each == FOR {
            let Some(ids) = &query.rfm_entity_ids else {
                response.errors.push(structure(
                    "When using \"FOR\" in foundation model queries, you must specify one or \
                     multiple entities to make the prediction for (e.g., \"FOR table.column = \
                     <ID>\")"
                        .to_string(),
                    codes::MODEL_FOR_REQUIRES_ENTITY_IDS,
                ));
                return response;
            };
            let message = loc_start(ids);
            let (op, value) = match &ids.borrow().kind {
                Kind::Condition { op, value, .. } => (*op, value.clone()),
                _ => return response,
            };
            let entity_ids = typed(&value);
            if op == Op::Rel(RelOp::Eq) {
                if let Typed::List(_) = entity_ids {
                    let text = entity_ids.py_str().replace('[', "(").replace(']', ")");
                    response.errors.push(structure(
                        format!("{message}: Expected a single entity ID, but got a list: {text}"),
                        codes::MODEL_ENTITY_ID_LIST_NOT_ALLOWED,
                    ));
                    return response;
                }
            } else if op == Op::Member(MemberOp::In) {
                match &entity_ids {
                    Typed::List(items) => {
                        if items.len() > MAX_ENTITY_LIST_LENGTH {
                            let shown =
                                Typed::List(items[..MAX_ENTITY_LIST_LENGTH].to_vec()).py_str();
                            let shown = format!("{}, ... ]", &shown[..shown.len() - 1]);
                            response.errors.push(ValidationError::new(
                                "The entity id list is too long",
                                format!(
                                    "{message}: List of entity IDs {shown} has length {} which \
                                     exceeds maximum allowed length of {MAX_ENTITY_LIST_LENGTH}.",
                                    items.len()
                                ),
                                codes::MODEL_ENTITY_ID_LIST_TOO_LONG,
                                None,
                            ));
                        }
                    }
                    other => response.errors.push(structure(
                        format!(
                            "{message}: Expected a list of entity IDs, but got {}.",
                            other.py_str()
                        ),
                        codes::MODEL_ENTITY_ID_LIST_REQUIRED,
                    )),
                }
            } else {
                response.errors.push(structure(
                    format!(
                        "{message}: Expected entity IDs, with operators IN or =, but got {}. \
                         Valid syntax for specifying entity IDs is \"FOR table.column = <ID>\" or \
                         \"FOR table.column IN (<ID>, <ID>, ...)\".",
                        op.value()
                    ),
                    codes::MODEL_ENTITY_ID_OPERATOR,
                ));
            }
        } else if let Some(ids) = &query.rfm_entity_ids {
            response.errors.push(structure(
                format!(
                    "{}: Entity IDs are not expected when `FOR EACH` syntax is used. Got {}.",
                    loc_start(ids),
                    ast::to_string(ids)
                ),
                codes::MODEL_ENTITY_IDS_WITH_FOR_EACH,
            ));
        }
        response
    }

    fn find_disabled_joins(&self, node: &NodeRef) -> ValidationResponse {
        let mut response = ValidationResponse::default();
        if let Kind::Join {
            rhs_target,
            lhs_key,
            rhs_key,
        } = &node.borrow().kind
            && ast::is_column(rhs_target)
        {
            response.errors.push(ValidationError::new(
                "Unsupported implicit join",
                format!(
                    "{}: Static references to columns from other tables that implicitly contain \
                     a foreign key to primary key connection are not supported in foundation \
                     model queries. Your query implicitly requires a join {lhs_key} -> \
                     {rhs_key}. Please remove the reference to table {} and retry, or train a \
                     new model instead of the foundation model.",
                    loc_start(node),
                    table_name(rhs_key)
                ),
                codes::MODEL_NO_IMPLICIT_JOIN,
                span(node),
            ));
        }
        for child in ast::children(node) {
            response = response.merge(self.find_disabled_joins(&child));
        }
        response
    }

    /// `_is_key_col`: whether a column is a primary key, and whether a
    /// foreign key.
    pub fn is_key_col(&self, fqn: &str) -> (bool, bool) {
        let (table, column) = (table_name(fqn), col_name(fqn));
        let is_pkey = self
            .schema
            .table(&table)
            .is_some_and(|t| t.pkey.as_deref() == Some(column.as_str()));
        let is_fkey = self
            .schema
            .all_keys()
            .contains(&&ColumnKey::new(&table, &column))
            && !is_pkey;
        (is_pkey, is_fkey)
    }

    fn find_disabled_ops(&self, node: &NodeRef) -> ValidationResponse {
        let mut response = ValidationResponse::default();
        let condition = match &node.borrow().kind {
            Kind::Condition {
                target,
                op,
                value,
                input_op,
            } => Some((target.clone(), *op, value.clone(), input_op.clone())),
            _ => None,
        };
        if let Some((target, op, value, input_op)) = condition {
            let string_value = ast::dtype(&value) == Some(ast::NodeDtype::Plain(Dtype::String));
            let unsupported = matches!(
                op,
                Op::Str(_) | Op::Rel(RelOp::Leq | RelOp::Geq | RelOp::Lt | RelOp::Gt)
            );
            if string_value && unsupported {
                let op_rep = input_op.unwrap_or_else(|| op.value().to_string());
                response.errors.push(ValidationError::new(
                    "Unsupported operation",
                    format!(
                        "{}: Operation {op_rep} between strings is not supported in the \
                         foundation model queries.",
                        loc_start(node)
                    ),
                    codes::MODEL_NO_STRING_COMPARISON,
                    span(node),
                ));
            }
            let target_stype = ast::stype(&target);
            let (is_pkey, is_fkey) = self.is_key_col(&ast::fqn(&target));
            if ast::is_column(&target)
                && target_stype == Some(Stype::Id)
                && (is_pkey || is_fkey)
                && !self.can.key_columns_in_filters
            {
                response.errors.push(ValidationError::new(
                    "Unsupported operation",
                    format!(
                        "{}: Operations on primary key and foreign key columns are not supported \
                         in the foundation model queries.",
                        loc_start(node)
                    ),
                    codes::MODEL_NO_KEY_COLUMN_FILTER,
                    span(node),
                ));
            } else if !(matches!(target_stype, Some(Stype::Numerical | Stype::Categorical))
                || (target_stype == Some(Stype::Timestamp)
                    && self.can.timestamp_columns_in_filters)
                || target_stype == Some(Stype::Id))
            {
                response.errors.push(ValidationError::new(
                    "Unsupported operation",
                    format!(
                        "{}: Operations on a {} column are not supported in the foundation model \
                         queries yet.",
                        loc_start(node),
                        target_stype.map_or("None", Stype::value)
                    ),
                    codes::MODEL_UNSUPPORTED_COLUMN_OPERATION,
                    span(node),
                ));
            }
        }
        for child in ast::children(node) {
            response = response.merge(self.find_disabled_ops(&child));
        }
        response
    }

    fn find_disabled_aggrs(
        &self,
        node: &NodeRef,
        mut permit_list_distinct: bool,
        mut nested_aggr: bool,
    ) -> ValidationResponse {
        let mut response = ValidationResponse::default();
        if let Some(aggr) = ast::aggr(node) {
            let target = ast::target(node).expect("an aggregation has a target");
            if matches!(
                aggr,
                AggregationType::First | AggregationType::Last | AggregationType::CountDistinct
            ) {
                response.errors.push(ValidationError::new(
                    "Unsupported aggregation",
                    format!(
                        "{}: Aggregation {aggr} type is not supported in the foundation model \
                         queries.",
                        loc_start(node)
                    ),
                    codes::MODEL_UNSUPPORTED_AGGREGATION,
                    span(node),
                ));
            }
            let numerical = ast::dtype(&target).is_some_and(|d| d.is_numerical());
            if !numerical && aggr != AggregationType::ListDistinct {
                response.errors.push(ValidationError::new(
                    "Unsupported aggregation",
                    format!(
                        "{}: Aggregations of {} columns are not supported in the foundation \
                         model queries.",
                        loc_start(node),
                        ast::stype(&target).map_or("None", Stype::value)
                    ),
                    codes::MODEL_UNSUPPORTED_AGGREGATION_COLUMN,
                    span(node),
                ));
            }
            if aggr == AggregationType::ListDistinct && !permit_list_distinct {
                response.errors.push(ValidationError::new(
                    "Unsupported aggregation",
                    format!(
                        "{}: Aggregation {aggr} is not supported in the foundation model queries \
                         except for foreign key targets.",
                        loc_start(node)
                    ),
                    codes::MODEL_AGGREGATION_NEEDS_FOREIGN_KEY,
                    span(node),
                ));
            }
            if nested_aggr {
                response.errors.push(ValidationError::new(
                    "Unsupported nested aggregation",
                    format!(
                        "{}: Nesting aggregations is not supported in the foundation model \
                         queries.",
                        loc_start(node)
                    ),
                    codes::MODEL_NO_NESTED_AGGREGATION,
                    span(node),
                ));
            }
            nested_aggr = true;
        }
        if ast::is_condition(node) {
            permit_list_distinct = false;
        }
        for child in ast::children(node) {
            response =
                response.merge(self.find_disabled_aggrs(&child, permit_list_distinct, nested_aggr));
        }
        response
    }

    fn find_key_cols(
        &self,
        node: &NodeRef,
        mut permit_list_distinct: bool,
        mut permit_fkey: bool,
        mut permit_pkey: bool,
    ) -> ValidationResponse {
        let mut response = ValidationResponse::default();
        let aggr = ast::aggr(node);
        if aggr == Some(AggregationType::ListDistinct) && permit_list_distinct {
            permit_fkey = true;
        }
        if aggr == Some(AggregationType::Count) {
            permit_fkey = true;
            permit_pkey = true;
        }
        if ast::is_condition(node) {
            permit_list_distinct = false;
            permit_fkey = self.can.key_columns_in_filters;
            permit_pkey = self.can.key_columns_in_filters;
        }
        if ast::is_column(node) && col_name(&ast::fqn(node)) != "*" {
            let (is_pkey, is_fkey) = self.is_key_col(&ast::fqn(node));
            if (is_pkey && !permit_pkey) || (is_fkey && !permit_fkey) {
                response.errors.push(ValidationError::new(
                    "Unsupported column reference",
                    format!(
                        "{}: References to primary keys and foreign keys are not supported in the \
                         foundation model queries except for foreign key targets in COUNT and \
                         LIST_DISTINCT targets.",
                        loc_start(node)
                    ),
                    codes::MODEL_NO_KEY_COLUMN_REFERENCE,
                    span(node),
                ));
            }
        }
        if let Kind::Filter { target, condition } = &node.borrow().kind {
            response =
                response.merge(self.find_key_cols(condition, permit_list_distinct, false, false));
            return response.merge(self.find_key_cols(
                target,
                permit_list_distinct,
                permit_fkey,
                permit_pkey,
            ));
        }
        for child in ast::children(node) {
            response = response.merge(self.find_key_cols(
                &child,
                permit_list_distinct,
                permit_fkey,
                permit_pkey,
            ));
        }
        response
    }

    /// Shifts every location right by a stripped `EXPLAIN` or `EVALUATE`
    /// prefix. A node reached twice is shifted twice, as in Python.
    pub fn update_location_interval(&self, query: &ParsedQuery) {
        let mut offset = 0;
        if query.explain {
            offset += "EXPLAIN ".len();
        }
        if query.evaluate {
            offset += "EVALUATE ".len();
        }
        if offset == 0 {
            return;
        }
        shift(&query.target_ast, offset);
        shift(&query.entity_ast, offset);
        if let Some(ids) = &query.rfm_entity_ids {
            shift(ids, offset);
        }
        if let Some(whatif) = &query.whatif_ast {
            shift(whatif, offset);
        }
    }
}

fn shift(node: &NodeRef, offset: usize) {
    {
        let mut n = node.borrow_mut();
        if let Some(mut location) = n.base.location
            && location.data_available
        {
            if location.start_row == 1 {
                location = Location::new(
                    location.start_row,
                    location.start_col + offset,
                    location.end_row,
                    location.end_col,
                );
                n.base.location = Some(location);
            }
            if location.end_row == 1 {
                location = Location::new(
                    location.start_row,
                    location.start_col,
                    location.end_row,
                    location.end_col + offset,
                );
                n.base.location = Some(location);
            }
        }
    }
    // An array's elements are not its children, so they keep their
    // locations, as in Python.
    for child in ast::children(node) {
        shift(&child, offset);
    }
}
