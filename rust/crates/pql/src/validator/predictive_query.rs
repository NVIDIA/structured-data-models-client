// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Turns a parsed query into a validated one: every schema-dependent check
//! and every inferred type and join (`validator/predictive_query_validator.py`).

use crate::ast::{self, Kind, NodeRef};
use crate::codes;
use crate::profiles::{Capabilities, Mode};
use crate::query::{ParsedQuery, ValidatedQuery};
use crate::response::{Span, ValidationError, ValidationResponse, ValidationWarning};
use crate::schema::{ColumnKey, SchemaDefinition};
use crate::types::{AggregationType, ProblemType, TaskType};

use super::join::{JoinFailure, JoinValidator};
use super::model::ModelValidator;
use super::problem_type;
use super::time_range::TimeRangeValidator;
use super::type_validator::TypeValidator;
use super::{col_name, get_rhs_lp_table, table_name};

pub struct PredictiveQueryValidator<'a> {
    pub schema: &'a SchemaDefinition,
    pub allow_array_targets: bool,
    pub allow_timestamp_arrays: bool,
    pub mode: Mode,
    pub can: Capabilities,
}

/// Validation stopped where Python raises rather than reports.
#[derive(Debug)]
pub struct Raised(pub String);

impl From<JoinFailure> for Raised {
    fn from(failure: JoinFailure) -> Self {
        Raised(failure.0)
    }
}

fn loc_start(node: &NodeRef) -> String {
    ast::get_location(node).message_start()
}

fn span(node: &NodeRef) -> Option<Span> {
    Span::at(ast::get_location(node))
}

impl<'a> PredictiveQueryValidator<'a> {
    pub fn new(schema: &'a SchemaDefinition, mode: Mode, can: Capabilities) -> Self {
        PredictiveQueryValidator {
            schema,
            allow_array_targets: false,
            allow_timestamp_arrays: true,
            mode,
            can,
        }
    }

    /// Infers and checks types, windows, joins and what the model needs, in
    /// that order, stopping at the first stage that finds an error.
    pub fn validate_predictive_query(
        &self,
        mut query: ParsedQuery,
    ) -> Result<(Option<ValidatedQuery>, ValidationResponse), Raised> {
        let model = ModelValidator {
            schema: self.schema,
            mode: self.mode,
            can: self.can.clone(),
        };
        model.update_location_interval(&query);
        let mut response = ValidationResponse::default()
            .merge(self.validate_columns(&query))
            .merge(self.validate_wildcard(&query));
        if !response.ok() {
            return Ok((None, response));
        }
        let types = TypeValidator {
            schema: self.schema,
            allow_array_targets: self.allow_array_targets,
            allow_timestamp_arrays: self.allow_timestamp_arrays,
        };
        types.infer_dtypes(&query);
        types.infer_stypes(&query);
        response = response.merge(types.validate_dtypes(&query));
        if !response.ok() {
            return Ok((None, response));
        }
        response = response.merge(types.validate_stypes(&query));
        if !response.ok() {
            return Ok((None, response));
        }
        response = response.merge(problem_type::validate(&query, self.mode));
        if !response.ok() {
            return Ok((None, response));
        }
        let times = TimeRangeValidator {
            schema: self.schema,
            mode: self.mode,
        };
        response = response.merge(times.validate_time_ranges(&query));
        if !response.ok() {
            return Ok((None, response));
        }
        response = response.merge(self.validate_target_filters(&query));
        if !response.ok() {
            return Ok((None, response));
        }
        let mut joins = JoinValidator::new(self.schema);
        response = response.merge(joins.infer_and_validate_joins(&mut query)?);
        if !response.ok() {
            return Ok((None, response));
        }
        response = response.merge(self.validate_list_distinct(&query));
        if !response.ok() {
            return Ok((None, response));
        }
        response = response.merge(model.validate(&query));
        if !response.ok() {
            return Ok((None, response));
        }
        response = response.merge(self.target_not_entity_pkey(&query));
        if !response.ok() {
            return Ok((None, response));
        }
        Ok((
            Some(ValidatedQuery {
                query,
                validation_response: response.clone(),
            }),
            response,
        ))
    }

    /// Every table and column the query names is in the schema, and the
    /// entity is its table's primary key.
    pub fn validate_columns(&self, query: &ParsedQuery) -> ValidationResponse {
        let mut response = ValidationResponse::default();
        let mut unknown_tables = Vec::new();
        let mut unknown_columns = Vec::new();
        for (fqn, location) in query.all_query_columns_with_locations() {
            let (column, table) = (col_name(&fqn), table_name(&fqn));
            let Some(definition) = self.schema.table(&table) else {
                unknown_tables.push((table, location));
                continue;
            };
            if column != "*" && !definition.cols.iter().any(|c| c.name == column) {
                unknown_columns.push((table, column, location));
            }
        }
        for (table, location) in unknown_tables {
            response.errors.push(ValidationError::new(
                "Unknown table name",
                format!(
                    "{}: Table '{table}' does not exist in the schema. Please be mindful of case \
                     sensitivity and ensure that the name is spelled correctly.",
                    location.message_start()
                ),
                codes::SCHEMA_UNKNOWN_TABLE,
                Span::at(location),
            ));
        }
        for (table, column, location) in unknown_columns {
            response.errors.push(ValidationError::new(
                "Unknown column name",
                format!(
                    "{}: Column '{column}' does not exist in table '{table}'. Please be mindful \
                     of case sensitivity and ensure that the name is spelled correctly.",
                    location.message_start()
                ),
                codes::SCHEMA_UNKNOWN_COLUMN,
                Span::at(location),
            ));
        }
        let entity_column = query.entity_column();
        let (entity_table, entity_pkey) = (table_name(&entity_column), col_name(&entity_column));
        if response.ok()
            && self
                .schema
                .table(&entity_table)
                .and_then(|t| t.pkey.clone())
                != Some(entity_pkey)
        {
            let location = ast::get_location(&query.entity_column_obj());
            response.errors.push(ValidationError::new(
                "Invalid entity",
                format!(
                    "{}: Entity '{entity_column}' is not a primary key. The column used in 'FOR \
                     EACH' has to be a primary key.",
                    location.message_start()
                ),
                codes::SCHEMA_ENTITY_NOT_PRIMARY_KEY,
                Span::at(location),
            ));
        }
        response
    }

    /// A wildcard column only inside `COUNT`.
    pub fn validate_wildcard(&self, query: &ParsedQuery) -> ValidationResponse {
        let mut response = validate_wildcard_col(&query.entity_ast);
        response = response.merge(validate_wildcard_col(&query.target_ast));
        if let Some(whatif) = &query.whatif_ast {
            response = response.merge(validate_wildcard_col(whatif));
        }
        response
    }

    /// Warns of a target filter that reads only the entity's table.
    pub fn validate_target_filters(&self, query: &ParsedQuery) -> ValidationResponse {
        let mut response = ValidationResponse::default();
        let entity_table = table_name(&query.entity_column());
        let mut filters = Vec::new();
        find_filters(&query.target_ast, &mut filters);
        let mut entity_filters = Vec::new();
        for filter in &filters {
            let Kind::Filter { condition, .. } = &filter.borrow().kind else {
                continue;
            };
            let tables: Vec<String> = ast::all_query_columns(condition)
                .iter()
                .map(|c| table_name(c))
                .collect();
            if !tables.is_empty() && tables.iter().all(|t| *t == entity_table) {
                entity_filters.push(ast::to_string(condition));
            }
        }
        if !entity_filters.is_empty() {
            let mut distinct: Vec<String> = Vec::new();
            for f in &entity_filters {
                if !distinct.contains(f) {
                    distinct.push(f.clone());
                }
            }
            let listed: Vec<String> = entity_filters
                .iter()
                .map(|f| crate::pyrepr::str_repr(f))
                .collect();
            let location = ast::get_location(&query.target_ast);
            response.warnings.push(ValidationWarning::new(
                "Entity filter in the target definition",
                format!(
                    "{}: WHERE conditions [{}] in the target aggregation only contains references \
                     to entity table {entity_table}. It can likely be moved into the entity \
                     definition as follows: \"FOR EACH {} WHERE {}\". Leaving it in the target \
                     definition may result in unnecessary slowdowns or even incorrect labels.",
                    location.message_start(),
                    listed.join(", "),
                    query.entity_column(),
                    distinct.join(" AND ")
                ),
                codes::TARGET_FILTER_ONLY_ENTITY_COLUMNS,
                Span::at(location),
            ));
        }
        response
    }

    /// The checks particular to a `LIST_DISTINCT` target: link prediction
    /// and multilabel tasks.
    pub fn validate_list_distinct(&self, query: &ParsedQuery) -> ValidationResponse {
        let mut response = ValidationResponse::default();
        let mut target = query.target_ast.clone();
        if ast::is_join(&target)
            && let Some(rhs) = ast::target(&target)
        {
            target = rhs;
        }
        let Some(aggr) = ast::aggr(&target) else {
            return response;
        };
        let target_fqn = ast::get_target_column_name(&target);
        if col_name(&target_fqn) == "*" {
            return response;
        }
        let (fqn_table, fqn_col) = (table_name(&target_fqn), col_name(&target_fqn));
        if aggr == AggregationType::ListDistinct
            && query.problem_type == Some(ProblemType::Classify)
            && ast::is_static(&target)
        {
            response.errors.push(ValidationError::new(
                "Unsupported syntax",
                format!(
                    "{}: The combination of LIST_DISTINCT and CLASSIFY is not supported right \
                     now when there is no time range in LIST_DISTINCT.",
                    loc_start(&target)
                ),
                codes::TARGET_LIST_DISTINCT_CLASSIFY_STATIC,
                span(&target),
            ));
        }
        let table = self.schema.table_or_empty(&fqn_table);
        if table.pkey.as_deref() == Some(fqn_col.as_str()) && aggr == AggregationType::ListDistinct
        {
            response.errors.push(ValidationError::new(
                "Aggregation not supported on primary key columns",
                format!(
                    "{}: LIST_DISTINCT target aggregations are not supported onprimary key \
                     columns like {target_fqn}.",
                    loc_start(&target)
                ),
                codes::TARGET_AGGREGATION_ON_PRIMARY_KEY,
                span(&target),
            ));
            return response;
        }
        if !self
            .schema
            .all_keys()
            .contains(&&ColumnKey::new(&fqn_table, &fqn_col))
        {
            return response;
        }
        if matches!(aggr, AggregationType::First | AggregationType::Last) {
            response.errors.push(ValidationError::new(
                "Aggregation not supported on foreign key columns",
                format!(
                    "{}: LAST and FIRST aggregations are not supported onforeign key columns like \
                     {target_fqn} yet. Currently, only LIST_DISTINCT is supported.",
                    loc_start(&target)
                ),
                codes::TARGET_AGGREGATION_ON_FOREIGN_KEY,
                span(&target),
            ));
        }
        if aggr != AggregationType::ListDistinct {
            return response;
        }
        let rhs_key = match &query.target_ast.borrow().kind {
            Kind::Join { rhs_key, .. } => rhs_key.clone(),
            _ => String::new(),
        };
        if rhs_key == target_fqn {
            let mut valid_entities = Vec::new();
            let target_key = ColumnKey::new(&fqn_table, &fqn_col);
            for group in &self.schema.col_groups {
                if group.columns.contains(&target_key) {
                    continue;
                }
                if !group.columns.iter().any(|c| c.table_name == fqn_table) {
                    continue;
                }
                for col in &group.columns {
                    if self
                        .schema
                        .table(&col.table_name)
                        .and_then(|t| t.pkey.as_ref())
                        != Some(&col.col_name)
                    {
                        continue;
                    }
                    valid_entities.push(format!("FOR EACH {}.{}", col.table_name, col.col_name));
                }
            }
            let mut message = format!(
                "{}: Target {target_fqn} creates a loop by pointing to the entity table. A foreign \
                 key used in the target expression must not point to the entity table to define \
                 a valid predictive problem.",
                loc_start(&target)
            );
            if !valid_entities.is_empty() {
                message.push_str(&format!(
                    " To fix this, you can change the entity to one of the following: {}.",
                    valid_entities.join(" ")
                ));
            }
            response.errors.push(ValidationError::new(
                "Nothing to predict",
                message,
                codes::TARGET_EMPTY,
                None,
            ));
        }
        if query.problem_type == Some(ProblemType::Classify) {
            return response;
        }
        let filter_condition = match &ast::target(&target).map(|t| t.borrow().kind_condition()) {
            Some(Some(condition)) => Some(condition.clone()),
            _ => None,
        };
        if let Some(condition) = filter_condition {
            let entity_table = table_name(&query.entity_column());
            for (fqn, location) in ast::all_query_columns_with_locations(&condition) {
                if table_name(&fqn) == entity_table {
                    response.errors.push(ValidationError::new(
                        "Invalid target filter",
                        format!(
                            "{}: Entity table {entity_table} appears inside of the target filter \
                             {}. References to the entity table are only supported in the entity \
                             definition.",
                            location.message_start(),
                            ast::to_string(&condition)
                        ),
                        codes::TARGET_FILTER_REFERENCES_ENTITY,
                        None,
                    ));
                    break;
                }
            }
        }
        let target_col_name = ast::get_target_column_name(&target);
        let target_table = table_name(&target_col_name);
        if query.target_timeframe().is_none() {
            let location_msg = loc_start(&query.target_ast);
            if let Some(time_col) = &self.schema.table_or_empty(&target_table).time_col {
                response.errors.push(ValidationError::new(
                    "Time range required for LIST_DISTINCT with time column",
                    format!(
                        "{location_msg}: When using LIST_DISTINCT on a table with a time column \
                         ({time_col}), you must specify a time range. For example, use \
                         LIST_DISTINCT({target_col_name}, 0, 30, days) instead of \
                         LIST_DISTINCT({target_col_name}). Alternatively, you can remove \
                         {time_col} from {target_table} if time is not relevant."
                    ),
                    codes::TARGET_LIST_DISTINCT_NEEDS_TIME_RANGE,
                    None,
                ));
            }
            let time_cols: Vec<String> = self
                .schema
                .tables
                .iter()
                .filter(|(name, _)| *name != target_table)
                .filter_map(|(name, t)| t.time_col.as_ref().map(|c| format!("{name}.{c}")))
                .collect();
            if !time_cols.is_empty() {
                let listed = time_cols.join(" ");
                response.errors.push(ValidationError::new(
                    "Unexpected time columns in tables.",
                    format!(
                        "{location_msg}: Cannot perform LIST_DISTINCT query without temporal \
                         aggregation when the following time columns {listed} are observed in \
                         schema. Since LIST_DISTINCT and the columns do not contain any time \
                         information, this is likely to result in an incorrect data split and \
                         information leakage. To fix this, you can add the time interval in \
                         LIST_DISTINCT command or drop the columns {listed} from the schema."
                    ),
                    codes::TARGET_LIST_DISTINCT_TIME_COLUMNS,
                    None,
                ));
            }
        }
        response
    }

    /// The target does not predict the entity's own primary key.
    pub fn target_not_entity_pkey(&self, query: &ParsedQuery) -> ValidationResponse {
        let mut response = ValidationResponse::default();
        let entity = query.entity_column();
        let mut columns = Vec::new();
        entity_pkey_columns(&query.target_ast, &entity, &mut columns);
        for column in columns {
            response.errors.push(ValidationError::new(
                "Predicting entity primary key column",
                format!(
                    "{}: Predicting the entity primary key column is not allowed.",
                    loc_start(&column)
                ),
                codes::TARGET_IS_ENTITY_KEY,
                span(&column),
            ));
        }
        response
    }

    /// `validate_bp_filter_overrides`: during batch prediction, a link
    /// prediction's target filter may only read the table it ranks.
    pub fn validate_bp_filter_overrides(
        &self,
        query: &ValidatedQuery,
        task_type: TaskType,
    ) -> Result<ValidationResponse, String> {
        let mut response = ValidationResponse::default();
        if !task_type.is_link_pred() {
            return Ok(response);
        }
        let rhs_table = get_rhs_lp_table(query, self.schema)?;
        let mut aggr = query.query.target_ast.clone();
        if ast::is_join(&aggr)
            && let Some(rhs) = ast::target(&aggr)
        {
            aggr = rhs;
        }
        let condition = ast::target(&aggr).and_then(|t| t.borrow().kind_condition());
        if let Some(condition) = condition {
            for (column, location) in ast::all_query_columns_with_locations(&condition) {
                if table_name(&column) != rhs_table {
                    response.errors.push(ValidationError::new(
                        "Invalid target filter",
                        format!(
                            "{}: Table {} appears in the target filter edit {}. During batch \
                             prediction, only conditions that refer to {rhs_table} table can be \
                             added.",
                            location.message_start(),
                            table_name(&column),
                            ast::to_string(&condition)
                        ),
                        codes::REWRITE_TABLE_NOT_AVAILABLE,
                        None,
                    ));
                    break;
                }
            }
        }
        Ok(response)
    }
}

impl ast::Node {
    /// A filter's condition.
    pub fn kind_condition(&self) -> Option<NodeRef> {
        match &self.kind {
            Kind::Filter { condition, .. } => Some(condition.clone()),
            _ => None,
        }
    }
}

/// `_set_count_cols`: marks the column each `COUNT` counts.
pub(crate) fn set_count_cols(node: &NodeRef) {
    for child in ast::children(node) {
        set_count_cols(&child);
    }
    if ast::aggr(node) == Some(AggregationType::Count)
        && let Some(target) = ast::target(node)
    {
        let column = if ast::is_column(&target) {
            Some(target)
        } else {
            ast::target(&target)
        };
        if let Some(column) = column
            && let Kind::Column { count_col, .. } = &mut column.borrow_mut().kind
        {
            *count_col = true;
        }
    }
}

fn validate_wildcard_col(node: &NodeRef) -> ValidationResponse {
    set_count_cols(node);
    let mut response = ValidationResponse::default();
    for child in ast::children(node) {
        response = response.merge(validate_wildcard_col(&child));
    }
    if let Kind::Column { fqn, count_col } = &node.borrow().kind
        && col_name(fqn) == "*"
        && !count_col
    {
        response.errors.push(ValidationError::new(
            "Invalid wildcard column",
            format!(
                "{}: Wildcard column {fqn} appeared in an invalid position. It can only be used \
                 in a `COUNT` aggregation.",
                loc_start(node)
            ),
            codes::SCHEMA_WILDCARD_POSITION,
            span(node),
        ));
    }
    response
}

fn find_filters(node: &NodeRef, out: &mut Vec<NodeRef>) {
    if ast::is_filter(node) {
        out.push(node.clone());
        return;
    }
    for child in ast::children(node) {
        find_filters(&child, out);
    }
}

/// `_target_not_entity_pkey`: the target's columns that are the entity's
/// key, outside conditions' values, filters and aggregations.
fn entity_pkey_columns(node: &NodeRef, entity: &str, out: &mut Vec<NodeRef>) {
    match &node.borrow().kind {
        Kind::Column { fqn, .. } => {
            if fqn == entity {
                out.push(node.clone());
            }
        }
        Kind::Condition { target, .. } => entity_pkey_columns(target, entity, out),
        Kind::Filter { .. } | Kind::Aggregation { .. } => {}
        _ => {
            for child in ast::children(node) {
                entity_pkey_columns(&child, entity, out);
            }
        }
    }
}
