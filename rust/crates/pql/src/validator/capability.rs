// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Whether the model on the other end can answer a query the language
//! accepts (`validator/capability.py`). A capability only ever refuses.

use std::collections::BTreeSet;

use crate::ast::{self, NodeRef};
use crate::codes;
use crate::profiles::Profile;
use crate::query::ParsedQuery;
use crate::response::{Span, ValidationError, ValidationResponse};
use crate::schema::{ColumnKey, SchemaDefinition};
use crate::types::AggregationType;

use super::{col_name, table_name};

/// What the model cannot answer, asked of the parsed query.
pub fn check_parsed_capabilities(query: &ParsedQuery, profile: &Profile) -> ValidationResponse {
    let mut response = ValidationResponse::default();
    let can = &profile.capabilities;
    if query.target_timeframe().is_some() && !can.temporal {
        let model = &can.model;
        response.errors.push(ValidationError::new(
            "Time range not supported",
            format!(
                "The target carries a time range, and {model} predicts a value as it stands. No \
                 schema makes this answerable by {model}; ask for the column itself, or use a \
                 model that reads relationships."
            ),
            codes::MODEL_NO_TEMPORAL_TARGET,
            None,
        ));
    }
    response
}

/// One `PQL_MODEL_*` error for each capability the resolved query needs
/// and the model lacks.
pub fn check_capabilities(
    query: &ParsedQuery,
    schema: &SchemaDefinition,
    profile: &Profile,
) -> ValidationResponse {
    let can = &profile.capabilities;
    let model = &can.model;
    let mut response = ValidationResponse::default();
    let span = Span::at(ast::get_location(&query.target_ast));
    let tables: BTreeSet<String> = query
        .all_query_columns()
        .iter()
        .map(|c| table_name(c))
        .collect();
    let entity_table = table_name(&query.entity_column());
    if !can.multiple_tables && tables.len() > 1 {
        let others: Vec<&String> = tables.iter().filter(|t| **t != entity_table).collect();
        response.errors.push(ValidationError::new(
            "More than one table",
            format!(
                "This query reads {} as well as '{entity_table}', and {model} predicts from one \
                 table. Move the columns into '{entity_table}', or use a model that reads \
                 relationships.",
                and_list(&others)
            ),
            codes::MODEL_SINGLE_TABLE_ONLY,
            span,
        ));
    } else if !can.relationships && joins(query) {
        response.errors.push(ValidationError::new(
            "Relationship not supported",
            format!("This query follows a foreign key, and {model} does not read relationships."),
            codes::MODEL_NO_RELATIONSHIPS,
            span,
        ));
    }
    let temporal = query.target_timeframe().is_some();
    if temporal && !can.temporal {
        response.errors.push(ValidationError::new(
            "Time range not supported",
            format!("The target carries a time range, and {model} predicts a value as it stands."),
            codes::MODEL_NO_TEMPORAL_TARGET,
            span,
        ));
    }
    if !temporal && !can.static_ {
        response.errors.push(ValidationError::new(
            "Time range required",
            format!(
                "{model} predicts over a time range, and this target carries none. Add one, for \
                 example (0, 30, days)."
            ),
            codes::MODEL_NO_STATIC_TARGET,
            span,
        ));
    }
    let mut target = query.target_ast.clone();
    if ast::is_join(&target)
        && let Some(rhs) = ast::target(&target)
    {
        target = rhs;
    }
    if !can.derived_targets && !ast::is_column(&target) {
        response.errors.push(ValidationError::new(
            "Derived target not supported",
            format!(
                "The target is computed from the table rather than stored in it, and {model} \
                 predicts a column as it stands. Add the derived value as a column of \
                 '{entity_table}' and predict that column."
            ),
            codes::MODEL_NO_DERIVED_TARGET,
            span,
        ));
    }
    if !can.entity_filters && ast::is_filter(&query.entity_ast) {
        response.errors.push(ValidationError::new(
            "Filter not supported",
            format!(
                "This query filters the entities it predicts for, and {model} scores the rows it \
                 is given. Filter the rows before constructing the query."
            ),
            codes::MODEL_NO_ENTITY_FILTER,
            span,
        ));
    }
    if !can.counterfactuals && query.whatif_ast.is_some() {
        response.errors.push(ValidationError::new(
            "ASSUMING not supported",
            format!(
                "This query asks what would happen under an ASSUMING clause, and {model} scores \
                 the rows it is given as they stand. Change the values in those rows instead."
            ),
            codes::MODEL_NO_COUNTERFACTUAL,
            span,
        ));
    }
    if !can.link_prediction && ranks_a_foreign_key(query, schema) {
        response.errors.push(ValidationError::new(
            "Link prediction not supported",
            format!("This query ranks a foreign key, and {model} does not do link prediction."),
            codes::MODEL_CANNOT_RANK_FOREIGN_KEY,
            span,
        ));
    }
    if !can.forecasting && query.num_forecasts.gt(1) {
        response.errors.push(ValidationError::new(
            "Forecasting not supported",
            format!(
                "This query asks for {} timeframes, and {model} does not forecast.",
                query.num_forecasts
            ),
            codes::MODEL_NO_FORECASTING,
            span,
        ));
    }
    response
}

/// `'a'`, `'a' and 'b'`, `'a', 'b' and 'c'`.
fn and_list(names: &[&String]) -> String {
    let quoted: Vec<String> = names.iter().map(|n| format!("'{n}'")).collect();
    match quoted.as_slice() {
        [] => String::new(),
        [one] => one.clone(),
        [rest @ .., last] => format!("{} and {last}", rest.join(", ")),
    }
}

/// Whether resolution followed a foreign key to reach anything.
fn joins(query: &ParsedQuery) -> bool {
    fn walk(node: &NodeRef) -> bool {
        ast::is_join(node) || ast::children(node).iter().any(walk)
    }
    walk(&query.target_ast)
        || walk(&query.entity_ast)
        || query.whatif_ast.as_ref().is_some_and(walk)
}

/// Whether the target is a `LIST_DISTINCT` over a foreign key column.
fn ranks_a_foreign_key(query: &ParsedQuery, schema: &SchemaDefinition) -> bool {
    let mut target = query.target_ast.clone();
    if ast::is_join(&target)
        && let Some(rhs) = ast::target(&target)
    {
        target = rhs;
    }
    if ast::aggr(&target) != Some(AggregationType::ListDistinct) {
        return false;
    }
    let fqn = ast::get_target_column_name(&target);
    let (table, column) = (table_name(&fqn), col_name(&fqn));
    let Some(definition) = schema.table(&table) else {
        return false;
    };
    if definition.pkey.as_deref() == Some(column.as_str()) {
        return false;
    }
    schema
        .all_keys()
        .contains(&&ColumnKey::new(&table, &column))
}
