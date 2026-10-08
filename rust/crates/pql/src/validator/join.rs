// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Joins: the foreign key each reference across tables follows, inserted
//! into the tree as `Join` nodes (`validator/join_validator.py`).

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use crate::ast::{self, Kind, NodeRef};
use crate::codes;
use crate::query::ParsedQuery;
use crate::response::{Span, ValidationError, ValidationResponse};
use crate::schema::{ColumnKey, SchemaDefinition};

use super::{col_name, table_name};

/// A join could not be built: the merge of two windows overflowed, which
/// Python raises rather than reports.
#[derive(Debug)]
pub struct JoinFailure(pub String);

pub struct JoinValidator<'a> {
    pub schema: &'a SchemaDefinition,
    target_filter_example: Option<String>,
    entity_filter_tables: BTreeSet<String>,
    entity_filter_is_row_level: bool,
    allow_temporal_entity_filter: bool,
}

type Joined = Result<(NodeRef, ValidationResponse), JoinFailure>;

fn loc_start(node: &NodeRef) -> String {
    ast::get_location(node).message_start()
}

fn span(node: &NodeRef) -> Option<Span> {
    Span::at(ast::get_location(node))
}

impl<'a> JoinValidator<'a> {
    pub fn new(schema: &'a SchemaDefinition) -> Self {
        JoinValidator {
            schema,
            target_filter_example: None,
            entity_filter_tables: BTreeSet::new(),
            entity_filter_is_row_level: false,
            allow_temporal_entity_filter: false,
        }
    }

    /// Puts a `Join` wherever the tree reaches across a foreign key, and
    /// reports the references no single key resolves.
    pub fn infer_and_validate_joins(
        &mut self,
        query: &mut ParsedQuery,
    ) -> Result<ValidationResponse, JoinFailure> {
        let entity_table = table_name(&query.entity_column());
        let mut entity_filter_condition = None;
        self.entity_filter_tables = BTreeSet::new();
        self.entity_filter_is_row_level = false;
        let entity_condition = match &query.entity_ast.borrow().kind {
            Kind::Filter { condition, .. } => Some(condition.clone()),
            _ => None,
        };
        if let Some(condition) = &entity_condition {
            entity_filter_condition = Some(ast::to_string(condition));
            self.entity_filter_tables = ast::all_query_columns(condition)
                .iter()
                .map(|c| table_name(c))
                .collect();
            self.entity_filter_is_row_level = !contains_aggregation(condition);
        }
        self.allow_temporal_entity_filter = query.target_timeframe().is_some();
        self.target_filter_example = self.get_target_filter_example(query);
        let in_filter = entity_condition.is_some();
        let (entity_ast, entity_response) = self.infer_joins(
            &query.entity_ast,
            &entity_table,
            in_filter,
            entity_filter_condition.clone(),
        )?;
        query.entity_ast = entity_ast;
        let (target_ast, target_response) =
            self.infer_joins(&query.target_ast, &entity_table, false, None)?;
        query.target_ast = target_ast;
        let mut response = entity_response.merge(target_response);
        if let Some(whatif) = query.whatif_ast.clone() {
            let (whatif_ast, whatif_response) =
                self.infer_joins(&whatif, &entity_table, false, None)?;
            query.whatif_ast = Some(whatif_ast);
            response = response.merge(whatif_response);
        }
        Ok(response)
    }

    /// The entity filter moved into the target's aggregation, when that
    /// means the same.
    fn get_target_filter_example(&self, query: &ParsedQuery) -> Option<String> {
        let condition = match &query.entity_ast.borrow().kind {
            Kind::Filter { condition, .. } => condition.clone(),
            _ => return None,
        };
        let target_column = match &query.target_ast.borrow().kind {
            Kind::Aggregation { target, .. } if ast::is_column(target) => ast::fqn(target),
            _ => return None,
        };
        if contains_aggregation(&condition) {
            return None;
        }
        let tables: BTreeSet<String> = ast::all_query_columns(&condition)
            .iter()
            .map(|c| table_name(c))
            .collect();
        if tables != BTreeSet::from([table_name(&target_column)]) {
            return None;
        }
        let filtered = format!("{target_column} WHERE {}", ast::to_string(&condition));
        Some(ast::to_string(&query.target_ast).replacen(&target_column, &filtered, 1))
    }

    /// `_route_between`: the shortest route of foreign keys, child to
    /// parent, from one table to another.
    fn route_between(&self, start: &str, goal: &str) -> Vec<String> {
        let mut points_at: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        for group in &self.schema.col_groups {
            let tables: BTreeSet<String> =
                group.columns.iter().map(|c| c.table_name.clone()).collect();
            let parents: Vec<&String> = tables
                .iter()
                .filter(|t| {
                    self.schema
                        .table(t)
                        .and_then(|d| d.pkey.as_ref())
                        .is_some_and(|pkey| group.columns.contains(&ColumnKey::new(t, pkey)))
                })
                .collect();
            if parents.len() != 1 {
                continue;
            }
            let parent = parents[0].clone();
            for child in tables.iter().filter(|t| **t != parent) {
                points_at
                    .entry(child.clone())
                    .or_default()
                    .insert(parent.clone());
            }
        }
        let mut seen = BTreeSet::from([start.to_string()]);
        let mut queue = VecDeque::from([vec![start.to_string()]]);
        while let Some(route) = queue.pop_front() {
            let last = route.last().cloned().unwrap_or_default();
            for parent in points_at.get(&last).cloned().unwrap_or_default() {
                if parent == goal {
                    let mut found = route.clone();
                    found.push(parent);
                    return found;
                }
                if seen.insert(parent.clone()) {
                    let mut next = route.clone();
                    next.push(parent);
                    queue.push_back(next);
                }
            }
        }
        Vec::new()
    }

    /// `_find_keys`: the foreign keys from `fkey_table` to `pkey_table`,
    /// and an error unless there is exactly one.
    fn find_keys(
        &self,
        node: &NodeRef,
        fkey_table: &str,
        pkey_table: &str,
        is_target_entity: bool,
        in_entity_filter: bool,
        entity_filter_condition: Option<&str>,
    ) -> (Vec<String>, ValidationResponse) {
        let mut response = ValidationResponse::default();
        let mut keys: Vec<String> = Vec::new();
        let mut reverse_keys: Vec<String> = Vec::new();
        let pkey_of = |t: &str| self.schema.table(t).and_then(|d| d.pkey.clone());
        let pkey_of_pkey_table = pkey_of(pkey_table);
        let pkey_of_fkey_table = pkey_of(fkey_table);
        for group in &self.schema.col_groups {
            let in_group = |t: &str, k: &Option<String>| {
                k.as_ref()
                    .is_some_and(|k| group.columns.contains(&ColumnKey::new(t, k)))
            };
            let (is_keys_group, table_of_interest) = if in_group(pkey_table, &pkey_of_pkey_table) {
                (true, fkey_table)
            } else if in_group(fkey_table, &pkey_of_fkey_table) {
                (false, pkey_table)
            } else {
                continue;
            };
            let pkey_col = pkey_of_pkey_table.clone();
            let current: Vec<String> = group
                .columns
                .iter()
                .filter(|c| {
                    c.table_name == table_of_interest
                        && !(fkey_table == pkey_table && Some(&c.col_name) == pkey_col.as_ref())
                })
                .map(|c| format!("{}.{}", c.table_name, c.col_name))
                .collect();
            let into = if is_keys_group {
                &mut keys
            } else {
                &mut reverse_keys
            };
            for key in current {
                if !into.contains(&key) {
                    into.push(key);
                }
            }
        }
        let node_text = ast::to_string(node);
        let is_aggr = ast::is_aggregation(node);
        let node_fqn = ast::fqn(node);
        let mut additional = if is_aggr {
            let pkey_name = pkey_of_pkey_table
                .clone()
                .unwrap_or_else(|| "None".to_string());
            format!(
                "To unambiguously link each row in '{fkey_table}' to a \
                 '{pkey_table}.{pkey_name}', exactly one foreign key should point from \
                 '{fkey_table}' to '{pkey_table}.{pkey_name}'."
            )
        } else {
            format!(
                "To unambiguously link each row in '{fkey_table}' to a '{node_text}', exactly \
                 one foreign key should point from '{fkey_table}' to '{pkey_table}'."
            )
        };
        if !is_aggr && is_target_entity {
            additional.push_str(&format!(
                " This may also mean there is an incorrect target column specification. Column \
                 {node_fqn} is not reachable through a foreign key in table {fkey_table}."
            ));
        }
        if keys.is_empty() {
            let mut is_aggregation = false;
            if !reverse_keys.is_empty() {
                if is_aggr {
                    is_aggregation = true;
                }
                if !is_aggr && in_entity_filter {
                    let condition = entity_filter_condition
                        .map(str::to_string)
                        .unwrap_or_else(|| node_text.clone());
                    let mut fixes = Vec::new();
                    if let Some(example) = &self.target_filter_example {
                        fixes.push(format!(
                            "To filter events used by the prediction target, use '{example}'."
                        ));
                    }
                    if self.allow_temporal_entity_filter
                        && entity_filter_condition.is_some()
                        && self.entity_filter_is_row_level
                        && self.entity_filter_tables == BTreeSet::from([pkey_table.to_string()])
                        && self
                            .schema
                            .table(pkey_table)
                            .is_some_and(|t| t.time_col.is_some())
                    {
                        fixes.push(format!(
                            "To select entities that have matching events, use \
                             'COUNT({pkey_table}.* WHERE {condition}, -INF, 0) > 0'."
                        ));
                    }
                    let mut message = format!(
                        "{}: The entity filter condition '{condition}' references event column \
                         '{node_fqn}' directly. Table '{pkey_table}' can contain multiple rows \
                         for each '{fkey_table}', so there is no single value to compare for each \
                         entity. Move this condition inside an aggregation.",
                        loc_start(node)
                    );
                    if !fixes.is_empty() {
                        message.push(' ');
                        message.push_str(&fixes.join(" "));
                    }
                    response.errors.push(ValidationError::new(
                        "Invalid entity filter",
                        message,
                        codes::JOIN_EVENT_COLUMN_IN_ENTITY_FILTER,
                        None,
                    ));
                    return (keys, response);
                }
                if !is_aggr {
                    let column_name = col_name(&node_fqn);
                    response.errors.push(ValidationError::new(
                        format!("Couldn't pick a unique {node_fqn} for each {fkey_table}"),
                        format!(
                            "{}: Encountered an error, we have found multiple '{column_name}' \
                             values in the '{pkey_table}' table for a single '{fkey_table}'. To \
                             fix this, you have two options:\n1. Aggregate the values: If your \
                             '{pkey_table}' table has a 'created at' column, use an aggregation \
                             function to reduce them to one value. For example: PREDICT \
                             LAST({node_fqn}, 0, 30, days).\n2. Rewrite your query: If you are \
                             writing a static node prediction predictive query, make sure that \
                             you either use the same table to define your target and entity, or \
                             you have a foreign key from the entity pointing to the target table, \
                             not the other way around.",
                            loc_start(node)
                        ),
                        codes::JOIN_AMBIGUOUS_TARGET_VALUE,
                        span(node),
                    ));
                }
            }
            if is_aggregation || reverse_keys.is_empty() {
                let through = self.route_between(fkey_table, pkey_table);
                let explanation = if !through.is_empty() && is_aggr {
                    format!(
                        "{fkey_table} reaches {pkey_table} through {}, and an aggregation joins \
                         on one key rather than following a path, so its target has to be a \
                         table that points at {pkey_table} directly. Aggregate {fkey_table} for \
                         each {} instead, or carry {pkey_table}'s key on {fkey_table} as well.",
                        through.join(" -> "),
                        through[1]
                    )
                } else if !through.is_empty() {
                    format!(
                        "{node_fqn} is on {pkey_table}, which {fkey_table} reaches through {}, \
                         and a query follows one foreign key rather than a path. Ask for a \
                         column of {} instead, or carry {pkey_table}'s key on {fkey_table} as \
                         well.",
                        through.join(" -> "),
                        through[1]
                    )
                } else {
                    format!(
                        "There is no foreign key from table {fkey_table} to table {pkey_table}. \
                         {additional}"
                    )
                };
                response.errors.push(ValidationError::new(
                    "Missing foreign key",
                    format!(
                        "{}: Encountered an error when processing '{node_text}'. {explanation}",
                        loc_start(node)
                    ),
                    codes::JOIN_MISSING_FOREIGN_KEY,
                    span(node),
                ));
            }
        } else if keys.len() > 1 {
            let drops: Vec<String> = keys[1..]
                .iter()
                .map(|key| {
                    format!(
                        "schema.unlink('{fkey_table}', '{}', '{pkey_table}')",
                        key.rsplit('.').next().unwrap_or_default()
                    )
                })
                .collect();
            let listed: Vec<String> = keys.iter().map(|k| crate::pyrepr::str_repr(k)).collect();
            response.errors.push(ValidationError::new(
                "Ambiguous link between tables",
                format!(
                    "{}: Encountered an error when processing '{node_text}'. The foreign key from \
                     table {fkey_table} to table {pkey_table} is not unique (got [{}]). \
                     {additional} Keep the one this query means and drop the rest, for example: \
                     {}",
                    loc_start(node),
                    listed.join(", "),
                    drops.join("; ")
                ),
                codes::JOIN_AMBIGUOUS_FOREIGN_KEY,
                span(node),
            ));
        }
        (keys, response)
    }

    /// `_infer_joins`: makes `node` reachable from `source_table`.
    fn infer_joins(
        &self,
        node: &NodeRef,
        source_table: &str,
        in_entity_filter: bool,
        entity_filter_condition: Option<String>,
    ) -> Joined {
        enum Shape {
            Column,
            Aggregation(NodeRef),
            Condition(NodeRef),
            Filter(NodeRef, NodeRef),
            Other,
        }
        let shape = match &node.borrow().kind {
            Kind::Column { .. } => Shape::Column,
            Kind::Aggregation { target, .. } => Shape::Aggregation(target.clone()),
            Kind::Condition { target, .. } => Shape::Condition(target.clone()),
            Kind::Filter { target, condition } => Shape::Filter(target.clone(), condition.clone()),
            _ => Shape::Other,
        };
        let mut response = ValidationResponse::default();
        match shape {
            Shape::Column => {
                let col_table = table_name(&ast::fqn(node));
                if source_table != col_table {
                    let (keys, found) = self.find_keys(
                        node,
                        source_table,
                        &col_table,
                        true,
                        in_entity_filter,
                        entity_filter_condition.as_deref(),
                    );
                    if !found.ok() {
                        return Ok((node.clone(), found));
                    }
                    let rhs_pkey = self
                        .schema
                        .table(&col_table)
                        .and_then(|t| t.pkey.clone())
                        .unwrap_or_default();
                    let joined = ast::join(
                        node.clone(),
                        keys[0].clone(),
                        format!("{col_table}.{rhs_pkey}"),
                    )
                    .map_err(|e| JoinFailure(e.message().to_string()))?;
                    return Ok((joined, found));
                }
            }
            Shape::Aggregation(target) => {
                let target_table = table_name(&ast::get_target_column_name(node));
                let (new_target, target_response) =
                    self.infer_joins(&target, &target_table, false, None)?;
                if let Kind::Aggregation { target, .. } = &mut node.borrow_mut().kind {
                    *target = new_target;
                }
                response = response.merge(target_response);
                let (keys, join_response) =
                    self.find_keys(node, &target_table, source_table, false, false, None);
                response = response.merge(join_response);
                if !response.ok() {
                    return Ok((node.clone(), response));
                }
                if let Kind::Aggregation { group_by, .. } = &mut node.borrow_mut().kind {
                    *group_by = Some(keys[0].clone());
                }
                let lhs_pkey = self
                    .schema
                    .table(source_table)
                    .and_then(|t| t.pkey.clone())
                    .unwrap_or_default();
                let joined = ast::join(
                    node.clone(),
                    format!("{source_table}.{lhs_pkey}"),
                    keys[0].clone(),
                )
                .map_err(|e| JoinFailure(e.message().to_string()))?;
                return Ok((joined, response));
            }
            Shape::Condition(target) => {
                let mut filter_condition = entity_filter_condition;
                if in_entity_filter && filter_condition.is_none() {
                    filter_condition = Some(ast::to_string(node));
                }
                let (new_target, target_response) =
                    self.infer_joins(&target, source_table, in_entity_filter, filter_condition)?;
                if let Kind::Condition { target, .. } = &mut node.borrow_mut().kind {
                    *target = new_target;
                }
                response = response.merge(target_response);
            }
            Shape::Filter(target, condition) => {
                let target_table = table_name(&ast::fqn(&target));
                let (new_target, target_response) =
                    self.infer_joins(&target, source_table, false, None)?;
                if !ast::is_column(&new_target) {
                    return Err(JoinFailure(
                        "a filter's column must be on the table it filters".to_string(),
                    ));
                }
                if let Kind::Filter { target, .. } = &mut node.borrow_mut().kind {
                    *target = new_target;
                }
                response = response.merge(target_response);
                let (new_condition, condition_response) = self.infer_joins(
                    &condition,
                    &target_table,
                    in_entity_filter,
                    entity_filter_condition,
                )?;
                if let Kind::Filter { condition, .. } = &mut node.borrow_mut().kind {
                    *condition = new_condition;
                }
                response = response.merge(condition_response);
            }
            Shape::Other => {
                for child in ast::children(node) {
                    let (_, child_response) = self.infer_joins(
                        &child,
                        source_table,
                        in_entity_filter,
                        entity_filter_condition.clone(),
                    )?;
                    response = response.merge(child_response);
                }
            }
        }
        Ok((node.clone(), response))
    }
}

/// Whether the subtree holds an aggregation.
pub fn contains_aggregation(node: &NodeRef) -> bool {
    ast::is_aggregation(node) || ast::children(node).iter().any(contains_aggregation)
}
