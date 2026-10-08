// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The validated query as the sampler reads it: read once from the tree the
//! host gives (`json_serde.to_json_dict`), through the PQL core, into the
//! facts `sample_target` asks of it, and the label executor's query.

use sdm_labels::date_offset::{DateOffset, Unit};
use sdm_pql::ast::{self, NodeRef};
use sdm_pql::types::{ProblemType, TimeUnit};
use sdm_pql::value::Typed;
use serde_json::Value;

use super::Raised;

/// A window `(start, end]` in units of the anchor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Window {
    pub start: Option<i64>,
    pub end: i64,
    pub unit: Unit,
}

impl Window {
    /// `start_date_offset`: `None` for an open start.
    pub fn start_offset(&self) -> Option<DateOffset> {
        self.start.map(|v| DateOffset::single(self.unit, v))
    }

    pub fn end_offset(&self) -> DateOffset {
        DateOffset::single(self.unit, self.end)
    }

    /// `DateOffsetRange.timeframe`: `(1970 + end) - (1970 + start)` in
    /// nanoseconds, `None` for an open start.
    pub fn timeframe_ns(&self) -> Result<Option<i64>, Raised> {
        let Some(start) = self.start_offset() else {
            return Ok(None);
        };
        let end = self.end_offset().apply(0, 1)?;
        let start = start.apply(0, 1)?;
        Ok(Some(end - start))
    }
}

fn unit_of(unit: TimeUnit) -> Unit {
    match unit {
        TimeUnit::Seconds => Unit::Seconds,
        TimeUnit::Minutes => Unit::Minutes,
        TimeUnit::Hours => Unit::Hours,
        TimeUnit::Days => Unit::Days,
        TimeUnit::Weeks => Unit::Weeks,
        TimeUnit::Months => Unit::Months,
    }
}

fn window_of(range: &Option<sdm_pql::dor::DorRef>) -> Option<Window> {
    range.as_ref().map(|r| {
        let r = r.borrow();
        Window {
            start: r.start,
            end: r.end,
            unit: unit_of(r.unit),
        }
    })
}

/// An aggregation the query computes: the table of its target column, and
/// its window.
#[derive(Clone, Debug)]
pub struct Aggregation {
    pub table: String,
    pub window: Option<Window>,
}

fn aggregation(node: &NodeRef) -> Aggregation {
    let name = ast::get_target_column_name(node);
    Aggregation {
        table: name.split('.').next().unwrap_or_default().to_string(),
        window: window_of(&ast::aggr_time_range(node)),
    }
}

/// An entity id a query names (`FOR t.c = ...`), as its typed value.
#[derive(Clone, Debug, PartialEq)]
pub enum EntityId {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Text(String),
}

/// What the target is, as `_get_task_type` reads it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TargetKind {
    /// A condition or a logical operation: a yes or no.
    Condition,
    /// An aggregation (through a join, if any): its type and its target
    /// column, `table.column`.
    Aggregation { aggr: String, column: String },
    /// A column: its stype.
    Column { stype: Option<String> },
    /// Anything else.
    Other,
}

/// What the sampler asks of a validated query.
#[derive(Clone, Debug)]
pub struct SamplerQuery {
    pub entity_table: String,
    pub is_forecasting: bool,
    pub num_forecasts: i64,
    /// `get_rfm_entity_id_list()`, where the query names ids.
    pub rfm_entity_ids: Option<Vec<EntityId>>,
    /// `target_timeframe`: present for a temporal query.
    pub target_timeframe: Option<Window>,
    /// `all_query_columns`.
    pub all_query_columns: Vec<String>,
    /// `get_all_target_aggregations()`.
    pub target_aggregations: Vec<Aggregation>,
    /// Every aggregation of the target, the entity filter and the
    /// `ASSUMING`, in the order `_get_query_time_offset_dict` meets them.
    pub windowed_aggregations: Vec<Aggregation>,
    /// The entity filter's condition, as `to_string()` writes it.
    pub entity_filter_condition: Option<String>,
    /// `get_final_target_column()`: the column the label is the value of,
    /// where it is one.
    pub final_target_column: Option<String>,
    /// `get_final_target_aggregation().get_target_column_name()`, where the
    /// target is an aggregation.
    pub final_aggregation_column: Option<String>,
    /// The target, as `_get_task_type` reads it.
    pub target_kind: TargetKind,
    /// The entity column, `table.column`.
    pub entity_column: String,
    /// The target's `date_offset_range`'s end, where it has one.
    pub target_end_offset: Option<DateOffset>,
    /// The label executor's query.
    pub labels: sdm_labels::query::Query,
}

impl SamplerQuery {
    pub fn is_temporal(&self) -> bool {
        self.target_timeframe.is_some()
    }

    pub fn from_tree(tree: &Value) -> Result<Self, Raised> {
        let validated = sdm_pql::tree::read_query(tree).map_err(Raised::value_error)?;
        Self::from_validated(&validated)
    }

    /// `query.to_autoregressive_query()` of the query a tree holds: no
    /// `ASSUMING`, no entity filter, and a single aggregation's condition
    /// dropped.
    pub fn autoregressive_from_tree(tree: &Value) -> Result<Self, Raised> {
        let validated = sdm_pql::tree::read_query(tree).map_err(Raised::value_error)?;
        let autoregressive = sdm_pql::ValidatedQuery {
            query: validated.query.to_autoregressive_query(),
            validation_response: validated.validation_response.clone(),
        };
        Self::from_validated(&autoregressive)
    }

    /// The sampler's view of a validated query.
    pub fn from_validated(validated: &sdm_pql::ValidatedQuery) -> Result<Self, Raised> {
        let query = &validated.query;
        let tree = query.to_json(&validated.validation_response).render();
        let mut windowed = Vec::new();
        fn walk(node: &NodeRef, out: &mut Vec<Aggregation>) {
            if ast::is_aggregation(node) {
                out.push(aggregation(node));
            }
            for child in ast::children(node) {
                walk(&child, out);
            }
        }
        walk(&query.target_ast, &mut windowed);
        walk(&query.entity_ast, &mut windowed);
        if let Some(whatif) = &query.whatif_ast {
            walk(whatif, &mut windowed);
        }
        let entity_filter_condition = match &query.entity_ast.borrow().kind {
            sdm_pql::ast::Kind::Filter { condition, .. } => Some(ast::to_string(condition)),
            _ => None,
        };
        let rfm_entity_ids = match &query.rfm_entity_ids {
            None => None,
            Some(_) => Some(
                validated
                    .get_rfm_entity_id_list()
                    .map_err(Raised::value_error)?
                    .iter()
                    .map(entity_id)
                    .collect::<Result<_, _>>()?,
            ),
        };
        let labels = sdm_labels::query::Query::from_json(&tree)?;
        Ok(Self {
            entity_table: query.entity_table(),
            is_forecasting: query.problem_type == Some(ProblemType::Forecast),
            num_forecasts: query.num_forecasts.to_i64().unwrap_or(1),
            rfm_entity_ids,
            target_timeframe: window_of(&query.target_timeframe()),
            all_query_columns: query.all_query_columns(),
            target_aggregations: validated
                .get_all_target_aggregations()
                .iter()
                .map(aggregation)
                .collect(),
            windowed_aggregations: windowed,
            entity_filter_condition,
            final_target_column: validated
                .get_final_target_column()
                .map(|column| ast::fqn(&column)),
            final_aggregation_column: validated
                .get_final_target_aggregation()
                .map(|aggregation| ast::get_target_column_name(&aggregation)),
            target_kind: target_kind(&query.target_ast),
            entity_column: query.entity_column(),
            target_end_offset: window_of(&ast::date_offset_range(&query.target_ast))
                .map(|w| w.end_offset()),
            labels,
        })
    }
}

fn target_kind(target: &NodeRef) -> TargetKind {
    if ast::is_condition(target) || ast::is_logical(target) {
        return TargetKind::Condition;
    }
    let target = if ast::is_join(target) {
        match &target.borrow().kind {
            sdm_pql::ast::Kind::Join { rhs_target, .. } => rhs_target.clone(),
            _ => target.clone(),
        }
    } else {
        target.clone()
    };
    if let Some(aggr) = ast::aggr(&target) {
        return TargetKind::Aggregation {
            aggr: aggr.value().to_string(),
            column: ast::get_target_column_name(&target),
        };
    }
    if ast::is_column(&target) {
        return TargetKind::Column {
            stype: ast::stype(&target).map(|s| s.value().to_string()),
        };
    }
    TargetKind::Other
}

fn entity_id(value: &Typed) -> Result<EntityId, Raised> {
    Ok(match value {
        Typed::None => EntityId::None,
        Typed::Bool(b) => EntityId::Bool(*b),
        Typed::Int(i) => EntityId::Int(
            i.to_i64()
                .ok_or_else(|| Raised::value_error(format!("the entity id {i} is past int64")))?,
        ),
        Typed::Float(f) => EntityId::Float(*f),
        Typed::Str(s) => EntityId::Text(s.clone()),
        other => {
            return Err(Raised::value_error(format!(
                "an entity id of {other:?} is not supported by the core"
            )));
        }
    })
}
