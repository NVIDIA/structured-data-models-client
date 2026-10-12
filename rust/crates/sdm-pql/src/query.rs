// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A predictive query as the parser produces it, and once validated
//! (`predictive_query.query`).

use std::collections::BTreeMap;

use crate::ast::{self, Kind, Location, NodeDtype, NodeRef};
use crate::dor::{DateOffsetRange, DorRef};
use crate::literal::Literal;
use crate::pyint::PyInt;
use crate::response::ValidationResponse;
use crate::types::{BoolOp, Dtype, Op, ProblemType, QueryType, Stype};

/// A query as the parser produced it: no types, no joins.
#[derive(Clone, Debug)]
pub struct ParsedQuery {
    pub entity_ast: NodeRef,
    pub target_ast: NodeRef,
    pub whatif_ast: Option<NodeRef>,
    pub top_k: Option<PyInt>,
    pub problem_type: Option<ProblemType>,
    pub rfm_query: bool,
    /// `FOR EACH` or `FOR`.
    pub for_each: String,
    pub rfm_entity_ids: Option<NodeRef>,
    pub num_forecasts: PyInt,
    pub evaluate: bool,
    pub explain: bool,
}

impl ParsedQuery {
    /// The entity column, `table.column`.
    pub fn entity_column(&self) -> String {
        ast::fqn(&self.entity_column_obj())
    }

    pub fn entity_table(&self) -> String {
        self.entity_column()
            .split('.')
            .next()
            .unwrap_or_default()
            .to_string()
    }

    /// The entity column's node, through a filter if there is one.
    pub fn entity_column_obj(&self) -> NodeRef {
        if ast::is_filter(&self.entity_ast) {
            return ast::target(&self.entity_ast).unwrap_or_else(|| self.entity_ast.clone());
        }
        self.entity_ast.clone()
    }

    pub fn entity_timeframe(&self) -> Option<DorRef> {
        ast::date_offset_range(&self.entity_ast)
    }

    pub fn target_timeframe(&self) -> Option<DorRef> {
        ast::date_offset_range(&self.target_ast)
    }

    pub fn whatif_timeframe(&self) -> Option<DorRef> {
        self.whatif_ast.as_ref().and_then(ast::date_offset_range)
    }

    fn parts(&self) -> Vec<NodeRef> {
        let mut parts = vec![self.entity_ast.clone(), self.target_ast.clone()];
        parts.extend(self.whatif_ast.clone());
        parts
    }

    /// Every column the query names, with where, each once. Python's set
    /// gives them in an order that changes from run to run; here they come
    /// entity first, then target, then `ASSUMING`, in the order written.
    pub fn all_query_columns_with_locations(&self) -> Vec<(String, Location)> {
        let mut out: Vec<(String, Location)> = Vec::new();
        for part in self.parts() {
            for item in ast::all_query_columns_with_locations(&part) {
                if !out.contains(&item) {
                    out.push(item);
                }
            }
        }
        out
    }

    /// Every column the query names, each once, in the order above.
    pub fn all_query_columns(&self) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        for part in self.parts() {
            for column in ast::all_query_columns(&part) {
                if !out.contains(&column) {
                    out.push(column);
                }
            }
        }
        out
    }

    /// The query written back, `PREDICT` first unless left out, and its
    /// keywords in bold for a rich console when `rich`.
    pub fn to_string_with(&self, rich: bool, exclude_predict: bool) -> String {
        let bold = |t: &str| ast::maybe_bold(t, rich);
        let mut text = if exclude_predict {
            String::new()
        } else {
            format!("{} ", bold("PREDICT"))
        };
        text.push_str(&ast::render(&self.target_ast, rich));
        if self.evaluate && !exclude_predict {
            text = format!("EVALUATE {text}");
        }
        if self.explain && !exclude_predict {
            text = format!("EXPLAIN {text}");
        }
        if let Some(problem_type) = self.problem_type {
            if problem_type == ProblemType::Forecast {
                text.push_str(&format!(
                    " {} {} {}",
                    bold("FORECAST"),
                    self.num_forecasts,
                    bold("TIMEFRAMES")
                ));
            } else {
                text.push_str(&format!(" {}", problem_type.value()));
            }
        }
        if let Some(top_k) = &self.top_k {
            text.push_str(&format!(" {}", bold(&format!("TOP {top_k}"))));
        }
        text.push_str(&format!(" {}", bold(&self.for_each)));
        match &self.rfm_entity_ids {
            None => text.push_str(&format!(" {}", ast::render(&self.entity_ast, rich))),
            Some(ids) => {
                text.push_str(&format!(" {}", ast::render(ids, rich)));
                if let Kind::Filter { condition, .. } = &self.entity_ast.borrow().kind {
                    text.push_str(&format!(
                        " {} {}",
                        bold("WHERE"),
                        ast::render(condition, rich)
                    ));
                }
            }
        }
        if let Some(whatif) = &self.whatif_ast {
            text.push_str(&format!(
                " {} {}",
                bold("ASSUMING"),
                ast::render(whatif, rich)
            ));
        }
        text
    }

    /// The query as a label generator for autoregression: no `ASSUMING`,
    /// no entity filter, and a single aggregation's condition dropped.
    pub fn to_autoregressive_query(&self) -> Self {
        let mut out = self.clone();
        out.whatif_ast = None;
        if ast::is_filter(&out.entity_ast)
            && let Some(target) = ast::target(&out.entity_ast)
        {
            out.entity_ast = target;
        }
        let not_left = match &out.target_ast.borrow().kind {
            Kind::Logical {
                left,
                op: BoolOp::Not,
                ..
            } => Some(left.clone()),
            _ => None,
        };
        if let Some(left) = not_left {
            out.target_ast = left;
        }
        let inner = match &out.target_ast.borrow().kind {
            Kind::Condition {
                target,
                op: Op::Rel(_),
                ..
            } if ast::is_aggregation(target) => Some(target.clone()),
            _ => None,
        };
        if let Some(inner) = inner {
            out.target_ast = inner;
        }
        out
    }
}

/// A query the validator accepted against a schema, with what it found.
#[derive(Clone, Debug)]
pub struct ValidatedQuery {
    pub query: ParsedQuery,
    pub validation_response: ValidationResponse,
}

impl ValidatedQuery {
    pub fn entity_dtype(&self) -> Option<Dtype> {
        ast::dtype(&self.query.entity_ast).and_then(NodeDtype::plain)
    }

    /// The target's type, an array's as the list type of its elements.
    pub fn target_dtype(&self) -> Option<Dtype> {
        ast::dtype(&self.query.target_ast).and_then(|d| d.to_dtype().ok())
    }

    pub fn target_stype(&self) -> Option<Stype> {
        ast::stype(&self.query.target_ast)
    }

    /// The target column, or the target written out if it is computed.
    pub fn target_column(&self) -> String {
        match self.get_final_target_column() {
            Some(column) => ast::fqn(&column),
            None => ast::to_string(&self.query.target_ast),
        }
    }

    pub fn query_type(&self) -> QueryType {
        if self.query.target_timeframe().is_some() {
            QueryType::Temporal
        } else {
            QueryType::Static
        }
    }

    /// The key joining the target's table to the entity's: the first in
    /// sort order of those found, `None` if the tables are one.
    pub fn get_single_target_fkey(&self, lhs: bool) -> Option<String> {
        single_target_fkey(&self.query.target_ast, lhs)
    }

    /// The target's aggregation, if aggregating is the last thing it does.
    pub fn get_final_target_aggregation(&self) -> Option<NodeRef> {
        final_node(&self.query.target_ast, ast::is_aggregation)
    }

    /// The target's column, if the label is a column as it stands.
    pub fn get_final_target_column(&self) -> Option<NodeRef> {
        final_node(&self.query.target_ast, ast::is_column)
    }

    /// Every aggregation the label is computed from, outside filters.
    pub fn get_all_target_aggregations(&self) -> Vec<NodeRef> {
        let mut out = Vec::new();
        target_aggregations(&self.query.target_ast, &mut out);
        out
    }

    /// The target aggregations' windows, merged.
    pub fn get_combined_date_offset_range(&self) -> Result<DorRef, String> {
        let mut combined: Option<DorRef> = None;
        let aggregations = self.get_all_target_aggregations();
        if aggregations.is_empty() {
            return Err(
                "Cannot compute combined time range because target does not have \
                        any aggregations."
                    .to_string(),
            );
        }
        for aggregation in aggregations {
            let window =
                ast::aggr_time_range(&aggregation).ok_or("an aggregation without a window")?;
            combined = Some(match combined {
                None => window,
                Some(c) => DateOffsetRange::merge_ranges(&c, &window)?,
            });
        }
        combined.ok_or_else(String::new)
    }

    /// The entity IDs a `FOR t.c = ...` query names, as values.
    pub fn get_rfm_entity_id_list(&self) -> Result<Vec<Literal>, String> {
        let Some(ids) = &self.query.rfm_entity_ids else {
            return Ok(Vec::new());
        };
        let Some(value) = ast::children(ids).get(1).cloned() else {
            return Ok(Vec::new());
        };
        match ast::typed_value(&value)? {
            Literal::List(items) => Ok(items),
            one => Ok(vec![one]),
        }
    }

    /// The columns a static target reads directly, by table, which a model
    /// must not see.
    pub fn get_exclude_cols_dict(&self) -> BTreeMap<String, Vec<String>> {
        exclude_cols(&self.query.target_ast)
    }
}

fn single_target_fkey(node: &NodeRef, lhs: bool) -> Option<String> {
    if let Kind::Join {
        lhs_key, rhs_key, ..
    } = &node.borrow().kind
    {
        return Some(if lhs { lhs_key } else { rhs_key }.clone());
    }
    if ast::is_filter(node) {
        return ast::target(node).and_then(|t| single_target_fkey(&t, lhs));
    }
    ast::children(node)
        .iter()
        .filter_map(|c| single_target_fkey(c, lhs))
        .min()
}

fn final_node(node: &NodeRef, wanted: fn(&NodeRef) -> bool) -> Option<NodeRef> {
    if wanted(node) {
        return Some(node.clone());
    }
    if ast::is_join(node) {
        return ast::target(node).and_then(|t| final_node(&t, wanted));
    }
    None
}

fn target_aggregations(node: &NodeRef, out: &mut Vec<NodeRef>) {
    if ast::is_aggregation(node) {
        out.push(node.clone());
        return;
    }
    if ast::is_filter(node) {
        if let Some(target) = ast::target(node) {
            target_aggregations(&target, out);
        }
        return;
    }
    for child in ast::children(node) {
        target_aggregations(&child, out);
    }
}

fn exclude_cols(node: &NodeRef) -> BTreeMap<String, Vec<String>> {
    if ast::is_column(node) {
        let fqn = ast::fqn(node);
        let mut parts = fqn.split('.');
        let table = parts.next().unwrap_or_default().to_string();
        let column = parts.next().unwrap_or_default().to_string();
        return BTreeMap::from([(table, vec![column])]);
    }
    if ast::is_aggregation(node) {
        return BTreeMap::new();
    }
    if ast::is_filter(node) {
        return ast::target(node)
            .map(|t| exclude_cols(&t))
            .unwrap_or_default();
    }
    let mut out: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for child in ast::children(node) {
        for (table, cols) in exclude_cols(&child) {
            let entry = out.entry(table).or_default();
            for col in cols {
                if !entry.contains(&col) {
                    entry.push(col);
                }
            }
        }
    }
    out
}
