// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The first steps of a metric view's conversion (T3.5, M2), each a pure
//! composition of the schema's order and the data layer's reading: the
//! source table's name, and the joins kept or skipped, in the old client's
//! order (`DB/metric_view.py:784-909`). The ordered routine of stage 2
//! (`view.rs`) asks them in turn, between its reads.

use sdm_data::view_expr::metric::{join_condition, table_reference};
use sdm_data::view_expr::strip_comments;
use sdm_schema::view::joins::{self, JoinWalk};
use sdm_schema::view::spec::{Source, TableItem, ViewRefusal, ViewSpec};
use sdm_schema::view::yaml::NodeId;

/// Why the data layer refuses a join's source or condition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JoinJudgement {
    /// Its source is no plain one-to-three-part table name.
    NotPlainSource,
    /// Its condition is no single-column equality.
    NotSingleEquality,
}

/// The source table's name parts, or the node of a source that is no
/// plain table name, which refuses the view (`:784-791`).
pub fn fact_source(spec: &ViewSpec) -> Result<Vec<String>, NodeId> {
    let Some(TableItem::Table(table)) = spec.tables.first() else {
        unreachable!("a metric view's first table is its source")
    };
    let Source::Name(Some(node)) = table.source else {
        unreachable!("a metric view's source is a name")
    };
    spec.text(node).and_then(table_reference).ok_or(node)
}

/// The joins kept and skipped (`_add_joins`): a source read as a plain
/// table name, a condition with its comments dropped read as a
/// single-column equality, each refusal skipping the join.
pub fn walk_joins(spec: &mut ViewSpec) -> Result<JoinWalk<JoinJudgement>, ViewRefusal> {
    joins::walk(
        spec,
        |spec, source| {
            source
                .and_then(|node| spec.text(node))
                .and_then(table_reference)
                .map(|_| ())
                .ok_or(JoinJudgement::NotPlainSource)
        },
        |spec, on| {
            on.and_then(|node| spec.text(node))
                .and_then(|text| strip_comments(text).ok())
                .and_then(|text| join_condition(&text))
                .map(|_| ())
                .ok_or(JoinJudgement::NotSingleEquality)
        },
    )
}
