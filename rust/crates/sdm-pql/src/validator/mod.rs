// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The validators that resolve a parsed query against a schema
//! (`predictive_query.validator`), and what they share.

pub mod join;
pub mod model;
pub mod predictive_query;
pub mod problem_type;
pub mod time_range;
pub mod type_validator;

use crate::ast;
use crate::query::ValidatedQuery;
use crate::schema::{ColumnKey, SchemaDefinition};

/// Marks the column each `COUNT` counts, as validation does
/// (`_set_count_cols`).
pub fn mark_count_cols(node: &ast::NodeRef) {
    predictive_query::set_count_cols(node);
}

/// `get_rhs_lp_table`: the table a link-prediction target's foreign key
/// points at.
pub fn get_rhs_lp_table(
    query: &ValidatedQuery,
    schema: &SchemaDefinition,
) -> Result<String, String> {
    let aggr = query
        .get_final_target_aggregation()
        .ok_or("the target is not an aggregation")?;
    let key = ast::get_target_column_name(&aggr).unwrap_or_else(|| ColumnKey::new("", ""));
    for group in &schema.col_groups {
        if !group.columns.contains(&key) {
            continue;
        }
        for col in &group.columns {
            if schema
                .table(&col.table_name)
                .is_some_and(|t| t.pkey.as_deref() == Some(col.col_name.as_str()))
            {
                return Ok(col.table_name.clone());
            }
        }
        return Err(format!("col group {group:?} has no primary key"));
    }
    Err(format!("Foreign Key {key} not part of any group"))
}
