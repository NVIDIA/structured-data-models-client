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

/// The column of `table.column`.
pub fn col_name(full_name: &str) -> String {
    full_name.rsplit('.').next().unwrap_or_default().to_string()
}

/// The table of `table.column`, which may itself hold dots.
pub fn table_name(full_name: &str) -> String {
    match full_name.rsplit_once('.') {
        Some((table, _)) => table.to_string(),
        None => String::new(),
    }
}

/// `table.column`.
pub fn fqn(table: &str, column: &str) -> String {
    format!("{table}.{column}")
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
    let aggregated = ast::get_target_column_name(&aggr);
    let key = ColumnKey::new(&table_name(&aggregated), &col_name(&aggregated));
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
    Err(format!("Foreign Key {aggregated} not part of any group"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_split_on_the_last_dot() {
        assert_eq!(col_name("a.b.c"), "c");
        assert_eq!(table_name("a.b.c"), "a.b");
        assert_eq!(table_name("abc"), "");
        assert_eq!(fqn("t", "c"), "t.c");
    }
}
