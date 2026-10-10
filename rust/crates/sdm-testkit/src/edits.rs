// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The user's edits to the schema, as this host makes them to the
//! description its own `describe_source` answered (decision E50, T1.21d):
//! the case names only the edits, never the schema `materialize` samples.
//!
//! An edit is what the Python client's schema does with the same call,
//! as its host then hands the schema to `materialize`, where that is plain
//! data: a time column cleared, a column left out, a key named, the
//! relationships replaced, each once. A key named keeps the count the
//! description gave its table, as the Python host sends it (its
//! `_rust/materialize.py` takes each table's `duplicate_keys` from the
//! core's description, the schema holding none for a key named after it
//! was built); following an edit past the description is T1.24's (the
//! `schema_edited` refusal). Which columns the schema joins on is not
//! re-derived here: the core marks them again from what the host holds
//! (`materialize`'s consistency check, decision E55). An edit the Python
//! client refuses (a table or column the schema lacks, a key or time
//! column left out, a time column named as the key) is this host's
//! failure, not a case outcome.

use anyhow::{Result, bail};
use sdm_contract::ops::{ColumnRef, SourceDescription};
use serde::Deserialize;

use crate::case;

/// One edit, in the case format's spelling: the case's own two
/// (`clear_time_column`, `delete_column`) and the refusal map's probes'
/// two more (`set_primary_key`, `set_relationships`).
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum SchemaEdit {
    /// `schema[table].time_column = None`.
    ClearTimeColumn { table: String },
    /// `del schema[table][column]`.
    DeleteColumn { table: String, column: String },
    /// `schema[table].primary_key = column`.
    SetPrimaryKey { table: String, column: String },
    /// `schema.relationships = [...]`.
    SetRelationships { relationships: Vec<Link> },
}

/// A relationship as the case format writes one.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Link {
    pub source_table: String,
    pub source_column: String,
    pub target_table: String,
    pub target_column: String,
}

impl From<&case::Edit> for SchemaEdit {
    fn from(edit: &case::Edit) -> SchemaEdit {
        match edit {
            case::Edit::ClearTimeColumn { table } => SchemaEdit::ClearTimeColumn {
                table: table.0.clone(),
            },
            case::Edit::DeleteColumn { table, column } => SchemaEdit::DeleteColumn {
                table: table.0.clone(),
                column: column.0.clone(),
            },
        }
    }
}

/// `described` with `edits` made, in order.
pub fn edited(described: &SourceDescription, edits: &[SchemaEdit]) -> Result<SourceDescription> {
    let mut schema = described.clone();
    for edit in edits {
        match edit {
            SchemaEdit::ClearTimeColumn { table } => {
                let Some(at) = schema.tables.iter_mut().find(|t| &t.name == table) else {
                    bail!("an edit names a table the schema lacks");
                };
                at.time_column = None;
            }
            SchemaEdit::DeleteColumn { table, column } => {
                let Some(at) = schema.tables.iter_mut().find(|t| &t.name == table) else {
                    bail!("an edit names a table the schema lacks");
                };
                if at.primary_key.contains(column) || at.time_column.as_ref() == Some(column) {
                    bail!("an edit leaves out a table's key or time column");
                }
                let before = at.columns.len();
                at.columns.retain(|c| &c.name != column);
                if at.columns.len() == before {
                    bail!("an edit names a column the schema lacks");
                }
            }
            SchemaEdit::SetPrimaryKey { table, column } => {
                let Some(at) = schema.tables.iter_mut().find(|t| &t.name == table) else {
                    bail!("an edit names a table the schema lacks");
                };
                if !at.columns.iter().any(|c| &c.name == column) {
                    bail!("an edit names a column the schema lacks");
                }
                if at.time_column.as_ref() == Some(column) {
                    bail!("an edit names a table's time column as its key");
                }
                at.primary_key = vec![column.clone()];
            }
            SchemaEdit::SetRelationships { relationships } => {
                let mut links: Vec<Vec<ColumnRef>> = Vec::new();
                for link in relationships {
                    let pair = vec![
                        ColumnRef {
                            table: link.source_table.clone(),
                            column: link.source_column.clone(),
                        },
                        ColumnRef {
                            table: link.target_table.clone(),
                            column: link.target_column.clone(),
                        },
                    ];
                    if !links.contains(&pair) {
                        links.push(pair);
                    }
                }
                schema.links = links;
            }
        }
    }
    Ok(schema)
}
