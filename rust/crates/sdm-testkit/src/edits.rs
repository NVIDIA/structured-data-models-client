// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The user's edits to the schema, as this host makes them (decision E67,
//! T1.24b): each passed to the core's `edit_schema` with the description
//! the core last answered and the schema as this host holds it, the core
//! function the Python client's setters pass theirs to, so the core checks
//! and makes every edit and this host decides none (`crate::host` runs
//! them). The case names only the edits, never the schema `materialize`
//! samples (decision E50). An edit the core refuses is the case's outcome,
//! at the from_source stage, as the Python runner records it.
//!
//! This host holds a schema as the description the core last answered
//! ([`held`]): it changes nothing of it the core did not answer, so it
//! never holds a schema changed past its description.

use sdm_contract::ops::{
    self, ColumnRef, Given, HeldColumn, HeldSchema, HeldTable, SourceDescription,
};
use serde::Deserialize;

use crate::case;

/// One edit, in the case format's spelling, and the refusal map's probes'.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum SchemaEdit {
    /// `schema[table].time_column = None`.
    ClearTimeColumn { table: String },
    /// `del schema[table][column]`.
    DeleteColumn { table: String, column: String },
    /// `schema[table].primary_key = column`.
    SetPrimaryKey { table: String, column: String },
    /// `schema[table].time_column = column`.
    SetTimeColumn { table: String, column: String },
    /// `schema[table][column].stype = stype`.
    SetStype {
        table: String,
        column: String,
        stype: String,
    },
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

impl SchemaEdit {
    /// The case format's edit as this host makes it, every edit of the
    /// format (from T1.27, whose gate's checkpoint loader holds T1.24b's
    /// four: `set_primary_key`, `set_time_column`, `set_stype`,
    /// `set_relationships`). An `Option` for [`crate::host::run`]'s
    /// refusal of an edit a checkpoint's loader might not name.
    pub fn from_case(edit: &case::Edit) -> Option<SchemaEdit> {
        Some(match edit {
            case::Edit::ClearTimeColumn { table } => SchemaEdit::ClearTimeColumn {
                table: table.0.clone(),
            },
            case::Edit::DeleteColumn { table, column } => SchemaEdit::DeleteColumn {
                table: table.0.clone(),
                column: column.0.clone(),
            },
            case::Edit::SetPrimaryKey { table, column } => SchemaEdit::SetPrimaryKey {
                table: table.0.clone(),
                column: column.0.clone(),
            },
            case::Edit::SetTimeColumn { table, column } => SchemaEdit::SetTimeColumn {
                table: table.0.clone(),
                column: column.0.clone(),
            },
            case::Edit::SetStype {
                table,
                column,
                stype,
            } => SchemaEdit::SetStype {
                table: table.0.clone(),
                column: column.0.clone(),
                stype: stype.0.clone(),
            },
            case::Edit::SetRelationships { relationships } => SchemaEdit::SetRelationships {
                relationships: relationships
                    .iter()
                    .map(|link| Link {
                        source_table: link.source_table.0.clone(),
                        source_column: link.source_column.0.clone(),
                        target_table: link.target_table.0.clone(),
                        target_column: link.target_column.0.clone(),
                    })
                    .collect(),
            },
        })
    }

    /// The edit as the contract's `edit_schema` takes it: the same edit
    /// the Python client's setter passes for the same call.
    pub fn to_contract(&self) -> ops::SchemaEdit {
        match self {
            SchemaEdit::ClearTimeColumn { table } => ops::SchemaEdit::TimeColumn {
                table: table.clone(),
                column: Given::Value(None),
            },
            SchemaEdit::DeleteColumn { table, column } => ops::SchemaEdit::LeaveOut {
                table: table.clone(),
                column: Given::Value(column.clone()),
            },
            SchemaEdit::SetPrimaryKey { table, column } => ops::SchemaEdit::PrimaryKey {
                table: table.clone(),
                key: Given::Value(Some(vec![column.clone()])),
            },
            SchemaEdit::SetTimeColumn { table, column } => ops::SchemaEdit::TimeColumn {
                table: table.clone(),
                column: Given::Value(Some(column.clone())),
            },
            SchemaEdit::SetStype {
                table,
                column,
                stype,
            } => ops::SchemaEdit::Stype {
                table: table.clone(),
                column: column.clone(),
                stype: Given::Value(stype.clone()),
            },
            SchemaEdit::SetRelationships { relationships } => ops::SchemaEdit::Relationships {
                links: relationships
                    .iter()
                    .map(|link| {
                        Given::Value(vec![
                            ColumnRef {
                                table: link.source_table.clone(),
                                column: link.source_column.clone(),
                            },
                            ColumnRef {
                                table: link.target_table.clone(),
                                column: link.target_column.clone(),
                            },
                        ])
                    })
                    .collect(),
            },
        }
    }
}

/// The schema as this host holds it: the description the core last
/// answered, as it is, each column's data type by its wire name.
pub fn held(described: &SourceDescription) -> HeldSchema {
    HeldSchema {
        tables: described
            .tables
            .iter()
            .map(|table| HeldTable {
                name: table.name.clone(),
                source: table.source.clone(),
                columns: table
                    .columns
                    .iter()
                    .map(|column| HeldColumn {
                        name: column.name.clone(),
                        dtype: column.dtype.wire_name().to_owned(),
                        stype: column.stype,
                        stype_before_key: column.stype_before_key,
                    })
                    .collect(),
                primary_key: table.primary_key.clone(),
                time_column: table.time_column.clone(),
            })
            .collect(),
        links: described.links.clone(),
    }
}
