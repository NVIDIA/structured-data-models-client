// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Host-held metadata beside the Core-owned schema, as the base SDK retains it.

use std::collections::BTreeMap;

use anyhow::{Result, anyhow};
use sdm_contract::ops::{ColumnRef, HeldSchema, SourceDescription};
use serde_json::Value;

use crate::edits;

pub(super) struct SchemaOwner {
    description: SourceDescription,
    held: HeldSchema,
}

impl SchemaOwner {
    pub(super) fn new(description: SourceDescription) -> Self {
        let mut held = edits::held(&description);
        for (table, described) in held.tables.iter_mut().zip(&description.tables) {
            for column in &mut table.columns {
                // Exactly the initial host construction in SDK schema.build.
                column.stype_declared = described.time_column_inferred.as_ref()
                    == Some(&column.name)
                    || described.time_column.as_ref() == Some(&column.name);
            }
        }
        Self { description, held }
    }

    pub(super) fn description(&self) -> &SourceDescription {
        &self.description
    }

    pub(super) fn held(&self) -> &HeldSchema {
        &self.held
    }

    pub(super) fn adopt(
        &mut self,
        mut answer: HeldSchema,
        declared: Option<&ColumnRef>,
    ) -> Result<()> {
        let mut assigned = declared.is_none();
        {
            let tables: BTreeMap<_, _> = self
                .held
                .tables
                .iter()
                .map(|table| (table.name.as_str(), table))
                .collect();
            for table in &mut answer.tables {
                let previous = tables
                    .get(table.name.as_str())
                    .ok_or_else(|| anyhow!("the edited schema has no prior table owner"))?;
                let columns: BTreeMap<_, _> = previous
                    .columns
                    .iter()
                    .map(|column| (column.name.as_str(), column))
                    .collect();
                for column in &mut table.columns {
                    let previous = columns
                        .get(column.name.as_str())
                        .ok_or_else(|| anyhow!("the edited schema has no prior column owner"))?;
                    // SDK edits._store keeps these host flags rather than the
                    // reset flags in contract34's actual Core answer.
                    column.untyped = previous.untyped;
                    column.stype_declared = previous.stype_declared;
                    if declared.is_some_and(|reference| {
                        reference.table == table.name && reference.column == column.name
                    }) {
                        column.stype_declared = true;
                        assigned = true;
                    }
                }
            }
        }
        if !assigned {
            return Err(anyhow!(
                "the edited declaration has no returned column owner"
            ));
        }
        let description = edits::described(&answer)
            .ok_or_else(|| anyhow!("the edited schema holds a column with no source facts"))?;
        self.description = description;
        self.held = answer;
        Ok(())
    }
}

pub(super) fn encode(held: &HeldSchema) -> Result<Vec<u8>> {
    let mut value = serde_json::to_value(held)?;
    for (at, table) in held.tables.iter().enumerate() {
        for (column_at, column) in table.columns.iter().enumerate() {
            let target = value["tables"][at]["columns"][column_at]
                .as_object_mut()
                .ok_or_else(|| anyhow!("a held column did not encode as an object"))?;
            // The SDK codec writes both actual booleans, including false.
            target.insert("untyped".into(), Value::Bool(column.untyped));
            target.insert("stype_declared".into(), Value::Bool(column.stype_declared));
        }
    }
    Ok(serde_json::to_vec(&value)?)
}

#[cfg(test)]
mod tests;
