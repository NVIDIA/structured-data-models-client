// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Explicit schema refinement over fresh observations, without changing data or sampling.
use crate::assemble::{self, HostTypes};
use sdm_contract::CoreError;
use sdm_contract::ops::{InferenceOrigin, LinkOrigin, SchemaInference, SourceDescription, Stype};
use std::collections::BTreeSet;

pub fn state(schema: &SourceDescription) -> SchemaInference {
    schema.inference.clone().unwrap_or_else(|| SchemaInference {
        unknown_tables: schema
            .tables
            .iter()
            .map(|table| table.name.clone())
            .collect(),
        links: schema
            .links
            .iter()
            .map(|link| LinkOrigin {
                link: link.clone(),
                origin: InferenceOrigin::Unknown,
            })
            .collect(),
        ..Default::default()
    })
}

pub fn retain(schema: &SourceDescription, state: &mut SchemaInference) {
    state
        .links
        .retain(|origin| schema.links.contains(&origin.link));
    for link in &schema.links {
        if !state.links.iter().any(|origin| &origin.link == link) {
            state.links.push(LinkOrigin {
                link: link.clone(),
                origin: InferenceOrigin::Unknown,
            });
        }
    }
    // Follow schema ordering, including idempotent edits and reordered relationships.
    state
        .links
        .sort_by_key(|origin| schema.links.iter().position(|link| link == &origin.link));
    state.times.retain(|name, choice| {
        schema.tables.iter().any(|table| {
            &table.name == name
                && choice
                    .column
                    .as_ref()
                    .is_none_or(|column| table.columns.iter().any(|c| &c.name == column))
        })
    });
}

pub fn settle(schema: &mut SourceDescription, state: &SchemaInference) {
    let linked: BTreeSet<_> = schema
        .links
        .iter()
        .filter_map(|link| link.get(1))
        .map(|column| column.table.clone())
        .collect();
    for table in &mut schema.tables {
        if state.time_declared.contains(&table.name) || state.unknown_tables.contains(&table.name) {
            continue;
        }
        if let Some(choice) = state.times.get(&table.name) {
            let chosen = if choice.only_unlinked && linked.contains(&table.name) {
                None
            } else {
                choice.column.clone()
            };
            if chosen
                .as_ref()
                .is_some_and(|c| table.primary_key.contains(c))
            {
                continue;
            }
            table.time_column = chosen;
        }
    }
}

pub fn infer(
    schema: &mut SourceDescription,
    state: &mut SchemaInference,
    candidate: &SourceDescription,
    links: bool,
    types: &HostTypes,
) -> Result<(), CoreError> {
    if links {
        let catalog = candidate
            .inference
            .as_ref()
            .map(|s| {
                s.links
                    .iter()
                    .filter(|v| v.origin == InferenceOrigin::Declared)
                    .map(|v| v.link.clone())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        for link in catalog
            .iter()
            .cloned()
            .chain(assemble::by_name(&schema.tables))
        {
            if schema.links.iter().any(|old| old.first() == link.first()) {
                continue;
            }
            if assemble::checked(&schema.tables, &link, types).is_err() {
                continue;
            }
            let origin = if catalog.contains(&link) {
                InferenceOrigin::Declared
            } else {
                InferenceOrigin::Inferred
            };
            schema.links.push(link.clone());
            state.links.push(LinkOrigin { link, origin });
        }
    } else {
        for table in &mut schema.tables {
            if state.unknown_tables.contains(&table.name) {
                continue;
            }
            let Some(fresh) = candidate
                .tables
                .iter()
                .find(|fresh| fresh.name == table.name)
            else {
                continue;
            };
            if table.primary_key.is_empty()
                && !state.key_declared.contains(&table.name)
                && !fresh
                    .primary_key
                    .iter()
                    .any(|key| table.time_column.as_ref() == Some(key))
                && fresh
                    .primary_key
                    .iter()
                    .all(|key| table.columns.iter().any(|column| &column.name == key))
            {
                table.primary_key.clone_from(&fresh.primary_key);
                table.duplicate_keys.clone_from(&fresh.duplicate_keys);
            }
            if table.time_column.is_none()
                && !state.time_declared.contains(&table.name)
                && let Some(choice) = candidate
                    .inference
                    .as_ref()
                    .and_then(|s| s.times.get(&table.name))
                && choice.column.as_ref().is_none_or(|name| {
                    table
                        .columns
                        .iter()
                        .any(|column| &column.name == name && column.stype == Stype::Timestamp)
                })
            {
                state.times.insert(table.name.clone(), choice.clone());
            }
        }
    }
    Ok(())
}
