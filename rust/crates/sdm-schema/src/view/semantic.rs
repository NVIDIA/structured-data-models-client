// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A semantic view's definition, as the warehouse writes it out in YAML,
//! read into a [`ViewSpec`] (design S5, T3.5's V2, V3 and V6): its logical
//! tables, each with its base table and primary key, their columns by
//! section, and the relationships, read by the paths the old client reads
//! them by (`RE/graph.py:2069-2222`).
//!
//! Only what the old client read before its first warehouse read is read
//! here: each table's name, base table and the size of its primary key, and
//! each column's and relationship's name. Where it failed with a raw
//! `KeyError` or `TypeError`, the view is refused by name. A column's
//! expression and type, a key's member, and a relationship's tables and keys
//! are read when the conversion comes to them, so one it never uses cannot
//! refuse the view.

use super::spec::{
    ColumnItem, ColumnSpec, PrimaryKey, RefusalReason, RelationshipItem, Section, Source,
    TableItem, TableSpec, ViewKind, ViewRefusal, ViewSpec,
};
use super::yaml::{self, NodeId, Value};

/// Reads a semantic view's definition.
pub fn parse(definition: &str) -> Result<ViewSpec, ViewRefusal> {
    let refusal = |reason| ViewRefusal {
        kind: ViewKind::Semantic,
        reason,
    };
    let document = yaml::load(definition).map_err(|e| refusal(RefusalReason::Yaml(e)))?;
    let root = document.root();
    let mut spec = ViewSpec {
        kind: ViewKind::Semantic,
        document,
        tables: Vec::new(),
        columns: Vec::new(),
        relationships: Vec::new(),
        filter: None,
        parameters: Vec::new(),
        measures: Vec::new(),
        ignored: Vec::new(),
        window_orders: Vec::new(),
    };
    let tables = required(&spec, root, "tables")?;
    let tables = sequence(&mut spec, tables, "tables")?;
    // The old client reads every table's name first (`:2069`).
    let mut aliases = Vec::new();
    for &table in &tables {
        let name = required(&spec, table, "name")?;
        let alias = spec
            .text(name)
            .ok_or_else(|| spec.refusal(RefusalReason::Misses("name".into())))?;
        aliases.push(alias.to_owned());
    }
    for (index, (&cfg, alias)) in tables.iter().zip(aliases).enumerate() {
        let base = required(&spec, cfg, "base_table")?;
        let source = Source::Parts {
            table: required(&spec, base, "table")?,
            database: required(&spec, base, "database")?,
            schema: required(&spec, base, "schema")?,
        };
        let primary_key = match spec.document.get(cfg, "primary_key") {
            None => PrimaryKey::None,
            Some(key) => {
                let columns = required(&spec, key, "columns")?;
                match sequence(&mut spec, columns, "columns")?.as_slice() {
                    [] => PrimaryKey::None,
                    [one] => PrimaryKey::One(*one),
                    _ => PrimaryKey::Composite,
                }
            }
        };
        spec.tables.push(TableItem::Table(TableSpec {
            alias,
            source,
            parent: None,
            join: None,
            primary_key,
            cfg,
        }));
        for (section, key) in [
            (Section::Dimension, "dimensions"),
            (Section::TimeDimension, "time_dimensions"),
            (Section::Fact, "facts"),
        ] {
            let Some(entries) = spec.document.get(cfg, key) else {
                continue;
            };
            for column in sequence(&mut spec, entries, key)? {
                let name = required(&spec, column, "name")?;
                let item = ColumnItem::Column(ColumnSpec {
                    table: Some(index),
                    section,
                    name: Some(name),
                    expr: spec
                        .document
                        .get(column, "expr")
                        .filter(|&n| !spec.document.is_null(n)),
                    data_type: spec
                        .document
                        .get(column, "data_type")
                        .filter(|&n| !spec.document.is_null(n)),
                    cfg: column,
                });
                spec.columns.push(item);
            }
        }
    }
    if let Some(relationships) = spec.document.get(root, "relationships") {
        for cfg in sequence(&mut spec, relationships, "relationships")? {
            let name = required(&spec, cfg, "name")?;
            let pairs = required(&spec, cfg, "relationship_columns")?;
            let pairs = sequence(&mut spec, pairs, "relationship_columns")?.len();
            spec.relationships
                .push(RelationshipItem { name, pairs, cfg });
        }
    }
    Ok(spec)
}

/// `cfg[entry]`, a mapping's entry that must be there.
fn required(spec: &ViewSpec, cfg: NodeId, entry: &str) -> Result<NodeId, ViewRefusal> {
    spec.document
        .get(cfg, entry)
        .ok_or_else(|| spec.refusal(RefusalReason::Misses(entry.to_owned())))
}

/// The members of a section the old client iterates, as Python iterates
/// it; a null or anything else that cannot be iterated is refused.
fn sequence(spec: &mut ViewSpec, node: NodeId, entry: &str) -> Result<Vec<NodeId>, ViewRefusal> {
    if matches!(spec.value(node), Value::Null) {
        return Err(spec.refusal(RefusalReason::NotAList(entry.to_owned())));
    }
    spec.document
        .iterate(node)
        .ok_or_else(|| spec.refusal(RefusalReason::NotAList(entry.to_owned())))
}
