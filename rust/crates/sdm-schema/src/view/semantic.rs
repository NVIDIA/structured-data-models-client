// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A semantic view's definition, as the warehouse writes it out in YAML,
//! read into a [`ViewSpec`] (design S5, T3.5's V2, V3 and V6), in the old
//! client's order (`RE/graph.py:2069-2222`).
//!
//! [`parse`] reads only what the old client read before anything else:
//! the tables, and every table's name (`:2069`). The rest is read when the
//! conversion comes to it, between its warehouse reads: a table's base
//! table, primary key and the names its patterns are made of ([`table`]),
//! then its columns, one by one ([`columns`]), and after every table the
//! relationships ([`relationships`]), each relationship's tables and keys
//! only for one pair ([`ViewSpec::relationship`]). So a later table's
//! missing entry cannot refuse the view before an earlier table is read,
//! and an entry the conversion never reaches cannot refuse it at all.
//! Where the old client failed with a raw `KeyError` or `TypeError`, the
//! view is refused by name. [`read_all`] reads everything in that order,
//! for a caller with nothing to do in between.

use super::spec::{
    ColumnItem, ColumnSpec, PrimaryKey, RefusalReason, RelationshipItem, Section, Source,
    TableItem, TableSpec, ViewKind, ViewRefusal, ViewSpec,
};
use super::yaml::{self, NodeId, Value};

/// Reads a semantic view's definition: its tables and their names.
pub fn parse(definition: &str, rules: &yaml::Rules) -> Result<ViewSpec, ViewRefusal> {
    let refusal = |reason| ViewRefusal {
        kind: ViewKind::Semantic,
        reason,
    };
    let document = yaml::load(definition, rules).map_err(|e| refusal(RefusalReason::Yaml(e)))?;
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
    let tables = spec.required(root, "tables")?;
    // `{table_cfg['name'] for table_cfg in cfg['tables']}`: each name must
    // be there, and be a set's member.
    for cfg in sequence(&mut spec, tables, "tables")? {
        let name = spec.required(cfg, "name")?;
        if matches!(spec.value(name), Value::Seq(_) | Value::Map(_)) {
            return Err(spec.refusal(RefusalReason::Unhashable("name".into())));
        }
        spec.tables.push(TableItem::Declared { name, cfg });
    }
    Ok(spec)
}

/// The table at `index` of a parsed semantic view, read in the old
/// client's order (`:2071-2096`): its base table's `table`, `database` and
/// `schema`, its primary key (`len` and `[0]` as Python takes them), and
/// the names its patterns are made of, its own and every other table's,
/// which must be text. The table as already read for one that is.
pub fn table(spec: &mut ViewSpec, index: usize) -> Result<TableSpec, ViewRefusal> {
    let (name, cfg) = match spec.tables.get(index) {
        Some(TableItem::Declared { name, cfg }) => (*name, *cfg),
        Some(TableItem::Table(table)) => return Ok(table.clone()),
        _ => return Err(spec.refusal(RefusalReason::Misses("tables".into()))),
    };
    let base = spec.required(cfg, "base_table")?;
    let source = Source::Parts {
        table: spec.required(base, "table")?,
        database: spec.required(base, "database")?,
        schema: spec.required(base, "schema")?,
    };
    let primary_key = match spec.document.get(cfg, "primary_key") {
        None => PrimaryKey::None,
        Some(key) => {
            let columns = spec.required(key, "columns")?;
            match spec.document.len(columns) {
                None => return Err(spec.refusal(RefusalReason::NotAList("columns".into()))),
                Some(0) => PrimaryKey::None,
                Some(1) => PrimaryKey::One(
                    spec.document
                        .first(columns)
                        .ok_or_else(|| spec.refusal(RefusalReason::Misses("columns".into())))?,
                ),
                Some(_) => PrimaryKey::Composite,
            }
        }
    };
    // The patterns of its own name and of every other's (`:2087-2092`).
    let names: Vec<NodeId> = spec
        .tables
        .iter()
        .filter_map(|item| match item {
            TableItem::Declared { name, .. } => Some(*name),
            _ => None,
        })
        .chain(std::iter::once(name))
        .collect();
    if names.iter().any(|&n| spec.text(n).is_none()) {
        return Err(spec.refusal(RefusalReason::NotText("name".into())));
    }
    let alias = spec.text(name).unwrap_or_default().to_owned();
    Ok(TableSpec {
        alias,
        source,
        parent: None,
        join: None,
        primary_key,
        cfg,
    })
}

/// The columns of the table at `index` (`:2109-2114`): its `dimensions`,
/// `time_dimensions` and `facts`, each column's name required, in order,
/// ending at the first refusal, which comes when the old client came to
/// it.
pub fn columns(spec: &mut ViewSpec, index: usize) -> Vec<Result<ColumnItem, ViewRefusal>> {
    let cfg = match spec.tables.get(index) {
        Some(TableItem::Declared { cfg, .. }) => *cfg,
        Some(TableItem::Table(table)) => table.cfg,
        _ => return Vec::new(),
    };
    let mut read = Vec::new();
    for (section, key) in [
        (Section::Dimension, "dimensions"),
        (Section::TimeDimension, "time_dimensions"),
        (Section::Fact, "facts"),
    ] {
        let Some(entries) = spec.document.get(cfg, key) else {
            continue;
        };
        let entries = match sequence(spec, entries, key) {
            Ok(entries) => entries,
            Err(refusal) => {
                read.push(Err(refusal));
                return read;
            }
        };
        for column in entries {
            let name = match spec.required(column, "name") {
                Ok(name) => name,
                Err(refusal) => {
                    read.push(Err(refusal));
                    return read;
                }
            };
            let present = |entry: &str| {
                spec.document
                    .get(column, entry)
                    .filter(|&n| !spec.document.is_null(n))
            };
            read.push(Ok(ColumnItem::Column(ColumnSpec {
                table: Some(index),
                section,
                name: Some(name),
                expr: present("expr"),
                data_type: present("data_type"),
                cfg: column,
            })));
        }
    }
    read
}

/// The relationships (`:2200-2206`), each with its name and how many
/// column pairs it has, in order, ending at the first refusal.
pub fn relationships(spec: &mut ViewSpec) -> Vec<Result<RelationshipItem, ViewRefusal>> {
    let root = spec.document.root();
    let Some(section) = spec.document.get(root, "relationships") else {
        return Vec::new();
    };
    let entries = match sequence(spec, section, "relationships") {
        Ok(entries) => entries,
        Err(refusal) => return vec![Err(refusal)],
    };
    let mut read = Vec::new();
    for cfg in entries {
        let item = spec.required(cfg, "name").and_then(|name| {
            let pairs = spec.required(cfg, "relationship_columns")?;
            let pairs = spec.document.len(pairs).ok_or_else(|| {
                spec.refusal(RefusalReason::NotAList("relationship_columns".into()))
            })?;
            Ok(RelationshipItem { name, pairs, cfg })
        });
        let refused = item.is_err();
        read.push(item);
        if refused {
            break;
        }
    }
    read
}

/// Reads every table, its columns, and the relationships, in the old
/// client's order, into `spec`.
pub fn read_all(spec: &mut ViewSpec) -> Result<(), ViewRefusal> {
    for index in 0..spec.tables.len() {
        let table = table(spec, index)?;
        let columns = columns(spec, index);
        spec.tables[index] = TableItem::Table(table);
        for column in columns {
            spec.columns.push(column?);
        }
    }
    for relationship in relationships(spec) {
        spec.relationships.push(relationship?);
    }
    Ok(())
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
