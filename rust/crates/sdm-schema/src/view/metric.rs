// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A metric view's definition read into a [`ViewSpec`] (design S5, T3.5's
//! M2): its source, parameters, joins with the joins nested in them, filter,
//! dimensions and measures, every expression and name kept as written.
//!
//! This is the part of the old client's `parse_metric_view`
//! (`DB/metric_view.py:727-1082`) that reads the YAML's structure; what it
//! decided by reading an expression or a name, the data layer decides, and
//! the core asks in the old client's order ([`super::joins`] walks the
//! joins). A section is read as the old client iterates it (`for x in
//! section or []`), and an entry that is no mapping or misses a name is
//! kept in its place, with its value for the message. A nameless join's
//! nested joins are not read here: the old client reads them only when a
//! skipped join above it names them.

use super::spec::{
    Cardinality, ColumnItem, ColumnSpec, Ignored, JoinKeys, JoinSpec, MeasureItem, ParameterItem,
    PrimaryKey, RefusalReason, Section, Source, TableItem, TableSpec, ViewKind, ViewRefusal,
    ViewSpec,
};
use super::yaml::{self, NodeId, Value};

/// The alias the source table goes by in a metric view.
pub const SOURCE_ALIAS: &str = "source";

/// Reads a metric view's definition (`DB/metric_view.py:773-1082`).
pub fn parse(definition: &str) -> Result<ViewSpec, ViewRefusal> {
    let refusal = |reason| ViewRefusal {
        kind: ViewKind::Metric,
        reason,
    };
    let document = yaml::load(definition).map_err(|e| refusal(RefusalReason::Yaml(e)))?;
    let root = document.root();
    if !matches!(document.value(root), Value::Map(_)) {
        return Err(refusal(RefusalReason::NotAMapping));
    }
    let source = document
        .get(root, "source")
        .filter(|&n| !document.is_null(n))
        .ok_or_else(|| refusal(RefusalReason::MissesSource))?;
    let mut spec = ViewSpec {
        kind: ViewKind::Metric,
        document,
        tables: vec![TableItem::Table(TableSpec {
            alias: SOURCE_ALIAS.to_owned(),
            source: Source::Name(Some(source)),
            parent: None,
            join: None,
            primary_key: PrimaryKey::None,
            cfg: root,
        })],
        columns: Vec::new(),
        relationships: Vec::new(),
        filter: None,
        parameters: Vec::new(),
        measures: Vec::new(),
        ignored: Vec::new(),
        window_orders: Vec::new(),
    };
    if spec.document.has(root, "materialization") {
        spec.ignored.push(Ignored::Materialization);
    }
    spec.parameters = parameters(&spec, spec.document.get(root, "parameters"));
    let joins = spec.document.get(root, "joins");
    add_joins(&mut spec, joins, 0, &mut Vec::new())?;
    spec.filter = spec
        .document
        .get(root, "filter")
        .filter(|&n| !spec.document.is_null(n));
    let dimensions = spec
        .document
        .get(root, "dimensions")
        .filter(|&n| spec.document.is_set(n));
    let fields = spec
        .document
        .get(root, "fields")
        .filter(|&n| spec.document.is_set(n));
    let (section, columns) = match (dimensions, fields) {
        (Some(dimensions), _) => ("dimensions", Some(dimensions)),
        (None, fields) => ("fields", fields),
    };
    if dimensions.is_some() && fields.is_some() {
        spec.ignored.push(Ignored::Fields);
    }
    for entry in spec.entries(columns, section)? {
        let item = if matches!(spec.value(entry), Value::Map(_)) {
            ColumnItem::Column(ColumnSpec {
                table: None,
                section: Section::Dimension,
                name: present(&spec, entry, "name"),
                expr: present(&spec, entry, "expr"),
                data_type: None,
                cfg: entry,
            })
        } else {
            ColumnItem::NotAMapping { cfg: entry }
        };
        spec.columns.push(item);
    }
    let measures = spec.document.get(root, "measures");
    spec.measures = measure_items(&spec, measures);
    spec.window_orders = window_orders(&spec, measures);
    Ok(spec)
}

/// `cfg.get(key)`, unless absent or null.
fn present(spec: &ViewSpec, cfg: NodeId, key: &str) -> Option<NodeId> {
    spec.document
        .get(cfg, key)
        .filter(|&n| !spec.document.is_null(n))
}

fn list(spec: &ViewSpec, section: Option<NodeId>) -> Vec<NodeId> {
    match section.map(|n| spec.value(n)) {
        Some(Value::Seq(items)) => items.clone(),
        _ => Vec::new(),
    }
}

/// Each entry of `parameters` that is a mapping with a text name, in
/// order, a name given twice included (`:569-605`).
fn parameters(spec: &ViewSpec, section: Option<NodeId>) -> Vec<ParameterItem> {
    list(spec, section)
        .into_iter()
        .filter_map(|cfg| {
            let name = spec.text(spec.document.get(cfg, "name")?)?;
            Some(ParameterItem {
                name: name.to_owned(),
                data_type: present(spec, cfg, "data_type"),
                default: present(spec, cfg, "default"),
            })
        })
        .collect()
}

/// The joins under the table at `parent`, each before the joins nested in
/// it (`_add_joins`, `:825-909`): a join that is no mapping or has no text
/// name is kept in its place, its nested joins unread.
fn add_joins(
    spec: &mut ViewSpec,
    section: Option<NodeId>,
    parent: usize,
    visiting: &mut Vec<NodeId>,
) -> Result<(), ViewRefusal> {
    for cfg in spec.entries(section, "joins")? {
        if !matches!(spec.value(cfg), Value::Map(_)) {
            spec.tables.push(TableItem::NotAMapping { cfg, parent });
            continue;
        }
        let Some(name) = spec
            .document
            .get(cfg, "name")
            .and_then(|n| spec.text(n))
            .map(str::to_owned)
        else {
            spec.tables.push(TableItem::Nameless { cfg, parent });
            continue;
        };
        if visiting.contains(&cfg) {
            return Err(spec.refusal(RefusalReason::NestsItself));
        }
        let cardinality = match spec.document.get(cfg, "cardinality") {
            None => Cardinality::ManyToOne,
            Some(node) => match spec.text(node) {
                Some("many_to_one") => Cardinality::ManyToOne,
                Some("one_to_many") => Cardinality::OneToMany,
                _ => Cardinality::Unsupported(node),
            },
        };
        let keys = match present(spec, cfg, "using") {
            None => JoinKeys::On(
                spec.document
                    .get(cfg, "on")
                    .or_else(|| spec.document.get_true(cfg)),
            ),
            Some(using) => match spec.value(using) {
                Value::Str(_) => JoinKeys::Using(using),
                Value::Seq(columns) if columns.len() == 1 => JoinKeys::Using(columns[0]),
                _ => JoinKeys::UsingSeveral,
            },
        };
        let source = spec.document.get(cfg, "source");
        spec.tables.push(TableItem::Table(TableSpec {
            alias: name,
            source: Source::Name(source),
            parent: Some(parent),
            join: Some(JoinSpec { cardinality, keys }),
            primary_key: PrimaryKey::None,
            cfg,
        }));
        let at = spec.tables.len() - 1;
        visiting.push(cfg);
        let nested = spec.document.get(cfg, "joins");
        add_joins(spec, nested, at, visiting)?;
        visiting.pop();
    }
    Ok(())
}

/// Each entry of `measures` that is a mapping with a text name, in order
/// (`:1111-1126`).
fn measure_items(spec: &ViewSpec, section: Option<NodeId>) -> Vec<MeasureItem> {
    list(spec, section)
        .into_iter()
        .filter_map(|cfg| {
            let name = spec.text(spec.document.get(cfg, "name")?)?;
            Some(MeasureItem {
                name: name.to_owned(),
                expr: present(spec, cfg, "expr"),
                window: spec.document.has(cfg, "window"),
            })
        })
        .collect()
}

/// Each window's `order` text, a measure's `window` being one mapping or a
/// list of them (`_window_orders`, `:1222-1235`).
fn window_orders(spec: &ViewSpec, section: Option<NodeId>) -> Vec<String> {
    let mut orders = Vec::new();
    for cfg in list(spec, section) {
        let windows = match spec.document.get(cfg, "window").map(|w| (w, spec.value(w))) {
            Some((window, Value::Map(_))) => vec![window],
            Some((_, Value::Seq(windows))) => windows.clone(),
            _ => Vec::new(),
        };
        for window in windows {
            if let Some(order) = spec
                .document
                .get(window, "order")
                .and_then(|o| spec.text(o))
            {
                orders.push(order.to_owned());
            }
        }
    }
    orders
}
