// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The metric view's model (T3.5, design S5 and part 5's M2) against
//! `contract/view-metric-vectors.json`: each definition the old client's
//! `parse_metric_view` read in its own tests, and in the T3.5 cases, with
//! its whole answer.
//!
//! The model is the definition's structure, its expressions opaque text
//! (design S5); what the old client decided by reading an expression (a
//! join's source or condition, a dimension's tables, the filter) the data
//! layer decides, so here the model is held to every part of the old answer
//! that its structure settles: the two refusals of the definition itself,
//! the joins' tree and order as the old client walked it, the messages of
//! a join or dimension that is no mapping or misses a name, the
//! `materialization` and `fields` messages, the measures, the parameters
//! and the window measures' order dimensions.

use std::collections::BTreeSet;

use sdm_schema::view::metric::{self, Cardinality, DimensionItem, JoinItem, MetricView};
use sdm_schema::view::yaml::Value;
use serde_json::Value as Json;

const VECTORS: &str = include_str!("../../../../contract/view-metric-vectors.json");

fn vectors() -> Vec<Json> {
    let document: Json = serde_json::from_str(VECTORS).unwrap();
    assert_eq!(document["format"], "sdm-view-metric-vectors/1");
    document["metric_views"].as_array().unwrap().clone()
}

fn texts(value: &Json) -> Vec<String> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|item| item.as_str().unwrap().to_owned())
        .collect()
}

/// `unquote_ident` (`DB/metric_view.py:352-357`): a name in backticks,
/// unquoted, else as written.
fn unquoted(name: &str) -> String {
    let name = name.trim();
    match name
        .strip_prefix('`')
        .and_then(|rest| rest.strip_suffix('`'))
    {
        Some(inner) if name.len() >= 2 => inner.replace("``", "`"),
        _ => name.to_owned(),
    }
}

/// The joins the old client kept, walked as `_add_joins` walks them
/// (`DB/metric_view.py:825-909`): parents before children, each kept
/// join's nested joins visited, a skipped join's not. Gives each kept
/// join's name, parent and path, and the message of each join that is no
/// mapping or misses a name, in order. A join the old client skipped is
/// one it did not keep, or one named as a join kept before it.
fn walk(
    items: &[JoinItem],
    parent: &[String],
    kept: &BTreeSet<String>,
    joins: &mut Vec<(String, String, Vec<String>, Cardinality)>,
    messages: &mut Vec<String>,
) {
    for item in items {
        match item {
            JoinItem::NotAMapping { repr } => {
                messages.push(format!(
                    "Failed to add join {repr} since it is not a mapping"
                ));
            }
            JoinItem::Nameless { repr, .. } => {
                messages.push(format!("Failed to add join {repr} since it misses a name"));
            }
            JoinItem::Join(join) => {
                // A name already taken skips the join (`:849-851`).
                if !kept.contains(&join.name) || joins.iter().any(|(name, ..)| *name == join.name) {
                    continue;
                }
                let mut path = parent.to_vec();
                path.push(join.name.clone());
                let parent_alias = parent.last().cloned().unwrap_or_else(|| "source".into());
                joins.push((
                    join.name.clone(),
                    parent_alias,
                    path.clone(),
                    join.cardinality.clone(),
                ));
                walk(&join.joins, &path, kept, joins, messages);
            }
        }
    }
}

/// The messages of the dimensions the structure alone drops
/// (`DB/metric_view.py:958-996`): one that is no mapping, and one whose
/// expression is no text (one whose name is no text may still be a
/// wildcard, which the data layer reads).
fn dimension_messages(view: &MetricView) -> Vec<String> {
    view.dimensions
        .iter()
        .filter_map(|item| match item {
            DimensionItem::NotAMapping { repr } => Some(format!(
                "Failed to add dimension {repr} since it is not a mapping"
            )),
            DimensionItem::Dimension { expr, repr, .. } if !matches!(expr, Value::Str(_)) => Some(
                format!("Failed to add dimension {repr} since it misses a name or expression"),
            ),
            DimensionItem::Dimension { .. } => None,
        })
        .collect()
}

/// Whether `part` appears in `whole` in order, not necessarily together.
fn in_order(part: &[String], whole: &[String]) -> bool {
    let mut rest = whole.iter();
    part.iter().all(|wanted| rest.any(|item| item == wanted))
}

#[test]
fn each_definition_reads_as_the_old_client_read_its_structure() {
    let vectors = vectors();
    assert!(vectors.len() > 100, "{} definitions", vectors.len());
    let mut wrong = Vec::new();
    for vector in &vectors {
        let definition = vector["definition"].as_str().unwrap();
        let read = metric::parse(definition);
        let refused = vector["error"]["message"].as_str();
        // The two refusals of the definition itself; the old client's
        // others (`DB/metric_view.py:784-791`, the filter) read an
        // expression, so the structure is read and the data layer refuses.
        let own_refusal = refused.filter(|message| {
            *message == "The metric view definition is not a YAML mapping"
                || *message == "The metric view definition misses a 'source'"
        });
        let view = match (read, own_refusal) {
            (Err(refusal), Some(message)) => {
                if refusal.to_string() != message {
                    wrong.push(format!("{definition:?}: {refusal} where it said {message}"));
                }
                continue;
            }
            (Ok(_), Some(message)) => {
                wrong.push(format!("{definition:?}: read where it refused: {message}"));
                continue;
            }
            (Err(refusal), None) => {
                wrong.push(format!("{definition:?}: refused ({refusal})"));
                continue;
            }
            (Ok(view), None) => view,
        };
        let Some(spec) = vector.get("spec") else {
            continue;
        };
        let old_messages = texts(&spec["messages"]);

        // The joins the old client kept, in its order, with its parents,
        // paths and cardinalities.
        let kept: BTreeSet<String> = spec["joins"]
            .as_array()
            .unwrap()
            .iter()
            .map(|join| join["alias"].as_str().unwrap().to_owned())
            .collect();
        let (mut joins, mut join_messages) = (Vec::new(), Vec::new());
        walk(&view.joins, &[], &kept, &mut joins, &mut join_messages);
        let old_joins: Vec<_> = spec["joins"]
            .as_array()
            .unwrap()
            .iter()
            .map(|join| {
                (
                    join["alias"].as_str().unwrap().to_owned(),
                    join["parent_alias"].as_str().unwrap().to_owned(),
                    texts(&join["path"]),
                    match join["cardinality"].as_str().unwrap() {
                        "many_to_one" => Cardinality::ManyToOne,
                        "one_to_many" => Cardinality::OneToMany,
                        other => panic!("{other}"),
                    },
                )
            })
            .collect();
        if joins != old_joins {
            wrong.push(format!(
                "{definition:?}: joins {joins:?} where {old_joins:?}"
            ));
        }
        let old_join_messages: Vec<String> = old_messages
            .iter()
            .filter(|message| message.starts_with("Failed to add join "))
            .cloned()
            .collect();
        if join_messages != old_join_messages {
            wrong.push(format!(
                "{definition:?}: join messages {join_messages:?} where {old_join_messages:?}"
            ));
        }

        // The skipped joins carry their nested joins as the old client
        // collected them (`DB/metric_view.py:817-823`).
        for skipped in spec["skipped_joins"].as_array().unwrap() {
            let alias = skipped["alias"].as_str().unwrap();
            let nested = texts(&skipped["nested"]);
            let found = find(&view.joins, alias).map(|join| join.nested_names());
            if found.as_ref() != Some(&nested) {
                wrong.push(format!(
                    "{definition:?}: skipped join {alias} nests {found:?} where {nested:?}"
                ));
            }
        }

        // The messages of the sections and of the dimensions the structure
        // drops, in the old client's order.
        let mut own = Vec::new();
        if view.materialization {
            own.push(
                "Ignored the 'materialization' section since it cannot be represented in a graph"
                    .to_owned(),
            );
        }
        if view.fields_ignored {
            own.push(
                "Ignored the 'fields' section since the view also has 'dimensions', and \
                 Databricks reads only those"
                    .to_owned(),
            );
        }
        own.extend(dimension_messages(&view));
        if !in_order(&own, &old_messages) {
            wrong.push(format!(
                "{definition:?}: messages {own:?} not in {old_messages:?}"
            ));
        }
        for section in [
            "Ignored the 'materialization' section",
            "Ignored the 'fields' section",
        ] {
            let said = old_messages.iter().any(|m| m.starts_with(section));
            let says = own.iter().any(|m| m.starts_with(section));
            if said != says {
                wrong.push(format!("{definition:?}: {section}: {says} where {said}"));
            }
        }

        // The measures the old client read are the model's, in order.
        let measures: Vec<String> = view
            .measures
            .iter()
            .filter(|measure| matches!(measure.expr, Value::Str(_)))
            .map(|measure| measure.name.clone())
            .collect();
        let old_measures: Vec<String> = spec["measures"]
            .as_array()
            .unwrap()
            .iter()
            .map(|measure| measure["name"].as_str().unwrap().to_owned())
            .collect();
        if !in_order(&old_measures, &measures) {
            wrong.push(format!(
                "{definition:?}: measures {measures:?} lack {old_measures:?}"
            ));
        }

        // Each window measure's order dimension, unquoted, once.
        let mut orders: Vec<String> = Vec::new();
        for order in &view.window_orders {
            let order = unquoted(order);
            if !orders.contains(&order) {
                orders.push(order);
            }
        }
        if orders != texts(&spec["window_orders"]) {
            wrong.push(format!(
                "{definition:?}: window orders {orders:?} where {}",
                spec["window_orders"]
            ));
        }

        // Each parameter read as its default is a parameter of the model.
        let parameters: BTreeSet<String> = view.parameters.iter().map(|p| p.name.clone()).collect();
        for note in spec["notes"].as_array().unwrap() {
            let name = note[0].as_str().unwrap();
            if !parameters.contains(name) {
                wrong.push(format!("{definition:?}: no parameter {name}"));
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "{} differences:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

/// The join named `alias`, wherever it is nested.
fn find<'a>(items: &'a [JoinItem], alias: &str) -> Option<&'a metric::Join> {
    items.iter().find_map(|item| match item {
        JoinItem::Join(join) if join.name == alias => Some(join),
        JoinItem::Join(join) => find(&join.joins, alias),
        JoinItem::Nameless { joins, .. } => find(joins, alias),
        JoinItem::NotAMapping { .. } => None,
    })
}

/// The structure the data layer reads further, kept as written: a join's
/// source, `on` (or the key `true` a bare `on:` is, row 31), `using` and
/// cardinality; the filter; a parameter's type and default as YAML 1.1
/// values.
#[test]
fn a_join_keeps_what_it_says_as_written() {
    let view = metric::parse(
        "source: c.s.orders\n\
         filter: source.price > 50 -- a comment\n\
         parameters:\n\
         \x20 - name: since\n\
         \x20   data_type: DATE\n\
         \x20   default: 2024-01-01\n\
         \x20 - name: flag\n\
         \x20   default: yes\n\
         joins:\n\
         \x20 - name: users\n\
         \x20   source: c.s.users\n\
         \x20   on: source.user_id = users.user_id\n\
         \x20   joins:\n\
         \x20     - name: countries\n\
         \x20       source: c.s.countries\n\
         \x20       using: country\n\
         \x20       cardinality: one_to_many\n\
         \x20 - name: shops\n\
         \x20   source: c.s.shops\n\
         \x20   using: [a, b]\n\
         \x20   cardinality: many_to_many\n",
    )
    .unwrap();
    assert_eq!(view.source, Value::Str("c.s.orders".into()));
    assert_eq!(
        view.filter,
        Some(Value::Str("source.price > 50 -- a comment".into()))
    );
    assert_eq!(view.parameters.len(), 2);
    assert_eq!(view.parameters[0].data_type, Value::Str("DATE".into()));
    assert!(matches!(view.parameters[0].default, Value::Date(_)));
    assert_eq!(view.parameters[1].default, Value::Bool(true));
    let [JoinItem::Join(users), JoinItem::Join(shops)] = view.joins.as_slice() else {
        panic!("{:?}", view.joins)
    };
    assert_eq!(users.source, Value::Str("c.s.users".into()));
    assert_eq!(
        users.keys,
        metric::JoinKeys::On(Value::Str("source.user_id = users.user_id".into()))
    );
    assert_eq!(users.cardinality, Cardinality::ManyToOne);
    assert_eq!(users.nested_names(), vec!["countries".to_owned()]);
    let [JoinItem::Join(countries)] = users.joins.as_slice() else {
        panic!("{:?}", users.joins)
    };
    assert_eq!(countries.keys, metric::JoinKeys::Using("country".into()));
    assert_eq!(countries.cardinality, Cardinality::OneToMany);
    assert_eq!(shops.keys, metric::JoinKeys::UsingSeveral);
    assert_eq!(
        shops.cardinality,
        Cardinality::Unsupported("many_to_many".into())
    );
}
