// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A metric view's definition and its joins (T3.5, design S5 and part 5's
//! M2) against `contract/view-metric-vectors.json`: each definition the old
//! client's `parse_metric_view` read in its own tests, in the T3.5 cases and
//! in the first review's cases, with its whole answer.
//!
//! The joins are walked as the core walks them (`view_steps::walk_joins`:
//! the schema's order, the data layer's reading of each source and
//! condition), and held to the old answer whole: each join kept, with its
//! parent, path, cardinality, table and keys; each join skipped, with its
//! reason and the joins nested in it; and the message of each join that is
//! no mapping or has no name. The definition's own refusals, the source's,
//! the sections it ignores, the dimensions it cannot read, the measures,
//! the parameters and the window measures' order dimensions are held too.
//! The messages quote a value as the core writes it (`view_words`).

use std::collections::BTreeSet;

use sdm_contract::CoreError;
use sdm_contract::data::{DataLayer, ProviderChoice};
use sdm_contract::view::{ViewTextAnswer, ViewTextAsk};
use sdm_core::pql::{spaces, yaml_rules};
use sdm_core::view::definition::{self as view_definition, Dimension};
use sdm_core::view_steps::{self, JoinJudgement};
use sdm_core::view_words;
use sdm_data::Data;
use sdm_data::view_expr::metric as data;
use sdm_data::view_expr::strip_comments;
use sdm_schema::view::joins::JoinWalk;
use sdm_schema::view::metric;
use sdm_schema::view::spec::{
    Cardinality, ColumnItem, JoinKeys, RefusalReason, Source, TableItem, ViewRefusal, ViewSpec,
};
use sdm_schema::view::yaml::{Outside, Value, YamlError};
use serde_json::Value as Json;

/// The data layer's answers in Databricks's dialect, as the routine asks
/// them (`DataLayer::view_text`).
fn ask(question: ViewTextAsk) -> Result<ViewTextAnswer, CoreError> {
    Data::reading_views(spaces()).view_text(&ProviderChoice::sql("databricks"), question)
}

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

/// `unquote_ident`: a name in backticks, unquoted, else as written.
fn unquoted(name: &str) -> String {
    data::unquote(name, spaces())
}

/// Whether `part` appears in `whole` in order, not necessarily together.
fn in_order(part: &[String], whole: &[String]) -> bool {
    let mut rest = whole.iter();
    part.iter().all(|wanted| rest.any(|item| item == wanted))
}

/// Each kept join as the old answer writes it.
fn kept_joins(spec: &ViewSpec, walk: &JoinWalk<JoinJudgement>) -> Vec<Json> {
    walk.kept
        .iter()
        .map(|&at| {
            let TableItem::Table(table) = &spec.tables[at] else {
                unreachable!()
            };
            let mut path = vec![table.alias.clone()];
            let mut parent = table.parent;
            while let Some(index) = parent.filter(|&p| p > 0) {
                let TableItem::Table(up) = &spec.tables[index] else {
                    unreachable!()
                };
                path.insert(0, up.alias.clone());
                parent = up.parent;
            }
            let parent_alias = if path.len() > 1 {
                path[path.len() - 2].clone()
            } else {
                metric::SOURCE_ALIAS.to_owned()
            };
            let join = table.join.as_ref().unwrap();
            let Source::Name(Some(source)) = table.source else {
                unreachable!()
            };
            let parts = data::table_reference(spec.text(source).unwrap(), spaces()).unwrap();
            let condition = match join.keys {
                JoinKeys::Using(column) => {
                    let column = unquoted(&view_words::value_text(&spec.document, column).unwrap());
                    serde_json::json!([
                        {"qualifier": parent_alias, "column": column},
                        {"qualifier": table.alias, "column": column},
                    ])
                }
                JoinKeys::On(Some(on)) => {
                    let on = strip_comments(spec.text(on).unwrap(), spaces()).unwrap();
                    let [a, b] = data::join_condition(&on, spaces()).unwrap();
                    serde_json::json!([
                        {"qualifier": a.qualifier, "column": a.column},
                        {"qualifier": b.qualifier, "column": b.column},
                    ])
                }
                _ => unreachable!(),
            };
            serde_json::json!({
                "alias": table.alias,
                "table": parts,
                "parent_alias": parent_alias,
                "path": path,
                "condition": condition,
                "cardinality": match join.cardinality {
                    Cardinality::ManyToOne => "many_to_one",
                    Cardinality::OneToMany => "one_to_many",
                    Cardinality::Unsupported(_) | Cardinality::Unhashable => unreachable!(),
                },
            })
        })
        .collect()
}

/// The messages of the dimensions the structure alone drops: one that is
/// no mapping, and one whose expression is no text.
fn dimension_messages(spec: &ViewSpec) -> Vec<String> {
    spec.columns
        .iter()
        .filter_map(|item| match item {
            ColumnItem::NotAMapping { cfg } => {
                Some(view_words::dimension_not_a_mapping(spec, *cfg).unwrap())
            }
            ColumnItem::Column(column) if column.expr.and_then(|e| spec.text(e)).is_none() => {
                Some(view_words::dimension_misses_text(spec, column.cfg).unwrap())
            }
            ColumnItem::Column(_) => None,
        })
        .collect()
}

#[test]
fn each_definition_reads_as_the_old_client_read_it() {
    let vectors = vectors();
    assert!(vectors.len() > 140, "{} definitions", vectors.len());
    let mut wrong = Vec::new();
    for vector in &vectors {
        let definition = vector["definition"].as_str().unwrap();
        let refused = vector["error"]["message"].as_str();
        let mut spec = match metric::parse(definition, &yaml_rules()) {
            Ok(spec) => spec,
            // YAML outside what view definitions use, refused by name
            // where the old client read it (register P3J-9).
            Err(ViewRefusal {
                reason:
                    RefusalReason::Yaml(YamlError {
                        outside: Some(what),
                        ..
                    }),
                ..
            }) => {
                let written = match what {
                    Outside::Tag(_) => definition.contains('!'),
                    Outside::Merge => definition.contains("<<"),
                    _ => true,
                };
                if !written {
                    wrong.push(format!("{definition:?}: refused as outside ({what})"));
                }
                continue;
            }
            // A text PyYAML refused too, in its own words (P3J-1).
            Err(ViewRefusal {
                reason: RefusalReason::Yaml(_),
                ..
            }) if vector["error"]["type"]
                .as_str()
                .is_some_and(|class| class.ends_with("Error") && class != "ValueError") =>
            {
                continue;
            }
            Err(refusal) => {
                if Some(refusal.to_string().as_str()) != refused {
                    wrong.push(format!("{definition:?}: {refusal} where {refused:?}"));
                }
                continue;
            }
        };
        // The source, which the old client parses before anything else.
        if let Err(node) = view_steps::fact_source(&spec, &ask).unwrap() {
            let text = view_words::unsupported_source_text(&spec, node.unwrap()).unwrap();
            if Some(text.as_str()) != refused {
                wrong.push(format!("{definition:?}: {text} where {refused:?}"));
            }
            continue;
        }
        let walk = match view_steps::walk_joins(&mut spec, &ask).unwrap() {
            Ok(walk) => walk,
            Err(refusal) => {
                wrong.push(format!("{definition:?}: joins refused ({refusal})"));
                continue;
            }
        };
        let Some(old) = vector.get("spec") else {
            // Refused later by the old client, for its filter: the data
            // layer's step, held by sdm-data's and view_words' tests.
            continue;
        };
        let old_messages = texts(&old["messages"]);

        let kept = kept_joins(&spec, &walk);
        if Json::Array(kept.clone()) != old["joins"] {
            wrong.push(format!(
                "{definition:?}: joins {kept:?} where {}",
                old["joins"]
            ));
        }
        let skipped: Vec<Json> = walk
            .skipped
            .iter()
            .map(|s| {
                serde_json::json!({
                    "alias": s.alias,
                    "reason": view_words::skip_reason_text(&spec, &s.alias, &s.reason).unwrap(),
                    "nested": s.nested,
                })
            })
            .collect();
        let old_skipped: Vec<Json> = old["skipped_joins"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| serde_json::json!({"alias": s["alias"], "reason": s["reason"], "nested": s["nested"]}))
            .collect();
        if skipped != old_skipped {
            wrong.push(format!(
                "{definition:?}: skipped {skipped:?} where {old_skipped:?}"
            ));
        }
        let join_messages: Vec<String> = walk
            .messages
            .iter()
            .map(|m| view_words::join_message_text(&spec, m).unwrap())
            .collect();
        let old_join_messages: Vec<String> = old_messages
            .iter()
            .filter(|m| m.starts_with("Failed to add join "))
            .cloned()
            .collect();
        if join_messages != old_join_messages {
            wrong.push(format!(
                "{definition:?}: {join_messages:?} where {old_join_messages:?}"
            ));
        }

        let mut own: Vec<String> = spec
            .ignored
            .iter()
            .map(|i| view_words::ignored_text(*i))
            .collect();
        own.extend(dimension_messages(&spec));
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

        let measures: Vec<String> = spec
            .measures
            .iter()
            .filter(|m| m.expr.is_some_and(|e| spec.text(e).is_some()))
            .map(|m| m.name.clone())
            .collect();
        let old_measures: Vec<String> = old["measures"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m["name"].as_str().unwrap().to_owned())
            .collect();
        if !in_order(&old_measures, &measures) {
            wrong.push(format!(
                "{definition:?}: measures {measures:?} lack {old_measures:?}"
            ));
        }

        let mut orders: Vec<String> = Vec::new();
        for order in &spec.window_orders {
            let order = unquoted(order);
            if !orders.contains(&order) {
                orders.push(order);
            }
        }
        if orders != texts(&old["window_orders"]) {
            wrong.push(format!(
                "{definition:?}: orders {orders:?} where {}",
                old["window_orders"]
            ));
        }

        let parameters: BTreeSet<String> = spec.parameters.iter().map(|p| p.name.clone()).collect();
        // Each parameter read as its default, with its default's text, the
        // literal's (Codex review 2, P0-1).
        let read = view_steps::parameters(&spec, &[], &ask).unwrap();
        for note in old["notes"].as_array().unwrap() {
            let name = note[0].as_str().unwrap();
            if !parameters.contains(name) {
                wrong.push(format!("{definition:?}: no parameter {name}"));
            }
            let text = read
                .iter()
                .find(|(folded, _)| *folded == name.to_lowercase())
                .and_then(|(_, parameter)| parameter.read.as_ref().ok())
                .map(|read| read.text.as_str());
            if text != note[1].as_str() {
                wrong.push(format!(
                    "{definition:?}: {name} read as {text:?} where {}",
                    note[1]
                ));
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

/// The structure the data layer reads further, kept as written: a join's
/// source, `on` (or the key `true` a bare `on:` is, row 31), `using` and
/// cardinality; the filter; a parameter's type and default.
#[test]
fn a_join_keeps_what_it_says_as_written() {
    let spec = metric::parse(
        "source: c.s.orders\n\
         filter: source.price > 50 -- a comment\n\
         parameters:\n\
         \x20 - name: since\n\
         \x20   data_type: DATE\n\
         \x20   default: 2024-01-01\n\
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
        &yaml_rules(),
    )
    .unwrap();
    assert_eq!(
        spec.text(spec.filter.unwrap()),
        Some("source.price > 50 -- a comment")
    );
    assert!(matches!(
        spec.value(spec.parameters[0].default.unwrap()),
        Value::Date(_)
    ));
    let tables: Vec<&str> = spec
        .tables
        .iter()
        .map(|t| match t {
            TableItem::Table(t) => t.alias.as_str(),
            _ => "?",
        })
        .collect();
    assert_eq!(tables, ["source", "users", "countries", "shops"]);
    let TableItem::Table(countries) = &spec.tables[2] else {
        panic!()
    };
    assert_eq!(countries.parent, Some(1));
    let join = countries.join.as_ref().unwrap();
    assert_eq!(join.cardinality, Cardinality::OneToMany);
    let JoinKeys::Using(column) = join.keys else {
        panic!()
    };
    assert_eq!(spec.text(column), Some("country"));
    let TableItem::Table(shops) = &spec.tables[3] else {
        panic!()
    };
    assert_eq!(shops.join.as_ref().unwrap().keys, JoinKeys::UsingSeveral);
}

/// Each dimension as the old answer writes it.
fn dimension_json(dimension: &Dimension) -> Json {
    match dimension {
        Dimension::Wildcard { alias, excluded } => {
            serde_json::json!({"wildcard": alias, "excluded": excluded})
        }
        Dimension::Column { name, alias, expr } => {
            serde_json::json!({"name": name, "alias": alias, "expr": expr})
        }
    }
}

/// The definition read whole as the routine reads it (`view::definition`,
/// stage 2): its parameters read with the types the warehouse described,
/// a source column of a parameter's name taking it out, then the joins,
/// the filter, the dimensions and the measures, each with what it says, as
/// the old client's `parse_metric_view` answered.
#[test]
fn each_definition_reads_whole_as_the_old_client_read_it() {
    let mut wrong = Vec::new();
    let mut read_whole = 0;
    for vector in &vectors() {
        let definition = vector["definition"].as_str().unwrap();
        let Ok(mut spec) = metric::parse(definition, &yaml_rules()) else {
            continue;
        };
        if view_steps::fact_source(&spec, &ask).unwrap().is_err() {
            continue;
        }
        let described: Vec<(String, String)> = vector["parameter_types"]
            .as_object()
            .map(|types| {
                types
                    .iter()
                    .map(|(name, t)| (name.clone(), t.as_str().unwrap().to_owned()))
                    .collect()
            })
            .unwrap_or_default();
        let mut parameters = view_steps::parameters(&spec, &described, &ask).unwrap();
        if !parameters.is_empty() {
            let columns = if vector["source_columns"].is_array() {
                texts(&vector["source_columns"])
            } else {
                Vec::new()
            };
            parameters.retain(|(folded, _)| !columns.iter().any(|c| c.to_lowercase() == *folded));
        }
        let read = match view_definition::read(&mut spec, &parameters, &ask, "databricks") {
            Ok(read) => read,
            Err(error) => {
                if vector.get("spec").is_some() {
                    wrong.push(format!("{definition:?}: refused {error:?}"));
                }
                continue;
            }
        };
        let read = match read {
            Ok(read) => read,
            Err(refused) => {
                let said = refused.error().message;
                if vector["error"]["type"] != "UnsupportedFilterError"
                    || vector["error"]["message"].as_str() != Some(said.as_str())
                {
                    wrong.push(format!("{definition:?}: {said} where {}", vector["error"]));
                }
                continue;
            }
        };
        let Some(old) = vector.get("spec") else {
            wrong.push(format!("{definition:?}: read where {}", vector["error"]));
            continue;
        };
        read_whole += 1;
        let columns: Vec<Json> = read.dimensions.iter().map(dimension_json).collect();
        if Json::Array(columns.clone()) != old["columns"] {
            wrong.push(format!(
                "{definition:?}: columns {columns:?} where {}",
                old["columns"]
            ));
        }
        if read.filter.as_deref() != old["filter"].as_str() {
            wrong.push(format!(
                "{definition:?}: filter {:?} where {}",
                read.filter, old["filter"]
            ));
        }
        let notes: Vec<Json> = read
            .notes
            .iter()
            .map(|(n, t)| serde_json::json!([n, t]))
            .collect();
        if Json::Array(notes.clone()) != old["notes"] {
            wrong.push(format!(
                "{definition:?}: notes {notes:?} where {}",
                old["notes"]
            ));
        }
        let measures: Vec<Json> = read
            .measures
            .iter()
            .map(|m| {
                serde_json::json!({
                    "name": m.name,
                    "inputs": m.inputs.iter().map(|(a, c)| vec![a, c]).collect::<Vec<_>>(),
                    "aggregation": m.aggregation.as_ref().map(|(f, c)| serde_json::json!([f, c])),
                })
            })
            .collect();
        if Json::Array(measures.clone()) != old["measures"] {
            wrong.push(format!(
                "{definition:?}: measures {measures:?} where {}",
                old["measures"]
            ));
        }
        let messages: Vec<String> = read.messages.iter().map(|m| m.message.clone()).collect();
        if messages != texts(&old["messages"]) {
            wrong.push(format!(
                "{definition:?}: messages {messages:?} where {}",
                old["messages"]
            ));
        }
        let skipped: Vec<Json> = read
            .skipped
            .iter()
            .map(|s| {
                serde_json::json!({
                    "alias": s.alias, "reason": s.reason, "nested": s.nested,
                    "dimensions": s.dimensions,
                    "message": view_definition::skipped_warning(s).message,
                })
            })
            .collect();
        let old_skipped: Vec<Json> = old["skipped_joins"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| {
                serde_json::json!({
                    "alias": s["alias"], "reason": s["reason"], "nested": s["nested"],
                    "dimensions": s["dimensions"], "message": s["message"],
                })
            })
            .collect();
        if skipped != old_skipped {
            wrong.push(format!(
                "{definition:?}: skipped {skipped:?} where {old_skipped:?}"
            ));
        }
        if read.window_orders != texts(&old["window_orders"]) {
            wrong.push(format!(
                "{definition:?}: orders {:?} where {}",
                read.window_orders, old["window_orders"]
            ));
        }
    }
    assert!(read_whole > 140, "{read_whole} definitions read whole");
    assert!(
        wrong.is_empty(),
        "{} differences:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}
