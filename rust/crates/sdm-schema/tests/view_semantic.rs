// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The semantic view's [`ViewSpec`] (T3.5, design part 5's V2, V3 and V6)
//! against the semantic-view definitions of `contract/view-yaml-vectors.json`,
//! as PyYAML read them: each logical table's name, base table and primary
//! key, its columns from `dimensions`, `time_dimensions` and `facts` in that
//! order, and each relationship, read by the paths the old client reads them
//! by (`RE/graph.py:2069-2222`).
//!
//! From T3.5 fix 1 (Codex review 1, P1-9), an entry the old client reads
//! only on the branch that uses it is read there: a composite relationship's
//! tables, a physical column's type, a composite key's members. One it never
//! reads cannot refuse the view.

use sdm_schema::view::semantic;
use sdm_schema::view::spec::{
    ColumnItem, PrimaryKey, RefusalReason, RelationshipItem, Section, Source, TableItem,
    ViewRefusal, ViewSpec,
};
use sdm_schema::view::yaml::Value;
use serde_json::Value as Json;

mod rules;

use rules::RULES;

/// A definition read whole, in the old client's order.
fn read(source: &str) -> Result<ViewSpec, ViewRefusal> {
    let mut spec = semantic::parse(source, &RULES)?;
    semantic::read_all(&mut spec)?;
    Ok(spec)
}

const VECTORS: &str = include_str!("../../../../contract/view-yaml-vectors.json");

/// The entry `key` of a mapping in the vectors' typed form, if present.
fn get<'a>(map: &'a Json, key: &str) -> Option<&'a Json> {
    assert_eq!(map["t"], "map");
    map["v"]
        .as_array()
        .unwrap()
        .iter()
        .find(|pair| pair[0]["t"] == "str" && pair[0]["v"] == key)
        .map(|pair| &pair[1])
}

fn text(value: &Json) -> String {
    assert_eq!(value["t"], "str", "{value}");
    value["v"].as_str().unwrap().to_owned()
}

fn items(value: Option<&Json>) -> Vec<Json> {
    value.map_or_else(Vec::new, |value| {
        assert_eq!(value["t"], "list");
        value["v"].as_array().unwrap().clone()
    })
}

fn optional_text(value: Option<&Json>) -> Option<String> {
    value.filter(|value| value["t"] != "null").map(text)
}

fn node_text(spec: &ViewSpec, node: Option<sdm_schema::view::yaml::NodeId>) -> Option<String> {
    node.and_then(|n| spec.text(n)).map(str::to_owned)
}

#[test]
fn each_definition_reads_as_the_old_client_reads_it() {
    let document: Json = serde_json::from_str(VECTORS).unwrap();
    let definitions: Vec<&Json> = document["vectors"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|vector| vector["value"]["t"] == "map" && get(&vector["value"], "tables").is_some())
        .collect();
    assert!(definitions.len() >= 5, "{} definitions", definitions.len());
    for vector in definitions {
        let source = vector["yaml"].as_str().unwrap();
        let cfg = &vector["value"];
        let mut spec = read(source).unwrap_or_else(|error| panic!("{source}: {error}"));

        let tables = items(get(cfg, "tables"));
        assert_eq!(spec.tables.len(), tables.len(), "{source}");
        for (index, (item, table_cfg)) in spec.tables.iter().zip(&tables).enumerate() {
            let TableItem::Table(table) = item else {
                panic!("{item:?}")
            };
            assert_eq!(table.alias, text(get(table_cfg, "name").unwrap()));
            let base = get(table_cfg, "base_table").unwrap();
            let Source::Parts {
                database,
                schema,
                table: base_table,
            } = table.source
            else {
                panic!()
            };
            assert_eq!(
                node_text(&spec, Some(database)),
                optional_text(get(base, "database"))
            );
            assert_eq!(
                node_text(&spec, Some(schema)),
                optional_text(get(base, "schema"))
            );
            assert_eq!(
                node_text(&spec, Some(base_table)),
                optional_text(get(base, "table"))
            );
            // `RE/graph.py:2081-2090`: one column is the key; more are
            // composite, dropped with a message; none is no key.
            let key: Vec<String> = get(table_cfg, "primary_key")
                .map(|key| items(get(key, "columns")).iter().map(text).collect())
                .unwrap_or_default();
            match (&table.primary_key, key.as_slice()) {
                (PrimaryKey::None, []) | (PrimaryKey::Composite, [_, _, ..]) => {}
                (PrimaryKey::One(node), [one]) => assert_eq!(spec.text(*node), Some(one.as_str())),
                (got, want) => panic!("{source}: {got:?} where {want:?}"),
            }
            // `RE/graph.py:2105-2113`: dimensions, then time dimensions,
            // then facts.
            let mut expected = Vec::new();
            for (section, key) in [
                (Section::Dimension, "dimensions"),
                (Section::TimeDimension, "time_dimensions"),
                (Section::Fact, "facts"),
            ] {
                for column in items(get(table_cfg, key)) {
                    expected.push((
                        text(get(&column, "name").unwrap()),
                        optional_text(get(&column, "expr")),
                        optional_text(get(&column, "data_type")),
                        section,
                    ));
                }
            }
            let columns: Vec<_> = spec
                .columns
                .iter()
                .filter_map(|c| match c {
                    ColumnItem::Column(c) if c.table == Some(index) => Some((
                        node_text(&spec, c.name).unwrap(),
                        node_text(&spec, c.expr),
                        node_text(&spec, c.data_type),
                        c.section,
                    )),
                    _ => None,
                })
                .collect();
            assert_eq!(columns, expected, "{source}");
        }

        // `RE/graph.py:2209-2222`.
        let relationships = items(get(cfg, "relationships"));
        assert_eq!(spec.relationships.len(), relationships.len(), "{source}");
        for (relationship, cfg) in spec.relationships.clone().iter().zip(&relationships) {
            assert_eq!(
                node_text(&spec, Some(relationship.name)),
                Some(text(get(cfg, "name").unwrap()))
            );
            let pairs = items(get(cfg, "relationship_columns"));
            assert_eq!(relationship.pairs, pairs.len());
            if pairs.len() == 1 {
                let read = spec.relationship(relationship).unwrap();
                assert_eq!(
                    node_text(&spec, Some(read.left_table)),
                    Some(text(get(cfg, "left_table").unwrap()))
                );
                assert_eq!(
                    node_text(&spec, Some(read.right_table)),
                    Some(text(get(cfg, "right_table").unwrap()))
                );
                assert_eq!(
                    node_text(&spec, Some(read.left_column)),
                    Some(text(get(&pairs[0], "left_column").unwrap()))
                );
                assert_eq!(
                    node_text(&spec, Some(read.right_column)),
                    Some(text(get(&pairs[0], "right_column").unwrap()))
                );
            }
        }
    }
}

/// What the old client fails on with a raw `KeyError` or `TypeError`
/// (old-behaviour row 53) is refused by name, saying what is missing.
#[test]
fn a_definition_missing_its_tables_is_refused_by_name() {
    for (source, missing) in [
        ("name: v\n", "tables"),
        ("tables:\n  - name: A\n", "base_table"),
        (
            "tables:\n  - name: A\n    base_table: {database: D, schema: S}\n",
            "table",
        ),
        ("- a\n", "tables"),
    ] {
        let refusal = read(source).unwrap_err();
        assert!(
            refusal.to_string().contains(&format!("'{missing}'")),
            "{source}: {refusal}"
        );
    }
}

/// Codex review 1, P1-9: each of these is converted by the old client,
/// which never reads the entry that is no text.
#[test]
fn an_entry_the_old_client_never_reads_cannot_refuse_the_view() {
    let base = "tables:\n  - name: A\n    base_table: {database: D, schema: S, table: T}\n";
    // A relationship of no pair: the old client notes it as composite and
    // reads neither of its tables (`RE/graph.py:2211-2215`).
    let spec = read(&format!(
        "{base}relationships:\n  - name: R\n    relationship_columns: []\n"
    ))
    .unwrap();
    assert_eq!(
        spec.relationships,
        vec![RelationshipItem {
            name: spec.relationships[0].name,
            pairs: 0,
            cfg: spec.relationships[0].cfg,
        }]
    );
    // A physical column's type, which it never uses (`:2117-2119`).
    let spec = read(&format!(
        "{base}    dimensions:\n      - name: X\n        data_type: 123\n"
    ))
    .unwrap();
    let ColumnItem::Column(column) = &spec.columns[0] else {
        panic!()
    };
    assert!(column.expr.is_none() && column.data_type.is_some());
    // A composite key's members, which it never inspects (`:2086-2090`).
    let spec = read(&format!("{base}    primary_key: {{columns: [1, 2]}}\n")).unwrap();
    let TableItem::Table(table) = &spec.tables[0] else {
        panic!()
    };
    assert_eq!(table.primary_key, PrimaryKey::Composite);
}

/// Codex review 2, P1-8: the old client reads every table's name first,
/// then finishes each table, reading its catalog, before it reads the
/// next; a later table's missing entry refuses the view only when the
/// conversion comes to it.
#[test]
fn each_table_is_read_when_the_conversion_comes_to_it() {
    let source = "tables:\n- name: A\n  base_table: {database: D, schema: S, table: T}\n  \
                  dimensions: [{name: X, expr: X + 1}]\n- name: B\n";
    let mut spec = semantic::parse(source, &RULES).unwrap();
    assert_eq!(spec.tables.len(), 2);
    let a = semantic::table(&mut spec, 0).unwrap();
    assert_eq!(a.alias, "A");
    let columns = semantic::columns(&mut spec, 0);
    assert!(matches!(columns.as_slice(), [Ok(ColumnItem::Column(_))]));
    let refusal = semantic::table(&mut spec, 1).unwrap_err();
    assert_eq!(refusal.reason, RefusalReason::Misses("base_table".into()));
    // A name missing anywhere refuses before any table is read.
    let refusal = semantic::parse("tables:\n- name: A\n- base_table: {}\n", &RULES).unwrap_err();
    assert_eq!(refusal.reason, RefusalReason::Misses("name".into()));
    // A name no set holds refuses then; one no pattern is made of, when
    // the first table makes its patterns.
    let refusal = semantic::parse("tables:\n- name: [A]\n", &RULES).unwrap_err();
    assert_eq!(refusal.reason, RefusalReason::Unhashable("name".into()));
    let source =
        "tables:\n- name: A\n  base_table: {database: D, schema: S, table: T}\n- name: 1\n";
    let mut spec = semantic::parse(source, &RULES).unwrap();
    let refusal = semantic::table(&mut spec, 0).unwrap_err();
    assert_eq!(refusal.reason, RefusalReason::NotText("name".into()));
}

/// Codex review 2, P1-6: `len(...)` and `[0]` as Python takes them, on a
/// mapping (its entry keyed 0) and on a text (its first character).
#[test]
fn a_key_and_a_pair_are_indexed_as_python_indexes_them() {
    let base = "tables:\n  - name: A\n    base_table: {database: D, schema: S, table: T}\n";
    let mut spec = read(&format!(
        "{base}relationships:\n  - name: R\n    left_table: A\n    right_table: B\n    \
         relationship_columns: {{0: {{left_column: X, right_column: Y}}}}\n"
    ))
    .unwrap();
    let item = spec.relationships[0].clone();
    assert_eq!(item.pairs, 1);
    let pair = spec.relationship(&item).unwrap();
    assert_eq!(spec.text(pair.left_column), Some("X"));
    assert_eq!(spec.text(pair.right_column), Some("Y"));
    for (columns, key) in [
        ("{0: K}", "K"),
        ("[K]", "K"),
        ("K", "K"),
        ("{false: K}", "K"),
    ] {
        let spec = read(&format!("{base}    primary_key: {{columns: {columns}}}\n")).unwrap();
        let TableItem::Table(table) = &spec.tables[0] else {
            panic!()
        };
        let PrimaryKey::One(node) = table.primary_key else {
            panic!("{columns}: {:?}", table.primary_key)
        };
        assert_eq!(spec.value(node), &Value::Str(key.into()), "{columns}");
    }
    // `{1: K}` has a length of one and no entry keyed 0: Python's
    // `KeyError`, refused by name.
    let refusal = read(&format!("{base}    primary_key: {{columns: {{1: K}}}}\n")).unwrap_err();
    assert_eq!(refusal.reason, RefusalReason::Misses("columns".into()));
    let refusal = read(&format!("{base}    primary_key: {{columns: 7}}\n")).unwrap_err();
    assert_eq!(refusal.reason, RefusalReason::NotAList("columns".into()));
}
