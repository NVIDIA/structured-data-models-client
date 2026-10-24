// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The semantic view's model (T3.5, design part 5's V2, V3 and V6) against
//! the semantic-view definitions of `contract/view-yaml-vectors.json`, as
//! PyYAML read them: each logical table's name, base table and primary key,
//! its columns from `dimensions`, `time_dimensions` and `facts` in that
//! order, and each relationship, read by the paths the old client reads
//! them by (`RE/graph.py:2072-2252`). Expressions stay as written; which
//! table one reads is the data layer's to say.

use sdm_schema::view::semantic::{self, PrimaryKey, RelationshipKeys, Section};
use serde_json::Value as Json;

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
        let view = semantic::parse(source).unwrap_or_else(|error| panic!("{source}: {error}"));

        let tables = items(get(cfg, "tables"));
        assert_eq!(view.tables.len(), tables.len(), "{source}");
        for (table, table_cfg) in view.tables.iter().zip(&tables) {
            assert_eq!(table.name, text(get(table_cfg, "name").unwrap()));
            let base = get(table_cfg, "base_table").unwrap();
            assert_eq!(table.base.database, text(get(base, "database").unwrap()));
            assert_eq!(table.base.schema, text(get(base, "schema").unwrap()));
            assert_eq!(table.base.table, text(get(base, "table").unwrap()));
            // `RE/graph.py:2081-2090`: one column is the key; more are
            // composite, dropped with a message; none is no key.
            let key: Vec<String> = get(table_cfg, "primary_key")
                .map(|key| items(get(key, "columns")).iter().map(text).collect())
                .unwrap_or_default();
            let expected = match key.len() {
                0 => PrimaryKey::None,
                1 => PrimaryKey::One(key[0].clone()),
                _ => PrimaryKey::Composite(key.clone()),
            };
            assert_eq!(table.primary_key, expected, "{source}");
            // `RE/graph.py:2105-2113`: dimensions, then time dimensions,
            // then facts.
            let mut expected_columns = Vec::new();
            for (section, key) in [
                (Section::Dimension, "dimensions"),
                (Section::TimeDimension, "time_dimensions"),
                (Section::Fact, "facts"),
            ] {
                for column in items(get(table_cfg, key)) {
                    expected_columns.push((
                        text(get(&column, "name").unwrap()),
                        optional_text(get(&column, "expr")),
                        optional_text(get(&column, "data_type")),
                        section,
                    ));
                }
            }
            let columns: Vec<_> = table
                .columns
                .iter()
                .map(|c| {
                    (
                        c.name.clone(),
                        c.expr.clone(),
                        c.data_type.clone(),
                        c.section,
                    )
                })
                .collect();
            assert_eq!(columns, expected_columns, "{source}");
        }

        // `RE/graph.py:2209-2222`.
        let relationships = items(get(cfg, "relationships"));
        assert_eq!(view.relationships.len(), relationships.len(), "{source}");
        for (relationship, cfg) in view.relationships.iter().zip(&relationships) {
            assert_eq!(relationship.name, text(get(cfg, "name").unwrap()));
            assert_eq!(
                relationship.left_table,
                text(get(cfg, "left_table").unwrap())
            );
            assert_eq!(
                relationship.right_table,
                text(get(cfg, "right_table").unwrap())
            );
            let columns = items(get(cfg, "relationship_columns"));
            let expected = if columns.len() == 1 {
                RelationshipKeys::One {
                    left: text(get(&columns[0], "left_column").unwrap()),
                    right: text(get(&columns[0], "right_column").unwrap()),
                }
            } else {
                RelationshipKeys::Composite
            };
            assert_eq!(relationship.keys, expected, "{source}");
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
        let refusal = semantic::parse(source).unwrap_err();
        assert!(
            refusal.to_string().contains(&format!("'{missing}'")),
            "{source}: {refusal}"
        );
    }
}
