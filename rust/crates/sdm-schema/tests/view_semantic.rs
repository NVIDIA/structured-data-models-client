// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Semantic view definitions: physical identities, section order and lazy refusals.

use sdm_schema::view::semantic;
use sdm_schema::view::spec::{
    ColumnItem, PrimaryKey, RefusalReason, RelationshipItem, Section, Source, TableItem,
    ViewRefusal, ViewSpec,
};
use sdm_schema::view::yaml::Value;

mod rules;

use rules::RULES;

/// A definition read whole, in the old client's order.
fn read(source: &str) -> Result<ViewSpec, ViewRefusal> {
    let mut spec = semantic::parse(source, &RULES)?;
    semantic::read_all(&mut spec)?;
    Ok(spec)
}

#[test]
fn tables_columns_and_relationships_preserve_explicit_definition() {
    let mut spec = read(
        r#"
tables:
- name: People
  base_table: {database: D, schema: S, table: users}
  primary_key: {columns: [id]}
  dimensions: [{name: identifier, expr: id, data_type: BIGINT}]
  time_dimensions: [{name: created, expr: created_at, data_type: TIMESTAMP_NTZ}]
  facts: [{name: amount, expr: price * 2, data_type: DOUBLE}]
- name: Orders
  base_table: {database: D, schema: S, table: orders}
relationships:
- name: ownership
  left_table: Orders
  right_table: People
  relationship_columns: [{left_column: user_id, right_column: id}]
"#,
    )
    .unwrap();
    assert_eq!(spec.tables.len(), 2);
    for (index, alias, physical) in [(0, "People", "users"), (1, "Orders", "orders")] {
        let TableItem::Table(table) = &spec.tables[index] else {
            panic!("table expected")
        };
        assert_eq!(table.alias, alias);
        let Source::Parts {
            database,
            schema,
            table: base,
        } = table.source
        else {
            panic!("physical source expected")
        };
        assert_eq!(spec.text(database), Some("D"));
        assert_eq!(spec.text(schema), Some("S"));
        assert_eq!(spec.text(base), Some(physical));
        if index == 0 {
            let PrimaryKey::One(key) = table.primary_key else {
                panic!("single key expected")
            };
            assert_eq!(spec.text(key), Some("id"));
        } else {
            assert_eq!(table.primary_key, PrimaryKey::None);
        }
    }
    let columns: Vec<_> = spec
        .columns
        .iter()
        .map(|c| {
            let ColumnItem::Column(c) = c else {
                panic!("column expected")
            };
            (
                c.table,
                c.name.and_then(|n| spec.text(n)),
                c.expr.and_then(|n| spec.text(n)),
                c.data_type.and_then(|n| spec.text(n)),
                c.section,
            )
        })
        .collect();
    assert_eq!(
        columns,
        vec![
            (
                Some(0),
                Some("identifier"),
                Some("id"),
                Some("BIGINT"),
                Section::Dimension
            ),
            (
                Some(0),
                Some("created"),
                Some("created_at"),
                Some("TIMESTAMP_NTZ"),
                Section::TimeDimension
            ),
            (
                Some(0),
                Some("amount"),
                Some("price * 2"),
                Some("DOUBLE"),
                Section::Fact
            ),
        ]
    );
    assert_eq!(spec.relationships.len(), 1);
    let relationship = spec.relationships[0].clone();
    assert_eq!(spec.text(relationship.name), Some("ownership"));
    assert_eq!(relationship.pairs, 1);
    let pair = spec.relationship(&relationship).unwrap();
    assert_eq!(spec.text(pair.left_table), Some("Orders"));
    assert_eq!(spec.text(pair.right_table), Some("People"));
    assert_eq!(spec.text(pair.left_column), Some("user_id"));
    assert_eq!(spec.text(pair.right_column), Some("id"));
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

/// each of these is converted by the old client,
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
    // A composite relationship is retained for an omission note without
    // inspecting its endpoints or trying to interpret either member.
    let spec = read(&format!(
        "{base}relationships:\n  - name: composite\n    left_table: [not, text]\n    relationship_columns: [1, 2]\n"
    )).unwrap();
    assert_eq!(spec.relationships.len(), 1);
    assert_eq!(spec.relationships[0].pairs, 2);
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

/// the old client reads every table's name first,
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

/// `len(...)` and `[0]` as Python takes them, on a
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
