// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The schema's edit semantics (T1.24b, decision E67), edit by edit, each
//! held to what the reference client's setters make of the same assignment
//! on a table in its source (`Column.__setattr__`,
//! `_TableShape.__setattr__`, `SourceTable.__delitem__`,
//! `Schema.__setattr__` and its list); each refusal seeded, so a core that
//! took an edit the reference refuses fails here; and a schema as a host
//! holds it, with its facts, read as the description (T3.16).

use std::collections::{BTreeMap, BTreeSet};

use sdm_contract::ops::{
    ColumnRef, DescribedColumn, DescribedTable, Dtype, Given, SchemaEdit, SourceDescription, Stype,
};
use sdm_contract::{ColumnType, CoreError, Detail, ErrorCode, Scalar};
use sdm_schema::edit::{self, Edited};

/// The language's answer, as `sdm-core` passes it: the schema's tests hold
/// only what the edit rules do with it.
fn supports(stype: Stype, dtype: Dtype) -> bool {
    let number = matches!(
        dtype,
        Dtype::Int | Dtype::Int32 | Dtype::Int64 | Dtype::Float | Dtype::Float64
    );
    match stype {
        Stype::Numerical => number,
        Stype::Categorical => number || matches!(dtype, Dtype::Bool | Dtype::String),
        Stype::Id => number || dtype == Dtype::String,
        Stype::Text | Stype::Multicategorical | Stype::Sequence => dtype == Dtype::String,
        Stype::Timestamp => matches!(dtype, Dtype::Date | Dtype::Time | Dtype::String),
        Stype::Image | Stype::Unsupported => true,
    }
}

fn column(name: &str, dtype: Dtype, stype: Stype) -> DescribedColumn {
    DescribedColumn {
        name: name.into(),
        received: match dtype {
            Dtype::String => ColumnType::Utf8,
            Dtype::Float => ColumnType::Float64,
            Dtype::Date => ColumnType::Timestamp {
                unit: sdm_contract::Unit::Us,
                zone: None,
            },
            _ => ColumnType::Int64,
        },
        dtype,
        stype,
        stype_before_key: None,
        nullable: true,
        unique: false,
        declared_type: None,
        exact_number: None,
    }
}

fn at(table: &str, column: &str) -> ColumnRef {
    ColumnRef {
        table: table.into(),
        column: column.into(),
    }
}

/// The shop: users keyed by `user_id`, ordered by `signup`; orders keyed by
/// `order_id`, ordered by `ts`, each naming its user.
fn shop() -> SourceDescription {
    let mut users = vec![
        column("user_id", Dtype::Int, Stype::Id),
        column("age", Dtype::Int, Stype::Numerical),
        column("country", Dtype::String, Stype::Categorical),
        column("signup", Dtype::Date, Stype::Timestamp),
        column("seen", Dtype::Date, Stype::Timestamp),
    ];
    users[0].stype_before_key = Some(Stype::Numerical);
    let mut orders = vec![
        column("order_id", Dtype::Int, Stype::Id),
        column("user_id", Dtype::Int, Stype::Id),
        column("price", Dtype::Float, Stype::Numerical),
        column("ts", Dtype::Date, Stype::Timestamp),
    ];
    orders[1].stype_before_key = Some(Stype::Numerical);
    SourceDescription {
        tables: vec![
            DescribedTable {
                name: "users".into(),
                source: "users".into(),
                columns: users,
                primary_key: vec!["user_id".into()],
                time_column: Some("signup".into()),
                time_column_inferred: None,
                duplicate_keys: Some(Scalar::UInt64(0)),
            },
            DescribedTable {
                name: "orders".into(),
                source: "orders".into(),
                columns: orders,
                primary_key: vec!["order_id".into()],
                time_column: Some("ts".into()),
                time_column_inferred: None,
                duplicate_keys: Some(Scalar::UInt64(0)),
            },
        ],
        links: vec![vec![at("orders", "user_id"), at("users", "user_id")]],
    }
}

fn made(description: SourceDescription, edit: &SchemaEdit) -> Result<Edited, CoreError> {
    edit::edit(description, edit, &supports)
}

fn table<'a>(description: &'a SourceDescription, name: &str) -> &'a DescribedTable {
    description.tables.iter().find(|t| t.name == name).unwrap()
}

fn meaning(description: &SourceDescription, name: &str, of: &str) -> (Stype, Option<Stype>) {
    let column = table(description, name)
        .columns
        .iter()
        .find(|c| c.name == of)
        .unwrap();
    (column.stype, column.stype_before_key)
}

/// A refusal is the reference's at the assignment: `INVALID_REQUEST`
/// naming the field.
fn refused(result: Result<Edited, CoreError>, field: &str) {
    let error = result.expect_err("the reference refuses this edit");
    assert_eq!(error.code, ErrorCode::InvalidRequest);
    assert_eq!(error.fields.get("field"), Some(&Detail::text(field)));
}

fn key(table: &str, key: Option<&[&str]>) -> SchemaEdit {
    SchemaEdit::PrimaryKey {
        table: table.into(),
        key: Given::Value(key.map(|k| k.iter().map(|s| (*s).to_owned()).collect())),
    }
}

fn time(table: &str, column: Option<&str>) -> SchemaEdit {
    SchemaEdit::TimeColumn {
        table: table.into(),
        column: Given::Value(column.map(Into::into)),
    }
}

fn stype(table: &str, column: &str, text: &str) -> SchemaEdit {
    SchemaEdit::Stype {
        table: table.into(),
        column: column.into(),
        stype: Given::Value(text.into()),
    }
}

fn links(given: &[(&str, &str, &str, &str)]) -> SchemaEdit {
    SchemaEdit::Relationships {
        links: given
            .iter()
            .map(|(a, b, c, d)| Given::Value(vec![at(a, b), at(c, d)]))
            .collect(),
    }
}

#[test]
fn a_key_set_marks_its_column_and_gives_the_old_key_back() {
    // `schema['users'].primary_key = 'age'`: age is an identifier, what it
    // meant kept; user_id, no longer the key, means what it meant (the
    // schema joins on a link's foreign key, not the key it names); the
    // table's repeats are uncounted (`_adopt_key` on a table in its
    // source).
    let done = made(shop(), &key("users", Some(&["age"]))).unwrap();
    let users = table(&done.description, "users");
    assert_eq!(users.primary_key, ["age"]);
    assert_eq!(users.duplicate_keys, None);
    assert_eq!(
        meaning(&done.description, "users", "age"),
        (Stype::Id, Some(Stype::Numerical))
    );
    assert_eq!(
        meaning(&done.description, "users", "user_id"),
        (Stype::Numerical, None)
    );
    assert_eq!(done.declared, None);
    assert!(done.warnings.is_empty());
    // The foreign key orders names stays one until the link goes.
    assert_eq!(
        meaning(&done.description, "orders", "user_id"),
        (Stype::Id, Some(Stype::Numerical))
    );
    let unlinked = made(done.description, &links(&[])).unwrap();
    assert_eq!(
        meaning(&unlinked.description, "orders", "user_id"),
        (Stype::Numerical, None)
    );
}

#[test]
fn a_key_cleared_counts_no_repeats_and_releases_its_column() {
    let unlinked = made(shop(), &links(&[])).unwrap().description;
    let done = made(unlinked, &key("users", None)).unwrap();
    let users = table(&done.description, "users");
    assert!(users.primary_key.is_empty());
    assert_eq!(users.duplicate_keys, Some(Scalar::UInt64(0)));
    assert_eq!(
        meaning(&done.description, "users", "user_id"),
        (Stype::Numerical, None)
    );
}

#[test]
fn a_released_timestamp_orders_a_table_that_has_no_time_column() {
    // A key that meant a timestamp, released, orders its table when the
    // table has none and it is not the key (`_release_as_key` then
    // `_adopt_as_time_column`).
    let mut shop = shop();
    shop.links.clear();
    let users = &mut shop.tables[0];
    users.time_column = None;
    users.columns[0].stype_before_key = Some(Stype::Timestamp);
    let done = made(shop, &key("users", Some(&["age"]))).unwrap();
    let users = table(&done.description, "users");
    assert_eq!(users.time_column.as_deref(), Some("user_id"));
    assert_eq!(
        meaning(&done.description, "users", "user_id"),
        (Stype::Timestamp, None)
    );
}

#[test]
fn a_key_the_reference_refuses_is_refused_at_the_assignment() {
    for (edit, why) in [
        (key("users", Some(&[])), "empty"),
        (key("users", Some(&["nope"])), "a column the table lacks"),
        (
            key("users", Some(&["signup"])),
            "a time column, which no key holds",
        ),
        (key("orders", Some(&["ts"])), "the time column"),
        (key("nope", Some(&["age"])), "a table the schema lacks"),
    ] {
        let result = made(shop(), &edit);
        assert!(result.is_err(), "{why}");
        refused(result, "primary_key");
    }
    // A value no contract type holds (a set, a number) is refused as the
    // reference refuses it.
    refused(
        made(
            shop(),
            &SchemaEdit::PrimaryKey {
                table: "users".into(),
                key: Given::Unreadable,
            },
        ),
        "primary_key",
    );
    // A column that is the time column is refused for that, not its type,
    // where the type would identify a row: text named both.
    let mut text_time = shop();
    text_time.tables[0].time_column = Some("country".into());
    let error = made(text_time, &key("users", Some(&["country"]))).unwrap_err();
    assert_eq!(error.fields.get("column"), Some(&Detail::text("country")));
}

#[test]
fn a_composite_key_is_taken_as_the_reference_takes_it() {
    let done = made(shop(), &key("users", Some(&["user_id", "age"]))).unwrap();
    assert_eq!(
        table(&done.description, "users").primary_key,
        ["user_id", "age"]
    );
    assert_eq!(meaning(&done.description, "users", "age").0, Stype::Id);
}

#[test]
fn a_time_column_set_means_a_timestamp_the_caller_chose() {
    let done = made(shop(), &time("users", Some("seen"))).unwrap();
    assert_eq!(
        table(&done.description, "users").time_column.as_deref(),
        Some("seen")
    );
    assert_eq!(
        meaning(&done.description, "users", "seen"),
        (Stype::Timestamp, None)
    );
    assert_eq!(done.declared, Some(at("users", "seen")));
    // Text can hold a time: its meaning becomes timestamp.
    let done = made(shop(), &time("users", Some("country"))).unwrap();
    assert_eq!(
        meaning(&done.description, "users", "country").0,
        Stype::Timestamp
    );
    // A foreign key named as the time column means a timestamp before the
    // schema joins on it, and stays an identifier (`_mark_key_columns`).
    let mut shop = shop();
    shop.tables[1].columns[1] = column("user_id", Dtype::Date, Stype::Id);
    shop.tables[1].columns[1].stype_before_key = Some(Stype::Timestamp);
    shop.tables[0].columns[0] = column("user_id", Dtype::Date, Stype::Id);
    let done = made(shop, &time("orders", Some("user_id"))).unwrap();
    assert_eq!(
        meaning(&done.description, "orders", "user_id"),
        (Stype::Id, Some(Stype::Timestamp))
    );
}

#[test]
fn a_time_column_cleared_changes_nothing_else() {
    let done = made(shop(), &time("users", None)).unwrap();
    let users = table(&done.description, "users");
    assert_eq!(users.time_column, None);
    assert_eq!(
        meaning(&done.description, "users", "signup"),
        (Stype::Timestamp, None)
    );
    assert_eq!(done.declared, None);
}

#[test]
fn a_time_column_the_reference_refuses_is_refused_at_the_assignment() {
    for edit in [
        time("users", Some("nope")),
        time("users", Some("user_id")),
        time("users", Some("age")),
        time("nope", Some("signup")),
    ] {
        refused(made(shop(), &edit), "time_column");
    }
    refused(
        made(
            shop(),
            &SchemaEdit::TimeColumn {
                table: "users".into(),
                column: Given::Unreadable,
            },
        ),
        "time_column",
    );
}

#[test]
fn a_meaning_is_held_as_the_reference_holds_it() {
    let done = made(shop(), &stype("users", "age", "categorical")).unwrap();
    assert_eq!(
        meaning(&done.description, "users", "age"),
        (Stype::Categorical, None)
    );
    assert_eq!(done.declared, Some(at("users", "age")));
    // A key's meaning declared: it stays an identifier, the declared
    // meaning what it gives back (`_stay_a_key_if_the_schema_joins_on_this`).
    let done = made(shop(), &stype("orders", "user_id", "categorical")).unwrap();
    assert_eq!(
        meaning(&done.description, "orders", "user_id"),
        (Stype::Id, Some(Stype::Categorical))
    );
    // ID forgets what the column meant before it was a key.
    let done = made(shop(), &stype("orders", "user_id", "ID")).unwrap();
    assert_eq!(
        meaning(&done.description, "orders", "user_id"),
        (Stype::Id, None)
    );
    // A timestamp orders a table that has none, unless it is the key.
    let cleared = made(shop(), &time("users", None)).unwrap().description;
    let done = made(cleared, &stype("users", "seen", "timestamp")).unwrap();
    assert_eq!(
        table(&done.description, "users").time_column.as_deref(),
        Some("seen")
    );
    let done = made(shop(), &stype("users", "seen", "timestamp")).unwrap();
    assert_eq!(
        table(&done.description, "users").time_column.as_deref(),
        Some("signup")
    );
}

#[test]
fn a_meaning_the_reference_refuses_is_refused_at_the_assignment() {
    for edit in [
        stype("users", "age", "banana"),
        stype("users", "age", "image"),
        stype("users", "country", "numerical"),
        stype("users", "age", "timestamp"),
        stype("users", "nope", "categorical"),
    ] {
        refused(made(shop(), &edit), "stype");
    }
    refused(
        made(
            shop(),
            &SchemaEdit::Stype {
                table: "users".into(),
                column: "age".into(),
                stype: Given::Unreadable,
            },
        ),
        "stype",
    );
}

#[test]
fn a_column_left_out_is_gone_and_the_links_are_kept() {
    let done = made(
        shop(),
        &SchemaEdit::LeaveOut {
            table: "orders".into(),
            column: Given::Value("user_id".into()),
        },
    )
    .unwrap();
    assert!(
        !table(&done.description, "orders")
            .columns
            .iter()
            .any(|c| c.name == "user_id")
    );
    assert_eq!(done.description.links.len(), 1);
    for (column, why) in [
        ("order_id", "the key"),
        ("ts", "the time column"),
        ("nope", ""),
    ] {
        let result = made(
            shop(),
            &SchemaEdit::LeaveOut {
                table: "orders".into(),
                column: Given::Value(column.into()),
            },
        );
        assert!(result.is_err(), "{why}");
        refused(result, "column");
    }
}

#[test]
fn the_relationships_are_checked_settled_and_warned_of() {
    // Each once, in order; one table reaching another two ways is said.
    let done = made(
        shop(),
        &links(&[
            ("orders", "user_id", "users", "user_id"),
            ("orders", "order_id", "users", "user_id"),
            ("orders", "user_id", "users", "user_id"),
        ]),
    )
    .unwrap();
    assert_eq!(done.description.links.len(), 2);
    let codes: Vec<&str> = done.warnings.iter().map(|w| w.code.as_str()).collect();
    assert_eq!(codes, ["TWO_WAYS_TO_A_TABLE"]);
    assert_eq!(
        meaning(&done.description, "orders", "order_id").0,
        Stype::Id
    );
    // Two tables naming each other are said first.
    let mut keyed = made(shop(), &key("orders", Some(&["user_id"])))
        .unwrap()
        .description;
    keyed = made(keyed, &links(&[])).unwrap().description;
    let done = made(
        keyed,
        &links(&[
            ("orders", "user_id", "users", "user_id"),
            ("users", "user_id", "orders", "user_id"),
        ]),
    )
    .unwrap();
    let codes: Vec<&str> = done.warnings.iter().map(|w| w.code.as_str()).collect();
    assert_eq!(codes, ["TABLES_NAME_EACH_OTHER"]);
    for refused_links in [
        links(&[("orders", "user_id", "users", "age")]),
        links(&[("orders", "nope", "users", "user_id")]),
        links(&[("orders", "user_id", "nope", "user_id")]),
        links(&[("orders", "ts", "users", "user_id")]),
        SchemaEdit::Relationships {
            links: vec![Given::Unreadable],
        },
    ] {
        refused(made(shop(), &refused_links), "relationships");
    }
}

#[test]
fn a_refused_edit_changes_nothing() {
    // The edit is checked whole before anything of the schema changes, so
    // a host that keeps its schema on a refusal keeps the reference's.
    let bad = links(&[
        ("orders", "order_id", "users", "user_id"),
        ("orders", "user_id", "users", "age"),
    ]);
    assert!(made(shop(), &bad).is_err());
    let bad = key("users", Some(&["age", "nope"]));
    assert!(made(shop(), &bad).is_err());
}

#[test]
fn a_schema_held_with_its_facts_reads_as_the_description() {
    // T3.16: the schema a host holds, each column with its facts, is the
    // schema the edits work on, whatever the core answered of it before.
    let shop = shop();
    let held = sdm_schema::held::held(&shop, &BTreeSet::new(), &BTreeMap::new());
    let read = sdm_schema::held::read(held, &BTreeMap::new(), false).unwrap();
    assert!(read.unfacted.is_empty());
    let mut expected = shop;
    for table in &mut expected.tables {
        table.time_column_inferred = None;
    }
    assert_eq!(read.description, expected);
}
