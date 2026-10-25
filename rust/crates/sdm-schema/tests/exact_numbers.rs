// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A column of numbers handed over as their exact text (a Postgres
//! `numeric` declared without a precision) read by its role in the schema
//! as it stands (T2.21, learnings lesson 19): a key its catalog declares
//! is always its text, as the reference reads it; any other is its text
//! while the schema keys it or joins it to another column read as text,
//! and a double otherwise. Keys named and links given or edited by the
//! caller are as exact as the catalog's.

use sdm_contract::ops::{
    ColumnRef, DescribedColumn, DescribedTable, Dtype, ExactNumber, Given, SchemaEdit,
    SourceDescription, Stype,
};
use sdm_contract::{ColumnType, ErrorCode, Scalar};
use sdm_schema::assemble::{Link, mark_keys, read_exact_numbers};
use sdm_schema::edit;

fn supports(stype: Stype, dtype: Dtype) -> bool {
    let number = matches!(dtype, Dtype::Int64 | Dtype::Float64);
    match stype {
        Stype::Numerical => number,
        Stype::Id => number || dtype == Dtype::String,
        _ => true,
    }
}

fn plain(name: &str, dtype: Dtype, received: ColumnType, stype: Stype) -> DescribedColumn {
    DescribedColumn {
        name: name.into(),
        received,
        dtype,
        stype,
        stype_before_key: None,
        nullable: true,
        unique: false,
        declared_type: None,
        exact_number: None,
    }
}

/// An exact column as `describe_source` reads it for inference: a declared
/// key as its text, any other as a double.
fn exact(name: &str, kind: ExactNumber) -> DescribedColumn {
    let mut column = match kind {
        ExactNumber::DeclaredKey => plain(name, Dtype::String, ColumnType::Utf8, Stype::Id),
        ExactNumber::Other => plain(name, Dtype::Float64, ColumnType::Float64, Stype::Numerical),
    };
    column.exact_number = Some(kind);
    column
}

fn int(name: &str) -> DescribedColumn {
    plain(name, Dtype::Int64, ColumnType::Int64, Stype::Numerical)
}

fn table(name: &str, columns: Vec<DescribedColumn>, key: Option<&str>) -> DescribedTable {
    DescribedTable {
        name: name.into(),
        source: name.into(),
        columns,
        frame_key_form: None,
        primary_key: key.map(Into::into).into_iter().collect(),
        time_column: None,
        time_column_inferred: None,
        duplicate_keys: Some(Scalar::UInt64(0)),
    }
}

fn at(table: &str, column: &str) -> ColumnRef {
    ColumnRef {
        table: table.into(),
        column: column.into(),
    }
}

fn link(foreign: (&str, &str), referenced: (&str, &str)) -> Link {
    vec![at(foreign.0, foreign.1), at(referenced.0, referenced.1)]
}

fn links(given: &[Link]) -> SchemaEdit {
    SchemaEdit::Relationships {
        links: given.iter().cloned().map(Given::Value).collect(),
    }
}

fn read(description: &SourceDescription, name: &str, of: &str) -> (Dtype, ColumnType) {
    let column = description
        .tables
        .iter()
        .find(|t| t.name == name)
        .unwrap()
        .columns
        .iter()
        .find(|c| c.name == of)
        .unwrap();
    (column.dtype, column.received.clone())
}

const TEXT: (Dtype, ColumnType) = (Dtype::String, ColumnType::Utf8);
const DOUBLE: (Dtype, ColumnType) = (Dtype::Float64, ColumnType::Float64);

/// users keyed by `user_id` of `users_key`'s kind (or an integer), orders
/// naming a user in `user_id` of `orders_key`'s kind (or an integer), with
/// no link yet; users' key the caller's where it is no declared key.
fn shop(users_key: Option<ExactNumber>, orders_key: Option<ExactNumber>) -> SourceDescription {
    let column =
        |kind: Option<ExactNumber>| kind.map_or_else(|| int("user_id"), |k| exact("user_id", k));
    SourceDescription {
        tables: vec![
            table(
                "users",
                vec![column(users_key), exact("balance", ExactNumber::Other)],
                Some("user_id"),
            ),
            table(
                "orders",
                vec![int("order_id"), column(orders_key)],
                Some("order_id"),
            ),
        ],
        links: Vec::new(),
    }
}

#[test]
fn a_key_the_caller_names_is_its_text_and_a_double_again_once_cleared() {
    let mut schema = shop(Some(ExactNumber::Other), None);
    mark_keys(&mut schema);
    assert_eq!(read(&schema, "users", "user_id"), TEXT);
    // A feature stays a double.
    assert_eq!(read(&schema, "users", "balance"), DOUBLE);
    let cleared = edit::edit(
        schema,
        &SchemaEdit::PrimaryKey {
            table: "users".into(),
            key: Given::Value(None),
        },
        &supports,
        &sdm_schema::assemble::HostTypes::default(),
    )
    .unwrap()
    .description;
    assert_eq!(read(&cleared, "users", "user_id"), DOUBLE);
    let keyed = edit::edit(
        cleared,
        &SchemaEdit::PrimaryKey {
            table: "users".into(),
            key: Given::Value(Some(vec!["user_id".into()])),
        },
        &supports,
        &sdm_schema::assemble::HostTypes::default(),
    )
    .unwrap()
    .description;
    assert_eq!(read(&keyed, "users", "user_id"), TEXT);
}

#[test]
fn a_declared_key_is_its_text_whatever_the_schema_joins_on() {
    let mut schema = shop(Some(ExactNumber::Other), Some(ExactNumber::DeclaredKey));
    // orders.user_id is a declared foreign key the caller's links leave out.
    mark_keys(&mut schema);
    assert_eq!(read(&schema, "orders", "user_id"), TEXT);
}

#[test]
fn a_link_edited_onto_a_declared_key_joins_both_ends_as_text() {
    // The reproduction's edited variant: users keyed by a declared numeric
    // key, orders naming it in a numeric column no constraint declares,
    // which the reference reads as a double and refuses to join.
    let schema = shop(Some(ExactNumber::DeclaredKey), Some(ExactNumber::Other));
    let edited = edit::edit(
        schema,
        &links(&[link(("orders", "user_id"), ("users", "user_id"))]),
        &supports,
        &sdm_schema::assemble::HostTypes::default(),
    )
    .unwrap()
    .description;
    assert_eq!(read(&edited, "orders", "user_id"), TEXT);
    assert_eq!(read(&edited, "users", "user_id"), TEXT);
    // Cleared, the foreign key is a double again.
    let cleared = edit::edit(
        edited,
        &links(&[]),
        &supports,
        &sdm_schema::assemble::HostTypes::default(),
    )
    .unwrap()
    .description;
    assert_eq!(read(&cleared, "orders", "user_id"), DOUBLE);
    assert_eq!(read(&cleared, "users", "user_id"), TEXT);
}

#[test]
fn a_link_given_between_two_keys_no_catalog_declares_joins_them_as_text() {
    let mut schema = shop(Some(ExactNumber::Other), Some(ExactNumber::Other));
    let given = [link(("orders", "user_id"), ("users", "user_id"))];
    read_exact_numbers(&mut schema.tables, &given);
    assert_eq!(read(&schema, "orders", "user_id"), TEXT);
    assert_eq!(read(&schema, "users", "user_id"), TEXT);
}

#[test]
fn a_key_joined_to_an_integer_is_read_as_the_reference_reads_it() {
    // An integer foreign key on a key the caller named: both a number, as
    // the reference joins them, rather than a text no integer matches.
    let schema = shop(Some(ExactNumber::Other), None);
    let edited = edit::edit(
        schema,
        &links(&[link(("orders", "user_id"), ("users", "user_id"))]),
        &supports,
        &sdm_schema::assemble::HostTypes::default(),
    )
    .unwrap()
    .description;
    assert_eq!(read(&edited, "users", "user_id"), DOUBLE);
    assert_eq!(read(&edited, "orders", "user_id").0, Dtype::Int64);
}

#[test]
fn a_column_joined_through_a_key_an_integer_joins_too_is_a_double() {
    // payments.user_id -> users.user_id <- orders.user_id (an integer): the
    // key is a double, so the other column joined to it is one too, both
    // ends of each link decided together.
    let mut schema = shop(Some(ExactNumber::Other), None);
    schema.tables.push(table(
        "payments",
        vec![int("payment_id"), exact("user_id", ExactNumber::Other)],
        Some("payment_id"),
    ));
    let given = [
        link(("payments", "user_id"), ("users", "user_id")),
        link(("orders", "user_id"), ("users", "user_id")),
    ];
    read_exact_numbers(&mut schema.tables, &given);
    assert_eq!(read(&schema, "users", "user_id"), DOUBLE);
    assert_eq!(read(&schema, "payments", "user_id"), DOUBLE);
}

#[test]
fn a_link_still_refused_leaves_the_schema_as_it_was() {
    // A declared numeric key joined from an integer: the reference refuses
    // the join, and so does the edit, naming the field.
    let schema = shop(Some(ExactNumber::DeclaredKey), None);
    let refused = edit::edit(
        schema,
        &links(&[link(("orders", "user_id"), ("users", "user_id"))]),
        &supports,
        &sdm_schema::assemble::HostTypes::default(),
    )
    .unwrap_err();
    assert_eq!(refused.code, ErrorCode::InvalidRequest);
}
