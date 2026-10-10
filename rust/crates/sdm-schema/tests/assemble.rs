// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The schema's assembly (T1.21d, decision E55), rule by rule, each held to
//! what the reference client's `Schema.from_source` makes of the same
//! tables, its warnings word for word; and the consistency check that holds
//! the schema `materialize` is handed, each refusal seeded.

use std::collections::BTreeSet;

use sdm_contract::ops::{
    ColumnRef, DescribedColumn, DescribedTable, Dtype, SourceDescription, Stype,
};
use sdm_contract::{ColumnType, CoreWarning, Detail, ErrorCode, Scalar};
use sdm_schema::assemble::{self, Link};
use sdm_schema::infer::{InferredColumn, InferredTable, PassedOver, Taken, TimeChoice};

fn column(name: &str, dtype: Dtype, stype: Stype) -> InferredColumn {
    InferredColumn {
        name: name.into(),
        dtype,
        stype,
        stype_before_key: None,
        received: match dtype {
            Dtype::String => ColumnType::Utf8,
            Dtype::Float => ColumnType::Float64,
            Dtype::Date => ColumnType::Timestamp {
                unit: sdm_contract::Unit::Us,
                zone: None,
            },
            _ => ColumnType::Int64,
        },
        nullable: true,
        unique: false,
    }
}

fn table(name: &str, columns: Vec<InferredColumn>, key: Option<&str>) -> InferredTable {
    InferredTable {
        name: name.into(),
        columns,
        primary_key: key.map(Into::into),
        time: TimeChoice::default(),
        unique: Vec::new(),
    }
}

fn link(source: (&str, &str), target: (&str, &str)) -> Link {
    vec![
        ColumnRef {
            table: source.0.into(),
            column: source.1.into(),
        },
        ColumnRef {
            table: target.0.into(),
            column: target.1.into(),
        },
    ]
}

fn links(description: &SourceDescription) -> Vec<(String, String, String, String)> {
    description
        .links
        .iter()
        .map(|l| {
            (
                l[0].table.clone(),
                l[0].column.clone(),
                l[1].table.clone(),
                l[1].column.clone(),
            )
        })
        .collect()
}

fn described<'a>(description: &'a SourceDescription, name: &str) -> &'a DescribedTable {
    description.tables.iter().find(|t| t.name == name).unwrap()
}

fn stype(description: &SourceDescription, table: &str, name: &str) -> (Stype, Option<Stype>) {
    let c = described(description, table)
        .columns
        .iter()
        .find(|c| c.name == name)
        .unwrap();
    (c.stype, c.stype_before_key)
}

fn codes(warnings: &[CoreWarning]) -> Vec<&str> {
    warnings.iter().map(|w| w.code.as_str()).collect()
}

fn users() -> InferredTable {
    table(
        "users",
        vec![
            column("user_id", Dtype::Int, Stype::Id),
            column("age", Dtype::Int, Stype::Numerical),
        ],
        Some("user_id"),
    )
}

fn orders(user_id: Dtype) -> InferredTable {
    table(
        "orders",
        vec![
            column("order_id", Dtype::Int, Stype::Id),
            column("user_id", user_id, Stype::Numerical),
            column("price", Dtype::Float, Stype::Numerical),
        ],
        Some("order_id"),
    )
}

#[test]
fn a_column_named_as_another_table_s_key_links_to_it_and_is_marked() {
    let made = assemble::assemble(vec![orders(Dtype::Int), users()], vec![], None);
    assert_eq!(
        links(&made.description),
        [(
            "orders".into(),
            "user_id".into(),
            "users".into(),
            "user_id".into()
        )]
    );
    // Joined on, so an identifier, what it meant kept to be given back.
    assert_eq!(
        stype(&made.description, "orders", "user_id"),
        (Stype::Id, Some(Stype::Numerical))
    );
    assert_eq!(
        stype(&made.description, "users", "user_id"),
        (Stype::Id, None)
    );
    assert!(made.warnings.is_empty());
}

#[test]
fn a_key_name_two_tables_share_links_nothing() {
    let twin = table(
        "twins",
        vec![column("user_id", Dtype::Int, Stype::Id)],
        Some("user_id"),
    );
    let made = assemble::assemble(vec![orders(Dtype::Int), users(), twin], vec![], None);
    assert!(made.description.links.is_empty());
    assert_eq!(
        stype(&made.description, "orders", "user_id"),
        (Stype::Numerical, None)
    );
}

#[test]
fn a_link_that_cannot_join_is_dropped_and_said() {
    let made = assemble::assemble(vec![orders(Dtype::String), users()], vec![], None);
    assert!(made.description.links.is_empty());
    assert_eq!(codes(&made.warnings), ["LINK_CANNOT_JOIN"]);
    assert_eq!(
        made.warnings[0].message,
        "orders.user_id names the key of 'users' and holds 'string' where that key holds \
         'int64', which cannot match, so no relationship was made from it. Store the two as \
         one type if they are the same identifier."
    );
    // A declared foreign key to a column that is not its table's key.
    let items = table(
        "items",
        vec![
            column("item_id", Dtype::Int, Stype::Id),
            column("sku", Dtype::String, Stype::Categorical),
        ],
        Some("item_id"),
    );
    let lines = table(
        "lines",
        vec![
            column("line_id", Dtype::Int, Stype::Id),
            column("sku", Dtype::String, Stype::Categorical),
        ],
        None,
    );
    let made = assemble::assemble(
        vec![items, lines],
        vec![link(("lines", "sku"), ("items", "sku"))],
        None,
    );
    assert!(made.description.links.is_empty());
    assert_eq!(
        made.warnings[0].message,
        "lines.sku refers to items.sku, but the primary key of 'items' is 'item_id'. A \
         relationship joins to a primary key, so none was made from it. Pass \
         primary_keys={'items': 'sku'} if 'sku' identifies a row of 'items'."
    );
    assert_eq!(
        made.warnings[0].fields["reason"],
        Detail::text("not_the_key")
    );
}

#[test]
fn two_ways_and_tables_naming_each_other_are_said_in_the_reference_s_words() {
    let accounts = table(
        "accounts",
        vec![
            column("account_id", Dtype::Int, Stype::Id),
            column("refund_id", Dtype::Int, Stype::Id),
        ],
        Some("account_id"),
    );
    let refunds = table(
        "refunds",
        vec![
            column("refund_id", Dtype::Int, Stype::Id),
            column("account_id", Dtype::Int, Stype::Id),
            column("payer", Dtype::Int, Stype::Numerical),
        ],
        Some("refund_id"),
    );
    let made = assemble::assemble(
        vec![accounts, refunds],
        vec![link(("refunds", "payer"), ("accounts", "account_id"))],
        None,
    );
    assert_eq!(
        links(&made.description),
        [
            (
                "refunds".into(),
                "payer".into(),
                "accounts".into(),
                "account_id".into()
            ),
            (
                "accounts".into(),
                "refund_id".into(),
                "refunds".into(),
                "refund_id".into()
            ),
            (
                "refunds".into(),
                "account_id".into(),
                "accounts".into(),
                "account_id".into()
            ),
        ]
    );
    assert_eq!(
        codes(&made.warnings),
        ["TABLES_NAME_EACH_OTHER", "TWO_WAYS_TO_A_TABLE"]
    );
    assert_eq!(
        made.warnings[0].message,
        "'accounts' and 'refunds' name each other, and a relationship is crossed in both \
         directions, so this is two ways over one pair and an aggregation between them is \
         refused for not having a single one. Declare whichever of the two a question is about \
         and leave the other out."
    );
    assert_eq!(
        made.warnings[1].message,
        "'refunds' reaches 'accounts' through 'account_id' and 'payer', and an aggregation \
         across the two tables has to be resolved to one of them, so a query that crosses this \
         pair is refused until it is. They are all kept because they are all real, and which \
         one a question means is not something this can decide. To keep 'account_id', drop the \
         others: schema.relationships = [link for link in schema.relationships if not \
         (link.source_table == 'refunds' and link.target_table == 'accounts' and \
         link.source_column in ['payer'])]."
    );
    assert_eq!(
        made.warnings[1].fields["columns"],
        Detail::List(vec![
            Scalar::Utf8("account_id".into()),
            Scalar::Utf8("payer".into())
        ])
    );
}

/// A table whose one timestamp is an edit stamp, as inference leaves it:
/// taken, pending the links.
fn plans() -> InferredTable {
    let mut plans = table(
        "plans",
        vec![
            column("plan_id", Dtype::Int, Stype::Id),
            column("modified_at", Dtype::Date, Stype::Timestamp),
            column("_loaded_at", Dtype::Date, Stype::Timestamp),
        ],
        Some("plan_id"),
    );
    plans.time = TimeChoice {
        column: None,
        how: None,
        refused: vec![
            ("modified_at".into(), PassedOver::EditStamp),
            ("_loaded_at".into(), PassedOver::LoadStamp),
        ],
        fallback: Some("modified_at".into()),
    }
    .for_links(false);
    assert_eq!(plans.time.how, Some(Taken::EditStamp));
    plans
}

#[test]
fn a_lone_edit_stamp_is_taken_in_a_log_and_refused_in_an_entity() {
    // Nothing links to plans: a log of changes, the stamp taken.
    let made = assemble::assemble(vec![plans(), users()], vec![], None);
    assert_eq!(
        described(&made.description, "plans").time_column.as_deref(),
        Some("modified_at")
    );
    assert_eq!(codes(&made.warnings), ["TIME_COLUMNS_GUESSED"]);
    assert_eq!(
        made.warnings[0].message,
        "Some time columns were inferred, or refused, on a guess about what a column holds:\n\
         - plans.modified_at was taken as the time column although it looks like when a row \
         was last changed, because the table has no other timestamp and no table links to it, \
         so it reads as a log of changes. If it does not say when a row became true, clear it: \
         time_columns={'plans': None}."
    );
    // Something links to plans: an entity edited in place, the stamp refused.
    let subscriptions = table(
        "subscriptions",
        vec![
            column("sub_id", Dtype::Int, Stype::Id),
            column("plan_id", Dtype::Int, Stype::Id),
        ],
        Some("sub_id"),
    );
    let made = assemble::assemble(vec![plans(), subscriptions], vec![], None);
    assert_eq!(described(&made.description, "plans").time_column, None);
    assert_eq!(
        made.warnings[0].message,
        "Some time columns were inferred, or refused, on a guess about what a column holds:\n\
         - plans.modified_at was not taken as the time column because it looks like when a row \
         was last changed, not when it began. If it does say when a row became true, name it: \
         time_columns={'plans': 'modified_at'}.\n\
         - plans._loaded_at was not taken as the time column because it looks like when a row \
         was loaded, not when it began. If it does say when a row became true, name it: \
         time_columns={'plans': '_loaded_at'}."
    );
    // A lone table with no link has nothing to window: nothing is said.
    let made = assemble::assemble(vec![plans()], vec![], None);
    assert!(made.warnings.is_empty());
}

#[test]
fn a_keyless_table_is_keyed_by_the_one_column_foreign_keys_refer_to() {
    let codes_table = || {
        table(
            "codes",
            vec![
                column("tag", Dtype::String, Stype::Categorical),
                column("label", Dtype::String, Stype::Categorical),
            ],
            None,
        )
    };
    let mut tables = vec![codes_table()];
    let declared = vec![link(("coded", "tag"), ("codes", "tag"))];
    assemble::key_by_reference(&mut tables, &declared).unwrap();
    assert_eq!(tables[0].primary_key.as_deref(), Some("tag"));
    assert_eq!(tables[0].columns[0].stype, Stype::Id);
    assert_eq!(
        tables[0].columns[0].stype_before_key,
        Some(Stype::Categorical)
    );
    // Foreign keys naming two of its columns leave it keyless.
    let mut tables = vec![codes_table()];
    let two = vec![
        link(("coded", "tag"), ("codes", "tag")),
        link(("other", "label"), ("codes", "label")),
    ];
    assemble::key_by_reference(&mut tables, &two).unwrap();
    assert_eq!(tables[0].primary_key, None);
    // A column that cannot identify a row, or the time column, is refused.
    let mut dated = vec![table(
        "days",
        vec![column("day", Dtype::Date, Stype::Timestamp)],
        None,
    )];
    let err =
        assemble::key_by_reference(&mut dated, &[link(("x", "day"), ("days", "day"))]).unwrap_err();
    assert_eq!(err.code, ErrorCode::InvalidRequest);
    let mut timed = vec![codes_table()];
    timed[0].time.column = Some("tag".into());
    let err = assemble::key_by_reference(&mut timed, &declared).unwrap_err();
    assert_eq!(err.code, ErrorCode::InvalidRequest);
}

fn description() -> SourceDescription {
    assemble::assemble(vec![orders(Dtype::Int), users()], vec![], None).description
}

#[test]
fn the_columns_joined_on_are_marked_again_from_what_the_host_holds() {
    // A host that dropped the link: the foreign key gets back what it meant.
    let mut unlinked = description();
    unlinked.links.clear();
    assemble::mark_keys(&mut unlinked);
    assert_eq!(
        stype(&unlinked, "orders", "user_id"),
        (Stype::Numerical, None)
    );
    // A host that never marked it: marked here.
    let mut unmarked = description();
    let user_id = &mut unmarked.tables[0].columns[1];
    user_id.stype = Stype::Numerical;
    user_id.stype_before_key = None;
    assemble::mark_keys(&mut unmarked);
    assert_eq!(
        stype(&unmarked, "orders", "user_id"),
        (Stype::Id, Some(Stype::Numerical))
    );
    // A timestamp given back orders a table that has no time column.
    let mut stamped = description();
    stamped.links.clear();
    stamped.tables[0].columns[1] = DescribedColumn {
        name: "user_id".into(),
        received: ColumnType::Int64,
        dtype: Dtype::Date,
        stype: Stype::Id,
        stype_before_key: Some(Stype::Timestamp),
        nullable: true,
        unique: false,
    };
    assemble::mark_keys(&mut stamped);
    assert_eq!(stamped.tables[0].time_column.as_deref(), Some("user_id"));
}

#[test]
fn a_link_that_cannot_join_is_refused_by_the_consistency_check() {
    assert!(assemble::check_links(&description()).is_ok());
    let refused = |change: &dyn Fn(&mut SourceDescription)| {
        let mut schema = description();
        change(&mut schema);
        let err = assemble::check_links(&schema).unwrap_err();
        assert_eq!(err.code, ErrorCode::InvalidRequest);
        assert_eq!(err.fields["field"], Detail::text("relationships"));
    };
    // Each way the evidence can be missing, empty or retyped.
    refused(&|s| s.links[0][0].column = "gone".into());
    refused(&|s| s.links[0][1].table = "gone".into());
    refused(&|s| s.links[0].pop().map(|_| ()).unwrap());
    refused(&|s| s.links[0].clear());
    refused(&|s| s.tables[1].primary_key.clear());
    refused(&|s| s.tables[1].primary_key.push("age".into()));
    refused(&|s| s.tables[0].columns[1].dtype = Dtype::String);
    // A number joins a number.
    let mut widened = description();
    widened.tables[0].columns[1].dtype = Dtype::Float;
    assert!(assemble::check_links(&widened).is_ok());
}

#[test]
fn names_are_quoted_and_listed_as_the_reference_s_messages_do() {
    let three = table(
        "c",
        vec![
            column("a", Dtype::Int, Stype::Id),
            column("x", Dtype::Int, Stype::Id),
            column("y", Dtype::Int, Stype::Id),
        ],
        Some("a"),
    );
    let referred = table("t", vec![column("k", Dtype::Int, Stype::Id)], Some("k"));
    let made = assemble::assemble(
        vec![three, referred],
        vec![
            link(("c", "x"), ("t", "k")),
            link(("c", "y"), ("t", "k")),
            link(("c", "a"), ("t", "k")),
        ],
        None,
    );
    assert_eq!(codes(&made.warnings), ["TWO_WAYS_TO_A_TABLE"]);
    assert!(
        made.warnings[0]
            .message
            .starts_with("'c' reaches 't' through 'a', 'x' and 'y', and"),
        "{}",
        made.warnings[0].message
    );
    assert!(
        made.warnings[0]
            .message
            .ends_with("link.source_column in ['x', 'y'])].")
    );
    let names: BTreeSet<&str> = made.warnings[0]
        .fields
        .values()
        .filter_map(|d| match d {
            Detail::Scalar(Scalar::Utf8(t)) => Some(t.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(names, BTreeSet::from(["c", "t"]));
}
