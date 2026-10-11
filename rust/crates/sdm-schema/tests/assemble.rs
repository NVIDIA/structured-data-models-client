// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The schema's assembly (T1.21d, decision E55), rule by rule, each held to
//! what the reference client's `Schema.from_source` makes of the same
//! tables, its warnings by code and fields (their words are the host's,
//! decision E59, held word for word by the client's tests); the keys, time
//! columns and relationships a caller names (T1.23); and the consistency
//! check that holds the schema `materialize` is handed, each refusal
//! seeded.

use std::collections::BTreeSet;

use sdm_contract::ops::{
    ColumnRef, DescribedColumn, DescribedTable, Dtype, SourceDescription, Stype,
};
use sdm_contract::{ColumnType, CoreWarning, Detail, ErrorCode, Scalar};
use sdm_schema::assemble::{self, Given, Link};
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
        source: name.into(),
        columns,
        left_out: Vec::new(),
        unread_times: Vec::new(),
        primary_key: key.map(Into::into),
        time: TimeChoice::default(),
        time_column: None,
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

/// The schema assembled of `tables` and `declared`, with nothing given.
fn made_of(tables: Vec<InferredTable>, declared: Vec<Link>) -> assemble::Assembled {
    assemble::assemble(tables, declared, Given::default(), None).unwrap()
}

/// The text of field `name` of `warning`.
fn said<'a>(warning: &'a CoreWarning, name: &str) -> &'a str {
    match &warning.fields[name] {
        Detail::Scalar(Scalar::Utf8(text)) => text,
        other => panic!("{name}: {other:?}"),
    }
}

fn names(list: &[&str]) -> Detail {
    Detail::List(list.iter().map(|n| Scalar::Utf8((*n).into())).collect())
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
    let made = made_of(vec![orders(Dtype::Int), users()], vec![]);
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
    let made = made_of(vec![orders(Dtype::Int), users(), twin], vec![]);
    assert!(made.description.links.is_empty());
    assert_eq!(
        stype(&made.description, "orders", "user_id"),
        (Stype::Numerical, None)
    );
}

#[test]
fn a_link_that_cannot_join_is_dropped_and_said() {
    let made = made_of(vec![orders(Dtype::String), users()], vec![]);
    assert!(made.description.links.is_empty());
    assert_eq!(codes(&made.warnings), ["LINK_CANNOT_JOIN"]);
    let warning = &made.warnings[0];
    assert_eq!(said(warning, "variant"), "types");
    assert_eq!(said(warning, "source_table"), "orders");
    assert_eq!(said(warning, "source_column"), "user_id");
    assert_eq!(said(warning, "target_table"), "users");
    assert_eq!(said(warning, "source_dtype"), "string");
    assert_eq!(said(warning, "target_dtype"), "int64");
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
    let made = made_of(
        vec![items, lines],
        vec![link(("lines", "sku"), ("items", "sku"))],
    );
    assert!(made.description.links.is_empty());
    assert_eq!(said(&made.warnings[0], "variant"), "not_the_key");
    assert_eq!(made.warnings[0].fields["key"], names(&["item_id"]));
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
    let made = made_of(
        vec![accounts, refunds],
        vec![link(("refunds", "payer"), ("accounts", "account_id"))],
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
    assert_eq!(said(&made.warnings[0], "source_table"), "accounts");
    assert_eq!(said(&made.warnings[0], "target_table"), "refunds");
    assert_eq!(said(&made.warnings[1], "source_table"), "refunds");
    assert_eq!(said(&made.warnings[1], "target_table"), "accounts");
    assert_eq!(said(&made.warnings[1], "kept"), "account_id");
    assert_eq!(made.warnings[1].fields["others"], names(&["payer"]));
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
        among: vec![],
        refused: vec![
            ("modified_at".into(), PassedOver::EditStamp),
            ("_loaded_at".into(), PassedOver::LoadStamp),
        ],
        fallback: Some("modified_at".into()),
    }
    .for_links(false);
    plans.time_column.clone_from(&plans.time.column);
    assert_eq!(plans.time.how, Some(Taken::EditStamp));
    plans
}

#[test]
fn a_lone_edit_stamp_is_taken_in_a_log_and_refused_in_an_entity() {
    // Nothing links to plans: a log of changes, the stamp taken.
    let made = made_of(vec![plans(), users()], vec![]);
    assert_eq!(
        described(&made.description, "plans").time_column.as_deref(),
        Some("modified_at")
    );
    assert_eq!(
        described(&made.description, "plans").time_column_inferred,
        None
    );
    assert_eq!(codes(&made.warnings), ["TIME_COLUMNS_GUESSED"]);
    assert_eq!(said(&made.warnings[0], "variant"), "taken_edit_stamp");
    assert_eq!(said(&made.warnings[0], "table"), "plans");
    assert_eq!(said(&made.warnings[0], "column"), "modified_at");
    // Something links to plans: an entity edited in place, the stamp refused.
    let subscriptions = table(
        "subscriptions",
        vec![
            column("sub_id", Dtype::Int, Stype::Id),
            column("plan_id", Dtype::Int, Stype::Id),
        ],
        Some("sub_id"),
    );
    let made = made_of(vec![plans(), subscriptions], vec![]);
    assert_eq!(described(&made.description, "plans").time_column, None);
    assert_eq!(
        described(&made.description, "plans")
            .time_column_inferred
            .as_deref(),
        Some("modified_at")
    );
    // One warning a line of the reference's, in its order.
    assert_eq!(
        made.warnings
            .iter()
            .map(|w| (said(w, "column"), said(w, "variant")))
            .collect::<Vec<_>>(),
        [
            ("modified_at", "refused_edit_stamp"),
            ("_loaded_at", "refused_load_stamp")
        ]
    );
    // A lone table with no link has nothing to window: nothing is said.
    let made = made_of(vec![plans()], vec![]);
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
    timed[0].time_column = Some("tag".into());
    let err = assemble::key_by_reference(&mut timed, &declared).unwrap_err();
    assert_eq!(err.code, ErrorCode::InvalidRequest);
}

fn description() -> SourceDescription {
    made_of(vec![orders(Dtype::Int), users()], vec![]).description
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
        // Named, when the link has both ends.
        if schema.links[0].len() == 2 {
            assert_eq!(
                err.fields["source_table"],
                Detail::text(&schema.links[0][0].table)
            );
            assert_eq!(
                err.fields["target_column"],
                Detail::text(&schema.links[0][1].column)
            );
        }
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
    let made = made_of(
        vec![three, referred],
        vec![
            link(("c", "x"), ("t", "k")),
            link(("c", "y"), ("t", "k")),
            link(("c", "a"), ("t", "k")),
        ],
    );
    assert_eq!(codes(&made.warnings), ["TWO_WAYS_TO_A_TABLE"]);
    assert_eq!(made.warnings[0].fields["columns"], names(&["a", "x", "y"]));
    assert_eq!(made.warnings[0].fields["others"], names(&["x", "y"]));
    let texts: BTreeSet<&str> = made.warnings[0]
        .fields
        .values()
        .filter_map(|d| match d {
            Detail::Scalar(Scalar::Utf8(t)) => Some(t.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(texts, BTreeSet::from(["c", "t", "a"]));
}

/// A warning's message names nothing: every name it speaks of is a field,
/// for the host to word (decision E59).
#[test]
fn a_warning_s_message_names_nothing_and_its_fields_name_everything() {
    let made = made_of(
        vec![
            table(
                "odd'name",
                vec![column("user_id", Dtype::String, Stype::Id)],
                None,
            ),
            users(),
        ],
        vec![],
    );
    let [warning] = made.warnings.as_slice() else {
        panic!("{:?}", made.warnings)
    };
    assert_eq!(said(warning, "source_table"), "odd'name");
    for name in ["odd'name", "user_id", "users", "string", "int64"] {
        assert!(!warning.message.contains(name), "{}", warning.message);
    }
}

/// The keys and time columns a caller names (T1.23), applied between the
/// reading and the count as the reference applies them, each refusal
/// seeded: a table or column the schema does not hold, a key that cannot
/// identify a row or is the time column, a time column that is the key or
/// cannot hold a time, an empty key, and a key of two columns.
#[test]
fn keys_and_time_columns_named_are_applied_and_checked() {
    let named = |keys: &[(&str, &[&str])], times: &[(&str, &str)]| {
        let mut tables = vec![orders(Dtype::Int), users()];
        tables[0]
            .columns
            .push(column("ts", Dtype::Date, Stype::Timestamp));
        tables[0]
            .columns
            .push(column("note", Dtype::String, Stype::Text));
        tables[0].time_column = Some("ts".into());
        let keys = keys
            .iter()
            .map(|(t, k)| ((*t).to_owned(), k.iter().map(|c| (*c).to_owned()).collect()))
            .collect();
        let times = times
            .iter()
            .map(|(t, c)| ((*t).to_owned(), (*c).to_owned()))
            .collect();
        assemble::apply_keys_and_times(&mut tables, &keys, &times).map(|()| tables)
    };
    // A key moved: the new column an identifier, what it meant kept.
    let tables = named(&[("users", &["age"])], &[]).unwrap();
    assert_eq!(tables[1].primary_key.as_deref(), Some("age"));
    assert_eq!(tables[1].columns[1].stype, Stype::Id);
    assert_eq!(
        tables[1].columns[1].stype_before_key,
        Some(Stype::Numerical)
    );
    // Text named as the time column means a timestamp.
    let tables = named(&[], &[("orders", "note")]).unwrap();
    assert_eq!(tables[0].time_column.as_deref(), Some("note"));
    assert_eq!(tables[0].columns[4].stype, Stype::Timestamp);
    let rule = |result: Result<Vec<InferredTable>, sdm_contract::CoreError>| {
        let err = result.unwrap_err();
        (
            err.code,
            err.fields.get("field").or(err.fields.get("rule")).cloned(),
        )
    };
    let invalid = |field: &str| (ErrorCode::InvalidRequest, Some(Detail::text(field)));
    assert_eq!(
        rule(named(&[("nowhere", &["x"])], &[])),
        invalid("primary_keys")
    );
    assert_eq!(
        rule(named(&[("users", &["nothing"])], &[])),
        invalid("primary_keys")
    );
    assert_eq!(
        rule(named(&[("orders", &["ts"])], &[])),
        invalid("primary_keys")
    );
    assert_eq!(rule(named(&[("users", &[])], &[])), invalid("primary_keys"));
    assert_eq!(
        rule(named(&[], &[("nowhere", "ts")])),
        invalid("time_columns")
    );
    assert_eq!(
        rule(named(&[], &[("orders", "nothing")])),
        invalid("time_columns")
    );
    assert_eq!(
        rule(named(&[], &[("orders", "order_id")])),
        invalid("time_columns")
    );
    assert_eq!(
        rule(named(&[], &[("orders", "price")])),
        invalid("time_columns")
    );
    assert_eq!(
        rule(named(&[("orders", &["order_id", "user_id"])], &[])),
        (
            ErrorCode::UnsupportedRequest,
            Some(Detail::text("composite_key"))
        )
    );
}

/// The relationships a caller gives (T1.23) are the schema's, in place of
/// the declared and inferred ones, each checked as the reference checks
/// them, each refusal seeded; an empty list says there are none. A table
/// whose time column the caller named is neither settled by the links nor
/// warned of.
#[test]
fn relationships_given_are_the_schema_s_and_checked() {
    let given = |links: &[Link], times: Option<&std::collections::BTreeMap<String, String>>| {
        assemble::assemble(
            vec![orders(Dtype::Int), users()],
            vec![],
            Given {
                time_columns: times,
                relationships: Some(links),
            },
            None,
        )
    };
    let none = given(&[], None).unwrap();
    assert!(none.description.links.is_empty());
    assert_eq!(
        stype(&none.description, "orders", "user_id"),
        (Stype::Numerical, None)
    );
    let one = [link(("orders", "user_id"), ("users", "user_id"))];
    let made = given(&one, None).unwrap();
    assert_eq!(links(&made.description).len(), 1);
    for bad in [
        link(("orders", "gone"), ("users", "user_id")),
        link(("orders", "user_id"), ("users", "age")),
        link(("orders", "price"), ("nowhere", "user_id")),
    ] {
        let err = given(&[bad], None).unwrap_err();
        assert_eq!(err.code, ErrorCode::InvalidRequest);
        assert_eq!(err.fields["field"], Detail::text("relationships"));
    }
    let mut strings = vec![orders(Dtype::String), users()];
    strings.truncate(2);
    let err = assemble::assemble(
        strings,
        vec![],
        Given {
            time_columns: None,
            relationships: Some(&one),
        },
        None,
    )
    .unwrap_err();
    assert_eq!(err.code, ErrorCode::InvalidRequest);
    // A lone edit stamp a link would refuse, kept where its table's time
    // column was named, and not warned of.
    let mut subscriptions = table(
        "subscriptions",
        vec![
            column("sub_id", Dtype::Int, Stype::Id),
            column("plan_id", Dtype::Int, Stype::Id),
        ],
        Some("sub_id"),
    );
    subscriptions.time_column = None;
    let timed = std::collections::BTreeMap::from([("plans".to_owned(), "modified_at".to_owned())]);
    let made = assemble::assemble(
        vec![plans(), subscriptions],
        vec![],
        Given {
            time_columns: Some(&timed),
            relationships: None,
        },
        None,
    )
    .unwrap();
    assert_eq!(
        described(&made.description, "plans").time_column.as_deref(),
        Some("modified_at")
    );
    assert!(made.warnings.is_empty(), "{:?}", made.warnings);
}
