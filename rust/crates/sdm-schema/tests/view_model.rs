// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Logical view-model decisions and rollback, asserted from explicit facts.
//! Time selection tests verify callback requests, choices, retained columns and notes.

use sdm_contract::ops::{Dtype, Stype};
use sdm_schema::view::model::{
    self, Added, Choice, Edge, EdgeFacts, EdgeReason, JoinNote, JoinRefusal, JoinStep, KeyCheck,
    KeyReason, LinkFailure, MeasureColumn, ModelColumn, ModelTable, RawAdd, RelationshipRefusal,
    RelationshipStep, TimeInput, ViewModel,
};
#[test]
fn join_key_counts_distinguish_unknown_empty_unique_and_repeated() {
    assert_eq!(model::join_key_reason(None, None, 0), None);
    assert_eq!(model::join_key_reason(Some(10), None, 0), None);
    assert_eq!(
        model::join_key_reason(Some(0), Some(0), 0),
        Some(KeyReason::NoValue)
    );
    assert_eq!(
        model::join_key_reason(Some(10), Some(0), 0),
        Some(KeyReason::NoValue)
    );
    assert_eq!(model::join_key_reason(Some(10), Some(8), 0), None); // Null keys are not repeats.
    assert_eq!(
        model::join_key_reason(Some(10), Some(8), 2),
        Some(KeyReason::NotUnique {
            repeats: 2,
            rows: 10
        })
    );
}

#[test]
fn measure_notes_resolve_renames_and_refuse_shadowed_source_columns() {
    let columns = [
        MeasureColumn {
            name: "Amount".into(),
            expression: false,
            renames: None,
        },
        MeasureColumn {
            name: "price".into(),
            expression: true,
            renames: None,
        },
        MeasureColumn {
            name: "Revenue".into(),
            expression: true,
            renames: Some("raw_revenue".into()),
        },
    ];
    let aggregation =
        |function: &str, name: Option<&str>| Some((function.into(), name.map(str::to_owned)));
    let notes = model::measure_notes(
        "orders",
        &columns,
        &["Amount".into(), "price".into(), "raw_revenue".into()],
        &[
            ("total".into(), aggregation("SUM", Some("amount"))),
            ("shadow".into(), aggregation("SUM", Some("price"))),
            ("renamed".into(), aggregation("SUM", Some("RAW_REVENUE"))),
            ("rows".into(), aggregation("COUNT", None)),
            ("missing".into(), aggregation("AVG", Some("absent"))),
            ("complex".into(), None),
        ],
    );
    let got: Vec<_> = notes
        .iter()
        .map(|n| {
            (
                n.measure.as_str(),
                n.table.as_str(),
                n.function.as_str(),
                n.column.as_deref(),
            )
        })
        .collect();
    assert_eq!(
        got,
        [
            ("total", "orders", "SUM", Some("Amount")),
            ("renamed", "orders", "SUM", Some("Revenue")),
            ("rows", "orders", "COUNT", None)
        ]
    );
}

fn time_input(name: &str, source: bool, derived: Option<&str>) -> TimeInput {
    TimeInput {
        name: name.into(),
        timestamp: true,
        source,
        derived: derived.map(str::to_owned),
    }
}

#[test]
fn derived_time_adds_raw_source_once_and_reports_replaced_dimensions() {
    let columns = [
        time_input("day", false, Some("event_time")),
        time_input("month", false, Some("EVENT_TIME")),
    ];
    let mut additions = Vec::new();
    let result = model::time_column(
        "orders",
        &columns,
        &[("event_time".into(), true)],
        &[],
        |request: &RawAdd| {
            additions.push(request.clone());
            Added {
                added: true,
                timestamp: true,
            }
        },
        |candidates| {
            assert_eq!(candidates, ["event_time"]);
            Choice {
                column: Some("event_time".into()),
                fallback: None,
            }
        },
    );
    assert_eq!(
        additions,
        [RawAdd {
            name: "event_time".into(),
            raw: "event_time".into()
        }]
    );
    assert_eq!(result.chosen.as_deref(), Some("event_time"));
    assert_eq!(result.kept, ["event_time"]);
    assert!(result.removed.is_empty());
    assert_eq!(result.instead_of, ["day", "month"]);
    let note = result.note.unwrap();
    assert_eq!(note.table, "orders");
    assert!(note.added);
    assert!(!note.window_order);
    assert_eq!(note.source_column, None);
}

#[test]
fn time_name_collisions_are_disambiguated_and_unused_additions_are_removed() {
    let columns = [
        time_input("event_time", false, Some("event_time")),
        time_input("event_time_raw", false, None),
        time_input("other", true, None),
    ];
    let result = model::time_column(
        "orders",
        &columns,
        &[("event_time".into(), true)],
        &[],
        |request| {
            assert_eq!(
                request,
                &RawAdd {
                    name: "event_time_raw_2".into(),
                    raw: "event_time".into()
                }
            );
            Added {
                added: true,
                timestamp: true,
            }
        },
        |candidates| {
            assert_eq!(candidates, ["event_time_raw_2", "event_time_raw", "other"]);
            Choice {
                column: Some("other".into()),
                fallback: None,
            }
        },
    );
    assert!(result.kept.is_empty());
    assert_eq!(result.removed, ["event_time_raw_2"]);
    assert_eq!(result.chosen.as_deref(), Some("other"));
    assert_eq!(result.note, None);
}

#[test]
fn window_time_is_offered_first_then_all_candidates_and_fallback_is_kept() {
    let columns = [
        time_input("day", false, Some("event_time")),
        time_input("received", true, None),
    ];
    let mut calls = 0;
    let result = model::time_column(
        "orders",
        &columns,
        &[("event_time".into(), true)],
        &["received".into()],
        |_| Added {
            added: true,
            timestamp: true,
        },
        |candidates| {
            calls += 1;
            if calls == 1 {
                assert_eq!(candidates, ["received"]);
                Choice::default()
            } else {
                assert_eq!(candidates, ["event_time", "received"]);
                Choice {
                    column: Some("received".into()),
                    fallback: Some("event_time".into()),
                }
            }
        },
    );
    assert_eq!(calls, 2);
    assert_eq!(result.kept, ["event_time"]);
    assert!(result.removed.is_empty());
    assert!(result.note.unwrap().window_order);
}

#[test]
fn unsuccessful_raw_additions_leave_original_candidates_and_no_spurious_note() {
    for added in [false, true] {
        let result = model::time_column(
            "orders",
            &[time_input("day", false, Some("event_time"))],
            &[("event_time".into(), true)],
            &[],
            |_| Added {
                added,
                timestamp: false,
            },
            |candidates| {
                assert_eq!(candidates, ["day"]);
                Choice {
                    column: Some("day".into()),
                    fallback: None,
                }
            },
        );
        assert_eq!(result.chosen.as_deref(), Some("day"));
        assert_eq!(
            result.removed,
            if added { vec!["event_time"] } else { vec![] }
        );
        assert!(result.kept.is_empty());
        assert_eq!(result.note, None);
    }
    let result = model::time_column(
        "orders",
        &[],
        &[],
        &[],
        |_| panic!("nothing to add"),
        |_| panic!("nothing to choose"),
    );
    assert_eq!(result.chosen, None);
}

/// `_edge_error`'s reasons by name (`RE/graph.py:3283-3348`).
#[test]
fn each_edge_reason_is_given_where_the_old_client_gave_it() {
    let facts =
        |fkey_dtype: Option<Dtype>, target: Option<Dtype>, source_key: Option<&str>| EdgeFacts {
            fkey: "k".into(),
            fkey_dtype,
            target_key: target.map(|d| ("id".to_owned(), d)),
            source_key: source_key.map(str::to_owned),
        };
    assert_eq!(
        model::edge_reason(&facts(None, Some(Dtype::Int), None)),
        Some(EdgeReason::NoColumn)
    );
    assert_eq!(
        model::edge_reason(&facts(Some(Dtype::Int), None, None)),
        Some(EdgeReason::NoPrimaryKey)
    );
    assert_eq!(
        model::edge_reason(&facts(Some(Dtype::Int), Some(Dtype::Int), Some("k"))),
        Some(EdgeReason::KeyIsPrimary)
    );
    assert_eq!(
        model::edge_reason(&facts(Some(Dtype::Date), Some(Dtype::Int), None)),
        Some(EdgeReason::NotNumberOrString { dtype: Dtype::Date })
    );
    assert_eq!(
        model::edge_reason(&facts(Some(Dtype::String), Some(Dtype::Float), None)),
        Some(EdgeReason::Incompatible {
            fkey_dtype: Dtype::String,
            pkey: "id".into(),
            pkey_dtype: Dtype::Float
        })
    );
    assert_eq!(
        model::edge_reason(&facts(Some(Dtype::Int), Some(Dtype::Float), None)),
        None
    );
    assert_eq!(
        model::edge_reason(&facts(Some(Dtype::Binary), Some(Dtype::String), None)),
        None
    );
}

fn column(name: &str, dtype: Dtype, stype: Stype) -> ModelColumn {
    ModelColumn {
        name: name.into(),
        expr: None,
        dtype,
        stype,
    }
}

fn table(name: &str, columns: Vec<ModelColumn>) -> ModelTable {
    ModelTable {
        name: name.into(),
        columns,
        primary_key: None,
    }
}

/// The fact table `orders` and a join `users` on `user_id`, many orders to
/// one user.
fn orders_and_users() -> (ViewModel, JoinStep) {
    let mut view = ViewModel::new();
    view.add_table(table(
        "orders",
        vec![
            column("order_id", Dtype::Int, Stype::Numerical),
            column("user_id", Dtype::Int, Stype::Numerical),
        ],
    ))
    .unwrap();
    let users = table(
        "users",
        vec![
            column("user_id", Dtype::Int, Stype::Numerical),
            column("country", Dtype::String, Stype::Categorical),
        ],
    );
    let step = JoinStep {
        table: users,
        one: ("users".into(), "user_id".into()),
        many: ("orders".into(), "user_id".into()),
        count: KeyCheck::Counted {
            rows: Some(10),
            distinct: Some(10),
            repeats: 0,
        },
        source_columns: vec![],
    };
    (view, step)
}

#[test]
fn a_join_on_a_unique_key_links_its_table() {
    let (mut view, step) = orders_and_users();
    assert_eq!(view.add_join(step), Ok(vec![]));
    let users = view.table("users").unwrap();
    assert_eq!(users.primary_key.as_deref(), Some("user_id"));
    assert_eq!(users.column("user_id").unwrap().stype, Stype::Id);
    assert_eq!(
        view.edges,
        vec![Edge {
            source: "orders".into(),
            key: "user_id".into(),
            target: "users".into()
        }]
    );
    assert_eq!(
        view.table("orders")
            .unwrap()
            .column("user_id")
            .unwrap()
            .stype,
        Stype::Id
    );
    assert!(view.linked("users"));
    assert!(!view.linked("orders"));
}

#[test]
fn a_join_on_a_repeating_key_is_refused_and_leaves_no_trace() {
    let (mut view, mut step) = orders_and_users();
    step.count = KeyCheck::Counted {
        rows: Some(10),
        distinct: Some(7),
        repeats: 3,
    };
    let before = view.clone();
    assert_eq!(
        view.add_join(step),
        Err(JoinRefusal::Key {
            table: "users".into(),
            key: "user_id".into(),
            reason: KeyReason::NotUnique {
                repeats: 3,
                rows: 10
            }
        })
    );
    assert_eq!(view, before);
}

#[test]
fn a_key_that_could_not_be_counted_is_kept_unchecked() {
    let (mut view, mut step) = orders_and_users();
    step.count = KeyCheck::Failed;
    assert_eq!(
        view.add_join(step),
        Ok(vec![JoinNote::KeptUnchecked {
            key: "user_id".into()
        }])
    );
    assert!(view.table("users").is_some());
}

#[test]
fn a_table_keyed_otherwise_refuses_the_join() {
    let (mut view, mut step) = orders_and_users();
    // A one-to-many join whose one side is the fact table, already keyed.
    view.tables[0].primary_key = Some("order_id".into());
    step.one = ("orders".into(), "user_id".into());
    step.many = ("users".into(), "user_id".into());
    let before = view.clone();
    assert_eq!(
        view.add_join(step),
        Err(JoinRefusal::AlreadyKeyed {
            table: "orders".into(),
            key: "order_id".into()
        })
    );
    assert_eq!(view, before);
}

#[test]
fn a_derived_key_gives_way_to_its_source_column_and_comes_back_on_a_refusal() {
    let (mut view, mut step) = orders_and_users();
    // The fact table holds `user_id` as an expression; the join reads the
    // source column of that name.
    view.tables[0].columns[1].expr = Some("user_id + 0".into());
    step.source_columns = vec![(
        "orders".into(),
        column("user_id", Dtype::Int, Stype::Numerical),
    )];
    let notes = view.clone().add_join(step.clone()).unwrap();
    assert_eq!(
        notes,
        vec![JoinNote::ReplacedDerived {
            key: "user_id".into(),
            table: "orders".into()
        }]
    );
    // Refused later, at the link, every change is undone.
    step.source_columns[0].1.dtype = Dtype::Date;
    let before = view.clone();
    let refused = view.add_join(step).unwrap_err();
    assert!(matches!(
        refused,
        JoinRefusal::NotLinked(LinkFailure::ForeignKeyType { .. })
    ));
    assert_eq!(view.table("users"), None);
    let orders = view.table("orders").unwrap();
    assert_eq!(
        orders.column("user_id"),
        before.table("orders").unwrap().column("user_id")
    );
    assert_eq!(orders.primary_key, None);
}

#[test]
fn a_key_the_table_lacks_is_added_from_its_source() {
    let (mut view, mut step) = orders_and_users();
    view.tables[0].columns.remove(1);
    step.source_columns = vec![(
        "orders".into(),
        column("user_id", Dtype::Int, Stype::Numerical),
    )];
    view.add_join(step.clone()).unwrap();
    assert!(view.table("orders").unwrap().column("user_id").is_some());

    // With no such source column, the keys cannot be linked.
    let (mut view, mut step) = orders_and_users();
    view.tables[0].columns.remove(1);
    step.source_columns.clear();
    let before = view.clone();
    assert_eq!(
        view.add_join(step),
        Err(JoinRefusal::NotLinked(LinkFailure::NoSourceColumn {
            key: "user_id".into()
        }))
    );
    assert_eq!(view, before);
}

#[test]
fn a_relationship_links_two_logical_tables_as_the_old_client_linked_them() {
    let mut view = ViewModel::new();
    let mut users = table(
        "users",
        vec![column("user_id", Dtype::Int, Stype::Numerical)],
    );
    users.primary_key = Some("user_id".into());
    users.columns[0].stype = Stype::Id;
    let mut orders = table("orders", vec![column("order_id", Dtype::Int, Stype::Id)]);
    orders.primary_key = Some("order_id".into());
    view.add_table(users).unwrap();
    view.add_table(orders).unwrap();
    let step = |left_key: &str, right_key: &str| RelationshipStep {
        left_table: "orders".into(),
        left_key: left_key.into(),
        right_table: "users".into(),
        right_key: right_key.into(),
        left_key_column: Some(column("user_id", Dtype::Int, Stype::Numerical)),
    };
    assert_eq!(
        view.add_relationship(step("user_id", "country")),
        Err(RelationshipRefusal::NotPrimaryKey {
            key: "country".into(),
            table: "users".into()
        })
    );
    assert_eq!(
        view.add_relationship(step("order_id", "user_id")),
        Err(RelationshipRefusal::IsPrimaryKey {
            key: "order_id".into(),
            table: "orders".into()
        })
    );
    // The referencing key, missing, is added from the base table.
    view.add_relationship(step("user_id", "user_id")).unwrap();
    assert!(view.table("orders").unwrap().column("user_id").is_some());
    assert_eq!(view.edges.len(), 1);
    // The same link again is refused, and the column it added stays, as
    // the first link added it.
    assert!(matches!(
        view.add_relationship(step("user_id", "user_id")),
        Err(RelationshipRefusal::NotLinked(LinkFailure::EdgeExists(_)))
    ));
}

#[test]
fn a_relationship_that_cannot_link_takes_back_the_column_it_added() {
    let mut view = ViewModel::new();
    let mut users = table("users", vec![column("user_id", Dtype::Int, Stype::Id)]);
    users.primary_key = Some("user_id".into());
    view.add_table(users).unwrap();
    view.add_table(table("orders", vec![])).unwrap();
    let refused = view.add_relationship(RelationshipStep {
        left_table: "orders".into(),
        left_key: "placed".into(),
        right_table: "users".into(),
        right_key: "user_id".into(),
        left_key_column: Some(column("placed", Dtype::Date, Stype::Timestamp)),
    });
    assert!(matches!(
        refused,
        Err(RelationshipRefusal::NotLinked(
            LinkFailure::ForeignKeyType { .. }
        ))
    ));
    assert!(view.table("orders").unwrap().columns.is_empty());
}

#[test]
fn invalid_edges_are_dropped_and_an_empty_model_is_refused() {
    assert!(ViewModel::new().validate().is_err());
    let (mut view, step) = orders_and_users();
    view.add_join(step).unwrap();
    assert!(view.validate().is_ok());
    // The key a link reads leaves its table: the edge is dropped, saying
    // why.
    let orders = view.tables.iter_mut().find(|t| t.name == "orders").unwrap();
    orders.columns.retain(|c| c.name != "user_id");
    assert!(view.validate().is_err());
    let dropped = view.drop_invalid_edges();
    assert_eq!(dropped.len(), 1);
    assert_eq!(dropped[0].1, EdgeReason::NoColumn);
    assert!(view.edges.is_empty());
    assert!(view.validate().is_ok());
}
