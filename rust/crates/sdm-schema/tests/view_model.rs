// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The logical model's steps (T3.5, design S5: `add_table`, `add_join`
//! with its undo, `add_relationship`, and the finish: measure notes, time
//! columns, invalid edges, validation), the graph's part of the old
//! client's conversion (`RE/graph.py:250-421`, `:1820-1943`, `:2209-2256`,
//! `:3256-3371`).
//!
//! Against `contract/view-step-vectors.json` where the old client's tests
//! reached a step it records: a join key's count, the measure notes, the
//! time column (driven with the reads and choices the old client made on
//! the way, which the core makes from the rows it read), and each edge's
//! error. The join and relationship steps, which change several tables at
//! once and undo all of it on a refusal, are held to the old routine's
//! outcomes one scenario each.

use std::collections::BTreeMap;

use sdm_contract::ops::{Dtype, Stype};
use sdm_schema::view::model::{
    self, Added, Choice, Edge, EdgeFacts, EdgeReason, JoinNote, JoinRefusal, JoinStep, KeyCheck,
    KeyReason, LinkFailure, MeasureColumn, ModelColumn, ModelTable, RawAdd, RelationshipRefusal,
    RelationshipStep, TimeInput, ViewModel,
};
use serde_json::Value as Json;

const VECTORS: &str = include_str!("../../../../contract/view-step-vectors.json");

fn document() -> Json {
    serde_json::from_str(VECTORS).unwrap()
}

fn texts(value: &Json) -> Vec<String> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|item| item.as_str().unwrap().to_owned())
        .collect()
}

fn dtype(name: &str) -> Dtype {
    serde_json::from_value(Json::String(name.to_owned())).unwrap()
}

fn optional_text(value: &Json) -> Option<String> {
    value.as_str().map(str::to_owned)
}

#[test]
fn a_join_key_s_count_is_judged_as_the_old_client_judged_it() {
    let document = document();
    let vectors = document["join_key_reason"].as_array().unwrap();
    assert!(vectors.len() >= 10);
    for vector in vectors {
        let count = &vector["count"];
        let reason = model::join_key_reason(
            count["rows"].as_u64(),
            count["distinct"].as_u64(),
            count["repeats"].as_u64().unwrap(),
        );
        let expected = vector["reason"].as_str();
        match (&reason, expected) {
            (None, None) => {}
            (Some(KeyReason::NoValue), Some(text)) => assert!(text.contains("holds no value")),
            (Some(KeyReason::NotUnique { repeats, rows }), Some(text)) => {
                assert!(text.contains("is not unique"), "{text}");
                assert_eq!(*repeats, count["repeats"].as_u64().unwrap());
                assert_eq!(*rows, count["rows"].as_u64().unwrap());
            }
            _ => panic!("{vector}: {reason:?}"),
        }
    }
}

#[test]
fn measure_notes_are_the_old_client_s() {
    let document = document();
    let renamed: BTreeMap<String, Option<String>> = document["renamed_column"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| {
            (
                v["expr"].as_str().unwrap().to_owned(),
                optional_text(&v["column"]),
            )
        })
        .collect();
    let vectors = document["measure_notes"].as_array().unwrap();
    assert!(vectors.len() >= 20);
    for vector in vectors {
        let columns: Vec<MeasureColumn> = vector["columns"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| {
                let expr = optional_text(&c["expr"]);
                MeasureColumn {
                    name: c["name"].as_str().unwrap().to_owned(),
                    renames: expr.as_ref().and_then(|e| renamed[e].clone()),
                    expression: expr.is_some(),
                }
            })
            .collect();
        let measures: Vec<(String, Option<model::Aggregation>)> = vector["measures"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| {
                (
                    m[0].as_str().unwrap().to_owned(),
                    m[1].as_array()
                        .map(|a| (a[0].as_str().unwrap().to_owned(), optional_text(&a[1]))),
                )
            })
            .collect();
        let notes = model::measure_notes(
            vector["table"].as_str().unwrap(),
            &columns,
            &texts(&vector["source_columns"]),
            &measures,
        );
        let got: Vec<Json> = notes
            .iter()
            .map(|n| serde_json::json!([n.measure, n.table, n.function, n.column]))
            .collect();
        let expected: Vec<Json> = vector["notes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|n| serde_json::json!([n["measure"], n["table"], n["function"], n["column"]]))
            .collect();
        assert_eq!(got, expected, "{vector}");
    }
}

#[test]
fn a_time_column_is_taken_as_the_old_client_took_it() {
    let document = document();
    let derived: BTreeMap<String, Option<String>> = document["derived_column"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| {
            (
                v["expr"].as_str().unwrap().to_owned(),
                optional_text(&v["column"]),
            )
        })
        .collect();
    let vectors = document["time_column"].as_array().unwrap();
    assert!(vectors.len() >= 30);
    for vector in vectors {
        let columns: Vec<TimeInput> = vector["columns"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| {
                let expr = optional_text(&c["expr"]);
                TimeInput {
                    name: c["name"].as_str().unwrap().to_owned(),
                    timestamp: c["stype"] == "timestamp",
                    source: expr.is_none(),
                    derived: expr
                        .as_ref()
                        .and_then(|e| derived.get(e).cloned().flatten()),
                }
            })
            .collect();
        let source: Vec<(String, bool)> = vector["source_columns"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| {
                (
                    c[0].as_str().unwrap().to_owned(),
                    matches!(c[1].as_str().unwrap(), "date" | "time"),
                )
            })
            .collect();
        let mut adds = vector["adds"].as_array().unwrap().iter();
        let mut choices = vector["choices"].as_array().unwrap().iter();
        let outcome = model::time_column(
            vector["table"].as_str().unwrap(),
            &columns,
            &source,
            &texts(&vector["window_orders"]),
            |add: &RawAdd| {
                let recorded = adds.next().expect("an add the old client made");
                assert_eq!(recorded["name"], add.name.as_str(), "{vector}");
                let expr =
                    (add.name != add.raw).then(|| format!("`{}`", add.raw.replace('`', "``")));
                assert_eq!(optional_text(&recorded["expr"]), expr, "{vector}");
                Added {
                    added: recorded["added"].as_bool().unwrap(),
                    timestamp: recorded["stype"] == "timestamp",
                }
            },
            |candidates: &[String]| {
                let recorded = choices.next().expect("a choice the old client made");
                assert_eq!(texts(&recorded["candidates"]), candidates, "{vector}");
                Choice {
                    column: optional_text(&recorded["chosen"]),
                    fallback: optional_text(&recorded["fallback"]),
                }
            },
        );
        assert!(adds.next().is_none(), "{vector}");
        assert!(choices.next().is_none(), "{vector}");
        let mut after: Vec<String> = columns.iter().map(|c| c.name.clone()).collect();
        after.extend(outcome.kept.iter().cloned());
        assert_eq!(after, texts(&vector["columns_after"]), "{vector}");
        if !vector["choices"].as_array().unwrap().is_empty() {
            assert_eq!(outcome.instead_of, texts(&vector["instead_of"]), "{vector}");
        }
        let note = outcome.note.as_ref().map(|n| {
            serde_json::json!({
                "table": n.table, "column": n.column, "instead_of": n.instead_of,
                "source_column": n.source_column, "added": n.added,
                "window_order": n.window_order,
            })
        });
        let expected = vector["note"].as_object().map(|n| {
            serde_json::json!({
                "table": n["table"], "column": n["column"], "instead_of": n["instead_of"],
                "source_column": n["source_column"], "added": n["added"],
                "window_order": n["window_order"],
            })
        });
        assert_eq!(note, expected, "{vector}");
    }
}

#[test]
fn an_edge_is_judged_as_the_old_client_judged_it() {
    let document = document();
    for vector in document["edge_error"].as_array().unwrap() {
        let facts = EdgeFacts {
            fkey_dtype: vector["fkey_dtype"].as_str().map(dtype),
            target_key: vector["target_key"].as_array().map(|k| {
                (
                    k[0].as_str().unwrap().to_owned(),
                    dtype(k[1].as_str().unwrap()),
                )
            }),
            source_key: optional_text(&vector["source_key"]),
            fkey: vector["edge"][1].as_str().unwrap().to_owned(),
        };
        let reason = model::edge_reason(&facts);
        assert_eq!(
            reason.is_some(),
            vector["reason"].is_string(),
            "{vector}: {reason:?}"
        );
    }
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
