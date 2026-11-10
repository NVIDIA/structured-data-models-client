// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Classification labels, leakage and limits through the real Rust/DuckDB host.

use std::path::PathBuf;

use sdm_testkit::case::Case;
use sdm_testkit::host::{self, Answer};
use sdm_testkit::{bases, case};
use serde_json::{Value, json};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

fn the_case(id: &str) -> (Case, bases::Bases) {
    let contract = root().join("contract");
    let loaded = case::load_for(&[contract.join("cases")], case::Database::Duckdb).unwrap();
    let found = loaded
        .iter()
        .find(|l| l.case.id == id)
        .unwrap_or_else(|| panic!("{id} is a case"))
        .case
        .clone();
    (found, bases::load(&contract.join("bases.yaml")).unwrap())
}

/// The inverse of `host::numbers_as_text`: `{"number": text}` read back as
/// the number written.
fn numbers(value: Value) -> Value {
    match value {
        Value::Object(members) => match members.get("number") {
            Some(Value::String(text)) if members.len() == 1 => {
                serde_json::from_str(text).expect("a number's text")
            }
            _ => Value::Object(members.into_iter().map(|(k, v)| (k, numbers(v))).collect()),
        },
        Value::Array(items) => Value::Array(items.into_iter().map(numbers).collect()),
        other => other,
    }
}

/// The payload `rust-host` materializes for `id`, or the refusal's rule.
fn payload(id: &str) -> Value {
    let (case, bases) = the_case(id);
    let observed = host::run(&case, &bases).unwrap();
    match observed.materialized {
        Some(state) => numbers(state["payload"].clone()),
        None => match observed.answer {
            Answer::Error { code, fields, .. } => panic!(
                "{id}: rust-host refused it ({code}, rule {:?}); this case requires \
                 classification_task and target_leakage",
                fields.get("rule")
            ),
            other => panic!("{id}: no payload ({other:?})"),
        },
    }
}

fn classification(id: &str) {
    let got = payload(id);
    let binary = id == "base.binary";
    let (_, bases) = the_case(id);
    let db = sdm_testkit::duck::open().unwrap();
    for statement in &bases["shop"] {
        db.execute_batch(statement).unwrap();
    }
    let target = if binary {
        json!({"column_name":"TARGET","dtype":"bool","classes":["false","true"],"positive_class":"true"})
    } else {
        json!({"column_name":"TARGET","dtype":"string","classes":["IN","DE","US"]})
    };
    assert_eq!(
        got["task"],
        json!({"kind":if binary {"binary_classification"} else {"multiclass_classification"},"target":target,"entity_table_names":["users"],"anchor_time_column":"__kumo_anchor_time"})
    );
    for half in ["context", "predict"] {
        let table = &got[half]["instance_table"];
        let expected_columns = if half == "context" {
            json!([
                "instance_id",
                "__kumo_entity_ref_0",
                "__kumo_anchor_time",
                "TARGET"
            ])
        } else {
            json!(["instance_id", "__kumo_entity_ref_0", "__kumo_anchor_time"])
        };
        assert_eq!(table["columns"], expected_columns);
        let rows = table["rows"].as_array().unwrap();
        if half == "predict" {
            assert_eq!(
                rows.iter()
                    .map(|r| r[1].as_i64().unwrap())
                    .collect::<Vec<_>>(),
                [0, 1]
            );
        } else {
            assert_eq!(rows.len(), if binary { 112 } else { 6 });
            for row in rows {
                let entity = row[1].as_i64().unwrap();
                if binary {
                    let anchor = row[2].as_str().unwrap();
                    let expected: bool = db.query_row("SELECT COUNT(*) > 0 FROM orders WHERE user_id = ? AND ts > CAST(? AS TIMESTAMP) AND ts <= CAST(? AS TIMESTAMP) + INTERVAL 30 DAY", duckdb::params![entity, anchor, anchor], |r|r.get(0)).unwrap();
                    assert_eq!(row[3].as_bool(), Some(expected));
                } else {
                    assert!(
                        ![0, 1].contains(&entity),
                        "scored entities cannot train their own static labels"
                    );
                    let expected: String = db
                        .query_row(
                            "SELECT country FROM users WHERE user_id = ?",
                            [entity],
                            |r| r.get(0),
                        )
                        .unwrap();
                    assert_eq!(row[3].as_str(), Some(expected.as_str()));
                }
            }
        }
        let ids: std::collections::BTreeSet<_> =
            rows.iter().map(|r| r[0].as_i64().unwrap()).collect();
        assert_eq!(ids.len(), rows.len());
        let related_tables = got[half]["related_tables"].as_object().unwrap();
        assert_eq!(related_tables.len(), 2);
        assert!(related_tables.contains_key("users") && related_tables.contains_key("orders"));
        for (name, table) in related_tables {
            let columns = table["columns"].as_array().unwrap();
            let at = columns.iter().position(|c| c == "instance_id").unwrap();
            if !binary && name == "users" {
                assert!(!columns.iter().any(|c| c == "country"));
            }
            let related = table["rows"].as_array().unwrap();
            assert!(!related.is_empty(), "{id}/{half}/{name}");
            let actual: std::collections::BTreeSet<_> =
                related.iter().map(|r| r[at].as_i64().unwrap()).collect();
            if name == "users" {
                assert_eq!(actual, ids);
            } else {
                let expected: std::collections::BTreeSet<_> = rows.iter().filter(|r| {
                    let count: i64 = db.query_row("SELECT COUNT(*) FROM orders WHERE user_id = ? AND ts <= CAST(? AS TIMESTAMP)", duckdb::params![r[1].as_i64().unwrap(), r[2].as_str().unwrap()], |v| v.get(0)).unwrap();
                    count > 0
                }).map(|r| r[0].as_i64().unwrap()).collect();
                assert_eq!(actual, expected);
            }
            assert!(
                related
                    .iter()
                    .all(|r| ids.contains(&r[at].as_i64().unwrap())),
                "related rows from leaked instances"
            );
        }
    }
    assert_eq!(
        got["metadata"]["num_context_examples"],
        if binary { 112 } else { 8 }
    );
    assert_eq!(got["metadata"]["num_prediction_examples"], 2);
}

#[test]
fn binary_labels_match_source_counts() {
    classification("base.binary");
}
#[test]
fn multiclass_labels_and_leakage_match_source_entities() {
    classification("base.multiclass");
}

#[test]
fn no_entity_count_is_refused_by_a_limit() {
    // The core holds a materialize to no entity count. Ids the base does
    // not hold are refused for that reason, at 1,001 and at 100,001 alike.
    let with = |n: u64| {
        let (mut case, bases) = the_case("base.regression");
        let materialize = case
            .core
            .as_mut()
            .and_then(|core| core.materialize.as_mut())
            .unwrap();
        materialize.indices = (0..n)
            .map(|i| json!({"t": "i64", "v": i.to_string()}))
            .collect();
        host::run(&case, &bases).unwrap()
    };
    for n in [1001, 100_001] {
        let observed = with(n);
        let Answer::Error { code, fields, .. } = &observed.answer else {
            panic!("not refused: {:?}", observed.answer);
        };
        assert_eq!(code, "INVALID_REQUEST", "{n}");
        assert_eq!(fields.get("field").map(String::as_str), Some("indices"));
    }
}
