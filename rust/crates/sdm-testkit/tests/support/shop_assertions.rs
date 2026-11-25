// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Shared behavioral assertions for the explicitly authored eight-user shop.

use serde_json::{Value, json};
use std::collections::BTreeMap;

pub fn wire_number(value: &Value) -> i64 {
    value["number"].as_str().unwrap().parse().unwrap()
}

pub fn assert_shop_history(body: &Value) {
    let calendar = duckdb::Connection::open_in_memory().unwrap();
    let day_of = |value: &Value| -> i64 {
        calendar
            .query_row(
                "SELECT date_diff('day', DATE '2023-01-01', CAST(? AS TIMESTAMP))",
                [value.as_str().unwrap()],
                |row| row.get(0),
            )
            .unwrap()
    };
    assert_eq!(body["model"], "kumo-relational");
    assert!(!body["metadata"]["request_id"].as_str().unwrap().is_empty());
    let schema = &body["schema"];
    for tables in [
        &schema["related_tables"],
        &body["context"]["related_tables"],
        &body["predict"]["related_tables"],
    ] {
        let names: std::collections::BTreeSet<_> = tables
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(names, std::collections::BTreeSet::from(["orders", "users"]));
    }
    for (column, dtype) in [
        ("instance_id", "int64"),
        ("__kumo_entity_ref_0", "int64"),
        ("__kumo_anchor_time", "timestamp[us]"),
        ("TARGET", "float32"),
    ] {
        assert_eq!(schema["instance_table"]["columns"][column]["dtype"], dtype);
    }
    for (table, columns) in [
        (
            "users",
            vec![
                ("instance_id", "int64"),
                ("user_id", "int64"),
                ("age", "int64"),
                ("country", "string"),
            ],
        ),
        (
            "orders",
            vec![
                ("instance_id", "int64"),
                ("__node_id", "int64"),
                ("user_id", "int64"),
                ("price", "float64"),
                ("ts", "timestamp[us]"),
            ],
        ),
    ] {
        let actual = schema["related_tables"][table]["columns"]
            .as_object()
            .unwrap();
        assert_eq!(actual.len(), columns.len());
        for (column, dtype) in columns {
            assert_eq!(actual[column]["dtype"], dtype);
        }
    }
    assert_eq!(
        body["output"],
        json!({"fields":["prediction"],"prediction_statistic":"median"})
    );
    assert_eq!(
        body["schema"]["related_tables"]["users"]["primary_key"],
        json!(["instance_id", "user_id"])
    );
    assert_eq!(
        body["schema"]["related_tables"]["orders"]["primary_key"],
        json!(["instance_id", "__node_id"])
    );
    assert_eq!(
        body["schema"]["relationships"],
        json!([
            {"source_columns":["__kumo_entity_ref_0"],"target_table":"users","target_columns":["user_id"]},
            {"source_table":"orders","source_columns":["user_id"],"target_table":"users","target_columns":["user_id"]}
        ])
    );
    let mut instance_ids = std::collections::BTreeSet::new();
    for side in ["context", "predict"] {
        let table = &body[side];
        let instances: BTreeMap<_, _> = table["instance_table"]["rows"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| {
                let id = wire_number(&row[0]);
                assert!(
                    instance_ids.insert(id),
                    "duplicate instance id across context/predict"
                );
                (id, (wire_number(&row[1]), day_of(&row[2])))
            })
            .collect();
        let metadata_key = if side == "context" {
            "num_context_examples"
        } else {
            "num_prediction_examples"
        };
        assert_eq!(
            wire_number(&body["metadata"][metadata_key]),
            instances.len() as i64
        );
        let users = &table["related_tables"]["users"];
        assert_eq!(
            users["columns"],
            json!(["instance_id", "user_id", "age", "country"])
        );
        assert_eq!(users["rows"].as_array().unwrap().len(), instances.len());
        let mut seen_users = std::collections::BTreeSet::new();
        for row in users["rows"].as_array().unwrap() {
            let id = wire_number(&row[0]);
            assert!(seen_users.insert(id));
            let entity = instances[&id].0;
            assert_eq!(wire_number(&row[1]), entity);
            assert_eq!(wire_number(&row[2]), 20 + entity);
            assert_eq!(row[3], ["US", "DE", "IN"][(entity % 3) as usize]);
        }
        let orders = &table["related_tables"]["orders"];
        assert_eq!(
            orders["columns"],
            json!(["instance_id", "__node_id", "price", "ts", "user_id"])
        );
        let mut actual = BTreeMap::<i64, Vec<i64>>::new();
        let mut nodes = std::collections::BTreeSet::new();
        for row in orders["rows"].as_array().unwrap() {
            let id = wire_number(&row[0]);
            assert!(nodes.insert(wire_number(&row[1])));
            let (entity, anchor) = instances[&id];
            let day = day_of(&row[3]);
            assert_eq!(day % 7, 0);
            let order = day / 7;
            assert!((0..80).contains(&order));
            assert_eq!(order % 8, entity);
            assert_eq!(wire_number(&row[4]), entity);
            assert!(day <= anchor);
            let price: f64 = row[2]["number"].as_str().unwrap().parse().unwrap();
            assert_eq!(price, 10.0 + 1.5 * order as f64);
            actual.entry(id).or_default().push(order);
        }
        for (id, (entity, anchor)) in instances {
            let expected: Vec<_> = (0..80_i64)
                .rev()
                .filter(|i| i % 8 == entity && i * 7 <= anchor)
                .collect();
            assert_eq!(
                actual.remove(&id).unwrap_or_default(),
                expected,
                "{side} history instance={id}"
            );
        }
        assert!(actual.is_empty());
    }
}
