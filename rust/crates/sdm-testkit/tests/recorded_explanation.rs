// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Result observations preserve authored endpoint explanations and per-item metadata.

use std::path::PathBuf;

use sdm_testkit::bases;
use sdm_testkit::case;
use sdm_testkit::host;
use sdm_testkit::observation::{self, Identity};
use serde_json::Value;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

fn expected_explanation(call: u64) -> Value {
    // Authored endpoint answers for row zero of calls zero and one.
    let (rank, score) = match call {
        0 => (1, 0.1),
        1 => (4, 0.4),
        _ => unreachable!(),
    };
    serde_json::json!({"format":"scripted", "summary":format!("row 0 of call {call}"),
        "score":score, "rank":rank, "path":[call,0],
        "detail":{"call":call,"row_index":0,"note":null}})
}

/// The case's observation on DuckDB, as `sdm-contract-run` writes it: its
/// `core.predicted`.
fn predicted(id: &str) -> Value {
    let contract = root().join("contract");
    let bases = bases::load(&contract.join("bases.yaml")).unwrap();
    let loaded = case::load(&[contract.join(format!("cases/s1/{id}.json"))]).unwrap();
    let observed = host::run(&loaded[0].case, &bases).unwrap();
    let identity = Identity::new(&root()).unwrap();
    let line = observation::write(&identity, &loaded[0], &observed, None);
    assert_eq!(
        line["outcome"]["result"], "success",
        "{id}: {}",
        line["outcome"]
    );
    line["core"]["predicted"].clone()
}

/// Two JSON values the same, a number by its value.
fn same(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Object(x), Value::Object(y)) => {
            x.len() == y.len() && x.iter().all(|(k, v)| y.get(k).is_some_and(|w| same(v, w)))
        }
        (Value::Array(x), Value::Array(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(v, w)| same(v, w))
        }
        (Value::Number(x), Value::Number(y)) => x.as_f64() == y.as_f64(),
        _ => a == b,
    }
}

#[test]
fn the_result_s_explanation_is_recorded() {
    for id in [
        "explain.answered",
        "explain.forecast",
        "metadata.items-explained",
    ] {
        let wanted = if id == "explain.forecast" {
            serde_json::json!({"by_forecast_step":{"1":expected_explanation(0),"2":expected_explanation(1)}})
        } else {
            expected_explanation(0)
        };
        let answer = predicted(id);
        let Some(text) = answer["result"]["explanation"].as_str() else {
            panic!("{id}: no explanation recorded: {}", answer["result"]);
        };
        let read: Value = serde_json::from_str(text).unwrap();
        assert!(
            same(&read, &wanted),
            "{id}: {read} where the reference has {wanted}"
        );
    }
}

#[test]
fn a_result_with_no_explanation_records_none() {
    for id in ["explain.not-answered", "metadata.items"] {
        let answer = predicted(id);
        assert!(
            answer["result"].get("explanation").is_none(),
            "{id}: {}",
            answer["result"]["explanation"]
        );
    }
}

#[test]
fn each_item_s_metadata_is_recorded() {
    for id in ["metadata.items", "metadata.items-across-calls"] {
        let answer = predicted(id);
        let result = &answer["result"];
        let at = result["columns"]
            .as_array()
            .unwrap()
            .iter()
            .position(|c| c["name"] == "metadata")
            .unwrap_or_else(|| panic!("{id}: no metadata column: {}", result["columns"]));
        let recorded: Vec<Value> = result["rows"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| match &row[at] {
                Value::Null => Value::Null,
                cell => serde_json::from_str(cell["utf8"].as_str().unwrap()).unwrap(),
            })
            .collect();
        let positions = if id == "metadata.items-across-calls" {
            vec![(0, 0), (0, 1), (1, 0)]
        } else {
            vec![(0, 0), (0, 1)]
        };
        let wanted: Vec<_> = positions
            .iter()
            .map(|(call, row)| serde_json::json!({"call":call,"row_index":row,"source":"scripted"}))
            .collect();
        assert_eq!(recorded, wanted);
        let index = result["columns"]
            .as_array()
            .unwrap()
            .iter()
            .position(|column| column["name"] == "row_index")
            .unwrap();
        for (i, row) in result["rows"].as_array().unwrap().iter().enumerate() {
            assert_eq!(row[index], serde_json::json!({"i64":i.to_string()}));
        }
    }
}
