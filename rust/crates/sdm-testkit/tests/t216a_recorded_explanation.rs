// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! T2.16a's exit on `rust-host` (the package's test author's), beside
//! `contract/tests/test_t216a_recorded_explanation.py`, which holds
//! `python-core` to the same: the host records a result's explanation and
//! its items' metadata, so the gate's `core` rule compares them across
//! hosts (T2.13's exits; T2.1n round 1, P3 3).
//!
//! Where: `core.predicted.result.explanation`, the JSON text the core put
//! on the result's Arrow schema metadata under `explanation`, absent where
//! the core gives none; each item's metadata in the result rows'
//! `metadata` column. The values are the reference's
//! (`contract/t213-option-vectors.json`); a forecast's whole explanation
//! (`{"by_forecast_step": ...}`) is on the schema metadata only, so
//! `explain.forecast` tells a recorded explanation from the rows' column.
//!
//! Each is read from the observation as `sdm-contract-run` writes it
//! (`observation::write`), not from the in-process answer: the gate
//! overlays the checkpoint's writer, so the explanation reaches the
//! observation through the writer alone.

use std::path::{Path, PathBuf};

use sdm_testkit::bases;
use sdm_testkit::case;
use sdm_testkit::host;
use sdm_testkit::observation::{self, Identity};
use serde_json::Value;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

fn read_json(path: &Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn vector(id: &str) -> Value {
    let all = read_json(&root().join("contract/t213-option-vectors.json"));
    assert_eq!(all["recorded_by"], "contract/tools/option_record.py");
    all["vectors"][id].clone()
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
        let wanted = vector(id)["explanation"].clone();
        assert!(!wanted.is_null(), "{id}: the vector holds no explanation");
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
        assert!(vector(id)["explanation"].is_null());
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
        let wanted = vector(id)["item_metadata"].clone();
        let wanted = wanted.as_array().unwrap();
        assert_eq!(recorded.len(), wanted.len(), "{id}");
        for pair in wanted {
            assert!(
                recorded.iter().any(|got| same(&pair[1], got)),
                "{id}: no item metadata {} in {recorded:?}",
                pair[1]
            );
        }
    }
}
