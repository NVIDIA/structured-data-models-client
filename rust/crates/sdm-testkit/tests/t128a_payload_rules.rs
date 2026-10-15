// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! T1.28a's exits on `rust-host`, held from outside the implementation (the
//! package's independent test author), against what the reference gave:
//! the vectors are `t128a-payload-rules.json`, recorded from the reference
//! by `t128a_payload_rules_record.py` beside it, which the Python test
//! holds `python-core` to as well, so the two hosts agree through them.
//!
//! - Exit "on both core hosts, over `base.binary` and `base.multiclass`,
//!   the classes, their order, the positive class and the context-leakage
//!   payload fields match the reference", from the real payload (not the
//!   test-only labels operation): the task, the context's and predicted
//!   rows, and each related table's columns and the instance ids it keeps
//!   (the context that answers the question taken out; a static target's
//!   column left out).
//! - Exit "python-core and rust-host public outcomes match for context
//!   leakage and limits": the leakage fields above, and the entity count's
//!   limit, 1,001 entities refused as the reference refuses them
//!   (`UNSUPPORTED_REQUEST`), which `python-core` is held to by the same
//!   recorded case.
//!
//! Before T1.28a the core refuses both classification cases by its rule
//! `classification_task`, and these tests say so.

use std::path::PathBuf;

use sdm_testkit::case::Case;
use sdm_testkit::host::{self, Answer};
use sdm_testkit::{bases, case};
use serde_json::{Map, Value, json};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

fn recorded() -> Value {
    let path = root().join("packages/structured-data-models-client/tests/t128a-payload-rules.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn expected(name: &str) -> Value {
    recorded()["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["name"] == name)
        .unwrap_or_else(|| panic!("{name} is recorded"))["expected"]
        .clone()
}

fn the_case(id: &str) -> (Case, bases::Bases) {
    let contract = root().join("contract");
    let loaded = case::load(&[contract.join("cases")]).unwrap();
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

fn rows(table: &Value) -> Value {
    json!({"columns": table["columns"], "rows": table["rows"]})
}

/// Each related table's columns and the sorted instance ids its rows hang
/// off, as the recorder keeps them.
fn instances(related: &Value) -> Value {
    let mut out = Map::new();
    let Some(tables) = related.as_object() else {
        return Value::Null;
    };
    for (name, table) in tables {
        let columns = table["columns"].as_array().cloned().unwrap_or_default();
        let at = columns.iter().position(|c| c == "instance_id");
        let mut ids: Vec<Value> = Vec::new();
        if let Some(at) = at {
            for row in table["rows"].as_array().into_iter().flatten() {
                let id = row[at].clone();
                if !ids.contains(&id) {
                    ids.push(id);
                }
            }
        }
        ids.sort_by_key(|v| v.as_i64());
        out.insert(
            name.clone(),
            json!({"columns": columns, "instance_ids": ids}),
        );
    }
    Value::Object(out)
}

/// The payload `rust-host` materializes for `id`, or the refusal's rule.
fn payload(id: &str) -> Value {
    let (case, bases) = the_case(id);
    let observed = host::run(&case, &bases).unwrap();
    match observed.materialized {
        Some(state) => numbers(state["payload"].clone()),
        None => match observed.answer {
            Answer::Error { code, fields, .. } => panic!(
                "{id}: rust-host refused it ({code}, rule {:?}); T1.28a lifts \
                 classification_task and target_leakage",
                fields.get("rule")
            ),
            other => panic!("{id}: no payload ({other:?})"),
        },
    }
}

fn classification(id: &str) {
    let want = expected(id);
    let got = payload(id);
    assert_eq!(got["task"], want["task"], "{id}: the task");
    assert_eq!(
        rows(&got["context"]["instance_table"]),
        want["context"],
        "{id}: the context's rows"
    );
    assert_eq!(
        instances(&got["context"]["related_tables"]),
        want["context_related"],
        "{id}: the context's related tables"
    );
    assert_eq!(
        rows(&got["predict"]["instance_table"]),
        want["predict"],
        "{id}: the predicted rows"
    );
    assert_eq!(
        instances(&got["predict"]["related_tables"]),
        want["predict_related"],
        "{id}: the predicted rows' related tables"
    );
}

#[test]
fn the_recording_names_the_two_base_cases() {
    // The vectors are the reference's, and hold what the exit compares.
    let all = recorded();
    assert_eq!(all["recorded_by"], "t128a_payload_rules_record.py");
    assert_eq!(
        expected("base.binary")["task"]["kind"],
        "binary_classification"
    );
    assert_eq!(
        expected("base.binary")["task"]["target"]["positive_class"],
        "true"
    );
    assert_eq!(
        expected("base.multiclass")["task"]["kind"],
        "multiclass_classification"
    );
    // base.multiclass's static target leaks: the reference took rows out.
    let leaked = expected("base.multiclass")["notices"]
        .as_array()
        .unwrap()
        .iter()
        .any(|n| n.as_str().unwrap().contains("rows sampled as context"));
    assert!(leaked);
}

#[test]
fn base_binary_s_classification_payload_is_the_reference_s() {
    classification("base.binary");
}

#[test]
fn base_multiclass_s_classification_payload_and_leakage_are_the_reference_s() {
    classification("base.multiclass");
}

#[test]
fn more_entities_than_the_endpoint_scores_are_refused_as_the_reference_refuses_them() {
    // The entity count's limit (1,000), on base.regression with 1,001
    // entities: the reference's UNSUPPORTED_REQUEST, which python-core is
    // held to by the same recorded case.
    let want = expected("base.regression.entities-1001");
    assert_eq!(want["outcome"], "refused");
    let (mut case, bases) = the_case("base.regression");
    let materialize = case
        .core
        .as_mut()
        .and_then(|core| core.materialize.as_mut())
        .unwrap();
    materialize.indices = (0..1001)
        .map(|i| json!({"t": "i64", "v": i.to_string()}))
        .collect();
    let observed = host::run(&case, &bases).unwrap();
    assert!(observed.materialized.is_none(), "1,001 entities were taken");
    let Answer::Error { code, fields, .. } = &observed.answer else {
        panic!("not refused: {:?}", observed.answer);
    };
    // The core's own limit, not a rule it does not hold yet (`not_yet`
    // names one), which would refuse with the same code for another reason.
    assert_eq!(fields.get("rule"), None, "refused by a rule not held yet");
    assert_eq!(Some(code.as_str()), want["code"].as_str());
}
