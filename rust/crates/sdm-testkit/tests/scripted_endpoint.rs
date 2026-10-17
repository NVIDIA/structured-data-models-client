// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The Rust scripted endpoint against the shared vector file
//! (`contract/scripted-endpoint.json`), which the Python endpoint answers
//! too (`contract/tests/test_scripted_endpoint.py`): every vector's status
//! and answer, and a vector for every reason the rule refuses with.

use std::collections::BTreeSet;
use std::path::PathBuf;

use sdm_testkit::endpoint::{Endpoint, FAILED, REASONS, answer, answer_call};
use serde_json::Value;

fn vectors() -> Value {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../contract/scripted-endpoint.json");
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

#[test]
fn the_rule_answers_each_vector() {
    let document = vectors();
    assert_eq!(document["format"], "sdm-scripted-endpoint/1");
    let vectors = document["vectors"].as_array().unwrap();
    // Not vacuous: the file holds the vectors T1.11a recorded.
    assert!(vectors.len() >= 20, "{}", vectors.len());
    let mut names = BTreeSet::new();
    for vector in vectors {
        let name = vector["name"].as_str().unwrap();
        assert!(names.insert(name), "{name} twice");
        let (status, answered) = answer(
            vector["method"].as_str().unwrap(),
            vector["path"].as_str().unwrap(),
            vector["body"].as_str().unwrap().as_bytes(),
        );
        assert_eq!(
            u64::from(status),
            vector["status"].as_u64().unwrap(),
            "{name}"
        );
        assert_eq!(answered, vector["answer"], "{name}");
        // The answer's bytes read back as the same JSON, a float as a float.
        let bytes = serde_json::to_vec(&answered).unwrap();
        assert_eq!(
            serde_json::from_slice::<Value>(&bytes).unwrap(),
            vector["answer"],
            "{name}"
        );
    }
}

#[test]
fn every_reason_has_a_vector() {
    let document = vectors();
    let refused: BTreeSet<&str> = document["vectors"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|v| v["status"] == 400)
        .map(|v| v["answer"]["error"].as_str().unwrap())
        .collect();
    assert_eq!(refused, REASONS.into_iter().collect());
}

/// Four different requests, so an answer given to the wrong call shows
/// (the calls of `tests/test_t21a_scripted_calls.py`).
const CALLS: [&str; 4] = [
    "regression",
    "regression, instance_id not the first column",
    "binary classification answers the first class",
    "regression",
];

fn vector(name: &str) -> Value {
    vectors()["vectors"]
        .as_array()
        .unwrap()
        .iter()
        .find(|vector| vector["name"] == name)
        .unwrap()
        .clone()
}

fn asked(endpoint: &mut Endpoint, name: &str) -> (u64, Value) {
    let vector = vector(name);
    let (status, answered) = endpoint.answer(
        vector["method"].as_str().unwrap(),
        vector["path"].as_str().unwrap(),
        vector["body"].as_str().unwrap().as_bytes(),
    );
    (u64::from(status), answered)
}

fn expected(fail_call: Option<u64>) -> Vec<(u64, Value)> {
    CALLS
        .iter()
        .zip(0_u64..)
        .map(|(name, call)| {
            if fail_call == Some(call) {
                (
                    u64::from(FAILED),
                    serde_json::json!({"error": "scripted_failure", "call": call}),
                )
            } else {
                let vector = vector(name);
                (vector["status"].as_u64().unwrap(), vector["answer"].clone())
            }
        })
        .collect()
}

/// T2.1a: calls are numbered from zero, the chosen one (the first and the
/// last included) answered 500 naming its number and every other by the
/// rule; a call past the last, or none, fails nothing.
#[test]
fn the_endpoint_numbers_its_calls_and_fails_the_chosen_one() {
    let distinct: BTreeSet<String> = expected(None)
        .iter()
        .map(|(_, answer)| answer.to_string())
        .collect();
    assert_eq!(distinct.len(), 3);
    for fail_call in [None, Some(0), Some(2), Some(3), Some(4)] {
        let mut endpoint = Endpoint::new(fail_call);
        let answers: Vec<_> = CALLS
            .iter()
            .map(|name| asked(&mut endpoint, name))
            .collect();
        let failed = fail_call.filter(|&call| call < 4);
        assert_eq!(answers, expected(failed), "{fail_call:?}");
        assert_eq!(endpoint.calls(), 4);
    }
}

#[test]
fn the_rule_numbers_the_call_it_is_given() {
    let vector = vector("regression");
    let asked = (
        vector["method"].as_str().unwrap(),
        vector["path"].as_str().unwrap(),
        vector["body"].as_str().unwrap().as_bytes(),
    );
    assert_eq!(
        answer_call(asked.0, asked.1, asked.2, 1, Some(1)),
        (
            FAILED,
            serde_json::json!({"error": "scripted_failure", "call": 1})
        )
    );
    assert_eq!(
        answer_call(asked.0, asked.1, asked.2, 0, Some(1)),
        answer(asked.0, asked.1, asked.2)
    );
    // A failure is the chosen call's whatever it asks, a 404 included.
    assert_eq!(answer_call("GET", "/", b"", 0, Some(0)).0, FAILED);
}
