// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The Rust scripted endpoint against the shared vector file
//! (`contract/scripted-endpoint.json`), which the Python endpoint answers
//! too (`contract/tests/test_scripted_endpoint.py`): every vector's status
//! and answer, and a vector for every reason the rule refuses with.

use std::collections::BTreeSet;
use std::path::PathBuf;

use sdm_testkit::endpoint::{REASONS, answer};
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
