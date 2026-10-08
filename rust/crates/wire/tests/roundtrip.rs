// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Every recorded request goes through the wire types unchanged: the golden
//! payloads in `tests/golden/wire_payloads.json`, and every payload of every
//! parity transcript in `tests/parity/transcripts/`.
//!
//! Unchanged means parsed-equal, by value and by JSON type, so an integer
//! that comes back a float, or a key that comes back absent or null, fails.

use std::fs;
use std::path::{Path, PathBuf};

use sdm_wire::{Value, parse_prediction_request};

/// What the corpus held when these tests were written; it may only grow.
const MIN_GOLDEN: usize = 9;
const MIN_CORPUS: usize = 57;

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

fn read_json(path: &Path) -> Value {
    let text =
        fs::read_to_string(path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{} is not JSON: {e}", path.display()))
}

fn transcripts(dir: &Path, found: &mut Vec<PathBuf>) {
    let entries =
        fs::read_dir(dir).unwrap_or_else(|e| panic!("cannot list {}: {e}", dir.display()));
    for entry in entries {
        let path = entry.expect("a directory entry").path();
        if path.is_dir() {
            transcripts(&path, found);
        } else if path.file_name().is_some_and(|n| n == "transcript.json") {
            found.push(path);
        }
    }
}

/// Where a payload does not survive a parse and a serialization, if it
/// does not.
fn roundtrip_failure(payload: &Value) -> Option<String> {
    let text = payload.to_string();
    let request = match parse_prediction_request(&text) {
        Ok(request) => request,
        Err(e) => return Some(format!("does not parse: {e}")),
    };
    let back = match serde_json::to_value(&request) {
        Ok(back) => back,
        Err(e) => return Some(format!("does not serialize: {e}")),
    };
    (back != *payload).then(|| format!("comes back as {back}"))
}

fn check_all(payloads: &[(String, Value)]) {
    let failures: Vec<_> = payloads
        .iter()
        .filter_map(|(name, payload)| roundtrip_failure(payload).map(|why| format!("{name} {why}")))
        .collect();
    assert!(
        failures.is_empty(),
        "{} of {} payloads do not round-trip:\n{}",
        failures.len(),
        payloads.len(),
        failures.join("\n")
    );
}

#[test]
fn every_golden_payload_round_trips() {
    let path = repo().join("tests/golden/wire_payloads.json");
    let Value::Object(cases) = read_json(&path) else {
        panic!("{} is not an object", path.display());
    };
    let mut payloads = Vec::new();
    for (case, list) in cases {
        let Value::Array(list) = list else {
            panic!("{case} is not a list of payloads");
        };
        for (i, payload) in list.into_iter().enumerate() {
            payloads.push((format!("golden {case}[{i}]"), payload));
        }
    }
    assert!(
        payloads.len() >= MIN_GOLDEN,
        "only {} golden payloads",
        payloads.len()
    );
    check_all(&payloads);
}

#[test]
fn every_corpus_payload_round_trips() {
    let root = repo().join("tests/parity/transcripts");
    let mut files = Vec::new();
    transcripts(&root, &mut files);
    files.sort();
    let mut payloads = Vec::new();
    for file in files {
        let name = file
            .strip_prefix(&root)
            .unwrap_or(&file)
            .display()
            .to_string();
        let Some(Value::Array(list)) = read_json(&file).get_mut("payloads").map(Value::take) else {
            panic!("{name} has no list of payloads");
        };
        for (i, payload) in list.into_iter().enumerate() {
            payloads.push((format!("corpus {name}[{i}]"), payload));
        }
    }
    assert!(
        payloads.len() >= MIN_CORPUS,
        "only {} corpus payloads",
        payloads.len()
    );
    check_all(&payloads);
}

#[test]
fn a_payload_that_does_not_round_trip_is_caught() {
    let mut payload = read_json(&repo().join("tests/golden/wire_payloads.json"))
        .as_object()
        .and_then(|cases| cases.values().next())
        .and_then(|list| list.get(0))
        .cloned()
        .expect("a golden payload");
    assert_eq!(roundtrip_failure(&payload), None);
    payload["unexpected"] = Value::from(1);
    assert!(roundtrip_failure(&payload).is_some());
}
