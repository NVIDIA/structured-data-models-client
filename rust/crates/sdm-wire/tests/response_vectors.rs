// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The prediction responses of `contract/response-vectors.json`, which the
//! client's bindings read alike (`contract/tests/test_response_vectors.py`):
//! each body read, or refused naming the field the vector names.

#![allow(
    clippy::disallowed_methods,
    reason = "test code: reads the vector file"
)]

use std::path::PathBuf;

use sdm_wire::json;
use sdm_wire::response::PredictionResponse;
use serde_json::Value;

#[test]
fn the_reader_reads_each_vector_as_the_bindings_do() {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../contract/response-vectors.json");
    let document: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    assert_eq!(document["format"], "sdm-response-vectors/1");
    let vectors = document["vectors"].as_array().unwrap();
    assert!(vectors.len() >= 20, "{}", vectors.len());
    for vector in vectors {
        let name = vector["name"].as_str().unwrap();
        let read = json::read(vector["body"].as_str().unwrap().as_bytes()).unwrap();
        match (PredictionResponse::read(&read), vector["refused"].as_str()) {
            (Ok(_), None) => {}
            (Err(refused), Some(field)) => assert_eq!(refused.field, field, "{name}"),
            (outcome, wanted) => panic!("{name}: {outcome:?}, where {wanted:?}"),
        }
    }
}
