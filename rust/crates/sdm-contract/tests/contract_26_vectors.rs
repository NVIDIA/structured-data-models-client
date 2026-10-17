// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The Rust loader over contract v26's vectors (T2.1b; design choices 1
//! and 12): `contract-26-vectors.json`, which the Python loader validates
//! against the committed schema (`tests/test_t21b_contract_v26.py`), read
//! here into the type each vector names: an accepted vector deserializes
//! and serializes back to the same JSON, and a refused one does not
//! deserialize.

#![allow(
    clippy::disallowed_methods,
    reason = "test code: reads the committed vectors"
)]

use sdm_contract::data::{DataRequest, Relation};
use sdm_contract::ops::{DescribedTable, HeldTable, NamedTable, SourceSpec};
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::Value;

fn vectors() -> Value {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/contract-26-vectors.json"
    );
    let document: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    assert_eq!(document["format"], "sdm-contract-vectors/1");
    assert_eq!(document["contract"], sdm_contract::CONTRACT_VERSION);
    document
}

/// `value` read as `T`, and written back.
fn round_trip<T: DeserializeOwned + Serialize>(value: &Value) -> Result<Value, String> {
    let typed: T = serde_json::from_value(value.clone()).map_err(|e| e.to_string())?;
    serde_json::to_value(&typed).map_err(|e| e.to_string())
}

/// `value` read as the definition `name` names, and written back.
fn load(name: &str, value: &Value) -> Result<Value, String> {
    match name {
        "DataRequest" => round_trip::<DataRequest>(value),
        "Relation" => round_trip::<Relation>(value),
        "NamedTable" => round_trip::<NamedTable>(value),
        "SourceSpec" => round_trip::<SourceSpec>(value),
        "DescribedTable" => round_trip::<DescribedTable>(value),
        "HeldTable" => round_trip::<HeldTable>(value),
        other => panic!("a vector names a definition this loader does not read: {other}"),
    }
}

#[test]
fn each_accepted_vector_reads_and_writes_back_the_same() {
    let document = vectors();
    let accepted = document["accept"].as_array().unwrap();
    assert!(!accepted.is_empty());
    for vector in accepted {
        let (name, value) = (vector["definition"].as_str().unwrap(), &vector["value"]);
        let written = load(name, value).unwrap_or_else(|e| panic!("{}: {e}", vector["name"]));
        assert_eq!(&written, value, "{}", vector["name"]);
    }
}

#[test]
fn each_refused_vector_is_refused() {
    let document = vectors();
    let refused = document["refuse"].as_array().unwrap();
    assert!(!refused.is_empty());
    for vector in refused {
        let (name, value) = (vector["definition"].as_str().unwrap(), &vector["value"]);
        assert!(load(name, value).is_err(), "{} was read", vector["name"]);
    }
}
