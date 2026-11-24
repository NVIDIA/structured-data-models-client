// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The shared host-provider vectors (`contract/host-provider-vectors.json`),
//! which the Java and Python hosts read into their public request types,
//! held to the contract: each request's wire is a `DataRequest` the core
//! writes back unchanged, each expected schema an `ExpectedSchema`, and
//! each declared capabilities' wire a `ProviderCapabilities`.
#![allow(
    clippy::disallowed_methods,
    reason = "a test reads its vectors from the contract directory"
)]

use std::path::Path;

use sdm_contract::data::{DataRequest, ExpectedSchema, FailureFacts, ProviderCapabilities};
use serde_json::Value;

fn vectors() -> Value {
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../contract/host-provider-vectors.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn cases<'a>(vectors: &'a Value, part: &str) -> &'a Vec<Value> {
    vectors[part].as_array().unwrap()
}

#[test]
fn each_request_is_one_the_core_writes_as_the_vectors_do() {
    let vectors = vectors();
    for case in cases(&vectors, "requests") {
        let wire = &case["wire"];
        let read: DataRequest = serde_json::from_value(wire.clone())
            .unwrap_or_else(|e| panic!("{}: {e}", case["name"]));
        assert_eq!(
            &serde_json::to_value(&read).unwrap(),
            wire,
            "{}",
            case["name"]
        );
    }
}

#[test]
fn each_key_set_a_request_names_is_in_the_vectors() {
    let vectors = vectors();
    for case in cases(&vectors, "requests") {
        let read: DataRequest = serde_json::from_value(case["wire"].clone()).unwrap();
        for id in read.key_sets() {
            assert!(
                vectors["key_sets"].get(id.0.to_string()).is_some(),
                "{}: key set {}",
                case["name"],
                id.0
            );
        }
    }
}

#[test]
fn each_expected_schema_and_capabilities_is_the_contract_s() {
    let vectors = vectors();
    for case in cases(&vectors, "expected") {
        let read: ExpectedSchema = serde_json::from_value(case["wire"].clone())
            .unwrap_or_else(|e| panic!("{}: {e}", case["name"]));
        assert_eq!(
            &serde_json::to_value(&read).unwrap(),
            &case["wire"],
            "{}",
            case["name"]
        );
    }
    for case in cases(&vectors, "capabilities") {
        let _: ProviderCapabilities = serde_json::from_value(case["wire"].clone())
            .unwrap_or_else(|e| panic!("{}: {e}", case["name"]));
    }
}

/// Each round trip the hosts' tests read into their public types and write
/// back is the contract's own: a request, the capabilities and a failure's
/// facts the core writes exactly so.
#[test]
fn each_round_trip_is_written_as_the_core_writes_it() {
    let vectors = vectors();
    let trips = &vectors["round_trips"];
    for case in cases(trips, "requests") {
        let wire = &case["wire"];
        let read: DataRequest = serde_json::from_value(wire.clone())
            .unwrap_or_else(|e| panic!("{}: {e}", case["name"]));
        assert_eq!(
            &serde_json::to_value(&read).unwrap(),
            wire,
            "{}",
            case["name"]
        );
        for id in read.key_sets() {
            assert!(vectors["key_sets"].get(id.0.to_string()).is_some());
        }
    }
    // The hosts write a set of names sorted, which the core reads in any
    // order; every field is written as the core writes it.
    let sorted = |mut wire: Value| {
        for set in ["optional", "filters"] {
            if let Some(names) = wire.get_mut(set).and_then(Value::as_array_mut) {
                names.sort_by_key(|name| name.as_str().unwrap_or_default().to_owned());
            }
        }
        wire
    };
    for case in cases(trips, "capabilities") {
        let wire = &case["wire"];
        let read: ProviderCapabilities = serde_json::from_value(wire.clone())
            .unwrap_or_else(|e| panic!("{}: {e}", case["name"]));
        assert_eq!(
            sorted(serde_json::to_value(&read).unwrap()),
            sorted(wire.clone()),
            "{}",
            case["name"]
        );
    }
    for case in cases(trips, "failures") {
        let facts = &case["wire"]["facts"];
        if facts.is_null() {
            continue;
        }
        let read: FailureFacts = serde_json::from_value(facts.clone())
            .unwrap_or_else(|e| panic!("{}: {e}", case["name"]));
        assert_eq!(
            &serde_json::to_value(&read).unwrap(),
            facts,
            "{}",
            case["name"]
        );
    }
}
