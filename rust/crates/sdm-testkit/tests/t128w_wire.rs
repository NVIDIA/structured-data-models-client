// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! T1.28w's exits, held from outside the implementation (the package's
//! independent test author), at `rust-host`'s public route:
//!
//! - Exit 1, the lock refuses a payload whose shape changed: the payload
//!   `rust-host` materializes for each case that reaches materialize holds
//!   to the contract's locked `$defs/StatePayload`, and every one-step
//!   change of its shape (a key the format requires dropped, a key it does
//!   not name added, a value retyped, a dtype or row format the wire does
//!   not name) is refused, at every path outside the open data (a row's
//!   cells). A lock that took any of them would not be the typed payload's.
//! - Exit 2, rust-host follows the core's word on a held sampler state: a
//!   materializer handed a state the core refuses (another version, a memo
//!   that does not read, a memo no run left) answers that call with the
//!   refusal, and its next call starts afresh, exactly as a materializer
//!   that never held one (the same fetches and answer); the one after it is
//!   handed the state that fresh run left, as a materializer's second call.
//!   A host that ignored `drop_held` would hand the refused state back and
//!   be refused again.

use std::path::PathBuf;

use sdm_contract::ops::SAMPLER_STATE_VERSION;
use sdm_contract::schema_check::check;
use sdm_testkit::case::Case;
use sdm_testkit::host::{self, Answer, Observed};
use sdm_testkit::{bases, case};
use serde_json::{Value, json};

fn contract() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../contract")
}

fn loaded() -> (Vec<case::Loaded>, bases::Bases) {
    (
        case::load(&[contract().join("cases")]).unwrap(),
        bases::load(&contract().join("bases.yaml")).unwrap(),
    )
}

fn regression(loaded: &[case::Loaded]) -> Case {
    loaded
        .iter()
        .find(|l| l.case.id == "base.regression")
        .expect("base.regression is a case")
        .case
        .clone()
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

/// The keys the wire's payload writes only when it has them: each may be
/// left out, so dropping one is not a change of shape.
const OPTIONAL: [&str; 7] = [
    "anchor_time_column",
    "stype",
    "nullable",
    "source_table",
    "source_query",
    "prediction_statistic",
    "class_shuffle",
];

/// Whether `path` is inside the open data: a row's cells, any JSON value.
fn in_cells(path: &[String]) -> bool {
    path.windows(2)
        .any(|w| w[0] == "rows" && w[1].starts_with('['))
}

/// Every one-step change of `value`'s shape, each with what it changed.
fn mutations(value: &Value, path: &mut Vec<String>, out: &mut Vec<(String, Value)>) {
    if in_cells(path) {
        return;
    }
    let here = path.join(".");
    // Retyped: a value of another JSON type in its place.
    let retyped = match value {
        Value::Object(_) => json!([]),
        Value::Array(_) => json!({}),
        Value::String(_) => json!(7),
        Value::Bool(_) => json!("true"),
        Value::Number(_) => json!("1"),
        Value::Null => json!({}),
    };
    out.push((format!("{here} retyped"), retyped));
    match value {
        Value::Object(members) => {
            // A key it does not name.
            let mut added = members.clone();
            added.insert("t128w_unnamed".into(), json!(1));
            out.push((format!("{here} gains a key"), Value::Object(added)));
            for (key, item) in members {
                if !OPTIONAL.contains(&key.as_str()) && !open_map(path) {
                    let mut dropped = members.clone();
                    dropped.remove(key);
                    out.push((format!("{here} loses {key}"), Value::Object(dropped)));
                }
                path.push(key.clone());
                let mut inner = Vec::new();
                mutations(item, path, &mut inner);
                path.pop();
                for (what, changed) in inner {
                    let mut whole = members.clone();
                    whole.insert(key.clone(), changed);
                    out.push((what, Value::Object(whole)));
                }
            }
        }
        Value::Array(items) => {
            for (n, item) in items.iter().enumerate() {
                path.push(format!("[{n}]"));
                let mut inner = Vec::new();
                mutations(item, path, &mut inner);
                path.pop();
                for (what, changed) in inner {
                    let mut whole = items.clone();
                    whole[n] = changed;
                    out.push((what, Value::Array(whole)));
                }
            }
        }
        _ => {}
    }
}

/// Whether the object at `path` is one of the open maps of names (the
/// related tables, a table's columns), whose entries are the source's: an
/// entry may be left out, so dropping one is no change of shape.
fn open_map(path: &[String]) -> bool {
    matches!(
        path.last().map(String::as_str),
        Some("related_tables" | "columns")
    )
}

/// The payload of the first case of each family (`base.`, `key.`, ...)
/// that materializes: every shape the cases give (keys of both kinds,
/// keyless tables, links, anchors or none), at a cost the test can carry.
fn payloads() -> Vec<(String, Value)> {
    let (loaded, bases) = loaded();
    let mut found: Vec<(String, Value)> = Vec::new();
    for one in &loaded {
        let family = one.case.id.split('.').next().unwrap_or_default();
        if found
            .iter()
            .any(|(id, _)| id.split('.').next() == Some(family))
            && one.case.id != "base.regression"
        {
            continue;
        }
        let observed = host::run(&one.case, &bases).unwrap();
        if let Some(state) = observed.materialized {
            found.push((one.case.id.clone(), numbers(state["payload"].clone())));
        }
    }
    found
}

#[test]
fn every_one_step_change_of_the_payload_s_shape_is_refused_by_the_lock() {
    // Exit 1: the producer's own output holds, and every change of its
    // shape (outside a row's cells) is refused by the locked definition.
    let payloads = payloads();
    assert!(
        payloads.iter().any(|(id, _)| id == "base.regression"),
        "base.regression materializes"
    );
    let mut refused = 0;
    for (id, payload) in &payloads {
        assert_eq!(check("StatePayload", payload), Ok(()), "{id} as produced");
        let mut changed = Vec::new();
        mutations(payload, &mut Vec::new(), &mut changed);
        assert!(changed.len() > 50, "{id}: {} changes", changed.len());
        for (what, value) in changed {
            assert!(
                check("StatePayload", &value).is_err(),
                "{id}: {what} was taken by the lock"
            );
            refused += 1;
        }
        // Dropping a key the payload writes only when it has one is no
        // change of shape: the lock takes it.
        for key in OPTIONAL {
            let mut without = payload.clone();
            strip(&mut without, key);
            assert_eq!(
                check("StatePayload", &without),
                Ok(()),
                "{id} without {key}"
            );
        }
    }
    assert!(refused > 1000, "{refused} changes refused");
}

fn strip(value: &mut Value, key: &str) {
    match value {
        Value::Object(members) => {
            members.remove(key);
            members.values_mut().for_each(|v| strip(v, key));
        }
        Value::Array(items) => items.iter_mut().for_each(|v| strip(v, key)),
        _ => {}
    }
}

#[test]
fn a_dtype_or_row_format_the_wire_does_not_name_is_refused() {
    // Exit 1, the dtype serialization: the seven wire dtypes the reference's
    // engine writes are taken in a target and a column; any other name,
    // including the Arrow spellings of the same types, is refused; so is a
    // row format other than `arrays`.
    let (_, payload) = payloads()
        .into_iter()
        .find(|(id, _)| id == "base.regression")
        .unwrap();
    for dtype in [
        "int64",
        "float32",
        "float64",
        "string",
        "stringlist",
        "bool",
        "timestamp[us]",
    ] {
        let mut target = payload.clone();
        target["task"]["target"]["dtype"] = json!(dtype);
        assert_eq!(check("StatePayload", &target), Ok(()), "{dtype}");
    }
    for dtype in [
        "int32",
        "double",
        "utf8",
        "boolean",
        "timestamp[ns]",
        "timestamp[us, tz=UTC]",
        "Int64",
        "",
    ] {
        let mut target = payload.clone();
        target["task"]["target"]["dtype"] = json!(dtype);
        assert!(check("StatePayload", &target).is_err(), "target {dtype}");
        let mut column = payload.clone();
        let columns = column["schema"]["instance_table"]["columns"]
            .as_object_mut()
            .unwrap();
        let first = columns.keys().next().unwrap().clone();
        columns[&first]["dtype"] = json!(dtype);
        assert!(check("StatePayload", &column).is_err(), "column {dtype}");
    }
    for format in ["records", "split", "ARRAYS"] {
        let mut rows = payload.clone();
        rows["context"]["instance_table"]["format"] = json!(format);
        assert!(check("StatePayload", &rows).is_err(), "format {format}");
    }
}

fn calls(
    case: &Case,
    bases: &bases::Bases,
    n: usize,
    held: Option<host::HeldSampler>,
) -> Vec<Observed> {
    host::materialize_repeatedly_holding(case, bases, n, held).unwrap()
}

fn is_refusal(observed: &Observed) -> bool {
    matches!(observed.answer, Answer::Error { .. })
}

#[test]
fn a_refused_held_state_is_dropped_and_the_next_call_starts_afresh() {
    // Exit 2 on rust-host: three refusals of a handed state, each dropped
    // on the core's word.
    let (loaded, bases) = loaded();
    let case = regression(&loaded);
    let fresh = calls(&case, &bases, 2, None);
    assert!(matches!(fresh[0].answer, Answer::Materialized(_)));
    // The control: a held state is used, so the second call reads less
    // than the first. Without it a dropped state could not be told from a
    // kept one.
    assert!(
        fresh[1].fetches.len() < fresh[0].fetches.len() || fresh[1].driver != fresh[0].driver,
        "a held state changes nothing the host can see"
    );
    let unsound = serde_json::to_vec(&json!({
        "source": "t128w",
        "indexes_checked": true,
        "bounds": [
            {"table": "orders", "column": "created_at", "min": 2, "max": 1}
        ]
    }))
    .unwrap();
    for (what, held) in [
        (
            "another version",
            (SAMPLER_STATE_VERSION + 1, b"{}".to_vec()),
        ),
        ("version 0", (0, b"{}".to_vec())),
        (
            "a memo that is not JSON",
            (SAMPLER_STATE_VERSION, b"\x00not json".to_vec()),
        ),
        (
            "a memo of another shape",
            (
                SAMPLER_STATE_VERSION,
                br#"{"source":"x","bounds":[],"extra":1}"#.to_vec(),
            ),
        ),
        (
            "a memo no run left",
            (SAMPLER_STATE_VERSION, unsound.clone()),
        ),
    ] {
        let seen = calls(&case, &bases, 3, Some(held));
        assert!(
            is_refusal(&seen[0]),
            "{what}: the first call was not refused: {:?}",
            seen[0].answer
        );
        assert_eq!(
            seen[0].fetches,
            Vec::<Value>::new(),
            "{what}: the refused call fetched"
        );
        for (n, (got, want)) in seen[1..].iter().zip(&fresh).enumerate() {
            assert_eq!(
                got.answer,
                want.answer,
                "{what}: call {} after the refusal",
                n + 1
            );
            assert_eq!(
                got.fetches,
                want.fetches,
                "{what}: call {} after the refusal",
                n + 1
            );
        }
    }
}
