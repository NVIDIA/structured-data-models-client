// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What the generator writes for the shapes the pinned contract does not use
//! yet, so a contract that starts using one gets types already tested.
//!
//! `edge/generated.rs` is the generator's output for `edge/contract.json`,
//! built here against the crate's own `fields.rs`, as `generated.rs` is.
//! `rust-wire` checks it is current; `test_generate_wire_rs.py` beside this
//! file tests the generator's refusals.

#[path = "../src/fields.rs"]
mod fields;
#[allow(dead_code, reason = "a test uses only some of the generated items")]
#[path = "edge/generated.rs"]
mod generated;

use generated::{
    Closed, ClosedLevel, Either, Mode, OpenDefault, OpenEmpty, Pet, Scalar, TypedExtra,
};
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::{Value, json};

fn parse<T: DeserializeOwned>(value: &Value) -> Result<T, String> {
    serde_json::from_str(&value.to_string()).map_err(|e| e.to_string())
}

/// `value` parsed as a `T` and serialized back.
fn roundtrip<T: DeserializeOwned + Serialize>(value: &Value) -> Result<Value, String> {
    serde_json::to_value(parse::<T>(value)?).map_err(|e| e.to_string())
}

fn closed() -> Value {
    json!({
        "type": "t",
        "self": true,
        "camelCase": 3,
        "maybe": "m",
        "undeclared": [1, "a"]
    })
}

#[test]
fn a_closed_object_round_trips_every_field_it_names() {
    let mut payload = closed();
    let extra = json!({
        "2nd": "second",
        "perhaps": 4,
        "plain": 5,
        "mode": "kebab-case",
        "level": "high",
        "inner": {"n": 1, "other": [true]},
        "counts": {"a": 1, "b": 2},
        "bag": {"x": {"y": null}},
        "cells": [1.5, null, 2]
    });
    for (key, value) in extra.as_object().expect("an object") {
        payload[key] = value.clone();
    }
    assert_eq!(roundtrip::<Closed>(&payload), Ok(payload.clone()));

    let parsed: Closed = parse(&payload).expect("a Closed");
    assert_eq!(parsed.r#type, "t");
    assert!(parsed.self_);
    assert_eq!(parsed.camel_case, 3);
    assert_eq!(parsed.f_2nd.as_deref(), Some("second"));
    assert_eq!(parsed.mode, Some(Mode::KebabCase));
    assert_eq!(parsed.level, Some(ClosedLevel::High));
    assert_eq!(parsed.undeclared, json!([1, "a"]));
    let cells = parsed.cells.expect("cells");
    assert!(cells[0].as_ref().is_some_and(|n| n.is_f64()));
    assert!(cells[1].is_none());
    assert!(cells[2].as_ref().is_some_and(|n| n.is_u64()));
}

#[test]
fn a_required_nullable_key_must_be_present_and_may_be_null() {
    let mut payload = closed();
    payload["maybe"] = Value::Null;
    assert_eq!(roundtrip::<Closed>(&payload), Ok(payload.clone()));
    assert_eq!(parse::<Closed>(&payload).map(|c| c.maybe), Ok(None));

    payload.as_object_mut().map(|o| o.remove("maybe"));
    let error = parse::<Closed>(&payload).expect_err("an absent required key");
    assert!(error.contains("missing field `maybe`"), "{error}");
}

#[test]
fn an_optional_nullable_key_keeps_absent_and_null_apart() {
    let absent = closed();
    let mut null = closed();
    null["perhaps"] = Value::Null;
    assert_eq!(parse::<Closed>(&absent).map(|c| c.perhaps), Ok(None));
    assert_eq!(parse::<Closed>(&null).map(|c| c.perhaps), Ok(Some(None)));
    assert_eq!(roundtrip::<Closed>(&absent), Ok(absent));
    assert_eq!(roundtrip::<Closed>(&null), Ok(null));
}

type Change = fn(&mut Value);

#[test]
fn a_closed_object_refuses_what_the_contract_does_not_allow() {
    let cases: [(&str, Change); 10] = [
        ("an unknown key", |p| p["other"] = json!(1)),
        ("null for a key that is never null", |p| {
            p["plain"] = Value::Null
        }),
        ("null for a required key that is never null", |p| {
            p["type"] = Value::Null
        }),
        ("an absent required key", |p| {
            p.as_object_mut().map(|o| o.remove("self"));
        }),
        ("an absent required key the properties do not name", |p| {
            p.as_object_mut().map(|o| o.remove("undeclared"));
        }),
        ("a value outside a named enum", |p| {
            p["mode"] = json!("slow")
        }),
        ("a value outside an inline enum", |p| {
            p["level"] = json!("mid")
        }),
        ("a string in a map of integers", |p| {
            p["counts"] = json!({"a": "1"})
        }),
        ("an inline object without its required key", |p| {
            p["inner"] = json!({})
        }),
        ("a string among nullable numbers", |p| {
            p["cells"] = json!([1, "2"])
        }),
    ];
    for (case, change) in cases {
        let mut payload = closed();
        change(&mut payload);
        assert!(parse::<Closed>(&payload).is_err(), "{case} was accepted");
    }
}

#[test]
fn an_enum_spells_each_value_as_the_contract_does() {
    let values = [
        (Mode::Fast, "fast"),
        (Mode::KebabCase, "kebab-case"),
        (Mode::V1x, "1x"),
    ];
    for (mode, text) in values {
        assert_eq!(serde_json::to_value(mode).ok(), Some(json!(text)));
        assert_eq!(parse::<Mode>(&json!(text)), Ok(mode));
    }
    assert!(parse::<Mode>(&json!("Fast")).is_err());
}

#[test]
fn an_object_open_through_an_empty_schema_keeps_unknown_keys() {
    let payload = json!({"a": "x", "b": 1, "c": {"d": [null]}});
    assert_eq!(roundtrip::<OpenEmpty>(&payload), Ok(payload.clone()));
    let parsed: OpenEmpty = parse(&payload).expect("an OpenEmpty");
    assert_eq!(parsed.additional_properties.len(), 2);
}

#[test]
fn the_field_for_unknown_keys_steps_aside_for_a_property_of_its_name() {
    let payload = json!({"additional_properties": "named", "other": 2});
    assert_eq!(roundtrip::<OpenDefault>(&payload), Ok(payload.clone()));
    let parsed: OpenDefault = parse(&payload).expect("an OpenDefault");
    assert_eq!(parsed.additional_properties.as_deref(), Some("named"));
    assert_eq!(parsed.additional_properties_.get("other"), Some(&json!(2)));
}

#[test]
fn typed_additional_properties_take_only_their_type() {
    let payload = json!({"a": "x", "b": 1, "c": -2});
    assert_eq!(roundtrip::<TypedExtra>(&payload), Ok(payload));
    assert!(parse::<TypedExtra>(&json!({"b": "1"})).is_err());
    assert!(parse::<TypedExtra>(&json!({"b": 1.5})).is_err());
}

#[test]
fn a_discriminator_selects_by_its_mapping_and_by_a_members_own_name() {
    let cases = [
        (json!({"kind": "cat", "lives": 9}), "cat"),
        (json!({"kind": "kitten", "lives": 9}), "cat"),
        (json!({"kind": "Dog", "bark": true}), "dog"),
    ];
    for (payload, variant) in cases {
        assert_eq!(roundtrip::<Pet>(&payload), Ok(payload.clone()));
        let parsed: Pet = parse(&payload).expect("a Pet");
        let got = match parsed {
            Pet::Cat(_) => "cat",
            Pet::Dog(_) => "dog",
        };
        assert_eq!(got, variant, "{payload}");
    }
    assert_eq!(Pet::TAGS, ["cat", "kitten", "Dog"]);
}

#[test]
fn a_discriminator_refuses_a_tag_it_cannot_place() {
    let cases = [
        (json!({"kind": "dog"}), "unknown `kind` \"dog\""),
        (json!({"lives": 9}), "missing field `kind`"),
        (json!({"kind": 1}), "`kind` must be a string"),
        (json!(["cat"]), "expected an object"),
        // The tag selects Cat, which is closed and requires `lives`.
        (json!({"kind": "cat"}), "missing field `lives`"),
        (
            json!({"kind": "cat", "lives": 9, "x": 1}),
            "unknown field `x`",
        ),
    ];
    for (payload, message) in cases {
        let error = parse::<Pet>(&payload).expect_err("refused");
        assert!(error.contains(message), "{payload}: {error}");
    }
}

#[test]
fn an_untagged_union_takes_each_member_and_null_when_one_is_nullable() {
    let cases = [
        (json!("s"), Scalar::String("s".to_owned())),
        (json!(7), Scalar::Integer(7)),
        (json!(false), Scalar::Boolean(false)),
        (Value::Null, Scalar::Null),
    ];
    for (payload, expected) in cases {
        assert_eq!(parse::<Scalar>(&payload), Ok(expected));
        assert_eq!(roundtrip::<Scalar>(&payload), Ok(payload));
    }
    assert!(parse::<Scalar>(&json!(1.5)).is_err());
    assert!(parse::<Scalar>(&json!([])).is_err());
}

#[test]
fn an_any_of_union_takes_each_member() {
    let names = json!(["a", "b"]);
    assert!(matches!(parse::<Either>(&names), Ok(Either::Array(_))));
    assert_eq!(roundtrip::<Either>(&names), Ok(names));
    let cat = json!({"kind": "cat", "lives": 1});
    assert!(matches!(parse::<Either>(&cat), Ok(Either::Cat(_))));
    assert!(parse::<Either>(&Value::Null).is_err());
}
