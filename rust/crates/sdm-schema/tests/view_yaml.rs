// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The view definitions' YAML reader (T3.5, design part 5, old-behaviour
//! row 69) against `contract/view-yaml-vectors.json`: what PyYAML 6.0.3's
//! `safe_load` made of each text, YAML 1.1's values included (`yes`, `on`
//! and `off` booleans, `017` octal, `1:30` base 60, `1e3` text and
//! `1.0e+3` a float), with the value's Python `repr` and `str`, which the
//! old client's messages quote. A text PyYAML refused is refused.

use sdm_schema::view::yaml::{self, Value};
use serde_json::Value as Json;

/// The vectors, read at build time: the core's crates do no I/O.
const VECTORS: &str = include_str!("../../../../contract/view-yaml-vectors.json");

/// `value` as the vectors write a loaded value.
fn typed(value: &Value) -> Json {
    let (kind, inner) = match value {
        Value::Null => return serde_json::json!({"t": "null"}),
        Value::Bool(flag) => ("bool", Json::Bool(*flag)),
        Value::Int(number) => ("int", Json::String(number.to_string())),
        Value::Float(_) => ("float", Json::String(value.repr())),
        Value::Str(text) => ("str", Json::String(text.clone())),
        Value::Bytes(_) => ("bytes", Json::String(base64(value))),
        Value::Date(date) => ("date", Json::String(date.isoformat())),
        Value::Timestamp(moment) => ("datetime", Json::String(moment.isoformat())),
        Value::Seq(items) => ("list", Json::Array(items.iter().map(typed).collect())),
        Value::Map(pairs) => (
            "map",
            Json::Array(
                pairs
                    .iter()
                    .map(|(key, entry)| Json::Array(vec![typed(key), typed(entry)]))
                    .collect(),
            ),
        ),
    };
    serde_json::json!({"t": kind, "v": inner})
}

/// The standard base64 of a bytes value, as the recorder writes it.
fn base64(value: &Value) -> String {
    let Value::Bytes(bytes) = value else {
        unreachable!()
    };
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::new();
    for chunk in bytes.chunks(3) {
        let n = chunk
            .iter()
            .enumerate()
            .fold(0u32, |n, (i, &b)| n | (u32::from(b) << (16 - 8 * i)));
        for i in 0..4 {
            if i <= chunk.len() {
                out.push(char::from(ALPHABET[((n >> (18 - 6 * i)) & 63) as usize]));
            } else {
                out.push('=');
            }
        }
    }
    out
}

#[test]
fn each_text_reads_as_pyyaml_read_it() {
    let document: Json = serde_json::from_str(VECTORS).unwrap();
    assert_eq!(document["format"], "sdm-view-yaml-vectors/1");
    let vectors = document["vectors"].as_array().unwrap();
    assert!(vectors.len() > 250, "{} vectors", vectors.len());
    let mut wrong = Vec::new();
    for vector in vectors {
        let text = vector["yaml"].as_str().unwrap();
        let read = yaml::load(text);
        match (&vector["error"], read) {
            (Json::String(_), Err(_)) => {}
            (Json::String(class), Ok(value)) => {
                wrong.push(format!(
                    "{text:?}: PyYAML refused it ({class}), read {value:?}"
                ));
            }
            (_, Err(error)) => wrong.push(format!("{text:?}: refused ({error})")),
            (_, Ok(value)) => {
                if typed(&value) != vector["value"] {
                    wrong.push(format!(
                        "{text:?}: {} where PyYAML read {}",
                        typed(&value),
                        vector["value"]
                    ));
                }
                if value.repr() != vector["repr"].as_str().unwrap() {
                    wrong.push(format!(
                        "{text:?}: repr {} where Python's is {}",
                        value.repr(),
                        vector["repr"]
                    ));
                }
                if let Some(text_form) = vector["str"].as_str()
                    && value.text_form() != text_form
                {
                    wrong.push(format!(
                        "{text:?}: str {} where Python's is {text_form}",
                        value.text_form()
                    ));
                }
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "{} of {} vectors differ:\n{}",
        wrong.len(),
        vectors.len(),
        wrong.join("\n")
    );
}

/// The design's own vectors (part 5), named so a regression says which.
#[test]
fn yaml_1_1_values_hold() {
    let value = |text: &str| match yaml::load(&format!("v: {text}")).unwrap() {
        Value::Map(mut pairs) => pairs.remove(0).1,
        other => panic!("{other:?}"),
    };
    assert_eq!(value("yes"), Value::Bool(true));
    assert_eq!(value("on"), Value::Bool(true));
    assert_eq!(value("off"), Value::Bool(false));
    assert_eq!(value("y"), Value::Str("y".into()));
    assert_eq!(value("n"), Value::Str("n".into()));
    assert_eq!(value("~"), Value::Null);
    assert_eq!(value("017"), Value::Int(15));
    assert_eq!(value("1:30"), Value::Int(90));
    assert_eq!(value("1e3"), Value::Str("1e3".into()));
    assert_eq!(value("1.0e3"), Value::Str("1.0e3".into()));
    assert_eq!(value("1.0e+3"), Value::Float(1000.0));
    assert_eq!(value("'yes'"), Value::Str("yes".into()));
    // A bare `on:` key is the boolean true, as a join's `on` is read.
    assert_eq!(
        yaml::load("on: x").unwrap(),
        Value::Map(vec![(Value::Bool(true), Value::Str("x".into()))])
    );
}
