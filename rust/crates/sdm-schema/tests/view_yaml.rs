// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The view definitions' YAML reader (T3.5, design part 5, old-behaviour
//! row 69) against `contract/view-yaml-vectors.json`: what PyYAML 6.0.3's
//! `safe_load` made of each text, YAML 1.1's values included (`yes`, `on`
//! and `off` booleans, `017` octal, `1:30` base 60, `1e3` text and
//! `1.0e+3` a float), and from T3.5 fix 1 mapping keys Python takes as one,
//! integers of any size, `!!set`, `!!omap`, `!!pairs`, base64 as Python's
//! decoder reads it, aliases of their own anchor, block scalars at the end
//! of the text, an anchor named twice and a byte order mark. A text PyYAML
//! refused is refused. The texts the old client's messages quote a value by
//! are the core's to write (`sdm-pql/tests/shown.rs`,
//! `sdm-core/tests/view_words.rs`).
//!
//! The few texts `yaml-rust2` scans otherwise than PyYAML are registered
//! differences, each named below and held to what the core does.

use sdm_schema::view::yaml::{self, Document, NodeId, Value};
use serde_json::Value as Json;

/// The vectors, read at build time: the core's crates do no I/O.
const VECTORS: &str = include_str!("../../../../contract/view-yaml-vectors.json");

/// The texts the core reads otherwise than PyYAML, registered (P3J-6):
/// a tab after a plain scalar, which PyYAML refuses and `yaml-rust2`
/// reads; and a tab starting a quoted scalar's next line, and an alias of
/// an anchor on a flow mapping's key, which PyYAML reads and `yaml-rust2`
/// refuses.
const REGISTERED: &[&str] = &[
    "v: a\tb",
    "v: a \tb",
    "v: \"a\n\tb\"",
    "{&n .nan: a, *n: b}",
];

/// A float as the comparison reads it: the value, its sign and NaN told
/// apart, whatever digits wrote it.
fn float_key(x: f64) -> String {
    if x.is_nan() {
        "nan".into()
    } else {
        format!("{x:?}")
    }
}

/// `node` as the vectors write a loaded value, `open` the collections it
/// is inside.
fn typed(document: &Document, node: NodeId, open: &mut Vec<NodeId>) -> Json {
    if let Some(at) = open.iter().position(|n| *n == node) {
        return serde_json::json!({"t": "cycle", "v": open.len() - at});
    }
    let (kind, inner) = match document.value(node) {
        Value::Null => return serde_json::json!({"t": "null"}),
        Value::Bool(flag) => ("bool", Json::Bool(*flag)),
        Value::Int(n) => ("int", Json::String(n.text().to_owned())),
        Value::Float(x) => ("float", Json::String(float_key(*x))),
        Value::Str(text) => ("str", Json::String(text.clone())),
        Value::Bytes(bytes) => ("bytes", Json::String(base64(bytes))),
        Value::Date(date) => ("date", Json::String(date.isoformat())),
        Value::Timestamp(moment) => ("datetime", Json::String(moment.isoformat())),
        Value::Seq(items) => {
            open.push(node);
            let items = items.iter().map(|n| typed(document, *n, open)).collect();
            open.pop();
            ("list", Json::Array(items))
        }
        Value::Map(pairs) => {
            open.push(node);
            let pairs = pairs
                .iter()
                .map(|(k, v)| {
                    Json::Array(vec![typed(document, *k, open), typed(document, *v, open)])
                })
                .collect();
            open.pop();
            ("map", Json::Array(pairs))
        }
        Value::Set(members) => {
            open.push(node);
            let mut members: Vec<Json> =
                members.iter().map(|n| typed(document, *n, open)).collect();
            open.pop();
            members.sort_by_key(|m| m.to_string());
            ("set", Json::Array(members))
        }
        Value::Pairs(pairs) => {
            open.push(node);
            let pairs = pairs
                .iter()
                .map(|(k, v)| {
                    serde_json::json!({"t": "tuple", "v": [typed(document, *k, open), typed(document, *v, open)]})
                })
                .collect();
            open.pop();
            ("list", Json::Array(pairs))
        }
    };
    serde_json::json!({"t": kind, "v": inner})
}

/// The recorded typed value, its floats keyed as [`float_key`] writes, its
/// sets' members in the order [`typed`] sorts them.
fn recorded(value: &Json) -> Json {
    match value["t"].as_str() {
        Some("float") => {
            let x: f64 = value["v"].as_str().unwrap().parse().unwrap();
            serde_json::json!({"t": "float", "v": float_key(x)})
        }
        Some(kind @ ("list" | "tuple" | "set")) => {
            let mut items: Vec<Json> = value["v"]
                .as_array()
                .unwrap()
                .iter()
                .map(recorded)
                .collect();
            if kind == "set" {
                items.sort_by_key(|m| m.to_string());
            }
            serde_json::json!({"t": kind, "v": items})
        }
        Some("map") => serde_json::json!({
            "t": "map",
            "v": value["v"].as_array().unwrap().iter()
                .map(|pair| Json::Array(vec![recorded(&pair[0]), recorded(&pair[1])]))
                .collect::<Vec<_>>(),
        }),
        _ => value.clone(),
    }
}

/// The standard base64 of bytes, as the recorder writes them.
fn base64(bytes: &[u8]) -> String {
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
    assert!(vectors.len() > 350, "{} vectors", vectors.len());
    let mut wrong = Vec::new();
    for vector in vectors {
        let text = vector["yaml"].as_str().unwrap();
        if REGISTERED.contains(&text) {
            continue;
        }
        match (&vector["error"], yaml::load(text)) {
            (Json::String(_), Err(_)) => {}
            (Json::String(class), Ok(loaded)) => wrong.push(format!(
                "{text:?}: PyYAML refused it ({class}), read {}",
                typed(&loaded, loaded.root(), &mut Vec::new())
            )),
            (_, Err(error)) => wrong.push(format!("{text:?}: refused ({error})")),
            (_, Ok(loaded)) => {
                let got = typed(&loaded, loaded.root(), &mut Vec::new());
                let expected = recorded(&vector["value"]);
                if got != expected {
                    wrong.push(format!("{text:?}: {got} where PyYAML read {expected}"));
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

/// What the core does with each registered difference (P3J-6).
#[test]
fn the_registered_scanner_differences_hold() {
    // PyYAML refuses a tab after a plain scalar; `yaml-rust2` keeps it.
    for text in ["v: a\tb", "v: a \tb"] {
        let loaded = yaml::load(text).unwrap();
        let entry = loaded.get(loaded.root(), "v").unwrap();
        assert!(
            loaded.text(entry).is_some_and(|t| t.contains('\t')),
            "{text:?}"
        );
    }
    // PyYAML reads these; `yaml-rust2` refuses them.
    assert!(yaml::load("v: \"a\n\tb\"").is_err());
    assert!(yaml::load("{&n .nan: a, *n: b}").is_err());
}

/// The design's own vectors (part 5), named so a regression says which.
#[test]
fn yaml_1_1_values_hold() {
    let value = |text: &str| {
        let loaded = yaml::load(&format!("v: {text}")).unwrap();
        loaded
            .value(loaded.get(loaded.root(), "v").unwrap())
            .clone()
    };
    assert_eq!(value("yes"), Value::Bool(true));
    assert_eq!(value("on"), Value::Bool(true));
    assert_eq!(value("off"), Value::Bool(false));
    assert_eq!(value("y"), Value::Str("y".into()));
    assert_eq!(value("n"), Value::Str("n".into()));
    assert_eq!(value("~"), Value::Null);
    assert!(matches!(value("017"), Value::Int(n) if n.text() == "15"));
    assert!(matches!(value("1:30"), Value::Int(n) if n.text() == "90"));
    assert_eq!(value("1e3"), Value::Str("1e3".into()));
    assert_eq!(value("1.0e3"), Value::Str("1.0e3".into()));
    assert_eq!(value("1.0e+3"), Value::Float(1000.0));
    assert_eq!(value("'yes'"), Value::Str("yes".into()));
    // A bare `on:` key is the boolean true, as a join's `on` is read.
    let loaded = yaml::load("on: x").unwrap();
    assert!(loaded.get_true(loaded.root()).is_some());
}

/// An alias is its anchor's node, held once: eighteen doublings of an alias
/// are read in no time and no memory, and a list may hold itself (Codex
/// review 1, P1-7).
#[test]
fn an_alias_is_its_anchor_s_node() {
    let mut text = String::from("source: c.s.t\nmaterialization:\n  a0: &a0 [x]\n");
    for i in 1..19 {
        text.push_str(&format!("  a{i}: &a{i} [*a{}, *a{}]\n", i - 1, i - 1));
    }
    let loaded = yaml::load(&text).unwrap();
    let materialization = loaded.get(loaded.root(), "materialization").unwrap();
    let Value::Map(pairs) = loaded.value(materialization) else {
        panic!()
    };
    let Value::Seq(last) = loaded.value(pairs[18].1) else {
        panic!()
    };
    assert_eq!(last[0], last[1]);
    assert_eq!(last[0], pairs[17].1);

    let loaded = yaml::load("v: &a [*a]").unwrap();
    let list = loaded.get(loaded.root(), "v").unwrap();
    assert_eq!(loaded.value(list), &Value::Seq(vec![list]));
}
