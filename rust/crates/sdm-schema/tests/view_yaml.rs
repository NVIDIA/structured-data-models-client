// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The view definitions' YAML reader (T3.5, design part 5, old-behaviour
//! row 69) against `contract/view-yaml-vectors.json`: what PyYAML 6.0.3's
//! `safe_load` made of each text, YAML 1.1's values included (`yes`, `on`
//! and `off` booleans, `017` octal, `1:30` base 60, `1e3` text and
//! `1.0e+3` a float), and from T3.5 fix 1 mapping keys Python takes as one,
//! integers of any size, aliases of their own anchor, block scalars at the
//! end of the text, an anchor named twice and a byte order mark; from fix 2
//! block scalars whose last line looks like a comment or ends the
//! document, and deep nesting. A text PyYAML refused is refused. The texts
//! the old client's messages quote a value by are the core's to write
//! (`sdm-pql/tests/shown.rs`, `sdm-core/tests/view_words.rs`).
//!
//! The YAML outside what view definitions use is refused by name
//! (register P3J-9): an explicit tag, a merge, a key that is a collection,
//! nesting past [`MAX_DEPTH`], an integer Python would not write. The few
//! texts `yaml-rust2` scans otherwise than PyYAML are registered
//! differences (P3J-6), each named below and held to what the core does.

use sdm_schema::view::yaml::{self, Document, MAX_DEPTH, NodeId, Outside, Rules, Value};
use serde_json::Value as Json;

/// Python's limit on an integer's digits, which the core hands in from
/// `sdm-pql` (`sdm_core::pql::yaml_rules`).
const RULES: Rules = Rules { most_digits: 4300 };

fn load(text: &str) -> Result<Document, yaml::YamlError> {
    yaml::load(text, &RULES)
}

/// The vectors, read at build time: the core's crates do no I/O.
const VECTORS: &str = include_str!("../../../../contract/view-yaml-vectors.json");

/// The texts the core reads otherwise than PyYAML, registered (P3J-6):
/// a tab after a plain scalar or alone on the line after a block scalar,
/// which PyYAML refuses and `yaml-rust2` reads; and a tab starting a quoted scalar's next line, and an alias of
/// an anchor on a flow mapping's key, which PyYAML reads and `yaml-rust2`
/// refuses.
const REGISTERED: &[&str] = &[
    "v: a\tb",
    "v: a \tb",
    "v: |\n  a\n\t",
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
    };
    serde_json::json!({"t": kind, "v": inner})
}

/// The recorded typed value, its floats keyed as [`float_key`] writes.
fn recorded(value: &Json) -> Json {
    match value["t"].as_str() {
        Some("float") => {
            let x: f64 = value["v"].as_str().unwrap().parse().unwrap();
            serde_json::json!({"t": "float", "v": float_key(x)})
        }
        Some("list") => serde_json::json!({
            "t": "list",
            "v": value["v"].as_array().unwrap().iter().map(recorded).collect::<Vec<_>>(),
        }),
        Some("map") => serde_json::json!({
            "t": "map",
            "v": value["v"].as_array().unwrap().iter()
                .map(|pair| Json::Array(vec![recorded(&pair[0]), recorded(&pair[1])]))
                .collect::<Vec<_>>(),
        }),
        _ => value.clone(),
    }
}

/// How many collections deep a value is, a collection inside itself
/// counted once, as the recorder counts.
fn depth(document: &Document, node: NodeId, open: &mut Vec<NodeId>) -> usize {
    if open.contains(&node) {
        return 0;
    }
    let members: Vec<NodeId> = match document.value(node) {
        Value::Seq(items) => items.clone(),
        Value::Map(pairs) => pairs.iter().flat_map(|(k, v)| [*k, *v]).collect(),
        _ => return 0,
    };
    open.push(node);
    let deepest = members
        .iter()
        .map(|&m| depth(document, m, open))
        .max()
        .unwrap_or(0);
    open.pop();
    deepest + 1
}

/// Whether `vector` is outside the YAML view definitions use, as `what`
/// says: a tag or a merge written, a collection key PyYAML refused too,
/// nesting past the bound, an integer Python would not write.
fn outside_as_recorded(vector: &Json, what: &Outside) -> bool {
    let text = vector["yaml"].as_str().unwrap();
    match what {
        Outside::Tag(_) => text.contains('!'),
        Outside::Merge => text.contains("<<"),
        Outside::CollectionKey => vector["error"].is_string(),
        Outside::Nesting => vector["depth"]
            .as_u64()
            .is_some_and(|d| d > MAX_DEPTH as u64),
        Outside::Expansion => false,
        Outside::LongInteger => vector["repr_error"] == true,
        Outside::LineBreak => text.contains(['\u{85}', '\u{2028}', '\u{2029}']),
    }
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
        match (&vector["error"], load(text)) {
            (_, Err(error)) if error.outside.is_some() => {
                let what = error.outside.as_ref().unwrap();
                if !outside_as_recorded(vector, what) {
                    wrong.push(format!("{text:?}: refused as outside ({what})"));
                }
            }
            (Json::String(_), Err(_)) => {}
            (Json::String(class), Ok(loaded)) => wrong.push(format!(
                "{text:?}: PyYAML refused it ({class}), read {}",
                typed(&loaded, loaded.root(), &mut Vec::new())
            )),
            (_, Err(error)) => wrong.push(format!("{text:?}: refused ({error})")),
            (_, Ok(loaded)) if vector.get("depth").is_some() => {
                let got = depth(&loaded, loaded.root(), &mut Vec::new());
                if Some(got as u64) != vector["depth"].as_u64() {
                    wrong.push(format!("{text:?}: {got} deep where {}", vector["depth"]));
                }
            }
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
        let loaded = load(text).unwrap();
        let entry = loaded.get(loaded.root(), "v").unwrap();
        assert!(
            loaded.text(entry).is_some_and(|t| t.contains('\t')),
            "{text:?}"
        );
    }
    // PyYAML refuses a tab alone after a block scalar; `yaml-rust2` ends
    // the scalar there.
    let loaded = load("v: |\n  a\n\t").unwrap();
    assert_eq!(
        loaded.text(loaded.get(loaded.root(), "v").unwrap()),
        Some("a\n")
    );
    // PyYAML reads these; `yaml-rust2` refuses them.
    assert!(load("v: \"a\n\tb\"").is_err());
    assert!(load("{&n .nan: a, *n: b}").is_err());
}

/// The design's own vectors (part 5), named so a regression says which.
#[test]
fn yaml_1_1_values_hold() {
    let value = |text: &str| {
        let loaded = load(&format!("v: {text}")).unwrap();
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
    let loaded = load("on: x").unwrap();
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
    let loaded = load(&text).unwrap();
    let materialization = loaded.get(loaded.root(), "materialization").unwrap();
    let Value::Map(pairs) = loaded.value(materialization) else {
        panic!()
    };
    let Value::Seq(last) = loaded.value(pairs[18].1) else {
        panic!()
    };
    assert_eq!(last[0], last[1]);
    assert_eq!(last[0], pairs[17].1);

    let loaded = load("v: &a [*a]").unwrap();
    let list = loaded.get(loaded.root(), "v").unwrap();
    assert_eq!(loaded.value(list), &Value::Seq(vec![list]));
}

/// A block scalar ending the text keeps the line break PyYAML keeps,
/// judged from where the scalar ends, not from how its last line looks
/// (Codex review 2, P0-1).
#[test]
fn a_block_scalar_at_the_end_keeps_python_s_breaks() {
    let value = |text: &str| {
        let loaded = load(text).unwrap();
        loaded
            .text(loaded.get(loaded.root(), "v").unwrap())
            .unwrap()
            .to_owned()
    };
    assert_eq!(value("v: |\n  #hello"), "#hello");
    assert_eq!(value("v: >\n  #hello"), "#hello");
    assert_eq!(value("v: |\n  hi\n..."), "hi\n");
    assert_eq!(value("v: |\n  a\n# c"), "a\n");
    assert_eq!(value("v: |\n  a\n  "), "a\n");
    assert_eq!(value("v: |+\n  a\n  "), "a\n");
    assert_eq!(value("v: |+\n  a"), "a");
    assert_eq!(value("v: |-\n  a\n\n"), "a");
    assert_eq!(value("v: |2\n   a"), " a");
}

/// The YAML outside what view definitions use is refused by name, each
/// for what it is (register P3J-9), a mapping keyed by itself before it is
/// made (Codex review 2, P1-2).
#[test]
fn yaml_outside_the_subset_is_refused_by_name() {
    let refused = |text: &str| load(text).unwrap_err().outside;
    assert_eq!(
        refused("v: !!binary YQ=="),
        Some(Outside::Tag("!!binary".into()))
    );
    assert_eq!(
        refused("v: !!set {a: null}"),
        Some(Outside::Tag("!!set".into()))
    );
    assert_eq!(
        refused("v: !!omap [{a: 1}]"),
        Some(Outside::Tag("!!omap".into()))
    );
    assert_eq!(refused("v: !!str x"), Some(Outside::Tag("!!str".into())));
    assert_eq!(refused("v: ! x"), Some(Outside::Tag("!".into())));
    assert_eq!(refused("b: &b {k: 1}\nv:\n  <<: *b"), Some(Outside::Merge));
    assert_eq!(refused("v: &a {? *a : x}"), Some(Outside::CollectionKey));
    assert_eq!(refused("{[1]: x}"), Some(Outside::CollectionKey));
    let deep = format!("v: {}{}", "[".repeat(MAX_DEPTH), "]".repeat(MAX_DEPTH));
    assert_eq!(refused(&deep), Some(Outside::Nesting));
    let shallow = format!(
        "v: {}{}",
        "[".repeat(MAX_DEPTH - 1),
        "]".repeat(MAX_DEPTH - 1)
    );
    assert!(load(&shallow).is_ok());
    let mut chain = String::from("a0: &a0 [x]\n");
    for i in 1..=MAX_DEPTH {
        chain.push_str(&format!("a{i}: &a{i} [*a{}]\n", i - 1));
    }
    assert_eq!(refused(&chain), Some(Outside::Nesting));
    let mut bomb = String::from("a0: &a0 [x, x]\n");
    for i in 1..30 {
        bomb.push_str(&format!("a{i}: &a{i} [*a{}, *a{}]\n", i - 1, i - 1));
    }
    assert_eq!(refused(&bomb), Some(Outside::Expansion));
    assert_eq!(
        refused(&format!("v: 0x{}", "f".repeat(4000))),
        Some(Outside::LongInteger)
    );
    // Python's own limit on a decimal integer's digits is a refusal, not
    // a subset's.
    let decimal = load(&format!("v: {}", "9".repeat(4301))).unwrap_err();
    assert_eq!(decimal.outside, None);
    assert!(load(&format!("v: {}", "9".repeat(4300))).is_ok());
}

/// A value a later key replaces is still made, so it refuses the text as
/// PyYAML's constructor did (Codex review 2, P2).
#[test]
fn a_replaced_value_is_still_read() {
    assert!(load("v: {a: 2024-13-01, a: 2}").is_err());
    let loaded = load("v: {a: 1, a: 2}").unwrap();
    let v = loaded.get(loaded.root(), "v").unwrap();
    assert!(matches!(loaded.value(loaded.get(v, "a").unwrap()), Value::Int(n) if n.text() == "2"));
}
