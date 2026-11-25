// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! YAML view parsing: types, aliases, bounded expansion and scalar semantics.

use sdm_schema::view::yaml::{self, Document, MAX_DEPTH, Outside, Value};

mod rules;

use rules::RULES;

fn load(text: &str) -> Result<Document, yaml::YamlError> {
    yaml::load(text, &RULES)
}

#[test]
fn typed_scalars_and_equal_mapping_keys_keep_their_semantics() {
    for text in [
        "123456789012345678901234567890",
        "-123456789012345678901234567890",
    ] {
        let loaded = load(text).unwrap();
        assert!(matches!(loaded.value(loaded.root()), Value::Int(n) if n.text() == text));
    }
    for (text, expected) in [
        ("-0.0", -0.0_f64),
        (".inf", f64::INFINITY),
        ("-.inf", f64::NEG_INFINITY),
    ] {
        let loaded = load(text).unwrap();
        let Value::Float(value) = loaded.value(loaded.root()) else {
            panic!("float expected")
        };
        assert_eq!(value.to_bits(), expected.to_bits());
    }
    let loaded = load(".nan").unwrap();
    assert!(matches!(loaded.value(loaded.root()),Value::Float(v) if v.is_nan()));
    let loaded = load("2024-02-29").unwrap();
    let Value::Date(value) = loaded.value(loaded.root()) else {
        panic!("date expected")
    };
    assert_eq!(rules::date_iso(value), "2024-02-29");
    let loaded = load("2024-01-02 03:04:05.123456+05:30").unwrap();
    let Value::Timestamp(value) = loaded.value(loaded.root()) else {
        panic!("timestamp expected")
    };
    assert_eq!(
        rules::datetime_iso(value),
        "2024-01-02T03:04:05.123456+05:30"
    );
    // Python treats true and 1 as the same mapping key. The last value
    // wins without changing the first key's type or insertion position.
    let loaded = load("{true: first, other: middle, 1: last}").unwrap();
    let Value::Map(pairs) = loaded.value(loaded.root()) else {
        panic!("map expected")
    };
    assert_eq!(pairs.len(), 2);
    assert_eq!(loaded.value(pairs[0].0), &Value::Bool(true));
    assert_eq!(loaded.text(pairs[0].1), Some("last"));
    assert_eq!(loaded.text(pairs[1].0), Some("other"));
    let loaded = load("{x: [null, 'yes', 2]}").unwrap();
    let list = loaded.get(loaded.root(), "x").unwrap();
    let Value::Seq(values) = loaded.value(list) else {
        panic!("sequence expected")
    };
    assert_eq!(values.len(), 3);
    assert_eq!(loaded.value(values[0]), &Value::Null);
    assert_eq!(loaded.text(values[1]), Some("yes"));
    assert!(matches!(loaded.value(values[2]),Value::Int(n) if n.text()=="2"));
    for malformed in ["a: [", "{a: b", "---\na\n---\nb"] {
        assert!(load(malformed).is_err(), "{malformed}");
    }
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
/// are read in no time and no memory, and a list may hold itself.
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
///.
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
/// for what it is (J-9), a mapping keyed by itself before it is
/// made.
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
/// PyYAML's constructor did.
#[test]
fn a_replaced_value_is_still_read() {
    assert!(load("v: {a: 2024-13-01, a: 2}").is_err());
    let loaded = load("v: {a: 1, a: 2}").unwrap();
    let v = loaded.get(loaded.root(), "v").unwrap();
    assert!(matches!(loaded.value(loaded.get(v, "a").unwrap()), Value::Int(n) if n.text() == "2"));
}

/// A block scalar's chomping is its header's, read from its indicator,
/// never from an indicator in the header's comment.
#[test]
fn a_block_header_s_comment_is_no_indicator() {
    let value = |text: &str| {
        let loaded = load(text).unwrap();
        loaded
            .text(loaded.get(loaded.root(), "v").unwrap())
            .unwrap()
            .to_owned()
    };
    assert_eq!(value("v: |- # |+\n  hello\n"), "hello");
    assert_eq!(value("v: |+ # |-\n  hello\n\n"), "hello\n\n");
    assert_eq!(value("v: | # >-\n  hello\n"), "hello\n");
    // PyYAML refuses a comment that touches the header, as `yaml-rust2`.
    assert!(load("v: |-#c\n  a\n").is_err());
    assert_eq!(value("\"a|b\": x\nv: |+ # |\n  a\n\n"), "a\n\n");
}

/// `!!str` makes text of a scalar, as PyYAML does; a lone `-` in a flow
/// collection is refused by name (the third reviews).
#[test]
fn a_str_tag_reads_as_text_and_a_flow_dash_is_named() {
    let value = |text: &str| {
        let loaded = load(text).unwrap();
        loaded
            .value(loaded.get(loaded.root(), "v").unwrap())
            .clone()
    };
    assert_eq!(
        value("v: !!str 2024-01-01"),
        Value::Str("2024-01-01".into())
    );
    assert_eq!(value("v: !!str 0x1f"), Value::Str("0x1f".into()));
    assert_eq!(value("v: !!str"), Value::Str(String::new()));
    assert_eq!(value("v: !!str |\n  a"), Value::Str("a".into()));
    let refused = load("v: !!str [a]").unwrap_err();
    assert_eq!(refused.outside, None);
    for text in ["v: [-]", "v: {a: -}", "v: [x, -, y]", "v: [ - ]"] {
        assert_eq!(
            load(text).unwrap_err().outside,
            Some(Outside::FlowDash),
            "{text}"
        );
    }
}

/// Block nesting is refused by name past the bound, a million levels as
/// promptly as two hundred and one, with no recursion to overflow.
#[test]
fn deep_block_nesting_is_refused_by_name() {
    for levels in [MAX_DEPTH + 1, 3_000, 1_000_000] {
        let mut text = String::from("v:\n");
        text.push_str(&"- ".repeat(levels));
        text.push_str("1\n");
        assert_eq!(
            load(&text).unwrap_err().outside,
            Some(Outside::Nesting),
            "{levels}"
        );
        let mut text = String::from("source: c.s.t\n");
        for level in 0..levels.min(1_000) {
            text.push_str(&" ".repeat(level));
            text.push_str(&format!("k{level}:\n"));
        }
        assert_eq!(
            load(&text).unwrap_err().outside,
            Some(Outside::Nesting),
            "{levels}"
        );
    }
    let mut shallow = String::from("v:\n");
    shallow.push_str(&"- ".repeat(MAX_DEPTH - 2));
    shallow.push_str("1\n");
    assert!(load(&shallow).is_ok());
}

/// Many block scalars read in time linear in the text: eight times as many take not much over eight times as long,
/// where each block once re-walked the text to its end (20,000 took 224
/// s).
#[test]
#[allow(
    clippy::disallowed_methods,
    clippy::disallowed_types,
    reason = "a test of growth measures its own time; the core reads no clock"
)]
fn many_block_scalars_read_in_linear_time() {
    let text = |n: usize| "- |\n  a\n".repeat(n);
    let best = |text: &str| {
        (0..3)
            .map(|_| {
                let started = std::time::Instant::now();
                let loaded = load(text).unwrap();
                let took = started.elapsed().as_secs_f64();
                assert!(
                    matches!(loaded.value(loaded.root()), Value::Seq(items) if !items.is_empty())
                );
                took
            })
            .fold(f64::INFINITY, f64::min)
    };
    let (small, large) = (best(&text(2_500)), best(&text(20_000)));
    assert!(large / small.max(1e-6) < 24.0, "{small} s, then {large} s");
}
