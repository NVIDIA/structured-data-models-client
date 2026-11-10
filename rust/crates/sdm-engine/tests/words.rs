// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Text feature normalization: explicit behavior and independent Unicode mappings.

use sdm_engine::words::{lower_case, normalize};
use std::collections::BTreeMap;

#[test]
fn unicode_lowering_matches_independent_arrow_mapping() {
    let document: serde_json::Value = serde_json::from_str(include_str!("words.json")).unwrap();
    let changed: BTreeMap<u32, Vec<u32>> =
        serde_json::from_value(document["lower"].clone()).unwrap();
    assert!(!changed.is_empty());
    for codepoint in 0..=0x10ffff {
        let Some(character) = char::from_u32(codepoint) else {
            continue;
        };
        let expected = changed
            .get(&codepoint)
            .cloned()
            .unwrap_or_else(|| vec![codepoint]);
        assert_eq!(
            lower_case(character).map(u32::from).collect::<Vec<_>>(),
            expected,
            "U+{codepoint:04X}"
        );
    }
}

#[test]
fn punctuation_and_missing_values_have_explicit_words() {
    assert!(normalize(None).is_empty());
    assert!(normalize(Some("")).is_empty());
    assert_eq!(
        normalize(Some("Hello, (WORLD)! 'Yes' \"NO\"; why?: done.")),
        ["hello", "world", "yes", "no", "why", "done"]
    );
    assert_eq!(
        normalize(Some("a-b c_d e/f g+h @i #j $k %l &m =n [o] {p}")),
        [
            "a-b", "c_d", "e/f", "g+h", "@i", "#j", "$k", "%l", "&m", "=n", "[o]", "{p}"
        ]
    );
}

#[test]
fn all_supported_whitespace_splits_but_zero_width_characters_do_not() {
    // Unicode White_Space plus the four Python information separators.
    let spaces = [
        0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x1c, 0x1d, 0x1e, 0x1f, 0x20, 0x85, 0xa0, 0x1680, 0x2000,
        0x2001, 0x2002, 0x2003, 0x2004, 0x2005, 0x2006, 0x2007, 0x2008, 0x2009, 0x200a, 0x2028,
        0x2029, 0x202f, 0x205f, 0x3000,
    ];
    for point in spaces {
        let c = char::from_u32(point).unwrap();
        assert_eq!(
            normalize(Some(&format!("A{c}{c}B"))),
            ["a", "b"],
            "U+{point:04X}"
        );
    }
    for c in ['\u{200b}', '\u{feff}', '\u{180e}'] {
        assert_eq!(normalize(Some(&format!("A{c}B"))), [format!("a{c}b")]);
    }
}

#[test]
fn html_breaks_are_case_insensitive_and_require_complete_tags() {
    for tag in [
        "<br>",
        "<BR/>",
        "<br />",
        "<Br\t/>",
        "<br\n>",
        "<br\u{1c}/>",
    ] {
        assert_eq!(normalize(Some(&format!("A{tag}B"))), ["a", "b"]);
    }
    for tag in ["<br", "<br//>", "<brx>", "<br/ >", "< br>"] {
        assert_eq!(
            normalize(Some(tag)),
            tag.split_whitespace().collect::<Vec<_>>()
        );
    }
    assert_eq!(normalize(Some("A<br><br/>B")), ["a", "b"]);
}

#[test]
fn limits_count_unicode_characters_before_words_and_markup() {
    for count in [299, 300, 301] {
        assert_eq!(
            normalize(Some(&"É".repeat(count))),
            ["é".repeat(count.min(300))]
        );
    }
    for count in [49, 50, 51] {
        let words: Vec<_> = (0..count).map(|i| format!("w{i}")).collect();
        let input = words.join(" ");
        assert!(input.len() < 300);
        assert_eq!(normalize(Some(&input)), words[..count.min(50)]);
    }
    assert_eq!(
        normalize(Some(&format!("{}<br>", "X".repeat(298)))),
        [format!("{}<b", "x".repeat(298))]
    );
    assert_eq!(
        normalize(Some(&format!("{}<br>", "X".repeat(296)))),
        ["x".repeat(296)]
    );
    assert_eq!(normalize(Some("İ ΣΟΣ ẞ K")), ["i", "σοσ", "ß", "k"]);
}
