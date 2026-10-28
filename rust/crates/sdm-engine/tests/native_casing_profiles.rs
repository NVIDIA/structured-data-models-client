// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Existing complete native profile receipts through the public casing kernel.

use std::collections::BTreeMap;

use serde_json::Value;

use sdm_contract::scalar::unicode::UnicodeText;
use sdm_engine::words::{CasingProfile, UnicodeDataVersion, lower_native, native_space};

fn lower(text: &[char], profile: CasingProfile) -> Vec<char> {
    let text = UnicodeText::from_points(text.iter().copied().map(u32::from)).unwrap();
    lower_native(text.as_slice(), profile)
        .points()
        .map(|point| char::from_u32(point).unwrap())
        .collect()
}

fn space(character: char) -> bool {
    native_space(u32::from(character))
}

#[test]
fn simple_17_reuses_the_original_protected_arrow_scalar_and_word_receipt() {
    // Keep this producer identity distinct from the new metadata proof.
    // The original receipt is never regenerated to fit this kernel.
    let observed: Value = serde_json::from_str(include_str!("words.json")).unwrap();
    assert_eq!(observed["environment"]["python"], "3.12.13");
    assert_eq!(observed["environment"]["pandas"], "3.0.6");
    assert_eq!(observed["environment"]["pyarrow"], "25.0.1");
    let mut changed = BTreeMap::new();
    for (point, values) in observed["lower"].as_object().unwrap() {
        let point = point.parse::<u32>().unwrap();
        let values: Vec<_> = values
            .as_array()
            .unwrap()
            .iter()
            .map(|value| char::from_u32(value.as_u64().unwrap() as u32).unwrap())
            .collect();
        changed.insert(point, values);
    }
    assert_eq!(changed.len(), 1488);
    let profile = CasingProfile::Simple(UnicodeDataVersion::V17_0);
    for point in 0..=0x10ffff {
        let Some(character) = char::from_u32(point) else {
            continue;
        };
        let expected = changed
            .get(&point)
            .cloned()
            .unwrap_or_else(|| vec![character]);
        assert_eq!(
            lower(&[character], profile),
            expected,
            "original Arrow scalar U+{point:04X}"
        );
    }
    let words = observed["values"].as_array().unwrap();
    assert_eq!(words.len(), 772);
    for value in words {
        assert_eq!(
            serde_json::json!(sdm_engine::words::normalize_with_profile(
                value[0].as_str(),
                profile
            )),
            value[1]
        );
    }
}

struct ObservedRanges {
    values: Vec<(u32, u32)>,
    at: usize,
}

impl ObservedRanges {
    fn new(value: &Value) -> Self {
        Self {
            values: value
                .as_array()
                .unwrap()
                .iter()
                .map(|range| {
                    (
                        range[0].as_u64().unwrap() as u32,
                        range[1].as_u64().unwrap() as u32,
                    )
                })
                .collect(),
            at: 0,
        }
    }

    // The native producer compressed actual per-scalar outcomes into
    // sorted ranges. This only decodes them as the scalar scan advances.
    fn contains(&mut self, point: u32) -> bool {
        while self
            .values
            .get(self.at)
            .is_some_and(|&(_, end)| end < point)
        {
            self.at += 1;
        }
        self.values
            .get(self.at)
            .is_some_and(|&(first, end)| first <= point && point <= end)
    }
}

#[test]
fn historical_profiles_match_every_scalar_and_effective_native_sigma_context() {
    for (source, version, database) in [
        (
            include_str!("t44_builtin_unicode_13_reference.json"),
            UnicodeDataVersion::V13_0,
            "13.0.0",
        ),
        (
            include_str!("t44_builtin_unicode_14_reference.json"),
            UnicodeDataVersion::V14_0,
            "14.0.0",
        ),
        (
            include_str!("t44_builtin_unicode_15_reference.json"),
            UnicodeDataVersion::V15_0,
            "15.0.0",
        ),
        (
            include_str!("t44_builtin_unicode_15_1_reference.json"),
            UnicodeDataVersion::V15_1,
            "15.1.0",
        ),
        (
            include_str!("t44_builtin_unicode_16_reference.json"),
            UnicodeDataVersion::V16_0,
            "16.0.0",
        ),
    ] {
        let observed: Value = serde_json::from_str(source).unwrap();
        assert_eq!(observed["unicode_database"], database);
        assert_eq!(observed["scalar_count"], 1_112_064);
        let profile = CasingProfile::FullRoot(version);
        let mut differences = observed["full_lower_differences"]
            .as_array()
            .unwrap()
            .iter()
            .peekable();
        let mut after_scalar = ObservedRanges::new(&observed["sigma_after_scalar_final_ranges"]);
        let mut after_a_scalar =
            ObservedRanges::new(&observed["sigma_after_a_scalar_final_ranges"]);
        let mut whitespace = ObservedRanges::new(&observed["whitespace_ranges"]);
        for point in 0..=0x10ffff {
            let Some(character) = char::from_u32(point) else {
                continue;
            };
            let expected: Vec<_> = if differences
                .peek()
                .is_some_and(|entry| entry[0].as_u64() == Some(u64::from(point)))
            {
                differences.next().unwrap()[1]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|value| char::from_u32(value.as_u64().unwrap() as u32).unwrap())
                    .collect()
            } else {
                vec![character]
            };
            assert_eq!(
                lower(&[character], profile),
                expected,
                "{database} U+{point:04X}"
            );
            let mut context = expected.clone();
            context.push(if after_scalar.contains(point) {
                'ς'
            } else {
                'σ'
            });
            assert_eq!(
                lower(&[character, 'Σ'], profile),
                context,
                "{database} scalar+Sigma U+{point:04X}"
            );
            let mut context = vec!['a'];
            context.extend(&expected);
            context.push(if after_a_scalar.contains(point) {
                'ς'
            } else {
                'σ'
            });
            assert_eq!(
                lower(&['A', character, 'Σ'], profile),
                context,
                "{database} A+scalar+Sigma U+{point:04X}"
            );
            assert_eq!(
                space(character),
                whitespace.contains(point),
                "{database} whitespace U+{point:04X}"
            );
        }
        assert!(differences.next().is_none());
        for example in observed["examples"].as_array().unwrap() {
            let text: Vec<_> = example["input"]
                .as_str()
                .unwrap()
                .chars()
                .take(sdm_engine::words::MAX_CHARS)
                .collect();
            assert_eq!(
                lower(&text, profile).iter().collect::<String>(),
                example["first_300_lower"].as_str().unwrap()
            );
        }
    }
}
