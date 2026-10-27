// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native lowercase over pinned Unicode data, without native callbacks.
//! Input is already clipped to the sampler's bounded character count.

mod data;

/// Actual database version of a native lowercase kernel.
/// Neither the Python nor the Arrow package version identifies this fact.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnicodeDataVersion {
    V13_0,
    V14_0,
    V15_0,
    V15_1,
    V16_0,
    V17_0,
}

/// The two native lowercase kernels selected by built-in string storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CasingProfile {
    /// Arrow's per-scalar lowercase, including the simple dotted-I mapping.
    Simple(UnicodeDataVersion),
    /// Python's full root lowercase, including contextual final sigma.
    FullRoot(UnicodeDataVersion),
}

impl CasingProfile {
    /// Resolve the raw result of the fixed native Arrow kernel identity.
    /// Equal complete simple tables share one representative data profile;
    /// this does not infer the native library's package or database version.
    pub fn from_simple_identity(identity: &str) -> Option<Self> {
        data::IDENTITIES
            .iter()
            .find(|&&(observed, _)| observed == identity)
            .map(|&(_, version)| Self::Simple(version))
    }
}

/// Fixed public-kernel input used only to observe native identity metadata.
/// User values are never part of this bounded observation.
pub fn simple_identity_input() -> &'static str {
    data::IDENTITY_INPUT
}

#[derive(Clone, Copy)]
struct Tables {
    simple: &'static [(char, char)],
    overrides: &'static [(char, char)],
    full: &'static [(char, &'static str)],
    cased: &'static [(char, char)],
    ignorable: &'static [(char, char)],
}

impl Tables {
    fn simple(self, character: char) -> char {
        if let Ok(index) = self
            .overrides
            .binary_search_by_key(&character, |&(key, _)| key)
        {
            return self.overrides[index].1;
        }
        self.simple
            .binary_search_by_key(&character, |&(key, _)| key)
            .map(|index| self.simple[index].1)
            .unwrap_or(character)
    }

    fn extend_full<T: From<char>>(self, character: char, result: &mut Vec<T>) {
        match self.full.binary_search_by_key(&character, |&(key, _)| key) {
            Ok(index) => result.extend(self.full[index].1.chars().map(T::from)),
            Err(_) => result.push(T::from(self.simple(character))),
        }
    }

    fn cased(self, character: char) -> bool {
        in_ranges(self.cased, character)
    }

    fn ignorable(self, character: char) -> bool {
        in_ranges(self.ignorable, character)
    }
}

fn in_ranges(ranges: &[(char, char)], character: char) -> bool {
    let end = ranges.partition_point(|&(first, _)| first <= character);
    end > 0 && character <= ranges[end - 1].1
}

/// All supported Python databases must have the same whitespace set before
/// the generator emits this shared predicate. Arrow casing still uses this
/// Python predicate for the later regex and split stages.
pub(super) fn space(character: char) -> bool {
    data::SPACE.binary_search(&character).is_ok()
}

pub(super) fn lower(text: &[char], profile: CasingProfile) -> Vec<char> {
    lower_points(text, Some, profile)
}

/// The same kernel over native codepoints. A non-scalar point is unchanged,
/// neither Cased nor Case_Ignorable, so it is a real context boundary.
/// This avoids separate kernels or per-scalar-run allocation on native data.
pub(super) fn lower_points<T: Copy + From<char>>(
    text: &[T],
    scalar: fn(T) -> Option<char>,
    profile: CasingProfile,
) -> Vec<T> {
    let (version, full) = match profile {
        CasingProfile::Simple(version) => (version, false),
        CasingProfile::FullRoot(version) => (version, true),
    };
    let tables = data::tables(version);
    if !full {
        return text
            .iter()
            .map(|&character| scalar(character)
                .map(|character| T::from(tables.simple(character)))
                .unwrap_or(character))
            .collect();
    }

    // Final_Sigma examines the original, clipped string. Case_Ignorable
    // characters are skipped even when also Cased. Compute the suffix once
    // so a long sequence of sigmas/combining marks remains linear.
    let mut following = vec![false; text.len()];
    let mut next_cased = false;
    for (index, &character) in text.iter().enumerate().rev() {
        following[index] = next_cased;
        let scalar = scalar(character);
        if !scalar.is_some_and(|character| tables.ignorable(character)) {
            next_cased = scalar.is_some_and(|character| tables.cased(character));
        }
    }
    let mut result = Vec::with_capacity(text.len());
    let mut preceding_cased = false;
    for (index, &character) in text.iter().enumerate() {
        let scalar = scalar(character);
        if scalar == Some('\u{3a3}') && preceding_cased && !following[index] {
            result.push(T::from('\u{3c2}'));
        } else if let Some(character) = scalar {
            tables.extend_full(character, &mut result);
        } else {
            result.push(character);
        }
        if !scalar.is_some_and(|character| tables.ignorable(character)) {
            preceding_cased = scalar.is_some_and(|character| tables.cased(character));
        }
    }
    result
}

#[cfg(test)]
mod native_profiles {
    use std::collections::BTreeMap;

    use serde_json::Value;

    use super::{CasingProfile, UnicodeDataVersion, lower, space};

    #[test]
    fn simple_17_reuses_the_original_protected_arrow_scalar_and_word_receipt() {
        // Keep this producer identity distinct from the new metadata proof.
        // The original receipt is never regenerated to fit this kernel.
        let observed: Value = serde_json::from_str(include_str!("../../tests/words.json")).unwrap();
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
                serde_json::json!(crate::words::normalize_with_profile(
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
                include_str!("../../tests/t44_builtin_unicode_13_reference.json"),
                UnicodeDataVersion::V13_0,
                "13.0.0",
            ),
            (
                include_str!("../../tests/t44_builtin_unicode_14_reference.json"),
                UnicodeDataVersion::V14_0,
                "14.0.0",
            ),
            (
                include_str!("../../tests/t44_builtin_unicode_15_reference.json"),
                UnicodeDataVersion::V15_0,
                "15.0.0",
            ),
            (
                include_str!("../../tests/t44_builtin_unicode_15_1_reference.json"),
                UnicodeDataVersion::V15_1,
                "15.1.0",
            ),
            (
                include_str!("../../tests/t44_builtin_unicode_16_reference.json"),
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
            let mut after_scalar =
                ObservedRanges::new(&observed["sigma_after_scalar_final_ranges"]);
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
                    .take(crate::words::MAX_CHARS)
                    .collect();
                assert_eq!(
                    lower(&text, profile).iter().collect::<String>(),
                    example["first_300_lower"].as_str().unwrap()
                );
            }
        }
    }
}
