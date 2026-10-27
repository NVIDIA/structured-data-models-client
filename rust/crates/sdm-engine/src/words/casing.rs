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

    fn extend_full(self, character: char, result: &mut Vec<char>) {
        match self.full.binary_search_by_key(&character, |&(key, _)| key) {
            Ok(index) => result.extend(self.full[index].1.chars()),
            Err(_) => result.push(self.simple(character)),
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
    let (version, full) = match profile {
        CasingProfile::Simple(version) => (version, false),
        CasingProfile::FullRoot(version) => (version, true),
    };
    let tables = data::tables(version);
    if !full {
        return text
            .iter()
            .map(|&character| tables.simple(character))
            .collect();
    }

    // Final_Sigma examines the original, clipped string. Case_Ignorable
    // characters are skipped even when also Cased. Compute the suffix once
    // so a long sequence of sigmas/combining marks remains linear.
    let mut following = vec![false; text.len()];
    let mut next_cased = false;
    for (index, &character) in text.iter().enumerate().rev() {
        following[index] = next_cased;
        if !tables.ignorable(character) {
            next_cased = tables.cased(character);
        }
    }
    let mut result = Vec::with_capacity(text.len());
    let mut preceding_cased = false;
    for (index, &character) in text.iter().enumerate() {
        if character == '\u{3a3}' && preceding_cased && !following[index] {
            result.push('\u{3c2}');
        } else {
            tables.extend_full(character, &mut result);
        }
        if !tables.ignorable(character) {
            preceding_cased = tables.cased(character);
        }
    }
    result
}
