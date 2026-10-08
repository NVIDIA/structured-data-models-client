// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python's character classes and the string methods built on them.

use crate::unicode_tables::{DECIMAL, SPACE, WORD};

fn within(table: &[(u32, u32)], c: char) -> bool {
    let cp = c as u32;
    table
        .binary_search_by(|&(lo, hi)| {
            if hi < cp {
                std::cmp::Ordering::Less
            } else if lo > cp {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Equal
            }
        })
        .is_ok()
}

/// Whether `\w` matches `c`.
pub fn is_word(c: char) -> bool {
    within(WORD, c)
}

/// Whether `\s` matches `c`, as `str.isspace` holds for it.
pub fn is_space(c: char) -> bool {
    within(SPACE, c)
}

/// The value of `c` if `\d` matches it.
pub fn decimal_value(c: char) -> Option<u32> {
    let cp = c as u32;
    DECIMAL
        .iter()
        .find(|&&(lo, hi)| lo <= cp && cp <= hi)
        .map(|&(lo, _)| (cp - lo) % 10)
}

/// `text.strip()`.
pub fn strip(text: &str) -> &str {
    text.trim_matches(is_space)
}

/// Whether `c` matches the lower-case ASCII letter `letter` under
/// `re.IGNORECASE`: its simple lower case is the letter, or Python counts it
/// as the same letter (the dotless and dotted `i`, the long `s`).
pub fn matches_folded(c: char, letter: char) -> bool {
    if c.to_ascii_lowercase() == letter {
        return true;
    }
    matches!(
        (letter, c),
        ('i', '\u{130}' | '\u{131}') | ('k', '\u{212a}') | ('s', '\u{17f}')
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classes_follow_python() {
        for c in [
            'a',
            'Z',
            '_',
            '0',
            'é',
            'ǅ',
            '٣',
            '²',
            '½',
            'ª',
            '中',
            '\u{1d400}',
        ] {
            assert!(is_word(c), "{c:?}");
        }
        for c in ['$', '.', ' ', '\u{200d}', '\u{301}', '-', '\u{e000}'] {
            assert!(!is_word(c), "{c:?}");
        }
        for c in [
            ' ', '\t', '\x0b', '\x1c', '\x1f', '\u{85}', '\u{a0}', '\u{3000}',
        ] {
            assert!(is_space(c), "{c:?}");
        }
        for c in ['\u{200b}', 'a', '\u{feff}'] {
            assert!(!is_space(c), "{c:?}");
        }
        assert_eq!(decimal_value('7'), Some(7));
        assert_eq!(decimal_value('٣'), Some(3));
        assert_eq!(decimal_value('\u{1d7d8}'), Some(0));
        assert_eq!(decimal_value('²'), None);
        assert_eq!(strip("\u{85} a b\u{3000}"), "a b");
    }

    #[test]
    fn folding_follows_python() {
        assert!(matches_folded('S', 's'));
        assert!(matches_folded('ſ', 's'));
        assert!(matches_folded('İ', 'i'));
        assert!(matches_folded('K', 'k'));
        assert!(!matches_folded('ß', 's'));
    }
}
