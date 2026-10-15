// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A text feature as the reference's sampler sends it: each value as a list
//! of lower-case words (`_normalize_text`, `rfm/base/sampler.py`), which the
//! payload writes as a `stringlist` column.
//!
//! The reference's steps, on each value: a missing value is the empty
//! text; the text is cut to its first 300 characters (`6 * max_words`,
//! counted in code points); lower-cased code point by code point, by
//! Unicode's simple mapping ([`lower_case`]), as the reference's frame
//! lowers the Arrow-backed text `astype(str)` gives it (Arrow's
//! `utf8_lower`: no final sigma, and U+0130 lowered to `i` alone); each of
//! `'".,()!?;:` and each `<br>`, `<br/>`, `<br />` (any whitespace before
//! the slash) becomes a space; and the text is split on whitespace, the
//! first 50 words kept. Whitespace is what `str.isspace` counts, which the
//! reference's `\s` and `split` both use ([`space`]): Unicode's
//! `White_Space`, as `char::is_whitespace` reads it, and the four
//! separators U+001C to U+001F, which Unicode does not count.
//!
//! `words.json`, recorded in the reference's environment, holds the
//! lowering of every code point and the words of 772 values (T1.26b).

/// The most words a value keeps (`max_words`).
pub const MAX_WORDS: usize = 50;
/// The most characters a value is cut to before it is split
/// (`6 * max_words`).
pub const MAX_CHARS: usize = 6 * MAX_WORDS;

/// Whether `c` is whitespace as the reference's splitting reads it
/// (`str.isspace`): Unicode's `White_Space` and U+001C to U+001F.
pub fn space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

/// Whether `c` is one of the punctuation marks the reference turns into a
/// space (`PUNCTUATION`).
fn punctuation(c: char) -> bool {
    matches!(
        c,
        '\'' | '"' | '.' | ',' | '(' | ')' | '!' | '?' | ';' | ':'
    )
}

/// `c` by Unicode's simple lower-case mapping: `char::to_lowercase`,
/// which is the simple mapping for every code point but U+0130, whose full
/// mapping (`i` and a combining dot) the simple one is not.
pub fn lower_case(c: char) -> impl Iterator<Item = char> {
    let simple = (c == '\u{130}').then_some('i');
    simple
        .into_iter()
        .chain(c.to_lowercase().filter(move |_| simple.is_none()))
}

/// `value`, a missing one as the empty text, as the reference's words.
pub fn normalize(value: Option<&str>) -> Vec<String> {
    let lower: Vec<char> = value
        .unwrap_or("")
        .chars()
        .take(MAX_CHARS)
        .flat_map(lower_case)
        .map(|c| if punctuation(c) { ' ' } else { c })
        .collect();
    // `<br\s*/?>`, each match a space, scanned left to right without
    // overlap as `re.sub` scans.
    let mut text = String::with_capacity(lower.len());
    let mut at = 0;
    while at < lower.len() {
        if let Some(end) = line_break(&lower, at) {
            text.push(' ');
            at = end;
        } else {
            text.push(lower[at]);
            at += 1;
        }
    }
    text.split(space)
        .filter(|word| !word.is_empty())
        .take(MAX_WORDS)
        .map(str::to_owned)
        .collect()
}

/// The end of a `<br\s*/?>` starting at `at`, if one does.
fn line_break(text: &[char], at: usize) -> Option<usize> {
    if text.get(at..at + 3)? != ['<', 'b', 'r'] {
        return None;
    }
    let mut end = at + 3;
    while text.get(end).copied().is_some_and(space) {
        end += 1;
    }
    if text.get(end) == Some(&'/') {
        end += 1;
    }
    (text.get(end) == Some(&'>')).then_some(end + 1)
}

#[cfg(test)]
mod tests {
    use super::normalize;

    fn words(text: &str) -> Vec<String> {
        normalize(Some(text))
    }

    #[test]
    fn values_become_the_reference_s_words() {
        assert_eq!(
            words("Sport-100 Helmet, Red"),
            ["sport-100", "helmet", "red"]
        );
        assert_eq!(words("a<br>b<BR />c<br\t/>d"), ["a", "b", "c", "d"]);
        assert_eq!(words("<br x>"), ["<br", "x>"]);
        assert_eq!(words("x<br/ >y"), ["x<br/", ">y"]);
        assert_eq!(words("x\u{1c}y\u{a0}z\u{85}w"), ["x", "y", "z", "w"]);
        assert_eq!(words("a\u{200b}b"), ["a\u{200b}b"]);
        assert!(normalize(None).is_empty());
        assert!(words("  ...  ").is_empty());
        assert_eq!(words(&"w ".repeat(200)).len(), 50);
        assert_eq!(
            words("\u{130}x \u{3a3}\u{39f}\u{3a3}"),
            ["ix", "\u{3c3}\u{3bf}\u{3c3}"]
        );
        let cut = format!("{}\u{130}b", "a".repeat(299));
        assert_eq!(words(&cut), [format!("{}i", "a".repeat(299))]);
    }
}
