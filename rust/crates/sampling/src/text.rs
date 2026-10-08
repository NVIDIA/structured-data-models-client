// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `base/sampler._normalize_text`: text cut into lower-case words.
//!
//! Its steps cross three definitions of a character, and each is kept:
//! the cut to `6 * max_words` and the lower case are pandas' string column's
//! (Arrow: code points, and [`crate::lower_table`]'s one-to-one mapping),
//! while punctuation, `<br>` tags and the word split are Python's `re` and
//! `str.split` on Python's own whitespace (`str.isspace`, which holds for
//! U+001C to U+001F and U+0085, unlike Rust's `char::is_whitespace`).

use sdm_dialects::unicode::is_space;

use crate::lower_table::LOWER;

/// The lower case of `c` as pandas' Arrow-backed strings give it.
pub fn arrow_lower(c: char) -> char {
    if c.is_ascii() {
        return c.to_ascii_lowercase();
    }
    match LOWER.binary_search_by_key(&(c as u32), |&(from, _)| from) {
        Ok(at) => char::from_u32(LOWER[at].1).unwrap_or(c),
        Err(_) => c,
    }
}

/// `PUNCTUATION`: `['".,()!?;:]`.
fn is_punctuation(c: char) -> bool {
    matches!(
        c,
        '\'' | '"' | '.' | ',' | '(' | ')' | '!' | '?' | ';' | ':'
    )
}

/// `re.sub(r'<br\s*/?>', ' ', text)`.
fn replace_line_breaks(text: &[char]) -> Vec<char> {
    let mut out = Vec::with_capacity(text.len());
    let mut i = 0;
    while i < text.len() {
        if text[i..].starts_with(&['<', 'b', 'r']) {
            let mut j = i + 3;
            while j < text.len() && is_space(text[j]) {
                j += 1;
            }
            if j < text.len() && text[j] == '/' {
                j += 1;
            }
            if j < text.len() && text[j] == '>' {
                out.push(' ');
                i = j + 1;
                continue;
            }
        }
        out.push(text[i]);
        i += 1;
    }
    out
}

/// `items[:stop]`, as Python slices: a negative `stop` counts from the end.
fn cut<T>(items: &mut Vec<T>, stop: i64) {
    let keep = if stop < 0 {
        items
            .len()
            .saturating_sub(usize::try_from(stop.unsigned_abs()).unwrap_or(usize::MAX))
    } else {
        usize::try_from(stop).unwrap_or(usize::MAX)
    };
    items.truncate(keep);
}

/// One value normalized: `None` is the empty text. A negative `max_words`
/// is a Python slice's negative stop, as it is in `_normalize_text`.
pub fn normalize_one(value: Option<&str>, max_words: Option<i64>) -> Vec<String> {
    let mut chars: Vec<char> = value.unwrap_or("").chars().collect();
    if let Some(max_words) = max_words {
        cut(&mut chars, max_words.saturating_mul(6));
    }
    let lowered: Vec<char> = chars
        .into_iter()
        .map(arrow_lower)
        .map(|c| if is_punctuation(c) { ' ' } else { c })
        .collect();
    let line: String = replace_line_breaks(&lowered).into_iter().collect();
    let mut words: Vec<String> = line
        .split(is_space)
        .filter(|word| !word.is_empty())
        .map(str::to_string)
        .collect();
    if let Some(max_words) = max_words {
        cut(&mut words, max_words);
    }
    words
}

/// `_normalize_text(pd.Series(values, dtype=object), max_words).tolist()`
/// for text and nulls.
pub fn normalize_text(values: &[Option<String>], max_words: Option<i64>) -> Vec<Vec<String>> {
    values
        .iter()
        .map(|value| normalize_one(value.as_deref(), max_words))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn words_are_lowered_and_split_on_python_whitespace() {
        assert_eq!(
            normalize_one(Some("Hello,\u{1c}WORLD<br />and\u{85}more!"), Some(50)),
            ["hello", "world", "and", "more"]
        );
    }

    #[test]
    fn the_cut_and_the_word_limit() {
        assert_eq!(normalize_one(Some("a b c d e"), Some(2)), ["a", "b"]);
        assert_eq!(
            normalize_one(Some("abcdefghijklm"), Some(2)),
            ["abcdefghijkl"]
        );
        assert!(normalize_one(None, None).is_empty());
    }

    #[test]
    fn a_negative_word_limit_slices_from_the_end_as_python_does() {
        // "a b c d e"[:-6] is "a b", and ["a", "b"][:-1] is ["a"].
        assert_eq!(normalize_one(Some("a b c d e"), Some(-1)), ["a"]);
        assert!(normalize_one(Some("a b c d e"), Some(-2)).is_empty());
        assert!(normalize_one(Some("a b c"), Some(0)).is_empty());
        assert!(normalize_one(Some("a b c"), Some(i64::MIN)).is_empty());
        assert_eq!(
            normalize_one(Some("a b c"), Some(i64::MAX)),
            ["a", "b", "c"]
        );
    }

    #[test]
    fn a_dotted_capital_i_lowers_to_one_code_point() {
        assert_eq!(normalize_one(Some("\u{130}"), None), ["i"]);
    }
}
