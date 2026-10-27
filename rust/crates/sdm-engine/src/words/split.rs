// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared sampler word boundaries over ordinary or native codepoints.
//! Punctuation becomes whitespace before the HTML line-break substitution.

use std::ops::Range;

/// The first bounded words as exact ranges of already lowercased input.
/// A non-scalar native surrogate is an ordinary non-whitespace character.
/// No Unicode replacement or pair merging occurs during boundary selection.
pub(super) fn ranges<T: Copy>(
    text: &[T],
    scalar: impl Fn(T) -> Option<char>,
    space: fn(char) -> bool,
) -> Vec<Range<usize>> {
    let mut words = Vec::new();
    let mut start = None;
    let mut at = 0;
    while at < text.len() {
        let boundary = scalar(text[at]).is_some_and(|point| space(point) || super::punctuation(point));
        let next = if boundary {
            Some(at + 1)
        } else {
            line_break(text, at, &scalar, space)
        };
        if let Some(next) = next {
            if let Some(start) = start.take() {
                words.push(start..at);
                if words.len() == super::MAX_WORDS {
                    return words;
                }
            }
            at = next;
        } else {
            start.get_or_insert(at);
            at += 1;
        }
    }
    if let Some(start) = start {
        words.push(start..at);
    }
    words
}

fn line_break<T: Copy>(
    text: &[T],
    at: usize,
    scalar: &impl Fn(T) -> Option<char>,
    space: fn(char) -> bool,
) -> Option<usize> {
    let prefix = text.get(at..at + 3)?;
    if !prefix.iter().copied().map(scalar).eq(['<', 'b', 'r'].map(Some)) {
        return None;
    }
    let mut end = at + 3;
    // PUNCTUATION is replaced before the reference's HTML regex. Treat it
    // as whitespace here too, without rewriting or copying the input.
    while text.get(end).copied().and_then(scalar)
        .is_some_and(|point| space(point) || super::punctuation(point))
    {
        end += 1;
    }
    if text.get(end).copied().and_then(scalar) == Some('/') {
        end += 1;
    }
    (text.get(end).copied().and_then(scalar) == Some('>')).then_some(end + 1)
}

#[cfg(test)]
mod tests {
    use super::ranges;

    #[test]
    fn punctuation_precedes_html_matching_without_splitting_native_units() {
        let text = "a<br'/>b<BR />c<br\t/>d";
        let text: Vec<_> = text.chars().collect();
        let words: Vec<String> = ranges(&text, Some, crate::words::space)
            .into_iter()
            .map(|range| text[range].iter().collect())
            .collect();
        // Input is already lowercased at this stage, so an uppercase HTML
        // tag here is ordinary text. Punctuation inside a lowercased tag
        // becomes whitespace before that tag's regex is applied.
        assert_eq!(words, ["a", "b<BR", "/>c", "d"]);
        let points = [65, 0xd83d, 0xde3a, 32, 0x1f63a, 0, 66];
        assert_eq!(
            ranges(&points, char::from_u32, crate::words::space),
            [0..3, 4..7]
        );
    }

    #[test]
    fn word_cap_and_nonmatching_tags_keep_exact_existing_boundaries() {
        for (text, expected) in [
            ("<br x>", vec!["<br", "x>"]),
            ("x<br/ >y", vec!["x<br/", ">y"]),
            ("x\u{1c}y\u{a0}z\u{85}w", vec!["x", "y", "z", "w"]),
            ("a\u{200b}b", vec!["a\u{200b}b"]),
            ("  ...  ", vec![]),
        ] {
            let points: Vec<_> = text.chars().map(u32::from).collect();
            let words: Vec<String> = ranges(&points, char::from_u32, crate::words::space)
                .into_iter()
                .map(|range| points[range].iter().map(|&point| char::from_u32(point).unwrap()).collect())
                .collect();
            assert_eq!(words, expected);
        }
        let text: Vec<_> = "w ".repeat(200).chars().collect();
        assert_eq!(ranges(&text, Some, crate::words::space).len(), 50);
    }
}
