// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared sampler word boundaries over ordinary or native codepoints.
//! Punctuation becomes whitespace before the HTML line-break substitution.

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
