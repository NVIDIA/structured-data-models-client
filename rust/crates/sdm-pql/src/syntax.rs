// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Small text rules the parser and the date reader share.

/// White space as the reference strips it from around a query: the
/// characters Unicode gives the white-space bidirectional classes.
pub fn is_space(c: char) -> bool {
    matches!(
        c,
        '\t' | '\n'
            | '\u{0b}'
            | '\u{0c}'
            | '\r'
            | '\u{1c}'..='\u{1f}'
                | ' '
                | '\u{85}'
                | '\u{a0}'
                | '\u{1680}'
                | '\u{2000}'..='\u{200a}'
                | '\u{2028}'
                | '\u{2029}'
                | '\u{202f}'
                | '\u{205f}'
                | '\u{3000}'
    )
}

/// Words a syntax message must not repeat, because they belong to a parser
/// generator or a validation library rather than to PQL; quoting a token
/// that holds one would make the message read as if they did.
const FOREIGN: &[&str] = &[
    "mismatched input",
    "extraneous input",
    "input mismatch",
    "no viable alternative",
    "token recognition error",
    "antlr",
    "recognitionexception",
    "<eof>",
    "fqn_column",
    "wildcard_column",
    "rel_op",
    "neg_infty",
    "time_unit",
    "quoted_id",
    "for_each",
    "pydantic",
    "value_error",
    "input_value=",
    "input_type=",
    "argskwargs",
    "validation error",
    "missing ",
    "atn",
];

/// Whether `text` can be quoted in a syntax message as it stands: no control
/// characters, on one line, and free of the words above.
pub fn is_plain(text: &str) -> bool {
    let lower = text.to_lowercase();
    !text.is_empty()
        && text
            .chars()
            .all(|c| !c.is_control() && (c == ' ' || !c.is_whitespace()))
        && !FOREIGN.iter().any(|w| lower.contains(w))
}

/// `text` in single quotes, or in double quotes when it holds a single one.
pub fn quoted(text: &str) -> String {
    if text.contains('\'') && !text.contains('"') {
        format!("\"{text}\"")
    } else {
        format!("'{text}'")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_text_is_quoted_and_the_rest_is_not() {
        assert!(is_plain("users.age"));
        assert!(!is_plain("for_each.x"));
        assert!(!is_plain("a\tb"));
        assert!(!is_plain(""));
        assert_eq!(quoted("'s'"), "\"'s'\"");
        assert_eq!(quoted("x"), "'x'");
    }
}
