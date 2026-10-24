// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The one way the view expressions' checks match a pattern. Each pattern
//! is the old client's, written as it wrote it; two things make it match
//! as its `re` matched:
//!
//! - `\s` is the old client's white space, which holds the four separators
//!   U+001C to U+001F that Rust's `\s` does not (T3.5 fix 1): read either
//!   way, `WITH\x1cx AS (x)` would open a sub-query for the warehouse and
//!   none for the check. This is a pattern's class, written into each
//!   fixed pattern once it is compiled; the same white space as a test of
//!   a character, for `strip` and `split`, is `sdm-pql`'s rule, which the
//!   core hands in ([`super::Spaces`]);
//! - a match is total: a pattern that runs out of its backtracking budget
//!   answers [`Unreadable::TooDeep`], never "no match", so no check can
//!   accept an expression because a pattern failed on it. Each caller
//!   refuses on it. So does a pattern made of a definition's text that is
//!   past the compiler's bounds ([`Pattern::build`]).

use std::ops::Range;

use fancy_regex::{Regex, RegexBuilder};

use super::names::Unreadable;

/// How many backtracking steps a match may take: far past what any
/// expression the old client read in time needs, yet bounded.
const BACKTRACK_LIMIT: usize = 1 << 31;

pub(crate) type Read<T> = Result<T, Unreadable>;

/// A compiled pattern.
pub(crate) struct Pattern(Regex);

/// The old client's `\s` in a pattern: outside a class `[\s\x1C-\x1F]`,
/// inside one `\s\x1C-\x1F`.
fn white_space(pattern: &str) -> String {
    let mut out = String::with_capacity(pattern.len());
    let mut chars = pattern.chars();
    let mut class = false;
    while let Some(c) = chars.next() {
        match c {
            '\\' => {
                let Some(next) = chars.next() else {
                    out.push(c);
                    break;
                };
                match (next, class) {
                    ('s', false) => out.push_str(r"[\s\x1C-\x1F]"),
                    ('s', true) => out.push_str(r"\s\x1C-\x1F"),
                    _ => {
                        out.push('\\');
                        out.push(next);
                    }
                }
            }
            '[' if !class => {
                class = true;
                out.push(c);
                // A `]` or `^]` first in a class is a member, not its end.
                let rest = chars.as_str();
                let skip = if rest.starts_with("^]") {
                    2
                } else if rest.starts_with(']') || rest.starts_with('^') {
                    1
                } else {
                    0
                };
                for _ in 0..skip {
                    out.extend(chars.next());
                }
            }
            ']' if class => {
                class = false;
                out.push(c);
            }
            _ => out.push(c),
        }
    }
    out
}

fn unreadable(_: fancy_regex::Error) -> Unreadable {
    Unreadable::TooDeep
}

impl Pattern {
    /// Compiles one of the old client's fixed patterns, read with its
    /// `\s`; each compiles, as the tests of the patterns show.
    pub(crate) fn new(pattern: &str) -> Pattern {
        Pattern::build(pattern).expect("a fixed pattern compiles")
    }

    /// Compiles a pattern made of a definition's text, a table's name in
    /// it: one past the compiler's bounds is unreadable, never a panic.
    pub(crate) fn build(pattern: &str) -> Read<Pattern> {
        RegexBuilder::new(&white_space(pattern))
            .backtrack_limit(BACKTRACK_LIMIT)
            .build()
            .map(Pattern)
            .map_err(unreadable)
    }

    pub(crate) fn is_match(&self, text: &str) -> Read<bool> {
        self.0.is_match(text).map_err(unreadable)
    }

    /// Where the first match is.
    pub(crate) fn find(&self, text: &str) -> Read<Option<Range<usize>>> {
        Ok(self.0.find(text).map_err(unreadable)?.map(|m| m.range()))
    }

    /// Where the first match starting at or after `from` is, the text
    /// before `from` still seen by a lookbehind.
    pub(crate) fn find_from(&self, text: &str, from: usize) -> Read<Option<Range<usize>>> {
        Ok(self
            .0
            .find_from_pos(text, from)
            .map_err(unreadable)?
            .map(|m| m.range()))
    }

    /// Where each match is, in order.
    pub(crate) fn find_all(&self, text: &str) -> Read<Vec<Range<usize>>> {
        self.0
            .find_iter(text)
            .map(|m| m.map(|m| m.range()).map_err(unreadable))
            .collect()
    }

    /// The spans of the first match's groups, the whole match first; a
    /// group that took no part is `None`.
    pub(crate) fn groups(&self, text: &str) -> Read<Option<Groups>> {
        let Some(captures) = self.0.captures(text).map_err(unreadable)? else {
            return Ok(None);
        };
        Ok(Some(Groups {
            spans: (0..captures.len())
                .map(|i| captures.get(i).map(|m| m.range()))
                .collect(),
            names: self
                .0
                .capture_names()
                .enumerate()
                .filter_map(|(i, name)| Some((name?.to_owned(), i)))
                .collect(),
        }))
    }

    /// The spans of each match's groups, in order.
    pub(crate) fn groups_all(&self, text: &str) -> Read<Vec<Groups>> {
        let names: Vec<(String, usize)> = self
            .0
            .capture_names()
            .enumerate()
            .filter_map(|(i, name)| Some((name?.to_owned(), i)))
            .collect();
        self.0
            .captures_iter(text)
            .map(|captures| {
                let captures = captures.map_err(unreadable)?;
                Ok(Groups {
                    spans: (0..captures.len())
                        .map(|i| captures.get(i).map(|m| m.range()))
                        .collect(),
                    names: names.clone(),
                })
            })
            .collect()
    }

    /// `text` with each match replaced by `with`.
    pub(crate) fn replace_all(&self, text: &str, with: &str) -> Read<String> {
        let mut out = String::with_capacity(text.len());
        let mut last = 0;
        for span in self.find_all(text)? {
            out.push_str(&text[last..span.start]);
            out.push_str(with);
            last = span.end;
        }
        out.push_str(&text[last..]);
        Ok(out)
    }
}

/// The spans of a match's groups.
pub(crate) struct Groups {
    spans: Vec<Option<Range<usize>>>,
    names: Vec<(String, usize)>,
}

impl Groups {
    pub(crate) fn get(&self, group: usize) -> Option<Range<usize>> {
        self.spans.get(group).cloned().flatten()
    }

    pub(crate) fn name(&self, name: &str) -> Option<Range<usize>> {
        let (_, group) = self.names.iter().find(|(n, _)| n == name)?;
        self.get(*group)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn white_space_is_the_old_client_s_inside_and_outside_a_class() {
        assert_eq!(white_space(r"a\sb"), r"a[\s\x1C-\x1F]b");
        assert_eq!(white_space(r"[^\s.]"), r"[^\s\x1C-\x1F.]");
        assert_eq!(white_space(r"[\s\S]"), r"[\s\x1C-\x1F\S]");
        assert_eq!(white_space(r"\\s"), r"\\s");
        assert_eq!(white_space(r"[]\s]"), r"[]\s\x1C-\x1F]");
        let p = Pattern::new(r"^with\s+x$");
        assert!(p.is_match("with\u{1c}x").unwrap());
    }
}
