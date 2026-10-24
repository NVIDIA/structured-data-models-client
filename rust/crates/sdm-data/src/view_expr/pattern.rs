// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The one way the view expressions' checks match a pattern. Each pattern
//! is the old client's, written as it wrote it; two things make it match
//! as its `re` matched:
//!
//! - `\s` is the old client's white space, Python's (`str.isspace`), which
//!   holds the four separators U+001C to U+001F that Rust's `\s` does not
//!   (T3.5 fix 1): read either way, `WITH\x1cx AS (x)` would open a
//!   sub-query for the warehouse and none for the check. The rule is
//!   `sdm-pql`'s, handed in by the core as [`Spaces`]; a pattern's `\s` is
//!   the class derived from it (every code point the rule holds), and `\S`
//!   its complement, so no pattern holds the engine's own `\s`. Each fixed
//!   pattern is compiled on first use with that class, and kept
//!   ([`Lazy`]);
//! - a match is total: a pattern that runs out of its backtracking budget
//!   answers [`Unreadable::TooDeep`], never "no match", so no check can
//!   accept an expression because a pattern failed on it. Each caller
//!   refuses on it. So does a pattern made of a definition's text that is
//!   past the compiler's bounds ([`Pattern::build`]).

use std::borrow::Cow;
use std::ops::Range;
use std::sync::OnceLock;

use fancy_regex::{Regex, RegexBuilder};

use super::Spaces;
use super::names::Unreadable;

/// How many backtracking steps a match may take: far past what any
/// expression the old client read in time needs, yet bounded.
const BACKTRACK_LIMIT: usize = 1 << 31;

pub(crate) type Read<T> = Result<T, Unreadable>;

/// A compiled pattern.
#[derive(Clone)]
pub(crate) struct Pattern(Regex);

/// A fixed pattern (or patterns), compiled on first use with the white
/// space handed in, and kept for each later use with the same white space;
/// with another, compiled again for that use.
pub(crate) struct Lazy<T> {
    make: fn(&Spaces) -> T,
    made: OnceLock<(String, T)>,
}

impl<T: Clone> Lazy<T> {
    pub(crate) const fn new(make: fn(&Spaces) -> T) -> Lazy<T> {
        Lazy {
            make,
            made: OnceLock::new(),
        }
    }

    pub(crate) fn get(&self, spaces: &Spaces) -> Cow<'_, T> {
        let (class, made) = self
            .made
            .get_or_init(|| (spaces.class().to_owned(), (self.make)(spaces)));
        if class == spaces.class() {
            Cow::Borrowed(made)
        } else {
            Cow::Owned((self.make)(spaces))
        }
    }
}

/// The old client's `\s` and `\S` in a pattern, by `class`, the white
/// space's members: outside a class `[class]` and `[^class]`, inside one
/// `class`; `None` for a `\S` inside a class, which no fixed pattern holds.
fn white_space(pattern: &str, class: &str) -> Option<String> {
    let mut out = String::with_capacity(pattern.len());
    let mut chars = pattern.chars();
    let mut in_class = false;
    while let Some(c) = chars.next() {
        match c {
            '\\' => {
                let Some(next) = chars.next() else {
                    out.push(c);
                    break;
                };
                match (next, in_class) {
                    ('s', false) => out.push_str(&format!("[{class}]")),
                    ('s', true) => out.push_str(class),
                    ('S', false) => out.push_str(&format!("[^{class}]")),
                    ('S', true) => return None,
                    _ => {
                        out.push('\\');
                        out.push(next);
                    }
                }
            }
            '[' if !in_class => {
                in_class = true;
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
            ']' if in_class => {
                in_class = false;
                out.push(c);
            }
            _ => out.push(c),
        }
    }
    Some(out)
}

fn unreadable(_: fancy_regex::Error) -> Unreadable {
    Unreadable::TooDeep
}

impl Pattern {
    /// Compiles one of the old client's fixed patterns, read with its
    /// `\s`; each compiles, as the tests of the patterns show.
    pub(crate) fn new(pattern: &str, spaces: &Spaces) -> Pattern {
        Pattern::build(pattern, spaces).expect("a fixed pattern compiles")
    }

    /// Compiles a pattern made of a definition's text, a table's name in
    /// it: one past the compiler's bounds is unreadable, never a panic.
    pub(crate) fn build(pattern: &str, spaces: &Spaces) -> Read<Pattern> {
        let text = white_space(pattern, spaces.class()).ok_or(Unreadable::NotUnderstood)?;
        RegexBuilder::new(&text)
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
    fn white_space_is_the_class_inside_and_outside_a_class() {
        let class = r"\x{9}-\x{d}";
        assert_eq!(white_space(r"a\sb", class).unwrap(), r"a[\x{9}-\x{d}]b");
        assert_eq!(white_space(r"[^\s.]", class).unwrap(), r"[^\x{9}-\x{d}.]");
        assert_eq!(white_space(r"a\S", class).unwrap(), r"a[^\x{9}-\x{d}]");
        assert_eq!(white_space(r"\\s", class).unwrap(), r"\\s");
        assert_eq!(white_space(r"[]\s]", class).unwrap(), r"[]\x{9}-\x{d}]");
        assert_eq!(white_space(r"[\S]", class), None);
    }

    /// A pattern's `\s` and `\S`, outside a class and inside one, match
    /// exactly what the white space handed in holds, over every code
    /// point.
    #[test]
    fn a_pattern_s_white_space_is_the_rule_s() {
        let spaces = Spaces::new(|c| c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c));
        let inside = Pattern::new(r"^[\sx]$", &spaces);
        let outside = Pattern::new(r"^\s$", &spaces);
        let not = Pattern::new(r"^\S$", &spaces);
        for code in 0..=u32::from(char::MAX) {
            let Some(c) = char::from_u32(code) else {
                continue;
            };
            let text = c.to_string();
            let holds = spaces.holds(c);
            assert_eq!(outside.is_match(&text), Ok(holds), "U+{code:04X}");
            assert_eq!(not.is_match(&text), Ok(!holds), "U+{code:04X}");
            assert_eq!(
                inside.is_match(&text),
                Ok(holds || c == 'x'),
                "U+{code:04X}"
            );
        }
    }
}
