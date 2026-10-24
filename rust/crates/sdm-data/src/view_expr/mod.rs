// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The checks on a view's SQL expressions (T3.5, old-behaviour rows 61 to
//! 68), ported from the old client's `rfm/view_expr.py` and the Snowflake
//! qualifier helpers of `rfm/graph.py:107-191`.
//!
//! A metric or semantic view is authored by someone else, yet its
//! expressions are spliced into the statements a graph samples with and
//! run under the caller's credentials. So each expression is read as its
//! dialect lexes it ([`lex`]): text no scalar expression needs (a
//! statement separator, a comment, a sub-query, an unpaired quote or
//! bracket) is refused whatever the caller trusts ([`unsafe_reason`]), and
//! unless the caller trusts the view, every call must be to a pure,
//! built-in function on the dialect's allow-list and no session value may
//! be read ([`untrusted_call_reason`], [`words`]). Which words of a
//! Databricks expression name values is settled by a scan of its grammar
//! ([`name_references`]), so a function, type, keyword or datetime unit is
//! never read as a column or a parameter.
//!
//! [`metric`] holds a metric view's expression helpers (names, join
//! conditions, qualifiers, parameters, fields, the filter, measures, join
//! keys) and [`describe`] reads what `DESCRIBE TABLE EXTENDED` says of one.
//!
//! These are pure functions of the expression's text, and each refusal is
//! typed: the core words it as the old client did, quoting a value as
//! Python writes it (`sdm-core::view_words`), which no code here does.

mod check;
pub mod describe;
mod lex;
pub mod metric;
mod names;
mod pattern;
mod qualifier;
mod words;

pub use check::{ExprRefusal, unsafe_reason, untrusted_call_reason};
pub use lex::{string_literals, strip_comments};
pub use names::{NameReference, Unreadable, name_references};
pub use qualifier::{mask_literals, qualifier_found, strip_qualifier};

/// The dialect an expression is written in, which decides how it lexes
/// and which functions it may call.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ExprDialect {
    Databricks,
    Snowflake,
}

/// The kind of view an expression is taken from, as a dropped column's
/// message names it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ViewKind {
    MetricView,
    SemanticView,
}

impl ViewKind {
    fn noun(self) -> &'static str {
        match self {
            ViewKind::MetricView => "metric view",
            ViewKind::SemanticView => "semantic view",
        }
    }
}

/// Why a view's column expression is dropped, or `None` to keep it
/// (`_view_expr_message`, `RE/graph.py:159-191`): unsafe text whatever the
/// trust, and unless `trust_expressions`, a call that is not trusted, with
/// the way out. `column` names the column as the message does (`column
/// 'x'`, or `column 'x' of table 't'`); `columns` are the columns of the
/// table the expression is evaluated on; `text` holds Python's rules the
/// core hands in.
pub fn expression_message<S: AsRef<str>>(
    column: &str,
    expr: &str,
    view: ViewKind,
    dialect: ExprDialect,
    trust_expressions: bool,
    columns: &[S],
    text: TextRules<'_>,
) -> Option<String> {
    let mut hatch = "";
    let mut reason = unsafe_reason(expr, Some(dialect), text.spaces);
    if reason.is_none() && !trust_expressions {
        reason = untrusted_call_reason(expr, dialect, columns, text.spaces);
        hatch = ". Pass trust_expressions=True to keep it if you trust whoever authored the view";
    }
    let reason = reason?.words(text.quote);
    Some(format!(
        "Failed to add {column} since {reason}, and expressions taken from the {} are executed \
         as SQL under your warehouse credentials{hatch}",
        view.noun()
    ))
}

/// The Python text rules the core hands in to word an expression's
/// refusal: how the old client quoted a piece of it (`repr`), and white
/// space.
#[derive(Clone, Copy)]
pub struct TextRules<'a> {
    pub quote: &'a dyn Fn(&str) -> String,
    pub spaces: &'a Spaces,
}

/// White space as the old client's Python reads it (`str.isspace`, what
/// `strip`, `rstrip` and `split` take off, and `re`'s `\s`): the rule is
/// `sdm-pql`'s, the one place it is written, and the core hands it in.
/// Its class for patterns is derived from it, by testing every code point
/// once, so a pattern's `\s` holds exactly what the rule holds.
#[derive(Clone, Debug)]
pub struct Spaces {
    is: fn(char) -> bool,
    class: String,
}

impl Spaces {
    /// The rule `is`, with its class: each run of code points it holds, as
    /// a pattern's range.
    pub fn new(is: fn(char) -> bool) -> Spaces {
        let mut class = String::new();
        let mut run: Option<(u32, u32)> = None;
        let close = |run: (u32, u32), class: &mut String| {
            if run.0 == run.1 {
                class.push_str(&format!("\\x{{{:x}}}", run.0));
            } else {
                class.push_str(&format!("\\x{{{:x}}}-\\x{{{:x}}}", run.0, run.1));
            }
        };
        for code in 0..=u32::from(char::MAX) {
            let holds = char::from_u32(code).is_some_and(is);
            run = match (run, holds) {
                (Some((first, last)), true) if last + 1 == code => Some((first, code)),
                (Some(done), true) => {
                    close(done, &mut class);
                    Some((code, code))
                }
                (None, true) => Some((code, code)),
                (Some(done), false) => {
                    close(done, &mut class);
                    None
                }
                (None, false) => None,
            };
        }
        if let Some(done) = run {
            close(done, &mut class);
        }
        Spaces { is, class }
    }

    /// Whether `c` is white space.
    pub fn holds(&self, c: char) -> bool {
        (self.is)(c)
    }

    /// The members of a pattern's class of white space.
    pub fn class(&self) -> &str {
        &self.class
    }

    /// `text.strip()`.
    pub(crate) fn strip<'a>(&self, text: &'a str) -> &'a str {
        text.trim_matches(self.is)
    }

    /// `text.rstrip()`.
    pub(crate) fn rstrip<'a>(&self, text: &'a str) -> &'a str {
        text.trim_end_matches(self.is)
    }

    /// `text.split()`: the words between runs of white space.
    pub(crate) fn split<'a>(&self, text: &'a str) -> impl Iterator<Item = &'a str> {
        text.split(self.is).filter(|part| !part.is_empty())
    }
}
