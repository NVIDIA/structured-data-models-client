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
//! These are pure functions of the expression's text; the reasons are the
//! old client's words. The old client writes a few of them with Python's
//! `repr` of a piece of the expression ([`quoted`]).

mod check;
mod lex;
mod names;
mod qualifier;
mod words;

pub use check::{unsafe_reason, untrusted_call_reason};
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
/// the way out.
///
/// `column` is how the message names the column (`column 'x'`, or `column
/// 'x' of table 't'`); `columns` are the columns of the table the
/// expression is evaluated on.
pub fn expression_message<S: AsRef<str>>(
    column: &str,
    expr: &str,
    view: ViewKind,
    dialect: ExprDialect,
    trust_expressions: bool,
    columns: &[S],
) -> Option<String> {
    let mut hatch = "";
    let mut reason = unsafe_reason(expr, Some(dialect));
    if reason.is_none() && !trust_expressions {
        reason = untrusted_call_reason(expr, dialect, columns);
        hatch = ". Pass trust_expressions=True to keep it if you trust whoever authored the view";
    }
    let reason = reason?;
    Some(format!(
        "Failed to add {column} since {reason}, and expressions taken from the {} are \
         executed as SQL under your warehouse credentials{hatch}",
        view.noun()
    ))
}

/// The text as the old client's messages quote a piece of an expression,
/// Python's `repr` of it: in single quotes unless it holds one and no
/// double quote, a backslash and that quote escaped, `\t`, `\n` and `\r`
/// as such, and a character Python does not print as `\xhh`, `\uhhhh` or
/// `\Uhhhhhhhh`.
pub(crate) fn quoted(text: &str) -> String {
    let quote = if text.contains('\'') && !text.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::with_capacity(text.len() + 2);
    out.push(quote);
    for c in text.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\t' => out.push_str("\\t"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            _ if c == quote => {
                out.push('\\');
                out.push(c);
            }
            _ if printable(c) => out.push(c),
            _ => {
                let code = u32::from(c);
                if code < 0x100 {
                    out.push_str(&format!("\\x{code:02x}"));
                } else if code < 0x10000 {
                    out.push_str(&format!("\\u{code:04x}"));
                } else {
                    out.push_str(&format!("\\U{code:08x}"));
                }
            }
        }
    }
    out.push(quote);
    out
}

/// Whether the old client's messages show `c` as itself: not a control or
/// format character, a separator other than the space, a private-use
/// character or a noncharacter. Unassigned code points, which it also
/// escapes, are not told apart here.
fn printable(c: char) -> bool {
    let code = u32::from(c);
    if code < 0x80 {
        return (0x20..0x7F).contains(&code);
    }
    !(c.is_control()
        || matches!(code,
            0xA0 | 0xAD | 0x1680 | 0x2000..=0x200F | 0x2028..=0x202F | 0x205F..=0x2064
            | 0x2066..=0x206F | 0x3000 | 0xFEFF | 0xFFF9..=0xFFFB | 0x600..=0x605 | 0x61C
            | 0x6DD | 0x70F | 0x890..=0x891 | 0x8E2 | 0x180E | 0x110BD | 0x110CD
            | 0x13430..=0x1343F | 0x1BCA0..=0x1BCA3 | 0x1D173..=0x1D17A | 0xE0001
            | 0xE0020..=0xE007F | 0xE000..=0xF8FF | 0xF0000..=0xFFFFD
            | 0x100000..=0x10FFFD | 0xFDD0..=0xFDEF)
        || code & 0xFFFE == 0xFFFE)
}

/// Whether `c` is white space as the old client's `str.strip` reads it:
/// Unicode's white space, and the four separators U+001C to U+001F.
fn spacing(c: char) -> bool {
    c.is_whitespace() || ('\u{1C}'..='\u{1F}').contains(&c)
}

/// `text.rstrip()`.
fn rstrip(text: &str) -> &str {
    text.trim_end_matches(spacing)
}

/// `text.strip()`.
fn stripped(text: &str) -> &str {
    text.trim_matches(spacing)
}
