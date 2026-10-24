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
/// table the expression is evaluated on; `quote` writes a piece of the
/// expression as the old client quoted it, which the core hands in.
pub fn expression_message<S: AsRef<str>>(
    column: &str,
    expr: &str,
    view: ViewKind,
    dialect: ExprDialect,
    trust_expressions: bool,
    columns: &[S],
    quote: &dyn Fn(&str) -> String,
) -> Option<String> {
    let mut hatch = "";
    let mut reason = unsafe_reason(expr, Some(dialect));
    if reason.is_none() && !trust_expressions {
        reason = untrusted_call_reason(expr, dialect, columns);
        hatch = ". Pass trust_expressions=True to keep it if you trust whoever authored the view";
    }
    let reason = reason?.words(quote);
    Some(format!(
        "Failed to add {column} since {reason}, and expressions taken from the {} are executed \
         as SQL under your warehouse credentials{hatch}",
        view.noun()
    ))
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
