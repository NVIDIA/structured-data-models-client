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

pub use check::{ExprRefusal, reason_words, unsafe_reason, untrusted_call_reason};
pub use lex::{string_literals, strip_comments};
pub use names::{NameReference, Unreadable, name_references};
pub use qualifier::{mask_literals, qualifier_found, strip_qualifier};
use sdm_contract::data::Relation;
pub use sdm_contract::view::{DeclaredType, ExprVerdict, ViewTextAnswer, ViewTextAsk};

/// The dialect an expression is written in, which decides how it lexes
/// and which functions it may call.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ExprDialect {
    Databricks,
    Snowflake,
}
/// Why a view's column expression is refused, and whether trusting the
/// view would keep it, or `None` to keep it (`_view_expr_message`'s
/// reason, `RE/graph.py:159-191`): unsafe text whatever the trust, and
/// unless `trust_expressions`, a call that is not trusted. `columns` are
/// the columns of the table the expression is evaluated on.
pub fn expression_verdict<S: AsRef<str>>(
    expr: &str,
    dialect: ExprDialect,
    trust_expressions: bool,
    columns: &[S],
    spaces: &Spaces,
) -> Option<ExprVerdict> {
    if let Some(reason) = unsafe_reason(expr, Some(dialect), spaces) {
        return Some(ExprVerdict {
            reason,
            untrusted: false,
        });
    }
    if trust_expressions {
        return None;
    }
    untrusted_call_reason(expr, dialect, columns, spaces).map(|reason| ExprVerdict {
        reason,
        untrusted: true,
    })
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

/// The answer to one question of a semantic model's text
/// (`DataLayer::view_text`, T3.27), in `dialect`, its white space
/// `spaces`, each the helper the question is named after; `declared` reads
/// a declared type by the dialect's type map. A [`ViewTextAsk::Derive`]
/// is the relation itself, which the dialect renders where it reads it.
pub fn answer(
    ask: ViewTextAsk,
    dialect: ExprDialect,
    spaces: &Spaces,
    declared: &dyn Fn(&str) -> DeclaredType,
) -> ViewTextAnswer {
    use metric::{
        check_filter, derived_column, join_condition, read_measures, read_parameters, reads_star,
        references_alias, renamed_column, resolve_join_keys, strip_alias, substitute_fields,
        substitute_parameters, table_reference, type_class, view_name, wildcard,
    };
    match ask {
        ViewTextAsk::ViewName { name } => ViewTextAnswer::Names(match dialect {
            ExprDialect::Databricks => view_name(&name, spaces),
            // Snowflake's semantic lookup takes the supplied name as bound
            // text, including its own quotes, spaces and dots. The warehouse
            // resolves it; the metric-view identifier grammar does not.
            ExprDialect::Snowflake => Some(vec![name]),
        }),
        ViewTextAsk::TableReference { name } => {
            ViewTextAnswer::Names(table_reference(&name, spaces))
        }
        ViewTextAsk::JoinCondition { condition } => {
            ViewTextAnswer::Condition(join_condition(&condition, spaces))
        }
        ViewTextAsk::StripComments { expr } => ViewTextAnswer::Text(strip_comments(&expr, spaces)),
        ViewTextAsk::TypeClass { data_type } => {
            ViewTextAnswer::TypeClass(type_class(&data_type, spaces))
        }
        ViewTextAsk::Parameters { inputs, described } => {
            ViewTextAnswer::Parameters(read_parameters(&inputs, &described, spaces))
        }
        ViewTextAsk::SubstituteParameters { expr, parameters } => {
            ViewTextAnswer::Substituted(substitute_parameters(&expr, &parameters, spaces))
        }
        ViewTextAsk::SubstituteFields {
            expr,
            fields,
            qualifiers,
        } => ViewTextAnswer::Fields(substitute_fields(&expr, &fields, &qualifiers, spaces)),
        ViewTextAsk::Filter { filter, candidates } => {
            ViewTextAnswer::Filter(check_filter(filter.as_deref(), &candidates, spaces))
        }
        ViewTextAsk::Wildcard { expr } => ViewTextAnswer::Wildcard(wildcard(&expr, spaces)),
        ViewTextAsk::ReadsStar { expr } => ViewTextAnswer::Flag(reads_star(&expr, spaces)),
        ViewTextAsk::ReferencesAlias { expr, path } => {
            ViewTextAnswer::Flag(references_alias(&expr, &path, spaces))
        }
        ViewTextAsk::StripAlias { expr, path } => {
            ViewTextAnswer::Text(strip_alias(&expr, &path, spaces))
        }
        ViewTextAsk::DerivedColumn { expr } => {
            ViewTextAnswer::Column(derived_column(&expr, spaces))
        }
        ViewTextAsk::RenamedColumn { expr } => {
            ViewTextAnswer::Column(renamed_column(&expr, spaces))
        }
        ViewTextAsk::Measures {
            measures,
            candidates,
            kept,
            dimensions,
            refused,
            parameters,
        } => {
            let (read, left_out) = read_measures(
                &measures,
                &candidates,
                &kept,
                &dimensions,
                &refused,
                &parameters,
                spaces,
            );
            ViewTextAnswer::Measures(read, left_out)
        }
        ViewTextAsk::JoinKeys {
            join,
            child_columns,
            parent_columns,
        } => ViewTextAnswer::JoinKeys(resolve_join_keys(&join, &child_columns, &parent_columns)),
        ViewTextAsk::DeclaredType { data_type } => {
            ViewTextAnswer::DeclaredType(declared(&data_type))
        }
        ViewTextAsk::ReasonWords { reason } => ViewTextAnswer::Words(reason_words(&reason)),
        ViewTextAsk::Expression {
            expr,
            trust_expressions,
            columns,
        } => ViewTextAnswer::Expression(expression_verdict(
            &expr,
            dialect,
            trust_expressions,
            &columns,
            spaces,
        )),
        ViewTextAsk::CrossTable { expr, others } => {
            ViewTextAnswer::Flag(cross_table(&expr, &others, spaces))
        }
        ViewTextAsk::StripQualifier { table, expr } => {
            ViewTextAnswer::Text(strip_qualifier(&table, &expr, spaces))
        }
        ViewTextAsk::Derive {
            table,
            columns,
            filter,
        } => ViewTextAnswer::Relation(Relation {
            source: table.source,
            columns,
            filter,
            found: false,
        }),
    }
}

/// Whether `expr` names any of the logical tables `others`, outside
/// single-quoted literals (`RE/graph.py:2134`).
fn cross_table(expr: &str, others: &[String], spaces: &Spaces) -> Result<bool, Unreadable> {
    let masked = mask_literals(expr, spaces)?;
    for other in others {
        if qualifier_found(other, &masked, spaces)? {
            return Ok(true);
        }
    }
    Ok(false)
}
