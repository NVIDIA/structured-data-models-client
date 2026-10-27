// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What the core asks the data layer about a semantic model's text, and
//! what it answers (T3.27, design S5): [`DataLayer::view_text`]'s
//! [`ViewTextAsk`] and [`ViewTextAnswer`].
//!
//! A semantic model's conversion is one ordered routine in the core, in the
//! old client's read order; every question it has of an expression, a name
//! or a condition is a pure question of text in the run's provider's
//! dialect, which the data layer answers, so the core never reads SQL and
//! the data layer never sees the definition's YAML or the run. Expressions
//! cross as opaque text. The types are the T3.5 pure modules' own
//! (`sdm-data`'s `view_expr` and `view_expr::metric`, at lane j's fix 2),
//! taken as they are; each refusal is typed, and the core words it as the
//! old client did.
//!
//! These types never travel: they are in-process, as [`crate::catalog`]'s.
//!
//! [`DataLayer::view_text`]: crate::data::DataLayer::view_text

use std::fmt;

use crate::data::{Relation, RelationColumn};
use crate::ops::Dtype;

/// The kind of semantic model a definition is: a dialect reads one kind
/// (Databricks metric views, Snowflake semantic views), and a sentence
/// names it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ViewKind {
    Metric,
    Semantic,
}

impl ViewKind {
    /// The kind as a sentence names it.
    pub fn noun(self) -> &'static str {
        match self {
            ViewKind::Metric => "metric view",
            ViewKind::Semantic => "semantic view",
        }
    }
}

/// Why an expression is refused (old-behaviour rows 64 to 66), typed so
/// the core words it as the old client did (`sdm-core`'s view words).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExprRefusal {
    /// It holds `;`, `--`, `/*`, `*/`, `|>` or `->>`, anywhere.
    Contains(&'static str),
    /// `$$` on Databricks, which quotes nothing with it.
    DollarQuotes,
    /// A Snowflake comment, `//` or another.
    SlashComment,
    /// A backslash inside a Snowflake double-quoted name.
    BackslashInQuotedName,
    /// A character outside quotes that is not ASCII.
    NotAscii(char),
    /// A quote left unpaired: `$$`, `'`, `"` or a backtick.
    Unpaired(&'static str),
    ClosesUnopened,
    LeavesUnclosed,
    /// A statement keyword outside quotes, as written.
    Keyword(String),
    SubQuery,
    /// A call of what a bracket or an index gives.
    CallsResult,
    /// A call of something that is no name, with the ten characters before
    /// its `(`, quoted text masked.
    CallsAfter(String),
    /// A call of a quoted name.
    QuotedFunction,
    /// A call of a qualified name: its last part.
    QualifiedFunction(String),
    /// A function named through `IDENTIFIER()`.
    Identifier,
    /// A call of a function off the dialect's allow-list, as written.
    NotPure(String),
    /// A session word read bare, as written.
    SessionValue(String),
    /// A session word read quoted, its quotes included.
    QuotedSessionValue(String),
    /// A pattern the check reads it by failed on it, so it cannot be read
    /// safely (T3.5 fix 1; the old client has no such limit).
    Unreadable(Unreadable),
}

/// An expression whose names could not be settled within the bounds the
/// scan keeps, so no reading of it is safe (`UnreadableExpressionError`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unreadable {
    TooDeep,
    NotUnderstood,
}

impl fmt::Display for Unreadable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Unreadable::TooDeep => "it is too deeply nested or too long to read safely",
            Unreadable::NotUnderstood => {
                "it has syntax that is not understood, so which of its words name columns \
                 cannot be settled"
            }
        })
    }
}

impl std::error::Error for Unreadable {}

/// Why an expression is refused, and whether trusting the view would keep
/// it: an untrusted call, which `trust_expressions` accepts, or unsafe
/// text, which nothing does (`_view_expr_message`, `RE/graph.py:159-191`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExprVerdict {
    pub reason: ExprRefusal,
    /// The refusal is of a call the caller does not trust, so the message
    /// says how to trust it.
    pub untrusted: bool,
}

/// One side of a join condition: the (dot-joined) table alias or join path
/// qualifying the column, if any, and the column.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KeyRef {
    pub qualifier: Option<String>,
    pub column: String,
}

/// The literal a parameter's type takes, as `_default_text` tells types
/// apart (`DB/metric_view.py:513-566`): by the type in lower case, its
/// white space single.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TypeClass {
    Boolean,
    /// `int`, `bigint` and the like.
    Integer,
    /// `double`, `float`, `real`, and `decimal` with or without its
    /// precision.
    Decimal,
    /// `string`, `char(n)` and `varchar(n)`.
    String,
    Date,
    /// `timestamp`, `timestamp_ntz` and `timestamp_ltz`.
    Timestamp,
    /// Any other type, which has no literal here.
    NoLiteral,
}

/// Why a default gives no literal of its type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DefaultRefusal {
    /// The type has no literal here.
    NoLiteral,
    /// The default is no value of the type.
    NotOfType,
    /// An integer of more digits than Python reads, so many.
    TooManyDigits(usize),
    /// A text `int` cannot read though the old client's pattern took it.
    NotAnInteger,
    /// A value whose text would be longer than the core writes.
    TooLong,
}

/// What a parameter of the view is given, as the core hands it over: its
/// name, its type as the definition writes it if text, and its default
/// unless null, as the core holds it (`D`).
#[derive(Clone, Debug, PartialEq)]
pub struct ParameterInput<D> {
    pub name: String,
    pub data_type: Option<String>,
    pub default: Option<D>,
}

/// A parameter's default as the core converted it (`_default_text`, which
/// reads the definition's value, the core's): the text it casts from, or
/// why it has none, for each type it may be read as, by that type as
/// written (the definition's, and any the warehouse describes for its
/// name, which wins). The data layer picks the winning type's
/// ([`ViewTextAsk::Parameters`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DefaultTexts(pub Vec<(String, Result<String, DefaultRefusal>)>);

/// A parameter as a query of the view without arguments reads it.
#[derive(Clone, Debug, PartialEq)]
pub struct Parameter {
    pub name: String,
    pub read: Result<ParameterRead, Unread>,
}

/// A parameter's default as a typed SQL literal, and as text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParameterRead {
    pub literal: String,
    pub text: String,
}

/// Why a parameter has no literal.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Unread {
    NoDefault,
    NoType,
    /// Its default is refused for its type, which is given.
    Refused {
        refusal: DefaultRefusal,
        data_type: String,
    },
}

/// Why the filter cannot be applied to the source table alone.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FilterRefusal {
    /// It is no text, or blank.
    NotACondition,
    Unsafe(ExprRefusal),
    /// It reads a joined table, by alias.
    JoinedTable(String),
    /// A pattern the check reads it by failed on it.
    Unreadable(Unreadable),
}

/// A measure as the core hands it over: an entry of `measures` that is a
/// mapping with a text name, its `expr` if text, and whether it has a
/// `window`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MeasureInput {
    pub name: String,
    pub expr: Option<String>,
    pub window: bool,
}

/// A dimension resolved onto one table: its name, the table's alias, and
/// its expression with that table's qualifier taken out.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DimensionRef {
    pub name: String,
    pub alias: String,
    pub expr: String,
}

/// What a measure reads: each column with its table's alias, and the
/// predictive query aggregation it is, if one plain aggregate of the
/// source table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MeasureRead {
    pub name: String,
    pub inputs: Vec<(String, String)>,
    pub aggregation: Option<(String, Option<String>)>,
}

/// A measure left out, and why.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MeasureRefusal {
    pub name: String,
    pub reason: MeasureReason,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MeasureReason {
    Unreadable(Unreadable),
    /// It names a refused dimension, as written.
    NamesRefused(String),
}

/// A join as its keys are resolved: its alias, the alias it is declared
/// on, its path, and its condition's two sides.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JoinRef {
    pub alias: String,
    pub parent_alias: String,
    pub path: Vec<String>,
    pub condition: [KeyRef; 2],
}

/// A view column's declared type read by the dialect's type map (the old
/// client's `DatabricksTable._to_dtype`, `SnowTable._to_dtype`, rows 41 and
/// 55 to 57).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeclaredType {
    /// A type of the map's.
    Known(Dtype),
    /// A type the map refuses, as the old client refuses such a column.
    Unsupported,
    /// A type the map does not read, left to the sampled values.
    Unknown,
}

/// One question about a semantic model's text, in the run's provider's
/// dialect ([`crate::data::DataLayer::view_text`]): each the old client's
/// helper it is named after (`RE/view_expr.py`, `RE/graph.py:107-191`,
/// `DB/metric_view.py`), answered by the [`ViewTextAnswer`] its doc names.
#[derive(Clone, Debug, PartialEq)]
pub enum ViewTextAsk {
    /// The definition lookup's name as the dialect reads it:
    /// [`ViewTextAnswer::Names`], none for an invalid name.
    /// A metric view uses one to three identifier parts (`_split_view_name`);
    /// a semantic view uses its supplied name as one opaque bound-text part.
    ViewName { name: String },
    /// A source's or a join's `source` as a plain one-to-three-part table
    /// name (`_table_reference`): [`ViewTextAnswer::Names`], none for one
    /// that is not (a query).
    TableReference { name: String },
    /// A join's condition as a single-column equality (`_join_condition`):
    /// [`ViewTextAnswer::Condition`].
    JoinCondition { condition: String },
    /// An expression with its comments replaced by a space, as Databricks
    /// drops them (`strip_comments`): [`ViewTextAnswer::Text`].
    StripComments { expr: String },
    /// The class of a parameter's type, by which the core converts its
    /// default (`_default_text`): [`ViewTextAnswer::TypeClass`].
    TypeClass { data_type: String },
    /// The `parameters` section read, the types the warehouse describes
    /// winning, each default the text the core converted it to for the
    /// winning type (`_read_parameters`): [`ViewTextAnswer::Parameters`].
    Parameters {
        inputs: Vec<ParameterInput<DefaultTexts>>,
        described: Vec<(String, String)>,
    },
    /// `expr` with each bare reference to a parameter that has a literal
    /// replaced by it, and the parameters it names
    /// (`_substitute_parameters`); a parameter a source column of its name
    /// shadows is left out of `parameters` by the core
    /// (`DB/metric_view.py:799-802`): [`ViewTextAnswer::Substituted`].
    SubstituteParameters {
        expr: String,
        parameters: Vec<(String, Parameter)>,
    },
    /// `expr` with each reference to an earlier field replaced by its
    /// expression in brackets, `None` for a refused one, and `qualifiers`
    /// every join path or alias, folded (`_substitute_fields`):
    /// [`ViewTextAnswer::Fields`].
    SubstituteFields {
        expr: String,
        fields: Vec<(String, Option<String>)>,
        qualifiers: Vec<Vec<String>>,
    },
    /// The filter, checked or, where the caller trusts the view, trusted
    /// but for unsafe text (`_check_filter`), `candidates` each join path
    /// or alias with the alias it resolves to, longest first:
    /// [`ViewTextAnswer::Filter`].
    Filter {
        filter: Option<String>,
        candidates: Vec<(Vec<String>, String)>,
    },
    /// `alias.*` or `alias.* EXCEPT (a, b)` (`_wildcard`): the path and the
    /// columns left out, [`ViewTextAnswer::Wildcard`].
    Wildcard { expr: String },
    /// Whether `expr` reads every column of a table through `*`
    /// (`_reads_star`): [`ViewTextAnswer::Flag`].
    ReadsStar { expr: String },
    /// Whether `expr` names the join path `path` (`_references_alias`):
    /// [`ViewTextAnswer::Flag`].
    ReferencesAlias { expr: String, path: Vec<String> },
    /// `expr` with the join path `path` taken out as its qualifier
    /// (`_strip_alias`): [`ViewTextAnswer::Text`].
    StripAlias { expr: String, path: Vec<String> },
    /// The source column a derived column reads, if one alone
    /// (`_derived_column`): [`ViewTextAnswer::Column`].
    DerivedColumn { expr: String },
    /// The source column a dimension names as itself, in any case
    /// (`_renamed_column`): [`ViewTextAnswer::Column`].
    RenamedColumn { expr: String },
    /// Which columns each measure reads, never running it
    /// (`_read_measures`): [`ViewTextAnswer::Measures`].
    Measures {
        measures: Vec<MeasureInput>,
        candidates: Vec<(Vec<String>, String)>,
        kept: Vec<String>,
        dimensions: Vec<(String, DimensionRef)>,
        refused: Vec<String>,
        parameters: Vec<String>,
    },
    /// A join's condition resolved into `(other_alias, other_key,
    /// child_key)`, unqualified names the child's first
    /// (`resolve_join_keys`): [`ViewTextAnswer::JoinKeys`].
    JoinKeys {
        join: JoinRef,
        child_columns: Vec<String>,
        parent_columns: Vec<String>,
    },
    /// A view column's declared type (a metric view's described column, a
    /// semantic view's `data_type`), read by the dialect's type map:
    /// [`ViewTextAnswer::DeclaredType`].
    DeclaredType { data_type: String },
    /// The dialect's own words for a refusal's reason that names its
    /// warehouse (Databricks's `$$`, old-behaviour row 64), a clause:
    /// [`ViewTextAnswer::Words`], none for a reason in no warehouse's words,
    /// which the core words.
    ReasonWords { reason: ExprRefusal },
    /// One column's expression checked against `columns`, its table's:
    /// unsafe text whatever the trust, and unless `trust_expressions`, a
    /// call that is not trusted (`_view_expr_message`'s reason):
    /// [`ViewTextAnswer::Expression`], none to keep it.
    Expression {
        expr: String,
        trust_expressions: bool,
        columns: Vec<String>,
    },
    /// Whether `expr` names any of the logical tables `others`, outside
    /// single-quoted literals (`_qualifier_pattern` over `_mask_literals`,
    /// `RE/graph.py:107-156`): [`ViewTextAnswer::Flag`].
    CrossTable { expr: String, others: Vec<String> },
    /// `expr` with its own table's qualifier taken out, literals left as
    /// they are (`_strip_qualifier`): [`ViewTextAnswer::Text`].
    StripQualifier { table: String, expr: String },
    /// The relation a logical table reads: its base table, the columns it
    /// keeps, each a source column or an expression, and the view's filter
    /// on a fact table, rendered as the dialect reads a derived relation:
    /// [`ViewTextAnswer::Relation`].
    Derive {
        table: Relation,
        columns: Vec<RelationColumn>,
        filter: Option<String>,
    },
}

/// The answer to a [`ViewTextAsk`], of the kind its doc names.
#[derive(Clone, Debug, PartialEq)]
pub enum ViewTextAnswer {
    /// A name's parts, unquoted, or none for text that is no such name.
    Names(Option<Vec<String>>),
    /// A condition's two sides, or none for one that is no single-column
    /// equality.
    Condition(Option<[KeyRef; 2]>),
    /// Text rewritten, or why it could not be read.
    Text(Result<String, Unreadable>),
    /// A parameter type's class.
    TypeClass(TypeClass),
    /// The parameters, by folded name, in order.
    Parameters(Vec<(String, Parameter)>),
    /// The text substituted, and the parameters it names, once each.
    Substituted(Result<(String, Vec<String>), Unreadable>),
    /// The text substituted, and a refused field it names, as written.
    Fields(Result<(String, Option<String>), Unreadable>),
    /// The filter, trimmed, or why it cannot be applied.
    Filter(Result<String, FilterRefusal>),
    /// A wildcard's path and the columns it leaves out, or none.
    Wildcard(Option<(Vec<String>, Vec<String>)>),
    /// A yes or no, or why it could not be read.
    Flag(Result<bool, Unreadable>),
    /// A source column, or none.
    Column(Option<String>),
    /// The measures read, and those left out.
    Measures(Vec<MeasureRead>, Vec<MeasureRefusal>),
    /// `(other_alias, other_key, child_key)`, or none.
    JoinKeys(Option<(String, String, String)>),
    /// Why an expression is refused, or none to keep it.
    Expression(Option<ExprVerdict>),
    /// A declared type, read.
    DeclaredType(DeclaredType),
    /// A clause in the dialect's words, or none.
    Words(Option<String>),
    /// A derived relation, as the dialect reads it.
    Relation(Relation),
}
