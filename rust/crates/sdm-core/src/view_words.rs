// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The words of a view's conversion (T3.5): the old client's sentences, as
//! it wrote them, from the typed facts `sdm-schema` and `sdm-data` answer
//! with. A value or a piece of an expression is quoted as the old client
//! quoted it, Python's `repr` or `str`, through the parser's text helpers
//! ([`crate::pql`], the one module that names the parser), so no other crate
//! writes Python's text.
//!
//! Each function gives the sentence's text; T3.27 turns each into a
//! `Sentence` of the contract, its template and fields, which every host
//! words alike. A sentence that quotes a value too large to quote is
//! refused by name ([`TooLarge`]).

use sdm_contract::ops::Dtype;
use sdm_contract::view::ViewKind;
use sdm_contract::{CoreWarning, Detail, Sentence};
use sdm_data::view_expr::ExprRefusal;
use sdm_data::view_expr::metric::{
    DefaultRefusal, FilterRefusal, MeasureReason, MeasureRefusal, Parameter, Unread,
};
use sdm_schema::view::joins::{JoinMessage, SkipReason};
use sdm_schema::view::model::{
    Edge, EdgeReason, JoinNote, JoinRefusal, KeyReason, LinkFailure, MeasureNote,
    RelationshipRefusal, TimeColumnNote,
};
use sdm_schema::view::spec::{Ignored, ViewSpec};
use sdm_schema::view::yaml::{Document, NodeId};

pub use crate::pql::TooLarge;
use crate::pql::conversions::{not_an_integer_text, too_many_digits_text};
use crate::pql::{text_repr, value_repr as repr, value_text as text};
use crate::view_steps::JoinJudgement;

/// The text the old client's messages quote a view definition's value by:
/// Python's `repr` of it.
pub fn value_repr(document: &Document, node: NodeId) -> Result<String, TooLarge> {
    repr(document, node)
}

/// Python's `str` of a view definition's value.
pub fn value_text(document: &Document, node: NodeId) -> Result<String, TooLarge> {
    text(document, node)
}

/// A text as the old client quotes a piece of an expression or a name:
/// Python's `repr` of it. The data layer's sentences take it.
pub fn quote(text: &str) -> String {
    text_repr(text)
}

/// Why an expression is refused, in the old client's words
/// (`RE/view_expr.py`, old-behaviour rows 64 to 66), quoted as it quoted:
/// `words` the data layer's own words for a reason that names its warehouse
/// (`ViewTextAsk::ReasonWords`), which win.
pub fn expr_reason(reason: &ExprRefusal, words: Option<String>) -> String {
    if let Some(words) = words {
        return words;
    }
    match reason {
        ExprRefusal::Contains(token) => format!("it contains {}", quote(token)),
        // A warehouse's own: the dialect words it.
        ExprRefusal::DollarQuotes => "it contains '$$'".into(),
        ExprRefusal::SlashComment => "it contains '//'".into(),
        ExprRefusal::BackslashInQuotedName => {
            "it has a backslash inside a double-quoted name".into()
        }
        ExprRefusal::NotAscii(c) => format!(
            "it has {} outside quotes, which is not ASCII",
            quote(&c.to_string())
        ),
        ExprRefusal::Unpaired(delimiter) => format!(
            "it leaves {} unpaired, so which part of it is quoted text cannot be established",
            quote(delimiter)
        ),
        ExprRefusal::ClosesUnopened => "it closes a '(' it never opened".into(),
        ExprRefusal::LeavesUnclosed => "it leaves a '(' unclosed".into(),
        ExprRefusal::Keyword(keyword) => format!("it contains the SQL keyword '{keyword}'"),
        ExprRefusal::SubQuery => "it contains a sub-query".into(),
        ExprRefusal::CallsResult => "it calls the result of an expression".into(),
        ExprRefusal::CallsAfter(before) => {
            format!("it calls what precedes '(' after {}", quote(before))
        }
        ExprRefusal::QuotedFunction => "it calls a function by a quoted name".into(),
        ExprRefusal::QualifiedFunction(name) => format!("it calls the qualified function '{name}'"),
        ExprRefusal::Identifier => "it names a function through IDENTIFIER()".into(),
        ExprRefusal::NotPure(name) => {
            format!("it calls '{name}', which is not a pure built-in SQL function")
        }
        ExprRefusal::SessionValue(word) => format!("it reads the session value '{word}'"),
        ExprRefusal::QuotedSessionValue(quoted) => format!("it reads the session value {quoted}"),
        ExprRefusal::Unreadable(unreadable) => unreadable.to_string(),
    }
}

/// A view column whose expression is refused, left out
/// (`_view_expr_message`, `RE/graph.py:159-191`, rows 41, 55 and 61): a
/// metric view's dimension by its name, a semantic view's column with its
/// table; `reason` the refusal's clause, and the way out for a call the
/// caller does not trust.
pub fn expression_dropped(
    kind: ViewKind,
    column: &str,
    table: Option<&str>,
    reason: &str,
    untrusted: bool,
) -> CoreWarning {
    let hatch = if untrusted {
        ". Pass trust_expressions=True to keep it if you trust whoever authored the view"
    } else {
        ""
    };
    let named = match table {
        Some(table) => format!("column '{column}' of table '{table}'"),
        None => format!("column '{column}'"),
    };
    let sentence = match (table.is_some(), untrusted) {
        (false, false) => Sentence::ViewExpressionDropped,
        (false, true) => Sentence::ViewExpressionDroppedUntrusted,
        (true, false) => Sentence::ViewTableExpressionDropped,
        (true, true) => Sentence::ViewTableExpressionDroppedUntrusted,
    };
    let mut warning = CoreWarning::new(
        sentence,
        format!(
            "Failed to add {named} since {reason}, and expressions taken from the {} are \
             executed as SQL under your warehouse credentials{hatch}",
            kind.noun()
        ),
    )
    .with_field("column", Detail::text(column))
    .with_field("reason", Detail::text(reason));
    if let Some(table) = table {
        warning = warning.with_field("table", Detail::text(table));
    }
    warning
}

/// Why a default gives no literal of `data_type` (`_default_text`'s
/// refusals).
pub fn default_refusal_text(
    refusal: &DefaultRefusal,
    data_type: &str,
    document: &Document,
    default: NodeId,
) -> Result<String, TooLarge> {
    Ok(match refusal {
        DefaultRefusal::NoLiteral => format!("its type '{data_type}' has no literal here"),
        DefaultRefusal::NotOfType => {
            format!(
                "its default {} is not of type {data_type}",
                repr(document, default)?
            )
        }
        // Python's own words, which the old client passed on.
        DefaultRefusal::TooManyDigits(digits) => too_many_digits_text(*digits),
        DefaultRefusal::NotAnInteger => not_an_integer_text(&text(document, default)?),
        DefaultRefusal::TooLong => {
            format!("its default is too long to write as a literal of type {data_type}")
        }
    })
}

/// A parameter's text: its default's, or why it has no literal
/// (`_Parameter.text`). `default` is its default as the definition gives
/// it, if it has one.
pub fn parameter_text(
    parameter: &Parameter,
    document: &Document,
    default: Option<NodeId>,
) -> Result<String, TooLarge> {
    Ok(match (&parameter.read, default) {
        (Ok(read), _) => read.text.clone(),
        (Err(Unread::Refused { refusal, data_type }), Some(default)) => {
            default_refusal_text(refusal, data_type, document, default)?
        }
        (Err(Unread::NoType), _) => "has no type".into(),
        (Err(_), _) => "has no default".into(),
    })
}

/// The note of a parameter read as its default (`ParameterNote`).
pub fn parameter_note_text(parameter: &str, default: &str) -> String {
    format!(
        "Read the parameter '{parameter}' as its default '{default}', as a query of the view \
         without arguments does"
    )
}

/// How a refused filter can be read after all.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FilterHatch {
    None,
    /// A default for the parameter it reads.
    Default,
    /// `trust_expressions=True`, for a call that is not trusted.
    Trust,
}

/// The refusal of a view for its filter (`_refuse_filter`), `reason` a
/// clause.
pub fn filter_message(
    document: &Document,
    filter: NodeId,
    reason: &str,
    hatch: FilterHatch,
) -> Result<String, TooLarge> {
    let hatch = match hatch {
        FilterHatch::None => "",
        FilterHatch::Default => ", or give the parameter a default of its type",
        FilterHatch::Trust => {
            ", or pass trust_expressions=True to apply it if you trust whoever authored the view"
        }
    };
    Ok(format!(
        "Cannot apply the metric view filter {} since {reason}. The graph would otherwise read \
         rows the view leaves out. Create a regular view with this WHERE clause and point the \
         metric view's source at it{hatch}",
        repr(document, filter)?
    ))
}

/// The refusal of a filter the source table alone cannot apply
/// (`_check_filter`), `words` the data layer's own words for an unsafe
/// reason that names its warehouse.
pub fn filter_refusal_text(
    document: &Document,
    filter: NodeId,
    refusal: &FilterRefusal,
    words: Option<String>,
) -> Result<String, TooLarge> {
    let reason = match refusal {
        FilterRefusal::NotACondition => "it is not a SQL condition".to_owned(),
        FilterRefusal::Unsafe(reason) => expr_reason(reason, words),
        FilterRefusal::JoinedTable(alias) => format!("it references the joined table '{alias}'"),
        FilterRefusal::Unreadable(unreadable) => unreadable.to_string(),
    };
    filter_message(document, filter, &reason, FilterHatch::None)
}

/// A measure left out (`_read_measures`).
pub fn measure_refusal_text(refusal: &MeasureRefusal) -> String {
    let reason = match &refusal.reason {
        MeasureReason::Unreadable(unreadable) => unreadable.to_string(),
        MeasureReason::NamesRefused(dimension) => {
            format!("it names the dimension '{dimension}', which is refused")
        }
    };
    format!("Failed to add measure '{}' since {reason}", refusal.name)
}

/// `n` with a comma between each three digits, as Python's `{n:,}`.
fn grouped(n: u64) -> String {
    let digits = n.to_string();
    let mut out = String::new();
    for (at, c) in digits.chars().enumerate() {
        if at > 0 && (digits.len() - at).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// Why a join's key cannot key its table (`_join_key_reason`), with a
/// repeating value if one was read, as Python's `repr` writes it.
pub fn key_reason_text(
    reason: &KeyReason,
    table: &str,
    key: &str,
    example: Option<&str>,
) -> String {
    match reason {
        KeyReason::NoValue => format!("its key '{key}' holds no value in table '{table}'"),
        KeyReason::NotUnique { repeats, rows } => {
            let example = example.map_or(String::new(), |shown| format!(", e.g. {shown}"));
            format!(
                "its key '{key}' is not unique in table '{table}': {} of {} rows repeat a \
                 value{example}. A graph table needs one row per key, so point the join at a \
                 table or view with one row per '{key}'",
                grouped(*repeats),
                grouped(*rows)
            )
        }
    }
}

/// `'a'`, `'a' and 'b'`, or `'a', 'b' and 'c'` (`quoted_list`).
fn quoted_list(names: &[String]) -> String {
    let quoted: Vec<String> = names.iter().map(|n| format!("'{n}'")).collect();
    match quoted.split_last() {
        None => String::new(),
        Some((last, [])) => last.clone(),
        Some((last, rest)) => format!("{} and {last}", rest.join(", ")),
    }
}

/// A join left out, with the joins nested in it and the dimensions that go
/// with it (`SkippedJoin.__str__`), `reason` a clause.
pub fn skipped_join_text(
    alias: &str,
    reason: &str,
    nested: &[String],
    dimensions: &[String],
) -> String {
    let mut lost = Vec::new();
    if !nested.is_empty() {
        let noun = if nested.len() == 1 { "join" } else { "joins" };
        lost.push(format!("the {noun} {} nested in it", quoted_list(nested)));
    }
    if !dimensions.is_empty() {
        let noun = if dimensions.len() == 1 {
            "dimension"
        } else {
            "dimensions"
        };
        lost.push(format!("the {noun} {}", quoted_list(dimensions)));
    }
    let with_it = if lost.is_empty() {
        String::new()
    } else {
        format!(", and with it {},", lost.join(" and "))
    };
    format!("Failed to add join '{alias}'{with_it} since {reason}")
}

/// A name as a predictive query writes it, in backticks unless a bare name
/// (`_pql_name`).
fn pql_name(name: &str) -> String {
    let mut chars = name.chars();
    let bare = chars
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_alphanumeric() || c == '_');
    if bare {
        name.to_owned()
    } else {
        format!("`{name}`")
    }
}

/// A measure's note (`MeasureNote.__str__`).
pub fn measure_note_text(note: &MeasureNote) -> String {
    let column = note.column.as_deref().map_or("*".to_owned(), pql_name);
    format!(
        "'{}' is {}({}.{column}, 0, 30, days) over the next 30 days",
        note.measure,
        note.function,
        pql_name(&note.table)
    )
}

/// A time column's note (`TimeColumnNote.__str__`).
pub fn time_note_text(note: &TimeColumnNote) -> String {
    let mut reasons = Vec::new();
    if note.window_order {
        reasons.push("a window measure of the view orders by it".to_owned());
    }
    if !note.instead_of.is_empty() {
        let verbs = if note.instead_of.len() == 1 {
            "truncates or casts"
        } else {
            "truncate or cast"
        };
        reasons.push(format!(
            "{} {verbs} it, and it says when a row happened more precisely",
            quoted_list(&note.instead_of)
        ));
    }
    let (added, renamed) = match &note.source_column {
        Some(source) => (
            String::new(),
            format!(
                ". It is the source column '{source}', added under a name of its own because a \
                 dimension has its name"
            ),
        ),
        None if note.added => (", and added it as a column".to_owned(), String::new()),
        None => (String::new(), String::new()),
    };
    format!(
        "Took '{}' as the time column of table '{}'{added}, since {}{renamed}",
        note.column,
        note.table,
        reasons.join(" and ")
    )
}

fn dtype_name(dtype: Dtype) -> String {
    serde_json::to_value(dtype)
        .ok()
        .and_then(|v| v.as_str().map(str::to_owned))
        .unwrap_or_default()
}

/// An edge as the old client prints it: `Edge(src_table='a', fkey='b',
/// dst_table='c')`.
fn edge_text(edge: &Edge) -> String {
    format!(
        "Edge(src_table={}, fkey={}, dst_table={})",
        text_repr(&edge.source),
        text_repr(&edge.key),
        text_repr(&edge.target)
    )
}

/// Why an edge cannot be used (`_edge_error`).
pub fn edge_reason_text(edge: &Edge, reason: &EdgeReason) -> String {
    let shown = edge_text(edge);
    match reason {
        EdgeReason::NoColumn => format!(
            "Edge {shown} is invalid since table '{}' no longer has a column '{}'. Remove the \
             link with `unlink()` before removing the column.",
            edge.source, edge.key
        ),
        EdgeReason::NoPrimaryKey => format!(
            "Edge {shown} is invalid since table '{}' does not have a primary key. Add either a \
             primary key or remove the link before proceeding.",
            edge.target
        ),
        EdgeReason::KeyIsPrimary => format!(
            "Cannot treat the primary key of table '{}' as a foreign key. Remove either the \
             primary key or the link before proceeding.",
            edge.source
        ),
        EdgeReason::NotNumberOrString { dtype } => format!(
            "{shown} is invalid as foreign key must be a number or string (got '{}')",
            dtype_name(*dtype)
        ),
        EdgeReason::Incompatible {
            fkey_dtype,
            pkey,
            pkey_dtype,
        } => format!(
            "{shown} is invalid as foreign key '{}' and primary key '{pkey}' have incompatible \
             data types (got foreign key data type '{}' and primary key data type '{}')",
            edge.key,
            dtype_name(*fkey_dtype),
            dtype_name(*pkey_dtype)
        ),
    }
}

/// An edge dropped for its reason (`_drop_invalid_edges`).
pub fn dropped_edge_text(edge: &Edge, reason: &EdgeReason) -> String {
    format!(
        "Failed to add the relationship between '{}' and '{}' since {}",
        edge.source,
        edge.target,
        edge_reason_text(edge, reason)
    )
}

/// The old client's text of a link that failed, the engine's `ValueError`.
fn link_failure_text(failure: &LinkFailure) -> String {
    match failure {
        LinkFailure::NoSourceColumn { key } => {
            format!("Column '{key}' does not exist in the underlying source table")
        }
        LinkFailure::UnsupportedType { table, key } => format!(
            "Encountered unsupported data type for column '{key}' in table '{table}'. Please \
             either change the column's data type or remove the column from this table."
        ),
        LinkFailure::KeyType { key, dtype } => format!(
            "Column '{key}' received an incompatible semantic type (got dtype='{}' and \
             stype='ID')",
            dtype_name(*dtype)
        ),
        LinkFailure::EdgeExists(edge) => format!("{} already exists in the graph", edge_text(edge)),
        LinkFailure::ForeignKeyType { key, table, dtype } => format!(
            "Cannot use '{key}' in source table '{table}' as a foreign key due to its \
             incompatible data type. Foreign keys must have data type 'int', 'float' or \
             'string' (got '{}')",
            dtype_name(*dtype)
        ),
    }
}

/// Why a join is left out, as a clause of [`skipped_join_text`]; a key's
/// repeating value, if one was read, for its count's reason.
pub fn join_refusal_text(refusal: &JoinRefusal, example: Option<&str>) -> String {
    match refusal {
        JoinRefusal::AlreadyKeyed { table, key } => {
            format!("table '{table}' already uses '{key}' as its primary key")
        }
        JoinRefusal::Key { table, key, reason } => key_reason_text(reason, table, key, example),
        JoinRefusal::NotLinked(failure) => {
            format!(
                "its keys could not be linked: {}",
                link_failure_text(failure)
            )
        }
    }
}

/// What an added join says. `failure` names how counting the keys failed.
pub fn join_note_text(note: &JoinNote, join: &str, failure: &str) -> String {
    match note {
        JoinNote::KeptUnchecked { key } => format!(
            "Kept join '{join}' with its key '{key}' unchecked, since counting the keys failed: \
             {failure}"
        ),
        JoinNote::ReplacedDerived { key, table } => format!(
            "Replaced the derived column '{key}' of table '{table}' with its physical source \
             column since join '{join}' references it as a key"
        ),
    }
}

/// A relationship left out (`RE/graph.py:2224-2252`).
pub fn relationship_refusal_text(name: &str, refusal: &RelationshipRefusal) -> String {
    let reason = match refusal {
        RelationshipRefusal::NotPrimaryKey { key, table } => {
            format!("the referenced key '{key}' of table '{table}' is not a primary key")
        }
        RelationshipRefusal::IsPrimaryKey { key, table } => {
            format!("the referencing key '{key}' of table '{table}' is a primary key")
        }
        RelationshipRefusal::NoTable(table) => format!(
            "its keys could not be linked: {}",
            text_repr(&format!("Table '{table}' not found in graph"))
        ),
        RelationshipRefusal::NotLinked(failure) => {
            format!(
                "its keys could not be linked: {}",
                link_failure_text(failure)
            )
        }
    };
    format!("Failed to add relationship '{name}' since {reason}")
}

/// A section the conversion leaves out, as it says so.
pub fn ignored_text(ignored: Ignored) -> String {
    match ignored {
        Ignored::Materialization => {
            "Ignored the 'materialization' section since it cannot be represented in a graph".into()
        }
        Ignored::Fields => sdm_data::view_expr::metric::FIELDS_IGNORED.into(),
    }
}

/// A source that is no plain table name, which refuses the view
/// (`DB/metric_view.py:784-791`).
pub fn unsupported_source_text(spec: &ViewSpec, source: NodeId) -> Result<String, TooLarge> {
    Ok(format!(
        "Unsupported metric view source {}. Only plain table names are supported as the source \
         of a metric view when converting it into a graph. Create a regular view with this query \
         and point the metric view's source at it",
        repr(&spec.document, source)?
    ))
}

/// A join entry the conversion cannot read (`:828-839`).
pub fn join_message_text(spec: &ViewSpec, message: &JoinMessage) -> Result<String, TooLarge> {
    Ok(match message {
        JoinMessage::NotAMapping(cfg) => format!(
            "Failed to add join {} since it is not a mapping",
            repr(&spec.document, *cfg)?
        ),
        JoinMessage::Nameless(cfg) => format!(
            "Failed to add join {} since it misses a name",
            repr(&spec.document, *cfg)?
        ),
    })
}

/// Why a join is skipped, as a clause (`:849-889`).
pub fn skip_reason_text(
    spec: &ViewSpec,
    alias: &str,
    reason: &SkipReason<JoinJudgement>,
) -> Result<String, TooLarge> {
    Ok(match reason {
        SkipReason::NameInUse => format!("the name '{alias}' is already in use"),
        SkipReason::Judged(JoinJudgement::NotPlainSource) => {
            "only plain table names are supported as join sources. Create a regular view with its \
             query and join that instead"
                .into()
        }
        SkipReason::Cardinality(node) => {
            format!(
                "of its unsupported cardinality '{}'",
                text(&spec.document, *node)?
            )
        }
        SkipReason::UsingSeveral => {
            "it joins on more than one column, which a graph link cannot express".into()
        }
        SkipReason::Judged(JoinJudgement::NotSingleEquality) => {
            "only single-column equality join conditions are supported".into()
        }
    })
}

/// A dimension that is no mapping (`:959-963`).
pub fn dimension_not_a_mapping(spec: &ViewSpec, cfg: NodeId) -> Result<String, TooLarge> {
    Ok(format!(
        "Failed to add dimension {} since it is not a mapping",
        repr(&spec.document, cfg)?
    ))
}

/// A dimension that misses its name or its expression (`:991-996`).
pub fn dimension_misses_text(spec: &ViewSpec, cfg: NodeId) -> Result<String, TooLarge> {
    Ok(format!(
        "Failed to add dimension {} since it misses a name or expression",
        repr(&spec.document, cfg)?
    ))
}
