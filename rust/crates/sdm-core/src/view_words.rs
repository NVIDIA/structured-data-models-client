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
//! words alike.

use sdm_contract::ops::Dtype;
use sdm_data::view_expr::ExprRefusal;
use sdm_data::view_expr::metric::{
    DefaultKind, DefaultRefusal, FilterRefusal, MeasureReason, MeasureRefusal, Parameter,
    ParameterDefault, ParameterInput, Unread,
};
use sdm_schema::view::metric::Parameter as ViewParameter;
use sdm_schema::view::model::{
    Edge, EdgeReason, JoinNote, JoinRefusal, KeyReason, LinkFailure, MeasureNote,
    RelationshipRefusal, TimeColumnNote,
};
use sdm_schema::view::yaml::Value;

use crate::pql::{text_repr, value_repr as repr, value_text as text};

/// The text the old client's messages quote a view definition's value by:
/// Python's `repr` of it.
pub fn value_repr(value: &Value) -> String {
    repr(value)
}

/// Python's `str` of a view definition's value.
pub fn value_text(value: &Value) -> String {
    text(value)
}

/// A text as the old client quotes a piece of an expression or a name:
/// Python's `repr` of it. The data layer's sentences take it.
pub fn quote(text: &str) -> String {
    text_repr(text)
}

/// Why an expression is refused, in the old client's words
/// (`RE/view_expr.py`, old-behaviour rows 64 to 66), quoted as it quoted.
pub fn expr_reason(reason: &ExprRefusal) -> String {
    reason.words(&quote)
}

/// A parameter's default as the data layer reads it: its kind, and
/// Python's `str` of it.
pub fn parameter_default(value: &Value) -> ParameterDefault {
    let kind = match value {
        Value::Bool(flag) => DefaultKind::Bool(*flag),
        Value::Int(n) => DefaultKind::Int(*n),
        Value::Date(_) => DefaultKind::Date,
        Value::Timestamp(moment) => DefaultKind::DateTime {
            iso: moment.isoformat(),
        },
        _ => DefaultKind::Other,
    };
    ParameterDefault {
        kind,
        text: text(value),
    }
}

/// Each parameter of the view as the data layer reads it: its type if the
/// definition writes it as text, its default unless null.
pub fn parameter_inputs(parameters: &[ViewParameter]) -> Vec<ParameterInput> {
    parameters
        .iter()
        .map(|parameter| ParameterInput {
            name: parameter.name.clone(),
            data_type: parameter.data_type.as_str().map(str::to_owned),
            default: (parameter.default != Value::Null)
                .then(|| parameter_default(&parameter.default)),
        })
        .collect()
}

/// Why a default gives no literal of `data_type` (`_default_text`'s
/// refusals).
pub fn default_refusal_text(refusal: &DefaultRefusal, data_type: &str, default: &Value) -> String {
    match refusal {
        DefaultRefusal::NoLiteral => format!("its type '{data_type}' has no literal here"),
        DefaultRefusal::NotOfType => {
            format!("its default {} is not of type {data_type}", repr(default))
        }
    }
}

/// A parameter's text: its default's, or why it has no literal
/// (`_Parameter.text`). `default` is its default as the definition gives
/// it.
pub fn parameter_text(parameter: &Parameter, default: &Value) -> String {
    match &parameter.read {
        Ok(read) => read.text.clone(),
        Err(Unread::NoDefault) => "has no default".into(),
        Err(Unread::NoType) => "has no type".into(),
        Err(Unread::Refused { refusal, data_type }) => {
            default_refusal_text(refusal, data_type, default)
        }
    }
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
pub fn filter_message(filter: &Value, reason: &str, hatch: FilterHatch) -> String {
    let hatch = match hatch {
        FilterHatch::None => "",
        FilterHatch::Default => ", or give the parameter a default of its type",
        FilterHatch::Trust => {
            ", or pass trust_expressions=True to apply it if you trust whoever authored the view"
        }
    };
    format!(
        "Cannot apply the metric view filter {} since {reason}. The graph would otherwise read \
         rows the view leaves out. Create a regular view with this WHERE clause and point the \
         metric view's source at it{hatch}",
        repr(filter)
    )
}

/// The refusal of a filter the source table alone cannot apply
/// (`_check_filter`).
pub fn filter_refusal_text(filter: &Value, refusal: &FilterRefusal) -> String {
    let reason = match refusal {
        FilterRefusal::NotACondition => "it is not a SQL condition".to_owned(),
        FilterRefusal::Unsafe(reason) => expr_reason(reason),
        FilterRefusal::JoinedTable(alias) => format!("it references the joined table '{alias}'"),
    };
    filter_message(filter, &reason, FilterHatch::None)
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
/// repeating value if one was read.
pub fn key_reason_text(
    reason: &KeyReason,
    table: &str,
    key: &str,
    example: Option<&Value>,
) -> String {
    match reason {
        KeyReason::NoValue => format!("its key '{key}' holds no value in table '{table}'"),
        KeyReason::NotUnique { repeats, rows } => {
            let example = example.map_or(String::new(), |v| format!(", e.g. {}", repr(v)));
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
pub fn join_refusal_text(refusal: &JoinRefusal, example: Option<&Value>) -> String {
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
