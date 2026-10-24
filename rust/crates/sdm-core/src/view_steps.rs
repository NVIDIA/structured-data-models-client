// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The first steps of a metric view's conversion (T3.5, M2), each a pure
//! composition of the schema's order and the data layer's reading: the
//! source table's name, the parameters' defaults, and the joins kept or
//! skipped, in the old client's order (`DB/metric_view.py:513-909`). The
//! ordered routine of stage 2 (`view.rs`) asks them in turn, between its
//! reads.

use sdm_contract::view::{
    DefaultRefusal, DefaultTexts, Parameter, ParameterInput, TypeClass, ViewTextAnswer, ViewTextAsk,
};
use sdm_contract::{CoreError, Sentence};
use sdm_schema::view::joins::{self, JoinWalk};
use sdm_schema::view::spec::{Source, TableItem, TableSpec, ViewRefusal, ViewSpec};
use sdm_schema::view::yaml::{Document, NodeId, Value};

use crate::pql::conversions::{
    Unconverted, date_text, datetime_text, decimal_text, int_text, strip,
};
use crate::pql::{date_iso, datetime_iso, value_text};

/// The data layer's answers to the routine's questions of text
/// (`DataLayer::view_text`), in the run provider's dialect.
pub type Ask<'a> = &'a dyn Fn(ViewTextAsk) -> Result<ViewTextAnswer, CoreError>;

/// An answer of another kind than the question's, which no data layer
/// gives: the core's fault, refused as an internal error.
fn mismatched(asked: &str) -> CoreError {
    CoreError::new(
        Sentence::Internal,
        "the data layer answered a question of a view's text with another kind of answer",
    )
    .with_field("asked", sdm_contract::Detail::text(asked))
}

/// A name's parts, or none (`ViewName`, `TableReference`).
pub fn names(ask: Ask<'_>, question: ViewTextAsk) -> Result<Option<Vec<String>>, CoreError> {
    match ask(question)? {
        ViewTextAnswer::Names(names) => Ok(names),
        _ => Err(mismatched("names")),
    }
}

/// A type's class (`TypeClass`).
pub fn type_class(ask: Ask<'_>, data_type: &str) -> Result<TypeClass, CoreError> {
    match ask(ViewTextAsk::TypeClass {
        data_type: data_type.to_owned(),
    })? {
        ViewTextAnswer::TypeClass(class) => Ok(class),
        _ => Err(mismatched("type_class")),
    }
}

/// Why the data layer refuses a join's source or condition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JoinJudgement {
    /// Its source is no plain one-to-three-part table name.
    NotPlainSource,
    /// Its condition is no single-column equality.
    NotSingleEquality,
}

/// The source table's name parts, or why there are none: the node of a
/// source that is no plain table name, which refuses the view
/// (`:784-791`), or `None` for a definition with no metric view's source
/// (a semantic view's).
pub fn fact_source(
    spec: &ViewSpec,
    ask: Ask<'_>,
) -> Result<Result<Vec<String>, Option<NodeId>>, CoreError> {
    let Some(TableItem::Table(TableSpec {
        source: Source::Name(Some(node)),
        ..
    })) = spec.tables.first()
    else {
        return Ok(Err(None));
    };
    let Some(name) = spec.text(*node) else {
        return Ok(Err(Some(*node)));
    };
    Ok(names(
        ask,
        ViewTextAsk::TableReference {
            name: name.to_owned(),
        },
    )?
    .ok_or(Some(*node)))
}

/// The text a parameter's default casts from, checked against its type so
/// that nothing but the value reaches the SQL (`_default_text`,
/// `DB/metric_view.py:513-566`): the data layer tells the type's class,
/// and the default is converted as the old client's Python converted it.
pub fn default_text(
    class: TypeClass,
    document: &Document,
    default: NodeId,
) -> Result<String, DefaultRefusal> {
    let value = document.value(default);
    // `str(default)`, which a text too large to write refuses.
    let text = || value_text(document, default).map_err(|_| DefaultRefusal::TooLong);
    let converted = |answer: Result<String, Unconverted>| {
        answer.map_err(|unconverted| match unconverted {
            Unconverted::Invalid => DefaultRefusal::NotOfType,
            Unconverted::TooManyDigits(digits) => DefaultRefusal::TooManyDigits(digits),
            Unconverted::NotAnInteger => DefaultRefusal::NotAnInteger,
            Unconverted::TooLong => DefaultRefusal::TooLong,
        })
    };
    match (class, value) {
        (TypeClass::Boolean | TypeClass::String, Value::Bool(flag)) => Ok(flag.to_string()),
        (_, Value::Bool(_)) => Err(DefaultRefusal::NotOfType),
        (TypeClass::Boolean, _) => {
            let lower = strip(&text()?).to_lowercase();
            if lower == "true" || lower == "false" {
                Ok(lower)
            } else {
                Err(DefaultRefusal::NotOfType)
            }
        }
        (TypeClass::Integer, Value::Int(n)) => Ok(n.text().to_owned()),
        (TypeClass::Integer, _) => converted(int_text(&text()?)),
        (TypeClass::Decimal, _) => converted(decimal_text(&text()?)),
        (TypeClass::String, Value::Date(date)) => Ok(date_iso(date)),
        (TypeClass::String, Value::Timestamp(moment)) => Ok(datetime_iso(moment, 'T')),
        (TypeClass::String, _) => text(),
        (TypeClass::Date, Value::Date(date)) => Ok(date_iso(date)),
        (TypeClass::Date, _) => converted(date_text(strip(&text()?))),
        (TypeClass::Timestamp, Value::Timestamp(_)) => text(),
        (TypeClass::Timestamp, Value::Date(date)) => Ok(date_iso(date)),
        (TypeClass::Timestamp, _) => converted(datetime_text(strip(&text()?))),
        (TypeClass::NoLiteral, _) => Err(DefaultRefusal::NoLiteral),
    }
}

/// Each parameter of the view as the data layer reads it: its type if the
/// definition writes it as text, and its default, unless null, converted
/// for each type it may be read as, the definition's and any `described`
/// for its name (D3 of T3.27).
pub fn parameter_inputs(
    spec: &ViewSpec,
    described: &[(String, String)],
    ask: Ask<'_>,
) -> Result<Vec<ParameterInput<DefaultTexts>>, CoreError> {
    let mut inputs = Vec::new();
    for parameter in &spec.parameters {
        let data_type = parameter
            .data_type
            .and_then(|n| spec.text(n))
            .map(str::to_owned);
        let default = match parameter.default {
            None => None,
            Some(node) => {
                let folded = parameter.name.to_lowercase();
                let mut types: Vec<String> = data_type.iter().cloned().collect();
                for (name, written) in described {
                    if name.to_lowercase() == folded && !types.contains(written) {
                        types.push(written.clone());
                    }
                }
                let mut texts = Vec::new();
                for written in types {
                    let class = type_class(ask, &written)?;
                    texts.push((written, default_text(class, &spec.document, node)));
                }
                Some(DefaultTexts(texts))
            }
        };
        inputs.push(ParameterInput {
            name: parameter.name.clone(),
            data_type,
            default,
        });
    }
    Ok(inputs)
}

/// The `parameters` section read (`_read_parameters`), by folded name,
/// the types the warehouse describes, `described`, winning.
pub fn parameters(
    spec: &ViewSpec,
    described: &[(String, String)],
    ask: Ask<'_>,
) -> Result<Vec<(String, Parameter)>, CoreError> {
    let inputs = parameter_inputs(spec, described, ask)?;
    match ask(ViewTextAsk::Parameters {
        inputs,
        described: described.to_vec(),
    })? {
        ViewTextAnswer::Parameters(read) => Ok(read),
        _ => Err(mismatched("parameters")),
    }
}

/// The joins kept and skipped (`_add_joins`): a source read as a plain
/// table name, a condition with its comments dropped read as a
/// single-column equality, each refusal skipping the join.
pub fn walk_joins(
    spec: &mut ViewSpec,
    ask: Ask<'_>,
) -> Result<Result<JoinWalk<JoinJudgement>, ViewRefusal>, CoreError> {
    // A question the data layer fails stops the walk, and is the run's
    // failure.
    let failed: std::cell::RefCell<Option<CoreError>> = std::cell::RefCell::new(None);
    let keep = |answer: Result<bool, CoreError>| match answer {
        Ok(read) => read,
        Err(error) => {
            failed.borrow_mut().get_or_insert(error);
            false
        }
    };
    let walked = joins::walk(
        spec,
        |spec, source| {
            let read = source.and_then(|node| spec.text(node)).map(|name| {
                names(
                    ask,
                    ViewTextAsk::TableReference {
                        name: name.to_owned(),
                    },
                )
                .map(|parts| parts.is_some())
            });
            if keep(read.unwrap_or(Ok(false))) {
                Ok(())
            } else {
                Err(JoinJudgement::NotPlainSource)
            }
        },
        |spec, on| {
            let read = on.and_then(|node| spec.text(node)).map(|text| {
                let stripped = match ask(ViewTextAsk::StripComments {
                    expr: text.to_owned(),
                })? {
                    ViewTextAnswer::Text(Ok(stripped)) => stripped,
                    ViewTextAnswer::Text(Err(_)) => return Ok(false),
                    _ => return Err(mismatched("strip_comments")),
                };
                match ask(ViewTextAsk::JoinCondition {
                    condition: stripped,
                })? {
                    ViewTextAnswer::Condition(condition) => Ok(condition.is_some()),
                    _ => Err(mismatched("join_condition")),
                }
            });
            if keep(read.unwrap_or(Ok(false))) {
                Ok(())
            } else {
                Err(JoinJudgement::NotSingleEquality)
            }
        },
    );
    if let Some(error) = failed.into_inner() {
        return Err(error);
    }
    Ok(walked)
}
