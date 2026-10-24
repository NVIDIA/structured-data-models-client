// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The first steps of a metric view's conversion (T3.5, M2), each a pure
//! composition of the schema's order and the data layer's reading: the
//! source table's name, the parameters' defaults, and the joins kept or
//! skipped, in the old client's order (`DB/metric_view.py:513-909`). The
//! ordered routine of stage 2 (`view.rs`) asks them in turn, between its
//! reads.

use sdm_data::view_expr::metric::{
    DefaultRefusal, Parameter, ParameterInput, TypeClass, join_condition, read_parameters,
    table_reference, type_class,
};
use sdm_data::view_expr::strip_comments;
use sdm_schema::view::joins::{self, JoinWalk};
use sdm_schema::view::spec::{Source, TableItem, TableSpec, ViewRefusal, ViewSpec};
use sdm_schema::view::yaml::{Document, NodeId, Value};

use crate::pql::conversions::{
    Unconverted, date_text, datetime_text, decimal_text, int_text, strip,
};
use crate::pql::value_text;

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
pub fn fact_source(spec: &ViewSpec) -> Result<Vec<String>, Option<NodeId>> {
    let Some(TableItem::Table(TableSpec {
        source: Source::Name(Some(node)),
        ..
    })) = spec.tables.first()
    else {
        return Err(None);
    };
    spec.text(*node)
        .and_then(table_reference)
        .ok_or(Some(*node))
}

/// The text a parameter's default casts from, checked against its type so
/// that nothing but the value reaches the SQL (`_default_text`,
/// `DB/metric_view.py:513-566`): the data layer tells the type's class,
/// and the default is converted as the old client's Python converted it.
pub fn default_text(
    data_type: &str,
    document: &Document,
    default: NodeId,
) -> Result<String, DefaultRefusal> {
    let class = type_class(data_type);
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
        (TypeClass::String, Value::Date(date)) => Ok(date.isoformat()),
        (TypeClass::String, Value::Timestamp(moment)) => Ok(moment.isoformat()),
        (TypeClass::String, _) => text(),
        (TypeClass::Date, Value::Date(date)) => Ok(date.isoformat()),
        (TypeClass::Date, _) => converted(date_text(strip(&text()?))),
        (TypeClass::Timestamp, Value::Timestamp(_)) => text(),
        (TypeClass::Timestamp, Value::Date(date)) => Ok(date.isoformat()),
        (TypeClass::Timestamp, _) => converted(datetime_text(strip(&text()?))),
        (TypeClass::NoLiteral, _) => Err(DefaultRefusal::NoLiteral),
    }
}

/// Each parameter of the view as the data layer reads it: its type if the
/// definition writes it as text, its default unless null.
pub fn parameter_inputs(spec: &ViewSpec) -> Vec<ParameterInput<NodeId>> {
    spec.parameters
        .iter()
        .map(|parameter| ParameterInput {
            name: parameter.name.clone(),
            data_type: parameter
                .data_type
                .and_then(|n| spec.text(n))
                .map(str::to_owned),
            default: parameter.default,
        })
        .collect()
}

/// The `parameters` section read (`_read_parameters`), by folded name,
/// the types Databricks describes, `described`, winning.
pub fn parameters(spec: &ViewSpec, described: &[(String, String)]) -> Vec<(String, Parameter)> {
    read_parameters(&parameter_inputs(spec), described, |data_type, default| {
        default_text(data_type, &spec.document, *default)
    })
}

/// The joins kept and skipped (`_add_joins`): a source read as a plain
/// table name, a condition with its comments dropped read as a
/// single-column equality, each refusal skipping the join.
pub fn walk_joins(spec: &mut ViewSpec) -> Result<JoinWalk<JoinJudgement>, ViewRefusal> {
    joins::walk(
        spec,
        |spec, source| {
            source
                .and_then(|node| spec.text(node))
                .and_then(table_reference)
                .map(|_| ())
                .ok_or(JoinJudgement::NotPlainSource)
        },
        |spec, on| {
            on.and_then(|node| spec.text(node))
                .and_then(|text| strip_comments(text).ok())
                .and_then(|text| join_condition(&text))
                .map(|_| ())
                .ok_or(JoinJudgement::NotSingleEquality)
        },
    )
}
