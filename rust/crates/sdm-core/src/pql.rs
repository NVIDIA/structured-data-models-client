// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The PQL adapter: the one module that names a PQL implementation.
//!
//! Behind it is the interim parser, `sdm-pql-interim`, until the PQL team's
//! crate takes its place. Whatever sits behind it answers in the
//! validated-query format ([`sdm_contract::pql`]), and the model capability
//! checks ([`crate::profiles`]) read that format, so nothing past this module
//! changes when the implementation does.
//!
//! A query is validated as the reference's `predictive_query.validate`
//! validates it: parsed; refused if the model cannot answer a target with a
//! time range, which no schema changes; resolved against the schema, with the
//! two filter capabilities resolution consults; and refused for each
//! capability the resolved query needs and the model lacks.

use std::collections::BTreeSet;

use sdm_contract::ops::{
    ColumnRef, Dtype, PqlDiagnostic, QuerySchema, QueryValidation, Severity, Span, Stype,
};
use sdm_contract::pql::{
    Aggregation, ConditionOp, EntityClause, LogicalOp, NodeKind, NodeType, PqlLiteral, PqlNode,
    ProblemType, QueryMode, TimeUnit, ValidatedQuery, Window,
};
use sdm_contract::{CoreError, Detail, ErrorCode, ErrorKind, HostFacts, Scalar, Unit};
use sdm_pql_interim as interim;
use sdm_pql_interim::ast::{self, ConstValue, Kind, NodeRef};
use sdm_pql_interim::literal::Literal;
use sdm_pql_interim::types as t;

use crate::profiles::{
    Mode, Profile, QueryFeatures, check_capabilities, check_parsed_capabilities,
};

/// A query's validation, with the validated query when it was accepted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Validation {
    /// Errors first, then warnings, each in the order PQL gave them.
    pub diagnostics: Vec<PqlDiagnostic>,
    /// The query, when it has no errors.
    pub query: Option<ValidatedQuery>,
}

impl Validation {
    fn refused(diagnostics: Vec<PqlDiagnostic>) -> Self {
        Validation {
            diagnostics,
            query: None,
        }
    }

    /// The answer `validate_query` gives.
    pub fn to_result(&self) -> QueryValidation {
        QueryValidation {
            accepted: self.query.is_some(),
            diagnostics: self.diagnostics.clone(),
        }
    }
}

/// Validates `query` against `schema` for `profile`.
pub fn validate(
    query: &str,
    schema: &QuerySchema,
    profile: &Profile,
    facts: &HostFacts,
) -> Result<QueryValidation, CoreError> {
    resolve(query, schema, profile, facts).map(|v| v.to_result())
}

/// The stack a validation runs on, as deep as the deepest query the parser
/// takes needs in a debug build; reserved, not committed, so only what a
/// query touches is used.
const STACK_BYTES: usize = 256 << 20;

/// Validates `query` against `schema` for `profile`, and gives the
/// validated query when it is accepted.
///
/// Parsing, resolving and reading the tree recurse as deep as the query
/// nests, so they run on a stack of their own: a caller on any thread,
/// however small its stack, is answered.
pub fn resolve(
    query: &str,
    schema: &QuerySchema,
    profile: &Profile,
    facts: &HostFacts,
) -> Result<Validation, CoreError> {
    stacker::grow(STACK_BYTES, || resolve_here(query, schema, profile, facts))
}

fn resolve_here(
    query: &str,
    schema: &QuerySchema,
    profile: &Profile,
    facts: &HostFacts,
) -> Result<Validation, CoreError> {
    let reader = interim::dates::Reader {
        year: clock_year(&facts.now)?,
    };
    let mode = match profile.mode {
        Mode::Training => interim::Mode::Training,
        Mode::Inference => interim::Mode::Inference,
    };
    let parsed = match interim::parse(query, mode, reader).map_err(unsupported)? {
        Ok(parsed) => parsed,
        Err(response) => return Ok(Validation::refused(diagnostics(&response))),
    };
    let refused = check_parsed_capabilities(parsed.target_has_time_range(), profile);
    if !refused.is_empty() {
        return Ok(Validation::refused(refused));
    }
    let filters = interim::Filters {
        key_columns: profile.capabilities.key_columns_in_filters,
        timestamp_columns: profile.capabilities.timestamp_columns_in_filters,
    };
    let validated = match interim::resolve(parsed, &schema_definition(schema), mode, filters) {
        interim::Resolved::Accepted(validated) => validated,
        interim::Resolved::Refused(response) => {
            return Ok(Validation::refused(diagnostics(&response)));
        }
    };
    let query = validated_query(&validated)?;
    let mut found = diagnostics(&validated.validation_response);
    let refusals = check_capabilities(&features(&query, schema), profile);
    if refusals.is_empty() {
        return Ok(Validation {
            diagnostics: found,
            query: Some(query),
        });
    }
    let warnings = found.split_off(found.iter().take_while(|d| is_error(d)).count());
    found.extend(refusals);
    found.extend(warnings);
    Ok(Validation::refused(found))
}

fn is_error(diagnostic: &PqlDiagnostic) -> bool {
    diagnostic.severity == Severity::Error
}

fn unsupported(interrupted: interim::Interrupted) -> CoreError {
    CoreError::new(
        ErrorKind::Unsupported,
        ErrorCode::UnsupportedRequest,
        format!(
            "{}; the interim PQL parser cannot validate this query",
            interrupted.0
        ),
    )
}

fn internal(message: impl Into<String>) -> CoreError {
    CoreError::new(ErrorKind::Internal, ErrorCode::InternalError, message)
}

// --- Host facts --------------------------------------------------------------

/// The year of the host's clock, in UTC, which places a two-digit year in a
/// date literal. Counted with civil arithmetic, so any clock an int64 holds
/// has a year (decision E6).
fn clock_year(now: &Scalar) -> Result<i16, CoreError> {
    let Scalar::Timestamp { unit, value, .. } = now else {
        return Err(internal("the host's clock is not a ts scalar"));
    };
    let per_second: i64 = match unit {
        Unit::Ns => 1_000_000_000,
        Unit::Us => 1_000_000,
    };
    let days = value.div_euclid(per_second).div_euclid(86_400);
    let year = year_of_day(days);
    i16::try_from(year).map_err(|_| {
        CoreError::new(
            ErrorKind::InvalidRequest,
            ErrorCode::InvalidRequest,
            "the host's clock is past the years a date literal can be read in",
        )
        .with_field("field", Detail::text("facts.now"))
    })
}

/// The proleptic Gregorian year of a day counted from 1970-01-01 (Howard
/// Hinnant's `civil_from_days`).
fn year_of_day(days: i64) -> i64 {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    yoe + era * 400 + i64::from(month <= 2)
}

// --- The schema, into the interim parser's form ------------------------------

/// A name, where an empty one means none, as the reference reads a schema.
fn name(value: &Option<String>) -> Option<String> {
    value.clone().filter(|v| !v.is_empty())
}

fn schema_definition(schema: &QuerySchema) -> interim::SchemaDefinition {
    use interim::schema::{Column, ColumnKey, ColumnKeyGroup, TableDefinition};
    interim::SchemaDefinition {
        tables: schema
            .tables
            .iter()
            .map(|table| {
                let cols = table
                    .columns
                    .iter()
                    .map(|c| Column {
                        name: c.name.clone(),
                        stype: interim_stype(c.stype),
                        dtype: interim_dtype(c.dtype),
                    })
                    .collect();
                let definition = TableDefinition {
                    cols,
                    pkey: name(&table.primary_key),
                    time_col: name(&table.time_column),
                    end_time_col: name(&table.end_time_column),
                };
                (table.name.clone(), definition)
            })
            .collect(),
        col_groups: schema
            .links
            .iter()
            .map(|group| ColumnKeyGroup {
                columns: group
                    .iter()
                    .map(|c| ColumnKey::new(&c.table, &c.column))
                    .collect(),
            })
            .collect(),
    }
}

macro_rules! same_names {
    ($to:ident, $from:ident, $a:ty, $b:ty, [$($v:ident),+ $(,)?]) => {
        fn $to(value: $a) -> $b {
            match value {
                $(<$a>::$v => <$b>::$v,)+
            }
        }
        fn $from(value: $b) -> $a {
            match value {
                $(<$b>::$v => <$a>::$v,)+
            }
        }
    };
}

same_names!(
    interim_dtype,
    contract_dtype,
    Dtype,
    t::Dtype,
    [
        Bool,
        Int,
        Byte,
        Int16,
        Int32,
        Int64,
        Float,
        Float32,
        Float64,
        String,
        Binary,
        Date,
        Time,
        Timedelta,
        Floatlist,
        Intlist,
        Stringlist,
        Unsupported,
    ]
);

same_names!(
    interim_stype,
    contract_stype,
    Stype,
    t::Stype,
    [
        Numerical,
        Categorical,
        Multicategorical,
        Id,
        Text,
        Timestamp,
        Sequence,
        Image,
        Unsupported,
    ]
);

// --- Diagnostics -------------------------------------------------------------

fn span(span: Option<interim::Span>) -> Option<Span> {
    let span = span?;
    let at = |v: usize| u32::try_from(v).unwrap_or(u32::MAX);
    Some(Span {
        start_row: at(span.start_row),
        start_col: at(span.start_col),
        end_row: at(span.end_row),
        end_col: at(span.end_col),
    })
}

/// A response's errors, then its warnings.
fn diagnostics(response: &interim::ValidationResponse) -> Vec<PqlDiagnostic> {
    let errors = response.errors.iter().map(|e| PqlDiagnostic {
        code: e.code.to_owned(),
        message: e.message.clone(),
        severity: Severity::Error,
        span: span(e.span),
    });
    let warnings = response.warnings.iter().map(|w| PqlDiagnostic {
        code: w.code.to_owned(),
        message: w.message.clone(),
        severity: Severity::Warning,
        span: span(w.span),
    });
    errors.chain(warnings).collect()
}

// --- The validated query -----------------------------------------------------

fn validated_query(validated: &interim::ValidatedQuery) -> Result<ValidatedQuery, CoreError> {
    let q = &validated.query;
    let optional = |n: &Option<NodeRef>| n.as_ref().map(node).transpose();
    Ok(ValidatedQuery {
        entity: node(&q.entity_ast)?,
        target: node(&q.target_ast)?,
        assuming: optional(&q.whatif_ast)?,
        entity_ids: optional(&q.rfm_entity_ids)?,
        problem_type: q.problem_type.map(|p| match p {
            t::ProblemType::Rank => ProblemType::Rank,
            t::ProblemType::Classify => ProblemType::Classify,
            t::ProblemType::Forecast => ProblemType::Forecast,
        }),
        top_k: q.top_k.as_ref().map(ToString::to_string),
        num_forecasts: q
            .num_forecasts
            .to_i64()
            .and_then(|n| u32::try_from(n).ok())
            .ok_or_else(|| internal("a validated query forecasts past u32"))?,
        entity_clause: if q.for_each == interim::validator::model::FOR {
            EntityClause::For
        } else {
            EntityClause::ForEach
        },
        mode: if q.rfm_query {
            QueryMode::Inference
        } else {
            QueryMode::Training
        },
        evaluate: q.evaluate,
        explain: q.explain,
    })
}

fn column_ref(fqn: &str) -> Result<ColumnRef, CoreError> {
    let (table, column) = fqn
        .rsplit_once('.')
        .ok_or_else(|| internal(format!("{fqn} is not table.column")))?;
    Ok(ColumnRef {
        table: table.to_owned(),
        column: column.to_owned(),
    })
}

fn window(range: &interim::dor::DateOffsetRange) -> Result<Window, CoreError> {
    let narrow = |v: i64| i32::try_from(v).map_err(|_| internal("a window past int32"));
    Ok(Window {
        start: range.start.map(narrow).transpose()?,
        end: narrow(range.end)?,
        unit: match range.unit {
            t::TimeUnit::Seconds => TimeUnit::Seconds,
            t::TimeUnit::Minutes => TimeUnit::Minutes,
            t::TimeUnit::Hours => TimeUnit::Hours,
            t::TimeUnit::Days => TimeUnit::Days,
            t::TimeUnit::Weeks => TimeUnit::Weeks,
            t::TimeUnit::Months => TimeUnit::Months,
        },
    })
}

fn condition_op(op: t::Op) -> ConditionOp {
    match op {
        t::Op::Rel(t::RelOp::Eq) => ConditionOp::Eq,
        t::Op::Rel(t::RelOp::Neq) => ConditionOp::Ne,
        t::Op::Rel(t::RelOp::Lt) => ConditionOp::Lt,
        t::Op::Rel(t::RelOp::Leq) => ConditionOp::Le,
        t::Op::Rel(t::RelOp::Gt) => ConditionOp::Gt,
        t::Op::Rel(t::RelOp::Geq) => ConditionOp::Ge,
        t::Op::Member(t::MemberOp::In) => ConditionOp::In,
        t::Op::Member(t::MemberOp::IsIn) => ConditionOp::IsIn,
        t::Op::Str(t::StrOp::StartsWith) => ConditionOp::StartsWith,
        t::Op::Str(t::StrOp::EndsWith) => ConditionOp::EndsWith,
        t::Op::Str(t::StrOp::Contains) => ConditionOp::Contains,
        t::Op::Str(t::StrOp::NotContains) => ConditionOp::NotContains,
    }
}

fn aggregation(aggr: t::AggregationType) -> Aggregation {
    match aggr {
        t::AggregationType::Sum => Aggregation::Sum,
        t::AggregationType::Avg => Aggregation::Avg,
        t::AggregationType::Min => Aggregation::Min,
        t::AggregationType::Max => Aggregation::Max,
        t::AggregationType::Count => Aggregation::Count,
        t::AggregationType::CountDistinct => Aggregation::CountDistinct,
        t::AggregationType::First => Aggregation::First,
        t::AggregationType::Last => Aggregation::Last,
        t::AggregationType::ListDistinct => Aggregation::ListDistinct,
    }
}

fn boxed(n: &NodeRef) -> Result<Box<PqlNode>, CoreError> {
    node(n).map(Box::new)
}

/// A constant's literal: its text read as its type gives it.
fn literal(n: &NodeRef, value: &ConstValue) -> Result<PqlLiteral, CoreError> {
    let text = match value {
        ConstValue::List(items) => {
            return Ok(PqlLiteral::Array {
                items: items.iter().map(node).collect::<Result<_, _>>()?,
            });
        }
        ConstValue::Text(text) => text,
    };
    let read = ast::typed_value(n).map_err(|e| internal(format!("a validated constant: {e}")))?;
    Ok(match read {
        Literal::Null => PqlLiteral::Null,
        Literal::Bool(value) => PqlLiteral::Bool { value },
        Literal::Int(value) => PqlLiteral::Int {
            value: value.to_string(),
        },
        Literal::Float(_) => PqlLiteral::Float { text: text.clone() },
        Literal::Str(value) => PqlLiteral::String { value },
        Literal::Time(time) => PqlLiteral::Time {
            text: text.clone(),
            value: time.to_string(),
            offset: None,
        },
        Literal::List(_) => return Err(internal("a list constant written as text")),
    })
}

/// One node of the interim parser's tree, in the validated-query format.
fn node(n: &NodeRef) -> Result<PqlNode, CoreError> {
    let borrowed = n.borrow();
    let base = &borrowed.base;
    let kind = match &borrowed.kind {
        Kind::Column { fqn, .. } => NodeKind::Column {
            column: column_ref(fqn)?,
        },
        Kind::Constant { value, .. } => NodeKind::Constant {
            literal: literal(n, value)?,
        },
        Kind::Condition {
            target,
            op,
            value,
            input_op,
        } => NodeKind::Condition {
            target: boxed(target)?,
            op: condition_op(*op),
            value: boxed(value)?,
            from_like: input_op.is_some(),
        },
        Kind::Logical { left, op, right } => NodeKind::Logical {
            op: match op {
                t::BoolOp::And => LogicalOp::And,
                t::BoolOp::Or => LogicalOp::Or,
                t::BoolOp::Not => LogicalOp::Not,
            },
            left: boxed(left)?,
            right: right.as_ref().map(boxed).transpose()?,
        },
        Kind::Filter { target, condition } => NodeKind::Filter {
            target: boxed(target)?,
            condition: boxed(condition)?,
        },
        Kind::Aggregation {
            target,
            aggr,
            aggr_time_range,
            group_by,
            time_col,
        } => NodeKind::Aggregation {
            aggregation: aggregation(*aggr),
            target: boxed(target)?,
            window: aggr_time_range
                .as_ref()
                .map(|r| window(&r.borrow()))
                .transpose()?,
            group_by: group_by.as_deref().map(column_ref).transpose()?,
            time_column: time_col.as_deref().map(column_ref).transpose()?,
        },
        Kind::Join {
            rhs_target,
            lhs_key,
            rhs_key,
        } => NodeKind::Join {
            target: boxed(rhs_target)?,
            lhs_key: column_ref(lhs_key)?,
            rhs_key: column_ref(rhs_key)?,
        },
    };
    Ok(PqlNode {
        dtype: base.dtype.map(|d| match d {
            ast::NodeDtype::Plain(d) => NodeType::Value(contract_dtype(d)),
            ast::NodeDtype::Array(d) => NodeType::Array(contract_dtype(d)),
        }),
        stype: base.stype.map(contract_stype),
        range: base
            .date_offset_range
            .as_ref()
            .map(|r| window(&r.borrow()))
            .transpose()?,
        span: base.location.and_then(|l| span(interim::Span::at(l))),
        kind,
    })
}

// --- What a validated query needs of a model ---------------------------------

/// A node's children, in the order the reference's tree gives them; an
/// array's elements are not children.
fn children(node: &PqlNode) -> Vec<&PqlNode> {
    match &node.kind {
        NodeKind::Column { .. } | NodeKind::Constant { .. } => Vec::new(),
        NodeKind::Condition { target, value, .. } => vec![target, value],
        NodeKind::Logical { left, right, .. } => {
            std::iter::once(&**left).chain(right.as_deref()).collect()
        }
        NodeKind::Filter { target, condition } => vec![target, condition],
        NodeKind::Aggregation { target, .. } | NodeKind::Join { target, .. } => vec![target],
    }
}

fn walk<'a>(node: &'a PqlNode, visit: &mut impl FnMut(&'a PqlNode)) {
    visit(node);
    for child in children(node) {
        walk(child, visit);
    }
}

/// The earlier start and the later end of two spans.
fn merge(a: Span, b: Span) -> Span {
    let (start_row, start_col) = (a.start_row, a.start_col).min((b.start_row, b.start_col));
    let (end_row, end_col) = (a.end_row, a.end_col).max((b.end_row, b.end_col));
    Span {
        start_row,
        start_col,
        end_row,
        end_col,
    }
}

/// Where a node is: its own span merged with its children's.
fn location(node: &PqlNode) -> Option<Span> {
    children(node)
        .into_iter()
        .fold(node.span, |at, child| match (at, location(child)) {
            (Some(a), Some(b)) => Some(merge(a, b)),
            (a, b) => a.or(b),
        })
}

/// The node a join reaches, or the node itself.
fn through_join(node: &PqlNode) -> &PqlNode {
    match &node.kind {
        NodeKind::Join { target, .. } => target,
        _ => node,
    }
}

/// What `query` needs of a model, read off the validated query.
pub fn features(query: &ValidatedQuery, schema: &QuerySchema) -> QueryFeatures {
    let parts = [
        Some(&query.entity),
        Some(&query.target),
        query.assuming.as_ref(),
    ];
    let mut tables = BTreeSet::new();
    let mut follows_foreign_key = false;
    for part in parts.into_iter().flatten() {
        walk(part, &mut |n| match &n.kind {
            NodeKind::Column { column } => {
                tables.insert(column.table.clone());
            }
            NodeKind::Join { .. } => follows_foreign_key = true,
            _ => {}
        });
    }
    let entity_column = match &query.entity.kind {
        NodeKind::Filter { target, .. } => target,
        _ => &query.entity,
    };
    let entity_table = match &entity_column.kind {
        NodeKind::Column { column } => column.table.clone(),
        _ => String::new(),
    };
    let target = through_join(&query.target);
    QueryFeatures {
        tables,
        entity_table,
        follows_foreign_key,
        temporal_target: query.target.range.is_some(),
        derived_target: !matches!(target.kind, NodeKind::Column { .. }),
        entity_filter: matches!(query.entity.kind, NodeKind::Filter { .. }),
        counterfactual: query.assuming.is_some(),
        ranks_foreign_key: ranks_foreign_key(target, schema),
        num_forecasts: query.num_forecasts,
        target_span: location(&query.target),
    }
}

/// Whether `target` is a `LIST_DISTINCT` over a linked column that is not
/// its table's primary key.
fn ranks_foreign_key(target: &PqlNode, schema: &QuerySchema) -> bool {
    let NodeKind::Aggregation {
        aggregation: Aggregation::ListDistinct,
        target: aggregated,
        ..
    } = &target.kind
    else {
        return false;
    };
    let column = match &aggregated.kind {
        NodeKind::Column { column } => column,
        NodeKind::Filter { target, .. } => match &target.kind {
            NodeKind::Column { column } => column,
            _ => return false,
        },
        _ => return false,
    };
    let Some(table) = schema.tables.iter().find(|t| t.name == column.table) else {
        return false;
    };
    if name(&table.primary_key).as_deref() == Some(column.column.as_str()) {
        return false;
    }
    schema.links.iter().flatten().any(|c| c == column)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_clock_year_is_civil() {
        let days = |y: i64, m: i64, d: i64| {
            // Days from 1970-01-01, by counting forward from a known date.
            let y = if m <= 2 { y - 1 } else { y };
            let era = y.div_euclid(400);
            let yoe = y - era * 400;
            let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
            era * 146_097 + yoe * 365 + yoe / 4 - yoe / 100 + doy - 719_468
        };
        for (y, m, d) in [
            (1970, 1, 1),
            (1969, 12, 31),
            (2000, 2, 29),
            (2026, 10, 8),
            (9999, 12, 31),
            (1, 1, 1),
            (0, 3, 1),
        ] {
            assert_eq!(year_of_day(days(y, m, d)), y, "{y}-{m}-{d}");
        }
        let now = Scalar::Timestamp {
            unit: Unit::Ns,
            zone: Some("UTC".into()),
            value: i64::MIN,
        };
        assert_eq!(clock_year(&now).unwrap(), 1677);
        let far = Scalar::Timestamp {
            unit: Unit::Us,
            zone: Some("UTC".into()),
            value: i64::MAX,
        };
        assert!(clock_year(&far).is_err());
    }
}
