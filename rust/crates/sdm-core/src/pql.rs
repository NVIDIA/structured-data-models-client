// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The PQL adapter: the one module that names the PQL parser.
//!
//! Behind it is `sdm-pql-interim`, this client's own parser (decision E18;
//! its name awaits a rename the gate can judge, E69). It
//! answers in the validated-query format ([`sdm_contract::pql`]), and the
//! model capability checks ([`crate::profiles`]) read that format, so
//! nothing past this module depends on the parser's own types.
//!
//! Date literals are read by the host's facts: two-digit years around the
//! current year in the host's local zone, and D4's day-first warning by the
//! platform's year padding (`date-rules.md`, "The PQL literal rules"). A
//! day-first literal whose format can be named gives a
//! [`DATE_LITERAL_DAY_FIRST`] warning, which a `validate_query` run gives on
//! its warnings channel, once per such literal.
//!
//! A query is validated as the reference's `predictive_query.validate`
//! validates it: parsed; refused if the model cannot answer a target with a
//! time range, which no schema changes; resolved against the schema, with the
//! two filter capabilities resolution consults; and refused for each
//! capability the resolved query needs and the model lacks.

use std::collections::BTreeSet;

use sdm_contract::facts::LocalZone;
use sdm_contract::ops::{
    ColumnRef, Dtype, PqlDiagnostic, QuerySchema, QueryValidation, Severity, Span, Stype,
};
use sdm_contract::pql::{
    Aggregation, ConditionOp, EntityClause, LogicalOp, NodeKind, NodeType, PqlLiteral, PqlNode,
    ProblemType, QueryMode, TimeUnit, ValidatedQuery, Window,
};
use sdm_contract::{CoreError, CoreWarning, Detail, ErrorCode, ErrorKind, HostFacts, Scalar, Unit};
use sdm_engine::time::{Instant, Zones};
use sdm_pql_interim as parser;
use sdm_pql_interim::ast::{self, ConstValue, Kind, NodeRef};
use sdm_pql_interim::literal::Literal;
use sdm_pql_interim::types as t;

use crate::profiles::{
    Mode, Profile, QueryFeatures, check_capabilities, check_parsed_capabilities,
};

/// D4's warning: a date literal read day first, whose format, as `strftime`
/// writes it, is the `format` field, and whose place in the query is
/// `start_row`, `start_col`, `end_row` and `end_col`.
pub const DATE_LITERAL_DAY_FIRST: &str = "DATE_LITERAL_DAY_FIRST";

/// Every warning the adapter gives, with its variants (none).
pub const WARNINGS: &[(&str, &[&str])] = &[(DATE_LITERAL_DAY_FIRST, &[])];

/// A query's validation, with the validated query when it was accepted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Validation {
    /// Errors first, then warnings, each in the order PQL gave them.
    pub diagnostics: Vec<PqlDiagnostic>,
    /// The query, when it has no errors.
    pub query: Option<ValidatedQuery>,
    /// The query, when it has no errors, written back as the reference's
    /// `ValidatedPredictiveQuery.to_string()` writes it.
    pub text: Option<String>,
    /// D4's warning for each day-first date literal read, in the order
    /// read, accepted or not.
    pub warnings: Vec<CoreWarning>,
}

impl Validation {
    fn refused(diagnostics: Vec<PqlDiagnostic>, warnings: Vec<CoreWarning>) -> Self {
        Validation {
            diagnostics,
            query: None,
            text: None,
            warnings,
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
///
/// Everything runs on the adapter's stack ([`on_adapter_stack`]): parsing,
/// resolving, building the answer and dropping the trees, so a query as
/// deep as the parser takes is answered and freed on a caller's thread of
/// any size.
pub fn validate(
    query: &str,
    schema: &QuerySchema,
    profile: &Profile,
    facts: &HostFacts,
    zones: &dyn Zones,
) -> Result<(QueryValidation, Vec<CoreWarning>), CoreError> {
    on_adapter_stack(|| {
        resolve_here(query, schema, profile, facts, zones).map(|v| (v.to_result(), v.warnings))
    })
}

/// The stack a validation runs on, as deep as the deepest query the parser
/// takes needs in a debug build; reserved, not committed, so only what a
/// query touches is used.
const STACK_BYTES: usize = 256 << 20;

/// Runs `f` on the stack the adapter validates on: grown once per adapter
/// call, here and nowhere else (the parser grows none of its own),
/// so parsing, resolving, reading the tree and dropping it all fit in it.
pub fn on_adapter_stack<R>(f: impl FnOnce() -> R) -> R {
    stacker::grow(STACK_BYTES, f)
}

/// Validates `query` against `schema` for `profile`, and gives the
/// validated query when it is accepted.
///
/// Parsing and resolving run on the adapter's stack ([`on_adapter_stack`]);
/// the validated query comes back to the caller, which drops it where it
/// drops it: a caller holding a deep one drops it on the adapter's stack
/// too, as [`validate`] does.
pub fn resolve(
    query: &str,
    schema: &QuerySchema,
    profile: &Profile,
    facts: &HostFacts,
    zones: &dyn Zones,
) -> Result<Validation, CoreError> {
    on_adapter_stack(|| resolve_here(query, schema, profile, facts, zones))
}

fn resolve_here(
    query: &str,
    schema: &QuerySchema,
    profile: &Profile,
    facts: &HostFacts,
    zones: &dyn Zones,
) -> Result<Validation, CoreError> {
    // The current year is read only for a literal that needs it (D5): a
    // zone or clock the run cannot read refuses only such a query.
    let year = local_year(facts, zones);
    let reader = parser::dates::Reader {
        year: year.as_ref().ok().copied(),
        pads_year: facts.platform.strftime_pads_year,
    };
    let mode = match profile.mode {
        Mode::Training => parser::Mode::Training,
        Mode::Inference => parser::Mode::Inference,
    };
    let parse = parser::parse(query, mode, reader);
    let warnings: Vec<CoreWarning> = parse.warnings.iter().map(day_first_warning).collect();
    let parsed = match parse.outcome {
        Ok(parsed) => parsed,
        Err(parser::Unparsed::Refused(response)) => {
            return Ok(Validation::refused(diagnostics(&response), warnings));
        }
        Err(parser::Unparsed::Unplaced) => {
            return Err(year.err().unwrap_or_else(|| {
                internal("a two-digit year went unplaced with the current year known")
            }));
        }
    };
    let refused = check_parsed_capabilities(parsed.target_has_time_range(), profile);
    if !refused.is_empty() {
        return Ok(Validation::refused(refused, warnings));
    }
    let filters = parser::Filters {
        key_columns: profile.capabilities.key_columns_in_filters,
        timestamp_columns: profile.capabilities.timestamp_columns_in_filters,
    };
    let validated = match parser::resolve(parsed, &schema_definition(schema), mode, filters) {
        parser::Resolved::Accepted(validated) => validated,
        parser::Resolved::Refused(response) => {
            return Ok(Validation::refused(diagnostics(&response), warnings));
        }
    };
    let query = validated_query(&validated)?;
    let mut found = diagnostics(&validated.validation_response);
    let refusals = check_capabilities(&features(&query, schema), profile);
    if refusals.is_empty() {
        return Ok(Validation {
            diagnostics: found,
            query: Some(query),
            text: Some(validated.query.to_string_with(false, false)),
            warnings,
        });
    }
    let later = found.split_off(found.iter().take_while(|d| is_error(d)).count());
    found.extend(refusals);
    found.extend(later);
    Ok(Validation::refused(found, warnings))
}

fn is_error(diagnostic: &PqlDiagnostic) -> bool {
    diagnostic.severity == Severity::Error
}

fn internal(message: impl Into<String>) -> CoreError {
    CoreError::new(ErrorKind::Internal, ErrorCode::InternalError, message)
}

// --- Host facts --------------------------------------------------------------

/// D4's warning for one literal, as the run gives it: the format and where
/// the literal is, never its text.
fn day_first_warning(warning: &parser::DateWarning) -> CoreWarning {
    let mut fields = std::collections::BTreeMap::new();
    fields.insert(
        "format".to_owned(),
        Detail::text(warning.day_first.format()),
    );
    if let Some(at) = warning.location.and_then(parser::Span::at) {
        let count = |n: usize| Detail::Scalar(Scalar::UInt64(u64::try_from(n).unwrap_or(u64::MAX)));
        fields.insert("start_row".to_owned(), count(at.start_row));
        fields.insert("start_col".to_owned(), count(at.start_col));
        fields.insert("end_row".to_owned(), count(at.end_row));
        fields.insert("end_col".to_owned(), count(at.end_col));
    }
    CoreWarning {
        code: DATE_LITERAL_DAY_FIRST.to_owned(),
        message: "a date literal was read day first".to_owned(),
        fields,
    }
}

/// The current year in the host's local zone, which places a two-digit
/// year in a date literal, as the reference's `dateutil` places it by the
/// machine's local time: the clock, `facts.now`, in `facts.local_zone`,
/// resolved through the run's zones (an IANA name) or read as a POSIX `TZ`
/// string; in UTC where the host names no zone. A zone the run cannot
/// resolve is refused by name.
pub fn local_year(facts: &HostFacts, zones: &dyn Zones) -> Result<i16, CoreError> {
    let Scalar::Timestamp { unit, value, .. } = &facts.now else {
        return Err(internal("the host's clock is not a ts scalar"));
    };
    let unit = match unit {
        Unit::Ns => arrow_schema::TimeUnit::Nanosecond,
        Unit::Us => arrow_schema::TimeUnit::Microsecond,
    };
    let now = Instant::from_count(*value, unit);
    let unread = || {
        CoreError::new(
            ErrorKind::InvalidRequest,
            ErrorCode::InvalidRequest,
            "the host's local zone is not one this run can read",
        )
        .with_field("field", Detail::text("facts.local_zone"))
    };
    let year = match &facts.local_zone {
        None => now.civil(0).year,
        Some(LocalZone::Iana(name)) => {
            let zone = zones.zone(name).map_err(|_| unread())?;
            now.to_local(&zone).civil.year
        }
        Some(LocalZone::Posix(text)) => {
            let zone = jiff::tz::TimeZone::posix(text).map_err(|_| unread())?;
            now.to_local(&zone).civil.year
        }
    };
    i16::try_from(year).map_err(|_| {
        CoreError::new(
            ErrorKind::InvalidRequest,
            ErrorCode::InvalidRequest,
            "the host's clock is past the years a date literal can be read in",
        )
        .with_field("field", Detail::text("facts.now"))
    })
}

// --- The schema, into the parser's form ------------------------------------

/// A name, where an empty one means none, as the reference reads a schema.
fn name(value: &Option<String>) -> Option<String> {
    value.clone().filter(|v| !v.is_empty())
}

fn schema_definition(schema: &QuerySchema) -> parser::SchemaDefinition {
    use parser::schema::{Column, ColumnKey, ColumnKeyGroup, TableDefinition};
    parser::SchemaDefinition {
        tables: schema
            .tables
            .iter()
            .map(|table| {
                let cols = table
                    .columns
                    .iter()
                    .map(|c| Column {
                        name: c.name.clone(),
                        stype: parser_stype(c.stype),
                        dtype: parser_dtype(c.dtype),
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
        pub(crate) fn $to(value: $a) -> $b {
            match value {
                $(<$a>::$v => <$b>::$v,)+
            }
        }
        pub(crate) fn $from(value: $b) -> $a {
            match value {
                $(<$b>::$v => <$a>::$v,)+
            }
        }
    };
}

same_names!(
    parser_dtype,
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
    parser_stype,
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

fn span(span: Option<parser::Span>) -> Option<Span> {
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
fn diagnostics(response: &parser::ValidationResponse) -> Vec<PqlDiagnostic> {
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

fn validated_query(validated: &parser::ValidatedQuery) -> Result<ValidatedQuery, CoreError> {
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
        entity_clause: if q.for_each == parser::validator::model::FOR {
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

fn window(range: &parser::dor::DateOffsetRange) -> Result<Window, CoreError> {
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
    let read = ast::typed_value(n)
        // The reader's text may quote the constant, which is input.
        .map_err(|_| internal("a validated constant could not be read as its type"))?;
    Ok(match read {
        Literal::Null => PqlLiteral::Null,
        Literal::Bool(value) => PqlLiteral::Bool { value },
        Literal::Int(value) => PqlLiteral::Int {
            value: value.to_string(),
        },
        Literal::Float(_) => PqlLiteral::Float { text: text.clone() },
        Literal::Str(value) => PqlLiteral::String { value },
        Literal::Time(stamp) => PqlLiteral::Time {
            text: text.clone(),
            value: stamp.at.to_string(),
            offset: stamp.offset,
        },
        Literal::List(_) => return Err(internal("a list constant written as text")),
    })
}

/// One node of the parser's tree, in the validated-query format.
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
        span: base.location.and_then(|l| span(parser::Span::at(l))),
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
    use sdm_contract::facts::PlatformFacts;
    use sdm_engine::time::UnknownZone;

    /// Zones that hold only New York, from `jiff`'s bundled database.
    struct NewYork;

    impl Zones for NewYork {
        fn zone(&self, name: &str) -> Result<jiff::tz::TimeZone, UnknownZone> {
            if name != "America/New_York" {
                return Err(UnknownZone);
            }
            jiff::tz::TimeZone::posix("EST5EDT,M3.2.0,M11.1.0").map_err(|_| UnknownZone)
        }
    }

    fn facts(micros: i64, local_zone: Option<LocalZone>) -> HostFacts {
        HostFacts {
            now: Scalar::Timestamp {
                unit: Unit::Us,
                zone: Some("UTC".into()),
                value: micros,
            },
            local_zone,
            zone_roots: vec![],
            platform: PlatformFacts {
                os: "linux".into(),
                strftime_pads_year: false,
            },
        }
    }

    #[test]
    fn the_year_is_the_clock_in_the_local_zone() {
        // 2027-01-01T04:30Z is 2026-12-31T23:30 in New York.
        let new_year = 1_798_777_800_000_000;
        let iana = Some(LocalZone::Iana("America/New_York".into()));
        let posix = Some(LocalZone::Posix("EST5EDT,M3.2.0,M11.1.0".into()));
        assert_eq!(local_year(&facts(new_year, None), &NewYork).unwrap(), 2027);
        assert_eq!(local_year(&facts(new_year, iana), &NewYork).unwrap(), 2026);
        assert_eq!(local_year(&facts(new_year, posix), &NewYork).unwrap(), 2026);
        let unknown = Some(LocalZone::Iana("Mars/Olympus".into()));
        let refused = local_year(&facts(new_year, unknown), &NewYork).unwrap_err();
        assert_eq!(refused.fields["field"], Detail::text("facts.local_zone"));
        assert!(local_year(&facts(i64::MIN, None), &NewYork).is_err());
    }
}
