// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The PQL adapter: the one module that names the PQL parser.
//!
//! Behind it is `sdm-pql`, this client's own parser (decision E18). It
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
    ColumnRef, Dtype, PqlDiagnostic, QueryFacts, QuerySchema, QueryValidation, Severity, Span,
    Stype,
};
use sdm_contract::pql::{
    Aggregation, ConditionOp, EntityClause, LogicalOp, NodeKind, NodeType, PqlLiteral, PqlNode,
    ProblemType, QueryMode, TimeUnit, ValidatedQuery, Window,
};
use sdm_contract::{
    CoreError, CoreWarning, Detail, HostFacts, Scalar, Sentence, Unit, WarningCode,
};
use sdm_engine::time::{Instant, Zones};
use sdm_pql as parser;
use sdm_pql::ast::{self, ConstValue, Kind, NodeRef};
use sdm_pql::literal::Literal;
use sdm_pql::types as t;

use crate::profiles::{
    Mode, Profile, QueryFeatures, check_capabilities, check_parsed_capabilities,
};

/// D4's warning's code: a date literal read day first.
pub const DATE_LITERAL_DAY_FIRST: &str = WarningCode::DateLiteralDayFirst.as_str();

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
    /// The warnings of the literals read, in order, accepted or not.
    pub warnings: Vec<CoreWarning>,
    /// A forecast's plan, when it has no errors: how many steps it asks,
    /// and what each step's regression is written from ([`per_step`]),
    /// never the steps themselves, which a caller writes within its own
    /// bound ([`ForecastPlan::step`], T1.36 fix 23).
    pub steps: Option<ForecastPlan>,
    /// A ranking's plan, when it has no errors: what each candidate's
    /// question is written from ([`per_candidate`], T2.8).
    pub ranking: Option<RankingPlan>,
    /// The entity filter's condition, when the query has no errors and
    /// filters its entity, written as the reference writes it
    /// (`str(query.entity_ast.condition)`), which its notices and its
    /// refusal of a sample left empty quote (T2.13b).
    pub entity_filter: Option<String>,
}

impl Validation {
    fn refused(diagnostics: Vec<PqlDiagnostic>, warnings: Vec<CoreWarning>) -> Self {
        Validation {
            diagnostics,
            query: None,
            text: None,
            warnings,
            steps: None,
            ranking: None,
            entity_filter: None,
        }
    }

    /// The answer `validate_query` gives over `schema`: an accepted
    /// query's facts and its text as written back with it (T3.8).
    pub fn to_result(&self, schema: &QuerySchema) -> QueryValidation {
        QueryValidation {
            accepted: self.query.is_some(),
            diagnostics: self.diagnostics.clone(),
            query: self.facts(schema).map(|facts| QueryFacts {
                text: self.text.clone(),
                ..facts
            }),
        }
    }

    /// The facts of the accepted query a host reads ([`QueryFacts`],
    /// T2.18); `None` when the query was refused.
    pub fn facts(&self, schema: &QuerySchema) -> Option<QueryFacts> {
        let query = self.query.as_ref()?;
        let features = features(query, schema);
        Some(QueryFacts {
            entity_table: features.entity_table,
            problem_type: query.problem_type,
            windowed: features.temporal_target,
            num_forecasts: query.num_forecasts,
            text: None,
        })
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
    let (validation, warnings) = validate_warned(query, schema, profile, facts, zones);
    validation.map(|validation| (validation, warnings))
}

/// [`validate`], with D4's warnings for the day-first literals read beside
/// the outcome, whether the query was answered or the run ends with an
/// error (a packed literal, a two-digit year the run's zone cannot place):
/// the literals read before the error keep their warnings, as the
/// reference's reader gives each as it reads it (carried row U1, T2.13).
pub fn validate_warned(
    query: &str,
    schema: &QuerySchema,
    profile: &Profile,
    facts: &HostFacts,
    zones: &dyn Zones,
) -> (Result<QueryValidation, CoreError>, Vec<CoreWarning>) {
    on_adapter_stack(|| {
        let resolved = resolve_here(query, schema, profile, facts, zones);
        match resolved {
            Ok(validation) => (Ok(validation.to_result(schema)), validation.warnings),
            Err(Halted { error, warnings }) => (Err(*error), warnings),
        }
    })
}

/// A validation that ended with an error, and D4's warnings for the
/// day-first literals read before it.
struct Halted {
    error: Box<CoreError>,
    warnings: Vec<CoreWarning>,
}

impl From<CoreError> for Halted {
    fn from(error: CoreError) -> Self {
        Halted {
            error: Box::new(error),
            warnings: vec![],
        }
    }
}

/// `query` as the PQL library writes it back (its `to_string()`, the text a
/// host sends materialize after validating, redundant parentheses dropped),
/// or `None` when it does not parse. Read for training with no current year,
/// on the adapter's stack ([`on_adapter_stack`]).
pub fn normalized(query: &str) -> Option<String> {
    on_adapter_stack(|| {
        let reader = parser::dates::Reader {
            year: None,
            pads_year: true,
        };
        let parse = parser::parse(query, parser::Mode::Training, reader);
        parse
            .outcome
            .ok()
            .map(|parsed| parsed.query.to_string_with(false, false))
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
    on_adapter_stack(|| resolve_here(query, schema, profile, facts, zones).map_err(|h| *h.error))
}

fn resolve_here(
    query: &str,
    schema: &QuerySchema,
    profile: &Profile,
    facts: &HostFacts,
    zones: &dyn Zones,
) -> Result<Validation, Halted> {
    // Resolve the current year up front; propagate a zone or clock error
    // only if a literal needs a two-digit year placed.
    let year = local_year(facts, zones);
    let reader = parser::dates::Reader {
        year: year.as_ref().ok().copied(),
        pads_year: facts.platform.strftime_pads_year,
    };
    let mode = match profile.mode {
        Mode::Training => parser::Mode::Training,
        Mode::Inference => parser::Mode::Inference,
    };
    let held = facts.date_literal_range.map(|range| parser::dates::Held {
        min_ns: range.min_ns,
        max_ns: range.max_ns,
    });
    let parse = parser::parse_held(query, mode, reader, held);
    let warnings: Vec<CoreWarning> = parse.warnings.iter().map(literal_warning).collect();
    // An error after a literal was read keeps the literal's warning (U1).
    let halted = |error: CoreError| Halted {
        error: Box::new(error),
        warnings: warnings.clone(),
    };
    let parsed = match parse.outcome {
        Ok(parsed) => parsed,
        Err(parser::Unparsed::Refused(response)) => {
            return Ok(Validation::refused(diagnostics(&response), warnings));
        }
        Err(parser::Unparsed::Packed(location, literal)) => {
            return Err(halted(packed(query, location, &literal)));
        }
        Err(parser::Unparsed::Unplaced) => {
            return Err(halted(year.err().unwrap_or_else(|| {
                internal("a two-digit year went unplaced with the current year known")
            })));
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
    let query = validated_query(&validated).map_err(halted)?;
    let mut found = diagnostics(&validated.validation_response);
    let refusals = check_capabilities(&features(&query, schema), profile);
    if refusals.is_empty() {
        let steps = per_step(&validated).map_err(halted)?;
        return Ok(Validation {
            diagnostics: found,
            query: Some(query),
            text: Some(validated.query.to_string_with(false, false)),
            warnings,
            steps,
            ranking: per_candidate(&validated),
            entity_filter: entity_filter(&validated),
        });
    }
    let later = found.split_off(found.iter().take_while(|d| is_error(d)).count());
    found.extend(refusals);
    found.extend(later);
    Ok(Validation::refused(found, warnings))
}

/// A forecast as the client's `_per_step` splits it, kept compact: how
/// many steps it asks, and the parts every step's regression is written
/// from. Nothing here grows with the step count (T1.36 fix 23).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ForecastPlan {
    /// `FORECAST n TIMEFRAMES`'s n, saturated at `u64::MAX`.
    pub count: u64,
    aggr: &'static str,
    target: String,
    start: i128,
    end: i128,
    unit: &'static str,
    entity: String,
}

impl ForecastPlan {
    /// Step `step`'s regression (from 1): the same aggregation of what it
    /// aggregates, filter and all, its window moved along by `step - 1`
    /// times its span, and the same entity clause, filter and all.
    pub fn step(&self, step: u64) -> String {
        let span = self.end - self.start;
        let moved = i128::from(step.saturating_sub(1)) * span;
        format!(
            "PREDICT {}({}, {}, {}, {}) {}",
            self.aggr,
            self.target,
            self.start + moved,
            self.end + moved,
            self.unit,
            self.entity,
        )
    }
}

/// A forecast's plan, the regressions the client's `_per_step` asks it as
/// (T1.36 fix 21), written by [`ForecastPlan::step`]; none for a query that
/// is not a forecast. A forecast whose target ends in no aggregation the
/// reference cannot split (an `AttributeError`), refused by name, as the
/// regression it would ask is; one whose window has no start the query's
/// validation refuses first ("TARGET cannot have an aggregation in the
/// past"), so reaching it here is a fault of this build (T2.10). Nothing is
/// written per step here (T1.36 fix 23).
fn per_step(validated: &parser::ValidatedQuery) -> Result<Option<ForecastPlan>, CoreError> {
    let query = &validated.query;
    if query.problem_type != Some(t::ProblemType::Forecast) {
        return Ok(None);
    }
    let aggregation = validated
        .get_final_target_aggregation()
        .ok_or_else(|| sdm_engine::not_yet("problem_type"))?;
    let (aggr, target) = match &aggregation.borrow().kind {
        Kind::Aggregation { aggr, target, .. } => (aggr.value(), ast::to_string(target)),
        _ => return Err(internal("a final aggregation is not an aggregation")),
    };
    let window = ast::date_offset_range(&aggregation)
        .ok_or_else(|| internal("a forecast's aggregation has no window"))?;
    let window = window.borrow();
    let start = i128::from(
        window
            .start
            .ok_or_else(|| internal("a forecast's window has no start"))?,
    );
    let count = query
        .num_forecasts
        .to_i64()
        .map_or(u64::MAX, |n| u64::try_from(n).unwrap_or(0));
    Ok(Some(ForecastPlan {
        count,
        aggr,
        target,
        start,
        end: i128::from(window.end),
        unit: window.unit.value(),
        entity: format!("{} {}", query.for_each, ast::to_string(&query.entity_ast)),
    }))
}

/// A ranking as the client's `_per_candidate` asks it (T2.8): what every
/// candidate's binary question is written from, read off the validated
/// query; nothing is refused here, the core refusing in the client's order
/// what the client refuses as it writes the questions
/// (`sdm_request::ranking`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RankingPlan {
    /// The query's own `TOP k`, its digits as written.
    pub top_k: Option<String>,
    /// The final aggregation the ranking is over; none where the target
    /// ends in none, which the client refuses.
    pub ranked: Option<sdm_request::ranking::Ranked>,
}

/// A ranking's plan ([`RankingPlan`]); none for a query that is not a
/// ranking.
fn per_candidate(validated: &parser::ValidatedQuery) -> Option<RankingPlan> {
    let query = &validated.query;
    if query.problem_type != Some(t::ProblemType::Rank) {
        return None;
    }
    let top_k = query.top_k.as_ref().map(ToString::to_string);
    let ranked = validated
        .get_final_target_aggregation()
        .and_then(|aggregation| {
            let Kind::Aggregation { target, .. } = &aggregation.borrow().kind else {
                return None;
            };
            let text = ast::to_string(target);
            // The table and the column ranked: the column node's own, held
            // apart (KO2-36); the text split at its first dot where the target
            // is no lone column, as the client splits it.
            let (table, column) = match &target.borrow().kind {
                Kind::Column { name, .. } => (name.table_name.clone(), name.col_name.clone()),
                _ => match text.split_once('.') {
                    Some((table, column)) => (table.to_owned(), column.to_owned()),
                    None => (text.clone(), String::new()),
                },
            };
            let window = ast::date_offset_range(&aggregation).map(|window| {
                let window = window.borrow();
                sdm_request::ranking::Window {
                    start: window.start,
                    end: window.end,
                    unit: window.unit.value().to_owned(),
                }
            });
            Some(sdm_request::ranking::Ranked {
                target: text,
                table,
                column,
                window,
                entity: format!("{} {}", query.for_each, ast::to_string(&query.entity_ast)),
            })
        });
    Some(RankingPlan { top_k, ranked })
}

/// The entity filter's condition as the reference writes it
/// (`str(query.entity_ast.condition)`), when the entity is filtered.
fn entity_filter(validated: &parser::ValidatedQuery) -> Option<String> {
    match &validated.query.entity_ast.borrow().kind {
        Kind::Filter { condition, .. } => Some(ast::to_string(condition)),
        _ => None,
    }
}

fn is_error(diagnostic: &PqlDiagnostic) -> bool {
    diagnostic.severity == Severity::Error
}

/// The refusal of a date literal with a run of five digits or more (D8),
/// by name and where it is: the reference reads such a literal through
/// `dateutil`, or refuses it. It names the literal and the query, as the
/// caller wrote them (`literal`, `query`), as the reference's message does
/// (KO2-86, narrowing decision E79: a value of the caller's own call).
fn packed(query: &str, location: Option<ast::Location>, literal: &str) -> CoreError {
    let mut error = CoreError::new(
        Sentence::DateLiteralPacked,
        "a date literal with a run of five digits or more is refused",
    )
    .with_field("operation", Detail::text("validate_query"))
    .with_field("field", Detail::text("query"))
    .with_field("query", Detail::text(query))
    .with_field("literal", Detail::text(literal));
    if let Some(at) = location.and_then(parser::Span::at) {
        error = error
            .with_field("start_row", count(at.start_row))
            .with_field("start_col", count(at.start_col))
            .with_field("end_row", count(at.end_row))
            .with_field("end_col", count(at.end_col));
    }
    error
}

/// A count as a detail.
fn count(n: usize) -> Detail {
    Detail::Scalar(Scalar::UInt64(u64::try_from(n).unwrap_or(u64::MAX)))
}

fn internal(message: impl Into<String>) -> CoreError {
    CoreError::new(Sentence::Internal, message)
}

// --- Host facts --------------------------------------------------------------

/// Reading a deprecated string escape preserves its first diagnostic;
/// validation's warning channel, not the materializer, presents it.
fn literal_warning(warning: &parser::LiteralWarning) -> CoreWarning {
    match warning {
        parser::LiteralWarning::DayFirst(warning) => day_first_warning(warning),
        parser::LiteralWarning::InvalidStringEscape { location } => {
            let mut warning = CoreWarning::new(
                Sentence::InvalidStringEscape,
                "a string literal uses a deprecated escape",
            );
            warning
                .fields
                .insert("escape".to_owned(), Detail::text("\\/"));
            literal_location(&mut warning.fields, *location);
            warning
        }
    }
}

/// D4's warning for one literal, as the run gives it: the format and where
/// the literal is, never its text.
fn day_first_warning(warning: &parser::DateWarning) -> CoreWarning {
    let mut fields = std::collections::BTreeMap::new();
    fields.insert(
        "format".to_owned(),
        Detail::text(warning.day_first.format()),
    );
    literal_location(&mut fields, warning.location);
    // D4's warning: a date literal read day first, whose format, as
    // `strftime` writes it, is the `format` field, and whose place in the
    // query is `start_row`, `start_col`, `end_row` and `end_col`.
    let mut warning = CoreWarning::new(
        Sentence::DateLiteralDayFirst,
        "a date literal was read day first",
    );
    warning.fields = fields;
    warning
}

fn literal_location(
    fields: &mut std::collections::BTreeMap<String, Detail>,
    location: Option<parser::ast::Location>,
) {
    if let Some(at) = location.and_then(parser::Span::at) {
        let count = |n: usize| Detail::Scalar(Scalar::UInt64(u64::try_from(n).unwrap_or(u64::MAX)));
        fields.insert("start_row".to_owned(), count(at.start_row));
        fields.insert("start_col".to_owned(), count(at.start_col));
        fields.insert("end_row".to_owned(), count(at.end_row));
        fields.insert("end_col".to_owned(), count(at.end_col));
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
            Sentence::LocalZoneUnread,
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
            Sentence::ClockPastYears,
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

/// The parser holds a column's table and column apart, as the contract
/// does, so neither is ever split on a dot (KO2-36).
fn column_ref(name: &parser::schema::ColumnKey) -> ColumnRef {
    ColumnRef {
        table: name.table_name.clone(),
        column: name.col_name.clone(),
    }
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
        Kind::Column { name, .. } => NodeKind::Column {
            column: column_ref(name),
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
            group_by: group_by.as_ref().map(column_ref),
            time_column: time_col.as_ref().map(column_ref),
        },
        Kind::Join {
            rhs_target,
            lhs_key,
            rhs_key,
        } => NodeKind::Join {
            target: boxed(rhs_target)?,
            lhs_key: column_ref(lhs_key),
            rhs_key: column_ref(rhs_key),
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

/// A name as the query language spells it, in backticks where a bare name
/// cannot (T2.8: a ranking's questions name the ranked table so), from
/// here, the one module that names the parser.
pub(crate) fn quote_name(name: &str) -> String {
    parser::names::quote_name(name)
}

/// The text a float class is spelled in, as the reference spells it: the
/// parser's float text, the shortest digits that read back to the double
/// ([`sdm_pql::float_repr`]). The payload and the answer's layout take it
/// from here, the one module that names the parser.
pub fn float_text(x: f64) -> String {
    parser::float_repr::float_repr(x)
}

/// A view definition's value too large to quote: its text would pass
/// [`sdm_pql::conversions::LONGEST_TEXT`], or its nesting, through the
/// aliases of collections inside themselves, the depth a quote follows.
/// Python wrote such a text or ran out of memory; it is refused by name
/// (register P3J-10).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TooLarge;

/// How deep a quote follows a value: twice the nesting a definition may
/// have, for the collections met again inside themselves.
const QUOTE_DEPTH: usize = 2 * sdm_schema::view::yaml::MAX_DEPTH;

/// What a quote may still spend, and the nodes it is inside. Each value
/// spends the fewest characters it can be written in (a text its own,
/// a date ten), so a quote that runs out would be longer than the bound;
/// the text itself is then written up to the bound exactly
/// ([`parser::float_repr::Shown::repr_within`]).
struct Quote {
    budget: usize,
    open: Vec<sdm_schema::view::yaml::NodeId>,
    on_path: BTreeSet<sdm_schema::view::yaml::NodeId>,
}

impl Quote {
    fn new() -> Quote {
        Quote {
            budget: parser::conversions::LONGEST_TEXT,
            open: Vec::new(),
            on_path: BTreeSet::new(),
        }
    }

    fn spend(&mut self, cost: usize) -> Result<(), TooLarge> {
        self.budget = self.budget.checked_sub(cost).ok_or(TooLarge)?;
        Ok(())
    }
}

/// A node of a view definition, as the parser's text helpers take it: a
/// collection met again inside itself is Python's `[...]` or `{...}`.
fn shown(
    document: &sdm_schema::view::yaml::Document,
    node: sdm_schema::view::yaml::NodeId,
    quote: &mut Quote,
) -> Result<parser::float_repr::Shown, TooLarge> {
    use parser::float_repr::Shown;
    use sdm_schema::view::yaml::Value;
    let value = document.value(node);
    if quote.on_path.contains(&node) {
        quote.spend(1)?;
        return Ok(Shown::Recursive(!matches!(value, Value::Map(_))));
    }
    if quote.open.len() >= QUOTE_DEPTH {
        return Err(TooLarge);
    }
    quote.spend(match value {
        Value::Str(text) => text.chars().count(),
        Value::Int(n) => n.text().len(),
        Value::Date(_) => 10,
        Value::Timestamp(_) => 19,
        _ => 1,
    })?;
    quote.open.push(node);
    quote.on_path.insert(node);
    let shown = match value {
        Value::Null => Shown::None,
        Value::Bool(flag) => Shown::Bool(*flag),
        Value::Int(n) => Shown::Int(n.text().to_owned()),
        Value::Float(x) => Shown::Float(*x),
        Value::Str(text) => Shown::Text(text.clone()),
        Value::Date(date) => Shown::Date {
            year: date.year,
            month: date.month,
            day: date.day,
        },
        Value::Timestamp(moment) => Shown::DateTime {
            year: moment.date.year,
            month: moment.date.month,
            day: moment.date.day,
            hour: moment.hour,
            minute: moment.minute,
            second: moment.second,
            microsecond: moment.microsecond,
            offset: moment.offset,
        },
        Value::Seq(items) => Shown::List(
            items
                .iter()
                .map(|n| shown(document, *n, quote))
                .collect::<Result<_, _>>()?,
        ),
        Value::Map(pairs) => Shown::Dict(
            pairs
                .iter()
                .map(|(k, v)| Ok((shown(document, *k, quote)?, shown(document, *v, quote)?)))
                .collect::<Result<_, _>>()?,
        ),
    };
    quote.open.pop();
    quote.on_path.remove(&node);
    Ok(shown)
}

/// The text the old client's messages quote a view definition's value by,
/// Python's `repr` of what PyYAML made of it ([`sdm_pql::float_repr`]),
/// from here, the one module that names the parser (T3.5).
pub(crate) fn value_repr(
    document: &sdm_schema::view::yaml::Document,
    node: sdm_schema::view::yaml::NodeId,
) -> Result<String, TooLarge> {
    shown(document, node, &mut Quote::new())?
        .repr_within(parser::conversions::LONGEST_TEXT)
        .ok_or(TooLarge)
}

/// Python's `str` of a view definition's value, as the old client wrote
/// it into a message or a parameter's default (T3.5).
pub(crate) fn value_text(
    document: &sdm_schema::view::yaml::Document,
    node: sdm_schema::view::yaml::NodeId,
) -> Result<String, TooLarge> {
    shown(document, node, &mut Quote::new())?
        .text_within(parser::conversions::LONGEST_TEXT)
        .ok_or(TooLarge)
}

/// Python's `repr` of a text, as the old client quotes a piece of an
/// expression or a name (T3.5).
pub(crate) fn text_repr(text: &str) -> String {
    parser::float_repr::text_repr(text)
}

/// How a view definition's YAML is read: Python's limit on an integer's
/// digits, from the parser (T3.5).
pub fn yaml_rules() -> sdm_schema::view::yaml::Rules {
    sdm_schema::view::yaml::Rules {
        most_digits: parser::conversions::MAX_STR_DIGITS,
        day_ordinal: parser::conversions::day_ordinal,
    }
}

/// Python's white space (`str.isspace`), from the parser's one rule of
/// it, as the data layer's view checks take it, with the class its
/// patterns are compiled with derived from it once (T3.5).
pub fn spaces() -> &'static sdm_data::view_expr::Spaces {
    static SPACES: std::sync::LazyLock<sdm_data::view_expr::Spaces> =
        std::sync::LazyLock::new(|| sdm_data::view_expr::Spaces::new(parser::syntax::is_space));
    &SPACES
}

/// `date.isoformat()` of a view definition's date, from the parser's one
/// writer of it (T3.5).
pub(crate) fn date_iso(date: &sdm_schema::view::yaml::Date) -> String {
    parser::conversions::date_iso(date.year, date.month, date.day)
}

/// `datetime.isoformat(separator)` of a view definition's date and time,
/// from the parser's one writer of it (T3.5).
pub(crate) fn datetime_iso(moment: &sdm_schema::view::yaml::Timestamp, separator: char) -> String {
    parser::conversions::datetime_iso(
        (moment.date.year, moment.date.month, moment.date.day),
        parser::conversions::Clock {
            hour: moment.hour,
            minute: moment.minute,
            second: moment.second,
            microsecond: moment.microsecond,
        },
        moment.offset.map(|seconds| i64::from(seconds) * 1_000_000),
        separator,
    )
}

/// The conversions of a parameter's default, as the old client's Python
/// made them ([`sdm_pql::conversions`]): `text.strip()`, `str(int(text))`,
/// `format(Decimal(text), 'f')`, and the ISO readers and writers.
pub(crate) mod conversions {
    pub(crate) use sdm_pql::conversions::{
        Unconverted, date_text, datetime_text, decimal_text, int_text, not_an_integer_text, strip,
        too_many_digits_text,
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use sdm_contract::ErrorCode;
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
            date_literal_range: None,
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

    /// Carried row U1 (T2.13): a day-first literal read before the error a
    /// query then ends with (a packed literal, a two-digit year the run's
    /// zone cannot place) keeps its warning; one read after it is never
    /// read, as the reference's reader stops there too.
    #[test]
    fn a_day_first_literal_read_before_an_error_keeps_its_warning() {
        let schema: QuerySchema = serde_json::from_value(serde_json::json!({
            "tables": [{"name": "t", "columns": [
                {"name": "id", "dtype": "int", "stype": "ID"},
                {"name": "c", "dtype": "int", "stype": "categorical"},
                {"name": "ts", "dtype": "time", "stype": "timestamp"}],
             "primary_key": "id", "time_column": "ts", "end_time_column": null}],
            "links": [],
        }))
        .unwrap();
        let profile = Profile::resolve(&sdm_contract::ops::ProfileSpec {
            name: sdm_contract::ops::ProfileName::Relational,
            model: None,
            capabilities: Default::default(),
        });
        let known = facts(1_767_225_600_000_000, None);
        let unknown = facts(
            1_767_225_600_000_000,
            Some(LocalZone::Iana("Mars/Olympus".into())),
        );
        for (query, facts, rule, warned) in [
            (
                "PREDICT t.c FOR EACH t.id WHERE t.ts > 13/01/2024 AND t.ts < 20240101-01-01",
                &known,
                Some("date_literal_packed"),
                1,
            ),
            (
                "PREDICT t.c FOR EACH t.id WHERE t.ts > 13/01/2024 AND t.ts > 14/01/2024 \
                 AND t.ts < 20240101-01-01",
                &known,
                Some("date_literal_packed"),
                2,
            ),
            (
                "PREDICT t.c FOR EACH t.id WHERE t.ts > 20240101-01-01 AND t.ts < 13/01/2024",
                &known,
                Some("date_literal_packed"),
                0,
            ),
            (
                "PREDICT t.c FOR EACH t.id WHERE t.ts > 13/01/2024 AND t.ts < 1/2/76",
                &unknown,
                None,
                1,
            ),
        ] {
            let (outcome, warnings) = validate_warned(query, &schema, &profile, facts, &NewYork);
            let err = outcome.expect_err(query);
            assert_eq!(err.code, ErrorCode::InvalidRequest, "{query}");
            match rule {
                Some(rule) => assert_eq!(err.fields["rule"], Detail::text(rule), "{query}"),
                None => assert_eq!(err.fields["field"], Detail::text("facts.local_zone")),
            }
            assert_eq!(warnings.len(), warned, "{query}");
            assert!(
                warnings.iter().all(|w| w.code == DATE_LITERAL_DAY_FIRST),
                "{query}"
            );
            // `validate` ends with the same error.
            let plain = validate(query, &schema, &profile, facts, &NewYork).unwrap_err();
            assert_eq!(plain, err, "{query}");
        }
    }
}
