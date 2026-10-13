// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The two calls the adapter makes: text to a parsed query, and a parsed
//! query resolved against a schema (`predictive_query.api.validate`, less
//! the capability checks, which the core asks).
//!
//! Both recurse as deep as the query nests ([`crate::parser::MAX_DEPTH`]
//! levels) and run on the caller's stack: the core's adapter grows one
//! stack for a whole call (`sdm_core::pql::on_adapter_stack`), and this
//! crate grows none of its own.

use crate::ast;
use crate::codes;
use crate::dates::Reader;
use crate::profiles::{Filters, Mode};
use crate::query::{ParsedQuery, ValidatedQuery};
use crate::response::{ValidationError, ValidationResponse};
use crate::schema::SchemaDefinition;
use crate::validator::predictive_query::PredictiveQueryValidator;
use crate::visitor::{ParseError, parse_query};

pub use crate::visitor::DateWarning;

/// A query the grammar accepts, not yet resolved against a schema.
#[derive(Debug)]
pub struct Parsed {
    pub query: ParsedQuery,
}

impl Parsed {
    /// Whether the target carries a time range, which a model may refuse
    /// before resolution (`check_parsed_capabilities`).
    pub fn target_has_time_range(&self) -> bool {
        ast::date_offset_range(&self.query.target_ast).is_some()
    }
}

/// What resolving a query gives.
#[derive(Debug)]
pub enum Resolved {
    /// The validated query, and the warnings resolution gave.
    Accepted(ValidatedQuery),
    /// Why the query does not resolve against the schema.
    Refused(ValidationResponse),
}

/// What parsing gives: the parsed query, or why not; and D4's warning for
/// each day-first date literal read, in the order read, whether or not the
/// query parses.
#[derive(Debug)]
pub struct Parse {
    pub outcome: Result<Parsed, Unparsed>,
    pub warnings: Vec<DateWarning>,
}

/// Why text is no parsed query.
#[derive(Debug)]
pub enum Unparsed {
    /// Refused: one diagnostic without a span.
    Refused(ValidationResponse),
    /// A date literal's two-digit year, with no current year to place it
    /// (`Reader::year` none): never a diagnostic about the query.
    Unplaced,
    /// A date literal with a run of five digits or more (D8), refused by
    /// name, and where it is: never a diagnostic, never its text
    /// (decision E78).
    Packed(Option<ast::Location>),
}

/// `text` parsed for `mode`, date literals read with `reader`.
pub fn parse(text: &str, mode: Mode, reader: Reader) -> Parse {
    let (parsed, warnings) = parse_query(text, mode, reader);
    let outcome = match parsed {
        Ok(query) => Ok(Parsed { query }),
        Err(ParseError::Failure(failure)) => Err(Unparsed::Refused(ValidationResponse::error(
            ValidationError::new(failure.title, failure.message, failure.code, None),
        ))),
        Err(ParseError::Unplaced) => Err(Unparsed::Unplaced),
        Err(ParseError::Packed(location)) => Err(Unparsed::Packed(location)),
    };
    Parse { outcome, warnings }
}

/// `parsed` resolved against `schema` for `mode`, with the filter
/// capabilities resolution consults. Where the reference raises instead of
/// answering (a window that grows past int32 as it is merged into a join),
/// the answer is a `PQL_VALUE_INT_OUT_OF_RANGE` error.
pub fn resolve(
    parsed: Parsed,
    schema: &SchemaDefinition,
    mode: Mode,
    filters: Filters,
) -> Resolved {
    let validator = PredictiveQueryValidator::new(schema, mode, filters);
    match validator.validate_predictive_query(parsed.query) {
        Err(raised) => Resolved::Refused(ValidationResponse::error(ValidationError::new(
            "Unsupported constant",
            raised.0,
            codes::VALUE_INT_OUT_OF_RANGE,
            None,
        ))),
        Ok((Some(resolved), response)) if response.ok() => Resolved::Accepted(resolved),
        Ok((_, response)) => Resolved::Refused(response),
    }
}
