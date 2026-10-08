// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! PQL, the predictive query language: parsing and validating a query
//! against a schema, ported from Python's `predictive_query`.
//!
//! [`validate`] resolves a query and [`validate_json`] gives the same answer
//! as the JSON today's Python corpus records (`tests/rust_core/pql/CONTRACT.md`).
//! Date literals are read as pandas reads them ([`dates`], `DATES.md`).

pub mod api;
pub mod ast;
pub mod codes;
pub mod context;
pub mod dates;
pub mod dor;
pub mod json;
pub mod lexer;
pub mod names;
pub mod parser;
mod printable;
pub mod profiles;
pub mod pyint;
pub mod pyrepr;
pub mod query;
pub mod response;
pub mod schema;
pub mod syntax;
pub mod tree;
pub mod types;
pub mod validator;
pub mod value;
pub mod visitor;

use std::fmt;

use json::Json;

pub use api::{Validation, validate, validate_in};
pub use profiles::{Capabilities, Mode, Profile};
pub use query::{ParsedQuery, ValidatedQuery};
pub use response::{Span, ValidationError, ValidationResponse, ValidationWarning};
pub use schema::SchemaDefinition;

/// Input [`validate_json`] cannot read: not a schema, or an unknown profile
/// or mode. A query, whatever its text, is always answered.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InputError {
    Schema(String),
    Profile(String),
    Mode(String),
    /// Validation failed inside the core; a bug, never an answer.
    Internal(String),
}

impl fmt::Display for InputError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InputError::Schema(e) => write!(f, "not a schema: {e}"),
            InputError::Profile(name) => write!(
                f,
                "unknown profile {name:?}; expected RELATIONAL, TABULAR, TIMESERIES, ANOMALY or \
                 FINE_TUNING"
            ),
            InputError::Mode(mode) => {
                write!(f, "unknown mode {mode:?}; expected TRAINING or INFERENCE")
            }
            InputError::Internal(e) => write!(f, "PQL validation failed: {e}"),
        }
    }
}

impl std::error::Error for InputError {}

/// The stack a validation runs on: deep enough for the deepest query the
/// parser takes ([`parser::MAX_DEPTH`]), in debug builds too. Reserved, not
/// committed: only what a validation touches is.
const STACK_BYTES: usize = 256 << 20;

/// [`validate`] as JSON text: `{"tree": ..., "warnings": [...]}` for a
/// query that resolves, else `{"errors": [...], "warnings": [...]}`, as
/// `tests/parity/test_pql_corpus.py` writes today's answers.
///
/// `profile` names one of the five profiles; `mode`, if given, replaces its
/// mode (`TRAINING` or `INFERENCE`).
pub fn validate_json(
    query: &str,
    schema_json: &str,
    profile: &str,
    mode: Option<&str>,
) -> Result<String, InputError> {
    validate_json_in(query, schema_json, profile, mode, dates::Host::assumed())
}

/// [`validate_json`], reading dates as pandas reads them in the Python
/// process `host` describes.
pub fn validate_json_in(
    query: &str,
    schema_json: &str,
    profile: &str,
    mode: Option<&str>,
    host: dates::Host,
) -> Result<String, InputError> {
    let schema = SchemaDefinition::from_json(schema_json).map_err(InputError::Schema)?;
    let mut profile =
        Profile::by_name(profile).ok_or_else(|| InputError::Profile(profile.to_string()))?;
    if let Some(mode) = mode {
        profile.mode = Mode::parse(mode).ok_or_else(|| InputError::Mode(mode.to_string()))?;
    }
    // A panic would be a bug in the port: an error, never an abort.
    std::panic::catch_unwind(|| {
        validate_in(query, &schema, &profile, host)
            .to_json()
            .render()
    })
    .map_err(|_| InputError::Internal("validation panicked".to_string()))
}

/// The reading of a date literal, `{"value", "warnings"}`, as JSON text
/// (`tests/rust_core/pql/CONTRACT.md`): `value` is `null` when the literal
/// is refused, else `YYYY-MM-DDTHH:MM:SS` and any UTC offset.
pub fn parse_date_json(literal: &str) -> String {
    parse_date_json_in(literal, dates::Host::assumed())
}

/// [`parse_date_json`], as pandas reads the literal in the Python process
/// `host` describes.
pub fn parse_date_json_in(literal: &str, host: dates::Host) -> String {
    let reading = dates::to_datetime(literal, host);
    let value = reading
        .value
        .map_or(Json::Null, |stamp| Json::String(stamp.iso()));
    let warnings = reading
        .warnings
        .into_iter()
        .map(|message| {
            Json::object(vec![
                ("category", Json::str("builtins.UserWarning")),
                ("message", Json::String(message)),
            ])
        })
        .collect();
    Json::object(vec![("value", value), ("warnings", Json::Array(warnings))]).render()
}
