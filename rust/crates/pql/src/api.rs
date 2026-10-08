// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The one call: text and a schema in, a resolved query or diagnostics out
//! (`predictive_query.api.validate`).

use crate::codes;
use crate::context;
use crate::dates::Host;
use crate::json::Json;
use crate::profiles::Profile;
use crate::query::ValidatedQuery;
use crate::response::{Span, ValidationError, ValidationResponse};
use crate::schema::SchemaDefinition;
use crate::validator::capability::{check_capabilities, check_parsed_capabilities};
use crate::validator::predictive_query::PredictiveQueryValidator;
use crate::visitor::parse_query;

/// What [`validate`] gives.
#[derive(Debug)]
pub struct Validation {
    /// The resolved query, or `None` when `response` has errors.
    pub query: Option<ValidatedQuery>,
    /// The diagnostics: errors, or the warnings of a resolved query.
    pub response: ValidationResponse,
    /// The `UserWarning`s pandas gives as the query's date literals are read,
    /// in order, which today's Python gives through its warnings module.
    pub warnings: Vec<String>,
}

impl Validation {
    /// The answer as the PQL corpus records today's (`tests/parity/pql/`):
    /// `{"tree", "warnings"}` for a resolved query, else
    /// `{"errors", "warnings"}`.
    pub fn to_json(&self) -> Json {
        let warnings = Json::Array(
            self.warnings
                .iter()
                .map(|message| {
                    Json::object(vec![
                        ("category", Json::str("builtins.UserWarning")),
                        ("message", Json::str(message)),
                    ])
                })
                .collect(),
        );
        match &self.query {
            Some(resolved) => Json::object(vec![
                (
                    "tree",
                    resolved.query.to_json(&resolved.validation_response),
                ),
                ("warnings", warnings),
            ]),
            None => Json::object(vec![
                (
                    "errors",
                    Json::Array(
                        self.response
                            .errors
                            .iter()
                            .map(|e| {
                                Json::object(vec![
                                    ("code", Json::str(e.code)),
                                    ("span", e.span.map_or(Json::Null, Span::to_json)),
                                    ("message", Json::str(&e.message)),
                                ])
                            })
                            .collect(),
                    ),
                ),
                ("warnings", warnings),
            ]),
        }
    }
}

/// Resolves `query` against `schema`, then asks whether `profile` can run
/// it. Capability is asked of the parsed query first, since a model that
/// cannot answer a query's shape cannot answer it on any schema.
///
/// Dates are read as pandas reads them on Linux with Python 3.12 or 3.13
/// ([`Host::assumed`]); [`validate_in`] says which Python to follow.
///
/// Where today's Python raises instead of answering (a window that grows
/// past int32 as it is merged into a join), the answer is a
/// `PQL_VALUE_INT_OUT_OF_RANGE` error.
///
/// Validation recurses as deep as the query nests, up to
/// [`crate::parser::MAX_DEPTH`] levels, which needs up to 8 MiB of stack
/// in a debug build; it runs on a stack of its own, so a caller on any
/// thread, however small its stack, is answered.
pub fn validate(query: &str, schema: &SchemaDefinition, profile: &Profile) -> Validation {
    validate_in(query, schema, profile, Host::assumed())
}

/// [`validate`], reading dates as pandas reads them in the Python process
/// `host` describes.
pub fn validate_in(
    query: &str,
    schema: &SchemaDefinition,
    profile: &Profile,
    host: Host,
) -> Validation {
    let ((query, response), warnings) = stacker::grow(crate::STACK_BYTES, || {
        context::run(host, || resolve(query, schema, profile))
    });
    Validation {
        query,
        response,
        warnings,
    }
}

fn resolve(
    query: &str,
    schema: &SchemaDefinition,
    profile: &Profile,
) -> (Option<ValidatedQuery>, ValidationResponse) {
    let parsed = match parse_query(query, profile.mode) {
        Ok(parsed) => parsed,
        Err(failure) => {
            return (
                None,
                ValidationResponse::error(ValidationError::new(
                    failure.title,
                    failure.message,
                    failure.code,
                    None,
                )),
            );
        }
    };
    let parsed_capability = check_parsed_capabilities(&parsed, profile);
    if !parsed_capability.ok() {
        return (None, parsed_capability);
    }
    let validator =
        PredictiveQueryValidator::new(schema, profile.mode, profile.capabilities.clone());
    let (resolved, mut response) = match validator.validate_predictive_query(parsed) {
        Ok(result) => result,
        Err(raised) => {
            return (
                None,
                ValidationResponse::error(ValidationError::new(
                    "Unsupported constant",
                    raised.0,
                    codes::VALUE_INT_OUT_OF_RANGE,
                    None,
                )),
            );
        }
    };
    let Some(mut resolved) = resolved.filter(|_| response.ok()) else {
        return (None, response);
    };
    let capability = check_capabilities(&resolved.query, schema, profile);
    response.errors.extend(capability.errors);
    response.warnings.extend(capability.warnings);
    if !response.ok() {
        return (None, response);
    }
    resolved.validation_response = response.clone();
    (Some(resolved), response)
}
