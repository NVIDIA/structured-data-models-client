// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Computing labels from the predictive query, as today's
//! `PQueryPandasExecutor` computes them over the rows the sampler fetched,
//! and the `pd.DateOffset` arithmetic their windows (and the sampler) use.
//!
//! The executor's arithmetic is pandas': [`column`] holds a series by its
//! dtype, [`aggregate`] groups and aggregates it with pandas' kernels,
//! [`executor`] walks the query ([`query`]) as `PQueryPandasExecutor` does,
//! and [`arrow`] reads the sampler's Arrow tables and writes the labels.

pub mod aggregate;
pub mod arrow;
pub mod column;
pub mod date_offset;
pub mod datetime;
pub mod executor;
pub mod fixture;
pub mod query;

use std::fmt;

use serde_json::{Value, json};

#[derive(Clone, Debug, PartialEq)]
pub enum Warning {
    User(String),
    Float32Overflow(f64),
    MaskedFloat32Overflow(f64),
}

impl Warning {
    pub fn category(&self) -> &'static str {
        match self {
            Self::User(_) => "builtins.UserWarning",
            Self::Float32Overflow(_) | Self::MaskedFloat32Overflow(_) => "builtins.RuntimeWarning",
        }
    }

    pub fn message(&self) -> &str {
        match self {
            Self::User(message) => message,
            Self::Float32Overflow(_) | Self::MaskedFloat32Overflow(_) => {
                "overflow encountered in cast"
            }
        }
    }

    pub fn masked_float32_cast(&self) -> bool {
        matches!(self, Self::MaskedFloat32Overflow(_))
    }
}

/// What the executor raised: the Python exception's class and text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LabelError {
    pub code: String,
    pub message: String,
    /// A `KeyError`'s argument, whose `repr` is its text.
    pub key: Option<String>,
}

impl LabelError {
    pub fn new(code: &str, message: impl Into<String>) -> Self {
        Self {
            code: code.to_string(),
            message: message.into(),
            key: None,
        }
    }

    /// `KeyError(key)`, whose text is the key's `repr`.
    pub fn key_error(key: &str) -> Self {
        Self {
            code: "KeyError".to_string(),
            message: sdm_schema::py::str_repr(key),
            key: Some(key.to_string()),
        }
    }
}

impl fmt::Display for LabelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.code, self.message)
    }
}

impl std::error::Error for LabelError {}

/// The `labels.*` operations of the parity fixtures
/// (`tests/rust_core/sampling/CONTRACT.md`, Labels): the case's input in,
/// `{"output": ...}` or `{"error": {"code", "message"}}` out, as JSON text.
/// `Err` is for an unknown operation or input that isn't a JSON object.
pub fn fixture_op(operation: &str, input_json: &str) -> Result<String, String> {
    fixture_op_in(operation, input_json, &sdm_schema::dt::Host::assumed())
}

/// [`fixture_op`] with what reading a date takes from the Python process.
pub fn fixture_op_in(
    operation: &str,
    input_json: &str,
    host: &sdm_schema::dt::Host,
) -> Result<String, String> {
    fixture_op_with_warnings(operation, input_json, host).map(|(output, _warnings)| output)
}

pub fn fixture_op_with_warnings(
    operation: &str,
    input_json: &str,
    host: &sdm_schema::dt::Host,
) -> Result<(String, Vec<Warning>), String> {
    let input: Value =
        serde_json::from_str(input_json).map_err(|e| format!("the input is not JSON: {e}"))?;
    let Value::Object(input) = input else {
        return Err("the input is not a JSON object".to_string());
    };
    let mut warnings = Vec::new();
    let output = match fixture::run_with_warnings(operation, &input, host, &mut warnings) {
        None => return Err(format!("no labels fixture operation {operation:?}")),
        Some(Ok(output)) => json!({"output": output}).to_string(),
        Some(Err(error)) => {
            json!({"error": {"code": error.code, "message": error.message}}).to_string()
        }
    };
    Ok((output, warnings))
}

/// `PQueryPandasExecutor().execute_query(query, ...)` of a resolved query's
/// `tree` (JSON) over one batch: the labels and the mask of the examples
/// they belong to.
pub fn execute(
    query_json: &str,
    batch: &executor::Batch,
) -> Result<executor::Labelled, LabelError> {
    let query = query::Query::from_json(query_json)?;
    executor::execute_query(&query, batch)
}

#[cfg(test)]
mod tests {
    use super::LabelError;

    #[test]
    fn a_key_error_keeps_its_key_and_shows_its_repr() {
        let error = LabelError::key_error("a\u{85}b");
        assert_eq!(error.code, "KeyError");
        assert_eq!(error.message, "'a\\x85b'");
        assert_eq!(error.key.as_deref(), Some("a\u{85}b"));
    }
}
