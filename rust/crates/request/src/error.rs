// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The refusals and warnings the request code gives, as the core's
//! [`Diagnostic`]s with today's class, code, message and details.

use sdm_protocol::Diagnostic;
use serde_json::{Map, Value};

/// `sdm_client.errors.StructuredDataError`.
pub const STRUCTURED: &str = "sdm_client.errors.StructuredDataError";
/// `sdm_client.errors.UnsupportedRequestError`.
pub const UNSUPPORTED: &str = "sdm_client.errors.UnsupportedRequestError";
/// `sdm_client.errors.UnknownModelError`.
pub const UNKNOWN_MODEL: &str = "sdm_client.errors.UnknownModelError";
/// `sdm_client.errors.NimRequestError`.
pub const NIM_REQUEST: &str = "sdm_client.errors.NimRequestError";
/// `sdm_client.errors.MissingExtraError`.
pub const MISSING_EXTRA: &str = "sdm_client.errors.MissingExtraError";

/// `sdm_client.core.warnings.SDMWarning`.
pub const SDM_WARNING: &str = "sdm_client.core.warnings.SDMWarning";
/// `builtins.UserWarning`.
pub const USER_WARNING: &str = "builtins.UserWarning";

/// A refusal, or another exception today's code raises.
pub type R<T> = Result<T, Diagnostic>;

/// `StructuredDataError(message, code=code)`.
pub fn structured(code: &str, message: impl Into<String>) -> Diagnostic {
    Diagnostic::refusal(STRUCTURED, Some(code), message)
}

/// `StructuredDataError(message, code='INVALID_REQUEST')`.
pub fn invalid(message: impl Into<String>) -> Diagnostic {
    structured("INVALID_REQUEST", message)
}

/// `UnsupportedRequestError(message, details=details)`.
pub fn unsupported(message: impl Into<String>, details: Map<String, Value>) -> Diagnostic {
    Diagnostic::refusal(UNSUPPORTED, Some("UNSUPPORTED_REQUEST"), message).with_details(details)
}

/// A builtin exception, such as `ValueError`, which carries no code.
pub fn builtin(class: &str, message: impl Into<String>) -> Diagnostic {
    Diagnostic::refusal(&format!("builtins.{class}"), None, message)
}

/// `ValueError(message)`.
pub fn value_error(message: impl Into<String>) -> Diagnostic {
    builtin("ValueError", message)
}

/// A mapping of details, from pairs.
pub fn details<const N: usize>(pairs: [(&str, Value); N]) -> Map<String, Value> {
    pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect()
}

/// The short name of a diagnostic's class: `ValueError` of
/// `builtins.ValueError`.
pub fn short_class(diagnostic: &Diagnostic) -> String {
    let class = diagnostic.class.as_deref().unwrap_or("CoreError");
    class.rsplit('.').next().unwrap_or(class).to_string()
}

/// A refusal as a run's `done` value carries it, when the run leaves
/// something behind that the host keeps before raising it: `{"class",
/// "code", "message", "details", "status"}`.
pub fn refusal_json(refusal: &Diagnostic) -> Value {
    let mut found = serde_json::json!({
        "class": refusal.class,
        "code": refusal.code,
        "message": refusal.message,
        "details": Value::Object((*refusal.details).clone()),
        "status": refusal.status,
    });
    if !refusal.chain.is_empty() {
        found["chain"] = Value::Array((*refusal.chain).clone());
    }
    found
}

/// The refusal [`refusal_json`] wrote.
pub fn refusal_of(json: &Value) -> Diagnostic {
    let text = |key: &str| json.get(key).and_then(Value::as_str).map(str::to_owned);
    Diagnostic {
        class: text("class"),
        category: None,
        code: text("code"),
        message: text("message").unwrap_or_default(),
        details: Box::new(
            json.get("details")
                .and_then(Value::as_object)
                .cloned()
                .unwrap_or_default(),
        ),
        status: json
            .get("status")
            .and_then(Value::as_i64)
            .and_then(|s| i32::try_from(s).ok()),
        chain: Box::new(
            json.get("chain")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default(),
        ),
    }
}

/// A warning as hosts and fixtures write it: `{"category", "message"}`.
pub fn warning_json(warning: &Diagnostic) -> Value {
    let mut result = serde_json::json!({
        "category": warning.category.as_deref().unwrap_or(USER_WARNING),
        "message": warning.message,
    });
    if warning.details.get("masked_float32_cast") == Some(&Value::Bool(true)) {
        result["masked_float32_cast"] = Value::Bool(true);
    }
    result
}

/// Where warnings go, in the order they were given: kept, or handed to a
/// run as they are given, so each step carries the warnings before it.
#[derive(Default)]
pub struct Warnings {
    kept: Vec<Diagnostic>,
    run: Option<sdm_protocol::Asker>,
}

impl std::fmt::Debug for Warnings {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Warnings")
            .field("kept", &self.kept)
            .finish()
    }
}

impl Warnings {
    /// Warnings handed to the run `asker` asks through as they are given.
    pub fn live(asker: sdm_protocol::Asker) -> Self {
        Self {
            kept: Vec::new(),
            run: Some(asker),
        }
    }

    pub fn sdm(&mut self, message: impl Into<String>) {
        self.push(Diagnostic::warning(SDM_WARNING, message));
    }

    pub fn user(&mut self, message: impl Into<String>) {
        self.push(Diagnostic::warning(USER_WARNING, message));
    }

    pub fn push(&mut self, warning: Diagnostic) {
        match &self.run {
            Some(asker) => asker.warn(warning),
            None => self.kept.push(warning),
        }
    }

    /// The warnings kept so far, taken.
    pub fn take(&mut self) -> Vec<Diagnostic> {
        std::mem::take(&mut self.kept)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A refusal a run's `done` value carries reads back as itself.
    #[test]
    fn a_refusal_reads_back_as_written() {
        let refusal = invalid("no such row")
            .with_detail("table", Value::from("users"))
            .with_detail("rows", Value::from(1));
        assert_eq!(refusal_of(&refusal_json(&refusal)), refusal);
        let mut sent = Diagnostic::refusal(NIM_REQUEST, None, "the endpoint said no");
        sent.status = Some(422);
        assert_eq!(refusal_of(&refusal_json(&sent)), sent);
    }
}
