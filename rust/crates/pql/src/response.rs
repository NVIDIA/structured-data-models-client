// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What validation says about a query (`predictive_query.schema.common`):
//! errors, warnings and information, each with a stable code.

use crate::ast::Location;
use crate::json::Json;

/// Where in the query a diagnostic points. Rows count from 1 and columns
/// from 0, and the end is the first character of the last token.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Span {
    pub start_row: usize,
    pub start_col: usize,
    pub end_row: usize,
    pub end_col: usize,
}

impl Span {
    /// `Span.at`: a location's span, or `None` for one without data.
    pub fn at(location: Location) -> Option<Span> {
        if !location.data_available {
            return None;
        }
        Some(Span {
            start_row: location.start_row,
            start_col: location.start_col,
            end_row: location.end_row,
            end_col: location.end_col,
        })
    }

    pub fn to_json(self) -> Json {
        Json::object(vec![
            ("start_row", Json::int(self.start_row)),
            ("start_col", Json::int(self.start_col)),
            ("end_row", Json::int(self.end_row)),
            ("end_col", Json::int(self.end_col)),
        ])
    }
}

fn span_json(span: Option<Span>) -> Json {
    span.map_or(Json::Null, Span::to_json)
}

/// A decision validation made on the author's behalf.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValidationInfo {
    pub title: String,
    pub message: String,
    pub id: String,
}

/// Something questionable about a query that does not stop it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValidationWarning {
    pub title: String,
    pub message: String,
    pub code: &'static str,
    pub span: Option<Span>,
}

/// A reason a query cannot run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValidationError {
    pub title: String,
    pub message: String,
    pub code: &'static str,
    pub span: Option<Span>,
}

impl ValidationError {
    pub fn new(
        title: impl Into<String>,
        message: impl Into<String>,
        code: &'static str,
        span: Option<Span>,
    ) -> Self {
        ValidationError {
            title: title.into(),
            message: message.into(),
            code,
            span,
        }
    }
}

impl ValidationWarning {
    pub fn new(
        title: impl Into<String>,
        message: impl Into<String>,
        code: &'static str,
        span: Option<Span>,
    ) -> Self {
        ValidationWarning {
            title: title.into(),
            message: message.into(),
            code,
            span,
        }
    }
}

/// Everything validation has to say about one query.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ValidationResponse {
    pub warnings: Vec<ValidationWarning>,
    pub errors: Vec<ValidationError>,
    pub info_items: Vec<ValidationInfo>,
}

impl ValidationResponse {
    /// Whether the query can run: no errors.
    pub fn ok(&self) -> bool {
        self.errors.is_empty()
    }

    /// Whether there is nothing to say at all.
    pub fn empty(&self) -> bool {
        self.errors.is_empty() && self.warnings.is_empty() && self.info_items.is_empty()
    }

    pub fn error(error: ValidationError) -> Self {
        ValidationResponse {
            errors: vec![error],
            ..Self::default()
        }
    }

    /// `merge`: this response, with `other`'s items after its own.
    pub fn merge(mut self, other: ValidationResponse) -> Self {
        self.warnings.extend(other.warnings);
        self.errors.extend(other.errors);
        self.info_items.extend(other.info_items);
        self
    }

    /// Every error, warning and item of information, numbered.
    pub fn message(&self) -> String {
        let mut message = self.error_message();
        if !self.warnings.is_empty() {
            if !self.errors.is_empty() {
                message.push('\n');
            }
            message.push_str("Warnings:\n");
            let lines: Vec<String> = self
                .warnings
                .iter()
                .enumerate()
                .map(|(i, w)| format!("{}. {}", i + 1, w.message))
                .collect();
            message.push_str(&lines.join("\n"));
        }
        if !self.info_items.is_empty() {
            if !message.is_empty() {
                message.push('\n');
            }
            message.push_str("Info:\n");
            let lines: Vec<String> = self
                .info_items
                .iter()
                .enumerate()
                .map(|(i, info)| format!("{}. {}. {}", i + 1, info.title, info.message))
                .collect();
            message.push_str(&lines.join("\n"));
        }
        message
    }

    /// Every error, numbered when there is more than one.
    pub fn error_message(&self) -> String {
        match self.errors.as_slice() {
            [] => String::new(),
            [only] => format!("{}. {}", only.title, only.message),
            errors => errors
                .iter()
                .enumerate()
                .map(|(i, e)| format!("{}. {}. {}", i + 1, e.title, e.message))
                .collect::<Vec<_>>()
                .join("\n"),
        }
    }

    /// The response as `json_serde.to_json_dict` writes it.
    pub fn to_json(&self) -> Json {
        Json::object(vec![
            (
                "warnings",
                Json::Array(
                    self.warnings
                        .iter()
                        .map(|w| {
                            Json::object(vec![
                                ("title", Json::str(&w.title)),
                                ("message", Json::str(&w.message)),
                                ("code", Json::str(w.code)),
                                ("span", span_json(w.span)),
                            ])
                        })
                        .collect(),
                ),
            ),
            (
                "errors",
                Json::Array(
                    self.errors
                        .iter()
                        .map(|e| {
                            Json::object(vec![
                                ("title", Json::str(&e.title)),
                                ("message", Json::str(&e.message)),
                                ("code", Json::str(e.code)),
                                ("span", span_json(e.span)),
                            ])
                        })
                        .collect(),
                ),
            ),
            (
                "info_items",
                Json::Array(
                    self.info_items
                        .iter()
                        .map(|i| {
                            Json::object(vec![
                                ("title", Json::str(&i.title)),
                                ("message", Json::str(&i.message)),
                                ("id", Json::str(&i.id)),
                            ])
                        })
                        .collect(),
                ),
            ),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn messages_number_errors() {
        let mut r = ValidationResponse::error(ValidationError::new("A", "first", "X", None));
        assert_eq!(r.error_message(), "A. first");
        r.errors
            .push(ValidationError::new("B", "second", "Y", None));
        r.warnings
            .push(ValidationWarning::new("W", "careful", "Z", None));
        assert_eq!(
            r.message(),
            "1. A. first\n2. B. second\nWarnings:\n1. careful"
        );
        assert!(!r.ok() && !r.empty());
    }
}
