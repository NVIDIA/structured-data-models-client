// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use sdm_protocol::Diagnostic;
use serde_json::{Value, json};

pub const INTERNAL_CAUSE: &str = "_sdm_internal_cause";
pub const INTERNAL_WARNING: &str = "_sdm_internal_warning";
pub const TIME_COLUMN_WARNING: &str = "TimeColumnWarning";
pub const VIEW_CONVERSION_WARNING: &str = "ViewConversionWarning";

fn external_class(class: &str) -> Option<String> {
    match class {
        "ValueError"
        | "KeyError"
        | "TypeError"
        | "RuntimeError"
        | "NotImplementedError"
        | "IndexError"
        | "AssertionError"
        | "OSError"
        | "OverflowError"
        | "ZeroDivisionError"
        | "AttributeError"
        | "UnicodeDecodeError"
        | "MemoryError"
        | "Exception" => Some(format!("builtins.{class}")),
        "OutOfBoundsDatetime" | "ParserError" | "DateParseError" => {
            Some(format!("pandas.errors.{class}"))
        }
        "ArrowInvalid" | "ArrowTypeError" | "ArrowNotImplementedError" => {
            Some(format!("pyarrow.lib.{class}"))
        }
        "ResponseContractError" => Some("sdm_api.generated.tfm_api.ResponseContractError".into()),
        other if other.contains('.') => Some(other.to_owned()),
        _ => None,
    }
}

pub fn public_class(class: &str) -> String {
    external_class(class).unwrap_or_else(|| "builtins.ValueError".to_owned())
}

pub fn cause_link(class: &str, message: impl Into<String>) -> Value {
    let mut result = json!({"class": public_class(class), "message": message.into()});
    if external_class(class).is_none() {
        result[INTERNAL_CAUSE] = json!(class);
    }
    result
}

pub fn warning_category(category: &str) -> &str {
    match category {
        TIME_COLUMN_WARNING => "builtins.UserWarning",
        VIEW_CONVERSION_WARNING => "sdm_connectors._warnings.ViewConversionWarning",
        other => other,
    }
}

pub fn warning(category: &str, message: String) -> Diagnostic {
    let mut result = Diagnostic::warning(warning_category(category), message);
    if matches!(category, TIME_COLUMN_WARNING | VIEW_CONVERSION_WARNING) {
        result
            .details
            .insert(INTERNAL_WARNING.to_owned(), json!(category));
    }
    result
}

pub fn public_observation(diagnostic: &Diagnostic) -> Value {
    let mut details = (*diagnostic.details).clone();
    if diagnostic.category.is_some() {
        details.remove(INTERNAL_WARNING);
    }
    let structured_error = diagnostic.category.is_none()
        && matches!(
            diagnostic.class.as_deref(),
            Some(
                "sdm_client.errors.StructuredDataError"
                    | "sdm_client.errors.NimRequestError"
                    | "sdm_client.errors.UnsupportedRequestError"
                    | "sdm_client.errors.UnknownModelError"
                    | "sdm_client.errors.MissingExtraError"
            )
        );
    json!({
        "class": diagnostic.class.as_ref().or(diagnostic.category.as_ref()),
        "code": diagnostic.code,
        "message": diagnostic.message,
        "details": if details.is_empty() && !structured_error { Value::Null } else { json!(details) },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn owned_causes_have_explicit_visibility() {
        assert_eq!(public_class("TableNotFoundError"), "builtins.ValueError");
        assert_eq!(public_class("TaskReferenceError"), "builtins.ValueError");
        assert_eq!(
            cause_link("ValueError", "external"),
            json!({"class":"builtins.ValueError","message":"external"})
        );
        assert_eq!(
            cause_link("TableNotFoundError", "internal")[INTERNAL_CAUSE],
            "TableNotFoundError"
        );
    }

    #[test]
    fn public_warnings_hide_presentation_metadata() {
        for (kind, category) in [
            (TIME_COLUMN_WARNING, "builtins.UserWarning"),
            (
                VIEW_CONVERSION_WARNING,
                "sdm_connectors._warnings.ViewConversionWarning",
            ),
        ] {
            let diagnostic = warning(kind, "message".into());
            let value = public_observation(&diagnostic);
            assert_eq!(
                value,
                json!({"class":category,"code":null,"message":"message","details":null})
            );
            assert_eq!(diagnostic.details[INTERNAL_WARNING], kind);
        }
    }

    #[test]
    fn public_structured_errors_retain_empty_details() {
        for class in [
            "sdm_client.errors.StructuredDataError",
            "sdm_client.errors.NimRequestError",
            "sdm_client.errors.UnsupportedRequestError",
            "sdm_client.errors.UnknownModelError",
            "sdm_client.errors.MissingExtraError",
        ] {
            let diagnostic = Diagnostic::refusal(class, Some("INVALID_REQUEST"), "refused");
            assert_eq!(public_observation(&diagnostic)["details"], json!({}));
            let detailed =
                diagnostic.with_details(serde_json::from_value(json!({"retained": 7})).unwrap());
            assert_eq!(
                public_observation(&detailed)["details"],
                json!({"retained": 7})
            );
        }
        for class in [
            "builtins.ValueError",
            "pyarrow.lib.ArrowNotImplementedError",
        ] {
            let diagnostic = Diagnostic::refusal(class, None, "refused");
            assert!(public_observation(&diagnostic)["details"].is_null());
        }
    }
}
