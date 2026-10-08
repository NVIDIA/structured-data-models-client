// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use sdm_protocol::Diagnostic;
use serde_json::{Value, json};

use crate::diagnostics::{
    INTERNAL_CAUSE, INTERNAL_WARNING, TIME_COLUMN_WARNING, VIEW_CONVERSION_WARNING,
};

fn historical_class(class: &str) -> String {
    if class == "TaskReferenceError" {
        "sdm_relational_engine.rfm.diagnostics.TaskReferenceError".to_owned()
    } else {
        format!("sdm_relational_engine.exceptions.{class}")
    }
}

fn historical_cause(value: &mut Value) {
    if let Some(fields) = value.as_object_mut() {
        if let Some(Value::String(class)) = fields.remove(INTERNAL_CAUSE) {
            fields.insert("class".to_owned(), json!(historical_class(&class)));
        }
        if let Some(Value::Array(chain)) = fields.get_mut("chain") {
            for link in chain {
                historical_cause(link);
            }
        }
    }
}

pub fn historical_fixture(value: &mut Value) {
    if let Some(error) = value.get_mut("error") {
        historical_cause(error);
    }
}

pub fn historical_materialize(value: &mut Value) {
    if let Some(refused) = value.get_mut("refused") {
        historical_cause(refused);
    }
}

pub fn fixture_warning(warning: &Diagnostic) -> Value {
    let category = match warning
        .details
        .get(INTERNAL_WARNING)
        .and_then(Value::as_str)
    {
        Some(TIME_COLUMN_WARNING) => "sdm_relational_engine.rfm.base.table.TimeColumnWarning",
        Some(VIEW_CONVERSION_WARNING) => "sdm_relational_engine.rfm.graph.ViewConversionWarning",
        _ => warning
            .category
            .as_deref()
            .unwrap_or("builtins.UserWarning"),
    };
    json!({"category": category, "message": warning.message})
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diagnostics::cause_link;

    #[test]
    fn historical_fields_are_restored_without_touching_payloads() {
        let payload = json!({"class":"payload","_sdm_internal_cause":"opaque"});
        let warning = crate::diagnostics::warning(TIME_COLUMN_WARNING, "time".into());
        let mut value = json!({"output":payload,"error":{"chain":[cause_link("TaskReferenceError","missing")]},"warnings":[fixture_warning(&warning)]});
        historical_fixture(&mut value);
        assert_eq!(value["output"], payload);
        assert_eq!(
            value["error"]["chain"][0],
            json!({"class":"sdm_relational_engine.rfm.diagnostics.TaskReferenceError","message":"missing"})
        );
        assert_eq!(
            value["warnings"][0],
            json!({"category":"sdm_relational_engine.rfm.base.table.TimeColumnWarning","message":"time"})
        );
        let once = value.clone();
        historical_fixture(&mut value);
        assert_eq!(value, once);
    }

    #[test]
    fn historical_warning_names_are_confined_to_fixture_presentation() {
        for (kind, expected) in [
            (
                TIME_COLUMN_WARNING,
                "sdm_relational_engine.rfm.base.table.TimeColumnWarning",
            ),
            (
                VIEW_CONVERSION_WARNING,
                "sdm_relational_engine.rfm.graph.ViewConversionWarning",
            ),
        ] {
            let warning = crate::diagnostics::warning(kind, "message".into());
            assert_eq!(
                fixture_warning(&warning),
                json!({"category": expected, "message": "message"})
            );
            assert!(
                !crate::diagnostics::public_observation(&warning)
                    .to_string()
                    .contains("sdm_relational_engine")
            );
        }
        let warning = Diagnostic::warning("builtins.DeprecationWarning", "prefix");
        assert_eq!(
            fixture_warning(&warning),
            json!({"category": "builtins.DeprecationWarning", "message": "prefix"})
        );
    }

    #[test]
    fn materialize_compatibility_requires_a_structured_marked_cause() {
        let payload = cause_link("TaskReferenceError", "payload");
        let external = json!({"class":"builtins.ValueError","message":"external"});
        let mut value = json!({"payload":payload,"refused":{"chain":[cause_link("TaskReferenceError","missing"),external]}});
        historical_materialize(&mut value);
        assert_eq!(value["payload"], payload);
        assert_eq!(value["refused"]["chain"][1], external);
        assert_eq!(
            value["refused"]["chain"][0]["class"],
            "sdm_relational_engine.rfm.diagnostics.TaskReferenceError"
        );
        assert!(value["refused"]["chain"][0].get(INTERNAL_CAUSE).is_none());
        let once = value.clone();
        historical_materialize(&mut value);
        assert_eq!(value, once);
    }
}
