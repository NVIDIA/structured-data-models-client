// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared wire-option admission, independent of model capability policy.

use sdm_contract::{CoreError, Detail, Sentence};
use sdm_wire::json::Value;

/// Normalize only the common typed fields, preserving unknown model options.
/// The caller owns the copied outgoing envelope, never the user's request.
pub fn normalize(value: &mut Value) -> Result<(), CoreError> {
    if let Some(mode) = value.get("run_mode")
        && !matches!(mode, Value::Null)
        && !matches!(mode, Value::Text(text) if text.chars().count() <= 1024)
    {
        return Err(CoreError::new(
            Sentence::InferenceRunModeInvalid,
            "inference run_mode must be text of at most 1024 characters or null",
        )
        .with_field("field", Detail::text("run_mode")));
    }
    if let Some(given) = value.get("use_prediction_time") {
        let flag = crate::options::lax_boolean("use_prediction_time", given)?;
        value.set("use_prediction_time", Value::Bool(flag));
    }
    if value
        .get("inference_config")
        .is_some_and(|v| !matches!(v, Value::Object(_)))
    {
        return Err(CoreError::new(
            Sentence::InferenceConfigNotAnObject,
            "inference_config must be a mapping",
        )
        .with_field("field", Detail::text("inference_config")));
    }
    Ok(())
}

/// Apply admission to an outgoing request or session envelope.
pub fn envelope(payload: &mut Value) -> Result<(), CoreError> {
    if let Some(mut inference) = payload.get("inference").cloned() {
        normalize(&mut inference)?;
        payload.set("inference", inference);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use sdm_contract::ErrorCode;

    #[test]
    fn admission_normalizes_only_known_fields() {
        let mut value = sdm_wire::json::read(br#"{"use_prediction_time":"NO","run_mode":null,"future":{"x":1},"inference_config":{}}"#).unwrap();
        normalize(&mut value).unwrap();
        assert_eq!(value.get("use_prediction_time"), Some(&Value::Bool(false)));
        assert_eq!(
            value.get("future").unwrap().get("x"),
            Some(&Value::Number("1".into()))
        );
        for token in ["null", "2", "[]", "\" no\""] {
            let mut value =
                sdm_wire::json::read(format!(r#"{{"use_prediction_time":{token}}}"#).as_bytes())
                    .unwrap();
            assert_eq!(
                normalize(&mut value).unwrap_err().code,
                ErrorCode::InvalidRequest
            );
        }
        for text in [r#"{"run_mode":7}"#, r#"{"inference_config":null}"#] {
            let mut value = sdm_wire::json::read(text.as_bytes()).unwrap();
            assert_eq!(
                normalize(&mut value).unwrap_err().code,
                ErrorCode::InvalidRequest
            );
        }
    }

    #[test]
    fn mode_limit_counts_unicode_characters() {
        let mut value = Value::Object(vec![("run_mode".into(), Value::text(&"é".repeat(1024)))]);
        normalize(&mut value).unwrap();
        value.set("run_mode", Value::text(&"é".repeat(1025)));
        assert_eq!(
            normalize(&mut value).unwrap_err().code,
            ErrorCode::InvalidRequest
        );
    }
}
