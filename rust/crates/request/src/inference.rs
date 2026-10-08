// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The engine's inference configurations (`rfm/inference/config.py`): the
//! defaults each task gets, and a configuration a caller handed over as a
//! dataclass, written `{"$inference_config": {"class": C, "fields": {...}}}`.

use serde_json::{Map, Value, json};

/// The fields of configuration class `class`, with their defaults.
pub fn defaults(class: &str) -> Vec<(&'static str, Value)> {
    let mut fields = vec![
        ("num_estimators", json!(1)),
        ("column_shuffle", json!(false)),
        ("category_shuffle", json!(false)),
        ("hop_shuffle", json!(false)),
    ];
    match class {
        "ClassificationInferenceConfig" => fields.push(("class_shuffle", json!(false))),
        "RegressionInferenceConfig" => {
            fields.push(("target_transforms", json!(["quantile"])));
            fields.push(("output_type", json!("median")));
        }
        _ => {}
    }
    fields
}

/// `_keys_the_caller_set(dataclass)`: the fields set to other than their
/// default.
pub fn set_fields(dataclass: &Map<String, Value>) -> Vec<String> {
    let class = dataclass
        .get("class")
        .and_then(Value::as_str)
        .unwrap_or("InferenceConfig");
    let given = dataclass
        .get("fields")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    defaults(class)
        .into_iter()
        .filter(|(name, default)| given.get(*name).is_some_and(|v| v != default))
        .map(|(name, _)| name.to_string())
        .collect()
}
