// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The models this client can send to (`sdm_client/models.py`): what each
//! may be asked (its profile), what its deployment serves (its runtime
//! contract), and `spec_for`, which names one.

use sdm_schema::pyval::PyVal;
use serde_json::json;

use crate::error::{R, UNKNOWN_MODEL, invalid};

pub const KUMO_TABULAR: &str = "kumo-tabular";
pub const KUMO_RELATIONAL: &str = "kumo-relational";

/// `RELATIONAL_LIMITS`.
pub mod relational_limits {
    pub const CONTEXT_ROWS_FAST: i64 = 1_000;
    pub const CONTEXT_ROWS_MAX: i64 = 10_000;
    pub const PREDICT_ROWS_MAX: i64 = 1_000;
    pub const PREDICT_ROWS_TEMPORAL_LINK_PREDICTION: i64 = 200;
    pub const EXPLANATION_PREDICT_ROWS: i64 = 1;
    pub const HOPS_MAX: i64 = 6;
    pub const TABLES_MAX: i64 = 15;
    pub const CLASSES_MAX: i64 = 10;
}

/// `TABULAR_LIMITS['context_rows_max']`.
pub const TABULAR_CONTEXT_ROWS_MAX: i64 = 10_000;

pub const DEFAULT_PREDICTION_STATISTIC: &str = "median";
pub const PREDICTION_STATISTICS: [&str; 2] = ["mean", "median"];
pub const SUMMARISED_TASK_KINDS: [&str; 2] = ["regression", "forecasting"];

/// Which of the profiles a model may be asked: `RELATIONAL` or `TABULAR`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Profile {
    Relational,
    Tabular,
}

impl Profile {
    /// The name `predictive_query` knows the profile by.
    pub fn name(self) -> &'static str {
        match self {
            Profile::Relational => "RELATIONAL",
            Profile::Tabular => "TABULAR",
        }
    }

    /// `profile.capabilities.model`.
    pub fn model(self) -> &'static str {
        match self {
            Profile::Relational => KUMO_RELATIONAL,
            Profile::Tabular => KUMO_TABULAR,
        }
    }

    /// The shape capabilities of `SHAPE_CAPABILITIES` the profile has.
    pub fn supports(self, capability: &str) -> bool {
        match self {
            Profile::Relational => true,
            Profile::Tabular => matches!(capability, "static"),
        }
    }
}

/// A model's specification.
#[derive(Clone, Debug, PartialEq)]
pub struct Spec {
    pub model_id: &'static str,
    pub profile: Profile,
    pub tasks: &'static [&'static str],
    pub outputs: &'static [&'static str],
    pub inference_options: &'static [&'static str],
    pub unread_feature_stypes: &'static [&'static str],
    pub context_rows_max: Option<i64>,
}

pub const TABULAR: Spec = Spec {
    model_id: KUMO_TABULAR,
    profile: Profile::Tabular,
    tasks: &["classification", "regression"],
    outputs: &["prediction", "probabilities", "quantiles"],
    inference_options: &[],
    unread_feature_stypes: &[],
    context_rows_max: Some(TABULAR_CONTEXT_ROWS_MAX),
};

pub const RELATIONAL: Spec = Spec {
    model_id: KUMO_RELATIONAL,
    profile: Profile::Relational,
    tasks: &[
        "classification",
        "binary_classification",
        "multiclass_classification",
        "regression",
        "forecasting",
        "temporal_link_prediction",
    ],
    outputs: &[
        "prediction",
        "probabilities",
        "scores",
        "rankings",
        "embeddings",
        "quantiles",
        "explanation",
    ],
    inference_options: &[
        "kind",
        "num_estimators",
        "column_shuffle",
        "category_shuffle",
        "hop_shuffle",
        "class_shuffle",
    ],
    unread_feature_stypes: &["text"],
    context_rows_max: Some(relational_limits::CONTEXT_ROWS_MAX),
};

/// Every model, as `MODELS` lists them.
pub const MODELS: [&Spec; 2] = [&TABULAR, &RELATIONAL];

/// The model ids, sorted, as `sorted(MODELS)` gives them.
pub fn known_models() -> Vec<&'static str> {
    let mut known: Vec<&'static str> = MODELS.iter().map(|m| m.model_id).collect();
    known.sort_unstable();
    known
}

/// `UnknownModelError(model, list(MODELS))`.
pub fn unknown_model(model: &str) -> sdm_protocol::Diagnostic {
    let known = known_models();
    let shown: Vec<String> = known.iter().map(|k| sdm_pql::pyrepr::str_repr(k)).collect();
    sdm_protocol::Diagnostic::refusal(
        UNKNOWN_MODEL,
        Some("UNKNOWN_MODEL"),
        format!(
            "Unknown model {}; this client can send: [{}]. The Universal TFM contract declares \
             these as a closed set, so a new model id has to be added to the contract and the \
             bindings regenerated before it can be sent.",
            sdm_pql::pyrepr::str_repr(model),
            shown.join(", ")
        ),
    )
    .with_detail("model", json!(model))
    .with_detail("known_models", json!(known))
}

/// The handle classes `sdm_client.models` defines, by name, and the spec
/// each carries.
fn handle(class: &str) -> Option<&'static Spec> {
    match class {
        "KumoTabular" => Some(&TABULAR),
        "KumoRelational" => Some(&RELATIONAL),
        _ => None,
    }
}

/// `spec_for(model)`: a model id, or a handle written `{"$model": CLASS}`.
pub fn spec_for(model: &PyVal) -> R<&'static Spec> {
    match model {
        PyVal::Str(id) => MODELS
            .iter()
            .find(|m| m.model_id == id)
            .copied()
            .ok_or_else(|| unknown_model(id)),
        PyVal::Dict(items) => {
            let class = items.iter().find_map(|(k, v)| match (k, v) {
                (PyVal::Str(k), PyVal::Str(v)) if k == "$model" => Some(v.clone()),
                _ => None,
            });
            match class {
                Some(class) => handle(&class).ok_or_else(|| unknown_model(&class)),
                None => Err(invalid(format!(
                    "model must be a model handle or a model id, got {}",
                    type_name(model)
                ))),
            }
        }
        other => Err(invalid(format!(
            "model must be a model handle or a model id, got {}",
            type_name(other)
        ))),
    }
}

/// `type(value).__name__` of a value a caller passed.
pub fn type_name(value: &PyVal) -> String {
    if let PyVal::Held {
        type_name: Some(name),
        ..
    } = value
    {
        return name.clone();
    }
    match value.semantic() {
        PyVal::None => "NoneType".to_string(),
        PyVal::Bool(_) => "bool".to_string(),
        PyVal::Int(_) | PyVal::BigInt(_) => "int".to_string(),
        PyVal::Float(_) => "float".to_string(),
        PyVal::Str(_) => "str".to_string(),
        PyVal::Bytes(_) => "bytes".to_string(),
        PyVal::List(_) => "list".to_string(),
        PyVal::Tuple(_) => "tuple".to_string(),
        PyVal::Dict(_) => "dict".to_string(),
        PyVal::Decimal(_) => "Decimal".to_string(),
        PyVal::Timestamp(_) | PyVal::TimestampTz(..) => "Timestamp".to_string(),
        PyVal::DateTime(..) => "datetime".to_string(),
        PyVal::Date(..) => "date".to_string(),
        PyVal::Timedelta(_) => "timedelta".to_string(),
        PyVal::Uuid(_) => "UUID".to_string(),
        PyVal::Na => "NAType".to_string(),
        PyVal::NaT => "NaTType".to_string(),
        PyVal::NdArray(..) => "ndarray".to_string(),
        PyVal::Numpy(name, _) => name.clone(),
        PyVal::Opaque { type_name, .. } => type_name.clone(),
        _ => "object".to_string(),
    }
}
