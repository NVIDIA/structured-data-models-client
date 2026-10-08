// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What a query is for, and what the model on the other end can do
//! (`predictive_query.profiles`).
//!
//! The mode is what a query means; capabilities are what a model will
//! answer, and a capability may only ever refuse.

/// Whether a query defines a task or asks one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Training,
    Inference,
}

impl Mode {
    pub fn value(self) -> &'static str {
        match self {
            Mode::Training => "TRAINING",
            Mode::Inference => "INFERENCE",
        }
    }

    pub fn parse(text: &str) -> Option<Self> {
        match text {
            "TRAINING" => Some(Mode::Training),
            "INFERENCE" => Some(Mode::Inference),
            _ => None,
        }
    }
}

/// What a model can answer. Every field defaults to permitting.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Capabilities {
    pub model: String,
    pub relationships: bool,
    pub multiple_tables: bool,
    pub temporal: bool,
    pub static_: bool,
    pub entity_filters: bool,
    pub counterfactuals: bool,
    pub derived_targets: bool,
    pub link_prediction: bool,
    pub forecasting: bool,
    pub key_columns_in_filters: bool,
    pub timestamp_columns_in_filters: bool,
}

impl Default for Capabilities {
    fn default() -> Self {
        Capabilities {
            model: "this model".to_string(),
            relationships: true,
            multiple_tables: true,
            temporal: true,
            static_: true,
            entity_filters: true,
            counterfactuals: true,
            derived_targets: true,
            link_prediction: true,
            forecasting: true,
            key_columns_in_filters: true,
            timestamp_columns_in_filters: true,
        }
    }
}

/// The capability names `Capabilities.also` takes.
pub const CAPABILITY_NAMES: &[&str] = &[
    "relationships",
    "multiple_tables",
    "temporal",
    "static",
    "entity_filters",
    "counterfactuals",
    "derived_targets",
    "link_prediction",
    "forecasting",
    "key_columns_in_filters",
    "timestamp_columns_in_filters",
];

impl Capabilities {
    /// A model that supports nothing until it says otherwise.
    pub fn nothing(model: &str) -> Self {
        Capabilities {
            model: model.to_string(),
            relationships: false,
            multiple_tables: false,
            temporal: false,
            static_: false,
            entity_filters: false,
            counterfactuals: false,
            derived_targets: false,
            link_prediction: false,
            forecasting: false,
            key_columns_in_filters: false,
            timestamp_columns_in_filters: false,
        }
    }

    /// The same capabilities, with each named one set as given.
    pub fn also(&self, supported: &[(&str, bool)]) -> Result<Self, String> {
        let mut unknown: Vec<&str> = supported
            .iter()
            .map(|(name, _)| *name)
            .filter(|name| !CAPABILITY_NAMES.contains(name))
            .collect();
        if !unknown.is_empty() {
            unknown.sort_unstable();
            let quoted: Vec<String> = unknown.iter().map(|n| format!("'{n}'")).collect();
            return Err(format!("not a capability: [{}]", quoted.join(", ")));
        }
        let mut out = self.clone();
        for (name, value) in supported {
            let field = match *name {
                "relationships" => &mut out.relationships,
                "multiple_tables" => &mut out.multiple_tables,
                "temporal" => &mut out.temporal,
                "static" => &mut out.static_,
                "entity_filters" => &mut out.entity_filters,
                "counterfactuals" => &mut out.counterfactuals,
                "derived_targets" => &mut out.derived_targets,
                "link_prediction" => &mut out.link_prediction,
                "forecasting" => &mut out.forecasting,
                "key_columns_in_filters" => &mut out.key_columns_in_filters,
                _ => &mut out.timestamp_columns_in_filters,
            };
            *field = *value;
        }
        Ok(out)
    }
}

/// A mode and a set of capabilities: everything a caller declares.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Profile {
    pub mode: Mode,
    pub capabilities: Capabilities,
}

impl Profile {
    /// The same profile, naming a particular model in its diagnostics.
    pub fn named(&self, model: &str) -> Self {
        let mut out = self.clone();
        out.capabilities.model = model.to_string();
        out
    }

    /// One of the five profiles, by name.
    pub fn by_name(name: &str) -> Option<Self> {
        let caps = |model: &str| Capabilities {
            model: model.to_string(),
            ..Capabilities::default()
        };
        let inference = |capabilities| Profile {
            mode: Mode::Inference,
            capabilities,
        };
        match name {
            "FINE_TUNING" => Some(Profile {
                mode: Mode::Training,
                capabilities: caps("fine-tuning"),
            }),
            "RELATIONAL" => Some(inference(caps("a relational model"))),
            "TABULAR" => Some(inference(Capabilities {
                relationships: false,
                multiple_tables: false,
                temporal: false,
                entity_filters: false,
                counterfactuals: false,
                derived_targets: false,
                link_prediction: false,
                forecasting: false,
                ..caps("a tabular model")
            })),
            "TIMESERIES" => Some(inference(Capabilities {
                relationships: false,
                multiple_tables: false,
                static_: false,
                link_prediction: false,
                ..caps("a time-series model")
            })),
            "ANOMALY" => Some(inference(Capabilities {
                relationships: false,
                multiple_tables: false,
                temporal: false,
                link_prediction: false,
                forecasting: false,
                ..caps("an anomaly-detection model")
            })),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profiles_and_capabilities() {
        let tabular = Profile::by_name("TABULAR").unwrap();
        assert!(!tabular.capabilities.temporal && tabular.capabilities.static_);
        assert_eq!(tabular.named("m").capabilities.model, "m");
        let some = Capabilities::nothing("x")
            .also(&[("temporal", true)])
            .unwrap();
        assert!(some.temporal && !some.static_);
        assert_eq!(
            Capabilities::default()
                .also(&[("b", true), ("a", true)])
                .unwrap_err(),
            "not a capability: ['a', 'b']"
        );
        assert!(Profile::by_name("relational").is_none());
    }
}
