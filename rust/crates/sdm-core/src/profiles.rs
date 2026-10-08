// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Profiles and model capability checks.
//!
//! A profile is a mode and a set of capabilities: the mode is what a query
//! means, the capabilities are what a model will answer, and a capability
//! only ever refuses. The capability checks run on top of a validated query,
//! through the features the PQL adapter reads off it, so they stay the same
//! when the PQL team's crate replaces the interim parser.

use std::collections::BTreeSet;

use sdm_contract::ops::{Capability, PqlDiagnostic, ProfileName, ProfileSpec, Severity, Span};

/// Whether a query defines a task or asks one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Training,
    Inference,
}

/// What a model can answer. Every capability defaults to permitting.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Capabilities {
    /// How diagnostics name the model.
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

impl Capabilities {
    fn permitting(model: &str) -> Self {
        Self {
            model: model.to_owned(),
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

    fn flag(&mut self, capability: Capability) -> &mut bool {
        match capability {
            Capability::Relationships => &mut self.relationships,
            Capability::MultipleTables => &mut self.multiple_tables,
            Capability::Temporal => &mut self.temporal,
            Capability::Static => &mut self.static_,
            Capability::EntityFilters => &mut self.entity_filters,
            Capability::Counterfactuals => &mut self.counterfactuals,
            Capability::DerivedTargets => &mut self.derived_targets,
            Capability::LinkPrediction => &mut self.link_prediction,
            Capability::Forecasting => &mut self.forecasting,
            Capability::KeyColumnsInFilters => &mut self.key_columns_in_filters,
            Capability::TimestampColumnsInFilters => &mut self.timestamp_columns_in_filters,
        }
    }
}

/// A mode and a set of capabilities.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Profile {
    pub mode: Mode,
    pub capabilities: Capabilities,
}

impl Profile {
    /// One of the named profiles.
    pub fn named(name: ProfileName) -> Self {
        let inference = |capabilities| Profile {
            mode: Mode::Inference,
            capabilities,
        };
        match name {
            ProfileName::FineTuning => Profile {
                mode: Mode::Training,
                capabilities: Capabilities::permitting("fine-tuning"),
            },
            ProfileName::Relational => inference(Capabilities::permitting("a relational model")),
            ProfileName::Tabular => inference(Capabilities {
                relationships: false,
                multiple_tables: false,
                temporal: false,
                entity_filters: false,
                counterfactuals: false,
                derived_targets: false,
                link_prediction: false,
                forecasting: false,
                ..Capabilities::permitting("a tabular model")
            }),
            ProfileName::Timeseries => inference(Capabilities {
                relationships: false,
                multiple_tables: false,
                static_: false,
                link_prediction: false,
                ..Capabilities::permitting("a time-series model")
            }),
            ProfileName::Anomaly => inference(Capabilities {
                relationships: false,
                multiple_tables: false,
                temporal: false,
                link_prediction: false,
                forecasting: false,
                ..Capabilities::permitting("an anomaly-detection model")
            }),
        }
    }

    /// The profile a request names: the named profile, naming `model` in
    /// diagnostics if given, with each listed capability set as given.
    pub fn resolve(spec: &ProfileSpec) -> Self {
        let mut profile = Self::named(spec.name);
        if let Some(model) = &spec.model {
            profile.capabilities.model.clone_from(model);
        }
        for (capability, allowed) in &spec.capabilities {
            *profile.capabilities.flag(*capability) = *allowed;
        }
        profile
    }
}

/// What a validated query needs of a model, as the PQL adapter reads it off
/// the query.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct QueryFeatures {
    /// Every table the query reads.
    pub tables: BTreeSet<String>,
    /// The table of the entities predicted for.
    pub entity_table: String,
    /// Whether resolution followed a foreign key.
    pub follows_foreign_key: bool,
    /// Whether the target carries a time range.
    pub temporal_target: bool,
    /// Whether the target is computed rather than a stored column.
    pub derived_target: bool,
    pub entity_filter: bool,
    /// Whether the query has an `ASSUMING` clause.
    pub counterfactual: bool,
    /// Whether the target is a `LIST_DISTINCT` over a foreign key.
    pub ranks_foreign_key: bool,
    pub num_forecasts: u32,
    /// Where the target is in the query text.
    pub target_span: Option<Span>,
}

/// The codes of the capability refusals.
pub mod codes {
    pub const MODEL_SINGLE_TABLE_ONLY: &str = "PQL_MODEL_SINGLE_TABLE_ONLY";
    pub const MODEL_NO_RELATIONSHIPS: &str = "PQL_MODEL_NO_RELATIONSHIPS";
    pub const MODEL_NO_TEMPORAL_TARGET: &str = "PQL_MODEL_NO_TEMPORAL_TARGET";
    pub const MODEL_NO_STATIC_TARGET: &str = "PQL_MODEL_NO_STATIC_TARGET";
    pub const MODEL_NO_DERIVED_TARGET: &str = "PQL_MODEL_NO_DERIVED_TARGET";
    pub const MODEL_NO_ENTITY_FILTER: &str = "PQL_MODEL_NO_ENTITY_FILTER";
    pub const MODEL_NO_COUNTERFACTUAL: &str = "PQL_MODEL_NO_COUNTERFACTUAL";
    pub const MODEL_CANNOT_RANK_FOREIGN_KEY: &str = "PQL_MODEL_CANNOT_RANK_FOREIGN_KEY";
    pub const MODEL_NO_FORECASTING: &str = "PQL_MODEL_NO_FORECASTING";
}

/// `'a'`, `'a' and 'b'`, `'a', 'b' and 'c'`.
fn and_list(names: &[&String]) -> String {
    let quoted: Vec<String> = names.iter().map(|n| format!("'{n}'")).collect();
    match quoted.as_slice() {
        [] => String::new(),
        [one] => one.clone(),
        [rest @ .., last] => format!("{} and {last}", rest.join(", ")),
    }
}

/// One refusal per capability the query needs and the profile lacks, in a
/// fixed order.
pub fn check_capabilities(query: &QueryFeatures, profile: &Profile) -> Vec<PqlDiagnostic> {
    let can = &profile.capabilities;
    let model = &can.model;
    let entity = &query.entity_table;
    let mut out = Vec::new();
    let mut refuse = |code: &str, message: String| {
        out.push(PqlDiagnostic {
            code: code.to_owned(),
            message,
            severity: Severity::Error,
            span: query.target_span,
        });
    };
    if !can.multiple_tables && query.tables.len() > 1 {
        let others: Vec<&String> = query.tables.iter().filter(|t| *t != entity).collect();
        refuse(
            codes::MODEL_SINGLE_TABLE_ONLY,
            format!(
                "This query reads {} as well as '{entity}', and {model} predicts from one table.",
                and_list(&others)
            ),
        );
    } else if !can.relationships && query.follows_foreign_key {
        refuse(
            codes::MODEL_NO_RELATIONSHIPS,
            format!("This query follows a foreign key, and {model} does not read relationships."),
        );
    }
    if query.temporal_target && !can.temporal {
        refuse(
            codes::MODEL_NO_TEMPORAL_TARGET,
            format!("The target carries a time range, and {model} predicts a value as it stands."),
        );
    }
    if !query.temporal_target && !can.static_ {
        refuse(
            codes::MODEL_NO_STATIC_TARGET,
            format!("{model} predicts over a time range, and this target carries none."),
        );
    }
    if !can.derived_targets && query.derived_target {
        refuse(
            codes::MODEL_NO_DERIVED_TARGET,
            format!(
                "The target is computed from the table rather than stored in it, and {model} \
                 predicts a column as it stands."
            ),
        );
    }
    if !can.entity_filters && query.entity_filter {
        refuse(
            codes::MODEL_NO_ENTITY_FILTER,
            format!(
                "This query filters the entities it predicts for, and {model} scores the rows \
                 it is given."
            ),
        );
    }
    if !can.counterfactuals && query.counterfactual {
        refuse(
            codes::MODEL_NO_COUNTERFACTUAL,
            format!(
                "This query asks what would happen under an ASSUMING clause, and {model} scores \
                 the rows it is given as they stand."
            ),
        );
    }
    if !can.link_prediction && query.ranks_foreign_key {
        refuse(
            codes::MODEL_CANNOT_RANK_FOREIGN_KEY,
            format!("This query ranks a foreign key, and {model} does not do link prediction."),
        );
    }
    if !can.forecasting && query.num_forecasts > 1 {
        refuse(
            codes::MODEL_NO_FORECASTING,
            format!(
                "This query asks for {} timeframes, and {model} does not forecast.",
                query.num_forecasts
            ),
        );
    }
    out
}
