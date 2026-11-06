// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The size limits a client holds a request to before it is sent, each a
//! setting a caller may raise or lower to match the deployment it sends to.
//!
//! Each limit mirrors one the serving layer enforces, so the client refuses
//! or batches first and names the limit, rather than sending a request the
//! endpoint refuses without saying how to meet it. A deployment configures
//! its own (the serving layer's `TFM_MAX_TRAIN_ROWS`, `TFM_MAX_TEST_ROWS`
//! and `TFM_MAX_PAYLOAD_BYTES`), so these are settings with the serving
//! layer's defaults, not constants.
//!
//! [`SETTINGS`] is the one table of them: each setting's name, the
//! environment variable a host reads it from, and its default. A host
//! passes what its caller gave and what its environment holds with every
//! run ([`GivenLimits`]), reading nothing else; the core resolves them,
//! the caller's value before the environment's before the default, and
//! refuses a value that is not a whole number of at least one by name.

use std::collections::BTreeMap;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::error::{CoreError, Detail};

/// The prefix of every setting's environment variable.
pub const ENVIRONMENT_PREFIX: &str = "SDM_CLIENT_";

/// One limit a caller may set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Setting {
    /// The setting's name, as a host's argument names it.
    pub name: &'static str,
    /// The environment variable a host reads it from, the name uppercased
    /// after [`ENVIRONMENT_PREFIX`].
    pub env: &'static str,
    /// The value where neither the caller nor the environment names one.
    pub default: u64,
    /// What the limit holds a request to.
    pub about: &'static str,
}

/// The most context rows `kumo-relational` reads: the serving layer's
/// `TFM_MAX_TRAIN_ROWS` default.
pub const MAX_RELATIONAL_CONTEXT_ROWS: Setting = Setting {
    name: "max_relational_context_rows",
    env: "SDM_CLIENT_MAX_RELATIONAL_CONTEXT_ROWS",
    default: 10_000,
    about: "the most context rows a relational request carries; num_context_samples asks for at most this many",
};

/// The most rows one `kumo-relational` call scores: the serving layer's
/// `TFM_MAX_TEST_ROWS` default. A request of more is asked in several calls.
pub const MAX_RELATIONAL_PREDICT_ROWS: Setting = Setting {
    name: "max_relational_predict_rows",
    env: "SDM_CLIENT_MAX_RELATIONAL_PREDICT_ROWS",
    default: 10_000,
    about: "the most rows one relational call scores; more are asked in several calls",
};

/// The most rows one temporal link prediction call scores.
pub const MAX_TEMPORAL_LINK_PREDICT_ROWS: Setting = Setting {
    name: "max_temporal_link_predict_rows",
    env: "SDM_CLIENT_MAX_TEMPORAL_LINK_PREDICT_ROWS",
    default: 200,
    about: "the most rows one temporal link prediction call scores; more are asked in several calls",
};

/// The most context rows `kumo-tabular` reads: the serving layer's
/// `TFM_MAX_TRAIN_ROWS` default.
pub const MAX_TABULAR_CONTEXT_ROWS: Setting = Setting {
    name: "max_tabular_context_rows",
    env: "SDM_CLIENT_MAX_TABULAR_CONTEXT_ROWS",
    default: 10_000,
    about: "the most context rows a single-table request carries; materialize() samples down to it",
};

/// The most bytes one call's body takes, as written and sent: the serving
/// layer's `TFM_MAX_PAYLOAD_BYTES` default, 30 MiB. A batch of calls past it
/// is asked as smaller calls; a context past it on its own is refused.
pub const MAX_REQUEST_BYTES: Setting = Setting {
    name: "max_request_bytes",
    env: "SDM_CLIENT_MAX_REQUEST_BYTES",
    default: 30 * 1024 * 1024,
    about: "the most bytes one call's body takes; a call past it is split into smaller calls",
};

impl Setting {
    /// `error`, naming this setting and its environment variable, which
    /// its sentence tells the caller to raise (`setting`, `env`).
    pub fn named(&self, error: CoreError) -> CoreError {
        error
            .with_field("setting", Detail::text(self.name))
            .with_field("env", Detail::text(self.env))
    }
}

/// Every setting, in the order a host lists them.
pub const SETTINGS: [Setting; 5] = [
    MAX_RELATIONAL_CONTEXT_ROWS,
    MAX_RELATIONAL_PREDICT_ROWS,
    MAX_TEMPORAL_LINK_PREDICT_ROWS,
    MAX_TABULAR_CONTEXT_ROWS,
    MAX_REQUEST_BYTES,
];

/// The setting named `name`, if there is one.
pub fn setting(name: &str) -> Option<&'static Setting> {
    SETTINGS.iter().find(|setting| setting.name == name)
}

/// What a host passes with every run (contract v72): the values its caller
/// set, by [`Setting::name`], and the text of each setting's environment
/// variable that is set, by [`Setting::env`], each as the host read it.
/// Left out, every limit is its default.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct GivenLimits {
    /// The values the caller set, each a whole number of at least one.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub given: BTreeMap<String, u64>,
    /// The environment variables set, each as its text.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub environment: BTreeMap<String, String>,
}

/// The input of `resolve_limits`, which reads nothing but the run's
/// limits: none.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ResolveLimits {}

/// The limits a run holds a request to, resolved; `resolve_limits` answers
/// them, so a host can say what its caller's settings came to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Limits {
    pub max_relational_context_rows: u64,
    pub max_relational_predict_rows: u64,
    pub max_temporal_link_predict_rows: u64,
    pub max_tabular_context_rows: u64,
    pub max_request_bytes: u64,
}

impl Default for Limits {
    fn default() -> Self {
        Limits {
            max_relational_context_rows: MAX_RELATIONAL_CONTEXT_ROWS.default,
            max_relational_predict_rows: MAX_RELATIONAL_PREDICT_ROWS.default,
            max_temporal_link_predict_rows: MAX_TEMPORAL_LINK_PREDICT_ROWS.default,
            max_tabular_context_rows: MAX_TABULAR_CONTEXT_ROWS.default,
            max_request_bytes: MAX_REQUEST_BYTES.default,
        }
    }
}

impl Limits {
    /// The value of the setting named `name`, if there is one.
    pub fn get(&self, name: &str) -> Option<u64> {
        Some(match name {
            n if n == MAX_RELATIONAL_CONTEXT_ROWS.name => self.max_relational_context_rows,
            n if n == MAX_RELATIONAL_PREDICT_ROWS.name => self.max_relational_predict_rows,
            n if n == MAX_TEMPORAL_LINK_PREDICT_ROWS.name => self.max_temporal_link_predict_rows,
            n if n == MAX_TABULAR_CONTEXT_ROWS.name => self.max_tabular_context_rows,
            n if n == MAX_REQUEST_BYTES.name => self.max_request_bytes,
            _ => return None,
        })
    }

    /// The setting named `name` set to `value`; false where there is none.
    pub fn set(&mut self, name: &str, value: u64) -> bool {
        let slot = match name {
            n if n == MAX_RELATIONAL_CONTEXT_ROWS.name => &mut self.max_relational_context_rows,
            n if n == MAX_RELATIONAL_PREDICT_ROWS.name => &mut self.max_relational_predict_rows,
            n if n == MAX_TEMPORAL_LINK_PREDICT_ROWS.name => {
                &mut self.max_temporal_link_predict_rows
            }
            n if n == MAX_TABULAR_CONTEXT_ROWS.name => &mut self.max_tabular_context_rows,
            n if n == MAX_REQUEST_BYTES.name => &mut self.max_request_bytes,
            _ => return false,
        };
        *slot = value;
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_setting_is_named_once_and_read_from_its_own_variable() {
        let mut names: Vec<&str> = SETTINGS.iter().map(|s| s.name).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), SETTINGS.len());
        for setting in SETTINGS {
            assert_eq!(
                setting.env,
                format!("{ENVIRONMENT_PREFIX}{}", setting.name.to_ascii_uppercase())
            );
            assert!(setting.default >= 1);
            assert_eq!(Limits::default().get(setting.name), Some(setting.default));
            let mut limits = Limits::default();
            assert!(limits.set(setting.name, setting.default + 1));
            assert_eq!(limits.get(setting.name), Some(setting.default + 1));
        }
        assert_eq!(Limits::default().get("max_tables"), None);
        assert!(!Limits::default().set("max_tables", 1));
    }

    #[test]
    fn the_defaults_are_the_serving_layers() {
        let limits = Limits::default();
        assert_eq!(limits.max_relational_context_rows, 10_000);
        assert_eq!(limits.max_relational_predict_rows, 10_000);
        assert_eq!(limits.max_temporal_link_predict_rows, 200);
        assert_eq!(limits.max_tabular_context_rows, 10_000);
        assert_eq!(limits.max_request_bytes, 31_457_280);
    }
}
