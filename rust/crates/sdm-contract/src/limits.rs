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

/// One limit a caller may set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Setting {
    /// The setting's name, as a host's argument names it.
    pub name: &'static str,
    /// The environment variable a host reads it from: `SDM_CLIENT_` and the
    /// name in capitals.
    pub env: &'static str,
    /// The value where neither the caller nor the environment names one.
    pub default: u64,
}

/// The most context rows `kumo-relational` reads: the serving layer's
/// `TFM_MAX_TRAIN_ROWS` default.
pub const MAX_RELATIONAL_CONTEXT_ROWS: Setting = Setting {
    name: "max_relational_context_rows",
    env: "SDM_CLIENT_MAX_RELATIONAL_CONTEXT_ROWS",
    default: 10_000,
};

/// The most rows one `kumo-relational` call scores: the serving layer's
/// `TFM_MAX_TEST_ROWS` default. A request of more is asked in several calls.
pub const MAX_RELATIONAL_PREDICT_ROWS: Setting = Setting {
    name: "max_relational_predict_rows",
    env: "SDM_CLIENT_MAX_RELATIONAL_PREDICT_ROWS",
    default: 10_000,
};

/// The most rows one temporal link prediction call scores.
pub const MAX_TEMPORAL_LINK_PREDICT_ROWS: Setting = Setting {
    name: "max_temporal_link_predict_rows",
    env: "SDM_CLIENT_MAX_TEMPORAL_LINK_PREDICT_ROWS",
    default: 200,
};

/// The most context rows `kumo-tabular` reads: the serving layer's
/// `TFM_MAX_TRAIN_ROWS` default.
pub const MAX_TABULAR_CONTEXT_ROWS: Setting = Setting {
    name: "max_tabular_context_rows",
    env: "SDM_CLIENT_MAX_TABULAR_CONTEXT_ROWS",
    default: 10_000,
};

/// The most bytes one call's body takes, as written and sent: the serving
/// layer's `TFM_MAX_PAYLOAD_BYTES` default, 30 MiB. A batch of calls past it
/// is asked as smaller calls; a context past it on its own is refused.
pub const MAX_REQUEST_BYTES: Setting = Setting {
    name: "max_request_bytes",
    env: "SDM_CLIENT_MAX_REQUEST_BYTES",
    default: 30 * 1024 * 1024,
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

/// The limits a run holds a request to, resolved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_setting_has_its_own_name_variable_and_serving_layer_default() {
        let mut names: Vec<&str> = SETTINGS.iter().map(|s| s.name).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), SETTINGS.len());
        for setting in SETTINGS {
            assert_eq!(
                setting.env,
                format!("SDM_CLIENT_{}", setting.name.to_ascii_uppercase())
            );
        }
        assert_eq!(
            Limits::default(),
            Limits {
                max_relational_context_rows: 10_000,
                max_relational_predict_rows: 10_000,
                max_temporal_link_predict_rows: 200,
                max_tabular_context_rows: 10_000,
                max_request_bytes: 31_457_280,
            }
        );
    }
}
