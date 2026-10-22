// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The observation writer: what the pure-Rust host saw of one case, in the
//! format of `contract/observation.schema.json` (`sdm-observation/1`). A
//! field this host cannot observe is null, never filled in: it has no
//! stage, exception class, warnings section or Python runtime cell. From
//! T1.11e a case whose operations read data has a `driver` section, what
//! the DuckDB provider ran ([`crate::driver`]).

use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result, bail};
use sdm_contract::CONTRACT_VERSION;
use serde_json::{Value, json};

use crate::case::Loaded;
use crate::host::{Answer, Observed, os};

/// The observation format this writer writes.
pub const FORMAT: &str = "sdm-observation/1";

/// What every observation of one run shares.
#[derive(Debug, Clone)]
pub struct Identity {
    pub platform: String,
    pub commit: Value,
    pub duckdb: String,
}

/// The checkout at `directory`: its commit and whether its tree is clean.
pub fn commit(directory: &Path) -> Result<Value> {
    let git = |arguments: &[&str]| -> Result<String> {
        let out = Command::new("git")
            .arg("-C")
            .arg(directory)
            .args(arguments)
            .output()
            .context("running git")?;
        if !out.status.success() {
            bail!("git {} failed", arguments[0]);
        }
        Ok(String::from_utf8(out.stdout)?)
    };
    Ok(json!({
        "sha": git(&["rev-parse", "HEAD"])?.trim(),
        "clean": git(&["status", "--porcelain"])?.trim().is_empty(),
    }))
}

/// The root of the checkout holding `directory`.
pub fn repository(directory: &Path) -> Result<std::path::PathBuf> {
    let out = Command::new("git")
        .arg("-C")
        .arg(directory)
        .args(["rev-parse", "--show-toplevel"])
        .output()
        .context("running git")?;
    if !out.status.success() {
        bail!("git rev-parse failed");
    }
    Ok(String::from_utf8(out.stdout)?.trim().into())
}

/// The DuckDB version this host links, from `SELECT version()`, without
/// its leading `v`.
pub fn duckdb_version() -> Result<String> {
    let connection = duckdb::Connection::open_in_memory()?;
    let version: String = connection.query_row("SELECT version()", [], |row| row.get(0))?;
    Ok(version.trim_start_matches('v').to_string())
}

impl Identity {
    /// This run's identity, for the checkout at `repository`.
    pub fn new(repository: &Path) -> Result<Identity> {
        let arch = match (std::env::consts::OS, std::env::consts::ARCH) {
            ("macos", "aarch64") => "arm64",
            (_, arch) => arch,
        };
        Ok(Identity {
            platform: format!("{}-{arch}", os()),
            commit: commit(repository)?,
            duckdb: duckdb_version()?,
        })
    }
}

/// One observation of `loaded`, as `observed`; `bases` is the SHA-256 of
/// `contract/bases.yaml` when the case's data was built from it.
pub fn write(
    identity: &Identity,
    loaded: &Loaded,
    observed: &Observed,
    bases: Option<&str>,
) -> Value {
    let (result, code) = observed.answer.outcome();
    let (accepted, predicted, error) = match &observed.answer {
        Answer::Validated(accepted) => (Some(*accepted), Value::Null, Value::Null),
        Answer::Described(_) | Answer::Materialized(_) => {
            (observed.accepted, Value::Null, Value::Null)
        }
        Answer::Predicted(answer) => {
            // `core.predicted` holds python-core's shape: its `body` the
            // last call's (T1.36 fix 22) and, where there were several
            // calls, `calls`, every call's body in order (T2.1g); its
            // result's explanation where the core gave one, the JSON text
            // it wrote (T2.16a), as python-core records it.
            let mut predicted = answer.clone();
            if let (Some(explanation), Some(result)) = (
                &observed.explanation,
                predicted.get_mut("result").and_then(Value::as_object_mut),
            ) {
                result.insert("explanation".into(), Value::String(explanation.clone()));
            }
            (observed.accepted, predicted, Value::Null)
        }
        Answer::Error { kind, code, .. } => (
            observed.accepted,
            Value::Null,
            json!({"kind": kind, "code": code}),
        ),
    };
    let materialized = observed.materialized.clone().unwrap_or(Value::Null);
    let described = observed.described.clone().unwrap_or(Value::Null);
    let mut observation = json!({
        "format": FORMAT,
        "case": loaded.case.id,
        "identity": {
            "runner": "rust-host",
            "cell": null,
            "platform": identity.platform,
            "commit": identity.commit,
            "reference": null,
            "native": null,
            "native_loaded": null,
            "operations": observed.operations,
            "contract": CONTRACT_VERSION,
            "versions": {"duckdb": identity.duckdb},
            "inputs": {"case": loaded.digest, "bases": bases},
        },
        "outcome": {"result": result, "stage": null, "error": null, "code": code},
        "host": null,
        "core": {
            "inputs": observed.inputs,
            "fetches": observed.fetches,
            "accepted": accepted,
            "described": described,
            "materialized": materialized,
            "predicted": predicted,
            "error": error,
        },
    });
    // What the provider ran, for a case whose operations read data; a
    // case that reads none has no driver section (T1.11e).
    if let Some(driver) = &observed.driver {
        observation["driver"] = driver.clone();
    }
    observation
}

/// The stand-in's verdict on a case of a warehouse leg judged offline
/// (Phase 3 design choice 2), which only the stand-in assigns: the reason
/// code when neither exact replay nor the local engine could answer (a row
/// of the design's blind-spot table), and the codes of answers the local
/// engine gave that the warehouse may give otherwise (B4). The observation
/// carries it as `standin` (`contract/observation.schema.json`), and
/// `compare.py` holds each code to `contract/compare-policy.yaml`'s list.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Standin {
    pub live_only: Option<String>,
    pub flagged: Vec<String>,
}

impl Standin {
    /// The verdict as the observation's `standin` field.
    pub fn to_value(&self) -> Value {
        json!({"live_only": self.live_only, "flagged": self.flagged})
    }
}

/// `observation`, one [`write`] made of a warehouse leg on the stand-in,
/// with the stand-in's verdict.
pub fn with_standin(observation: &mut Value, verdict: &Standin) {
    observation["standin"] = verdict.to_value();
}

#[cfg(test)]
mod standin_tests {
    use super::*;

    #[test]
    fn the_verdict_is_the_schema_s_field() {
        let mut observation = json!({"case": "base.binary@databricks-sql"});
        with_standin(
            &mut observation,
            &Standin {
                live_only: None,
                flagged: vec!["B4".into()],
            },
        );
        assert_eq!(
            observation["standin"],
            json!({"live_only": null, "flagged": ["B4"]})
        );
        with_standin(
            &mut observation,
            &Standin {
                live_only: Some("B1".into()),
                flagged: Vec::new(),
            },
        );
        assert_eq!(observation["standin"]["live_only"], "B1");
    }
}
