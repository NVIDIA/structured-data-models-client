// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The case loader: `contract/cases/*.json`, in the format of
//! `contract/case.schema.json` (`sdm-case/1`). Every key of the format is
//! read, an unknown one refused, so the loader cannot drift from the schema
//! the Python runner validates against.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::de::Error as _;
use serde::{Deserialize, Deserializer};
use serde_json::Value;
use serde_json::value::RawValue;
use sha2::{Digest, Sha256};

/// The case format this loader reads.
pub const FORMAT: &str = "sdm-case/1";

/// One contract case.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Case {
    pub format: String,
    pub id: String,
    pub status: Status,
    pub slice: String,
    pub paths: Vec<String>,
    pub routes: Vec<String>,
    #[serde(default)]
    pub seam: Option<String>,
    pub cells: Vec<String>,
    pub input: Input,
    pub until: Until,
    #[serde(default)]
    pub core: Option<CoreInputs>,
    pub expect: Expect,
    #[serde(default)]
    pub register: Option<String>,
    pub why: String,
    pub source: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    Acceptance,
    Diagnostic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Until {
    FromSource,
    Materialize,
    Predict,
}

/// The public call's inputs. The pure-Rust host reads `pql` for S0, and the
/// base and the setup for an operation that reads data (from T1.11b); the
/// public call's arguments are the Python runner's.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Input {
    pub base: String,
    #[serde(default)]
    pub setup: Option<Vec<String>>,
    #[serde(default)]
    pub from_source: Option<FromSource>,
    pub pql: String,
    pub materialize: Materialize,
}

/// Keyword arguments to `Schema.from_source`, each as the case gives it.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FromSource {
    #[serde(default)]
    pub tables: Option<Tables>,
    #[serde(default)]
    pub primary_keys: Option<serde_json::Map<String, Value>>,
    #[serde(default)]
    pub time_columns: Option<serde_json::Map<String, Value>>,
}

/// `from_source`'s tables: a list of names, or names by alias.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum Tables {
    List(Vec<Value>),
    Named(serde_json::Map<String, Value>),
}

/// Keyword arguments to `materialize`.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Materialize {
    pub indices: Vec<Index>,
    #[serde(default)]
    pub indices_as: Option<IndicesAs>,
    #[serde(default)]
    pub anchor_time: Option<String>,
}

/// One entity key: a JSON integer literal of any size, kept as its
/// digits, or text. A number with a fraction or an exponent, `1.0`
/// included, is refused, as the Python loader refuses it; nothing passes
/// through a double.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Index {
    Integer(String),
    Text(String),
}

/// Whether `text` is a JSON integer literal: `-?(0|[1-9][0-9]*)`.
fn is_integer_literal(text: &str) -> bool {
    let digits = text.strip_prefix('-').unwrap_or(text);
    !digits.is_empty()
        && digits.bytes().all(|byte| byte.is_ascii_digit())
        && (digits == "0" || !digits.starts_with('0'))
}

impl<'de> Deserialize<'de> for Index {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Index, D::Error> {
        let raw = Box::<RawValue>::deserialize(deserializer)?;
        let text = raw.get().trim();
        if text.starts_with('"') {
            serde_json::from_str(text)
                .map(Index::Text)
                .map_err(D::Error::custom)
        } else if is_integer_literal(text) {
            Ok(Index::Integer(text.to_string()))
        } else {
            Err(D::Error::custom(
                "an index is a JSON integer literal or a string",
            ))
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IndicesAs {
    Decimal,
}

/// The core operations a case's call starts, with what the host passes.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CoreInputs {
    #[serde(default)]
    pub validate_query: Option<ValidateQueryInput>,
    #[serde(default)]
    pub describe_source: Option<DescribeSourceInput>,
}

/// `describe_source`'s inputs, kept as written: the provider, the source's
/// tables and overrides (the contract's `SourceSpec`) and whether keys are
/// checked. The data are the case's base and setup.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DescribeSourceInput {
    pub provider: Value,
    pub source: Value,
    pub check_keys: bool,
}

/// `validate_query`'s inputs beside the query text, which is `input.pql`:
/// kept as written, so an observation records exactly what the case gave.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ValidateQueryInput {
    pub profile: Value,
    pub schema: Value,
}

/// The reference's outcome.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Expect {
    pub outcome: String,
    pub stage: String,
    #[serde(default)]
    pub error: Option<String>,
    #[serde(default)]
    pub code: Option<String>,
    #[serde(default)]
    pub message: Option<String>,
    #[serde(default)]
    pub today: Option<String>,
}

/// A case with the SHA-256 of its file's bytes.
#[derive(Debug, Clone)]
pub struct Loaded {
    pub case: Case,
    pub digest: String,
}

/// The SHA-256 of `bytes`, as lowercase hexadecimal.
pub fn sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn json_files(directory: &Path, found: &mut Vec<PathBuf>) -> Result<()> {
    let entries =
        std::fs::read_dir(directory).with_context(|| format!("reading {}", directory.display()))?;
    for entry in entries {
        let path = entry?.path();
        if path.is_dir() {
            json_files(&path, found)?;
        } else if path
            .extension()
            .is_some_and(|extension| extension == "json")
        {
            found.push(path);
        }
    }
    Ok(())
}

/// The case files named, a directory standing for every `*.json` under it,
/// in path order.
pub fn case_files(paths: &[PathBuf]) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for path in paths {
        if path.is_dir() {
            let mut found = Vec::new();
            json_files(path, &mut found)?;
            found.sort();
            files.extend(found);
        } else {
            files.push(path.clone());
        }
    }
    Ok(files)
}

/// Every case named, each read whole, its ids unique. The error names the
/// file and the position, never a value read.
pub fn load(paths: &[PathBuf]) -> Result<Vec<Loaded>> {
    let mut loaded = Vec::new();
    let mut ids = BTreeSet::new();
    for path in case_files(paths)? {
        let name = path.display().to_string();
        let bytes = std::fs::read(&path).with_context(|| format!("reading {name}"))?;
        let case: Case = serde_json::from_slice(&bytes).map_err(|error| {
            anyhow::anyhow!(
                "{name}: not a case (line {}, column {})",
                error.line(),
                error.column()
            )
        })?;
        if case.format != FORMAT {
            bail!("{name}: not {FORMAT}");
        }
        if !ids.insert(case.id.clone()) {
            bail!("{name}: case id {} repeats", case.id);
        }
        loaded.push(Loaded {
            case,
            digest: sha256(&bytes),
        });
    }
    Ok(loaded)
}
