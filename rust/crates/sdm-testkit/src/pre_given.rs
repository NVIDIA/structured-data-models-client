// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Cases written before contract 23: `sdm-contract-run --pre-given-specs`.
//!
//! Until T1.32h (contract 23) a case's `core.describe_source.source` wrote
//! each declared key as a bare list of column names and each time column as
//! a bare name or `null`; since then the contract writes both as `Given`
//! (`{"value": ...}` or `"unreadable"`). The host's preflight refuses a spec
//! in the old shape as a runner failure, so a public case left in it is
//! caught (the P1B-12 design's seed 12). A bundle frozen before T1.32h
//! cannot be rewritten, though, and the preflight would stop every run of
//! it at its first declared key or time column. Under the flag, and only
//! under it, each case's spec is read as contract 22's `SourceSpec` and
//! rewritten into the current shape, the one a host writes for the same
//! call today; the host then records the spec it sent, as it always does.

use std::collections::BTreeMap;

use anyhow::{Result, bail};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::case::Case;

/// Contract 22's `SourceSpec`, read only to tell its shape: the fields
/// T1.32h left alone are kept as written and checked by the host's own
/// preflight after the rewrite.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
#[allow(dead_code)]
struct Contract22Spec {
    tables: Value,
    primary_keys: BTreeMap<String, Vec<String>>,
    time_columns: BTreeMap<String, Option<String>>,
    #[serde(default)]
    relationships: Option<Value>,
}

/// Rewrites `case`'s `core.describe_source.source`, written in contract
/// 22's shape, into the current one: each key list and each time column
/// name or `null` becomes `{"value": ...}`; nothing else changes. A case
/// without `describe_source` is left as it is. A spec not in contract
/// 22's shape, one already written as `Given` among them, is refused.
pub fn upgrade(case: &mut Case) -> Result<()> {
    let Some(spec) = case
        .core
        .as_mut()
        .and_then(|core| core.describe_source.as_mut())
    else {
        return Ok(());
    };
    if serde_json::from_value::<Contract22Spec>(spec.source.clone()).is_err() {
        bail!(
            "{}: core.describe_source.source is not a contract-22 SourceSpec",
            case.id
        );
    }
    for field in ["primary_keys", "time_columns"] {
        if let Some(by_table) = spec.source[field].as_object_mut() {
            for value in by_table.values_mut() {
                *value = json!({ "value": value.take() });
            }
        }
    }
    Ok(())
}
