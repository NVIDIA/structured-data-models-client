// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The pure-Rust test host: runs a case's core operations through
//! `sdm_host::HostRun`, as a binding does, with no Python in the process.
//!
//! Phase 1a runs S0: `validate_query` with the case's query, schema and
//! profile. Its answer is the case's outcome as the S0 route maps it: an
//! accepted query succeeds, a refused one is `INVALID_REQUEST`, and a core
//! error is a refusal with its own code (design, "Routing seam"). A step
//! other than `Done` is this host's failure: S0 fetches nothing.

use std::collections::BTreeMap;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow, bail};
use sdm_contract::data::AttachmentId;
use sdm_contract::facts::PlatformFacts;
use sdm_contract::ops::{Operation, OperationResult, ProfileSpec, QuerySchema, ValidateQuery};
use sdm_contract::run::StartRun;
use sdm_contract::{CONTRACT_VERSION, ControlMessage, CoreError, HostFacts, Scalar, Step, Unit};
use sdm_host::{Attachment, HostRun, Outgoing, default_root, encode, read};
use serde_json::{Value, json};

use crate::case::Case;

/// What the host saw of one case: the operations it started with their
/// inputs, and the core's answer.
#[derive(Debug, Clone, PartialEq)]
pub struct Observed {
    pub operations: Vec<String>,
    pub inputs: Vec<Value>,
    pub answer: Answer,
}

/// The core's answer to the case's operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Answer {
    /// `validate_query`'s acceptance.
    Validated(bool),
    /// A core error's kind and code, as the contract spells them.
    Error { kind: String, code: String },
}

impl Answer {
    /// The case's outcome as the S0 route maps the answer: `success` or
    /// `refusal`, with the code a refusal carries.
    pub fn outcome(&self) -> (&'static str, Option<String>) {
        match self {
            Answer::Validated(true) => ("success", None),
            Answer::Validated(false) => ("refusal", Some("INVALID_REQUEST".into())),
            Answer::Error { code, .. } => ("refusal", Some(code.clone())),
        }
    }
}

/// This host's facts: the clock now, no local zone, no lookup roots (the
/// core's bundled zone database), and the platform's own `strftime` year
/// padding, which macOS's C library pads and glibc does not.
pub fn facts() -> Result<HostFacts> {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("the clock is before 1970")?;
    Ok(HostFacts {
        now: Scalar::Timestamp {
            unit: Unit::Us,
            zone: Some("UTC".into()),
            value: i64::try_from(now.as_micros()).context("the clock is past int64")?,
        },
        local_zone: None,
        zone_roots: vec![],
        platform: PlatformFacts {
            os: os().into(),
            strftime_pads_year: cfg!(target_os = "macos"),
        },
    })
}

/// The operating system as Python's `sys.platform` names it.
pub fn os() -> &'static str {
    match std::env::consts::OS {
        "macos" => "darwin",
        other => other,
    }
}

fn spelled(value: impl serde::Serialize) -> Result<String> {
    match serde_json::to_value(value)? {
        Value::String(text) => Ok(text),
        _ => bail!("not a string"),
    }
}

fn answer_of(error: &CoreError) -> Result<Answer> {
    Ok(Answer::Error {
        kind: spelled(error.kind)?,
        code: spelled(error.code)?,
    })
}

/// The error an `error` control message carries.
fn error_of(out: &Outgoing) -> Result<Answer> {
    match read(out).map_err(|_| anyhow!("the host wrote an unreadable message"))? {
        ControlMessage::Error { error } => answer_of(&error),
        _ => bail!("the host gave a message that is not an error"),
    }
}

/// Runs `case`'s core operations; an `Err` is this host's failure, never
/// the case's outcome.
pub fn run(case: &Case) -> Result<Observed> {
    let Some(spec) = case
        .core
        .as_ref()
        .and_then(|core| core.validate_query.as_ref())
    else {
        bail!(
            "{}: names no core operation this host runs (S0's validate_query)",
            case.id
        );
    };
    let profile: ProfileSpec = serde_json::from_value(spec.profile.clone()).map_err(|_| {
        anyhow!(
            "{}: core.validate_query.profile is not a ProfileSpec",
            case.id
        )
    })?;
    serde_json::from_value::<QuerySchema>(spec.schema.clone()).map_err(|_| {
        anyhow!(
            "{}: core.validate_query.schema is not a QuerySchema",
            case.id
        )
    })?;
    let start = ControlMessage::Start(Box::new(StartRun {
        contract: CONTRACT_VERSION,
        facts: facts()?,
        operation: Operation::ValidateQuery(ValidateQuery {
            query: AttachmentId(1),
            schema: AttachmentId(2),
            profile,
        }),
    }));
    let attachments = BTreeMap::from([
        (
            AttachmentId(1),
            Attachment::Bytes(serde_json::to_vec(&case.input.pql)?),
        ),
        (
            AttachmentId(2),
            Attachment::Bytes(serde_json::to_vec(&spec.schema)?),
        ),
    ]);
    let input = json!({
        "op": "validate_query",
        "query": case.input.pql,
        "profile": spec.profile,
        "schema": spec.schema,
    });
    let message = encode(&start).map_err(|_| anyhow!("{}: its start does not encode", case.id))?;
    let answer = match HostRun::start(&default_root(), &message, attachments) {
        Err(refused) => error_of(&refused)?,
        Ok(handle) => match handle.next() {
            Err(failed) => error_of(&failed)?,
            Ok(out) => match read(&out).map_err(|_| anyhow!("an unreadable step"))? {
                ControlMessage::Step(Step::Done {
                    result: OperationResult::ValidateQuery(validation),
                }) => Answer::Validated(validation.accepted),
                _ => bail!("{}: the core asked for a step S0 never takes", case.id),
            },
        },
    };
    Ok(Observed {
        operations: vec!["validate_query".into()],
        inputs: vec![input],
        answer,
    })
}
