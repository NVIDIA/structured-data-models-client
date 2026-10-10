// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The pure-Rust test host: runs a case's core operations through
//! `sdm_host::HostRun`, as a binding does, with no Python in the process.
//!
//! It runs S0, `validate_query` with the case's query, schema and profile,
//! whose answer is the case's outcome as the S0 route maps it: an accepted
//! query succeeds, a refused one is `INVALID_REQUEST`, and a core error is
//! a refusal with its own code (design, "Routing seam"); a step other than
//! `Done` is this host's failure, since S0 fetches nothing. From T1.11b it
//! runs `describe_source` too, over the case's DuckDB base built with
//! `duckdb-rs`, answering each fetch with the DuckDB provider
//! ([`crate::duck`]); its answer is the source's description, or the core's
//! error. Either way it records the operation's inputs and every fetch the
//! core asked for, the step trace its fields come from.

use std::collections::BTreeMap;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow, bail};
use sdm_contract::data::{AttachmentId, ProviderChoice};
use sdm_contract::facts::PlatformFacts;
use sdm_contract::ops::{
    DescribeSource, Operation, OperationResult, ProfileSpec, QuerySchema, SourceSpec, ValidateQuery,
};
use sdm_contract::run::{FetchOutcome, FetchReply, StartRun};
use sdm_contract::{CONTRACT_VERSION, ControlMessage, CoreError, HostFacts, Scalar, Step, Unit};
use sdm_host::{ArrowData, Attachment, HostRun, Outgoing, default_root, encode, read};
use serde_json::{Value, json};

use crate::bases::Bases;
use crate::case::{Case, DescribeSourceInput};
use crate::duck;

/// What the host saw of one case: the operations it started with their
/// inputs, each fetch the core asked for (its request and plan, in order),
/// and the core's answer.
#[derive(Debug, Clone, PartialEq)]
pub struct Observed {
    pub operations: Vec<String>,
    pub inputs: Vec<Value>,
    pub fetches: Vec<Value>,
    pub answer: Answer,
}

/// The core's answer to the case's operation.
#[derive(Debug, Clone, PartialEq)]
pub enum Answer {
    /// `validate_query`'s acceptance.
    Validated(bool),
    /// `describe_source`'s description, as the JSON it attached.
    Described(Value),
    /// A core error's kind and code, as the contract spells them.
    Error { kind: String, code: String },
}

impl Answer {
    /// The case's outcome as the route maps the answer: `success` or
    /// `refusal`, with the code a refusal carries.
    pub fn outcome(&self) -> (&'static str, Option<String>) {
        match self {
            Answer::Validated(true) | Answer::Described(_) => ("success", None),
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

/// Runs `case`'s core operation, building its data from `bases`; an `Err`
/// is this host's failure, never the case's outcome.
pub fn run(case: &Case, bases: &Bases) -> Result<Observed> {
    let core = case.core.as_ref();
    if let Some(spec) = core.and_then(|core| core.describe_source.as_ref()) {
        return describe(case, spec, bases);
    }
    let Some(spec) = core.and_then(|core| core.validate_query.as_ref()) else {
        bail!(
            "{}: names no core operation this host runs (validate_query, describe_source)",
            case.id
        );
    };
    let profile: ProfileSpec = serde_json::from_value(spec.profile.clone()).map_err(|_| {
        anyhow!(
            "{}: core.validate_query.profile is not a ProfileSpec",
            case.id
        )
    })?;
    let schema: QuerySchema = serde_json::from_value(spec.schema.clone()).map_err(|_| {
        anyhow!(
            "{}: core.validate_query.schema is not a QuerySchema",
            case.id
        )
    })?;
    // What this host sends, as it encodes it, is what it records.
    let query_bytes = serde_json::to_vec(&case.input.pql)?;
    let schema_bytes = serde_json::to_vec(&schema)?;
    let input = json!({
        "op": "validate_query",
        "query": serde_json::from_slice::<Value>(&query_bytes)?,
        "profile": serde_json::to_value(&profile)?,
        "schema": serde_json::from_slice::<Value>(&schema_bytes)?,
    });
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
        (AttachmentId(1), Attachment::Bytes(query_bytes)),
        (AttachmentId(2), Attachment::Bytes(schema_bytes)),
    ]);
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
        fetches: vec![],
        answer,
    })
}

/// `describe_source` over the case's DuckDB base, each fetch answered by
/// the DuckDB provider.
fn describe(case: &Case, spec: &DescribeSourceInput, bases: &Bases) -> Result<Observed> {
    let provider: ProviderChoice = serde_json::from_value(spec.provider.clone()).map_err(|_| {
        anyhow!(
            "{}: core.describe_source.provider is not a ProviderChoice",
            case.id
        )
    })?;
    let source: SourceSpec = serde_json::from_value(spec.source.clone()).map_err(|_| {
        anyhow!(
            "{}: core.describe_source.source is not a SourceSpec",
            case.id
        )
    })?;
    let Some(base) = bases.get(&case.input.base) else {
        bail!("{}: unknown base", case.id);
    };
    let connection = duck::open()?;
    for statement in base.iter().chain(case.input.setup.iter().flatten()) {
        connection
            .execute_batch(statement)
            .map_err(|_| anyhow!("{}: its data could not be built", case.id))?;
    }
    // What this host sends, as it encodes it, is what it records.
    let source_bytes = serde_json::to_vec(&source)?;
    let input = json!({
        "op": "describe_source",
        "provider": serde_json::to_value(&provider)?,
        "source": serde_json::from_slice::<Value>(&source_bytes)?,
        "check_keys": spec.check_keys,
    });
    let start = ControlMessage::Start(Box::new(StartRun {
        contract: CONTRACT_VERSION,
        facts: facts()?,
        operation: Operation::DescribeSource(DescribeSource {
            provider,
            source: AttachmentId(1),
            check_keys: spec.check_keys,
        }),
    }));
    let message = encode(&start).map_err(|_| anyhow!("{}: its start does not encode", case.id))?;
    let attachments = BTreeMap::from([(AttachmentId(1), Attachment::Bytes(source_bytes))]);
    let mut fetches = Vec::new();
    let handle = match HostRun::start(&default_root(), &message, attachments) {
        Ok(handle) => handle,
        Err(refused) => {
            return Ok(Observed {
                operations: vec!["describe_source".into()],
                inputs: vec![input],
                fetches,
                answer: error_of(&refused)?,
            });
        }
    };
    let answer = loop {
        let out = match handle.next() {
            Ok(out) => out,
            Err(failed) => break error_of(&failed)?,
        };
        match read(&out).map_err(|_| anyhow!("an unreadable step"))? {
            ControlMessage::Step(Step::Done {
                result: OperationResult::DescribeSource(described),
            }) => {
                let Ok(Some(Attachment::Bytes(bytes))) =
                    handle.take_attachment(described.description)
                else {
                    bail!("{}: the description is not attached", case.id);
                };
                break Answer::Described(serde_json::from_slice(&bytes)?);
            }
            ControlMessage::Step(Step::Fetch { requests }) => {
                let mut refused = None;
                for step in &requests {
                    fetches.push(json!({"request": step.request, "plan": step.plan}));
                    let (outcome, attachments) = match duck::answer(&connection, step) {
                        Ok(fetched) => (
                            FetchOutcome::Data {
                                data: AttachmentId(1),
                            },
                            BTreeMap::from([(
                                AttachmentId(1),
                                Attachment::Arrow(ArrowData::new(fetched.schema, fetched.batches)),
                            )]),
                        ),
                        Err(error) => (FetchOutcome::Error { error }, BTreeMap::new()),
                    };
                    let reply = encode(&ControlMessage::FetchReply(FetchReply {
                        id: step.id,
                        outcome,
                    }))
                    .map_err(|_| anyhow!("a reply does not encode"))?;
                    if let Err(failed) = handle.reply(&reply, attachments) {
                        refused = Some(error_of(&failed)?);
                        break;
                    }
                }
                if let Some(answer) = refused {
                    break answer;
                }
            }
            _ => bail!(
                "{}: the core asked for a step describe_source never takes",
                case.id
            ),
        }
    };
    Ok(Observed {
        operations: vec!["describe_source".into()],
        inputs: vec![input],
        fetches,
        answer,
    })
}
