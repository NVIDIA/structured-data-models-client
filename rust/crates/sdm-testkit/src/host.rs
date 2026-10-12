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
//! core asked for, the step trace its fields come from. From T1.21d its
//! `materialize` samples the description its own `describe_source`
//! answered, with the case's edits made ([`crate::edits`]): the case names
//! no schema for it, and its loader refuses one (decision E50).

use std::collections::BTreeMap;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow, bail};
use std::sync::Arc;

use duckdb::arrow::array::{ArrayRef, RecordBatch};
use duckdb::arrow::datatypes::{Field, Schema};
use sdm_contract::data::{AttachmentId, FetchPlan, ProviderChoice, SqlParam};
use sdm_contract::facts::PlatformFacts;
use sdm_contract::ops::{
    DescribeSource, EditSchema, Materialize, Operation, OperationResult, Predict, ProfileSpec,
    QuerySchema, RequestState, SamplerState, Seed, SourceDescription, SourceSpec, ValidateQuery,
};
use sdm_contract::run::{FetchOutcome, FetchReply, ModelOutcome, ModelReply, StartRun};
use sdm_contract::{
    CONTRACT_VERSION, ControlMessage, CoreError, Detail, HostFacts, Scalar, Step, Unit,
};
use sdm_host::{ArrowData, Attachment, HostRun, Outgoing, default_root, encode, read};
use serde_json::{Value, json};

use crate::bases::Bases;
use crate::case::{Case, DescribeSourceInput, MaterializeInput, PredictInput};
use crate::driver::{self, DriverLog};
use crate::edits::{self, SchemaEdit};
use crate::{duck, endpoint};

/// What the host saw of one case: the operations it started with their
/// inputs, each fetch the core asked for (its request and plan, in order),
/// `describe_source`'s description when it gave one, and the core's last
/// answer.
#[derive(Debug, Clone, PartialEq)]
pub struct Observed {
    pub operations: Vec<String>,
    pub inputs: Vec<Value>,
    pub fetches: Vec<Value>,
    pub described: Option<Value>,
    /// `materialize`'s request state, when it gave one: kept apart from
    /// the answer, which a later `predict` replaces.
    pub materialized: Option<Value>,
    pub answer: Answer,
    /// The core's warnings, in the order given, each its code and fields
    /// as the `warnings` control message carries them (T1.23): what the
    /// hosts are held to agree on, each wording them itself.
    pub warnings: Vec<Value>,
    /// What the DuckDB provider ran, and DuckDB's query log, for a case
    /// whose operations read data (the observation's `driver`, T1.11e).
    pub driver: Option<Value>,
    /// What this host passed each `edit_schema` and `materialize`'s held
    /// schema, in order (T1.24b), as the Python runner records them in
    /// `core.edits`: apart from `inputs`, whose shape a gate's checkpoint
    /// fixes, and written by no observation yet.
    pub edits: Vec<Value>,
}

/// The core's answer to the case's operation.
#[derive(Debug, Clone, PartialEq)]
pub enum Answer {
    /// `validate_query`'s acceptance.
    Validated(bool),
    /// `describe_source`'s description, as the JSON it attached.
    Described(Value),
    /// `materialize`'s request state: its version, the query and the
    /// payload as the JSON it attached, each number as its text.
    Materialized(Value),
    /// `predict`'s answer: the call's body as the endpoint received it,
    /// each number as its text, the model and call id the endpoint named,
    /// its metadata, and the result's columns and rows.
    Predicted(Value),
    /// A core error's kind and code, as the contract spells them, and its
    /// text fields by name (`operation` and `rule` for a refusal of a rule
    /// not ported), which the refusal maps read (T1.21b); the observation
    /// records the kind and code.
    Error {
        kind: String,
        code: String,
        fields: BTreeMap<String, String>,
    },
}

impl Answer {
    /// The case's outcome as the route maps the answer: `success` or
    /// `refusal`, with the code a refusal carries.
    pub fn outcome(&self) -> (&'static str, Option<String>) {
        match self {
            Answer::Validated(true)
            | Answer::Described(_)
            | Answer::Materialized(_)
            | Answer::Predicted(_) => ("success", None),
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
    let fields = error
        .fields
        .iter()
        .filter_map(|(name, detail)| match detail {
            Detail::Scalar(Scalar::Utf8(text)) => Some((name.clone(), text.clone())),
            _ => None,
        })
        .collect();
    Ok(Answer::Error {
        kind: spelled(error.kind)?,
        code: spelled(error.code)?,
        fields,
    })
}

/// The error an `error` control message carries.
fn error_of(out: &Outgoing) -> Result<Answer> {
    match read(out).map_err(|_| anyhow!("the host wrote an unreadable message"))? {
        ControlMessage::Error { error } => answer_of(&error),
        _ => bail!("the host gave a message that is not an error"),
    }
}

/// Whether an `error` control message tells the host to drop the sampler
/// state it handed the run (`drop_held`, decision E74).
fn drops_sampler(out: &Outgoing) -> Result<bool> {
    match read(out).map_err(|_| anyhow!("the host wrote an unreadable message"))? {
        ControlMessage::Error { error } => Ok(error
            .drop_held
            .contains(&sdm_contract::HeldState::SamplerState)),
        _ => bail!("the host gave a message that is not an error"),
    }
}

/// Runs `case`'s core operation, building its data from `bases`; an `Err`
/// is this host's failure, never the case's outcome. The case's edits are
/// made to the description before `materialize` samples it.
pub fn run(case: &Case, bases: &Bases) -> Result<Observed> {
    let mut edits: Vec<SchemaEdit> = Vec::new();
    for edit in case.input.edits.iter().flatten() {
        let Some(made) = SchemaEdit::from_case(edit) else {
            bail!("{}: an edit this host does not convert yet", case.id);
        };
        edits.push(made);
    }
    run_edited(case, bases, &edits)
}

/// Runs `case`'s core operation as [`run`] does, with `edits` made to the
/// description in place of the case's own (a refusal map's probe, whose
/// edits may also name a key or the relationships).
pub fn run_edited(case: &Case, bases: &Bases, edits: &[SchemaEdit]) -> Result<Observed> {
    let core = case.core.as_ref();
    if let Some(spec) = core.and_then(|core| core.describe_source.as_ref()) {
        return describe(case, spec, bases, edits);
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
        described: None,
        materialized: None,
        answer,
        warnings: vec![],
        driver: None,
        edits: vec![],
    })
}

/// The case's DuckDB base, built on a fresh connection.
fn database(case: &Case, bases: &Bases) -> Result<duckdb::Connection> {
    let Some(base) = bases.get(&case.input.base) else {
        bail!("{}: unknown base", case.id);
    };
    let connection = duck::open()?;
    for statement in base.iter().chain(case.input.setup.iter().flatten()) {
        connection
            .execute_batch(statement)
            .map_err(|_| anyhow!("{}: its data could not be built", case.id))?;
    }
    Ok(connection)
}

/// The warnings `handle` has given since they were last taken, each its
/// code and fields, added to `into`.
fn take_warnings(handle: &HostRun, into: &mut Vec<Value>) -> Result<()> {
    let Some(out) = handle
        .take_warnings()
        .map_err(|_| anyhow!("the run's warnings could not be taken"))?
    else {
        return Ok(());
    };
    let ControlMessage::Warnings { warnings } =
        read(&out).map_err(|_| anyhow!("unreadable warnings"))?
    else {
        bail!("the run's warnings are another message");
    };
    for warning in warnings {
        into.push(json!({"code": warning.code, "fields": warning.fields}));
    }
    Ok(())
}

/// Runs a started operation to its result, answering each fetch with the
/// DuckDB provider and the key sets the step names, and recording each
/// fetch in `fetches`: the result, or the core's error.
fn drive(
    case: &Case,
    handle: &HostRun,
    connection: &duckdb::Connection,
    observed: &mut Observed,
    log: &mut DriverLog,
) -> Result<std::result::Result<OperationResult, Answer>> {
    loop {
        let next = handle.next();
        take_warnings(handle, &mut observed.warnings)?;
        let out = match next {
            Ok(out) => out,
            Err(failed) => return Ok(Err(error_of(&failed)?)),
        };
        match read(&out).map_err(|_| anyhow!("an unreadable step"))? {
            ControlMessage::Step(Step::Done { result }) => return Ok(Ok(result)),
            ControlMessage::Step(Step::Fetch { requests }) => {
                for step in &requests {
                    observed
                        .fetches
                        .push(json!({"request": step.request, "plan": step.plan}));
                    let mut keys = BTreeMap::new();
                    if let FetchPlan::Sql { statements, .. } = &step.plan {
                        for param in statements.iter().flat_map(|s| &s.params) {
                            if let SqlParam::KeySet { keys: set, .. } = param
                                && !keys.contains_key(&set.attachment)
                            {
                                let Ok(Some(Attachment::Arrow(data))) =
                                    handle.take_attachment(set.attachment)
                                else {
                                    bail!("{}: a key set is not attached", case.id);
                                };
                                keys.insert(
                                    set.attachment,
                                    duck::Fetched {
                                        schema: data.schema,
                                        batches: data.batches,
                                    },
                                );
                            }
                        }
                    }
                    let (outcome, attachments) =
                        match duck::answer_logged(connection, step, &keys, log) {
                            Ok(fetched) => (
                                FetchOutcome::Data {
                                    data: AttachmentId(1),
                                },
                                BTreeMap::from([(
                                    AttachmentId(1),
                                    Attachment::Arrow(ArrowData::new(
                                        fetched.schema,
                                        fetched.batches,
                                    )),
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
                        return Ok(Err(error_of(&failed)?));
                    }
                }
            }
            _ => bail!(
                "{}: the core asked for a step this host never takes",
                case.id
            ),
        }
    }
}

/// `describe_source` over the case's DuckDB base, each fetch answered by
/// the DuckDB provider; then, when the case names it, `materialize` over
/// the same base.
fn describe(
    case: &Case,
    spec: &DescribeSourceInput,
    bases: &Bases,
    edits: &[SchemaEdit],
) -> Result<Observed> {
    let connection = database(case, bases)?;
    // After the setup, as the Python runner turns it on (design, "Driver
    // capture"), so the log holds what the case's operations ran.
    driver::enable_query_log(&connection)?;
    let mut log = DriverLog::default();
    let mut observed = describe_on(case, spec, &connection, &mut log, edits)?;
    let logged = driver::query_log(&connection)?;
    observed.driver = Some(log.record(&logged));
    Ok(observed)
}

/// `describe_source`, then `materialize` and `predict` when the case names
/// them, on `connection`, recording what the provider ran in `log`;
/// `materialize` samples the description `describe_source` answered, with
/// `edits` made.
fn describe_on(
    case: &Case,
    spec: &DescribeSourceInput,
    connection: &duckdb::Connection,
    log: &mut DriverLog,
    edits: &[SchemaEdit],
) -> Result<Observed> {
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
    let mut observed = Observed {
        operations: vec!["describe_source".into()],
        inputs: vec![input],
        fetches: Vec::new(),
        described: None,
        materialized: None,
        answer: Answer::Validated(false),
        warnings: Vec::new(),
        driver: None,
        edits: Vec::new(),
    };
    let handle = match HostRun::start(&default_root(), &message, attachments) {
        Ok(handle) => handle,
        Err(refused) => {
            observed.answer = error_of(&refused)?;
            return Ok(observed);
        }
    };
    log.stage = "from_source";
    let described = match drive(case, &handle, connection, &mut observed, log)? {
        Err(answer) => {
            observed.answer = answer;
            return Ok(observed);
        }
        Ok(OperationResult::DescribeSource(described)) => {
            let Ok(Some(Attachment::Bytes(bytes))) = handle.take_attachment(described.description)
            else {
                bail!("{}: the description is not attached", case.id);
            };
            serde_json::from_slice::<Value>(&bytes)?
        }
        Ok(_) => bail!("{}: describe_source answered another operation", case.id),
    };
    observed.described = Some(described.clone());
    observed.answer = Answer::Described(described.clone());
    // The user's edits, each checked and made by the core's edit_schema
    // (decision E67), finish describing the schema: a refusal of one is the
    // case's outcome, as the Python runner records it at from_source.
    let mut schema: SourceDescription = serde_json::from_value(described)
        .map_err(|_| anyhow!("{}: the description is not a SourceDescription", case.id))?;
    for edit in edits {
        match edit_schema(case, &schema, edit, connection, &mut observed, log)? {
            Ok(edited) => schema = edited,
            Err(answer) => {
                observed.answer = answer;
                return Ok(observed);
            }
        }
    }
    let Some(spec) = case.core.as_ref().and_then(|c| c.materialize.as_ref()) else {
        return Ok(observed);
    };
    // The schema materialize samples is this host's own description, with
    // the user's edits made: never one the case carries (decision E50).
    let Some(state) = materialize(case, spec, &schema, connection, &mut observed, log, None)?
    else {
        return Ok(observed);
    };
    if let Some(spec) = case.core.as_ref().and_then(|c| c.predict.as_ref()) {
        log.stage = "predict";
        predict(case, spec, state, &mut observed)?;
    }
    Ok(observed)
}

/// One edit made to `schema`, the description the core last answered, by
/// the core's `edit_schema` over the schema as this host holds it
/// ([`edits::held`]): the schema it answered, or the core's refusal, the
/// case's answer.
fn edit_schema(
    case: &Case,
    schema: &SourceDescription,
    edit: &SchemaEdit,
    connection: &duckdb::Connection,
    observed: &mut Observed,
    log: &mut DriverLog,
) -> Result<std::result::Result<SourceDescription, Answer>> {
    // What this host sends, as it encodes it, is what it records.
    let schema_bytes = serde_json::to_vec(schema)?;
    let held_bytes = serde_json::to_vec(&edits::held(schema))?;
    let edit_bytes = serde_json::to_vec(&edit.to_contract())?;
    observed.operations.push("edit_schema".into());
    observed.edits.push(json!({
        "op": "edit_schema",
        "schema": serde_json::from_slice::<Value>(&schema_bytes)?,
        "held": serde_json::from_slice::<Value>(&held_bytes)?,
        "edit": serde_json::from_slice::<Value>(&edit_bytes)?,
    }));
    let start = ControlMessage::Start(Box::new(StartRun {
        contract: CONTRACT_VERSION,
        facts: facts()?,
        operation: Operation::EditSchema(EditSchema {
            schema: AttachmentId(1),
            held: Some(AttachmentId(2)),
            edit: AttachmentId(3),
        }),
    }));
    let message = encode(&start).map_err(|_| anyhow!("{}: its start does not encode", case.id))?;
    let attachments = BTreeMap::from([
        (AttachmentId(1), Attachment::Bytes(schema_bytes)),
        (AttachmentId(2), Attachment::Bytes(held_bytes)),
        (AttachmentId(3), Attachment::Bytes(edit_bytes)),
    ]);
    let handle = match HostRun::start(&default_root(), &message, attachments) {
        Ok(handle) => handle,
        Err(refused) => return Ok(Err(error_of(&refused)?)),
    };
    match drive(case, &handle, connection, observed, log)? {
        Err(answer) => Ok(Err(answer)),
        Ok(OperationResult::EditSchema(edited)) => {
            let Ok(Some(Attachment::Bytes(bytes))) = handle.take_attachment(edited.schema) else {
                bail!("{}: the schema edited is not attached", case.id);
            };
            Ok(Ok(serde_json::from_slice(&bytes).map_err(|_| {
                anyhow!("{}: the schema edited is not a SourceDescription", case.id)
            })?))
        }
        Ok(_) => bail!("{}: edit_schema answered another operation", case.id),
    }
}

/// One Arrow column of contract scalars, all of one integer or text type.
fn column_of(values: &[Scalar]) -> Result<ArrayRef> {
    macro_rules! ints {
        ($array:ty, $variant:ident) => {
            Arc::new(<$array>::from_iter_values(
                values
                    .iter()
                    .map(|v| match v {
                        Scalar::$variant(v) => Ok(*v),
                        _ => Err(anyhow!("the indices are not of one type")),
                    })
                    .collect::<Result<Vec<_>>>()?,
            )) as ArrayRef
        };
    }
    use duckdb::arrow::array::{
        Int8Array, Int16Array, Int32Array, Int64Array, StringArray, UInt8Array, UInt16Array,
        UInt32Array, UInt64Array,
    };
    Ok(match values.first() {
        Some(Scalar::Int8(_)) => ints!(Int8Array, Int8),
        Some(Scalar::Int16(_)) => ints!(Int16Array, Int16),
        Some(Scalar::Int32(_)) => ints!(Int32Array, Int32),
        Some(Scalar::Int64(_)) => ints!(Int64Array, Int64),
        Some(Scalar::UInt8(_)) => ints!(UInt8Array, UInt8),
        Some(Scalar::UInt16(_)) => ints!(UInt16Array, UInt16),
        Some(Scalar::UInt32(_)) => ints!(UInt32Array, UInt32),
        Some(Scalar::UInt64(_)) => ints!(UInt64Array, UInt64),
        Some(Scalar::Utf8(_)) => Arc::new(StringArray::from(
            values
                .iter()
                .map(|v| match v {
                    Scalar::Utf8(v) => Ok(v.clone()),
                    _ => Err(anyhow!("the indices are not of one type")),
                })
                .collect::<Result<Vec<_>>>()?,
        )),
        _ => bail!("the indices are integers or text, of one type"),
    })
}

/// A JSON value with each number as the text it was written with, as the
/// Python runner reads a payload (`{"number": text}`), so no value passes
/// through a float: the core writes a float as the shortest text that
/// reads back as its double, which reading and writing it again gives.
pub fn numbers_as_text(value: Value) -> Value {
    match value {
        Value::Number(number) => {
            let text = match (number.as_i64(), number.as_u64(), number.as_f64()) {
                (Some(i), _, _) => i.to_string(),
                (_, Some(u), _) => u.to_string(),
                (_, _, Some(f)) => serde_json::to_string(&f).unwrap_or_default(),
                _ => number.to_string(),
            };
            json!({"number": text})
        }
        Value::Array(items) => Value::Array(items.into_iter().map(numbers_as_text).collect()),
        Value::Object(members) => Value::Object(
            members
                .into_iter()
                .map(|(k, v)| (k, numbers_as_text(v)))
                .collect(),
        ),
        other => other,
    }
}

/// What a `materialize` run answered, as this host holds it until the
/// `predict` run it starts: the state's version, and its payload and query
/// as the bytes the run attached.
pub struct HeldState {
    pub version: u32,
    pub payload: Vec<u8>,
    pub query: Vec<u8>,
}

/// A materializer's sampler state as this host keeps it between runs
/// (decision E52): its version and its memo's bytes, as the run left them.
pub type HeldSampler = (u32, Vec<u8>);

/// `materialize` over the case's DuckDB base, after its `describe_source`,
/// sampling `schema`: the request state it answered, held for `predict`,
/// or none when it answered an error. With `sampler`, the sampler state
/// held there is handed to the run, and the one the run leaves, done or
/// failed, is put there in its place.
fn materialize(
    case: &Case,
    spec: &MaterializeInput,
    schema: &SourceDescription,
    connection: &duckdb::Connection,
    observed: &mut Observed,
    log: &mut DriverLog,
    sampler: Option<&mut Option<HeldSampler>>,
) -> Result<Option<HeldState>> {
    let Some((handle, driven)) = start_materialize(
        case,
        spec,
        schema,
        connection,
        observed,
        log,
        sampler,
        &Operation::Materialize,
    )?
    else {
        return Ok(None);
    };
    match driven {
        Err(answer) => {
            observed.answer = answer;
            Ok(None)
        }
        Ok(OperationResult::Materialize(materialized)) => {
            let state = materialized.state;
            let bytes = |id| -> Result<Vec<u8>> {
                match handle.take_attachment(id) {
                    Ok(Some(Attachment::Bytes(bytes))) => Ok(bytes),
                    _ => bail!("{}: the request state is not attached", case.id),
                }
            };
            let held = HeldState {
                version: state.version,
                payload: bytes(state.payload)?,
                query: bytes(state.query)?,
            };
            let recorded = json!({
                "version": held.version,
                "query": serde_json::from_slice::<Value>(&held.query)?,
                "payload": numbers_as_text(serde_json::from_slice(&held.payload)?),
            });
            observed.materialized = Some(recorded.clone());
            observed.answer = Answer::Materialized(recorded);
            Ok(Some(held))
        }
        Ok(_) => bail!("{}: materialize answered another operation", case.id),
    }
}

/// What a driven run ended with: the handle, and its result or the core's
/// error.
type Driven = (HostRun, std::result::Result<OperationResult, Answer>);

/// Starts the operation `as_op` makes of `case`'s `materialize` over
/// `schema` and drives it to its end: the run and how it ended, or none
/// when the start was refused (the refusal the case's answer).
#[allow(clippy::too_many_arguments)]
fn start_materialize(
    case: &Case,
    spec: &MaterializeInput,
    schema: &SourceDescription,
    connection: &duckdb::Connection,
    observed: &mut Observed,
    log: &mut DriverLog,
    sampler: Option<&mut Option<HeldSampler>>,
    as_op: &dyn Fn(Materialize) -> Operation,
) -> Result<Option<Driven>> {
    log.stage = "materialize";
    let provider: ProviderChoice = serde_json::from_value(spec.provider.clone()).map_err(|_| {
        anyhow!(
            "{}: core.materialize.provider is not a ProviderChoice",
            case.id
        )
    })?;
    let indices: Vec<Scalar> = serde_json::from_value(Value::from(spec.indices.clone()))
        .map_err(|_| anyhow!("{}: core.materialize.indices are not scalars", case.id))?;
    let anchor_time: Option<Scalar> = serde_json::from_value(spec.anchor_time.clone())
        .map_err(|_| anyhow!("{}: core.materialize.anchor_time is not a scalar", case.id))?;
    let seed: Option<Seed> = match &spec.seed {
        None => None,
        Some(text) => Some(
            Seed::try_from(text.clone())
                .map_err(|_| anyhow!("{}: core.materialize.seed is not a seed", case.id))?,
        ),
    };
    let column = column_of(&indices)?;
    let field = Field::new("indices", column.data_type().clone(), false);
    let batch = RecordBatch::try_new(Arc::new(Schema::new(vec![field])), vec![column])?;
    // What this host sends, as it encodes it, is what it records: the
    // description the core last answered, and the schema as this host
    // holds it (decision E67).
    let schema_bytes = serde_json::to_vec(schema)?;
    let held_bytes = serde_json::to_vec(&edits::held(schema))?;
    let query_bytes = serde_json::to_vec(&case.input.pql)?;
    observed.operations.push("materialize".into());
    observed.edits.push(json!({
        "op": "materialize",
        "held": serde_json::from_slice::<Value>(&held_bytes)?,
    }));
    observed.inputs.push(json!({
        "op": "materialize",
        "provider": serde_json::to_value(&provider)?,
        "schema": serde_json::from_slice::<Value>(&schema_bytes)?,
        "query": serde_json::from_slice::<Value>(&query_bytes)?,
        "indices": serde_json::to_value(&indices)?,
        "anchor_time": serde_json::to_value(&anchor_time)?,
        "seed": serde_json::to_value(&seed)?,
    }));
    let start = ControlMessage::Start(Box::new(StartRun {
        contract: CONTRACT_VERSION,
        facts: facts()?,
        operation: as_op(Materialize {
            provider,
            schema: AttachmentId(1),
            held: Some(AttachmentId(5)),
            query: AttachmentId(2),
            indices: Some(AttachmentId(3)),
            anchor_time,
            seed,
            unseeded: false,
            anchor_entity: false,
            num_context_samples: None,
            num_hops: None,
            num_neighbors: None,
            prediction_statistic: None,
            inference_config: None,
            options: vec![],
            sampler_state: sampler
                .as_ref()
                .and_then(|held| held.as_ref())
                .map(|(version, _)| SamplerState {
                    version: *version,
                    memo: AttachmentId(4),
                }),
        }),
    }));
    let message = encode(&start).map_err(|_| anyhow!("{}: its start does not encode", case.id))?;
    let mut attachments = BTreeMap::from([
        (AttachmentId(1), Attachment::Bytes(schema_bytes)),
        (AttachmentId(5), Attachment::Bytes(held_bytes)),
        (AttachmentId(2), Attachment::Bytes(query_bytes)),
        (
            AttachmentId(3),
            Attachment::Arrow(ArrowData::new(batch.schema(), vec![batch])),
        ),
    ]);
    if let Some(Some((_, memo))) = sampler.as_ref().map(|held| held.as_ref()) {
        attachments.insert(AttachmentId(4), Attachment::Bytes(memo.clone()));
    }
    let handle = match HostRun::start(&default_root(), &message, attachments) {
        Ok(handle) => handle,
        Err(refused) => {
            // The core's word on the sampler state handed to it (decision
            // E74): one it says to drop is not handed back again.
            if let Some(held) = sampler
                && drops_sampler(&refused)?
            {
                *held = None;
            }
            observed.answer = error_of(&refused)?;
            return Ok(None);
        }
    };
    let driven = drive(case, &handle, connection, observed, log)?;
    if let Some(held) = sampler {
        keep_sampler(&handle, held)?;
    }
    Ok(Some((handle, driven)))
}

/// `case`'s `describe_source`, then the operation `as_op` makes of its
/// `materialize` in its place, over one DuckDB base, for a case without
/// edits: what the host observed, and the run and how it ended, or none
/// when the case was answered before (T1.27: the test-only `labels`
/// operation, whose result `as_op`'s caller reads).
pub fn materialize_as(
    case: &Case,
    bases: &Bases,
    as_op: &dyn Fn(Materialize) -> Operation,
) -> Result<(Observed, Option<Driven>)> {
    let Some(spec) = case.core.as_ref().and_then(|c| c.describe_source.as_ref()) else {
        bail!("{}: names no describe_source", case.id);
    };
    let Some(materialize_spec) = case.core.as_ref().and_then(|c| c.materialize.as_ref()) else {
        bail!("{}: names no materialize", case.id);
    };
    if case.input.edits.as_ref().is_some_and(|e| !e.is_empty()) {
        bail!("{}: materialize_as takes a case without edits", case.id);
    }
    let mut described_only = case.clone();
    if let Some(core) = described_only.core.as_mut() {
        core.materialize = None;
        core.predict = None;
    }
    let connection = database(case, bases)?;
    let mut log = DriverLog::default();
    let mut observed = describe_on(&described_only, spec, &connection, &mut log, &[])?;
    let Answer::Described(described) = &observed.answer else {
        return Ok((observed, None));
    };
    let schema: SourceDescription = serde_json::from_value(described.clone())
        .map_err(|_| anyhow!("{}: the description is not a SourceDescription", case.id))?;
    let driven = start_materialize(
        case,
        materialize_spec,
        &schema,
        &connection,
        &mut observed,
        &mut log,
        None,
        as_op,
    )?;
    Ok((observed, driven))
}

/// Puts the sampler state `handle`'s run left in `held`, as it was taken.
fn keep_sampler(handle: &HostRun, held: &mut Option<HeldSampler>) -> Result<()> {
    let Some(out) = handle
        .take_sampler_state()
        .map_err(|_| anyhow!("the sampler state could not be taken"))?
    else {
        return Ok(());
    };
    let ControlMessage::SamplerState { state } =
        read(&out).map_err(|_| anyhow!("an unreadable sampler state"))?
    else {
        bail!("the run left a message that is not a sampler state");
    };
    let Ok(Some(Attachment::Bytes(memo))) = handle.take_attachment(state.memo) else {
        bail!("the sampler state's memo is not attached");
    };
    *held = Some((state.version, memo));
    Ok(())
}

/// `case`'s `describe_source`, then its `materialize` `calls` times on one
/// materializer, each run handed the sampler state the one before it left
/// (decision E52), over one DuckDB base: what each call observed, its
/// fetches and warnings, as a binding's materializer makes its calls.
pub fn materialize_repeatedly(case: &Case, bases: &Bases, calls: usize) -> Result<Vec<Observed>> {
    materialize_repeatedly_holding(case, bases, calls, None)
}

/// [`materialize_repeatedly`], the first call handed `held` as the
/// sampler state its materializer holds, as a host would hand back one
/// kept from before.
pub fn materialize_repeatedly_holding(
    case: &Case,
    bases: &Bases,
    calls: usize,
    held: Option<HeldSampler>,
) -> Result<Vec<Observed>> {
    let core = case.core.as_ref();
    let (Some(describe_spec), Some(spec)) = (
        core.and_then(|c| c.describe_source.as_ref()),
        core.and_then(|c| c.materialize.as_ref()),
    ) else {
        bail!("{}: names no describe_source and materialize", case.id);
    };
    let connection = database(case, bases)?;
    let mut log = DriverLog::default();
    // The description alone: the calls below are this function's.
    let mut describing = case.clone();
    if let Some(core) = describing.core.as_mut() {
        core.materialize = None;
        core.predict = None;
    }
    let mut first = describe_on(&describing, describe_spec, &connection, &mut log, &[])?;
    let described: SourceDescription = serde_json::from_value(
        first
            .described
            .take()
            .ok_or_else(|| anyhow!("{}: nothing was described", case.id))?,
    )?;
    let mut held = held;
    let mut seen = Vec::with_capacity(calls);
    for _ in 0..calls {
        let mut observed = Observed {
            operations: Vec::new(),
            inputs: Vec::new(),
            fetches: Vec::new(),
            described: None,
            materialized: None,
            answer: Answer::Validated(false),
            warnings: Vec::new(),
            driver: None,
            edits: Vec::new(),
        };
        materialize(
            case,
            spec,
            &described,
            &connection,
            &mut observed,
            &mut log,
            Some(&mut held),
        )?;
        seen.push(observed);
    }
    Ok(seen)
}

/// The id this host's `predict` requests carry. The reference's client
/// draws one at random for every request; this host draws none, and the
/// comparison leaves its value out under decision E35 (approved by the
/// owner on 2026-10-10; `contract/compare-policy.yaml`).
pub const REQUEST_ID: &str = "sdm-000000000000";

/// One Arrow result as rows of tagged cells, the same for both core
/// runners: an integer as `{"i64": digits}`, a float as its IEEE-754 bits
/// `{"f64_bits": 16 hex digits}`, so no text form of a float is compared,
/// and text as `{"utf8": text}`.
pub fn result_rows(data: &ArrowData) -> Result<Value> {
    use duckdb::arrow::array::{Array, Float64Array, Int64Array, StringArray};
    use duckdb::arrow::datatypes::DataType;
    let columns: Vec<Value> = data
        .schema
        .fields()
        .iter()
        .map(|f| {
            let kind = match f.data_type() {
                DataType::Int64 => "int64",
                DataType::Float64 => "float64",
                DataType::Utf8 => "utf8",
                _ => "other",
            };
            json!({"name": f.name(), "type": kind})
        })
        .collect();
    let mut rows = Vec::new();
    for batch in &data.batches {
        for row in 0..batch.num_rows() {
            let mut cells = Vec::new();
            for column in batch.columns() {
                if column.is_null(row) {
                    cells.push(Value::Null);
                } else if let Some(a) = column.as_any().downcast_ref::<Int64Array>() {
                    cells.push(json!({"i64": a.value(row).to_string()}));
                } else if let Some(a) = column.as_any().downcast_ref::<Float64Array>() {
                    cells.push(json!({"f64_bits": format!("{:016x}", a.value(row).to_bits())}));
                } else if let Some(a) = column.as_any().downcast_ref::<StringArray>() {
                    cells.push(json!({"utf8": a.value(row)}));
                } else {
                    bail!("a result column of a type this host does not record");
                }
            }
            rows.push(Value::Array(cells));
        }
    }
    Ok(json!({"columns": columns, "rows": rows}))
}

/// `predict` of the state `materialize` answered, handed back as this host
/// held it: its model call answered by the scripted endpoint, in process.
fn predict(
    case: &Case,
    spec: &PredictInput,
    state: HeldState,
    observed: &mut Observed,
) -> Result<()> {
    observed.operations.push("predict".into());
    observed.inputs.push(json!({
        "op": "predict",
        "model": spec.model,
        "state": {
            "version": state.version,
            "query": serde_json::from_slice::<Value>(&state.query)?,
            "payload": numbers_as_text(serde_json::from_slice(&state.payload)?),
        },
    }));
    let start = ControlMessage::Start(Box::new(StartRun {
        contract: CONTRACT_VERSION,
        facts: facts()?,
        operation: Operation::Predict(Predict {
            state: RequestState {
                version: state.version,
                payload: AttachmentId(1),
                query: AttachmentId(2),
            },
            model: spec.model.clone(),
            request_id: REQUEST_ID.into(),
            outputs: vec!["prediction".into()],
            inference_asked: vec![],
        }),
    }));
    let message = encode(&start).map_err(|_| anyhow!("{}: its start does not encode", case.id))?;
    let attachments = BTreeMap::from([
        (AttachmentId(1), Attachment::Bytes(state.payload)),
        (AttachmentId(2), Attachment::Bytes(state.query)),
    ]);
    let handle = match HostRun::start(&default_root(), &message, attachments) {
        Ok(handle) => handle,
        Err(refused) => {
            observed.answer = error_of(&refused)?;
            return Ok(());
        }
    };
    let next = handle.next();
    take_warnings(&handle, &mut observed.warnings)?;
    let call = match next {
        Err(failed) => {
            observed.answer = error_of(&failed)?;
            return Ok(());
        }
        Ok(out) => match read(&out).map_err(|_| anyhow!("an unreadable step"))? {
            ControlMessage::Step(Step::CallModel(call)) => call,
            _ => bail!(
                "{}: predict asked for a step other than a model call",
                case.id
            ),
        },
    };
    let Ok(Some(Attachment::Bytes(body))) = handle.take_attachment(call.body) else {
        bail!("{}: the model call's body is not attached", case.id);
    };
    let sent = numbers_as_text(serde_json::from_slice(&body)?);
    // The id this host started predict with is the one the call carries.
    if sent["metadata"]["request_id"] != REQUEST_ID {
        bail!(
            "{}: the call does not carry the request id it was given",
            case.id
        );
    }
    let (status, answered) = endpoint::answer("POST", endpoint::PREDICTIONS, &body);
    let reply = encode(&ControlMessage::ModelReply(ModelReply {
        id: call.id,
        outcome: ModelOutcome::Response {
            status,
            content_type: "application/json".into(),
            body: AttachmentId(1),
        },
    }))
    .map_err(|_| anyhow!("a reply does not encode"))?;
    let answer = BTreeMap::from([(
        AttachmentId(1),
        Attachment::Bytes(serde_json::to_vec(&answered)?),
    )]);
    if let Err(failed) = handle.reply(&reply, answer) {
        observed.answer = error_of(&failed)?;
        return Ok(());
    }
    let next = handle.next();
    take_warnings(&handle, &mut observed.warnings)?;
    observed.answer = match next {
        Err(failed) => error_of(&failed)?,
        Ok(out) => match read(&out).map_err(|_| anyhow!("an unreadable step"))? {
            ControlMessage::Step(Step::Done {
                result: OperationResult::Predict(predicted),
            }) => {
                let Ok(Some(Attachment::Arrow(result))) = handle.take_attachment(predicted.result)
                else {
                    bail!("{}: the result is not attached", case.id);
                };
                let Ok(Some(Attachment::Bytes(metadata))) =
                    handle.take_attachment(predicted.metadata)
                else {
                    bail!("{}: the metadata is not attached", case.id);
                };
                Answer::Predicted(json!({
                    "body": sent,
                    "model": predicted.model,
                    "request_id": predicted.request_id,
                    "metadata": serde_json::from_slice::<Value>(&metadata)?,
                    "result": result_rows(&result)?,
                }))
            }
            _ => bail!("{}: predict answered another step", case.id),
        },
    };
    Ok(())
}
