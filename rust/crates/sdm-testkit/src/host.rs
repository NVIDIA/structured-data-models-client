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
use std::collections::btree_map::Entry;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow, bail};
use std::sync::Arc;

use duckdb::arrow::array::{ArrayRef, RecordBatch};
use duckdb::arrow::compute::concat_batches;
use duckdb::arrow::datatypes::{Field, Schema};
use sdm_contract::data::{AttachmentId, Dialect, FetchPlan, KeySet, ProviderChoice, SqlParam};
use sdm_contract::facts::PlatformFacts;
use sdm_contract::ops::{
    DescribeSource, Dtype, EditSchema, ForecastStep, GivenCandidate, Materialize, Operation,
    OperationResult, Predict, ProfileSpec, QueryColumn, QuerySchema, QueryTable, Ranking,
    RankingPart, RequestBatch, RequestState, SamplerState, Seed, SourceDescription, SourceSpec,
    ValidateQuery,
};
use sdm_contract::run::{FetchOutcome, FetchReply, ModelOutcome, ModelReply, StartRun};
use sdm_contract::{
    CONTRACT_VERSION, ControlMessage, CoreError, Detail, ErrorCode, ErrorKind, HostFacts, Scalar,
    Step, Unit,
};
use sdm_host::{
    ArrowData, Attachment, CompositionRoot, HostRun, Outgoing, default_root, encode, read,
};
use serde_json::{Value, json};

use crate::bases::Bases;
use crate::case::{
    Case, CoreInputs, Database, DescribeSourceInput, Index, MaterializeInput, PredictInput, Until,
    ValidateQueryInput,
};
use crate::driver::{self, DriverLog};
use crate::edits::{self, SchemaEdit};
use crate::memory::Memory;
use crate::replay::{Recording, Replay, Replayed};
use crate::sqlite::Sqlite;
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
    /// `validate_query`'s acceptance where the call went on past it (off
    /// DuckDB, T2.1d): kept apart from the answer, which a later
    /// `materialize` replaces, as python-core records both.
    pub accepted: Option<bool>,
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
    /// What a failed `predict` had answered before it failed, as the core
    /// carried it on its error (`CoreError.answered`, T2.1g): the result's
    /// columns and rows, as a predicted answer's, and the rows those calls
    /// score. In process only; no observation writes it yet.
    pub answered: Option<Value>,
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
        date_literal_range: None,
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

/// What a failed run had answered before it failed, as its error carries
/// it (`CoreError.answered`, T2.1g), the result taken from the run: its
/// columns and rows, as a predicted answer's, and the rows those calls
/// score; none where the error carries none.
fn answered_of(handle: &HostRun, out: &Outgoing) -> Result<Option<Value>> {
    let ControlMessage::Error { error } =
        read(out).map_err(|_| anyhow!("the host wrote an unreadable message"))?
    else {
        bail!("the host gave a message that is not an error");
    };
    let Some(answered) = error.answered else {
        return Ok(None);
    };
    let Ok(Some(Attachment::Arrow(result))) = handle.take_attachment(answered.result) else {
        bail!("the answers an error carries are not attached");
    };
    Ok(Some(json!({
        "result": result_rows(&result)?,
        "rows": answered.rows,
    })))
}

/// A JSON scalar a case wrote, as the Python host sends it on: read as
/// Python's `json.loads` reads it, then written as its `json.dumps` writes
/// it (`_rust/codec.py`'s `attachment`): text as its decoded characters
/// (`"\u0033"` is `"3"`), `true` and `false` as themselves, a whole number
/// as its digits of any size (`-0` is `0`), and a number with a fraction or
/// an exponent as Python's `repr` of the double it reads as (`3.0`,
/// `1e+16`). None for a number past a double's range, which Python reads as
/// an infinity and its host refuses to send (INVALID_REQUEST).
pub fn as_python_sends(written: &str) -> Result<Option<String>> {
    let text = written.trim();
    match text.as_bytes().first() {
        Some(b'"') => {
            let decoded: String = serde_json::from_str(text)?;
            Ok(Some(serde_json::to_string(&decoded)?))
        }
        _ if text == "true" || text == "false" => Ok(Some(text.to_owned())),
        Some(b'-' | b'0'..=b'9') if !text.contains(['.', 'e', 'E']) => {
            let (negative, digits) = match text.strip_prefix('-') {
                Some(digits) => (true, digits),
                None => (false, text),
            };
            if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
                bail!("rows_per_call is not a JSON number: {text}");
            }
            let zero = digits.bytes().all(|b| b == b'0');
            Ok(Some(if negative && !zero {
                text.to_owned()
            } else {
                digits.to_owned()
            }))
        }
        Some(b'-' | b'0'..=b'9') => {
            let read: f64 = text
                .parse()
                .map_err(|_| anyhow!("rows_per_call is not a JSON number: {text}"))?;
            Ok(read.is_finite().then(|| python_repr(read)))
        }
        _ => bail!("rows_per_call is not a JSON scalar: {text}"),
    }
}

/// A finite double as Python's `repr` writes it: the shortest digits that
/// read back as it, positional from `1e-4` to below `1e16` (always with a
/// fraction, `3.0`), and otherwise with an exponent of a sign and at least
/// two digits (`1e+16`, `1e-05`).
fn python_repr(value: f64) -> String {
    let shortest = format!("{value:e}");
    let (mantissa, exponent) = shortest
        .split_once('e')
        .map_or((shortest.as_str(), 0_i32), |(m, e)| {
            (m, e.parse().unwrap_or(0))
        });
    let (negative, mantissa) = match mantissa.strip_prefix('-') {
        Some(rest) => ("-", rest),
        None => ("", mantissa),
    };
    let digits: String = mantissa.chars().filter(char::is_ascii_digit).collect();
    if (-4..16).contains(&exponent) {
        let point = i64::from(exponent) + 1;
        let positional = if point <= 0 {
            format!(
                "0.{}{digits}",
                "0".repeat(usize::try_from(-point).unwrap_or(0))
            )
        } else {
            let point = usize::try_from(point).unwrap_or(0);
            if digits.len() <= point {
                format!("{digits}{}.0", "0".repeat(point - digits.len()))
            } else {
                format!("{}.{}", &digits[..point], &digits[point..])
            }
        };
        format!("{negative}{positional}")
    } else {
        let sign = if exponent < 0 { '-' } else { '+' };
        format!("{negative}{mantissa}e{sign}{:02}", exponent.unsigned_abs())
    }
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
    let (input, answer) = validate(case, spec, &default_root())?;
    Ok(Observed {
        operations: vec!["validate_query".into()],
        inputs: vec![input],
        fetches: vec![],
        described: None,
        materialized: None,
        accepted: None,
        answer,
        warnings: vec![],
        driver: None,
        edits: vec![],
        answered: None,
    })
}

/// `pql` as the PQL library writes a query back (`to_string()`), read
/// through the core's adapter: the Python host sends materialize that text; `pql` as
/// it is where it does not parse (or holds a date literal whose year needs
/// the current one, which no host fact here places).
fn written_back(pql: &str) -> String {
    sdm_host::normalized_query(pql).unwrap_or_else(|| pql.to_owned())
}

/// The query schema of `description`, as the core reads a description for
/// a query (`sdm-core`'s materialize) and the Python host builds one from a
/// schema it described: each table's columns with their data and semantic
/// types, its key where it has one column, its time column, and each link
/// its columns in order.
fn query_schema(description: &SourceDescription) -> QuerySchema {
    QuerySchema {
        tables: description
            .tables
            .iter()
            .map(|t| QueryTable {
                name: t.name.clone(),
                columns: t
                    .columns
                    .iter()
                    .map(|c| QueryColumn {
                        name: c.name.clone(),
                        dtype: query_dtype(c.dtype),
                        stype: c.stype,
                    })
                    .collect(),
                primary_key: match t.primary_key.as_slice() {
                    [one] => Some(one.clone()),
                    _ => None,
                },
                time_column: t.time_column.clone(),
                end_time_column: None,
            })
            .collect(),
        links: description
            .links
            .iter()
            .map(|link| {
                let mut group = link.clone();
                group.sort();
                group
            })
            .collect(),
    }
}

/// A described column's data type as the Python host writes it in a query
/// schema: through the wire's name of it, which the language spells
/// differently for a few (`sdm_client.schema`'s `_WIRE_TO_PQL_DTYPE`):
/// every integer `int`, every float `float`, a date or a time `time`; any
/// other as described.
fn query_dtype(dtype: Dtype) -> Dtype {
    match dtype {
        Dtype::Int | Dtype::Int64 | Dtype::Byte | Dtype::Int16 | Dtype::Int32 => Dtype::Int,
        Dtype::Float | Dtype::Float64 | Dtype::Float32 => Dtype::Float,
        Dtype::Date | Dtype::Time => Dtype::Time,
        other => other,
    }
}

/// The days since 1970-01-01 of `date`, written `YYYY-MM-DD` (Howard
/// Hinnant's `days_from_civil`); `None` for any other text or a day the
/// month does not hold.
fn days_of(date: &str) -> Option<i64> {
    let parts: Vec<&str> = date.split('-').collect();
    let [year, month, day] = parts.as_slice() else {
        return None;
    };
    let digits = |text: &str, width: usize| {
        (text.len() == width && text.bytes().all(|b| b.is_ascii_digit()))
            .then(|| text.parse::<i64>().ok())
            .flatten()
    };
    let (year, month, day) = (digits(year, 4)?, digits(month, 2)?, digits(day, 2)?);
    let leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    let length = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if leap => 29,
        2 => 28,
        _ => return None,
    };
    if !(1..=length).contains(&day) {
        return None;
    }
    let y = if month <= 2 { year - 1 } else { year };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = (month + 9) % 12;
    let doy = (153 * mp + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    Some(era * 146_097 + doe - 719_468)
}

/// The model every case's `predict` names (`core.predict.model`).
const MODEL: &str = "kumo-relational";

/// `case` with every call it makes through its `until` named in its core
/// operations, on `dialect`'s SQL provider, as python-core makes them off
/// DuckDB (T2.1d, `run_python.routes_on`), where a case routes on the core
/// only the calls DuckDB lets it make (S0's validation, a source DuckDB
/// refuses at `from_source`): each operation the case names, its provider
/// on `dialect`; `describe_source` of every table with no key or time
/// column given, keys checked, where the case names none; `materialize`
/// with the case's entities and anchor time, the default seed, where the
/// case names none and goes past `from_source`; and `predict` with the
/// model every case names, where it goes to `predict`. Each derived as the
/// recorders write a case's core inputs: an entity an `i64` where it holds
/// one and its digits otherwise, a text one as text, a date anchor its
/// midnight in nanoseconds. A call the case's input passes more than that
/// to (tables named, decimal entities, options) is refused, as one this
/// derivation does not write.
pub fn through_until(case: &Case, dialect: &str) -> Result<Case> {
    let mut case = case.clone();
    let provider = json!({"provider": "sql", "dialect": dialect});
    let rewrite = |given: &mut Value| {
        if given == &json!({"provider": "sql", "dialect": "duckdb"}) {
            *given = provider.clone();
        }
    };
    let mut core = case.core.take().unwrap_or(CoreInputs {
        validate_query: None,
        describe_source: None,
        materialize: None,
        predict: None,
    });
    if let Some(spec) = core.describe_source.as_mut() {
        rewrite(&mut spec.provider);
    }
    if let Some(spec) = core.materialize.as_mut() {
        rewrite(&mut spec.provider);
    }
    let not_derived = |what: &str| anyhow!("{}: {what}, which no derivation writes", case.id);
    if core.describe_source.is_none() {
        if case.input.from_source.is_some() {
            return Err(not_derived("from_source's arguments"));
        }
        core.describe_source = Some(DescribeSourceInput {
            provider: provider.clone(),
            source: json!({"tables": [], "primary_keys": {}, "time_columns": {}}),
            check_keys: true,
        });
    }
    let given = &case.input.materialize;
    if case.until != Until::FromSource && core.materialize.is_none() {
        // A ranking's candidates and `top_k` are read from the case's
        // input, as on DuckDB (T2.8).
        if given.indices_as.is_some()
            || given.inference_config.is_some()
            || given.rows_per_call.is_some()
        {
            return Err(not_derived("materialize's options"));
        }
        let indices = given
            .indices
            .iter()
            .map(|index| match index {
                Index::Integer(digits) if digits.parse::<i64>().is_ok() => {
                    json!({"t": "i64", "v": digits})
                }
                Index::Integer(digits) | Index::Text(digits) => json!({"t": "utf8", "v": digits}),
            })
            .collect();
        let anchor_time = match &given.anchor_time {
            None => Value::Null,
            Some(date) => {
                let days =
                    days_of(date).ok_or_else(|| not_derived("an anchor time other than a date"))?;
                let nanos = i128::from(days) * 86_400 * 1_000_000_000;
                json!({"t": "ts", "v": nanos.to_string(), "unit": "ns", "zone": null})
            }
        };
        core.materialize = Some(MaterializeInput {
            provider: provider.clone(),
            indices,
            anchor_time,
            seed: None,
        });
    }
    if case.until == Until::Predict && core.predict.is_none() {
        core.predict = Some(PredictInput {
            model: MODEL.to_owned(),
        });
    }
    case.core = Some(core);
    Ok(case)
}

/// S0's `validate_query` with the case's query, schema and profile: what
/// this host sent, as it recorded it, and the core's answer.
fn validate(
    case: &Case,
    spec: &ValidateQueryInput,
    root: &CompositionRoot,
) -> Result<(Value, Answer)> {
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
    let answer = match HostRun::start(root, &message, attachments) {
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
    Ok((input, answer))
}

/// Where a case's fetches are answered: its DuckDB base, through the
/// DuckDB provider; its tables loaded into SQLite, through the dev-only
/// `rusqlite` provider ([`crate::sqlite`], T2.2); or a Postgres server's
/// recorded answers, replayed ([`crate::replay`], KO2-20).
#[derive(Clone, Copy)]
enum Source<'a> {
    Duck(&'a duckdb::Connection),
    Sqlite(&'a Sqlite),
    Replay(&'a Replay<'a>),
    /// The non-SQL provider, through the composition root given.
    Memory(&'a Memory, &'a CompositionRoot),
}

impl Source<'_> {
    /// The provider choice a run over this source names: the case's own on
    /// DuckDB; on a Postgres replay, Postgres's SQL dialect, as the Python
    /// host names it for a Postgres source.
    fn provider(self, given: ProviderChoice) -> ProviderChoice {
        match self {
            Source::Duck(_) => given,
            Source::Sqlite(_) => ProviderChoice::Sql {
                dialect: Dialect::new(Database::Sqlite.name()),
            },
            Source::Replay(_) => ProviderChoice::Sql {
                dialect: Dialect::new(Database::Postgres.name()),
            },
            // The non-SQL provider, a host provider, as it declares itself.
            Source::Memory(..) => ProviderChoice::Host {
                capabilities: Memory::capabilities(),
            },
        }
    }

    /// The composition root a run over this source starts with: the one
    /// the non-SQL provider's run was given (T2.7); `sdm-data`'s for every
    /// other.
    fn root(self) -> CompositionRoot {
        match self {
            Source::Memory(_, root) => root.clone(),
            _ => default_root(),
        }
    }
}

/// Runs `case` over its data loaded into a new SQLite file by the contract
/// loader ([`crate::sqlite::load`], run from `repository`) and read through
/// the dev-only `rusqlite` provider (T2.2): every call the case makes
/// through its `until`, on SQLite's provider ([`through_until`], T2.1d), as
/// python-core makes them over a SQLite source; the record of what ran is
/// the provider's own log of its statements and bindings.
pub fn run_sqlite(case: &Case, bases: &Bases, repository: &std::path::Path) -> Result<Observed> {
    let case = &through_until(case, Database::Sqlite.name())?;
    let Some(spec) = case.core.as_ref().and_then(|c| c.describe_source.as_ref()) else {
        bail!("{}: names no describe_source after its derivation", case.id);
    };
    let mut edits: Vec<SchemaEdit> = Vec::new();
    for edit in case.input.edits.iter().flatten() {
        let Some(made) = SchemaEdit::from_case(edit) else {
            bail!("{}: an edit this host does not convert yet", case.id);
        };
        edits.push(made);
    }
    if bases.get(&case.input.base).is_none() {
        bail!("{}: unknown base", case.id);
    }
    let directory = std::env::temp_dir().join(format!(
        "sdm-case-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or_default()
    ));
    std::fs::create_dir_all(&directory)?;
    let loaded = (|| -> Result<Observed> {
        // What the loader reads of the case: its base and setup.
        let input = directory.join("case.json");
        std::fs::write(
            &input,
            serde_json::to_vec(&json!({
                "id": case.id,
                "input": {
                    "base": case.input.base,
                    "setup": case.input.setup.clone().unwrap_or_default(),
                },
            }))?,
        )?;
        let file = directory.join("case.sqlite");
        crate::sqlite::load(repository, &input, &file)?;
        let source = Sqlite::open(&file)?;
        let mut log = DriverLog::default();
        let mut observed = describe_on(case, spec, Source::Sqlite(&source), &mut log, &edits)?;
        observed.driver = Some(log.record(&source.logged()));
        Ok(observed)
    })();
    std::fs::remove_dir_all(&directory).ok();
    loaded
}

/// Runs `case` on Postgres as [`run`] runs it on DuckDB, each fetch
/// answered from `recording`'s fetches for the case's Postgres key
/// (`<case>@postgres`), replayed in order: a case that reads no data runs
/// as on DuckDB; one whose fetches are not those recorded, or that leaves
/// recorded fetches unasked, is this host's failure. Its driver record is
/// the statements the recording names, each with this host's read of the
/// answer replayed.
pub fn run_replayed(case: &Case, recording: &Recording) -> Result<Observed> {
    // Every call the case makes through its until, as python-core makes
    // them off DuckDB, whose fetches the recording holds (T2.1d).
    let case = &through_until(case, Database::Postgres.name())?;
    let Some(spec) = case.core.as_ref().and_then(|c| c.describe_source.as_ref()) else {
        return run_edited(case, &Bases::default(), &[]);
    };
    let mut edits: Vec<SchemaEdit> = Vec::new();
    for edit in case.input.edits.iter().flatten() {
        let Some(made) = SchemaEdit::from_case(edit) else {
            bail!("{}: an edit this host does not convert yet", case.id);
        };
        edits.push(made);
    }
    let key = format!("{}@{}", case.id, Database::Postgres.name());
    // A diagnostic case's fetch past the recording is answered as the
    // provider's error, so the case is observed (T2.8 fix 4).
    let replay = recording.replay(&key)?;
    let replay = if case.status == crate::case::Status::Diagnostic {
        replay.past_the_end()
    } else {
        replay
    };
    let mut log = DriverLog::default();
    let mut observed = describe_on(case, spec, Source::Replay(&replay), &mut log, &edits)?;
    if !replay.finished() {
        bail!("{key}: the core asked fewer fetches than were recorded: record the case again");
    }
    // Its query log is the statements the recording names, in the order
    // run, as the Postgres driver's own log gave them to python-core.
    observed.driver = Some(log.record(&replay.ran()));
    Ok(observed)
}

/// Runs `case` as [`run`] runs it on DuckDB, over its DuckDB base copied
/// into the minimal non-SQL test provider ([`crate::memory`], T2.7), every
/// run started with `root`: the second data layer's
/// ([`crate::layer::root`]) to run it swapped, or `default_root()`, over
/// `sdm-data`, to run the same provider on the default layer. Every
/// operation the case names, its fetches answered from in-memory Arrow
/// tables, never by a statement; its run records no driver.
pub fn run_memory(case: &Case, bases: &Bases, root: &CompositionRoot) -> Result<Observed> {
    // The base built once on DuckDB and copied out, the connection then
    // closed: the run reads the copy alone.
    let memory = Memory::from_duckdb(&database(case, bases)?)?;
    run_memory_over(case, &memory, root)
}

/// Runs `case` as [`run_memory`] does, over `memory` as given in place of
/// the case's base: a source a test built or changed (T2.21: a column
/// handed over as a host provider hands it, such as Postgres's exact
/// numbers as their marked text).
pub fn run_memory_over(case: &Case, memory: &Memory, root: &CompositionRoot) -> Result<Observed> {
    let Some(spec) = case.core.as_ref().and_then(|c| c.describe_source.as_ref()) else {
        bail!("{}: reads no data (names no describe_source)", case.id);
    };
    let mut edits: Vec<SchemaEdit> = Vec::new();
    for edit in case.input.edits.iter().flatten() {
        let Some(made) = SchemaEdit::from_case(edit) else {
            bail!("{}: an edit this host does not convert yet", case.id);
        };
        edits.push(made);
    }
    describe_on(
        case,
        spec,
        Source::Memory(memory, root),
        &mut DriverLog::default(),
        &edits,
    )
}

/// What the driver ran for a replayed fetch, recorded as run: each
/// statement the recording names on the fetch's own cursor, the last with
/// this host's read of the answer replayed (or its failure), as the DuckDB
/// provider records a fetch it runs.
fn log_replayed(log: &mut DriverLog, replayed: &Replayed<'_>) {
    let handle = log.cursor();
    let Some((last, before)) = replayed.statements.split_last() else {
        return;
    };
    for text in before {
        log.statement(handle, text);
    }
    log.statement(handle, last);
    match &replayed.answer {
        Ok(fetched) => log.answered(&fetched.schema, &fetched.batches),
        Err(_) => log.failed(),
    }
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
    source: Source<'_>,
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
                    // The key sets the step names: a SQL plan's parameters,
                    // or, for a host provider, the request's own.
                    let sets: Vec<&KeySet> = match &step.plan {
                        FetchPlan::Sql { statements, .. } => statements
                            .iter()
                            .flat_map(|s| &s.params)
                            .filter_map(|param| match param {
                                SqlParam::KeySet { keys, .. } => Some(keys),
                                SqlParam::Value { .. } => None,
                            })
                            .collect(),
                        FetchPlan::Host { .. } => Memory::key_sets(&step.request),
                    };
                    let mut keys = BTreeMap::new();
                    for set in sets {
                        if let Entry::Vacant(entry) = keys.entry(set.attachment) {
                            let Ok(Some(Attachment::Arrow(data))) =
                                handle.take_attachment(set.attachment)
                            else {
                                bail!("{}: a key set is not attached", case.id);
                            };
                            entry.insert(duck::Fetched {
                                schema: data.schema,
                                batches: data.batches,
                            });
                        }
                    }
                    let answered = match source {
                        Source::Duck(connection) => {
                            duck::answer_logged(connection, step, &keys, log)
                        }
                        Source::Sqlite(source) => source.answer_logged(step, &keys, log),
                        Source::Replay(replay) => {
                            let replayed = replay.answer(step)?;
                            log_replayed(log, &replayed);
                            replayed.answer
                        }
                        Source::Memory(memory, _) => {
                            let mut sets = BTreeMap::new();
                            for (id, fetched) in &keys {
                                sets.insert(
                                    *id,
                                    concat_batches(&fetched.schema, &fetched.batches)?,
                                );
                            }
                            memory
                                .answer(&step.request, &sets)
                                .map(|batch| duck::Fetched {
                                    schema: batch.schema(),
                                    batches: vec![batch],
                                })
                        }
                    };
                    let (outcome, attachments) = match answered {
                        Ok(fetched) => (
                            FetchOutcome::Data {
                                data: AttachmentId(1),
                                continuation: None,
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
    let mut observed = describe_on(case, spec, Source::Duck(&connection), &mut log, edits)?;
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
    source: Source<'_>,
    log: &mut DriverLog,
    edits: &[SchemaEdit],
) -> Result<Observed> {
    let provider: ProviderChoice = serde_json::from_value(spec.provider.clone()).map_err(|_| {
        anyhow!(
            "{}: core.describe_source.provider is not a ProviderChoice",
            case.id
        )
    })?;
    let provider = source.provider(provider);
    // Off DuckDB a case runs every call through its `until` (T2.1d).
    let off_duckdb = !matches!(
        &provider,
        ProviderChoice::Sql { dialect } if dialect.name() == Database::Duckdb.name()
    );
    let spec_given: SourceSpec = serde_json::from_value(spec.source.clone()).map_err(|_| {
        anyhow!(
            "{}: core.describe_source.source is not a SourceSpec",
            case.id
        )
    })?;
    // What this host sends, as it encodes it, is what it records.
    let source_bytes = serde_json::to_vec(&spec_given)?;
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
        accepted: None,
        answer: Answer::Validated(false),
        warnings: Vec::new(),
        driver: None,
        edits: Vec::new(),
        answered: None,
    };
    let handle = match HostRun::start(&source.root(), &message, attachments) {
        Ok(handle) => handle,
        Err(refused) => {
            observed.answer = error_of(&refused)?;
            return Ok(observed);
        }
    };
    log.stage = "from_source";
    let described = match drive(case, &handle, source, &mut observed, log)? {
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
        match edit_schema(case, &schema, edit, source, &mut observed, log)? {
            Ok(edited) => schema = edited,
            Err(answer) => {
                observed.answer = answer;
                return Ok(observed);
            }
        }
    }
    // A query the client refuses before the core's materialize: the client
    // validates it first, on S0's route by the core's validate_query
    // (design, "Routing seam"), as the reference refuses it before any
    // data is read; so a case that names validate_query beside
    // describe_source runs it after the description, and runs materialize
    // only once it is accepted (refuse.unknown-column, T1.32d; whether or
    // not the case names materialize, carried row G5). On DuckDB it is
    // validated with the schema and profile the case gives; off DuckDB a
    // case runs every call through its `until` (T2.1d, [`through_until`]),
    // and the query is validated on the schema this host described, as
    // the Python host builds its query schema from the source's.
    let materializing = case.core.as_ref().and_then(|c| c.materialize.as_ref());
    if let Some(validating) = case.core.as_ref().and_then(|c| c.validate_query.as_ref()) {
        let described;
        let given = if off_duckdb {
            described = ValidateQueryInput {
                profile: validating.profile.clone(),
                schema: serde_json::to_value(query_schema(&schema))?,
            };
            &described
        } else {
            validating
        };
        let (input, answer) = validate(case, given, &source.root())?;
        observed.operations.push("validate_query".into());
        observed.inputs.push(input);
        let accepted = matches!(answer, Answer::Validated(true));
        observed.answer = answer;
        if !accepted {
            return Ok(observed);
        }
        if off_duckdb || materializing.is_some() {
            observed.accepted = Some(true);
        }
    }
    let Some(spec) = materializing else {
        return Ok(observed);
    };
    // The schema materialize samples is this host's own description, with
    // the user's edits made: never one the case carries (decision E50).
    let Some(state) = materialize(case, spec, &schema, source, &mut observed, log, None)? else {
        return Ok(observed);
    };
    if let Some(spec) = case.core.as_ref().and_then(|c| c.predict.as_ref()) {
        log.stage = "predict";
        predict(case, spec, state, &mut observed, &source.root())?;
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
    source: Source<'_>,
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
    let handle = match HostRun::start(&source.root(), &message, attachments) {
        Ok(handle) => handle,
        Err(refused) => return Ok(Err(error_of(&refused)?)),
    };
    match drive(case, &handle, source, observed, log)? {
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

/// A ranking's candidates as the case names them, each the contract scalar
/// the Python host writes of the same value, as a given value (T2.8): an integer an `i64`, a
/// `u64` past it, and past both a decimal of scale 0; text `utf8`.
fn candidates_of(given: &[Index]) -> Result<Vec<GivenCandidate>> {
    given
        .iter()
        .map(|index| -> Result<Scalar> {
            match index {
                Index::Text(text) => Ok(Scalar::Utf8(text.clone())),
                Index::Integer(digits) => {
                    if let Ok(value) = digits.parse::<i64>() {
                        return Ok(Scalar::Int64(value));
                    }
                    if let Ok(value) = digits.parse::<u64>() {
                        return Ok(Scalar::UInt64(value));
                    }
                    let value: i128 = digits
                        .parse()
                        .map_err(|_| anyhow!("a candidate past 38 digits, which no case writes"))?;
                    let precision = u8::try_from(digits.trim_start_matches('-').len())?;
                    Ok(Scalar::Decimal128 {
                        precision,
                        scale: 0,
                        value,
                    })
                }
            }
        })
        .map(|scalar| scalar.map(GivenCandidate::Value))
        .collect()
}

/// One Arrow column of contract scalars, all of one type, nulls among
/// them; a column of nulls alone is Arrow's null type. This host hands the
/// core the entities as the case types them, each scalar's type its own
/// (T1.32a: the core casts them to the key's type). None at all is a
/// column of the null type and no rows, which the core refuses as naming
/// no entity (`empty.indices`, T1.32d).
fn column_of(values: &[Scalar]) -> Result<ArrayRef> {
    use duckdb::arrow::array::NullArray;
    use duckdb::arrow::array::builder::{
        BinaryBuilder, BooleanBuilder, Date32Builder, Decimal128Builder, Decimal256Builder,
        Float32Builder, Float64Builder, Int8Builder, Int16Builder, Int32Builder, Int64Builder,
        StringBuilder, TimestampMicrosecondBuilder, TimestampNanosecondBuilder, UInt8Builder,
        UInt16Builder, UInt32Builder, UInt64Builder,
    };
    let Some(first) = values.iter().find(|v| !matches!(v, Scalar::Null)) else {
        return Ok(Arc::new(NullArray::new(values.len())));
    };
    let mixed = || anyhow!("the indices are not of one type");
    macro_rules! column {
        ($builder:expr, $pattern:pat => $value:expr) => {{
            let mut built = $builder;
            for value in values {
                match value {
                    Scalar::Null => built.append_null(),
                    $pattern => built.append_value($value),
                    _ => return Err(mixed()),
                }
            }
            Arc::new(built.finish()) as ArrayRef
        }};
    }
    Ok(match first {
        Scalar::Null => unreachable!("a non-null scalar was found"),
        Scalar::Bool(_) => column!(BooleanBuilder::new(), Scalar::Bool(v) => *v),
        Scalar::Int8(_) => column!(Int8Builder::new(), Scalar::Int8(v) => *v),
        Scalar::Int16(_) => column!(Int16Builder::new(), Scalar::Int16(v) => *v),
        Scalar::Int32(_) => column!(Int32Builder::new(), Scalar::Int32(v) => *v),
        Scalar::Int64(_) => column!(Int64Builder::new(), Scalar::Int64(v) => *v),
        Scalar::UInt8(_) => column!(UInt8Builder::new(), Scalar::UInt8(v) => *v),
        Scalar::UInt16(_) => column!(UInt16Builder::new(), Scalar::UInt16(v) => *v),
        Scalar::UInt32(_) => column!(UInt32Builder::new(), Scalar::UInt32(v) => *v),
        Scalar::UInt64(_) => column!(UInt64Builder::new(), Scalar::UInt64(v) => *v),
        Scalar::Float32(_) => column!(Float32Builder::new(), Scalar::Float32(v) => *v),
        Scalar::Float64(_) => column!(Float64Builder::new(), Scalar::Float64(v) => *v),
        Scalar::Utf8(_) => column!(StringBuilder::new(), Scalar::Utf8(v) => v),
        Scalar::Binary(_) => column!(BinaryBuilder::new(), Scalar::Binary(v) => v),
        Scalar::Date32(_) => column!(Date32Builder::new(), Scalar::Date32(v) => *v),
        Scalar::Decimal128 {
            precision, scale, ..
        } => {
            let (precision, scale) = (*precision, *scale);
            let mut built = Decimal128Builder::new();
            for value in values {
                match value {
                    Scalar::Null => built.append_null(),
                    Scalar::Decimal128 {
                        precision: p,
                        scale: s,
                        value,
                    } if (*p, *s) == (precision, scale) => built.append_value(*value),
                    _ => return Err(mixed()),
                }
            }
            Arc::new(built.finish().with_precision_and_scale(precision, scale)?)
        }
        Scalar::Decimal256 {
            precision, scale, ..
        } => {
            let (precision, scale) = (*precision, *scale);
            let mut built = Decimal256Builder::new();
            for value in values {
                match value {
                    Scalar::Null => built.append_null(),
                    Scalar::Decimal256 {
                        precision: p,
                        scale: s,
                        value,
                    } if (*p, *s) == (precision, scale) => built.append_value(*value),
                    _ => return Err(mixed()),
                }
            }
            Arc::new(built.finish().with_precision_and_scale(precision, scale)?)
        }
        Scalar::Timestamp { unit, zone, .. } => {
            let (unit, zone) = (*unit, zone.clone());
            let same = |u: &Unit, z: &Option<String>| *u == unit && *z == zone;
            match unit {
                Unit::Ns => {
                    let mut built = TimestampNanosecondBuilder::new();
                    for value in values {
                        match value {
                            Scalar::Null => built.append_null(),
                            Scalar::Timestamp { unit, zone, value } if same(unit, zone) => {
                                built.append_value(*value)
                            }
                            _ => return Err(mixed()),
                        }
                    }
                    Arc::new(built.finish().with_timezone_opt(zone))
                }
                Unit::Us => {
                    let mut built = TimestampMicrosecondBuilder::new();
                    for value in values {
                        match value {
                            Scalar::Null => built.append_null(),
                            Scalar::Timestamp { unit, zone, value } if same(unit, zone) => {
                                built.append_value(*value)
                            }
                            _ => return Err(mixed()),
                        }
                    }
                    Arc::new(built.finish().with_timezone_opt(zone))
                }
            }
        }
        _ => bail!("the indices are of a type this host does not build"),
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
    /// A forecast's later steps, each its number, payload and query
    /// (T1.36 fix 21).
    pub forecast: Option<Vec<HeldStep>>,
    /// A request asked in several calls (`RequestState.batches`, contract
    /// v26): each later call's payload and the rows it scores, in order.
    pub batches: Option<Vec<HeldBatch>>,
    /// A ranking's state (T2.8), as the core answered it.
    pub ranking: Option<HeldRanking>,
}

/// A ranking's state as this host holds it: how many candidates a row
/// keeps, the first candidate by its position and as the result names it,
/// and each later
/// question's candidate, by its position and as the result names it, and
/// its payload's and query's bytes.
pub struct HeldRanking {
    pub top_k: Option<u32>,
    pub candidate: u32,
    pub value: String,
    pub parts: Vec<(u32, String, Vec<u8>, Vec<u8>)>,
}

/// A later call of a request asked in several, as this host holds it: its
/// payload's bytes and the rows it scores.
pub type HeldBatch = (Vec<u8>, u64);

/// A forecast's later step as this host holds it: its number, and its
/// payload's and query's bytes.
pub type HeldStep = (u32, Vec<u8>, Vec<u8>);

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
    source: Source<'_>,
    observed: &mut Observed,
    log: &mut DriverLog,
    sampler: Option<&mut Option<HeldSampler>>,
) -> Result<Option<HeldState>> {
    let Some((handle, driven)) = start_materialize(
        case,
        spec,
        schema,
        source,
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
            let mut forecast = None;
            if let Some(steps) = &state.forecast {
                let mut later = Vec::new();
                for step in steps {
                    later.push((step.step, bytes(step.payload)?, bytes(step.query)?));
                }
                forecast = Some(later);
            }
            let mut batches = None;
            if let Some(calls) = &state.batches {
                let mut later = Vec::new();
                for call in calls {
                    later.push((bytes(call.payload)?, call.rows));
                }
                batches = Some(later);
            }
            let mut ranking = None;
            if let Some(ranked) = &state.ranking {
                let mut parts = Vec::new();
                for part in &ranked.parts {
                    parts.push((
                        part.candidate,
                        part.value.clone(),
                        bytes(part.payload)?,
                        bytes(part.query)?,
                    ));
                }
                ranking = Some(HeldRanking {
                    top_k: ranked.top_k,
                    candidate: ranked.candidate,
                    value: ranked.value.clone(),
                    parts,
                });
            }
            let held = HeldState {
                version: state.version,
                payload: bytes(state.payload)?,
                query: bytes(state.query)?,
                forecast,
                batches,
                ranking,
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
    source: Source<'_>,
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
    let provider = source.provider(provider);
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
    let field = Field::new("indices", column.data_type().clone(), true);
    let batch = RecordBatch::try_new(Arc::new(Schema::new(vec![field])), vec![column])?;
    // What this host sends, as it encodes it, is what it records: the
    // description the core last answered, and the schema as this host
    // holds it (decision E67).
    let schema_bytes = serde_json::to_vec(schema)?;
    let held_bytes = serde_json::to_vec(&edits::held(schema))?;
    // Off DuckDB the call is the Python host's on the core route (T2.1d):
    // it sends the validated query as the PQL library writes it back
    // (`to_string()`), which `sdm-pql` writes the same way; a query it does
    // not parse is sent as the case gives it.
    let off_duckdb = !matches!(
        &provider,
        ProviderChoice::Sql { dialect } if dialect.name() == Database::Duckdb.name()
    );
    let query = if off_duckdb {
        written_back(&case.input.pql)
    } else {
        case.input.pql.clone()
    };
    let query_bytes = serde_json::to_vec(&query)?;
    // The call's option values, as the client passes them (T1.28a's
    // `Materialize.values`, `_rust/materialize.py`): the context size, none
    // here, since the case format carries none, and of the options whose
    // values the core reads, those the case's call gives: its inference
    // options as written (T1.29, for `base.adventureworks`).
    let mut values = serde_json::Map::from_iter([("num_context_samples".to_owned(), Value::Null)]);
    if let Some(options) = &case.input.materialize.inference_config {
        values.insert(
            "inference_config".to_owned(),
            Value::Object(options.clone()),
        );
    }
    // The sampling options the case's call gives (T2.11), as the Python
    // host passes them (`VALUED`), spliced in below as the case wrote them;
    // `context_anchor_time` as that host writes it (a `ts` scalar, or the
    // literal `entity`).
    let given = &case.input.materialize;
    if let Some(context) = &given.context_anchor_time {
        let written = if context == "entity" {
            Value::from("entity")
        } else {
            let ns = iso_date_ns(context)
                .ok_or_else(|| anyhow!("{}: context_anchor_time is not a date", case.id))?;
            json!({"t": "ts", "v": ns.to_string(), "unit": "ns", "zone": null})
        };
        values.insert("context_anchor_time".to_owned(), written);
    }
    // An anchor at each entity's own time, which the Python host passes as
    // `anchor_entity` beside no anchor time.
    let anchor_entity = given.anchor_time.as_deref() == Some("entity");
    let mut values_bytes = serde_json::to_vec(&Value::Object(values))?;
    for (name, value) in [
        ("random_seed", &given.random_seed),
        ("num_hops", &given.num_hops),
        ("num_neighbors", &given.num_neighbors),
        ("max_pq_iterations", &given.max_pq_iterations),
        ("use_prediction_time", &given.use_prediction_time),
        ("return_embeddings", &given.return_embeddings),
        ("exclude_cols_dict", &given.exclude_cols_dict),
    ] {
        if let Some(written) = value {
            values_bytes.pop();
            values_bytes.extend_from_slice(format!(r#","{name}":{}}}"#, written.0).as_bytes());
        }
    }
    // `rows_per_call` as the case wrote it (T2.8c's cases), whatever JSON
    // scalar it is, read as the Python host reads and hands the caller's
    // value over (`as_python_sends`): the core holds its type and size to
    // what the reference takes. Spliced in as text, so a whole number past
    // 64 bits is not read through a double. A value the Python host cannot
    // send as JSON (a number past a double's range, read as an infinity)
    // is refused before the run, as that host refuses it.
    if let Some(written) = &case.input.materialize.rows_per_call {
        let Some(sent) = as_python_sends(&written.0)? else {
            observed.answer = Answer::Error {
                kind: spelled(ErrorKind::InvalidRequest)?,
                code: spelled(ErrorCode::InvalidRequest)?,
                fields: BTreeMap::from([("field".to_owned(), "rows_per_call".to_owned())]),
            };
            return Ok(None);
        };
        values_bytes.pop();
        values_bytes.extend_from_slice(format!(r#","rows_per_call":{sent}}}"#).as_bytes());
    }
    // A ranking's `top_k` as the case wrote it (T2.8), whatever integer it
    // is, as the Python host hands the caller's value over: the core holds
    // it to what the reference takes. Spliced in as text, so an integer
    // past 64 bits is not read through a double.
    if let Some(top_k) = &case.input.materialize.top_k {
        values_bytes.pop();
        values_bytes.extend_from_slice(format!(r#","top_k":{}}}"#, top_k.0).as_bytes());
    }
    // A ranking's candidates, as the Python host hands them over (T2.8).
    let candidates = case
        .input
        .materialize
        .candidates
        .as_ref()
        .map(|given| candidates_of(given))
        .transpose()?;
    // `lag_timesteps` as the case wrote it (T2.9's cases), as `rows_per_call`
    // is, read as the Python host reads and hands the caller's value over
    // (`as_python_sends`): the core holds its type and sign to what the
    // reference's engine takes; one the Python host cannot send as JSON is
    // refused before the run naming the option, as that host refuses it.
    if let Some(written) = &case.input.materialize.lag_timesteps {
        let Some(sent) = as_python_sends(&written.0)? else {
            observed.answer = Answer::Error {
                kind: spelled(ErrorKind::InvalidRequest)?,
                code: spelled(ErrorCode::InvalidRequest)?,
                fields: BTreeMap::from([("field".to_owned(), "lag_timesteps".to_owned())]),
            };
            return Ok(None);
        };
        values_bytes.pop();
        values_bytes.extend_from_slice(format!(r#","lag_timesteps":{sent}}}"#).as_bytes());
    }
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
            candidates: candidates.as_ref().map(|_| AttachmentId(7)),
            anchor_time,
            seed,
            unseeded: false,
            anchor_entity,
            values: Some(AttachmentId(6)),
            options: vec![],
            most_parts: None,
            most_entities: None,
            most_calls: None,
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
        (AttachmentId(6), Attachment::Bytes(values_bytes)),
        (
            AttachmentId(3),
            Attachment::Arrow(ArrowData::new(batch.schema(), vec![batch])),
        ),
    ]);
    if let Some(Some((_, memo))) = sampler.as_ref().map(|held| held.as_ref()) {
        attachments.insert(AttachmentId(4), Attachment::Bytes(memo.clone()));
    }
    if let Some(candidates) = &candidates {
        attachments.insert(
            AttachmentId(7),
            Attachment::Bytes(serde_json::to_vec(candidates)?),
        );
    }
    let handle = match HostRun::start(&source.root(), &message, attachments) {
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
    let driven = drive(case, &handle, source, observed, log)?;
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
    let mut observed = describe_on(
        &described_only,
        spec,
        Source::Duck(&connection),
        &mut log,
        &[],
    )?;
    let Answer::Described(described) = &observed.answer else {
        return Ok((observed, None));
    };
    let schema: SourceDescription = serde_json::from_value(described.clone())
        .map_err(|_| anyhow!("{}: the description is not a SourceDescription", case.id))?;
    let driven = start_materialize(
        case,
        materialize_spec,
        &schema,
        Source::Duck(&connection),
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
    let mut first = describe_on(
        &describing,
        describe_spec,
        Source::Duck(&connection),
        &mut log,
        &[],
    )?;
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
            accepted: None,
            answer: Answer::Validated(false),
            warnings: Vec::new(),
            driver: None,
            edits: Vec::new(),
            answered: None,
        };
        materialize(
            case,
            spec,
            &described,
            Source::Duck(&connection),
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
/// and text as `{"utf8": text}`; from T1.28b, which lays out a
/// classification's answers, a boolean as `{"bool": value}` and a class's
/// probabilities as `{"map": [[class, its bits], ...]}`, in order; from
/// T2.8, which lays out a ranking's `scores`, `rankings` and `candidates`
/// (contract v27),
/// a list as `{"list": [cell, ...]}`, each member a cell as above.
pub fn result_rows(data: &ArrowData) -> Result<Value> {
    use duckdb::arrow::array::{
        Array, BooleanArray, Float64Array, Int64Array, LargeListArray, ListArray, MapArray,
        StringArray,
    };
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
                DataType::Boolean => "bool",
                DataType::Map(..) => "map",
                DataType::List(..) | DataType::LargeList(..) => "list",
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
                } else if let Some(a) = column.as_any().downcast_ref::<BooleanArray>() {
                    cells.push(json!({"bool": a.value(row)}));
                } else if let Some(a) = column.as_any().downcast_ref::<MapArray>() {
                    let entries = a.value(row);
                    let (Some(classes), Some(probabilities)) = (
                        entries.column(0).as_any().downcast_ref::<StringArray>(),
                        entries.column(1).as_any().downcast_ref::<Float64Array>(),
                    ) else {
                        bail!("a map column of a type this host does not record");
                    };
                    let pairs: Vec<Value> = (0..entries.len())
                        .map(|i| {
                            json!([
                                classes.value(i),
                                format!("{:016x}", probabilities.value(i).to_bits())
                            ])
                        })
                        .collect();
                    cells.push(json!({"map": pairs}));
                } else if let Some(a) = column.as_any().downcast_ref::<ListArray>() {
                    cells.push(json!({"list": list_cells(a.value(row).as_ref())?}));
                } else if let Some(a) = column.as_any().downcast_ref::<LargeListArray>() {
                    cells.push(json!({"list": list_cells(a.value(row).as_ref())?}));
                } else {
                    bail!("a result column of a type this host does not record");
                }
            }
            rows.push(Value::Array(cells));
        }
    }
    let mut recorded = json!({"columns": columns, "rows": rows});
    // The result's explanation (T2.13), on the schema's metadata as the JSON
    // text the core wrote, recorded as text so the hosts are held to the
    // same bytes (T2.16a); an item's explanation and metadata are the
    // result's own columns, among the rows.
    if let Some(explanation) = data.schema.metadata().get("explanation") {
        recorded["explanation"] = Value::String(explanation.clone());
    }
    Ok(recorded)
}

/// A list cell's members as [`result_rows`] records a cell: an integer,
/// a float's bits, text or a boolean, a missing member `null`.
fn list_cells(values: &dyn duckdb::arrow::array::Array) -> Result<Vec<Value>> {
    use duckdb::arrow::array::{
        BooleanArray, Float64Array, Int64Array, LargeStringArray, StringArray,
    };
    let any = values.as_any();
    (0..values.len())
        .map(|i| {
            if values.is_null(i) {
                Ok(Value::Null)
            } else if let Some(a) = any.downcast_ref::<Int64Array>() {
                Ok(json!({"i64": a.value(i).to_string()}))
            } else if let Some(a) = any.downcast_ref::<Float64Array>() {
                Ok(json!({"f64_bits": format!("{:016x}", a.value(i).to_bits())}))
            } else if let Some(a) = any.downcast_ref::<StringArray>() {
                Ok(json!({"utf8": a.value(i)}))
            } else if let Some(a) = any.downcast_ref::<LargeStringArray>() {
                Ok(json!({"utf8": a.value(i)}))
            } else if let Some(a) = any.downcast_ref::<BooleanArray>() {
                Ok(json!({"bool": a.value(i)}))
            } else {
                bail!("a list member of a type this host does not record")
            }
        })
        .collect()
}

/// `predict` of the state `materialize` answered, handed back as this host
/// held it: its model call answered by the scripted endpoint, in process.
fn predict(
    case: &Case,
    spec: &PredictInput,
    state: HeldState,
    observed: &mut Observed,
    root: &CompositionRoot,
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
                forecast: state.forecast.as_ref().map(|steps| {
                    (0u32..)
                        .zip(steps)
                        .map(|(at, (step, ..))| ForecastStep {
                            step: *step,
                            payload: AttachmentId(3 + 2 * at),
                            query: AttachmentId(4 + 2 * at),
                            // This host's one constant id, for every step.
                            request_id: Some(REQUEST_ID.into()),
                        })
                        .collect()
                }),
                // Each later call after the forecast's steps' attachments
                // (none where there are batches), with this host's one
                // constant id, as each step's.
                batches: state.batches.as_ref().map(|calls| {
                    (0u32..)
                        .zip(calls)
                        .map(|(at, (_, rows))| RequestBatch {
                            payload: AttachmentId(3 + at),
                            rows: *rows,
                            request_id: Some(REQUEST_ID.into()),
                        })
                        .collect()
                }),
                // Each later question after the other attachments (none
                // where there is a ranking), with this host's one constant
                // id, as each step's (T2.8).
                ranking: state.ranking.as_ref().map(|held| Ranking {
                    top_k: held.top_k,
                    candidate: held.candidate,
                    value: held.value.clone(),
                    parts: (0u32..)
                        .zip(&held.parts)
                        .map(|(at, (candidate, value, ..))| RankingPart {
                            candidate: *candidate,
                            value: value.clone(),
                            payload: AttachmentId(3 + 2 * at),
                            query: AttachmentId(4 + 2 * at),
                            request_id: Some(REQUEST_ID.into()),
                        })
                        .collect(),
                }),
            },
            model: spec.model.clone(),
            request_id: REQUEST_ID.into(),
            // The outputs the case's caller named, as the Python host
            // hands them over; none named, the task's default ones, as the
            // client asks when its caller names none (T1.28b).
            outputs: case.input.outputs.clone().unwrap_or_default(),
            default_outputs: case.input.outputs.is_none(),
            // The inference options the case's caller set, which alone the
            // call sends (the reference's 26ecdd28, T2.0c), as the Python
            // host hands over the keys of the request's asked config.
            inference_asked: case
                .input
                .materialize
                .inference_config
                .iter()
                .flat_map(|options| options.keys().cloned())
                .collect(),
        }),
    }));
    let message = encode(&start).map_err(|_| anyhow!("{}: its start does not encode", case.id))?;
    let mut attachments = BTreeMap::from([
        (AttachmentId(1), Attachment::Bytes(state.payload)),
        (AttachmentId(2), Attachment::Bytes(state.query)),
    ]);
    for (at, (_, payload, query)) in (0u32..).zip(state.forecast.unwrap_or_default()) {
        attachments.insert(AttachmentId(3 + 2 * at), Attachment::Bytes(payload));
        attachments.insert(AttachmentId(4 + 2 * at), Attachment::Bytes(query));
    }
    for (at, (payload, _)) in (0u32..).zip(state.batches.unwrap_or_default()) {
        attachments.insert(AttachmentId(3 + at), Attachment::Bytes(payload));
    }
    for (at, (_, _, payload, query)) in
        (0u32..).zip(state.ranking.map(|held| held.parts).unwrap_or_default())
    {
        attachments.insert(AttachmentId(3 + 2 * at), Attachment::Bytes(payload));
        attachments.insert(AttachmentId(4 + 2 * at), Attachment::Bytes(query));
    }
    let handle = match HostRun::start(root, &message, attachments) {
        Ok(handle) => handle,
        Err(refused) => {
            observed.answer = error_of(&refused)?;
            return Ok(());
        }
    };
    // Each model call answered in turn, a forecast's one per step (T1.36
    // fix 21); the body recorded is the last call's, as the Python runner
    // records `core.predicted.body`, and every call's is kept in the
    // answer's `calls`, which the observation leaves out (T1.36 fix 22).
    let mut bodies: Vec<Value> = Vec::new();
    // The scripted endpoint over the calls in order, failing the case's
    // `fail_call` (T2.1a), as the Python runner serves it.
    let mut scripted = endpoint::Endpoint::new(case.input.fail_call);
    let mut next = handle.next();
    take_warnings(&handle, &mut observed.warnings)?;
    loop {
        let call = match &next {
            Err(failed) => {
                observed.answer = error_of(failed)?;
                observed.answered = answered_of(&handle, failed)?;
                return Ok(());
            }
            Ok(out) => match read(out).map_err(|_| anyhow!("an unreadable step"))? {
                ControlMessage::Step(Step::CallModel(call)) => call,
                _ if !bodies.is_empty() => break,
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
        bodies.push(sent);
        // The test's script where one stands in for the rule on this thread
        // (`endpoint::answered_by`), its bytes as they travel; else the
        // scripted endpoint, failing the case's call, and answering beside
        // the rule's prediction what the case tells it to (probabilities,
        // quantiles, an explanation, item metadata, embeddings), as the
        // Python runner serves it (T2.8, T2.9, T2.13, T2.11), by the call's
        // number.
        let (status, answered) = match endpoint::scripted(&body) {
            Some(given) => given,
            None => {
                let call = scripted.calls();
                let told = endpoint::Told {
                    probabilities: case.input.probabilities.unwrap_or(false),
                    quantiles: case.input.quantiles.unwrap_or(false),
                    explanation: case.input.explanation.unwrap_or(false),
                    item_metadata: case.input.item_metadata.unwrap_or(false),
                    embeddings: case.input.embeddings.unwrap_or(false),
                };
                let (status, answered) = scripted.answer("POST", endpoint::PREDICTIONS, &body);
                let (status, answered) =
                    if told != endpoint::Told::default() && case.input.fail_call != Some(call) {
                        endpoint::answer_told("POST", endpoint::PREDICTIONS, &body, call, told)
                    } else {
                        (status, answered)
                    };
                (status, serde_json::to_vec(&answered)?)
            }
        };
        let reply = encode(&ControlMessage::ModelReply(ModelReply {
            id: call.id,
            outcome: ModelOutcome::Response {
                status,
                content_type: "application/json".into(),
                body: AttachmentId(1),
            },
        }))
        .map_err(|_| anyhow!("a reply does not encode"))?;
        let answer = BTreeMap::from([(AttachmentId(1), Attachment::Bytes(answered))]);
        if let Err(failed) = handle.reply(&reply, answer) {
            observed.answer = error_of(&failed)?;
            return Ok(());
        }
        next = handle.next();
        take_warnings(&handle, &mut observed.warnings)?;
    }
    let sent = bodies.last().cloned().unwrap_or(Value::Null);
    observed.answer = match next {
        Err(failed) => {
            observed.answered = answered_of(&handle, &failed)?;
            error_of(&failed)?
        }
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
                let mut answer = json!({
                    "body": sent,
                    "model": predicted.model,
                    "request_id": predicted.request_id,
                    "metadata": serde_json::from_slice::<Value>(&metadata)?,
                    "result": result_rows(&result)?,
                });
                // Every call's body, in order, where there were several (a
                // forecast's, one per step); in process only, the
                // observation keeping `body`, the last call's, as
                // python-core records it.
                if bodies.len() > 1 {
                    answer["calls"] = Value::Array(bodies);
                }
                Answer::Predicted(answer)
            }
            _ => bail!("{}: predict answered another step", case.id),
        },
    };
    Ok(())
}

#[cfg(test)]
mod query_dtype_tests {
    use super::*;

    /// As the Python host writes a query schema's data types: the S0 cases'
    /// pinned schemas (`int`, `float`, `time`, `string`) from the
    /// description's (`int64`, `float64`, `date`, `string`).
    #[test]
    fn a_query_schema_names_a_type_as_the_python_host_does() {
        for (described, written) in [
            (Dtype::Int64, Dtype::Int),
            (Dtype::Int32, Dtype::Int),
            (Dtype::Byte, Dtype::Int),
            (Dtype::Float32, Dtype::Float),
            (Dtype::Float64, Dtype::Float),
            (Dtype::Date, Dtype::Time),
            (Dtype::Time, Dtype::Time),
            (Dtype::String, Dtype::String),
            (Dtype::Bool, Dtype::Bool),
        ] {
            assert_eq!(query_dtype(described), written, "{described:?}");
        }
    }
}

#[cfg(test)]
mod written_back_tests {
    use super::written_back;

    #[test]
    fn a_query_is_written_back_as_the_python_host_sends_it() {
        // s0.boundary.deepest's query, its 400 redundant parentheses as the
        // PQL library's to_string() drops them (python-core's materialize
        // input on SQLite).
        let deepest = format!(
            "PREDICT COUNT(orders.* WHERE {}orders.price > 1{}, 0, 30, days) FOR EACH users.user_id",
            "(".repeat(400),
            ")".repeat(400)
        );
        assert_eq!(
            written_back(&deepest),
            "PREDICT COUNT(orders.* WHERE orders.price > 1, 0, 30, days) FOR EACH users.user_id"
        );
        let plain = "PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id";
        assert_eq!(written_back(plain), plain);
        // A query that does not parse is sent as it is.
        assert_eq!(written_back("PREDICT COUNT("), "PREDICT COUNT(");
    }
}

#[cfg(test)]
mod rows_per_call_tests {
    use super::as_python_sends;

    /// What the Python host sends for each scalar a case may write, as
    /// `json.loads` then `json.dumps` give it (T2.8c, its round 1 review).
    #[test]
    fn rows_per_call_is_sent_as_the_python_host_sends_it() {
        for (written, sent) in [
            (r#""3""#, Some(r#""3""#)),
            (r#""\u0033""#, Some(r#""3""#)),
            ("-0", Some("0")),
            ("0", Some("0")),
            ("-1", Some("-1")),
            ("18446744073709551616", Some("18446744073709551616")),
            ("3.0", Some("3.0")),
            ("2.5", Some("2.5")),
            ("3e0", Some("3.0")),
            ("1E16", Some("1e+16")),
            ("0.00001", Some("1e-05")),
            ("-0.0", Some("-0.0")),
            ("true", Some("true")),
            ("false", Some("false")),
            // Past a double's range: Python reads an infinity, which its
            // host refuses to send.
            ("1e400", None),
            ("-1e400", None),
        ] {
            assert_eq!(
                as_python_sends(written).unwrap().as_deref(),
                sent,
                "{written}"
            );
        }
    }
}

/// A date or time the case writes (`YYYY-MM-DD`, with `THH:MM:SS` or a
/// space before the time), in nanoseconds since the epoch, as
/// `pd.Timestamp` reads it; `None` for any other text.
fn iso_date_ns(text: &str) -> Option<i64> {
    let (date, time) = match text.split_once(['T', ' ']) {
        Some((date, time)) => (date, Some(time)),
        None => (text, None),
    };
    let mut parts = date.splitn(3, '-');
    let year: i64 = parts.next()?.parse().ok()?;
    let month: i64 = parts.next()?.parse().ok()?;
    let day: i64 = parts.next()?.parse().ok()?;
    if !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return None;
    }
    // Days from the civil date (Howard Hinnant's algorithm).
    let y = if month <= 2 { year - 1 } else { year };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = (month + 9) % 12;
    let doy = (153 * mp + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146_097 + doe - 719_468;
    let seconds = match time {
        None => 0,
        Some(time) => {
            let mut hms = time.splitn(3, ':');
            let h: i64 = hms.next()?.parse().ok()?;
            let m: i64 = hms.next()?.parse().ok()?;
            let s: i64 = hms.next().map_or(Some(0), |s| s.parse().ok())?;
            h * 3_600 + m * 60 + s
        }
    };
    (days * 86_400 + seconds).checked_mul(1_000_000_000)
}
