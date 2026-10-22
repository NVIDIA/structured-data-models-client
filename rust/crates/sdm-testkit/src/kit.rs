// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The provider conformance kit's runner on the pure-Rust host (T2.5,
//! T2.6), `sdm-contract-run --kit DATABASE [--fault F] --out REPORT`.
//!
//! It builds `contract/fixtures/provider-kit/dataset.sql` on DATABASE
//! (DuckDB from the file; SQLite loaded from it by the one contract loader,
//! through `contract/runners/provider_kit.py --load sqlite`, as
//! `--database sqlite` loads a case; `memory`, the minimal non-SQL test
//! provider, T2.7, its tables copied whole from DuckDB's into Arrow), and
//! runs each case of `expected.json` through the core on this host's
//! provider for it ([`crate::duck`], [`crate::sqlite`], [`crate::memory`],
//! the last a host provider, declaring what it declares), as the file's
//! `about` says: the
//! core plans the case's request for the provider (the test-only fetch,
//! which a build with `sdm-host`'s `test-ops` feature carries, as `cargo
//! test` builds this crate), with what the case withholds taken out of
//! what the provider declares and the fallback's limits it declares, and
//! admits the answer. The provider path between the run and the provider
//! pages every answer in pages of `page_rows` rows, each but the last
//! carrying a continuation; fails the page a case's `inject` names, with a
//! provider error retryable as it says; and seeds the fault named, as
//! `expected.json`'s `faults` say. The report and the judgment are the
//! Python runner's (`contract/runners/provider_kit.py`), `host`
//! `rust-host`. The start and the test-only result are read as JSON, so
//! this module builds without the test-only operations, and a build
//! without them refuses the kit's start by name.
//!
//! Postgres is not a provider of this host (its Postgres is a replay of
//! the Python provider's answers, KO2-20): the Python runner runs the kit
//! on the Python host's Postgres provider.

use std::collections::BTreeMap;
use std::collections::btree_map::Entry;
use std::path::Path;
use std::process::{Command, Stdio};

use anyhow::{Context, Result, anyhow, bail};
use duckdb::arrow::array::{Array, ArrayRef, Int64Array, RecordBatch};
use duckdb::arrow::compute::{cast, concat_batches};
use duckdb::arrow::datatypes::{DataType, Field, Schema};
use sdm_contract::data::{AttachmentId, FetchPlan, KeySet, ProviderError, SqlParam};
use sdm_contract::run::{FetchOutcome, FetchReply, FetchStep};
use sdm_contract::{CONTRACT_VERSION, ControlMessage, CoreError, ErrorKind, Step};
use sdm_host::{ArrowData, Attachment, HostRun, Outgoing, default_root, encode, read};
use serde_json::{Value, json};

use crate::driver::DriverLog;
use crate::duck::{self, Fetched};
use crate::host::facts;
use crate::memory::Memory;
use crate::sqlite::Sqlite;

/// The report's format, `expected.json`'s.
pub const FORMAT: &str = "sdm-provider-kit/1";
/// The databases this host runs the kit on.
/// `memory` is the minimal non-SQL test provider ([`crate::memory`],
/// T2.7) over the dataset's tables copied from DuckDB, a host provider.
pub const DATABASES: &[&str] = &["duckdb", "sqlite", "memory"];
/// The faults a run may seed.
pub const FAULTS: &[&str] = &["truncate", "silent_capability"];

/// The kit's directory, under the repository.
const KIT: &str = "contract/fixtures/provider-kit";

/// Where a case's fetches are answered.
enum Source {
    Duck(duckdb::Connection),
    Sqlite(Sqlite),
    Memory(Memory),
}

impl Source {
    fn answer(
        &self,
        step: &FetchStep,
        keys: &BTreeMap<AttachmentId, Fetched>,
    ) -> Result<Fetched, ProviderError> {
        match self {
            Source::Duck(connection) => duck::answer_with(connection, step, keys),
            Source::Sqlite(source) => source.answer_logged(step, keys, &mut DriverLog::default()),
            Source::Memory(source) => {
                let mut sets = BTreeMap::new();
                for (id, fetched) in keys {
                    let batch =
                        concat_batches(&fetched.schema, &fetched.batches).map_err(|_| {
                            ProviderError {
                                refuses_expression: false,
                                kind: ErrorKind::Internal,
                                retryable: false,
                                sqlstate: None,
                                vendor_code: None,
                            }
                        })?;
                    sets.insert(*id, batch);
                }
                let batch = source.answer(&step.request, &sets)?;
                Ok(Fetched {
                    schema: batch.schema(),
                    batches: vec![batch],
                })
            }
        }
    }
}

/// Runs the kit on `database`, `fault` seeded, from the repository at
/// `repository`: the report, and whether every case passed.
pub fn run(repository: &Path, database: &str, fault: Option<&str>) -> Result<(Value, bool)> {
    if !DATABASES.contains(&database) {
        bail!(
            "--kit {database}: this host runs the kit on DuckDB, SQLite or the non-SQL memory provider"
        );
    }
    if let Some(fault) = fault
        && !FAULTS.contains(&fault)
    {
        bail!("--fault {fault}: one of {FAULTS:?}");
    }
    let kit: Value = serde_json::from_str(
        &std::fs::read_to_string(repository.join(KIT).join("expected.json"))
            .context("reading the kit's expected.json")?,
    )?;
    let page_rows = kit["page_rows"]
        .as_u64()
        .filter(|&rows| rows > 0)
        .ok_or_else(|| anyhow!("expected.json's page_rows is not a positive count"))?
        as usize;
    let cases = kit["cases"]
        .as_array()
        .ok_or_else(|| anyhow!("expected.json holds no cases"))?;
    let by_id: BTreeMap<&str, &Value> = cases
        .iter()
        .map(|c| (c["id"].as_str().unwrap_or_default(), c))
        .collect();
    let directory = tempdir()?;
    let built = (|| -> Result<(Value, bool)> {
        let source = source(repository, database, &directory)?;
        let mut reported = Vec::new();
        let mut passed = true;
        for case in cases {
            let seen = run_case(case, &by_id, database, &source, page_rows, fault)?;
            let problems = judge(case, &seen, &by_id, page_rows);
            if !problems.is_empty() {
                passed = false;
                eprintln!("{}: {}", case["id"], problems.join("; "));
            }
            reported.push(seen);
        }
        Ok((
            json!({
                "format": FORMAT,
                "host": "rust-host",
                "database": database,
                "fault": fault,
                "cases": reported,
            }),
            passed,
        ))
    })();
    std::fs::remove_dir_all(&directory).ok();
    built
}

fn tempdir() -> Result<std::path::PathBuf> {
    let directory = std::env::temp_dir().join(format!(
        "sdm-provider-kit-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or_default()
    ));
    std::fs::create_dir_all(&directory)?;
    Ok(directory)
}

/// The dataset on `database`.
fn source(repository: &Path, database: &str, directory: &Path) -> Result<Source> {
    if database == "duckdb" || database == "memory" {
        let connection = duck::open()?;
        let dataset = std::fs::read_to_string(repository.join(KIT).join("dataset.sql"))?;
        connection
            .execute_batch(&dataset)
            .context("building the kit's dataset on DuckDB")?;
        if database == "memory" {
            return Ok(Source::Memory(Memory::from_duckdb(&connection)?));
        }
        return Ok(Source::Duck(connection));
    }
    let file = directory.join("kit.sqlite");
    let python = std::env::var_os("SDM_PYTHON").unwrap_or_else(|| "python3".into());
    let done = Command::new(&python)
        .arg("-I")
        .arg(repository.join("contract/runners/provider_kit.py"))
        .args(["--load", "sqlite"])
        .arg(&file)
        .current_dir(repository)
        .stdin(Stdio::null())
        .output()
        .with_context(|| {
            format!(
                "running {} for the contract loader",
                python.to_string_lossy()
            )
        })?;
    if !done.status.success() {
        bail!(
            "the contract loader did not load the kit's dataset: {}",
            String::from_utf8_lossy(&done.stderr).trim()
        );
    }
    Ok(Source::Sqlite(Sqlite::open(&file)?))
}

/// The case whose request and seeds `case` runs.
fn resolved<'a>(case: &'a Value, by_id: &BTreeMap<&str, &'a Value>) -> Result<&'a Value> {
    match case
        .get("same_as")
        .or_else(|| case.get("request_of"))
        .and_then(Value::as_str)
    {
        Some(named) => by_id
            .get(named)
            .copied()
            .ok_or_else(|| anyhow!("{}: no case {named}", case["id"])),
        None => Ok(case),
    }
}

/// Between the run and the provider: each answer in pages, the fault
/// seeded, the injected page failed, and every fetch's request recorded.
struct ProviderPath<'a> {
    source: &'a Source,
    page_rows: usize,
    fault: Option<&'a str>,
    inject: Option<(usize, bool)>,
    held: BTreeMap<String, Vec<ArrowData>>,
    next: u64,
    fetched: Vec<String>,
}

impl ProviderPath<'_> {
    /// The outcome of one fetch, and its attachments.
    fn answer(
        &mut self,
        step: &FetchStep,
        keys: &BTreeMap<AttachmentId, Fetched>,
    ) -> Result<(FetchOutcome, BTreeMap<AttachmentId, Attachment>)> {
        let name = step.request.name();
        self.fetched.push(name.to_owned());
        let failed = |error| Ok((FetchOutcome::Error { error }, BTreeMap::new()));
        if let Some((page, retryable)) = self.inject
            && self.fetched.len() == page
        {
            return failed(ProviderError {
                refuses_expression: false,
                kind: ErrorKind::DataSource,
                retryable,
                sqlstate: None,
                vendor_code: None,
            });
        }
        let token = match &step.continuation {
            Some(token) => token.clone(),
            None => {
                let fetched = match self.source.answer(step, keys) {
                    Ok(fetched) => fetched,
                    Err(error) => return failed(error),
                };
                let mut whole = concat_batches(&fetched.schema, &fetched.batches)?;
                if self.fault == Some("silent_capability") && name == "neighbors" {
                    whole = whole.slice(0, 0);
                }
                let mut pages: Vec<ArrowData> = (0..whole.num_rows())
                    .step_by(self.page_rows)
                    .map(|at| {
                        let rows = self.page_rows.min(whole.num_rows() - at);
                        ArrowData::new(fetched.schema.clone(), vec![whole.slice(at, rows)])
                    })
                    .collect();
                if pages.is_empty() {
                    pages.push(ArrowData::new(fetched.schema.clone(), vec![whole]));
                }
                if self.fault == Some("truncate") && pages.len() > 1 {
                    pages.pop();
                }
                self.next += 1;
                let token = format!("page-{}", self.next);
                self.held.insert(token.clone(), pages);
                token
            }
        };
        let mut pages = self
            .held
            .remove(&token)
            .ok_or_else(|| anyhow!("a continuation this path never gave"))?;
        let page = pages.remove(0);
        let continuation = (!pages.is_empty()).then(|| token.clone());
        if continuation.is_some() {
            self.held.insert(token, pages);
        }
        Ok((
            FetchOutcome::Data {
                data: AttachmentId(1),
                continuation,
            },
            BTreeMap::from([(AttachmentId(1), Attachment::Arrow(page))]),
        ))
    }
}

/// What `case` gave on `source`: its fetches, and its answer or error.
fn run_case(
    case: &Value,
    by_id: &BTreeMap<&str, &Value>,
    database: &str,
    source: &Source,
    page_rows: usize,
    fault: Option<&str>,
) -> Result<Value> {
    let id = case["id"].as_str().unwrap_or_default();
    let given = resolved(case, by_id)?;
    let mut request = given["request"].clone();
    let seeds = given.get("seeds").and_then(Value::as_array);
    if seeds.is_some() {
        request["seeds"] = json!({
            "attachment": 2,
            "columns": [
                {"name": "key", "type": {"type": "int64"}, "nullable": false},
                {"name": "batch", "type": {"type": "int64"}, "nullable": false},
            ],
        });
    }
    let capabilities = case.get("capabilities").cloned().unwrap_or(json!({}));
    let mut fetch = json!({
        "request": request,
        "provider": match source {
            Source::Memory(_) => json!({"provider": "host", "capabilities": Memory::capabilities()}),
            _ => json!({"provider": "sql", "dialect": database}),
        },
        "echo": true,
        "undeclared": capabilities.get("withhold").cloned().unwrap_or(json!([])),
    });
    if let Some(limits) = capabilities.get("fallback") {
        fetch["fallback"] = limits.clone();
    }
    let mut operation = json!({"op": "test", "test": "fetch", "fetch": 1});
    let mut attachments = BTreeMap::from([(
        AttachmentId(1),
        Attachment::Bytes(serde_json::to_vec(&fetch)?),
    )]);
    if let Some(seeds) = seeds {
        operation = json!({"op": "test", "test": "keyed", "fetch": 1, "keys": 2});
        let column = |at: usize| -> Result<ArrayRef> {
            let values = seeds
                .iter()
                .map(|s| {
                    s[at]
                        .as_i64()
                        .ok_or_else(|| anyhow!("{id}: a seed is not [key, batch]"))
                })
                .collect::<Result<Vec<i64>>>()?;
            Ok(std::sync::Arc::new(Int64Array::from(values)))
        };
        let schema = std::sync::Arc::new(Schema::new(vec![
            Field::new("key", DataType::Int64, false),
            Field::new("batch", DataType::Int64, false),
        ]));
        let batch = RecordBatch::try_new(schema.clone(), vec![column(0)?, column(1)?])?;
        attachments.insert(
            AttachmentId(2),
            Attachment::Arrow(ArrowData::new(schema, vec![batch])),
        );
    }
    let start = json!({
        "message": "start",
        "contract": CONTRACT_VERSION,
        "facts": serde_json::to_value(facts()?)?,
        "operation": operation,
    });
    let mut path = ProviderPath {
        source,
        page_rows,
        fault,
        inject: case.get("inject").map(|inject| {
            (
                inject["page"].as_u64().unwrap_or(0) as usize,
                inject["retryable"] == Value::Bool(true),
            )
        }),
        held: BTreeMap::new(),
        next: 0,
        fetched: Vec::new(),
    };
    let ended = match HostRun::start(&default_root(), &serde_json::to_vec(&start)?, attachments) {
        Err(refused) => Err(error_of(&refused)?),
        Ok(handle) => drive(&handle, &mut path)?,
    };
    let mut seen = json!({"id": id, "fetched": path.fetched});
    match ended {
        Ok(answer) => seen["answer"] = answer,
        Err(error) => seen["error"] = error,
    }
    Ok(seen)
}

/// The run to its end: the admitted answer, as columns and rows, or the
/// core's error.
fn drive(
    handle: &HostRun,
    path: &mut ProviderPath<'_>,
) -> Result<std::result::Result<Value, Value>> {
    loop {
        let out = match handle.next() {
            Ok(out) => out,
            Err(failed) => return Ok(Err(error_of(&failed)?)),
        };
        // The test-only result, read as JSON (see the module's text).
        let message: Value = serde_json::from_slice(&out.control).unwrap_or(Value::Null);
        if message["message"] == "step" && message["step"] == "done" {
            let id = message["result"]["data"]
                .as_u64()
                .ok_or_else(|| anyhow!("the fetch's answer is not attached"))?;
            let Ok(Some(Attachment::Arrow(data))) = handle.take_attachment(AttachmentId(id as u32))
            else {
                bail!("the fetch's answer is not attached as Arrow");
            };
            return Ok(Ok(answer_of(&data)?));
        }
        let ControlMessage::Step(Step::Fetch { requests }) =
            read(&out).map_err(|_| anyhow!("an unreadable step"))?
        else {
            bail!("the core asked for a step the kit never takes");
        };
        for step in &requests {
            let mut keys = BTreeMap::new();
            if step.continuation.is_none() {
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
                for set in sets {
                    if let Entry::Vacant(entry) = keys.entry(set.attachment) {
                        let Ok(Some(Attachment::Arrow(data))) =
                            handle.take_attachment(set.attachment)
                        else {
                            bail!("a key set is not attached");
                        };
                        entry.insert(Fetched {
                            schema: data.schema,
                            batches: data.batches,
                        });
                    }
                }
            }
            let (outcome, attachments) = path.answer(step, &keys)?;
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
}

/// The core's error an `error` control message carries, as the report
/// holds it.
fn error_of(out: &Outgoing) -> Result<Value> {
    let ControlMessage::Error { error } =
        read(out).map_err(|_| anyhow!("the host wrote an unreadable message"))?
    else {
        bail!("the host gave a message that is not an error");
    };
    report_error(&error)
}

fn report_error(error: &CoreError) -> Result<Value> {
    Ok(json!({
        "code": serde_json::to_value(error.code)?,
        "kind": serde_json::to_value(error.kind)?,
        "retryable": error.retryable,
        "fields": serde_json::to_value(&error.fields)?,
    }))
}

/// Admitted data as the report's columns and rows: each value a JSON
/// number, text, boolean or null.
fn answer_of(data: &ArrowData) -> Result<Value> {
    let columns: Vec<&String> = data.schema.fields().iter().map(|f| f.name()).collect();
    let mut rows = Vec::new();
    for batch in &data.batches {
        let cells = batch
            .columns()
            .iter()
            .map(cells_of)
            .collect::<Result<Vec<_>>>()?;
        for row in 0..batch.num_rows() {
            rows.push(Value::Array(cells.iter().map(|c| c[row].clone()).collect()));
        }
    }
    Ok(json!({"columns": columns, "rows": rows}))
}

fn cells_of(column: &ArrayRef) -> Result<Vec<Value>> {
    use duckdb::arrow::array::{BooleanArray, Float64Array, StringArray};
    let data_type = column.data_type();
    let cells = if data_type.is_integer() {
        let values = cast(column, &DataType::Int64)?;
        let values = values
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| anyhow!("an integer column that does not cast"))?;
        values.iter().map(|v| json!(v)).collect()
    } else if data_type.is_floating() || matches!(data_type, DataType::Decimal128(..)) {
        let values = cast(column, &DataType::Float64)?;
        let values = values
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| anyhow!("a number column that does not cast"))?;
        values.iter().map(|v| json!(v)).collect()
    } else if *data_type == DataType::Boolean {
        let values = column
            .as_any()
            .downcast_ref::<BooleanArray>()
            .ok_or_else(|| anyhow!("a boolean column that is not one"))?;
        values.iter().map(|v| json!(v)).collect()
    } else {
        let values = cast(column, &DataType::Utf8)?;
        let values = values
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| anyhow!("a column that does not cast to text"))?;
        values.iter().map(|v| json!(v)).collect()
    };
    Ok(cells)
}

// --- the judgment, by expected.json's rules ------------------------------------------

/// A value as compared: a number by its value, a zero of either sign one
/// value, anything else as it is.
fn compared(value: &Value) -> Value {
    match value.as_f64() {
        Some(number) if value.is_number() => json!(number + 0.0),
        _ => value.clone(),
    }
}

fn row_key(row: &[Value]) -> String {
    serde_json::to_string(&row.iter().map(compared).collect::<Vec<_>>()).unwrap_or_default()
}

fn rows_of(value: &Value) -> Vec<Vec<Value>> {
    value
        .as_array()
        .map(|rows| {
            rows.iter()
                .map(|row| row.as_array().cloned().unwrap_or_default())
                .collect()
        })
        .unwrap_or_default()
}

fn answer_problem(got: &Value, wanted: &Value) -> Option<String> {
    let names: Vec<&str> = got["columns"]
        .as_array()
        .map(|c| c.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    let mut at = Vec::new();
    for column in wanted["columns"].as_array().into_iter().flatten() {
        let column = column.as_str().unwrap_or_default();
        match names.iter().position(|n| *n == column) {
            Some(i) => at.push(i),
            None => return Some(format!("no column {column} among {names:?}")),
        }
    }
    let rows: Vec<Vec<Value>> = rows_of(&got["rows"])
        .iter()
        .map(|row| {
            at.iter()
                .map(|&i| row.get(i).cloned().unwrap_or(Value::Null))
                .collect()
        })
        .collect();
    if let Some(any_of) = wanted.get("any_of") {
        let count = wanted["count"].as_u64().unwrap_or(0) as usize;
        if rows.len() != count {
            return Some(format!("{} rows, not {count}", rows.len()));
        }
        let mut pool: Vec<String> = rows_of(any_of).iter().map(|r| row_key(r)).collect();
        for row in &rows {
            match pool.iter().position(|k| *k == row_key(row)) {
                Some(i) => {
                    pool.remove(i);
                }
                None => return Some(format!("{row:?} is not one of the rows it may be")),
            }
        }
        return None;
    }
    let keys = |rows: &[Vec<Value>]| rows.iter().map(|r| row_key(r)).collect::<Vec<_>>();
    let (mut have, mut want) = (keys(&rows), keys(&rows_of(&wanted["rows"])));
    if wanted["ordered"] != Value::Bool(true) {
        have.sort();
        want.sort();
    }
    (have != want).then(|| format!("rows {have:?}, not {want:?}"))
}

/// An error field's value as text: the value, or a scalar's value.
fn text(detail: Option<&Value>) -> String {
    match detail {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Object(scalar)) if scalar.contains_key("v") => text(scalar.get("v")),
        Some(other) => other.to_string(),
        None => "None".to_owned(),
    }
}

/// Where a case names its fetches exactly, how `fetched` differs.
fn fetched_problem(exactly: &Value, fetched: &[&str]) -> Vec<String> {
    let wanted: Vec<&str> = exactly
        .as_array()
        .map(|f| f.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    if fetched == wanted.as_slice() {
        return vec![];
    }
    vec![format!("fetched {fetched:?}, not exactly {wanted:?}")]
}

/// Why `seen` does not pass `case`; empty when it does.
fn judge(
    case: &Value,
    seen: &Value,
    by_id: &BTreeMap<&str, &Value>,
    page_rows: usize,
) -> Vec<String> {
    let fetched: Vec<&str> = seen["fetched"]
        .as_array()
        .map(|f| f.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    let wanted = match case.get("same_as").and_then(Value::as_str) {
        Some(named) => by_id.get(named).and_then(|c| c.get("answer")),
        None => case.get("answer"),
    };
    if let Some(wanted) = wanted {
        let Some(got) = seen.get("answer") else {
            return vec![format!("an error, {}, not an answer", seen["error"])];
        };
        if let Some(problem) = answer_problem(got, wanted) {
            return vec![problem];
        }
        if let Some(exactly) = case.get("fetched") {
            return fetched_problem(exactly, &fetched);
        }
        let withheld: Vec<&str> = case
            .pointer("/capabilities/withhold")
            .and_then(Value::as_array)
            .map(|w| w.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default();
        if !withheld.is_empty() {
            if fetched.is_empty() || fetched.iter().any(|f| withheld.contains(f)) {
                return vec![format!(
                    "fetched {fetched:?}: the fallback asks none of {withheld:?}"
                )];
            }
            return vec![];
        }
        let rows = match wanted.get("any_of") {
            Some(_) => wanted["count"].as_u64().unwrap_or(0) as usize,
            None => wanted["rows"].as_array().map_or(0, Vec::len),
        };
        let name = resolved(case, by_id)
            .ok()
            .and_then(|c| c["request"]["request"].as_str())
            .unwrap_or_default();
        if fetched.iter().filter(|f| **f == name).count() < rows.div_ceil(page_rows) {
            return vec![format!(
                "fetched {fetched:?}: {rows} rows in fewer than pages of {page_rows}"
            )];
        }
        return vec![];
    }
    let Some(got) = seen.get("error") else {
        return vec![format!("an answer, not the error {}", case["error"])];
    };
    let error = &case["error"];
    for name in ["code", "kind", "retryable"] {
        if got.get(name) != error.get(name) {
            return vec![format!("{name} {:?}, not {}", got.get(name), error[name])];
        }
    }
    for (name, value) in error["fields"].as_object().into_iter().flatten() {
        let have = text(got.pointer(&format!("/fields/{name}")));
        if have != text(Some(value)) {
            return vec![format!("field {name} {have}, not {value}")];
        }
    }
    if let Some(exactly) = case.get("fetched") {
        return fetched_problem(exactly, &fetched);
    }
    if let Some(page) = case.pointer("/inject/page").and_then(Value::as_u64)
        && (fetched.len() as u64) < page
    {
        return vec![format!("fetched {fetched:?}: page {page} was never asked")];
    }
    vec![]
}
