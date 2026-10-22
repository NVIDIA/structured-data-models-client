// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The stand-in provider (Phase 3 design, choice 2; T3.3): it answers the
//! core's data requests for a warehouse leg offline, from what the old
//! client did and got on that leg in the early capture (C-1), so a
//! warehouse case is judged by its answers against the old client's,
//! never by matching the core's SQL.
//!
//! It answers a fetch step in three parts, in order:
//!
//! 1. **Exact replay.** When every statement of the step's plan, and its
//!    parameters, equal a statement the old client sent in the capture,
//!    after the text's whitespace outside quotes is normalised (the
//!    capture already holds placeholders, `{catalog}` and the like, where
//!    the warehouse's names were), the step is answered with the last
//!    one's recorded result, or its recorded error.
//! 2. **The local engine.** Otherwise the request, each relation read by
//!    its last name part, is planned through the DuckDB dialect and run on
//!    a DuckDB loaded with the capture's tables, its session set to the
//!    capture's NULL order and time zone. An entity-sampling request is
//!    answered only from the old client's recorded draw of the same table,
//!    seed and fraction text, which the step's plan writes; any other is
//!    refused by name. A request reading a table the capture does not
//!    hold is refused by name. A request binding a key set whose column
//!    types differ from the recorded table's is answered and flagged
//!    [`B4`].
//! 3. **Live only.** What neither part answers is [`Answer::LiveOnly`]
//!    with its reason: [`B1`], a statement the old client never sent over
//!    SQL the local engine cannot run (a catalog read, a request the
//!    DuckDB dialect cannot render or DuckDB cannot run), or [`B2`], a
//!    relation derived from a view whose column the warehouse refused.
//!
//! A refusal ([`Refused`]) is a runner failure, never a provider error the
//! core could take for the warehouse's: a missing table or draw fails by
//! name, never passes silently (design, Risks).
//!
//! The capture is read from its directory as `contract/runners/captures.py`
//! writes it (format [`FORMAT`]): `capture.json` and each table an Arrow
//! IPC stream named by its SHA-256 under `tables/`, its content checked
//! against its name. Lakebase's local engine is the local Postgres CI runs,
//! which the Python host reads directly; this module serves the SQL
//! warehouse, Spark and Snowflake legs.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::io::Cursor;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use duckdb::Connection;
use duckdb::arrow::compute::concat_batches;
use duckdb::arrow::datatypes::{DataType, SchemaRef};
use duckdb::arrow::record_batch::RecordBatch;
use duckdb::vtab::arrow::{ArrowVTab, arrow_recordbatch_to_query_params};
use sdm_contract::ErrorKind;
use sdm_contract::Scalar;
use sdm_contract::data::{
    AttachmentId, DataRequest, Dialect, FetchPlan, Filter, ProviderChoice, ProviderError, Relation,
    SqlParam,
};
use sdm_contract::run::FetchStep;
use serde::Deserialize;
use serde_json::Value;

use crate::duck::{self, Fetched};

/// The capture format this reader reads.
pub const FORMAT: &str = "sdm-capture/1";

/// A statement the old client never sent, over SQL the local engine cannot
/// run.
pub const B1: &str = "B1";
/// A relation derived from a view, holding a column the warehouse refused.
pub const B2: &str = "B2";
/// A request comparing keys of different recorded types, answered on the
/// local engine.
pub const B4: &str = "B4";

/// The legs this module stands in for.
pub const LEGS: &[&str] = &["databricks-sql", "databricks-spark", "snowflake"];

/// What the stand-in refuses by name: a recorded table or draw missing, a
/// draw asked with another seed or fraction text, a capture it cannot
/// read. A runner failure, never an answer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Refused(pub String);

impl fmt::Display for Refused {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Refused {}

fn refused(message: impl Into<String>) -> Refused {
    Refused(message.into())
}

/// How the stand-in answered one fetch step.
pub enum Answer {
    /// Rows: replayed, from a recorded draw, or from the local engine.
    Data(Fetched),
    /// The provider's error: a replayed statement the warehouse refused,
    /// as the contract carries it.
    Error(ProviderError),
    /// Neither part can judge the step: the reason code, one of
    /// [`B1`] and [`B2`] (or [`B4`]'s, where the caller decides it).
    LiveOnly(&'static str),
}

/// An answer, how it was reached, and the codes it was flagged with.
pub struct Answered {
    pub answer: Answer,
    /// `replay`, `draw`, `local` or `live_only`: diagnostic.
    pub by: &'static str,
    pub flagged: Vec<&'static str>,
}

// --- the capture --------------------------------------------------------------------

#[derive(Deserialize)]
struct Document {
    format: String,
    case: String,
    leg: String,
    session: Session,
    tables: Vec<NamedTable>,
    #[serde(default)]
    derived: Vec<DerivedEntry>,
    #[serde(default)]
    draws: Vec<DrawEntry>,
    #[serde(default)]
    statements: Vec<StatementEntry>,
}

/// The warehouse session the capture was recorded in (B5).
#[derive(Deserialize, Clone)]
pub struct Session {
    pub time_zone: String,
    pub null_order: String,
}

#[derive(Deserialize)]
struct NamedTable {
    name: String,
    data: String,
}

#[derive(Deserialize)]
struct DerivedEntry {
    #[serde(default)]
    view: Option<String>,
    table: String,
    #[serde(default)]
    refused: Vec<String>,
    data: Option<String>,
}

#[derive(Deserialize)]
struct DrawEntry {
    table: String,
    seed: Option<Value>,
    fraction: String,
    rows: String,
}

#[derive(Deserialize)]
struct StatementEntry {
    text: String,
    #[serde(default)]
    parameters: Option<Value>,
    result: Option<String>,
    #[serde(default)]
    error: Option<Value>,
}

/// One random draw the old client's sampler made.
pub struct Draw {
    pub table: String,
    pub seed: Option<String>,
    pub fraction: String,
    pub rows: Fetched,
}

/// One statement the old client sent, its text normalised, and its answer.
pub struct Sent {
    pub text: String,
    pub parameters: Vec<Value>,
    pub answer: std::result::Result<Fetched, ()>,
}

/// One case's capture on one leg.
pub struct Capture {
    pub case: String,
    pub leg: String,
    pub session: Session,
    pub tables: BTreeMap<String, Fetched>,
    /// Each derived relation's table name, and whether the warehouse
    /// refused one of its columns (no warehouse-evaluated table).
    pub derived: BTreeMap<String, Option<Fetched>>,
    pub draws: Vec<Draw>,
    pub sent: Vec<Sent>,
    /// The views the old client read, by last name part: each named by a
    /// view's own read it sent (Databricks' `DESCRIBE TABLE EXTENDED`,
    /// Snowflake's `SYSTEM$READ_YAML_FROM_SEMANTIC_VIEW`), or by a derived
    /// relation's entry.
    pub views: BTreeSet<String>,
    pub directory: PathBuf,
}

/// The table file `name` of the capture in `directory`, its content
/// checked against its name.
fn table_file(directory: &Path, name: &str) -> Result<Fetched> {
    let digest = name
        .strip_prefix("tables/")
        .and_then(|rest| rest.strip_suffix(".arrows"))
        .filter(|d| d.len() == 64 && d.bytes().all(|b| b.is_ascii_hexdigit()))
        .ok_or_else(|| anyhow!("{name:?} is not a capture table"))?;
    let bytes = std::fs::read(directory.join(name))
        .with_context(|| format!("the capture table {name} cannot be read"))?;
    if crate::case::sha256(&bytes) != digest {
        bail!("{name}: its content is not its name");
    }
    decode(&bytes).with_context(|| format!("{name}: not an Arrow IPC stream"))
}

/// An Arrow IPC stream's schema and batches.
pub fn decode(bytes: &[u8]) -> Result<Fetched> {
    let reader = arrow_ipc::reader::StreamReader::try_new(Cursor::new(bytes), None)
        .map_err(|e| anyhow!("{e}"))?;
    let schema = reader.schema();
    let batches = reader
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|e| anyhow!("{e}"))?;
    Ok(Fetched { schema, batches })
}

/// An Arrow IPC stream of `fetched`.
pub fn encode(fetched: &Fetched) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    let mut writer = arrow_ipc::writer::StreamWriter::try_new(&mut bytes, &fetched.schema)
        .map_err(|e| anyhow!("{e}"))?;
    for batch in &fetched.batches {
        writer.write(batch).map_err(|e| anyhow!("{e}"))?;
    }
    writer.finish().map_err(|e| anyhow!("{e}"))?;
    drop(writer);
    Ok(bytes)
}

impl Capture {
    /// The capture at `path`, its directory (or its `capture.json`).
    pub fn read(path: &Path) -> Result<Capture> {
        let (directory, document) = if path.is_dir() {
            (path.to_path_buf(), path.join("capture.json"))
        } else {
            (
                path.parent().map(Path::to_path_buf).unwrap_or_default(),
                path.to_path_buf(),
            )
        };
        let text = std::fs::read_to_string(&document)
            .with_context(|| format!("the capture {} cannot be read", document.display()))?;
        let format = serde_json::from_str::<Value>(&text)
            .ok()
            .and_then(|v| v.get("format").and_then(Value::as_str).map(str::to_owned));
        match format.as_deref() {
            Some(FORMAT) => {}
            other => bail!(
                "{}: the capture format {:?} is not {FORMAT}; convert it with \
                 contract/runners/captures.py convert",
                document.display(),
                other.unwrap_or("(none)")
            ),
        }
        let document: Document = serde_json::from_str(&text)
            .with_context(|| format!("{}: not a capture", document.display()))?;
        debug_assert_eq!(document.format, FORMAT);
        let mut tables = BTreeMap::new();
        for table in &document.tables {
            tables.insert(table.name.clone(), table_file(&directory, &table.data)?);
        }
        let mut derived = BTreeMap::new();
        for entry in &document.derived {
            let data = match (&entry.data, entry.refused.is_empty()) {
                (Some(name), _) => Some(table_file(&directory, name)?),
                (None, false) => None,
                (None, true) => bail!(
                    "the derived relation {} has no table and no refused column",
                    entry.table
                ),
            };
            derived.insert(entry.table.clone(), data);
        }
        let mut draws = Vec::new();
        for draw in &document.draws {
            let seed = match &draw.seed {
                None | Some(Value::Null) => None,
                Some(Value::Number(n)) => Some(n.to_string()),
                Some(other) => bail!("a draw's seed {other} is not an integer"),
            };
            draws.push(Draw {
                table: draw.table.clone(),
                seed,
                fraction: draw.fraction.clone(),
                rows: table_file(&directory, &draw.rows)?,
            });
        }
        let mut sent = Vec::new();
        for statement in &document.statements {
            let parameters = match &statement.parameters {
                None | Some(Value::Null) => Vec::new(),
                Some(Value::Array(items)) => items.clone(),
                Some(other) => vec![other.clone()],
            };
            let answer = match (&statement.result, &statement.error) {
                (Some(name), _) => Ok(table_file(&directory, name)?),
                (None, _) => Err(()),
            };
            sent.push(Sent {
                text: normalised(&statement.text),
                parameters,
                answer,
            });
        }
        let mut views: BTreeSet<String> = document
            .derived
            .iter()
            .filter_map(|entry| entry.view.as_deref())
            .filter(|view| !view.is_empty())
            .map(last_part)
            .collect();
        for statement in &document.statements {
            views.extend(view_read(&statement.text));
        }
        Ok(Capture {
            case: document.case,
            leg: document.leg,
            session: document.session,
            tables,
            derived,
            draws,
            sent,
            views,
            directory,
        })
    }
}

// --- text and values ----------------------------------------------------------------

/// A statement's text with each run of whitespace outside quotes made one
/// space and none at either end, so the old client's layout and the
/// core's compare equal; quoted text (`'...'`, `"..."`, `` `...` ``) is
/// kept as written.
pub fn normalised(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut quote: Option<char> = None;
    let mut space = false;
    for c in text.chars() {
        match quote {
            Some(q) => {
                out.push(c);
                if c == q {
                    quote = None;
                }
            }
            None if c.is_whitespace() => space = true,
            None => {
                if space && !out.is_empty() {
                    out.push(' ');
                }
                space = false;
                if matches!(c, '\'' | '"' | '`') {
                    quote = Some(c);
                }
                out.push(c);
            }
        }
    }
    out
}

/// A scalar parameter as the JSON value the old client passed: integers as
/// numbers, text as text; `None` for one no recorded parameter can equal.
fn parameter(value: &Scalar) -> Option<Value> {
    use Scalar as S;
    Some(match value {
        S::Null => Value::Null,
        S::Bool(b) => Value::Bool(*b),
        S::Int8(v) => Value::from(*v),
        S::Int16(v) => Value::from(*v),
        S::Int32(v) => Value::from(*v),
        S::Int64(v) => Value::from(*v),
        S::UInt8(v) => Value::from(*v),
        S::UInt16(v) => Value::from(*v),
        S::UInt32(v) => Value::from(*v),
        S::UInt64(v) => Value::from(*v),
        S::Float64(v) => serde_json::Number::from_f64(*v).map(Value::Number)?,
        S::Utf8(text) => Value::String(text.clone()),
        _ => return None,
    })
}

/// Whether a recorded parameter equals the plan's: numbers by value.
fn same_parameter(recorded: &Value, asked: &Value) -> bool {
    match (recorded, asked) {
        (Value::Number(a), Value::Number(b)) => {
            match (a.as_i64(), b.as_i64(), a.as_u64(), b.as_u64()) {
                (Some(x), Some(y), _, _) => x == y,
                (_, _, Some(x), Some(y)) => x == y,
                _ => a.as_f64() == b.as_f64() && a.is_f64() == b.is_f64(),
            }
        }
        _ => recorded == asked,
    }
}

/// The last part of a name as a warehouse writes it, unquoted.
fn last_part(name: &str) -> String {
    let mut parts = Vec::new();
    let mut current = String::new();
    let mut quote: Option<char> = None;
    let mut chars = name.chars().peekable();
    while let Some(c) = chars.next() {
        match quote {
            Some(q) if c == q => {
                if chars.peek() == Some(&q) {
                    current.push(q);
                    chars.next();
                } else {
                    quote = None;
                }
            }
            Some(_) => current.push(c),
            None if matches!(c, '"' | '`') => quote = Some(c),
            None if c == '.' => parts.push(std::mem::take(&mut current)),
            None => current.push(c),
        }
    }
    parts.push(current);
    parts.pop().unwrap_or_default()
}

/// The view a statement reads as a view (Databricks' `DESCRIBE TABLE
/// EXTENDED <view>`, Snowflake's `SYSTEM$READ_YAML_FROM_SEMANTIC_VIEW('<view>')`),
/// by its last name part.
fn view_read(text: &str) -> Option<String> {
    let upper = text.to_ascii_uppercase();
    if let Some(at) = upper.find("SYSTEM$READ_YAML_FROM_SEMANTIC_VIEW") {
        let rest = &text[at..];
        let open = rest.find('\'')? + 1;
        let close = rest[open..].find('\'')? + open;
        return Some(last_part(&rest[open..close]));
    }
    let at = upper.find("DESCRIBE TABLE EXTENDED")?;
    let name = text[at + "DESCRIBE TABLE EXTENDED".len()..]
        .split_whitespace()
        .next()?;
    Some(last_part(name))
}

/// The sampling clause a plan's statements write: the old client's
/// Databricks `TABLESAMPLE (<fraction> PERCENT) REPEATABLE (<seed>)`, or
/// Snowflake's `SAMPLE ROW (<n> ROWS)`, its argument as written.
fn sampling_clause(text: &str) -> Option<(String, Option<String>)> {
    let upper = text.to_ascii_uppercase();
    if let Some(at) = upper.find("TABLESAMPLE") {
        let rest = &text[at + "TABLESAMPLE".len()..];
        let open = rest.find('(')?;
        let close = rest[open..].find(')')? + open;
        let inside = rest[open + 1..close].trim();
        let fraction = inside
            .strip_suffix("PERCENT")
            .or_else(|| inside.strip_suffix("percent"))
            .map(str::trim)?
            .to_owned();
        let after = &rest[close + 1..];
        let seed = after
            .to_ascii_uppercase()
            .find("REPEATABLE")
            .and_then(|at| {
                let tail = &after[at + "REPEATABLE".len()..];
                let open = tail.find('(')?;
                let close = tail[open..].find(')')? + open;
                Some(tail[open + 1..close].trim().to_owned())
            });
        return Some((fraction, seed));
    }
    if let Some(at) = upper.find("SAMPLE ROW") {
        let rest = &text[at + "SAMPLE ROW".len()..];
        let open = rest.find('(')?;
        let close = rest[open..].find(')')? + open;
        return Some((rest[open + 1..close].trim().to_owned(), None));
    }
    None
}

/// The seed of a sampling request as decimal digits.
fn seed_text(seed: &Scalar) -> Option<String> {
    sdm_contract::data::seed_integer(seed).map(|seed| seed.to_string())
}

/// Every relation a request reads.
fn relations(request: &DataRequest) -> Vec<&Relation> {
    use DataRequest as R;
    match request {
        R::Describe { tables, .. } => tables.iter().collect(),
        R::Scan { table, .. }
        | R::Preview { table, .. }
        | R::CountRows { table, .. }
        | R::ColumnRanges { table, .. }
        | R::RepeatedKey { table, .. }
        | R::SampleEntities { table, .. }
        | R::Neighbors { table, .. }
        | R::Lookup { table, .. } => vec![table],
        R::KeyCounts { keys } => keys.iter().map(|key| &key.table).collect(),
        R::TimeBounds { columns } => columns.iter().map(|column| &column.table).collect(),
    }
}

/// `request` with each relation read by its last name part, the name the
/// capture's tables and the local engine's have.
fn local(request: &DataRequest) -> DataRequest {
    use DataRequest as R;
    let short =
        |relation: &Relation| Relation::named(relation.source.last().cloned().unwrap_or_default());
    let mut request = request.clone();
    match &mut request {
        R::Describe { tables, .. } => tables.iter_mut().for_each(|t| *t = short(t)),
        R::Scan { table, .. }
        | R::Preview { table, .. }
        | R::CountRows { table, .. }
        | R::ColumnRanges { table, .. }
        | R::RepeatedKey { table, .. }
        | R::SampleEntities { table, .. }
        | R::Neighbors { table, .. }
        | R::Lookup { table, .. } => *table = short(table),
        R::KeyCounts { keys } => keys.iter_mut().for_each(|k| k.table = short(&k.table)),
        R::TimeBounds { columns } => columns.iter_mut().for_each(|c| c.table = short(&c.table)),
    }
    request
}

/// The key sets a request binds, with the table columns each is compared
/// with.
fn compared_keys(request: &DataRequest) -> Vec<(AttachmentId, Vec<(String, String)>)> {
    use DataRequest as R;
    let mut found = Vec::new();
    let filters: &[Filter] = match request {
        R::Scan { filters, .. }
        | R::CountRows { filters, .. }
        | R::SampleEntities { filters, .. } => filters,
        _ => &[],
    };
    for filter in filters {
        if let Filter::KeyIn { columns, keys } = filter {
            let pairs = columns
                .iter()
                .zip(&keys.columns)
                .map(|(column, field)| (column.clone(), field.name.clone()))
                .collect();
            found.push((keys.attachment, pairs));
        }
    }
    if let R::Neighbors { key, seeds, .. } | R::Lookup { key, seeds, .. } = request {
        let roles = sdm_contract::data::seed_key_columns(key.len());
        found.push((seeds.attachment, key.iter().cloned().zip(roles).collect()));
    }
    found
}

/// Whether two recorded types are compared as keys of one type: equal, or
/// both integers, or both text.
fn same_key_type(a: &DataType, b: &DataType) -> bool {
    let integer = |t: &DataType| t.is_integer();
    let text =
        |t: &DataType| matches!(t, DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View);
    a == b || (integer(a) && integer(b)) || (text(a) && text(b))
}

// --- the stand-in -------------------------------------------------------------------

/// The stand-in over one capture: the local engine loaded, ready to answer.
pub struct StandIn {
    capture: Capture,
    duck: Connection,
}

fn quoted(name: &str) -> String {
    format!("\"{}\"", name.replace('"', "\"\""))
}

/// Loads `fetched` into `connection` as the table `name`, every value
/// through Arrow.
fn load_table(
    connection: &Connection,
    name: &str,
    fetched: &Fetched,
    temporary: bool,
) -> Result<()> {
    connection
        .register_table_function::<ArrowVTab>("arrow")
        .map_err(|e| anyhow!("{e}"))?;
    let batch = concat_batches(&fetched.schema, &fetched.batches).map_err(|e| anyhow!("{e}"))?;
    let what = if temporary {
        "TEMPORARY TABLE"
    } else {
        "TABLE"
    };
    let sql = format!(
        "CREATE OR REPLACE {what} {} AS SELECT * FROM arrow(?, ?)",
        quoted(name)
    );
    if batch.num_rows() == 0 {
        // The Arrow table function takes no batch of no rows: an empty
        // table is created from one row of nulls of its columns' types,
        // none of it kept.
        let nulls = fetched
            .schema
            .fields()
            .iter()
            .map(|field| duckdb::arrow::array::new_null_array(field.data_type(), 1))
            .collect::<Vec<_>>();
        let nullable = std::sync::Arc::new(duckdb::arrow::datatypes::Schema::new(
            fetched
                .schema
                .fields()
                .iter()
                .map(|field| field.as_ref().clone().with_nullable(true))
                .collect::<Vec<_>>(),
        ));
        let row = RecordBatch::try_new(nullable, nulls).map_err(|e| anyhow!("{name}: {e}"))?;
        let params = arrow_recordbatch_to_query_params(row);
        connection
            .execute(&format!("{} LIMIT 0", sql), params)
            .map_err(|e| anyhow!("{name}: {e}"))?;
        return Ok(());
    }
    let params = arrow_recordbatch_to_query_params(batch);
    connection
        .execute(&sql, params)
        .map_err(|e| anyhow!("{name}: {e}"))?;
    Ok(())
}

impl StandIn {
    /// The stand-in over `capture`: a DuckDB with each recorded table, its
    /// session's NULL order and time zone the capture's.
    pub fn open(capture: Capture) -> Result<StandIn> {
        if !LEGS.contains(&capture.leg.as_str()) {
            bail!(
                "the stand-in serves {LEGS:?}; {} is served by its own local engine",
                capture.leg
            );
        }
        let duck = duck::open()?;
        let order = match capture.session.null_order.as_str() {
            "nulls_first" => "nulls_first",
            "nulls_last" => "nulls_last",
            other => bail!("the capture's NULL order {other:?} is not nulls_first or nulls_last"),
        };
        duck.execute_batch(&format!(
            "SET GLOBAL default_null_order = '{order}'; SET GLOBAL TimeZone = '{}'",
            capture.session.time_zone.replace('\'', "''")
        ))
        .context("setting the capture's session")?;
        for (name, table) in &capture.tables {
            load_table(&duck, name, table, false)?;
        }
        for (name, data) in &capture.derived {
            if let Some(table) = data {
                load_table(&duck, name, table, false)?;
            }
        }
        Ok(StandIn { capture, duck })
    }

    /// The capture this stand-in serves.
    pub fn capture(&self) -> &Capture {
        &self.capture
    }

    /// Answers one fetch step, its key sets in `keys` by attachment, or
    /// refuses it by name.
    pub fn fetch(
        &self,
        step: &FetchStep,
        keys: &BTreeMap<AttachmentId, Fetched>,
    ) -> std::result::Result<Answered, Refused> {
        let request = &step.request;
        // What the request depends on is checked first, so a recorded
        // table or draw the capture lacks is refused by name, never served
        // by a replayed statement.
        if let DataRequest::SampleEntities { table, .. } = request {
            let name = table.source.last().cloned().unwrap_or_default();
            if !self
                .capture
                .draws
                .iter()
                .any(|draw| last_part(&draw.table) == name)
            {
                return Err(refused(format!(
                    "sample_entities: the capture of {} on {} holds no recorded draw of table {name}",
                    self.capture.case, self.capture.leg
                )));
            }
        }
        for relation in relations(request) {
            let name = relation.source.last().cloned().unwrap_or_default();
            // A view the old client read is no recorded table: its reads
            // are served by exact replay of its recorded statements alone.
            if !self.capture.tables.contains_key(&name)
                && !self.capture.derived.contains_key(&name)
                && !self.capture.views.contains(&name)
            {
                return Err(refused(format!(
                    "{}: the capture of {} on {} holds no recorded table {name}",
                    request.name(),
                    self.capture.case,
                    self.capture.leg
                )));
            }
        }
        if !matches!(request, DataRequest::SampleEntities { .. })
            && let Some(answer) = self.replayed(&step.plan)
        {
            return Ok(Answered {
                answer,
                by: "replay",
                flagged: Vec::new(),
            });
        }
        for relation in relations(request) {
            let name = relation.source.last().cloned().unwrap_or_default();
            if let Some(None) = self.capture.derived.get(&name) {
                return Ok(Answered {
                    answer: Answer::LiveOnly(B2),
                    by: "live_only",
                    flagged: Vec::new(),
                });
            }
            if self.capture.views.contains(&name)
                && !self.capture.tables.contains_key(&name)
                && !self.capture.derived.contains_key(&name)
            {
                // A view read the old client never sent: only the
                // warehouse evaluates the view.
                return Ok(Answered {
                    answer: Answer::LiveOnly(B1),
                    by: "live_only",
                    flagged: Vec::new(),
                });
            }
            if !self.capture.tables.contains_key(&name) && !self.capture.derived.contains_key(&name)
            {
                return Err(refused(format!(
                    "{}: the capture of {} on {} holds no recorded table {name}",
                    request.name(),
                    self.capture.case,
                    self.capture.leg
                )));
            }
        }
        if let DataRequest::Describe { .. } = request {
            // A catalog read answers in the leg's own catalog shape, which
            // only the warehouse's recorded output has.
            return Ok(Answered {
                answer: Answer::LiveOnly(B1),
                by: "live_only",
                flagged: Vec::new(),
            });
        }
        let flagged = self.flags(request, keys);
        if let DataRequest::SampleEntities { .. } = request {
            return self.drawn(step, keys).map(|answer| Answered {
                answer,
                by: "draw",
                flagged,
            });
        }
        let answer = self.run_local(step.id, &local(request), keys);
        Ok(match answer {
            Some(fetched) => Answered {
                answer: Answer::Data(fetched),
                by: "local",
                flagged,
            },
            None => Answered {
                answer: Answer::LiveOnly(B1),
                by: "live_only",
                flagged,
            },
        })
    }

    /// The recorded answer of a plan whose every statement the old client
    /// sent with the same parameters, the last one's.
    fn replayed(&self, plan: &FetchPlan) -> Option<Answer> {
        let FetchPlan::Sql { statements, .. } = plan else {
            return None;
        };
        let mut last = None;
        for statement in statements {
            let mut asked = Vec::new();
            for param in &statement.params {
                match param {
                    SqlParam::Value { value } => asked.push(parameter(value)?),
                    SqlParam::KeySet { .. } => return None,
                }
            }
            let text = normalised(&statement.text);
            let sent = self.capture.sent.iter().find(|sent| {
                sent.text == text
                    && sent.parameters.len() == asked.len()
                    && sent
                        .parameters
                        .iter()
                        .zip(&asked)
                        .all(|(recorded, asked)| same_parameter(recorded, asked))
            })?;
            last = Some(sent);
        }
        let sent = last?;
        Some(match &sent.answer {
            Ok(fetched) => Answer::Data(Fetched {
                schema: fetched.schema.clone(),
                batches: fetched.batches.clone(),
            }),
            Err(()) => Answer::Error(ProviderError {
                kind: ErrorKind::DataSource,
                retryable: false,
                sqlstate: None,
                vendor_code: None,
            }),
        })
    }

    /// The codes a request is flagged with: [`B4`] where a key set's column
    /// is compared with a table column of another recorded type.
    fn flags(
        &self,
        request: &DataRequest,
        keys: &BTreeMap<AttachmentId, Fetched>,
    ) -> Vec<&'static str> {
        let Some(table) = relations(request)
            .first()
            .and_then(|r| r.source.last())
            .and_then(|name| self.capture.tables.get(name))
        else {
            return Vec::new();
        };
        for (attachment, pairs) in compared_keys(request) {
            let Some(set) = keys.get(&attachment) else {
                continue;
            };
            for (column, role) in pairs {
                let (Ok(recorded), Ok(bound)) = (
                    table.schema.field_with_name(&column),
                    set.schema.field_with_name(&role),
                ) else {
                    continue;
                };
                if !same_key_type(recorded.data_type(), bound.data_type()) {
                    return vec![B4];
                }
            }
        }
        Vec::new()
    }

    /// An entity-sampling request answered from the old client's recorded
    /// draw of the same table, seed and fraction text, or refused by name.
    fn drawn(
        &self,
        step: &FetchStep,
        keys: &BTreeMap<AttachmentId, Fetched>,
    ) -> std::result::Result<Answer, Refused> {
        let DataRequest::SampleEntities {
            table,
            columns,
            filters,
            n,
            seed,
            ..
        } = &step.request
        else {
            unreachable!("a sampling request");
        };
        let name = table.source.last().cloned().unwrap_or_default();
        let clause = match &step.plan {
            FetchPlan::Sql { statements, .. } => {
                statements.iter().find_map(|s| sampling_clause(&s.text))
            }
            FetchPlan::Host { .. } => None,
        };
        // A seeded draw is served for the request's own seed, which the
        // plan, where it writes one, writes too; an unseeded draw
        // (Snowflake's `SAMPLE ROW (n ROWS)`) only for a plan writing an
        // unseeded clause, whatever seed the request carries.
        let requested = seed_text(seed);
        let (fraction, written, clause_written) = match clause {
            Some((fraction, seed)) => (Some(fraction), seed, true),
            None => (None, None, false),
        };
        let asked_seed = written.clone().or_else(|| requested.clone());
        let recorded: Vec<&Draw> = self
            .capture
            .draws
            .iter()
            .filter(|draw| last_part(&draw.table) == name)
            .collect();
        if recorded.is_empty() {
            return Err(refused(format!(
                "sample_entities: the capture of {} on {} holds no recorded draw of table {name}",
                self.capture.case, self.capture.leg
            )));
        }
        let Some(draw) = recorded.iter().find(|draw| {
            let seeds = match &draw.seed {
                Some(recorded) => {
                    requested.as_deref() == Some(recorded.as_str())
                        && written.as_deref().is_none_or(|w| w == recorded)
                }
                None => clause_written && written.is_none(),
            };
            seeds
                && fraction
                    .as_deref()
                    .is_none_or(|fraction| fraction == draw.fraction)
        }) else {
            let held = recorded
                .iter()
                .map(|draw| {
                    format!(
                        "seed {} and fraction {:?}",
                        draw.seed.as_deref().unwrap_or("none"),
                        draw.fraction
                    )
                })
                .collect::<Vec<_>>()
                .join("; ");
            return Err(refused(format!(
                "sample_entities: a draw of table {name} asked with seed {} (the request's \
                 {}) and fraction {:?}, where the old client drew with {held}",
                asked_seed.as_deref().unwrap_or("none"),
                requested.as_deref().unwrap_or("none"),
                fraction.as_deref().unwrap_or("(none written)")
            )));
        };
        // The drawn rows, read as a scan of them with the request's
        // columns, filters and size.
        // A table of the database, not of the session: the step runs on a
        // cursor of its own, which sees no other session's temporary table.
        let drawn = "__sdm_standin_draw";
        let cursor = &self.duck;
        load_table(cursor, drawn, &draw.rows, false)
            .map_err(|e| refused(format!("sample_entities: the draw of {name}: {e}")))?;
        let scan = DataRequest::Scan {
            table: Relation::named(drawn),
            columns: columns.clone(),
            filters: filters.clone(),
            limit: Some(*n),
        };
        let answer = self.run_on(cursor, step.id, &scan, keys);
        let _ = cursor.execute_batch(&format!("DROP TABLE IF EXISTS {}", quoted(drawn)));
        answer.map(Answer::Data).ok_or_else(|| {
            refused(format!(
                "sample_entities: the draw of {name} could not be read"
            ))
        })
    }

    fn run_local(
        &self,
        id: sdm_contract::run::ReplyId,
        request: &DataRequest,
        keys: &BTreeMap<AttachmentId, Fetched>,
    ) -> Option<Fetched> {
        self.run_on(&self.duck, id, request, keys)
    }

    /// `request` planned through the DuckDB dialect and run on
    /// `connection`; `None` where it cannot be planned or run.
    fn run_on(
        &self,
        connection: &Connection,
        id: sdm_contract::run::ReplyId,
        request: &DataRequest,
        keys: &BTreeMap<AttachmentId, Fetched>,
    ) -> Option<Fetched> {
        let choice = ProviderChoice::Sql {
            dialect: Dialect("duckdb".to_owned()),
        };
        let plan = sdm_host::default_root()
            .data_access(&choice)
            .ok()?
            .plan(request)
            .ok()?;
        let step = FetchStep {
            id,
            request: request.clone(),
            plan,
            continuation: None,
        };
        duck::answer_with(connection, &step, keys).ok()
    }
}

/// A schema's field names, for messages.
pub fn columns(schema: &SchemaRef) -> Vec<String> {
    schema.fields().iter().map(|f| f.name().clone()).collect()
}

/// One batch of every row of `fetched`.
pub fn concatenated(fetched: &Fetched) -> Result<RecordBatch> {
    concat_batches(&fetched.schema, &fetched.batches).map_err(|e| anyhow!("{e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn whitespace_outside_quotes_is_one_space() {
        assert_eq!(
            normalised("SELECT   `a  b`  FROM\n\n `{catalog}`.`t`\tLIMIT  ?"),
            "SELECT `a  b` FROM `{catalog}`.`t` LIMIT ?"
        );
        assert_eq!(normalised("  x 'a   b'  "), "x 'a   b'");
    }

    #[test]
    fn the_sampling_clause_is_read_as_written() {
        assert_eq!(
            sampling_clause("SELECT 1 FROM t TABLESAMPLE (1e-05 PERCENT) REPEATABLE (2147483655)"),
            Some(("1e-05".to_owned(), Some("2147483655".to_owned())))
        );
        assert_eq!(
            sampling_clause("SELECT * FROM t SAMPLE ROW (5 ROWS)"),
            Some(("5 ROWS".to_owned(), None))
        );
        assert_eq!(sampling_clause("SELECT 1"), None);
    }

    #[test]
    fn a_name_s_last_part_is_unquoted() {
        assert_eq!(last_part("`{catalog}`.`{schema}`.`users`"), "users");
        assert_eq!(last_part("\"DB\".\"S\".\"a\"\"b\""), "a\"b");
        assert_eq!(last_part("users"), "users");
    }

    #[test]
    fn parameters_compare_by_value() {
        assert!(same_parameter(&Value::from(3), &Value::from(3u64)));
        assert!(!same_parameter(&Value::from(3), &Value::from(4)));
        assert!(!same_parameter(&Value::from("3"), &Value::from(3)));
    }
}
