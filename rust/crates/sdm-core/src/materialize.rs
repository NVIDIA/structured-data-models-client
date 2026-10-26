// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `materialize`: a query over a described source, to the payload of its
//! prediction request, as the reference's materializer gathers it through
//! its relational engine (`KumoRelational._get_task_table`, then
//! `materialize_task`) over DuckDB.
//!
//! The steps are the reference's, in its order, one data request for each
//! statement it runs (design, T1.11a row): the source's indexes, for the
//! sampler's "Missing N indices" warning, when a key or foreign key is not
//! one the source enforces and the provider declares index advice
//! (T1.26a); the time bounds of the aggregated tables, or of every table
//! with a time column for a static query; the reservoir sample of the
//! entity table (`sample_entities`), with the entity table's columns the
//! query reads; for each aggregated table, the windowed rows each
//! example's labels aggregate (`neighbors` with a window, `_by_time`);
//! the entity rows of every example (`lookup`); and then the subgraph,
//! walked hop by hop as the reference walks it ([`crate::subgraph`]: the
//! keys each table's rows refer to looked up, the latest rows referring to
//! them read, `_by_fkey`, nulls placed as the provider declares its source
//! places them). The examples' order is numpy's (`sdm_engine::random`), the
//! labels the reference executor's (`sdm_engine::labels`, over the fetched
//! Arrow, decision E19; a window's months by `sdm_engine::time::offset`),
//! the payload the engine's layout (`sdm_request::payload`).
//!
//! What the reference's sampler reads once and keeps, the index check and
//! the time bounds, a run reads only where the memo its host hands back
//! does not hold it (decision E52): the run keeps what it read in a memo it
//! leaves the host, done or failed (`Asker::keep`), and the core holds no
//! state across runs. The coverage warning, which the reference gives once
//! in a process, is given on every run that meets it, and once in a process
//! by the host ([`ONCE_PER_PROCESS`]).
//!
//! It labels, for each entity of a table that refers to no other, a
//! windowed `SUM`, `COUNT`, `AVG`, `MIN` or `MAX` of a table referring to
//! it, a comparison of one with a number or `NULL` (binary), or a column
//! of the entity's own row (static: multiclass or regression), from T1.27;
//! from T2.10 every target the query's validation lets through for the
//! relational model: an aggregation of the rows a filter keeps (`WHERE`), a
//! comparison or `IN` of any column with constants of its type, and `NOT`,
//! `AND` and `OR` of them (binary), each table aggregated read once over
//! the window its aggregations span together;
//! from T1.28a it materializes each of those tasks' requests, with the
//! client's rules the core holds (design, "Host-held semantics"): the run
//! mode the context size selects (a fast run of 1,000 examples and two
//! hops of 32 neighbors, or a best run of 10,000 and 64), the context
//! narrowed to the size asked, the hops the options ask bounded, the
//! entity count, the reserved names, the statistic, the inference
//! options, a static target's columns kept out of what is read and sent,
//! the subgraph's 15 tables, the payload's layout and byte limit
//! ([`sdm_request::payload`]), and the notices of what the request leaves
//! out, each a code and fields the host words; and it refuses with
//! `UNSUPPORTED_REQUEST` naming the rule ([`sdm_engine::not_yet`]) every
//! input whose request takes a rule it does not port yet (register entry
//! P1B-3), and the requests and policies it needs of its provider before
//! anything is fetched.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex, PoisonError};

use arrow_array::cast::AsArray;
use arrow_array::types::{
    Date32Type, Decimal128Type, Float32Type, Float64Type, Int8Type, Int16Type, Int32Type,
    Int64Type, Time64MicrosecondType, Time64NanosecondType, TimestampMicrosecondType,
    TimestampMillisecondType, TimestampNanosecondType, TimestampSecondType, UInt8Type, UInt16Type,
    UInt32Type, UInt64Type,
};
use arrow_array::{Array, ArrayRef, RecordBatch, TimestampNanosecondArray, UInt32Array};
use arrow_schema::{DataType, TimeUnit as ArrowUnit};
use sdm_contract::catalog::{CatalogAnswer, IndexCoverage};
use sdm_contract::data::{
    AttachmentId, BATCH_COLUMN, DataRequest, DescribePart, Filter, IndexAdvice, KeySet,
    MATCHES_COLUMN, NullOrder, OptionalRequest, ProviderChoice, SEED_STORED_TIME, TableColumn,
    TextTimes,
};
use sdm_contract::ops::{
    ColumnRef, ForecastStep, Materialized, MemoBounds, MemoRows, MemoStoredTime, OperationResult,
    ProfileName, ProfileSpec, QueryColumn, QueryFacts, QuerySchema, QueryTable, RequestBatch,
    RequestState, SamplerMemo, Seed, SourceDescription,
};
use sdm_contract::ops::{Dtype, Stype};
use sdm_contract::pql::{Aggregation, ConditionOp, NodeKind, PqlLiteral, PqlNode, ValidatedQuery};
use sdm_contract::{
    ColumnType, CoreError, CoreWarning, Detail, Field, HostFacts, Scalar, Sentence, TimestampUnit,
    Unit, WarningCode,
};
use sdm_engine::graph::{Edge, Graph, Table};
use sdm_engine::keys;
use sdm_engine::labels;
use sdm_engine::not_yet;
use sdm_engine::random::{Entropy, default_rng};
use sdm_engine::sample::{self, SEED_END, SEED_START};
use sdm_engine::time::Instant;
use sdm_engine::time::offset::{Offset, timeframe};
use sdm_engine::time::text::{Reader, Unread};
use sdm_request::options::{self, AskedAnchor, AskedSeed, Kind as Task};
use sdm_request::payload::{self, RESERVED, Sampled};

use crate::describe::{ask, describe};
use crate::profiles::Profile;
use crate::run::{ArrowData, Asker, Attachment, internal};

/// The warning that indices the sampler's reads would use are missing.
pub const MISSING_INDICES: &str = WarningCode::MissingIndices.as_str();
/// The warning that a large entity table gave fewer than half the context
/// examples asked for, which the reference gives once in a process
/// (`_coverage_warned`).
pub const CONTEXT_COVERAGE: &str = WarningCode::ContextCoverage.as_str();

/// The warnings the reference gives once in a process: a host gives each
/// once in its process (decision E52), `sdm-host` for every binding, so
/// the core, which holds no state across runs, gives it on every run that
/// meets it.
pub const ONCE_PER_PROCESS: &[&str] = &[CONTEXT_COVERAGE];

/// The notice that the default run mode gathered as many context examples
/// as it takes (the client's `_warn_of_a_context_the_default_mode_capped`).
pub const CONTEXT_CAPPED: &str = WarningCode::ContextCapped.as_str();
/// The notice that tables of the schema no relationship connects to the
/// entity table are not in the request (`_warn_of_tables_left_out`).
pub const TABLES_UNLINKED: &str = WarningCode::TablesUnlinked.as_str();
/// The notice that tables linked into the schema were reached by no row
/// the sampler read (`_warn_of_tables_left_out`).
pub const TABLES_UNREACHED: &str = WarningCode::TablesUnreached.as_str();
/// The notice that columns the request sends are of a meaning the model
/// does not read (`_warn_of_columns_the_model_does_not_read`).
pub const COLUMNS_UNREAD: &str = WarningCode::ColumnsUnread.as_str();
/// The notice that context examples answering the question were taken out
/// (`_drop_context_that_answers_the_question`).
pub const CONTEXT_ANSWERS: &str = WarningCode::ContextAnswers.as_str();
/// The seed the reference samples with when none is given.
const DEFAULT_SEED: u128 = 42;
/// How many times as many candidates as examples a run tries
/// (`max_pq_iterations`' default).
const ITERATIONS: usize = 10;
/// The most tables a request's subgraph carries (`_MAX_SUBGRAPH_TABLES`).
const SUBGRAPH_TABLES: usize = 15;
/// The most entities one `materialize` gathers where the host names no
/// limit (`main`'s `MOST_ENTITIES`): every call's payload is built, with
/// its own context, before any is sent, so the whole population is held at
/// once.
pub const MOST_ENTITIES: u64 = 100_000;
/// The most model calls one request is gathered as where the host names no
/// limit (`main`'s `MOST_CALLS`): a context is gathered for every call.
pub const MOST_CALLS: u32 = 100;
/// The most rows an instance table may hold (`MAX_TABLE_ROWS`).
const TABLE_ROWS: usize = 10_000;
/// How many examples the reference labels at once, stopping once it has
/// the run mode's (`_sample_target_set`'s `batch_size`): the first batch
/// holds every example a context takes whenever all of it is labelled.
const LABEL_BATCH: usize = 10_000;
/// `pd.Timestamp.max` and `min`, in nanoseconds: the bounds of a table
/// whose time column holds no value.
const TIME_MAX: i64 = i64::MAX;
const TIME_MIN: i64 = -i64::MAX;
const DAY_NS: i128 = 86_400_000_000_000;
/// The version of [`RequestState`]'s format this build writes: the
/// contract's, which `predict` reads (decision H23).
pub use sdm_contract::ops::REQUEST_STATE_VERSION;

/// What a `materialize` start gives its body.
#[derive(Clone)]
pub(crate) struct Inputs {
    /// Fixed frame consumption, not a DataLayer or optional-provider fallback.
    pub frames: Option<crate::frame_materialize::Context>,
    pub description: SourceDescription,
    pub query: String,
    pub indices: Option<ArrowData>,
    /// The rows a ranking is over (`Materialize.candidates`), as the host
    /// gave them (T2.8).
    pub candidates: Option<Vec<sdm_contract::ops::GivenCandidate>>,
    pub anchor_time: Option<Scalar>,
    pub seed: Option<Seed>,
    pub unseeded: bool,
    pub anchor_entity: bool,
    /// The call's option values (`Materialize.values`), as read.
    pub values: Option<sdm_wire::json::Value>,
    pub options: Vec<String>,
    /// The most steps a forecast is asked as (`Materialize.most_parts`).
    pub most_parts: Option<u32>,
    /// The most entities one `materialize` gathers
    /// (`Materialize.most_entities`).
    pub most_entities: Option<u64>,
    /// The most calls one request is gathered as
    /// (`Materialize.most_calls`).
    pub most_calls: Option<u32>,
    pub facts: HostFacts,
    /// The zones the run resolves names with: the host's local zone, which
    /// places a date literal's two-digit year (`pql::local_year`).
    pub zones: Arc<dyn sdm_engine::time::Zones>,
    /// The memo the host handed back, made under this provider and
    /// description (decision E52).
    pub memo: Option<SamplerMemo>,
    /// The fingerprint of this provider and description ([`fingerprint`]).
    pub source: String,
    /// The schema as the host holds it, where the run reads its tables once
    /// the request's inputs are checked (`describe::initialize`): every
    /// table of a source as it stands at this run (the owner's plan B), or
    /// a host provider's columns that hold no facts; `description` is read
    /// until then by the columns' facts, or data types where they hold
    /// none (T3.4).
    pub pending: Option<sdm_contract::ops::HeldSchema>,
    /// The data type names the host gave its columns (T3.16), which the
    /// request's checks before the source is read take, as the reference's
    /// `check_consistent` reads the caller's schema.
    pub types: sdm_schema::assemble::HostTypes,
    /// The test-only `labels` operation's run: it finishes with the
    /// context's labels, before anything of the task's layout is read.
    #[cfg(feature = "test-ops")]
    pub labels_only: bool,
}

/// The fingerprint a sampler memo is made under: FNV-1a, as sixteen hex
/// digits, over the JSON of what the memo's reads depend on, the provider
/// and, of each table, its name, source name, key, time column and which
/// columns the source keeps unique, and the links (the index check reads
/// those, the time bounds the table and its time column). A host keeps a
/// materializer's memo for that materializer; this only keeps a memo of
/// another source, or of a schema whose keys, links or time columns were
/// edited since, from being read, the threat being honest drift, not a
/// forger. A column's type or meaning edited is not in it: the reference's
/// sampler, built once, keeps what it read across such an edit too.
pub fn fingerprint(
    provider: &ProviderChoice,
    description: &SourceDescription,
) -> Result<String, CoreError> {
    let tables: Vec<_> = description
        .tables
        .iter()
        .map(|t| {
            let unique: Vec<&str> = t
                .columns
                .iter()
                .filter(|c| c.unique)
                .map(|c| c.name.as_str())
                .collect();
            (&t.name, &t.source, &t.primary_key, &t.time_column, unique)
        })
        .collect();
    let text = serde_json::to_vec(&(provider, tables, &description.links))
        .map_err(|_| internal("a description does not serialize"))?;
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in text {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    Ok(format!("{hash:016x}"))
}

/// Refuses a sampler memo no run left, naming `sampler_state`, before
/// anything of it is used: one naming a table's bounds twice, or bounds no
/// read gives (a smallest time after the largest, or one of the two
/// missing without the other: a time column holding no time gives
/// neither); or one holding bounds or stored times with the index check
/// not made, which every run makes before it reads either (M6).
pub fn check_memo(memo: &SamplerMemo) -> Result<(), CoreError> {
    let mut tables = BTreeSet::new();
    let read = !memo.bounds.is_empty() || !memo.stored_times.is_empty();
    let sound = (memo.indexes_checked || !read)
        && memo.bounds.iter().all(|b| {
            tables.insert(&b.table)
                && match (b.min, b.max) {
                    (Some(min), Some(max)) => min <= max,
                    (None, None) => true,
                    _ => false,
                }
        });
    let mut spelled = BTreeSet::new();
    if sound && memo.stored_times.iter().all(|s| spelled.insert(&s.table)) {
        return Ok(());
    }
    Err(invalid(
        Sentence::SamplerStateUnread,
        "the sampler state is not one a run left",
        "sampler_state",
    ))
}

/// `error`, said of the forecast's step it happened in, `step` of `steps`
/// (`forecast_step`, `forecast_steps`): with the remedy of asking for fewer
/// steps where the step's window outran the data (`window_outran_the_data`),
/// as the client says each part's refusal (`about_the_part`, T3.17 fix 2).
pub(crate) fn in_step(error: CoreError, step: u32, steps: u32) -> CoreError {
    let outran =
        error.fields.get("window_outran_the_data") == Some(&Detail::Scalar(Scalar::Bool(true)));
    error
        .with_field("forecast_step", Detail::Scalar(Scalar::UInt32(step)))
        .with_field("forecast_steps", Detail::Scalar(Scalar::UInt32(steps)))
        .in_part(if outran {
            Sentence::PartForecastStepOutran
        } else {
            Sentence::PartForecastStep
        })
}

fn invalid(sentence: Sentence, message: &str, field: &str) -> CoreError {
    CoreError::new(sentence, message).with_field("field", Detail::text(field))
}

fn count(n: usize) -> Detail {
    Detail::Scalar(Scalar::UInt64(u64::try_from(n).unwrap_or(u64::MAX)))
}

/// A seed's value as decimal text, `-` before a negative one, of any size,
/// for the host's words.
pub(crate) fn seed_text(seed: Option<&Seed>) -> String {
    let Some(seed) = seed else {
        return String::new();
    };
    // Each hexadecimal digit into a little-endian vector of decimal digits.
    let mut digits: Vec<u8> = vec![0];
    for hex in seed.magnitude_hex().chars() {
        let mut carry = hex.to_digit(16).unwrap_or(0);
        for digit in &mut digits {
            let value = u32::from(*digit) * 16 + carry;
            *digit = (value % 10) as u8;
            carry = value / 10;
        }
        while carry > 0 {
            digits.push((carry % 10) as u8);
            carry /= 10;
        }
    }
    while digits.len() > 1 && digits.last() == Some(&0) {
        digits.pop();
    }
    let text: String = digits.iter().rev().map(|d| char::from(b'0' + d)).collect();
    if seed.is_negative() && text != "0" {
        format!("-{text}")
    } else {
        text
    }
}

/// The scalar a sample's seed travels as: as exact as the integer is, the
/// narrowest of `u64` and a 38-digit decimal that holds it.
pub fn seed_scalar(seed: &Seed) -> Result<Scalar, CoreError> {
    let value = seed
        .to_u128()
        .filter(|v| *v < 10_u128.pow(38))
        .ok_or_else(|| not_yet("seed_size"))?;
    Ok(match u64::try_from(value) {
        Ok(small) => Scalar::UInt64(small),
        Err(_) => Scalar::Decimal128 {
            precision: 38,
            scale: 0,
            value: value as i128,
        },
    })
}

/// The schema PQL resolves the query against, as the client's
/// `to_schema_definition` writes it from the schema the description is:
/// a table's key where it has one column, its time column, and a group of
/// the two columns of each link, ordered by table and column.
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
                        dtype: c.dtype,
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
                let mut group: Vec<ColumnRef> = link.clone();
                group.sort();
                group
            })
            .collect(),
    }
}

/// A windowed aggregation of a table referring to the entity's.
#[derive(Debug, Clone)]
struct Windowed {
    aggregation: labels::Aggregation,
    /// The aggregated table, and its column; none for the table's `*`.
    table: String,
    column: Option<String>,
    /// The condition choosing the rows aggregated (`WHERE`), over the
    /// table's columns: a [`Shape`] whose columns are the table's.
    filter: Option<Box<Shape>>,
    /// The window's ends, as offsets from each example's anchor.
    start: Offset,
    end: Offset,
}

/// What a condition tests its operand against.
#[derive(Debug, Clone)]
enum Test {
    /// `op constant`; no constant is `NULL`.
    Compare(labels::RelOp, Option<labels::Constant>),
    /// `IN (constants)`.
    In(Vec<labels::Constant>),
}

/// What a query predicts, in the shapes this build labels.
#[derive(Debug, Clone)]
enum Shape {
    /// A column of the entity's own row: a static target, or a condition's
    /// operand; in a filter, a column of the rows aggregated.
    Column(String),
    Aggregated(Windowed),
    /// A column or an aggregation tested.
    Condition {
        target: Box<Shape>,
        test: Test,
    },
    Not(Box<Shape>),
    Both {
        op: labels::BoolOp,
        left: Box<Shape>,
        right: Box<Shape>,
    },
}

impl Shape {
    /// The target's windowed aggregations, in the order the reference's
    /// sampler meets them (`_add_time_offset`, depth first, left to
    /// right); a query with one is temporal (`QueryType.TEMPORAL`), one
    /// without static. A filter holds none.
    fn windowed<'a>(&'a self, into: &mut Vec<&'a Windowed>) {
        match self {
            Shape::Column(_) => {}
            Shape::Aggregated(windowed) => into.push(windowed),
            Shape::Condition { target, .. } | Shape::Not(target) => target.windowed(into),
            Shape::Both { left, right, .. } => {
                left.windowed(into);
                right.windowed(into);
            }
        }
    }

    /// The entity table's columns the target reads outside its
    /// aggregations (`get_exclude_cols_dict`'s).
    fn entity_columns(&self, into: &mut BTreeSet<String>) {
        match self {
            Shape::Column(column) => {
                into.insert(column.clone());
            }
            Shape::Aggregated(_) => {}
            Shape::Condition { target, .. } | Shape::Not(target) => target.entity_columns(into),
            Shape::Both { left, right, .. } => {
                left.entity_columns(into);
                right.entity_columns(into);
            }
        }
    }
}

/// The parts of a validated query this build materializes.
struct Asked {
    entity_table: String,
    entity_column: String,
    target: Shape,
    /// The window the target's aggregations span together, merged as the
    /// query's validation merges them (`target_ast.date_offset_range`);
    /// none for a static target.
    range: Option<sdm_contract::pql::Window>,
    task: Task,
    /// The conditions that choose the examples, each over the entity's row
    /// and its windowed aggregations: the entity filter (`WHERE`), then
    /// the assumption (`ASSUMING`), and-ed into the examples' mask as the
    /// reference's executor and-s them (`execute`, T2.13b).
    choosing: Vec<Shape>,
}

impl Asked {
    /// The windowed aggregations the labels read, in the order the
    /// reference's sampler meets them (`_get_query_time_offset_dict`): the
    /// target's, then the entity filter's, then the assumption's.
    fn every_windowed(&self) -> Vec<&Windowed> {
        let mut windows = Vec::new();
        self.target.windowed(&mut windows);
        for shape in &self.choosing {
            shape.windowed(&mut windows);
        }
        windows
    }
}

fn column_of(node: &PqlNode) -> Option<&ColumnRef> {
    match &node.kind {
        NodeKind::Column { column } => Some(column),
        _ => None,
    }
}

/// The query's shape, or the rule it takes that this build does not port.
///
/// `EVALUATE` and `EXPLAIN`, and the entities `FOR t.c = 1` or
/// `FOR t.c IN (1, 2)` names, gather what the same query without them
/// gathers, as the reference's engine gathers it: its task table reads
/// neither prefix, and the entities scored are the call's indices
/// (`_get_task_table`, a forecast's named entity replaced by the first
/// index). The query text the payload carries keeps them as written.
fn asked(query: &ValidatedQuery) -> Result<Asked, CoreError> {
    if query.problem_type.is_some() || query.num_forecasts != 1 || query.top_k.is_some() {
        return Err(not_yet("problem_type"));
    }
    // The entity column, or the column a filter chooses the examples of
    // (`execute_filter` over the entity's row and its aggregations).
    let (entity, filter) = match &query.entity.kind {
        NodeKind::Filter { target, condition } => (
            column_of(target).ok_or_else(|| unvalidated("an entity filter of no column"))?,
            Some(condition.as_ref()),
        ),
        _ => (
            column_of(&query.entity).ok_or_else(|| unvalidated("an entity that is no column"))?,
            None,
        ),
    };
    let operand = |node: &PqlNode| shape(node, &entity.table);
    let choosing = filter
        .into_iter()
        .chain(query.assuming.as_ref())
        .map(|condition| condition_shape(condition, &operand))
        .collect::<Result<Vec<_>, _>>()?;
    let target = shape(&query.target, &entity.table)?;
    let task = match &target {
        Shape::Condition { .. } | Shape::Not(_) | Shape::Both { .. } => Task::Binary,
        Shape::Aggregated(windowed)
            if windowed.aggregation == labels::Aggregation::ListDistinct =>
        {
            Task::TemporalLink
        }
        Shape::Aggregated(_) => Task::Regression,
        Shape::Column(_) => match query.target.stype {
            Some(Stype::Id | Stype::Categorical) => Task::Multiclass,
            Some(Stype::Numerical) => Task::Regression,
            // The reference raises `NotImplementedError` here.
            _ => return Err(not_yet("target_stype")),
        },
    };
    let mut windows = Vec::new();
    target.windowed(&mut windows);
    let range = match (windows.is_empty(), query.target.range) {
        (true, _) => None,
        (false, Some(range)) => Some(range),
        (false, None) => return Err(internal("a windowed target spans no window")),
    };
    Ok(Asked {
        entity_table: entity.table.clone(),
        entity_column: entity.column.clone(),
        target,
        range,
        task,
        choosing,
    })
}

/// A target the query's validation refuses for the relational model before
/// it reaches here (`COUNT_DISTINCT`, `FIRST` and `LAST`; a window with no start; a column of another table than the
/// entity's, outside an aggregation, or of another than the aggregated
/// table's in a filter; a table's `*` or a constant as the target; an
/// operator other than a comparison and `IN`; a condition of a condition):
/// a fault of this build, never the caller's.
fn unvalidated(what: &str) -> CoreError {
    internal(&format!(
        "the query's validation let through what it refuses for the relational model: {what}"
    ))
}

/// The target `node` as a [`Shape`]: a column of the entity's row; a
/// windowed `SUM`, `COUNT`, `AVG`, `MIN` or `MAX` of a column or of the
/// rows a filter keeps; a comparison or `IN` of either with constants;
/// `NOT`, `AND` and `OR` of those (T2.10).
fn shape(node: &PqlNode, entity_table: &str) -> Result<Shape, CoreError> {
    match &node.kind {
        // A join holds an aggregation (`execute_join`).
        NodeKind::Join { target, .. } => match &target.kind {
            NodeKind::Aggregation { .. } => shape(target, entity_table),
            _ => Err(unvalidated("a join to a table's column")),
        },
        NodeKind::Column { column } => {
            if column.table == entity_table && column.column != "*" {
                Ok(Shape::Column(column.column.clone()))
            } else {
                Err(unvalidated("a column of another table, or a table's *"))
            }
        }
        NodeKind::Aggregation {
            aggregation,
            target,
            window,
            ..
        } => {
            let aggregation = match aggregation {
                Aggregation::Sum => labels::Aggregation::Sum,
                Aggregation::Count => labels::Aggregation::Count,
                Aggregation::Avg => labels::Aggregation::Avg,
                Aggregation::Min => labels::Aggregation::Min,
                Aggregation::Max => labels::Aggregation::Max,
                // `LIST_DISTINCT` of a foreign key, which the validation lets
                // through as a temporal link prediction (`_get_task_type`):
                // its ordered FK labels are gathered below.
                Aggregation::ListDistinct => labels::Aggregation::ListDistinct,
                _ => return Err(unvalidated("an aggregation the model does not take")),
            };
            // A column of the aggregated table, or its rows a filter
            // chooses (`execute_filter`).
            let (column, filter) = match &target.kind {
                NodeKind::Filter { target, condition } => {
                    let column =
                        column_of(target).ok_or_else(|| unvalidated("a filter of no column"))?;
                    let filter =
                        condition_shape(condition, &|node| filter_column(node, &column.table))?;
                    (column, Some(Box::new(filter)))
                }
                _ => (
                    column_of(target).ok_or_else(|| unvalidated("an aggregation of no column"))?,
                    None,
                ),
            };
            let window = window.ok_or_else(|| unvalidated("an aggregation with no window"))?;
            let start = window
                .start
                .ok_or_else(|| unvalidated("a window with no start"))?;
            Ok(Shape::Aggregated(Windowed {
                aggregation,
                table: column.table.clone(),
                column: (column.column != "*").then(|| column.column.clone()),
                filter,
                start: Offset::of(start, window.unit),
                end: Offset::of(window.end, window.unit),
            }))
        }
        NodeKind::Condition { .. } | NodeKind::Logical { .. } => {
            condition_shape(node, &|node| shape(node, entity_table))
        }
        NodeKind::Filter { .. } => Err(unvalidated("a filter outside an aggregation")),
        NodeKind::Constant { .. } => Err(unvalidated("a constant target")),
    }
}

/// A filter's column: one of the aggregated table's, the validation
/// refusing any other.
fn filter_column(node: &PqlNode, table: &str) -> Result<Shape, CoreError> {
    match column_of(node) {
        Some(column) if column.table == table && column.column != "*" => {
            Ok(Shape::Column(column.column.clone()))
        }
        _ => Err(unvalidated(
            "a filter's operand that is not the table's column",
        )),
    }
}

/// A condition, `NOT`, `AND` or `OR` as a [`Shape`], each condition's
/// operand read by `operand`.
fn condition_shape(
    node: &PqlNode,
    operand: &dyn Fn(&PqlNode) -> Result<Shape, CoreError>,
) -> Result<Shape, CoreError> {
    match &node.kind {
        NodeKind::Condition {
            target, op, value, ..
        } => {
            let target = operand(target)?;
            if !matches!(target, Shape::Column(_) | Shape::Aggregated(_)) {
                return Err(unvalidated("a condition of a condition"));
            }
            Ok(Shape::Condition {
                target: Box::new(target),
                test: test_of(*op, value)?,
            })
        }
        NodeKind::Logical { op, left, right } => {
            let left = Box::new(condition_shape(left, operand)?);
            let op = match op {
                sdm_contract::pql::LogicalOp::Not => return Ok(Shape::Not(left)),
                sdm_contract::pql::LogicalOp::And => labels::BoolOp::And,
                sdm_contract::pql::LogicalOp::Or => labels::BoolOp::Or,
            };
            let right = right
                .as_ref()
                .ok_or_else(|| unvalidated("AND or OR with one side"))?;
            Ok(Shape::Both {
                op,
                left,
                right: Box::new(condition_shape(right, operand)?),
            })
        }
        _ => Err(unvalidated(
            "a logical operation of something not a condition",
        )),
    }
}

/// What a condition of operator `op` tests its operand against: `value`,
/// a constant, or a list of them for `IN`.
fn test_of(op: ConditionOp, value: &PqlNode) -> Result<Test, CoreError> {
    let NodeKind::Constant { literal } = &value.kind else {
        return Err(unvalidated(
            "a condition compared with something not a constant",
        ));
    };
    let rel = match op {
        ConditionOp::Eq => labels::RelOp::Eq,
        ConditionOp::Ne => labels::RelOp::Ne,
        ConditionOp::Lt => labels::RelOp::Lt,
        ConditionOp::Le => labels::RelOp::Le,
        ConditionOp::Gt => labels::RelOp::Gt,
        ConditionOp::Ge => labels::RelOp::Ge,
        ConditionOp::In => {
            let PqlLiteral::Array { items } = literal else {
                return Err(unvalidated("IN with no list"));
            };
            let values = items
                .iter()
                .map(|item| match &item.kind {
                    NodeKind::Constant { literal } => {
                        constant(literal)?.ok_or_else(|| unvalidated("NULL in an IN list"))
                    }
                    _ => Err(unvalidated("an IN list of something not a constant")),
                })
                .collect::<Result<Vec<_>, _>>()?;
            return Ok(Test::In(values));
        }
        _ => return Err(unvalidated("an operator other than a comparison and IN")),
    };
    Ok(Test::Compare(rel, constant(literal)?))
}

/// A condition's constant as the executor reads it (`typed_value`); none
/// for `NULL`. An integer past what a condition compares is refused by name
/// (`condition_value`).
fn constant(literal: &PqlLiteral) -> Result<Option<labels::Constant>, CoreError> {
    use labels::Constant;
    Ok(Some(match literal {
        PqlLiteral::Null => return Ok(None),
        PqlLiteral::Int { value } => {
            Constant::Int(value.parse().map_err(|_| not_yet("condition_value"))?)
        }
        PqlLiteral::Float { text } => {
            Constant::Float(text.parse().map_err(|_| not_yet("condition_value"))?)
        }
        PqlLiteral::Bool { value } => Constant::Bool(*value),
        PqlLiteral::String { value } => Constant::Text(value.clone()),
        PqlLiteral::Time { value, offset, .. } => Constant::Time(time_constant(value, *offset)?),
        PqlLiteral::Array { .. } => return Err(unvalidated("a list compared with one value")),
    }))
}

/// A date literal's time, `YYYY-MM-DDTHH:MM:SS` as validation reads it, in
/// nanoseconds, as `pd.to_datetime` reads the literal: a time past what a
/// `Timestamp` holds is the caller's error, as pandas' out-of-bounds
/// refusal is; so is a literal with a UTC offset, which the reference
/// cannot compare with times that have none (the query's grammar writes
/// none).
fn time_constant(value: &str, offset: Option<i32>) -> Result<i64, CoreError> {
    let refused = |sentence, message: &str| {
        CoreError::new(sentence, message).with_field("field", Detail::text("query"))
    };
    if offset.is_some() {
        return Err(refused(
            Sentence::TimeConstantOffset,
            "a time constant with a UTC offset cannot be compared with times that have none",
        ));
    }
    let civil = (|| {
        let (date, time) = value.split_once('T')?;
        let mut date = date.rsplitn(3, '-');
        let day: u8 = date.next()?.parse().ok()?;
        let month: u8 = date.next()?.parse().ok()?;
        let year: i64 = date.next()?.parse().ok()?;
        let mut time = time.splitn(3, ':');
        let hour: u8 = time.next()?.parse().ok()?;
        let minute: u8 = time.next()?.parse().ok()?;
        let second: u8 = time.next()?.parse().ok()?;
        sdm_engine::time::Civil::new(year, month, day, hour, minute, second, 0)
    })()
    .ok_or_else(|| internal("a time constant validation read does not read as a date"))?;
    i64::try_from(Instant::from_civil(&civil, 0).nanos())
        .ok()
        .filter(|nanos| *nanos != i64::MIN)
        .ok_or_else(|| {
            refused(
                Sentence::TimeConstantPast,
                "a time constant past what a timestamp of nanoseconds holds",
            )
        })
}

/// One batch of a fetch's answer, all its rows.
pub(crate) fn one_batch(data: &ArrowData) -> Result<RecordBatch, CoreError> {
    arrow_select::concat::concat_batches(&data.schema, &data.batches)
        .map_err(|_| internal("a fetch's batches do not concatenate"))
}

pub(crate) fn named<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a ArrayRef, CoreError> {
    batch.column_by_name(name).ok_or_else(|| {
        crate::run::schema_mismatch(
            "expected",
            Some(name),
            "an answer lacks a column it asked for",
        )
    })
}

/// The column of an answer the core names by its own alias, `name`, as
/// the source folds an unquoted one (`sdm_schema::catalog::result_column`).
pub(crate) fn aliased<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a ArrayRef, CoreError> {
    sdm_schema::catalog::result_column(batch, name).ok_or_else(|| {
        crate::run::schema_mismatch(
            "expected",
            Some(name),
            "an answer lacks a column it asked for",
        )
    })
}

/// A time column's values in nanoseconds, a zone-less timestamp's only.
fn times_ns(array: &ArrayRef, column: &str) -> Result<Vec<Option<i64>>, CoreError> {
    times_ns_from(array, column, false)
}

/// DuckDB's `'-infinity'` in a timestamp of any unit: `-(2^63 - 1)`.
const NEGATIVE_INFINITY: i64 = -i64::MAX;

/// A time column's values in nanoseconds, as [`times_ns`]; with
/// `earliest`, for the smallest times of the time bounds, [`NEGATIVE_INFINITY`] in a coarser unit read as
/// [`TIME_MIN`], `pd.Timestamp.min`: the reference reads that bound (the
/// time bounds' cast to `TIMESTAMP` turns a `TIMESTAMP_NS`, `_S` or `_MS`
/// `'-infinity'` into the microsecond one) as `datetime.min`, 0001-01-01,
/// which is before every anchor a nanosecond count holds, as `TIME_MIN`
/// is, and goes on (T1.36 fix 3). Any other time past what nanoseconds
/// hold, a finite one or `'infinity'`, is refused as before (DT-14).
fn times_ns_from(
    array: &ArrayRef,
    column: &str,
    earliest: bool,
) -> Result<Vec<Option<i64>>, CoreError> {
    let DataType::Timestamp(unit, None) = array.data_type() else {
        return Err(not_yet("time_type"));
    };
    let per: i64 = match unit {
        ArrowUnit::Second => 1_000_000_000,
        ArrowUnit::Millisecond => 1_000_000,
        ArrowUnit::Microsecond => 1_000,
        ArrowUnit::Nanosecond => 1,
    };
    let raw: Vec<Option<i64>> = match unit {
        ArrowUnit::Second => array.as_primitive::<TimestampSecondType>().iter().collect(),
        ArrowUnit::Millisecond => array
            .as_primitive::<TimestampMillisecondType>()
            .iter()
            .collect(),
        ArrowUnit::Microsecond => array
            .as_primitive::<TimestampMicrosecondType>()
            .iter()
            .collect(),
        ArrowUnit::Nanosecond => array
            .as_primitive::<TimestampNanosecondType>()
            .iter()
            .collect(),
    };
    raw.into_iter()
        .map(|v| {
            v.map(|v| match v.checked_mul(per) {
                Some(nanos) => Ok(nanos),
                None if earliest && v == NEGATIVE_INFINITY => Ok(TIME_MIN),
                None => Err(sample::out_of_range(column)),
            })
            .transpose()
        })
        .collect()
}

/// How `materialize` reads the times a source returns as anything but a
/// zone-less timestamp, as the reference's engine reads every column a
/// fetch returns whose meaning is a timestamp (`Table._sanitize`, its
/// `to_datetime`): a timestamp with a zone at its UTC instants, the zone
/// dropped; a date at its midnight; text by the reading the provider
/// declares (`ProviderCapabilities::text_times`, register DT-7), never by
/// a dialect: by the core's list (`date-rules.md`, "The core's text
/// formats"), refusing a value outside it, an order of day and month the
/// fetch leaves open and offsets mixed by name ([`text_unread`]), and a
/// blank value dropped and warned of (`TIME_VALUES_UNREAD`), as the
/// reference drops it and warns.
///
/// It also keeps, for a provider whose source compares a window's bounds
/// with its stored text (`DataLayer::compares_stored_times`, SQLite's),
/// one stored value of each windowed table's time column, read once before
/// that table's first windowed read and kept in the sampler's memo.
/// A fresh SQL source read expires it under plan B; host providers keep
/// it across the materializer's runs (decision E52). The stored value is
/// the reference SQLite sampler's `_time_spelling`, named
/// it on each window bound's seed column ([`SEED_STORED_TIME`]) for the
/// data layer to spell the bound by.
pub(crate) struct Times {
    reader: Reader,
    text: Option<TextTimes>,
    /// Each windowed table's stored value, by the schema's name of the
    /// table: the text read, or none where its column held no text.
    stored: Mutex<BTreeMap<String, Option<String>>>,
}

impl Times {
    /// The reading of a run's times, holding the memo's stored values
    /// after the source refresh policy has expired any stale data facts.
    pub(crate) fn new(reader: Reader, text: Option<TextTimes>, memo: &SamplerMemo) -> Self {
        Times {
            reader,
            text,
            stored: Mutex::new(
                memo.stored_times
                    .iter()
                    .map(|s| (s.table.clone(), s.value.clone()))
                    .collect(),
            ),
        }
    }

    /// Each table's stored value read so far, for the memo the run leaves.
    fn stored_times(&self) -> Vec<MemoStoredTime> {
        self.stored
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .iter()
            .map(|(table, value)| MemoStoredTime {
                table: table.clone(),
                value: value.clone(),
            })
            .collect()
    }

    /// `request`, a windowed `neighbors` of `table`, with each window
    /// bound's seed column naming one stored value of the window's column
    /// ([`SEED_STORED_TIME`]), where the run's provider compares a bound
    /// with that stored text; the value read first, once in the run, by a
    /// `scan` of one non-null value (`_probe_time_spelling`'s statement).
    /// Any other request, and any provider comparing times as times, is
    /// asked as it is.
    pub(crate) async fn windowed(
        &self,
        asker: &Asker,
        table: &Table,
        mut request: DataRequest,
    ) -> Result<DataRequest, CoreError> {
        if !asker.compares_stored_times() {
            return Ok(request);
        }
        let DataRequest::Neighbors {
            window: Some(window),
            seeds,
            ..
        } = &mut request
        else {
            return Ok(request);
        };
        let known = self
            .stored
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(&table.name)
            .cloned();
        let stored = match known {
            Some(stored) => stored,
            None => {
                let read = ask(
                    asker,
                    DataRequest::Scan {
                        table: table.source.clone(),
                        columns: vec![window.column.clone()],
                        filters: vec![Filter::NotNull {
                            column: window.column.clone(),
                        }],
                        limit: Some(1),
                    },
                )
                .await?;
                // The reference spells the bounds by the value only when it
                // is text, and by its default otherwise.
                let stored = read
                    .batches
                    .iter()
                    .find(|b| b.num_rows() > 0)
                    .and_then(|b| {
                        let column = b.column(0);
                        let text = column.as_string_opt::<i32>()?;
                        (!text.is_null(0)).then(|| text.value(0).to_owned())
                    });
                self.stored
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .insert(table.name.clone(), stored.clone());
                stored
            }
        };
        if let Some(stored) = stored {
            let bounds: Vec<String> = window
                .lower
                .iter()
                .chain(&window.upper)
                .map(|b| b.seed_column.clone())
                .collect();
            for column in &mut seeds.columns {
                if bounds.contains(&column.name) {
                    column
                        .metadata
                        .insert(SEED_STORED_TIME.to_owned(), stored.clone());
                }
            }
        }
        Ok(request)
    }

    /// A bound of the time bounds in nanoseconds (`earliest` for the
    /// smallest, [`times_ns_from`]), read as the reference reads each bound
    /// its source returns (`Timestamp(value)`), whatever the column's type:
    /// a zone-less timestamp as it is; a timestamp with a zone at its UTC
    /// instant, the zone dropped, as the reference's engine reads an aware
    /// time; a date at its midnight; a column holding no time, of whatever
    /// type the source's driver gave it, as none; text, as a source with no
    /// time type answers its stored values (SQLite's) or a text column's
    /// (Postgres's), read by the reading the provider declares, a value no
    /// reading takes refused by name ([`text_unread`]).
    fn bounds(
        &self,
        array: &ArrayRef,
        column: &str,
        earliest: bool,
    ) -> Result<Vec<Option<i64>>, CoreError> {
        macro_rules! unzoned {
            ($t:ty) => {
                Arc::new(
                    array
                        .as_primitive::<$t>()
                        .clone()
                        .with_timezone_opt::<Arc<str>>(None),
                ) as ArrayRef
            };
        }
        match array.data_type() {
            DataType::Timestamp(_, None) => times_ns_from(array, column, earliest),
            DataType::Timestamp(unit, Some(_)) => {
                let unzoned = match unit {
                    ArrowUnit::Second => unzoned!(TimestampSecondType),
                    ArrowUnit::Millisecond => unzoned!(TimestampMillisecondType),
                    ArrowUnit::Microsecond => unzoned!(TimestampMicrosecondType),
                    ArrowUnit::Nanosecond => unzoned!(TimestampNanosecondType),
                };
                times_ns_from(&unzoned, column, earliest)
            }
            DataType::Date32 => array
                .as_primitive::<Date32Type>()
                .iter()
                .map(|d| {
                    d.map(|d| Instant::from_days(d).to_count(ArrowUnit::Nanosecond, column))
                        .transpose()
                })
                .collect(),
            _ if array.null_count() == array.len() => Ok(vec![None; array.len()]),
            DataType::Utf8 => {
                let values: Vec<Option<&str>> = array.as_string::<i32>().iter().collect();
                let read = match self.text {
                    Some(TextTimes::ValueByValue) => self.reader.each(&values),
                    Some(TextTimes::ClientRules) => return Err(not_yet("text_time_rules")),
                    None => return Err(not_yet("text_times")),
                }
                .map_err(text_unread)?;
                values
                    .iter()
                    .zip(read)
                    .map(|(value, at)| match (value, at) {
                        (_, Some(at)) => at.to_count(ArrowUnit::Nanosecond, column).map(Some),
                        (None, None) => Ok(None),
                        (Some(_), None) => Err(not_yet("time_type")),
                    })
                    .collect()
            }
            _ => Err(not_yet("time_type")),
        }
    }

    /// `batch`, fetched from `table`, with each column whose meaning is a
    /// timestamp read so.
    pub(crate) fn read(
        &self,
        asker: &Asker,
        table: &Table,
        batch: RecordBatch,
    ) -> Result<RecordBatch, CoreError> {
        // Its big integers in their columns' tiers first (big_integers).
        let (batch, mut read_any) = big_integers(table, batch)?;
        let (schema, mut fields) = (batch.schema(), Vec::new());
        let mut columns = Vec::with_capacity(schema.fields().len());
        for (field, array) in schema.fields().iter().zip(batch.columns()) {
            let timestamp = table
                .column(field.name())
                .is_some_and(|c| c.stype == sdm_contract::ops::Stype::Timestamp);
            match timestamp
                .then(|| self.column(asker, table, field.name(), array))
                .transpose()?
                .flatten()
            {
                Some(read) => {
                    read_any = true;
                    fields.push(arrow_schema::Field::new(
                        field.name(),
                        read.data_type().clone(),
                        field.is_nullable(),
                    ));
                    columns.push(read);
                }
                None => {
                    fields.push(field.as_ref().clone());
                    columns.push(Arc::clone(array));
                }
            }
        }
        if !read_any {
            return Ok(batch);
        }
        RecordBatch::try_new(Arc::new(arrow_schema::Schema::new(fields)), columns)
            .map_err(|_| internal("a fetch's times do not read back into its rows"))
    }

    /// Column `name` of `table` read to a zone-less timestamp, or `None`
    /// when it is one already or of a type no reading takes.
    fn column(
        &self,
        asker: &Asker,
        table: &Table,
        name: &str,
        array: &ArrayRef,
    ) -> Result<Option<ArrayRef>, CoreError> {
        macro_rules! unzoned {
            ($t:ty) => {
                Arc::new(
                    array
                        .as_primitive::<$t>()
                        .clone()
                        .with_timezone_opt::<Arc<str>>(None),
                ) as ArrayRef
            };
        }
        Ok(Some(match array.data_type() {
            DataType::Timestamp(_, None) => return Ok(None),
            // A zoned column keeps its unit, empty or not, so each of a
            // table's answers reads it alike.
            DataType::Timestamp(ArrowUnit::Second, Some(_)) => unzoned!(TimestampSecondType),
            DataType::Timestamp(ArrowUnit::Millisecond, Some(_)) => {
                unzoned!(TimestampMillisecondType)
            }
            DataType::Timestamp(ArrowUnit::Microsecond, Some(_)) => {
                unzoned!(TimestampMicrosecondType)
            }
            DataType::Timestamp(ArrowUnit::Nanosecond, Some(_)) => {
                unzoned!(TimestampNanosecondType)
            }
            // A column holding no value says nothing of its type: a source
            // whose driver types a column by its values (SQLite's) gives an
            // empty answer's columns as integers. It holds no time.
            _ if array.null_count() == array.len() => {
                Arc::new(TimestampNanosecondArray::from(vec![None; array.len()]))
            }
            DataType::Date32 => {
                let days = array.as_primitive::<Date32Type>();
                let nanos = days
                    .iter()
                    .map(|d| {
                        d.map(|d| Instant::from_days(d).to_count(ArrowUnit::Nanosecond, name))
                            .transpose()
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Arc::new(TimestampNanosecondArray::from(nanos))
            }
            DataType::Utf8 => {
                let text = array.as_string::<i32>();
                let values: Vec<Option<&str>> = text.iter().collect();
                let read = match self.text {
                    Some(TextTimes::ValueByValue) => self.reader.each(&values),
                    Some(TextTimes::ClientRules) => return Err(not_yet("text_time_rules")),
                    None => return Err(not_yet("text_times")),
                }
                .map_err(text_unread)?;
                let mut unread = 0;
                let mut examples: Vec<Scalar> = Vec::new();
                for (value, at) in values.iter().zip(&read) {
                    if let (Some(value), None) = (value, at) {
                        unread += 1;
                        let example = Scalar::Utf8((*value).to_owned());
                        if examples.len() < 3 && !examples.contains(&example) {
                            examples.push(example);
                        }
                    }
                }
                if unread > 0 {
                    asker.warn(CoreWarning {
                        fields: [
                            ("table".to_owned(), Detail::text(&table.name)),
                            ("column".to_owned(), Detail::text(name)),
                            ("unread".to_owned(), count(unread)),
                            ("rows".to_owned(), count(values.len())),
                            ("examples".to_owned(), Detail::List(examples)),
                        ]
                        .into(),
                        ..CoreWarning::new(
                            Sentence::TimeValuesUnread,
                            "a time column's values were read as no time and dropped",
                        )
                    });
                }
                let nanos = read
                    .iter()
                    .map(|at| {
                        at.map(|at| at.to_count(ArrowUnit::Nanosecond, name))
                            .transpose()
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Arc::new(TimestampNanosecondArray::from(nanos))
            }
            DataType::Time64(_) => {
                times_of_day_unread(asker, table, name, array);
                return Ok(None);
            }
            _ => return Ok(None),
        }))
    }
}

/// A time of day whose meaning is a timestamp holds no time as the
/// reference's engine reads it (its `to_datetime`): every value is dropped
/// (the request sends each cell null, `sdm_request::payload`), and, as the
/// reference warns, `TIME_VALUES_UNREAD` says how many of the fetch's rows
/// were, with three distinct values (the contract's `time64` scalars).
/// The reference's three are the first in the fetch's row order, which
/// DuckDB varies from run to run; the core's are the earliest three times
/// of day, the same on every run.
fn times_of_day_unread(asker: &Asker, table: &Table, name: &str, array: &ArrayRef) {
    let (unit, values): (Unit, BTreeSet<i64>) = match array.data_type() {
        DataType::Time64(ArrowUnit::Microsecond) => (
            Unit::Us,
            array
                .as_primitive::<Time64MicrosecondType>()
                .iter()
                .flatten()
                .collect(),
        ),
        DataType::Time64(ArrowUnit::Nanosecond) => (
            Unit::Ns,
            array
                .as_primitive::<Time64NanosecondType>()
                .iter()
                .flatten()
                .collect(),
        ),
        _ => return,
    };
    let examples: Vec<Scalar> = values
        .into_iter()
        .take(3)
        .map(|value| Scalar::Time64 { unit, value })
        .collect();
    let unread = array.len() - array.null_count();
    if unread > 0 {
        asker.warn(CoreWarning {
            fields: [
                ("table".to_owned(), Detail::text(&table.name)),
                ("column".to_owned(), Detail::text(name)),
                ("unread".to_owned(), count(unread)),
                ("rows".to_owned(), count(array.len())),
                ("examples".to_owned(), Detail::List(examples)),
            ]
            .into(),
            ..CoreWarning::new(
                Sentence::TimeValuesUnread,
                "a time of day's values were read as no time and dropped",
            )
        });
    }
}

/// The batch positions a `neighbors` or `lookup` answer names.
/// Its column is the core's own alias, found as the source folds it
/// (Snowflake's `__KUMO_BATCH__`, the reference's unquoted alias).
pub(crate) fn batches(batch: &RecordBatch, seeds: usize) -> Result<Vec<usize>, CoreError> {
    let column = aliased(batch, BATCH_COLUMN)?;
    let values = column
        .as_primitive_opt::<Int64Type>()
        .ok_or_else(|| crate::run::schema_mismatch("expected", Some(BATCH_COLUMN), "not int64"))?;
    values
        .iter()
        .map(|v| {
            v.and_then(|v| usize::try_from(v).ok())
                .filter(|v| *v < seeds)
                .ok_or_else(|| {
                    crate::run::schema_mismatch(
                        "expected",
                        Some(BATCH_COLUMN),
                        "a row names no seed asked for",
                    )
                })
        })
        .collect()
}

/// An aggregated column's values, as the reference's executor reads them
/// (`execute_column`): a float at float32, an integer at int64, a decimal
/// exactly; any other type is refused by name (`target_type`).
fn label_values(array: &ArrayRef) -> Result<labels::Values, CoreError> {
    use labels::Values;
    macro_rules! ints {
        ($t:ty) => {
            Values::Int(
                array
                    .as_primitive::<$t>()
                    .iter()
                    .map(|v| v.map(i64::from))
                    .collect(),
            )
        };
    }
    Ok(match array.data_type() {
        DataType::Float64 => Values::Float(
            array
                .as_primitive::<Float64Type>()
                .iter()
                .map(|v| v.map(|v| v as f32))
                .collect(),
        ),
        DataType::Float32 => Values::Float(array.as_primitive::<Float32Type>().iter().collect()),
        DataType::Int8 => ints!(Int8Type),
        DataType::Int16 => ints!(Int16Type),
        DataType::Int32 => ints!(Int32Type),
        DataType::Int64 => ints!(Int64Type),
        DataType::UInt8 => ints!(UInt8Type),
        DataType::UInt16 => ints!(UInt16Type),
        DataType::UInt32 => ints!(UInt32Type),
        DataType::UInt64 => Values::Int(unsigned(array)?),
        DataType::Decimal128(precision, scale) => Values::Decimal {
            values: array.as_primitive::<Decimal128Type>().iter().collect(),
            precision: *precision,
            scale: *scale,
        },
        _ => return Err(not_yet("target_type")),
    })
}

/// Whether a double of `array` is finite and past float32's range, so the
/// executor's cast of it to float32 ([`label_values`], [`static_column`])
/// makes it an infinity, which NumPy warns of (FX-1).
fn float32_overflows(array: &ArrayRef) -> bool {
    array.data_type() == &DataType::Float64
        && array
            .as_primitive::<Float64Type>()
            .iter()
            .flatten()
            .any(|v| v.is_finite() && (v as f32).is_infinite())
}

/// The warning NumPy gives where the executor's cast of a double to
/// float32 overflows (FX-1), which the host shows as the reference's
/// `RuntimeWarning`. Not over a source whose floats the reference reads as
/// pandas' nullable `Float64` (Postgres's, `_sanitize_postgres_frame`, the
/// reading that takes a NaN as missing), whose cast to float32 warns of
/// nothing.
fn warn_float32_overflow(asker: &Asker) {
    if asker.reads_nan_as_missing() {
        return;
    }
    warn(
        asker,
        Sentence::Float32Overflow,
        "a double past float32's range was cast to float32, as an infinity",
    );
}

/// A `UInt64` column at int64, as pandas' `astype('int64')` takes it: a
/// value past int64, which that cast refuses, is refused by name
/// (`target_type`).
fn unsigned(array: &ArrayRef) -> Result<Vec<Option<i64>>, CoreError> {
    array
        .as_primitive::<UInt64Type>()
        .iter()
        .map(|v| {
            v.map(|v| i64::try_from(v).map_err(|_| not_yet("target_type")))
                .transpose()
        })
        .collect()
}

/// A static target's values at `rows` of `array`, the entity rows sampled,
/// as the executor reads them: a float at float32, an integer at int64,
/// a boolean, text, a decimal exactly (compared exactly in a condition,
/// read as a target by `labels::label` and `labels::label_classes`,
/// T2.0c); a time or any other type is refused by name (`target_type`):
/// the reference's task table takes none of them.
fn static_column(array: &ArrayRef, rows: &[usize]) -> Result<labels::Column, CoreError> {
    use labels::Column;
    let taken = take(array, rows)?;
    let array = &taken;
    macro_rules! ints {
        ($t:ty) => {
            Column::Int(
                array
                    .as_primitive::<$t>()
                    .iter()
                    .map(|v| v.map(i64::from))
                    .collect(),
            )
        };
    }
    Ok(match array.data_type() {
        DataType::Float64 => Column::Float(
            array
                .as_primitive::<Float64Type>()
                .iter()
                .map(|v| v.map(|v| v as f32))
                .collect(),
        ),
        DataType::Float32 => Column::Float(array.as_primitive::<Float32Type>().iter().collect()),
        DataType::Int8 => ints!(Int8Type),
        DataType::Int16 => ints!(Int16Type),
        DataType::Int32 => ints!(Int32Type),
        DataType::Int64 => ints!(Int64Type),
        DataType::UInt8 => ints!(UInt8Type),
        DataType::UInt16 => ints!(UInt16Type),
        DataType::UInt32 => ints!(UInt32Type),
        DataType::UInt64 => Column::Int(unsigned(array)?),
        DataType::Boolean => Column::Bool(array.as_boolean().iter().collect()),
        DataType::Decimal128(precision, scale) => Column::Decimal {
            values: array.as_primitive::<Decimal128Type>().iter().collect(),
            precision: *precision,
            scale: *scale,
        },
        DataType::Utf8 => Column::Text(
            array
                .as_string::<i32>()
                .iter()
                .map(|v| v.map(str::to_owned))
                .collect(),
        ),
        DataType::LargeUtf8 => Column::Text(
            array
                .as_string::<i64>()
                .iter()
                .map(|v| v.map(str::to_owned))
                .collect(),
        ),
        DataType::Utf8View => Column::Text(
            array
                .as_string_view()
                .iter()
                .map(|v| v.map(str::to_owned))
                .collect(),
        ),
        _ => return Err(not_yet("target_type")),
    })
}

/// A condition's column at `rows` of `array` (an entity's static column,
/// or a column of the rows a filter tests), as the executor reads it: as
/// [`static_column`] reads a target; a time too, which the reads have
/// made a zone-less timestamp ([`Times::read`]), in nanoseconds; and an
/// unsigned 64-bit column whole, past int64 too, which a condition
/// compares without the target's signed reading (T2.10 fix 2).
fn condition_column(array: &ArrayRef, rows: &[usize]) -> Result<labels::Column, CoreError> {
    match array.data_type() {
        DataType::Timestamp(_, None) => {
            let all = times_ns(array, "")?;
            Ok(labels::Column::Time(rows.iter().map(|&r| all[r]).collect()))
        }
        DataType::UInt64 => Ok(labels::Column::UInt(
            take(array, rows)?
                .as_primitive::<UInt64Type>()
                .iter()
                .collect(),
        )),
        _ => static_column(array, rows),
    }
}

/// The rows `rows` of `array`, in that order.
fn take(array: &ArrayRef, rows: &[usize]) -> Result<ArrayRef, CoreError> {
    let indices = UInt32Array::from_iter_values(rows.iter().map(|r| *r as u32));
    arrow_select::take::take(array.as_ref(), &indices, None)
        .map_err(|_| internal("rows could not be taken"))
}

fn take_batch(batch: &RecordBatch, rows: &[usize]) -> Result<RecordBatch, CoreError> {
    let indices = UInt32Array::from_iter_values(rows.iter().map(|r| *r as u32));
    arrow_select::take::take_record_batch(batch, &indices)
        .map_err(|_| internal("rows could not be taken"))
}

/// Attaches `seeds` and names them as a request's key set.
pub(crate) fn key_set(asker: &Asker, seeds: RecordBatch) -> Result<KeySet, CoreError> {
    let columns = seeds
        .schema()
        .fields()
        .iter()
        .map(|f| {
            Ok(Field {
                name: f.name().clone(),
                column_type: ColumnType::from_arrow(f.name(), f.data_type())?,
                nullable: f.is_nullable(),
                metadata: Default::default(),
            })
        })
        .collect::<Result<Vec<_>, CoreError>>()?;
    let schema = seeds.schema();
    let attachment: AttachmentId =
        asker.attach(Attachment::Arrow(ArrowData::new(schema, vec![seeds])))?;
    Ok(KeySet {
        attachment,
        columns,
    })
}

/// The smallest and largest time of `table`'s time column, as the
/// reference's sampler reads them, once.
struct Bounds {
    min: i64,
    max: i64,
}

/// What the run has read of its source, which it leaves the host
/// (decision E52): the memo handed back, or a new one.
struct Memo<'a> {
    asker: &'a Asker,
    memo: SamplerMemo,
}

impl Memo<'_> {
    /// Keeps the memo as what the run leaves.
    fn keep(&self) {
        self.asker.keep(self.memo.clone());
    }
}

/// The stored time values `times` read, kept in the memo the run leaves
/// where they are new. A SQL source refresh expires them at the next run;
/// a host provider keeps its E52 lifetime.
fn keep_stored_times(memo: &mut Memo<'_>, times: &Times) {
    let stored = times.stored_times();
    if stored != memo.memo.stored_times {
        memo.memo.stored_times = stored;
        memo.keep();
    }
}

/// The time bounds of `tables` together, the smallest of their smallest
/// times and the largest of their largest (`get_min_time` and
/// `get_max_time` over those tables, a table whose time column holds no
/// time counting as `pd.Timestamp.max` and `min`): each as the memo holds
/// it, read by a run of this materializer before, as the reference's
/// sampler reads them once and keeps them; the others read, in one
/// request, and kept. A fresh SQL source read expires earlier bounds, so
/// the memo reuses them only within that run; host providers retain E52.
async fn time_bounds(
    memo: &mut Memo<'_>,
    times: &Times,
    tables: &[&Table],
) -> Result<Bounds, CoreError> {
    let mut unknown: Vec<(&Table, String)> = Vec::new();
    for table in tables {
        let column = table
            .time_column
            .clone()
            .ok_or_else(|| internal("a table bounded in time has no time column"))?;
        let kept = memo
            .memo
            .bounds
            .iter()
            .any(|b| b.table == table.name && b.column == column);
        if !kept {
            unknown.push((table, column));
        }
    }
    if !unknown.is_empty() {
        let read = read_bounds(memo.asker, times, &unknown).await?;
        for ((table, column), (min, max)) in unknown.into_iter().zip(read) {
            memo.memo.bounds.push(MemoBounds {
                table: table.name.clone(),
                column,
                min,
                max,
            });
        }
        memo.keep();
    }
    let mut bounds = Bounds {
        min: TIME_MAX,
        max: TIME_MIN,
    };
    for table in tables {
        let column = table.time_column.as_deref();
        let Some(kept) = memo
            .memo
            .bounds
            .iter()
            .find(|b| b.table == table.name && Some(b.column.as_str()) == column)
        else {
            return Err(internal("a table's time bounds were not kept"));
        };
        bounds.min = bounds.min.min(kept.min.unwrap_or(TIME_MAX));
        bounds.max = bounds.max.max(kept.max.unwrap_or(TIME_MIN));
    }
    Ok(bounds)
}

/// The smallest and largest time of each table's time column, in the
/// order asked, one request for them all.
async fn read_bounds(
    asker: &Asker,
    times: &Times,
    tables: &[(&Table, String)],
) -> Result<Vec<(Option<i64>, Option<i64>)>, CoreError> {
    let answer = ask(
        asker,
        DataRequest::TimeBounds {
            columns: tables
                .iter()
                .map(|(table, column)| TableColumn {
                    table: table.source.clone(),
                    column: column.clone(),
                })
                .collect(),
        },
    )
    .await?;
    let batch = one_batch(&answer)?;
    if batch.num_rows() != tables.len() {
        return Err(crate::run::schema_mismatch(
            "expected",
            None,
            "the time bounds do not hold one row per column",
        ));
    }
    use sdm_contract::data::results::{BOUNDS_MAX, BOUNDS_MIN, BOUNDS_POSITION};
    // A position of either width: DuckDB writes the literal as an `int32`,
    // SQLite's driver reads every integer as an `int64`.
    let positions: Vec<Option<i64>> = {
        let column = aliased(&batch, BOUNDS_POSITION)?;
        if let Some(values) = column.as_primitive_opt::<Int32Type>() {
            values.iter().map(|v| v.map(i64::from)).collect()
        } else if let Some(values) = column.as_primitive_opt::<Int64Type>() {
            values.iter().collect()
        } else {
            return Err(crate::run::schema_mismatch(
                "expected",
                Some(BOUNDS_POSITION),
                "not an integer",
            ));
        }
    };
    let mins = times.bounds(aliased(&batch, BOUNDS_MIN)?, BOUNDS_MIN, true)?;
    let maxs = times.bounds(aliased(&batch, BOUNDS_MAX)?, BOUNDS_MAX, false)?;
    let mut out = vec![None; tables.len()];
    for row in 0..batch.num_rows() {
        let at = positions[row]
            .and_then(|p| usize::try_from(p).ok())
            .filter(|at| *at < tables.len() && out[*at].is_none())
            .ok_or_else(|| {
                crate::run::schema_mismatch(
                    "expected",
                    Some(BOUNDS_POSITION),
                    "a row names no column asked for",
                )
            })?;
        out[at] = Some((mins[row], maxs[row]));
    }
    Ok(out.into_iter().map(|b| b.unwrap_or((None, None))).collect())
}

/// The requests and policies the sampler needs of its provider, each
/// refused by name before anything is fetched, as a capability is refused
/// wherever the core reads it (decision E63): the optional requests it
/// asks (`UNSUPPORTED_REQUEST` naming the request, as `DataAccess::plan`
/// refuses one undeclared), then where its source puts a null when it
/// orders (`null_order`), without which it is not asked for rows in an
/// order. The order the sampler keeps neighbors in, which a provider
/// declaring none would leave to the source.
fn sampling_capabilities(asker: &Asker) -> Result<NullOrder, CoreError> {
    let capabilities = asker.capabilities()?;
    for needed in [
        OptionalRequest::TimeBounds,
        OptionalRequest::SampleEntities,
        OptionalRequest::Neighbors,
        OptionalRequest::Lookup,
    ] {
        if !capabilities.optional.contains(&needed) && capabilities.fallback.is_none() {
            return Err(crate::compose::undeclared(needed.name()));
        }
    }
    capabilities.null_order.ok_or_else(|| not_yet("null_order"))
}

/// The sampler's check of the indexes its reads would use, under the
/// provider's index advice (`ProviderCapabilities::index_advice`): every
/// key and foreign key the source does not enforce as unique, less those
/// an index the advice counts covers, warned of as the reference warns of
/// them. A provider declaring no advice is not asked for its indexes and
/// gets no warning. The data layer reads the source's indexes (decision
/// E54): which columns an index covers, or that it cannot tell, is its
/// dialect's reading, never this module's.
///
/// The check is made once for a materializer (decision E52): a run whose
/// memo says it was made skips it, as the reference builds its sampler,
/// and checks, once; a run that makes it keeps that in its memo, warning
/// given or not.
/// Each table's rows, counted once for a materializer where the provider
/// sizes its sampling by them (`ProviderCapabilities::sample_rows`,
/// `neighbor_lookback`; contract v28), as the reference's warehouse
/// samplers count every table of the graph, in its order, as they are
/// built (`num_rows_dict`); kept in the memo, so a later run counts none
/// again unless the run refreshes its SQL source (plan B), which expires
/// source-data facts. A provider declaring neither is asked nothing.
async fn table_rows(memo: &mut Memo<'_>, graph: &Graph) -> Result<(), CoreError> {
    let capabilities = memo.asker.capabilities()?;
    if !(capabilities.sample_rows || capabilities.neighbor_lookback) {
        return Ok(());
    }
    let mut counted = false;
    for table in &graph.tables {
        if memo.memo.rows.iter().any(|r| r.table == table.name) {
            continue;
        }
        let answer = ask(
            memo.asker,
            DataRequest::CountRows {
                table: table.source.clone(),
                filters: vec![],
            },
        )
        .await?;
        let rows = row_count(&answer)?;
        // A table of no rows, as the reference's warehouse table refuses
        // it when its sampler counts it (`databricks/table.py:402-411`).
        if rows == 0 {
            return Err(sdm_schema::infer::empty_table(
                &memo.asker.read_as(&table.source),
            ));
        }
        memo.memo.rows.push(MemoRows {
            table: table.name.clone(),
            rows,
        });
        counted = true;
    }
    if counted {
        memo.keep();
    }
    Ok(())
}

/// One aggregate row whose first column holds a nonnegative integer count.
fn row_count(answer: &ArrowData) -> Result<u64, CoreError> {
    let batch = one_batch(answer)?;
    let unread =
        || crate::run::schema_mismatch("expected", None, "a row count is not one row of one count");
    if batch.num_rows() != 1 || batch.num_columns() == 0 {
        return Err(unread());
    }
    let column = batch.column(0);
    if column.is_null(0) {
        return Err(unread());
    }
    let count: i128 = if let Some(values) = column.as_primitive_opt::<Int64Type>() {
        values.value(0).into()
    } else if let Some(values) = column.as_primitive_opt::<Int32Type>() {
        values.value(0).into()
    } else if let Some(values) = column.as_primitive_opt::<arrow_array::types::UInt64Type>() {
        values.value(0).into()
    } else if let Some(values) = column.as_primitive_opt::<arrow_array::types::Decimal128Type>() {
        values.value(0)
    } else {
        return Err(unread());
    };
    u64::try_from(count).map_err(|_| unread())
}

/// The rows the memo counted of `table`, if it counted them.
fn rows_of(memo: &SamplerMemo, table: &str) -> Option<u64> {
    memo.rows.iter().find(|r| r.table == table).map(|r| r.rows)
}

/// The rate of rows per referenced row and second of each table with a
/// time column referring to another, for a provider whose neighbor reads
/// look back no further than the reference's warehouse samplers do
/// (`ProviderCapabilities::neighbor_lookback`, `_by_fkey`): the table's
/// rows over the referenced table's, over the seconds its time column
/// spans (at least one), each step a double as the reference's arithmetic
/// rounds it, the span its `Timedelta.total_seconds()` (whole microseconds
/// over a million). None for any other provider, and none for a table
/// whose times or counts are not known.
async fn lookbacks(
    memo: &mut Memo<'_>,
    times: &Times,
    graph: &Graph,
) -> Result<Vec<crate::subgraph::Lookback>, CoreError> {
    if !memo.asker.capabilities()?.neighbor_lookback {
        return Ok(vec![]);
    }
    let mut timed: Vec<&Table> = Vec::new();
    for table in &graph.tables {
        if table.time_column.is_some()
            && graph.tables.iter().any(|dst| {
                graph
                    .referring(&dst.name)
                    .iter()
                    .any(|(src, _)| *src == table.name)
            })
        {
            timed.push(table);
        }
    }
    if timed.is_empty() {
        return Ok(vec![]);
    }
    // Every bound read in one request, as the reference reads its
    // sampler's bounds together.
    time_bounds(memo, times, &timed).await?;
    let mut out = Vec::new();
    for dst in &graph.tables {
        for (src, _) in graph.referring(&dst.name) {
            let Some(src_table) = timed.iter().find(|t| t.name == src) else {
                continue;
            };
            let bounds = time_bounds(memo, times, &[*src_table]).await?;
            let (Some(facts), Some(entities)) =
                (rows_of(&memo.memo, src), rows_of(&memo.memo, &dst.name))
            else {
                continue;
            };
            if entities == 0 || bounds.min > bounds.max {
                continue;
            }
            let micros = (i128::from(bounds.max) - i128::from(bounds.min)).div_euclid(1_000);
            #[allow(
                clippy::cast_precision_loss,
                reason = "integers to the nearest double, as the reference computes them"
            )]
            let (span, rate) = {
                let span = micros as f64 / 1e6;
                let rate = facts as f64 / entities as f64;
                (span, rate)
            };
            let span = if span < 1.0 { 1.0 } else { span };
            if out
                .iter()
                .any(|l: &crate::subgraph::Lookback| l.src == src && l.dst == dst.name)
            {
                continue;
            }
            out.push(crate::subgraph::Lookback {
                src: src.to_owned(),
                dst: dst.name.clone(),
                rate: rate / span,
            });
        }
    }
    Ok(out)
}

async fn missing_indexes(
    memo: &mut Memo<'_>,
    graph: &Graph,
    advice: Option<IndexAdvice>,
) -> Result<(), CoreError> {
    if memo.memo.indexes_checked {
        return Ok(());
    }
    check_indexes(memo.asker, graph, advice).await?;
    memo.memo.indexes_checked = true;
    memo.keep();
    Ok(())
}

async fn check_indexes(
    asker: &Asker,
    graph: &Graph,
    advice: Option<IndexAdvice>,
) -> Result<(), CoreError> {
    match advice {
        None => Ok(()),
        Some(IndexAdvice::SingleColumn) => check_single_columns(asker, graph).await,
        Some(IndexAdvice::LeadingColumn) => check_leading_columns(asker, graph).await,
    }
}

/// The reference's warning of the indexes missing, by its two counts: the
/// indices and the tables they are missing from; the host words it.
fn warn_missing(asker: &Asker, indexes: usize, tables: usize) {
    // Which of the reference's sentences, by its two counts' number
    // (decision E59(d)).
    let (variant, sentence) = match (indexes == 1, tables == 1) {
        (true, true) => ("index_table", Sentence::MissingIndexTable),
        (true, false) => ("index_tables", Sentence::MissingIndexTables),
        (false, true) => ("indices_table", Sentence::MissingIndicesTable),
        (false, false) => ("indices_tables", Sentence::MissingIndicesTables),
    };
    asker.warn(CoreWarning {
        fields: [
            ("indices".to_owned(), count(indexes)),
            ("tables".to_owned(), count(tables)),
            ("variant".to_owned(), Detail::text(variant)),
        ]
        .into(),
        ..CoreWarning::new(
            sentence,
            "indices the sampler's reads would use are missing",
        )
    });
}

/// [`IndexAdvice::LeadingColumn`], the reference's SQLite sampler's check
/// (`SQLiteSampler.__init__`): each table's key, and each foreign key with
/// its table's time column where it has one, that the source does not
/// enforce as unique, in the reference's order (the tables with keys, then
/// the links); then each such table's indexes, read one table at a time,
/// each index's columns by its name where the list does not give them; an
/// index whose first column is a read's first serves it. What is left is
/// warned of, by tuple and table.
async fn check_leading_columns(asker: &Asker, graph: &Graph) -> Result<(), CoreError> {
    let mut wanted: Vec<(&Table, Vec<Vec<String>>)> = Vec::new();
    for table in &graph.tables {
        if let Some(key) = &table.primary_key
            && let Some(column) = table.column(key)
            && !column.unique
        {
            push_read(&mut wanted, table, vec![key.clone()]);
        }
    }
    for edge in &graph.edges {
        let Some(table) = graph.table(&edge.src) else {
            continue;
        };
        if let Some(column) = table.column(&edge.fkey)
            && !column.unique
        {
            let read = match &table.time_column {
                Some(time) if table.column(time).is_some() => {
                    vec![edge.fkey.clone(), time.clone()]
                }
                _ => vec![edge.fkey.clone()],
            };
            push_read(&mut wanted, table, read);
        }
    }
    let mut missing: Vec<usize> = Vec::new();
    for (table, mut reads) in wanted {
        let answer = describe(
            asker,
            DataRequest::Describe {
                tables: vec![table.source.clone()],
                part: DescribePart::Indexes,
            },
        )
        .await?;
        let CatalogAnswer::Indexes(indexes) = answer else {
            return Err(internal(
                "the data layer read another part of the catalog than was asked for",
            ));
        };
        for index in indexes {
            let first = match index.columns {
                IndexCoverage::Resolved(columns) => columns.into_iter().next(),
                IndexCoverage::Unresolved => None,
                IndexCoverage::Named(name) => {
                    // The index is its table's schema's object.
                    let mut relation = table.source.clone();
                    if let Some(last) = relation.source.last_mut() {
                        *last = name;
                    }
                    let answer = describe(
                        asker,
                        DataRequest::Describe {
                            tables: vec![relation],
                            part: DescribePart::IndexColumns,
                        },
                    )
                    .await?;
                    let CatalogAnswer::IndexColumns(columns) = answer else {
                        return Err(internal(
                            "the data layer read another part of the catalog than was asked for",
                        ));
                    };
                    columns.into_iter().next().flatten()
                }
            };
            reads.retain(|read| Some(&read[0]) != first.as_ref());
        }
        if !reads.is_empty() {
            missing.push(reads.len());
        }
    }
    if !missing.is_empty() {
        warn_missing(asker, missing.iter().sum(), missing.len());
    }
    Ok(())
}

/// `read`, a read of `table` wanting an index, added once, its table's
/// reads kept in the order first wanted.
fn push_read<'a>(
    wanted: &mut Vec<(&'a Table, Vec<Vec<String>>)>,
    table: &'a Table,
    read: Vec<String>,
) {
    match wanted.iter_mut().find(|(t, _)| t.name == table.name) {
        Some((_, reads)) => {
            if !reads.contains(&read) {
                reads.push(read);
            }
        }
        None => wanted.push((table, vec![read])),
    }
}

/// [`IndexAdvice::SingleColumn`], DuckDB's: every key and foreign key the
/// source does not enforce as unique, less those an index over exactly
/// that one column covers, read from the source's whole index list.
async fn check_single_columns(asker: &Asker, graph: &Graph) -> Result<(), CoreError> {
    let mut missing: Vec<(String, Vec<String>)> = Vec::new();
    let mut add = |table: &str, column: &str| match missing.iter_mut().find(|(t, _)| t == table) {
        Some((_, columns)) => {
            if !columns.iter().any(|c| c == column) {
                columns.push(column.to_owned());
            }
        }
        None => missing.push((table.to_owned(), vec![column.to_owned()])),
    };
    for table in &graph.tables {
        if let Some(key) = &table.primary_key
            && let Some(column) = table.column(key)
            && !column.unique
        {
            add(&table.name, key);
        }
    }
    for edge in &graph.edges {
        if let Some(column) = graph.table(&edge.src).and_then(|t| t.column(&edge.fkey))
            && !column.unique
        {
            add(&edge.src, &edge.fkey);
        }
    }
    if missing.is_empty() {
        return Ok(());
    }
    let answer = describe(
        asker,
        DataRequest::Describe {
            tables: vec![],
            part: DescribePart::Indexes,
        },
    )
    .await?;
    let CatalogAnswer::Indexes(indexes) = answer else {
        return Err(internal(
            "the data layer read another part of the catalog than was asked for",
        ));
    };
    // An index is the source's, named by the source's name of its table:
    // it covers each table of the graph read from that one.
    for index in indexes {
        // Only an index over exactly one column, resolved, serves a read;
        // one the reader could not resolve serves none, as the reference
        // reads no column from it.
        let IndexCoverage::Resolved(columns) = &index.columns else {
            continue;
        };
        let [column] = columns.as_slice() else {
            continue;
        };
        // A relation of several parts names its table by them, which an
        // index's one name does not say: it is covered by none (design
        // choice 12; Phase 2's hosts give one part).
        for table in graph
            .tables
            .iter()
            .filter(|t| t.source.single() == Some(index.table.as_str()))
        {
            if let Some(at) = missing.iter().position(|(t, _)| *t == table.name) {
                missing[at].1.retain(|c| c != column);
                if missing[at].1.is_empty() {
                    missing.remove(at);
                }
            }
        }
    }
    if !missing.is_empty() {
        let indexes: usize = missing.iter().map(|(_, c)| c.len()).sum();
        warn_missing(asker, indexes, missing.len());
    }
    Ok(())
}

fn warn(asker: &Asker, sentence: Sentence, message: &str) {
    asker.warn(CoreWarning::new(sentence, message));
}

/// Holds the schema `materialize` is handed as the reference holds a
/// schema it is asked to make a request of (`Schema.check_consistent`,
/// decision E55): the columns it joins on marked again, from what the
/// host holds, so a host's edit is followed as the reference's schema
/// follows it; each column's meaning one its values can carry, as the
/// language answers it (`Stype.supports_dtype`); each time column of the
/// timestamp meaning, as the engine's table takes one (from T1.27's fix
/// round 1, found by the differential: a time column set on a column the
/// schema joins on keeps its identifier's meaning); and each link able to
/// join. Before anything else of the call is read, as the reference asks
/// it before the gathering starts.
/// Refuses a schema whose column means what its data type cannot carry,
/// before anything else of the call is read, as the reference refuses it
/// (T3.16): with `types`, as the caller holds it, as its `check_consistent`
/// reads the caller's schema, a column named by no data type the client
/// knows let through; without, as the source gave it, as its engine refuses
/// a column of an incompatible semantic type when it reads the table in
/// place.
pub(crate) fn meanings_carried(
    description: &SourceDescription,
    types: Option<&sdm_schema::assemble::HostTypes>,
) -> Result<(), CoreError> {
    let mut marked = description.clone();
    sdm_schema::assemble::mark_keys(&mut marked);
    for table in &marked.tables {
        for column in &table.columns {
            let unknown = types.is_some_and(|t| t.is_unknown(&table.name, &column.name));
            if unknown || crate::edit::supports(column.stype, column.dtype) {
                continue;
            }
            let stype = name_of(serde_json::to_value(column.stype));
            return Err(match types {
                // As `check_consistent` says it, quoting the type the
                // caller's schema holds.
                Some(types) => not_carried(Sentence::MeaningNotCarried, &table.name, &column.name)
                    .with_field("reason", Detail::text("meaning_not_carried"))
                    .with_field(
                        "dtype",
                        Detail::text(types.name(&table.name, &column.name, column.dtype)),
                    )
                    .with_field("stype", Detail::text(stype)),
                // As the reference's engine says it when it reads the table
                // in place, naming the source's type in the language's
                // name.
                None => not_carried(
                    Sentence::IncompatibleSemanticType,
                    &table.name,
                    &column.name,
                )
                .with_field("reason", Detail::text("incompatible_semantic_type"))
                .with_field(
                    "dtype",
                    Detail::text(name_of(serde_json::to_value(column.dtype))),
                )
                .with_field("stype", Detail::text(stype)),
            });
        }
    }
    Ok(())
}

/// A contract enum's name, as its JSON spells it.
fn name_of(value: Result<serde_json::Value, serde_json::Error>) -> String {
    value
        .ok()
        .and_then(|v| v.as_str().map(str::to_owned))
        .unwrap_or_default()
}

fn not_carried(sentence: Sentence, table: &str, column: &str) -> CoreError {
    invalid(
        sentence,
        "a column's meaning is not one its values can carry",
        "schema",
    )
    .with_field("table", Detail::text(table))
    .with_field("column", Detail::text(column))
}

/// The reference's `Schema.check_consistent` over the caller's schema,
/// before its engine reads anything (`relational.py:514`): the key columns
/// marked as the schema joins on them (`_mark_key_columns`), then each link
/// able to join, each column's type read by the name `types` gives it (the
/// host's). Each column's meaning against the type the caller gave it is
/// that check's other half, [`meanings_carried`] with the host's types,
/// made where the run is asked (dispatch).
fn consistent(
    description: &mut SourceDescription,
    types: &sdm_schema::assemble::HostTypes,
) -> Result<(), CoreError> {
    sdm_schema::assemble::mark_keys(description);
    sdm_schema::assemble::check_links(description, types)
}

/// The reference's engine's refusals of the schema over the tables it read
/// and kept, each over the types the source holds, in its order: each
/// column's meaning set on the engine's table (`relational.py:551`,
/// `Column.__setattr__`), one its values can carry, and a time column's
/// the timestamp meaning (the engine's table takes a time column only of
/// it: one the schema joins on keeps its identifier's); then the edges, as
/// `Graph.validate` refuses them (`relational.py:418`). `description`'s
/// key columns are marked ([`consistent`]).
fn kept_checks(description: &SourceDescription) -> Result<(), CoreError> {
    meanings_carried(description, None)?;
    for table in &description.tables {
        if let Some(time) = &table.time_column
            && table
                .columns
                .iter()
                .any(|c| c.name == *time && c.stype != Stype::Timestamp)
        {
            return Err(invalid(
                Sentence::TimeColumnNotATime,
                "a table's time column does not mean a time",
                "time_column",
            )
            .with_field("table", Detail::text(&table.name))
            .with_field("column", Detail::text(time)));
        }
    }
    sdm_schema::assemble::check_edges(description)
}

/// The names of options the host passes without their values
/// (`Materialize::options`): from T2.11 every sampling option the
/// reference takes travels in the call's values, so any name here is the
/// host's fault, `INVALID_REQUEST`, refused before anything of the call is
/// read (decision E61).
fn options_taken(inputs: &Inputs) -> Result<(), CoreError> {
    if inputs.options.is_empty() {
        return Ok(());
    }
    Err(invalid(
        Sentence::OptionsWithoutValues,
        "an option named is not one of the sampling options",
        "options",
    ))
}

/// A seed for an unseeded sample (`random_seed=None`), a new one each
/// call, as the reference draws a new sample each call: the host's clock
/// (`HostFacts.now`, the core reading none of its own), mixed by
/// SplitMix64's finalizer, at most the provider's largest seed. Every
/// provider then draws as it draws under a seed, and the sample is judged
/// by its properties (KO2-76, register P2I-2).
fn fresh_seed(now: &Scalar, max_seed: u64) -> u64 {
    let clock = match now {
        Scalar::Timestamp { value, .. } => *value as u64,
        _ => 0,
    };
    let mut z = clock.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    let drawn = z ^ (z >> 31);
    match max_seed.checked_add(1) {
        Some(bound) => drawn % bound,
        None => drawn,
    }
}

/// The rows of `array` holding a NaN: none but in a float column.
fn nan_rows(array: &ArrayRef) -> BTreeSet<usize> {
    use arrow_array::types::{Float32Type, Float64Type};
    match array.data_type() {
        DataType::Float32 => array
            .as_primitive::<Float32Type>()
            .iter()
            .enumerate()
            .filter(|(_, v)| v.is_some_and(f32::is_nan))
            .map(|(row, _)| row)
            .collect(),
        DataType::Float64 => array
            .as_primitive::<Float64Type>()
            .iter()
            .enumerate()
            .filter(|(_, v)| v.is_some_and(f64::is_nan))
            .map(|(row, _)| row)
            .collect(),
        _ => BTreeSet::new(),
    }
}

/// The prefix of the column the reference's engine folds a key of several
/// columns into (`composite_key.DERIVED_PREFIX`), and the longest name it
/// gives one (`MAX_DERIVED_NAME`).
const FOLDED_NAME_MOST: usize = 200;

/// A schema keyed by several columns somewhere, refused as the reference
/// refuses it at materialize, after its query resolved: its engine folds
/// each such key into one identity as the materializer keys its tables, in
/// table order (`Table._set_composite_primary_key`), refusing a float among
/// the key's columns (`refuse_unfoldable_dtypes`), a derived name past 200
/// characters (`encode_identity`) or one a column holds already; then its
/// SQL samplers refuse every such table, whose folded column is named with
/// the prefix they reserve (`_reject_reserved_columns`). Each refusal is
/// `INVALID_REQUEST` naming the table and the key's columns. Between the
/// two the reference reads the source's fold of the first rows
/// (`_assert_warehouse_folds_like_pandas`), which chooses only between two
/// refusals of that code; the core makes no such read, one statement fewer
/// (KO2-68). `Ok` for a schema with no such key.
fn refuse_composite_keys(description: &SourceDescription) -> Result<(), CoreError> {
    let refused = |sentence, table: &str, key: &[String], message: &str| {
        invalid(sentence, message, "schema")
            .with_field("table", Detail::text(table))
            .with_field(
                "columns",
                Detail::List(key.iter().map(|c| Scalar::Utf8(c.clone())).collect()),
            )
    };
    let mut first = None;
    for table in &description.tables {
        let key = table.primary_key.as_slice();
        if key.len() < 2 {
            continue;
        }
        let float = key.iter().any(|name| {
            table.columns.iter().any(|c| {
                &c.name == name && matches!(c.dtype, Dtype::Float | Dtype::Float32 | Dtype::Float64)
            })
        });
        if float {
            return Err(refused(
                Sentence::CompositeKeyFloat,
                &table.name,
                key,
                "a float cannot be part of a key of several columns: its text is not reproducible",
            ));
        }
        let name = sdm_engine::graph::folded_name(key);
        if name.len() > FOLDED_NAME_MOST {
            return Err(refused(
                Sentence::CompositeKeyNameLong,
                &table.name,
                key,
                "a key of several columns needs a folded column's name longer than a source takes",
            ));
        }
        if table.columns.iter().any(|c| c.name == name) {
            return Err(refused(
                Sentence::CompositeKeyNameTaken,
                &table.name,
                key,
                "a column holds the name a key of several columns is folded into",
            ));
        }
        first.get_or_insert((table.name.clone(), key.to_vec()));
    }
    match first {
        None => Ok(()),
        Some((table, key)) => Err(sdm_engine::graph::with_reserved(
            refused(
                Sentence::ReservedName,
                &table,
                &key,
                "a table keyed by several columns cannot be sampled: the SQL samplers fold its key into a column of the name they reserve",
            ),
            description,
        )),
    }
}

/// Refuses a column of the schema named as the wire names a column of its
/// own (`instance_id`, `__kumo_anchor_time`), as the client refuses it
/// (`_refuse_a_name_the_wire_reserves`): `INVALID_SCHEMA`, naming each
/// table and column, in order.
fn reserved(description: &SourceDescription) -> Result<(), CoreError> {
    let mut taken: Vec<(&str, &str)> = description
        .tables
        .iter()
        .flat_map(|t| {
            t.columns
                .iter()
                .filter(|c| RESERVED.contains(&c.name.as_str()))
                .map(move |c| (t.name.as_str(), c.name.as_str()))
        })
        .collect();
    if taken.is_empty() {
        return Ok(());
    }
    taken.sort_unstable();
    let list = |names: Vec<&str>| {
        Detail::List(
            names
                .into_iter()
                .map(|n| Scalar::Utf8(n.to_owned()))
                .collect(),
        )
    };
    // Each named as `table.column`, and a name for it that the request does
    // not need (`_unreserved`), for the host's words.
    let unreserved = |column: &str| {
        let suggested = format!("{}_from_the_data", column.trim_start_matches('_'));
        if RESERVED.contains(&suggested.as_str()) {
            format!("{column}_")
        } else {
            suggested
        }
    };
    Err(CoreError::new(
        if taken.len() > 1 {
            Sentence::WireReservedSeveral
        } else {
            Sentence::WireReservedOne
        },
        "a column of the schema has a name the request needs for its own",
    )
    .with_field("tables", list(taken.iter().map(|(t, _)| *t).collect()))
    .with_field("columns", list(taken.iter().map(|(_, c)| *c).collect()))
    .with_field(
        "reserved",
        Detail::List(
            taken
                .iter()
                .map(|(t, c)| Scalar::Utf8(format!("{t}.{c}")))
                .collect(),
        ),
    )
    .with_field(
        "renamed",
        Detail::List(
            taken
                .iter()
                .map(|(_, c)| Scalar::Utf8(unreserved(c)))
                .collect(),
        ),
    ))
}

/// A list of names as a warning's field.
fn names<'a>(names: impl IntoIterator<Item = &'a str>) -> Detail {
    Detail::List(
        names
            .into_iter()
            .map(|n| Scalar::Utf8(n.to_owned()))
            .collect(),
    )
}

/// The notices of what the request leaves out, given as the client gives
/// them after it gathers, each a code and fields the host words: the
/// schema's tables the request does not carry, those no relationship
/// connects to the entity table apart from those linked and reached by no
/// row (`_warn_of_tables_left_out`); the columns it sends that the model
/// does not read, text (`_warn_of_columns_the_model_does_not_read`, a
/// static target's columns left out); and the context examples that
/// answered the question, taken out (`_drop_context_that_answers_the_question`).
fn notices(
    asker: &Asker,
    description: &SourceDescription,
    asked: &Asked,
    sent: &BTreeSet<String>,
    excluded: &[(String, String)],
    answering: usize,
    windowed: bool,
) {
    let mut missing: Vec<&str> = description
        .tables
        .iter()
        .map(|t| t.name.as_str())
        .filter(|n| !sent.contains(*n))
        .collect();
    missing.sort_unstable();
    // The tables a relationship path connects to the entity table, read in
    // both directions (`_tables_linked_to`).
    let mut linked = BTreeSet::from([asked.entity_table.as_str()]);
    let mut queue = vec![asked.entity_table.as_str()];
    while let Some(at) = queue.pop() {
        for link in &description.links {
            if let [a, b] = link.as_slice() {
                for (from, to) in [(a, b), (b, a)] {
                    if from.table == at && linked.insert(to.table.as_str()) {
                        queue.push(to.table.as_str());
                    }
                }
            }
        }
    }
    let (unreached, unlinked): (Vec<&str>, Vec<&str>) =
        missing.into_iter().partition(|n| linked.contains(n));
    let several = |n: usize| if n > 1 { "several" } else { "one" };
    if !unlinked.is_empty() {
        asker.warn(CoreWarning {
            fields: [
                ("variant".to_owned(), Detail::text(several(unlinked.len()))),
                ("tables".to_owned(), names(unlinked.iter().copied())),
                (
                    "entity_table".to_owned(),
                    Detail::text(asked.entity_table.as_str()),
                ),
            ]
            .into(),
            ..CoreWarning::new(
                if unlinked.len() > 1 {
                    Sentence::TablesUnlinkedSeveral
                } else {
                    Sentence::TablesUnlinkedOne
                },
                "tables of the schema no relationship connects are not in the request",
            )
        });
    }
    if !unreached.is_empty() {
        asker.warn(CoreWarning {
            fields: [
                ("variant".to_owned(), Detail::text(several(unreached.len()))),
                ("tables".to_owned(), names(unreached.iter().copied())),
            ]
            .into(),
            ..CoreWarning::new(
                if unreached.len() > 1 {
                    Sentence::TablesUnreachedSeveral
                } else {
                    Sentence::TablesUnreachedOne
                },
                "tables linked into the schema were reached by no row sampled",
            )
        });
    }
    let mut unread: Vec<String> = description
        .tables
        .iter()
        .filter(|t| sent.contains(&t.name))
        .flat_map(|t| {
            t.columns
                .iter()
                .filter(|c| c.stype == Stype::Text)
                .filter(|c| {
                    !excluded
                        .iter()
                        .any(|(et, ec)| *et == t.name && *ec == c.name)
                })
                .map(move |c| format!("{}.{}", t.name, c.name))
        })
        .collect();
    unread.sort_unstable();
    if !unread.is_empty() {
        asker.warn(CoreWarning {
            fields: [
                ("variant".to_owned(), Detail::text(several(unread.len()))),
                (
                    "columns".to_owned(),
                    names(unread.iter().map(String::as_str)),
                ),
                ("stypes".to_owned(), names(["text"])),
            ]
            .into(),
            ..CoreWarning::new(
                if unread.len() > 1 {
                    Sentence::ColumnsUnreadSeveral
                } else {
                    Sentence::ColumnsUnreadOne
                },
                "columns the request sends are of a meaning the model does not read",
            )
        });
    }
    if answering > 0 {
        asker.warn(CoreWarning {
            fields: [
                (
                    "variant".to_owned(),
                    Detail::text(if windowed { "windowed" } else { "static" }),
                ),
                ("examples".to_owned(), count(answering)),
            ]
            .into(),
            ..CoreWarning::new(
                if windowed {
                    Sentence::ContextAnswersWindowed
                } else {
                    Sentence::ContextAnswersStatic
                },
                "context examples answering the question were taken out",
            )
        });
    }
}

/// The body of a `materialize` run.
/// The most steps a forecast is asked as where the host names no limit:
/// the client's `decompose.MOST_PARTS` as it ships (T1.36 fix 21).
pub const MOST_PARTS: u32 = 200;

/// One call's payload as gathered: its bytes, and the rows it scores.
pub(crate) type Call = (Vec<u8>, u64);

/// What one question's run gathered: each call's payload, in order (one
/// for a request of no more rows than one call scores), and its query's
/// text; or, for the test-only `labels` operation, its result.
pub(crate) enum Gathered {
    Request(Vec<Call>, String),
    #[cfg_attr(not(feature = "test-ops"), allow(dead_code))]
    Done(OperationResult),
}

pub(crate) async fn run(asker: Asker, inputs: Inputs) -> Result<OperationResult, CoreError> {
    let (steps, ranked, facts) = resolved(&inputs);
    // A ranking, asked as one binary question per candidate (T2.8).
    if let Some(plan) = ranked {
        return crate::ranking::materialize(asker, inputs, plan, facts).await;
    }
    // Candidates on a question that ranks nothing, refused before anything
    // is gathered, as the client's `parts_of` refuses them; a query that
    // does not resolve is refused as itself, below.
    if inputs.candidates.is_some() && facts.is_some() {
        return Err(crate::ranking::candidates_on_no_ranking());
    }
    let Some(steps) = steps else {
        return match one(&asker, inputs).await? {
            Gathered::Done(result) => Ok(result),
            Gathered::Request(calls, query) => {
                // The first call's payload is the state's, and each later
                // call's a batch of it, in order (`main`'s multi-call
                // `predict`): `predict` sends them one after another.
                let mut calls = calls.into_iter();
                let (first, _) = calls
                    .next()
                    .ok_or_else(|| internal("a request gathered as no call"))?;
                let (payload, query) = attach_request(&asker, first, &query)?;
                let mut batches = Vec::new();
                for (later, rows) in calls {
                    batches.push(RequestBatch {
                        payload: asker.attach(Attachment::Bytes(later))?,
                        rows,
                        request_id: None,
                    });
                }
                Ok(OperationResult::Materialize(Materialized {
                    state: RequestState {
                        version: REQUEST_STATE_VERSION,
                        payload,
                        query,
                        forecast: None,
                        batches: (!batches.is_empty()).then_some(batches),
                        ranking: None,
                    },
                    query: facts,
                }))
            }
        };
    };
    // A forecast, asked as one regression per step, each gathered as the
    // client's `_in_parts` gathers it (T1.36 fix 21): in order, each by a
    // materializer of its own, so with no memo of an earlier one's reads,
    // each refusal naming the step it happened in (`forecast_step`).
    // The materializer's own memo is left as it was handed, done or failed,
    // as the client's per-step materializers leave the caller's untouched.
    // The client's refusals of a forecast before it gathers anything
    // (`parts_of`): `top_k`, which ranks, then more steps than the host's
    // limit, 200 by default (`_refuse_too_many`).
    if inputs
        .values
        .as_ref()
        .is_some_and(|values| values.get("top_k").is_some())
    {
        return Err(CoreError::new(
            Sentence::ForecastTopK,
            "top_k is a ranking's, and a forecast ranks nothing",
        )
        .with_field("field", Detail::text("top_k")));
    }
    // The host's limit is checked against the plan's count before any step
    // is written, so nothing grows with a count past it (T1.36 fix 23).
    let most = inputs.most_parts.unwrap_or(MOST_PARTS);
    if steps.count > u64::from(most) {
        return Err(CoreError::new(
            Sentence::TooManySteps,
            "a forecast asks for more steps than the host's limit, one call each",
        )
        .with_field("field", Detail::text("query"))
        .with_field("steps", Detail::Scalar(Scalar::UInt64(steps.count)))
        .with_field("limit", Detail::Scalar(Scalar::UInt32(most))));
    }
    // Then, before anything is gathered, as the client's `_in_parts`
    // refuses them (`main`'s 1fc5a686): the batch size asked for, a step
    // being one call (a forecast over more entities than one call scores
    // would make the steps times the batches), and the population each step
    // gathers, counted across the steps.
    let per_call = options::rows_one_call_scores(asked_rows_per_call(inputs.values.as_ref())?);
    let entities = inputs.indices.as_ref().map_or(0, |data| {
        data.batches
            .iter()
            .map(RecordBatch::num_rows)
            .sum::<usize>()
    });
    let entities = u64::try_from(entities).unwrap_or(u64::MAX);
    if entities > per_call {
        return Err(CoreError::new(
            Sentence::BatchedPartForecast,
            "a forecast over more entities than one call scores would batch each step",
        )
        .with_field("field", Detail::text("indices"))
        .with_field("part_kind", Detail::text("forecast"))
        .with_field("entities", Detail::Scalar(Scalar::UInt64(entities)))
        .with_field("parts", Detail::Scalar(Scalar::UInt64(steps.count)))
        .with_field("rows_per_call", Detail::Scalar(Scalar::UInt64(per_call))));
    }
    refuse_entities(entities, steps.count, inputs.most_entities)?;
    let handed = inputs.memo.clone().unwrap_or_else(|| SamplerMemo {
        source: inputs.source.clone(),
        ..SamplerMemo::default()
    });
    // Within the checked bound, a `u32` (the limit's type).
    let bound = u32::try_from(steps.count).map_err(|_| internal("a forecast's steps past u32"))?;
    let mut gathered = Vec::with_capacity(bound as usize);
    for step in 1..=bound {
        let mut asked = inputs.clone();
        asked.query = steps.step(u64::from(step));
        asked.memo = None;
        let tagged = |e: CoreError| in_step(e, step, bound);
        let done = one(&asker, asked).await;
        asker.keep(handed.clone());
        match done.map_err(tagged)? {
            Gathered::Done(result) => return Ok(result),
            // A step scores no more rows than one call (refused above), so
            // it is gathered as one call.
            Gathered::Request(calls, query) => match <[Call; 1]>::try_from(calls) {
                Ok([(payload, _)]) => gathered.push((step, payload, query)),
                Err(_) => return Err(internal("a forecast's step gathered as several calls")),
            },
        }
    }
    let mut attached = Vec::with_capacity(gathered.len());
    for (step, payload, query) in gathered {
        let (payload, query) = attach_request(&asker, payload, &query)?;
        attached.push(ForecastStep {
            step,
            payload,
            query,
            request_id: None,
        });
    }
    let first = attached.remove(0);
    Ok(OperationResult::Materialize(Materialized {
        state: RequestState {
            version: REQUEST_STATE_VERSION,
            payload: first.payload,
            query: first.query,
            forecast: Some(attached),
            batches: None,
            ranking: None,
        },
        query: facts,
    }))
}

/// The batch size the call's values ask for (`rows_per_call`), read alone,
/// as the client's `_in_parts` reads it before a part is gathered: none
/// where the values name none.
pub(crate) fn asked_rows_per_call(
    values: Option<&sdm_wire::json::Value>,
) -> Result<Option<u64>, CoreError> {
    match values.and_then(|v| v.get("rows_per_call")) {
        None => Ok(None),
        Some(given) => options::rows_per_call(given),
    }
}

/// Refuses a population too large to hold (`main`'s
/// `refuse_more_entities_than_fit_in_memory`), from the count alone, before
/// anything is read: `entities` gathered `times` over (a forecast's steps
/// each gather it whole), past the host's limit, 100,000 where it names
/// none, `UNSUPPORTED_REQUEST`, rule `most_entities`.
pub(crate) fn refuse_entities(
    entities: u64,
    times: u64,
    most: Option<u64>,
) -> Result<(), CoreError> {
    let most = most.unwrap_or(MOST_ENTITIES);
    let total = entities.saturating_mul(times);
    if total <= most {
        return Ok(());
    }
    Err(CoreError::new(
        if times == 1 {
            Sentence::MostEntities
        } else {
            Sentence::MostEntitiesParts
        },
        "more entities were asked for than one materialize holds at once",
    )
    .with_field("field", Detail::text("indices"))
    .with_field("entities", Detail::Scalar(Scalar::UInt64(entities)))
    .with_field("gathered_times", Detail::Scalar(Scalar::UInt64(times)))
    .with_field("total", Detail::Scalar(Scalar::UInt64(total)))
    .with_field("limit", Detail::Scalar(Scalar::UInt64(most))))
}

/// A request's payload and query, attached.
pub(crate) fn attach_request(
    asker: &Asker,
    payload: Vec<u8>,
    query: &str,
) -> Result<(AttachmentId, AttachmentId), CoreError> {
    let payload = asker.attach(Attachment::Bytes(payload))?;
    let query = asker.attach(Attachment::Bytes(
        serde_json::to_vec(query).map_err(|_| internal("a query does not serialize"))?,
    ))?;
    Ok((payload, query))
}

/// A forecast's plan ([`crate::pql::resolve`]'s `steps`), when the query
/// resolves to one, none otherwise, the query then gathered as itself,
/// which refuses it where it does not resolve as before; and the facts of
/// the query a host reads ([`QueryFacts`], T2.18), when it resolves.
#[allow(
    clippy::type_complexity,
    reason = "three plain parts, named where they are taken"
)]
fn resolved(
    inputs: &Inputs,
) -> (
    Option<crate::pql::ForecastPlan>,
    Option<crate::pql::RankingPlan>,
    Option<QueryFacts>,
) {
    let mut description = inputs.description.clone();
    if consistent(&mut description, &inputs.types).is_err() {
        return (None, None, None);
    }
    let profile = Profile::resolve(&ProfileSpec {
        name: ProfileName::Relational,
        model: None,
        capabilities: Default::default(),
    });
    let schema = query_schema(&description);
    match crate::pql::resolve(
        &inputs.query,
        &schema,
        &profile,
        &inputs.facts,
        &*inputs.zones,
    ) {
        Ok(validation) => {
            let facts = validation.facts(&schema);
            (validation.steps, validation.ranking, facts)
        }
        Err(_) => (None, None, None),
    }
}

/// One question gathered: the body of `materialize` before T1.36 fix 21.
pub(crate) async fn one(asker: &Asker, inputs: Inputs) -> Result<Gathered, CoreError> {
    let frames = inputs.frames.clone();
    // The memo the run leaves the host: the one handed back, or a new one
    // (decision E52), left as it is by a run refused before it reads.
    let mut memo = Memo {
        asker,
        memo: inputs.memo.clone().unwrap_or_else(|| SamplerMemo {
            source: inputs.source.clone(),
            ..SamplerMemo::default()
        }),
    };
    memo.keep();
    // The call's option values, each of the type the client takes; then
    // the context size's bounds and the run mode it selects (T1.28a). The
    // client checks these in another order (the context size's lower
    // bound, the indices, the statistic, the reserved names, then its
    // upper bound, the entity count and the shapes); a call with several
    // faults may be refused here for another of them than the client
    // names, which the core does not claim to match (decision E41(k)(iii)).
    let asked_values = options::values(inputs.values.as_ref())?;
    let mode = options::run_mode(
        asked_values.num_context_samples,
        asked_values.num_context_samples_given.as_deref(),
    )?;
    // The hops the sampling options ask, bounded as the client bounds them,
    // before the options this core does not take are refused by name.
    options::hops(asked_values.num_hops, asked_values.num_neighbors)?;
    // The seed the call names, in place of the host's `seed` and
    // `unseeded` where it names one.
    let (seed, unseeded) = match &asked_values.seed {
        AskedSeed::Absent => (inputs.seed.clone(), inputs.unseeded),
        AskedSeed::Unseeded => (None, true),
        AskedSeed::Given(text) => (
            Some(
                Seed::try_from(text.clone())
                    .map_err(|_| internal("a seed the call named does not read"))?,
            ),
            false,
        ),
    };
    options_taken(&inputs)?;
    if seed.as_ref().is_some_and(Seed::is_negative) {
        return Err(
            CoreError::new(Sentence::SeedNegative, "a sampling seed cannot be negative")
                .with_field("field", Detail::text("seed"))
                .with_field("reason", Detail::text("negative"))
                .with_field("given", Detail::text(seed_text(seed.as_ref()))),
        );
    }
    #[cfg(feature = "test-ops")]
    let labels_only = inputs.labels_only;
    let inputs_anchor_entity = inputs.anchor_entity;
    let Inputs {
        mut description,
        pending,
        query,
        indices,
        anchor_time,
        facts,
        zones,
        most_entities,
        most_calls,
        types,
        ..
    } = inputs;
    // The sampling options the engine reads later than the client's
    // checks, each read where the reference reads it (T2.11).
    let deferred = asked_values.clone();
    let options::Asked {
        num_context_samples,
        prediction_statistic,
        inference_config,
        top_k,
        rows_per_call,
        lag_timesteps,
        ..
    } = asked_values;
    let anchor_entity = inputs_anchor_entity;
    // The most context examples asked for, which narrows the context the
    // mode gathers (`_narrow_context`); the bound above holds it to the
    // best mode's.
    let asked_examples = num_context_samples
        .and_then(|n| n.magnitude)
        .and_then(|n| usize::try_from(n).ok());
    // A table the schema names otherwise than the source does is read by
    // its source name (`Table::source`), as the reference's engine reads
    // it; the query and the payload name it as the schema does.
    // The schema as the caller holds it, its types by the host's names, as
    // the reference's `check_consistent` reads it before anything is read
    // (`relational.py:514`); where a column holds no facts of the source,
    // the schema is read by what the source says of it once the inputs are
    // checked (below). The reference's engine's checks wait for its reads.
    consistent(&mut description, &types)?;
    // The seed reaches this entry as the integer the caller gave, of any
    // size (decision H7); numpy's generator takes it whole, and the sample
    // takes it as the narrowest exact scalar.
    // An unseeded sample draws under a seed of its own, new each call.
    let seed = match seed {
        Some(seed) => seed,
        None if unseeded => Seed::from(u128::from(match &frames {
            Some(frames) => crate::frame_materialize::lock(frames)
                .entropy
                .ok_or_else(|| internal("an unseeded frame call lacks its OS entropy fact"))?,
            None => fresh_seed(&facts.now, asker.capabilities()?.max_seed),
        })),
        None => Seed::from(DEFAULT_SEED),
    };
    let entropy = Entropy::from_hex(seed.magnitude_hex())
        .ok_or_else(|| internal("a seed's digits do not read"))?;

    let indices = indices.ok_or_else(|| {
        invalid(
            Sentence::IndicesAbsent,
            "materialize names the entities to predict for, and none were given",
            "indices",
        )
    })?;
    let indices = one_batch(&indices)?;
    if indices.num_columns() != 1 {
        return Err(invalid(
            Sentence::IndicesNotOneColumn,
            "the entities to predict for are one column",
            "indices",
        ));
    }
    let indices = Arc::clone(indices.column(0));
    if indices.is_empty() {
        return Err(invalid(
            Sentence::IndicesEmpty,
            "the entities to predict for are none",
            "indices",
        )
        .with_field("reason", Detail::text("empty")));
    }
    // More entities than one call scores are gathered as several calls
    // (`main`'s 1fc5a686), up to a population the host's limit holds,
    // refused from the count alone before anything is read.
    refuse_entities(
        u64::try_from(indices.len()).unwrap_or(u64::MAX),
        1,
        most_entities,
    )?;
    // The statistic the caller chose, one the endpoint reports, and no
    // column named as the wire names its own (the client's checks before
    // it gathers; their order among the checks above is not the client's,
    // decision E41(k)(iii)).
    let statistic = options::statistic(prediction_statistic.as_deref())?;
    reserved(&description)?;
    // Every input check of the request made, the tables are read, as the
    // reference's engine reads its tables (`relational.py:520-522`,
    // `materializers/sources.py:61-71`): a source's tables as they stand at
    // this run, a host provider's columns that held no facts. On every
    // path, every check over what the source holds comes after: the
    // engine's meanings, time columns and edges ([`kept_checks`]), the
    // graph and what the sampler needs.
    let refresh_source_facts = asker.located().is_none()
        && asker
            .data()
            .is_some_and(crate::compose::DataAccess::reads_a_source);
    if let Some(located) = crate::describe::initialize(asker, pending.as_ref(), &facts).await? {
        // Plan B reads current rows and view definitions. Neither the
        // held schema's fingerprint nor a matching fresh preview proves
        // that their bounds, counts or stored time spelling are unchanged.
        // Expire only data facts, once at the fresh source boundary: E52's
        // index/warning lifetime and reuse within this run remain intact.
        // Providers without source refresh retain E52's
        // cross-run data memo as well.
        if refresh_source_facts {
            memo.memo.bounds.clear();
            memo.memo.rows.clear();
            memo.memo.stored_times.clear();
            memo.keep();
        }
        description = located;
        sdm_schema::assemble::mark_keys(&mut description);
    }
    kept_checks(&description)?;
    if description.tables.iter().any(|t| t.primary_key.len() > 1) {
        // The reference resolves its query against a schema with no key of
        // several columns (`to_schema_definition`) before its engine folds
        // one (T2.12).
        let resolved = crate::pql::resolve(
            &query,
            &query_schema(&description),
            &Profile::resolve(&ProfileSpec {
                name: ProfileName::Relational,
                model: None,
                capabilities: Default::default(),
            }),
            &facts,
            &*zones,
        )?;
        if resolved.query.is_none() {
            return Err(unresolved(&query, &resolved.diagnostics));
        }
        refuse_composite_keys(&description)?;
    }

    let graph = Graph::new(&description)?;
    if let Some(frames) = &frames {
        let mut frames = crate::frame_materialize::lock(frames);
        frames.initialize(asker, &description, &graph)?;
        memo.memo.source = frames.source();
        memo.keep();
    }
    let profile = Profile::resolve(&ProfileSpec {
        name: ProfileName::Relational,
        model: None,
        capabilities: Default::default(),
    });
    let validation = crate::pql::resolve(
        &query,
        &query_schema(&description),
        &profile,
        &facts,
        &*zones,
    )?;
    // Literal warnings belong to validate_query: a public call validates its query
    // before it materializes it (`PredictiveQuery._resolve`), so this run,
    // which reads the same literals, does not give it again.
    let (Some(validated), Some(source_query)) = (validation.query, validation.text) else {
        return Err(unresolved(&query, &validation.diagnostics));
    };
    // An entity filter chooses the labelled examples, never the entities
    // scored, which the client says before it gathers
    // (`_warn_the_filter_chose_only_the_examples`), as a code and the
    // filter's condition and the entities' count the host words.
    if let Some(condition) = &validation.entity_filter {
        asker.warn(CoreWarning {
            fields: [
                ("condition".to_owned(), Detail::text(condition.as_str())),
                ("entities".to_owned(), count(indices.len())),
            ]
            .into(),
            ..CoreWarning::new(
                Sentence::FilterChoseExamples,
                "the entity filter chooses the labelled examples, not the entities scored",
            )
        });
    }
    // What the sampler needs of its provider, refused before anything is
    // fetched.
    let nulls = if frames.is_some() {
        NullOrder::First
    } else {
        sampling_capabilities(asker)?
    };
    // The index read, the sampler's check of the indexes it would use
    // (only under the provider's index advice, once for a materializer),
    // comes before anything of the task is read, as the reference builds
    // its sampler, and gives its warning, before its task table. The checks
    // after it (null entities, then the seed, then the task's shape) are in
    // this core's own order, not the reference's: an input with several
    // errors may be refused for another of them than the reference's first
    // (decision E41(k)(i)).
    if frames.is_none() {
        missing_indexes(&mut memo, &graph, asker.capabilities()?.index_advice).await?;
    }
    // Each table's rows, where the provider sizes its sampling by them,
    // as the reference's warehouse sampler counts them as it is built.
    if frames.is_none() {
        table_rows(&mut memo, &graph).await?;
    }
    // The context's anchor and the candidates per example, read with the
    // task (`_get_task_table`).
    let (context_anchor, max_pq_iterations) = options::task_table(&deferred)?;
    if keys::missing(&indices) {
        return Err(invalid(
            Sentence::IndexMissing,
            "an entity to predict for is missing",
            "indices",
        ));
    }
    // A seed past what its scalar carries (10**38 or more) is past every
    // largest seed a provider can declare (a `u64`), which every provider
    // declares from contract v26 (design choice 1): RS-7's refusal, as for
    // any seed past it (T1.36 fix 9).
    let seed_value = match seed_scalar(&seed) {
        Ok(value) => value,
        Err(_) if !seed.is_negative() => {
            return Err(crate::compose::past_seed_range(
                OptionalRequest::SampleEntities.name(),
                if frames.is_some() {
                    u32::MAX as u64
                } else {
                    asker.capabilities()?.max_seed
                },
            ));
        }
        Err(err) => return Err(err),
    };
    let asked = asked(&validated)?;
    let link_target = if let Shape::Aggregated(windowed) = &asked.target
        && windowed.aggregation == labels::Aggregation::ListDistinct
    {
        let column = windowed
            .column
            .as_deref()
            .ok_or_else(|| unvalidated("a list target with no column"))?;
        let edges: Vec<_> = graph
            .edges
            .iter()
            .filter(|e| e.src == windowed.table && e.fkey == column)
            .collect();
        match edges.as_slice() {
            [edge] => Some(edge.dst.clone()),
            _ => {
                return Err(CoreError::new(
                    Sentence::LinkTargetNotForeignKey,
                    "the list target is not one foreign key",
                )
                .with_field(
                    "column",
                    Detail::text(format!("{}.{}", windowed.table, column)),
                ));
            }
        }
    } else {
        None
    };
    options::statistic_for(asked.task, statistic.as_ref())?;
    let entity = graph
        .table(&asked.entity_table)
        .ok_or_else(|| internal("the query's entity table is not in the graph"))?;
    let key = entity
        .primary_key
        .clone()
        .filter(|k| *k == asked.entity_column)
        .ok_or_else(|| not_yet("entity_key"))?;
    // A temporal query's aggregated tables, each with the foreign key by
    // which it refers to the entity table and the window its rows are read
    // over (`_get_query_time_offset_dict`).
    // The labels read every table the target, the entity filter and the
    // assumption aggregate, each over the windows of them all pooled; the
    // lags and the anchor's bounds the target's alone
    // (`to_autoregressive_query`, `get_all_target_aggregations`).
    let labelled_fetches = fetches(&graph, &asked.entity_table, &asked.every_windowed())?;
    let mut windows = Vec::new();
    asked.target.windowed(&mut windows);
    let fetches = fetches(&graph, &asked.entity_table, &windows)?;

    // The times every fetch below returns, read as the reference's engine
    // reads them.
    let read_times = Times::new(
        crate::describe::text_reader(&facts)?,
        if frames.is_some() {
            None
        } else {
            asker.capabilities()?.text_times
        },
        &memo.memo,
    );

    // The tables whose times bound the anchor: a temporal query's
    // aggregated tables; for a static query, every table with a time
    // column (`get_min_time()`, `get_max_time()`), none of which leaves the
    // anchor unchecked (`_validate_time` returns) and, by default, no
    // anchor at all (`pd.Timestamp.min`).
    let bounded: Vec<&Table> = if fetches.is_empty() {
        graph
            .tables
            .iter()
            .filter(|t| t.time_column.is_some())
            .collect()
    } else {
        fetches.iter().map(|f| f.table).collect()
    };
    // The window the target spans (`target_ast.date_offset_range`), whose
    // end steps the context's anchor back and whose length steps the
    // examples back.
    let span = match &asked.range {
        None => None,
        Some(range) => Some((
            Offset::of(
                range
                    .start
                    .ok_or_else(|| unvalidated("a window with no start"))?,
                range.unit,
            ),
            Offset::of(range.end, range.unit),
        )),
    };

    // An anchor at each entity's own time (`anchor_time='entity'`) is a
    // static query's only, of an entity table with a time column, with the
    // context anchored the same way; no time bounds are read or checked
    // for it (`_get_task_table`). Otherwise the anchor and the context's
    // anchor, checked against the data's times.
    let anchor_time = if frames.is_some() && anchor_time.is_none() {
        Some(Scalar::Timestamp {
            value: TIME_MAX,
            unit: TimestampUnit::Ns,
            zone: None,
        })
    } else {
        anchor_time
    };
    let (anchor, context_anchor, earliest) = if anchor_entity {
        if span.is_some() {
            return Err(invalid(
                Sentence::AnchorEntityNotStatic,
                "an anchor at each entity's own time is for a static query only",
                "anchor_time",
            )
            .with_field("reason", Detail::text("entity_not_static")));
        }
        if entity.time_column.is_none() {
            return Err(invalid(
                Sentence::AnchorEntityNoTimeColumn,
                "an anchor at each entity's own time needs the entity table to have a time column",
                "anchor_time",
            )
            .with_field("reason", Detail::text("entity_without_time_column"))
            .with_field("table", Detail::text(&entity.name)));
        }
        if matches!(context_anchor, AskedAnchor::At(_)) {
            return Err(shared_anchor());
        }
        (TIME_MIN, TIME_MIN, None)
    } else {
        anchors_checked(
            asker,
            &mut memo,
            &read_times,
            &bounded,
            span.map(|(_, end)| end),
            anchor_time,
            &context_anchor,
        )
        .await?
    };

    // The examples: a reservoir sample of the entity table, kept one row
    // per entity, with the entity table's columns the query reads; for a
    // temporal query, walked back from the context's anchor by the window.
    let context_examples = mode.context_examples();
    // `max_pq_iterations` times as many candidates as examples (T2.11); a
    // float's count is no slice bound for a temporal query's walk, the
    // engine's `TypeError`, the caller's error.
    let trials = max_pq_iterations
        .unwrap_or(options::Iterations::Whole(ITERATIONS as u64))
        .trials(context_examples, span.is_some())
        .ok_or_else(|| {
            invalid(
                Sentence::IterationsWholeForWindow,
                "max_pq_iterations is a whole number for a query over a window",
                "max_pq_iterations",
            )
        })?;
    let mut entity_columns = BTreeSet::from([key.clone()]);
    if let Some(time) = &entity.time_column {
        entity_columns.insert(time.clone());
    }
    asked.target.entity_columns(&mut entity_columns);
    for shape in &asked.choosing {
        shape.entity_columns(&mut entity_columns);
    }
    let mut filters = Vec::new();
    if entity.column(&key).is_none_or(|c| c.nullable) {
        filters.push(Filter::NotNull {
            column: key.clone(),
        });
    }
    if let Some(time) = &entity.time_column
        && entity.column(time).is_none_or(|c| c.nullable)
    {
        filters.push(Filter::NotNull {
            column: time.clone(),
        });
    }
    // Seeds are optional. An unseeded provider documents its limitation
    // without issuing a routine warning for the default seed (P3D-13).
    let sampled = if let Some(frames) = &frames {
        crate::frame_materialize::lock(frames).entities(
            entity,
            &entity.ordered(&entity_columns),
            trials,
            seed.to_u128().unwrap_or(0),
        )?
    } else {
        read_times.read(
            asker,
            entity,
            one_batch(
                &ask(
                    asker,
                    DataRequest::SampleEntities {
                        table: entity.source.clone(),
                        key: vec![key.clone()],
                        n: trials as u64,
                        seed: seed_value,
                        columns: entity.ordered(&entity_columns),
                        filters,
                        // The schema's type of the key, for a provider that
                        // orders the draw by it (SQLite's, which reads the
                        // type the key's declaration fixes first).
                        key_dtype: asker
                            .samples_by_key_dtype()
                            .then(|| entity.column(&key).map(|c| c.dtype))
                            .flatten(),
                        // The table's rows, for a provider sizing its draw by
                        // them (the reference's sampled percentage).
                        rows: if asker.capabilities()?.sample_rows {
                            rows_of(&memo.memo, &entity.name)
                        } else {
                            None
                        },
                    },
                )
                .await?,
            )?,
        )?
    };
    if sampled.num_rows() == 0 {
        return Err(invalid(
            Sentence::EntityTableEmpty,
            "the entity table holds no row to sample",
            "query",
        )
        .with_field("table", Detail::text(&entity.name)));
    }
    // One entity per key value: `1.0` and `1.00` of a column of numbers
    // handed over as their text are one key (`subgraph::key_values`).
    let sampled_keys = crate::subgraph::key_values(entity, &key, named(&sampled, &key)?)?;
    let mut seen = BTreeSet::new();
    let kept: Vec<usize> = (0..sampled_keys.len())
        .filter(|&row| seen.insert(sampled_keys[row].clone()))
        .collect();
    let entity_key = take(named(&sampled, &key)?, &kept)?;
    let entity_time = match &entity.time_column {
        Some(time) => {
            let column = named(&sampled, time)?;
            // A time of day holds no instant: the reference's engine reads
            // each as none (`to_datetime`, see `times_of_day_unread`), so no
            // entity is valid at any anchor and a temporal query collects no
            // context example, the caller's error below (T1.36 fix 9). A
            // static query never gets here with one: its time bounds read
            // every time column first and fail on it (register DT-12).
            let all = if matches!(
                column.data_type(),
                DataType::Time32(_) | DataType::Time64(_)
            ) {
                vec![None; column.len()]
            } else {
                times_ns(column, time)?
            };
            Some(kept.iter().map(|&r| all[r]).collect::<Vec<_>>())
        }
        None => None,
    };
    // The entities, cast to the sampled key's type under the core's rule
    // table (T1.32a), where the reference casts them, after its sample
    // (`pd.Series(indices, dtype=train_pkey.dtype)`): an entity that does
    // not cast is the caller's error, never the core's (E41(l)).
    let indices = keys::cast(&indices, entity_key.data_type())?;
    let mut rng = default_rng(&entropy);
    let examples = match &span {
        Some((start, end)) => sample::temporal_examples(
            &mut rng,
            kept.len(),
            entity_time.as_deref(),
            context_anchor,
            timeframe(*start, *end, "anchor_time")?,
            earliest.unwrap_or(TIME_MAX),
            trials,
        )?,
        None if anchor_entity => sample::entity_examples(
            &mut rng,
            kept.len(),
            entity_time.as_deref().unwrap_or_default(),
            sample::MISSING_ANCHOR,
        ),
        None => {
            sample::static_examples(&mut rng, kept.len(), entity_time.as_deref(), context_anchor)
        }
    };

    // A NaN key a source hands over, which the reference reads as missing
    // where it reads floats into NumPy-backed pandas (Postgres's rows), is
    // a missing key its sampler cannot coerce (`_coerce_id`'s `int` or
    // `float` of `NA`) where an example's key seeds the read of a windowed
    // target's rows (`_by_time`): refused as the caller's error before that
    // read. A static target reads no rows by the key, and such an example
    // is left out once labelled, below (T2.12).
    let nan_key = nan_rows(&entity_key);
    if asker.reads_nan_as_missing()
        && !labelled_fetches.is_empty()
        && examples.index.iter().any(|i| nan_key.contains(i))
    {
        return Err(invalid(
            Sentence::EntityKeyMissing,
            "an entity sampled has no value for its table's primary key",
            "query",
        )
        .with_field("table", Detail::text(&entity.name))
        .with_field("column", Detail::text(&key)));
    }
    // Their labels, as the reference's executor computes them over the
    // rows read for them (decision E19).
    let entity_rows: Vec<usize> = examples.index.iter().map(|&i| kept[i]).collect();
    let labelling = Labelling {
        asker,
        times: &read_times,
        sampled: &sampled,
        entity_rows: &entity_rows,
        entity_key: &entity_key,
        examples: &examples,
    };
    let fetched = labelling.read(&labelled_fetches).await?;
    let target = labelling.build(&asked.target, &fetched, true)?;
    // The examples the entity filter and the assumption choose: each
    // condition's truth over the examples, and-ed (`execute`'s `mask`).
    let mut chosen: Option<Vec<bool>> = None;
    for shape in &asked.choosing {
        let holds = labels::truth(&labelling.build(shape, &fetched, false)?)?.values;
        chosen = Some(match chosen {
            None => holds,
            Some(before) => before.iter().zip(&holds).map(|(&a, &b)| a && b).collect(),
        });
    }
    drop(fetched);
    // A classification keeps a decimal column's distinct values apart as
    // classes; any other task reads them as numbers (T2.0c, round 1).
    let mut labelled = if matches!(asked.task, Task::Multiclass) {
        labels::label_classes(&target)?
    } else {
        labels::label(&target)?
    };
    null_compared(&asked.target, &labelled.mask)?;
    // The examples the entity filter or the assumption leaves out keep no
    // label (`out[mask[_mask]]`, `mask & _mask`).
    if let Some(chosen) = &chosen {
        let keep: Vec<bool> = (0..labelled.mask.len())
            .filter(|&row| labelled.mask[row])
            .map(|row| chosen[row])
            .collect();
        for (row, holds) in chosen.iter().enumerate() {
            labelled.mask[row] &= *holds;
        }
        labelled.labels = labelled.labels.kept(&keep);
    }
    // An example of an entity whose key is NaN is left out of the context
    // once labelled, as the reference leaves it out (T2.12).
    if !nan_key.is_empty() {
        let keep: Vec<bool> = (0..labelled.mask.len())
            .filter(|&row| labelled.mask[row])
            .map(|row| !nan_key.contains(&examples.index[row]))
            .collect();
        for row in 0..labelled.mask.len() {
            labelled.mask[row] &= !nan_key.contains(&examples.index[row]);
        }
        labelled.labels = labelled.labels.kept(&keep);
    }
    // The examples with a label, in order, the first a context holds.
    let labelled_rows: Vec<usize> = (0..labelled.mask.len())
        .filter(|&row| labelled.mask[row])
        .collect();
    let gathered = labelled_rows.len().min(context_examples);
    if gathered == 0 {
        let ends: Vec<Offset> = windows.iter().map(|w| w.end).collect();
        let mut aggregated: Vec<&Table> = Vec::new();
        for fetch in &fetches {
            if !aggregated.iter().any(|t| t.name == fetch.table.name) {
                aggregated.push(fetch.table);
            }
        }
        aggregated.sort_by(|a, b| a.name.cmp(&b.name));
        return Err(empty_sample(
            &mut memo,
            &read_times,
            entity,
            &aggregated,
            &ends,
            (!anchor_entity).then_some(context_anchor),
            validation.entity_filter.as_deref(),
        )
        .await);
    }
    if kept.len() >= trials && gathered < context_examples / 2 {
        // The reference warns of it once in a process: given on every run
        // that meets it, and once in a process by the host
        // ([`ONCE_PER_PROCESS`], decision E52).
        asker.warn(CoreWarning {
            fields: [
                ("examples".to_owned(), count(context_examples)),
                ("candidates".to_owned(), count(trials)),
            ]
            .into(),
            ..CoreWarning::new(
                Sentence::ContextCoverage,
                "fewer than half the context examples asked for were collected",
            )
        });
    }
    // The target's own past as features (`lag_timesteps`, T2.9), where the
    // reference's engine adds them: its value refused, as the engine
    // refuses it, once the context is labelled; for a temporal query, each
    // aggregated table's rows the lags read, in the labels' order, for the
    // context's examples and the entities predicted for, in that order,
    // before the context is narrowed ([`crate::lag`]).
    // The stored values the labels' reads spelled their bounds by, kept.
    keep_stored_times(&mut memo, &read_times);
    let steps = match lag_timesteps {
        None => 0,
        Some(asked) => asked.refusal(!fetches.is_empty() && link_target.is_none())?,
    };
    let lag = match steps {
        0 => None,
        steps => {
            // Each step moves by the target's window's end
            // (`target_timeframe.end_date_offset`).
            let (_, end) = span.ok_or_else(|| internal("a lag without its target's window"))?;
            let context: Vec<usize> = labelled_rows[..gathered]
                .iter()
                .map(|&row| examples.index[row])
                .collect();
            let keys = arrow_select::concat::concat(&[
                take(&entity_key, &context)?.as_ref(),
                indices.as_ref(),
            ])
            .map_err(|_| internal("the examples' keys do not concatenate"))?;
            let mut lag_anchors: Vec<i64> = labelled_rows[..gathered]
                .iter()
                .map(|&row| examples.anchors[row])
                .collect();
            lag_anchors.extend(std::iter::repeat_n(anchor, indices.len()));
            let back = crate::lag::shifted(steps, end, &lag_anchors)?;
            // Each table read once from the moved anchors, over its pooled
            // window stretched by the steps (`sample_lagged_target`'s
            // `time_offset_dict`).
            let mut fetched = BTreeMap::new();
            for fetch in &fetches {
                let (starts, ends) = crate::lag::reach(steps, fetch.start, fetch.end, &back)?;
                let rows =
                    fetch_rows(asker, &read_times, fetch, Arc::clone(&keys), starts, ends).await?;
                fetched.insert(fetch.table.name.clone(), rows);
            }
            keep_stored_times(&mut memo, &read_times);
            let shape = autoregressive(&asked.target);
            let lagged = lag_anchors.len();
            Some(crate::lag::Lag {
                steps,
                end,
                shifted: back,
                target: Box::new(move |anchors: &[i64]| {
                    temporal_target(&shape, &fetched, anchors, lagged, "lag_timesteps")
                }),
            })
        }
    };
    // The default mode capped the context at as many examples as it takes:
    // said, as the client says it, the caller having chosen no size.
    if num_context_samples.is_none() && gathered >= options::RunMode::Fast.context_examples() {
        asker.warn(CoreWarning {
            fields: [
                (
                    "cap".to_owned(),
                    count(options::RunMode::Fast.context_examples()),
                ),
                (
                    "most".to_owned(),
                    count(options::RunMode::Best.context_examples()),
                ),
            ]
            .into(),
            ..CoreWarning::new(
                Sentence::ContextCapped,
                "the default run mode gathered as many context examples as it takes",
            )
        });
    }
    // Then narrowed to the size asked for (`_narrow_context`).
    let num_train = asked_examples.map_or(gathered, |n| gathered.min(n));
    let train_rows: Vec<usize> = labelled_rows[..num_train]
        .iter()
        .map(|&row| examples.index[row])
        .collect();
    let train_keys = take(&entity_key, &train_rows)?;
    let mut anchors: Vec<i64> = labelled_rows[..num_train]
        .iter()
        .map(|&row| examples.anchors[row])
        .collect();
    let labels = labelled.labels.truncated(num_train);
    #[cfg(feature = "test-ops")]
    if labels_only {
        return labels_result(asker, &train_keys, &anchors, &labels).map(Gathered::Done);
    }
    // How many entities one call scores: the endpoint's 10,000, or fewer
    // where the caller asked (`rows_per_call`); a request of more is
    // gathered as several calls, each with a context of its own, at most as
    // many as the host's limit, 100 where it names none, refused once the
    // context is gathered, as the client refuses it after its engine's task
    // table (`_refuse_more_entities_than_fit_in_memory`, `main`'s 1fc5a686).
    let per_call = usize::try_from(asked.task.rows_one_call_scores(rows_per_call))
        .map_err(|_| internal("a call's rows past usize"))?;
    let num_test = indices.len();
    let calls = num_test.div_ceil(per_call);
    let most = most_calls.unwrap_or(MOST_CALLS);
    if calls > most as usize {
        return Err(CoreError::new(
            Sentence::MostCalls,
            "the entities asked for are gathered as more calls than one materialize holds at once",
        )
        .with_field("field", Detail::text("indices"))
        .with_field("entities", count(num_test))
        .with_field("calls", count(calls))
        .with_field("rows_per_call", count(per_call))
        .with_field("limit", Detail::Scalar(Scalar::UInt32(most))));
    }
    // How many candidates the call asked be kept (`_candidates_to_rank`,
    // T2.8), which the engine writes into the task; then the inference
    // options, read for the task's config class, as the engine reads them
    // before it lays out any request.
    let top_k = top_k.as_ref().map(options::top_k).transpose()?;
    let options = options::options(asked.task, inference_config.as_ref())?;
    // A static target's columns, kept out of what the request reads and
    // sends (`get_exclude_cols_dict`); each must be one of its table's
    // features, as the reference's sampler requires.
    let mut excluded_columns = BTreeSet::new();
    asked.target.entity_columns(&mut excluded_columns);
    let mut excluded: Vec<(String, String)> = excluded_columns
        .into_iter()
        .map(|column| (entity.name.clone(), column))
        .collect();
    for (_, column) in &excluded {
        if !graph.features(entity).iter().any(|(n, _)| n == column) {
            return Err(invalid(
                Sentence::TargetNotAFeature,
                "the column a static target reads is not one of its table's features",
                "query",
            ));
        }
    }
    // Then the columns the caller keeps out (`exclude_cols_dict`, T2.11),
    // merged after the query's (`_columns_to_leave_out`), each a feature of
    // a table of the graph, as the reference's sampler requires.
    // (`_columns_to_leave_out`, then the hops `_resolve_options` resolves,
    // then the names `sample_subgraph` checks.)
    let exclude_cols = options::excluded(&deferred)?;
    let walk_hops = if frames.is_some() && deferred.neighbors.is_some() {
        // The native frame sampler validates its signed array after
        // seed reset; SQL's numeric-comparison coercions do not apply.
        vec![]
    } else {
        options::neighbors(&deferred, mode)?
    };
    // Each fact table's look-back rate, where the provider's neighbor
    // reads look back no further than the reference's warehouse samplers.
    let lookback = if frames.is_some() {
        vec![]
    } else {
        lookbacks(&mut memo, &read_times, &graph).await?
    };
    let hops_ignored = options::hops_ignored(&deferred);
    // The engine says it ignores `num_hops` where `num_neighbors` is given
    // (`_resolve_options`), once the task is labelled.
    if let Some(num_hops) = hops_ignored {
        asker.warn(CoreWarning {
            fields: [("num_hops".to_owned(), Detail::text(&num_hops))].into(),
            ..CoreWarning::new(
                Sentence::HopsIgnored,
                "num_hops is ignored where num_neighbors is given",
            )
        });
    }
    excluded_by_caller(&graph, &exclude_cols, &mut excluded)?;
    if num_train > TABLE_ROWS {
        return Err(invalid(
            Sentence::ContextPastRows,
            "the context holds more rows than a request takes",
            "query",
        ));
    }
    // The rows the largest call scores (`_the_largest_batch`), which a
    // payload past the endpoint's body names as the size to come down from.
    let largest = per_call.min(num_test);
    let train_anchors = anchors.len();
    let entity_names = crate::subgraph::sample_columns(&graph, entity, &excluded);
    // The labelled context's examples before narrowing, then the entities
    // predicted for: the examples the lags were read for (T2.9).
    let labelled_examples = gathered;
    // The request's lagged columns, made once (T2.9 fix 3), and how many
    // entities the whole request scores.
    let mut lag_columns: Option<Vec<sdm_request::payload::LagColumn>> = None;
    let all_scored = num_test;
    // The columns each call reads of its examples' entity rows: with an
    // anchor at each entity's own time, its time column too, which anchors
    // each row (`_sample_subgraph`), sent only where it is a feature.
    let mut entity_read = entity_names.clone();
    if anchor_entity
        && let Some(time) = &entity.time_column
        && !entity_read.contains(time)
    {
        entity_read.push(time.clone());
    }
    let mut gathered = Vec::with_capacity(calls);
    let mut sent = BTreeSet::new();
    let mut answering = 0;
    // Each call's payload, in order: the context's examples and labels, the
    // same for every call, with the call's own entities, and the subgraph
    // walked from both, as the reference's engine builds each batch
    // (`_iter_task_requests`: `narrow_prediction`, then `_get_context`),
    // every one before any is sent.
    for start in (0..num_test).step_by(per_call) {
        let scored = indices.slice(start, per_call.min(num_test - start));
        let num_test = scored.len();
        let examples_total = num_train + num_test;
        let all_keys = arrow_select::concat::concat(&[train_keys.as_ref(), scored.as_ref()])
            .map_err(|_| internal("the examples' keys do not concatenate"))?;
        anchors.truncate(train_anchors);
        anchors.extend(std::iter::repeat_n(anchor, num_test));

        // The subgraph: each example's entity row.
        let looked_up = if let Some(frames) = &frames {
            crate::frame_materialize::lock(frames).lookup(entity, &all_keys, &entity_read)?
        } else {
            read_times.read(
                asker,
                entity,
                one_batch(
                    &ask(
                        asker,
                        sample::by_pkey(
                            entity,
                            entity_read.clone(),
                            key_set(asker, sample::seeds(Arc::clone(&all_keys), vec![])?)?,
                            true,
                        )?,
                    )
                    .await?,
                )?,
            )?
        };
        let positions = batches(&looked_up, examples_total)?;
        // A key the source enforces unique is held by one row each: a
        // dialect may leave its count out (`SEED_KEY_UNIQUE`), and any
        // other count is read.
        let unique = entity.column(&key).is_some_and(|c| c.unique);
        let matches = match looked_up.column_by_name(MATCHES_COLUMN) {
            None if unique => None,
            _ => Some(
                named(&looked_up, MATCHES_COLUMN)?
                    .as_primitive_opt::<Int64Type>()
                    .ok_or_else(|| {
                        crate::run::schema_mismatch("expected", Some(MATCHES_COLUMN), "not int64")
                    })?,
            ),
        };
        // Each entity to score that more than one row holds, in the order
        // the caller gave them, each value once, as the reference names
        // them (`_refuse_an_id_scored_twice`): the table, its key and the
        // values, values of the caller's own call (KO2-86).
        let mut repeated: Vec<usize> = positions
            .iter()
            .enumerate()
            .filter(|&(row, &at)| at >= num_train && matches.is_some_and(|m| m.value(row) > 1))
            .map(|(_, &at)| at)
            .collect();
        if !repeated.is_empty() {
            repeated.sort_unstable();
            repeated.dedup();
            let mut keys: Vec<Scalar> = Vec::with_capacity(repeated.len());
            for at in repeated {
                let value = key_scalar(scored.as_ref(), at - num_train);
                if !keys.contains(&value) {
                    keys.push(value);
                }
            }
            // Said of the first five, and how many more, as the reference
            // says them (`_refuse_an_id_scored_twice`).
            let (sentence, more) = match keys.len() {
                1 => (Sentence::HeldBySeveralRowsOne, 0),
                2..=5 => (Sentence::HeldBySeveralRows, 0),
                n => (Sentence::HeldBySeveralRowsMany, n - 5),
            };
            let mut refused = invalid(
                sentence,
                "an entity to predict for is held by more than one row, so which is meant \
                 would be a guess",
                "indices",
            );
            if more > 0 {
                refused = refused
                    .with_field("shown", Detail::List(keys[..5].to_vec()))
                    .with_field("more", count(more));
            }
            return Err(refused
                .with_field("reason", Detail::text("held_by_several_rows"))
                .with_field("table", Detail::text(&entity.name))
                .with_field("key", Detail::text(&key))
                .with_field("keys", Detail::List(keys)));
        }
        if positions.len() != examples_total {
            // The table and the keys not found, in the order the caller gave
            // them: values of the caller's own call, which a refusal may name
            // (KO2-86, narrowing decision H17), as the reference names them.
            let found: BTreeSet<usize> = positions.iter().copied().collect();
            let keys = (num_train..examples_total)
                .filter(|at| !found.contains(at))
                .map(|at| key_scalar(scored.as_ref(), at - num_train))
                .collect();
            return Err(invalid(
                Sentence::NotInTable,
                "an entity to predict for is not in the entity table",
                "indices",
            )
            .with_field("reason", Detail::text("not_in_table"))
            .with_field("table", Detail::text(&entity.name))
            .with_field("keys", Detail::List(keys)));
        }
        let mut order: Vec<usize> = (0..positions.len()).collect();
        order.sort_by_key(|&row| positions[row]);
        let entity_rows = take_batch(&looked_up, &order)?;
        // Each predicted entity anchored at its own time, read from its row.
        if anchor_entity && let Some(time) = &entity.time_column {
            let times = times_ns(named(&entity_rows, time)?, time)?;
            for (at, time) in anchors[train_anchors..].iter_mut().zip(&times[num_train..]) {
                // A null time is a missing anchor (`NaT`), never the
                // no-anchor sentinel.
                *at = time.unwrap_or(sample::MISSING_ANCHOR);
            }
        }
        let entity_rows = RecordBatch::try_new(
            Arc::new(arrow_schema::Schema::new(
                entity_names
                    .iter()
                    .map(|name| {
                        Ok(arrow_schema::Field::new(
                            name,
                            named(&entity_rows, name)?.data_type().clone(),
                            true,
                        ))
                    })
                    .collect::<Result<Vec<_>, CoreError>>()?,
            )),
            entity_names
                .iter()
                .map(|name| named(&entity_rows, name).map(Arc::clone))
                .collect::<Result<Vec<_>, _>>()?,
        )
        .map_err(|_| internal("the entity rows do not make one batch"))?;

        // The neighbors of each example's entity, hop by hop.
        let walked = if let Some(frames) = &frames {
            crate::frame_materialize::lock(frames).walk(
                asker,
                &graph,
                entity,
                &all_keys,
                &walk_hops,
                deferred.neighbors.as_deref(),
                &excluded,
                (!unseeded).then(|| seed.to_u128()).flatten(),
            )?
        } else {
            crate::subgraph::Walk {
                asker,
                graph: &graph,
                times: &read_times,
                anchors: &anchors,
                nulls,
                entity,
                link_target: link_target.as_deref(),
                neighbors: &walk_hops,
                excluded: &excluded,
                lookback: &lookback,
            }
            .run(entity_rows)
            .await?
        };
        keep_stored_times(&mut memo, &read_times);
        // The subgraph carries at most as many tables as the endpoint
        // reads, refused as the reference's engine refuses it
        // (`_get_context`).
        if walked.tables.len() > SUBGRAPH_TABLES {
            return Err(invalid(
                Sentence::SubgraphPastTables,
                "the request's subgraph carries more tables than the endpoint reads",
                "tables",
            )
            .with_field("tables", count(walked.tables.len()))
            .with_field("limit", count(SUBGRAPH_TABLES)));
        }
        if let Some(rhs) = &link_target
            && !walked.tables.iter().any(|table| &table.name == rhs)
        {
            return Err(CoreError::new(
                Sentence::LinkHistoryMissing,
                "the sampled subgraph has no historical target entities",
            ));
        }
        sent.extend(walked.tables.iter().map(|t| t.name.clone()));
        // The lagged columns of this call's examples (T2.9): the narrowed
        // context's, then this call's own entities, each its own row's,
        // sliced from the whole request's columns, so each column's type
        // is the request's whatever this call's rows hold (fix 3), typed
        // over the gathered context before narrowing (fix 4); the
        // columns are computed at the first call's payload, where the
        // reference's size refusal falls, held to the largest call's rows.
        if lag_columns.is_none()
            && let Some(lag) = &lag
        {
            lag.fits(num_train + largest)?;
            lag_columns = Some(
                lag.columns(
                    &(0..num_train)
                        .chain(labelled_examples..labelled_examples + all_scored)
                        .collect::<Vec<_>>(),
                )?,
            );
        }
        let rows: Vec<usize> = (0..num_train)
            .chain(num_train + start..num_train + start + num_test)
            .collect();
        let lags = lag_columns
            .iter()
            .flatten()
            .map(|column| column.rows(&rows))
            .collect();
        // The options the engine's request model holds to a boolean, read
        // as the request is built.
        let (use_prediction_time, return_embeddings) = options::request_flags(&deferred)?;

        let built = payload::build(&Sampled {
            num_train,
            num_test,
            anchors: anchors.clone(),
            kind: asked.task,
            labels: labels.clone(),
            windowed: asked.range.is_some(),
            tables: walked.tables,
            links: walked.links,
            entity_table: entity.name.clone(),
            link_target: link_target.clone(),
            source_query: source_query.clone(),
            run_mode: mode,
            use_prediction_time,
            return_embeddings,
            options: options.clone(),
            statistic: statistic.clone(),
            rows_per_call: largest,
            top_k,
            lags,
            float_text: crate::pql::float_text,
        })?;
        answering += built.answering;
        gathered.push((
            built.bytes,
            u64::try_from(num_test).map_err(|_| internal("a call's rows past u64"))?,
        ));
    }
    // What the request leaves out, said once as the client says it of all
    // its calls, in its order: the tables no call carries, the columns the
    // model does not read, and the context that answers the question,
    // counted across the calls.
    notices(
        asker,
        &description,
        &asked,
        &sent,
        &excluded,
        answering,
        asked.range.is_some(),
    );
    Ok(Gathered::Request(gathered, source_query))
}

/// The refusal of a query that does not resolve against the schema, with
/// the query and each error PQL found (its message names where), for the
/// host's words (`PredictiveQuery._refusal`).
fn unresolved(query: &str, diagnostics: &[sdm_contract::ops::PqlDiagnostic]) -> CoreError {
    let reasons: Vec<Scalar> = diagnostics
        .iter()
        .filter(|d| d.severity == sdm_contract::ops::Severity::Error)
        .map(|d| Scalar::Utf8(d.message.clone()))
        .collect();
    invalid(
        Sentence::Unresolved,
        "the query does not resolve against the schema",
        "query",
    )
    .with_field("reason", Detail::text("unresolved"))
    .with_field("query", Detail::text(query))
    .with_field("reasons", Detail::List(reasons))
}

/// The refusal of anchors set apart where one is each entity's own time
/// (`_get_task_table`'s "needs to be shared").
fn shared_anchor() -> CoreError {
    invalid(
        Sentence::AnchorEntityNotShared,
        "an anchor at each entity's own time is shared by the context and the predictions",
        "context_anchor_time",
    )
    .with_field("reason", Detail::text("entity_not_shared"))
}

/// The anchor and the context's anchor, in nanoseconds, checked against
/// the times of the tables that bound them (`bounded`, `_validate_time`):
/// the anchor given, or the latest time of those tables; the context's
/// anchor given (`context_anchor_time`, T2.11), or the window's end
/// (`end`, a temporal query's) before the anchor, by the calendar for
/// months, a static query's being the anchor itself. Also the earliest
/// time of those tables, where it was read.
async fn anchors_checked(
    asker: &Asker,
    memo: &mut Memo<'_>,
    read_times: &Times,
    bounded: &[&Table],
    end: Option<Offset>,
    anchor_time: Option<Scalar>,
    context_given: &AskedAnchor,
) -> Result<(i64, i64, Option<i64>), CoreError> {
    let mut bounds: Option<Bounds> = None;
    let anchor = match anchor_time {
        None if bounded.is_empty() => TIME_MIN,
        None => {
            let read = time_bounds(memo, read_times, bounded).await?;
            let max = read.max;
            bounds = Some(read);
            max
        }
        // A zoned anchor counts its UTC instant, the zone only how it is
        // shown: read at its UTC wall clock, as the reference's engine
        // reads one (`to_naive_utc`).
        Some(Scalar::Timestamp { unit, value, .. }) => match unit {
            TimestampUnit::Ns => value,
            TimestampUnit::S => value
                .checked_mul(1_000_000_000)
                .ok_or_else(|| sample::out_of_range("anchor_time"))?,
            TimestampUnit::Ms => value
                .checked_mul(1_000_000)
                .ok_or_else(|| sample::out_of_range("anchor_time"))?,
            TimestampUnit::Us => value
                .checked_mul(1_000)
                .ok_or_else(|| sample::out_of_range("anchor_time"))?,
        },
        Some(_) => {
            return Err(invalid(
                Sentence::AnchorNotATime,
                "an anchor time is a ts scalar",
                "anchor_time",
            ));
        }
    };
    let context_anchor = match context_given {
        AskedAnchor::Entity => return Err(shared_anchor()),
        AskedAnchor::At(at) => *at,
        AskedAnchor::Absent => match end {
            Some(end) => end.negated().add_to(anchor, "anchor_time")?,
            None => anchor,
        },
    };
    let bounds = match bounds {
        Some(bounds) => Some(bounds),
        None if bounded.is_empty() => None,
        None => Some(time_bounds(memo, read_times, bounded).await?),
    };
    if let Some(bounds) = &bounds {
        // Each names the times, as the reference's words show them.
        let at = |value: i64| {
            Detail::Scalar(Scalar::Timestamp {
                unit: TimestampUnit::Ns,
                zone: None,
                value,
            })
        };
        if anchor < bounds.min {
            return Err(invalid(
                Sentence::AnchorBeforeData,
                "the anchor time is before the earliest time in the data",
                "anchor_time",
            )
            .with_field("reason", Detail::text("anchor_before_data"))
            .with_field("anchor", at(anchor))
            .with_field("earliest", at(bounds.min)));
        }
        if context_anchor < bounds.min {
            return Err(invalid(
                Sentence::ContextBeforeDataRefused,
                "the context's anchor time is before the earliest time in the data",
                "anchor_time",
            )
            .with_field("reason", Detail::text("context_before_data"))
            .with_field("context_anchor", at(context_anchor))
            .with_field("earliest", at(bounds.min))
            // As the reference's client marks this refusal
            // (`window_outran_the_data`), so a forecast's step says its
            // window is that step's (T2.14b fix 7).
            .with_field("window_outran_the_data", Detail::Scalar(Scalar::Bool(true))));
        }
        // The reference's three warnings of the context's anchor, of which
        // it gives the first that holds: a context anchored after the
        // predictions; a temporal context whose window reaches past the
        // predictions' anchor; a context reaching back before the data. A
        // static query's window is empty, so its context reaches back no
        // further than its anchor, which was checked above.
        // Each names the times it compares, as the reference's words show
        // them.
        let at = |value: i64| {
            Detail::Scalar(Scalar::Timestamp {
                unit: TimestampUnit::Ns,
                zone: None,
                value,
            })
        };
        let timed = |sentence: Sentence, message: &str, fields: [(&str, i64); 2]| {
            asker.warn(
                CoreWarning::new(sentence, message)
                    .with_fields(fields.map(|(name, value)| (name, at(value)))),
            );
        };
        let anchored = |sentence: Sentence, message: &str| {
            timed(
                sentence,
                message,
                [("context_anchor", context_anchor), ("anchor", anchor)],
            );
        };
        if context_anchor > anchor {
            anchored(
                Sentence::ContextAfterAnchor,
                "the context's anchor time is after the predictions' anchor time",
            );
        } else if let Some(end) = end
            && reaches(end, context_anchor)? > i128::from(anchor)
        {
            anchored(
                Sentence::ContextLeaks,
                "the context's window reaches past the predictions' anchor time",
            );
        } else if let Some(end) = end
            && let back = reaches(end.negated(), context_anchor)?
            && back < i128::from(bounds.min)
        {
            // How far back the context reaches, and the earliest time of
            // the data (M4): a reach past what a time holds names neither.
            match i64::try_from(back) {
                Ok(back) => timed(
                    Sentence::ContextBeforeData,
                    "the context reaches back before the data",
                    [("back_to", back), ("earliest", bounds.min)],
                ),
                Err(_) => warn(
                    asker,
                    Sentence::ContextBeforeDataUntimed,
                    "the context reaches back before the data",
                ),
            }
        }
        if i128::from(anchor) > i128::from(bounds.max) + DAY_NS {
            // The anchor and the latest time of the data (M4).
            timed(
                Sentence::AnchorAfterData,
                "the anchor time is after the data",
                [("anchor", anchor), ("latest", bounds.max)],
            );
        }
    }
    Ok((anchor, context_anchor, bounds.map(|b| b.min)))
}

/// Adds the columns the caller keeps out of the request (`exclude_cols_dict`)
/// to `excluded`, each once, in the order given, as the reference merges
/// them after the query's (`_columns_to_leave_out`) and its sampler reads
/// them (`sample_subgraph`): a table the graph does not hold, or a name
/// that is not one of its table's features (its key and foreign keys
/// never are), is the caller's error, `INVALID_REQUEST` naming
/// `exclude_cols_dict`, the table and the column.
fn excluded_by_caller(
    graph: &Graph,
    given: &[(String, Vec<sdm_wire::json::Value>)],
    excluded: &mut Vec<(String, String)>,
) -> Result<(), CoreError> {
    for (table_name, columns) in given {
        let table = graph.table(table_name).ok_or_else(|| {
            let mut tables: Vec<&str> = graph.tables.iter().map(|t| t.name.as_str()).collect();
            tables.sort_unstable();
            invalid(
                Sentence::ExcludedUnknownTable,
                "a table the caller keeps columns of is not in the graph",
                "exclude_cols_dict",
            )
            .with_field("reason", Detail::text("unknown_table"))
            .with_field("table", Detail::text(table_name))
            .with_field("tables", names(tables))
        })?;
        let features = graph.features(table);
        for column in columns {
            let name = column
                .as_text()
                .filter(|name| features.iter().any(|(n, _)| n == name));
            let Some(name) = name else {
                let mut excludable: Vec<&str> = features.iter().map(|(n, _)| n.as_str()).collect();
                excludable.sort_unstable();
                // The column as the caller gave it, its JSON text, for the
                // host's words.
                return Err(invalid(
                    Sentence::ExcludedNotAFeature,
                    "a column the caller keeps out is not one of its table's features",
                    "exclude_cols_dict",
                )
                .with_field("reason", Detail::text("not_a_feature"))
                .with_field("table", Detail::text(table_name))
                .with_field(
                    "given",
                    Detail::text(String::from_utf8_lossy(&column.to_bytes()).into_owned()),
                )
                .with_field("columns", names(excludable)));
            };
            let pair = (table.name.clone(), name.to_owned());
            if !excluded.contains(&pair) {
                excluded.push(pair);
            }
        }
    }
    Ok(())
}

/// `at + offset` in nanoseconds: a fixed span added without bound, as the
/// reference compares such times; a calendar one as pandas adds it.
fn reaches(offset: Offset, at: i64) -> Result<i128, CoreError> {
    match offset.fixed_nanos() {
        Some(span) => Ok(i128::from(at) + span),
        None => offset.add_to(at, "anchor_time").map(i128::from),
    }
}

/// An average, a minimum or a maximum compared with `NULL`, checked over
/// the examples [`labels::label`] gave a label (`mask`), as the reference
/// meets it: its executor labels only the examples with a value in their
/// window, then indexes every example with those labels (`execute`), so
/// its first batch of examples ([`LABEL_BATCH`]) decides. Each with a
/// value: labelled. None with one: numpy's broadcast fails, the caller's
/// error, `INVALID_REQUEST`. Some without: an `IndexError` the reference
/// gives as `INTERNAL_ERROR`, refused by name here (`average_null`,
/// `min_max_null`; register P1B-21, the class of decision E66(b)). An
/// example past the batch without a value is left unlabelled, never in a
/// context, as there.
fn null_compared(target: &Shape, mask: &[bool]) -> Result<(), CoreError> {
    let Shape::Condition {
        target,
        test: Test::Compare(_, None),
    } = target
    else {
        return Ok(());
    };
    let Shape::Aggregated(Windowed { aggregation, .. }) = target.as_ref() else {
        return Ok(());
    };
    let rule = match aggregation {
        labels::Aggregation::Avg => "average_null",
        labels::Aggregation::Min | labels::Aggregation::Max => "min_max_null",
        labels::Aggregation::Sum
        | labels::Aggregation::Count
        | labels::Aggregation::ListDistinct => return Ok(()),
    };
    let batch = &mask[..mask.len().min(LABEL_BATCH)];
    if batch.is_empty() {
        return Ok(());
    }
    if !batch.contains(&true) {
        return Err(invalid(
            Sentence::NullComparedNowhere,
            "no example's window holds a value to compare with NULL",
            "query",
        ));
    }
    if batch.contains(&false) {
        return Err(not_yet(rule));
    }
    Ok(())
}

/// The refusal of a context no example was collected for, with the reason
/// the reference's sampler finds for it (`_diagnose_empty_sample`), as a
/// `reason` and the facts it names, which the host words: `anchor` is the
/// context's anchor asked, none for an anchor at each entity's own time;
/// `aggregated` the tables the target aggregates, by name, and `ends` the
/// ends of its aggregations' windows; `condition` the entity filter's.
///
/// - `no_entity_at_anchor`: the entity table's earliest time is after the
///   anchor (`entity_table`, `column`, `anchor`, `earliest`, and `later`
///   where an anchor after it leaves the target's window room in the data);
/// - `no_entity_passed`: the data leaves room after the anchor for the
///   target's whole window, and an entity filter chose no example
///   (`anchor`, `tables`, `min_time`, `max_time`, `condition`);
/// - `anchor_too_late`: the data leaves no such room (the same facts but
///   the condition);
/// - `too_restrictive`: anything else, as the reference's generic reason.
///
/// A time bound that cannot be read gives the reason without it, as the
/// reference's diagnosis falls back when its own reads fail.
async fn empty_sample(
    memo: &mut Memo<'_>,
    times: &Times,
    entity: &Table,
    aggregated: &[&Table],
    ends: &[Offset],
    anchor: Option<i64>,
    condition: Option<&str>,
) -> CoreError {
    let refused = |sentence: Sentence, reason: &str| {
        invalid(sentence, "no context example was collected", "anchor_time")
            .with_field("reason", Detail::text(reason))
    };
    let too_restrictive = || refused(Sentence::ContextTooRestrictive, "too_restrictive");
    let several = aggregated.len() > 1;
    let Some(anchor) = anchor else {
        return too_restrictive();
    };
    let at = |value: i64| {
        Detail::Scalar(Scalar::Timestamp {
            unit: TimestampUnit::Ns,
            zone: None,
            value,
        })
    };
    // Whether `from` plus the latest of the ends is not after `max`
    // (`_window_fits_after`).
    let fits = |from: i64, max: i64| {
        let Some(first) = ends.first() else {
            return false;
        };
        let end = ends[1..]
            .iter()
            .fold(*first, |end, other| pooled(end, *other, false));
        reaches(end, from).is_ok_and(|reached| reached <= i128::from(max))
    };
    let bounds = if aggregated.is_empty() {
        None
    } else {
        time_bounds(memo, times, aggregated).await.ok()
    };
    if let Some(column) = &entity.time_column {
        let earliest = time_bounds(memo, times, &[entity])
            .await
            .ok()
            .map(|b| b.min)
            .filter(|&min| min != TIME_MAX && min > anchor);
        if let Some(earliest) = earliest {
            let later = bounds.as_ref().is_some_and(|b| fits(earliest, b.max));
            let sentence = if later {
                Sentence::NoEntityAtAnchorLater
            } else {
                Sentence::NoEntityAtAnchor
            };
            return refused(sentence, "no_entity_at_anchor")
                .with_field("entity_table", Detail::text(entity.name.as_str()))
                .with_field("column", Detail::text(column.as_str()))
                .with_field("anchor", at(anchor))
                .with_field("earliest", at(earliest))
                .with_field("later", Detail::Scalar(Scalar::Bool(later)));
        }
    }
    let Some(bounds) = bounds else {
        return too_restrictive();
    };
    let facts = |error: CoreError| {
        error
            .with_field("anchor", at(anchor))
            .with_field("tables", names(aggregated.iter().map(|t| t.name.as_str())))
            .with_field("min_time", at(bounds.min))
            .with_field("max_time", at(bounds.max))
    };
    if fits(anchor, bounds.max) {
        return match condition {
            Some(condition) => {
                let sentence = if several {
                    Sentence::NoEntityPassedSeveral
                } else {
                    Sentence::NoEntityPassed
                };
                facts(refused(sentence, "no_entity_passed"))
                    .with_field("condition", Detail::text(condition))
            }
            None => too_restrictive(),
        };
    }
    let sentence = if several {
        Sentence::AnchorTooLateSeveral
    } else {
        Sentence::AnchorTooLate
    };
    facts(refused(sentence, "anchor_too_late"))
}

/// One aggregated table's read of the rows the target's aggregations of it
/// take (`_by_time`): the foreign key by which it refers to the entity
/// table, the window its rows are read over from each example's anchor,
/// pooled over those aggregations, and the columns they and their filters
/// read, with the table's time column.
struct Fetch<'g> {
    table: &'g Table,
    edge: Edge,
    start: Offset,
    end: Offset,
    columns: BTreeSet<String>,
}

/// The reads the target's aggregations take, one for each table they
/// aggregate, in the order the reference's sampler meets the tables
/// (`_get_query_time_offset_dict`): each table's foreign key to the entity
/// table must be its only one, or the query is refused as there; the
/// window is the earliest start and the latest end of its aggregations'
/// (`min_date_offset`, `max_date_offset`).
fn fetches<'g>(
    graph: &'g Graph,
    entity_table: &str,
    windows: &[&Windowed],
) -> Result<Vec<Fetch<'g>>, CoreError> {
    let mut out: Vec<Fetch<'g>> = Vec::new();
    for windowed in windows {
        let found: Vec<_> = graph
            .edges
            .iter()
            .filter(|e| e.src == windowed.table && e.dst == entity_table)
            .collect();
        let edge = match found.as_slice() {
            [one] => (*one).clone(),
            _ => {
                return Err(invalid(
                    Sentence::NoUniqueForeignKey,
                    "no unique foreign key leads from the aggregated table to the entity table",
                    "query",
                ));
            }
        };
        let mut columns = BTreeSet::new();
        if let Some(column) = &windowed.column {
            columns.insert(column.clone());
        }
        if let Some(filter) = &windowed.filter {
            filter.entity_columns(&mut columns);
        }
        match out.iter_mut().find(|f| f.edge == edge) {
            Some(fetch) => {
                fetch.start = pooled(fetch.start, windowed.start, true);
                fetch.end = pooled(fetch.end, windowed.end, false);
                fetch.columns.extend(columns);
            }
            None => {
                let table = graph
                    .table(&windowed.table)
                    .ok_or_else(|| internal("the aggregated table is not in the graph"))?;
                columns.insert(
                    table
                        .time_column
                        .clone()
                        .ok_or_else(|| internal("the aggregated table has no time column"))?,
                );
                out.push(Fetch {
                    table,
                    edge,
                    start: windowed.start,
                    end: windowed.end,
                    columns,
                });
            }
        }
    }
    Ok(out)
}

/// The most and the fewest seconds `offset` can span, as the reference's
/// sampler bounds a calendar offset (`date_offset_to_seconds`): a month
/// from 28 to 31 days.
fn offset_seconds(offset: Offset, latest: bool) -> i128 {
    let count = i128::from(offset.count);
    let days = |shortest: i128, longest: i128| {
        if (count > 0) == latest {
            longest
        } else {
            shortest
        }
    };
    match offset.unit {
        sdm_contract::pql::TimeUnit::Months => count * days(28, 31) * 86_400,
        _ => offset
            .fixed_nanos()
            .map_or(0, |nanos| nanos / 1_000_000_000),
    }
}

/// Of two offsets, the one that reaches furthest back (`earliest`) or
/// forward for every anchor, as the reference pools a table's windows
/// (`_pooled_offset`): one of them where it bounds the other at every
/// anchor, the first such; otherwise the extreme span either can take, in
/// seconds.
fn pooled(first: Offset, second: Offset, earliest: bool) -> Offset {
    if first == second {
        return first;
    }
    // The candidate's span at the far end from the pool's own bound,
    // against the other's at that bound.
    let bounds = |candidate: Offset, other: Offset| {
        let mine = offset_seconds(candidate, earliest);
        let theirs = offset_seconds(other, !earliest);
        if earliest {
            mine <= theirs
        } else {
            mine >= theirs
        }
    };
    if bounds(first, second) {
        return first;
    }
    if bounds(second, first) {
        return second;
    }
    let spans = [
        offset_seconds(first, !earliest),
        offset_seconds(second, !earliest),
    ];
    let span = if earliest {
        spans[0].min(spans[1])
    } else {
        spans[0].max(spans[1])
    };
    Offset {
        count: i64::try_from(span).unwrap_or(if earliest { i64::MIN } else { i64::MAX }),
        unit: sdm_contract::pql::TimeUnit::Seconds,
    }
}

/// What a run's labels are read from: the entity rows sampled and, for
/// each table the target aggregates, its rows each example's windows hold
/// (`_sample_query_data`).
struct Labelling<'a> {
    asker: &'a Asker,
    times: &'a Times,
    /// The entity sample, and each example's row of it.
    sampled: &'a RecordBatch,
    entity_rows: &'a [usize],
    /// Each kept entity's key, which the examples index.
    entity_key: &'a ArrayRef,
    examples: &'a sample::Examples,
}

/// One aggregated table's rows read for the examples: the rows, the
/// example each is for, and its time.
struct Fetched {
    rows: RecordBatch,
    batch: Vec<usize>,
    times: Vec<Option<i64>>,
}

impl Labelling<'_> {
    /// Each table in `fetches` read for the examples, in order, by name,
    /// which [`Labelling::build`] puts a target or a condition together
    /// from.
    async fn read(&self, fetches: &[Fetch<'_>]) -> Result<BTreeMap<String, Fetched>, CoreError> {
        let keys = take(self.entity_key, &self.examples.index)?;
        let anchors = &self.examples.anchors;
        let mut fetched = BTreeMap::new();
        for fetch in fetches {
            let starts = shifted(anchors, fetch.start, "anchor_time")?;
            let ends = shifted(anchors, fetch.end, "anchor_time")?;
            let rows = fetch_rows(
                self.asker,
                self.times,
                fetch,
                Arc::clone(&keys),
                starts,
                ends,
            )
            .await?;
            fetched.insert(fetch.table.name.clone(), rows);
        }
        Ok(fetched)
    }

    /// `shape` over the examples, from the rows read; `whole` where it is
    /// the whole target, whose column is a static target read as one.
    fn build(
        &self,
        shape: &Shape,
        fetched: &BTreeMap<String, Fetched>,
        whole: bool,
    ) -> Result<labels::Target, CoreError> {
        Ok(match shape {
            Shape::Column(column) => {
                let array = named(self.sampled, column)?;
                if float32_overflows(&take(array, self.entity_rows)?) {
                    warn_float32_overflow(self.asker);
                }
                labels::Target::Column(if whole {
                    static_column(array, self.entity_rows)?
                } else {
                    condition_column(array, self.entity_rows)?
                })
            }
            Shape::Aggregated(windowed) => aggregate_target(
                windowed,
                {
                    let rows = fetched
                        .get(&windowed.table)
                        .ok_or_else(|| internal("an aggregation's table was not read"))?;
                    if let Some(column) = &windowed.column
                        && float32_overflows(named(&rows.rows, column)?)
                    {
                        warn_float32_overflow(self.asker);
                    }
                    // The columns its filter reads, which the executor casts
                    // to float32 too (`execute_column`).
                    if let Some(filter) = &windowed.filter {
                        let mut read = BTreeSet::new();
                        filter.entity_columns(&mut read);
                        for column in read {
                            if float32_overflows(named(&rows.rows, &column)?) {
                                warn_float32_overflow(self.asker);
                                break;
                            }
                        }
                    }
                    rows
                },
                &self.examples.anchors,
                self.examples.index.len(),
                "anchor_time",
            )?,
            Shape::Condition { target, test } => tested(self.build(target, fetched, false)?, test),
            Shape::Not(inner) => labels::Target::Not(Box::new(self.build(inner, fetched, false)?)),
            Shape::Both { op, left, right } => labels::Target::Both {
                op: *op,
                left: Box::new(self.build(left, fetched, false)?),
                right: Box::new(self.build(right, fetched, false)?),
            },
        })
    }
}

/// The target the lags label, as `to_autoregressive_query` makes it: a
/// `NOT` around the whole target dropped, so a negated condition's lags
/// are the condition's own; a `NOT` inside the target kept. (Its other
/// step, dropping a comparison of an aggregation, never applies: the
/// reference's comparison holds the aggregation's join, not the
/// aggregation.)
fn autoregressive(target: &Shape) -> Shape {
    match target {
        Shape::Not(inner) => inner.as_ref().clone(),
        shape => shape.clone(),
    }
}

/// A temporal target `shape` over `examples` examples anchored at
/// `anchors`, from each aggregated table's rows read for them (`fetched`):
/// each aggregation over its own window from the example's anchor, its
/// conditions and logical operations over those (`execute`, as the
/// reference's executor puts the target together). The reference's
/// validation lets no static column into a temporal target. A time
/// outside what nanoseconds hold is refused naming `field`.
fn temporal_target(
    shape: &Shape,
    fetched: &BTreeMap<String, Fetched>,
    anchors: &[i64],
    examples: usize,
    field: &str,
) -> Result<labels::Target, CoreError> {
    let part = |inner: &Shape| temporal_target(inner, fetched, anchors, examples, field);
    Ok(match shape {
        Shape::Aggregated(windowed) => aggregate_target(
            windowed,
            fetched
                .get(&windowed.table)
                .ok_or_else(|| internal("an aggregation's table was not read"))?,
            anchors,
            examples,
            field,
        )?,
        Shape::Condition { target, test } => tested(part(target)?, test),
        Shape::Not(inner) => labels::Target::Not(Box::new(part(inner)?)),
        Shape::Both { op, left, right } => labels::Target::Both {
            op: *op,
            left: Box::new(part(left)?),
            right: Box::new(part(right)?),
        },
        Shape::Column(_) => return Err(unvalidated("a static column in a temporal target")),
    })
}

/// `fetch`'s rows: each example's rows of the table whose foreign key
/// names the example's entity (`keys`, one per example), after its
/// `starts` and not after its `ends` (`_by_time`, one statement).
async fn fetch_rows(
    asker: &Asker,
    times: &Times,
    fetch: &Fetch<'_>,
    keys: ArrayRef,
    starts: Vec<i64>,
    ends: Vec<i64>,
) -> Result<Fetched, CoreError> {
    let examples = keys.len();
    let table = fetch.table;
    let time_column = table
        .time_column
        .clone()
        .ok_or_else(|| internal("the aggregated table has no time column"))?;
    let seeds = sample::seeds(keys, vec![(SEED_END, ends), (SEED_START, starts)])?;
    let rows = times.read(
        asker,
        table,
        one_batch(
            &ask(
                asker,
                times
                    .windowed(
                        asker,
                        table,
                        sample::by_time(
                            table,
                            &fetch.edge.fkey,
                            table.ordered(&fetch.columns),
                            key_set(asker, seeds)?,
                            true,
                        )?,
                    )
                    .await?,
            )
            .await?,
        )?,
    )?;
    Ok(Fetched {
        batch: batches(&rows, examples)?,
        times: times_ns(named(&rows, &time_column)?, &time_column)?,
        rows,
    })
}

/// Each of `anchors` moved by `offset`, a time outside what nanoseconds
/// hold refused naming `field`.
fn shifted(anchors: &[i64], offset: Offset, field: &str) -> Result<Vec<i64>, CoreError> {
    anchors.iter().map(|a| offset.add_to(*a, field)).collect()
}

/// `windowed` over each of `examples` examples' rows of its table in
/// `fetched`: those its filter keeps, after its own window's start and
/// not after its end from the example's anchor (`execute_aggregation`).
fn aggregate_target(
    windowed: &Windowed,
    fetched: &Fetched,
    anchors: &[i64],
    examples: usize,
    field: &str,
) -> Result<labels::Target, CoreError> {
    let ends = shifted(anchors, windowed.end, field)?;
    let starts = shifted(anchors, windowed.start, field)?;
    let kept = match &windowed.filter {
        None => None,
        Some(filter) => Some(labels::truth(&row_condition(filter, &fetched.rows)?)?.values),
    };
    let bounds = labels::Bounds {
        starts: Some(&starts),
        ends: &ends,
    };
    if windowed.aggregation == labels::Aggregation::ListDistinct {
        let name = windowed
            .column
            .as_deref()
            .ok_or_else(|| unvalidated("a list target with no column"))?;
        let array = named(&fetched.rows, name)?;
        let rows: Vec<usize> = (0..array.len()).collect();
        let values = static_column(array, &rows)?;
        return labels::distinct(
            examples,
            &fetched.batch,
            &values,
            &fetched.times,
            bounds,
            kept.as_deref(),
        )
        .map(labels::Target::Listed);
    }
    let values = match &windowed.column {
        Some(column) => label_values(named(&fetched.rows, column)?)?,
        None => labels::Values::Ones(fetched.rows.num_rows()),
    };
    labels::aggregate_kept(
        windowed.aggregation,
        examples,
        &fetched.batch,
        &values,
        &fetched.times,
        labels::Bounds {
            starts: Some(&starts),
            ends: &ends,
        },
        kept.as_deref(),
    )
    .map(labels::Target::Aggregated)
}

/// `target` tested as `test` asks.
fn tested(target: labels::Target, test: &Test) -> labels::Target {
    let target = Box::new(target);
    match test {
        Test::Compare(op, constant) => labels::Target::Condition {
            target,
            op: *op,
            constant: constant.clone(),
        },
        Test::In(values) => labels::Target::Member {
            target,
            values: values.clone(),
        },
    }
}

/// A filter, over every row read of the table it filters.
fn row_condition(filter: &Shape, rows: &RecordBatch) -> Result<labels::Target, CoreError> {
    let every: Vec<usize> = (0..rows.num_rows()).collect();
    Ok(match filter {
        Shape::Column(column) => {
            labels::Target::Column(condition_column(named(rows, column)?, &every)?)
        }
        Shape::Condition { target, test } => tested(row_condition(target, rows)?, test),
        Shape::Not(inner) => labels::Target::Not(Box::new(row_condition(inner, rows)?)),
        Shape::Both { op, left, right } => labels::Target::Both {
            op: *op,
            left: Box::new(row_condition(left, rows)?),
            right: Box::new(row_condition(right, rows)?),
        },
        Shape::Aggregated(_) => return Err(unvalidated("an aggregation in a filter")),
    })
}

/// The test-only `labels` operation's result: the context's examples, each
/// its entity's key (`entity`), its anchor (`anchor`, zone-less
/// nanoseconds) and its label (`target`, at the type the executor gives
/// it), attached as one Arrow batch.
#[cfg(feature = "test-ops")]
fn labels_result(
    asker: &Asker,
    keys: &ArrayRef,
    anchors: &[i64],
    labels: &labels::Labels,
) -> Result<OperationResult, CoreError> {
    use arrow_array::{BooleanArray, Float32Array, Int64Array, StringArray};
    let target: ArrayRef = match labels {
        labels::Labels::Float(v) => Arc::new(Float32Array::from(v.clone())),
        labels::Labels::Int(v) => Arc::new(Int64Array::from(v.clone())),
        labels::Labels::Bool(v) => Arc::new(BooleanArray::from(v.clone())),
        labels::Labels::Text(v) => Arc::new(StringArray::from(v.clone())),
        labels::Labels::Lists(_) => {
            return Err(internal("the scalar label echo does not carry list labels"));
        }
    };
    let schema = Arc::new(arrow_schema::Schema::new(vec![
        arrow_schema::Field::new("entity", keys.data_type().clone(), true),
        arrow_schema::Field::new(
            "anchor",
            DataType::Timestamp(ArrowUnit::Nanosecond, None),
            false,
        ),
        arrow_schema::Field::new("target", target.data_type().clone(), false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::clone(keys),
            Arc::new(TimestampNanosecondArray::from(anchors.to_vec())),
            target,
        ],
    )
    .map_err(|_| internal("the labels do not make one batch"))?;
    let rows = batch.num_rows() as u64;
    let data = asker.attach(Attachment::Arrow(ArrowData::new(schema, vec![batch])))?;
    Ok(OperationResult::Test(sdm_contract::test_ops::TestResult {
        rows,
        data: Some(data),
        offset_seconds: None,
    }))
}

/// A text time reading's refusal, by its rule's name.
fn text_unread(unread: Unread) -> CoreError {
    match unread {
        Unread::Outside => not_yet("text_time"),
        Unread::Order => not_yet("text_time_order"),
        Unread::MixedZones => not_yet("text_time_zones"),
    }
}

/// `batch`, fetched from `table`, with each big-integer column in the tier
/// its column was described in, or refused by name (register P1B-43,
/// [`crate::big_integer::conform`]), and each column of numbers handed over
/// as their text in the type it was described as
/// ([`crate::exact_number::conform`]); and whether any column changed.
fn big_integers(table: &Table, batch: RecordBatch) -> Result<(RecordBatch, bool), CoreError> {
    let schema = batch.schema();
    let mut fields = Vec::with_capacity(schema.fields().len());
    let mut columns = Vec::with_capacity(schema.fields().len());
    let mut changed = false;
    for (field, array) in schema.fields().iter().zip(batch.columns()) {
        let conformed = match table.column(field.name()) {
            Some(described) => {
                let described = described.received.to_arrow();
                match crate::big_integer::conform(field, array, &described)? {
                    Some(conformed) => Some(conformed),
                    None => crate::exact_number::conform(field, array, &described)?,
                }
            }
            None => None,
        };
        match conformed {
            Some((field, array)) => {
                changed = true;
                fields.push(field);
                columns.push(array);
            }
            None => {
                fields.push(field.as_ref().clone());
                columns.push(Arc::clone(array));
            }
        }
    }
    if !changed {
        return Ok((batch, false));
    }
    let schema = arrow_schema::Schema::new_with_metadata(fields, schema.metadata().clone());
    RecordBatch::try_new(Arc::new(schema), columns)
        .map(|batch| (batch, true))
        .map_err(|_| internal("a fetch's big integers do not read back into its rows"))
}

/// One entity the caller gave, as a scalar a refusal names it by: an
/// integer, a float or text as itself, any other unnamed (null).
fn key_scalar(array: &dyn Array, row: usize) -> Scalar {
    use arrow_array::types::{
        Float64Type, Int8Type, Int16Type, Int32Type, Int64Type, UInt8Type, UInt16Type, UInt32Type,
        UInt64Type,
    };
    use arrow_schema::DataType;
    if array.is_null(row) {
        return Scalar::Null;
    }
    match array.data_type() {
        DataType::Int8 => Scalar::Int8(array.as_primitive::<Int8Type>().value(row)),
        DataType::Int16 => Scalar::Int16(array.as_primitive::<Int16Type>().value(row)),
        DataType::Int32 => Scalar::Int32(array.as_primitive::<Int32Type>().value(row)),
        DataType::Int64 => Scalar::Int64(array.as_primitive::<Int64Type>().value(row)),
        DataType::UInt8 => Scalar::UInt8(array.as_primitive::<UInt8Type>().value(row)),
        DataType::UInt16 => Scalar::UInt16(array.as_primitive::<UInt16Type>().value(row)),
        DataType::UInt32 => Scalar::UInt32(array.as_primitive::<UInt32Type>().value(row)),
        DataType::UInt64 => Scalar::UInt64(array.as_primitive::<UInt64Type>().value(row)),
        DataType::Utf8 => Scalar::Utf8(array.as_string::<i32>().value(row).to_owned()),
        DataType::LargeUtf8 => Scalar::Utf8(array.as_string::<i64>().value(row).to_owned()),
        DataType::Utf8View => Scalar::Utf8(array.as_string_view().value(row).to_owned()),
        DataType::Float64 => Scalar::Float64(array.as_primitive::<Float64Type>().value(row)),
        _ => Scalar::Null,
    }
}

#[cfg(test)]
mod tests {

    /// An aggregate count is one row of one integer: a second row is the
    /// provider's answer not reading, never its first row taken (T3.6
    /// round 1; a catalog listing's first row is its dialect's to read).
    #[test]
    fn an_aggregate_count_is_exactly_one_row() {
        use arrow_array::Int64Array;
        let data = |values: Vec<i64>| {
            let schema =
                std::sync::Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new(
                    "rows",
                    DataType::Int64,
                    false,
                )]));
            let batch = RecordBatch::try_new(
                std::sync::Arc::clone(&schema),
                vec![std::sync::Arc::new(Int64Array::from(values))],
            )
            .unwrap();
            ArrowData::new(schema, vec![batch])
        };
        assert_eq!(super::row_count(&data(vec![8])).unwrap(), 8);
        for answer in [data(vec![8, 9]), data(vec![])] {
            assert_eq!(
                super::row_count(&answer).unwrap_err().sentence,
                Sentence::ProviderSchemaMismatch
            );
        }
    }
    use super::*;
    use sdm_contract::ErrorCode;
    use sdm_contract::pql::TimeUnit;

    /// A filter over an unsigned 64-bit key whose values pass int64
    /// (`9223372036854775808::UBIGINT + order_id`): `order_id > 0` and
    /// `order_id != NULL` keep every row with a value, as the reference's
    /// executor and the filter before T2.10 keep them, never refused as a
    /// target's type (T2.10 fix 2, round 2's finding).
    #[test]
    fn a_filter_tests_an_unsigned_key_past_int64() {
        let past = 1_u64 << 63;
        let keys: ArrayRef = Arc::new(arrow_array::UInt64Array::from(vec![
            Some(past),
            None,
            Some(u64::MAX),
            Some(5),
        ]));
        let rows = RecordBatch::try_from_iter([("order_id", keys)]).unwrap();
        let filter = |op, constant| Shape::Condition {
            target: Box::new(Shape::Column("order_id".to_owned())),
            test: Test::Compare(op, constant),
        };
        let kept = |shape: &Shape| {
            labels::truth(&row_condition(shape, &rows).unwrap())
                .unwrap()
                .values
        };
        let compared = filter(labels::RelOp::Gt, Some(labels::Constant::Int(0)));
        assert_eq!(kept(&compared), [true, false, true, true]);
        let past_int64 = filter(
            labels::RelOp::Ge,
            Some(labels::Constant::Int(i128::from(past))),
        );
        assert_eq!(kept(&past_int64), [true, false, true, false]);
        let present = filter(labels::RelOp::Ne, None);
        assert_eq!(kept(&present), [true, false, true, true]);
        let float = filter(labels::RelOp::Lt, Some(labels::Constant::Float(1e19)));
        assert_eq!(kept(&float), [true, false, false, true]);
        let member = Shape::Condition {
            target: Box::new(Shape::Column("order_id".to_owned())),
            test: Test::In(vec![
                labels::Constant::Int(i128::from(u64::MAX)),
                labels::Constant::Int(5),
            ]),
        };
        assert_eq!(kept(&member), [false, false, true, true]);
    }

    /// A count over the last 30 days, compared as `op 0`.
    fn counted(op: labels::RelOp) -> Shape {
        Shape::Condition {
            target: Box::new(Shape::Aggregated(Windowed {
                aggregation: labels::Aggregation::Count,
                table: "orders".to_owned(),
                column: None,
                filter: None,
                start: Offset::of(0, TimeUnit::Days),
                end: Offset::of(30, TimeUnit::Days),
            })),
            test: Test::Compare(op, Some(labels::Constant::Int(0))),
        }
    }

    /// The lags of `NOT (COUNT(...) > 0)` are those of `COUNT(...) > 0`, as
    /// the reference's `to_autoregressive_query` drops the whole target's
    /// `NOT` (T2.10, measured against the reference: the lagged columns of
    /// the two queries are the same); a `NOT` inside the target stays.
    #[test]
    fn the_lags_of_a_negated_target_are_its_condition_s() {
        let negated = Shape::Not(Box::new(counted(labels::RelOp::Gt)));
        assert!(matches!(
            autoregressive(&negated),
            Shape::Condition {
                test: Test::Compare(labels::RelOp::Gt, _),
                ..
            }
        ));
        let inside = Shape::Both {
            op: labels::BoolOp::And,
            left: Box::new(negated),
            right: Box::new(counted(labels::RelOp::Ge)),
        };
        match autoregressive(&inside) {
            Shape::Both { left, .. } => assert!(matches!(*left, Shape::Not(_))),
            other => panic!("not kept whole: {other:?}"),
        }
        assert!(matches!(
            autoregressive(&counted(labels::RelOp::Gt)),
            Shape::Condition { .. }
        ));
    }

    /// The pools the reference's `min_date_offset` and `max_date_offset`
    /// give, read from it at 6298913c: one offset where it bounds the other
    /// at every anchor, the first of two that bound each other; a rigid
    /// span of seconds where neither does.
    #[test]
    fn a_table_s_windows_pool_as_the_reference_pools_them() {
        let of = Offset::of;
        assert_eq!(
            pooled(of(30, TimeUnit::Days), of(1, TimeUnit::Months), false),
            Offset {
                count: 2_678_400,
                unit: TimeUnit::Seconds
            }
        );
        assert_eq!(
            pooled(of(0, TimeUnit::Days), of(0, TimeUnit::Months), true),
            of(0, TimeUnit::Days)
        );
        assert_eq!(
            pooled(of(7, TimeUnit::Days), of(1, TimeUnit::Weeks), false),
            of(7, TimeUnit::Days)
        );
        assert_eq!(
            pooled(of(35, TimeUnit::Days), of(1, TimeUnit::Months), false),
            of(35, TimeUnit::Days)
        );
        assert_eq!(
            pooled(of(5, TimeUnit::Days), of(1, TimeUnit::Days), true),
            of(1, TimeUnit::Days)
        );
        assert_eq!(
            pooled(of(48, TimeUnit::Hours), of(1, TimeUnit::Days), false),
            of(48, TimeUnit::Hours)
        );
    }

    #[test]
    fn a_date_literal_reads_as_its_wall_clock_in_nanoseconds() {
        assert_eq!(
            time_constant("2024-01-04T06:00:00", None).unwrap(),
            1_704_348_000_000_000_000
        );
        assert_eq!(
            time_constant("2024-01-04T06:00:00", Some(3_600))
                .unwrap_err()
                .code,
            ErrorCode::InvalidRequest
        );
        // Past what a timestamp of nanoseconds holds: the caller's error.
        assert_eq!(
            time_constant("2263-01-01T00:00:00", None).unwrap_err().code,
            ErrorCode::InvalidRequest
        );
    }
}
