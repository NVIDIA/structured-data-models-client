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
use sdm_contract::catalog::{CatalogAnswer, IndexColumns};
use sdm_contract::data::{
    AttachmentId, BATCH_COLUMN, DataRequest, DescribePart, Filter, IndexAdvice, KeySet,
    MATCHES_COLUMN, NullOrder, OptionalRequest, ProviderChoice, SEED_STORED_TIME, TableColumn,
    TextTimes,
};
use sdm_contract::ops::{
    ColumnRef, ForecastStep, Materialized, MemoBounds, MemoStoredTime, OperationResult,
    ProfileName, ProfileSpec, QueryColumn, QueryFacts, QuerySchema, QueryTable, RequestBatch,
    RequestState, SamplerMemo, Seed, SourceDescription,
};
use sdm_contract::ops::{Dtype, Stype};
use sdm_contract::pql::{Aggregation, ConditionOp, NodeKind, PqlLiteral, PqlNode, ValidatedQuery};
use sdm_contract::{
    ColumnType, CoreError, CoreWarning, Detail, ErrorCode, ErrorKind, Field, HostFacts, Scalar,
    Unit,
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
use sdm_request::options::{self, AskedSeed, Kind as Task};
use sdm_request::payload::{self, RESERVED, Sampled};

use crate::describe::{ask, describe};
use crate::profiles::Profile;
use crate::run::{ArrowData, Asker, Attachment, internal};

/// The warning that indices the sampler's reads would use are missing.
pub const MISSING_INDICES: &str = "MISSING_INDICES";
/// The warning that the context reaches back before the data.
pub const CONTEXT_BEFORE_DATA: &str = "CONTEXT_BEFORE_DATA";
/// The warning that the anchor time is after the data.
pub const ANCHOR_AFTER_DATA: &str = "ANCHOR_AFTER_DATA";
/// The warning that a large entity table gave fewer than half the context
/// examples asked for, which the reference gives once in a process
/// (`_coverage_warned`).
pub const CONTEXT_COVERAGE: &str = "CONTEXT_COVERAGE";

/// The warnings the reference gives once in a process: a host gives each
/// once in its process (decision E52), `sdm-host` for every binding, so
/// the core, which holds no state across runs, gives it on every run that
/// meets it.
pub const ONCE_PER_PROCESS: &[&str] = &[CONTEXT_COVERAGE];

/// The sampler's warnings, each with the variants it gives (none: one
/// sentence), which a host words (decision E59(d)).
pub const WARNINGS: &[(&str, &[&str])] = &[
    (
        MISSING_INDICES,
        &[
            "index_table",
            "index_tables",
            "indices_table",
            "indices_tables",
        ],
    ),
    (CONTEXT_BEFORE_DATA, &[]),
    (ANCHOR_AFTER_DATA, &[]),
    (CONTEXT_COVERAGE, &[]),
    (CONTEXT_CAPPED, &[]),
    (TABLES_UNLINKED, &["one", "several"]),
    (TABLES_UNREACHED, &["one", "several"]),
    (COLUMNS_UNREAD, &["one", "several"]),
    (CONTEXT_ANSWERS, &["windowed", "static"]),
    (crate::subgraph::NULL_KEYS_LEFT_OUT, &[]),
];
/// The notice that the default run mode gathered as many context examples
/// as it takes (the client's `_warn_of_a_context_the_default_mode_capped`).
pub const CONTEXT_CAPPED: &str = "CONTEXT_CAPPED";
/// The notice that tables of the schema no relationship connects to the
/// entity table are not in the request (`_warn_of_tables_left_out`).
pub const TABLES_UNLINKED: &str = "TABLES_UNLINKED";
/// The notice that tables linked into the schema were reached by no row
/// the sampler read (`_warn_of_tables_left_out`).
pub const TABLES_UNREACHED: &str = "TABLES_UNREACHED";
/// The notice that columns the request sends are of a meaning the model
/// does not read (`_warn_of_columns_the_model_does_not_read`).
pub const COLUMNS_UNREAD: &str = "COLUMNS_UNREAD";
/// The notice that context examples answering the question were taken out
/// (`_drop_context_that_answers_the_question`).
pub const CONTEXT_ANSWERS: &str = "CONTEXT_ANSWERS";
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
/// neither).
pub fn check_memo(memo: &SamplerMemo) -> Result<(), CoreError> {
    let mut tables = BTreeSet::new();
    let sound = memo.bounds.iter().all(|b| {
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
        "the sampler state is not one a run left",
        "sampler_state",
    ))
}

fn invalid(message: &str, field: &str) -> CoreError {
    CoreError::new(
        ErrorKind::InvalidRequest,
        ErrorCode::InvalidRequest,
        message,
    )
    .with_field("field", Detail::text(field))
}

fn count(n: usize) -> Detail {
    Detail::Scalar(Scalar::UInt64(u64::try_from(n).unwrap_or(u64::MAX)))
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
    if query.assuming.is_some() {
        return Err(not_yet("assuming"));
    }
    let entity = column_of(&query.entity).ok_or_else(|| not_yet("entity_filter"))?;
    let target = shape(&query.target, &entity.table)?;
    let task = match &target {
        Shape::Condition { .. } | Shape::Not(_) | Shape::Both { .. } => Task::Binary,
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
                // link prediction is Phase 4's (design, "Explicitly out"),
                // refused by name until then.
                Aggregation::ListDistinct => return Err(not_yet("aggregation")),
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
    let refused = |message: &str| {
        CoreError::new(
            ErrorKind::InvalidRequest,
            ErrorCode::InvalidRequest,
            message,
        )
        .with_field("field", Detail::text("query"))
    };
    if offset.is_some() {
        return Err(refused(
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
        .ok_or_else(|| refused("a time constant past what a timestamp of nanoseconds holds"))
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
/// that table's first windowed read and kept in the sampler's memo across
/// the materializer's runs (decision E52), as the reference's SQLite
/// sampler reads one per table and keeps it (`_time_spelling`), and names
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
    /// The reading of a run's times, holding the stored values `memo` kept
    /// from the materializer's earlier runs.
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
                        code: sdm_schema::assemble::TIME_VALUES_UNREAD.into(),
                        message: "a time column's values were read as no time and dropped".into(),
                        fields: [
                            ("table".to_owned(), Detail::text(&table.name)),
                            ("column".to_owned(), Detail::text(name)),
                            ("unread".to_owned(), count(unread)),
                            ("rows".to_owned(), count(values.len())),
                            ("examples".to_owned(), Detail::List(examples)),
                        ]
                        .into(),
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
            code: sdm_schema::assemble::TIME_VALUES_UNREAD.into(),
            message: "a time of day's values were read as no time and dropped".into(),
            fields: [
                ("table".to_owned(), Detail::text(&table.name)),
                ("column".to_owned(), Detail::text(name)),
                ("unread".to_owned(), count(unread)),
                ("rows".to_owned(), count(array.len())),
                ("examples".to_owned(), Detail::List(examples)),
            ]
            .into(),
        });
    }
}

/// The batch positions a `neighbors` or `lookup` answer names.
pub(crate) fn batches(batch: &RecordBatch, seeds: usize) -> Result<Vec<usize>, CoreError> {
    let column = named(batch, BATCH_COLUMN)?;
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
/// [`static_column`] reads a target, and a time too, which the reads have
/// made a zone-less timestamp ([`Times::read`]), in nanoseconds.
fn condition_column(array: &ArrayRef, rows: &[usize]) -> Result<labels::Column, CoreError> {
    if let DataType::Timestamp(_, None) = array.data_type() {
        let all = times_ns(array, "")?;
        return Ok(labels::Column::Time(rows.iter().map(|&r| all[r]).collect()));
    }
    static_column(array, rows)
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
/// where they are new, so the materializer's next run reads none again.
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
/// request, and kept.
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
        let column = named(&batch, BOUNDS_POSITION)?;
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
    let mins = times.bounds(named(&batch, BOUNDS_MIN)?, BOUNDS_MIN, true)?;
    let maxs = times.bounds(named(&batch, BOUNDS_MAX)?, BOUNDS_MAX, false)?;
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
        if !capabilities.optional.contains(&needed) {
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
    let variant = match (indexes == 1, tables == 1) {
        (true, true) => "index_table",
        (true, false) => "index_tables",
        (false, true) => "indices_table",
        (false, false) => "indices_tables",
    };
    asker.warn(CoreWarning {
        code: MISSING_INDICES.into(),
        message: "indices the sampler's reads would use are missing".into(),
        fields: [
            ("indices".to_owned(), count(indexes)),
            ("tables".to_owned(), count(tables)),
            ("variant".to_owned(), Detail::text(variant)),
        ]
        .into(),
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
                IndexColumns::Resolved(columns) => columns.into_iter().next(),
                IndexColumns::Unresolved => None,
                IndexColumns::Named(name) => {
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
        let IndexColumns::Resolved(columns) = &index.columns else {
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

fn warn(asker: &Asker, code: &str, message: &str) {
    asker.warn(CoreWarning {
        code: code.to_owned(),
        message: message.to_owned(),
        fields: Default::default(),
    });
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
fn consistent(description: &mut SourceDescription) -> Result<(), CoreError> {
    sdm_schema::assemble::mark_keys(description);
    for table in &description.tables {
        for column in &table.columns {
            // As the language answers it, as an edit's meaning is held.
            let supported = crate::edit::supports(column.stype, column.dtype);
            if !supported {
                return Err(invalid(
                    "a column's meaning is not one its values can carry",
                    "schema",
                )
                .with_field("table", Detail::text(&table.name))
                .with_field("column", Detail::text(&column.name)));
            }
        }
        // The engine's table takes a time column only of the timestamp
        // meaning (`Column.__setattr__`): one the schema joins on keeps
        // its identifier's, and the reference refuses the request.
        if let Some(time) = &table.time_column
            && table
                .columns
                .iter()
                .any(|c| c.name == *time && c.stype != Stype::Timestamp)
        {
            return Err(
                invalid("a table's time column does not mean a time", "time_column")
                    .with_field("table", Detail::text(&table.name))
                    .with_field("column", Detail::text(time)),
            );
        }
    }
    sdm_schema::assemble::check_links(description)
}

/// The refusal of one of the reference's sampling options this message
/// carries by name only (`Materialize::options`), which this core takes
/// none of yet, by its own name; a name that is not one of the
/// reference's options is the host's fault, `INVALID_REQUEST`.
fn option_not_taken(name: &str) -> CoreError {
    match name {
        "context_anchor_time" => not_yet("context_anchor_time"),
        "exclude_cols_dict" => not_yet("exclude_cols_dict"),
        "max_pq_iterations" => not_yet("max_pq_iterations"),
        "num_hops" => not_yet("num_hops"),
        "num_neighbors" => not_yet("num_neighbors"),
        "return_embeddings" => not_yet("return_embeddings"),
        "use_prediction_time" => not_yet("use_prediction_time"),
        _ => invalid(
            "an option named is not one of the sampling options",
            "options",
        ),
    }
}

/// The call's options this core does not take yet, each refused by its
/// name before anything else of the call is read, as the host's bridge
/// refused them before the core ran until T1.24 (decision E61): the other
/// sampling options, the first in name order; an unseeded sample; and an
/// anchor at each entity's own time; the hops they ask bounded first
/// ([`options::hops`]).
fn options_taken(inputs: &Inputs, unseeded: bool) -> Result<(), CoreError> {
    if let Some(first) = inputs.options.iter().min() {
        return Err(option_not_taken(first));
    }
    if unseeded {
        return Err(not_yet("unseeded"));
    }
    if inputs.anchor_entity {
        return Err(not_yet("anchor_entity"));
    }
    Ok(())
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
const FOLDED_PREFIX: &str = "__kumo_key_";
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
    let refused = |table: &str, key: &[String], message: &str| {
        invalid(message, "schema")
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
                &table.name,
                key,
                "a float cannot be part of a key of several columns: its text is not reproducible",
            ));
        }
        let name = format!(
            "{FOLDED_PREFIX}{}",
            key.iter()
                .map(|c| c.bytes().map(|b| format!("{b:02x}")).collect::<String>())
                .collect::<Vec<_>>()
                .join("_")
        );
        if name.len() > FOLDED_NAME_MOST {
            return Err(refused(
                &table.name,
                key,
                "a key of several columns needs a folded column's name longer than a source takes",
            ));
        }
        if table.columns.iter().any(|c| c.name == name) {
            return Err(refused(
                &table.name,
                key,
                "a column holds the name a key of several columns is folded into",
            ));
        }
        first.get_or_insert((table.name.clone(), key.to_vec()));
    }
    match first {
        None => Ok(()),
        Some((table, key)) => Err(refused(
            &table,
            &key,
            "a table keyed by several columns cannot be sampled: the SQL samplers fold its key into a column of the name they reserve",
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
    Err(CoreError::new(
        ErrorKind::InvalidRequest,
        ErrorCode::InvalidSchema,
        "a column of the schema has a name the request needs for its own",
    )
    .with_field("tables", list(taken.iter().map(|(t, _)| *t).collect()))
    .with_field("columns", list(taken.iter().map(|(_, c)| *c).collect())))
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
            code: TABLES_UNLINKED.into(),
            message: "tables of the schema no relationship connects are not in the request".into(),
            fields: [
                ("variant".to_owned(), Detail::text(several(unlinked.len()))),
                ("tables".to_owned(), names(unlinked.iter().copied())),
                (
                    "entity_table".to_owned(),
                    Detail::text(asked.entity_table.as_str()),
                ),
            ]
            .into(),
        });
    }
    if !unreached.is_empty() {
        asker.warn(CoreWarning {
            code: TABLES_UNREACHED.into(),
            message: "tables linked into the schema were reached by no row sampled".into(),
            fields: [
                ("variant".to_owned(), Detail::text(several(unreached.len()))),
                ("tables".to_owned(), names(unreached.iter().copied())),
            ]
            .into(),
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
            code: COLUMNS_UNREAD.into(),
            message: "columns the request sends are of a meaning the model does not read".into(),
            fields: [
                ("variant".to_owned(), Detail::text(several(unread.len()))),
                (
                    "columns".to_owned(),
                    names(unread.iter().map(String::as_str)),
                ),
                ("stypes".to_owned(), names(["text"])),
            ]
            .into(),
        });
    }
    if answering > 0 {
        asker.warn(CoreWarning {
            code: CONTEXT_ANSWERS.into(),
            message: "context examples answering the question were taken out".into(),
            fields: [
                (
                    "variant".to_owned(),
                    Detail::text(if windowed { "windowed" } else { "static" }),
                ),
                ("examples".to_owned(), count(answering)),
            ]
            .into(),
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
            ErrorKind::InvalidRequest,
            ErrorCode::InvalidRequest,
            "top_k is a ranking's, and a forecast ranks nothing",
        )
        .with_field("field", Detail::text("top_k"))
        .with_field("rule", Detail::text("forecast_top_k")));
    }
    // The host's limit is checked against the plan's count before any step
    // is written, so nothing grows with a count past it (T1.36 fix 23).
    let most = inputs.most_parts.unwrap_or(MOST_PARTS);
    if steps.count > u64::from(most) {
        return Err(CoreError::new(
            ErrorKind::InvalidRequest,
            ErrorCode::InvalidRequest,
            "a forecast asks for more steps than the host's limit, one call each",
        )
        .with_field("field", Detail::text("query"))
        .with_field("rule", Detail::text("too_many_parts"))
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
            ErrorKind::Unsupported,
            ErrorCode::UnsupportedRequest,
            "a forecast over more entities than one call scores would batch each step",
        )
        .with_field("field", Detail::text("indices"))
        .with_field("rule", Detail::text("batched_part"))
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
        let tagged =
            |e: CoreError| e.with_field("forecast_step", Detail::Scalar(Scalar::UInt32(step)));
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
        ErrorKind::Unsupported,
        ErrorCode::UnsupportedRequest,
        "more entities were asked for than one materialize holds at once",
    )
    .with_field("field", Detail::text("indices"))
    .with_field("rule", Detail::text("most_entities"))
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
    if consistent(&mut description).is_err() {
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
    let mode = options::run_mode(asked_values.num_context_samples)?;
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
    options_taken(&inputs, unseeded)?;
    if seed.as_ref().is_some_and(Seed::is_negative) {
        return Err(CoreError::new(
            ErrorKind::InvalidRequest,
            ErrorCode::InvalidRequest,
            "a sampling seed cannot be negative",
        )
        .with_field("field", Detail::text("seed")));
    }
    #[cfg(feature = "test-ops")]
    let labels_only = inputs.labels_only;
    let Inputs {
        mut description,
        query,
        indices,
        anchor_time,
        facts,
        zones,
        most_entities,
        most_calls,
        ..
    } = inputs;
    let options::Asked {
        num_context_samples,
        prediction_statistic,
        inference_config,
        top_k,
        rows_per_call,
        lag_timesteps,
        ..
    } = asked_values;
    // The most context examples asked for, which narrows the context the
    // mode gathers (`_narrow_context`); the bound above holds it to the
    // best mode's.
    let asked_examples = num_context_samples
        .and_then(|n| n.magnitude)
        .and_then(|n| usize::try_from(n).ok());
    // A table the schema names otherwise than the source does is read by
    // its source name (`Table::source`), as the reference's engine reads
    // it; the query and the payload name it as the schema does.
    consistent(&mut description)?;
    // The seed reaches this entry as the integer the caller gave, of any
    // size (decision H7); numpy's generator takes it whole, and the sample
    // takes it as the narrowest exact scalar.
    let seed = seed.unwrap_or_else(|| Seed::from(DEFAULT_SEED));
    let entropy = Entropy::from_hex(seed.magnitude_hex())
        .ok_or_else(|| internal("a seed's digits do not read"))?;

    let indices = indices.ok_or_else(|| {
        invalid(
            "materialize names the entities to predict for, and none were given",
            "indices",
        )
    })?;
    let indices = one_batch(&indices)?;
    if indices.num_columns() != 1 {
        return Err(invalid(
            "the entities to predict for are one column",
            "indices",
        ));
    }
    let indices = Arc::clone(indices.column(0));
    if indices.is_empty() {
        return Err(invalid("the entities to predict for are none", "indices"));
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
            return Err(invalid(
                "the query does not resolve against the schema",
                "query",
            ));
        }
        refuse_composite_keys(&description)?;
    }

    let graph = Graph::new(&description)?;
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
    // D4's warning is validate_query's: a public call validates its query
    // before it materializes it (`PredictiveQuery._resolve`), so this run,
    // which reads the same literals, does not give it again.
    let (Some(validated), Some(source_query)) = (validation.query, validation.text) else {
        return Err(invalid(
            "the query does not resolve against the schema",
            "query",
        ));
    };
    // What the sampler needs of its provider, refused before anything is
    // fetched.
    let nulls = sampling_capabilities(asker)?;
    // The index read, the sampler's check of the indexes it would use
    // (only under the provider's index advice, once for a materializer),
    // comes before anything of the task is read, as the reference builds
    // its sampler, and gives its warning, before its task table. The checks
    // after it (null entities, then the seed, then the task's shape) are in
    // this core's own order, not the reference's: an input with several
    // errors may be refused for another of them than the reference's first
    // (decision E41(k)(i)).
    missing_indexes(&mut memo, &graph, asker.capabilities()?.index_advice).await?;
    if keys::missing(&indices) {
        return Err(invalid("an entity to predict for is missing", "indices"));
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
                asker.capabilities()?.max_seed,
            ));
        }
        Err(err) => return Err(err),
    };
    let asked = asked(&validated)?;
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
    let mut windows = Vec::new();
    asked.target.windowed(&mut windows);
    let fetches = fetches(&graph, &asked.entity_table, &windows)?;
    // The subgraph is walked as the reference walks it from the entity
    // table (`subgraph`); an entity table referring to another is not
    // sampled yet.
    if !graph.foreign_keys(&entity.name).is_empty() {
        return Err(not_yet("parent_table"));
    }

    // The times every fetch below returns, read as the reference's engine
    // reads them.
    let read_times = Times::new(
        crate::describe::text_reader(&facts)?,
        asker.capabilities()?.text_times,
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

    // The anchor, and the times it is checked against.
    let mut bounds: Option<Bounds> = None;
    let anchor = match anchor_time {
        None if bounded.is_empty() => TIME_MIN,
        None => {
            let read = time_bounds(&mut memo, &read_times, &bounded).await?;
            let max = read.max;
            bounds = Some(read);
            max
        }
        // A zoned anchor counts its UTC instant, the zone only how it is
        // shown: read at its UTC wall clock, as the reference's engine
        // reads one (`to_naive_utc`).
        Some(Scalar::Timestamp { unit, value, .. }) => match unit {
            Unit::Ns => value,
            Unit::Us => value
                .checked_mul(1_000)
                .ok_or_else(|| sample::out_of_range("anchor_time"))?,
        },
        Some(_) => {
            return Err(invalid("an anchor time is a ts scalar", "anchor_time"));
        }
    };
    // The context's anchor: a window's end before the anchor, by the
    // calendar for months; a static query's is the anchor itself.
    let context_anchor = match &span {
        Some((_, end)) => end.negated().add_to(anchor, "anchor_time")?,
        None => anchor,
    };
    let bounds = match bounds {
        Some(bounds) => Some(bounds),
        None if bounded.is_empty() => None,
        None => Some(time_bounds(&mut memo, &read_times, &bounded).await?),
    };
    if let Some(bounds) = &bounds {
        if anchor < bounds.min {
            return Err(invalid(
                "the anchor time is before the earliest time in the data",
                "anchor_time",
            ));
        }
        if context_anchor < bounds.min {
            return Err(invalid(
                "the context's anchor time is before the earliest time in the data",
                "anchor_time",
            ));
        }
        // A static query's window is empty: its context reaches back no
        // further than its anchor, which was checked above.
        if let Some((_, end)) = &span
            && reaches(end.negated(), context_anchor)? < i128::from(bounds.min)
        {
            warn(
                asker,
                CONTEXT_BEFORE_DATA,
                "the context reaches back before the data",
            );
        }
        if i128::from(anchor) > i128::from(bounds.max) + DAY_NS {
            warn(
                asker,
                ANCHOR_AFTER_DATA,
                "the anchor time is after the data",
            );
        }
    }

    // The examples: a reservoir sample of the entity table, kept one row
    // per entity, with the entity table's columns the query reads; for a
    // temporal query, walked back from the context's anchor by the window.
    let context_examples = mode.context_examples();
    let trials = ITERATIONS * context_examples;
    let mut entity_columns = BTreeSet::from([key.clone()]);
    if let Some(time) = &entity.time_column {
        entity_columns.insert(time.clone());
    }
    asked.target.entity_columns(&mut entity_columns);
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
    let sampled = read_times.read(
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
                },
            )
            .await?,
        )?,
    )?;
    if sampled.num_rows() == 0 {
        return Err(invalid("the entity table holds no row to sample", "query"));
    }
    let sampled_keys = keys::read(named(&sampled, &key)?)?;
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
            bounds.as_ref().map_or(TIME_MAX, |b| b.min),
            trials,
        )?,
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
        && !fetches.is_empty()
        && examples.index.iter().any(|i| nan_key.contains(i))
    {
        return Err(invalid(
            "an entity sampled has no value for its table's primary key",
            "query",
        )
        .with_field("table", Detail::text(&entity.name))
        .with_field("column", Detail::text(&key)));
    }
    // Their labels, as the reference's executor computes them over the
    // rows read for them (decision E19).
    let entity_rows: Vec<usize> = examples.index.iter().map(|&i| kept[i]).collect();
    let target = Labelling {
        asker,
        times: &read_times,
        sampled: &sampled,
        entity_rows: &entity_rows,
        entity_key: &entity_key,
        examples: &examples,
    }
    .target(&asked.target, &fetches)
    .await?;
    // A classification keeps a decimal column's distinct values apart as
    // classes; any other task reads them as numbers (T2.0c, round 1).
    let mut labelled = if matches!(asked.task, Task::Multiclass) {
        labels::label_classes(&target)?
    } else {
        labels::label(&target)?
    };
    null_compared(&asked.target, &labelled.mask)?;
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
        return Err(invalid("no context example was collected", "anchor_time"));
    }
    if kept.len() >= trials && gathered < context_examples / 2 {
        // The reference warns of it once in a process: given on every run
        // that meets it, and once in a process by the host
        // ([`ONCE_PER_PROCESS`], decision E52).
        asker.warn(CoreWarning {
            code: CONTEXT_COVERAGE.into(),
            message: "fewer than half the context examples asked for were collected".into(),
            fields: [
                ("examples".to_owned(), count(context_examples)),
                ("candidates".to_owned(), count(trials)),
            ]
            .into(),
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
        Some(asked) => asked.refusal(!fetches.is_empty())?,
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
            code: CONTEXT_CAPPED.into(),
            message: "the default run mode gathered as many context examples as it takes".into(),
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
    let per_call = usize::try_from(options::rows_one_call_scores(rows_per_call))
        .map_err(|_| internal("a call's rows past usize"))?;
    let num_test = indices.len();
    let calls = num_test.div_ceil(per_call);
    let most = most_calls.unwrap_or(MOST_CALLS);
    if calls > most as usize {
        return Err(CoreError::new(
            ErrorKind::Unsupported,
            ErrorCode::UnsupportedRequest,
            "the entities asked for are gathered as more calls than one materialize holds at once",
        )
        .with_field("field", Detail::text("indices"))
        .with_field("rule", Detail::text("most_calls"))
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
    let excluded: Vec<(String, String)> = excluded_columns
        .into_iter()
        .map(|column| (entity.name.clone(), column))
        .collect();
    for (_, column) in &excluded {
        if !graph.features(entity).iter().any(|(n, _)| n == column) {
            return Err(invalid(
                "the column a static target reads is not one of its table's features",
                "query",
            ));
        }
    }
    if num_train > TABLE_ROWS {
        return Err(invalid(
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
        let looked_up = read_times.read(
            asker,
            entity,
            one_batch(
                &ask(
                    asker,
                    sample::by_pkey(
                        entity,
                        entity_names.clone(),
                        key_set(asker, sample::seeds(all_keys, vec![])?)?,
                        true,
                    )?,
                )
                .await?,
            )?,
        )?;
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
        for (row, &at) in positions.iter().enumerate() {
            if at >= num_train && matches.is_some_and(|m| m.value(row) > 1) {
                return Err(invalid(
                    "an entity to predict for is held by more than one row, so which is meant \
                     would be a guess",
                    "indices",
                ));
            }
        }
        if positions.len() != examples_total {
            return Err(invalid(
                "an entity to predict for is not in the entity table",
                "indices",
            ));
        }
        let mut order: Vec<usize> = (0..positions.len()).collect();
        order.sort_by_key(|&row| positions[row]);
        let entity_rows = take_batch(&looked_up, &order)?;
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
        let walked = crate::subgraph::Walk {
            asker,
            graph: &graph,
            times: &read_times,
            anchors: &anchors,
            nulls,
            entity,
            neighbors: mode.neighbors(),
            excluded: &excluded,
        }
        .run(entity_rows)
        .await?;
        keep_stored_times(&mut memo, &read_times);
        // The subgraph carries at most as many tables as the endpoint
        // reads, refused as the reference's engine refuses it
        // (`_get_context`).
        if walked.tables.len() > SUBGRAPH_TABLES {
            return Err(invalid(
                "the request's subgraph carries more tables than the endpoint reads",
                "tables",
            )
            .with_field("tables", count(walked.tables.len()))
            .with_field("limit", count(SUBGRAPH_TABLES)));
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
            source_query: source_query.clone(),
            run_mode: mode,
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
        labels::Aggregation::Sum | labels::Aggregation::Count => return Ok(()),
    };
    let batch = &mask[..mask.len().min(LABEL_BATCH)];
    if batch.is_empty() {
        return Ok(());
    }
    if !batch.contains(&true) {
        return Err(invalid(
            "no example's window holds a value to compare with NULL",
            "query",
        ));
    }
    if batch.contains(&false) {
        return Err(not_yet(rule));
    }
    Ok(())
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
    /// The target `shape` names, over the examples: each table in `fetches`
    /// read first, in order, then the target put together from the rows.
    async fn target(
        &self,
        shape: &Shape,
        fetches: &[Fetch<'_>],
    ) -> Result<labels::Target, CoreError> {
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
        self.build(shape, &fetched, true)
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
                labels::Target::Column(if whole {
                    static_column(array, self.entity_rows)?
                } else {
                    condition_column(array, self.entity_rows)?
                })
            }
            Shape::Aggregated(windowed) => labels::Target::Aggregated(aggregate(
                windowed,
                fetched
                    .get(&windowed.table)
                    .ok_or_else(|| internal("an aggregation's table was not read"))?,
                &self.examples.anchors,
                self.examples.index.len(),
                "anchor_time",
            )?),
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
        Shape::Aggregated(windowed) => labels::Target::Aggregated(aggregate(
            windowed,
            fetched
                .get(&windowed.table)
                .ok_or_else(|| internal("an aggregation's table was not read"))?,
            anchors,
            examples,
            field,
        )?),
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
fn aggregate(
    windowed: &Windowed,
    fetched: &Fetched,
    anchors: &[i64],
    examples: usize,
    field: &str,
) -> Result<labels::Aggregated, CoreError> {
    let ends = shifted(anchors, windowed.end, field)?;
    let starts = shifted(anchors, windowed.start, field)?;
    let values = match &windowed.column {
        Some(column) => label_values(named(&fetched.rows, column)?)?,
        None => labels::Values::Ones(fetched.rows.num_rows()),
    };
    let kept = match &windowed.filter {
        None => None,
        Some(filter) => Some(labels::truth(&row_condition(filter, &fetched.rows)?)?.values),
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

#[cfg(test)]
mod tests {
    use super::*;
    use sdm_contract::pql::TimeUnit;

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
