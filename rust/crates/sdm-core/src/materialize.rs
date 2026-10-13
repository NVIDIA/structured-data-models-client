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
//! (T1.26a); the time bounds of the aggregated table, or of every table
//! with a time column for a static query; the reservoir sample of the
//! entity table (`sample_entities`), with the entity table's columns the
//! query reads; the windowed rows each example's label aggregates
//! (`neighbors` with a window, `_by_time`);
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

use std::collections::BTreeSet;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{
    Date32Type, Decimal128Type, Float32Type, Float64Type, Int8Type, Int16Type, Int32Type,
    Int64Type, TimestampMicrosecondType, TimestampMillisecondType, TimestampNanosecondType,
    TimestampSecondType, UInt8Type, UInt16Type, UInt32Type, UInt64Type,
};
use arrow_array::{Array, ArrayRef, RecordBatch, TimestampNanosecondArray, UInt32Array};
use arrow_schema::{DataType, TimeUnit as ArrowUnit};
use sdm_contract::catalog::{CatalogAnswer, IndexColumns};
use sdm_contract::data::{
    AttachmentId, BATCH_COLUMN, DataRequest, DescribePart, Filter, IndexAdvice, KeySet,
    MATCHES_COLUMN, NullOrder, OptionalRequest, ProviderChoice, TableColumn, TextTimes,
};
use sdm_contract::ops::Stype;
use sdm_contract::ops::{
    ColumnRef, Materialized, MemoBounds, OperationResult, ProfileName, ProfileSpec, QueryColumn,
    QuerySchema, QueryTable, RequestState, SamplerMemo, Seed, SourceDescription,
};
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
/// The most entities one request predicts for.
const PREDICT_ROWS: usize = 1_000;
/// The most rows an instance table may hold (`MAX_TABLE_ROWS`).
const TABLE_ROWS: usize = 10_000;
/// `pd.Timestamp.max` and `min`, in nanoseconds: the bounds of a table
/// whose time column holds no value.
const TIME_MAX: i64 = i64::MAX;
const TIME_MIN: i64 = -i64::MAX;
const DAY_NS: i128 = 86_400_000_000_000;
/// The version of [`RequestState`]'s format this build writes: the
/// contract's, which `predict` reads (decision H23).
pub use sdm_contract::ops::REQUEST_STATE_VERSION;

/// What a `materialize` start gives its body.
pub(crate) struct Inputs {
    pub description: SourceDescription,
    pub query: String,
    pub indices: Option<ArrowData>,
    pub anchor_time: Option<Scalar>,
    pub seed: Option<Seed>,
    pub unseeded: bool,
    pub anchor_entity: bool,
    /// The call's option values (`Materialize.values`), as read.
    pub values: Option<sdm_wire::json::Value>,
    pub options: Vec<String>,
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
    if sound {
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
    /// The window's ends, as offsets from each example's anchor.
    start: Offset,
    end: Offset,
}

/// What a query predicts, in the shapes this build labels.
#[derive(Debug, Clone)]
enum Shape {
    /// A column of the entity's own row: a static target.
    Column(String),
    Aggregated(Windowed),
    /// `target op constant`; no constant is `NULL`.
    Condition {
        target: Box<Shape>,
        op: labels::RelOp,
        constant: Option<labels::Constant>,
    },
}

impl Shape {
    /// The aggregation the target takes, if it takes one: a query with one
    /// is temporal (`QueryType.TEMPORAL`), one without static.
    fn windowed(&self) -> Option<&Windowed> {
        match self {
            Shape::Column(_) => None,
            Shape::Aggregated(windowed) => Some(windowed),
            Shape::Condition { target, .. } => target.windowed(),
        }
    }

    /// The entity table's columns the target reads.
    fn entity_columns(&self, into: &mut BTreeSet<String>) {
        match self {
            Shape::Column(column) => {
                into.insert(column.clone());
            }
            Shape::Aggregated(_) => {}
            Shape::Condition { target, .. } => target.entity_columns(into),
        }
    }
}

/// The parts of a validated query this build materializes.
struct Asked {
    entity_table: String,
    entity_column: String,
    target: Shape,
    task: Task,
}

fn column_of(node: &PqlNode) -> Option<&ColumnRef> {
    match &node.kind {
        NodeKind::Column { column } => Some(column),
        _ => None,
    }
}

/// The query's shape, or the rule it takes that this build does not port.
fn asked(query: &ValidatedQuery) -> Result<Asked, CoreError> {
    if query.evaluate {
        return Err(not_yet("evaluate"));
    }
    if query.explain {
        return Err(not_yet("explain"));
    }
    if query.problem_type.is_some() || query.num_forecasts != 1 || query.top_k.is_some() {
        return Err(not_yet("problem_type"));
    }
    if query.assuming.is_some() {
        return Err(not_yet("assuming"));
    }
    if query.entity_ids.is_some() {
        return Err(not_yet("entity_ids"));
    }
    let entity = column_of(&query.entity).ok_or_else(|| not_yet("entity_filter"))?;
    let target = shape(&query.target, &entity.table)?;
    let task = match &target {
        Shape::Condition { .. } => Task::Binary,
        Shape::Aggregated(_) => Task::Regression,
        Shape::Column(_) => match query.target.stype {
            Some(Stype::Id | Stype::Categorical) => Task::Multiclass,
            Some(Stype::Numerical) => Task::Regression,
            // The reference raises `NotImplementedError` here.
            _ => return Err(not_yet("target_stype")),
        },
    };
    Ok(Asked {
        entity_table: entity.table.clone(),
        entity_column: entity.column.clone(),
        target,
        task,
    })
}

/// The target `node` as a [`Shape`], or the rule it takes that this build
/// does not port: an aggregation other than `SUM`, `COUNT`, `AVG`, `MIN`
/// and `MAX` (`aggregation`), of anything but a column (`target_filter`),
/// or without a start (`open_window`); a static column of another table
/// than the entity's (`static_target`); a condition other than a
/// comparison with a number or `NULL` of a column or an aggregation
/// (`condition`); `AND`, `OR` and `NOT` (`logical_target`).
fn shape(node: &PqlNode, entity_table: &str) -> Result<Shape, CoreError> {
    match &node.kind {
        // A join holds an aggregation (`execute_join`).
        NodeKind::Join { target, .. } => match &target.kind {
            NodeKind::Aggregation { .. } => shape(target, entity_table),
            _ => Err(not_yet("static_target")),
        },
        NodeKind::Column { column } => {
            if column.table == entity_table && column.column != "*" {
                Ok(Shape::Column(column.column.clone()))
            } else {
                Err(not_yet("static_target"))
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
                _ => return Err(not_yet("aggregation")),
            };
            let column = column_of(target).ok_or_else(|| not_yet("target_filter"))?;
            let window = window.ok_or_else(|| not_yet("open_window"))?;
            let start = window.start.ok_or_else(|| not_yet("open_window"))?;
            Ok(Shape::Aggregated(Windowed {
                aggregation,
                table: column.table.clone(),
                column: (column.column != "*").then(|| column.column.clone()),
                start: Offset::of(start, window.unit),
                end: Offset::of(window.end, window.unit),
            }))
        }
        NodeKind::Condition {
            target, op, value, ..
        } => {
            let op = match op {
                ConditionOp::Eq => labels::RelOp::Eq,
                ConditionOp::Ne => labels::RelOp::Ne,
                ConditionOp::Lt => labels::RelOp::Lt,
                ConditionOp::Le => labels::RelOp::Le,
                ConditionOp::Gt => labels::RelOp::Gt,
                ConditionOp::Ge => labels::RelOp::Ge,
                _ => return Err(not_yet("condition")),
            };
            let NodeKind::Constant { literal } = &value.kind else {
                return Err(not_yet("condition"));
            };
            let constant = match literal {
                PqlLiteral::Null => None,
                PqlLiteral::Int { value } => Some(labels::Constant::Int(
                    value.parse().map_err(|_| not_yet("condition_value"))?,
                )),
                PqlLiteral::Float { text } => Some(labels::Constant::Float(
                    text.parse().map_err(|_| not_yet("condition_value"))?,
                )),
                _ => return Err(not_yet("condition")),
            };
            let target = shape(target, entity_table)?;
            if matches!(target, Shape::Condition { .. }) {
                return Err(not_yet("condition"));
            }
            // An average compared with `NULL`, which the reference fails
            // with an internal error, is refused by name (register P1B-21,
            // the class of decision E66(b)), not labelled.
            if constant.is_none()
                && matches!(
                    &target,
                    Shape::Aggregated(Windowed {
                        aggregation: labels::Aggregation::Avg,
                        ..
                    })
                )
            {
                return Err(not_yet("average_null"));
            }
            Ok(Shape::Condition {
                target: Box::new(target),
                op,
                constant,
            })
        }
        NodeKind::Logical { .. } => Err(not_yet("logical_target")),
        NodeKind::Filter { .. } => Err(not_yet("target_filter")),
        NodeKind::Constant { .. } => Err(not_yet("static_target")),
    }
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
            v.map(|v| {
                v.checked_mul(per)
                    .ok_or_else(|| sample::out_of_range(column))
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
pub(crate) struct Times {
    reader: Reader,
    text: Option<TextTimes>,
}

impl Times {
    /// `batch`, fetched from `table`, with each column whose meaning is a
    /// timestamp read so.
    pub(crate) fn read(
        &self,
        asker: &Asker,
        table: &Table,
        batch: RecordBatch,
    ) -> Result<RecordBatch, CoreError> {
        let schema = batch.schema();
        let mut fields = Vec::with_capacity(schema.fields().len());
        let mut columns = Vec::with_capacity(schema.fields().len());
        let mut read_any = false;
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
            _ => return Ok(None),
        }))
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
/// a boolean, text, a decimal exactly (which only a condition compares,
/// `labels::label` refusing it as a target, P1B-18); a time or any other
/// type is refused by name (`target_type`): the reference's task table
/// takes none of them.
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

/// The time bounds of `tables` together, the smallest of their smallest
/// times and the largest of their largest (`get_min_time` and
/// `get_max_time` over those tables, a table whose time column holds no
/// time counting as `pd.Timestamp.max` and `min`): each as the memo holds
/// it, read by a run of this materializer before, as the reference's
/// sampler reads them once and keeps them; the others read, in one
/// request, and kept.
async fn time_bounds(memo: &mut Memo<'_>, tables: &[&Table]) -> Result<Bounds, CoreError> {
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
        let read = read_bounds(memo.asker, &unknown).await?;
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
    let positions = named(&batch, BOUNDS_POSITION)?
        .as_primitive_opt::<Int32Type>()
        .ok_or_else(|| crate::run::schema_mismatch("expected", Some(BOUNDS_POSITION), "not int32"))?
        .clone();
    let mins = times_ns(named(&batch, BOUNDS_MIN)?, BOUNDS_MIN)?;
    let maxs = times_ns(named(&batch, BOUNDS_MAX)?, BOUNDS_MAX)?;
    let mut out = vec![None; tables.len()];
    for row in 0..batch.num_rows() {
        let at = positions
            .is_valid(row)
            .then(|| usize::try_from(positions.value(row)).ok())
            .flatten()
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
    let Some(IndexAdvice::SingleColumn) = advice else {
        return Ok(());
    };
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
        for table in graph.tables.iter().filter(|t| t.source == index.table) {
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
        // Which of the reference's sentences, by its two counts' number
        // (decision E59(d)): the host words it.
        let variant = match (indexes == 1, missing.len() == 1) {
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
                ("tables".to_owned(), count(missing.len())),
                ("variant".to_owned(), Detail::text(variant)),
            ]
            .into(),
        });
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
        "lag_timesteps" => not_yet("lag_timesteps"),
        "max_pq_iterations" => not_yet("max_pq_iterations"),
        "num_hops" => not_yet("num_hops"),
        "num_neighbors" => not_yet("num_neighbors"),
        "return_embeddings" => not_yet("return_embeddings"),
        "top_k" => not_yet("top_k"),
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
pub(crate) async fn run(asker: Asker, inputs: Inputs) -> Result<OperationResult, CoreError> {
    // The memo the run leaves the host: the one handed back, or a new one
    // (decision E52), left as it is by a run refused before it reads.
    let mut memo = Memo {
        asker: &asker,
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
        ..
    } = inputs;
    let options::Asked {
        num_context_samples,
        prediction_statistic,
        inference_config,
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
    if indices.len() > PREDICT_ROWS {
        return Err(CoreError::new(
            ErrorKind::Unsupported,
            ErrorCode::UnsupportedRequest,
            "more entities were asked for than one request predicts for",
        )
        .with_field("field", Detail::text("indices")));
    }
    // The statistic the caller chose, one the endpoint reports, and no
    // column named as the wire names its own (the client's checks before
    // it gathers; their order among the checks above is not the client's,
    // decision E41(k)(iii)).
    let statistic = options::statistic(prediction_statistic.as_deref())?;
    reserved(&description)?;

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
    let nulls = sampling_capabilities(&asker)?;
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
    let seed_value = seed_scalar(&seed)?;
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
    // A temporal query's aggregated table, and the foreign key by which it
    // refers to the entity table (`_get_query_time_offset_dict`).
    let aggregated = match asked.target.windowed() {
        None => None,
        Some(windowed) => {
            let table = graph
                .table(&windowed.table)
                .ok_or_else(|| internal("the aggregated table is not in the graph"))?;
            let found: Vec<_> = graph
                .edges
                .iter()
                .filter(|e| e.src == windowed.table && e.dst == asked.entity_table)
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
            Some((windowed, table, edge))
        }
    };
    // The subgraph is walked as the reference walks it from the entity
    // table (`subgraph`); an entity table referring to another is not
    // sampled yet.
    if !graph.foreign_keys(&entity.name).is_empty() {
        return Err(not_yet("parent_table"));
    }

    // The times every fetch below returns, read as the reference's engine
    // reads them.
    let read_times = Times {
        reader: crate::describe::text_reader(&facts)?,
        text: asker.capabilities()?.text_times,
    };

    // The tables whose times bound the anchor: a temporal query's
    // aggregated table; for a static query, every table with a time column
    // (`get_min_time()`, `get_max_time()`), none of which leaves the anchor
    // unchecked (`_validate_time` returns) and, by default, no anchor at
    // all (`pd.Timestamp.min`).
    let bounded: Vec<&Table> = match &aggregated {
        Some((_, table, _)) => vec![*table],
        None => graph
            .tables
            .iter()
            .filter(|t| t.time_column.is_some())
            .collect(),
    };

    // The anchor, and the times it is checked against.
    let mut bounds: Option<Bounds> = None;
    let anchor = match anchor_time {
        None if bounded.is_empty() => TIME_MIN,
        None => {
            let read = time_bounds(&mut memo, &bounded).await?;
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
    let context_anchor = match &aggregated {
        Some((windowed, ..)) => windowed.end.negated().add_to(anchor, "anchor_time")?,
        None => anchor,
    };
    let bounds = match bounds {
        Some(bounds) => Some(bounds),
        None if bounded.is_empty() => None,
        None => Some(time_bounds(&mut memo, &bounded).await?),
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
        if let Some((windowed, ..)) = &aggregated
            && reaches(windowed.end.negated(), context_anchor)? < i128::from(bounds.min)
        {
            warn(
                &asker,
                CONTEXT_BEFORE_DATA,
                "the context reaches back before the data",
            );
        }
        if i128::from(anchor) > i128::from(bounds.max) + DAY_NS {
            warn(
                &asker,
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
        &asker,
        entity,
        one_batch(
            &ask(
                &asker,
                DataRequest::SampleEntities {
                    table: entity.source.clone(),
                    key: vec![key.clone()],
                    n: trials as u64,
                    seed: seed_value,
                    columns: entity.ordered(&entity_columns),
                    filters,
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
            let all = times_ns(named(&sampled, time)?, time)?;
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
    let examples = match &aggregated {
        Some((windowed, ..)) => sample::temporal_examples(
            &mut rng,
            kept.len(),
            entity_time.as_deref(),
            context_anchor,
            timeframe(windowed.start, windowed.end, "anchor_time")?,
            bounds.as_ref().map_or(TIME_MAX, |b| b.min),
            trials,
        )?,
        None => {
            sample::static_examples(&mut rng, kept.len(), entity_time.as_deref(), context_anchor)
        }
    };

    // Their labels, as the reference's executor computes them over the
    // rows read for them (decision E19).
    let entity_rows: Vec<usize> = examples.index.iter().map(|&i| kept[i]).collect();
    let target = Labelling {
        asker: &asker,
        times: &read_times,
        sampled: &sampled,
        entity_rows: &entity_rows,
        entity_key: &entity_key,
        examples: &examples,
    }
    .target(&asked.target, aggregated.as_ref())
    .await?;
    let labelled = labels::label(&target)?;
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
        return labels_result(&asker, &train_keys, &anchors, &labels);
    }
    // The inference options, read for the task's config class, as the
    // engine reads them before it lays out any request.
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
    let num_test = indices.len();
    let examples_total = num_train + num_test;
    if num_train > TABLE_ROWS {
        return Err(invalid(
            "the context holds more rows than a request takes",
            "query",
        ));
    }
    let all_keys = arrow_select::concat::concat(&[train_keys.as_ref(), indices.as_ref()])
        .map_err(|_| internal("the examples' keys do not concatenate"))?;
    anchors.extend(std::iter::repeat_n(anchor, num_test));

    // The subgraph: each example's entity row.
    let entity_names = crate::subgraph::sample_columns(&graph, entity, &excluded);
    let looked_up = read_times.read(
        &asker,
        entity,
        one_batch(
            &ask(
                &asker,
                sample::by_pkey(
                    entity,
                    entity_names.clone(),
                    key_set(&asker, sample::seeds(all_keys, vec![])?)?,
                )?,
            )
            .await?,
        )?,
    )?;
    let positions = batches(&looked_up, examples_total)?;
    let matches = named(&looked_up, MATCHES_COLUMN)?
        .as_primitive_opt::<Int64Type>()
        .ok_or_else(|| {
            crate::run::schema_mismatch("expected", Some(MATCHES_COLUMN), "not int64")
        })?;
    for (row, &at) in positions.iter().enumerate() {
        if at >= num_train && matches.value(row) > 1 {
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
        asker: &asker,
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
    // The subgraph carries at most as many tables as the endpoint reads,
    // refused as the reference's engine refuses it (`_get_context`).
    if walked.tables.len() > SUBGRAPH_TABLES {
        return Err(invalid(
            "the request's subgraph carries more tables than the endpoint reads",
            "tables",
        )
        .with_field("tables", count(walked.tables.len()))
        .with_field("limit", count(SUBGRAPH_TABLES)));
    }
    let sent: BTreeSet<String> = walked.tables.iter().map(|t| t.name.clone()).collect();

    let built = payload::build(&Sampled {
        num_train,
        num_test,
        anchors,
        kind: asked.task,
        labels,
        windowed: aggregated.is_some(),
        tables: walked.tables,
        links: walked.links,
        entity_table: entity.name.clone(),
        source_query: source_query.clone(),
        run_mode: mode,
        options,
        statistic,
    })?;
    // What the request leaves out, said as the client says it, in its
    // order: the tables, the columns the model does not read, and the
    // context that answers the question.
    notices(
        &asker,
        &description,
        &asked,
        &sent,
        &excluded,
        built.answering,
        aggregated.is_some(),
    );
    let payload = asker.attach(Attachment::Bytes(built.bytes))?;
    let query = asker.attach(Attachment::Bytes(
        serde_json::to_vec(&source_query).map_err(|_| internal("a query does not serialize"))?,
    ))?;
    Ok(OperationResult::Materialize(Materialized {
        state: RequestState {
            version: REQUEST_STATE_VERSION,
            payload,
            query,
        },
    }))
}

/// `at + offset` in nanoseconds: a fixed span added without bound, as the
/// reference compares such times; a calendar one as pandas adds it.
fn reaches(offset: Offset, at: i64) -> Result<i128, CoreError> {
    match offset.fixed_nanos() {
        Some(span) => Ok(i128::from(at) + span),
        None => offset.add_to(at, "anchor_time").map(i128::from),
    }
}

/// What a run's labels are read from: the entity rows sampled and, for a
/// windowed aggregation, the rows of the aggregated table its window holds
/// for each example (`_sample_query_data`).
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

impl Labelling<'_> {
    /// The target `shape` names, over the examples.
    async fn target(
        &self,
        shape: &Shape,
        aggregated: Option<&(&Windowed, &Table, Edge)>,
    ) -> Result<labels::Target, CoreError> {
        match shape {
            Shape::Condition {
                target,
                op,
                constant,
            } => Ok(labels::Target::Condition {
                target: Box::new(self.leaf(target, aggregated).await?),
                op: *op,
                constant: *constant,
            }),
            leaf => self.leaf(leaf, aggregated).await,
        }
    }

    async fn leaf(
        &self,
        shape: &Shape,
        aggregated: Option<&(&Windowed, &Table, Edge)>,
    ) -> Result<labels::Target, CoreError> {
        match shape {
            Shape::Column(column) => Ok(labels::Target::Column(static_column(
                named(self.sampled, column)?,
                self.entity_rows,
            )?)),
            Shape::Aggregated(_) => {
                let (windowed, table, edge) =
                    aggregated.ok_or_else(|| internal("an aggregation without its table"))?;
                self.aggregate(windowed, table, edge)
                    .await
                    .map(labels::Target::Aggregated)
            }
            Shape::Condition { .. } => Err(internal("a condition nested in a condition")),
        }
    }

    /// `windowed` over each example's rows of `aggregated`: those whose
    /// foreign key `edge` names the example's entity, after the window's
    /// start and not after its end from the example's anchor (`_by_time`).
    async fn aggregate(
        &self,
        windowed: &Windowed,
        aggregated: &Table,
        edge: &Edge,
    ) -> Result<labels::Aggregated, CoreError> {
        let examples = self.examples.index.len();
        let example_keys = take(self.entity_key, &self.examples.index)?;
        let shifted = |offset: Offset| -> Result<Vec<i64>, CoreError> {
            self.examples
                .anchors
                .iter()
                .map(|a| offset.add_to(*a, "anchor_time"))
                .collect()
        };
        let ends = shifted(windowed.end)?;
        let starts = shifted(windowed.start)?;
        let time_column = aggregated
            .time_column
            .clone()
            .ok_or_else(|| internal("the aggregated table has no time column"))?;
        let mut label_columns = BTreeSet::from([time_column.clone()]);
        if let Some(column) = &windowed.column {
            label_columns.insert(column.clone());
        }
        let seeds = sample::seeds(
            example_keys,
            vec![(SEED_END, ends.clone()), (SEED_START, starts.clone())],
        )?;
        let rows = self.times.read(
            self.asker,
            aggregated,
            one_batch(
                &ask(
                    self.asker,
                    sample::by_time(
                        aggregated,
                        &edge.fkey,
                        aggregated.ordered(&label_columns),
                        key_set(self.asker, seeds)?,
                        true,
                    )?,
                )
                .await?,
            )?,
        )?;
        let values = match &windowed.column {
            Some(column) => label_values(named(&rows, column)?)?,
            None => labels::Values::Ones(rows.num_rows()),
        };
        labels::aggregate(
            windowed.aggregation,
            examples,
            &batches(&rows, examples)?,
            &values,
            &times_ns(named(&rows, &time_column)?, &time_column)?,
            labels::Bounds {
                starts: Some(&starts),
                ends: &ends,
            },
        )
    }
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
