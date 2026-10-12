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
//! (T1.26a); the time bounds of the aggregated table; the
//! reservoir sample of the entity table (`sample_entities`); the windowed
//! rows each example's label sums (`neighbors` with a window, `_by_time`);
//! the entity rows of every example (`lookup`); and then the subgraph,
//! walked hop by hop as the reference walks it ([`crate::subgraph`]: the
//! keys each table's rows refer to looked up, the latest rows referring to
//! them read, `_by_fkey`, nulls placed as the provider declares its source
//! places them). The examples' order is numpy's (`sdm_engine::random`), the
//! labels the reference executor's (`sdm_engine::sample`), the payload the
//! engine's layout (`sdm_request::payload`).
//!
//! What the reference's sampler reads once and keeps, the index check and
//! the time bounds, a run reads only where the memo its host hands back
//! does not hold it (decision E52): the run keeps what it read in a memo it
//! leaves the host, done or failed (`Asker::keep`), and the core holds no
//! state across runs. The coverage warning, which the reference gives once
//! in a process, is given on every run that meets it, and once in a process
//! by the host ([`ONCE_PER_PROCESS`]).
//!
//! It materializes a regression over a windowed `SUM` for each entity of a
//! table that refers to no other, with the reference's defaults (a fast
//! run, two hops of 32 neighbors, the default seed or the one given), and
//! refuses with `UNSUPPORTED_REQUEST` naming the rule
//! ([`sdm_engine::not_yet`]) every input whose request takes a rule it does
//! not port yet (register entry P1B-3), and the requests and policies it
//! needs of its provider before anything is fetched.

use std::collections::BTreeSet;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{
    Date32Type, Float32Type, Float64Type, Int8Type, Int16Type, Int32Type, Int64Type,
    TimestampMicrosecondType, TimestampMillisecondType, TimestampNanosecondType,
    TimestampSecondType, UInt8Type, UInt16Type, UInt32Type, UInt64Type,
};
use arrow_array::{Array, ArrayRef, RecordBatch, TimestampNanosecondArray, UInt32Array};
use arrow_schema::{DataType, TimeUnit as ArrowUnit};
use sdm_contract::catalog::{CatalogAnswer, IndexColumns};
use sdm_contract::data::{
    AttachmentId, BATCH_COLUMN, DataRequest, DescribePart, Filter, IndexAdvice, KeySet,
    MATCHES_COLUMN, NullOrder, OptionalRequest, ProviderChoice, TableColumn, TextTimes,
};
use sdm_contract::ops::{
    ColumnRef, Materialized, MemoBounds, OperationResult, ProfileName, ProfileSpec, QueryColumn,
    QuerySchema, QueryTable, RequestState, SamplerMemo, Seed, SourceDescription,
};
use sdm_contract::pql::{Aggregation, NodeKind, PqlNode, ValidatedQuery};
use sdm_contract::{
    ColumnType, CoreError, CoreWarning, Detail, ErrorCode, ErrorKind, Field, HostFacts, Scalar,
    Unit,
};
use sdm_engine::graph::{Graph, Table};
use sdm_engine::keys;
use sdm_engine::not_yet;
use sdm_engine::random::{Entropy, default_rng};
use sdm_engine::sample::{self, SEED_END, SEED_START, Values};
use sdm_engine::time::Instant;
use sdm_engine::time::text::{Reader, Unread};
use sdm_request::payload::{self, Sampled};

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
];
/// The seed the reference samples with when none is given.
const DEFAULT_SEED: u128 = 42;
/// A fast run's most context examples (`_MAX_CONTEXT_SIZE`).
const CONTEXT_EXAMPLES: usize = 1_000;
/// How many times as many candidates as examples a run tries
/// (`max_pq_iterations`' default).
const ITERATIONS: usize = 10;
/// The most entities one request predicts for.
const PREDICT_ROWS: usize = 1_000;
/// The most rows an instance table may hold (`MAX_TABLE_ROWS`).
const TABLE_ROWS: usize = 10_000;
/// The largest payload the endpoint takes, in bytes (`_MAX_SIZE`).
const PAYLOAD_BYTES: usize = 30 * 1024 * 1024;
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
    pub num_context_samples: Option<u64>,
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

/// The parts of a validated query this build materializes.
struct Asked {
    entity_table: String,
    entity_column: String,
    /// The aggregated table and column.
    table: String,
    column: String,
    /// The window, from the anchor, in nanoseconds.
    start: i128,
    end: i128,
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
    let mut target = &query.target;
    if let NodeKind::Join { target: inner, .. } = &target.kind {
        target = inner;
    }
    let NodeKind::Aggregation {
        aggregation,
        target: aggregated,
        window,
        ..
    } = &target.kind
    else {
        return Err(not_yet("static_target"));
    };
    if *aggregation != Aggregation::Sum {
        return Err(not_yet("aggregation"));
    }
    let column = column_of(aggregated).ok_or_else(|| not_yet("target_filter"))?;
    let window = window.ok_or_else(|| not_yet("open_window"))?;
    let start = window.start.ok_or_else(|| not_yet("open_window"))?;
    let span = |n: i32| sample::span_ns(n, window.unit).ok_or_else(|| not_yet("month_window"));
    Ok(Asked {
        entity_table: entity.table.clone(),
        entity_column: entity.column.clone(),
        table: column.table.clone(),
        column: column.column.clone(),
        start: span(start)?,
        end: span(window.end)?,
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

/// A label column's values: a float read at float32, as the reference's
/// executor reads it, or an integer.
fn label_values(array: &ArrayRef) -> Result<Values, CoreError> {
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
        DataType::UInt64 => Values::Int(
            array
                .as_primitive::<UInt64Type>()
                .iter()
                .map(|v| v.map(|v| v as i64))
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

/// `table`'s time bounds: as the memo holds them, read by a run of this
/// materializer before, as the reference's sampler reads them once and
/// keeps them (`get_min_time`, `get_max_time`); else read, and kept.
async fn time_bounds(memo: &mut Memo<'_>, table: &Table) -> Result<Bounds, CoreError> {
    let column = table
        .time_column
        .clone()
        .ok_or_else(|| internal("the aggregated table has no time column"))?;
    let kept = memo
        .memo
        .bounds
        .iter()
        .find(|b| b.table == table.name && b.column == column);
    let (min, max) = match kept {
        Some(kept) => (kept.min, kept.max),
        None => {
            let (min, max) = read_bounds(memo.asker, table, &column).await?;
            memo.memo.bounds.push(MemoBounds {
                table: table.name.clone(),
                column,
                min,
                max,
            });
            memo.keep();
            (min, max)
        }
    };
    Ok(Bounds {
        min: min.unwrap_or(TIME_MAX),
        max: max.unwrap_or(TIME_MIN),
    })
}

async fn read_bounds(
    asker: &Asker,
    table: &Table,
    column: &str,
) -> Result<(Option<i64>, Option<i64>), CoreError> {
    let column = column.to_owned();
    let answer = ask(
        asker,
        DataRequest::TimeBounds {
            columns: vec![TableColumn {
                table: table.source.clone(),
                column,
            }],
        },
    )
    .await?;
    let batch = one_batch(&answer)?;
    if batch.num_rows() != 1 {
        return Err(crate::run::schema_mismatch(
            "expected",
            None,
            "the time bounds do not hold one row per column",
        ));
    }
    let read = |name: &str| -> Result<Option<i64>, CoreError> {
        Ok(times_ns(named(&batch, name)?, name)?[0])
    };
    use sdm_contract::data::results::{BOUNDS_MAX, BOUNDS_MIN};
    Ok((read(BOUNDS_MIN)?, read(BOUNDS_MAX)?))
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
/// language answers it (`Stype.supports_dtype`); and each link able to
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
        "inference_config" => not_yet("inference_config"),
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
/// sampling options, the first in name order; an unseeded sample; a
/// context size, whose run mode is T1.28a's; and an anchor at each
/// entity's own time.
fn options_taken(inputs: &Inputs) -> Result<(), CoreError> {
    if let Some(first) = inputs.options.iter().min() {
        return Err(option_not_taken(first));
    }
    if inputs.unseeded {
        return Err(not_yet("unseeded"));
    }
    if inputs.num_context_samples.is_some() {
        return Err(not_yet("num_context_samples"));
    }
    if inputs.anchor_entity {
        return Err(not_yet("anchor_entity"));
    }
    Ok(())
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
    options_taken(&inputs)?;
    let Inputs {
        mut description,
        query,
        indices,
        anchor_time,
        seed,
        facts,
        zones,
        ..
    } = inputs;
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
    // D4's warning, each time a run reads the query's literals.
    for warning in &validation.warnings {
        asker.warn(warning.clone());
    }
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
    if indices.null_count() > 0 {
        return Err(invalid("an entity to predict for is missing", "indices"));
    }
    let seed_value = seed_scalar(&seed)?;
    let asked = asked(&validated)?;
    let entity = graph
        .table(&asked.entity_table)
        .ok_or_else(|| internal("the query's entity table is not in the graph"))?;
    let key = entity
        .primary_key
        .clone()
        .filter(|k| *k == asked.entity_column)
        .ok_or_else(|| not_yet("entity_key"))?;
    let aggregated = graph
        .table(&asked.table)
        .ok_or_else(|| internal("the aggregated table is not in the graph"))?;
    let edge = {
        let found: Vec<_> = graph
            .edges
            .iter()
            .filter(|e| e.src == asked.table && e.dst == asked.entity_table)
            .collect();
        match found.as_slice() {
            [one] => (*one).clone(),
            _ => {
                return Err(invalid(
                    "no unique foreign key leads from the aggregated table to the entity table",
                    "query",
                ));
            }
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

    // The anchor, and the times it is checked against.
    let mut bounds: Option<Bounds> = None;
    let anchor = match anchor_time {
        None => {
            let read = time_bounds(&mut memo, aggregated).await?;
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
    let context_anchor = sample::shift(anchor, -asked.end, "anchor_time")?;
    let bounds = match bounds {
        Some(bounds) => bounds,
        None => time_bounds(&mut memo, aggregated).await?,
    };
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
    if i128::from(context_anchor) - asked.end < i128::from(bounds.min) {
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

    // The examples: a reservoir sample of the entity table, kept one row
    // per entity, walked back from the context's anchor by the window.
    let trials = ITERATIONS * CONTEXT_EXAMPLES;
    let mut entity_columns = BTreeSet::from([key.clone()]);
    if let Some(time) = &entity.time_column {
        entity_columns.insert(time.clone());
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
    if indices.data_type() != entity_key.data_type() {
        return Err(not_yet("indices_type"));
    }
    let mut rng = default_rng(&entropy);
    let examples = sample::temporal_examples(
        &mut rng,
        kept.len(),
        entity_time.as_deref(),
        context_anchor,
        asked.end - asked.start,
        bounds.min,
        trials,
    )?;

    // Their labels: the windowed rows of the aggregated table.
    let example_keys = take(&entity_key, &examples.index)?;
    let ends = examples
        .anchors
        .iter()
        .map(|a| sample::shift(*a, asked.end, "anchor_time"))
        .collect::<Result<Vec<_>, _>>()?;
    let starts = examples
        .anchors
        .iter()
        .map(|a| sample::shift(*a, asked.start, "anchor_time"))
        .collect::<Result<Vec<_>, _>>()?;
    let time_column = aggregated
        .time_column
        .clone()
        .ok_or_else(|| internal("the aggregated table has no time column"))?;
    let label_columns = BTreeSet::from([asked.column.clone(), time_column.clone()]);
    let seeds = sample::seeds(example_keys, vec![(SEED_END, ends), (SEED_START, starts)])?;
    let labelled = read_times.read(
        &asker,
        aggregated,
        one_batch(
            &ask(
                &asker,
                sample::by_time(
                    aggregated,
                    &edge.fkey,
                    aggregated.ordered(&label_columns),
                    key_set(&asker, seeds)?,
                    true,
                )?,
            )
            .await?,
        )?,
    )?;
    let labels = sample::sum_labels(
        examples.index.len(),
        &batches(&labelled, examples.index.len())?,
        &label_values(named(&labelled, &asked.column)?)?,
        &times_ns(named(&labelled, &time_column)?, &time_column)?,
        &examples.anchors,
        sample::Window {
            start: Some(asked.start),
            end: asked.end,
        },
    );
    let num_train = examples.index.len().min(CONTEXT_EXAMPLES);
    if num_train == 0 {
        return Err(invalid("no context example was collected", "anchor_time"));
    }
    if kept.len() >= trials && num_train < CONTEXT_EXAMPLES / 2 {
        // The reference warns of it once in a process: given on every run
        // that meets it, and once in a process by the host
        // ([`ONCE_PER_PROCESS`], decision E52).
        asker.warn(CoreWarning {
            code: CONTEXT_COVERAGE.into(),
            message: "fewer than half the context examples asked for were collected".into(),
            fields: [
                ("examples".to_owned(), count(CONTEXT_EXAMPLES)),
                ("candidates".to_owned(), count(trials)),
            ]
            .into(),
        });
    }
    let train_rows: Vec<usize> = (0..num_train).collect();
    let train_keys = take(&take(&entity_key, &examples.index)?, &train_rows)?;
    let targets = labels[..num_train].to_vec();
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
    let mut anchors = examples.anchors[..num_train].to_vec();
    anchors.extend(std::iter::repeat_n(anchor, num_test));

    // The subgraph: each example's entity row.
    let entity_names = crate::subgraph::sample_columns(&graph, entity);
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
    }
    .run(entity_rows)
    .await?;

    let payload = payload::regression(&Sampled {
        num_train,
        num_test,
        anchors,
        targets,
        tables: walked.tables,
        links: walked.links,
        entity_table: entity.name.clone(),
        source_query: source_query.clone(),
    })?;
    if payload.len() > PAYLOAD_BYTES {
        return Err(invalid(
            "the request is larger than the endpoint takes",
            "query",
        ));
    }
    let payload = asker.attach(Attachment::Bytes(payload))?;
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

/// A text time reading's refusal, by its rule's name.
fn text_unread(unread: Unread) -> CoreError {
    match unread {
        Unread::Outside => not_yet("text_time"),
        Unread::Order => not_yet("text_time_order"),
        Unread::MixedZones => not_yet("text_time_zones"),
    }
}
