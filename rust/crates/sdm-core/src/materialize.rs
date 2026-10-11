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
//! one the source enforces; the time bounds of the aggregated table; the
//! reservoir sample of the entity table (`sample_entities`); the windowed
//! rows each example's label sums (`neighbors` with a window); the entity
//! rows of every example (`lookup`); and the latest rows referring to each
//! (`neighbors` with a row limit and an order). The examples' order is
//! numpy's (`sdm_engine::random`), the labels the reference executor's
//! (`sdm_engine::sample`), the payload the engine's layout
//! (`sdm_request::payload`).
//!
//! As built at T1.11c it is thin: it materializes a regression over a
//! windowed `SUM` for each entity of a table that others refer to, with the
//! reference's defaults (a fast run, two hops of 32 neighbors, the default
//! seed or the one given), and refuses with `UNSUPPORTED_REQUEST` naming
//! the rule ([`sdm_engine::not_yet`]) every input whose request takes a
//! rule it does not port yet (register entry P1B-3).

use std::collections::BTreeSet;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{
    Float32Type, Float64Type, Int8Type, Int16Type, Int32Type, Int64Type, TimestampMicrosecondType,
    TimestampMillisecondType, TimestampNanosecondType, TimestampSecondType, UInt8Type, UInt16Type,
    UInt32Type, UInt64Type,
};
use arrow_array::{Array, ArrayRef, RecordBatch, UInt32Array};
use arrow_schema::{DataType, TimeUnit as ArrowUnit};
use sdm_contract::catalog::CatalogAnswer;
use sdm_contract::data::{
    AttachmentId, BATCH_COLUMN, DataRequest, DescribePart, Filter, KeySet, MATCHES_COLUMN,
    SeedBound, TableColumn, Window,
};
use sdm_contract::ops::{
    ColumnRef, Materialized, OperationResult, ProfileName, ProfileSpec, QueryColumn, QuerySchema,
    QueryTable, RequestState, Seed, SourceDescription,
};
use sdm_contract::pql::{Aggregation, NodeKind, PqlNode, ValidatedQuery};
use sdm_contract::{
    ColumnType, CoreError, CoreWarning, Detail, ErrorCode, ErrorKind, Field, HostFacts, Scalar,
    Unit,
};
use sdm_engine::graph::{Graph, Table};
use sdm_engine::keys::{self, Key};
use sdm_engine::not_yet;
use sdm_engine::random::{Entropy, default_rng};
use sdm_engine::sample::{self, Values};
use sdm_request::payload::{self, Sampled, SampledLink, SampledTable};

use crate::describe::{ask, describe};
use crate::profiles::Profile;
use crate::run::{ArrowData, Asker, Attachment, internal};

/// The warning that indices the sampler's reads would use are missing.
pub const MISSING_INDICES: &str = "MISSING_INDICES";
/// The warning that the context reaches back before the data.
pub const CONTEXT_BEFORE_DATA: &str = "CONTEXT_BEFORE_DATA";
/// The warning that the anchor time is after the data.
pub const ANCHOR_AFTER_DATA: &str = "ANCHOR_AFTER_DATA";

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
];
/// The seed the reference samples with when none is given.
const DEFAULT_SEED: u128 = 42;
/// A fast run's most context examples (`_MAX_CONTEXT_SIZE`).
const CONTEXT_EXAMPLES: usize = 1_000;
/// How many times as many candidates as examples a run tries
/// (`max_pq_iterations`' default).
const ITERATIONS: usize = 10;
/// A fast run's neighbors per hop, two hops (`_DEFAULT_NUM_NEIGHBORS`).
const NEIGHBORS: [u64; 2] = [32, 32];
/// The most entities one request predicts for.
const PREDICT_ROWS: usize = 1_000;
/// The most rows an instance table may hold (`MAX_TABLE_ROWS`).
const TABLE_ROWS: usize = 10_000;
/// The largest payload the endpoint takes, in bytes (`_MAX_SIZE`).
const PAYLOAD_BYTES: usize = 30 * 1024 * 1024;
/// The seed time columns this operation's requests carry.
const SEED_END: &str = "end";
const SEED_START: &str = "start";
const SEED_TIME: &str = "time";
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
    pub facts: HostFacts,
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
fn one_batch(data: &ArrowData) -> Result<RecordBatch, CoreError> {
    arrow_select::concat::concat_batches(&data.schema, &data.batches)
        .map_err(|_| internal("a fetch's batches do not concatenate"))
}

fn named<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a ArrayRef, CoreError> {
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

/// The batch positions a `neighbors` or `lookup` answer names.
fn batches(batch: &RecordBatch, seeds: usize) -> Result<Vec<usize>, CoreError> {
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
fn key_set(asker: &Asker, seeds: RecordBatch) -> Result<KeySet, CoreError> {
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

async fn time_bounds(asker: &Asker, table: &Table) -> Result<Bounds, CoreError> {
    let column = table
        .time_column
        .clone()
        .ok_or_else(|| internal("the aggregated table has no time column"))?;
    let answer = ask(
        asker,
        DataRequest::TimeBounds {
            columns: vec![TableColumn {
                table: table.name.clone(),
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
    Ok(Bounds {
        min: read(BOUNDS_MIN)?.unwrap_or(TIME_MAX),
        max: read(BOUNDS_MAX)?.unwrap_or(TIME_MIN),
    })
}

/// The sampler's check of the indexes its reads would use: every key and
/// foreign key the source does not enforce as unique, less those an index
/// over that one column covers, warned of as the reference warns of them.
/// The data layer reads the source's indexes (decision E54): which column
/// an index covers is its dialect's reading, never this module's.
async fn missing_indexes(asker: &Asker, graph: &Graph) -> Result<(), CoreError> {
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
    for index in indexes {
        if graph.table(&index.table).is_none() {
            continue;
        }
        if let [column] = index.columns.as_slice()
            && let Some(at) = missing.iter().position(|(t, _)| *t == index.table)
        {
            missing[at].1.retain(|c| c != column);
            if missing[at].1.is_empty() {
                missing.remove(at);
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
            let supported = crate::pql::interim_stype(column.stype)
                .supports_dtype(crate::pql::interim_dtype(column.dtype));
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

/// The body of a `materialize` run.
pub(crate) async fn run(asker: Asker, inputs: Inputs) -> Result<OperationResult, CoreError> {
    let Inputs {
        mut description,
        query,
        indices,
        anchor_time,
        seed,
        facts,
    } = inputs;
    // A table the schema names otherwise than the source does: its data
    // requests would name the schema's table, so it is refused by name
    // until the route that takes tables named samples them (T1.24).
    if description.tables.iter().any(|t| t.source != t.name) {
        return Err(not_yet("table_source"));
    }
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
    let validation = crate::pql::resolve(&query, &query_schema(&description), &profile, &facts)?;
    let (Some(validated), Some(source_query)) = (validation.query, validation.text) else {
        return Err(invalid(
            "the query does not resolve against the schema",
            "query",
        ));
    };
    // The index read, the sampler's check of the indexes it would use,
    // comes before anything of the task is read, as the reference builds
    // its sampler, and gives its warning, before its task table. The checks
    // after it (null entities, then the seed, then the task's shape) are in
    // this core's own order, not the reference's: an input with several
    // errors may be refused for another of them than the reference's first
    // (decision E41(k)(i)).
    missing_indexes(&asker, &graph).await?;
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
    // The subgraph this build samples: the entity table, which refers to
    // nothing, and the tables referring to it, which refer to nothing else
    // and nothing refers to.
    if !graph.foreign_keys(&entity.name).is_empty() {
        return Err(not_yet("parent_table"));
    }
    let referring: Vec<(String, String)> = graph
        .referring(&entity.name)
        .into_iter()
        .map(|(src, fkey)| (src.to_owned(), fkey.to_owned()))
        .collect();
    for (src, _) in &referring {
        if *src == entity.name
            || graph.foreign_keys(src).len() != 1
            || !graph.referring(src).is_empty()
        {
            return Err(not_yet("deeper_hop"));
        }
    }
    for table in
        std::iter::once(entity.name.as_str()).chain(referring.iter().map(|(s, _)| s.as_str()))
    {
        let table = graph
            .table(table)
            .ok_or_else(|| internal("a sampled table is missing"))?;
        if graph
            .features(table)
            .iter()
            .any(|(_, stype)| *stype == sdm_contract::ops::Stype::Text)
        {
            return Err(not_yet("text_column"));
        }
    }

    // The anchor, and the times it is checked against.
    let mut bounds: Option<Bounds> = None;
    let anchor = match anchor_time {
        None => {
            let read = time_bounds(&asker, aggregated).await?;
            let max = read.max;
            bounds = Some(read);
            max
        }
        Some(Scalar::Timestamp {
            unit,
            zone: None,
            value,
        }) => match unit {
            Unit::Ns => value,
            Unit::Us => value
                .checked_mul(1_000)
                .ok_or_else(|| sample::out_of_range("anchor_time"))?,
        },
        Some(Scalar::Timestamp { .. }) => return Err(not_yet("anchor_zone")),
        Some(_) => {
            return Err(invalid("an anchor time is a ts scalar", "anchor_time"));
        }
    };
    let context_anchor = sample::shift(anchor, -asked.end, "anchor_time")?;
    let bounds = match bounds {
        Some(bounds) => bounds,
        None => time_bounds(&asker, aggregated).await?,
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
    let sampled = one_batch(
        &ask(
            &asker,
            DataRequest::SampleEntities {
                table: entity.name.clone(),
                key: vec![key.clone()],
                n: trials as u64,
                seed: seed_value,
                columns: entity.ordered(&entity_columns),
                filters,
            },
        )
        .await?,
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
    let labelled = one_batch(
        &ask(
            &asker,
            DataRequest::Neighbors {
                table: aggregated.name.clone(),
                key: edge.fkey.clone(),
                columns: aggregated.ordered(&label_columns),
                seeds: key_set(&asker, seeds)?,
                window: Some(Window {
                    column: time_column.clone(),
                    lower: Some(SeedBound {
                        seed_column: SEED_START.into(),
                        inclusive: false,
                    }),
                    upper: Some(SeedBound {
                        seed_column: SEED_END.into(),
                        inclusive: true,
                    }),
                }),
                per_seed: None,
                order: None,
            },
        )
        .await?,
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
        // The reference warns of it once in a process, which a core that
        // holds no state across runs cannot do.
        return Err(not_yet("context_coverage"));
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
    let mut entity_names: BTreeSet<String> = graph
        .features(entity)
        .into_iter()
        .map(|(name, _)| name)
        .collect();
    entity_names.insert(key.clone());
    let looked_up = one_batch(
        &ask(
            &asker,
            DataRequest::Lookup {
                table: entity.name.clone(),
                key: key.clone(),
                columns: entity.ordered(&entity_names),
                seeds: key_set(&asker, sample::seeds(all_keys, vec![])?)?,
            },
        )
        .await?,
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
    let entity_keys = Arc::clone(named(&entity_rows, &key)?);
    let entity_table = SampledTable {
        name: entity.name.clone(),
        primary_key: Some(key.clone()),
        batch: (0..examples_total).collect(),
        columns: entity
            .ordered(&entity_names)
            .into_iter()
            .map(|name| Ok((name.clone(), Arc::clone(named(&entity_rows, &name)?))))
            .collect::<Result<_, CoreError>>()?,
        stypes: graph.features(entity),
    };

    // The first hop: the latest rows of each table referring to the
    // entities, up to the anchor; the second finds them all already
    // sampled, each row's foreign key its own example's entity.
    let mut tables = vec![entity_table];
    let mut links = Vec::new();
    let entity_key_values = keys::read(&entity_keys)?;
    for (src, fkey) in &referring {
        let table = graph
            .table(src)
            .ok_or_else(|| internal("a sampled table is missing"))?;
        let mut names: BTreeSet<String> =
            graph.features(table).into_iter().map(|(n, _)| n).collect();
        names.insert(fkey.clone());
        if let Some(pk) = &table.primary_key {
            names.insert(pk.clone());
        }
        let columns = table.ordered(&names);
        let (times, window) = match &table.time_column {
            Some(time) => (
                vec![(SEED_TIME, anchors.clone())],
                Some(Window {
                    column: time.clone(),
                    lower: None,
                    upper: Some(SeedBound {
                        seed_column: SEED_TIME.into(),
                        inclusive: true,
                    }),
                }),
            ),
            None => (vec![], None),
        };
        let neighbors = one_batch(
            &ask(
                &asker,
                DataRequest::Neighbors {
                    table: src.clone(),
                    key: fkey.clone(),
                    columns: columns.clone(),
                    seeds: key_set(&asker, sample::seeds(Arc::clone(&entity_keys), times)?)?,
                    window,
                    per_seed: Some(NEIGHBORS[0]),
                    order: Some(sample::tie_break(table)?),
                },
            )
            .await?,
        )?;
        if neighbors.num_rows() == 0 {
            continue;
        }
        let batch = batches(&neighbors, examples_total)?;
        // A row already sampled for its example is kept once.
        let rows: Vec<usize> = match &table.primary_key {
            Some(pk) => {
                let values = keys::read(named(&neighbors, pk)?)?;
                let mut seen = BTreeSet::new();
                (0..batch.len())
                    .filter(|&row| seen.insert((values[row].clone(), batch[row])))
                    .collect()
            }
            None => (0..batch.len()).collect(),
        };
        let kept_rows = take_batch(&neighbors, &rows)?;
        let kept_batch: Vec<usize> = rows.iter().map(|&r| batch[r]).collect();
        // The second hop: each row's foreign key is its example's entity.
        let foreign = keys::read(named(&kept_rows, fkey)?)?;
        let mut pairs = Vec::with_capacity(rows.len());
        for (row, (&example, value)) in kept_batch.iter().zip(&foreign).enumerate() {
            match (value, &entity_key_values[example]) {
                (Some(Key::Int(a)), Some(Key::Int(b))) if a == b => pairs.push((row, example)),
                (Some(Key::Text(a)), Some(Key::Text(b))) if a == b => pairs.push((row, example)),
                _ => return Err(not_yet("parent_lookup")),
            }
        }
        let features = graph.features(table);
        let sent: BTreeSet<String> = features.iter().map(|(n, _)| n.clone()).collect();
        tables.push(SampledTable {
            name: src.clone(),
            primary_key: None,
            batch: kept_batch,
            columns: columns
                .iter()
                .filter(|c| sent.contains(*c))
                .map(|name| Ok((name.clone(), Arc::clone(named(&kept_rows, name)?))))
                .collect::<Result<_, CoreError>>()?,
            stypes: features,
        });
        links.push(SampledLink {
            src: src.clone(),
            fkey: fkey.clone(),
            dst: entity.name.clone(),
            pairs,
        });
    }

    let payload = payload::regression(&Sampled {
        num_train,
        num_test,
        anchors,
        targets,
        tables,
        links,
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
