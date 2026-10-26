// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The in-memory fallback (design choice 1, T2.6).
//!
//! A provider may declare limits for it (`ProviderCapabilities::fallback`):
//! an optional request it does not declare is then answered here, from the
//! rows `scan`s read, instead of refused (`DataAccess::route`). The answer
//! has the shape the request's pushdown gives, so the body that asked reads
//! it as it reads the provider's: the requested columns with the fields the
//! scan's admission gave them, and the contract's fixed columns
//! (`DataRequest::expected`) for `key_counts` and `time_bounds`.
//!
//! Limits fail loudly by name, never by reading on or cutting short: each
//! scan asks for one row past what is left of `max_rows`, so no scan moves
//! more than the limit allows, each page of a scan is held to what is left
//! of both limits as it arrives, so none is asked for past them, and more
//! rows than the limit, or more bytes of Arrow data held (the seeds, every
//! scan and the answer) than `max_bytes`, end the run with `UNSUPPORTED_REQUEST` naming the request,
//! the rule `fallback_limit`, the `limit` and its `most`. A value the fallback does
//! not read (a time in a column that is neither a zone-less timestamp nor a
//! date, a key joined to seeds of another type, a column the row format
//! does not order) ends the run the same way, rule `fallback`, naming the
//! `field`, never silently answered otherwise. A provider's failure of a scan is the
//! request's answer, as the provider's own failure of the request would be.
//!
//! Ordering, grouping and distinct counts use Arrow's row format
//! (`arrow-row`), whose byte order is a defined total order of the values,
//! each term's direction and null placement its own; a key, a join and an
//! order term first write the floats SQL holds equal alike (`-0.0` and
//! `0.0`, every NaN), so they group, count and tie as SQL's do: a `neighbors` limit
//! per seed keeps the first rows in the request's order, the scan's order
//! breaking ties. A sample is the core's own draw over a defined ordering
//! of the same data (plan section 3, "Data access"): the scanned rows in
//! the row format's order, shuffled by the registered stream under the
//! request's seed (RS-1, `numpy.random.default_rng(seed)`), the first `n`
//! kept, so it is the same rows on every provider holding the same data.
//! Times are read as the pushdowns' `CAST(column AS TIMESTAMP)` reads them:
//! a zone-less timestamp at microseconds (a finer unit truncated toward
//! zero, as DuckDB casts it), a date at its midnight.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{
    TimestampMicrosecondType, TimestampMillisecondType, TimestampNanosecondType,
    TimestampSecondType,
};
use arrow_array::{
    Array, ArrayRef, Decimal128Array, Int32Array, Int64Array, RecordBatch,
    TimestampMicrosecondArray, UInt32Array,
};
use arrow_buffer::NullBuffer;
use arrow_row::{RowConverter, Rows, SortField};
use arrow_schema::{DataType, Field, Schema, SchemaRef, SortOptions, TimeUnit};
use arrow_select::concat::concat_batches;
use arrow_select::take::take;
use sdm_contract::catalog::CatalogAnswer;
use sdm_contract::data::{
    BATCH_COLUMN, DataRequest, DescribePart, ExpectedSchema, FallbackLimits, Filter,
    MATCHES_COLUMN, ProviderError, Relation, SEED_BATCH, TieBreak, Window, refused, seed_integer,
    seed_key_columns,
};
use sdm_contract::{CoreError, Detail, Scalar, Sentence};
use sdm_engine::random::{Entropy, default_rng};

use crate::run::{ArrowData, Asker, Budget, FetchResult, Reply, internal, schema_mismatch};

/// The answer to `request`, an optional request the run's provider does not
/// declare, within `limits`: the rows the fallback computed, or the
/// provider's failure of a scan it asked. Any other failure ends the run at
/// its next step ([`Asker::fail`]); the request is then never answered.
pub(crate) async fn answer(
    asker: Asker,
    request: DataRequest,
    limits: FallbackLimits,
    seeds: Option<ArrowData>,
) -> Reply {
    let mut held = Held {
        asker: &asker,
        request: &request,
        limits,
        rows: 0,
        bytes: 0,
    };
    match answered(&mut held, seeds).await {
        Ok(data) => Reply::Fetch(FetchResult::Data(data)),
        Err(Failed::Provider(error)) => Reply::Fetch(FetchResult::Error(error)),
        Err(Failed::Core(error)) => {
            asker.fail(error);
            std::future::pending().await
        }
    }
}

/// Why the fallback gave no answer: a scan the provider failed, or the
/// fallback's own refusal.
enum Failed {
    Provider(ProviderError),
    Core(CoreError),
}

impl From<CoreError> for Failed {
    fn from(error: CoreError) -> Self {
        Failed::Core(error)
    }
}

/// What the fallback has read and holds for one request, against its
/// limits.
struct Held<'a> {
    asker: &'a Asker,
    request: &'a DataRequest,
    limits: FallbackLimits,
    rows: u64,
    bytes: u64,
}

/// A scan's rows, as one batch, with the schema the provider handed them
/// over in (`ArrowData::received`).
struct Scanned {
    batch: RecordBatch,
    received: SchemaRef,
}

impl Held<'_> {
    /// The rows of `columns` of `table` meeting `filters`, at most `most`
    /// of them where given: asked as a `scan` of at most one row past what
    /// is left of the row limit.
    async fn scan(
        &mut self,
        table: &Relation,
        columns: Vec<String>,
        filters: Vec<Filter>,
        most: Option<u64>,
    ) -> Result<Scanned, Failed> {
        let left = self.limits.max_rows.saturating_sub(self.rows);
        let limit = left.saturating_add(1).min(most.unwrap_or(u64::MAX));
        // Held to what is left of both limits as each page arrives.
        let budget = Budget {
            rows: left,
            bytes: self.limits.max_bytes.saturating_sub(self.bytes),
            past_rows: over(self.request, "max_rows", self.limits.max_rows),
            past_bytes: over(self.request, "max_bytes", self.limits.max_bytes),
        };
        let asked = self
            .asker
            .fetch_within(
                DataRequest::Scan {
                    table: table.clone(),
                    columns,
                    filters,
                    limit: Some(limit),
                },
                budget,
            )?
            .await;
        let data = match asked {
            Reply::Fetch(FetchResult::Data(data)) => data,
            Reply::Fetch(FetchResult::Error(error)) => return Err(Failed::Provider(error)),
            _ => {
                return Err(internal("a scan the fallback asked is not answered with data").into());
            }
        };
        let batch = concat_batches(&data.schema, &data.batches)
            .map_err(|_| internal("a scan's pages do not make one batch"))?;
        self.rows = self.rows.saturating_add(batch.num_rows() as u64);
        if self.rows > self.limits.max_rows {
            return Err(over(self.request, "max_rows", self.limits.max_rows).into());
        }
        self.hold(&batch)?;
        let received = if data.received.fields().len() == batch.num_columns() {
            data.received
        } else {
            batch.schema()
        };
        Ok(Scanned { batch, received })
    }

    /// Holds `batch`'s Arrow data against the byte limit.
    fn hold(&mut self, batch: &RecordBatch) -> Result<(), CoreError> {
        self.bytes = self
            .bytes
            .saturating_add(batch.get_array_memory_size() as u64);
        if self.bytes > self.limits.max_bytes {
            return Err(over(self.request, "max_bytes", self.limits.max_bytes));
        }
        Ok(())
    }

    /// `answer`, held against the byte limit, as the request's answer.
    fn finish(&mut self, answer: Out) -> Result<ArrowData, Failed> {
        let data = answer.finish()?;
        for batch in &data.batches {
            self.hold(batch)?;
        }
        Ok(data)
    }
}

/// The fallback's answer to `held`'s request.
async fn answered(held: &mut Held<'_>, seeds: Option<ArrowData>) -> Result<ArrowData, Failed> {
    let request = held.request;
    match request {
        DataRequest::Preview {
            table,
            columns,
            limit,
        } => {
            named(request, "columns", columns)?;
            let scanned = held
                .scan(table, unique(columns), vec![], Some(*limit))
                .await?;
            let rows = scanned.batch.num_rows().min(usize_of(*limit));
            let kept: Vec<u32> = (0..rows).map(position).collect();
            let mut out = Out::default();
            out.project(&scanned, columns, &kept)?;
            held.finish(out)
        }
        DataRequest::ColumnRanges { table, columns } => {
            named(request, "columns", columns)?;
            let scanned = held.scan(table, unique(columns), vec![], None).await?;
            let mut out = Out::default();
            out.push_count("count", scanned.batch.num_rows());
            for name in columns {
                let at = scanned.index(name)?;
                let column = scanned.batch.column(at);
                let present = all_valid(std::slice::from_ref(column));
                let rows = ordered_rows(request, "columns", std::slice::from_ref(column))?;
                let mut held_rows: Vec<u32> = (0..column.len())
                    .filter(|&row| holds(&present, row))
                    .map(position)
                    .collect();
                held_rows.sort_by(|&a, &b| rows.row(a as usize).cmp(&rows.row(b as usize)));
                for (label, pick) in [("min", held_rows.first()), ("max", held_rows.last())] {
                    let picked = take(column, &UInt32Array::from(vec![pick.copied()]), None)
                        .map_err(|_| internal("a column's extreme value cannot be taken"))?;
                    let field = scanned.batch.schema_ref().field(at).clone();
                    let received = scanned.received.field(at).clone();
                    out.push(
                        field.with_name(label).with_nullable(true),
                        received.with_name(label).with_nullable(true),
                        picked,
                    );
                }
                out.push_count("count", held_rows.len());
            }
            held.finish(out)
        }
        DataRequest::RepeatedKey { table, key } => {
            named(request, "key", key)?;
            let scanned = held.scan(table, unique(key), vec![], None).await?;
            let arrays = scanned.columns(key)?;
            let present = all_valid(&arrays);
            let rows = equal_rows(request, "key", &arrays)?;
            let mut seen = BTreeSet::new();
            let repeated = (0..scanned.batch.num_rows())
                .filter(|&row| holds(&present, row))
                .find(|&row| !seen.insert(rows.row(row)));
            let kept: Vec<u32> = repeated.into_iter().map(position).collect();
            let mut out = Out::default();
            out.project(&scanned, key, &kept)?;
            held.finish(out)
        }
        DataRequest::KeyCounts { keys } => {
            let (mut positions, mut counts) = (vec![], [vec![], vec![], vec![]]);
            for (at, key) in keys.iter().enumerate() {
                let scanned = held
                    .scan(&key.table, unique(&key.columns), vec![], None)
                    .await?;
                let arrays = scanned.columns(&key.columns)?;
                let present = all_valid(&arrays);
                let rows = equal_rows(request, "keys", &arrays)?;
                let total = scanned.batch.num_rows();
                let distinct: BTreeSet<_> = (0..total)
                    .filter(|&row| holds(&present, row))
                    .map(|row| rows.row(row))
                    .collect();
                let missing = (0..total).filter(|&row| !holds(&present, row)).count();
                // Labelled as the dialects label it: its label, where it
                // gives one, else its position (T2.12).
                positions
                    .push(i32::try_from(key.label_at(at)).map_err(|_| internal("too many keys"))?);
                for (count, value) in counts.iter_mut().zip([total, distinct.len(), missing]) {
                    count.push(i64_of(value)?);
                }
            }
            let [n, d, z] = counts;
            let columns: Vec<ArrayRef> = vec![
                Arc::new(Int32Array::from(positions)),
                Arc::new(Int64Array::from(n)),
                Arc::new(Int64Array::from(d)),
                Arc::new(Int64Array::from(z)),
            ];
            held.finish(Out::fixed(request, columns)?)
        }
        DataRequest::TimeBounds { columns } => {
            let (mut positions, mut lows, mut highs) = (vec![], vec![], vec![]);
            for (at, column) in columns.iter().enumerate() {
                let scanned = held
                    .scan(&column.table, vec![column.column.clone()], vec![], None)
                    .await?;
                let index = scanned.index(&column.column)?;
                let times = wall_us(request, "columns", scanned.batch.column(index))?;
                positions.push(i32::try_from(at).map_err(|_| internal("too many columns"))?);
                lows.push(times.iter().flatten().min().copied());
                highs.push(times.iter().flatten().max().copied());
            }
            let columns: Vec<ArrayRef> = vec![
                Arc::new(Int32Array::from(positions)),
                Arc::new(TimestampMicrosecondArray::from(lows)),
                Arc::new(TimestampMicrosecondArray::from(highs)),
            ];
            held.finish(Out::fixed(request, columns)?)
        }
        DataRequest::SampleEntities {
            table,
            key,
            n,
            seed,
            columns,
            filters,
            ..
        } => {
            named(request, "columns", columns)?;
            if key.is_empty() || key.iter().any(|k| !columns.contains(k)) {
                return Err(refused(
                    request,
                    "key",
                    "a sample's key is not among the columns it projects",
                )
                .into());
            }
            let seed = seed_integer(seed)
                .and_then(|seed| u128::try_from(seed).ok())
                .ok_or_else(|| {
                    refused(request, "seed", "a sample's seed is a non-negative integer")
                })?;
            let projected = unique(columns);
            let scanned = held
                .scan(table, projected.clone(), filters.clone(), None)
                .await?;
            // A defined ordering of the same data: every projected column,
            // by the row format's order, so equal rows are equal in all.
            let rows = ordered_rows(request, "columns", &scanned.columns(&projected)?)?;
            let mut drawn: Vec<u32> = (0..scanned.batch.num_rows()).map(position).collect();
            drawn.sort_by(|&a, &b| rows.row(a as usize).cmp(&rows.row(b as usize)));
            default_rng(&Entropy::from_u128(seed)).shuffle(&mut drawn);
            drawn.truncate(usize_of(*n));
            let mut out = Out::default();
            out.project(&scanned, columns, &drawn)?;
            held.finish(out)
        }
        DataRequest::Neighbors {
            table,
            key,
            columns,
            window,
            per_seed,
            order,
            ..
        } => {
            let limit = match per_seed {
                Some(sdm_contract::data::NeighborLimit::Unsigned(limit)) => Some(*limit),
                // This non-SQL provider uses a row-position bound: no
                // nonnegative position meets a negative bound. SQL
                // providers keep their own native LIMIT semantics.
                Some(sdm_contract::data::NeighborLimit::Negative { .. }) => Some(0),
                None => None,
                Some(sdm_contract::data::NeighborLimit::SqlExpression { .. }) => {
                    return Err(not_kept(
                        Sentence::FallbackSqlExpression,
                        request,
                        "per_seed",
                        "the in-memory fallback does not execute caller-authored SQL expressions",
                    )
                    .into());
                }
            };
            named(request, "columns", columns)?;
            let seeds = seeds_of(held, seeds)?;
            let mut asked = key.clone();
            asked.extend(columns.iter().cloned());
            asked.extend(window.iter().map(|w| w.column.clone()));
            if let Some(TieBreak(terms)) = order {
                asked.extend(terms.iter().map(|t| t.column.clone()));
            }
            let scanned = held.scan(table, unique(&asked), vec![], None).await?;
            let mut pairs = joined(request, &scanned, key, &seeds)?;
            if let Some(window) = window {
                pairs = within(request, &scanned, &seeds, window, pairs)?;
            }
            if let (Some(limit), Some(TieBreak(terms))) = (limit, order) {
                pairs = first_per_seed(request, &scanned, &seeds, terms, limit, pairs)?;
            }
            let mut out = Out::default();
            out.batch(&seeds, &pairs)?;
            let rows: Vec<u32> = pairs.iter().map(|&(_, row)| row).collect();
            out.project(&scanned, columns, &rows)?;
            held.finish(out)
        }
        DataRequest::Lookup {
            table,
            key,
            columns,
            ..
        } => {
            named(request, "columns", columns)?;
            let seeds = seeds_of(held, seeds)?;
            let mut asked = key.clone();
            asked.extend(columns.iter().cloned());
            let scanned = held.scan(table, unique(&asked), vec![], None).await?;
            let pairs = joined(request, &scanned, key, &seeds)?;
            // One row per batch holding a match, with the count of matches
            // over the batch, in the order the batches first match.
            let batches = batch_rows(request, &seeds)?;
            let mut first: Vec<(u32, u32)> = vec![];
            let mut matches: BTreeMap<_, (usize, i64)> = BTreeMap::new();
            for &(seed, row) in &pairs {
                let entry = matches
                    .entry(batches.row(seed as usize))
                    .or_insert_with(|| {
                        first.push((seed, row));
                        (first.len() - 1, 0)
                    });
                entry.1 += 1;
            }
            let mut counts = vec![0; first.len()];
            for (at, count) in matches.into_values() {
                counts[at] = count;
            }
            let mut out = Out::default();
            out.batch(&seeds, &first)?;
            out.push(
                Field::new(MATCHES_COLUMN, DataType::Int64, false),
                Field::new(MATCHES_COLUMN, DataType::Int64, false),
                Arc::new(Int64Array::from(counts)),
            );
            let rows: Vec<u32> = first.iter().map(|&(_, row)| row).collect();
            out.project(&scanned, columns, &rows)?;
            held.finish(out)
        }
        DataRequest::CountRows { table, filters } => {
            // The columns the filters read, or, with none, the table's
            // first: a scan names some.
            let mut columns: Vec<String> = vec![];
            for filter in filters {
                match filter {
                    Filter::KeyIn { columns: named, .. } => columns.extend(named.iter().cloned()),
                    Filter::TimeRange { column, .. } | Filter::NotNull { column } => {
                        columns.push(column.clone())
                    }
                }
            }
            if columns.is_empty() {
                columns.push(first_column(held, table).await?);
            }
            let scanned = held
                .scan(table, unique(&columns), filters.clone(), None)
                .await?;
            let mut out = Out::default();
            out.push_count("rows", scanned.batch.num_rows());
            held.finish(out)
        }
        DataRequest::Describe { .. } | DataRequest::Scan { .. } => {
            Err(internal("a request every provider answers reached the fallback").into())
        }
    }
}

/// The first column of `table`, as the provider's catalog lists it.
async fn first_column(held: &Held<'_>, table: &Relation) -> Result<String, Failed> {
    let request = DataRequest::Describe {
        tables: vec![table.clone()],
        part: DescribePart::Columns,
    };
    let data = match held.asker.fetch(request.clone())?.await {
        Reply::Fetch(FetchResult::Data(data)) => data,
        Reply::Fetch(FetchResult::Error(error)) => return Err(Failed::Provider(error)),
        _ => {
            return Err(internal("a describe the fallback asked is not answered with data").into());
        }
    };
    match held.asker.read(&request, &data)? {
        CatalogAnswer::Columns(columns) => columns
            .into_iter()
            .next()
            .map(|column| column.name)
            .ok_or_else(|| {
                not_kept(
                    Sentence::FallbackNoColumn,
                    held.request,
                    "table",
                    "the table has no column to count rows by",
                )
                .into()
            }),
        _ => Err(internal("a describe of columns answered another part").into()),
    }
}

/// A request's seeds, as one batch, held against the byte limit.
fn seeds_of(held: &mut Held<'_>, seeds: Option<ArrowData>) -> Result<RecordBatch, CoreError> {
    let seeds =
        seeds.ok_or_else(|| internal("a request with seeds reached the fallback without them"))?;
    let batch = concat_batches(&seeds.schema, &seeds.batches)
        .map_err(|_| internal("the seeds do not make one batch"))?;
    held.hold(&batch)?;
    Ok(batch)
}

/// Each seed's row and each row of `scanned` holding its key, every part
/// of it present and equal, seed by seed, the scan's order within each:
/// the join of `neighbors` and `lookup`. A key is joined only to seed
/// columns of its own type, compared by value as the row format holds it.
fn joined(
    request: &DataRequest,
    scanned: &Scanned,
    key: &[String],
    seeds: &RecordBatch,
) -> Result<Vec<(u32, u32)>, CoreError> {
    let table = scanned.columns(key)?;
    let seed = seed_key_columns(key.len())
        .iter()
        .map(|name| seed_column(seeds, name))
        .collect::<Result<Vec<_>, _>>()?;
    let mut table = table;
    let mut seed = seed;
    for (t, s) in table.iter_mut().zip(seed.iter_mut()) {
        if t.data_type() == s.data_type() {
            continue;
        }
        // Integers of two widths, compared as the exact integers they are.
        let (Some(widened_t), Some(widened_s)) = (integers(t), integers(s)) else {
            return Err(not_kept(
                Sentence::FallbackSeedTypes,
                request,
                "seeds",
                "the in-memory fallback joins a key only to seeds of its own type, or integers to integers",
            ));
        };
        *t = widened_t;
        *s = widened_s;
    }
    let table: Vec<ArrayRef> = table.iter().map(compared).collect();
    let seed: Vec<ArrayRef> = seed.iter().map(compared).collect();
    let converter = converter(request, "key", &table)?;
    let table_rows = convert(request, "key", &converter, &table)?;
    let seed_rows = convert(request, "key", &converter, &seed)?;
    let table_present = all_valid(&table);
    let seed_present = all_valid(&seed);
    let mut holding: BTreeMap<_, Vec<u32>> = BTreeMap::new();
    for row in (0..scanned.batch.num_rows()).filter(|&row| holds(&table_present, row)) {
        holding
            .entry(table_rows.row(row))
            .or_default()
            .push(position(row));
    }
    let mut pairs = vec![];
    for at in (0..seeds.num_rows()).filter(|&at| holds(&seed_present, at)) {
        if let Some(rows) = holding.get(&seed_rows.row(at)) {
            pairs.extend(rows.iter().map(|&row| (position(at), row)));
        }
    }
    Ok(pairs)
}

/// An integer column (of any width, signed or not, or a decimal of scale
/// 0) as the `decimal128(38, 0)` of its exact values; `None` for any other.
fn integers(array: &ArrayRef) -> Option<ArrayRef> {
    use arrow_array::types::{
        Int8Type, Int16Type, Int32Type, Int64Type, UInt8Type, UInt16Type, UInt32Type, UInt64Type,
    };
    macro_rules! exact {
        ($t:ty) => {
            array
                .as_primitive::<$t>()
                .iter()
                .map(|v| v.map(i128::from))
                .collect::<Decimal128Array>()
        };
    }
    let values = match array.data_type() {
        DataType::Int8 => exact!(Int8Type),
        DataType::Int16 => exact!(Int16Type),
        DataType::Int32 => exact!(Int32Type),
        DataType::Int64 => exact!(Int64Type),
        DataType::UInt8 => exact!(UInt8Type),
        DataType::UInt16 => exact!(UInt16Type),
        DataType::UInt32 => exact!(UInt32Type),
        DataType::UInt64 => exact!(UInt64Type),
        DataType::Decimal128(_, 0) => array
            .as_primitive::<arrow_array::types::Decimal128Type>()
            .iter()
            .collect::<Decimal128Array>(),
        _ => return None,
    };
    Some(Arc::new(values.with_precision_and_scale(38, 0).ok()?))
}

/// The pairs whose row's window column, read as a time, is within its
/// seed's bounds; a null on either side is within none.
fn within(
    request: &DataRequest,
    scanned: &Scanned,
    seeds: &RecordBatch,
    window: &Window,
    pairs: Vec<(u32, u32)>,
) -> Result<Vec<(u32, u32)>, CoreError> {
    let times = wall_us(
        request,
        "window",
        scanned.batch.column(scanned.index(&window.column)?),
    )?;
    let mut bounds = vec![];
    for (bound, lower) in [(&window.lower, true), (&window.upper, false)] {
        let Some(bound) = bound else { continue };
        let column = seed_column(seeds, &bound.seed_column)?;
        let DataType::Timestamp(TimeUnit::Nanosecond, None) = column.data_type() else {
            return Err(internal(
                "a window bound's seed column is not a timestamp[ns]",
            ));
        };
        let values: Vec<Option<i64>> = column
            .as_primitive::<TimestampNanosecondType>()
            .iter()
            .collect();
        bounds.push((values, lower, bound.inclusive));
    }
    Ok(pairs
        .into_iter()
        .filter(|&(seed, row)| {
            let Some(time) = times[row as usize] else {
                return false;
            };
            let time = i128::from(time) * 1_000;
            bounds.iter().all(|(values, lower, inclusive)| {
                let Some(bound) = values[seed as usize].map(i128::from) else {
                    return false;
                };
                match (lower, inclusive) {
                    (true, true) => time >= bound,
                    (true, false) => time > bound,
                    (false, true) => time <= bound,
                    (false, false) => time < bound,
                }
            })
        })
        .collect())
}

/// At most `limit` pairs of each batch, the first in the request's order
/// (`order`'s terms over the pair's row, each read as a time where it says
/// so), the scan's order breaking ties, ordered by batch, then that order.
fn first_per_seed(
    request: &DataRequest,
    scanned: &Scanned,
    seeds: &RecordBatch,
    terms: &[sdm_contract::data::TieTerm],
    limit: u64,
    pairs: Vec<(u32, u32)>,
) -> Result<Vec<(u32, u32)>, CoreError> {
    let seed_at = UInt32Array::from(pairs.iter().map(|&(seed, _)| seed).collect::<Vec<_>>());
    let row_at = UInt32Array::from(pairs.iter().map(|&(_, row)| row).collect::<Vec<_>>());
    let taken = |array: &ArrayRef, at: &UInt32Array| {
        take(array, at, None).map_err(|_| internal("a pair's value cannot be taken"))
    };
    let mut fields = vec![SortField::new(
        seed_column(seeds, SEED_BATCH)?.data_type().clone(),
    )];
    let mut sorted = vec![taken(&seed_column(seeds, SEED_BATCH)?, &seed_at)?];
    for term in terms {
        let column = scanned.batch.column(scanned.index(&term.column)?);
        let values = if term.as_timestamp {
            let times = wall_us(request, "order", column)?;
            let times: ArrayRef = Arc::new(TimestampMicrosecondArray::from(times));
            taken(&times, &row_at)?
        } else {
            taken(&compared(column), &row_at)?
        };
        fields.push(SortField::new_with_options(
            values.data_type().clone(),
            SortOptions {
                descending: term.descending,
                nulls_first: !term.nulls_last,
            },
        ));
        sorted.push(values);
    }
    let converter = RowConverter::new(fields).map_err(|_| unordered(request, "order"))?;
    let rows = convert(request, "order", &converter, &sorted)?;
    let batch = convert(
        request,
        "order",
        &RowConverter::new(vec![SortField::new(sorted[0].data_type().clone())])
            .map_err(|_| unordered(request, "order"))?,
        &sorted[..1],
    )?;
    let mut order: Vec<usize> = (0..pairs.len()).collect();
    // Stable: equal rows keep the scan's order.
    order.sort_by(|&a, &b| rows.row(a).cmp(&rows.row(b)));
    let mut kept = vec![];
    let mut run: Option<(usize, u64)> = None;
    for at in order {
        let count = match run {
            Some((last, count)) if batch.row(last) == batch.row(at) => count + 1,
            _ => 1,
        };
        run = Some((at, count));
        if count <= limit {
            kept.push(pairs[at]);
        }
    }
    Ok(kept)
}

/// The row format of each seed's batch position, to group by.
fn batch_rows(request: &DataRequest, seeds: &RecordBatch) -> Result<Rows, CoreError> {
    let batch = [seed_column(seeds, SEED_BATCH)?];
    let converter = converter(request, "seeds", &batch)?;
    convert(request, "seeds", &converter, &batch)
}

/// The seed column `name`; the request's check already holds its presence.
fn seed_column(seeds: &RecordBatch, name: &str) -> Result<ArrayRef, CoreError> {
    seeds
        .column_by_name(name)
        .cloned()
        .ok_or_else(|| internal("the seeds lack a column the request names"))
}

/// `array`'s values in microseconds, as `CAST(column AS TIMESTAMP)` reads
/// them: a zone-less timestamp of any unit (a finer one truncated toward
/// zero) or a date at its midnight; any other type is not read, by name.
fn wall_us(
    request: &DataRequest,
    field: &str,
    array: &ArrayRef,
) -> Result<Vec<Option<i64>>, CoreError> {
    let past = || {
        not_kept(
            Sentence::FallbackTimePast,
            request,
            field,
            "a time is past what microseconds hold",
        )
    };
    let scaled = |values: Vec<Option<i64>>, per: i64| {
        values
            .into_iter()
            .map(|v| v.map(|v| v.checked_mul(per).ok_or_else(past)).transpose())
            .collect::<Result<Vec<_>, _>>()
    };
    match array.data_type() {
        DataType::Timestamp(unit, None) => match unit {
            TimeUnit::Second => scaled(
                array.as_primitive::<TimestampSecondType>().iter().collect(),
                1_000_000,
            ),
            TimeUnit::Millisecond => scaled(
                array
                    .as_primitive::<TimestampMillisecondType>()
                    .iter()
                    .collect(),
                1_000,
            ),
            TimeUnit::Microsecond => Ok(array
                .as_primitive::<TimestampMicrosecondType>()
                .iter()
                .collect()),
            TimeUnit::Nanosecond => Ok(array
                .as_primitive::<TimestampNanosecondType>()
                .iter()
                .map(|v| v.map(|v| v / 1_000))
                .collect()),
        },
        DataType::Date32 => scaled(
            array
                .as_primitive::<arrow_array::types::Date32Type>()
                .iter()
                .map(|v| v.map(i64::from))
                .collect(),
            86_400_000_000,
        ),
        DataType::Date64 => scaled(
            array
                .as_primitive::<arrow_array::types::Date64Type>()
                .iter()
                .collect(),
            1_000,
        ),
        _ => Err(not_kept(
            Sentence::FallbackTimeType,
            request,
            field,
            "the in-memory fallback reads a time only from a zone-less timestamp or a date",
        )),
    }
}

/// The answer the fallback builds: its fields as admitted and as received,
/// and its columns.
#[derive(Default)]
struct Out {
    fields: Vec<Field>,
    received: Vec<Field>,
    columns: Vec<ArrayRef>,
}

impl Out {
    fn push(&mut self, field: Field, received: Field, column: ArrayRef) {
        self.fields.push(field);
        self.received.push(received);
        self.columns.push(column);
    }

    /// A count the pushdown gives as a non-null 64-bit integer.
    fn push_count(&mut self, name: &str, count: usize) {
        let field = Field::new(name, DataType::Int64, false);
        let count = i64::try_from(count).unwrap_or(i64::MAX);
        self.push(
            field.clone(),
            field,
            Arc::new(Int64Array::from(vec![count])),
        );
    }

    /// `columns` of `scanned`, each by its name, at `rows`, in order, with
    /// the fields the scan's admission gave them.
    fn project(
        &mut self,
        scanned: &Scanned,
        columns: &[String],
        rows: &[u32],
    ) -> Result<(), CoreError> {
        let at = UInt32Array::from(rows.to_vec());
        for name in columns {
            let index = scanned.index(name)?;
            let column = take(scanned.batch.column(index), &at, None)
                .map_err(|_| internal("a scanned column's rows cannot be taken"))?;
            self.push(
                scanned.batch.schema_ref().field(index).clone(),
                scanned.received.field(index).clone(),
                column,
            );
        }
        Ok(())
    }

    /// Each pair's seed batch position, under the reference's name.
    fn batch(&mut self, seeds: &RecordBatch, pairs: &[(u32, u32)]) -> Result<(), CoreError> {
        let batch = seed_column(seeds, SEED_BATCH)?;
        let at = UInt32Array::from(pairs.iter().map(|&(seed, _)| seed).collect::<Vec<_>>());
        let column = take(&batch, &at, None)
            .map_err(|_| internal("a seed's batch position cannot be taken"))?;
        let field = Field::new(BATCH_COLUMN, batch.data_type().clone(), false);
        self.push(field.clone(), field, column);
        Ok(())
    }

    /// The contract's fixed columns of `request`'s answer, holding
    /// `columns`.
    fn fixed(request: &DataRequest, columns: Vec<ArrayRef>) -> Result<Out, CoreError> {
        let ExpectedSchema::Schema { fields } = request.expected() else {
            return Err(internal("a request of fixed columns expects none"));
        };
        let fields: Vec<Field> = fields.iter().map(|f| f.to_arrow()).collect();
        Ok(Out {
            received: fields.clone(),
            fields,
            columns,
        })
    }

    fn finish(self) -> Result<ArrowData, CoreError> {
        let schema = Arc::new(Schema::new(self.fields));
        let batch = RecordBatch::try_new(Arc::clone(&schema), self.columns)
            .map_err(|_| internal("the fallback's answer does not make a batch"))?;
        Ok(ArrowData {
            schema,
            batches: vec![batch],
            received: Arc::new(Schema::new(self.received)),
        })
    }
}

impl Scanned {
    /// The position of the scanned column `name`; a scan answered without
    /// a column it asked for is the provider's mismatch.
    fn index(&self, name: &str) -> Result<usize, CoreError> {
        self.batch.schema_ref().index_of(name).map_err(|_| {
            schema_mismatch(
                "expected",
                None,
                "a scan's answer lacks a column it asked for",
            )
        })
    }

    fn columns(&self, names: &[String]) -> Result<Vec<ArrayRef>, CoreError> {
        names
            .iter()
            .map(|name| Ok(Arc::clone(self.batch.column(self.index(name)?))))
            .collect()
    }
}

/// The refusal of a request past the fallback's declared `limit`.
fn over(request: &DataRequest, limit: &str, most: u64) -> CoreError {
    CoreError::new(
        Sentence::FallbackLimit,
        "the request is past a limit of the in-memory fallback the provider declares",
    )
    .with_field("request", Detail::text(request.name()))
    .with_field("limit", Detail::text(limit))
    .with_field("most", Detail::Scalar(Scalar::UInt64(most)))
}

/// The refusal of what the fallback does not read, naming the request's
/// `field`.
fn not_kept(sentence: Sentence, request: &DataRequest, field: &str, message: &str) -> CoreError {
    CoreError::new(sentence, message)
        .with_field("request", Detail::text(request.name()))
        .with_field("field", Detail::text(field))
}

/// The refusal of a column the fallback does not order, naming the
/// request's `field`.
fn unordered(request: &DataRequest, field: &str) -> CoreError {
    not_kept(
        Sentence::FallbackUnordered,
        request,
        field,
        "the in-memory fallback does not order a column of this type",
    )
}

/// Refuses a request naming no column where it must name some, as each
/// dialect refuses it.
fn named(request: &DataRequest, field: &str, names: &[String]) -> Result<(), CoreError> {
    if names.is_empty() {
        return Err(refused(request, field, "the request names no columns"));
    }
    Ok(())
}

/// `names` without repeats, in order: what one scan asks for.
fn unique(names: &[String]) -> Vec<String> {
    let mut seen = BTreeSet::new();
    names
        .iter()
        .filter(|name| seen.insert(name.as_str()))
        .cloned()
        .collect()
}

fn converter(
    request: &DataRequest,
    field: &str,
    arrays: &[ArrayRef],
) -> Result<RowConverter, CoreError> {
    RowConverter::new(
        arrays
            .iter()
            .map(|a| SortField::new(a.data_type().clone()))
            .collect(),
    )
    .map_err(|_| unordered(request, field))
}

fn convert(
    request: &DataRequest,
    field: &str,
    converter: &RowConverter,
    arrays: &[ArrayRef],
) -> Result<Rows, CoreError> {
    converter
        .convert_columns(arrays)
        .map_err(|_| unordered(request, field))
}

/// `arrays`' rows in the row format, each value as SQL compares it
/// ([`compared`]): to group, count distinct and find repeats by.
fn equal_rows(request: &DataRequest, field: &str, arrays: &[ArrayRef]) -> Result<Rows, CoreError> {
    let equal: Vec<ArrayRef> = arrays.iter().map(compared).collect();
    ordered_rows(request, field, &equal)
}

/// `array` with the values SQL holds equal written alike: a float's `-0.0`
/// as `0.0`, and every NaN as one NaN, as DuckDB and Postgres compare them
/// (SQLite stores a NaN as null). Arrow's row format orders by the bits,
/// telling them apart. Any other array is as it is.
fn compared(array: &ArrayRef) -> ArrayRef {
    use arrow_array::types::{Float32Type, Float64Type};
    macro_rules! alike {
        ($t:ty, $f:ty) => {
            Arc::new(array.as_primitive::<$t>().unary::<_, $t>(|v: $f| {
                if v == 0.0 {
                    0.0
                } else if v.is_nan() {
                    <$f>::NAN
                } else {
                    v
                }
            })) as ArrayRef
        };
    }
    match array.data_type() {
        DataType::Float32 => alike!(Float32Type, f32),
        DataType::Float64 => alike!(Float64Type, f64),
        _ => Arc::clone(array),
    }
}

/// `arrays`' rows in the row format, ascending, nulls first.
fn ordered_rows(
    request: &DataRequest,
    field: &str,
    arrays: &[ArrayRef],
) -> Result<Rows, CoreError> {
    convert(request, field, &converter(request, field, arrays)?, arrays)
}

/// The logical nulls of each of `arrays`, for [`holds`].
fn all_valid(arrays: &[ArrayRef]) -> Vec<Option<NullBuffer>> {
    arrays.iter().map(|array| array.logical_nulls()).collect()
}

/// Whether `row` holds a value in every array `present` was read from.
fn holds(present: &[Option<NullBuffer>], row: usize) -> bool {
    present
        .iter()
        .all(|nulls| nulls.as_ref().is_none_or(|nulls| nulls.is_valid(row)))
}

/// A row's position as the `u32` Arrow's `take` reads; a batch holds fewer
/// rows than that counts.
fn position(row: usize) -> u32 {
    u32::try_from(row).unwrap_or(u32::MAX)
}

fn usize_of(value: u64) -> usize {
    usize::try_from(value).unwrap_or(usize::MAX)
}

fn i64_of(value: usize) -> Result<i64, CoreError> {
    i64::try_from(value).map_err(|_| internal("a count past a 64-bit integer"))
}
