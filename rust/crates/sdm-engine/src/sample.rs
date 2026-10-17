// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The reference's sampler, as far as `materialize` reaches it at T1.11c:
//! the examples' entities and anchor times (`sample_target`'s temporal
//! walk; their labels are [`crate::labels`]'s from T1.27), the seeds a
//! `neighbors` or `lookup` request carries, the order `_by_fkey` keeps neighbors in, and, from T1.26b, the
//! sampler's three reads as typed requests ([`by_pkey`], [`by_fkey`],
//! [`by_time`]). It holds no SQL: what it asks for is typed requests,
//! which `sdm-data` plans.

use std::sync::Arc;

use arrow_array::{ArrayRef, Int64Array, RecordBatch, TimestampNanosecondArray};
use arrow_schema::{DataType, Field, Schema, TimeUnit as ArrowUnit};
use sdm_contract::data::{
    DataRequest, KeySet, MARKED, NullOrder, SEED_BATCH, SEED_COUNT_UNREAD, SEED_KEY, SEED_KEY_TYPE,
    SEED_KEY_UNIQUE, SeedBound, TieBreak, TieTerm,
};
use sdm_contract::{ColumnType, CoreError, Detail, ErrorCode, ErrorKind};

use crate::graph::Table;
use crate::random::Pcg64;

/// The seed time columns the sampler's requests carry: each seed's anchor
/// (`_by_fkey`'s `__kumo_time__`), and the start and end of its window
/// (`_by_time`'s `__kumo_start__` and `__kumo_end__`).
pub const SEED_TIME: &str = "time";
pub const SEED_START: &str = "start";
pub const SEED_END: &str = "end";

/// A time past what a nanosecond count since the epoch holds, which the
/// reference's pandas times cannot hold either.
pub fn out_of_range(field: &str) -> CoreError {
    CoreError::new(
        ErrorKind::Unsupported,
        ErrorCode::ValueOutOfRange,
        "a time is outside what nanoseconds since the epoch hold",
    )
    .with_field("field", Detail::text(field))
}

/// `time + span`, checked.
pub fn shift(time: i64, span: i128, field: &str) -> Result<i64, CoreError> {
    i64::try_from(i128::from(time) + span).map_err(|_| out_of_range(field))
}

/// The examples a temporal query collects: the position of each example's
/// entity among the sampled entities, and its anchor time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Examples {
    pub index: Vec<usize>,
    pub anchors: Vec<i64>,
}

/// `sample_target`'s walk back in time for a temporal query: at each
/// anchor, from `anchor` back by `step` while the anchor is not before
/// `earliest`, the entities valid then (all of them, or those whose time
/// is not after the anchor), shuffled by `rng` and cut to `trials`, until
/// `trials` examples are collected; then cut to `trials`.
pub fn temporal_examples(
    rng: &mut Pcg64,
    entities: usize,
    entity_time: Option<&[Option<i64>]>,
    anchor: i64,
    step: i128,
    earliest: i64,
    trials: usize,
) -> Result<Examples, CoreError> {
    let mut index = Vec::new();
    let mut anchors = Vec::new();
    let mut at = anchor;
    loop {
        let mut valid: Vec<usize> = match entity_time {
            None => (0..entities).collect(),
            Some(times) => times
                .iter()
                .enumerate()
                .filter(|(_, t)| t.is_some_and(|t| t <= at))
                .map(|(i, _)| i)
                .collect(),
        };
        rng.shuffle(&mut valid);
        valid.truncate(trials);
        anchors.extend(std::iter::repeat_n(at, valid.len()));
        index.extend(valid);
        if index.len() >= trials {
            break;
        }
        at = shift(at, -step, "anchor_time")?;
        if at < earliest {
            break;
        }
    }
    index.truncate(trials);
    anchors.truncate(trials);
    Ok(Examples { index, anchors })
}

/// `sample_target`'s examples for a static query, which shares its
/// examples between context and prediction: the entities valid at `anchor`
/// (all of them, or those whose time is not after it), shuffled by `rng`,
/// each at `anchor`.
pub fn static_examples(
    rng: &mut Pcg64,
    entities: usize,
    entity_time: Option<&[Option<i64>]>,
    anchor: i64,
) -> Examples {
    let mut index: Vec<usize> = match entity_time {
        None => (0..entities).collect(),
        Some(times) => times
            .iter()
            .enumerate()
            .filter(|(_, t)| t.is_some_and(|t| t <= anchor))
            .map(|(i, _)| i)
            .collect(),
    };
    rng.shuffle(&mut index);
    let anchors = vec![anchor; index.len()];
    Examples { index, anchors }
}

/// The seeds of a `neighbors` or `lookup` request: each seed's key
/// (`keys`, whose type is the key column's own), its batch position from
/// 0, and the seed time columns named, zone-less `timestamp[ns]`.
pub fn seeds(keys: ArrayRef, times: Vec<(&str, Vec<i64>)>) -> Result<RecordBatch, CoreError> {
    let rows = keys.len();
    let mut fields = vec![
        Field::new(SEED_KEY, keys.data_type().clone(), true),
        Field::new(SEED_BATCH, DataType::Int64, false),
    ];
    let mut columns: Vec<ArrayRef> =
        vec![keys, Arc::new(Int64Array::from_iter_values(0..rows as i64))];
    for (name, values) in times {
        fields.push(Field::new(
            name,
            DataType::Timestamp(ArrowUnit::Nanosecond, None),
            false,
        ));
        columns.push(Arc::new(TimestampNanosecondArray::from(values)));
    }
    RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).map_err(|_| {
        CoreError::new(
            ErrorKind::Internal,
            ErrorCode::InternalError,
            "the seeds do not make one batch",
        )
    })
}

/// The order `_by_fkey` keeps a table's neighbors in (design, "Data
/// access"): the time column as a timestamp, latest first, then the key;
/// with no key, every other non-list column by name. The reference writes
/// no null order, so each term's is the source's own, `nulls`, which the
/// provider declares (`ProviderCapabilities::null_order`), never a
/// dialect's default written here. Without a key, every column of the
/// table but the time column and lists, in name order, each ascending
/// (`_neighbor_order_by`'s keyless order, from T1.29: a fact table left
/// without a key, its key's nulls declining it, is read so), and the
/// foreign key alone where no column is left.
pub fn tie_break(table: &Table, fkey: &str, nulls: NullOrder) -> Result<TieBreak, CoreError> {
    let nulls_last = match nulls {
        NullOrder::First => false,
        NullOrder::Last => true,
    };
    let mut terms = Vec::new();
    if let Some(time) = &table.time_column {
        terms.push(TieTerm {
            column: time.clone(),
            as_timestamp: true,
            descending: true,
            nulls_last,
        });
    }
    let Some(key) = &table.primary_key else {
        let mut names: Vec<&str> = table
            .columns
            .iter()
            .filter(|c| {
                Some(&c.name) != table.time_column.as_ref()
                    && !matches!(
                        c.dtype,
                        sdm_contract::ops::Dtype::Floatlist
                            | sdm_contract::ops::Dtype::Intlist
                            | sdm_contract::ops::Dtype::Stringlist
                    )
            })
            .map(|c| c.name.as_str())
            .collect();
        names.sort_unstable();
        if names.is_empty() {
            names.push(fkey);
        }
        terms.extend(names.into_iter().map(|name| TieTerm {
            column: name.to_owned(),
            as_timestamp: false,
            descending: false,
            nulls_last,
        }));
        return Ok(TieBreak(terms));
    };
    terms.push(TieTerm {
        column: key.clone(),
        as_timestamp: false,
        descending: false,
        nulls_last,
    });
    Ok(TieBreak(terms))
}

/// `_by_fkey`'s request: the rows of `table` whose foreign key `fkey` is a
/// seed's key, `columns` in order; where the table has a time column, those
/// whose time is not after the seed's anchor (the seeds' [`SEED_TIME`]);
/// the first `per_seed` of each seed in the reference's order
/// ([`tie_break`], nulls as the provider declares), the result in that
/// order. A row whose time is null is not returned (register ORD-2).
pub fn by_fkey(
    table: &Table,
    fkey: &str,
    columns: Vec<String>,
    seeds: KeySet,
    per_seed: u64,
    nulls: NullOrder,
) -> Result<DataRequest, CoreError> {
    Ok(DataRequest::Neighbors {
        table: table.source.clone(),
        key: vec![fkey.to_owned()],
        columns,
        seeds: typed(seeds, table, fkey),
        window: table
            .time_column
            .as_ref()
            .map(|time| sdm_contract::data::Window {
                column: time.clone(),
                lower: None,
                upper: Some(SeedBound {
                    seed_column: SEED_TIME.into(),
                    inclusive: true,
                }),
            }),
        per_seed: Some(per_seed),
        order: Some(tie_break(table, fkey, nulls)?),
    })
}

/// `_by_time`'s request: every row of `table` whose foreign key `fkey` is
/// a seed's key and whose time is after the seed's start ([`SEED_START`],
/// when `start`) and not after its end ([`SEED_END`]), `columns` in order,
/// unordered.
pub fn by_time(
    table: &Table,
    fkey: &str,
    columns: Vec<String>,
    seeds: KeySet,
    start: bool,
) -> Result<DataRequest, CoreError> {
    let time = table.time_column.clone().ok_or_else(|| {
        CoreError::new(
            ErrorKind::Internal,
            ErrorCode::InternalError,
            "a windowed read names a table without a time column",
        )
    })?;
    Ok(DataRequest::Neighbors {
        table: table.source.clone(),
        key: vec![fkey.to_owned()],
        columns,
        seeds: typed(seeds, table, fkey),
        window: Some(sdm_contract::data::Window {
            column: time,
            lower: start.then(|| SeedBound {
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
    })
}

/// `_by_pkey`'s request: one row of `table` for each seed whose key is the
/// table's key, `columns` in order, with how many rows hold it where
/// `counted`, as the reference reads the count only where it scores (the
/// examples' entities); the seeds say so otherwise ([`SEED_COUNT_UNREAD`]),
/// and say a key the source enforces unique ([`SEED_KEY_UNIQUE`]), which
/// a SQL layer may read by a plain join, counting nothing.
pub fn by_pkey(
    table: &Table,
    columns: Vec<String>,
    seeds: KeySet,
    counted: bool,
) -> Result<DataRequest, CoreError> {
    let key = table.primary_key.clone().ok_or_else(|| {
        CoreError::new(
            ErrorKind::Internal,
            ErrorCode::InternalError,
            "a lookup names a table without a key",
        )
    })?;
    let mut seeds = typed(seeds, table, &key);
    let unique = table.column(&key).is_some_and(|c| c.unique);
    for column in &mut seeds.columns {
        let flag = match column.name.as_str() {
            SEED_KEY if unique => SEED_KEY_UNIQUE,
            SEED_BATCH if !counted => SEED_COUNT_UNREAD,
            _ => continue,
        };
        column.metadata.insert(flag.to_owned(), MARKED.to_owned());
    }
    Ok(DataRequest::Lookup {
        table: table.source.clone(),
        seeds,
        key: vec![key],
        columns,
    })
}

/// `seeds`, their key column naming the declared type of `table`'s `key`
/// ([`SEED_KEY_TYPE`]) where the described schema keeps one (a big-integer
/// key the host layer read and marked, `DescribedColumn::declared_type`) and
/// the seed key is of that type: the data layer joins such a seed in the
/// key's own type (register P1B-43), from what `describe_source` read, so
/// `materialize` asks the source nothing more.
fn typed(mut seeds: KeySet, table: &Table, key: &str) -> KeySet {
    let wide = ColumnType::Decimal128 {
        precision: 38,
        scale: 0,
    };
    if let Some(declared) = table.column(key).and_then(|c| c.declared_type.as_ref()) {
        for column in seeds
            .columns
            .iter_mut()
            .filter(|c| c.name == SEED_KEY && c.column_type == wide)
        {
            column
                .metadata
                .insert(SEED_KEY_TYPE.to_owned(), declared.clone());
        }
    }
    seeds
}
