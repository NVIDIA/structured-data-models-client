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
use sdm_contract::{ColumnType, CoreError, Detail, Sentence};

use crate::graph::Table;
use crate::random::Pcg64;

/// The seed time columns the sampler's requests carry: each seed's anchor
/// (`_by_fkey`'s `__kumo_time__`), and the start and end of its window
/// (`_by_time`'s `__kumo_start__` and `__kumo_end__`).
pub const SEED_TIME: &str = "time";
/// An example's anchor where its entity's own time is missing
/// (`anchor_time='entity'` over a null time, T2.11): pandas' `NaT` as the
/// reference's anchors hold it, `i64::MIN` nanoseconds, distinct from the
/// no-anchor sentinel (`pd.Timestamp.min`). It anchors no window: a read by
/// time reaches no row for it, and the payload sends it as null.
pub const MISSING_ANCHOR: i64 = i64::MIN;
pub const SEED_START: &str = "start";
pub const SEED_END: &str = "end";

/// A time past what a nanosecond count since the epoch holds, which the
/// reference's pandas times cannot hold either.
pub fn out_of_range(field: &str) -> CoreError {
    CoreError::new(
        Sentence::TimePastNanoseconds,
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

/// A static query's examples anchored at each entity's own time
/// (`anchor_time='entity'`, `sample_target`'s shared draw): every entity,
/// shuffled, then ordered by its time, latest first, as the reference's
/// `sort_values(ascending=False)` orders a `datetime64[ns]` column
/// ([`descending_order`]), each anchored at its time; an entity with no
/// time last, anchored at `missing`.
pub fn entity_examples(
    rng: &mut Pcg64,
    entities: usize,
    entity_time: &[Option<i64>],
    missing: i64,
) -> Examples {
    let mut index: Vec<usize> = (0..entities).collect();
    rng.shuffle(&mut index);
    let times: Vec<Option<i64>> = index
        .iter()
        .map(|&i| entity_time.get(i).copied().flatten())
        .collect();
    let index: Vec<usize> = descending_order(&times)
        .into_iter()
        .map(|at| index[at])
        .collect();
    let anchors = index
        .iter()
        .map(|&i| entity_time.get(i).copied().flatten().unwrap_or(missing))
        .collect();
    Examples { index, anchors }
}

/// The order pandas' `Series.sort_values(ascending=False)` gives the
/// positions of `values`, a `datetime64[ns]` column (`nargsort`): the
/// values present, reversed, sorted ascending by NumPy's default
/// `argsort` (`kind='quicksort'`, [`introsort_argsort`]), the result reversed;
/// then the missing ones (`NaT`), in order.
pub fn descending_order(values: &[Option<i64>]) -> Vec<usize> {
    let present: Vec<usize> = (0..values.len())
        .filter(|&i| values[i].is_some())
        .rev()
        .collect();
    let keys: Vec<i64> = present.iter().filter_map(|&i| values[i]).collect();
    let mut order: Vec<usize> = introsort_argsort(&keys)
        .into_iter()
        .map(|k| present[k])
        .collect();
    order.reverse();
    order.extend((0..values.len()).filter(|&i| values[i].is_none()));
    order
}

/// NumPy's generic `aquicksort` (`npysort/quicksort.cpp`), which its
/// `argsort(kind='quicksort')` runs on a `datetime64` array (no SIMD path
/// takes that type): introsort, a median-of-three quicksort partitioning
/// down to 16 elements or fewer, then insertion sort, and heapsort past a
/// depth of twice the length's highest bit. Unstable, and deterministic:
/// equal values come out in the order this exact algorithm leaves them.
pub fn introsort_argsort(v: &[i64]) -> Vec<usize> {
    const SMALL_QUICKSORT: usize = 16;
    let num = v.len();
    let mut t: Vec<usize> = (0..num).collect();
    if num < 2 {
        return t;
    }
    let less = |a: usize, b: usize, t: &[usize]| v[t[a]] < v[t[b]];
    let mut stack: Vec<(usize, usize, i64)> = Vec::new();
    let (mut pl, mut pr) = (0usize, num - 1);
    let mut cdepth = 2 * i64::from(usize::BITS - 1 - num.leading_zeros());
    loop {
        if cdepth < 0 {
            heapsort_indices(v, &mut t[pl..=pr]);
        } else {
            while pr - pl > SMALL_QUICKSORT - 1 {
                let pm = pl + ((pr - pl) >> 1);
                if less(pm, pl, &t) {
                    t.swap(pm, pl);
                }
                if less(pr, pm, &t) {
                    t.swap(pr, pm);
                }
                if less(pm, pl, &t) {
                    t.swap(pm, pl);
                }
                let vp = v[t[pm]];
                let mut pi = pl;
                let mut pj = pr - 1;
                t.swap(pm, pj);
                loop {
                    pi += 1;
                    while v[t[pi]] < vp {
                        pi += 1;
                    }
                    pj -= 1;
                    while vp < v[t[pj]] {
                        pj -= 1;
                    }
                    if pi >= pj {
                        break;
                    }
                    t.swap(pi, pj);
                }
                t.swap(pi, pr - 1);
                cdepth -= 1;
                // The larger part waits on the stack.
                if pi - pl < pr - pi {
                    stack.push((pi + 1, pr, cdepth));
                    pr = pi - 1;
                } else {
                    stack.push((pl, pi - 1, cdepth));
                    pl = pi + 1;
                }
            }
            for pi in pl + 1..=pr {
                let vi = t[pi];
                let vp = v[vi];
                let mut pj = pi;
                while pj > pl && vp < v[t[pj - 1]] {
                    t[pj] = t[pj - 1];
                    pj -= 1;
                }
                t[pj] = vi;
            }
        }
        let Some((l, r, d)) = stack.pop() else {
            break;
        };
        pl = l;
        pr = r;
        cdepth = d;
    }
    t
}

/// NumPy's generic `aheapsort` (`npysort/heapsort.cpp`) over the indices
/// `a`, which [`introsort_argsort`] falls back to past its depth limit.
fn heapsort_indices(v: &[i64], a: &mut [usize]) {
    let n = a.len();
    if n < 2 {
        return;
    }
    // One-based, as NumPy indexes its heap.
    let at = |a: &[usize], k: usize| a[k - 1];
    let mut l = n >> 1;
    while l > 0 {
        let tmp = at(a, l);
        let (mut i, mut j) = (l, l << 1);
        while j <= n {
            if j < n && v[at(a, j)] < v[at(a, j + 1)] {
                j += 1;
            }
            if v[tmp] < v[at(a, j)] {
                a[i - 1] = at(a, j);
                i = j;
                j += j;
            } else {
                break;
            }
        }
        a[i - 1] = tmp;
        l -= 1;
    }
    let mut n = n;
    while n > 1 {
        let tmp = at(a, n);
        a[n - 1] = at(a, 1);
        n -= 1;
        let (mut i, mut j) = (1usize, 2usize);
        while j <= n {
            if j < n && v[at(a, j)] < v[at(a, j + 1)] {
                j += 1;
            }
            if v[tmp] < v[at(a, j)] {
                a[i - 1] = at(a, j);
                i = j;
                j += j;
            } else {
                break;
            }
        }
        a[i - 1] = tmp;
    }
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
        // A missing anchor (`MISSING_ANCHOR`, an entity's own time null) is
        // a null bound, which no window holds (the reference's
        // `time <= NULL`): the column is nullable only where one is.
        let missing = values.contains(&MISSING_ANCHOR);
        fields.push(Field::new(
            name,
            DataType::Timestamp(ArrowUnit::Nanosecond, None),
            missing,
        ));
        columns.push(Arc::new(TimestampNanosecondArray::from_iter(
            values
                .into_iter()
                .map(|v| (v != MISSING_ANCHOR).then_some(v)),
        )));
    }
    RecordBatch::try_new(Arc::new(Schema::new(fields)), columns)
        .map_err(|_| CoreError::new(Sentence::Internal, "the seeds do not make one batch"))
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
    by_fkey_limit(table, fkey, columns, seeds, per_seed.into(), nulls)
}

/// A neighbor request with the caller's explicit signed or SQL bound.
pub fn by_fkey_limit(
    table: &Table,
    fkey: &str,
    columns: Vec<String>,
    seeds: KeySet,
    per_seed: sdm_contract::data::NeighborLimit,
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
                source_type: table.column(time).and_then(|c| c.declared_type.clone()),
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
            Sentence::Internal,
            "a windowed read names a table without a time column",
        )
    })?;
    Ok(DataRequest::Neighbors {
        table: table.source.clone(),
        key: vec![fkey.to_owned()],
        columns,
        seeds: typed(seeds, table, fkey),
        window: Some(sdm_contract::data::Window {
            source_type: table.column(&time).and_then(|c| c.declared_type.clone()),
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
        CoreError::new(Sentence::Internal, "a lookup names a table without a key")
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

/// `seeds`, their key column naming the destination key's catalog declaration
/// ([`SEED_KEY_TYPE`]) when the destination retains one and the seed is floating
/// or a wide decimal.
/// Only the data layer interprets that opaque declaration: a floating seed
/// must retain the destination's float coercion without rounding integer keys,
/// and a wide decimal must compare exactly with an opaque big integer.
fn typed(mut seeds: KeySet, table: &Table, key: &str) -> KeySet {
    if let Some(declared) = table.column(key).and_then(|c| c.declared_type.as_ref()) {
        for column in seeds.columns.iter_mut().filter(|c| {
            c.name == SEED_KEY
                && matches!(
                    c.column_type,
                    ColumnType::Float32
                        | ColumnType::Float64
                        | ColumnType::Decimal128 {
                            precision: 38,
                            scale: 0
                        }
                )
        }) {
            column
                .metadata
                .insert(SEED_KEY_TYPE.to_owned(), declared.clone());
        }
    }
    seeds
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random::{Entropy, default_rng};

    #[test]
    fn a_missing_anchor_is_a_null_bound_and_the_column_nullable_only_then() {
        use arrow_array::Array;
        let keys: ArrayRef = Arc::new(Int64Array::from(vec![1, 2]));
        let none = seeds(Arc::clone(&keys), vec![(SEED_TIME, vec![5, 6])]).unwrap();
        assert!(!none.schema().field(2).is_nullable());
        let one = seeds(keys, vec![(SEED_TIME, vec![5, MISSING_ANCHOR])]).unwrap();
        assert!(one.schema().field(2).is_nullable());
        assert!(one.column(2).is_null(1) && !one.column(2).is_null(0));
    }

    #[test]
    fn entity_examples_are_every_entity_latest_first_at_its_own_time() {
        let times = [Some(5), None, Some(9), Some(5), Some(1)];
        let mut rng = default_rng(&Entropy::from_hex("2a").unwrap());
        let examples = entity_examples(&mut rng, times.len(), &times, -7);
        assert_eq!(examples.index.len(), 5);
        assert_eq!(examples.anchors, vec![9, 5, 5, 1, -7]);
        assert_eq!(examples.index[0], 2);
        assert_eq!(examples.index[3], 4);
        assert_eq!(examples.index[4], 1);
    }
}
