// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The reference's sampler, as far as `materialize` reaches it at T1.11c:
//! the examples' entities and anchor times (`sample_target`'s temporal
//! walk), their labels (a windowed `SUM`, as the reference's pandas
//! executor computes it), the seeds a `neighbors` or `lookup` request
//! carries, and the order `_by_fkey` keeps neighbors in. It holds no SQL:
//! what it asks for is typed requests, which `sdm-data` plans.

use std::sync::Arc;

use arrow_array::{ArrayRef, Int64Array, RecordBatch, TimestampNanosecondArray};
use arrow_schema::{DataType, Field, Schema, TimeUnit as ArrowUnit};
use sdm_contract::data::{SEED_BATCH, SEED_KEY, TieBreak, TieTerm};
use sdm_contract::pql::TimeUnit;
use sdm_contract::{CoreError, Detail, ErrorCode, ErrorKind};

use crate::graph::Table;
use crate::random::Pcg64;

const NANOS_PER_SECOND: i128 = 1_000_000_000;

/// `count` units as a fixed span in nanoseconds, as `pd.DateOffset` adds it
/// to a time with no zone; `None` for months, which are calendar
/// arithmetic, not ported yet.
pub fn span_ns(count: i32, unit: TimeUnit) -> Option<i128> {
    let seconds = match unit {
        TimeUnit::Seconds => 1,
        TimeUnit::Minutes => 60,
        TimeUnit::Hours => 3_600,
        TimeUnit::Days => 86_400,
        TimeUnit::Weeks => 7 * 86_400,
        TimeUnit::Months => return None,
    };
    Some(i128::from(count) * seconds * NANOS_PER_SECOND)
}

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

/// A label's values: a float column, read at float32 as the reference's
/// executor reads one, or an integer column.
#[derive(Debug, Clone)]
pub enum Values {
    Float(Vec<Option<f32>>),
    Int(Vec<Option<i64>>),
}

/// The window a label sums over, from each example's anchor: times after
/// `anchor + start` (none: no lower bound) and not after `anchor + end`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Window {
    pub start: Option<i128>,
    pub end: i128,
}

/// A windowed `SUM` per example, as the reference's pandas executor gives
/// it: of each row whose value is present (not null, and for a float not
/// NaN, which pandas' sum skips) and whose time falls in its
/// example's window, summed in row order, a float column at float32 with
/// Kahan's compensation (register entry PK-2) and an integer column in
/// int64; then read at float32, an example with no row giving 0.
pub fn sum_labels(
    examples: usize,
    batch: &[usize],
    values: &Values,
    times: &[Option<i64>],
    anchors: &[i64],
    window: Window,
) -> Vec<f32> {
    let within = |row: usize| -> bool {
        let Some(time) = times[row] else {
            return false;
        };
        let anchor = i128::from(anchors[batch[row]]);
        let time = i128::from(time);
        time <= anchor + window.end && window.start.is_none_or(|start| time > anchor + start)
    };
    match values {
        Values::Float(values) => {
            let mut sum = vec![0_f32; examples];
            let mut compensation = vec![0_f32; examples];
            for row in 0..batch.len() {
                let Some(value) = values[row].filter(|v| !v.is_nan()) else {
                    continue;
                };
                if !within(row) {
                    continue;
                }
                let at = batch[row];
                let y = value - compensation[at];
                let t = sum[at] + y;
                compensation[at] = (t - sum[at]) - y;
                if compensation[at].is_nan() {
                    compensation[at] = 0.0;
                }
                sum[at] = t;
            }
            sum
        }
        Values::Int(values) => {
            let mut sum = vec![0_i64; examples];
            for row in 0..batch.len() {
                let Some(value) = values[row] else { continue };
                if within(row) {
                    sum[batch[row]] = sum[batch[row]].wrapping_add(value);
                }
            }
            // The reference reads the int64 sum at float32, rounding to
            // nearest, as `as` does.
            sum.into_iter().map(|s| s as f32).collect()
        }
    }
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
/// with no key, every other non-list column by name. All terms nulls last,
/// DuckDB's default the reference relies on. Without a key the order is not
/// sampled yet.
pub fn tie_break(table: &Table) -> Result<TieBreak, CoreError> {
    let mut terms = Vec::new();
    if let Some(time) = &table.time_column {
        terms.push(TieTerm {
            column: time.clone(),
            as_timestamp: true,
            descending: true,
            nulls_last: true,
        });
    }
    let Some(key) = &table.primary_key else {
        return Err(crate::not_yet("keyless_neighbors"));
    };
    terms.push(TieTerm {
        column: key.clone(),
        as_timestamp: false,
        descending: false,
        nulls_last: true,
    });
    Ok(TieBreak(terms))
}

#[cfg(test)]
mod tests {
    use super::{Values, Window, sum_labels};

    #[test]
    fn a_nan_is_skipped_as_pandas_skips_it() {
        // Example 0 holds only NaN, example 1 NaN and finite values, example
        // 2 a null: each sums what is present, an example with none 0.
        let values = Values::Float(vec![
            Some(f32::NAN),
            Some(1.5),
            Some(f32::NAN),
            Some(2.0),
            None,
        ]);
        let labels = sum_labels(
            3,
            &[0, 1, 1, 1, 2],
            &values,
            &[Some(5); 5],
            &[0, 0, 0],
            Window {
                start: Some(0),
                end: 10,
            },
        );
        assert_eq!(labels, [0.0, 3.5, 0.0]);
    }
}
