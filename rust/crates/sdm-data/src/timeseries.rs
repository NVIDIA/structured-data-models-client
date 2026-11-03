// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Aligned history expansion into ordinary relational input tables.
use arrow_array::{
    Array, ArrayRef, FixedSizeListArray, Float64Array, Int64Array, LargeListArray, ListArray,
    RecordBatch, TimestampNanosecondArray, UInt64Array,
};
use arrow_schema::{DataType, Field, Schema, TimeUnit};
use sdm_contract::{CoreError, Detail, Sentence, frames::TimeSeriesOptions};
use std::{collections::BTreeSet, sync::Arc};

pub struct Prepared {
    pub entity: RecordBatch,
    pub target: RecordBatch,
    pub key: String,
    pub step_ns: i64,
    pub irregular: bool,
}
fn bad(reason: &str) -> CoreError {
    CoreError::new(Sentence::TimeSeriesInput, reason).with_field("reason", Detail::text(reason))
}
fn list(array: &ArrayRef, row: usize) -> Result<ArrayRef, CoreError> {
    if array.is_null(row) {
        return Err(bad("history and timestamp lists cannot be null"));
    }
    if let Some(a) = array.as_any().downcast_ref::<ListArray>() {
        Ok(a.value(row))
    } else if let Some(a) = array.as_any().downcast_ref::<LargeListArray>() {
        Ok(a.value(row))
    } else if let Some(a) = array.as_any().downcast_ref::<FixedSizeListArray>() {
        Ok(a.value(row))
    } else {
        Err(bad("history and timestamps must be Arrow list columns"))
    }
}
fn column(batch: &RecordBatch, name: &str) -> Result<usize, CoreError> {
    batch
        .schema()
        .index_of(name)
        .map_err(|_| bad("requested time-series column does not exist"))
}

pub fn prepare(batch: &RecordBatch, opts: &TimeSeriesOptions) -> Result<Prepared, CoreError> {
    let names: BTreeSet<_> = batch
        .schema()
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect();
    if names.len() != batch.num_columns() || names.iter().any(String::is_empty) {
        return Err(bad("input columns must be unique nonempty names"));
    }
    let history = column(batch, &opts.history)?;
    let timestamps = opts
        .timestamps
        .as_ref()
        .map(|n| column(batch, n))
        .transpose()?;
    let key = opts
        .entity_key
        .clone()
        .unwrap_or_else(|| "entity_id".into());
    if opts.forecasts == 0 {
        return Err(bad("num_timeframes must be positive"));
    }
    if key == "timestamp"
        || key == "value"
        || key == opts.history
        || opts
            .timestamps
            .as_ref()
            .is_some_and(|n| n == &opts.history || n == &key)
    {
        return Err(bad(
            "entity, history and timestamp columns must be distinct and entity key cannot be value or timestamp",
        ));
    }
    let ids: ArrayRef = if opts.entity_key.is_some() {
        batch.column(column(batch, &key)?).clone()
    } else {
        if batch.schema().index_of(&key).is_ok() {
            return Err(bad("generated entity_id conflicts with an existing column"));
        }
        Arc::new(Int64Array::from_iter_values(
            0..i64::try_from(batch.num_rows()).map_err(|_| bad("too many entities"))?,
        ))
    };
    let mut values = Vec::<Option<f64>>::new();
    let mut times = Vec::<i64>::new();
    let mut rows = Vec::<u64>::new();
    let mut lengths = Vec::new();
    let mut deltas = Vec::<i64>::new();
    for row in 0..batch.num_rows() {
        let observations = list(batch.column(history), row)?;
        if !matches!(
            observations.data_type(),
            DataType::Null
                | DataType::Int8
                | DataType::Int16
                | DataType::Int32
                | DataType::Int64
                | DataType::UInt8
                | DataType::UInt16
                | DataType::UInt32
                | DataType::UInt64
                | DataType::Float16
                | DataType::Float32
                | DataType::Float64
                | DataType::Decimal128(..)
                | DataType::Decimal256(..)
        ) {
            return Err(bad("observations must be numeric scalar values"));
        }
        let numeric = arrow_cast::cast(&observations, &DataType::Float64)
            .map_err(|_| bad("observations cannot be represented as float64"))?;
        let numeric = numeric
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| bad("invalid numeric history"))?;
        for value in numeric.iter() {
            if value.is_some_and(|v| v.is_infinite()) {
                return Err(bad("observations cannot be infinite"));
            }
            values.push(value.filter(|v| !v.is_nan()));
        }
        lengths.push(numeric.len());
        rows.extend(std::iter::repeat_n(row as u64, numeric.len()));
        if let Some(index) = timestamps {
            let stamps = list(batch.column(index), row)?;
            if stamps.len() != numeric.len() {
                return Err(
                    bad("each entity's timestamp and observation lengths must match")
                        .with_field("row", Detail::text(row.to_string())),
                );
            }
            if !matches!(stamps.data_type(), DataType::Timestamp(..)) && !stamps.is_empty() {
                return Err(bad("timestamps must contain Arrow timestamp values"));
            }
            let ns = arrow_cast::cast(
                &stamps,
                &DataType::Timestamp(TimeUnit::Nanosecond, Some("UTC".into())),
            )
            .map_err(|_| bad("timestamp is outside nanosecond range"))?;
            let ns = ns
                .as_any()
                .downcast_ref::<TimestampNanosecondArray>()
                .ok_or_else(|| bad("invalid timestamp history"))?;
            if ns.null_count() != 0 {
                return Err(bad("timestamps cannot be null or outside nanosecond range"));
            }
            times.extend(ns.values().iter().copied());
            if opts.step_ns.is_none() {
                let mut sorted = ns.values().to_vec();
                sorted.sort_unstable();
                for pair in sorted.windows(2) {
                    deltas.push(
                        pair[1]
                            .checked_sub(pair[0])
                            .ok_or_else(|| bad("timestamp interval overflow"))?,
                    );
                }
            }
        }
    }
    if values.is_empty() {
        return Err(bad("at least one historical observation is required"));
    }
    let mut irregular = false;
    let step = if let Some(step) = opts.step_ns {
        step
    } else {
        if deltas.is_empty() {
            return Err(bad("cannot infer interval; supply time_delta"));
        }
        deltas.sort_unstable();
        irregular = deltas.first() != deltas.last();
        let mid = deltas.len() / 2;
        if deltas.len().is_multiple_of(2) {
            i64::try_from((i128::from(deltas[mid - 1]) + i128::from(deltas[mid])) / 2)
                .map_err(|_| bad("median interval overflow"))?
        } else {
            deltas[mid]
        }
    };
    if step <= 0 || step % 60_000_000_000 != 0 {
        return Err(bad(
            "time_delta must be a positive exact whole number of minutes",
        ));
    }
    step.checked_mul(i64::from(opts.forecasts))
        .ok_or_else(|| bad("forecast horizon overflow"))?;
    if timestamps.is_none() {
        let anchor = opts
            .anchor_ns
            .ok_or_else(|| bad("anchor_time is required for synthetic timestamps"))?;
        for length in lengths {
            for i in 0..length {
                let offset = i64::try_from(length - i)
                    .ok()
                    .and_then(|n| n.checked_mul(step))
                    .ok_or_else(|| bad("history interval overflow"))?;
                times.push(
                    anchor
                        .checked_sub(offset)
                        .ok_or_else(|| bad("history timestamp overflow"))?,
                );
            }
        }
    }
    let indices = UInt64Array::from(rows);
    let fk = arrow_select::take::take(&ids, &indices, None)
        .map_err(|_| bad("history entity expansion failed"))?;
    let target = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new(&key, ids.data_type().clone(), false),
            Field::new(
                "timestamp",
                DataType::Timestamp(TimeUnit::Nanosecond, Some("UTC".into())),
                false,
            ),
            Field::new("value", DataType::Float64, true),
        ])),
        vec![
            fk,
            Arc::new(TimestampNanosecondArray::from(times).with_timezone("UTC")),
            Arc::new(Float64Array::from(values)),
        ],
    )
    .map_err(|_| bad("history rows do not align"))?;
    let keep: Vec<_> = (0..batch.num_columns())
        .filter(|i| *i != history && Some(*i) != timestamps)
        .collect();
    let entity = batch
        .project(&keep)
        .map_err(|_| bad("entity projection failed"))?;
    let entity = if opts.entity_key.is_none() {
        let mut fields = vec![Arc::new(Field::new(&key, DataType::Int64, false))];
        fields.extend(entity.schema().fields().iter().cloned());
        let mut columns = vec![ids];
        columns.extend(entity.columns().iter().cloned());
        RecordBatch::try_new(Arc::new(Schema::new(fields)), columns)
            .map_err(|_| bad("entity rows do not align"))?
    } else {
        entity
    };
    Ok(Prepared {
        entity,
        target,
        key,
        step_ns: step,
        irregular,
    })
}
