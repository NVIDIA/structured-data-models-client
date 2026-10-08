// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `Sampler.sample_lagged_target`: a temporal query's labels at each of the
//! `lag_timesteps` timeframes before each example's anchor, which
//! `KumoRelational.add_lagged_target` adds to the task as features.

use sdm_labels::date_offset::DateOffset;
use sdm_schema::dt::Host;
use serde_json::Value;

use super::inputs::{Inputs, int};
use super::local;
use super::output::Warning;
use super::query::SamplerQuery;
use super::sampler::{Sampler, Through};
use super::subgraph::anchor_times;
use super::target::{query_columns, query_time_offsets};
use super::{Db, Raised};
use crate::frame::{Frame, Series};

/// `lag_timesteps * offset`: the offset with its `n` times as many.
fn times(offset: &DateOffset, factor: i64) -> Result<DateOffset, Raised> {
    let n = offset
        .n
        .checked_mul(factor)
        .ok_or_else(|| Raised::new("OverflowError", "Python int too large to convert to C long"))?;
    Ok(DateOffset {
        n,
        kwds: offset.kwds.clone(),
    })
}

/// What `sample_lagged_target` gives: the frame of lagged labels and the
/// warnings reading the rows gave.
#[derive(Clone, Debug)]
pub struct LaggedDone {
    pub frame: Frame,
    pub warnings: Vec<Warning>,
}

/// `Sampler.sample_lagged_target(query, entity_pkey, anchor_time,
/// lag_timesteps)`: a frame of the lagged labels, `__kumo_arl0__` the
/// furthest back.
pub async fn sample_lagged_target(
    inputs: Inputs,
    db: Db,
    host: Host,
) -> Result<LaggedDone, Raised> {
    let mut sampler = Sampler::new(&inputs, host)?;
    let get = |name: &str| inputs.arguments.get(name).unwrap_or(&Value::Null);
    let entity_pkey = inputs.series_argument("entity_pkey", "entity_pkey_dtype")?;
    let anchor = anchor_times(&inputs.series_argument("anchor_time", "anchor_time_dtype")?)?;
    let lag_timesteps = int(get("lag_timesteps"), "lag_timesteps")?;
    let lagged = lagged_targets(
        &mut sampler,
        &db,
        get("query"),
        &entity_pkey,
        &anchor,
        lag_timesteps,
    )
    .await;
    match lagged {
        Ok(frame) => Ok(LaggedDone {
            frame,
            warnings: sampler.take_warnings(),
        }),
        Err(raised) => Err(raised.after(sampler.take_warnings())),
    }
}

/// `sample_lagged_target` of a sampler, the query's tree, and the
/// examples' entities and anchor times.
pub async fn lagged_targets(
    sampler: &mut Sampler,
    db: &Db,
    tree: &Value,
    entity_pkey: &Series,
    anchor: &[Option<i64>],
    lag_timesteps: i64,
) -> Result<Frame, Raised> {
    let query = SamplerQuery::autoregressive_from_tree(tree)?;
    let time_offsets = query_time_offsets(sampler, &query)?
        .into_iter()
        .map(|(edge, (start, end))| Ok((edge, (start, times(&end, lag_timesteps)?))))
        .collect::<Result<Vec<_>, Raised>>()?;
    let mut columns_dict = query_columns(&query);
    for ((table, _, _), _) in &time_offsets {
        let column = sampler
            .time_column(table)
            .cloned()
            .ok_or_else(|| Raised::key_error(table))?;
        columns_dict.add(table, &column);
    }
    let offset = query
        .target_timeframe
        .as_ref()
        .ok_or_else(|| Raised::new("AssertionError", ""))?
        .end_offset();
    // `anchor_time - lag_timesteps * offset`.
    let back = times(&times(&offset, lag_timesteps)?, -1)?;
    let anchor = back.apply_array(anchor, 1_000_000_000)?;

    let (feat, time, batch) = match &sampler.through {
        Through::Local(_) => local::sample_query_data(
            sampler,
            &query.entity_table,
            entity_pkey,
            &anchor,
            &columns_dict,
            &time_offsets,
        )?,
        Through::Sql(_) => {
            super::sql::sample_query_data(
                sampler,
                db,
                &query.entity_table,
                entity_pkey,
                &anchor,
                &columns_dict,
                &time_offsets,
            )
            .await?
        }
    };

    let mut names = Vec::new();
    let mut columns: Vec<Series> = Vec::new();
    for i in (0..lag_timesteps.max(0)).rev() {
        let at = times(&offset, i)?.apply_array(&anchor, 1_000_000_000)?;
        let (labels, _) = sampler.execute_query(&query, &feat, &time, &batch, &at, false)?;
        names.push(format!("__kumo_arl{}__", lag_timesteps - 1 - i));
        columns.push(labels);
    }
    let rows = columns.first().map_or(0, Series::len);
    if columns.iter().any(|c| c.len() != rows) {
        return Err(Raised::value_error("All arrays must be of the same length"));
    }
    Ok(Frame::new(names, columns, rows))
}
