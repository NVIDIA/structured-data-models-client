// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `Sampler.sample_target`: draw the entities, walk the anchors back in
//! time, compute the labels and split them into training and test
//! examples, with the random draws in today's order (the forecast entity
//! first, then the test candidates, then the training candidates, then one
//! shuffle per temporal step).

use std::sync::Arc;

use arrow_array::TimestampNanosecondArray;
use sdm_dialects::int::Int;
use sdm_labels::date_offset::{DateOffset, max_date_offset, min_date_offset};
use sdm_random::RandomState;
use sdm_schema::dt::Host;
use serde_json::Value;

use super::inputs::{Backend, Inputs, int};
use super::output::{TargetDone, TargetPart, Warning};
use super::query::{Aggregation, SamplerQuery};
use super::sampler::{Sampler, TIMESTAMP_MAX, concat_labels, empty_float_series, head, tail};
use super::{Db, Raised};
use crate::frame::{Dtype, Frame, Series};
use crate::keys::{Key, KeyId, keys_of};

/// An anchor time: a timestamp, in nanoseconds, or `'entity'`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Anchor {
    Entity,
    At(i64),
}

pub fn anchor(value: &Value, what: &str) -> Result<Anchor, Raised> {
    match value {
        Value::String(s) if s == "entity" => Ok(Anchor::Entity),
        Value::Object(o) => Ok(Anchor::At(int(o.get("ns").unwrap_or(&Value::Null), what)?)),
        _ => Err(Raised::value_error(format!("{what} is not an anchor time"))),
    }
}

/// A seed from its decimal text, `None` for none.
pub fn seed(value: Option<&Value>) -> Result<Option<Int>, Raised> {
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(text)) => Int::parse(text)
            .map(Some)
            .ok_or_else(|| Raised::value_error(format!("the seed {text:?}"))),
        Some(_) => Err(Raised::value_error("the seed is not decimal text")),
    }
}

/// `random_state(random_seed)`: numpy's legacy generator for the seed; for
/// none, a generator seeded from 128 bits of the system's entropy, as
/// numpy's `RandomState(None)` seeds its own.
pub fn random_state(seed: Option<&Int>) -> Result<RandomState, Raised> {
    match seed {
        Some(seed) => {
            if seed.is_negative() {
                return Err(Raised::value_error("Seed must be between 0 and 2**32 - 1"));
            }
            sdm_random::random_state(&seed.magnitude_words())
                .map_err(|e| Raised::value_error(e.to_string()))
        }
        None => {
            let mut entropy = [0u8; 16];
            getrandom::fill(&mut entropy)
                .map_err(|e| Raised::new("OSError", format!("the system gave no entropy: {e}")))?;
            let words: Vec<u32> = entropy
                .chunks_exact(4)
                .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
                .collect();
            RandomState::from_words(&words).map_err(|e| Raised::value_error(e.to_string()))
        }
    }
}

/// A time column's values in nanoseconds, `None` for `NaT`.
pub fn times_of(series: &Series) -> Result<Vec<Option<i64>>, Raised> {
    use arrow_array::cast::AsArray;
    use arrow_array::types::{
        TimestampMicrosecondType, TimestampMillisecondType, TimestampNanosecondType,
        TimestampSecondType,
    };
    use arrow_schema::{DataType, TimeUnit};
    let array = series.array.as_ref();
    macro_rules! read {
        ($t:ty, $per:expr) => {{
            let values = array.as_primitive::<$t>();
            (0..array.len())
                .map(|i| {
                    if array.is_null(i) {
                        return Ok(None);
                    }
                    let v = values.value(i);
                    v.checked_mul($per).map(Some).ok_or_else(|| {
                        Raised::new(
                            "OutOfBoundsDatetime",
                            crate::graph_store::out_of_bounds_text(i128::from(v) * $per as i128),
                        )
                    })
                })
                .collect()
        }};
    }
    match array.data_type() {
        DataType::Timestamp(TimeUnit::Nanosecond, _) => read!(TimestampNanosecondType, 1),
        DataType::Timestamp(TimeUnit::Microsecond, _) => read!(TimestampMicrosecondType, 1_000),
        DataType::Timestamp(TimeUnit::Millisecond, _) => {
            read!(TimestampMillisecondType, 1_000_000)
        }
        DataType::Timestamp(TimeUnit::Second, _) => read!(TimestampSecondType, 1_000_000_000),
        DataType::Null => Ok(vec![None; array.len()]),
        other => Err(Raised::new(
            "TypeError",
            format!("times of Arrow type {other} are not supported by the core"),
        )),
    }
}

/// A `datetime64[ns]` column of `times`.
pub fn time_series(times: &[Option<i64>]) -> Series {
    Series::new(
        Dtype::Numpy("datetime64[ns]".into()),
        Arc::new(TimestampNanosecondArray::from(times.to_vec())),
    )
}

/// `to_ser(anchor, size)`.
fn to_ser(anchor: i64, size: usize) -> Vec<Option<i64>> {
    vec![Some(anchor); size]
}

/// The columns each table is read with, as `columns_dict` holds them: each
/// name once.
#[derive(Clone, Debug, Default)]
pub struct Columns(pub Vec<(String, Vec<String>)>);

impl Columns {
    pub fn get(&self, table: &str) -> Option<&Vec<String>> {
        self.0.iter().find(|(t, _)| t == table).map(|(_, c)| c)
    }

    /// `columns_dict[table]` of a `defaultdict(set)`: made empty if absent.
    pub fn entry(&mut self, table: &str) -> &mut Vec<String> {
        if let Some(i) = self.0.iter().position(|(t, _)| t == table) {
            return &mut self.0[i].1;
        }
        self.0.push((table.to_string(), Vec::new()));
        let last = self.0.len() - 1;
        &mut self.0[last].1
    }

    pub fn add(&mut self, table: &str, column: &str) {
        let columns = self.entry(table);
        if !columns.iter().any(|c| c == column) {
            columns.push(column.to_string());
        }
    }

    /// `{key: columns for key, columns in columns_dict.items() if key !=
    /// table}`.
    pub fn without(&self, table: &str) -> Columns {
        Columns(self.0.iter().filter(|(t, _)| t != table).cloned().collect())
    }
}

/// `_get_query_columns_dict(query)`.
pub fn query_columns(query: &SamplerQuery) -> Columns {
    let mut columns = Columns::default();
    for fqn in &query.all_query_columns {
        let mut parts = fqn.split('.');
        let table = parts.next().unwrap_or_default();
        let column = parts.next().unwrap_or_default();
        if column == "*" {
            // `columns_dict[table_name]` is not touched for a wildcard.
            continue;
        }
        columns.add(table, column);
    }
    columns
}

/// An edge type and the window its rows are read in.
pub type TimeOffsets = Vec<((String, String, String), (Option<DateOffset>, DateOffset))>;

/// `_get_query_time_offset_dict(query)`.
pub fn query_time_offsets(sampler: &Sampler, query: &SamplerQuery) -> Result<TimeOffsets, Raised> {
    let mut out: TimeOffsets = Vec::new();
    for Aggregation { table, window } in &query.windowed_aggregations {
        let edge_types: Vec<&(String, String, String)> = sampler
            .edge_types
            .iter()
            .filter(|e| &e.0 == table && e.2 == query.entity_table)
            .collect();
        if edge_types.len() != 1 {
            return Err(Raised::value_error(format!(
                "Could not find a unique foreign key from table '{table}' to '{}'",
                query.entity_table
            )));
        }
        let edge = edge_types[0].clone();
        let window = window
            .as_ref()
            .ok_or_else(|| Raised::new("AssertionError", ""))?;
        let (start, end) = match out.iter().position(|(e, _)| *e == edge) {
            None => (window.start_offset(), window.end_offset()),
            Some(i) => {
                let (start, end) = &out[i].1;
                (
                    min_date_offset(&[start.clone(), window.start_offset()])?,
                    max_date_offset(&[end.clone(), window.end_offset()])?,
                )
            }
        };
        match out.iter().position(|(e, _)| *e == edge) {
            Some(i) => out[i].1 = (start, end),
            None => out.push((edge, (start, end))),
        }
    }
    Ok(out)
}

/// What `_sample_target` gives: each set's labels and the mask of the
/// candidates they are for.
pub struct Labels {
    pub train_y: Series,
    pub train_mask: Vec<bool>,
    pub test_y: Series,
    pub test_mask: Vec<bool>,
}

fn select(values: &[i64], mask: &[bool]) -> Vec<i64> {
    values
        .iter()
        .zip(mask)
        .filter(|(_, m)| **m)
        .map(|(v, _)| *v)
        .collect()
}

fn select_times(values: &[Option<i64>], mask: &[bool]) -> Vec<Option<i64>> {
    values
        .iter()
        .zip(mask)
        .filter(|(_, m)| **m)
        .map(|(v, _)| *v)
        .collect()
}

fn positions(index: &[i64]) -> Vec<usize> {
    index.iter().map(|&i| i as usize).collect()
}

/// `Timestamp - Timedelta`, in nanoseconds.
fn minus(anchor: i64, step: i64) -> Result<i64, Raised> {
    anchor
        .checked_sub(step)
        .filter(|&t| t != i64::MIN)
        .ok_or_else(|| Raised::new("OverflowError", "Python int too large to convert to C long"))
}

fn plus(anchor: i64, step: i64) -> Result<i64, Raised> {
    anchor
        .checked_add(step)
        .filter(|&t| t != i64::MIN)
        .ok_or_else(|| Raised::new("OverflowError", "Python int too large to convert to C long"))
}

/// `Sampler.sample_target`.
pub async fn sample_target(inputs: Inputs, db: Db, host: Host) -> Result<TargetDone, Raised> {
    let mut sampler = Sampler::new(&inputs, host)?;
    let arguments = &inputs.arguments;
    let get = |name: &str| arguments.get(name).unwrap_or(&Value::Null);
    let query = SamplerQuery::from_tree(get("query"))?;
    let args = TargetArgs {
        num_train_examples: int(get("num_train_examples"), "num_train_examples")?,
        train_anchor: anchor(get("train_anchor_time"), "train_anchor_time")?,
        num_train_trials: int(get("num_train_trials"), "num_train_trials")?,
        num_test_examples: int(get("num_test_examples"), "num_test_examples")?,
        test_anchor: anchor(get("test_anchor_time"), "test_anchor_time")?,
        num_test_trials: int(get("num_test_trials"), "num_test_trials")?,
        random_seed: seed(arguments.get("random_seed"))?,
    };
    sample_target_with(&mut sampler, &db, &query, args).await
}

/// `sample_target`'s arguments besides the query.
#[derive(Clone, Debug)]
pub struct TargetArgs {
    pub num_train_examples: i64,
    pub train_anchor: Anchor,
    pub num_train_trials: i64,
    pub num_test_examples: i64,
    pub test_anchor: Anchor,
    pub num_test_trials: i64,
    pub random_seed: Option<Int>,
}

/// `Sampler.sample_target` of a sampler and query already read. What it
/// raises carries the warnings reading the rows gave before it raised.
pub async fn sample_target_with(
    sampler: &mut Sampler,
    db: &Db,
    query: &SamplerQuery,
    args: TargetArgs,
) -> Result<TargetDone, Raised> {
    match draw_targets(sampler, db, query, args).await {
        Ok(done) => Ok(done),
        Err(raised) => Err(raised.after(sampler.take_warnings())),
    }
}

async fn draw_targets(
    sampler: &mut Sampler,
    db: &Db,
    query: &SamplerQuery,
    args: TargetArgs,
) -> Result<TargetDone, Raised> {
    let TargetArgs {
        mut num_train_examples,
        train_anchor,
        mut num_train_trials,
        mut num_test_examples,
        test_anchor,
        mut num_test_trials,
        random_seed,
    } = args;
    let db = db.clone();
    let query = query.clone();

    let mut rng = random_state(random_seed.as_ref())?;
    let is_forecasting = query.is_forecasting;
    let forecast_num = query.num_forecasts;

    if num_train_examples == 0 || num_train_trials == 0 {
        num_train_examples = 0;
        num_train_trials = 0;
    }
    if num_test_examples == 0 || num_test_trials == 0 {
        num_test_examples = 0;
        num_test_trials = 0;
    }

    // 1. Collect information on what to query.
    let mut columns_dict = query_columns(&query);
    let time_offsets = query_time_offsets(sampler, &query)?;
    for ((table, _, _), _) in &time_offsets {
        let column = sampler
            .time_column(table)
            .cloned()
            .ok_or_else(|| Raised::key_error(table))?;
        columns_dict.add(table, &column);
    }

    // 2. Sample random rows from the entity table.
    let is_static = !query.is_temporal();
    let shared_train_test = is_static && train_anchor == test_anchor;
    let num_entity_rows = if shared_train_test {
        num_train_trials + num_test_trials
    } else {
        num_train_trials.max(num_test_trials)
    };
    if num_entity_rows <= 0 {
        return Err(Raised::new("AssertionError", ""));
    }
    let entity_table = query.entity_table.clone();
    let entity_ids = if is_forecasting {
        query.rfm_entity_ids.clone()
    } else {
        None
    };
    let mut entity_columns = columns_dict.entry(&entity_table).clone();
    for column in [
        sampler.time_column(&entity_table).cloned(),
        sampler.end_time_column(&entity_table).cloned(),
    ]
    .into_iter()
    .flatten()
    {
        if !entity_columns.contains(&column) {
            entity_columns.push(column);
        }
    }
    let entity_df = sampler
        .sample_entity_table(
            &db,
            &entity_table,
            &entity_columns,
            num_entity_rows,
            random_seed.as_ref(),
            entity_ids.as_deref(),
        )
        .await?;
    if entity_df.is_empty() {
        return Err(Raised::value_error(format!(
            "Failed to find any rows in the entity table '{entity_table}'."
        )));
    }
    // A key the source does not enforce can repeat: keep its first row.
    let pkey_name = sampler.primary_key(&entity_table)?.clone();
    let entity_df = {
        let keys = keys_of(entity_df.column(&pkey_name)?)?;
        let mut seen = std::collections::HashSet::new();
        let first: Vec<usize> = keys
            .iter()
            .enumerate()
            .filter(|(_, k)| seen.insert(KeyId::clone(&k.id())))
            .map(|(i, _)| i)
            .collect();
        if first.len() == entity_df.len() {
            entity_df
        } else {
            entity_df.take(&first)?
        }
    };
    let entity_pkey = entity_df.column(&pkey_name)?.clone();
    let entity_time = match sampler.time_column(&entity_table) {
        Some(column) => Some(times_of(entity_df.column(column)?)?),
        None => None,
    };
    let entity_end_time = match sampler.end_time_column(&entity_table) {
        Some(column) => Some(times_of(entity_df.column(column)?)?),
        None => None,
    };
    let num_entities = entity_pkey.len();

    let get_valid_entity_index =
        |rng: &mut RandomState, time: Anchor, max_size: Option<i64>| -> Vec<i64> {
            let mut index: Vec<i64> = match (time, &entity_time, &entity_end_time) {
                (Anchor::Entity, _, _) | (_, None, None) => (0..num_entities as i64).collect(),
                (Anchor::At(t), start, end) => (0..num_entities)
                    .filter(|&i| {
                        let born = start.as_ref().is_none_or(|s| s[i].is_some_and(|v| v <= t));
                        let alive = end.as_ref().is_none_or(|e| e[i].is_none_or(|v| v > t));
                        born && alive
                    })
                    .map(|i| i as i64)
                    .collect(),
            };
            rng.shuffle(&mut index);
            if let Some(max_size) = max_size {
                index.truncate(max_size.max(0) as usize);
            }
            index
        };

    // 3. Build training and test candidates.
    let requested_train_anchor = train_anchor;
    let mut train_anchor = train_anchor;
    let mut train_index: Vec<i64> = Vec::new();
    let mut test_index: Vec<i64> = Vec::new();
    let mut train_time: Vec<Option<i64>> = Vec::new();
    let mut test_time: Vec<Option<i64>> = Vec::new();

    let is_temporal = query.is_temporal();
    let mut aggr_table_names: Vec<String> = Vec::new();
    let mut step_offset: Option<i64> = None;
    if is_temporal {
        aggr_table_names = query
            .target_aggregations
            .iter()
            .map(|a| a.table.clone())
            .collect();
        let timeframe = query
            .target_timeframe
            .as_ref()
            .ok_or_else(|| Raised::new("AssertionError", ""))?;
        step_offset = Some(
            timeframe
                .timeframe_ns()?
                .ok_or_else(|| Raised::new("AssertionError", ""))?,
        );
    }

    // For forecasting, a single entity across all time steps.
    let mut forecast_entity_idx: Option<i64> = None;
    if is_forecasting {
        if let Some(ids) = &query.rfm_entity_ids {
            let first = ids
                .first()
                .ok_or_else(|| Raised::new("IndexError", "list index out of range"))?;
            let wanted = entity_id_key(first);
            let keys = keys_of(&entity_pkey)?;
            let found = keys.iter().position(|k| equal_keys(k, &wanted));
            match found {
                Some(i) => forecast_entity_idx = Some(i as i64),
                None => {
                    return Err(Raised::new(
                        "RuntimeError",
                        format!(
                            "Entity '{}' not found in entity table '{entity_table}'",
                            entity_id_str(first)
                        ),
                    ));
                }
            }
        } else {
            let valid = get_valid_entity_index(&mut rng, train_anchor, None);
            if valid.is_empty() {
                return Err(Raised::new(
                    "RuntimeError",
                    "No valid entities for forecasting",
                ));
            }
            forecast_entity_idx = Some(valid[0]);
        }
    }

    if shared_train_test {
        train_index = get_valid_entity_index(&mut rng, train_anchor, None);
        match train_anchor {
            Anchor::Entity => {
                let entity_time = entity_time
                    .as_ref()
                    .ok_or_else(|| Raised::new("AssertionError", ""))?;
                let times: Vec<Option<i64>> = train_index
                    .iter()
                    .map(|&i| entity_time[i as usize])
                    .collect();
                // `sort_values(ascending=False, kind='stable')`: ties keep
                // their order, `NaT` last.
                let mut perm: Vec<usize> = (0..times.len()).collect();
                perm.sort_by(|&a, &b| match (times[a], times[b]) {
                    (Some(x), Some(y)) => y.cmp(&x),
                    (Some(_), None) => std::cmp::Ordering::Less,
                    (None, Some(_)) => std::cmp::Ordering::Greater,
                    (None, None) => std::cmp::Ordering::Equal,
                });
                train_index = perm.iter().map(|&p| train_index[p]).collect();
                train_time = perm.iter().map(|&p| times[p]).collect();
            }
            Anchor::At(t) => train_time = to_ser(t, train_index.len()),
        }
    } else {
        if num_test_examples > 0 {
            if is_forecasting {
                let Anchor::At(mut current) = test_anchor else {
                    return Err(Raised::new("AssertionError", ""));
                };
                let step = step_offset.ok_or_else(|| Raised::new("AssertionError", ""))?;
                let idx = forecast_entity_idx.ok_or_else(|| Raised::new("AssertionError", ""))?;
                for _ in 0..forecast_num.max(0) {
                    test_index.push(idx);
                    test_time.push(Some(current));
                    current = plus(current, step)?;
                }
            } else {
                test_index = get_valid_entity_index(&mut rng, test_anchor, Some(num_test_trials));
                let Anchor::At(t) = test_anchor else {
                    return Err(Raised::new("AssertionError", ""));
                };
                test_time = to_ser(t, test_index.len());
            }
        }
        if is_static && num_train_examples > 0 {
            train_index = get_valid_entity_index(&mut rng, train_anchor, Some(num_train_trials));
            let Anchor::At(t) = train_anchor else {
                return Err(Raised::new("AssertionError", ""));
            };
            train_time = to_ser(t, train_index.len());
        } else if is_temporal && num_train_examples > 0 {
            let step = step_offset.ok_or_else(|| Raised::new("AssertionError", ""))?;
            let mut indices: Vec<Vec<i64>> = Vec::new();
            let mut times: Vec<Vec<Option<i64>>> = Vec::new();
            loop {
                let index = if is_forecasting {
                    vec![forecast_entity_idx.ok_or_else(|| Raised::new("AssertionError", ""))?]
                } else {
                    get_valid_entity_index(&mut rng, train_anchor, Some(num_train_trials))
                };
                let Anchor::At(t) = train_anchor else {
                    return Err(Raised::new("AssertionError", ""));
                };
                times.push(to_ser(t, index.len()));
                indices.push(index);
                if indices.iter().map(Vec::len).sum::<usize>() as i64 >= num_train_trials {
                    break;
                }
                train_anchor = Anchor::At(minus(t, step)?);
                let Anchor::At(moved) = train_anchor else {
                    unreachable!()
                };
                let min_time = sampler.get_min_time(&db, Some(&aggr_table_names)).await?;
                if moved < min_time {
                    break;
                }
            }
            if is_forecasting {
                indices.reverse();
                times.reverse();
            }
            train_index = indices.concat();
            train_index.truncate(num_train_trials.max(0) as usize);
            train_time = times.concat();
            train_time.truncate(num_train_trials.max(0) as usize);
        }
    }

    // 4. Sample training and test labels.
    let labels = if train_index.is_empty() && test_index.is_empty() {
        Labels {
            train_y: empty_float_series(),
            train_mask: Vec::new(),
            test_y: empty_float_series(),
            test_mask: Vec::new(),
        }
    } else {
        sample_target_labels(
            sampler,
            &db,
            &query,
            &entity_df,
            &entity_pkey,
            (&train_index, &train_time),
            if shared_train_test {
                num_train_examples + num_test_examples
            } else {
                num_train_examples
            },
            (&test_index, &test_time),
            if shared_train_test {
                0
            } else {
                num_test_examples
            },
            &columns_dict,
            &time_offsets,
        )
        .await?
    };

    // 5. Post-processing.
    let (train_pkey, train_y, test_pkey, test_y): (Series, Series, Series, Series);
    let (out_train_time, out_test_time): (Vec<Option<i64>>, Vec<Option<i64>>);
    if shared_train_test {
        let num_examples = (num_train_examples + num_test_examples).max(0) as usize;
        let mut index = select(&train_index, &labels.train_mask);
        index.truncate(num_examples);
        let mut time = select_times(&train_time, &labels.train_mask);
        time.truncate(num_examples);
        let y = head(&labels.train_y, num_examples);

        let mut num_test = num_test_examples;
        let num_train = num_train_examples.min(1000);
        let available = y.len() as i64;
        if num_test_examples > 0
            && num_train_examples > 0
            && (available as usize) < num_examples
            && available < num_test + num_train
        {
            // Not enough labels for the split asked for without losing many
            // training examples.
            num_test = available - num_train;
            if num_test < num_train {
                num_test = available / 2;
            }
        }
        let cut = slice_at(index.len(), num_test);
        test_pkey = entity_pkey.take(&positions(&index[..cut]))?;
        out_test_time = time[..slice_at(time.len(), num_test)].to_vec();
        test_y = head(&y, slice_at(y.len(), num_test));
        train_pkey = entity_pkey.take(&positions(&index[cut..]))?;
        out_train_time = time[slice_at(time.len(), num_test)..].to_vec();
        train_y = tail(&y, slice_at(y.len(), num_test));
    } else {
        let mut index = select(&train_index, &labels.train_mask);
        index.truncate(num_train_examples.max(0) as usize);
        train_pkey = entity_pkey.take(&positions(&index))?;
        let mut time = select_times(&train_time, &labels.train_mask);
        time.truncate(num_train_examples.max(0) as usize);
        out_train_time = time;
        train_y = head(&labels.train_y, num_train_examples.max(0) as usize);

        let mut index = select(&test_index, &labels.test_mask);
        index.truncate(num_test_examples.max(0) as usize);
        test_pkey = entity_pkey.take(&positions(&index))?;
        let mut time = select_times(&test_time, &labels.test_mask);
        time.truncate(num_test_examples.max(0) as usize);
        out_test_time = time;
        test_y = head(&labels.test_y, num_test_examples.max(0) as usize);
    }

    if num_train_examples > 0 && train_y.is_empty() {
        return Err(empty_sample_error(
            sampler,
            &db,
            &query,
            requested_train_anchor,
            "Failed to collect any context examples.",
            "context",
        )
        .await);
    }
    if num_test_examples > 0 && test_y.is_empty() {
        return Err(empty_sample_error(
            sampler,
            &db,
            &query,
            test_anchor,
            "Failed to collect any test examples for evaluation.",
            "test",
        )
        .await);
    }

    // The warnings reading the rows gave, then the coverage warning.
    let mut warnings = sampler.take_warnings();
    let enough_rows = entity_df.len() as i64 >= num_entity_rows;
    if num_train_examples > 0
        && !sampler.coverage_warned
        && enough_rows
        && (train_y.len() as i64) < num_train_examples / 2
    {
        sampler.coverage_warned = true;
        warnings.push(coverage_warning(
            num_train_examples,
            num_train_trials,
            "context",
        ));
    }
    if num_test_examples > 0
        && !sampler.coverage_warned
        && enough_rows
        && (test_y.len() as i64) < num_test_examples / 2
    {
        sampler.coverage_warned = true;
        warnings.push(coverage_warning(num_test_examples, num_test_trials, "test"));
    }

    Ok(TargetDone {
        train: TargetPart {
            entity_pkey: train_pkey,
            anchor_time: out_train_time,
            target: train_y,
        },
        test: TargetPart {
            entity_pkey: test_pkey,
            anchor_time: out_test_time,
            target: test_y,
        },
        warnings,
    })
}

/// Python's slice bound `[:n]` of a sequence of `len`, for any `n`.
fn slice_at(len: usize, n: i64) -> usize {
    if n >= 0 {
        (n as usize).min(len)
    } else {
        len.saturating_sub(n.unsigned_abs() as usize)
    }
}

fn coverage_warning(examples: i64, trials: i64, kind: &str) -> Warning {
    Warning::user(format!(
        "Failed to collect {} {kind} examples within {} candidates. To improve coverage, \
         consider increasing the number of PQ iterations using the 'max_pq_iterations' option. \
         This warning will not be shown again in this run.",
        sdm_schema::py::thousands(examples),
        sdm_schema::py::thousands(trials)
    ))
}

fn entity_id_key(id: &super::query::EntityId) -> Key {
    use super::query::EntityId;
    match id {
        EntityId::None => Key::Null,
        EntityId::Bool(b) => Key::Int(i64::from(*b)),
        EntityId::Int(i) => Key::Int(*i),
        EntityId::Float(f) => Key::Float(*f),
        EntityId::Text(s) => Key::Text(s.clone()),
    }
}

fn entity_id_str(id: &super::query::EntityId) -> String {
    use super::query::EntityId;
    match id {
        EntityId::None => "None".to_string(),
        EntityId::Bool(b) => if *b { "True" } else { "False" }.to_string(),
        EntityId::Int(i) => i.to_string(),
        EntityId::Float(f) => sdm_schema::py::float_repr(*f),
        EntityId::Text(s) => s.clone(),
    }
}

/// `entity_pkey.to_numpy() == entity_id`, as numpy compares a key with one.
fn equal_keys(key: &Key, wanted: &Key) -> bool {
    match (key, wanted) {
        (Key::Null, _) | (_, Key::Null) => false,
        (a, b) => a.id() == b.id(),
    }
}

/// `_sample_target`: the labels of the training and test candidates, by
/// the backend's family.
#[allow(clippy::too_many_arguments)]
async fn sample_target_labels(
    sampler: &mut Sampler,
    db: &Db,
    query: &SamplerQuery,
    entity_df: &Frame,
    entity_pkey: &Series,
    train: (&[i64], &[Option<i64>]),
    num_train_examples: i64,
    test: (&[i64], &[Option<i64>]),
    num_test_examples: i64,
    columns_dict: &Columns,
    time_offsets: &TimeOffsets,
) -> Result<Labels, Raised> {
    match sampler.backend {
        Backend::Local | Backend::Duckdb | Backend::Sqlite => {
            let (train_y, train_mask) = sample_target_set(
                sampler,
                db,
                query,
                entity_df,
                entity_pkey,
                train,
                num_train_examples,
                columns_dict,
                time_offsets,
            )
            .await?;
            let (test_y, test_mask) = sample_target_set(
                sampler,
                db,
                query,
                entity_df,
                entity_pkey,
                test,
                num_test_examples,
                columns_dict,
                time_offsets,
            )
            .await?;
            Ok(Labels {
                train_y,
                train_mask,
                test_y,
                test_mask,
            })
        }
        // PostgreSQL, Snowflake and Databricks ask for every candidate at
        // once, to pay the queries' start-up cost once.
        _ => {
            let index: Vec<i64> = [train.0, test.0].concat();
            let frame = entity_df.take(&positions(&index))?;
            let time: Vec<Option<i64>> = [train.1, test.1].concat();
            let pkey = sampler.primary_key(&query.entity_table)?.clone();
            let (y, mask) = labels_of(
                sampler,
                db,
                query,
                &frame,
                &pkey,
                &time,
                columns_dict,
                time_offsets,
            )
            .await?;
            let train_mask = mask[..train.0.len().min(mask.len())].to_vec();
            let test_mask = mask[train.0.len().min(mask.len())..].to_vec();
            let boundary = train_mask.iter().filter(|&&m| m).count();
            Ok(Labels {
                train_y: head(&y, boundary),
                train_mask,
                test_y: tail(&y, boundary),
                test_mask,
            })
        }
    }
}

/// The labels of the entities of `frame` at `time`, with the rows the
/// query reads (`_sample_query_data`, then the executor).
#[allow(clippy::too_many_arguments)]
async fn labels_of(
    sampler: &mut Sampler,
    db: &Db,
    query: &SamplerQuery,
    frame: &Frame,
    pkey: &str,
    time: &[Option<i64>],
    columns_dict: &Columns,
    time_offsets: &TimeOffsets,
) -> Result<(Series, Vec<bool>), Raised> {
    let entity_table = &query.entity_table;
    let entity_pkey = frame.column(pkey)?.clone();
    let others = columns_dict.without(entity_table);
    let (mut feat, mut times, batch) = super::sql::sample_query_data(
        sampler,
        db,
        entity_table,
        &entity_pkey,
        time,
        &others,
        time_offsets,
    )
    .await?;
    set_frame(&mut feat, entity_table, frame.clone());
    if let Some(column) = sampler.time_column(entity_table)
        && columns_dict
            .get(entity_table)
            .is_some_and(|c| c.contains(column))
    {
        set_series(&mut times, entity_table, frame.column(column)?.clone());
    }
    sampler.execute_query(query, &feat, &times, &batch, time, true)
}

fn set_frame(feat: &mut Vec<(String, Frame)>, table: &str, frame: Frame) {
    match feat.iter_mut().find(|(t, _)| t == table) {
        Some((_, f)) => *f = frame,
        None => feat.push((table.to_string(), frame)),
    }
}

fn set_series(times: &mut Vec<(String, Series)>, table: &str, series: Series) {
    match times.iter_mut().find(|(t, _)| t == table) {
        Some((_, s)) => *s = series,
        None => times.push((table.to_string(), series)),
    }
}

/// `_sample_target_set`: the labels of one set of candidates, in batches of
/// 10,000, stopping once enough are labelled.
#[allow(clippy::too_many_arguments)]
async fn sample_target_set(
    sampler: &mut Sampler,
    db: &Db,
    query: &SamplerQuery,
    entity_df: &Frame,
    entity_pkey: &Series,
    set: (&[i64], &[Option<i64>]),
    num_examples: i64,
    columns_dict: &Columns,
    time_offsets: &TimeOffsets,
) -> Result<(Series, Vec<bool>), Raised> {
    const BATCH_SIZE: usize = 10_000;
    let (index, anchor) = set;
    let mut count = 0i64;
    let mut ys: Vec<Series> = Vec::new();
    let mut mask = vec![false; index.len()];
    let mut start = 0;
    while start < index.len() {
        let stop = (start + BATCH_SIZE).min(index.len());
        let rows = positions(&index[start..stop]);
        let anchor_part = &anchor[start.min(anchor.len())..stop.min(anchor.len())];
        let (y, part_mask) = if sampler.backend == Backend::Local {
            let pkey = entity_pkey.take(&rows)?;
            super::local::target_batch(
                sampler,
                query,
                &pkey,
                anchor_part,
                columns_dict,
                time_offsets,
            )?
        } else {
            let frame = entity_df.take(&rows)?;
            let pkey = sampler.primary_key(&query.entity_table)?.clone();
            labels_of(
                sampler,
                db,
                query,
                &frame,
                &pkey,
                anchor_part,
                columns_dict,
                time_offsets,
            )
            .await?
        };
        // `mask[start:start + batch_size] = _mask`, which numpy takes only
        // of the slice's length.
        if part_mask.len() != stop - start {
            return Err(Raised::value_error(format!(
                "could not broadcast input array from shape ({},) into shape ({},)",
                part_mask.len(),
                stop - start
            )));
        }
        mask[start..stop].copy_from_slice(&part_mask);
        count += y.len() as i64;
        ys.push(y);
        if count >= num_examples {
            break;
        }
        start = stop;
    }
    let y = concat_labels(&ys)?;
    Ok((y, mask))
}

/// `_empty_sample_error`: the error an empty sample raises, typed by what
/// caused it.
async fn empty_sample_error(
    sampler: &mut Sampler,
    db: &Db,
    query: &SamplerQuery,
    anchor_time: Anchor,
    opening: &str,
    examples: &str,
) -> Raised {
    let (reason, earliest) =
        match diagnose_empty_sample(sampler, db, query, anchor_time, examples).await {
            Ok(found) => found,
            Err(raised) => return raised,
        };
    let class = if earliest.is_none() {
        "ValueError"
    } else {
        "NoEligibleEntityError"
    };
    Raised::new(class, format!("{opening}{reason}"))
}

/// `pd.Timestamp`'s `str`: `YYYY-MM-DD HH:MM:SS[.ffffff[fff]]`.
pub fn timestamp_text(ns: i64) -> String {
    let wall = sdm_dialects::time::Civil::from_epoch_nanos(i128::from(ns));
    let mut text = format!("{} {}", wall.date_text(), wall.time_text());
    let fraction = ns.rem_euclid(1_000_000_000);
    if fraction != 0 {
        if fraction % 1000 == 0 {
            text.push_str(&format!(".{:06}", fraction / 1000));
        } else {
            text.push_str(&format!(".{fraction:09}"));
        }
    }
    text
}

/// `_no_entity_at_anchor`: when the entity table starts after the anchor,
/// that start.
async fn no_entity_at_anchor(
    sampler: &mut Sampler,
    db: &Db,
    query: &SamplerQuery,
    anchor_time: Anchor,
) -> Option<i64> {
    let Anchor::At(anchor) = anchor_time else {
        return None;
    };
    let entity_table = query.entity_table.clone();
    sampler.time_column(&entity_table)?;
    let earliest = sampler.get_min_time(db, Some(&[entity_table])).await.ok()?;
    if earliest == TIMESTAMP_MAX {
        return None;
    }
    (earliest > anchor).then_some(earliest)
}

/// `_window_fits_after`.
fn window_fits_after(
    aggregations: &[Aggregation],
    anchor: i64,
    max_time: i64,
) -> Result<bool, Raised> {
    let mut ends = Vec::new();
    for aggregation in aggregations {
        let Some(window) = &aggregation.window else {
            return Ok(false);
        };
        ends.push(window.end_offset());
    }
    if ends.is_empty() {
        return Ok(false);
    }
    let offset = max_date_offset(&ends)?;
    Ok(offset.apply(anchor, 1)? <= max_time)
}

/// `_diagnose_empty_sample`.
async fn diagnose_empty_sample(
    sampler: &mut Sampler,
    db: &Db,
    query: &SamplerQuery,
    anchor_time: Anchor,
    examples: &str,
) -> Result<(String, Option<i64>), Raised> {
    let generic = " Is your predictive query too restrictive?".to_string();
    let Anchor::At(anchor) = anchor_time else {
        return Ok((generic, None));
    };
    let kind = if examples == "context" {
        "Context"
    } else {
        "Test"
    };

    let mut aggregations = query.target_aggregations.clone();
    let mut table_names: Vec<String> = aggregations.iter().map(|a| a.table.clone()).collect();
    table_names.sort();
    table_names.dedup();
    let (mut max_time, mut min_time) = (None, None);
    if !table_names.is_empty() {
        match sampler.get_max_time(db, Some(&table_names)).await {
            Ok(max) => match sampler.get_min_time(db, Some(&table_names)).await {
                Ok(min) => {
                    max_time = Some(max);
                    min_time = Some(min);
                }
                Err(_) => aggregations.clear(),
            },
            Err(_) => aggregations.clear(),
        }
    }

    if let Some(earliest) = no_entity_at_anchor(sampler, db, query, anchor_time).await {
        let entity_table = &query.entity_table;
        let column = sampler.time_column(entity_table).cloned();
        let later_fits = max_time
            .map(|max| window_fits_after(&aggregations, earliest, max).unwrap_or(false))
            .unwrap_or(false);
        let later = if later_fits {
            format!(
                ", or pass an anchor_time after {}",
                timestamp_text(earliest)
            )
        } else {
            String::new()
        };
        let table_repr = sdm_schema::py::str_repr(entity_table);
        let column_repr = column
            .as_deref()
            .map_or_else(|| "None".to_string(), sdm_schema::py::str_repr);
        return Ok((
            format!(
                " No row of {table_repr} existed at the anchor: it is {}, and the earliest \
                 {column_repr} is {}. {kind} examples only ask about an entity that existed by \
                 its anchor. If {column_repr} does not record when each entity began, set the \
                 time column of {table_repr} to None{later}.",
                timestamp_text(anchor),
                timestamp_text(earliest),
            ),
            Some(earliest),
        ));
    }

    let Some(max_time) = max_time else {
        return Ok((generic, None));
    };
    let listed = table_names
        .iter()
        .map(|n| sdm_schema::py::str_repr(n))
        .collect::<Vec<_>>()
        .join(", ");
    let holds = if table_names.len() == 1 {
        "holds"
    } else {
        "hold"
    };
    let facts = format!(
        "it is {}, and {listed} {holds} data from {} to {}",
        timestamp_text(anchor),
        min_time.map_or_else(|| "None".to_string(), timestamp_text),
        timestamp_text(max_time)
    );
    // An error here is raised, as Python doesn't catch it.
    if window_fits_after(&aggregations, anchor, max_time)? {
        let Some(condition) = &query.entity_filter_condition else {
            return Ok((generic, None));
        };
        return Ok((
            format!(
                " The anchor time is not the reason: {facts}, which leaves room after it for \
                 the query's whole window. No entity passed {}, so there was nothing to sample: \
                 check that the values it compares against occur in the data, including their \
                 case.",
                sdm_schema::py::str_repr(condition)
            ),
            None,
        ));
    }
    let place = if examples == "context" {
        "come from before the anchor"
    } else {
        "are asked at the anchor"
    };
    Ok((
        format!(
            " The anchor time is most likely the reason: {facts}. {kind} examples {place} and \
             each needs the query's full window of data after it, so an anchor at or near the \
             end of the range leaves none to draw from. Pass an earlier anchor_time, or shorten \
             the window the query asks about."
        ),
        None,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unseeded_streams_differ_however_close_together_they_start() {
        let streams: Vec<Vec<u32>> = (0..64)
            .map(|_| {
                let mut state = random_state(None).unwrap();
                (0..4).map(|_| state.next_u32()).collect()
            })
            .collect();
        for (i, a) in streams.iter().enumerate() {
            for b in &streams[i + 1..] {
                assert_ne!(a, b);
            }
        }
    }

    #[test]
    fn a_seed_gives_numpys_stream() {
        // np.random.RandomState(7).randint(0, 2**31, 2)
        let mut state = random_state(Some(&Int::parse("7").unwrap())).unwrap();
        assert_eq!(
            state.randint(0, 1 << 31, 2).unwrap(),
            [327_741_615, 976_413_892]
        );
    }
}
