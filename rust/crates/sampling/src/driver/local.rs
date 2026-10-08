// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `backend/local/sampler.py::LocalSampler`: the in-memory backend, over the
//! tables the host hands the run, through Phase 7's graph store and
//! neighbor sampler.

use std::sync::{Mutex, PoisonError};

use sdm_dialects::int::Int;
use sdm_labels::date_offset::Bound;
use sdm_schema::dt::Host;

use super::inputs::Inputs;
use super::query::{EntityId, SamplerQuery};
use super::sampler::{Sampler, Through};
use super::sql::{SamplerOutput, dtype_of_name, list_kind, to_datetime_series};
use super::target::{Columns, TimeOffsets, random_state};
use super::{Raised, inputs::frame_of};
use crate::frame::{Frame, Series};
use crate::graph_store::{
    ColumnSpec, EdgeSpec, GraphSpec, GraphStore, KeyValue, TableSpec, graph_store,
};
use crate::keys::{Index, Key, keys_of, list_repr, np_unique};
use crate::neighbor::{NeighborSampler, Seed, TimeRanges, cast_error, rel_type};
use crate::sequences::to_list_with_policy;
use sdm_dialects::py::Dtype as PqlDtype;

/// `LocalGraphStore` and the `relationallib.NeighborSampler` over it.
pub struct LocalBackend {
    store: GraphStore,
    /// `df_dict`: each table, sanitized.
    frames: Vec<(String, Frame)>,
    /// `pkey_map_dict`: each keyed table's valid keys and their rows.
    pkey_maps: Vec<(String, PkeyMap)>,
    neighbor: Mutex<NeighborSampler>,
    /// What reading the tables' times gave as the graph was built.
    built: Vec<super::output::Warning>,
}

struct PkeyMap {
    keys: Vec<Key>,
    index: Index,
    rows: Vec<i64>,
}

fn key_of(value: &KeyValue) -> Key {
    match value {
        KeyValue::Int(v) => Key::Int(*v),
        KeyValue::Float(v) => Key::Float(*v),
        KeyValue::Text(v) => Key::Text(v.clone()),
    }
}

/// The sampler's generator seeded as `seed(random_seed)` seeds it.
pub fn sampler_seed(seed: &Int) -> Result<Seed, Raised> {
    if let Some(value) = seed.to_i64()
        && (-(1i64 << 31)..(1i64 << 32)).contains(&value)
    {
        return Ok(Seed::Word(value));
    }
    if seed.is_negative() {
        return Err(Raised::new(
            "ValueError",
            "a seed is an integer from -2**31 up",
        ));
    }
    Ok(Seed::Words(seed.magnitude_words()))
}

impl LocalBackend {
    pub fn new(inputs: &Inputs, host: &Host) -> Result<Self, Raised> {
        let spec = GraphSpec {
            tables: inputs
                .tables
                .iter()
                .map(|t| TableSpec {
                    name: t.name.clone(),
                    primary_key: t.primary_key.clone(),
                    time_column: t.time_column.clone(),
                    end_time_column: t.end_time_column.clone(),
                    columns: t
                        .columns
                        .iter()
                        .map(|c| ColumnSpec {
                            name: c.name.clone(),
                            dtype: c.dtype.clone().unwrap_or_default(),
                            stype: c.stype.clone().unwrap_or_default(),
                        })
                        .collect(),
                })
                .collect(),
            edges: inputs
                .edges
                .iter()
                .map(|(src, fkey, dst)| EdgeSpec {
                    src_table: src.clone(),
                    fkey: fkey.clone(),
                    dst_table: dst.clone(),
                })
                .collect(),
        };
        let mut data = Vec::new();
        let mut frames = Vec::new();
        let mut built = Vec::new();
        for table in &inputs.tables {
            let batch = inputs.data(&table.name)?;
            let frame = frame_of(batch);
            data.push((
                table.name.clone(),
                frame
                    .names
                    .iter()
                    .cloned()
                    .zip(frame.columns.iter().map(|s| s.array.clone()))
                    .collect(),
            ));
            // `Table._sanitize`: each time column as `datetime64[ns]`, and
            // each list or sequence column held as text as lists. What it
            // warns of, it warned of when the graph was built, not here.
            let mut sanitized = frame;
            for (i, name) in sanitized.names.clone().iter().enumerate() {
                let Some(column) = table.column(name) else {
                    continue;
                };
                let dtype = column.dtype.as_deref().and_then(dtype_of_name);
                let stype = column.stype.as_deref();
                if dtype == Some(PqlDtype::Time) || stype == Some("timestamp") {
                    sanitized.columns[i] =
                        to_datetime_series(&sanitized.columns[i], name, host, &mut built)?;
                } else if let Some(kind) = list_kind(dtype, stype) {
                    sanitized.columns[i] = to_list_with_policy(
                        &sanitized.columns[i],
                        kind,
                        host.numpy_text_prefix,
                        &mut built,
                    )?;
                }
            }
            frames.push((table.name.clone(), sanitized));
        }
        let store = graph_store(&spec, &data, host).map_err(|e| match e.key {
            Some(crate::graph_store::MissingKey::Name(key)) => Raised::key_error(&key),
            Some(crate::graph_store::MissingKey::None) => Raised::key_error("None"),
            None => Raised::new(&e.code, e.message),
        })?;
        let pkey_maps = store
            .keys
            .iter()
            .zip(&store.node_ids)
            .map(|((table, keys), (_, rows))| {
                let keys: Vec<Key> = keys.iter().map(key_of).collect();
                let index = Index::new(&keys);
                (
                    table.clone(),
                    PkeyMap {
                        keys,
                        index,
                        rows: rows.clone(),
                    },
                )
            })
            .collect();
        let node_types: Vec<String> = inputs.tables.iter().map(|t| t.name.clone()).collect();
        let mut edge_types = Vec::new();
        for edge in &inputs.edges {
            edge_types.push(edge.clone());
            edge_types.push(crate::graph_store::rev_edge_type(edge));
        }
        let mut neighbor = NeighborSampler::new(
            node_types,
            edge_types,
            store
                .colptr
                .iter()
                .map(|(e, v)| (rel_type(e), v.clone()))
                .collect(),
            store
                .row
                .iter()
                .map(|(e, v)| (rel_type(e), v.clone()))
                .collect(),
            store.time.clone(),
            &Seed::Word(123),
        )
        .map_err(sample_error)?;
        if let Some((key, pos)) = &inputs.state.neighbor_generator {
            let state = sdm_random::RandomState::from_key(key, *pos)
                .map_err(|e| Raised::value_error(e.to_string()))?;
            neighbor.resume(state.into());
        }
        Ok(Self {
            store,
            frames,
            pkey_maps,
            neighbor: Mutex::new(neighbor),
            built,
        })
    }

    /// The neighbor sampler's generator as the draws so far left it.
    pub fn generator(&self) -> (Vec<u32>, usize) {
        let neighbor = self.neighbor.lock().unwrap_or_else(PoisonError::into_inner);
        let (key, pos) = neighbor.generator().state().key();
        (key.to_vec(), pos)
    }

    /// `LocalGraphStore.sanitization_report`'s tables: each table's rows in
    /// and out, and why the rest were dropped.
    pub fn sanitization_report(&self) -> &[(String, crate::graph_store::TableReport)] {
        &self.store.report
    }

    /// The warnings building the graph gave: reading its tables' times.
    pub fn built_warnings(&self) -> &[super::output::Warning] {
        &self.built
    }

    fn frame(&self, table: &str) -> Result<&Frame, Raised> {
        self.frames
            .iter()
            .find(|(t, _)| t == table)
            .map(|(_, f)| f)
            .ok_or_else(|| Raised::key_error(table))
    }

    /// `_get_min_max_time_dict(table_names)`.
    pub fn min_max_time_dict(&self, table_names: &[String]) -> Vec<(String, (i64, i64))> {
        self.store
            .time_range
            .iter()
            .filter(|(t, _)| table_names.contains(t))
            .cloned()
            .collect()
    }

    /// `LocalGraphStore.get_node_id(table, pkey)`.
    pub fn get_node_id(&self, table: &str, pkey: &Series) -> Result<Vec<i64>, Raised> {
        if self.frames.iter().all(|(t, _)| t != table) {
            return Err(Raised::key_error(&format!(
                "Table '{table}' does not exist"
            )));
        }
        let Some((_, map)) = self.pkey_maps.iter().find(|(t, _)| t == table) else {
            return Err(Raised::value_error(format!(
                "Table '{table}' does not have a primary key"
            )));
        };
        if pkey.is_empty() {
            return Err(Raised::key_error(&format!(
                "No primary keys passed for table '{table}'"
            )));
        }
        let given = keys_of(pkey)?;
        let target = map.keys.first().cloned().unwrap_or(Key::Null);
        let cast = given
            .iter()
            .map(|k| cast_key(k, &target))
            .collect::<Option<Vec<Key>>>()
            .ok_or_else(|| {
                Raised::value_error(format!(
                    "Could not cast primary keys {} to the expected data type '{}'",
                    list_repr(&given),
                    match target {
                        Key::Text(_) => "object",
                        Key::Float(_) => "float64",
                        _ => "int64",
                    }
                ))
            })?;
        let mut out = Vec::with_capacity(cast.len());
        let mut missing = Vec::new();
        for key in &cast {
            match map.index.get(key) {
                Some(position) => out.push(map.rows[position as usize]),
                None => missing.push(key.clone()),
            }
        }
        if !missing.is_empty() {
            return Err(Raised::key_error(&format!(
                "The primary keys {} do not exist in the '{table}' table",
                list_repr(&missing)
            )));
        }
        Ok(out)
    }

    /// `LocalGraphStore.validate_entity_references(table, pkey)`: a
    /// `TaskReferenceError` counting the keys no row the store kept holds.
    pub fn validate_entity_references(&self, table: &str, pkey: &Series) -> Result<(), Raised> {
        let Some((_, map)) = self.pkey_maps.iter().find(|(t, _)| t == table) else {
            return Ok(());
        };
        let target = map.keys.first().cloned().unwrap_or(Key::Null);
        let missing = keys_of(pkey)?
            .iter()
            .filter(|k| cast_key(k, &target).is_none_or(|k| map.index.get(&k).is_none()))
            .count();
        if missing > 0 {
            return Err(Raised::new(
                "TaskReferenceError",
                format!(
                    "Task references {} unresolved row(s) in table '{table}' after graph \
                     sanitization",
                    sdm_schema::py::thousands(missing as i64)
                ),
            ));
        }
        Ok(())
    }

    /// `_sample_entity_table(table, columns, num_rows, random_seed,
    /// entity_ids)`.
    pub fn sample_entity_table(
        &self,
        _sampler: &Sampler,
        table: &str,
        columns: &[String],
        num_rows: i64,
        random_seed: Option<&Int>,
        entity_ids: Option<&[EntityId]>,
    ) -> Result<Frame, Raised> {
        let (_, map) = self
            .pkey_maps
            .iter()
            .find(|(t, _)| t == table)
            .ok_or_else(|| Raised::key_error(table))?;
        let rows: Vec<i64> = match entity_ids {
            Some(ids) => {
                let wanted: Vec<Key> = ids.iter().map(entity_key).collect();
                let wanted = Index::new(&wanted);
                map.keys
                    .iter()
                    .zip(&map.rows)
                    .filter(|(k, _)| wanted.get(k).is_some())
                    .map(|(_, r)| *r)
                    .collect()
            }
            None if map.rows.len() as i64 > num_rows => {
                let mut rng = random_state(random_seed)?;
                let drawn = rng
                    .choice(map.rows.len() as i64, num_rows.max(0) as usize)
                    .map_err(|e| Raised::value_error(e.to_string()))?;
                drawn.iter().map(|&i| map.rows[i as usize]).collect()
            }
            None => map.rows.clone(),
        };
        let frame = self.frame(table)?;
        let positions: Vec<usize> = rows.iter().map(|&r| r as usize).collect();
        frame.take(&positions)?.select_in_order(columns)
    }
}

fn entity_key(id: &EntityId) -> Key {
    match id {
        EntityId::None => Key::Null,
        EntityId::Bool(b) => Key::Int(i64::from(*b)),
        EntityId::Int(i) => Key::Int(*i),
        EntityId::Float(f) => Key::Float(*f),
        EntityId::Text(s) => Key::Text(s.clone()),
    }
}

/// `pkey.astype(type(index[0]))`: a key cast to the kind of the table's
/// keys, `None` where the cast fails.
fn cast_key(key: &Key, target: &Key) -> Option<Key> {
    match (target, key) {
        (_, Key::Null) => None,
        (Key::Int(_), Key::Int(v)) => Some(Key::Int(*v)),
        (Key::Int(_), Key::Float(v)) if v.is_finite() => Some(Key::Int(v.trunc() as i64)),
        (Key::Int(_), Key::Text(s)) => s.trim().parse().ok().map(Key::Int),
        (Key::Float(_), Key::Int(v)) => Some(Key::Float(*v as f64)),
        (Key::Float(_), Key::Float(v)) => Some(Key::Float(*v)),
        (Key::Float(_), Key::Text(s)) => s.trim().parse().ok().map(Key::Float),
        (Key::Text(_), Key::Text(s)) => Some(Key::Text(s.clone())),
        (Key::Text(_), Key::Int(v)) => Some(Key::Text(v.to_string())),
        (Key::Text(_), Key::Float(v)) => Some(Key::Text(sdm_schema::py::float_repr(*v))),
        _ => None,
    }
}

fn sample_error(error: crate::neighbor::SampleError) -> Raised {
    use crate::neighbor::SampleError;
    match error {
        SampleError::Runtime(message) => Raised::new("RuntimeError", message),
        SampleError::Key(key) => Raised::key_error(&key),
        SampleError::Index(message) => Raised::new("IndexError", message),
    }
}

fn local_of(sampler: &Sampler) -> Result<&LocalBackend, Raised> {
    match &sampler.through {
        Through::Local(local) => Ok(local),
        Through::Sql(_) => Err(Raised::internal("a SQL sampler has no graph store")),
    }
}

/// Seconds of nanosecond times, floored, `NaT` as numpy floors its int64.
fn seconds_of(times: &[Option<i64>]) -> Vec<i64> {
    times
        .iter()
        .map(|t| t.unwrap_or(i64::MIN).div_euclid(1_000_000_000))
        .collect()
}

/// `LocalSampler._sample_query_data` then the executor: the labels of one
/// batch of candidates.
pub fn target_batch(
    sampler: &Sampler,
    query: &SamplerQuery,
    entity_pkey: &Series,
    anchor: &[Option<i64>],
    columns_dict: &Columns,
    time_offsets: &TimeOffsets,
) -> Result<(Series, Vec<bool>), Raised> {
    let (feat, time, batch) = sample_query_data(
        sampler,
        &query.entity_table,
        entity_pkey,
        anchor,
        columns_dict,
        time_offsets,
    )?;
    sampler.execute_query(query, &feat, &time, &batch, anchor, true)
}

/// The rows a query's labels are computed from, by table, with their times
/// and the examples they are for.
pub type QueryData = (
    Vec<(String, Frame)>,
    Vec<(String, Series)>,
    Vec<(String, Vec<i64>)>,
);

/// `LocalSampler._sample_query_data`.
pub fn sample_query_data(
    sampler: &Sampler,
    entity_table: &str,
    entity_pkey: &Series,
    anchor: &[Option<i64>],
    columns_dict: &Columns,
    time_offsets: &TimeOffsets,
) -> Result<QueryData, Raised> {
    let local = local_of(sampler)?;
    let num_hops = usize::from(!time_offsets.is_empty());
    let mut counts: Vec<(String, Vec<i64>)> = Vec::new();
    let mut ranges: Vec<(String, TimeRanges)> = Vec::new();
    for ((src, fkey, dst), (start, end)) in time_offsets {
        let start = match start {
            None => Ok(None),
            Some(start) => seconds_value(start.to_seconds(Bound::Earliest)?).map(Some),
        };
        let end = seconds_value(end.to_seconds(Bound::Latest)?);
        let range = match (start, end) {
            (Ok(start), Ok(end)) => Ok((start, end)),
            (Err(e), _) | (_, Err(e)) => Err(e),
        };
        ranges.push((
            rel_type(&(src.clone(), fkey.clone(), dst.clone())),
            TimeRanges {
                width: 2,
                ranges: vec![range],
            },
        ));
    }
    for edge in &sampler.edge_types {
        if time_offsets.iter().any(|(e, _)| e == edge) {
            continue;
        }
        counts.push((rel_type(edge), vec![0; num_hops]));
    }
    let node_ids = local.get_node_id(entity_table, entity_pkey)?;
    let seconds = seconds_of(anchor);
    let sampled = local
        .neighbor
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .sample(&counts, &ranges, entity_table, &node_ids, &seconds)
        .map_err(sample_error)?;

    let mut feat: Vec<(String, Frame)> = Vec::new();
    let mut time: Vec<(String, Series)> = Vec::new();
    for (table, columns) in &columns_dict.0 {
        let nodes = sampled
            .node_id
            .iter()
            .find(|(t, _)| t == table)
            .map(|(_, n)| n)
            .ok_or_else(|| Raised::key_error(table))?;
        let positions: Vec<usize> = nodes.iter().map(|&n| n as usize).collect();
        let frame = local
            .frame(table)?
            .take(&positions)?
            .select_in_order(columns)?;
        if let Some(column) = sampler.time_column(table)
            && columns.contains(column)
        {
            time.push((table.clone(), frame.column(column)?.clone()));
        }
        feat.push((table.clone(), frame));
    }
    Ok((feat, time, sampled.batch))
}

/// A span in seconds as the sampler casts it to `int64`, or pybind11's
/// error for one past it.
fn seconds_value(seconds: sdm_labels::date_offset::Seconds) -> Result<i64, String> {
    seconds
        .to_i128()
        .and_then(|s| i64::try_from(s).ok())
        .ok_or_else(|| cast_error("int"))
}

/// `LocalSampler._sample_subgraph`.
#[allow(clippy::too_many_arguments)]
pub fn sample_subgraph(
    sampler: &Sampler,
    entity_table: &str,
    entity_pkey: &Series,
    anchor_time: Option<&[Option<i64>]>,
    columns_dict: &Columns,
    num_neighbors: &[i64],
    random_seed: Option<&Int>,
) -> Result<SamplerOutput, Raised> {
    let local = local_of(sampler)?;
    let index = local.get_node_id(entity_table, entity_pkey)?;
    let time: Vec<i64> = match anchor_time {
        Some(anchor) => seconds_of(anchor),
        None => {
            let (_, times) = local
                .store
                .time
                .iter()
                .find(|(t, _)| t == entity_table)
                .ok_or_else(|| Raised::key_error(entity_table))?;
            index
                .iter()
                .map(|&i| {
                    times
                        .get(i as usize)
                        .copied()
                        .ok_or_else(|| Raised::new("IndexError", "index out of bounds"))
                })
                .collect::<Result<_, _>>()?
        }
    };
    let mut neighbor = local
        .neighbor
        .lock()
        .unwrap_or_else(PoisonError::into_inner);
    if let Some(seed) = random_seed {
        neighbor.seed(&sampler_seed(seed)?);
    }
    let counts: Vec<(String, Vec<i64>)> = sampler
        .edge_types
        .iter()
        .map(|e| (rel_type(e), num_neighbors.to_vec()))
        .collect();
    let sampled = neighbor
        .sample(&counts, &[], entity_table, &index, &time)
        .map_err(sample_error)?;
    drop(neighbor);

    let mut df_dict = Vec::new();
    let mut inverse_dict = Vec::new();
    for (table, nodes) in &sampled.node_id {
        let frame = local.frame(table)?;
        let columns = columns_dict
            .get(table)
            .cloned()
            .ok_or_else(|| Raised::key_error(table))?;
        let end_time_wanted = sampler
            .end_time_column(table)
            .is_some_and(|c| columns.contains(c));
        let positions: Vec<usize> = nodes.iter().map(|&n| n as usize).collect();
        let taken = if end_time_wanted || columns.is_empty() {
            frame.take(&positions)?
        } else {
            let keys: Vec<Key> = nodes.iter().map(|&n| Key::Int(n)).collect();
            let (first, inverse) = np_unique(&keys)?;
            if nodes.len() as f64 > 1.05 * first.len() as f64 {
                let unique: Vec<usize> = first.iter().map(|&i| positions[i]).collect();
                inverse_dict.push((table.clone(), inverse));
                frame.take(&unique)?
            } else {
                frame.take(&positions)?
            }
        };
        df_dict.push((table.clone(), taken.select_in_order(&columns)?));
    }
    let by_edge = |dict: &Vec<(String, Vec<i64>)>| -> Vec<((String, String, String), Vec<i64>)> {
        sampler
            .edge_types
            .iter()
            .map(|e| {
                let key = rel_type(e);
                let values = dict
                    .iter()
                    .find(|(k, _)| *k == key)
                    .map(|(_, v)| v.clone())
                    .unwrap_or_default();
                (e.clone(), values)
            })
            .collect()
    };
    Ok(SamplerOutput {
        anchor_time: time.iter().map(|t| t.wrapping_mul(1_000_000_000)).collect(),
        df_dict,
        inverse_dict,
        batch_dict: sampled.batch.clone(),
        num_sampled_nodes_dict: sampled.num_sampled_nodes.clone(),
        row_dict: by_edge(&sampled.row),
        col_dict: by_edge(&sampled.col),
        num_sampled_edges_dict: by_edge(&sampled.num_sampled_edges),
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{ArrayRef, Int64Array, RecordBatch};
    use serde_json::json;

    use super::*;

    fn ints(values: impl IntoIterator<Item = i64>) -> ArrayRef {
        Arc::new(Int64Array::from(values.into_iter().collect::<Vec<_>>()))
    }

    /// Four users with ten orders each, in memory, the sampler's state as
    /// `state` gives it.
    fn inputs(state: serde_json::Value) -> Inputs {
        let users = RecordBatch::try_from_iter([("user_id", ints(0..4))]).unwrap();
        let orders = RecordBatch::try_from_iter([
            ("order_id", ints(0..40)),
            ("user_id", ints((0..40).map(|i| i % 4))),
        ])
        .unwrap();
        let id = |name: &str| json!({"name": name, "dtype": "int", "stype": "ID"});
        let described = json!({
            "backend": "local",
            "tables": [
                {"name": "users", "primary_key": "user_id", "columns": [id("user_id")]},
                {
                    "name": "orders",
                    "primary_key": "order_id",
                    "columns": [id("order_id"), id("user_id")],
                },
            ],
            "edges": [["orders", "user_id", "users"]],
            "state": state,
            "arguments": {},
        });
        Inputs::parse(
            &described.to_string(),
            vec![("users".into(), users), ("orders".into(), orders)],
        )
        .unwrap()
    }

    /// Two of each user's orders, drawn with the sampler's generator.
    fn draw(sampler: &Sampler) -> Vec<(String, Vec<i64>)> {
        let local = local_of(sampler).unwrap();
        let mut neighbor = local.neighbor.lock().unwrap();
        let counts: Vec<(String, Vec<i64>)> = sampler
            .edge_types
            .iter()
            .map(|e| (rel_type(e), vec![2]))
            .collect();
        neighbor
            .sample(&counts, &[], "users", &[0, 1, 2, 3], &[])
            .unwrap()
            .node_id
    }

    /// Today's model keeps its neighbor sampler between calls, and a call
    /// without a seed draws on from where the last one left the generator:
    /// a sampler built again from the state a run left draws what the first
    /// sampler would have drawn next.
    #[test]
    fn a_sampler_built_from_its_state_draws_on_where_it_left_off() {
        let kept = Sampler::new(&inputs(json!({})), Host::assumed()).unwrap();
        let first = draw(&kept);
        let second = draw(&kept);
        assert_ne!(first, second, "two draws in a row are not the same");

        let once = Sampler::new(&inputs(json!({})), Host::assumed()).unwrap();
        assert_eq!(draw(&once), first);
        let mut left = inputs(json!({}));
        left.state = once.state();
        let again = Sampler::new(&left, Host::assumed()).unwrap();
        assert_eq!(draw(&again), second);
    }
}
