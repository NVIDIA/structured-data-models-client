// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `relationallib.NeighborSampler` (`csrc/neighbor_sampler.cpp`), draw for
//! draw: a heterogeneous graph in CSC form, sampled hop by hop from seed
//! nodes, each (example, node) pair numbered once per node type.
//!
//! An edge type `(dst, rel, src)` is walked from `src` to `dst`: its
//! `colptr` has an entry per `src` node and one more, and its `row` holds
//! `dst` nodes. A hop takes, for each node the last hop reached, either up
//! to a count of its neighbors (all of them for a negative count; at random
//! without replacement when there are more, by Floyd's algorithm on
//! `std::mt19937`; the latest ones before the seed time when the edge's
//! `dst` has times) or those within a time range of the seed time.
//!
//! What the C++ keeps between calls is kept too: the generator continues
//! from one call to the next, and a call that fails part way leaves the
//! per-type tallies it had started, which the next call appends to.

use std::collections::{HashMap, HashSet};
use std::fmt;

use sdm_random::{Mt19937, RandomState};

/// `(dst, rel, src)`.
pub type EdgeType = (String, String, String);

/// `dst__rel__src`, the key of an edge type's arrays.
pub fn rel_type(edge: &EdgeType) -> String {
    format!("{}__{}__{}", edge.0, edge.1, edge.2)
}

/// What the sampler raised, as the C++ raises it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SampleError {
    /// `std::runtime_error`, Python's `RuntimeError`.
    Runtime(String),
    /// The missing key of a Python dict, Python's `KeyError`, whose text is
    /// the key's `repr`.
    Key(String),
    /// `std::out_of_range`, which pybind11 raises as `IndexError`.
    Index(String),
}

impl fmt::Display for SampleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SampleError::Runtime(message) | SampleError::Index(message) => f.write_str(message),
            SampleError::Key(key) => f.write_str(&sdm_schema::py::str_repr(key)),
        }
    }
}

impl std::error::Error for SampleError {}

fn runtime(message: impl Into<String>) -> SampleError {
    SampleError::Runtime(message.into())
}

/// How the sampler's generator is seeded: as `base/utils.random_state`
/// seeds numpy's, from 0 up; a negative seed down to `-2**31` modulo
/// `2**32`, as the C `int` the sampler once took.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Seed {
    /// A seed from `-2**31` to `2**32 - 1`, kept modulo `2**32`.
    Word(i64),
    /// The 32-bit words of a seed from `2**32` up, least significant first.
    Words(Vec<u32>),
}

impl Seed {
    fn generator(&self) -> Mt19937 {
        match self {
            Seed::Word(seed) => Mt19937::new(*seed as u64),
            Seed::Words(words) => match RandomState::from_words(words) {
                Ok(state) => Mt19937::from(state),
                // A seed of two words or more is never empty.
                Err(_) => Mt19937::new(0),
            },
        }
    }
}

/// An edge type's `sampling_time_offsets` entry: its rows, one per hop, each
/// `(start, end)` relative to the seed time, and how wide the array was,
/// which must be 2.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TimeRanges {
    pub width: usize,
    pub ranges: Vec<TimeRange>,
}

/// One row of `sampling_time_offsets`: `(start, end)`, a `None` start open,
/// or the text of pybind11's error casting a value of it to `int64`. The C++
/// casts a row when it samples that hop, so the error is raised then, as a
/// `RuntimeError`.
pub type TimeRange = Result<(Option<i64>, i64), String>;

/// pybind11's text for a value of Python type `type_name` it can't cast to
/// `int64_t`.
pub fn cast_error(type_name: &str) -> String {
    format!("Unable to cast Python instance of type {type_name} to C++ type 'long'")
}

/// One hop's choice for one edge type: how many neighbors, or a time range
/// `(start, end)` relative to the seed time.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HopSpec {
    Count(i64),
    Range(Option<i64>, i64),
}

/// What `sample` gives: `(key, values)` pairs in the C++'s dict order.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Sampled {
    pub row: Vec<(String, Vec<i64>)>,
    pub col: Vec<(String, Vec<i64>)>,
    pub node_id: Vec<(String, Vec<i64>)>,
    pub batch: Vec<(String, Vec<i64>)>,
    pub num_sampled_nodes: Vec<(String, Vec<i64>)>,
    pub num_sampled_edges: Vec<(String, Vec<i64>)>,
}

/// The tallies the C++ keeps in members between `init_placeholders` and
/// `clear_placeholders`.
#[derive(Clone, Debug, Default)]
struct Tallies {
    nodes_per_hop: HashMap<String, Vec<i64>>,
    edges_per_hop: HashMap<String, Vec<i64>>,
    batch: HashMap<String, Vec<i64>>,
    node_ids: HashMap<String, Vec<i64>>,
    cols: HashMap<String, Vec<i64>>,
    rows: HashMap<String, Vec<i64>>,
}

/// `relationallib`'s `Mapper`: each (example, node) pair numbered in the
/// order it is first met.
#[derive(Default)]
struct Mapper {
    local: HashMap<(i64, i64), i64>,
}

impl Mapper {
    fn insert(&mut self, node: (i64, i64)) -> (i64, bool) {
        let next = self.local.len() as i64;
        match self.local.get(&node) {
            Some(&at) => (at, false),
            None => {
                self.local.insert(node, next);
                (next, true)
            }
        }
    }
}

pub struct NeighborSampler {
    node_types: Vec<String>,
    edge_types: Vec<EdgeType>,
    rel_types: Vec<String>,
    colptr: HashMap<String, Vec<i64>>,
    row: HashMap<String, Vec<i64>>,
    has_time: bool,
    node_time: HashMap<String, Vec<i64>>,
    rng: Mt19937,
    tallies: Tallies,
}

/// The C++'s `std::upper_bound` over `row[first..last]` by node time: the
/// first position whose time is past `value`, by its binary search, which
/// on an unsorted range lands where that search lands.
fn upper_bound(
    row: &[i64],
    first: i64,
    last: i64,
    value: i64,
    time: &[i64],
) -> Result<i64, SampleError> {
    let mut first = first;
    let mut len = last.wrapping_sub(first);
    while len > 0 {
        let half = len / 2;
        // Positions are `int64_t` in the C++, so they wrap as there.
        let middle = first.wrapping_add(half);
        if value < time_of(row, time, middle)? {
            len = half;
        } else {
            first = middle.wrapping_add(1);
            len -= half + 1;
        }
    }
    Ok(first)
}

fn at(values: &[i64], index: i64, what: &str) -> Result<i64, SampleError> {
    usize::try_from(index)
        .ok()
        .and_then(|i| values.get(i).copied())
        .ok_or_else(|| runtime(format!("{what} has no position {index}")))
}

/// The time of the node at `row[position]`.
fn time_of(row: &[i64], time: &[i64], position: i64) -> Result<i64, SampleError> {
    at(time, at(row, position, "row")?, "node_time")
}

impl NeighborSampler {
    /// `NeighborSampler(node_types, edge_types, colptr, row, node_time,
    /// seed)`, with its checks of the dicts' keys. `colptr`, `row` and
    /// `node_time` are in their dicts' order.
    pub fn new(
        node_types: Vec<String>,
        edge_types: Vec<EdgeType>,
        colptr: Vec<(String, Vec<i64>)>,
        row: Vec<(String, Vec<i64>)>,
        node_time: Vec<(String, Vec<i64>)>,
        seed: &Seed,
    ) -> Result<Self, SampleError> {
        let rel_types: Vec<String> = edge_types.iter().map(rel_type).collect();
        check_keys(&colptr, "colptr", &rel_types)?;
        check_keys(&row, "row", &rel_types)?;
        check_keys(&node_time, "node_time", &node_types)?;
        let has_time = !node_time.is_empty();
        let colptr: HashMap<_, _> = colptr.into_iter().collect();
        let row: HashMap<_, _> = row.into_iter().collect();
        for rel in &rel_types {
            if !colptr.contains_key(rel) || !row.contains_key(rel) {
                return Err(SampleError::Key(rel.clone()));
            }
        }
        let node_time: HashMap<_, _> = node_time
            .into_iter()
            .filter(|(name, _)| node_types.contains(name))
            .collect();
        Ok(Self {
            node_types,
            edge_types,
            rel_types,
            colptr,
            row,
            has_time,
            node_time,
            rng: seed.generator(),
            tallies: Tallies::default(),
        })
    }

    /// The generator as the draws so far left it.
    pub fn generator(&self) -> &Mt19937 {
        &self.rng
    }

    /// Draw on from `generator`, as a sampler kept between calls would.
    pub fn resume(&mut self, generator: Mt19937) {
        self.rng = generator;
    }

    /// `seed(seed)`: the generator restarted.
    pub fn seed(&mut self, seed: &Seed) {
        self.rng = seed.generator();
    }

    /// `init_placeholders`: an entry for each type the tallies lack.
    fn init_tallies(&mut self) {
        let t = &mut self.tallies;
        for name in &self.node_types {
            t.nodes_per_hop
                .entry(name.clone())
                .or_insert_with(|| vec![0]);
            t.batch.entry(name.clone()).or_default();
            t.node_ids.entry(name.clone()).or_default();
        }
        for rel in &self.rel_types {
            t.edges_per_hop
                .entry(rel.clone())
                .or_insert_with(|| vec![0]);
            t.cols.entry(rel.clone()).or_default();
            t.rows.entry(rel.clone()).or_default();
        }
    }

    /// `sample(num_neighbors, sampling_time_offsets, entity_table, seed_node,
    /// seed_time)`. Each dict maps an edge type's key to one entry per hop;
    /// an edge type is in one of them.
    pub fn sample(
        &mut self,
        num_neighbors: &[(String, Vec<i64>)],
        time_offsets: &[(String, TimeRanges)],
        entity_table: &str,
        seed_node: &[i64],
        seed_time: &[i64],
    ) -> Result<Sampled, SampleError> {
        check_keys(num_neighbors, "num_neighbors", &self.rel_types)?;
        check_keys(time_offsets, "sampling_time_offsets", &self.rel_types)?;
        if !self.node_types.iter().any(|n| n == entity_table) {
            return Err(runtime(format!("Invalid entity table '{entity_table}'.")));
        }
        if self.has_time && !seed_time.is_empty() && seed_time.len() != seed_node.len() {
            return Err(runtime("'seed_node' and 'seed_time' size mismatch."));
        }
        let counts: HashMap<&str, &Vec<i64>> =
            num_neighbors.iter().map(|(k, v)| (k.as_str(), v)).collect();
        let ranges: HashMap<&str, &TimeRanges> =
            time_offsets.iter().map(|(k, v)| (k.as_str(), v)).collect();

        self.init_tallies();
        let mut sampled: HashMap<String, Vec<(i64, i64)>> = HashMap::new();
        let mut mappers: HashMap<String, Mapper> = HashMap::new();
        let mut slices: HashMap<String, (usize, usize)> = HashMap::new();
        for name in &self.node_types {
            sampled.entry(name.clone()).or_default();
            mappers.insert(name.clone(), Mapper::default());
            slices.insert(name.clone(), (0, 0));
        }
        let mut layers = 0usize;
        for (edge, rel) in self.edge_types.iter().zip(&self.rel_types) {
            self.tallies
                .edges_per_hop
                .entry(rel.clone())
                .or_insert_with(|| vec![0]);
            let count = counts.get(rel.as_str());
            let range = ranges.get(rel.as_str());
            if count.is_some() && range.is_some() {
                return Err(runtime(format!(
                    "'sampling_time_offsets' and 'num_neighbors' may not contain the same edge \
                     type. '{rel}' appears in both."
                )));
            }
            if let Some(count) = count {
                layers = layers.max(count.len());
            } else {
                let Some(range) = range else {
                    return Err(SampleError::Key(rel.clone()));
                };
                layers = layers.max(range.ranges.len());
                if !self.node_time.contains_key(&edge.0) {
                    return Err(runtime(format!(
                        "'sampling_time_offsets' contains '{rel}' but {} is not included in \
                         'node_time'.",
                        edge.0
                    )));
                }
                if range.width != 2 {
                    return Err(runtime(format!(
                        "'sampling_time_offsets' at '{rel}' does not have shape (N, 2)."
                    )));
                }
            }
        }

        // The seeds, each its own example.
        slices.insert(entity_table.to_string(), (0, seed_node.len()));
        {
            let nodes = sampled.entry(entity_table.to_string()).or_default();
            let mapper = mappers.entry(entity_table.to_string()).or_default();
            for (batch, &node) in seed_node.iter().enumerate() {
                nodes.push((batch as i64, node));
                mapper.insert((batch as i64, node));
            }
        }
        let seed_times: Vec<i64> = if !seed_time.is_empty() {
            (0..seed_node.len())
                .map(|i| {
                    seed_time
                        .get(i)
                        .copied()
                        .ok_or_else(|| runtime("seed_time is shorter than seed_node"))
                })
                .collect::<Result<_, _>>()?
        } else if let Some(times) = self.node_time.get(entity_table) {
            seed_node
                .iter()
                .map(|&node| at(times, node, "node_time"))
                .collect::<Result<_, _>>()?
        } else {
            Vec::new()
        };
        {
            let seeds = &sampled[entity_table];
            let t = &mut self.tallies;
            if let Some(first) = t
                .nodes_per_hop
                .get_mut(entity_table)
                .and_then(|v| v.first_mut())
            {
                *first = seeds.len() as i64;
            }
            let batch = t.batch.entry(entity_table.to_string()).or_default();
            batch.extend(seeds.iter().map(|n| n.0));
            let ids = t.node_ids.entry(entity_table.to_string()).or_default();
            ids.extend(seeds.iter().map(|n| n.1));
        }

        for layer in 0..layers {
            for index in 0..self.edge_types.len() {
                let (dst, _, src) = self.edge_types[index].clone();
                let rel = self.rel_types[index].clone();
                // `slice_dict.at(src)`: an edge type from a node type the
                // sampler wasn't given has no slice. The message is libc++'s,
                // as `relationallib` built on macOS raises it; libstdc++ says
                // `unordered_map::at` (GCC 14) or `_Map_base::at` (GCC 10).
                let (begin, end) = *slices.get(&src).ok_or_else(|| {
                    SampleError::Index("unordered_map::at: key not found".to_string())
                })?;
                self.tallies
                    .edges_per_hop
                    .entry(rel.clone())
                    .or_default()
                    .push(0);
                let hop = match counts.get(rel.as_str()) {
                    Some(count) => HopSpec::Count(at(count, layer as i64, "num_neighbors")?),
                    None => {
                        // Either the counts or the ranges have the edge type,
                        // which the first loop checked.
                        let rows = &ranges[rel.as_str()].ranges;
                        // `sampling_time_offsets[rel][ell]`: numpy's error for
                        // an array with fewer rows than there are hops.
                        let row = rows.get(layer).ok_or_else(|| {
                            SampleError::Index(format!(
                                "index {layer} is out of bounds for axis 0 with size {}",
                                rows.len()
                            ))
                        })?;
                        let (start, end) = row.clone().map_err(runtime)?;
                        HopSpec::Range(start, end)
                    }
                };
                let is_static = self.node_time.is_empty()
                    || !self.node_time.contains_key(&dst)
                    || seed_times.is_empty();
                for i in begin..end {
                    let source = sampled[&src][i];
                    let mut out = std::mem::take(sampled.entry(dst.clone()).or_default());
                    let mapper = mappers.entry(dst.clone()).or_default();
                    let done = if is_static {
                        let HopSpec::Count(count) = hop else {
                            // A time range only reaches here without times,
                            // where the C++ samples none.
                            sampled.insert(dst.clone(), out);
                            continue;
                        };
                        self.uniform_sample(&rel, source, i as i64, count, mapper, &mut out)
                    } else {
                        let seed_time = at(&seed_times, source.0, "seed_time")?;
                        match hop {
                            HopSpec::Range(start, end) => self.temporal_range_sample(
                                &rel, &dst, source, i as i64, start, end, seed_time, mapper,
                                &mut out,
                            ),
                            HopSpec::Count(count) => self.temporal_sample(
                                &rel, &dst, source, i as i64, count, seed_time, mapper, &mut out,
                            ),
                        }
                    };
                    sampled.insert(dst.clone(), out);
                    done?;
                }
            }
            for name in &self.node_types {
                let (_, second) = slices[name];
                let size = sampled[name].len();
                slices.insert(name.clone(), (second, size));
                let t = &mut self.tallies;
                t.nodes_per_hop
                    .entry(name.clone())
                    .or_default()
                    .push((size - second) as i64);
                let nodes = &sampled[name][second..size];
                let batch = t.batch.entry(name.clone()).or_default();
                batch.extend(nodes.iter().map(|n| n.0));
                let ids = t.node_ids.entry(name.clone()).or_default();
                ids.extend(nodes.iter().map(|n| n.1));
            }
        }

        let mut tallies = std::mem::take(&mut self.tallies);
        let mut out = Sampled::default();
        for name in &self.node_types {
            set(
                &mut out.node_id,
                name,
                tallies.node_ids.get(name).cloned().unwrap_or_default(),
            );
            set(
                &mut out.num_sampled_nodes,
                name,
                tallies.nodes_per_hop.get(name).cloned().unwrap_or_default(),
            );
            set(
                &mut out.batch,
                name,
                tallies.batch.get(name).cloned().unwrap_or_default(),
            );
        }
        for rel in &self.rel_types {
            let edges = tallies.edges_per_hop.remove(rel).unwrap_or_default();
            set(
                &mut out.num_sampled_edges,
                rel,
                edges.get(1..).map(<[i64]>::to_vec).unwrap_or_default(),
            );
            tallies.edges_per_hop.insert(rel.clone(), edges);
            set(
                &mut out.row,
                rel,
                tallies.rows.get(rel).cloned().unwrap_or_default(),
            );
            set(
                &mut out.col,
                rel,
                tallies.cols.get(rel).cloned().unwrap_or_default(),
            );
        }
        Ok(out)
    }

    /// `colptr[node]` and `colptr[node + 1]`. The C++ does its arithmetic on
    /// positions in `int64_t`, so it wraps here too: it only overflows for a
    /// neighborhood outside `row`, which [`at`] then refuses.
    fn neighborhood(&self, rel: &str, node: i64) -> Result<(i64, i64), SampleError> {
        let colptr = self
            .colptr
            .get(rel)
            .ok_or_else(|| SampleError::Key(rel.to_string()))?;
        Ok((
            at(colptr, node, "colptr")?,
            at(colptr, node.wrapping_add(1), "colptr")?,
        ))
    }

    fn uniform_sample(
        &mut self,
        rel: &str,
        source: (i64, i64),
        local: i64,
        count: i64,
        mapper: &mut Mapper,
        out: &mut Vec<(i64, i64)>,
    ) -> Result<(), SampleError> {
        if count == 0 {
            return Ok(());
        }
        let (start, end) = self.neighborhood(rel, source.1)?;
        if end.wrapping_sub(start) == 0 {
            return Ok(());
        }
        self.pick(rel, source, local, start, end, count, mapper, out)
    }

    #[allow(clippy::too_many_arguments)]
    fn temporal_sample(
        &mut self,
        rel: &str,
        dst: &str,
        source: (i64, i64),
        local: i64,
        count: i64,
        seed_time: i64,
        mapper: &mut Mapper,
        out: &mut Vec<(i64, i64)>,
    ) -> Result<(), SampleError> {
        if count == 0 {
            return Ok(());
        }
        let (mut start, end) = self.neighborhood(rel, source.1)?;
        if end.wrapping_sub(start) == 0 {
            return Ok(());
        }
        let row = &self.row[rel];
        let time = &self.node_time[dst];
        let end = upper_bound(row, start, end, seed_time, time)?;
        if end.wrapping_sub(start) > 1
            && time_of(row, time, start)? > time_of(row, time, end.wrapping_sub(1))?
        {
            return Err(runtime("Found invalid non-sorted temporal neighborhood"));
        }
        if count > 0 {
            // `std::max(begin, end - count)` in `int64_t`, which wraps for a
            // neighborhood no `row` holds; `at` refuses its positions then.
            start = start.max(end.wrapping_sub(count));
        }
        if end.wrapping_sub(start) == 0 {
            return Ok(());
        }
        self.pick(rel, source, local, start, end, count, mapper, out)
    }

    #[allow(clippy::too_many_arguments)]
    fn temporal_range_sample(
        &mut self,
        rel: &str,
        dst: &str,
        source: (i64, i64),
        local: i64,
        range_start: Option<i64>,
        range_end: i64,
        seed_time: i64,
        mapper: &mut Mapper,
        out: &mut Vec<(i64, i64)>,
    ) -> Result<(), SampleError> {
        let (mut start, end) = self.neighborhood(rel, source.1)?;
        if end.wrapping_sub(start) == 0 {
            return Ok(());
        }
        let row = &self.row[rel];
        let time = &self.node_time[dst];
        if let Some(range_start) = range_start {
            start = upper_bound(row, start, end, range_start.wrapping_add(seed_time), time)?;
        }
        let end = upper_bound(row, start, end, range_end.wrapping_add(seed_time), time)?;
        if end.wrapping_sub(start) > 1
            && time_of(row, time, start)? > time_of(row, time, end.wrapping_sub(1))?
        {
            return Err(runtime("Found invalid non-sorted temporal neighborhood"));
        }
        if end.wrapping_sub(start) == 0 {
            return Ok(());
        }
        self.pick(
            rel,
            source,
            local,
            start,
            end,
            end.wrapping_sub(start),
            mapper,
            out,
        )
    }

    /// `_sample`: the whole neighborhood `start..end`, or `count` of it
    /// drawn without replacement.
    #[allow(clippy::too_many_arguments)]
    fn pick(
        &mut self,
        rel: &str,
        source: (i64, i64),
        local: i64,
        start: i64,
        end: i64,
        count: i64,
        mapper: &mut Mapper,
        out: &mut Vec<(i64, i64)>,
    ) -> Result<(), SampleError> {
        let population = end.wrapping_sub(start);
        if count < 0 || count >= population {
            for edge in start..end {
                self.add_edge(rel, edge, source, local, mapper, out)?;
            }
            return Ok(());
        }
        let mut picked = HashSet::new();
        for i in population - count..population {
            let mut offset = i64::from(self.rng.next_u32()) % (i + 1);
            if !picked.insert(offset) {
                offset = i;
                picked.insert(i);
            }
            self.add_edge(rel, start.wrapping_add(offset), source, local, mapper, out)?;
        }
        Ok(())
    }

    fn add_edge(
        &mut self,
        rel: &str,
        edge: i64,
        source: (i64, i64),
        local: i64,
        mapper: &mut Mapper,
        out: &mut Vec<(i64, i64)>,
    ) -> Result<(), SampleError> {
        let node = (source.0, at(&self.row[rel], edge, "row")?);
        let (index, new) = mapper.insert(node);
        if new {
            out.push(node);
        }
        let t = &mut self.tallies;
        if let Some(last) = t
            .edges_per_hop
            .entry(rel.to_string())
            .or_default()
            .last_mut()
        {
            *last += 1;
        }
        t.rows.entry(rel.to_string()).or_default().push(index);
        t.cols.entry(rel.to_string()).or_default().push(local);
        Ok(())
    }
}

/// `dict[key] = value`: a key keeps the place it was first given.
fn set(dict: &mut Vec<(String, Vec<i64>)>, key: &str, value: Vec<i64>) {
    match dict.iter_mut().find(|(k, _)| k == key) {
        Some(entry) => entry.1 = value,
        None => dict.push((key.to_string(), value)),
    }
}

/// `validate_arr_dict`'s check of a dict's keys.
fn check_keys<T>(dict: &[(String, T)], name: &str, keys: &[String]) -> Result<(), SampleError> {
    for (key, _) in dict {
        if !keys.contains(key) {
            return Err(runtime(format!("Invalid type '{key}' found in {name}.")));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn edge(dst: &str, src: &str) -> EdgeType {
        (dst.to_string(), "to".to_string(), src.to_string())
    }

    fn get<'a>(dict: &'a [(String, Vec<i64>)], key: &str) -> &'a [i64] {
        &dict.iter().find(|(k, _)| k == key).unwrap().1
    }

    /// `test_local_sampler.py::test_relationallib_sampler_static`.
    #[test]
    fn a_static_graph_samples_as_relationallib() {
        let mut sampler = NeighborSampler::new(
            vec!["A".into(), "B".into()],
            vec![edge("A", "B"), edge("B", "A")],
            vec![
                ("A__to__B".into(), vec![0, 2, 4]),
                ("B__to__A".into(), vec![0, 1, 2, 3]),
            ],
            vec![
                ("A__to__B".into(), vec![0, 2, 0, 2]),
                ("B__to__A".into(), vec![0, 1, 1]),
            ],
            Vec::new(),
            &Seed::Word(123),
        )
        .unwrap();
        let out = sampler
            .sample(
                &[
                    ("A__to__B".into(), vec![1, -1]),
                    ("B__to__A".into(), vec![1, 1]),
                ],
                &[],
                "A",
                &[0, 1, 2],
                &[],
            )
            .unwrap();
        assert_eq!(get(&out.row, "A__to__B"), [0, 3, 4, 5, 6, 2]);
        assert_eq!(get(&out.col, "A__to__B"), [0, 0, 1, 1, 2, 2]);
        assert_eq!(get(&out.node_id, "A"), [0, 1, 2, 2, 0, 2, 0]);
        assert_eq!(get(&out.batch, "A"), [0, 1, 2, 0, 1, 1, 2]);
        assert_eq!(get(&out.num_sampled_nodes, "B"), [0, 3, 0]);
        assert_eq!(get(&out.num_sampled_edges, "A__to__B"), [0, 6]);
    }

    #[test]
    fn an_unsorted_neighborhood_is_refused_and_its_tallies_stay() {
        let mut sampler = NeighborSampler::new(
            vec!["u".into(), "o".into()],
            vec![edge("o", "u")],
            vec![("o__to__u".into(), vec![0, 3])],
            vec![("o__to__u".into(), vec![2, 0, 1])],
            vec![("o".into(), vec![1, 2, 3]), ("u".into(), vec![9])],
            &Seed::Word(5),
        )
        .unwrap();
        let counts = [("o__to__u".to_string(), vec![-1])];
        let error = sampler.sample(&counts, &[], "u", &[0], &[]).unwrap_err();
        assert_eq!(
            error,
            SampleError::Runtime("Found invalid non-sorted temporal neighborhood".into())
        );
        // The C++ leaves the seed it had tallied, and the next call adds to it.
        let out = sampler.sample(&counts, &[], "u", &[0], &[]);
        assert_eq!(out.unwrap_err().to_string(), error.to_string());
        assert_eq!(sampler.tallies.batch["u"], [0, 0]);
    }

    #[test]
    fn a_missing_edge_type_is_a_key_error() {
        let error = NeighborSampler::new(
            vec!["a".into()],
            vec![edge("a", "a")],
            Vec::new(),
            Vec::new(),
            Vec::new(),
            &Seed::Word(0),
        )
        .err()
        .unwrap();
        assert_eq!(error.to_string(), "'a__to__a'");
    }

    #[test]
    fn a_key_error_shows_its_key_as_python_repr_does() {
        // Python escapes every character `str.isprintable` rejects.
        for (name, shown) in [
            ("t\u{85}", "'a__t\\x85__a'"),
            ("t\u{a0}", "'a__t\\xa0__a'"),
            ("t\u{2028}", "'a__t\\u2028__a'"),
            ("it's", "\"a__it's__a\""),
        ] {
            let error = NeighborSampler::new(
                vec!["a".into()],
                vec![("a".into(), name.into(), "a".into())],
                Vec::new(),
                Vec::new(),
                Vec::new(),
                &Seed::Word(0),
            )
            .err()
            .unwrap();
            assert_eq!(error.to_string(), shown);
        }
    }
}
