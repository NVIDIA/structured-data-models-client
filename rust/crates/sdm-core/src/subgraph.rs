// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The subgraph each example's request carries, walked as the reference's
//! SQL sampler walks it (`SQLSampler._sample_subgraph`, then
//! `Sampler.sample_subgraph`), hop by hop from the examples' entity rows.
//!
//! At each hop every table that gained rows at the hop before is visited,
//! in the order tables were first reached: its foreign keys first, each key
//! no row of the table it refers to holds yet for that example looked up
//! (`_by_pkey`, one `lookup`), then the tables referring to it, their
//! latest rows up to each example's anchor (`_by_fkey`, one `neighbors` for
//! each, asked together in one step). A table visited reads only the rows
//! the hop before reached: a table reached by two paths may gain rows while
//! the hop runs, which the next hop reads (`main`'s eb6c68eb). A column
//! linking two tables is one column read, and each of its links is looked
//! up. A row is told apart by its key and its example as values (decision
//! E41(d), H22), where the reference's
//! `Mapper` numbers them; each is kept once, numbered in the order first
//! reached. What each answer adds, and the links it makes, are the
//! reference's, in its order: so are the order of the tables the payload
//! sends and of their rows, and which link gives each row its foreign key.
//!
//! Then each table's rows: a table with a key whose rows repeat a key past
//! 1.05 times its distinct keys keeps each key's first row's values for
//! every row of that key, as the reference stores such a table's distinct
//! rows once; its columns cut to its features (and the entity table's key);
//! and each text feature as its words ([`sdm_engine::words`]).
//!
//! A null where the walk tells rows apart is no link (register P1B-13,
//! approved by the owner): a null foreign key the walk looks up adds no
//! row and is sent as null, and a row of a table read by foreign key whose
//! key is null is no other row's, reached anew. The reference takes such a
//! null for the last key it looked up, or numbers every such row as one,
//! and then numbers the rows it adds out of step with the rows it keeps,
//! so its request links rows across examples.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use arrow_array::{ArrayRef, RecordBatch, UInt32Array};
use arrow_schema::{Field, Schema};
use sdm_contract::data::NullOrder;
use sdm_contract::ops::Stype;
use sdm_contract::types::ColumnType;
use sdm_contract::{CoreError, CoreWarning, Detail, Scalar};
use sdm_engine::graph::{Graph, Table};
use sdm_engine::keys::{self, Key};
use sdm_engine::sample::{self, SEED_TIME};
use sdm_request::options::Hop;
use sdm_request::payload::{SampledColumn, SampledLink, SampledTable};

use crate::describe::{ask, ask_all_vec};
use crate::materialize::{Times, batches, key_set, named, one_batch};
use crate::run::{Asker, internal};

/// One table the walk reached: its rows, in the order numbered, as the
/// frames they came in, and the example each was reached for.
struct Visit {
    name: String,
    frames: Vec<RecordBatch>,
    batches: Vec<Vec<usize>>,
    /// Each row's number, by its key and example (the reference's
    /// `Mapper`), for a table with a key.
    nodes: BTreeMap<(Key, usize), usize>,
    /// The rows reached at each hop, the entity's first.
    counts: Vec<usize>,
    /// The frames visited so far.
    offset: usize,
}

impl Visit {
    fn new(name: &str, hops: usize) -> Visit {
        Visit {
            name: name.to_owned(),
            frames: Vec::new(),
            batches: Vec::new(),
            nodes: BTreeMap::new(),
            counts: vec![0; hops + 1],
            offset: 0,
        }
    }

    fn rows(&self) -> usize {
        self.counts.iter().sum()
    }
}

/// The rows of one foreign key, as the reference keeps an edge type: for a
/// table referring to the one visited, its rows and the rows they refer to
/// (`(src, fkey, dst)`); for a foreign key looked up, the rows found and the
/// rows referring to them (its reverse, `reversed`).
struct Edge {
    src: String,
    fkey: String,
    dst: String,
    reversed: bool,
    rows: Vec<usize>,
    cols: Vec<usize>,
}

/// What the walk sampled, as the payload takes it.
pub(crate) struct Walked {
    pub tables: Vec<SampledTable>,
    pub links: Vec<SampledLink>,
}

pub(crate) struct Walk<'a> {
    pub asker: &'a Asker,
    pub graph: &'a Graph,
    pub times: &'a Times,
    /// Each example's anchor time, nanoseconds.
    pub anchors: &'a [i64],
    pub nulls: NullOrder,
    pub entity: &'a Table,
    /// The hops walked, each with the neighbors it keeps of each row: the
    /// run mode's two by default, or as the sampling options ask
    /// (`num_hops`, `num_neighbors`, [`options::neighbors`]); the walk ends
    /// at a hop of size zero.
    ///
    /// [`options::neighbors`]: sdm_request::options::neighbors
    pub neighbors: &'a [Hop],
    /// The features left out, by table: a static target's columns, which
    /// the reference's sampler neither reads nor sends
    /// (`get_exclude_cols_dict`, `exclude_cols_dict`).
    pub excluded: &'a [(String, String)],
}

/// The features of `table` the walk reads and sends: those `excluded`
/// does not name for it.
pub(crate) fn features(
    graph: &Graph,
    table: &Table,
    excluded: &[(String, String)],
) -> Vec<(String, Stype)> {
    graph
        .features(table)
        .into_iter()
        .filter(|(n, _)| !excluded.iter().any(|(t, c)| *t == table.name && c == n))
        .collect()
}

/// The columns the walk reads of `table`: its features (but those
/// `excluded`), foreign keys and key, in the table's order
/// (`sample_columns_dict`).
pub(crate) fn sample_columns(
    graph: &Graph,
    table: &Table,
    excluded: &[(String, String)],
) -> Vec<String> {
    let mut names: BTreeSet<String> = features(graph, table, excluded)
        .into_iter()
        .map(|(n, _)| n)
        .collect();
    for (fkey, _) in graph.foreign_keys(&table.name) {
        names.insert(fkey.to_owned());
    }
    if let Some(key) = &table.primary_key {
        names.insert(key.clone());
    }
    table.ordered(&names)
}

fn table_named<'g>(graph: &'g Graph, name: &str) -> Result<&'g Table, CoreError> {
    graph
        .table(name)
        .ok_or_else(|| internal("a sampled table is not in the graph"))
}

/// `batch`'s columns `names`, in order.
fn project(batch: &RecordBatch, names: &[String]) -> Result<RecordBatch, CoreError> {
    let fields: Vec<Field> = names
        .iter()
        .map(|n| {
            let column = named(batch, n)?;
            Ok(Field::new(n, column.data_type().clone(), true))
        })
        .collect::<Result<_, CoreError>>()?;
    let columns = names
        .iter()
        .map(|n| named(batch, n).map(Arc::clone))
        .collect::<Result<Vec<_>, _>>()?;
    RecordBatch::try_new(Arc::new(Schema::new(fields)), columns)
        .map_err(|_| internal("an answer's columns do not make rows"))
}

/// Rows `rows` of `batch`, in that order.
fn take(batch: &RecordBatch, rows: &[usize]) -> Result<RecordBatch, CoreError> {
    let indices = UInt32Array::from_iter_values(rows.iter().map(|r| *r as u32));
    arrow_select::take::take_record_batch(batch, &indices)
        .map_err(|_| internal("rows could not be taken"))
}

/// `frames`, one after another, column by column.
fn concat(frames: &[RecordBatch]) -> Result<RecordBatch, CoreError> {
    let first = frames
        .first()
        .ok_or_else(|| internal("a table was visited with no rows"))?;
    if frames.len() == 1 {
        return Ok(first.clone());
    }
    let columns = (0..first.num_columns())
        .map(|i| {
            let arrays: Vec<&dyn arrow_array::Array> =
                frames.iter().map(|f| f.column(i).as_ref()).collect();
            arrow_select::concat::concat(&arrays)
                .map_err(|_| internal("a table's answers do not make one column"))
        })
        .collect::<Result<Vec<ArrayRef>, _>>()?;
    RecordBatch::try_new(first.schema(), columns)
        .map_err(|_| internal("a table's answers do not make rows"))
}

impl Walk<'_> {
    /// The subgraph of the examples whose entity rows `entity_rows` holds,
    /// one row per example, in example order, with the entity table's
    /// [`sample_columns`].
    pub(crate) async fn run(&self, entity_rows: RecordBatch) -> Result<Walked, CoreError> {
        let examples = entity_rows.num_rows();
        let mut visits = vec![Visit::new(&self.entity.name, self.neighbors.len())];
        let mut edges: Vec<Edge> = Vec::new();
        {
            let entity = &mut visits[0];
            let key = self
                .entity
                .primary_key
                .as_deref()
                .ok_or_else(|| internal("the entity table has no key"))?;
            for (row, value) in key_values(self.entity, key, named(&entity_rows, key)?)?
                .into_iter()
                .enumerate()
            {
                let value = value.ok_or_else(|| internal("an entity's key is null"))?;
                entity.nodes.insert((value, row), row);
            }
            entity.frames.push(entity_rows);
            entity.batches.push((0..examples).collect());
            entity.counts[0] = examples;
        }
        for (hop, &neighbors) in self.neighbors.iter().enumerate() {
            // A hop of size zero ends the walk (`neighbors == 0`).
            let Hop::Keep(neighbors) = neighbors else {
                break;
            };
            // The frames each table held when this hop began. A table
            // reached by two paths gains frames for the next hop while this
            // one runs, through whichever of its neighbors is visited first,
            // and those wait for the next hop, which counts them (`main`'s
            // eb6c68eb, `hop_end_dict`).
            let visiting: Vec<(String, usize)> = visits
                .iter()
                .map(|v| (v.name.clone(), v.frames.len()))
                .collect();
            for (name, end) in visiting {
                self.visit(&mut visits, &mut edges, &name, end, hop, neighbors)
                    .await?;
            }
        }
        self.walked(visits, edges)
    }

    async fn visit(
        &self,
        visits: &mut Vec<Visit>,
        edges: &mut Vec<Edge>,
        name: &str,
        end: usize,
        hop: usize,
        neighbors: u64,
    ) -> Result<(), CoreError> {
        let at = position(visits, name)?;
        if visits[at].counts[hop] == 0 {
            return Ok(());
        }
        let table = table_named(self.graph, name)?;
        let visit = &mut visits[at];
        // The rows the hop before reached: the frames from the last visit's
        // up to those the table held when this hop began.
        let frame = concat(&visit.frames[visit.offset..end])?;
        let batch: Vec<usize> = visit.batches[visit.offset..end].concat();
        // Each row's number is its place among the table's rows, in the
        // order they were reached (the reference's `Mapper` numbers a row
        // reached anew next, and a row of a table without a key, or whose
        // key is null, is always reached anew).
        let start: usize = visit.batches[..visit.offset].iter().map(Vec::len).sum();
        let index: Vec<usize> = (start..start + batch.len()).collect();
        visit.offset = end;

        // The foreign keys: each key not yet reached for its example looked
        // up in the table it refers to.
        for (fkey, dst) in self.graph.foreign_keys(name) {
            let dst_table = table_named(self.graph, dst)?;
            // Read as the key it refers to is read: by the numbers it writes
            // where that key is a column of numbers handed over as their
            // text, as text where that key is other text, and else by its
            // own column; so a spelling never tells a row apart from the
            // row it refers to.
            let fkeys = named(&frame, fkey)?;
            let values = match dst_table.primary_key.as_deref() {
                Some(dst_key) if exact_text(dst_table, dst_key) => keys::read_numbers(fkeys)?,
                Some(dst_key)
                    if dst_table
                        .column(dst_key)
                        .is_some_and(|c| c.received == ColumnType::Utf8) =>
                {
                    keys::read(fkeys)?
                }
                _ => key_values(table, fkey, fkeys)?,
            };
            let found = |visits: &[Visit], key: &Option<Key>, example: usize| {
                let at = visits.iter().position(|v| v.name == dst)?;
                visits[at].nodes.get(&(key.clone()?, example)).copied()
            };
            let mut row: Vec<Option<usize>> = values
                .iter()
                .zip(&batch)
                .map(|(key, &example)| found(visits, key, example))
                .collect();
            let missing: Vec<usize> = (0..row.len()).filter(|&i| row[i].is_none()).collect();
            if !missing.is_empty() {
                // The distinct (key, example) pairs, in the order first
                // reached, and the distinct keys among them, likewise; a
                // null refers to nothing, so it is looked up for no row and
                // adds none (register P1B-13), though a lookup is still
                // made, of no key when every one missing is null, as the
                // reference makes it.
                let mut pairs: Vec<(Key, usize)> = Vec::new();
                let mut paired: BTreeSet<(Key, usize)> = BTreeSet::new();
                let mut distinct: BTreeMap<Key, usize> = BTreeMap::new();
                let mut first_row: Vec<usize> = Vec::new();
                for &i in &missing {
                    let Some(key) = values[i].clone() else {
                        continue;
                    };
                    if !distinct.contains_key(&key) {
                        distinct.insert(key.clone(), first_row.len());
                        first_row.push(i);
                    }
                    if paired.insert((key.clone(), batch[i])) {
                        pairs.push((key, batch[i]));
                    }
                }
                let seeds = take_array(named(&frame, fkey)?, &first_row)?;
                let columns = sample_columns(self.graph, dst_table, self.excluded);
                let answer = self.times.read(
                    self.asker,
                    dst_table,
                    one_batch(
                        &ask(
                            self.asker,
                            sample::by_pkey(
                                dst_table,
                                columns.clone(),
                                key_set(self.asker, sample::seeds(seeds, vec![])?)?,
                                false,
                            )?,
                        )
                        .await?,
                    )?,
                )?;
                if answer.num_rows() > 0 {
                    let positions = batches(&answer, distinct.len())?;
                    let mut answered: Vec<Option<usize>> = vec![None; distinct.len()];
                    for (answer_row, &position) in positions.iter().enumerate() {
                        answered[position] = Some(answer_row);
                    }
                    let mut taken = Vec::new();
                    let mut examples = Vec::new();
                    for (key, example) in &pairs {
                        let position = *distinct
                            .get(key)
                            .ok_or_else(|| internal("a key looked up was not asked for"))?;
                        if let Some(answer_row) = answered[position] {
                            taken.push(answer_row);
                            examples.push(*example);
                        }
                    }
                    let added = take(&project(&answer, &columns)?, &taken)?;
                    let dst_key = dst_table
                        .primary_key
                        .as_deref()
                        .ok_or_else(|| internal("a table referred to has no key"))?;
                    let added_keys = key_values(dst_table, dst_key, named(&added, dst_key)?)?;
                    let dst_at = reach(visits, dst);
                    let dst_visit = &mut visits[dst_at];
                    let known = dst_visit.rows();
                    for (added_row, (key, &example)) in
                        added_keys.into_iter().zip(&examples).enumerate()
                    {
                        let key = key.ok_or_else(|| internal("a key looked up is null"))?;
                        if dst_visit
                            .nodes
                            .insert((key, example), known + added_row)
                            .is_some()
                        {
                            return Err(internal("a row looked up was already reached"));
                        }
                    }
                    dst_visit.counts[hop + 1] += examples.len();
                    dst_visit.frames.push(added);
                    dst_visit.batches.push(examples);
                    for &i in &missing {
                        row[i] = found(visits, &values[i], batch[i]);
                    }
                }
            }
            let mut rows = Vec::new();
            let mut cols = Vec::new();
            for (i, found) in row.iter().enumerate() {
                if let Some(found) = found {
                    rows.push(*found);
                    cols.push(index[i]);
                }
            }
            if !cols.is_empty() {
                edge(edges, name, fkey, dst, true).extend(rows, cols);
            }
        }

        // The tables referring to this one: their latest rows up to each
        // row's anchor, asked together.
        let referring: Vec<(String, String)> = self
            .graph
            .referring(name)
            .into_iter()
            .map(|(src, fkey)| (src.to_owned(), fkey.to_owned()))
            .collect();
        if referring.is_empty() {
            return Ok(());
        }
        let Some(pkey_column) = table.primary_key.as_deref() else {
            return Err(internal("a table referred to has no key"));
        };
        // A missing anchor travels as a null bound (`sample::seeds`): the
        // source reaches no row of a timed table for it.
        let times: Vec<i64> = batch.iter().map(|&b| self.anchors[b]).collect();
        let mut requests = Vec::with_capacity(referring.len());
        for (src, fkey) in &referring {
            let src_table = table_named(self.graph, src)?;
            let seeds = sample::seeds(
                Arc::clone(named(&frame, pkey_column)?),
                if src_table.time_column.is_some() {
                    vec![(SEED_TIME, times.clone())]
                } else {
                    vec![]
                },
            )?;
            requests.push(sample::by_fkey(
                src_table,
                fkey,
                sample_columns(self.graph, src_table, self.excluded),
                key_set(self.asker, seeds)?,
                neighbors,
                self.nulls,
            )?);
        }
        // A source comparing a window's bounds with its stored text reads
        // one stored value of a table before its first windowed read, as
        // the reference does table by table (`Times::windowed`): each read
        // is then asked after its own, in order.
        let answers = if self.asker.compares_stored_times() {
            let mut answers = Vec::with_capacity(requests.len());
            for ((src, _), request) in referring.iter().zip(requests) {
                let src_table = table_named(self.graph, src)?;
                let request = self.times.windowed(self.asker, src_table, request).await?;
                answers.push(ask(self.asker, request).await?);
            }
            answers
        } else {
            ask_all_vec(self.asker, requests).await?
        };
        for ((src, fkey), answer) in referring.iter().zip(answers) {
            let src_table = table_named(self.graph, src)?;
            let answer = self
                .times
                .read(self.asker, src_table, one_batch(&answer)?)?;
            if answer.num_rows() == 0 {
                continue;
            }
            let positions = batches(&answer, batch.len())?;
            let cols: Vec<usize> = positions.iter().map(|&p| index[p]).collect();
            let examples: Vec<usize> = positions.iter().map(|&p| batch[p]).collect();
            let columns = sample_columns(self.graph, src_table, self.excluded);
            let fetched = project(&answer, &columns)?;
            let src_at = reach(visits, src);
            let src_visit = &mut visits[src_at];
            let known = src_visit.rows();
            let (rows, kept) = match &src_table.primary_key {
                Some(key) => {
                    let values = key_values(src_table, key, named(&fetched, key)?)?;
                    let mut rows = Vec::with_capacity(values.len());
                    let mut kept = Vec::new();
                    for (i, (value, &example)) in values.into_iter().zip(&examples).enumerate() {
                        let next = known + kept.len();
                        // A null key is no row's but its own: the row is
                        // reached anew, never one already reached (register
                        // P1B-13).
                        let number = match value {
                            None => next,
                            Some(value) => *src_visit.nodes.entry((value, example)).or_insert(next),
                        };
                        if number == next {
                            kept.push(i);
                        }
                        rows.push(number);
                    }
                    (rows, kept)
                }
                None => (
                    (known..known + examples.len()).collect(),
                    (0..examples.len()).collect::<Vec<_>>(),
                ),
            };
            let kept_examples: Vec<usize> = kept.iter().map(|&i| examples[i]).collect();
            src_visit.counts[hop + 1] += kept.len();
            src_visit.frames.push(take(&fetched, &kept)?);
            src_visit.batches.push(kept_examples);
            edge(edges, src, fkey, name, false).extend(rows, cols);
        }
        Ok(())
    }

    /// The tables and links the payload takes, from what the walk reached.
    fn walked(&self, visits: Vec<Visit>, edges: Vec<Edge>) -> Result<Walked, CoreError> {
        let mut tables = Vec::new();
        // Each table whose rows reached were cut, each kept row's new place
        // by its place among the rows reached.
        let mut cut: BTreeMap<String, Vec<Option<usize>>> = BTreeMap::new();
        for visit in visits {
            let mut batch: Vec<usize> = visit.batches.concat();
            if batch.is_empty() {
                continue;
            }
            let table = table_named(self.graph, &visit.name)?;
            let mut frame = concat(&visit.frames)?;
            if let Some(key) = &table.primary_key {
                if let Some(kept) = null_keys_beside_values(&frame, key)? {
                    let left_out = kept.iter().filter(|k| !**k).count();
                    let rows: Vec<usize> = (0..kept.len()).filter(|&i| kept[i]).collect();
                    frame = take(&frame, &rows)?;
                    batch = rows.iter().map(|&i| batch[i]).collect();
                    let mut place = 0;
                    cut.insert(
                        table.name.clone(),
                        kept.iter()
                            .map(|&k| {
                                k.then(|| {
                                    place += 1;
                                    place - 1
                                })
                            })
                            .collect(),
                    );
                    self.asker.warn(CoreWarning {
                        code: NULL_KEYS_LEFT_OUT.to_owned(),
                        message: "rows reached with no value for their table's key were left out"
                            .to_owned(),
                        fields: [
                            ("table".to_owned(), Detail::text(&table.name)),
                            ("column".to_owned(), Detail::text(key)),
                            (
                                "rows".to_owned(),
                                Detail::Scalar(Scalar::UInt64(left_out as u64)),
                            ),
                        ]
                        .into_iter()
                        .collect(),
                    });
                }
                frame = first_of_each_key(frame, table, key)?;
            }
            let entity = table.name == self.entity.name;
            let features = features(self.graph, table, self.excluded);
            let mut sent: BTreeSet<String> = features.iter().map(|(n, _)| n.clone()).collect();
            if entity && let Some(key) = &table.primary_key {
                sent.insert(key.clone());
            }
            let columns = frame
                .schema()
                .fields()
                .iter()
                .filter(|f| sent.contains(f.name()))
                .map(|f| {
                    let array = Arc::clone(named(&frame, f.name())?);
                    let text = features
                        .iter()
                        .any(|(n, s)| n == f.name() && *s == Stype::Text);
                    let column = if text {
                        SampledColumn::Words(words(&array)?)
                    } else {
                        SampledColumn::Arrow(array)
                    };
                    Ok((f.name().clone(), column))
                })
                .collect::<Result<_, CoreError>>()?;
            tables.push(SampledTable {
                name: table.name.clone(),
                primary_key: if entity {
                    table.primary_key.clone()
                } else {
                    None
                },
                batch,
                columns,
                stypes: features,
            });
        }
        // A pair naming a row left out goes with it; the rest are renumbered.
        let renumber = |table: &str, row: usize| -> Option<usize> {
            match cut.get(table) {
                Some(places) => places.get(row).copied().flatten(),
                None => Some(row),
            }
        };
        let links = edges
            .into_iter()
            .filter(|e| !e.rows.is_empty())
            .map(|e| {
                let pairs: Vec<(usize, usize)> = if e.reversed {
                    e.cols.into_iter().zip(e.rows).collect()
                } else {
                    e.rows.into_iter().zip(e.cols).collect()
                };
                SampledLink {
                    pairs: pairs
                        .into_iter()
                        .filter_map(|(s, d)| Some((renumber(&e.src, s)?, renumber(&e.dst, d)?)))
                        .collect(),
                    src: e.src,
                    fkey: e.fkey,
                    dst: e.dst,
                }
            })
            .collect();
        Ok(Walked { tables, links })
    }
}

impl Edge {
    fn extend(&mut self, rows: Vec<usize>, cols: Vec<usize>) {
        self.rows.extend(rows);
        self.cols.extend(cols);
    }
}

/// The edge of `src.fkey` referring to `dst`, read in the direction
/// `reversed` says, made the first time it is asked for.
fn edge<'e>(
    edges: &'e mut Vec<Edge>,
    src: &str,
    fkey: &str,
    dst: &str,
    reversed: bool,
) -> &'e mut Edge {
    let at = match edges
        .iter()
        .position(|e| e.src == src && e.fkey == fkey && e.dst == dst && e.reversed == reversed)
    {
        Some(at) => at,
        None => {
            edges.push(Edge {
                src: src.to_owned(),
                fkey: fkey.to_owned(),
                dst: dst.to_owned(),
                reversed,
                rows: Vec::new(),
                cols: Vec::new(),
            });
            edges.len() - 1
        }
    };
    &mut edges[at]
}

/// Whether `column` of `table` is a column of numbers the host provider
/// handed over as their exact text (`sdm_contract::data::
/// EXACT_NUMBER_MARKER`), described as that text.
pub(crate) fn exact_text(table: &Table, column: &str) -> bool {
    table
        .column(column)
        .is_some_and(|c| c.exact_number.is_some() && c.received == ColumnType::Utf8)
}

/// The keys `array`, column `column` of `table`, holds: by the numbers it
/// writes for a column of numbers handed over as their exact text
/// ([`exact_text`], [`keys::read_numbers`]), so `1.0` and `1.00` are one
/// key, as the source compares them (the reference keys a caller-keyed
/// column by its double, one key, but a key the catalog declares by its
/// text, two: register P2G-4 and P2G-5); as [`keys::read`] reads any
/// other.
pub(crate) fn key_values(
    table: &Table,
    column: &str,
    array: &ArrayRef,
) -> Result<Vec<Option<Key>>, CoreError> {
    if exact_text(table, column) {
        keys::read_numbers(array)
    } else {
        keys::read(array)
    }
}

fn position(visits: &[Visit], name: &str) -> Result<usize, CoreError> {
    visits
        .iter()
        .position(|v| v.name == name)
        .ok_or_else(|| internal("a visited table was not reached"))
}

/// The visit of `name`, made, last in the walk's order, the first time a
/// table is reached.
fn reach(visits: &mut Vec<Visit>, name: &str) -> usize {
    match visits.iter().position(|v| v.name == name) {
        Some(at) => at,
        None => {
            let hops = visits[0].counts.len() - 1;
            visits.push(Visit::new(name, hops));
            visits.len() - 1
        }
    }
}

fn take_array(array: &ArrayRef, rows: &[usize]) -> Result<ArrayRef, CoreError> {
    let indices = UInt32Array::from_iter_values(rows.iter().map(|r| *r as u32));
    arrow_select::take::take(array.as_ref(), &indices, None)
        .map_err(|_| internal("rows could not be taken"))
}

/// The warning that rows a walk reached were left out for a null key
/// beside keys that are not null (register P1B-40, the owner's option (b)
/// of 2026-10-13).
pub const NULL_KEYS_LEFT_OUT: &str = "NULL_KEYS_LEFT_OUT";

/// Which rows of a keyed table's rows reached to keep, where they hold a
/// null text key beside a key that is not null: the reference sorts every
/// keyed table's keys reached (`np.unique` in `_sample_subgraph`), which
/// fails comparing a null with a text value (an unnamed error its client
/// reports as `INVALID_REQUEST`), so no request of it exists to match; the
/// core leaves the rows with a null key out and says so (P1B-40, approved
/// by the owner). `None` where nothing is left out: keys reached that are
/// all null sort, as one null alone does (measured on the reference at
/// b413e9ad), and a null integer, float or decimal key sorts (a decimal
/// since `main`'s eb6c68eb, which reads it as an integer or a float: the
/// reference keeps such a row as it keeps its integer twin's, T2.0c).
fn null_keys_beside_values(frame: &RecordBatch, key: &str) -> Result<Option<Vec<bool>>, CoreError> {
    let column = named(frame, key)?;
    let unsortable = matches!(
        column.data_type(),
        arrow_schema::DataType::Utf8
            | arrow_schema::DataType::LargeUtf8
            | arrow_schema::DataType::Utf8View
    );
    let nulls = column.null_count();
    if !unsortable || nulls == 0 || nulls == column.len() {
        return Ok(None);
    }
    Ok(Some(
        (0..column.len()).map(|i| column.is_valid(i)).collect(),
    ))
}

/// `frame`, its rows repeating a key past 1.05 times its distinct keys
/// each given the values of the first row of its key, as the reference
/// stores such a table's distinct rows once and reads each row through
/// them (`inverse_dict`).
fn first_of_each_key(
    frame: RecordBatch,
    table: &Table,
    key: &str,
) -> Result<RecordBatch, CoreError> {
    let values = key_values(table, key, named(&frame, key)?)?;
    let mut first: BTreeMap<Key, usize> = BTreeMap::new();
    let mut nulls = 0;
    // A null key is no other row's: each such row is its own.
    let rows: Vec<usize> = values
        .into_iter()
        .enumerate()
        .map(|(row, value)| match value {
            Some(value) => *first.entry(value).or_insert(row),
            None => {
                nulls += 1;
                row
            }
        })
        .collect();
    if (rows.len() as f64) > 1.05 * (first.len() + nulls) as f64 {
        take(&frame, &rows)
    } else {
        Ok(frame)
    }
}

/// A text feature's values as the reference's words.
fn words(array: &ArrayRef) -> Result<Vec<Vec<String>>, CoreError> {
    use arrow_array::Array;
    use arrow_array::cast::AsArray;
    if array.data_type() != &arrow_schema::DataType::Utf8 {
        // The schema's consistency check holds a text feature to text.
        return Err(internal("a text feature's values are not text"));
    }
    let text = array.as_string::<i32>();
    Ok((0..text.len())
        .map(|i| sdm_engine::words::normalize((!text.is_null(i)).then(|| text.value(i))))
        .collect())
}

#[cfg(test)]
mod null_key_tests {
    use std::sync::Arc;

    use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringArray};
    use arrow_schema::{Field, Schema};

    use super::null_keys_beside_values;

    fn frame(keys: ArrayRef) -> RecordBatch {
        let field = Field::new("k", keys.data_type().clone(), true);
        RecordBatch::try_new(Arc::new(Schema::new(vec![field])), vec![keys]).unwrap()
    }

    #[test]
    fn only_a_null_beside_a_value_is_left_out() {
        // Measured on the reference at b413e9ad: one null alone and two
        // nulls sort; a null beside text, or beside a decimal, does not.
        let text = |values: Vec<Option<&str>>| frame(Arc::new(StringArray::from(values)));
        assert_eq!(
            null_keys_beside_values(&text(vec![None]), "k").unwrap(),
            None
        );
        assert_eq!(
            null_keys_beside_values(&text(vec![None, None]), "k").unwrap(),
            None
        );
        assert_eq!(
            null_keys_beside_values(&text(vec![Some("a"), Some("b")]), "k").unwrap(),
            None
        );
        assert_eq!(
            null_keys_beside_values(&text(vec![None, Some("a"), None]), "k").unwrap(),
            Some(vec![false, true, false])
        );
        let decimal = arrow_array::Decimal128Array::from(vec![Some(1), None])
            .with_precision_and_scale(18, 0)
            .unwrap();
        // A decimal key with a null sorts since eb6c68eb, as an integer
        // key with a null does, and is read.
        assert_eq!(
            null_keys_beside_values(&frame(Arc::new(decimal)), "k").unwrap(),
            None
        );
        let ints = frame(Arc::new(Int64Array::from(vec![Some(1), None])));
        assert_eq!(null_keys_beside_values(&ints, "k").unwrap(), None);
    }
}
