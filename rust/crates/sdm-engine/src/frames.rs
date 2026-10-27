// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The typed numeric/time frame consumer: detached rows, sanitized key maps
//! and DF-1 adjacency, with RS-3 draws owned by its caller's checkpoint.
//! No provider, SQL renderer or host callback participates in this walk.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{
    TimestampMicrosecondType, TimestampMillisecondType, TimestampNanosecondType,
    TimestampSecondType,
};
use arrow_array::{
    Array, ArrayRef, Float32Array, Float64Array, Int64Array, RecordBatch, TimestampNanosecondArray,
    UInt64Array,
};
use arrow_schema::{DataType, Field, Schema, TimeUnit};
use sdm_contract::data::{BATCH_COLUMN, MATCHES_COLUMN};
use sdm_contract::{CoreError, Detail, Scalar, Sentence};

use crate::graph::{Edge, Graph};
use crate::keys::{self, Key};
use crate::random::Mt19937;
use crate::time::Instant;

const NANOS_PER_SECOND: i64 = 1_000_000_000;

fn invalid(message: &str) -> CoreError {
    CoreError::protocol(Sentence::InvalidMessage, message)
}

fn named<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a ArrayRef, CoreError> {
    batch
        .column_by_name(name)
        .ok_or_else(|| invalid("a frame lacks a described column"))
}

/// Reuse the shared numeric identifier reader, applying native missingness.
/// Boolean features do not enter this identifying-key path.
fn read_keys(array: &ArrayRef) -> Result<Vec<Option<Key>>, CoreError> {
    let mut values = keys::read(array)?;
    if let Some(array) = array.as_any().downcast_ref::<Float32Array>() {
        for (row, value) in values.iter_mut().enumerate() {
            if !array.is_null(row) && array.value(row).is_nan() {
                *value = None;
            }
        }
    }
    if let Some(array) = array.as_any().downcast_ref::<Float64Array>() {
        for (row, value) in values.iter_mut().enumerate() {
            if !array.is_null(row) && array.value(row).is_nan() {
                *value = None;
            }
        }
    }
    Ok(values)
}

fn take(batch: &RecordBatch, rows: &[usize], columns: &[String]) -> Result<RecordBatch, CoreError> {
    let indices = UInt64Array::from(rows.iter().map(|&row| row as u64).collect::<Vec<_>>());
    let arrays = columns
        .iter()
        .map(|name| {
            arrow_select::take::take(named(batch, name)?.as_ref(), &indices, None)
                .map_err(|_| invalid("frame row selection failed"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let fields = columns
        .iter()
        .zip(&arrays)
        .map(|(name, array)| Field::new(name, array.data_type().clone(), true));
    RecordBatch::try_new(Arc::new(Schema::new(fields.collect::<Vec<_>>())), arrays)
        .map_err(|_| invalid("selected frame columns do not align"))
}

struct Table {
    rows: RecordBatch,
    valid: Vec<usize>,
    index: BTreeMap<Key, usize>,
    /// Original instants in nanoseconds; cuts use their floor seconds.
    time: Option<Vec<Option<i64>>>,
}

struct Adjacency {
    edge: Edge,
    reversed: bool,
    groups: Vec<Vec<usize>>,
}

/// Pure derived indexes of one immutable, admitted snapshot.
pub struct Snapshot {
    tables: BTreeMap<String, Table>,
    adjacency: Vec<Adjacency>,
    pub dropped: Vec<DroppedRows>,
}

pub struct DroppedRows {
    pub table: String,
    pub input: usize,
    pub null_keys: usize,
    pub duplicate_keys: usize,
    pub null_times: usize,
}

/// Normalize the already typed input, not a native-language date parser.
/// Graph sanitization reads every timestamp before applying row masks.
fn timestamps(rows: RecordBatch) -> Result<RecordBatch, CoreError> {
    let mut fields = Vec::with_capacity(rows.num_columns());
    let mut columns = Vec::with_capacity(rows.num_columns());
    for (field, array) in rows.schema().fields().iter().zip(rows.columns()) {
        let DataType::Timestamp(unit, _) = field.data_type() else {
            fields.push(field.as_ref().clone());
            columns.push(Arc::clone(array));
            continue;
        };
        if field.data_type() == &DataType::Timestamp(TimeUnit::Nanosecond, None) {
            fields.push(field.as_ref().clone());
            columns.push(Arc::clone(array));
            continue;
        }
        let raw: Vec<Option<i64>> = match unit {
            TimeUnit::Second => array.as_primitive::<TimestampSecondType>().iter().collect(),
            TimeUnit::Millisecond => array
                .as_primitive::<TimestampMillisecondType>()
                .iter()
                .collect(),
            TimeUnit::Microsecond => array
                .as_primitive::<TimestampMicrosecondType>()
                .iter()
                .collect(),
            TimeUnit::Nanosecond => array
                .as_primitive::<TimestampNanosecondType>()
                .iter()
                .collect(),
        };
        let nanos = raw
            .into_iter()
            .map(|value| {
                value
                    .map(|value| {
                        Instant::from_count(value, *unit)
                            .to_count(TimeUnit::Nanosecond, field.name())
                            // The frozen frame route refuses these as
                            // INVALID_REQUEST with native date wording,
                            // rather than the shared provider range code.
                            // Its lossless wide constructor stays admitted;
                            // that materializer family remains unfinished.
                            .map_err(|_| crate::not_yet("frame_wide_timestamps"))
                    })
                    .transpose()
            })
            .collect::<Result<Vec<_>, _>>()?;
        fields.push(
            field
                .as_ref()
                .clone()
                .with_data_type(DataType::Timestamp(TimeUnit::Nanosecond, None)),
        );
        columns.push(Arc::new(TimestampNanosecondArray::from(nanos)) as ArrayRef);
    }
    RecordBatch::try_new(Arc::new(Schema::new(fields)), columns)
        .map_err(|_| invalid("normalized frame timestamp columns do not align"))
}

pub struct WalkedTable {
    pub name: String,
    pub rows: RecordBatch,
    pub batch: Vec<usize>,
}

pub struct WalkedLink {
    pub edge: Edge,
    pub pairs: Vec<(usize, usize)>,
}

pub struct Walked {
    pub tables: Vec<WalkedTable>,
    pub links: Vec<WalkedLink>,
}

impl Snapshot {
    pub fn new(graph: &Graph, mut input: BTreeMap<String, RecordBatch>) -> Result<Self, CoreError> {
        let mut tables = BTreeMap::new();
        let mut dropped = Vec::new();
        let timed = graph.tables.iter().any(|table| table.time_column.is_some());
        for table in &graph.tables {
            let rows = input
                .remove(&table.name)
                .ok_or_else(|| invalid("a frame source lacks one table"))?;
            // Timestamp features on an otherwise untimed graph retain the
            // separate no-time/DT-3 acceptance obligation of the full phase.
            if !timed
                && rows
                    .schema()
                    .fields()
                    .iter()
                    .any(|field| matches!(field.data_type(), DataType::Timestamp(_, _)))
            {
                return Err(crate::not_yet("frame_untimed_timestamp_features"));
            }
            if rows.schema().fields().iter().any(|f| {
                !matches!(
                    f.data_type(),
                    DataType::Boolean
                        | DataType::Int8
                        | DataType::Int16
                        | DataType::Int32
                        | DataType::Int64
                        | DataType::UInt8
                        | DataType::UInt16
                        | DataType::UInt32
                        | DataType::UInt64
                        | DataType::Float32
                        | DataType::Float64
                        | DataType::Timestamp(_, _)
                )
            }) {
                return Err(invalid(
                    "the numeric frame consumer received another native family",
                ));
            }
            let rows = timestamps(rows)?;
            let time = table
                .time_column
                .as_ref()
                .map(|column| {
                    let array = named(&rows, column)?;
                    if !matches!(
                        array.data_type(),
                        DataType::Timestamp(TimeUnit::Nanosecond, None)
                    ) {
                        return Err(crate::not_yet("frame_time_type"));
                    }
                    Ok(array
                        .as_primitive::<TimestampNanosecondType>()
                        .iter()
                        .collect::<Vec<_>>())
                })
                .transpose()?;
            let mut index = BTreeMap::new();
            let mut seen = BTreeSet::new();
            let mut null_keys = 0;
            let mut duplicate_keys = 0;
            let mut null_times = 0;
            let mut has_time = |row: usize| {
                if time.as_ref().is_some_and(|values| values[row].is_none()) {
                    null_times += 1;
                    false
                } else {
                    true
                }
            };
            let valid: Vec<usize> = match &table.primary_key {
                None => (0..rows.num_rows()).filter(|&row| has_time(row)).collect(),
                Some(key) => read_keys(named(&rows, key)?)?
                    .into_iter()
                    .enumerate()
                    .filter_map(|(row, key)| {
                        let Some(key) = key else {
                            null_keys += 1;
                            return None;
                        };
                        // Duplicate detection precedes time removal, so a
                        // later duplicate cannot replace a missing-time row.
                        if !seen.insert(key.clone()) {
                            duplicate_keys += 1;
                            None
                        } else if !has_time(row) {
                            None
                        } else {
                            index.insert(key, row);
                            Some(row)
                        }
                    })
                    .collect(),
            };
            if table.primary_key.is_some() && valid.is_empty() {
                return Err(CoreError::new(
                    Sentence::FrameNoValidRows,
                    "a frame has no valid primary-key rows",
                )
                .with_field("table", Detail::text(&table.name))
                .with_field(
                    "requirements",
                    Detail::text(if time.is_some() {
                        "primary key and time column pairs"
                    } else {
                        "primary keys"
                    }),
                ));
            }
            if null_keys + duplicate_keys + null_times > 0 {
                dropped.push(DroppedRows {
                    table: table.name.clone(),
                    input: rows.num_rows(),
                    null_keys,
                    duplicate_keys,
                    null_times,
                });
            }
            tables.insert(
                table.name.clone(),
                Table {
                    rows,
                    valid,
                    index,
                    time,
                },
            );
        }
        if !input.is_empty() {
            return Err(invalid("a snapshot binds an undescribed frame"));
        }
        let mut adjacency = Vec::new();
        for edge in &graph.edges {
            let src = &tables[&edge.src];
            let dst = &tables[&edge.dst];
            let keys = read_keys(named(&src.rows, &edge.fkey)?)?;
            let mut forward = vec![Vec::new(); dst.rows.num_rows()];
            let mut reverse = vec![Vec::new(); src.rows.num_rows()];
            // DF-1: stable grouping by destination, preserving source row
            // order within each neighborhood, independently of Torch.
            for &row in &src.valid {
                if let Some(key) = &keys[row]
                    && let Some(&at) = dst.index.get(key)
                {
                    forward[at].push(row);
                    reverse[row].push(at);
                }
            }
            if let Some(times) = &src.time {
                for group in &mut forward {
                    // DF-1: equal seconds keep source-row order, including
                    // timestamps on opposite sides of a subsecond anchor.
                    group.sort_by_key(|&row| {
                        times[row].map(|time| time.div_euclid(NANOS_PER_SECOND))
                    });
                }
            }
            adjacency.push(Adjacency {
                edge: edge.clone(),
                reversed: false,
                groups: forward,
            });
            adjacency.push(Adjacency {
                edge: edge.clone(),
                reversed: true,
                groups: reverse,
            });
        }
        Ok(Self {
            tables,
            adjacency,
            dropped,
        })
    }

    /// The graph's valid nanosecond values, before second sampler cuts.
    pub fn time_bounds(&self, table: &str) -> Result<(Option<i64>, Option<i64>), CoreError> {
        let table = self
            .tables
            .get(table)
            .ok_or_else(|| invalid("unknown timed frame"))?;
        let time = table
            .time
            .as_ref()
            .ok_or_else(|| invalid("an untimed frame was bounded in time"))?;
        let values = table.valid.iter().filter_map(|&row| time[row]);
        Ok((values.clone().min(), values.max()))
    }

    /// Entity sampling uses the sanitized primary-key map, before RS-1.
    pub fn entities(
        &self,
        table: &str,
        columns: &[String],
        limit: usize,
        seed: u128,
    ) -> Result<RecordBatch, CoreError> {
        let table = self
            .tables
            .get(table)
            .ok_or_else(|| invalid("unknown entity table"))?;
        let rows = if table.valid.len() > limit {
            let seed = u32::try_from(seed).map_err(|_| {
                CoreError::new(
                    Sentence::SeedPastRange,
                    "entity seed exceeds the legacy generator range",
                )
                .with_field("request", Detail::text("sample_entities"))
                .with_field("field", Detail::text("seed"))
                .with_field("max_seed", Detail::Scalar(Scalar::UInt64(u32::MAX as u64)))
            })?;
            Mt19937::seeded(seed)
                .permutation(table.valid.len())
                .into_iter()
                .take(limit)
                .map(|row| table.valid[row])
                .collect()
        } else {
            table.valid.clone()
        };
        take(&table.rows, &rows, columns)
    }

    pub fn entity_rows(&self, table: &str, keys: &ArrayRef) -> Result<Vec<usize>, CoreError> {
        let named_table = self
            .tables
            .get(table)
            .ok_or_else(|| invalid("unknown entity table"))?;
        let wanted = read_keys(keys)?;
        let found = wanted
            .iter()
            .map(|key| {
                key.as_ref()
                    .and_then(|key| named_table.index.get(key))
                    .copied()
            })
            .collect::<Vec<_>>();
        let missing = found.iter().filter(|row| row.is_none()).count();
        if missing > 0 {
            return Err(CoreError::new(
                Sentence::FrameTaskReferences,
                "task entity references are unresolved after graph sanitization",
            )
            .with_field("table", Detail::text(table))
            .with_field("rows", Detail::Scalar(Scalar::UInt64(missing as u64))));
        }
        Ok(found.into_iter().flatten().collect())
    }

    /// The existing materializer's example lookup shape, without a provider.
    pub fn lookup(
        &self,
        table: &str,
        keys: &ArrayRef,
        columns: &[String],
    ) -> Result<RecordBatch, CoreError> {
        let rows = self.entity_rows(table, keys)?;
        let selected = take(&self.tables[table].rows, &rows, columns)?;
        let mut fields = selected
            .schema()
            .fields()
            .iter()
            .map(|f| f.as_ref().clone())
            .collect::<Vec<_>>();
        let mut arrays = selected.columns().to_vec();
        fields.push(Field::new(BATCH_COLUMN, DataType::Int64, false));
        fields.push(Field::new(MATCHES_COLUMN, DataType::Int64, false));
        arrays.push(Arc::new(Int64Array::from(
            (0..rows.len()).map(|n| n as i64).collect::<Vec<_>>(),
        )));
        arrays.push(Arc::new(Int64Array::from(vec![1; rows.len()])));
        RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays)
            .map_err(|_| invalid("frame lookup columns do not align"))
    }

    /// The local sampler's node mapper and edge order, per example and hop.
    pub fn walk(
        &self,
        graph: &Graph,
        entity: &str,
        keys: &ArrayRef,
        anchors: &[i64],
        hops: &[i64],
        rng: &mut Mt19937,
    ) -> Result<Walked, CoreError> {
        let entity_rows = self.entity_rows(entity, keys)?;
        if anchors.len() != entity_rows.len() {
            return Err(invalid("frame anchors do not align with entity rows"));
        }
        let mut nodes: BTreeMap<String, Vec<(usize, usize)>> = graph
            .tables
            .iter()
            .map(|t| (t.name.clone(), Vec::new()))
            .collect();
        let mut maps: BTreeMap<String, BTreeMap<(usize, usize), usize>> = graph
            .tables
            .iter()
            .map(|t| (t.name.clone(), BTreeMap::new()))
            .collect();
        for (batch, row) in entity_rows.into_iter().enumerate() {
            let at = nodes[entity].len();
            nodes.get_mut(entity).unwrap().push((batch, row));
            maps.get_mut(entity).unwrap().insert((batch, row), at);
        }
        let mut slices = graph
            .tables
            .iter()
            .map(|t| (t.name.clone(), (0, nodes[&t.name].len())))
            .collect::<BTreeMap<_, _>>();
        let mut pairs = vec![Vec::new(); self.adjacency.len()];
        for &count in hops {
            for (which, adj) in self.adjacency.iter().enumerate() {
                let (from, to) = if adj.reversed {
                    (&adj.edge.src, &adj.edge.dst)
                } else {
                    (&adj.edge.dst, &adj.edge.src)
                };
                let (begin, end) = slices[from];
                for at in begin..end {
                    let (batch, row) = nodes[from][at];
                    let group = &adj.groups[row];
                    let selected = if let Some(times) = &self.tables[to].time {
                        let anchor = anchors[batch].div_euclid(NANOS_PER_SECOND);
                        let end = group.partition_point(|&row| {
                            times[row]
                                .is_some_and(|time| time.div_euclid(NANOS_PER_SECOND) <= anchor)
                        });
                        let begin = if count < 0 {
                            0
                        } else {
                            end.saturating_sub(count as usize)
                        };
                        (begin..end).collect()
                    } else if count < 0 {
                        (0..group.len()).collect()
                    } else {
                        rng.floyd(group.len(), count as usize)
                    };
                    for position in selected {
                        let key = (batch, group[position]);
                        let next = if let Some(&known) = maps[to].get(&key) {
                            known
                        } else {
                            let next = nodes[to].len();
                            nodes.get_mut(to).unwrap().push(key);
                            maps.get_mut(to).unwrap().insert(key, next);
                            next
                        };
                        pairs[which].push(if adj.reversed { (at, next) } else { (next, at) });
                    }
                }
            }
            for table in &graph.tables {
                let slice = slices.get_mut(&table.name).unwrap();
                *slice = (slice.1, nodes[&table.name].len());
            }
        }
        let mut output = Vec::new();
        for table in &graph.tables {
            let nodes = &nodes[&table.name];
            if nodes.is_empty() {
                continue;
            }
            let rows = nodes.iter().map(|&(_, row)| row).collect::<Vec<_>>();
            let columns = self.tables[&table.name]
                .rows
                .schema()
                .fields()
                .iter()
                .map(|f| f.name().clone())
                .collect::<Vec<_>>();
            output.push(WalkedTable {
                name: table.name.clone(),
                rows: take(&self.tables[&table.name].rows, &rows, &columns)?,
                batch: nodes.iter().map(|&(batch, _)| batch).collect(),
            });
        }
        let links = self
            .adjacency
            .iter()
            .zip(pairs)
            .filter(|(_, pairs)| !pairs.is_empty())
            .map(|(adj, pairs)| WalkedLink {
                edge: adj.edge.clone(),
                pairs,
            })
            .collect();
        Ok(Walked {
            tables: output,
            links,
        })
    }
}
