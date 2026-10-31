// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! In-memory graph ownership and reads. The shared materialization policy
//! composes these local operations without a provider or host callbacks.
use super::*;
use sdm_engine::frames::{Snapshot, TemporalWindow};
use sdm_engine::random::Mt19937;
use sdm_request::payload::{SampledColumn, SampledLink, SampledTable};
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};

struct Owner {
    description: SourceDescription,
    snapshot: Snapshot,
    graph: Graph,
    types: sdm_schema::assemble::HostTypes,
    neighbors: Mutex<Mt19937>,
    busy: AtomicBool,
}

/// One materializer's immutable graph and continuing neighbor stream.
/// Clones share ownership, never a global registry or a schema-wide cache.
#[derive(Clone)]
pub struct FrameGraph(Arc<Owner>);
impl fmt::Debug for FrameGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FrameGraph")
            .field("tables", &self.0.description.tables.len())
            .finish_non_exhaustive()
    }
}
struct Lease(FrameGraph);
impl Drop for Lease {
    fn drop(&mut self) {
        self.0.0.busy.store(false, Ordering::Release);
    }
}
impl FrameGraph {
    fn lease(&self) -> Result<Lease, CoreError> {
        if self
            .0
            .busy
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            return Err(CoreError::protocol(
                Sentence::RunBusy,
                "frame materializer is already in use",
            ));
        }
        Ok(Lease(self.clone()))
    }
    pub(super) fn graph(&self) -> Graph {
        self.0.graph.clone()
    }
    pub(super) fn time_bounds(&self, table: &str) -> Result<(Option<i64>, Option<i64>), CoreError> {
        self.0.snapshot.time_bounds(table)
    }
    pub(super) fn entities(
        &self,
        table: &Table,
        columns: &[String],
        limit: usize,
        seed: &Seed,
    ) -> Result<RecordBatch, CoreError> {
        self.0.snapshot.entities(
            &table.name,
            columns,
            limit,
            seed.to_u128()
                .ok_or_else(|| internal("frame entity seed is too large"))?,
        )
    }
    pub(super) fn lookup(
        &self,
        table: &Table,
        keys: &ArrayRef,
        columns: &[String],
    ) -> Result<RecordBatch, CoreError> {
        self.0.snapshot.lookup(&table.name, keys, columns)
    }
    pub(super) fn fetch(
        &self,
        fetch: &Fetch<'_>,
        keys: &ArrayRef,
        starts: &[i64],
        ends: &[i64],
    ) -> Result<Fetched, CoreError> {
        let windows = starts
            .iter()
            .zip(ends)
            .map(|(&lo, &hi)| TemporalWindow {
                lower: Some(i128::from(lo.div_euclid(1_000_000_000))),
                upper: i128::from(hi.div_euclid(1_000_000_000)),
            })
            .collect::<Vec<_>>();
        let selected = self.0.snapshot.temporal_rows(
            &fetch.edge,
            keys,
            &windows,
            &fetch.table.ordered(&fetch.columns),
        )?;
        let time = fetch
            .table
            .time_column
            .as_deref()
            .ok_or_else(|| internal("temporal frame has no time column"))?;
        Ok(Fetched {
            times: times_ns(named(&selected.rows, time)?, time)?,
            rows: selected.rows,
            batch: selected.batch,
        })
    }
    pub(super) fn walk(
        &self,
        graph: &Graph,
        entity: &Table,
        keys: &ArrayRef,
        anchors: &[i64],
        hops: &[options::Hop],
        excluded: &[(String, String)],
        seed: Option<u128>,
    ) -> Result<crate::subgraph::Walked, CoreError> {
        let rows = self.0.snapshot.entity_rows(&entity.name, keys)?;
        let mut counts = Vec::new();
        for hop in hops {
            match hop {
                options::Hop::Keep(n) => {
                    counts.push(i64::try_from(*n).map_err(|_| not_yet("frame_neighbor_count"))?)
                }
                options::Hop::Stop => break,
                options::Hop::Negative(_) => counts.push(-1),
                options::Hop::Expression(_) => return Err(not_yet("frame_neighbor_expression")),
            }
        }
        let mut rng = self
            .0
            .neighbors
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if let Some(seed) = seed {
            *rng = Mt19937::seeded(
                u32::try_from(seed).map_err(|_| internal("frame neighbor seed exceeds u32"))?,
            );
        }
        let walked = self.0.snapshot.walk_resolved(
            graph,
            &entity.name,
            &rows,
            anchors,
            &counts,
            &mut rng,
        )?;
        drop(rng);
        let mut tables = Vec::new();
        for table in walked.tables {
            let meaning = graph
                .table(&table.name)
                .ok_or_else(|| internal("sampled frame lost its schema"))?;
            let mut names = crate::subgraph::sample_columns(graph, meaning, excluded);
            if let Some(key) = &meaning.primary_key {
                if !names.contains(key) {
                    names.push(key.clone());
                }
            }
            let mut columns = Vec::new();
            for name in &names {
                let array = named(&table.rows, name)?.clone();
                let text = meaning.column(name).is_some_and(|c| c.stype == Stype::Text);
                columns.push((
                    name.clone(),
                    if text {
                        SampledColumn::Words(words(&array)?)
                    } else {
                        SampledColumn::Arrow(array)
                    },
                ));
            }
            tables.push(SampledTable {
                name: table.name,
                primary_key: meaning.primary_key.clone(),
                batch: table.batch,
                stypes: graph
                    .features(meaning)
                    .into_iter()
                    .filter(|(name, _)| names.contains(name))
                    .collect(),
                columns,
            });
        }
        let links = walked
            .links
            .into_iter()
            .map(|link| SampledLink {
                src: link.edge.src,
                fkey: link.edge.fkey,
                dst: link.edge.dst,
                pairs: link.pairs,
            })
            .collect();
        Ok(crate::subgraph::Walked { tables, links })
    }
}

pub(crate) fn prepare(
    asker: &Asker,
    input: sdm_contract::frames::PrepareFrameGraph,
    mut attachments: crate::run::Attachments,
    facts: &HostFacts,
) -> Result<OperationResult, CoreError> {
    let held = crate::attached::take_json(&mut attachments, input.schema)?;
    let names: Vec<String> = crate::attached::take_json(&mut attachments, input.names)?;
    if names.len() != input.tables.len() {
        return Err(internal("frame table names and attachments differ"));
    }
    let read = sdm_schema::held::read(held, &BTreeMap::new(), true)?;
    let description = read.description;
    let native = sdm_engine::graph::NativeGraph::new(&description)?;
    let graph = native.topology.clone();
    let times = Times::for_frames(crate::describe::text_reader(facts)?);
    let mut tables = BTreeMap::new();
    for (name, id) in names.into_iter().zip(input.tables) {
        let Some(Attachment::Arrow(data)) = attachments.remove(&id) else {
            return Err(internal("frame table is not Arrow"));
        };
        let table = graph
            .table(&name)
            .ok_or_else(|| internal("frame table is absent from schema"))?;
        let rows = times.read(asker, table, one_batch(&data)?)?;
        if tables.insert(name, rows).is_some() {
            return Err(internal("duplicate frame table name"));
        }
    }
    let identities = native
        .primary_keys
        .iter()
        .filter(|(_, keys)| keys.len() > 1)
        .map(|(name, keys)| {
            let rows = tables
                .get(name)
                .ok_or_else(|| internal("compound identity table is absent"))?;
            Ok((
                name.clone(),
                sdm_engine::frame_identity::encode(rows, keys, &BTreeMap::new())?,
            ))
        })
        .collect::<Result<BTreeMap<_, _>, CoreError>>()?;
    // Compound identities are real payload reference columns, not features.
    // Keep the original components and append the generated identity once.
    for (name, identity) in &identities {
        let rows = tables
            .get_mut(name)
            .ok_or_else(|| internal("compound table disappeared"))?;
        let key = graph
            .table(name)
            .and_then(|table| table.primary_key.as_ref())
            .ok_or_else(|| internal("compound key disappeared"))?;
        let mut fields = rows.schema().fields().to_vec();
        fields.push(Arc::new(arrow_schema::Field::new(
            key,
            identity.values.data_type().clone(),
            false,
        )));
        let mut columns = rows.columns().to_vec();
        columns.push(identity.values.clone());
        *rows = RecordBatch::try_new(Arc::new(arrow_schema::Schema::new(fields)), columns)
            .map_err(|_| internal("compound identity does not align with frame"))?;
    }
    let snapshot = Snapshot::new_frame(&native, tables, None, identities)?;
    let owner = FrameGraph(Arc::new(Owner {
        description,
        snapshot,
        graph,
        types: read.types,
        neighbors: Mutex::new(Mt19937::seeded(123)),
        busy: AtomicBool::new(false),
    }));
    let graph = asker.attach(Attachment::FrameGraph(owner))?;
    Ok(OperationResult::PreparedFrameGraph(
        sdm_contract::frames::PreparedFrameGraph { graph },
    ))
}

pub(crate) async fn materialize(
    asker: Asker,
    request: sdm_contract::frames::MaterializeFrames,
    mut attachments: crate::run::Attachments,
    facts: HostFacts,
    zones: Arc<dyn sdm_engine::time::Zones>,
) -> Result<OperationResult, CoreError> {
    let Some(Attachment::FrameGraph(owner)) = attachments.remove(&request.graph) else {
        return Err(internal("frame graph owner is absent"));
    };
    let _lease = owner.lease()?;
    let call: sdm_contract::frames::FrameCall =
        crate::attached::take_json(&mut attachments, request.input)?;
    let Some(Attachment::Arrow(indices)) = attachments.remove(&request.indices) else {
        return Err(internal("frame indices are absent"));
    };
    let values = match request.values {
        Some(id) => {
            let Some(Attachment::Bytes(bytes)) = attachments.remove(&id) else {
                return Err(internal("frame options are absent"));
            };
            Some(
                sdm_wire::json::read_options(&bytes)
                    .map_err(|_| internal("frame options are invalid JSON"))?,
            )
        }
        None => None,
    };
    let inputs = Inputs {
        description: owner.0.description.clone(),
        query: call.query,
        indices: Some(indices),
        candidates: call.candidates,
        anchor_time: call.anchor_time,
        seed: call.seed,
        unseeded: call.unseeded,
        anchor_entity: call.anchor_entity,
        values,
        options: call.options,
        most_parts: call.most_parts,
        most_entities: call.most_entities,
        most_calls: call.most_calls,
        facts,
        zones,
        memo: None,
        source: "in_memory".into(),
        pending: None,
        types: owner.0.types.clone(),
        frames: Some(owner),
        #[cfg(feature = "test-ops")]
        labels_only: false,
    };
    run(asker, inputs).await
}

fn words(array: &ArrayRef) -> Result<Vec<Vec<String>>, CoreError> {
    use arrow_array::cast::AsArray;
    match array.data_type() {
        arrow_schema::DataType::Utf8 => Ok(array
            .as_string::<i32>()
            .iter()
            .map(sdm_engine::words::normalize)
            .collect()),
        arrow_schema::DataType::LargeUtf8 => Ok(array
            .as_string::<i64>()
            .iter()
            .map(sdm_engine::words::normalize)
            .collect()),
        _ => Err(internal("a frame text feature is not UTF-8 text")),
    }
}
