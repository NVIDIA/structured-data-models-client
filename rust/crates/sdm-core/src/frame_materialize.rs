// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Materialize's fixed frame consumer and its owned checkpoint.
//! Metadata is JSON, rows are Arrow, and every exported id is call-local.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use arrow_array::{ArrayRef, RecordBatch, UInt64Array};
use sdm_contract::data::{AttachmentId, Relation};
use sdm_contract::frames::{
    FRAME_CHECKPOINT_VERSION, FrameCheckpoint, FrameCheckpointManifest, FrameCheckpointMetadata,
    MaterializeFrames,
};
use sdm_contract::ops::{HeldSchema, SourceDescription};
use sdm_contract::{CoreError, Detail, HeldState, Scalar, Sentence};
use sdm_engine::frames::Snapshot;
use sdm_engine::graph::{Graph, Table};
use sdm_engine::random::Mt19937;
use sdm_request::options::Hop;
use sdm_request::payload::{SampledColumn, SampledLink, SampledTable};

use crate::attached::take_json;
use crate::run::{ArrowData, Asker, Attachment, Attachments, internal};

pub(crate) type Context = Arc<Mutex<FrameRun>>;

pub(crate) fn lock(context: &Context) -> MutexGuard<'_, FrameRun> {
    context.lock().unwrap_or_else(PoisonError::into_inner)
}

fn invalid(message: &str) -> CoreError {
    CoreError::protocol(Sentence::InvalidMessage, message)
}

/// A value-only metadata record and independently owned detached Arrow tables.
#[derive(Clone)]
pub(crate) struct OwnedCheckpoint {
    pub metadata: FrameCheckpointMetadata,
    pub tables: Vec<(Relation, ArrowData)>,
}

pub(crate) struct FrameRun {
    tables: Vec<(Relation, ArrowData)>,
    metadata: Option<FrameCheckpointMetadata>,
    snapshot: Option<Snapshot>,
    neighbors: Mt19937,
    pub entropy: Option<u64>,
}

fn bound_tables(
    bindings: Vec<sdm_contract::frames::FrameInput>,
    attachments: &mut Attachments,
) -> Result<Vec<(Relation, ArrowData)>, CoreError> {
    bindings
        .into_iter()
        .map(|binding| {
            let Some(Attachment::Arrow(data)) = attachments.remove(&binding.rows) else {
                return Err(invalid("a frame binding lacks its Arrow rows"));
            };
            Ok((binding.source, data))
        })
        .collect()
}

/// Decode the operation-specific manifest; no graph is constructed yet.
pub(crate) fn take(
    manifest: AttachmentId,
    checkpoint: Option<FrameCheckpoint>,
    attachments: &mut Attachments,
) -> Result<Context, CoreError> {
    let input: MaterializeFrames = take_json(attachments, manifest)?;
    let (tables, metadata, neighbors) = match checkpoint {
        None => (
            bound_tables(input.tables, attachments)?,
            None,
            Mt19937::seeded(123),
        ),
        Some(checkpoint) => {
            if checkpoint.version != FRAME_CHECKPOINT_VERSION {
                return Err(CoreError::new(
                    Sentence::SamplerStateVersion,
                    "unknown frame checkpoint version",
                )
                .with_field(
                    "field",
                    Detail::text("sampler_state.frame_checkpoint.version"),
                )
                .with_field(
                    "version",
                    Detail::Scalar(Scalar::UInt32(checkpoint.version)),
                )
                .dropping(HeldState::SamplerState));
            }
            if !input.tables.is_empty() {
                return Err(invalid("a resumed frame run must not reread caller frames"));
            }
            let manifest: FrameCheckpointManifest = take_json(attachments, checkpoint.manifest)?;
            let metadata: FrameCheckpointMetadata = take_json(attachments, manifest.metadata)?;
            let neighbors = Mt19937::from_checkpoint(&metadata.neighbors).ok_or_else(|| {
                invalid("invalid frame neighbor checkpoint").dropping(HeldState::SamplerState)
            })?;
            (
                bound_tables(manifest.tables, attachments)?,
                Some(metadata),
                neighbors,
            )
        }
    };
    Ok(Arc::new(Mutex::new(FrameRun {
        tables,
        metadata,
        snapshot: None,
        neighbors,
        entropy: input.entropy,
    })))
}

/// The immutable graph facts of a resumed snapshot, or the caller's schema
/// for the first graph construction. Neither branch fetches from a provider.
pub(crate) fn description(
    context: &Context,
    held: HeldSchema,
) -> Result<(SourceDescription, sdm_schema::assemble::HostTypes), CoreError> {
    let read = sdm_schema::held::read(held, &BTreeMap::new(), true)?;
    let description = lock(context)
        .metadata
        .as_ref()
        .map(|state| state.description.clone())
        .unwrap_or_else(|| read.description.clone());
    Ok((description, read.types))
}

/// A fingerprint of logical fields and row values, independent of Arrow
/// chunking, buffer addresses, input ids and checkpoint rebinding ids.
fn fingerprint(
    description: &SourceDescription,
    tables: &[(Relation, ArrowData)],
) -> Result<String, CoreError> {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    let mut feed = |bytes: &[u8]| {
        for &byte in bytes {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    feed(
        &serde_json::to_vec(description)
            .map_err(|_| internal("frame graph facts do not serialize"))?,
    );
    for table in &description.tables {
        let (_, data) = tables
            .iter()
            .find(|(source, _)| source == &table.source)
            .ok_or_else(|| invalid("checkpoint lacks a logical source"))?;
        feed(
            &serde_json::to_vec(&table.source)
                .map_err(|_| internal("frame source does not serialize"))?,
        );
        let batch = crate::materialize::one_batch(data)?;
        for field in batch.schema().fields() {
            feed(field.name().as_bytes());
            feed(format!("{:?}", field.data_type()).as_bytes());
        }
        let converter = arrow_row::RowConverter::new(
            batch
                .schema()
                .fields()
                .iter()
                .map(|field| arrow_row::SortField::new(field.data_type().clone()))
                .collect(),
        )
        .map_err(|_| invalid("checkpoint contains a nonnumeric frame"))?;
        let rows = converter
            .convert_columns(batch.columns())
            .map_err(|_| invalid("checkpoint row values are unreadable"))?;
        for row in rows.iter() {
            let bytes = row.as_ref();
            feed(&(bytes.len() as u64).to_le_bytes());
            feed(bytes);
        }
    }
    Ok(format!("{hash:016x}"))
}

fn detached(data: &ArrowData) -> Result<ArrowData, CoreError> {
    let batch = crate::materialize::one_batch(data)?;
    let indices = UInt64Array::from(
        (0..batch.num_rows())
            .map(|row| row as u64)
            .collect::<Vec<_>>(),
    );
    let arrays = batch
        .columns()
        .iter()
        .map(|array| {
            arrow_select::take::take(array.as_ref(), &indices, None)
                .map_err(|_| internal("detaching a frame failed"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let copy = RecordBatch::try_new(batch.schema(), arrays)
        .map_err(|_| internal("detached frame columns do not align"))?;
    Ok(ArrowData::new(copy.schema(), vec![copy]))
}

impl FrameRun {
    pub fn source(&self) -> String {
        self.metadata
            .as_ref()
            .map(|metadata| metadata.source.clone())
            .unwrap_or_else(|| "frames".to_owned())
    }
    /// Called at the existing materializer's graph-build stage, before
    /// entity validation. Failed construction publishes no snapshot.
    pub fn initialize(
        &mut self,
        asker: &Asker,
        description: &SourceDescription,
        graph: &Graph,
    ) -> Result<(), CoreError> {
        if self.snapshot.is_some() {
            return Ok(());
        }
        if self.metadata.is_none() {
            self.tables = self
                .tables
                .iter()
                .map(|(source, data)| Ok((source.clone(), detached(data)?)))
                .collect::<Result<_, CoreError>>()?;
        }
        let source = fingerprint(description, &self.tables)?;
        if self
            .metadata
            .as_ref()
            .is_some_and(|metadata| metadata.source != source)
        {
            return Err(
                invalid("frame checkpoint no longer matches its logical snapshot")
                    .dropping(HeldState::SamplerState),
            );
        }
        let rows = description
            .tables
            .iter()
            .map(|table| {
                let (_, rows) = self
                    .tables
                    .iter()
                    .find(|(source, _)| source == &table.source)
                    .ok_or_else(|| invalid("a materialized frame lacks its source"))?;
                Ok((table.name.clone(), crate::materialize::one_batch(rows)?))
            })
            .collect::<Result<BTreeMap<_, _>, CoreError>>()?;
        self.snapshot = Some(Snapshot::new(graph, rows)?);
        self.metadata = Some(FrameCheckpointMetadata {
            source,
            description: description.clone(),
            neighbors: self.neighbors.checkpoint(),
        });
        self.keep(asker);
        Ok(())
    }

    pub fn keep(&mut self, asker: &Asker) {
        if let Some(metadata) = &mut self.metadata {
            metadata.neighbors = self.neighbors.checkpoint();
            asker.keep_frame(OwnedCheckpoint {
                metadata: metadata.clone(),
                tables: self.tables.clone(),
            });
        }
    }

    pub fn entities(
        &self,
        table: &Table,
        columns: &[String],
        limit: usize,
        seed: u128,
    ) -> Result<RecordBatch, CoreError> {
        self.snapshot
            .as_ref()
            .ok_or_else(|| internal("frame graph not initialized"))?
            .entities(&table.name, columns, limit, seed)
    }

    pub fn lookup(
        &self,
        table: &Table,
        keys: &ArrayRef,
        columns: &[String],
    ) -> Result<RecordBatch, CoreError> {
        self.snapshot
            .as_ref()
            .ok_or_else(|| internal("frame graph not initialized"))?
            .lookup(&table.name, keys, columns)
    }

    pub fn walk(
        &mut self,
        asker: &Asker,
        graph: &Graph,
        entity: &Table,
        keys: &ArrayRef,
        hops: &[Hop],
        excluded: &[(String, String)],
        seed: Option<u128>,
    ) -> Result<crate::subgraph::Walked, CoreError> {
        if let Some(seed) = seed {
            let seed = u32::try_from(seed)
                .ok()
                .filter(|&seed| seed <= i32::MAX as u32)
                .ok_or_else(|| {
                    CoreError::new(
                        Sentence::SeedPastRange,
                        "frame neighbor seed exceeds the native sampler range",
                    )
                    .with_field("request", Detail::text("sample_neighbors"))
                    .with_field("field", Detail::text("seed"))
                    .with_field("max_seed", Detail::Scalar(Scalar::UInt64(i32::MAX as u64)))
                })?;
            self.neighbors = Mt19937::seeded(seed);
        }
        let hops = hops
            .iter()
            .take_while(|hop| !matches!(hop, Hop::Stop))
            .map(|hop| match hop {
                Hop::Keep(count) => i64::try_from(*count).unwrap_or(i64::MAX),
                Hop::Stop => 0,
            })
            .collect::<Vec<_>>();
        let sampled = self
            .snapshot
            .as_ref()
            .ok_or_else(|| internal("frame graph not initialized"))?
            .walk(graph, &entity.name, keys, &hops, &mut self.neighbors);
        // A later cap, encoding error or cancellation must not replay draws.
        self.keep(asker);
        let sampled = sampled?;
        let tables = sampled
            .tables
            .into_iter()
            .map(|table| {
                let described = graph
                    .table(&table.name)
                    .ok_or_else(|| internal("sampled unknown frame"))?;
                let features = crate::subgraph::features(graph, described, excluded);
                let is_entity = table.name == entity.name;
                let columns = table
                    .rows
                    .schema()
                    .fields()
                    .iter()
                    .zip(table.rows.columns())
                    .filter(|(field, _)| {
                        features.iter().any(|(name, _)| name == field.name())
                            || (is_entity && described.primary_key.as_ref() == Some(field.name()))
                    })
                    .map(|(field, array)| {
                        (
                            field.name().clone(),
                            SampledColumn::Arrow(Arc::clone(array)),
                        )
                    })
                    .collect();
                Ok(SampledTable {
                    name: table.name,
                    primary_key: is_entity.then(|| described.primary_key.clone()).flatten(),
                    batch: table.batch,
                    columns,
                    stypes: features,
                })
            })
            .collect::<Result<Vec<_>, CoreError>>()?;
        let links = sampled
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
