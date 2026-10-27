// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Materialize's fixed frame consumer and its owned checkpoint.
//! Metadata is JSON, rows are Arrow, and every exported id is call-local.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use arrow_array::{ArrayRef, RecordBatch};
use sdm_contract::data::{AttachmentId, Relation};
use sdm_contract::frames::{
    FRAME_CHECKPOINT_VERSION, FrameCheckpoint, FrameCheckpointManifest, FrameCheckpointMetadata,
    MaterializeFrames,
};
use sdm_contract::ops::{Dtype, HeldSchema, SamplerMemo, SourceDescription};
use sdm_contract::{CoreError, CoreWarning, Detail, HeldState, Scalar, Sentence};
use sdm_engine::frames::Snapshot;
use sdm_engine::graph::{Graph, Table};
use sdm_engine::random::Mt19937;
use sdm_request::options::Hop;
use sdm_request::payload::{SampledColumn, SampledLink, SampledTable};
use sdm_wire::json::Value;

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
    if let Some(metadata) = &metadata {
        let source = fingerprint(&metadata.description, &tables)
            .map_err(|error| error.dropping(HeldState::SamplerState))?;
        if metadata.source != source {
            return Err(
                invalid("frame checkpoint no longer matches its logical snapshot")
                    .dropping(HeldState::SamplerState),
            );
        }
    }
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
    // This first opt-in consumer accepts numeric scalar identifiers.
    // Boolean and text feature columns do not imply identifying-key support.
    if description.tables.iter().any(|table| {
        table.primary_key.iter().any(|key| {
            table
                .columns
                .iter()
                .any(|column| column.name == *key && column.dtype == Dtype::Bool)
        })
    }) {
        return Err(sdm_engine::not_yet("frame_boolean_key"));
    }
    if description.tables.iter().any(|table| {
        table.primary_key.iter().any(|key| {
            table
                .columns
                .iter()
                .any(|column| column.name == *key && column.dtype == Dtype::String)
        })
    }) {
        return Err(sdm_engine::not_yet("frame_text_key"));
    }
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

/// The native sampler admits a signed int64 array, rather than the SQL
/// sampler's numeric comparison. NumPy promotes mixed bool/integer lists
/// to int64, but rejects an all-boolean or any float/text/object array.
fn native_hops(graph: &Graph, values: &[Value]) -> Result<Vec<i64>, CoreError> {
    let Some(edge) = graph.edges.first() else {
        return Ok(vec![]);
    };
    let refused = || {
        CoreError::new(
            Sentence::FrameNeighborDtype,
            "native neighbor counts lack int64 dtype",
        )
        .with_field(
            "edge",
            Detail::text(format!("{}__{}__{}", edge.src, edge.fkey, edge.dst)),
        )
    };
    let mut has_integer = false;
    let hops = values
        .iter()
        .map(|value| match value {
            Value::Bool(value) => Ok(i64::from(*value)),
            Value::Number(token) => {
                let number = token.parse::<i64>().map_err(|_| refused())?;
                has_integer = true;
                Ok(number)
            }
            _ => Err(refused()),
        })
        .collect::<Result<Vec<_>, CoreError>>()?;
    if !values.is_empty() && !has_integer {
        return Err(refused());
    }
    Ok(hops)
}

impl FrameRun {
    /// Publish an already validated retained owner at Run construction,
    /// before its body can fail or be cancelled without ever being polled.
    pub fn carry(&mut self, asker: &Asker, memo: Option<&SamplerMemo>) {
        if let Some(metadata) = &self.metadata {
            asker.keep(memo.cloned().unwrap_or_else(|| SamplerMemo {
                source: metadata.source.clone(),
                ..SamplerMemo::default()
            }));
            self.keep(asker);
        }
    }
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
        // HostRun has already detached every admitted native input under
        // the binding's ownership guard. Successful graph construction
        // retains those immutable rows; failed construction caches none.
        // Resumed rows and identity were validated once at the ownership
        // boundary, before carry. Immutable owners need no second full scan.
        let source = match &self.metadata {
            Some(metadata) => metadata.source.clone(),
            None => fingerprint(description, &self.tables)?,
        };
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

    /// The cached graph reports the same removals on every materialize call.
    pub fn warn_dropped(&self, asker: &Asker) -> bool {
        let Some(snapshot) = &self.snapshot else {
            return false;
        };
        if snapshot.dropped.is_empty() {
            return false;
        }
        let detail = snapshot
            .dropped
            .iter()
            .map(|row| {
                let mut causes = Vec::new();
                if row.null_keys > 0 {
                    causes.push(format!("{} with no primary key", row.null_keys));
                }
                if row.duplicate_keys > 0 {
                    causes.push(format!(
                        "{} with a repeated primary key",
                        row.duplicate_keys
                    ));
                }
                if row.null_times > 0 {
                    causes.push(format!("{} with no time", row.null_times));
                }
                format!(
                    "{}: {} of {} rows dropped ({})",
                    row.table,
                    row.null_keys + row.duplicate_keys + row.null_times,
                    row.input,
                    causes.join(", ")
                )
            })
            .collect::<Vec<_>>()
            .join("; ");
        asker.warn(
            CoreWarning::new(
                Sentence::FrameRowsDropped,
                "graph sanitization removed rows",
            )
            .with_field("detail", Detail::text(format!("[{detail}]"))),
        );
        true
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

    pub fn time_bounds(&self, table: &Table) -> Result<(Option<i64>, Option<i64>), CoreError> {
        self.snapshot
            .as_ref()
            .ok_or_else(|| internal("frame graph not initialized"))?
            .time_bounds(&table.name)
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
        anchors: &[i64],
        hops: &[Hop],
        neighbor_values: Option<&[Value]>,
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
        let hops = match neighbor_values {
            Some(values) => match native_hops(graph, values) {
                Ok(hops) => hops,
                Err(error) => {
                    // The actual local sampler resets before validating
                    // its count array, without consuming a neighbor draw.
                    self.keep(asker);
                    return Err(error);
                }
            },
            None => hops
                .iter()
                .map(|hop| match hop {
                    Hop::Keep(count) => i64::try_from(*count).unwrap_or(i64::MAX),
                    Hop::Stop => 0,
                })
                .collect::<Vec<_>>(),
        };
        let sampled = self
            .snapshot
            .as_ref()
            .ok_or_else(|| internal("frame graph not initialized"))?
            .walk(
                graph,
                &entity.name,
                keys,
                anchors,
                &hops,
                &mut self.neighbors,
            );
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
                        let text = features.iter().any(|(name, stype)| {
                            name == field.name() && *stype == sdm_contract::ops::Stype::Text
                        });
                        let column = if text {
                            SampledColumn::Words(crate::subgraph::words(array)?)
                        } else {
                            SampledColumn::Arrow(Arc::clone(array))
                        };
                        Ok((field.name().clone(), column))
                    })
                    .collect::<Result<_, CoreError>>()?;
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
