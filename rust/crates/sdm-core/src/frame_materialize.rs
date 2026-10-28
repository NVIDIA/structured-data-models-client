// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Materialize's fixed frame consumer and its owned checkpoint.
//! Metadata is JSON, rows are Arrow, and every exported id is call-local.

pub(crate) mod identity;
mod retained;
mod row_identity;
use row_identity::OwnedIdentity;
pub(crate) mod time_columns;
pub(crate) mod values;

pub use retained::RetainedFrameGraph;

#[cfg(test)]
mod identity_tests;
#[cfg(test)]
pub(crate) mod retained_tests;

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::DataType;
use sdm_contract::data::strings::StringEncoding;
use sdm_contract::data::{AttachmentId, Relation};
use sdm_contract::frames::materialize::{
    FrameCompositeDefinition, FrameIdentityCodec, FrameIdentityComponent, FrameNativeTable,
    FrameRowIdentity,
};
use sdm_contract::frames::{
    FRAME_CHECKPOINT_VERSION, FrameCheckpoint, FrameCheckpointManifest, FrameCheckpointMetadata,
    MaterializeFrames,
};
use sdm_contract::ops::{Dtype, HeldSchema, SamplerMemo, SourceDescription, Stype};
use sdm_contract::{CoreError, CoreWarning, Detail, HeldState, Scalar, Sentence};
use sdm_engine::frames::Snapshot;
use sdm_engine::graph::{Graph, NativeGraph, Table};
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
    pub row_identities: Vec<OwnedIdentity>,
}

/// A fully constructed immutable graph, independently of any task or walk.
/// The source binds its meanings and retained Arrow rows, never attachment ids.
pub(crate) struct PreparedGraph {
    pub source: String,
    pub description: SourceDescription,
    pub graph: NativeGraph,
}

/// Actual selected raw tables and links of one graph snapshot. The useful
/// context exists before native Text facts or request encoding are reached.
/// Arrow arrays remain independently owned and are not serialized as model JSON.
pub(crate) struct SampledContext {
    source: String,
    sampled: crate::subgraph::Walked,
    native_rows: BTreeMap<String, Vec<usize>>,
}

impl SampledContext {
    fn check_source(&self, source: &str) -> Result<(), CoreError> {
        if self.source != source {
            return Err(invalid("sampled context belongs to another frame graph"));
        }
        Ok(())
    }

    pub fn into_walked(self, source: &str) -> Result<crate::subgraph::Walked, CoreError> {
        self.check_source(source)?;
        Ok(self.sampled)
    }
}

pub(crate) struct FrameRun {
    row_identities: Vec<OwnedIdentity>,
    tables: Vec<(Relation, ArrowData)>,
    metadata: Option<FrameCheckpointMetadata>,
    snapshot: Option<Snapshot>,
    neighbors: Mt19937,
    pub entropy: Option<u64>,
    /// Core-created live owners retain the actual graph/state directly.
    /// Their standard state handoff must not re-export every raw table.
    retained_owner: bool,
}

/// Per-walk sampling options, borrowed from this materialization call.
pub(crate) struct FrameWalkOptions<'a> {
    pub hops: &'a [Hop],
    pub neighbor_values: Option<&'a [Value]>,
    pub excluded: &'a [(String, String)],
    pub seed: Option<u128>,
    pub link_target: Option<&'a str>,
}

fn bound_tables(
    bindings: Vec<sdm_contract::frames::FrameInput>,
    attachments: &mut Attachments,
) -> Result<Vec<(Relation, ArrowData)>, CoreError> {
    bindings
        .into_iter()
        .map(|binding| {
            // This unfinished materializer has not implemented logical
            // native Decimal values. Never silently treat a constructor's
            // scale-zero mantissa attachment as its model-input values.
            if !binding.decimals.is_empty() {
                return Err(invalid(
                    "native Decimal declarations require a materializer logical-value consumer",
                ));
            }
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
    let (tables, row_identities, metadata, neighbors) = match checkpoint {
        None => (
            bound_tables(input.tables, attachments)?,
            row_identity::bind(input.row_identities, attachments)?,
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
            if !input.tables.is_empty() || !input.row_identities.is_empty() {
                return Err(invalid("a resumed frame run must not reread caller frames"));
            }
            let manifest: FrameCheckpointManifest = take_json(attachments, checkpoint.manifest)?;
            if manifest.retained.is_some() {
                return Err(invalid(
                    "a retained graph must be rebound through its fixed frame value",
                ));
            }
            let metadata: FrameCheckpointMetadata = take_json(attachments, manifest.metadata)?;
            let neighbors = Mt19937::from_checkpoint(&metadata.neighbors).ok_or_else(|| {
                invalid("invalid frame neighbor checkpoint").dropping(HeldState::SamplerState)
            })?;
            (
                bound_tables(manifest.tables, attachments)?,
                row_identity::bind(manifest.row_identities, attachments)?,
                Some(metadata),
                neighbors,
            )
        }
    };
    if let Some(metadata) = &metadata {
        let source = fingerprint(
            &metadata.description,
            &tables,
            &row_identities,
            metadata.strings.as_ref(),
        )
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
        row_identities,
        metadata,
        snapshot: None,
        neighbors,
        entropy: input.entropy,
        retained_owner: false,
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
    row_identities: &[OwnedIdentity],
    strings: Option<&BTreeMap<String, BTreeMap<String, StringEncoding>>>,
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
    if let Some(strings) = strings {
        feed(b"native-string-encodings-v1");
        feed(
            &serde_json::to_vec(strings)
                .map_err(|_| internal("native text encodings do not serialize"))?,
        );
    }
    for identity in row_identities {
        feed(b"native-row-identity-v1");
        feed(
            &serde_json::to_vec(&(&identity.definition, identity.encoding))
                .map_err(|_| internal("native identity facts do not serialize"))?,
        );
        let batch = crate::materialize::one_batch(&identity.values)?;
        let converter = arrow_row::RowConverter::new(vec![arrow_row::SortField::new(
            batch.column(0).data_type().clone(),
        )])
        .map_err(|_| invalid("native identity type cannot be fingerprinted"))?;
        let values = converter
            .convert_columns(batch.columns())
            .map_err(|_| invalid("native identity values cannot be fingerprinted"))?;
        for row in values.iter() {
            feed(&(row.as_ref().len() as u64).to_le_bytes());
            feed(row.as_ref());
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

/// Meaning checks consume the same folded operational key as the actual
/// native graph. Full ordered source identity remains in the caller value.
pub(crate) fn native_projection(description: &SourceDescription) -> SourceDescription {
    let mut projected = description.clone();
    for table in &mut projected.tables {
        // Native columns already received their reached current metadata.
        // Caller-schema historical marks must not release those assignments.
        for column in &mut table.columns {
            column.stype_before_key = None;
        }
        if table.primary_key.len() > 1 {
            table.primary_key = vec![sdm_engine::graph::folded_name(&table.primary_key)];
        }
    }
    projected
}

impl FrameRun {
    pub(crate) fn retain_owner(&mut self) {
        self.retained_owner = true;
    }
    /// Native table construction infers the physical primitive family before
    /// applying Text meaning. A logical dtype edit cannot turn numeric rows
    /// into a text column. Reuse the shared semantic rule and diagnostic.
    pub fn validate_text_types(&self, description: &SourceDescription) -> Result<(), CoreError> {
        self.validate_text_types_with(description, None)
    }

    pub fn validate_native_text_types(
        &self,
        description: &SourceDescription,
        native: &[FrameNativeTable],
    ) -> Result<(), CoreError> {
        self.validate_text_types_with(description, Some(native))
    }

    fn validate_text_types_with(
        &self,
        description: &SourceDescription,
        native: Option<&[FrameNativeTable]>,
    ) -> Result<(), CoreError> {
        let mut physical = description.clone();
        for table in &mut physical.tables {
            let (_, rows) = self
                .tables
                .iter()
                .find(|(source, _)| source == &table.source)
                .ok_or_else(|| invalid("a materialized frame lacks its source"))?;
            for column in table
                .columns
                .iter_mut()
                .filter(|column| column.stype == Stype::Text)
            {
                if let Some(constructed) = native
                    .and_then(|tables| tables.iter().find(|native| native.source == table.source))
                    .and_then(|native| {
                        native
                            .columns
                            .iter()
                            .find(|native| native.name == column.name)
                    })
                    .and_then(|native| native.constructed)
                {
                    column.dtype = constructed;
                    continue;
                }
                let field = rows
                    .schema
                    .field_with_name(&column.name)
                    .map_err(|_| invalid("a frame lacks a described column"))?;
                let dtype = field.data_type();
                if dtype.is_integer() {
                    column.dtype = Dtype::Int;
                } else if dtype.is_floating() {
                    column.dtype = Dtype::Float;
                } else if dtype == &DataType::Boolean {
                    column.dtype = Dtype::Bool;
                }
            }
        }
        crate::materialize::meanings_carried(&native_projection(&physical), None)
    }

    fn table_shapes(&self, description: &SourceDescription) -> Result<(), CoreError> {
        for table in &description.tables {
            let (_, data) = self
                .tables
                .iter()
                .find(|(source, _)| source == &table.source)
                .ok_or_else(|| invalid("a native graph lacks its constructed source table"))?;
            if data.schema.fields().is_empty()
                || data.batches.iter().all(|batch| batch.num_rows() == 0)
            {
                return Err(sdm_engine::frame_identity::refusal(
                    "empty_frame",
                    &table.name,
                    "",
                ));
            }
        }
        Ok(())
    }

    /// Prepare a useful graph artifact before task-native input conversion.
    /// Construction retains the snapshot before its separate drop notice is
    /// presented, so warning-as-error does not reread mutated caller frames.
    pub fn prepare_graph(
        &mut self,
        asker: &Asker,
        description: SourceDescription,
    ) -> Result<PreparedGraph, CoreError> {
        self.table_shapes(&description)?;
        let graph = NativeGraph::new(&description)?;
        self.initialize_strings(asker, &description, &graph, None)?;
        Ok(PreparedGraph {
            source: self.source(),
            description,
            graph,
        })
    }

    /// Core-validated native origins supply explicit logical text encodings
    /// exactly once while building the same immutable graph/index owners.
    pub fn prepare_native_graph(
        &mut self,
        asker: &Asker,
        description: SourceDescription,
        native: &[FrameNativeTable],
    ) -> Result<PreparedGraph, CoreError> {
        self.table_shapes(&description)?;
        let graph = NativeGraph::new(&description)?;
        let strings = description
            .tables
            .iter()
            .map(|table| {
                let native = native
                    .iter()
                    .find(|native| native.source == table.source)
                    .ok_or_else(|| invalid("native frame graph lacks its source facts"))?;
                Ok((
                    table.name.clone(),
                    native
                        .columns
                        .iter()
                        .filter_map(|column| {
                            column
                                .strings
                                .map(|encoding| (column.name.clone(), encoding))
                        })
                        .collect(),
                ))
            })
            .collect::<Result<BTreeMap<_, _>, CoreError>>()?;
        self.initialize_strings(asker, &description, &graph, Some(strings))?;
        Ok(PreparedGraph {
            source: self.source(),
            description,
            graph,
        })
    }

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
    pub fn string_encodings(
        &self,
        table: &str,
    ) -> Result<BTreeMap<String, StringEncoding>, CoreError> {
        self.snapshot
            .as_ref()
            .ok_or_else(|| internal("frame graph is not constructed"))?
            .string_encodings(table)
    }

    pub fn source(&self) -> String {
        self.metadata
            .as_ref()
            .map(|metadata| metadata.source.clone())
            .unwrap_or_else(|| "frames".to_owned())
    }
    /// Called at the existing materializer's graph-build stage, before
    /// entity validation. Failed construction publishes no snapshot.
    #[cfg(test)]
    pub fn initialize(
        &mut self,
        asker: &Asker,
        description: &SourceDescription,
        graph: &Graph,
    ) -> Result<(), CoreError> {
        let native = NativeGraph::new(description)?;
        if &native.topology != graph {
            return Err(invalid(
                "native topology differs from its complete source identity",
            ));
        }
        self.initialize_strings(asker, description, &native, None)
    }

    fn initialize_strings(
        &mut self,
        asker: &Asker,
        description: &SourceDescription,
        graph: &NativeGraph,
        strings: Option<BTreeMap<String, BTreeMap<String, StringEncoding>>>,
    ) -> Result<(), CoreError> {
        if self.snapshot.is_some() {
            return Ok(());
        }
        // HostRun has already detached every admitted native input under
        // the binding's ownership guard. Successful graph construction
        // retains those immutable rows; failed construction caches none.
        // Resumed rows and identity were validated once at the ownership
        // boundary, before carry. Immutable owners need no second full scan.
        let strings = strings.or_else(|| {
            self.metadata
                .as_ref()
                .and_then(|metadata| metadata.strings.clone())
        });
        row_identity::complete(
            &mut self.row_identities,
            &self.tables,
            self.metadata.is_some(),
            description,
            graph,
            strings.as_ref(),
        )?;
        let source = match &self.metadata {
            Some(metadata) => metadata.source.clone(),
            None => fingerprint(
                description,
                &self.tables,
                &self.row_identities,
                strings.as_ref(),
            )?,
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
        let identities = self
            .row_identities
            .iter()
            .map(|identity| {
                let batch = crate::materialize::one_batch(&identity.values)?;
                Ok((
                    identity.definition.table.clone(),
                    sdm_engine::frame_identity::IdentityColumn {
                        values: Arc::clone(batch.column(0)),
                        encoding: identity.encoding,
                    },
                ))
            })
            .collect::<Result<BTreeMap<_, _>, CoreError>>()?;
        self.snapshot = Some(Snapshot::new_frame(
            graph,
            rows,
            strings.clone(),
            identities,
        )?);
        self.metadata = Some(FrameCheckpointMetadata {
            source,
            description: description.clone(),
            neighbors: self.neighbors.checkpoint(),
            strings,
        });
        self.keep(asker);
        Ok(())
    }

    pub fn keep(&mut self, asker: &Asker) {
        if let Some(metadata) = &mut self.metadata {
            metadata.neighbors = self.neighbors.checkpoint();
            if self.retained_owner {
                return;
            }
            asker.keep_frame(OwnedCheckpoint {
                metadata: metadata.clone(),
                tables: self.tables.clone(),
                row_identities: self.row_identities.clone(),
            });
        }
    }

    /// The cached graph reports the same removals on every materialize call.
    pub fn warn_dropped(&self, asker: &Asker) -> bool {
        let Some(snapshot) = &self.snapshot else {
            return false;
        };
        let Some(detail) = snapshot.dropped_summary() else {
            return false;
        };
        asker.warn(
            CoreWarning::new(
                Sentence::FrameRowsDropped,
                "graph sanitization removed rows",
            )
            .with_field("detail", Detail::text(detail)),
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

    pub fn source_keys(
        &self,
        table: &Table,
        column: &str,
        values: &ArrayRef,
    ) -> Result<Vec<Option<sdm_engine::keys::Key>>, CoreError> {
        self.snapshot
            .as_ref()
            .ok_or_else(|| internal("frame graph not initialized"))?
            .source_keys(&table.name, column, values)
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

    /// Useful node-reference rows of the complete native-converted task.
    /// This must finish before any later batch can reset or advance neighbors.
    pub fn resolve_references(
        &self,
        table: &Table,
        columns: &[&ArrayRef],
    ) -> Result<Vec<usize>, CoreError> {
        self.snapshot
            .as_ref()
            .ok_or_else(|| internal("frame graph not initialized"))?
            .entity_rows_many(&table.name, columns)
    }

    pub fn resolve_encoded_references(
        &self,
        table: &Table,
        columns: &[(&ArrayRef, Option<StringEncoding>)],
    ) -> Result<Vec<usize>, CoreError> {
        self.snapshot
            .as_ref()
            .ok_or_else(|| internal("frame graph not initialized"))?
            .entity_rows_encoded(&table.name, columns)
    }

    /// Actual immutable rows selected by the same existing primary-key map.
    /// Native candidate conversions change lookup values, not attribution IDs.
    pub fn source_rows(
        &self,
        table: &Table,
        rows: &[usize],
        columns: &[String],
    ) -> Result<RecordBatch, CoreError> {
        self.snapshot
            .as_ref()
            .ok_or_else(|| internal("frame graph not initialized"))?
            .selected_rows(&table.name, rows, columns)
    }

    /// Existing immutable frame adjacency supplies temporal label rows.
    /// This operation does not borrow or advance the neighbor generator.
    pub fn temporal_rows(
        &self,
        edge: &sdm_engine::graph::Edge,
        keys: &ArrayRef,
        windows: &[sdm_engine::frames::TemporalWindow],
        columns: &[String],
    ) -> Result<sdm_engine::frames::WalkedTable, CoreError> {
        self.snapshot
            .as_ref()
            .ok_or_else(|| internal("frame graph not initialized"))?
            .temporal_rows(edge, keys, windows, columns)
    }

    pub fn walk(
        &mut self,
        asker: &Asker,
        graph: &Graph,
        entity: &Table,
        keys: &ArrayRef,
        anchors: &[i64],
        options: FrameWalkOptions<'_>,
    ) -> Result<SampledContext, CoreError> {
        // Native key conversion must finish before a seed reset or draw.
        let entity_rows = self.resolve_references(entity, &[keys])?;
        self.walk_rows(asker, graph, entity, &entity_rows, anchors, options)
    }

    pub fn walk_rows(
        &mut self,
        asker: &Asker,
        graph: &Graph,
        entity: &Table,
        entity_rows: &[usize],
        anchors: &[i64],
        options: FrameWalkOptions<'_>,
    ) -> Result<SampledContext, CoreError> {
        let FrameWalkOptions {
            hops,
            neighbor_values,
            excluded,
            seed,
            link_target,
        } = options;
        // The caller has resolved this batch's actual converted values
        // before resetting the continuing neighbor stream, independently
        // of the earlier whole-task reference validation.
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
        // SQL hop bounds retain their own signed/expression meaning. When
        // no original native array was supplied, project those same values
        // through the existing frame count-array admission instead of
        // flattening a negative bound or expression into a positive count.
        let projected;
        let values = match neighbor_values {
            Some(values) => values,
            None => {
                projected = hops
                    .iter()
                    .map(|hop| match hop {
                        Hop::Keep(count) => {
                            Value::Number(i64::try_from(*count).unwrap_or(i64::MAX).to_string())
                        }
                        Hop::Stop => Value::Number("0".into()),
                        Hop::Negative(value) => Value::Number(value.as_str().into()),
                        Hop::Expression(text) => Value::Text(text.clone().into()),
                    })
                    .collect::<Vec<_>>();
                projected.as_slice()
            }
        };
        let hops = match native_hops(graph, values) {
            Ok(hops) => hops,
            Err(error) => {
                // The actual local sampler resets before validating
                // its count array, without consuming a neighbor draw.
                self.keep(asker);
                return Err(error);
            }
        };
        let sampled = self
            .snapshot
            .as_ref()
            .ok_or_else(|| internal("frame graph not initialized"))?
            .walk_resolved(
                graph,
                &entity.name,
                entity_rows,
                anchors,
                &hops,
                &mut self.neighbors,
            );
        // A later cap, encoding error or cancellation must not replay draws.
        self.keep(asker);
        let sampled = sampled?;
        let mut native_rows = BTreeMap::new();
        let tables = sampled
            .tables
            .into_iter()
            .map(|table| {
                native_rows.insert(table.name.clone(), table.source_rows);
                let described = graph
                    .table(&table.name)
                    .ok_or_else(|| internal("sampled unknown frame"))?;
                let features = crate::subgraph::features(graph, described, excluded);
                let is_entity =
                    table.name == entity.name || link_target == Some(table.name.as_str());
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
        Ok(SampledContext {
            source: self.source(),
            sampled: crate::subgraph::Walked { tables, links },
            native_rows,
        })
    }
}

#[cfg(test)]
mod parent_hop_tests {
    use super::*;
    use arrow_array::Int64Array;
    use sdm_contract::Step;
    use sdm_contract::data::NegativeNeighborBound;
    use sdm_contract::prepare::PreparedRequest;
    use serde_json::json;

    fn fixture() -> (FrameRun, SourceDescription, Graph) {
        let column = |name: &str| json!({"name":name,"dtype":"int64","stype":"ID"});
        let held: HeldSchema = serde_json::from_value(json!({"tables":[
            {"name":"users","source":{"source":["users"]},"columns":[column("user_id")],"primary_key":["user_id"],"time_column":null},
            {"name":"orders","source":{"source":["orders"]},"columns":[column("order_id"),column("user_id"),{"name":"row_value","dtype":"int64","stype":"numerical"}],"primary_key":["order_id"],"time_column":null}
        ],"links":[[{"table":"orders","column":"user_id"},{"table":"users","column":"user_id"}]]})).unwrap();
        let description = sdm_schema::held::read(held, &BTreeMap::new(), true)
            .unwrap()
            .description;
        let graph = Graph::new(&description).unwrap();
        let batches = [
            RecordBatch::try_from_iter([(
                "user_id",
                Arc::new(Int64Array::from(vec![1])) as ArrayRef,
            )])
            .unwrap(),
            RecordBatch::try_from_iter([
                (
                    "order_id",
                    Arc::new(Int64Array::from_iter_values(0..12)) as ArrayRef,
                ),
                (
                    "user_id",
                    Arc::new(Int64Array::from(vec![1; 12])) as ArrayRef,
                ),
                (
                    "row_value",
                    Arc::new(Int64Array::from_iter_values(0..12)) as ArrayRef,
                ),
            ])
            .unwrap(),
        ];
        let tables = description
            .tables
            .iter()
            .zip(batches)
            .map(|(table, batch)| {
                (
                    table.source.clone(),
                    ArrowData::new(batch.schema(), vec![batch]),
                )
            })
            .collect();
        (
            FrameRun {
                row_identities: Vec::new(),
                tables,
                metadata: None,
                snapshot: None,
                neighbors: Mt19937::seeded(123),
                entropy: None,
                retained_owner: false,
            },
            description,
            graph,
        )
    }

    fn order_ids(walked: &crate::subgraph::Walked) -> Vec<i64> {
        walked
            .tables
            .iter()
            .find(|table| table.name == "orders")
            .map(|table| {
                // Nonentity primary keys are not projected as model features.
                // This ordinary numeric feature carries those same source row
                // identities through the actual FrameRun projection.
                let (_, SampledColumn::Arrow(values)) = table
                    .columns
                    .iter()
                    .find(|(name, _)| name == "row_value")
                    .expect("the sampled source-row feature")
                else {
                    panic!("the numeric source-row feature must stay Arrow");
                };
                values
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap()
                    .values()
                    .to_vec()
            })
            .unwrap_or_default()
    }

    #[test]
    fn both_entity_roles_keep_keys_without_changing_history_features_or_draws() {
        let mut run = crate::Run::new(|asker| async move {
            let mut checkpoints = Vec::new();
            let mut features = Vec::new();
            for link_target in [None, Some("orders"), Some("users")] {
                let (mut frame, description, graph) = fixture();
                frame.initialize(&asker, &description, &graph)?;
                let keys: ArrayRef = Arc::new(Int64Array::from(vec![1]));
                let sampled = frame.walk(
                    &asker,
                    &graph,
                    &graph.tables[0],
                    &keys,
                    &[0],
                    FrameWalkOptions {
                        hops: &[Hop::Keep(3)],
                        neighbor_values: None,
                        excluded: &[],
                        seed: Some(7),
                        link_target,
                    },
                )?;
                let entity = sampled
                    .sampled
                    .tables
                    .iter()
                    .find(|table| table.name == "users")
                    .unwrap();
                assert_eq!(entity.primary_key.as_deref(), Some("user_id"));
                let history = sampled
                    .sampled
                    .tables
                    .iter()
                    .find(|table| table.name == "orders")
                    .unwrap();
                let rhs = link_target == Some("orders");
                assert_eq!(history.primary_key.as_deref(), rhs.then_some("order_id"));
                let names = history
                    .columns
                    .iter()
                    .map(|(name, _)| name.as_str())
                    .collect::<Vec<_>>();
                assert_eq!(
                    names.iter().position(|name| *name == "order_id"),
                    rhs.then_some(0)
                );
                features.push(
                    names
                        .into_iter()
                        .filter(|name| *name != "order_id")
                        .map(str::to_owned)
                        .collect::<Vec<_>>(),
                );
                checkpoints.push(frame.neighbors.checkpoint());
            }
            assert!(features.windows(2).all(|pair| pair[0] == pair[1]));
            assert!(checkpoints.windows(2).all(|pair| pair[0] == pair[1]));
            let value = asker.attach(Attachment::Bytes(Vec::new()))?;
            Ok(sdm_contract::ops::OperationResult::PrepareRequest(
                PreparedRequest { value },
            ))
        });
        assert!(matches!(run.next().unwrap(), Step::Done { .. }));
    }

    #[test]
    fn missing_batch_references_fail_before_a_requested_neighbor_reset() {
        let mut run = crate::Run::new(|asker| async move {
            let (mut frame, description, graph) = fixture();
            frame.initialize(&asker, &description, &graph)?;
            let keys: ArrayRef = Arc::new(Int64Array::from(vec![1]));
            frame.walk(
                &asker,
                &graph,
                &graph.tables[0],
                &keys,
                &[0],
                FrameWalkOptions {
                    hops: &[Hop::Keep(3)],
                    neighbor_values: None,
                    excluded: &[],
                    seed: Some(99),
                    link_target: None,
                },
            )?;
            let before = frame.neighbors.checkpoint();
            let missing: ArrayRef = Arc::new(Int64Array::from(vec![999]));
            assert!(
                frame
                    .walk(
                        &asker,
                        &graph,
                        &graph.tables[0],
                        &missing,
                        &[0],
                        FrameWalkOptions {
                            hops: &[Hop::Keep(3)],
                            neighbor_values: None,
                            excluded: &[],
                            seed: Some(7),
                            link_target: None,
                        }
                    )
                    .is_err()
            );
            assert_eq!(frame.neighbors.checkpoint(), before);
            assert_eq!(frame.metadata.as_ref().unwrap().neighbors, before);
            let value = asker.attach(Attachment::Bytes(Vec::new()))?;
            Ok(sdm_contract::ops::OperationResult::PrepareRequest(
                PreparedRequest { value },
            ))
        });
        assert!(matches!(run.next().unwrap(), Step::Done { .. }));
    }

    #[test]
    fn projected_hops_keep_native_signed_counts_errors_and_seed_continuation() {
        for (hop, refused, whole) in [
            (
                Hop::Negative(NegativeNeighborBound::new("-0.5".into()).unwrap()),
                true,
                false,
            ),
            (Hop::Expression("3 OR TRUE".into()), true, false),
            (
                Hop::Negative(NegativeNeighborBound::new("-1".into()).unwrap()),
                false,
                true,
            ),
            (Hop::Keep(3), false, false),
        ] {
            let mut run = crate::Run::new(move |asker| async move {
                let (mut frame, description, graph) = fixture();
                frame.initialize(&asker, &description, &graph)?;
                let keys = Arc::new(Int64Array::from(vec![1])) as ArrayRef;
                // A real earlier walk consumes the retained native stream.
                frame.walk(
                    &asker,
                    &graph,
                    &graph.tables[0],
                    &keys,
                    &[0],
                    FrameWalkOptions {
                        hops: &[Hop::Keep(3)],
                        neighbor_values: None,
                        excluded: &[],
                        seed: Some(99),
                        link_target: None,
                    },
                )?;
                assert_ne!(frame.metadata.as_ref().unwrap().neighbors.position, 624);
                let hops = [hop];
                let selected = frame.walk(
                    &asker,
                    &graph,
                    &graph.tables[0],
                    &keys,
                    &[0],
                    FrameWalkOptions {
                        hops: &hops,
                        neighbor_values: None,
                        excluded: &[],
                        seed: Some(7),
                        link_target: None,
                    },
                );
                if refused {
                    let error = selected
                        .err()
                        .expect("fractional/expression native count refusal");
                    assert_eq!(error.sentence, Sentence::FrameNeighborDtype);
                    assert_eq!(
                        frame.metadata.as_ref().unwrap().neighbors,
                        Mt19937::seeded(7).checkpoint()
                    );
                } else {
                    let first = selected
                        .unwrap_or_else(|error| panic!("{error:?}"))
                        .into_walked(&frame.source())?;
                    assert_eq!(
                        order_ids(&first),
                        if whole {
                            (0..12).collect()
                        } else {
                            vec![5, 3, 1]
                        }
                    );
                    if whole {
                        assert_eq!(
                            frame.metadata.as_ref().unwrap().neighbors,
                            Mt19937::seeded(7).checkpoint()
                        );
                    }
                }
                let continued = frame
                    .walk(
                        &asker,
                        &graph,
                        &graph.tables[0],
                        &keys,
                        &[0],
                        FrameWalkOptions {
                            hops: &[Hop::Keep(3)],
                            neighbor_values: None,
                            excluded: &[],
                            seed: None,
                            link_target: None,
                        },
                    )?
                    .into_walked(&frame.source())?;
                let selected = order_ids(&continued);
                // Recorded native sampler vectors: a reset/error or a whole
                // neighborhood consumes no draw; an earlier partial walk does.
                assert_eq!(
                    selected,
                    if refused || whole {
                        vec![5, 3, 1]
                    } else {
                        vec![6, 2, 3]
                    }
                );
                assert_eq!(
                    frame.metadata.as_ref().unwrap().neighbors,
                    frame.neighbors.checkpoint()
                );
                let value = asker.attach(Attachment::Bytes(serde_json::to_vec(&json!({"selected":selected,"neighbors":frame.metadata.as_ref().unwrap().neighbors})).unwrap()))?;
                Ok(sdm_contract::ops::OperationResult::PrepareRequest(
                    PreparedRequest { value },
                ))
            });
            let Step::Done {
                result: sdm_contract::ops::OperationResult::PrepareRequest(value),
            } = run.next().unwrap()
            else {
                panic!("a fixed native frame walk must finish without I/O")
            };
            let Some(Attachment::Bytes(value)) = run.take_attachment(value.value) else {
                panic!("useful selected rows")
            };
            let value: serde_json::Value = serde_json::from_slice(&value).unwrap();
            assert_eq!(value["selected"].as_array().unwrap().len(), 3);
        }
    }
}
