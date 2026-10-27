// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Ownership and Arrow rebinding for the fixed frame value consumers.
//! Pending final composition: this module is unlinked on the preserved base.

use std::collections::BTreeSet;

use arrow_array::cast::AsArray;
use arrow_array::types::UInt64Type;
use arrow_array::{Array, UInt64Array};
use sdm_contract::data::strings::StringData;
use sdm_contract::data::{AttachmentId, BATCH_COLUMN};
use sdm_contract::frames::materialize::{
    FRAME_MATERIALIZATION_VERSION, FrameNativeTable, PreparedFrameGraph, SampledFrameContext,
    SampledFrameLink, SampledFrameTable,
};
use sdm_contract::frames::{FrameCheckpointManifest, FrameInput};
use sdm_engine::strings::StringColumn;

use super::identity::Identity;
use super::*;

pub(crate) fn take_arrow(
    attachments: &mut Attachments,
    id: AttachmentId,
) -> Result<ArrowData, CoreError> {
    match attachments.remove(&id) {
        Some(Attachment::Arrow(data)) => Ok(data),
        _ => Err(invalid("a frame value lacks its named Arrow attachment")),
    }
}

pub(crate) fn take_array(
    attachments: &mut Attachments,
    id: AttachmentId,
) -> Result<ArrayRef, CoreError> {
    let batch = crate::materialize::one_batch(&take_arrow(attachments, id)?)?;
    if batch.num_columns() != 1 {
        return Err(invalid("a frame value column attachment is not one column"));
    }
    Ok(Arc::clone(batch.column(0)))
}

pub(crate) fn take_strings(
    attachments: &mut Attachments,
    data: &StringData,
) -> Result<ArrayRef, CoreError> {
    let array = take_array(attachments, data.values)?;
    let column = StringColumn::new(array.as_ref(), data.encoding)
        .map_err(|_| invalid("native strings do not match their explicit physical encoding"))?;
    for row in 0..column.len() {
        column
            .get(row)
            .map_err(|_| invalid("native strings contain invalid surrogatepass bytes"))?;
    }
    Ok(array)
}

pub(crate) fn attach_json<T: serde::Serialize>(
    asker: &Asker,
    value: &T,
) -> Result<AttachmentId, CoreError> {
    asker.attach(Attachment::Bytes(
        serde_json::to_vec(value).map_err(|_| internal("a frame value does not serialize"))?,
    ))
}

pub(crate) fn attach_batch(asker: &Asker, batch: RecordBatch) -> Result<AttachmentId, CoreError> {
    asker.attach(Attachment::Arrow(ArrowData::new(
        batch.schema(),
        vec![batch],
    )))
}

pub(crate) fn attach_array(
    asker: &Asker,
    name: &str,
    array: ArrayRef,
) -> Result<AttachmentId, CoreError> {
    let batch = RecordBatch::try_from_iter([(name, array)])
        .map_err(|_| internal("a frame value column does not make Arrow rows"))?;
    attach_batch(asker, batch)
}

pub(crate) fn positions(array: &ArrayRef) -> Result<Vec<usize>, CoreError> {
    let values = array
        .as_primitive_opt::<UInt64Type>()
        .filter(|values| values.null_count() == 0)
        .ok_or_else(|| invalid("frame source positions must be non-null UInt64 values"))?;
    values
        .values()
        .iter()
        .map(|&value| {
            usize::try_from(value)
                .map_err(|_| invalid("a frame source position exceeds this host's address space"))
        })
        .collect()
}

fn positional(values: &[usize]) -> Result<ArrayRef, CoreError> {
    Ok(Arc::new(UInt64Array::from(
        values
            .iter()
            .map(|&value| {
                u64::try_from(value).map_err(|_| internal("a frame source position exceeds UInt64"))
            })
            .collect::<Result<Vec<_>, _>>()?,
    )))
}

/// A useful graph has the actual domain graph and the actual detached owner.
/// The native facts are source meanings, independent of all attachment ids.
pub(crate) struct BoundGraph {
    pub prepared: PreparedGraph,
    pub frames: Context,
    pub native: Vec<FrameNativeTable>,
    pub source: String,
}

fn native_identity(source: &str, native: &[FrameNativeTable]) -> Result<String, CoreError> {
    let mut identity = Identity::new("graph");
    identity.value(source)?;
    identity.value(native)?;
    Ok(identity.finish())
}

fn validate_native(
    frames: &FrameRun,
    description: &SourceDescription,
    native: &[FrameNativeTable],
) -> Result<(), CoreError> {
    if native.len() != description.tables.len() {
        return Err(invalid(
            "frame native origins do not cover the immutable source",
        ));
    }
    let mut sources = BTreeSet::new();
    for table in native {
        if !sources.insert(&table.source) {
            return Err(invalid("frame native origins repeat a source"));
        }
        if !description
            .tables
            .iter()
            .any(|described| described.source == table.source)
        {
            return Err(invalid("frame native origins name an undescribed source"));
        }
        let (_, data) = frames
            .tables
            .iter()
            .find(|(source, _)| source == &table.source)
            .ok_or_else(|| invalid("frame native origins name an absent source"))?;
        if table.columns.len() != data.schema.fields().len() {
            return Err(invalid(
                "frame native origins do not cover their source columns",
            ));
        }
        for (column, field) in table.columns.iter().zip(data.schema.fields()) {
            if column.name != *field.name() {
                return Err(invalid("frame native origins change original column order"));
            }
            if let Some(encoding) = column.strings {
                for batch in &data.batches {
                    let values = batch
                        .column_by_name(&column.name)
                        .ok_or_else(|| invalid("native text column is absent"))?;
                    let strings = StringColumn::new(values.as_ref(), encoding).map_err(|_| {
                        invalid("native text origin does not match its Arrow encoding")
                    })?;
                    for row in 0..strings.len() {
                        strings.get(row).map_err(|_| {
                            invalid("native source text contains invalid surrogatepass bytes")
                        })?;
                    }
                }
            } else if matches!(field.data_type(), DataType::Utf8 | DataType::LargeUtf8) {
                return Err(invalid("native string source lacks its explicit encoding"));
            }
        }
    }
    Ok(())
}

impl BoundGraph {
    pub fn prepare(
        asker: &Asker,
        frames: Context,
        description: SourceDescription,
        native: Vec<FrameNativeTable>,
    ) -> Result<Self, CoreError> {
        let prepared = {
            let mut frame = lock(&frames);
            validate_native(&frame, &description, &native)?;
            frame.validate_text_types(&description)?;
            frame.prepare_graph(asker, description)?
        };
        let source = native_identity(&prepared.source, &native)?;
        Ok(Self {
            prepared,
            frames,
            native,
            source,
        })
    }

    /// Reconstruct only declared owners. No arbitrary integer can become an
    /// attachment id, and the checkpoint is verified before it is published.
    pub fn bind(
        asker: &Asker,
        value: PreparedFrameGraph,
        attachments: &mut Attachments,
    ) -> Result<Self, CoreError> {
        if value.version != FRAME_MATERIALIZATION_VERSION
            || value.checkpoint.version != FRAME_CHECKPOINT_VERSION
        {
            return Err(
                invalid("unknown prepared frame graph version").dropping(HeldState::SamplerState)
            );
        }
        let manifest: FrameCheckpointManifest = take_json(attachments, value.checkpoint.manifest)?;
        let metadata: FrameCheckpointMetadata = take_json(attachments, manifest.metadata)?;
        let tables = bound_tables(manifest.tables, attachments)?;
        let actual = fingerprint(&metadata.description, &tables)?;
        if actual != metadata.source {
            return Err(invalid(
                "prepared frame checkpoint no longer matches its immutable source",
            )
            .dropping(HeldState::SamplerState));
        }
        if native_identity(&actual, &value.native_tables)? != value.source {
            return Err(
                invalid("prepared frame graph no longer matches its native source facts")
                    .dropping(HeldState::SamplerState),
            );
        }
        let neighbors = Mt19937::from_checkpoint(&metadata.neighbors)
            .ok_or_else(|| invalid("invalid prepared frame neighbor state"))?;
        let description = metadata.description.clone();
        let frames = Arc::new(Mutex::new(FrameRun {
            tables,
            metadata: Some(metadata),
            snapshot: None,
            neighbors,
            entropy: None,
        }));
        let bound = Self::prepare(asker, frames, description, value.native_tables)?;
        Ok(bound)
    }

    pub fn publish(&self, asker: &Asker) -> Result<PreparedFrameGraph, CoreError> {
        let mut frames = lock(&self.frames);
        frames.keep(asker);
        let metadata = frames
            .metadata
            .as_ref()
            .ok_or_else(|| internal("prepared graph lacks checkpoint metadata"))?;
        let metadata = attach_json(asker, metadata)?;
        let tables = frames
            .tables
            .iter()
            .map(|(source, data)| {
                Ok(FrameInput {
                    source: source.clone(),
                    rows: asker.attach(Attachment::Arrow(data.clone()))?,
                })
            })
            .collect::<Result<Vec<_>, CoreError>>()?;
        let manifest = attach_json(asker, &FrameCheckpointManifest { metadata, tables })?;
        Ok(PreparedFrameGraph {
            version: FRAME_MATERIALIZATION_VERSION,
            source: self.source.clone(),
            checkpoint: FrameCheckpoint {
                version: FRAME_CHECKPOINT_VERSION,
                manifest,
            },
            native_tables: self.native.clone(),
        })
    }

    pub fn native_column(
        &self,
        column: &sdm_contract::ops::ColumnRef,
    ) -> Result<&sdm_contract::frames::materialize::FrameNativeColumn, CoreError> {
        let table = self
            .prepared
            .graph
            .table(&column.table)
            .ok_or_else(|| invalid("native column names an absent graph table"))?;
        self.native
            .iter()
            .find(|native| native.source == table.source)
            .and_then(|native| {
                native
                    .columns
                    .iter()
                    .find(|native| native.name == column.column)
            })
            .ok_or_else(|| invalid("native column lacks its original source dtype"))
    }

    pub fn selected_rows(
        &self,
        table: &str,
        rows: &[usize],
        columns: &[String],
    ) -> Result<RecordBatch, CoreError> {
        lock(&self.frames)
            .snapshot
            .as_ref()
            .ok_or_else(|| internal("bound graph lacks its immutable snapshot"))?
            .selected_rows(table, rows, columns)
    }

    pub fn selected_link_pairs(
        &self,
        edge: &sdm_engine::graph::Edge,
        source_rows: &[usize],
        source_batch: &[usize],
        target_rows: &[usize],
        target_batch: &[usize],
    ) -> Result<Vec<(usize, usize)>, CoreError> {
        lock(&self.frames)
            .snapshot
            .as_ref()
            .ok_or_else(|| internal("bound graph lacks its immutable snapshot"))?
            .selected_link_pairs(edge, source_rows, source_batch, target_rows, target_batch)
    }
}

impl SampledContext {
    /// Export only actual raw sampled columns, ownership, source rows and links.
    /// The payload stage independently rebinds them to their immutable source.
    pub(crate) fn publish(
        self,
        asker: &Asker,
        graph: &BoundGraph,
        plan: sdm_contract::frames::materialize::FrameRequestPlan,
        range: sdm_contract::frames::materialize::FrameBatchRange,
        anchors: ArrayRef,
    ) -> Result<SampledFrameContext, CoreError> {
        self.check_source(&graph.prepared.source)?;
        let mut identity = Identity::new("sampled-context");
        identity.value(&(&plan.identity, range))?;
        let anchor_rows = RecordBatch::try_from_iter([("anchor", anchors)])
            .map_err(|_| internal("frame context anchors do not align"))?;
        identity.arrow(&ArrowData::new(
            anchor_rows.schema(),
            vec![anchor_rows.clone()],
        ))?;
        let mut value = SampledFrameContext {
            version: FRAME_MATERIALIZATION_VERSION,
            plan,
            range,
            identity: String::new(),
            tables: Vec::new(),
            links: Vec::new(),
            anchors: attach_batch(asker, anchor_rows)?,
            text: Vec::new(),
        };
        for table in self.sampled.tables {
            let source_rows = self
                .native_rows
                .get(&table.name)
                .ok_or_else(|| internal("sampled table lacks original row positions"))?;
            let mut columns = table
                .columns
                .into_iter()
                .map(|(name, column)| match column {
                    SampledColumn::Arrow(array) => Ok((name, array)),
                    _ => Err(internal(
                        "raw frame context already contains normalized words",
                    )),
                })
                .collect::<Result<Vec<_>, CoreError>>()?;
            columns.push((BATCH_COLUMN.to_owned(), positional(&table.batch)?));
            let rows = RecordBatch::try_from_iter(columns)
                .map_err(|_| internal("sampled frame columns do not align"))?;
            identity.value(&(&table.name, &table.primary_key, &table.stypes))?;
            identity.arrow(&ArrowData::new(rows.schema(), vec![rows.clone()]))?;
            let source_data =
                RecordBatch::try_from_iter([("source_row", positional(source_rows)?)])
                    .map_err(|_| internal("sampled source positions do not align"))?;
            identity.arrow(&ArrowData::new(
                source_data.schema(),
                vec![source_data.clone()],
            ))?;
            let rows = attach_batch(asker, rows)?;
            let source_rows = attach_batch(asker, source_data)?;
            for (column, stype) in &table.stypes {
                if *stype == Stype::Text {
                    value.text.push(sdm_contract::ops::ColumnRef {
                        table: table.name.clone(),
                        column: column.clone(),
                    });
                }
            }
            value.tables.push(SampledFrameTable {
                name: table.name,
                primary_key: table.primary_key,
                rows,
                source_rows,
                stypes: table.stypes,
            });
        }
        for link in self.sampled.links {
            let pairs = RecordBatch::try_from_iter([
                (
                    "source",
                    positional(
                        &link
                            .pairs
                            .iter()
                            .map(|&(source, _)| source)
                            .collect::<Vec<_>>(),
                    )?,
                ),
                (
                    "target",
                    positional(
                        &link
                            .pairs
                            .iter()
                            .map(|&(_, target)| target)
                            .collect::<Vec<_>>(),
                    )?,
                ),
            ])
            .map_err(|_| internal("sampled link pairs do not align"))?;
            let source = sdm_contract::ops::ColumnRef {
                table: link.src,
                column: link.fkey,
            };
            let target_column = graph
                .prepared
                .graph
                .table(&link.dst)
                .and_then(|table| table.primary_key.clone())
                .ok_or_else(|| internal("sampled relationship target has no primary key"))?;
            let target = sdm_contract::ops::ColumnRef {
                table: link.dst,
                column: target_column,
            };
            identity.value(&(&source, &target))?;
            identity.arrow(&ArrowData::new(pairs.schema(), vec![pairs.clone()]))?;
            value.links.push(SampledFrameLink {
                source,
                target,
                pairs: attach_batch(asker, pairs)?,
            });
        }
        identity.value(&value.text)?;
        value.identity = identity.finish();
        Ok(value)
    }
}
