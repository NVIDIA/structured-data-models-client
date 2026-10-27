// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Actual immutable candidate populations and evaluated label-batch evidence.
//! Native lookups consume only reached ranges. Owners hold useful row data,
//! never a callback, mutable generator, pending action or program position.

use super::*;
use std::sync::Weak;

// LocalSampler._sample_target_set reads this many candidate rows before
// deciding whether the context quota has already been filled.
pub(super) const LABEL_BATCH_ROWS: usize = 10_000;

pub(super) struct PopulationData {
    pub graph: crate::RetainedFrameGraph,
    pub meaning: FrameTaskMeaning,
    pub prediction_type: FramePredictionType,
    pub population: task::ExamplePopulation,
    pub native_index: ArrayRef,
    pub asked: Arc<Asked>,
    pub memo: SamplerMemo,
    pub reader: sdm_engine::time::text::Reader,
    pub context_anchor: i64,
    pub span: Option<(Offset, Offset)>,
    pub earliest: Option<i64>,
    pub context_examples: usize,
    pub trials: usize,
    pub entity_filter: Option<String>,
}

struct OwnedPopulation {
    data: PopulationData,
    keys: ArrowData,
}

#[derive(Clone)]
pub struct RetainedFramePopulation {
    data: Arc<OwnedPopulation>,
}

impl std::fmt::Debug for RetainedFramePopulation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RetainedFramePopulation")
            .field("identity", &self.data.data.meaning.identity)
            .finish_non_exhaustive()
    }
}

fn candidate_range(start: u64, rows: usize) -> Option<FrameBatchRange> {
    let start_ = usize::try_from(start).ok()?;
    if start_ >= rows || start_ % LABEL_BATCH_ROWS != 0 {
        return None;
    }
    Some(FrameBatchRange {
        start,
        stop: rows.min(start_.saturating_add(LABEL_BATCH_ROWS)) as u64,
    })
}

impl RetainedFramePopulation {
    pub(super) fn new(data_: PopulationData) -> Result<Self, CoreError> {
        let keys = data(
            RecordBatch::try_from_iter([
                ("entity", data_.population.keys()?),
                ("index", Arc::clone(&data_.native_index)),
            ])
            .map_err(|_| internal("frame candidate keys do not form one Arrow column"))?,
        );
        Ok(Self {
            data: Arc::new(OwnedPopulation { data: data_, keys }),
        })
    }
    pub fn keys(&self) -> ArrowData {
        self.data.keys.clone()
    }
    fn same_owner(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.data, &other.data)
    }
}

struct LabelBatchData {
    range: FrameBatchRange,
    /// Domain lineage to the actual preceding evaluated rows. A weak link
    /// keeps no earlier row buffer or suspended computation alive.
    previous: Option<Weak<LabelBatchData>>,
    /// Accepted labels in this exact immutable prefix, including these rows.
    accepted_rows: usize,
    complete: bool,
    labelled: task::LabelledContext,
    native_target: Option<FrameNativeColumnRef>,
    native_labels: Option<task::LabelledContext>,
    target_encoding: Option<sdm_contract::data::strings::StringEncoding>,
}

/// Actual evaluated rows for one candidate range, including their original IDs.
#[derive(Clone)]
pub struct RetainedFrameLabelBatch {
    population: RetainedFramePopulation,
    data: Arc<LabelBatchData>,
}

impl LabelBatchData {
    /// Validate genuine ordered label data and cumulative accepted-row facts.
    /// Independent evaluations of the same range are distinct predecessors.
    fn follows(&self, previous: Option<&Arc<Self>>) -> bool {
        let same = match (&self.previous, previous) {
            (None, None) => true,
            (Some(found), Some(expected)) => found.ptr_eq(&Arc::downgrade(expected)),
            _ => false,
        };
        same && self.range.start == previous.map_or(0, |batch| batch.range.stop)
            && !previous.is_some_and(|batch| batch.complete)
            && previous
                .map_or(0, |batch| batch.accepted_rows)
                .checked_add(self.labelled.keys.len())
                == Some(self.accepted_rows)
    }
}

impl RetainedFrameLabelBatch {
    /// Actual list-label rows before entity/assumption masks and NaN-key
    /// exclusion, without another draw, aggregation or source lookup.
    pub fn rows(&self) -> Result<ArrowData, CoreError> {
        let labels = self
            .data
            .native_labels
            .as_ref()
            .ok_or_else(|| protocol("the label batch has no native list result"))?;
        Ok(data(labelled_rows(
            &labels.keys,
            &labels.anchors,
            &labels.labels,
        )?))
    }
}

impl std::fmt::Debug for RetainedFrameLabelBatch {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RetainedFrameLabelBatch")
            .field("range", &self.data.range)
            .finish_non_exhaustive()
    }
}

fn bind_population(
    asker: &Asker,
    value: PreparedFramePopulation,
    attachments: &mut Attachments,
) -> Result<(BoundGraph, RetainedFramePopulation), FrameStageError> {
    let Some(Attachment::FramePopulation(owner)) = attachments.remove(&value.owner) else {
        return Err(
            protocol("frame labels lack their actual immutable candidate population").into(),
        );
    };
    let mut graph = BoundGraph::bind(asker, value.graph, attachments)?;
    graph.release_access();
    let source = &owner.data.data;
    if source.meaning != value.meaning
        || source.prediction_type != value.prediction_type
        || !source.graph.same_owner(&graph.retained())
        || value.rows != source.population.examples.index.len() as u64
        || value.batch_rows != LABEL_BATCH_ROWS as u32
    {
        return Err(protocol(
            "candidate population meaning or graph no longer matches its actual immutable owner",
        )
        .into());
    }
    Ok((graph, owner))
}

fn bind_batch(
    owner: &RetainedFramePopulation,
    value: PreparedFrameLabelBatch,
    attachments: &mut Attachments,
) -> Result<RetainedFrameLabelBatch, FrameStageError> {
    let Some(Attachment::FrameLabelBatch(batch)) = attachments.remove(&value.owner) else {
        return Err(protocol("frame labels lack their actual evaluated candidate batch").into());
    };
    if !batch.population.same_owner(owner)
        || value.population != owner.data.data.meaning.identity
        || value.range != batch.data.range
        || value.complete != batch.data.complete
        || value.native_target != batch.data.native_target
        || value.target_encoding != batch.data.target_encoding
    {
        return Err(protocol(
            "native label evidence does not match its actual population and evaluated rows",
        )
        .into());
    }
    Ok(batch)
}

fn batches(
    owner: &RetainedFramePopulation,
    values: Vec<PreparedFrameLabelBatch>,
    attachments: &mut Attachments,
) -> Result<Vec<RetainedFrameLabelBatch>, FrameStageError> {
    let mut found = Vec::with_capacity(values.len());
    let mut end = 0;
    for value in values {
        let batch = bind_batch(owner, value, attachments)?;
        if candidate_range(end, owner.data.data.population.examples.index.len())
            != Some(batch.data.range)
            || !batch.data.follows(
                found
                    .last()
                    .map(|previous: &RetainedFrameLabelBatch| &previous.data),
            )
        {
            return Err(protocol("native label evidence does not cover a genuine ordered reached prefix of its actual population").into());
        }
        end = batch.data.range.stop;
        found.push(batch);
    }
    Ok(found)
}

fn converted(
    owner: &PopulationData,
    supplied: FrameReferenceValues,
    attachments: &mut Attachments,
    rows: usize,
) -> Result<
    Option<(
        ArrayRef,
        Option<sdm_contract::data::strings::StringEncoding>,
    )>,
    FrameStageError,
> {
    let (keys, encoding) = match (&owner.prediction_type, supplied) {
        (FramePredictionType::CoreKey { .. }, FrameReferenceValues::CoreKeys { values: None }) => {
            return Ok(None);
        }
        (
            FramePredictionType::NativeString { .. },
            FrameReferenceValues::NativeString {
                converted: NativeFrameStrings::Value { values },
            },
        ) => (take_strings(attachments, &values)?, Some(values.encoding)),
        (
            FramePredictionType::NativeString { .. },
            FrameReferenceValues::NativeString {
                converted: NativeFrameStrings::Failed { failure },
            },
        ) => {
            return Err(FrameStageError::Native(Box::new(
                FrameNativeFailureContext {
                    site: NativeSite::LabelReferences,
                    failure,
                    diagnostic: None,
                    dropped_summary: None,
                },
            )));
        }
        (
            FramePredictionType::NativeString { .. },
            FrameReferenceValues::NativeString {
                converted:
                    NativeFrameStrings::CastFailed {
                        failure,
                        expected_dtype,
                        input_repr,
                    },
            },
        ) => {
            return Err(FrameStageError::Native(Box::new(
                FrameNativeFailureContext {
                    site: NativeSite::LabelReferenceCast {
                        expected_dtype,
                        input_repr,
                    },
                    failure,
                    diagnostic: None,
                    dropped_summary: None,
                },
            )));
        }
        _ => {
            return Err(protocol(
                "candidate reference facts do not match their actual native source family",
            )
            .into());
        }
    };
    if keys.len() != rows {
        return Err(protocol(
            "native candidate keys do not cover the actual ordered candidate range",
        )
        .into());
    }
    Ok(Some((keys, encoding)))
}

async fn evaluate(
    asker: &Asker,
    graph: &BoundGraph,
    source: &PopulationData,
    population: task::ExamplePopulation,
    native: Option<(
        ArrayRef,
        Option<sdm_contract::data::strings::StringEncoding>,
    )>,
) -> Result<task::EvaluatedContext, FrameStageError> {
    let entity = graph
        .prepared
        .graph
        .table(&source.meaning.entity.table)
        .ok_or_else(|| protocol("candidate population lost its entity table"))?;
    let native_rows = match native {
        Some((keys, encoding)) => {
            let frames = crate::frame_materialize::lock(&graph.frames);
            let nodes = frames.resolve_encoded_references(entity, &[(&keys, encoding)])?;
            let columns = source
                .population
                .sampled
                .schema()
                .fields()
                .iter()
                .map(|field| field.name().clone())
                .collect::<Vec<_>>();
            Some(frames.source_rows(entity, &nodes, &columns)?)
        }
        None => None,
    };
    let mut memo = Memo {
        asker,
        frames: Some(graph.frames.clone()),
        memo: source.memo.clone(),
    };
    let times = Times::new(source.reader, None, &memo.memo);
    let mut windows = Vec::new();
    source.asked.target.windowed(&mut windows);
    let labelled_fetches = fetches(
        &graph.prepared.graph,
        &source.meaning.entity.table,
        &source.asked.every_windowed(),
    )?;
    let target_fetches = fetches(
        &graph.prepared.graph,
        &source.meaning.entity.table,
        &windows,
    )?;
    let entropy = Entropy::from_hex(source.meaning.seed.magnitude_hex())
        .ok_or_else(|| protocol("candidate population lost its original sampling entropy"))?;
    let labelled = task::evaluate_population(
        asker,
        &mut memo,
        LabelDraw {
            entity,
            key: &source.meaning.entity.column,
            asked: &source.asked,
            read_times: &times,
            labelled_fetches: &labelled_fetches,
            fetches: &target_fetches,
            windows: &windows,
            entropy: &entropy,
            context_anchor: source.context_anchor,
            anchor_entity: source.meaning.entity_anchor,
            span: source.span,
            earliest: source.earliest,
            context_examples: source.context_examples,
            trials: source.trials,
            entity_filter: source.entity_filter.as_deref(),
        },
        population,
        native_rows,
    )
    .await?;
    keep_stored_times(&mut memo, &times);
    Ok(labelled)
}

pub(super) async fn label_batch(
    asker: &Asker,
    value: PreparedFramePopulation,
    range: FrameBatchRange,
    previous: Option<PreparedFrameLabelBatch>,
    converted_: FrameReferenceValues,
    attachments: &mut Attachments,
) -> Result<PreparedFrameLabelBatch, FrameStageError> {
    let (graph, owner) = bind_population(asker, value, attachments)?;
    let previous = previous
        .map(|value| bind_batch(&owner, value, attachments))
        .transpose()?;
    let source = &owner.data.data;
    let expected_start = previous.as_ref().map_or(0, |batch| batch.data.range.stop);
    if range.start != expected_start
        || candidate_range(range.start, source.population.examples.index.len()) != Some(range)
        || previous.as_ref().is_some_and(|batch| batch.data.complete)
    {
        return Err(
            protocol("native candidate range does not follow its actual evaluated prefix").into(),
        );
    }
    let start = range.start as usize;
    let stop = range.stop as usize;
    let native = converted(source, converted_, attachments, stop - start)?;
    let population = source.population.slice(start..stop)?;
    let evaluated = evaluate(asker, &graph, source, population, native).await?;
    let labelled = evaluated.context;
    let gathered = previous
        .as_ref()
        .map_or(0, |batch| batch.data.accepted_rows)
        .checked_add(labelled.keys.len())
        .ok_or_else(|| protocol("evaluated candidate prefix exceeds this host's row capacity"))?;
    let complete = gathered >= source.context_examples;
    let identity = source.meaning.identity.clone();
    let native_target = match &source.asked.target {
        Shape::Aggregated(windowed)
            if windowed.aggregation == labels::Aggregation::ListDistinct =>
        {
            let column = windowed
                .column
                .as_ref()
                .ok_or_else(|| protocol("list labels lack their source column"))?;
            Some(graph.native_binding(&ColumnRef {
                table: windowed.table.clone(),
                column: column.clone(),
            })?)
        }
        _ => None,
    };
    let target_encoding = label_encoding(
        evaluated
            .native_labels
            .as_ref()
            .map_or(&labelled.labels, |labels| &labels.labels),
    );
    let batch = RetainedFrameLabelBatch {
        population: owner,
        data: Arc::new(LabelBatchData {
            range,
            previous: previous.as_ref().map(|batch| Arc::downgrade(&batch.data)),
            accepted_rows: gathered,
            complete,
            labelled,
            native_target: native_target.clone(),
            native_labels: evaluated.native_labels,
            target_encoding,
        }),
    };
    Ok(PreparedFrameLabelBatch {
        population: identity,
        range,
        complete,
        owner: asker.attach(Attachment::FrameLabelBatch(batch))?,
        native_target,
        target_encoding,
    })
}

pub(super) async fn labels(
    asker: &Asker,
    value: PreparedFramePopulation,
    values: Vec<PreparedFrameLabelBatch>,
    attachments: &mut Attachments,
) -> Result<LabelledFrameTask, FrameStageError> {
    let (graph, owner) = bind_population(asker, value, attachments)?;
    let batches = batches(&owner, values, attachments)?;
    let source = &owner.data.data;
    let exhausted = batches.last().map_or(0, |batch| batch.data.range.stop)
        == source.population.examples.index.len() as u64;
    if !exhausted && !batches.last().is_some_and(|batch| batch.data.complete) {
        return Err(protocol(
            "native label evidence is incomplete for this population and context quota",
        )
        .into());
    }
    let labelled = if let Some(first) = batches.first() {
        let arrays = batches
            .iter()
            .map(|batch| batch.data.labelled.keys.as_ref())
            .collect::<Vec<_>>();
        let keys = arrow_select::concat::concat(&arrays)
            .map_err(|_| internal("native label batch keys do not concatenate"))?;
        let mut labels = first.data.labelled.labels.clone();
        for batch in batches.iter().skip(1) {
            labels.extend(&batch.data.labelled.labels)?;
        }
        task::LabelledContext {
            keys,
            anchors: batches
                .iter()
                .flat_map(|batch| batch.data.labelled.anchors.iter().copied())
                .collect(),
            labels,
        }
    } else {
        evaluate(asker, &graph, source, source.population.slice(0..0)?, None)
            .await?
            .context
    };
    let entity = graph
        .prepared
        .graph
        .table(&source.meaning.entity.table)
        .ok_or_else(|| protocol("candidate population lost its entity table"))?;
    let mut memo = Memo {
        asker,
        frames: Some(graph.frames.clone()),
        memo: source.memo.clone(),
    };
    let times = Times::new(source.reader, None, &memo.memo);
    let mut windows = Vec::new();
    source.asked.target.windowed(&mut windows);
    let fetches = fetches(
        &graph.prepared.graph,
        &source.meaning.entity.table,
        &windows,
    )?;
    let entropy = Entropy::from_hex(source.meaning.seed.magnitude_hex())
        .ok_or_else(|| protocol("candidate population lost its original sampling entropy"))?;
    let labelled = task::finish_context(
        asker,
        &mut memo,
        LabelDraw {
            entity,
            key: &source.meaning.entity.column,
            asked: &source.asked,
            read_times: &times,
            labelled_fetches: &fetches,
            fetches: &fetches,
            windows: &windows,
            entropy: &entropy,
            context_anchor: source.context_anchor,
            anchor_entity: source.meaning.entity_anchor,
            span: source.span,
            earliest: source.earliest,
            context_examples: source.context_examples,
            trials: source.trials,
            entity_filter: source.entity_filter.as_deref(),
        },
        labelled,
        source.population.kept.len(),
    )
    .await?;
    keep_stored_times(&mut memo, &times);
    let rows = data(labelled_rows(
        &labelled.keys,
        &labelled.anchors,
        &labelled.labels,
    )?);
    let target_encoding = label_encoding(&labelled.labels);
    let mut meaning = source.meaning.clone();
    meaning.identity = task_identity(&meaning, &source.prediction_type, target_encoding, &rows)?;
    Ok(LabelledFrameTask {
        graph: graph.publish(asker)?,
        meaning,
        context: Some(asker.attach(Attachment::Arrow(rows))?),
        target_encoding,
        prediction_type: source.prediction_type.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::Int64Array;

    #[test]
    fn candidate_ranges_preserve_native_chunk_boundary_and_exhaustion() {
        for (start, stop) in [(0, 10000), (10000, 20000), (20000, 25000)] {
            assert_eq!(
                candidate_range(start, 25000),
                Some(FrameBatchRange { start, stop })
            );
        }
        assert_eq!(
            candidate_range(0, 10000),
            Some(FrameBatchRange {
                start: 0,
                stop: 10000
            })
        );
        for start in [1, 9999, 10001, 25000, 30000, u64::MAX] {
            assert!(candidate_range(start, 25000).is_none());
        }
        assert!(candidate_range(0, 0).is_none());
    }

    fn evaluated(
        previous: Option<&Arc<LabelBatchData>>,
        start: u64,
        rows: Vec<i64>,
    ) -> Arc<LabelBatchData> {
        let accepted_rows = previous.map_or(0, |batch| batch.accepted_rows) + rows.len();
        let anchors = vec![0; rows.len()];
        Arc::new(LabelBatchData {
            range: FrameBatchRange {
                start,
                stop: start + 10000,
            },
            previous: previous.map(Arc::downgrade),
            accepted_rows,
            complete: false,
            labelled: task::LabelledContext {
                keys: Arc::new(Int64Array::from(rows.clone())),
                anchors,
                labels: labels::Labels::Int(rows),
            },
            native_target: None,
            native_labels: None,
            target_encoding: None,
        })
    }

    #[test]
    fn final_prefix_uses_genuine_owners_and_checks_cumulative_accepted_rows() {
        let first = evaluated(None, 0, vec![1]);
        let second = evaluated(Some(&first), 10000, vec![2, 3]);
        let third = evaluated(Some(&second), 20000, vec![4]);
        assert!(first.follows(None));
        assert!(second.follows(Some(&first)));
        assert!(third.follows(Some(&second)));
        assert_eq!(third.accepted_rows, 4);
        let same_owner = Arc::clone(&first);
        assert!(second.follows(Some(&same_owner)));
        let independent_first = evaluated(None, 0, vec![1]);
        assert!(!second.follows(Some(&independent_first)));
        assert!(!third.follows(Some(&first)));
        assert!(!second.follows(None));
        assert!(!first.follows(Some(&second)));
        let mut incorrect_count = evaluated(Some(&second), 20000, vec![4]);
        Arc::get_mut(&mut incorrect_count).unwrap().accepted_rows = 1;
        assert!(!incorrect_count.follows(Some(&second)));
        let mut completed = evaluated(None, 0, vec![1]);
        Arc::get_mut(&mut completed).unwrap().complete = true;
        assert!(!evaluated(Some(&completed), 10000, vec![2]).follows(Some(&completed)));
    }

    #[test]
    fn domain_lineage_keeps_no_earlier_label_buffer_alive() {
        let first = evaluated(None, 0, vec![1]);
        let keys = Arc::downgrade(&first.labelled.keys);
        let second = evaluated(Some(&first), 10000, vec![2]);
        drop(first);
        assert!(keys.upgrade().is_none());
        assert!(second.previous.as_ref().unwrap().upgrade().is_none());
        assert_eq!(second.accepted_rows, 2);
        assert_eq!(second.labelled.labels.len(), 1);
    }
}
