// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Actual immutable candidate populations and evaluated label-batch evidence.
//! Native lookups consume only reached ranges. Owners hold useful row data,
//! never a callback, mutable generator, pending action or program position.

use super::*;

// LocalSampler._sample_target_set reads this many candidate rows before
// deciding whether the context quota has already been filled.
const LABEL_BATCH_ROWS: usize = 10_000;

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

pub(super) fn ranges(rows: usize) -> Vec<FrameBatchRange> {
    (0..rows)
        .step_by(LABEL_BATCH_ROWS)
        .map(|start| FrameBatchRange {
            start: start as u64,
            stop: rows.min(start + LABEL_BATCH_ROWS) as u64,
        })
        .collect()
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
    population: RetainedFramePopulation,
    range: FrameBatchRange,
    complete: bool,
    labelled: task::LabelledContext,
}

/// Actual evaluated rows for one candidate range, including their original IDs.
#[derive(Clone)]
pub struct RetainedFrameLabelBatch {
    data: Arc<LabelBatchData>,
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
        || value.batches != ranges(source.population.examples.index.len())
    {
        return Err(protocol(
            "candidate population meaning or graph no longer matches its actual immutable owner",
        )
        .into());
    }
    Ok((graph, owner))
}

fn batches(
    owner: &RetainedFramePopulation,
    values: Vec<PreparedFrameLabelBatch>,
    attachments: &mut Attachments,
) -> Result<Vec<RetainedFrameLabelBatch>, FrameStageError> {
    let mut found = Vec::with_capacity(values.len());
    let mut end = 0;
    for value in values {
        let Some(Attachment::FrameLabelBatch(batch)) = attachments.remove(&value.owner) else {
            return Err(
                protocol("frame labels lack their actual evaluated candidate batch").into(),
            );
        };
        if !batch.data.population.same_owner(owner)
            || value.population != owner.data.data.meaning.identity
            || value.range != batch.data.range
            || value.complete != batch.data.complete
            || value.range.start != end
            || found
                .last()
                .is_some_and(|previous: &RetainedFrameLabelBatch| previous.data.complete)
        {
            return Err(protocol("native label evidence does not cover an ordered reached prefix of its actual population").into());
        }
        end = value.range.stop;
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
            return Err(FrameStageError::Native {
                site: NativeSite::LabelReferences,
                failure,
                diagnostic: None,
                dropped_summary: None,
            });
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
            return Err(FrameStageError::Native {
                site: NativeSite::LabelReferenceCast {
                    expected_dtype,
                    input_repr,
                },
                failure,
                diagnostic: None,
                dropped_summary: None,
            });
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
) -> Result<task::LabelledContext, FrameStageError> {
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
    previous: Vec<PreparedFrameLabelBatch>,
    converted_: FrameReferenceValues,
    attachments: &mut Attachments,
) -> Result<PreparedFrameLabelBatch, FrameStageError> {
    let (graph, owner) = bind_population(asker, value, attachments)?;
    let previous = batches(&owner, previous, attachments)?;
    let source = &owner.data.data;
    let expected_start = previous.last().map_or(0, |batch| batch.data.range.stop);
    if range.start != expected_start
        || !ranges(source.population.examples.index.len()).contains(&range)
        || previous.last().is_some_and(|batch| batch.data.complete)
    {
        return Err(
            protocol("native candidate range does not follow its actual evaluated prefix").into(),
        );
    }
    let start = range.start as usize;
    let stop = range.stop as usize;
    let native = converted(source, converted_, attachments, stop - start)?;
    let mut population = source.population.clone();
    population.examples.index = population.examples.index[start..stop].to_vec();
    population.examples.anchors = population.examples.anchors[start..stop].to_vec();
    let labelled = evaluate(asker, &graph, source, population, native).await?;
    let gathered = previous
        .iter()
        .map(|batch| batch.data.labelled.keys.len())
        .sum::<usize>()
        .saturating_add(labelled.keys.len());
    let complete = gathered >= source.context_examples;
    let identity = source.meaning.identity.clone();
    let batch = RetainedFrameLabelBatch {
        data: Arc::new(LabelBatchData {
            population: owner,
            range,
            complete,
            labelled,
        }),
    };
    Ok(PreparedFrameLabelBatch {
        population: identity,
        range,
        complete,
        owner: asker.attach(Attachment::FrameLabelBatch(batch))?,
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
        evaluate(asker, &graph, source, source.population.clone(), None).await?
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

    #[test]
    fn candidate_ranges_preserve_native_chunk_boundary_and_exhaustion() {
        assert_eq!(
            ranges(25000),
            vec![
                FrameBatchRange {
                    start: 0,
                    stop: 10000
                },
                FrameBatchRange {
                    start: 10000,
                    stop: 20000
                },
                FrameBatchRange {
                    start: 20000,
                    stop: 25000
                }
            ]
        );
        assert_eq!(
            ranges(10000),
            vec![FrameBatchRange {
                start: 0,
                stop: 10000
            }]
        );
        assert!(ranges(0).is_empty());
    }
}
