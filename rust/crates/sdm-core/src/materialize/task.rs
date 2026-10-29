// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared typed entity/label draws, context narrowing and ordered task ranges.
//! These are useful task data and plans, with no native input callback or
//! request wire schema. Provider I/O remains the existing typed DataRequest.

use std::collections::BTreeSet;
use std::sync::Arc;

use arrow_array::{
    ArrayRef, BooleanArray, Float32Array, Int64Array, RecordBatch, TimestampNanosecondArray,
};
use arrow_schema::{DataType, Field, Schema, TimeUnit};
use sdm_contract::data::{DataRequest, Filter};
use sdm_contract::ops::{SamplerMemo, Seed};
use sdm_contract::{CoreError, CoreWarning, Detail, Scalar, Sentence};
use sdm_engine::graph::Table;
use sdm_engine::labels;
use sdm_engine::random::{Entropy, default_rng};
use sdm_engine::sample;
use sdm_engine::time::offset::{Offset, timeframe};
use sdm_request::options::Kind as Task;

use super::{
    Asked, Fetch, Labelling, MOST_CALLS, Memo, TIME_MAX, Times, Windowed, count, empty_sample,
    invalid, named, nan_rows, null_compared, one_batch, rows_of, take, times_ns,
};
use crate::run::{Asker, internal};

/// Typed rows obtained by the existing entity draw, before labels or any
/// caller-supplied prediction entity is converted.
pub(super) struct EntitySample {
    source_read: Option<sdm_contract::run::ReplyId>,
    sampled: RecordBatch,
    pub(super) entity_key: ArrayRef,
    entity_time: Option<Vec<Option<i64>>>,
    kept: Vec<usize>,
}

pub(super) struct EntityDraw<'a> {
    pub(super) entity: &'a Table,
    pub(super) key: &'a str,
    pub(super) asked: &'a Asked,
    pub(super) read_times: &'a Times,
    pub(super) memo: &'a SamplerMemo,
    pub(super) frames: Option<&'a crate::frame_materialize::Context>,
    pub(super) trials: usize,
    pub(super) seed: &'a Seed,
    pub(super) seed_value: Scalar,
    pub(super) unseeded: bool,
}

pub(super) async fn entity_sample(
    asker: &Asker,
    draw: EntityDraw<'_>,
) -> Result<EntitySample, CoreError> {
    let EntityDraw {
        entity,
        key,
        asked,
        read_times,
        memo,
        frames,
        trials,
        seed,
        seed_value,
        unseeded,
    } = draw;
    let mut entity_columns = BTreeSet::from([key.to_owned()]);
    if let Some(time) = &entity.time_column {
        entity_columns.insert(time.clone());
    }
    asked.target.entity_columns(&mut entity_columns);
    for shape in &asked.choosing {
        shape.entity_columns(&mut entity_columns);
    }
    let mut filters = Vec::new();
    if entity.column(key).is_none_or(|c| c.nullable) {
        filters.push(Filter::NotNull {
            column: key.to_owned(),
        });
    }
    if let Some(time) = &entity.time_column
        && entity.column(time).is_none_or(|c| c.nullable)
    {
        filters.push(Filter::NotNull {
            column: time.clone(),
        });
    }
    // Seeds are optional. An unseeded provider documents its limitation
    // without issuing a routine warning for the default seed (P3D-13).
    let mut source_read = None;
    let sampled = if let Some(frames) = frames {
        let entity_seed = seed.to_u128().unwrap_or(0);
        // The OS fact also seeds RS-1, whose entropy domain is wider.
        // RS-4's legacy MT seed uses its own 32-bit domain; caller-given
        // seeds retain the native range check when a draw is required.
        let entity_seed = if unseeded {
            entity_seed & u128::from(u32::MAX)
        } else {
            entity_seed
        };
        crate::frame_materialize::lock(frames).entities(
            entity,
            &entity.ordered(&entity_columns),
            trials,
            entity_seed,
        )?
    } else {
        let request = DataRequest::SampleEntities {
            table: entity.source.clone(),
            key: vec![key.to_owned()],
            n: trials as u64,
            seed: seed_value,
            columns: entity.ordered(&entity_columns),
            filters,
            // The schema's type of the key, for a provider that
            // orders the draw by it (SQLite's, which reads the
            // type the key's declaration fixes first).
            key_dtype: asker
                .samples_by_key_dtype()
                .then(|| entity.column(key).map(|c| c.dtype))
                .flatten(),
            // The table's rows, for a provider sizing its draw by
            // them (the reference's sampled percentage).
            rows: if asker.capabilities()?.sample_rows {
                rows_of(memo, &entity.name)
            } else {
                None
            },
        };
        let name = request.name();
        let pending = asker.fetch(request)?;
        let observed = pending.source_read();
        let raw = crate::describe::data(pending.await, name)?;
        source_read = observed.and_then(|read| read.first());
        read_times.read(asker, entity, one_batch(&raw)?).await?
    };
    if sampled.num_rows() == 0 {
        return Err(invalid(
            Sentence::EntityTableEmpty,
            "the entity table holds no row to sample",
            "query",
        )
        .with_field("table", Detail::text(&entity.name)));
    }
    // One entity per key value: `1.0` and `1.00` of a column of numbers
    // handed over as their text are one key (`subgraph::key_values`).
    let sampled_keys = match frames {
        Some(frames) => crate::frame_materialize::lock(frames).source_keys(
            entity,
            key,
            named(&sampled, key)?,
        )?,
        None => crate::subgraph::key_values(entity, key, named(&sampled, key)?)?,
    };
    let mut seen = BTreeSet::new();
    let kept: Vec<usize> = (0..sampled_keys.len())
        .filter(|&row| seen.insert(sampled_keys[row].clone()))
        .collect();
    let entity_key = take(named(&sampled, key)?, &kept)?;
    let entity_time = match &entity.time_column {
        Some(time) => {
            let column = named(&sampled, time)?;
            // A time of day holds no instant: the reference's engine reads
            // each as none (`to_datetime`, see `times_of_day_unread`), so no
            // entity is valid at any anchor and a temporal query collects no
            // context example, the caller's error below (T1.36 fix 9). A
            // static query never gets here with one: its time bounds read
            // every time column first and fail on it (register DT-12).
            let all = if matches!(
                column.data_type(),
                DataType::Time32(_) | DataType::Time64(_)
            ) {
                vec![None; column.len()]
            } else {
                times_ns(column, time)?
            };
            Some(kept.iter().map(|&r| all[r]).collect::<Vec<_>>())
        }
        None => None,
    };
    Ok(EntitySample {
        source_read,
        sampled,
        entity_key,
        entity_time,
        kept,
    })
}

/// Complete labeled context before the native prediction-entity constructor
/// and later client narrowing. It contains real task data, not a continuation.
#[derive(Clone)]
pub(super) struct LabelledContext {
    pub(super) keys: ArrayRef,
    pub(super) anchors: Vec<i64>,
    pub(super) labels: labels::Labels,
}

/// Evaluated labels and their original pre-mask native list-result population.
pub(super) struct EvaluatedContext {
    pub(super) context: LabelledContext,
    pub(super) native_labels: Option<LabelledContext>,
}

impl LabelledContext {
    pub(super) fn narrow(mut self, asked: Option<usize>) -> Self {
        let rows = asked.map_or(self.keys.len(), |count| self.keys.len().min(count));
        self.keys = self.keys.slice(0, rows);
        self.anchors.truncate(rows);
        self.labels = self.labels.truncated(rows);
        self
    }
}

/// Immutable context and prediction rows of a complete task. Native owners
/// are retained as Arrow arrays; each batch exposes its actual ordered range.
pub(super) struct TaskRows {
    context_keys: ArrayRef,
    context_anchors: Vec<i64>,
    predicted: ArrayRef,
    prediction_anchor: i64,
}

pub(super) struct TaskBatch {
    pub(super) range: std::ops::Range<usize>,
    pub(super) keys: ArrayRef,
    pub(super) anchors: Vec<i64>,
    pub(super) predicted: ArrayRef,
}

impl TaskRows {
    pub(super) fn new(
        context_keys: ArrayRef,
        context_anchors: Vec<i64>,
        predicted: ArrayRef,
        prediction_anchor: i64,
    ) -> Result<Self, CoreError> {
        if context_keys.len() != context_anchors.len() {
            return Err(CoreError::protocol(
                Sentence::InvalidMessage,
                "task context anchors do not align with its entity rows",
            ));
        }
        Ok(Self {
            context_keys,
            context_anchors,
            predicted,
            prediction_anchor,
        })
    }

    pub(super) fn batch(&self, range: std::ops::Range<usize>) -> Result<TaskBatch, CoreError> {
        if range.start > range.end || range.end > self.predicted.len() {
            return Err(CoreError::protocol(
                Sentence::InvalidMessage,
                "a task batch names an invalid prediction range",
            ));
        }
        let predicted = self.predicted.slice(range.start, range.len());
        let keys = arrow_select::concat::concat(&[self.context_keys.as_ref(), predicted.as_ref()])
            .map_err(|_| internal("the examples' keys do not concatenate"))?;
        let mut anchors = self.context_anchors.clone();
        anchors.extend(std::iter::repeat_n(self.prediction_anchor, predicted.len()));
        Ok(TaskBatch {
            range,
            keys,
            anchors,
            predicted,
        })
    }
}

/// Actual ordered prediction ranges of a complete task. Hosts can retain
/// these useful task facts without choosing rows or applying request policy.
pub(super) struct BatchLayout {
    rows: usize,
    per_call: usize,
    pub(super) calls: usize,
}

impl BatchLayout {
    pub(super) fn new(
        kind: Task,
        asked: Option<u64>,
        rows: usize,
        limit: Option<u32>,
    ) -> Result<Self, CoreError> {
        let per_call = usize::try_from(kind.rows_one_call_scores(asked))
            .map_err(|_| internal("a call's rows past usize"))?;
        let calls = rows.div_ceil(per_call);
        let most = limit.unwrap_or(MOST_CALLS);
        if calls > most as usize {
            return Err(CoreError::new(
                Sentence::MostCalls,
                "the entities asked for are gathered as more calls than one materialize holds at once",
            )
            .with_field("field", Detail::text("indices"))
            .with_field("entities", count(rows))
            .with_field("calls", count(calls))
            .with_field("rows_per_call", count(per_call))
            .with_field("limit", Detail::Scalar(Scalar::UInt32(most))));
        }
        Ok(Self {
            rows,
            per_call,
            calls,
        })
    }

    pub(super) fn largest(&self) -> usize {
        self.per_call.min(self.rows)
    }

    pub(super) fn ranges(&self) -> impl Iterator<Item = std::ops::Range<usize>> + '_ {
        (0..self.rows)
            .step_by(self.per_call)
            .map(|start| start..start + self.per_call.min(self.rows - start))
    }
}

#[derive(Clone, Copy)]
pub(super) struct LabelDraw<'a> {
    pub(super) entity: &'a Table,
    pub(super) key: &'a str,
    pub(super) asked: &'a Asked,
    pub(super) read_times: &'a Times,
    pub(super) labelled_fetches: &'a [Fetch<'a>],
    pub(super) fetches: &'a [Fetch<'a>],
    pub(super) windows: &'a [&'a Windowed],
    pub(super) entropy: &'a Entropy,
    pub(super) context_anchor: i64,
    pub(super) anchor_entity: bool,
    pub(super) span: Option<(Offset, Offset)>,
    pub(super) earliest: Option<i64>,
    pub(super) context_examples: usize,
    pub(super) trials: usize,
    pub(super) entity_filter: Option<&'a str>,
}

/// Complete candidate rows and anchors before their target is read.
/// This is independently useful sampled task data, without a callback or
/// suspended execution state.
#[derive(Clone)]
pub(super) struct ExamplePopulation {
    pub(super) source_read: Option<sdm_contract::run::ReplyId>,
    pub(super) sampled: RecordBatch,
    pub(super) entity_key: ArrayRef,
    pub(super) kept: Arc<[usize]>,
    // Pure facts of the retained immutable Arrow keys, read once without
    // native conversion; reached label reads still enforce missing-key policy.
    nan_key: Arc<BTreeSet<usize>>,
    pub(super) examples: sample::Examples,
}

impl ExamplePopulation {
    pub(super) fn keys(&self) -> Result<ArrayRef, CoreError> {
        take(&self.entity_key, &self.examples.index)
    }

    /// Share the retained source and clone only this reached candidate range.
    /// The complete population deliberately has no Clone implementation.
    pub(super) fn slice(&self, range: std::ops::Range<usize>) -> Result<Self, CoreError> {
        let index = self
            .examples
            .index
            .get(range.clone())
            .ok_or_else(|| internal("candidate index range exceeds its population"))?;
        let anchors = self
            .examples
            .anchors
            .get(range)
            .ok_or_else(|| internal("candidate anchors do not cover their index range"))?;
        Ok(Self {
            source_read: self.source_read,
            sampled: self.sampled.clone(),
            entity_key: Arc::clone(&self.entity_key),
            kept: Arc::clone(&self.kept),
            nan_key: Arc::clone(&self.nan_key),
            examples: sample::Examples {
                index: index.to_vec(),
                anchors: anchors.to_vec(),
            },
        })
    }
}

pub(super) fn example_population(
    draw: &LabelDraw<'_>,
    sampled: EntitySample,
) -> Result<ExamplePopulation, CoreError> {
    let mut rng = default_rng(draw.entropy);
    let examples = match &draw.span {
        Some((start, end)) => sample::temporal_examples(
            &mut rng,
            sampled.kept.len(),
            sampled.entity_time.as_deref(),
            draw.context_anchor,
            timeframe(*start, *end, "anchor_time")?,
            draw.earliest.unwrap_or(TIME_MAX),
            draw.trials,
        )?,
        None if draw.anchor_entity => sample::entity_examples(
            &mut rng,
            sampled.kept.len(),
            sampled.entity_time.as_deref().unwrap_or_default(),
            sample::MISSING_ANCHOR,
        ),
        None => sample::static_examples(
            &mut rng,
            sampled.kept.len(),
            sampled.entity_time.as_deref(),
            draw.context_anchor,
        ),
    };
    Ok(ExamplePopulation {
        source_read: sampled.source_read,
        sampled: sampled.sampled,
        nan_key: Arc::new(nan_rows(&sampled.entity_key)),
        entity_key: sampled.entity_key,
        kept: sampled.kept.into(),
        examples,
    })
}

pub(super) async fn label_population(
    asker: &Asker,
    memo: &mut Memo<'_>,
    draw: LabelDraw<'_>,
    population: ExamplePopulation,
    native_rows: Option<RecordBatch>,
) -> Result<LabelledContext, CoreError> {
    let kept = population.kept.len();
    let labelled = evaluate_population(asker, memo, draw, population, native_rows)
        .await?
        .context;
    finish_context(asker, memo, draw, labelled, kept).await
}

/// The actual reached source rows and earlier entity eligibility.
/// This is shared immutable task data, not a suspended label evaluation.
pub(super) struct LabelInputs {
    pub(super) population: ExamplePopulation,
    pub(super) target: super::Shape,
    pub(super) native_rows: Option<RecordBatch>,
    pub(super) entity_rows: Vec<usize>,
    pub(super) entity_positions: Vec<usize>,
    pub(super) strings:
        std::collections::BTreeMap<String, sdm_contract::data::strings::StringEncoding>,
    pub(super) fetched: std::collections::BTreeMap<String, super::Fetched>,
    entity_selection: Option<Vec<bool>>,
}

impl LabelInputs {
    fn labelling<'a>(&'a self, asker: &'a Asker, times: &'a Times, key: &str) -> Labelling<'a> {
        Labelling {
            asker,
            times,
            sampled: self
                .native_rows
                .as_ref()
                .unwrap_or(&self.population.sampled),
            entity_rows: &self.entity_rows,
            entity_key: &self.population.entity_key,
            native_keys: self
                .native_rows
                .as_ref()
                .and_then(|rows| rows.column_by_name(key)),
            examples: &self.population.examples,
            strings: self.strings.clone(),
        }
    }
}

/// Actual target groups retain the same reached input owner for later masks.
pub(super) struct EvaluatedLabelGroups {
    pub(super) inputs: Arc<LabelInputs>,
    pub(super) entity_selection: Option<Vec<bool>>,
    labelled: labels::Labelled,
    scalar_selection: Option<Vec<bool>>,
}

impl EvaluatedLabelGroups {
    pub(super) fn native_labels(&self) -> Result<Option<LabelledContext>, CoreError> {
        if !matches!(self.labelled.labels, labels::Labels::Lists(_)) {
            return Ok(None);
        }
        self.context_for(&self.labelled).map(Some)
    }

    /// The same population binds a separately grouped actual native operand.
    pub(super) fn projected_native_labels(
        &self,
        target: labels::Target,
    ) -> Result<LabelledContext, CoreError> {
        let labelled = labels::label(&target)?;
        if !matches!(labelled.labels, labels::Labels::Lists(_)) {
            return Err(CoreError::protocol(
                Sentence::InvalidMessage,
                "native group projection is not a complete list target",
            ));
        }
        self.context_for(&labelled)
    }

    fn context_for(&self, labelled: &labels::Labelled) -> Result<LabelledContext, CoreError> {
        if labelled.mask.len() != self.inputs.population.examples.index.len()
            || labelled.mask.iter().filter(|&&v| v).count() != labelled.labels.len()
        {
            return Err(CoreError::protocol(
                Sentence::InvalidMessage,
                "group mask does not cover its actual complete population",
            ));
        }
        let selected = labelled
            .mask
            .iter()
            .enumerate()
            .filter_map(|(row, &kept)| kept.then_some(row))
            .collect::<Vec<_>>();
        Ok(LabelledContext {
            keys: take(
                &self.inputs.population.entity_key,
                &selected
                    .iter()
                    .map(|&row| self.inputs.population.examples.index[row])
                    .collect::<Vec<_>>(),
            )?,
            anchors: selected
                .iter()
                .map(|&row| self.inputs.population.examples.anchors[row])
                .collect(),
            labels: labelled.labels.clone(),
        })
    }
}

/// Replace complete native groups without comparing their values to the
/// pre-restoration mathematical result. Source family admission is the
/// typed attachment reader's responsibility; count and list shape belong here.
fn restore_group_labels(
    mut before: labels::Labelled,
    restored: labels::Labels,
) -> Result<labels::Labelled, CoreError> {
    if !matches!(before.labels, labels::Labels::Lists(_))
        || !matches!(restored, labels::Labels::Lists(_))
        || before.labels.len() != restored.len()
        || before.mask.iter().filter(|&&kept| kept).count() != restored.len()
    {
        return Err(CoreError::protocol(
            sdm_contract::Sentence::InvalidMessage,
            "native list results do not cover their complete evaluated groups",
        ));
    }
    before.labels = restored;
    Ok(before)
}

fn select_labelled(
    mut labelled: labels::Labelled,
    selected: &[bool],
) -> Result<labels::Labelled, CoreError> {
    if labelled.mask.len() != selected.len()
        || labelled.mask.iter().filter(|&&kept| kept).count() != labelled.labels.len()
    {
        return Err(CoreError::protocol(
            sdm_contract::Sentence::InvalidMessage,
            "label selection does not cover its complete candidate population",
        ));
    }
    let keep = labelled
        .mask
        .iter()
        .zip(selected)
        .filter_map(|(&present, &selected)| present.then_some(selected))
        .collect::<Vec<_>>();
    for (present, selected) in labelled.mask.iter_mut().zip(selected) {
        *present &= *selected;
    }
    labelled.labels = labelled.labels.kept(&keep);
    Ok(labelled)
}

pub(super) async fn read_label_inputs(
    asker: &Asker,
    memo: &mut Memo<'_>,
    draw: LabelDraw<'_>,
    population: ExamplePopulation,
    native_rows: Option<RecordBatch>,
    observed_conditions: bool,
) -> Result<Arc<LabelInputs>, CoreError> {
    let LabelDraw {
        entity,
        key,
        asked,
        read_times,
        labelled_fetches,
        fetches: _,
        windows: _,
        entropy: _,
        context_anchor: _,
        anchor_entity: _,
        span: _,
        earliest: _,
        context_examples: _,
        trials: _,
        entity_filter,
    } = draw;
    let ExamplePopulation {
        sampled,
        entity_key,
        kept,
        nan_key,
        examples,
        source_read: _,
    } = &population;
    // A NaN key a source hands over, which the reference reads as missing
    // where it reads floats into NumPy-backed pandas (Postgres's rows), is
    // a missing key its sampler cannot coerce (`_coerce_id`'s `int` or
    // `float` of `NA`) where an example's key seeds the read of a windowed
    // target's rows (`_by_time`): refused as the caller's error before that
    // read. A static target reads no rows by the key, and such an example
    // is left out once labelled, below (T2.12).
    if asker.reads_nan_as_missing()
        && !labelled_fetches.is_empty()
        && examples.index.iter().any(|i| nan_key.contains(i))
    {
        return Err(invalid(
            Sentence::EntityKeyMissing,
            "an entity sampled has no value for its table's primary key",
            "query",
        )
        .with_field("table", Detail::text(&entity.name))
        .with_field("column", Detail::text(key)));
    }
    // Their labels, as the reference's executor computes them over the
    // rows read for them (decision E19).
    let entity_rows: Vec<usize> = match &native_rows {
        Some(rows) => (0..rows.num_rows()).collect(),
        None => examples.index.iter().map(|&i| kept[i]).collect(),
    };
    let sampled = native_rows.as_ref().unwrap_or(sampled);
    let labelling = Labelling {
        asker,
        times: read_times,
        sampled,
        entity_rows: &entity_rows,
        entity_key,
        native_keys: native_rows
            .as_ref()
            .and_then(|rows| rows.column_by_name(key)),
        examples,
        strings: match &memo.frames {
            Some(frames) => {
                crate::frame_materialize::lock(frames).string_encodings(&entity.name)?
            }
            None => Default::default(),
        },
    };
    let fetched = match &memo.frames {
        Some(frames) => super::frame_values::label_rows(&labelling, frames, labelled_fetches)?,
        None => labelling.read(labelled_fetches).await?,
    };
    // The entity condition is observed first, but execute() applies its
    // mask only after target restoration and what-if evaluation. A group
    // excluded by that condition still reaches the native converter.
    let entity_filtered = asked.task == Task::TemporalLink && entity_filter.is_some();
    let eligibility = if entity_filtered && !observed_conditions {
        let filter = asked
            .choosing
            .first()
            .ok_or_else(|| super::internal("an entity filter lost its validated condition"))?;
        Some(labels::truth(&labelling.build(filter, &fetched, false)?)?.values)
    } else {
        None
    };
    let strings = labelling.strings;
    Ok(Arc::new(LabelInputs {
        population,
        target: asked.target.clone(),
        native_rows,
        entity_positions: entity_rows.clone(),
        entity_rows,
        strings,
        fetched,
        entity_selection: eligibility,
    }))
}

pub(super) async fn evaluate_label_groups(
    asker: &Asker,
    memo: &mut Memo<'_>,
    draw: LabelDraw<'_>,
    population: ExamplePopulation,
    native_rows: Option<RecordBatch>,
) -> Result<EvaluatedLabelGroups, CoreError> {
    let inputs = read_label_inputs(asker, memo, draw, population, native_rows, false).await?;
    evaluate_input_groups(asker, draw, inputs)
}

pub(super) fn evaluate_input_groups(
    asker: &Asker,
    draw: LabelDraw<'_>,
    inputs: Arc<LabelInputs>,
) -> Result<EvaluatedLabelGroups, CoreError> {
    let labelling = inputs.labelling(asker, draw.read_times, draw.key);
    let target = labelling.build(&draw.asked.target, &inputs.fetched, true)?;
    evaluate_target_groups(asker, draw, inputs, target)
}

/// The same label/mask kernel consumes either the raw mathematical target or
/// the actual completed native target values. It performs no native cast.
pub(super) fn evaluate_target_groups(
    asker: &Asker,
    draw: LabelDraw<'_>,
    inputs: Arc<LabelInputs>,
    target: labels::Target,
) -> Result<EvaluatedLabelGroups, CoreError> {
    let asked = draw.asked;
    let labelling = inputs.labelling(asker, draw.read_times, draw.key);
    let fetched = &inputs.fetched;
    // Scalar operations keep their existing condition-before-label admission
    // order. Only actual list groups cross a native completion boundary.
    let mut scalar_selection: Option<Vec<bool>> = None;
    if asked.task != Task::TemporalLink {
        for shape in &asked.choosing {
            let selected = labels::truth(&labelling.build(shape, fetched, false)?)?.values;
            scalar_selection = Some(match scalar_selection {
                None => selected,
                Some(before) => before
                    .iter()
                    .zip(&selected)
                    .map(|(&a, &b)| a && b)
                    .collect(),
            });
        }
    }
    let labelled = if matches!(asked.task, Task::Multiclass) {
        labels::label_classes(&target)?
    } else {
        labels::label(&target)?
    };
    null_compared(&asked.target, &labelled.mask)?;
    Ok(EvaluatedLabelGroups {
        entity_selection: inputs.entity_selection.clone(),
        inputs,
        labelled,
        scalar_selection,
    })
}

pub(super) fn complete_label_groups(
    asker: &Asker,
    draw: LabelDraw<'_>,
    groups: &EvaluatedLabelGroups,
    restored: Option<labels::Labels>,
) -> Result<LabelledContext, CoreError> {
    complete_groups_selection(asker, draw, groups, restored, None)
}

pub(super) fn complete_observed_label_groups(
    asker: &Asker,
    draw: LabelDraw<'_>,
    groups: &EvaluatedLabelGroups,
    restored: labels::Labels,
    what_if: Option<&[bool]>,
) -> Result<LabelledContext, CoreError> {
    complete_groups_selection(asker, draw, groups, Some(restored), Some(what_if))
}

fn complete_groups_selection(
    asker: &Asker,
    draw: LabelDraw<'_>,
    groups: &EvaluatedLabelGroups,
    restored: Option<labels::Labels>,
    observed: Option<Option<&[bool]>>,
) -> Result<LabelledContext, CoreError> {
    let EvaluatedLabelGroups {
        inputs,
        labelled,
        scalar_selection,
        entity_selection,
    } = groups;
    let population = &inputs.population;
    let fetched = &inputs.fetched;
    let mut labelled = labelled.clone();
    if let Some(restored) = restored {
        labelled = restore_group_labels(labelled, restored)?;
    }
    let labelling = inputs.labelling(asker, draw.read_times, draw.key);
    // What-if conditions consume the retained rows only after a successful
    // native restoration. No source or later candidate slice is reread.
    let selection = if draw.asked.task == Task::TemporalLink {
        let mut selection = entity_selection.clone();
        if let Some(what_if) = observed {
            if let Some(next) = what_if {
                selection = Some(match selection {
                    None => next.to_vec(),
                    Some(before) => before.iter().zip(next).map(|(&a, &b)| a && b).collect(),
                });
            }
        } else {
            for shape in draw
                .asked
                .choosing
                .iter()
                .skip(usize::from(entity_selection.is_some()))
            {
                let next = labels::truth(&labelling.build(shape, fetched, false)?)?.values;
                selection = Some(match selection {
                    None => next,
                    Some(before) => before.iter().zip(&next).map(|(&a, &b)| a && b).collect(),
                });
            }
        }
        selection
    } else {
        scalar_selection.clone()
    };
    if let Some(selected) = selection {
        labelled = select_labelled(labelled, &selected)?;
    }
    if !population.nan_key.is_empty() {
        let selected = population
            .examples
            .index
            .iter()
            .map(|row| !population.nan_key.contains(row))
            .collect::<Vec<_>>();
        labelled = select_labelled(labelled, &selected)?;
    }
    let selected = labelled
        .mask
        .iter()
        .enumerate()
        .filter_map(|(row, &kept)| kept.then_some(row))
        .collect::<Vec<_>>();
    let rows = selected
        .iter()
        .map(|&row| population.examples.index[row])
        .collect::<Vec<_>>();
    Ok(LabelledContext {
        keys: take(&population.entity_key, &rows)?,
        anchors: selected
            .iter()
            .map(|&row| population.examples.anchors[row])
            .collect(),
        labels: labelled.labels,
    })
}

pub(super) async fn evaluate_population(
    asker: &Asker,
    memo: &mut Memo<'_>,
    draw: LabelDraw<'_>,
    population: ExamplePopulation,
    native_rows: Option<RecordBatch>,
) -> Result<EvaluatedContext, CoreError> {
    let groups = evaluate_label_groups(asker, memo, draw, population, native_rows).await?;
    let native_labels = groups.native_labels()?;
    let context = complete_label_groups(asker, draw, &groups, None)?;
    Ok(EvaluatedContext {
        context,
        native_labels,
    })
}

/// Existing context admission and coverage policy, after reached candidate reads.
pub(super) async fn finish_context(
    asker: &Asker,
    memo: &mut Memo<'_>,
    draw: LabelDraw<'_>,
    labelled: LabelledContext,
    kept: usize,
) -> Result<LabelledContext, CoreError> {
    let LabelDraw {
        entity,
        read_times,
        fetches,
        windows,
        context_anchor,
        anchor_entity,
        context_examples,
        trials,
        entity_filter,
        ..
    } = draw;
    let gathered = labelled.keys.len().min(context_examples);
    if gathered == 0 {
        let ends: Vec<Offset> = windows.iter().map(|w| w.end).collect();
        let mut aggregated: Vec<&Table> = Vec::new();
        for fetch in fetches {
            if !aggregated.iter().any(|t| t.name == fetch.table.name) {
                aggregated.push(fetch.table);
            }
        }
        aggregated.sort_by(|a, b| a.name.cmp(&b.name));
        return Err(empty_sample(
            memo,
            read_times,
            entity,
            &aggregated,
            &ends,
            (!anchor_entity).then_some(context_anchor),
            entity_filter,
        )
        .await);
    }
    if kept >= trials && gathered < context_examples / 2 {
        // The reference warns of it once in a process: given on every run
        // that meets it, and once in a process by the host
        // ([`ONCE_PER_PROCESS`], decision E52).
        asker.warn(CoreWarning {
            fields: [
                ("examples".to_owned(), count(context_examples)),
                ("candidates".to_owned(), count(trials)),
            ]
            .into(),
            ..CoreWarning::new(
                Sentence::ContextCoverage,
                "fewer than half the context examples asked for were collected",
            )
        });
    }
    Ok(labelled.narrow(Some(context_examples)))
}

/// The useful scalar labelled-context table before native prediction-entity
/// conversion. Its aligned entity/anchor/target columns are typed Arrow data,
/// not encoded model rows or an opaque materialization continuation.
pub(super) fn labelled_rows(
    keys: &ArrayRef,
    anchors: &[i64],
    labels: &sdm_engine::labels::Labels,
) -> Result<RecordBatch, CoreError> {
    use sdm_engine::labels::Labels;

    let target: ArrayRef = match labels {
        Labels::Float(values) => Arc::new(Float32Array::from(values.clone())),
        Labels::Int(values) => Arc::new(Int64Array::from(values.clone())),
        Labels::Bool(values) => Arc::new(BooleanArray::from(values.clone())),
        Labels::Text(values) => sdm_engine::strings::encode(values).0,
        Labels::Lists(_) => {
            return Err(internal("the scalar label echo does not carry list labels"));
        }
    };
    let schema = Arc::new(Schema::new(vec![
        Field::new("entity", keys.data_type().clone(), true),
        Field::new(
            "anchor",
            DataType::Timestamp(TimeUnit::Nanosecond, None),
            false,
        ),
        Field::new("target", target.data_type().clone(), false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::clone(keys),
            Arc::new(TimestampNanosecondArray::from(anchors.to_vec())),
            target,
        ],
    )
    .map_err(|_| internal("the labels do not make one batch"))
}

#[cfg(test)]
mod native_label_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::cast::AsArray;
    use arrow_array::types::Int64Type;
    use sdm_contract::ErrorCode;

    fn integers(values: Vec<i64>) -> ArrayRef {
        Arc::new(Int64Array::from(values))
    }

    #[test]
    fn native_nullable_float_missing_keys_keep_the_existing_nan_key_admission() {
        let keys: ArrayRef = Arc::new(arrow_array::Float64Array::from(vec![
            Some(f64::NAN),
            None,
            Some(3.0),
        ]));
        assert_eq!(nan_rows(&keys), BTreeSet::from([0, 1]));
        let keys: ArrayRef = Arc::new(arrow_array::Float32Array::from(vec![None, Some(1.0)]));
        assert_eq!(nan_rows(&keys), BTreeSet::from([0]));
        let integers: ArrayRef = Arc::new(Int64Array::from(vec![None, Some(1)]));
        assert!(nan_rows(&integers).is_empty());
    }

    #[test]
    fn candidate_slices_share_complete_source_allocations_and_copy_only_reached_examples() {
        let keys = integers((0..25_000).collect());
        let population = ExamplePopulation {
            source_read: None,
            sampled: RecordBatch::try_from_iter([("id", Arc::clone(&keys))]).unwrap(),
            entity_key: keys,
            kept: (0..25_000).collect::<Vec<_>>().into(),
            nan_key: Arc::new(BTreeSet::from([42, 20_001])),
            examples: sample::Examples {
                index: (0..25_000).rev().collect(),
                anchors: (0..25_000).collect(),
            },
        };
        let selected = population.slice(10_000..20_000).unwrap();
        assert!(Arc::ptr_eq(&selected.kept, &population.kept));
        assert!(Arc::ptr_eq(&selected.nan_key, &population.nan_key));
        assert!(Arc::ptr_eq(&selected.entity_key, &population.entity_key));
        assert!(Arc::ptr_eq(
            selected.sampled.column(0),
            population.sampled.column(0)
        ));
        assert_eq!(
            selected.examples.index,
            population.examples.index[10_000..20_000]
        );
        assert_eq!(
            selected.examples.anchors,
            population.examples.anchors[10_000..20_000]
        );
        assert_eq!(selected.examples.index.len(), 10_000);
        assert_eq!(selected.examples.anchors.len(), 10_000);
        assert_eq!(population.examples.index.len(), 25_000);
        let source = Arc::downgrade(&population.kept);
        drop(population);
        assert!(source.upgrade().is_some());
        drop(selected);
        assert!(source.upgrade().is_none());
    }

    #[test]
    fn immutable_task_rows_keep_context_and_exact_prediction_ranges() {
        let predicted = integers(vec![7, 8, 7]);
        let rows = TaskRows::new(
            integers(vec![1, 3]),
            vec![10, 20],
            Arc::clone(&predicted),
            99,
        )
        .unwrap();
        let batch = rows.batch(1..3).unwrap();
        assert_eq!(batch.range, 1..3);
        assert_eq!(
            batch.keys.as_primitive::<Int64Type>().values().as_ref(),
            &[1, 3, 8, 7]
        );
        assert_eq!(batch.anchors, [10, 20, 99, 99]);
        assert_eq!(
            batch
                .predicted
                .as_primitive::<Int64Type>()
                .values()
                .as_ptr(),
            predicted.as_primitive::<Int64Type>().values()[1..].as_ptr()
        );
        let earlier = rows.batch(0..1).unwrap();
        assert_eq!(earlier.range, 0..1);
        assert_eq!(
            earlier.keys.as_primitive::<Int64Type>().values().as_ref(),
            &[1, 3, 7]
        );
        assert_eq!(earlier.anchors, [10, 20, 99]);
        assert_eq!(rows.context_anchors, [10, 20]);
        assert_eq!(predicted.len(), 3);
    }

    #[test]
    fn task_row_shapes_and_ranges_are_checked_before_arrow_slicing() {
        let error = TaskRows::new(integers(vec![1]), vec![], integers(vec![7]), 0)
            .err()
            .expect("misaligned task anchors were accepted");
        assert_eq!(error.code, ErrorCode::InvalidMessage);
        let rows = TaskRows::new(integers(vec![1]), vec![0], integers(vec![7]), 0).unwrap();
        for (start, end) in [(2, 2), (0, 2), (1, 0)] {
            let error = rows
                .batch(start..end)
                .err()
                .expect("invalid task range was accepted");
            assert_eq!(error.code, ErrorCode::InvalidMessage);
        }
        let empty = rows.batch(1..1).unwrap();
        assert_eq!(empty.keys.len(), 1);
        assert!(empty.predicted.is_empty());
        assert_eq!(empty.anchors, [0]);
    }
}
