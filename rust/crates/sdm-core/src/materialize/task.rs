// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared typed entity/label draws, context narrowing and ordered task ranges.
//! These are useful task data and plans, with no native input callback or
//! request wire schema. Provider I/O remains the existing typed DataRequest.

use std::collections::BTreeSet;
use std::sync::Arc;

use arrow_array::{
    ArrayRef, BooleanArray, Float32Array, Int64Array, RecordBatch, StringArray,
    TimestampNanosecondArray,
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
use crate::describe::ask;
use crate::run::{Asker, internal};

/// Typed rows obtained by the existing entity draw, before labels or any
/// caller-supplied prediction entity is converted.
pub(super) struct EntitySample {
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
        read_times.read(
            asker,
            entity,
            one_batch(
                &ask(
                    asker,
                    DataRequest::SampleEntities {
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
                    },
                )
                .await?,
            )?,
        )?
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
    let sampled_keys = crate::subgraph::key_values(entity, key, named(&sampled, key)?)?;
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
        sampled,
        entity_key,
        entity_time,
        kept,
    })
}

/// Complete labeled context before the native prediction-entity constructor
/// and later client narrowing. It contains real task data, not a continuation.
pub(super) struct LabelledContext {
    pub(super) keys: ArrayRef,
    pub(super) anchors: Vec<i64>,
    pub(super) labels: labels::Labels,
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

pub(super) async fn label_context(
    asker: &Asker,
    memo: &mut Memo<'_>,
    draw: LabelDraw<'_>,
    sampled: EntitySample,
) -> Result<LabelledContext, CoreError> {
    let LabelDraw {
        entity,
        key,
        asked,
        read_times,
        labelled_fetches,
        fetches,
        windows,
        entropy,
        context_anchor,
        anchor_entity,
        span,
        earliest,
        context_examples,
        trials,
        entity_filter,
    } = draw;
    let EntitySample {
        sampled,
        entity_key,
        entity_time,
        kept,
    } = sampled;
    let mut rng = default_rng(entropy);
    let examples = match &span {
        Some((start, end)) => sample::temporal_examples(
            &mut rng,
            kept.len(),
            entity_time.as_deref(),
            context_anchor,
            timeframe(*start, *end, "anchor_time")?,
            earliest.unwrap_or(TIME_MAX),
            trials,
        )?,
        None if anchor_entity => sample::entity_examples(
            &mut rng,
            kept.len(),
            entity_time.as_deref().unwrap_or_default(),
            sample::MISSING_ANCHOR,
        ),
        None => {
            sample::static_examples(&mut rng, kept.len(), entity_time.as_deref(), context_anchor)
        }
    };

    // A NaN key a source hands over, which the reference reads as missing
    // where it reads floats into NumPy-backed pandas (Postgres's rows), is
    // a missing key its sampler cannot coerce (`_coerce_id`'s `int` or
    // `float` of `NA`) where an example's key seeds the read of a windowed
    // target's rows (`_by_time`): refused as the caller's error before that
    // read. A static target reads no rows by the key, and such an example
    // is left out once labelled, below (T2.12).
    let nan_key = nan_rows(&entity_key);
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
    let entity_rows: Vec<usize> = examples.index.iter().map(|&i| kept[i]).collect();
    let labelling = Labelling {
        asker,
        times: read_times,
        sampled: &sampled,
        entity_rows: &entity_rows,
        entity_key: &entity_key,
        examples: &examples,
    };
    let fetched = labelling.read(labelled_fetches).await?;
    let target = labelling.build(&asked.target, &fetched, true)?;
    // The examples the entity filter and the assumption choose: each
    // condition's truth over the examples, and-ed (`execute`'s `mask`).
    let mut chosen: Option<Vec<bool>> = None;
    for shape in &asked.choosing {
        let holds = labels::truth(&labelling.build(shape, &fetched, false)?)?.values;
        chosen = Some(match chosen {
            None => holds,
            Some(before) => before.iter().zip(&holds).map(|(&a, &b)| a && b).collect(),
        });
    }
    drop(fetched);
    // A classification keeps a decimal column's distinct values apart as
    // classes; any other task reads them as numbers (T2.0c, round 1).
    let mut labelled = if matches!(asked.task, Task::Multiclass) {
        labels::label_classes(&target)?
    } else {
        labels::label(&target)?
    };
    null_compared(&asked.target, &labelled.mask)?;
    // The examples the entity filter or the assumption leaves out keep no
    // label (`out[mask[_mask]]`, `mask & _mask`).
    if let Some(chosen) = &chosen {
        let keep: Vec<bool> = (0..labelled.mask.len())
            .filter(|&row| labelled.mask[row])
            .map(|row| chosen[row])
            .collect();
        for (row, holds) in chosen.iter().enumerate() {
            labelled.mask[row] &= *holds;
        }
        labelled.labels = labelled.labels.kept(&keep);
    }
    // An example of an entity whose key is NaN is left out of the context
    // once labelled, as the reference leaves it out (T2.12).
    if !nan_key.is_empty() {
        let keep: Vec<bool> = (0..labelled.mask.len())
            .filter(|&row| labelled.mask[row])
            .map(|row| !nan_key.contains(&examples.index[row]))
            .collect();
        for row in 0..labelled.mask.len() {
            labelled.mask[row] &= !nan_key.contains(&examples.index[row]);
        }
        labelled.labels = labelled.labels.kept(&keep);
    }
    // The examples with a label, in order, the first a context holds.
    let labelled_rows: Vec<usize> = (0..labelled.mask.len())
        .filter(|&row| labelled.mask[row])
        .collect();
    let gathered = labelled_rows.len().min(context_examples);
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
    if kept.len() >= trials && gathered < context_examples / 2 {
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
    let rows: Vec<usize> = labelled_rows[..gathered]
        .iter()
        .map(|&row| examples.index[row])
        .collect();
    let anchors = labelled_rows[..gathered]
        .iter()
        .map(|&row| examples.anchors[row])
        .collect();
    Ok(LabelledContext {
        keys: take(&entity_key, &rows)?,
        anchors,
        labels: labelled.labels.truncated(gathered),
    })
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
        Labels::Text(values) => Arc::new(StringArray::from(values.clone())),
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
mod tests {
    use super::*;
    use arrow_array::cast::AsArray;
    use arrow_array::types::Int64Type;
    use sdm_contract::ErrorCode;

    fn integers(values: Vec<i64>) -> ArrayRef {
        Arc::new(Int64Array::from(values))
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
        for range in [2..2, 0..2, 1..0] {
            let error = rows
                .batch(range)
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
