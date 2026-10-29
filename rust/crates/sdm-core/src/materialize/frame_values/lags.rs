// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Immutable lag inputs from the same local adjacency and shared label kernel.
//! The derived complete columns are mathematical data, computed once at the
//! original payload boundary; no native owner, workflow cursor or callback.

use super::*;
use std::sync::OnceLock;

pub(super) struct FrameLags {
    input: crate::lag::Lag,
    gathered: usize,
    predictions: usize,
    columns: OnceLock<Result<Vec<payload::LagColumn>, CoreError>>,
}

impl FrameLags {
    pub(super) fn prepare(
        bound: &BoundLabels,
        prediction: &ArrayRef,
        encoding: Option<sdm_contract::data::strings::StringEncoding>,
        steps: u64,
    ) -> Result<Option<Arc<Self>>, CoreError> {
        if steps == 0 {
            return Ok(None);
        }
        let range = bound
            .asked
            .range
            .as_ref()
            .ok_or_else(|| internal("native lags lack a temporal target window"))?;
        let end = Offset::of(range.end, range.unit);
        let mut windows = Vec::new();
        bound.asked.target.windowed(&mut windows);
        let fetches = fetches(
            &bound.graph.prepared.graph,
            &bound.value.meaning.entity.table,
            &windows,
        )?;
        let context = Arc::clone(&bound.labelled.keys);
        let prediction = Arc::clone(prediction);
        let (context, prediction) = match encoding {
            Some(encoding) => align_task_keys(
                context,
                bound
                    .graph
                    .native_column(&bound.value.meaning.entity)?
                    .strings
                    .ok_or_else(|| protocol("lag source key lost its explicit native encoding"))?,
                prediction,
                encoding,
            )?,
            None => (context, prediction),
        };
        let keys = arrow_select::concat::concat(&[context.as_ref(), prediction.as_ref()])
            .map_err(|_| internal("native lag keys do not concatenate"))?;
        let mut anchors = bound.labelled.anchors.clone();
        anchors.extend(std::iter::repeat_n(
            bound.value.meaning.prediction_anchor,
            prediction.len(),
        ));
        let shifted = crate::lag::shifted(steps, end, &anchors)?;
        let mut fetched = BTreeMap::new();
        for fetch in fetches {
            let upper = crate::lag::times(fetch.end, steps)?;
            // This is the original native query hop: floor each moved
            // anchor to seconds, use the registered earliest/latest offset
            // bounds, then apply exact calendar/nanosecond label predicates.
            let windows = label_windows(&shifted, fetch.start, upper);
            let selected = crate::frame_materialize::lock(&bound.graph.frames).temporal_rows(
                &fetch.edge,
                &keys,
                &windows,
                &fetch.table.ordered(&fetch.columns),
            )?;
            let time = fetch
                .table
                .time_column
                .as_deref()
                .ok_or_else(|| internal("native lag source lacks its time column"))?;
            fetched.insert(
                fetch.table.name.clone(),
                Fetched {
                    strings: crate::frame_materialize::lock(&bound.graph.frames)
                        .string_encodings(&fetch.table.name)?,
                    times: times_ns(named(&selected.rows, time)?, time)?,
                    rows: selected.rows,
                    batch: selected.batch,
                    source_positions: Some(selected.source_rows),
                    source_read: None,
                    time_column: time.to_owned(),
                },
            );
        }
        let shape = autoregressive(&bound.asked.target);
        let examples = anchors.len();
        Ok(Some(Arc::new(Self {
            input: crate::lag::Lag {
                steps,
                end,
                shifted,
                target: Box::new(move |anchors| {
                    temporal_target(&shape, &fetched, anchors, examples, "lag_timesteps")
                }),
            },
            gathered: context.len(),
            predictions: prediction.len(),
            columns: OnceLock::new(),
        })))
    }

    pub(super) fn columns(
        &self,
        trained: usize,
        largest: usize,
        range: FrameBatchRange,
    ) -> Result<Vec<payload::LagColumn>, CoreError> {
        let columns = self
            .columns
            .get_or_init(|| {
                self.input.fits(trained + largest)?;
                self.input.columns(
                    &(0..trained)
                        .chain(self.gathered..self.gathered + self.predictions)
                        .collect::<Vec<_>>(),
                )
            })
            .as_ref()
            .map_err(Clone::clone)?;
        let start = usize::try_from(range.start)
            .map_err(|_| protocol("native lag range exceeds host address space"))?;
        let stop = usize::try_from(range.stop)
            .map_err(|_| protocol("native lag range exceeds host address space"))?;
        let rows = (0..trained)
            .chain(trained + start..trained + stop)
            .collect::<Vec<_>>();
        Ok(columns.iter().map(|column| column.rows(&rows)).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    #[test]
    fn lag_values_keep_full_task_types_and_gathered_offsets_across_batches() {
        let evaluated = Arc::new(AtomicUsize::new(0));
        let seen = Arc::clone(&evaluated);
        let lags = FrameLags {
            input: crate::lag::Lag {
                steps: 2,
                end: Offset::of(0, sdm_contract::pql::TimeUnit::Seconds),
                shifted: vec![10, 11, 12, 20, 21, 22],
                target: Box::new(move |anchors| {
                    seen.fetch_add(1, Ordering::Relaxed);
                    Ok(labels::Target::Aggregated(labels::Aggregated {
                        values: anchors.iter().map(|&anchor| anchor as f32).collect(),
                        present: vec![true; anchors.len()],
                    }))
                }),
            },
            gathered: 3,
            predictions: 3,
            columns: OnceLock::new(),
        };
        let first = lags
            .columns(1, 2, FrameBatchRange { start: 0, stop: 2 })
            .unwrap();
        let second = lags
            .columns(1, 2, FrameBatchRange { start: 2, stop: 3 })
            .unwrap();
        assert_eq!(evaluated.load(Ordering::Relaxed), 2);
        for column in first {
            assert_eq!(
                column.values,
                payload::LagValues::Float(vec![Some(10.0), Some(20.0), Some(21.0)])
            );
        }
        for column in second {
            assert_eq!(
                column.values,
                payload::LagValues::Float(vec![Some(10.0), Some(22.0)])
            );
        }
    }
}
