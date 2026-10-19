// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `lag_timesteps` (T2.9): the target's own past as features of each
//! example, as the reference's engine adds them to a temporal task
//! (`KumoRelational.add_lagged_target`, `Sampler.sample_lagged_target`),
//! once the context is labelled and before it is narrowed.
//!
//! With `n` steps and the target's window `(start, end]`, each example's
//! anchor is moved back by `n` windows' ends (`anchor_time - n * offset`,
//! the offset the window's end, a month's by the calendar as pandas moves
//! one), and the rows of the aggregated table from there are read once,
//! after the moved anchor plus the window's start and not after it plus
//! `n` times the window's end (`_sample_query_data`'s `_by_time`, one
//! statement, as the reference runs it), for the context's examples and
//! the entities predicted for, in that order. Column `__kumo_arl<k>__` is
//! the target over step `n - 1 - k`'s window, whose anchor is the moved one
//! plus that many windows' ends: `__kumo_arl0__` the window just before
//! each example's own.
//!
//! Each step's target is labelled as the reference's executor labels it
//! there (`execute` with `filter_na=False` over a query whose aggregation
//! is joined, so `execute_join` drops what has no value): an aggregation
//! (a sum or a count of every example; an average, a minimum or a maximum
//! of only the examples whose window holds a value) or its comparison, each
//! value on its own example and an example whose window holds none
//! without one. The reference's engine lines the values up by position
//! instead (`pd.DataFrame`, then `pd.concat` with the examples), so an
//! average's, a minimum's or a maximum's move to the first examples rather
//! than their own; the core puts each on its own row (T2.9 fix 1, a
//! registered difference decided under KO2-70).

use sdm_contract::{CoreError, Detail, ErrorCode, ErrorKind, Scalar};
use sdm_engine::labels::{self, Aggregation, Constant, Labels, RelOp, Target};
use sdm_engine::sample::out_of_range;
use sdm_engine::time::offset::Offset;
use sdm_request::payload::{LagColumn, LagValues, PAYLOAD_BYTES};

/// The field a lag's refusal names.
const FIELD: &str = "lag_timesteps";

/// The fewest bytes a lagged column adds to a payload: its declaration in
/// the instance table's schema (`"__kumo_arl0__": {"dtype": "bool"}`).
const COLUMN_BYTES: usize = 30;

/// The fewest bytes one lagged cell adds: the shortest value (`0.0`, `true`
/// or `null`) and its separator.
const CELL_BYTES: usize = 4;

/// `offset` taken `times` times, as pandas multiplies a `DateOffset`: a
/// count of its unit `times` as large; past what the count holds, refused
/// as a time out of range naming `lag_timesteps`, as pandas overflows.
fn times(offset: Offset, times: u64) -> Result<Offset, CoreError> {
    i64::try_from(times)
        .ok()
        .and_then(|times| offset.count.checked_mul(times))
        .map(|count| Offset {
            count,
            unit: offset.unit,
        })
        .ok_or_else(|| out_of_range(FIELD))
}

/// Each example's anchor moved back by `steps` windows' ends (`anchor_time
/// - lag_timesteps * offset`), and the window the rows are read in from it:
/// after it plus `start`, not after it plus `steps` times `end`.
pub(crate) struct Window {
    pub shifted: Vec<i64>,
    pub starts: Vec<i64>,
    pub ends: Vec<i64>,
}

/// The window `steps` lags read for examples anchored at `anchors`, a
/// time outside what nanoseconds hold refused naming `lag_timesteps`.
pub(crate) fn window(
    steps: u64,
    start: Offset,
    end: Offset,
    anchors: &[i64],
) -> Result<Window, CoreError> {
    let back = times(end, steps)?.negated();
    let reach = times(end, steps)?;
    let shifted = anchors
        .iter()
        .map(|a| back.add_to(*a, FIELD))
        .collect::<Result<Vec<_>, _>>()?;
    let starts = shifted
        .iter()
        .map(|a| start.add_to(*a, FIELD))
        .collect::<Result<_, _>>()?;
    let ends = shifted
        .iter()
        .map(|a| reach.add_to(*a, FIELD))
        .collect::<Result<_, _>>()?;
    Ok(Window {
        shifted,
        starts,
        ends,
    })
}

/// The rows read for the lags: the example each belongs to, the aggregated
/// column's values (a table's `*`, one each) and each row's time.
pub(crate) struct Rows {
    pub batches: Vec<usize>,
    pub values: labels::Values,
    pub times: Vec<Option<i64>>,
}

/// What a request's lags are computed from, read once the context is
/// labelled; [`Lag::columns`] computes the columns of the examples a
/// payload sends.
pub(crate) struct Lag {
    pub steps: u64,
    pub aggregation: Aggregation,
    pub start: Offset,
    pub end: Offset,
    /// The target's comparison, where it makes one.
    pub condition: Option<(RelOp, Option<Constant>)>,
    /// Each example's moved anchor ([`Window::shifted`]): the context's
    /// examples, then the entities predicted for.
    pub shifted: Vec<i64>,
    pub rows: Rows,
}

impl Lag {
    /// Refuses lags that would make a payload of `rows` instance rows
    /// larger than the endpoint takes, before any column is computed, as
    /// the payload's limit refuses it.
    pub fn fits(&self, rows: usize) -> Result<(), CoreError> {
        let least = usize::try_from(self.steps)
            .ok()
            .and_then(|steps| {
                rows.checked_mul(CELL_BYTES)?
                    .checked_add(COLUMN_BYTES)?
                    .checked_mul(steps)
            })
            .unwrap_or(usize::MAX);
        if least > PAYLOAD_BYTES {
            return Err(CoreError::new(
                ErrorKind::InvalidRequest,
                ErrorCode::InvalidRequest,
                "the request is larger than the endpoint takes",
            )
            .with_field("field", Detail::text("payload"))
            .with_field("lag_timesteps", Detail::Scalar(Scalar::UInt64(self.steps)))
            .with_field(
                "limit",
                Detail::Scalar(Scalar::UInt64(PAYLOAD_BYTES as u64)),
            ));
        }
        Ok(())
    }

    /// The lagged columns of the examples at `kept` (positions among the
    /// examples the lags were read for: the whole request's), `__kumo_arl0__`
    /// first; a condition's that none of them has a value in
    /// [`LagValues::Empty`]. A call takes its slice with
    /// [`LagColumn::rows`], so each column's type is the request's (T2.9
    /// fix 3).
    pub fn columns(&self, kept: &[usize]) -> Result<Vec<LagColumn>, CoreError> {
        let examples = self.shifted.len();
        (0..self.steps)
            .map(|k| {
                // Column k is step steps - 1 - k, anchored that many windows'
                // ends after the moved anchor.
                let step = times(self.end, self.steps - 1 - k)?;
                let anchors = self
                    .shifted
                    .iter()
                    .map(|a| step.add_to(*a, FIELD))
                    .collect::<Result<Vec<_>, _>>()?;
                let ends = anchors
                    .iter()
                    .map(|a| self.end.add_to(*a, FIELD))
                    .collect::<Result<Vec<_>, _>>()?;
                let starts = anchors
                    .iter()
                    .map(|a| self.start.add_to(*a, FIELD))
                    .collect::<Result<Vec<_>, _>>()?;
                let aggregated = labels::aggregate(
                    self.aggregation,
                    examples,
                    &self.rows.batches,
                    &self.rows.values,
                    &self.rows.times,
                    labels::Bounds {
                        starts: Some(&starts),
                        ends: &ends,
                    },
                )?;
                let target = match self.condition {
                    None => Target::Aggregated(aggregated),
                    Some((op, constant)) => Target::Condition {
                        target: Box::new(Target::Aggregated(aggregated)),
                        op,
                        constant,
                    },
                };
                // Each value on its own example: the labels are those of the
                // examples `mask` marks, in order (see the module).
                let labelled = labels::label(&target)?;
                let mut at_example = vec![None; examples];
                for (slot, example) in (0..examples).filter(|&e| labelled.mask[e]).enumerate() {
                    at_example[example] = Some(slot);
                }
                let own = |at: usize| at_example.get(at).copied().flatten();
                let values = match labelled.labels {
                    Labels::Float(v) => LagValues::Float(
                        kept.iter()
                            .map(|&at| own(at).and_then(|i| v.get(i).copied()))
                            .collect(),
                    ),
                    Labels::Bool(v) => {
                        let values: Vec<Option<bool>> = kept
                            .iter()
                            .map(|&at| own(at).and_then(|i| v.get(i).copied()))
                            .collect();
                        if values.iter().any(Option::is_some) {
                            LagValues::Bool(values)
                        } else {
                            LagValues::Empty(values.len())
                        }
                    }
                    Labels::Int(_) | Labels::Text(_) => {
                        return Err(CoreError::new(
                            ErrorKind::Internal,
                            ErrorCode::InternalError,
                            "a lagged target is neither a number nor a condition",
                        ));
                    }
                };
                Ok(LagColumn {
                    name: format!("__kumo_arl{k}__"),
                    values,
                })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sdm_contract::pql::TimeUnit;

    const DAY: i64 = 86_400_000_000_000;

    fn days(n: i32) -> Offset {
        Offset::of(n, TimeUnit::Days)
    }

    /// Two examples anchored at days 100 and 50, a 10-day sum, two steps:
    /// the rows are read from day 80 (and 30) to day 100 (and 50); step 1's
    /// window, `__kumo_arl0__`, is (90, 100], step 0's (80, 90].
    #[test]
    fn each_step_is_the_window_that_many_ends_back() {
        let anchors = [100 * DAY, 50 * DAY];
        let w = window(2, days(0), days(10), &anchors).unwrap();
        assert_eq!(w.shifted, [80 * DAY, 30 * DAY]);
        assert_eq!(w.starts, [80 * DAY, 30 * DAY]);
        assert_eq!(w.ends, [100 * DAY, 50 * DAY]);
        // Rows of example 0 at days 85 and 95 (1.0, 2.0), of example 1 at
        // day 45 (4.0).
        let lag = Lag {
            steps: 2,
            aggregation: Aggregation::Sum,
            start: days(0),
            end: days(10),
            condition: None,
            shifted: w.shifted,
            rows: Rows {
                batches: vec![0, 0, 1],
                values: labels::Values::Float(vec![Some(1.0), Some(2.0), Some(4.0)]),
                times: vec![Some(85 * DAY), Some(95 * DAY), Some(45 * DAY)],
            },
        };
        let columns = lag.columns(&[0, 1]).unwrap();
        assert_eq!(columns[0].name, "__kumo_arl0__");
        assert_eq!(
            columns[0].values,
            LagValues::Float(vec![Some(2.0), Some(4.0)])
        );
        assert_eq!(
            columns[1].values,
            LagValues::Float(vec![Some(1.0), Some(0.0)])
        );
    }

    /// An average has a value only where its window holds one, each on
    /// its own example, the first example's none (the reference's frame
    /// would move the second's value to it; T2.9 fix 1); a comparison of
    /// it likewise.
    #[test]
    fn an_average_s_value_is_on_its_own_example() {
        let w = window(1, days(0), days(10), &[100 * DAY, 50 * DAY]).unwrap();
        let lag = |condition| Lag {
            steps: 1,
            aggregation: Aggregation::Avg,
            start: days(0),
            end: days(10),
            condition,
            shifted: w.shifted.clone(),
            rows: Rows {
                // Only the second example's window holds a row.
                batches: vec![1],
                values: labels::Values::Float(vec![Some(3.0)]),
                times: vec![Some(45 * DAY)],
            },
        };
        assert_eq!(
            lag(None).columns(&[0, 1]).unwrap()[0].values,
            LagValues::Float(vec![None, Some(3.0)])
        );
        assert_eq!(
            lag(Some((RelOp::Gt, Some(Constant::Int(1)))))
                .columns(&[0, 1])
                .unwrap()[0]
                .values,
            LagValues::Bool(vec![None, Some(true)])
        );
        assert_eq!(
            lag(Some((RelOp::Eq, None))).columns(&[1, 0]).unwrap()[0].values,
            LagValues::Bool(vec![Some(false), None])
        );
    }

    /// Examples whose windows hold a value, none, then a value, over two
    /// steps (T2.9 fix 1): a maximum's each on its own row, a gap where a
    /// window is empty, for the context and the entities predicted for
    /// alike; a sum's every row.
    #[test]
    fn a_maximum_s_values_keep_their_rows_across_gaps() {
        let anchors = [100 * DAY, 100 * DAY, 100 * DAY];
        let w = window(2, days(0), days(10), &anchors).unwrap();
        // Example 0: rows at days 85 (5.0) and 95 (7.0); example 1: none;
        // example 2: a row at day 88 (2.0) only.
        let rows = || Rows {
            batches: vec![0, 0, 2],
            values: labels::Values::Float(vec![Some(5.0), Some(7.0), Some(2.0)]),
            times: vec![Some(85 * DAY), Some(95 * DAY), Some(88 * DAY)],
        };
        let lag = |aggregation| Lag {
            steps: 2,
            aggregation,
            start: days(0),
            end: days(10),
            condition: None,
            shifted: w.shifted.clone(),
            rows: rows(),
        };
        let max = lag(Aggregation::Max).columns(&[0, 1, 2]).unwrap();
        // __kumo_arl0__ is (90, 100], __kumo_arl1__ (80, 90].
        assert_eq!(max[0].values, LagValues::Float(vec![Some(7.0), None, None]));
        assert_eq!(
            max[1].values,
            LagValues::Float(vec![Some(5.0), None, Some(2.0)])
        );
        // The rows a request keeps, in their own order.
        let kept = lag(Aggregation::Max).columns(&[2, 0]).unwrap();
        assert_eq!(kept[1].values, LagValues::Float(vec![Some(2.0), Some(5.0)]));
        let sum = lag(Aggregation::Sum).columns(&[0, 1, 2]).unwrap();
        assert_eq!(
            sum[1].values,
            LagValues::Float(vec![Some(5.0), Some(0.0), Some(2.0)])
        );
    }

    /// A condition's lag column takes its type from the whole request
    /// (T2.9 fix 3): `bool` where any example has a value, kept so in a
    /// call whose own rows hold none; `float64` nulls only where no example
    /// of the request has one.
    #[test]
    fn a_condition_s_lag_type_is_the_request_s_in_every_call() {
        // Example 0's window holds a row, example 1's none.
        let w = window(1, days(0), days(10), &[100 * DAY, 100 * DAY]).unwrap();
        let lag = |rows: Rows| Lag {
            steps: 1,
            aggregation: Aggregation::Avg,
            start: days(0),
            end: days(10),
            condition: Some((RelOp::Gt, Some(Constant::Int(1)))),
            shifted: w.shifted.clone(),
            rows,
        };
        let some = lag(Rows {
            batches: vec![0],
            values: labels::Values::Float(vec![Some(3.0)]),
            times: vec![Some(95 * DAY)],
        })
        .columns(&[0, 1])
        .unwrap();
        assert_eq!(some[0].values, LagValues::Bool(vec![Some(true), None]));
        // The call holding example 1 alone keeps the column's type.
        assert_eq!(some[0].rows(&[1]).values, LagValues::Bool(vec![None]));
        let none = lag(Rows {
            batches: vec![],
            values: labels::Values::Float(vec![]),
            times: vec![],
        })
        .columns(&[0, 1])
        .unwrap();
        assert_eq!(none[0].values, LagValues::Empty(2));
        assert_eq!(none[0].rows(&[1]).values, LagValues::Empty(1));
    }

    /// A month's window is moved by the calendar, as pandas moves it: from
    /// 31 March, two months back is 31 January, and the step after it ends
    /// on the last of February.
    #[test]
    fn months_are_moved_by_the_calendar() {
        let month = Offset::of(1, TimeUnit::Months);
        let march_31 = 1_680_220_800 * 1_000_000_000_i64;
        let january_31 = 1_675_123_200 * 1_000_000_000_i64;
        let w = window(2, Offset::of(0, TimeUnit::Months), month, &[march_31]).unwrap();
        assert_eq!(w.shifted, [january_31]);
        assert_eq!(w.ends, [march_31]);
    }

    /// Steps that reach outside what nanoseconds hold, or past what a count
    /// holds, are refused naming `lag_timesteps`; a payload they would make
    /// larger than the endpoint takes, before anything is computed.
    #[test]
    fn too_many_steps_are_refused_by_name() {
        for steps in [100_000, u64::MAX] {
            let e = window(steps, days(0), days(30), &[0]).err().unwrap();
            assert_eq!(e.code, ErrorCode::ValueOutOfRange);
            assert_eq!(e.fields["field"], Detail::text(FIELD));
        }
        let lag = Lag {
            steps: 10_000_000,
            aggregation: Aggregation::Count,
            start: Offset::of(0, TimeUnit::Seconds),
            end: Offset::of(1, TimeUnit::Seconds),
            condition: None,
            shifted: vec![0, 0],
            rows: Rows {
                batches: vec![],
                values: labels::Values::Ones(0),
                times: vec![],
            },
        };
        let e = lag.fits(2).err().unwrap();
        assert_eq!(e.code, ErrorCode::InvalidRequest);
        assert_eq!(e.fields["field"], Detail::text("payload"));
    }
}
