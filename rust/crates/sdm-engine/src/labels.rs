// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Labels over fetched rows, as the reference's pandas executor computes
//! them (`PQueryPandasExecutor`), in Rust, never compiled to SQL
//! (decision E19), compared exactly.
//!
//! - A column is read as the executor reads it (`execute_column`): a float
//!   at float32, an integer at int64, a decimal exactly; a null, and a
//!   float's NaN, is no value.
//! - A windowed aggregation ([`aggregate`], `execute_aggregation`) takes, of
//!   each example, the rows whose value is present and whose time is after
//!   its window's start, when it has one, and not after its end, in row
//!   order: `SUM` of a float at float32 with Kahan's compensation (register
//!   PK-2), of an integer in int64, of a decimal exactly; `COUNT` the rows;
//!   `AVG` a float's Kahan sum at float32 over the count, an integer's at
//!   float64; `MIN` and `MAX` exactly; then each read at float32, a decimal
//!   by Arrow's conversion to float64 first, as pandas' `astype` takes it.
//!   An example with no row is 0 for `SUM` and `COUNT` and has no label
//!   for the others.
//! - A condition ([`compare`], `execute_rel_op`) compares the values its
//!   target has with a constant read at the target's type: at float32 for a
//!   float32 target, at float64 for an integer target and a float constant,
//!   at int64 for an integer target and an integer constant.
//! - [`label`] puts them together as `execute` does: the examples whose
//!   target has a value keep their label, in order; the rest have none.
//!
//! What the executor does by a path this does not port is refused by name
//! ([`crate::not_yet`], register P1B-3).

use sdm_contract::{CoreError, ErrorCode, ErrorKind};

use crate::not_yet;

/// The values a column's fetched rows hold, as the executor reads them.
#[derive(Debug, Clone, PartialEq)]
pub enum Values {
    /// A float column at float32 (`astype('float32')`).
    Float(Vec<Option<f32>>),
    /// An integer column at int64.
    Int(Vec<Option<i64>>),
    /// A decimal column's unscaled values and its scale.
    Decimal {
        values: Vec<Option<i128>>,
        scale: i8,
    },
    /// A table's `*`: one for each row.
    Ones(usize),
}

impl Values {
    fn len(&self) -> usize {
        match self {
            Values::Float(v) => v.len(),
            Values::Int(v) => v.len(),
            Values::Decimal { values, .. } => values.len(),
            Values::Ones(n) => *n,
        }
    }

    /// Whether row `row` holds a value: not null, and for a float not NaN.
    fn present(&self, row: usize) -> bool {
        match self {
            Values::Float(v) => v[row].is_some_and(|v| !v.is_nan()),
            Values::Int(v) => v[row].is_some(),
            Values::Decimal { values, .. } => values[row].is_some(),
            Values::Ones(_) => true,
        }
    }
}

/// An aggregation the executor computes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Aggregation {
    Sum,
    Count,
    Avg,
    Min,
    Max,
}

/// Each example's window: times after `starts`, when given, and not after
/// `ends`, nanoseconds since the epoch, one of each per example.
#[derive(Debug, Clone, Copy)]
pub struct Bounds<'a> {
    pub starts: Option<&'a [i64]>,
    pub ends: &'a [i64],
}

/// An aggregation's answer for each example: its value at float32, and
/// whether it has one (`SUM` and `COUNT` always have).
#[derive(Debug, Clone, PartialEq)]
pub struct Aggregated {
    pub values: Vec<f32>,
    pub present: Vec<bool>,
}

/// The `f64` Arrow's `Decimal128::ToDouble` gives the decimal of unscaled
/// value `value` at `scale`: the integer's two halves at float64, scaled by
/// the power of ten, split into whole and fractional parts where the value
/// is past what a float64 holds exactly; pandas' `astype('float32')` of a
/// decimal takes this, then narrows it.
pub fn decimal_to_f64(value: i128, scale: i8) -> f64 {
    fn no_split(magnitude: u128, scale: i32) -> f64 {
        let high = (magnitude >> 64) as u64 as i64;
        let low = magnitude as u64;
        let mut x = (high as f64) * 18_446_744_073_709_551_616.0;
        x += low as f64;
        // Arrow's table of powers of ten holds each as its literal, the
        // closest double, which parsing the literal gives.
        let power: f64 = format!("1e{}", -scale).parse().unwrap_or(f64::NAN);
        x * power
    }
    let scale = i32::from(scale);
    let magnitude = value.unsigned_abs();
    let positive = if scale <= 0 || magnitude <= 1_u128 << 53 {
        no_split(magnitude, scale)
    } else {
        let power = 10_u128.pow(scale as u32);
        no_split(magnitude / power, 0) + no_split(magnitude % power, scale)
    };
    if value < 0 { -positive } else { positive }
}

/// The largest unscaled magnitude a `DECIMAL(38, _)` holds.
const DECIMAL_38: i128 = 99_999_999_999_999_999_999_999_999_999_999_999_999;

/// `aggregation` of `values` for each of `examples` examples: the rows of
/// example `batch[row]` whose value is present and whose time is in its
/// window ([`Bounds`]), in row order. A decimal's sum past `DECIMAL(38)`,
/// and a decimal's average, take rules this does not port, and are refused
/// by name (`target_type`).
pub fn aggregate(
    aggregation: Aggregation,
    examples: usize,
    batch: &[usize],
    values: &Values,
    times: &[Option<i64>],
    bounds: Bounds<'_>,
) -> Result<Aggregated, CoreError> {
    let rows = batch.len();
    if values.len() != rows || times.len() != rows || bounds.ends.len() != examples {
        return Err(CoreError::new(
            ErrorKind::Internal,
            ErrorCode::InternalError,
            "a label's rows, times and windows do not align",
        ));
    }
    let within = |row: usize| -> bool {
        let Some(time) = times[row] else {
            return false;
        };
        let at = batch[row];
        time <= bounds.ends[at] && bounds.starts.is_none_or(|starts| time > starts[at])
    };
    // The rows each example aggregates, in row order.
    let mut taken: Vec<Vec<usize>> = vec![Vec::new(); examples];
    for row in 0..rows {
        if values.present(row) && within(row) {
            taken[batch[row]].push(row);
        }
    }
    let present: Vec<bool> = match aggregation {
        Aggregation::Sum | Aggregation::Count => vec![true; examples],
        _ => taken.iter().map(|rows| !rows.is_empty()).collect(),
    };
    let mut out = vec![0_f32; examples];
    for (at, rows) in taken.iter().enumerate() {
        if !present[at] {
            continue;
        }
        out[at] = match aggregation {
            Aggregation::Count => rows.len() as i64 as f32,
            Aggregation::Sum => match values {
                Values::Float(v) => kahan_f32(rows.iter().map(|&r| v[r].unwrap_or(0.0))).0,
                Values::Int(v) => rows
                    .iter()
                    .fold(0_i64, |s, &r| s.wrapping_add(v[r].unwrap_or(0)))
                    as f32,
                Values::Ones(_) => rows.len() as i64 as f32,
                Values::Decimal { values: v, scale } => {
                    let mut sum: i128 = 0;
                    for &r in rows {
                        sum = sum
                            .checked_add(v[r].unwrap_or(0))
                            .filter(|s| s.abs() <= DECIMAL_38)
                            .ok_or_else(|| not_yet("target_type"))?;
                    }
                    decimal_to_f64(sum, *scale) as f32
                }
            },
            Aggregation::Avg => match values {
                Values::Float(v) => {
                    let (sum, n) = kahan_f32(rows.iter().map(|&r| v[r].unwrap_or(0.0)));
                    sum / (n as i64 as f32)
                }
                Values::Int(v) => {
                    let (sum, n) = kahan_f64(rows.iter().map(|&r| v[r].unwrap_or(0) as f64));
                    (sum / (n as i64 as f64)) as f32
                }
                Values::Ones(_) => 1.0,
                Values::Decimal { .. } => return Err(not_yet("target_type")),
            },
            Aggregation::Min | Aggregation::Max => {
                let max = aggregation == Aggregation::Max;
                match values {
                    Values::Float(v) => {
                        let mut best = if max {
                            f32::NEG_INFINITY
                        } else {
                            f32::INFINITY
                        };
                        for &r in rows {
                            let value = v[r].unwrap_or(0.0);
                            if (max && value > best) || (!max && value < best) {
                                best = value;
                            }
                        }
                        best
                    }
                    Values::Int(v) => {
                        let picked = rows.iter().filter_map(|&r| v[r]);
                        let best = if max { picked.max() } else { picked.min() };
                        best.unwrap_or(0) as f32
                    }
                    Values::Ones(_) => 1.0,
                    Values::Decimal { values: v, scale } => {
                        let picked = rows.iter().filter_map(|&r| v[r]);
                        let best = if max { picked.max() } else { picked.min() };
                        decimal_to_f64(best.unwrap_or(0), *scale) as f32
                    }
                }
            }
        };
    }
    Ok(Aggregated {
        values: out,
        present,
    })
}

/// The float32 Kahan sum of `values`, in order, and their count, as
/// pandas' grouped sum and mean compute it: a compensation that turns NaN
/// is reset to 0.
fn kahan_f32(values: impl Iterator<Item = f32>) -> (f32, usize) {
    let (mut sum, mut compensation, mut n) = (0_f32, 0_f32, 0_usize);
    for value in values {
        let y = value - compensation;
        let t = sum + y;
        compensation = (t - sum) - y;
        if compensation.is_nan() {
            compensation = 0.0;
        }
        sum = t;
        n += 1;
    }
    (sum, n)
}

/// [`kahan_f32`] at float64, for an integer's mean.
fn kahan_f64(values: impl Iterator<Item = f64>) -> (f64, usize) {
    let (mut sum, mut compensation, mut n) = (0_f64, 0_f64, 0_usize);
    for value in values {
        let y = value - compensation;
        let t = sum + y;
        compensation = (t - sum) - y;
        if compensation.is_nan() {
            compensation = 0.0;
        }
        sum = t;
        n += 1;
    }
    (sum, n)
}

/// A comparison a condition makes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RelOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

/// A condition's constant, as the query wrote it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Constant {
    Int(i128),
    Float(f64),
}

/// What a target gives each example it has a value for, in order.
#[derive(Debug, Clone, PartialEq)]
pub enum Labels {
    Float(Vec<f32>),
    Int(Vec<i64>),
    Bool(Vec<bool>),
    Text(Vec<String>),
}

impl Labels {
    pub fn len(&self) -> usize {
        match self {
            Labels::Float(v) => v.len(),
            Labels::Int(v) => v.len(),
            Labels::Bool(v) => v.len(),
            Labels::Text(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The first `n` labels.
    pub fn truncated(mut self, n: usize) -> Labels {
        match &mut self {
            Labels::Float(v) => v.truncate(n),
            Labels::Int(v) => v.truncate(n),
            Labels::Bool(v) => v.truncate(n),
            Labels::Text(v) => v.truncate(n),
        }
        self
    }
}

/// `left op constant` for each value, as `execute_rel_op` compares them: a
/// NaN equal to nothing and, unlike IEEE, not unequal either. An integer
/// constant past int64 against an integer target, which the reference
/// cannot read at the target's type, is refused by name
/// (`condition_value`); a target that is neither a float32 nor an int64,
/// by `condition`.
pub fn compare(left: &Labels, op: RelOp, constant: Constant) -> Result<Vec<bool>, CoreError> {
    fn apply<T: PartialOrd>(a: T, op: RelOp, b: T) -> bool {
        match op {
            RelOp::Eq => a == b,
            RelOp::Ne => a != b,
            RelOp::Lt => a < b,
            RelOp::Le => a <= b,
            RelOp::Gt => a > b,
            RelOp::Ge => a >= b,
        }
    }
    Ok(match left {
        Labels::Float(values) => {
            // The constant at the target's float32, by way of float64, as
            // numpy reads a Python number at float32.
            let value = match constant {
                Constant::Int(i) => i as f64 as f32,
                Constant::Float(f) => f as f32,
            };
            values
                .iter()
                .map(|&v| !v.is_nan() && apply(v, op, value))
                .collect()
        }
        Labels::Int(values) => match constant {
            Constant::Float(f) => values
                .iter()
                .map(|&v| {
                    let v = v as f64;
                    !f.is_nan() && apply(v, op, f)
                })
                .collect(),
            Constant::Int(i) => {
                let value = i64::try_from(i).map_err(|_| not_yet("condition_value"))?;
                values.iter().map(|&v| apply(v, op, value)).collect()
            }
        },
        Labels::Bool(_) | Labels::Text(_) => return Err(not_yet("condition")),
    })
}

/// A static column's values for each example, as the executor reads them.
#[derive(Debug, Clone, PartialEq)]
pub enum Column {
    Float(Vec<Option<f32>>),
    Int(Vec<Option<i64>>),
    Bool(Vec<Option<bool>>),
    Text(Vec<Option<String>>),
}

/// A query's target over the examples.
#[derive(Debug, Clone, PartialEq)]
pub enum Target {
    /// A column of the entity's row.
    Column(Column),
    /// A windowed aggregation.
    Aggregated(Aggregated),
    /// `target op constant`; no constant is `NULL`.
    Condition {
        target: Box<Target>,
        op: RelOp,
        constant: Option<Constant>,
    },
}

/// The examples' labels: which of them have one (`mask`), and the labels
/// of those, in order.
#[derive(Debug, Clone, PartialEq)]
pub struct Labelled {
    pub mask: Vec<bool>,
    pub labels: Labels,
}

/// `target`'s labels, as the executor's `execute` gives them.
pub fn label(target: &Target) -> Result<Labelled, CoreError> {
    match target {
        Target::Column(column) => {
            fn kept<T: Clone>(
                values: &[Option<T>],
                present: impl Fn(&T) -> bool,
            ) -> (Vec<bool>, Vec<T>) {
                let mask: Vec<bool> = values
                    .iter()
                    .map(|v| v.as_ref().is_some_and(&present))
                    .collect();
                let labels = values
                    .iter()
                    .zip(&mask)
                    .filter(|(_, keep)| **keep)
                    .filter_map(|(v, _)| v.clone())
                    .collect();
                (mask, labels)
            }
            let (mask, labels) = match column {
                Column::Float(v) => {
                    let (m, l) = kept(v, |f| !f.is_nan());
                    (m, Labels::Float(l))
                }
                Column::Int(v) => {
                    let (m, l) = kept(v, |_| true);
                    (m, Labels::Int(l))
                }
                Column::Bool(v) => {
                    let (m, l) = kept(v, |_| true);
                    (m, Labels::Bool(l))
                }
                Column::Text(v) => {
                    let (m, l) = kept(v, |_| true);
                    (m, Labels::Text(l))
                }
            };
            Ok(Labelled { mask, labels })
        }
        Target::Aggregated(aggregated) => Ok(Labelled {
            mask: aggregated.present.clone(),
            labels: Labels::Float(
                aggregated
                    .values
                    .iter()
                    .zip(&aggregated.present)
                    .filter(|(_, p)| **p)
                    .map(|(v, _)| *v)
                    .collect(),
            ),
        }),
        Target::Condition {
            target,
            op,
            constant,
        } => {
            let inner = label(target)?;
            match constant {
                Some(constant) => Ok(Labelled {
                    labels: Labels::Bool(compare(&inner.labels, *op, *constant)?),
                    mask: inner.mask,
                }),
                // `= NULL` and `!= NULL` ask whether the target has a value,
                // every example keeping its answer; another comparison with
                // NULL the executor does not make.
                None => {
                    let has: Vec<bool> = inner.mask;
                    let labels = match op {
                        RelOp::Eq => has.iter().map(|h| !h).collect(),
                        RelOp::Ne => has.clone(),
                        _ => return Err(not_yet("condition")),
                    };
                    Ok(Labelled {
                        mask: vec![true; has.len()],
                        labels: Labels::Bool(labels),
                    })
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bounds(ends: &[i64]) -> Bounds<'_> {
        Bounds { starts: None, ends }
    }

    #[test]
    fn a_nan_is_skipped_as_pandas_skips_it() {
        // Example 0 holds only NaN, example 1 NaN and finite values, example
        // 2 a null: each sums what is present, an example with none 0.
        let values = Values::Float(vec![
            Some(f32::NAN),
            Some(1.5),
            Some(f32::NAN),
            Some(2.0),
            None,
        ]);
        let out = aggregate(
            Aggregation::Sum,
            3,
            &[0, 1, 1, 1, 2],
            &values,
            &[Some(5); 5],
            bounds(&[10, 10, 10]),
        )
        .unwrap();
        assert_eq!(out.values, [0.0, 3.5, 0.0]);
        assert_eq!(out.present, [true; 3]);
        let avg = aggregate(
            Aggregation::Avg,
            3,
            &[0, 1, 1, 1, 2],
            &values,
            &[Some(5); 5],
            bounds(&[10, 10, 10]),
        )
        .unwrap();
        assert_eq!(avg.present, [false, true, false]);
        assert_eq!(avg.values[1], 1.75);
    }

    #[test]
    fn kahan_s_compensation_keeps_what_a_plain_sum_loses() {
        // A plain float32 sum of 1e8 and eight 1.0s stays 1e8; the
        // compensated one is 1e8 + 8, as pandas sums it.
        let mut v = vec![Some(1e8_f32)];
        v.extend(std::iter::repeat_n(Some(1.0_f32), 8));
        let n = v.len();
        let out = aggregate(
            Aggregation::Sum,
            1,
            &vec![0; n],
            &Values::Float(v),
            &vec![Some(0); n],
            bounds(&[0]),
        )
        .unwrap();
        assert_eq!(out.values[0], 100_000_008.0);
    }

    #[test]
    fn the_window_is_after_its_start_and_not_after_its_end() {
        let out = aggregate(
            Aggregation::Count,
            1,
            &[0, 0, 0, 0],
            &Values::Ones(4),
            &[Some(0), Some(1), Some(10), Some(11)],
            Bounds {
                starts: Some(&[0]),
                ends: &[10],
            },
        )
        .unwrap();
        assert_eq!(out.values, [2.0]);
    }

    #[test]
    fn a_decimal_reads_as_arrow_reads_it_to_float64() {
        // 5e-18 is the double closest to it, not five times 1e-18 at
        // float32 (Arrow's direct float32 conversion, which pandas does not
        // take).
        assert_eq!(decimal_to_f64(5, 18) as f32, 5e-18_f32);
        assert_eq!(decimal_to_f64(97_000_000_000_000_000_002, 18) as f32, 97.0);
        assert_eq!(
            decimal_to_f64(-49_476_625_724_350, 18) as f32,
            -4.947_662_5e-5
        );
    }

    #[test]
    fn a_condition_reads_its_constant_at_the_target_s_type() {
        let left = Labels::Float(vec![0.0, 0.1, 1.0, f32::NAN]);
        assert_eq!(
            compare(&left, RelOp::Gt, Constant::Int(0)).unwrap(),
            [false, true, true, false]
        );
        // 0.1 at float32 equals the float32 0.1.
        assert_eq!(
            compare(&left, RelOp::Eq, Constant::Float(0.1)).unwrap(),
            [false, true, false, false]
        );
        assert_eq!(
            compare(&left, RelOp::Ne, Constant::Int(1)).unwrap(),
            [true, true, false, false]
        );
        let ints = Labels::Int(vec![1, 2]);
        assert_eq!(
            compare(&ints, RelOp::Lt, Constant::Float(1.5)).unwrap(),
            [true, false]
        );
        assert!(compare(&ints, RelOp::Lt, Constant::Int(1 << 70)).is_err());
    }

    #[test]
    fn a_static_column_keeps_the_examples_it_has_a_value_for() {
        let out = label(&Target::Column(Column::Text(vec![
            Some("IN".into()),
            None,
            Some("DE".into()),
        ])))
        .unwrap();
        assert_eq!(out.mask, [true, false, true]);
        assert_eq!(out.labels, Labels::Text(vec!["IN".into(), "DE".into()]));
        let null = label(&Target::Condition {
            target: Box::new(Target::Column(Column::Float(vec![
                Some(1.0),
                Some(f32::NAN),
                None,
            ]))),
            op: RelOp::Eq,
            constant: None,
        })
        .unwrap();
        assert_eq!(null.mask, [true; 3]);
        assert_eq!(null.labels, Labels::Bool(vec![false, true, true]));
    }
}
