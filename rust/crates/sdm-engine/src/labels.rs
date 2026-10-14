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
//!   PK-2), of an integer in int64, of a decimal exactly, its total held to
//!   the column's declared precision; `COUNT` the rows; `AVG` a float's
//!   Kahan sum at float32 over the count, an integer's at float64, a
//!   decimal's exact sum over the count at the column's scale, rounded half
//!   away from zero; `MIN` and `MAX` exactly; then each read at float32, a
//!   decimal by Arrow's conversion to float64 first, as pandas' `astype`
//!   takes it.
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
    /// A decimal column's unscaled values, its declared precision and its
    /// scale.
    Decimal {
        values: Vec<Option<i128>>,
        precision: u8,
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

/// A decimal sum past the digits its column's type holds: the reference's
/// grouped sum fails on it (pyarrow's `hash_sum`, whose total keeps the
/// column's `decimal128(p, s)`), which its client reports as
/// `INVALID_REQUEST`; the core names the field the query is.
fn decimal_overflow() -> CoreError {
    CoreError::new(
        ErrorKind::InvalidRequest,
        ErrorCode::InvalidRequest,
        "a decimal sum is past the digits its column's type holds",
    )
    .with_field("field", sdm_contract::Detail::text("query"))
}

/// The exact sum of `values`, wrapping as Arrow's `Decimal128` adds them.
fn decimal_total(values: impl Iterator<Item = i128>) -> i128 {
    values.fold(0_i128, i128::wrapping_add)
}

/// The exact sum of `values`, refused when the total is past the
/// `precision` digits its column declares, as the reference's grouped sum
/// checks only its total (T1.27's fix round 3; T1.32b, at the declared
/// precision: `DECIMAL(4, 0)`'s 9999 + 1 refused, 9999 + 1 - 1 taken): a
/// running sum past them, or past `i128`, that comes back within them is
/// the reference's answer.
fn decimal_sum(values: impl Iterator<Item = i128>, precision: u8) -> Result<i128, CoreError> {
    let sum = decimal_total(values);
    // 10**p - 1, the largest unscaled magnitude `DECIMAL(p, _)` holds; past
    // 38 digits, which Arrow's `Decimal128` does not declare, `DECIMAL(38)`'s.
    let largest = 10_u128.pow(u32::from(precision.clamp(1, 38))) - 1;
    if sum.unsigned_abs() > largest {
        return Err(decimal_overflow());
    }
    Ok(sum)
}

/// The mean of `n` decimals whose exact total is `sum`, at their scale, as
/// Arrow's grouped mean of a `Decimal128` gives it: the quotient, moved one
/// away from zero, by the total's sign, where the remainder is at least
/// half the count. The total is not held to the column's precision, as the
/// reference's mean does not hold it.
fn decimal_mean(sum: i128, n: usize) -> i128 {
    let n = n as i128;
    let quotient = sum / n;
    let remainder = (sum % n).unsigned_abs();
    if remainder * 2 >= n.unsigned_abs() {
        if sum >= 0 { quotient + 1 } else { quotient - 1 }
    } else {
        quotient
    }
}

/// `aggregation` of `values` for each of `examples` examples: the rows of
/// example `batch[row]` whose value is present and whose time is in its
/// window ([`Bounds`]), in row order. A decimal's sum whose total is past
/// its column's declared precision is the caller's error,
/// `INVALID_REQUEST`, as the reference's grouped sum fails it and its
/// client reports it; a decimal's average is taken whatever its total, as
/// the reference averages values whose sum would not fit.
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
                Values::Decimal {
                    values: v,
                    precision,
                    scale,
                } => {
                    let sum = decimal_sum(rows.iter().map(|&r| v[r].unwrap_or(0)), *precision)?;
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
                // A decimal's mean, at its scale, whatever the total.
                Values::Decimal {
                    values: v, scale, ..
                } => {
                    let sum = decimal_total(rows.iter().map(|&r| v[r].unwrap_or(0)));
                    decimal_to_f64(decimal_mean(sum, rows.len()), *scale) as f32
                }
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
                    Values::Decimal {
                        values: v, scale, ..
                    } => {
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

/// `a op b`.
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

/// A float's exact value times ten to the `scale`, rounded half to even,
/// as an integer, or `None` when it is not finite or the integer is past
/// `i128`: the value pyarrow gives a float read at a decimal of that scale
/// (Arrow's `Decimal128::FromReal`, exact while the scale is small enough
/// for the float's digits and the power of ten to fit together; measured on
/// the reference's pyarrow: 0.5, 2.5 and 0.25 round to even, 0.15 and
/// 2.675, whose doubles are under the half, down).
fn float_at_scale(value: f64, scale: u32) -> Option<i128> {
    if !value.is_finite() {
        return None;
    }
    // value = mantissa * 2**exponent, exactly.
    let bits = value.abs().to_bits();
    let biased = ((bits >> 52) & 0x7ff) as i32;
    let fraction = u128::from(bits & ((1_u64 << 52) - 1));
    let (mantissa, exponent) = if biased == 0 {
        (fraction, -1074)
    } else {
        (fraction | (1 << 52), biased - 1075)
    };
    let scaled = mantissa.checked_mul(10_u128.checked_pow(scale)?)?;
    let magnitude = if exponent >= 0 {
        scaled.checked_mul(1_u128.checked_shl(exponent as u32)?)?
    } else {
        let shift = exponent.unsigned_abs();
        if shift >= 128 {
            // Under a quarter of the last place: nearer zero than one.
            0
        } else {
            let quotient = scaled >> shift;
            let remainder = scaled & ((1_u128 << shift) - 1);
            let half = 1_u128 << (shift - 1);
            if remainder > half || (remainder == half && quotient % 2 == 1) {
                quotient + 1
            } else {
                quotient
            }
        }
    };
    let magnitude = i128::try_from(magnitude).ok()?;
    Some(if value < 0.0 { -magnitude } else { magnitude })
}

/// The largest scale a float constant is read at exactly
/// ([`decimal_constant`]).
const EXACT_SCALE: u32 = 22;

/// A condition's constant read at a decimal column's `decimal128(precision,
/// scale)`, as `execute_rel_op` reads it (`pd.Series([value],
/// dtype=left.dtype)`), as an unscaled value at the scale: an integer of
/// int64 exactly; a float rounded half to even at a scale of up to 22
/// ([`float_at_scale`]), the scales Arrow's `Decimal128::FromReal` reads a
/// double at exactly (its 38 digits less a double's 16, multiplied at
/// once; measured on pyarrow 25: no difference through scale 25). A constant the type does not hold is the caller's
/// error, `INVALID_REQUEST`, as pyarrow refuses it (`ArrowInvalid`) and the
/// reference's client reports it. Refused by name (`condition_value`): an
/// integer past int64; a float that is not finite; a float at a scale past
/// 22, which pyarrow rounds by an approximation this does not port
/// (register P1B-71); a negative scale.
fn decimal_constant(constant: Constant, precision: u8, scale: i8) -> Result<i128, CoreError> {
    let refused = || not_yet("condition_value");
    let not_held = || {
        CoreError::new(
            ErrorKind::InvalidRequest,
            ErrorCode::InvalidRequest,
            "a condition's constant is past the digits its column's type holds",
        )
        .with_field("field", sdm_contract::Detail::text("query"))
    };
    let scale = u32::try_from(scale).map_err(|_| refused())?;
    let unscaled = match constant {
        Constant::Int(value) => {
            let value = i64::try_from(value).map_err(|_| refused())?;
            // Under 10**38 for any scale that holds it; past i128, held by
            // no DECIMAL(38).
            10_i128
                .checked_pow(scale)
                .and_then(|unit| i128::from(value).checked_mul(unit))
                .ok_or_else(not_held)?
        }
        Constant::Float(value) if !value.is_finite() || scale > EXACT_SCALE => {
            return Err(refused());
        }
        Constant::Float(value) => float_at_scale(value, scale).ok_or_else(not_held)?,
    };
    let largest = 10_u128.pow(u32::from(precision.clamp(1, 38))) - 1;
    if unscaled.unsigned_abs() > largest {
        return Err(not_held());
    }
    Ok(unscaled)
}

/// `= NULL` and `!= NULL` over the examples `has` says have a value: each
/// example keeps its answer, whether it has none, or one; another
/// comparison with NULL the executor does not make.
fn null_condition(has: Vec<bool>, op: RelOp) -> Result<Labelled, CoreError> {
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

/// A static decimal column compared with a constant, as `execute_rel_op`
/// compares it: exactly, the constant read at the column's type
/// ([`decimal_constant`]); the examples with a value keep their answer.
fn decimal_condition(
    values: &[Option<i128>],
    precision: u8,
    scale: i8,
    op: RelOp,
    constant: Option<Constant>,
) -> Result<Labelled, CoreError> {
    let mask: Vec<bool> = values.iter().map(Option::is_some).collect();
    let Some(constant) = constant else {
        return null_condition(mask, op);
    };
    let value = decimal_constant(constant, precision, scale)?;
    Ok(Labelled {
        labels: Labels::Bool(
            values
                .iter()
                .flatten()
                .map(|&v| apply(v, op, value))
                .collect(),
        ),
        mask,
    })
}

/// A static column's values for each example, as the executor reads them.
#[derive(Debug, Clone, PartialEq)]
pub enum Column {
    Float(Vec<Option<f32>>),
    Int(Vec<Option<i64>>),
    Bool(Vec<Option<bool>>),
    Text(Vec<Option<String>>),
    /// A decimal column's unscaled values, its declared precision and its
    /// scale: compared in a condition ([`decimal_condition`]); asked for
    /// itself, a target the reference's task table refuses ("unsupported
    /// data type ... for column 'TARGET'"), refused by name (`target_type`,
    /// register P1B-18).
    Decimal {
        values: Vec<Option<i128>>,
        precision: u8,
        scale: i8,
    },
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
                Column::Decimal { .. } => return Err(not_yet("target_type")),
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
            if let Target::Column(Column::Decimal {
                values,
                precision,
                scale,
            }) = target.as_ref()
            {
                return decimal_condition(values, *precision, *scale, *op, *constant);
            }
            let inner = label(target)?;
            match constant {
                Some(constant) => Ok(Labelled {
                    labels: Labels::Bool(compare(&inner.labels, *op, *constant)?),
                    mask: inner.mask,
                }),
                None => null_condition(inner.mask, *op),
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
    fn a_decimal_sum_past_38_digits_is_the_caller_s_error() {
        let big = 60_000_000_000_000_000_000_000_000_000_000_000_000_i128;
        let values = Values::Decimal {
            values: vec![Some(big), Some(big)],
            precision: 38,
            scale: 18,
        };
        let err = aggregate(
            Aggregation::Sum,
            1,
            &[0, 0],
            &values,
            &[Some(0); 2],
            bounds(&[0]),
        )
        .unwrap_err();
        assert_eq!(err.code, sdm_contract::ErrorCode::InvalidRequest);
        // One of them alone fits.
        let one = Values::Decimal {
            values: vec![Some(big)],
            precision: 38,
            scale: 18,
        };
        assert!(aggregate(Aggregation::Sum, 1, &[0], &one, &[Some(0)], bounds(&[0])).is_ok());
        // Only the total is checked, as the reference checks it: a running
        // sum past 38 digits, or past i128, that comes back is summed (fix
        // round 3; the reference's labels 6.000000208206194e19 and
        // 8.999999813669171e37), in any order.
        let nine = 90_000_000_000_000_000_000_000_000_000_000_000_000_i128;
        for (terms, scale, wanted) in [
            (vec![big, big, -big], 18, 6.000_000_208_206_194e19_f64),
            (vec![big, -big, big], 18, 6.000_000_208_206_194e19),
            (vec![big, big, -big, -big], 18, 0.0),
            (vec![nine, nine, -nine], 0, 8.999_999_813_669_171e37),
        ] {
            let n = terms.len();
            let values = Values::Decimal {
                values: terms.into_iter().map(Some).collect(),
                precision: 38,
                scale,
            };
            let out = aggregate(
                Aggregation::Sum,
                1,
                &vec![0; n],
                &values,
                &vec![Some(0); n],
                bounds(&[0]),
            )
            .unwrap();
            assert_eq!(f64::from(out.values[0]), wanted);
        }
        // An average of the same values is not the caller's error: the
        // reference averages them (T1.27's fix round 2), whatever the total
        // (T1.32b).
        let out = aggregate(
            Aggregation::Avg,
            1,
            &[0, 0],
            &values,
            &[Some(0); 2],
            bounds(&[0]),
        )
        .unwrap();
        assert_eq!(out.values[0], decimal_to_f64(big, 18) as f32);
    }

    fn decimal_aggregate(
        aggregation: Aggregation,
        terms: &[i128],
        precision: u8,
        scale: i8,
    ) -> Result<f32, CoreError> {
        let n = terms.len();
        let values = Values::Decimal {
            values: terms.iter().copied().map(Some).collect(),
            precision,
            scale,
        };
        aggregate(
            aggregation,
            1,
            &vec![0; n],
            &values,
            &vec![Some(0); n],
            bounds(&[0]),
        )
        .map(|out| out.values[0])
    }

    #[test]
    fn a_decimal_sum_is_held_to_its_column_s_declared_precision() {
        // As the reference's grouped sum of a decimal128(p, s) column
        // (pyarrow 25, measured): DECIMAL(4, 0)'s 9999 + 1 fails, 9998 + 1
        // and 9999 + 1 - 1 are taken; at DECIMAL(4, 2), 99.99 + 0.01 fails.
        for (terms, precision, scale) in [
            (vec![9_999, 1], 4, 0),
            (vec![-9_999, -1], 4, 0),
            (vec![9_999, 1], 4, 2),
            (vec![999_999_999, 1], 9, 0),
        ] {
            let err = decimal_aggregate(Aggregation::Sum, &terms, precision, scale).unwrap_err();
            assert_eq!(
                err.code,
                sdm_contract::ErrorCode::InvalidRequest,
                "{terms:?}"
            );
            assert_eq!(err.fields["field"], sdm_contract::Detail::text("query"));
        }
        assert_eq!(
            decimal_aggregate(Aggregation::Sum, &[9_998, 1], 4, 0).unwrap(),
            9_999.0
        );
        assert_eq!(
            decimal_aggregate(Aggregation::Sum, &[9_999, 1, -1], 4, 0).unwrap(),
            9_999.0
        );
        // Neither the mean nor the largest of the same values is held to it.
        assert_eq!(
            decimal_aggregate(Aggregation::Avg, &[9_999, 1], 4, 0).unwrap(),
            5_000.0
        );
        assert_eq!(
            decimal_aggregate(Aggregation::Max, &[9_999, 1], 4, 0).unwrap(),
            9_999.0
        );
    }

    #[test]
    fn a_float_constant_reads_at_a_decimal_s_scale_as_pyarrow_reads_it() {
        // Measured on the reference's pyarrow: pd.Series([value],
        // dtype=decimal128(10, scale)).
        for (value, scale, wanted) in [
            (0.5, 0, 0),
            (1.5, 0, 2),
            (2.5, 0, 2),
            (-0.5, 0, 0),
            (-1.5, 0, -2),
            (0.7, 0, 1),
            (-0.7, 0, -1),
            (0.15, 1, 1),
            (0.25, 1, 2),
            (0.35, 1, 3),
            (0.1, 1, 1),
            (2.675, 2, 267),
            (1.005, 2, 100),
            (0.045, 2, 4),
            (0.255, 2, 26),
            (1e-40, 2, 0),
            (5e-324, 0, 0),
        ] {
            assert_eq!(
                float_at_scale(value, scale),
                Some(wanted),
                "{value} at {scale}"
            );
        }
        assert_eq!(
            float_at_scale(1e30, 0),
            Some(1_000_000_000_000_000_019_884_624_838_656)
        );
        // At scale 22, the largest Arrow reads exactly (pyarrow 25,
        // decimal128(38, 22)).
        assert_eq!(float_at_scale(0.1, 22), Some(1_000_000_000_000_000_055_511));
        assert_eq!(
            float_at_scale(2.0 / 3.0, 22),
            Some(6_666_666_666_666_666_296_592)
        );
        assert_eq!(
            float_at_scale(-0.7, 22),
            Some(-6_999_999_999_999_999_555_911)
        );
        assert_eq!(
            decimal_constant(Constant::Float(0.1), 38, 22).unwrap(),
            1_000_000_000_000_000_055_511
        );
        assert_eq!(float_at_scale(f64::NAN, 0), None);
        assert_eq!(float_at_scale(f64::INFINITY, 0), None);
        assert_eq!(float_at_scale(1e300, 0), None);
    }

    #[test]
    fn a_static_decimal_is_compared_exactly_and_refused_as_a_target() {
        let column = Column::Decimal {
            values: vec![Some(-250), None, Some(0), Some(25), Some(100)],
            precision: 4,
            scale: 2,
        };
        let condition = |op, constant| Target::Condition {
            target: Box::new(Target::Column(column.clone())),
            op,
            constant,
        };
        let gt_zero = label(&condition(RelOp::Gt, Some(Constant::Int(0)))).unwrap();
        assert_eq!(gt_zero.mask, vec![true, false, true, true, true]);
        assert_eq!(gt_zero.labels, Labels::Bool(vec![false, false, true, true]));
        let ge_quarter = label(&condition(RelOp::Ge, Some(Constant::Float(0.25)))).unwrap();
        assert_eq!(
            ge_quarter.labels,
            Labels::Bool(vec![false, false, true, true])
        );
        let is_null = label(&condition(RelOp::Eq, None)).unwrap();
        assert_eq!(is_null.mask, vec![true; 5]);
        assert_eq!(
            is_null.labels,
            Labels::Bool(vec![false, true, false, false, false])
        );
        // A constant DECIMAL(4, 2) does not hold is the caller's error, as
        // the reference refuses it (measured: users.f > 100, INVALID_REQUEST
        // at materialize); an integer past int64 is refused by name. A float
        // at a scale past 22 is read, not refused (the owner's decision on
        // P1B-71, 2026-10-13): `a_float_constant_reads_exactly_at_scales_23_to_38`
        // and the tests after it hold that.
        for (constant, precision, scale, code) in [
            (Constant::Int(100), 4, 2, ErrorCode::InvalidRequest),
            (Constant::Float(100.0), 4, 2, ErrorCode::InvalidRequest),
            (Constant::Float(1e300), 4, 2, ErrorCode::InvalidRequest),
            (Constant::Int(5), 38, 38, ErrorCode::InvalidRequest),
            (
                Constant::Int(i128::from(i64::MAX) + 1),
                38,
                0,
                ErrorCode::UnsupportedRequest,
            ),
        ] {
            let column = Column::Decimal {
                values: vec![Some(1)],
                precision,
                scale,
            };
            let err = label(&Target::Condition {
                target: Box::new(Target::Column(column)),
                op: RelOp::Gt,
                constant: Some(constant),
            })
            .unwrap_err();
            assert_eq!(err.code, code, "{constant:?}");
            if code == ErrorCode::UnsupportedRequest {
                assert_eq!(
                    err.fields["rule"],
                    sdm_contract::Detail::text("condition_value")
                );
            } else {
                assert_eq!(err.fields["field"], sdm_contract::Detail::text("query"));
            }
        }
        // Asked for itself, a static decimal target is refused by name, as
        // the reference's task table refuses it.
        let err = label(&Target::Column(column)).unwrap_err();
        assert_eq!(err.code, sdm_contract::ErrorCode::UnsupportedRequest);
        assert_eq!(
            err.fields["rule"],
            sdm_contract::Detail::text("target_type")
        );
    }

    // --- T1.32 tests: a float constant at a decimal's scale past 22 -------
    //
    // The owner's decision on P1B-71 (2026-10-13): a float constant compared
    // with a static DECIMAL(p, s) column at a scale of 23 to 38 is read from
    // the float's exact binary value at the column's scale, rounded half to
    // even, as it is through scale 22; no refusal. Each expected value was
    // computed apart from the core, in Python:
    // `int((decimal.Decimal(value) * 10**s).quantize(1, ROUND_HALF_EVEN))`
    // under a context of 2000 digits. From scale 28 the reference's pyarrow
    // approximates; the core is exact there (P1B-71 rewritten as that
    // difference), so these are held to the exact value, not to pyarrow's.
    //
    // Gating: each is named in `contract/tests/t132b-pending.json`'s
    // `pending["exits"]` as `rust:<test name>`, failing when written (the
    // core refused every float past scale 22 by name, `condition_value`);
    // a pending test must still fail, any other must pass.

    fn t132b_pending(test: &str) -> bool {
        // Read at compile time: the core does no I/O, its tests included.
        let text = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contract/tests/t132b-pending.json"
        ));
        let gating: serde_json::Value = serde_json::from_str(text).unwrap();
        gating["pending"]["exits"]
            .as_array()
            .unwrap()
            .iter()
            .any(|t| t == &serde_json::Value::from(format!("rust:{test}")))
    }

    /// Runs `check`; a pending test must fail, any other must pass.
    fn t132b_gated(test: &str, check: impl FnOnce() + std::panic::UnwindSafe) {
        let outcome = std::panic::catch_unwind(check);
        match (outcome, t132b_pending(test)) {
            (Ok(()), true) => panic!(
                "{test} passes now: take rust:{test} out of \
                 contract/tests/t132b-pending.json's pending[\"exits\"]"
            ),
            (Err(failure), false) => std::panic::resume_unwind(failure),
            _ => {}
        }
    }

    fn read_at(value: f64, precision: u8, scale: i8) -> Result<i128, CoreError> {
        decimal_constant(Constant::Float(value), precision, scale)
    }

    #[test]
    fn a_float_constant_reads_exactly_at_scales_23_to_38() {
        t132b_gated("a_float_constant_reads_exactly_at_scales_23_to_38", || {
            for (value, precision, scale, wanted) in [
                (0.1, 38, 23, 10_000_000_000_000_000_555_112_i128),
                (-0.1, 38, 23, -10_000_000_000_000_000_555_112),
                (0.25, 38, 23, 25_000_000_000_000_000_000_000),
                (0.7, 38, 23, 69_999_999_999_999_995_559_108),
                (-0.7, 38, 23, -69_999_999_999_999_995_559_108),
                (2.0 / 3.0, 38, 23, 66_666_666_666_666_662_965_923),
                (123.456, 38, 23, 12_345_600_000_000_000_306_954_462),
                (0.1, 38, 28, 1_000_000_000_000_000_055_511_151_231),
                (0.25, 38, 28, 2_500_000_000_000_000_000_000_000_000),
                (0.7, 38, 28, 6_999_999_999_999_999_555_910_790_150),
                (-0.7, 38, 28, -6_999_999_999_999_999_555_910_790_150),
                (2.0 / 3.0, 38, 28, 6_666_666_666_666_666_296_592_325_125),
                (123.456, 38, 28, 1_234_560_000_000_000_030_695_446_184_836),
                // A precision under 38 at a scale past 22.
                (0.7, 30, 28, 6_999_999_999_999_999_555_910_790_150),
                (0.1, 38, 30, 100_000_000_000_000_005_551_115_123_126),
                (0.25, 38, 30, 250_000_000_000_000_000_000_000_000_000),
                (0.7, 38, 30, 699_999_999_999_999_955_591_079_014_994),
                (-0.7, 38, 30, -699_999_999_999_999_955_591_079_014_994),
                (2.0 / 3.0, 38, 30, 666_666_666_666_666_629_659_232_512_495),
                (123.456, 38, 30, 123_456_000_000_000_003_069_544_618_483_633),
                (
                    0.1,
                    38,
                    38,
                    10_000_000_000_000_000_555_111_512_312_578_270_212,
                ),
                (
                    -0.1,
                    38,
                    38,
                    -10_000_000_000_000_000_555_111_512_312_578_270_212,
                ),
                (
                    0.25,
                    38,
                    38,
                    25_000_000_000_000_000_000_000_000_000_000_000_000,
                ),
                (
                    0.7,
                    38,
                    38,
                    69_999_999_999_999_995_559_107_901_499_373_838_305,
                ),
                (
                    -0.7,
                    38,
                    38,
                    -69_999_999_999_999_995_559_107_901_499_373_838_305,
                ),
                (
                    2.0 / 3.0,
                    38,
                    38,
                    66_666_666_666_666_662_965_923_251_249_478_198_588,
                ),
                (
                    0.99999,
                    38,
                    38,
                    99_999_000_000_000_004_551_026_222_543_441_690_505,
                ),
                // The largest DECIMAL(38, 37) holds is under 10: 9 is read.
                (
                    9.0,
                    38,
                    37,
                    90_000_000_000_000_000_000_000_000_000_000_000_000,
                ),
                // Under half the last place, zero; over it, one.
                (5e-324, 38, 38, 0),
                (4e-39, 38, 38, 0),
                (6e-39, 38, 38, 1),
            ] {
                let read = read_at(value, precision, scale);
                assert!(
                    read.is_ok(),
                    "{value} at DECIMAL({precision}, {scale}) refused: {read:?}"
                );
                assert_eq!(
                    read.unwrap(),
                    wanted,
                    "{value} at DECIMAL({precision}, {scale})"
                );
            }
        });
    }

    #[test]
    fn a_float_constant_half_way_past_scale_22_rounds_to_even() {
        t132b_gated(
            "a_float_constant_half_way_past_scale_22_rounds_to_even",
            || {
                // 2**-(s + 1) and 3 * 2**-(s + 1) have exactly s + 1
                // decimal places, the last a 5: each lies exactly half-way
                // at scale s, and of the two neighbours the even one is
                // taken (5**s is odd, so one rounds down and one up).
                for (value, scale, wanted) in [
                    (2.0_f64.powi(-24), 23, 5_960_464_477_539_062_i128),
                    (3.0 * 2.0_f64.powi(-24), 23, 17_881_393_432_617_188),
                    (2.0_f64.powi(-29), 28, 18_626_451_492_309_570_312),
                    (3.0 * 2.0_f64.powi(-29), 28, 55_879_354_476_928_710_938),
                    (2.0_f64.powi(-31), 30, 465_661_287_307_739_257_812),
                    (3.0 * 2.0_f64.powi(-31), 30, 1_396_983_861_923_217_773_438),
                    (2.0_f64.powi(-39), 38, 181_898_940_354_585_647_583_007_812),
                    (
                        3.0 * 2.0_f64.powi(-39),
                        38,
                        545_696_821_063_756_942_749_023_438,
                    ),
                ] {
                    for sign in [1.0, -1.0] {
                        let read = read_at(sign * value, 38, scale);
                        assert!(
                            read.is_ok(),
                            "{} at DECIMAL(38, {scale}) refused: {read:?}",
                            sign * value
                        );
                        assert_eq!(
                            read.unwrap(),
                            if sign < 0.0 { -wanted } else { wanted },
                            "{} at DECIMAL(38, {scale})",
                            sign * value
                        );
                    }
                }
            },
        );
    }

    #[test]
    fn a_float_constant_its_decimal_cannot_hold_past_scale_22_is_the_caller_s_error() {
        t132b_gated(
            "a_float_constant_its_decimal_cannot_hold_past_scale_22_is_the_caller_s_error",
            || {
                // As at scale 22 and under: a constant past the column's
                // digits is INVALID_REQUEST naming the query, never a
                // refusal by name. Past the precision but inside i128
                // (12345.678 at DECIMAL(38, 34): 39 digits), and past i128.
                for (value, precision, scale) in [
                    (1.0, 38, 38),
                    (-1.0, 38, 38),
                    (100.0, 30, 28),
                    (12_345.678, 38, 34),
                    (123.456, 38, 38),
                    (1e16, 38, 23),
                    (-1e16, 38, 23),
                    (1e300, 38, 30),
                ] {
                    let err = read_at(value, precision, scale).unwrap_err();
                    assert_eq!(
                        err.code,
                        ErrorCode::InvalidRequest,
                        "{value} at DECIMAL({precision}, {scale})"
                    );
                    assert_eq!(
                        err.fields.get("field"),
                        Some(&sdm_contract::Detail::text("query")),
                        "{value} at DECIMAL({precision}, {scale})"
                    );
                    assert!(
                        !err.fields.contains_key("rule"),
                        "{value} at DECIMAL({precision}, {scale}) refused by name"
                    );
                }
            },
        );
    }

    #[test]
    fn a_static_decimal_past_scale_22_is_compared_with_a_float_exactly() {
        t132b_gated(
            "a_static_decimal_past_scale_22_is_compared_with_a_float_exactly",
            || {
                // users.f > 0.7 and users.f = 0.7 at DECIMAL(38, 30): the
                // constant is the double nearest 0.7 read exactly,
                // 0.699999999999999955591079014994 at scale 30, so the
                // column's 0.7 is greater, and only the exact reading equals.
                let exact = 699_999_999_999_999_955_591_079_014_994_i128;
                let column = Column::Decimal {
                    values: vec![
                        Some(700_000_000_000_000_000_000_000_000_000),
                        Some(exact),
                        Some(exact - 1),
                        None,
                        Some(exact + 1),
                    ],
                    precision: 38,
                    scale: 30,
                };
                let condition = |op| Target::Condition {
                    target: Box::new(Target::Column(column.clone())),
                    op,
                    constant: Some(Constant::Float(0.7)),
                };
                let gt = label(&condition(RelOp::Gt));
                assert!(gt.is_ok(), "refused: {gt:?}");
                let gt = gt.unwrap();
                assert_eq!(gt.mask, vec![true, true, true, false, true]);
                assert_eq!(gt.labels, Labels::Bool(vec![true, false, false, true]));
                let eq = label(&condition(RelOp::Eq)).unwrap();
                assert_eq!(eq.labels, Labels::Bool(vec![false, true, false, false]));
                // At DECIMAL(38, 23), the scale the diagnostic test refused
                // (P1B-71 before the owner's decision): 0.5 is read.
                let column = Column::Decimal {
                    values: vec![Some(50_000_000_000_000_000_000_000), Some(1), None],
                    precision: 38,
                    scale: 23,
                };
                let ge = label(&Target::Condition {
                    target: Box::new(Target::Column(column)),
                    op: RelOp::Ge,
                    constant: Some(Constant::Float(0.5)),
                });
                assert!(ge.is_ok(), "refused: {ge:?}");
                let ge = ge.unwrap();
                assert_eq!(ge.mask, vec![true, true, false]);
                assert_eq!(ge.labels, Labels::Bool(vec![true, false]));
            },
        );
    }

    #[test]
    fn a_decimal_mean_is_rounded_half_away_from_zero_at_its_scale() {
        // The reference's grouped means (pyarrow 25, measured), then read at
        // float32 as pandas' astype reads them.
        for (terms, scale, wanted) in [
            (vec![1, 2], 0, 2.0_f32),
            (vec![-1, -2], 0, -2.0),
            (vec![1, 2, 2], 0, 2.0),
            (vec![1, 1, 2], 0, 1.0),
            (vec![1, 2], 2, decimal_to_f64(2, 2) as f32),
            (vec![10, 20, 20], 2, decimal_to_f64(17, 2) as f32),
            (vec![-10, -20, -20], 2, decimal_to_f64(-17, 2) as f32),
        ] {
            assert_eq!(
                decimal_aggregate(Aggregation::Avg, &terms, 4, scale).unwrap(),
                wanted,
                "{terms:?}"
            );
        }
        // A total past DECIMAL(38) is averaged: (10**38 - 1 + 1) / 2.
        let nines = 99_999_999_999_999_999_999_999_999_999_999_999_999_i128;
        assert_eq!(
            decimal_aggregate(Aggregation::Avg, &[nines, 1], 38, 0).unwrap(),
            decimal_to_f64(50_000_000_000_000_000_000_000_000_000_000_000_000, 0) as f32
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
