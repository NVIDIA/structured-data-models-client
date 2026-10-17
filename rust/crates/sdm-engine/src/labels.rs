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
//!   PK-2), of an integer in int64, of a decimal exactly at any size
//!   (T2.0c); `COUNT` the rows; `AVG` a float's
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
//!   at int64 for an integer target and an integer constant; a decimal
//!   target exactly, at no scale but their own (T2.0c, KO2-50).
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

/// The exact sum of decimals' unscaled values, in 256 bits, two's
/// complement (`high * 2**128 + low`): no sum of `i128`s a request holds
/// passes it, so nothing wraps and nothing is refused (T2.0c, exact
/// decimals: the reference since `main`'s eb6c68eb sums a decimal column
/// as float64 and answers a sum past 38 digits).
#[derive(Clone, Copy, Default)]
struct ExactSum {
    high: i128,
    low: u128,
}

impl ExactSum {
    fn of(values: impl Iterator<Item = i128>) -> Self {
        values.fold(Self::default(), |sum, value| {
            let (low, carry) = sum.low.overflowing_add(value as u128);
            let high = sum.high + if value < 0 { -1 } else { 0 } + i128::from(carry);
            Self { high, low }
        })
    }

    /// The sum as an `i128`, where it is one.
    fn narrow(self) -> Option<i128> {
        let value = self.low as i128;
        let fits = (self.high == 0 && value >= 0) || (self.high == -1 && value < 0);
        fits.then_some(value)
    }

    /// Whether the sum is negative, and its magnitude as four 64-bit limbs,
    /// the most significant first.
    fn magnitude(self) -> (bool, [u64; 4]) {
        let negative = self.high < 0;
        let (mut high, mut low) = (self.high as u128, self.low);
        if negative {
            low = !low;
            high = !high;
            let (sum, carry) = low.overflowing_add(1);
            low = sum;
            high = high.wrapping_add(u128::from(carry));
        }
        (
            negative,
            [
                (high >> 64) as u64,
                high as u64,
                (low >> 64) as u64,
                low as u64,
            ],
        )
    }
}

/// `limbs / divisor` and the remainder.
fn divided(limbs: [u64; 4], divisor: u64) -> ([u64; 4], u64) {
    let mut quotient = [0_u64; 4];
    let mut remainder = 0_u128;
    for (at, limb) in limbs.iter().enumerate() {
        let current = (remainder << 64) | u128::from(*limb);
        quotient[at] = (current / u128::from(divisor)) as u64;
        remainder = current % u128::from(divisor);
    }
    (quotient, remainder as u64)
}

/// The limbs' decimal digits.
fn digits(mut limbs: [u64; 4]) -> String {
    const CHUNK: u64 = 10_000_000_000_000_000_000;
    let mut chunks = Vec::new();
    while limbs != [0; 4] {
        let (quotient, remainder) = divided(limbs, CHUNK);
        chunks.push(remainder);
        limbs = quotient;
    }
    match chunks.split_last() {
        None => "0".to_owned(),
        Some((first, rest)) => {
            let mut text = first.to_string();
            for chunk in rest.iter().rev() {
                text.push_str(&format!("{chunk:019}"));
            }
            text
        }
    }
}

/// A decimal past `i128` at `scale`, rounded once to the nearest double.
fn wide_to_f64(negative: bool, limbs: [u64; 4], scale: i8) -> f64 {
    let sign = if negative { "-" } else { "" };
    format!("{sign}{}e{}", digits(limbs), -i32::from(scale))
        .parse()
        .unwrap_or(f64::NAN)
}

/// The exact sum of decimals of `scale`, read at float32: within `i128`
/// as [`decimal_to_f64`] reads a decimal, past it rounded once to the
/// nearest double. A sum past the column's declared digits is a sum like
/// any other (T2.0c).
fn decimal_sum(values: impl Iterator<Item = i128>, scale: i8) -> f32 {
    let sum = ExactSum::of(values);
    match sum.narrow() {
        Some(sum) => decimal_to_f64(sum, scale) as f32,
        None => {
            let (negative, limbs) = sum.magnitude();
            wide_to_f64(negative, limbs, scale) as f32
        }
    }
}

/// The mean of `n` decimals, at their scale, as Arrow's grouped mean of a
/// `Decimal128` gives it: the quotient, moved one away from zero, by the
/// total's sign, where the remainder is at least half the count; the
/// total is exact whatever its size, so the mean is.
fn decimal_mean(values: impl Iterator<Item = i128>, n: usize) -> i128 {
    let (negative, limbs) = ExactSum::of(values).magnitude();
    let (quotient, remainder) = divided(limbs, n as u64);
    let mut magnitude = (u128::from(quotient[2]) << 64) | u128::from(quotient[3]);
    if u128::from(remainder) * 2 >= n as u128 {
        magnitude += 1;
    }
    // A mean is within its values' range, so within `i128`.
    let magnitude = magnitude as i128;
    if negative { -magnitude } else { magnitude }
}

/// `aggregation` of `values` for each of `examples` examples: the rows of
/// example `batch[row]` whose value is present and whose time is in its
/// window ([`Bounds`]), in row order. A decimal's sum and average are
/// exact whatever their size ([`decimal_sum`], [`decimal_mean`]).
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
                    values: v, scale, ..
                } => decimal_sum(rows.iter().map(|&r| v[r].unwrap_or(0)), *scale),
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
                } => decimal_to_f64(
                    decimal_mean(rows.iter().map(|&r| v[r].unwrap_or(0)), rows.len()),
                    *scale,
                ) as f32,
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

/// A condition's constant as a decimal, `digits * 10**exponent`: an
/// integer as written; a float as the shortest decimal that names its
/// double, which is the constant the query wrote whenever it wrote no more
/// than a double's 17 significant digits; `None` for a float that is not
/// finite.
fn constant_decimal(constant: Constant) -> Option<(i128, i32)> {
    match constant {
        Constant::Int(value) => Some((value, 0)),
        Constant::Float(value) if value.is_finite() => {
            // `{:e}` writes the shortest such decimal: `1.5e0`, `-1e308`.
            let text = format!("{value:e}");
            let (mantissa, exponent) = text.split_once('e')?;
            let exponent: i32 = exponent.parse().ok()?;
            let fraction = mantissa.split_once('.').map_or(0, |(_, f)| f.len());
            let digits: i128 = mantissa.replace('.', "").parse().ok()?;
            Some((digits, exponent - fraction as i32))
        }
        Constant::Float(_) => None,
    }
}

/// How a decimal of unscaled `value` at `scale` orders against `constant`,
/// exactly, at no scale but their own (T2.0c, KO2-50: `d > 1.5` holds for
/// a `DECIMAL(38, 0)` `d` of 2): the constant is never rounded to the
/// column's scale, and one past the column's digits orders by its sign.
/// An infinite constant orders past every value; a NaN, which the PQL
/// validator refuses before it reaches here, orders against none.
fn decimal_order(value: i128, scale: i8, constant: Constant) -> Option<std::cmp::Ordering> {
    use std::cmp::Ordering;
    let Some((digits, exponent)) = constant_decimal(constant) else {
        return match constant {
            Constant::Float(f) if f == f64::INFINITY => Some(Ordering::Less),
            Constant::Float(f) if f == f64::NEG_INFINITY => Some(Ordering::Greater),
            _ => None,
        };
    };
    // value * 10**-scale against digits * 10**exponent: the constant at the
    // value's scale is digits * 10**shift.
    let shift = exponent + i32::from(scale);
    // Past every value a 38-digit decimal holds.
    let beyond = || {
        Some(match digits.signum() {
            1 => Ordering::Less,
            -1 => Ordering::Greater,
            _ => value.cmp(&0),
        })
    };
    if shift >= 0 {
        return match 10_i128
            .checked_pow(shift as u32)
            .and_then(|unit| digits.checked_mul(unit))
        {
            Some(scaled) => Some(value.cmp(&scaled)),
            None => beyond(),
        };
    }
    // The constant at the value's scale lies in [floor, floor + 1), at floor
    // exactly when nothing is left over.
    let (floor, exact) = match 10_i128.checked_pow(shift.unsigned_abs()) {
        Some(unit) => (digits.div_euclid(unit), digits.rem_euclid(unit) == 0),
        // A unit past i128: the constant's digits are all below it.
        None => (if digits < 0 { -1 } else { 0 }, digits == 0),
    };
    Some(match value.cmp(&floor) {
        Ordering::Equal if !exact => Ordering::Less,
        other => other,
    })
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

/// A windowed aggregation compared with `= NULL` or `!= NULL`, as the
/// executor compares it: the examples whose window holds a value keep a
/// label, each `False` for `=` and `True` for `!=` (it drops the others
/// before it compares); the rest have none. Where some have none, the
/// reference's `execute` fails after this, which its caller refuses
/// (register P1B-21).
fn aggregated_null(present: Vec<bool>, op: RelOp) -> Result<Labelled, CoreError> {
    let label = match op {
        RelOp::Eq => false,
        RelOp::Ne => true,
        _ => return Err(not_yet("condition")),
    };
    let n = present.iter().filter(|p| **p).count();
    Ok(Labelled {
        mask: present,
        labels: Labels::Bool(vec![label; n]),
    })
}

/// A static decimal column compared with a constant: exactly
/// ([`decimal_order`]); the examples with a value keep their answer.
fn decimal_condition(
    values: &[Option<i128>],
    scale: i8,
    op: RelOp,
    constant: Option<Constant>,
) -> Result<Labelled, CoreError> {
    let mask: Vec<bool> = values.iter().map(Option::is_some).collect();
    let Some(constant) = constant else {
        return null_condition(mask, op);
    };
    Ok(Labelled {
        labels: Labels::Bool(
            values
                .iter()
                .flatten()
                .map(|&v| {
                    decimal_order(v, scale, constant).is_some_and(|order| match op {
                        RelOp::Eq => order.is_eq(),
                        RelOp::Ne => order.is_ne(),
                        RelOp::Lt => order.is_lt(),
                        RelOp::Le => order.is_le(),
                        RelOp::Gt => order.is_gt(),
                        RelOp::Ge => order.is_ge(),
                    })
                })
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
    /// scale: compared in a condition ([`decimal_condition`]), or a target
    /// ([`decimal_target`]).
    Decimal {
        values: Vec<Option<i128>>,
        precision: u8,
        scale: i8,
    },
}

/// A decimal column as a target, read as the reference reads it since
/// `main`'s eb6c68eb, so a class is spelled as the reference spells it:
/// at scale 0, where every value fits, an int64; otherwise each value at
/// float32 (float64, then float32, as [`decimal_to_f64`] reads it), and
/// where two distinct values meet at one float32, each value's own digits
/// instead, so no two values become one class (T2.0c).
fn decimal_target(values: &[Option<i128>], scale: i8) -> (Vec<bool>, Labels) {
    let mask: Vec<bool> = values.iter().map(Option::is_some).collect();
    let present: Vec<i128> = values.iter().flatten().copied().collect();
    if scale == 0 {
        if let Ok(ints) = present
            .iter()
            .map(|&v| i64::try_from(v))
            .collect::<Result<Vec<_>, _>>()
        {
            return (mask, Labels::Int(ints));
        }
    }
    let floats: Vec<f32> = present
        .iter()
        .map(|&v| decimal_to_f64(v, scale) as f32)
        .collect();
    let mut seen: std::collections::BTreeMap<u32, i128> = std::collections::BTreeMap::new();
    let merged = present.iter().zip(&floats).any(|(&value, float)| {
        // Adding zero makes a negative zero positive.
        *seen.entry((float + 0.0).to_bits()).or_insert(value) != value
    });
    let labels = if merged {
        Labels::Text(present.iter().map(|&v| decimal_text(v, scale)).collect())
    } else {
        Labels::Float(floats)
    };
    (mask, labels)
}

/// A decimal's own digits: its unscaled `value` at `scale`.
fn decimal_text(value: i128, scale: i8) -> String {
    let digits = value.unsigned_abs().to_string();
    let sign = if value < 0 { "-" } else { "" };
    let Ok(scale) = usize::try_from(scale) else {
        return format!(
            "{sign}{digits}{}",
            "0".repeat(usize::from(scale.unsigned_abs()))
        );
    };
    if scale == 0 {
        return format!("{sign}{digits}");
    }
    let padded = format!("{digits:0>width$}", width = scale + 1);
    let (whole, fraction) = padded.split_at(padded.len() - scale);
    format!("{sign}{whole}.{fraction}")
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
                Column::Decimal { values, scale, .. } => decimal_target(values, *scale),
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
            if let Target::Column(Column::Decimal { values, scale, .. }) = target.as_ref() {
                return decimal_condition(values, *scale, *op, *constant);
            }
            let inner = label(target)?;
            match constant {
                Some(constant) => Ok(Labelled {
                    labels: Labels::Bool(compare(&inner.labels, *op, *constant)?),
                    mask: inner.mask,
                }),
                None if matches!(target.as_ref(), Target::Aggregated(_)) => {
                    aggregated_null(inner.mask, *op)
                }
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
    fn a_decimal_sum_past_38_digits_is_summed_exactly() {
        // Past the column's 38 digits, and past i128, the total is exact
        // and read once at float32 (T2.0c; the reference since eb6c68eb
        // answers these as float64 sums).
        let big = 60_000_000_000_000_000_000_000_000_000_000_000_000_i128;
        let nine = 90_000_000_000_000_000_000_000_000_000_000_000_000_i128;
        for (terms, scale, wanted) in [
            (vec![big, big], 18, 1.2e20_f32),
            (vec![big, big, big, big], 18, 2.4e20),
            (vec![-big, -big, -big], 18, -1.8e20),
            (vec![big, big, -big], 18, 6e19),
            (vec![big, -big, big], 18, 6e19),
            (vec![big, big, -big, -big], 18, 0.0),
            (vec![nine, nine, -nine], 0, 9e37),
            (vec![nine, nine], 0, 1.8e38),
        ] {
            assert_eq!(
                decimal_aggregate(Aggregation::Sum, &terms, 38, scale).unwrap(),
                wanted,
                "{terms:?}"
            );
        }
        // Past the column's declared digits, within i128: summed.
        assert_eq!(
            decimal_aggregate(Aggregation::Sum, &[9_999, 1], 4, 0).unwrap(),
            10_000.0
        );
        assert_eq!(
            decimal_aggregate(Aggregation::Sum, &[9_999, 1], 4, 2).unwrap(),
            100.0
        );
        assert_eq!(
            decimal_aggregate(Aggregation::Sum, &[-9_999, -1], 4, 0).unwrap(),
            -10_000.0
        );
        // Their means are exact whatever the total: past i128 too.
        assert_eq!(
            decimal_aggregate(Aggregation::Avg, &[big, big], 38, 18).unwrap(),
            decimal_to_f64(big, 18) as f32
        );
        assert_eq!(
            decimal_aggregate(Aggregation::Avg, &[nine, nine, -nine], 38, 0).unwrap(),
            3e37
        );
        assert_eq!(
            decimal_aggregate(Aggregation::Avg, &[9_999, 1], 4, 0).unwrap(),
            5_000.0
        );
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
    fn a_static_decimal_is_compared_exactly_and_answered_as_a_target() {
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
        // Asked for itself, each value at float32, as the reference reads
        // it since eb6c68eb (at scale 0, an int64); distinct values meeting
        // at one float32 keep their own digits instead, one class each.
        assert_eq!(
            label(&Target::Column(column.clone())).unwrap(),
            Labelled {
                mask: vec![true, false, true, true, true],
                labels: Labels::Float(vec![-2.5, 0.0, 0.25, 1.0]),
            }
        );
        let whole = Column::Decimal {
            values: vec![Some(3), Some(-1), None],
            precision: 4,
            scale: 0,
        };
        assert_eq!(
            label(&Target::Column(whole)).unwrap().labels,
            Labels::Int(vec![3, -1])
        );
        let close = Column::Decimal {
            values: vec![Some(100_000_000_000), Some(100_000_000_001), None],
            precision: 38,
            scale: 18,
        };
        assert_eq!(
            label(&Target::Column(close)).unwrap().labels,
            Labels::Text(vec![
                "0.000000100000000000".to_owned(),
                "0.000000100000000001".to_owned(),
            ])
        );
    }

    /// `d op constant` over one `DECIMAL(38, scale)` value of unscaled `d`.
    fn holds(d: i128, scale: i8, op: RelOp, constant: Constant) -> bool {
        let labelled = label(&Target::Condition {
            target: Box::new(Target::Column(Column::Decimal {
                values: vec![Some(d)],
                precision: 38,
                scale,
            })),
            op,
            constant: Some(constant),
        })
        .unwrap();
        labelled.labels == Labels::Bool(vec![true])
    }

    #[test]
    fn a_decimal_is_compared_with_a_constant_of_any_scale_exactly() {
        use Constant::{Float, Int};
        use RelOp::{Eq, Ge, Gt, Le, Lt, Ne};
        // KO2-50: the constant is never rounded to the column's scale.
        assert!(holds(2, 0, Gt, Float(1.5)));
        assert!(!holds(1, 0, Gt, Float(1.5)));
        assert!(holds(1, 0, Lt, Float(1.5)));
        assert!(!holds(2, 0, Eq, Float(1.5)) && holds(2, 0, Ne, Float(1.5)));
        assert!(holds(-1, 0, Gt, Float(-1.5)) && !holds(-2, 0, Gt, Float(-1.5)));
        assert!(holds(-2, 0, Lt, Float(-1.5)) && holds(-2, 0, Le, Float(-1.5)));
        // At the column's scale, and finer: the literal the query wrote.
        assert!(holds(15, 1, Eq, Float(1.5)) && holds(15, 1, Ge, Float(1.5)));
        assert!(holds(1, 1, Eq, Float(0.1)));
        assert!(holds(100_000_000_000_000_000, 18, Eq, Float(0.1)));
        assert!(holds(15, 2, Lt, Float(0.155)) && holds(16, 2, Gt, Float(0.155)));
        assert!(holds(0, 0, Lt, Float(5e-324)) && holds(0, 0, Gt, Float(-5e-324)));
        assert!(holds(5, 38, Gt, Float(4.9e-38)) && holds(5, 38, Lt, Float(5.1e-38)));
        // Integers, exactly, past the column's digits and past int64.
        assert!(holds(100, 2, Eq, Int(1)) && holds(100, 2, Le, Int(1)));
        assert!(holds(99_999, 0, Lt, Int(i128::from(i64::MAX) + 1)));
        // A constant past every value the column holds orders by its sign.
        let largest = 10_i128.pow(38) - 1;
        assert!(holds(largest, 0, Lt, Float(1e308)) && holds(-largest, 0, Gt, Float(-1e308)));
        assert!(holds(largest, 0, Lt, Float(1e39)) && !holds(largest, 0, Ge, Float(1e39)));
        assert!(holds(1, 0, Lt, Float(f64::INFINITY)));
        assert!(holds(1, 0, Gt, Float(f64::NEG_INFINITY)));
        assert!(!holds(1, 0, Eq, Float(f64::INFINITY)));
        // A NaN, which the validator refuses first, holds no comparison.
        for op in [Eq, Ne, Lt, Le, Gt, Ge] {
            assert!(!holds(1, 0, op, Float(f64::NAN)));
        }
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
