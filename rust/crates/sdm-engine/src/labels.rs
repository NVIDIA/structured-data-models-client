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
//!   decimal's exact sum over the count, at any size; `MIN` and `MAX`
//!   exactly; then each read at float32, a decimal's exact value rounded
//!   once to the nearest float32 (T2.0c).
//!   An example with no row is 0 for `SUM` and `COUNT` and has no label
//!   for the others.
//! - A condition (`column_compare`, `execute_rel_op`) compares the values its
//!   target has with a constant read at the target's type: at float32 for a
//!   float32 target, at float64 for an integer target and a float constant,
//!   at int64 for an integer target and an integer constant; a decimal
//!   target exactly, at no scale but their own (T2.0c, KO2-50); a boolean,
//!   text or time with its own kind. `IN` (`execute_member_op`) reads its
//!   list at the target's type the same way, then tests membership.
//! - `NOT`, `AND` and `OR` ([`truth`], `execute_logical_operation`) join
//!   their conditions' answers row by row, and label only the rows every
//!   condition has a value for; a target's filter (`execute_filter`) keeps
//!   the rows its condition holds for, which [`aggregate_kept`] aggregates
//!   (T2.10).
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

/// The `f64` Arrow's `Decimal128::ToDouble` gives a decimal, shared with
/// inference (`sdm-schema`), which counts a scaled decimal as the
/// reference's cast to float64 gives it (T2.14b).
pub use sdm_contract::scalar::decimal_to_f64;

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

/// `sign * limbs / (divisor * 10**scale)`, exactly, rounded once to the
/// nearest float32 (ties to even): the quotient's digits, then enough of
/// its fraction's for any float32, then a last nonzero digit where any of
/// it is left, so the one rounding Rust's parse makes sees the exact value
/// on the right side of every midpoint.
fn exact_to_f32(negative: bool, limbs: [u64; 4], divisor: u64, scale: i8) -> f32 {
    let divisor = divisor.max(1);
    let (quotient, mut remainder) = divided(limbs, divisor);
    let mut fraction = String::new();
    // The divisor is under 2**64, so the fraction's first nonzero digit is
    // within its first 20, and 40 more are past what a float32 resolves.
    for _ in 0..60 {
        if remainder == 0 {
            break;
        }
        let next = u128::from(remainder) * 10;
        fraction.push(char::from(b'0' + (next / u128::from(divisor)) as u8));
        remainder = (next % u128::from(divisor)) as u64;
    }
    if remainder != 0 {
        fraction.push('1');
    }
    if fraction.is_empty() {
        fraction.push('0');
    }
    let sign = if negative { "-" } else { "" };
    format!(
        "{sign}{}.{fraction}e{}",
        digits(quotient),
        -i32::from(scale)
    )
    .parse()
    .unwrap_or(f32::NAN)
}

/// A decimal of unscaled `value` at `scale`, rounded once to the nearest
/// float32.
fn decimal_to_f32(value: i128, scale: i8) -> f32 {
    let (negative, limbs) = ExactSum::of(std::iter::once(value)).magnitude();
    exact_to_f32(negative, limbs, 1, scale)
}

/// The exact sum of decimals of `scale`, at any size, rounded once to the
/// nearest float32. A sum past the column's declared digits is a sum like
/// any other (T2.0c).
fn decimal_sum(values: impl Iterator<Item = i128>, scale: i8) -> f32 {
    let (negative, limbs) = ExactSum::of(values).magnitude();
    exact_to_f32(negative, limbs, 1, scale)
}

/// The exact mean of `n` decimals of `scale`, their exact sum over the
/// count, at any size, rounded once to the nearest float32: never first at
/// the column's scale (T2.0c, round 1).
fn decimal_mean(values: impl Iterator<Item = i128>, n: usize, scale: i8) -> f32 {
    let (negative, limbs) = ExactSum::of(values).magnitude();
    exact_to_f32(negative, limbs, n as u64, scale)
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
    aggregate_kept(aggregation, examples, batch, values, times, bounds, None)
}

/// [`aggregate`] over the rows a target's filter keeps (`kept`, one for
/// each row, from [`truth`]'s values), as `execute_filter` hands
/// `execute_aggregation` only the rows whose condition holds and whose
/// value is present; every row where `kept` is `None`.
pub fn aggregate_kept(
    aggregation: Aggregation,
    examples: usize,
    batch: &[usize],
    values: &Values,
    times: &[Option<i64>],
    bounds: Bounds<'_>,
    kept: Option<&[bool]>,
) -> Result<Aggregated, CoreError> {
    let rows = batch.len();
    if values.len() != rows
        || times.len() != rows
        || bounds.ends.len() != examples
        || kept.is_some_and(|k| k.len() != rows)
    {
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
        if values.present(row) && kept.is_none_or(|k| k[row]) && within(row) {
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
                } => decimal_mean(rows.iter().map(|&r| v[r].unwrap_or(0)), rows.len(), *scale),
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
                        decimal_to_f32(best.unwrap_or(0), *scale)
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

/// How `AND` and `OR` join two conditions (`execute_bool_op`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoolOp {
    And,
    Or,
}

/// A condition's constant, as the query wrote it.
#[derive(Debug, Clone, PartialEq)]
pub enum Constant {
    Int(i128),
    Float(f64),
    Bool(bool),
    Text(String),
    /// A date literal's time, nanoseconds since the epoch, with no zone.
    Time(i64),
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

    /// The labels whose place `keep` marks, in order (`keep` one flag a
    /// label).
    pub fn kept(self, keep: &[bool]) -> Labels {
        fn only<T>(values: Vec<T>, keep: &[bool]) -> Vec<T> {
            values
                .into_iter()
                .zip(keep)
                .filter_map(|(value, kept)| kept.then_some(value))
                .collect()
        }
        match self {
            Labels::Float(v) => Labels::Float(only(v, keep)),
            Labels::Int(v) => Labels::Int(only(v, keep)),
            Labels::Bool(v) => Labels::Bool(only(v, keep)),
            Labels::Text(v) => Labels::Text(only(v, keep)),
        }
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

/// A condition the query's validation lets through only for operands and
/// constants of matching types, met with others: a fault of this build.
fn unpaired() -> CoreError {
    CoreError::new(
        ErrorKind::Internal,
        ErrorCode::InternalError,
        "a condition pairs a value and a constant of types the query's validation keeps apart",
    )
}

/// A comparison with `NULL` other than `=` and `!=`, which the query's
/// validation refuses before it reaches here.
fn ordered_null() -> CoreError {
    CoreError::new(
        ErrorKind::Internal,
        ErrorCode::InternalError,
        "a comparison with NULL other than = and != reached the labels",
    )
}

/// A number constant at float32, by way of float64, as numpy reads a
/// Python number at a float32 target's type.
fn float32(constant: &Constant) -> Result<f32, CoreError> {
    match constant {
        Constant::Int(i) => Ok(*i as f64 as f32),
        Constant::Float(f) => Ok(*f as f32),
        _ => Err(unpaired()),
    }
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

/// Whether `order`, the left side's against the right, satisfies `op`.
fn holds(order: std::cmp::Ordering, op: RelOp) -> bool {
    match op {
        RelOp::Eq => order.is_eq(),
        RelOp::Ne => order.is_ne(),
        RelOp::Lt => order.is_lt(),
        RelOp::Le => order.is_le(),
        RelOp::Gt => order.is_gt(),
        RelOp::Ge => order.is_ge(),
    }
}

/// A condition's constant as a decimal, `digits * 10**exponent`: an
/// integer as written; a float as the shortest decimal that names its
/// double, which is the constant the query wrote whenever it wrote no more
/// than a double's 17 significant digits; `None` for a float that is not
/// finite, or a constant that is not a number.
fn constant_decimal(constant: &Constant) -> Option<(i128, i32)> {
    match *constant {
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
        _ => None,
    }
}

/// How a decimal of unscaled `value` at `scale` orders against `constant`,
/// exactly, at no scale but their own (T2.0c, KO2-50: `d > 1.5` holds for
/// a `DECIMAL(38, 0)` `d` of 2): the constant is never rounded to the
/// column's scale, and one past the column's digits orders by its sign.
/// An infinite constant orders past every value; a NaN, which the PQL
/// validator refuses before it reaches here, orders against none.
fn decimal_order(value: i128, scale: i8, constant: &Constant) -> Option<std::cmp::Ordering> {
    use std::cmp::Ordering;
    let Some((digits, exponent)) = constant_decimal(constant) else {
        return match *constant {
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
/// comparison with NULL the query's validation refuses.
fn null_condition(has: Vec<bool>, op: RelOp) -> Result<Labelled, CoreError> {
    let labels = match op {
        RelOp::Eq => has.iter().map(|h| !h).collect(),
        RelOp::Ne => has.clone(),
        _ => return Err(ordered_null()),
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
        _ => return Err(ordered_null()),
    };
    let n = present.iter().filter(|p| **p).count();
    Ok(Labelled {
        mask: present,
        labels: Labels::Bool(vec![label; n]),
    })
}

/// A column's values for each example (an entity's static column) or for
/// each row fetched (a column a target's filter tests), as the executor
/// reads them.
#[derive(Debug, Clone, PartialEq)]
pub enum Column {
    Float(Vec<Option<f32>>),
    Int(Vec<Option<i64>>),
    Bool(Vec<Option<bool>>),
    Text(Vec<Option<String>>),
    /// A decimal column's unscaled values, its declared precision and its
    /// scale: compared exactly in a condition, or a target
    /// ([`decimal_target`]).
    Decimal {
        values: Vec<Option<i128>>,
        precision: u8,
        scale: i8,
    },
    /// A time column's values, nanoseconds since the epoch with no zone:
    /// only compared in a condition, never a target.
    Time(Vec<Option<i64>>),
    /// An unsigned 64-bit column's values, past int64 too: only tested in
    /// a condition (a target's static column is read as [`Column::Int`]),
    /// against an integer exactly and against a float at float64, as
    /// the executor compares a uint64 column.
    UInt(Vec<Option<u64>>),
}

impl Column {
    /// Whether each row holds a value: not null, and for a float not NaN
    /// (`notna()` after `execute_column`'s float32).
    fn present(&self) -> Vec<bool> {
        match self {
            Column::Float(v) => v.iter().map(|v| v.is_some_and(|v| !v.is_nan())).collect(),
            Column::Int(v) => v.iter().map(Option::is_some).collect(),
            Column::Bool(v) => v.iter().map(Option::is_some).collect(),
            Column::Text(v) => v.iter().map(Option::is_some).collect(),
            Column::Decimal { values, .. } => values.iter().map(Option::is_some).collect(),
            Column::Time(v) => v.iter().map(Option::is_some).collect(),
            Column::UInt(v) => v.iter().map(Option::is_some).collect(),
        }
    }
}

/// A decimal column as a target. At scale 0, where every value fits, an
/// int64, as the reference reads it since `main`'s eb6c68eb. Otherwise:
/// - a regression's labels are numbers, each value rounded once to the
///   nearest float32 (round 1: a regression has no classes to keep apart);
/// - a classification's are spelled as the reference spells its classes
///   (float64, then float32, as [`decimal_to_f64`] reads it), and where two
///   distinct values meet at one float32, each value's own digits instead,
///   so no two values become one class (T2.0c).
fn decimal_target(values: &[Option<i128>], scale: i8, classes: bool) -> (Vec<bool>, Labels) {
    let mask: Vec<bool> = values.iter().map(Option::is_some).collect();
    let present: Vec<i128> = values.iter().flatten().copied().collect();
    if scale == 0
        && let Ok(ints) = present
            .iter()
            .map(|&v| i64::try_from(v))
            .collect::<Result<Vec<_>, _>>()
    {
        return (mask, Labels::Int(ints));
    }
    if !classes {
        let floats = present.iter().map(|&v| decimal_to_f32(v, scale)).collect();
        return (mask, Labels::Float(floats));
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

/// A query's target over the examples, or a target's filter over the rows
/// fetched for it (its leaves then the rows' columns).
#[derive(Debug, Clone, PartialEq)]
pub enum Target {
    /// A column of the entity's row, or of a row fetched.
    Column(Column),
    /// A windowed aggregation.
    Aggregated(Aggregated),
    /// `target op constant`; no constant is `NULL`.
    Condition {
        target: Box<Target>,
        op: RelOp,
        constant: Option<Constant>,
    },
    /// `target IN (values)`.
    Member {
        target: Box<Target>,
        values: Vec<Constant>,
    },
    /// `NOT condition`.
    Not(Box<Target>),
    /// `left AND right`, `left OR right`.
    Both {
        op: BoolOp,
        left: Box<Target>,
        right: Box<Target>,
    },
}

/// The examples' labels: which of them have one (`mask`), and the labels
/// of those, in order.
#[derive(Debug, Clone, PartialEq)]
pub struct Labelled {
    pub mask: Vec<bool>,
    pub labels: Labels,
}

/// A condition over every row (an example, or a row fetched), as the
/// executor's `execute_condition` and `execute_logical_operation` give it
/// without dropping a row (`filter_na=False`): whether it holds (`values`,
/// `False` where a value it compares is missing, so `NOT` of such a row
/// holds), and whether the row has a value for every column and
/// aggregation it tests (`mask`; for `= NULL` too, which holds exactly
/// where it has none).
#[derive(Debug, Clone, PartialEq)]
pub struct Truth {
    pub values: Vec<bool>,
    pub mask: Vec<bool>,
}

/// What a condition asks of its operand.
enum Test<'a> {
    Compare(RelOp, Option<&'a Constant>),
    In(&'a [Constant]),
}

/// `target`'s truth for each row, a condition, `NOT`, `AND` or `OR`
/// ([`Truth`]), as a target's filter (`execute_filter`, which keeps the
/// rows whose condition holds) and a logical target read it. `NOT` negates
/// a condition's values and keeps its mask; `AND` and `OR` join their
/// sides' values and require both masks (`execute_logical_operation`'s
/// `mask & right_mask`). An aggregation's example is joined to the same
/// example's other side: the reference's packs an average's, a minimum's
/// and a maximum's values onto the examples that have one before it joins
/// them, which misaligns them where an earlier example has none (register
/// entry of T2.10).
pub fn truth(target: &Target) -> Result<Truth, CoreError> {
    match target {
        Target::Condition {
            target,
            op,
            constant,
        } => leaf(target, &Test::Compare(*op, constant.as_ref()), false),
        Target::Member { target, values } => leaf(target, &Test::In(values), false),
        Target::Not(inner) => {
            let Truth { values, mask } = truth(inner)?;
            Ok(Truth {
                values: values.into_iter().map(|v| !v).collect(),
                mask,
            })
        }
        Target::Both { op, left, right } => {
            let left = truth(left)?;
            let right = truth(right)?;
            if left.values.len() != right.values.len() {
                return Err(CoreError::new(
                    ErrorKind::Internal,
                    ErrorCode::InternalError,
                    "a logical target's sides do not cover the same rows",
                ));
            }
            Ok(Truth {
                values: left
                    .values
                    .iter()
                    .zip(&right.values)
                    .map(|(&l, &r)| match op {
                        BoolOp::And => l && r,
                        BoolOp::Or => l || r,
                    })
                    .collect(),
                mask: left
                    .mask
                    .iter()
                    .zip(&right.mask)
                    .map(|(&l, &r)| l && r)
                    .collect(),
            })
        }
        Target::Column(_) | Target::Aggregated(_) => Err(CoreError::new(
            ErrorKind::Internal,
            ErrorCode::InternalError,
            "a column or an aggregation stands where a condition is read",
        )),
    }
}

/// One condition's truth over its operand, a column or an aggregation.
/// `top` is a condition that is the whole target, whose column the
/// executor reads without its missing values first (`filter_na=True`), so
/// an integer column is a numpy int64 that refuses a float with a
/// fraction in an `IN` list; elsewhere such a float is compared exactly
/// and matches no integer ([`int_member`], KO2-73).
fn leaf(target: &Target, test: &Test<'_>, top: bool) -> Result<Truth, CoreError> {
    match target {
        Target::Column(column) => {
            let mask = column.present();
            let values = match test {
                Test::Compare(op, None) => match op {
                    RelOp::Eq => mask.iter().map(|p| !p).collect(),
                    RelOp::Ne => mask.clone(),
                    _ => return Err(ordered_null()),
                },
                Test::Compare(op, Some(constant)) => column_compare(column, *op, constant)?,
                Test::In(values) => column_member(column, values, top)?,
            };
            Ok(Truth { values, mask })
        }
        Target::Aggregated(aggregated) => {
            let rows = aggregated.values.iter().zip(&aggregated.present);
            let values = match test {
                Test::Compare(op, None) => match op {
                    // The executor compares the values it packed, each
                    // present.
                    RelOp::Eq => vec![false; aggregated.present.len()],
                    RelOp::Ne => aggregated.present.clone(),
                    _ => return Err(ordered_null()),
                },
                Test::Compare(op, Some(constant)) => {
                    let value = float32(constant)?;
                    rows.map(|(&v, &p)| p && !v.is_nan() && apply(v, *op, value))
                        .collect()
                }
                Test::In(values) => {
                    let set = values.iter().map(float32).collect::<Result<Vec<_>, _>>()?;
                    rows.map(|(v, &p)| p && set.contains(v)).collect()
                }
            };
            Ok(Truth {
                values,
                mask: aggregated.present.clone(),
            })
        }
        _ => Err(CoreError::new(
            ErrorKind::Internal,
            ErrorCode::InternalError,
            "a condition of a condition reached the labels",
        )),
    }
}

/// Each row's `value op constant`, as `execute_rel_op` makes it: a float
/// column at float32, the constant cast to it; an integer column against
/// an integer exactly, against a float at float64; a decimal exactly
/// ([`decimal_order`]); a boolean, text or time against its own kind. A
/// row without a value holds no comparison.
fn column_compare(column: &Column, op: RelOp, constant: &Constant) -> Result<Vec<bool>, CoreError> {
    Ok(match (column, constant) {
        (Column::Float(v), Constant::Int(_) | Constant::Float(_)) => {
            let value = float32(constant)?;
            v.iter()
                .map(|x| x.is_some_and(|x| !x.is_nan() && apply(x, op, value)))
                .collect()
        }
        (Column::Int(v), Constant::Int(c)) => {
            let c = i64::try_from(*c).map_err(|_| not_yet("condition_value"))?;
            v.iter()
                .map(|x| x.is_some_and(|x| apply(x, op, c)))
                .collect()
        }
        (Column::Int(v), Constant::Float(f)) => v
            .iter()
            .map(|x| x.is_some_and(|x| apply(x as f64, op, *f)))
            .collect(),
        (Column::UInt(v), Constant::Int(c)) => v
            .iter()
            .map(|x| x.is_some_and(|x| apply(i128::from(x), op, *c)))
            .collect(),
        #[allow(
            clippy::cast_precision_loss,
            reason = "an unsigned value widened to compare with a float, as the executor does"
        )]
        (Column::UInt(v), Constant::Float(f)) => v
            .iter()
            .map(|x| x.is_some_and(|x| apply(x as f64, op, *f)))
            .collect(),
        (Column::Decimal { values, scale, .. }, Constant::Int(_) | Constant::Float(_)) => values
            .iter()
            .map(|x| {
                x.is_some_and(|x| decimal_order(x, *scale, constant).is_some_and(|o| holds(o, op)))
            })
            .collect(),
        (Column::Bool(v), Constant::Bool(b)) => v
            .iter()
            .map(|x| x.is_some_and(|x| apply(x, op, *b)))
            .collect(),
        (Column::Text(v), Constant::Text(t)) => v
            .iter()
            .map(|x| x.as_deref().is_some_and(|x| apply(x, op, t.as_str())))
            .collect(),
        (Column::Time(v), Constant::Time(t)) => v
            .iter()
            .map(|x| x.is_some_and(|x| apply(x, op, *t)))
            .collect(),
        _ => return Err(unpaired()),
    })
}

/// Each row's `value IN (values)`, as `execute_member_op` makes it: the
/// list read at the column's type (`pd.Series(values, dtype=...)`), then
/// `isin`. A float column's list at float32; a decimal's compared exactly;
/// an integer column's integers as written and its floats as
/// [`int_member`] reads them. A row without a value is in no list.
fn column_member(column: &Column, values: &[Constant], top: bool) -> Result<Vec<bool>, CoreError> {
    fn within<T: PartialEq>(rows: &[Option<T>], set: &[T]) -> Vec<bool> {
        rows.iter()
            .map(|x| x.as_ref().is_some_and(|x| set.contains(x)))
            .collect()
    }
    fn each<T>(
        values: &[Constant],
        read: impl Fn(&Constant) -> Option<T>,
    ) -> Result<Vec<T>, CoreError> {
        values
            .iter()
            .map(|c| read(c).ok_or_else(unpaired))
            .collect()
    }
    Ok(match column {
        Column::Float(v) => {
            let set = values.iter().map(float32).collect::<Result<Vec<_>, _>>()?;
            v.iter()
                .map(|x| x.is_some_and(|x| !x.is_nan() && set.contains(&x)))
                .collect()
        }
        Column::Int(v) => {
            let set: Vec<i64> = values
                .iter()
                .map(|c| int_member(c, top))
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .flatten()
                .collect();
            within(v, &set)
        }
        Column::Decimal {
            values: v, scale, ..
        } => {
            for c in values {
                if !matches!(c, Constant::Int(_) | Constant::Float(_)) {
                    return Err(unpaired());
                }
            }
            v.iter()
                .map(|x| {
                    x.is_some_and(|x| {
                        values
                            .iter()
                            .any(|c| decimal_order(x, *scale, c).is_some_and(|o| o.is_eq()))
                    })
                })
                .collect()
        }
        Column::Bool(v) => within(
            v,
            &each(values, |c| match c {
                Constant::Bool(b) => Some(*b),
                _ => None,
            })?,
        ),
        Column::Text(v) => within(
            v,
            &each(values, |c| match c {
                Constant::Text(t) => Some(t.clone()),
                _ => None,
            })?,
        ),
        Column::Time(v) => within(
            v,
            &each(values, |c| match c {
                Constant::Time(t) => Some(*t),
                _ => None,
            })?,
        ),
        // An integer as written, exactly; a float as [`int_member`] reads it.
        Column::UInt(v) => {
            let set: Vec<i128> = values
                .iter()
                .map(|c| match c {
                    Constant::Int(i) => Ok(Some(*i)),
                    c => int_member(c, top).map(|m| m.map(i128::from)),
                })
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .flatten()
                .collect();
            v.iter()
                .map(|x| x.is_some_and(|x| set.contains(&i128::from(x))))
                .collect()
        }
    })
}

/// An `IN` list's constant read at an integer column's type, or none where
/// it matches no integer: an integer as written. A float, where the
/// condition is the whole target, only when it has no fraction (numpy's
/// int64 refuses another, the caller's error, `INVALID_REQUEST`, as the
/// reference refuses it on every database), a float past int64 refused by
/// name (`condition_value`) where numpy's int64 raises an overflow.
/// Elsewhere (a target's filter, a logical target) a float is compared
/// exactly (KO2-73): a whole one matches its integer, and one with a
/// fraction or past int64 matches no row, where the reference takes it
/// toward zero on DuckDB and SQLite and refuses it on Postgres (register
/// P2E-17).
fn int_member(constant: &Constant, top: bool) -> Result<Option<i64>, CoreError> {
    // 2**63, the first float past int64.
    const PAST: f64 = 9_223_372_036_854_775_808.0;
    match *constant {
        Constant::Int(i) => i64::try_from(i)
            .map(Some)
            .map_err(|_| not_yet("condition_value")),
        Constant::Float(f) if top => {
            if f.fract() != 0.0 || !f.is_finite() {
                return Err(CoreError::new(
                    ErrorKind::InvalidRequest,
                    ErrorCode::InvalidRequest,
                    "an IN list of an integer column holds a float with a fraction",
                )
                .with_field("field", sdm_contract::Detail::text("query")));
            }
            if (-PAST..PAST).contains(&f) {
                Ok(Some(f as i64))
            } else {
                Err(not_yet("condition_value"))
            }
        }
        Constant::Float(f) => Ok(
            (f.is_finite() && f.fract() == 0.0 && (-PAST..PAST).contains(&f)).then_some(f as i64),
        ),
        _ => Err(unpaired()),
    }
}

/// `target`'s labels, as the executor's `execute` gives them, for any task
/// but a classification of a column ([`label_classes`]).
pub fn label(target: &Target) -> Result<Labelled, CoreError> {
    labelled(target, false)
}

/// `target`'s labels for a classification: a decimal column's distinct
/// values are kept as distinct classes ([`decimal_target`]).
pub fn label_classes(target: &Target) -> Result<Labelled, CoreError> {
    labelled(target, true)
}

/// The rows of `truth` its mask keeps, each labelled whether it holds.
fn kept_truth(truth: Truth) -> Labelled {
    let labels = truth
        .values
        .iter()
        .zip(&truth.mask)
        .filter(|(_, m)| **m)
        .map(|(v, _)| *v)
        .collect();
    Labelled {
        mask: truth.mask,
        labels: Labels::Bool(labels),
    }
}

fn labelled(target: &Target, classes: bool) -> Result<Labelled, CoreError> {
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
                Column::Decimal { values, scale, .. } => decimal_target(values, *scale, classes),
                // A time is no target the reference's task table takes.
                Column::Time(_) | Column::UInt(_) => return Err(not_yet("target_type")),
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
        // A condition that is the whole target: `= NULL` and `!= NULL`
        // label every example of a column, and of an aggregation those
        // with a value; any other, the examples with a value.
        Target::Condition {
            target,
            op,
            constant: None,
        } => match target.as_ref() {
            Target::Aggregated(aggregated) => aggregated_null(aggregated.present.clone(), *op),
            Target::Column(column) => null_condition(column.present(), *op),
            _ => Err(CoreError::new(
                ErrorKind::Internal,
                ErrorCode::InternalError,
                "a condition of a condition reached the labels",
            )),
        },
        Target::Condition {
            target: inner,
            op,
            constant: Some(constant),
        } => Ok(kept_truth(leaf(
            inner,
            &Test::Compare(*op, Some(constant)),
            true,
        )?)),
        Target::Member {
            target: inner,
            values,
        } => Ok(kept_truth(leaf(inner, &Test::In(values), true)?)),
        Target::Not(_) | Target::Both { .. } => Ok(kept_truth(truth(target)?)),
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
            label_classes(&Target::Column(close)).unwrap().labels,
            Labels::Text(vec![
                "0.000000100000000000".to_owned(),
                "0.000000100000000001".to_owned(),
            ])
        );
    }

    #[test]
    fn a_decimal_regression_target_meeting_at_one_float32_stays_numbers() {
        // Round 1 (Claude P1-1): a regression has no classes to keep apart,
        // so values meeting at one float32 are numbers, each rounded once to
        // the nearest float32, never text (which a regression's payload
        // refuses as an internal error). DECIMAL(12, 2) amounts past
        // 1,000,000 with cents meet at float32's step of 0.125 there.
        let amounts = Column::Decimal {
            values: vec![
                Some(123_456_701),
                Some(123_456_702),
                None,
                Some(-123_456_703),
            ],
            precision: 12,
            scale: 2,
        };
        let regression = label(&Target::Column(amounts.clone())).unwrap();
        assert_eq!(regression.mask, vec![true, true, false, true]);
        assert_eq!(
            regression.labels,
            Labels::Float(vec![1_234_567.0, 1_234_567.0, -1_234_567.0])
        );
        // As a classification, the same values stay three classes.
        assert_eq!(
            label_classes(&Target::Column(amounts)).unwrap().labels,
            Labels::Text(vec![
                "1234567.01".to_owned(),
                "1234567.02".to_owned(),
                "-1234567.03".to_owned(),
            ])
        );
        // Rounded once: a hair above a float32 midpoint rounds up.
        let above = Column::Decimal {
            values: vec![Some(1_000_000_059_604_644_775_390_625_000_001)],
            precision: 38,
            scale: 30,
        };
        assert_eq!(
            label(&Target::Column(above)).unwrap().labels,
            Labels::Float(vec![1.0 + f32::EPSILON])
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
    fn a_decimal_mean_is_exact_and_rounded_once() {
        // Round 1 (Codex P1): the mean is the exact sum over the count, never
        // first rounded to the column's scale: 1 and 2 at scale 0 average
        // 1.5, not 2.
        for (terms, scale, wanted) in [
            (vec![1, 2], 0, 1.5_f32),
            (vec![-1, -2], 0, -1.5),
            (vec![1, 2, 1, 2], 0, 1.5),
            (vec![1, 2, 2], 0, 5.0 / 3.0),
            (vec![1, 1, 2], 0, 4.0 / 3.0),
            (vec![-1, -1, -2], 0, -4.0 / 3.0),
            (vec![1, 2], 2, 0.015),
            (vec![10, 20, 20], 2, 50.0 / 300.0),
            (vec![-10, -20, -20], 2, -50.0 / 300.0),
        ] {
            assert_eq!(
                decimal_aggregate(Aggregation::Avg, &terms, 10, scale).unwrap(),
                wanted,
                "{terms:?} at {scale}"
            );
        }
        // A total past DECIMAL(38) and past i128 is averaged exactly.
        let nines = 99_999_999_999_999_999_999_999_999_999_999_999_999_i128;
        assert_eq!(
            decimal_aggregate(Aggregation::Avg, &[nines, 1], 38, 0).unwrap(),
            5e37
        );
        assert_eq!(
            decimal_aggregate(Aggregation::Avg, &[nines, nines, nines], 38, 0).unwrap(),
            1e38
        );
    }

    #[test]
    fn a_decimal_sum_never_wraps_nor_flips_its_sign() {
        // Round 1 (Codex P1): two values of -2**126 sum to i128::MIN, whose
        // magnitude is 2**127, which read as an i64 high half flipped sign.
        let half = -85_070_591_730_234_615_865_843_651_857_942_052_864_i128;
        assert_eq!(half * 2, i128::MIN);
        assert_eq!(
            decimal_aggregate(Aggregation::Sum, &[half, half], 38, 0).unwrap(),
            -1.701_411_8e38
        );
        assert_eq!(
            decimal_aggregate(Aggregation::Sum, &[-half, -half], 38, 0).unwrap(),
            1.701_411_8e38
        );
        // And its neighbours: past i128 on both sides, and back within it.
        assert_eq!(
            decimal_aggregate(Aggregation::Sum, &[half, half, half], 38, 0).unwrap(),
            -2.552_117_8e38
        );
        assert_eq!(
            decimal_aggregate(Aggregation::Sum, &[half, half, -1], 38, 0).unwrap(),
            -1.701_411_8e38
        );
        assert_eq!(
            decimal_aggregate(Aggregation::Sum, &[half, half, -half], 38, 0).unwrap(),
            -8.507_059e37
        );
        assert_eq!(decimal_to_f64(i128::MIN, 0), -1.701_411_834_604_692_3e38);
        assert_eq!(
            decimal_aggregate(Aggregation::Min, &[i128::MIN + 1, 0], 38, 0).unwrap(),
            -1.701_411_8e38
        );
    }

    #[test]
    fn a_decimal_aggregate_is_rounded_once_to_float32() {
        // Round 1 (Codex P1): 1 + 2**-24 is a float32 midpoint; a hair above
        // it rounds up, a hair below rounds down, where rounding first to
        // float64 lands on the midpoint and then to even, 1.0.
        let above = 1_000_000_059_604_644_775_390_625_000_001_i128;
        let below = 1_000_000_059_604_644_775_390_624_999_999_i128;
        let exact = 1_000_000_059_604_644_775_390_625_000_000_i128;
        let up = 1.0_f32 + f32::EPSILON;
        for aggregation in [Aggregation::Sum, Aggregation::Avg, Aggregation::Max] {
            assert_eq!(
                decimal_aggregate(aggregation, &[above], 38, 30).unwrap(),
                up,
                "{aggregation:?}"
            );
            assert_eq!(
                decimal_aggregate(aggregation, &[below], 38, 30).unwrap(),
                1.0,
                "{aggregation:?}"
            );
            // On the midpoint itself, to even.
            assert_eq!(
                decimal_aggregate(aggregation, &[exact], 38, 30).unwrap(),
                1.0,
                "{aggregation:?}"
            );
        }
        // Split over two rows, as a sum of two values and as their mean.
        assert_eq!(
            decimal_aggregate(Aggregation::Sum, &[above - 1, 1], 38, 30).unwrap(),
            up
        );
        assert_eq!(
            decimal_aggregate(Aggregation::Avg, &[above * 2 - 1, 1], 38, 30).unwrap(),
            up
        );
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

    // The reference's executor's answers below were read from it at
    // 6298913c over these values (T2.10's probe): six entities, `cat`
    // `['US', 'DE', 'IN', NULL, 'DE', 'IN']`, `flag`
    // `[true, false, false, true, NULL, false]`, `ni` `[0, 1, 2, 3, 0, NULL]`.
    fn cat() -> Box<Target> {
        Box::new(Target::Column(Column::Text(
            ["US", "DE", "IN", "", "DE", "IN"]
                .iter()
                .map(|t| (!t.is_empty()).then(|| (*t).to_owned()))
                .collect(),
        )))
    }

    fn flag() -> Box<Target> {
        Box::new(Target::Column(Column::Bool(vec![
            Some(true),
            Some(false),
            Some(false),
            Some(true),
            None,
            Some(false),
        ])))
    }

    fn ni() -> Box<Target> {
        Box::new(Target::Column(Column::Int(vec![
            Some(0),
            Some(1),
            Some(2),
            Some(3),
            Some(0),
            None,
        ])))
    }

    fn text(t: &str) -> Constant {
        Constant::Text(t.to_owned())
    }

    #[test]
    fn a_logical_target_labels_the_rows_every_condition_has_a_value_for() {
        // `e.cat != 'US' OR e.flag = true`: [T, T, T, T], mask 1 1 1 0 0 1.
        let or = label(&Target::Both {
            op: BoolOp::Or,
            left: Box::new(Target::Condition {
                target: cat(),
                op: RelOp::Ne,
                constant: Some(text("US")),
            }),
            right: Box::new(Target::Condition {
                target: flag(),
                op: RelOp::Eq,
                constant: Some(Constant::Bool(true)),
            }),
        })
        .unwrap();
        assert_eq!(or.mask, [true, true, true, false, false, true]);
        assert_eq!(or.labels, Labels::Bool(vec![true; 4]));
        // `NOT e.cat = NULL`: a NULL test inside a logical target keeps
        // only the rows with a value: [T; 5], mask 1 1 1 0 1 1.
        let not = label(&Target::Not(Box::new(Target::Condition {
            target: cat(),
            op: RelOp::Eq,
            constant: None,
        })))
        .unwrap();
        assert_eq!(not.mask, [true, true, true, false, true, true]);
        assert_eq!(not.labels, Labels::Bool(vec![true; 5]));
    }

    #[test]
    fn an_in_list_reads_a_float_at_an_integer_column_as_the_executor_does() {
        // `e.ni IN (1, 2)` and `IN (1, 2.0)`: [F, T, T, F, F], mask 1 1 1 1 1 0.
        for list in [
            vec![Constant::Int(1), Constant::Int(2)],
            vec![Constant::Int(1), Constant::Float(2.0)],
        ] {
            let out = label(&Target::Member {
                target: ni(),
                values: list,
            })
            .unwrap();
            assert_eq!(out.mask, [true, true, true, true, true, false]);
            assert_eq!(
                out.labels,
                Labels::Bool(vec![false, true, true, false, false])
            );
        }
        // `e.ni IN (1, 2.5)`, the whole target: numpy's int64 refuses 2.5.
        let refused = label(&Target::Member {
            target: ni(),
            values: vec![Constant::Int(1), Constant::Float(2.5)],
        })
        .unwrap_err();
        assert_eq!(refused.code, ErrorCode::InvalidRequest);
        // `e.ni IN (1, 2.5) OR e.flag = true`: 2.5 is compared exactly and
        // matches no integer (KO2-73), where the reference takes it as 2:
        // [T, T, F, T], mask 1 1 1 1 0 0.
        let or = label(&Target::Both {
            op: BoolOp::Or,
            left: Box::new(Target::Member {
                target: ni(),
                values: vec![Constant::Int(1), Constant::Float(2.5)],
            }),
            right: Box::new(Target::Condition {
                target: flag(),
                op: RelOp::Eq,
                constant: Some(Constant::Bool(true)),
            }),
        })
        .unwrap();
        assert_eq!(or.mask, [true, true, true, true, false, false]);
        assert_eq!(or.labels, Labels::Bool(vec![true, true, false, true]));
        // Inside a logical target a whole float matches its integer, and
        // one past int64 matches no row.
        for (float, wanted) in [(2.0, true), (1e20, false), (-2.5, false)] {
            let not = label(&Target::Not(Box::new(Target::Member {
                target: ni(),
                values: vec![Constant::Float(float)],
            })))
            .unwrap();
            assert_eq!(
                not.labels,
                Labels::Bool(vec![true, true, !wanted, true, true]),
                "{float}"
            );
        }
        // `e.score IN (0.25, 0.5)` at float32: [F, T, T, F, F, F].
        let score = label(&Target::Member {
            target: Box::new(Target::Column(Column::Float(
                (0..6).map(|i| Some(i as f32 * 0.25)).collect(),
            ))),
            values: vec![Constant::Float(0.25), Constant::Float(0.5)],
        })
        .unwrap();
        assert_eq!(
            score.labels,
            Labels::Bool(vec![false, true, true, false, false, false])
        );
    }

    #[test]
    fn a_filter_keeps_the_rows_its_condition_holds_for_a_missing_value_failing_it() {
        // `COUNT(a.* WHERE NOT a.v > 3, ...)`: a row whose `v` is null fails
        // `a.v > 3`, so `NOT` keeps it.
        let v = Target::Column(Column::Float(vec![Some(1.0), None, Some(5.0), Some(2.0)]));
        let kept = truth(&Target::Not(Box::new(Target::Condition {
            target: Box::new(v),
            op: RelOp::Gt,
            constant: Some(Constant::Int(3)),
        })))
        .unwrap()
        .values;
        assert_eq!(kept, [true, true, false, true]);
        let out = aggregate_kept(
            Aggregation::Count,
            2,
            &[0, 0, 1, 1],
            &Values::Ones(4),
            &[Some(5); 4],
            bounds(&[10, 10]),
            Some(&kept),
        )
        .unwrap();
        assert_eq!(out.values, [2.0, 1.0]);
        assert_eq!(out.present, [true, true]);
    }

    #[test]
    fn an_aggregation_in_a_logical_target_keeps_each_example_s_own_value() {
        // Examples 0 and 2 have an average, 1 none: each example's
        // condition stays its own, and 1 has no label.
        let avg = Box::new(Target::Aggregated(Aggregated {
            values: vec![1.5, 0.0, 2.5],
            present: vec![true, false, true],
        }));
        let count = Box::new(Target::Aggregated(Aggregated {
            values: vec![2.0, 0.0, 1.0],
            present: vec![true; 3],
        }));
        let and = label(&Target::Both {
            op: BoolOp::And,
            left: Box::new(Target::Condition {
                target: avg,
                op: RelOp::Gt,
                constant: Some(Constant::Int(2)),
            }),
            right: Box::new(Target::Condition {
                target: count,
                op: RelOp::Gt,
                constant: Some(Constant::Int(0)),
            }),
        })
        .unwrap();
        assert_eq!(and.mask, [true, false, true]);
        assert_eq!(and.labels, Labels::Bool(vec![false, true]));
    }

    #[test]
    fn a_time_is_compared_with_a_time() {
        let day = 86_400_000_000_000_i64;
        let out = label(&Target::Condition {
            target: Box::new(Target::Column(Column::Time(vec![
                Some(0),
                None,
                Some(2 * day),
            ]))),
            op: RelOp::Gt,
            constant: Some(Constant::Time(day)),
        })
        .unwrap();
        assert_eq!(out.mask, [true, false, true]);
        assert_eq!(out.labels, Labels::Bool(vec![false, true]));
    }
}
