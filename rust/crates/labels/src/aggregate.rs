// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `execute_aggregation_type`: a feature grouped by example and aggregated
//! as pandas' `groupby(...).agg(...)` does it, then cast as the executor
//! casts it.
//!
//! The arithmetic is pandas' cython kernels': sums by Kahan summation in
//! the column's own float type, in row order, the compensation reset when
//! it stops being finite; means the same, the compensation reset only when
//! it becomes `NaN`, divided by the count in that float type; integers
//! summed with wraparound and averaged in `float64`. Every result but a
//! datetime is then cast to `float32`.

use std::collections::{BTreeMap, HashSet};

use crate::LabelError;
use crate::column::{Column, Obj};
use sdm_schema::decimal;

/// `AggregationType`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Aggregation {
    Sum,
    Avg,
    Min,
    Max,
    Count,
    CountDistinct,
    First,
    Last,
    ListDistinct,
}

impl Aggregation {
    pub fn parse(name: &str) -> Option<Self> {
        Some(match name {
            "SUM" => Aggregation::Sum,
            "AVG" => Aggregation::Avg,
            "MIN" => Aggregation::Min,
            "MAX" => Aggregation::Max,
            "COUNT" => Aggregation::Count,
            "COUNT_DISTINCT" => Aggregation::CountDistinct,
            "FIRST" => Aggregation::First,
            "LAST" => Aggregation::Last,
            "LIST_DISTINCT" => Aggregation::ListDistinct,
            _ => return None,
        })
    }

    /// The name `agg` is given.
    fn pandas_name(self) -> &'static str {
        match self {
            Aggregation::Sum => "sum",
            Aggregation::Avg => "mean",
            Aggregation::Min => "min",
            Aggregation::Max => "max",
            Aggregation::Count => "size",
            Aggregation::CountDistinct => "count_distinct",
            Aggregation::First => "first",
            Aggregation::Last => "last",
            Aggregation::ListDistinct => "list",
        }
    }
}

/// `values[i]` for numpy's index `i` into `size` items: negative from the
/// end, or the `IndexError` numpy raises.
fn numpy_index(i: i64, size: usize) -> Result<usize, LabelError> {
    let n = size as i64;
    let at = if i < 0 { i + n } else { i };
    if (0..n).contains(&at) {
        Ok(at as usize)
    } else {
        Err(LabelError::new(
            "IndexError",
            format!("index {i} is out of bounds for axis 0 with size {size}"),
        ))
    }
}

/// Kahan summation as `group_sum` runs it over `float32`.
fn kahan_sum_f32(values: impl Iterator<Item = f32>) -> f32 {
    let (mut sum, mut comp) = (0f32, 0f32);
    for v in values {
        let y = v - comp;
        let t = sum + y;
        comp = t - sum - y;
        if !comp.is_finite() {
            comp = 0.0;
        }
        sum = t;
    }
    sum
}

fn kahan_sum_f64(values: impl Iterator<Item = f64>) -> f64 {
    let (mut sum, mut comp) = (0f64, 0f64);
    for v in values {
        let y = v - comp;
        let t = sum + y;
        comp = t - sum - y;
        if !comp.is_finite() {
            comp = 0.0;
        }
        sum = t;
    }
    sum
}

/// Kahan summation as `group_mean` runs it: the compensation is reset only
/// when it becomes `NaN`.
fn mean_f32(values: &[f32]) -> f32 {
    let (mut sum, mut comp) = (0f32, 0f32);
    for &v in values {
        let y = v - comp;
        let t = sum + y;
        comp = t - sum - y;
        if comp.is_nan() {
            comp = 0.0;
        }
        sum = t;
    }
    sum / values.len() as f32
}

fn mean_f64(values: &[f64]) -> f64 {
    let (mut sum, mut comp) = (0f64, 0f64);
    for &v in values {
        let y = v - comp;
        let t = sum + y;
        comp = t - sum - y;
        if comp.is_nan() {
            comp = 0.0;
        }
        sum = t;
    }
    sum / values.len() as f64
}

/// A hashable stand-in for a value `drop_duplicates` compares: Python's
/// `1 == 1.0 == True`, and `0.0 == -0.0`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum Identity {
    Int(i128),
    Float(u64),
    Text(String),
    Bytes(Vec<u8>),
    Duration(i128),
    Time(i128),
    /// A decimal no integer or float equals: its digits and scale, without
    /// trailing zeros (`sdm_schema::decimal::identity`).
    Decimal(i128, i8),
    None,
}

fn identity(obj: &Obj) -> Result<Identity, LabelError> {
    let float = |v: f64| {
        if v.fract() == 0.0 && v.abs() < 1e38 {
            Identity::Int(v as i128)
        } else {
            Identity::Float(v.to_bits())
        }
    };
    Ok(match obj {
        Obj::Bool(b) | Obj::NpBool(b) => Identity::Int(i128::from(u8::from(*b))),
        Obj::Int(v) | Obj::NpInt64(v) => Identity::Int(i128::from(*v)),
        Obj::Float(v) | Obj::NpFloat64(v) => float(*v),
        Obj::NpFloat32(v) => float(f64::from(*v)),
        Obj::Str(s) => Identity::Text(s.clone()),
        Obj::Bytes(bytes) => Identity::Bytes(bytes.clone()),
        Obj::Duration(unit, ticks) => {
            Identity::Duration(i128::from(*ticks) * i128::from(unit.nanos()))
        }
        Obj::Time(unit, t) => Identity::Time(i128::from(*t) * i128::from(unit.nanos())),
        Obj::Decimal(digits, scale) => match decimal::identity(*digits, *scale) {
            decimal::Identity::Int(v) => Identity::Int(v),
            decimal::Identity::Float(v) => float(v),
            decimal::Identity::Decimal(digits, scale) => Identity::Decimal(digits, scale),
        },
        Obj::None | Obj::Na | Obj::NaT => Identity::None,
        Obj::List(_) => {
            return Err(LabelError::new("TypeError", "unhashable type: 'list'"));
        }
    })
}

/// One group's aggregate, before the cast.
enum Scalar {
    F32(f32),
    F64(f64),
    I64(i64),
    Bool(bool),
    Time(i64),
    Obj(Obj),
}

/// `execute_aggregation_type(op, feat, batch, batch_size, filter_na)`.
pub fn aggregate(
    op: Aggregation,
    feat: &Column,
    batch: &[i64],
    batch_size: usize,
    filter_na: bool,
) -> Result<(Column, Vec<bool>), LabelError> {
    aggregate_with_decimal(op, feat, batch, batch_size, filter_na, None)
}

pub fn aggregate_with_decimal(
    op: Aggregation,
    feat: &Column,
    batch: &[i64],
    batch_size: usize,
    filter_na: bool,
    decimal_scale: Option<i8>,
) -> Result<(Column, Vec<bool>), LabelError> {
    aggregate_with_warnings(
        op,
        feat,
        batch,
        batch_size,
        filter_na,
        decimal_scale,
        &mut Vec::new(),
    )
}

pub fn aggregate_with_warnings(
    op: Aggregation,
    feat: &Column,
    batch: &[i64],
    batch_size: usize,
    filter_na: bool,
    decimal_scale: Option<i8>,
    warnings: &mut Vec<crate::Warning>,
) -> Result<(Column, Vec<bool>), LabelError> {
    if batch.len() != feat.len() {
        return Err(LabelError::new(
            "IndexError",
            format!(
                "boolean index did not match indexed array along axis 0; size of axis is {} but \
                 size of corresponding boolean axis is {}",
                batch.len(),
                feat.len()
            ),
        ));
    }
    if op == Aggregation::Sum && feat.is_datetime() {
        return Err(LabelError::new(
            "TypeError",
            "datetime64 type does not support operation 'sum'",
        ));
    }
    let notna = feat.notna();
    let feat = feat.filter(&notna);
    let batch: Vec<i64> = batch
        .iter()
        .zip(&notna)
        .filter(|(_, keep)| **keep)
        .map(|(b, _)| *b)
        .collect();

    let (out, keys) = if op == Aggregation::ListDistinct {
        list_distinct(&feat, &batch)?
    } else {
        let mut groups: BTreeMap<i64, Vec<usize>> = BTreeMap::new();
        for (row, key) in batch.iter().enumerate() {
            groups.entry(*key).or_default().push(row);
        }
        let keys: Vec<i64> = groups.keys().copied().collect();
        let scalars = groups
            .values()
            .map(|rows| {
                if op == Aggregation::Sum
                    && let Some(scale) = decimal_scale
                {
                    decimal_sum(&feat, rows, scale)
                } else {
                    reduce(op, &feat, rows)
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        let out = column_of(op, &feat, scalars)?;
        let out = if out.is_datetime() {
            out
        } else {
            out.to_f32_with_warnings(warnings)
                .map_err(|message| LabelError::new("ValueError", message))?
        };
        (out, keys)
    };

    if matches!(op, Aggregation::Sum | Aggregation::Count) {
        // reindex(range(batch_size), fill_value=0)
        let values = match &out {
            Column::F32(v) => v.clone(),
            _ => out
                .to_f32()
                .map_err(|m| LabelError::new("ValueError", m))
                .map(|c| match c {
                    Column::F32(v) => v,
                    _ => Vec::new(),
                })?,
        };
        let mut filled = filled(0f32, batch_size)?;
        for (key, value) in keys.iter().zip(values) {
            if (0..batch_size as i64).contains(key) {
                filled[*key as usize] = value;
            }
        }
        return Ok((Column::F32(filled), self::filled(true, batch_size)?));
    }

    let mut mask = filled(false, batch_size)?;
    for &b in &batch {
        mask[numpy_index(b, batch_size)?] = true;
    }
    if filter_na {
        return Ok((out, mask));
    }
    Ok((reindex(&out, &keys, batch_size)?, mask))
}

/// `n` copies of `value`, or the error numpy gives for an array it can't
/// make. `batch_size` comes from the caller, so a vector of its length must
/// not reach an infallible allocation, which would abort the process.
fn filled<T: Clone>(value: T, n: usize) -> Result<Vec<T>, LabelError> {
    let mut out = room(n)?;
    out.resize(n, value);
    Ok(out)
}

/// `f` of each of `at`, collected with [`room`].
fn gathered<T>(at: &[Option<usize>], f: impl Fn(Option<usize>) -> T) -> Result<Vec<T>, LabelError> {
    let mut out = room(at.len())?;
    out.extend(at.iter().map(|&i| f(i)));
    Ok(out)
}

/// An empty vector with room for `n` values: a `ValueError` past the
/// largest size an array can have, as numpy's, and a `MemoryError` when the
/// allocation fails.
fn room<T>(n: usize) -> Result<Vec<T>, LabelError> {
    let bytes = n
        .checked_mul(size_of::<T>())
        .filter(|&bytes| isize::try_from(bytes).is_ok());
    let Some(bytes) = bytes else {
        return Err(LabelError::new(
            "ValueError",
            "array is too big; `arr.size * arr.dtype.itemsize` is larger than the maximum \
             possible size.",
        ));
    };
    let mut out = Vec::new();
    out.try_reserve_exact(n).map_err(|_| {
        LabelError::new(
            "MemoryError",
            format!("Unable to allocate {bytes} bytes for an array of {n} values"),
        )
    })?;
    Ok(out)
}

/// `out.reindex(range(batch_size), fill_value=pd.NA)`.
fn reindex(out: &Column, keys: &[i64], batch_size: usize) -> Result<Column, LabelError> {
    let mut at: Vec<Option<usize>> = filled(None, batch_size)?;
    for (i, key) in keys.iter().enumerate() {
        if (0..batch_size as i64).contains(key) {
            at[*key as usize] = Some(i);
        }
    }
    Ok(match out {
        Column::F32(v) => Column::F32(gathered(&at, |i| i.map_or(f32::NAN, |i| v[i]))?),
        Column::F64(v) => Column::F64(gathered(&at, |i| i.map_or(f64::NAN, |i| v[i]))?),
        // An integer column can't hold the gap, so it becomes float64.
        Column::I64(v) => Column::F64(gathered(&at, |i| i.map_or(f64::NAN, |i| v[i] as f64))?),
        Column::MI64(v) => Column::MI64(gathered(&at, |i| i.and_then(|i| v[i]))?),
        Column::MF64(v) => Column::MF64(gathered(&at, |i| i.and_then(|i| v[i]))?),
        Column::MBool(v) => Column::MBool(gathered(&at, |i| i.and_then(|i| v[i]))?),
        Column::Time(u, v) => Column::Time(*u, gathered(&at, |i| i.and_then(|i| v[i]))?),
        Column::Str(v) => Column::Str(gathered(&at, |i| i.and_then(|i| v[i].clone()))?),
        Column::Bool(v) => Column::Obj(gathered(&at, |i| i.map_or(Obj::Na, |i| Obj::Bool(v[i])))?),
        Column::Obj(v) => Column::Obj(gathered(&at, |i| i.map_or(Obj::Na, |i| v[i].clone()))?),
    })
}

/// `drop_duplicates()` then `groupby('batch')['feat'].agg(list)`: each
/// example's distinct values in row order, examples in order.
fn list_distinct(feat: &Column, batch: &[i64]) -> Result<(Column, Vec<i64>), LabelError> {
    if feat.is_empty() {
        return Ok((feat.empty_like(), Vec::new()));
    }
    let values = feat.iter_objects();
    let mut seen = HashSet::new();
    let mut groups: BTreeMap<i64, Vec<Obj>> = BTreeMap::new();
    for (value, key) in values.into_iter().zip(batch) {
        if seen.insert((identity(&value)?, *key)) {
            groups.entry(*key).or_default().push(value);
        }
    }
    let keys = groups.keys().copied().collect();
    let lists = groups.into_values().map(Obj::List).collect();
    Ok((Column::Obj(lists), keys))
}

/// One group's aggregate of `feat`'s `rows`.
fn reduce(op: Aggregation, feat: &Column, rows: &[usize]) -> Result<Scalar, LabelError> {
    let unsupported = |dtype: &str| {
        LabelError::new(
            "TypeError",
            format!(
                "{dtype} type does not support operation '{}'",
                op.pandas_name()
            ),
        )
    };
    match op {
        Aggregation::Count => return Ok(Scalar::I64(rows.len() as i64)),
        Aggregation::CountDistinct => {
            return Err(LabelError::new(
                "AttributeError",
                "'SeriesGroupBy' object has no attribute 'count_distinct'",
            ));
        }
        Aggregation::First => return Ok(scalar_at(feat, rows[0])),
        Aggregation::Last => return Ok(scalar_at(feat, rows[rows.len() - 1])),
        Aggregation::ListDistinct => return Err(misrouted()),
        _ => {}
    }
    Ok(match feat {
        Column::F32(v) => {
            let vals: Vec<f32> = rows.iter().map(|&r| v[r]).collect();
            Scalar::F32(match op {
                Aggregation::Sum => kahan_sum_f32(vals.iter().copied()),
                Aggregation::Avg => mean_f32(&vals),
                Aggregation::Min => extreme(vals.iter().copied(), f32::INFINITY, false),
                _ => extreme(vals.iter().copied(), f32::NEG_INFINITY, true),
            })
        }
        Column::F64(_) | Column::MF64(_) => {
            let vals: Vec<f64> = rows
                .iter()
                .map(|&r| match feat {
                    Column::F64(v) => v[r],
                    Column::MF64(v) => v[r].unwrap_or(f64::NAN),
                    _ => f64::NAN,
                })
                .collect();
            Scalar::F64(match op {
                Aggregation::Sum => kahan_sum_f64(vals.iter().copied()),
                Aggregation::Avg => mean_f64(&vals),
                Aggregation::Min => extreme(vals.iter().copied(), f64::INFINITY, false),
                _ => extreme(vals.iter().copied(), f64::NEG_INFINITY, true),
            })
        }
        Column::I64(_) | Column::MI64(_) | Column::Bool(_) | Column::MBool(_) => {
            let vals: Vec<i64> = rows
                .iter()
                .map(|&r| match feat {
                    Column::I64(v) => v[r],
                    Column::MI64(v) => v[r].unwrap_or(0),
                    Column::Bool(v) => i64::from(v[r]),
                    Column::MBool(v) => i64::from(v[r].unwrap_or(false)),
                    _ => 0,
                })
                .collect();
            let is_bool = feat.is_bool_dtype();
            match op {
                Aggregation::Sum => Scalar::I64(vals.iter().fold(0i64, |a, b| a.wrapping_add(*b))),
                Aggregation::Avg => {
                    let floats: Vec<f64> = vals.iter().map(|v| *v as f64).collect();
                    Scalar::F64(mean_f64(&floats))
                }
                Aggregation::Min if is_bool => {
                    Scalar::Bool(vals.iter().min().is_some_and(|v| *v == 1))
                }
                Aggregation::Max if is_bool => {
                    Scalar::Bool(vals.iter().max().is_some_and(|v| *v == 1))
                }
                Aggregation::Min => Scalar::I64(vals.iter().copied().min().unwrap_or(0)),
                _ => Scalar::I64(vals.iter().copied().max().unwrap_or(0)),
            }
        }
        Column::Time(_, v) => {
            let vals: Vec<i64> = rows.iter().filter_map(|&r| v[r]).collect();
            match op {
                Aggregation::Sum => return Err(unsupported("datetime64")),
                Aggregation::Avg => {
                    let floats: Vec<f64> = vals.iter().map(|v| *v as f64).collect();
                    Scalar::Time(mean_f64(&floats) as i64)
                }
                Aggregation::Min => Scalar::Time(vals.iter().copied().min().unwrap_or(0)),
                _ => Scalar::Time(vals.iter().copied().max().unwrap_or(0)),
            }
        }
        Column::Str(_) | Column::Obj(_) => {
            let vals: Vec<Obj> = {
                let all = feat.objects();
                rows.iter().map(|&r| all[r].clone()).collect()
            };
            match op {
                Aggregation::Avg => {
                    return Err(LabelError::new(
                        "TypeError",
                        format!("agg function failed [how->mean,dtype->{}]", feat.dtype()),
                    ));
                }
                Aggregation::Sum => Scalar::Obj(object_sum(&vals)?),
                Aggregation::Min | Aggregation::Max => {
                    let mut best = vals[0].clone();
                    for v in &vals[1..] {
                        let ord =
                            v.py_cmp(&best, if op == Aggregation::Min { "<" } else { ">" })?;
                        let better = match op {
                            Aggregation::Min => ord == Some(std::cmp::Ordering::Less),
                            _ => ord == Some(std::cmp::Ordering::Greater),
                        };
                        if better {
                            best = v.clone();
                        }
                    }
                    Scalar::Obj(best)
                }
                _ => return Err(misrouted()),
            }
        }
    })
}

/// The callers route LIST_DISTINCT, COUNT, FIRST and LAST elsewhere.
fn misrouted() -> LabelError {
    LabelError::new("RuntimeError", "an aggregation reached the wrong kernel")
}

fn decimal_sum(feat: &Column, rows: &[usize], scale: i8) -> Result<Scalar, LabelError> {
    let Column::Obj(values) = feat else {
        return Err(misrouted());
    };
    let mut total = arrow_buffer::i256::ZERO;
    for row in rows {
        let Obj::Decimal(digits, actual_scale) = values[*row] else {
            return Err(misrouted());
        };
        if actual_scale != scale {
            return Err(misrouted());
        }
        total = total
            .checked_add(arrow_buffer::i256::from_i128(digits))
            .ok_or_else(|| {
                LabelError::new(
                    "OverflowError",
                    "decimal sum exceeds the supported accumulator",
                )
            })?;
    }
    let value = format!("{total}e{}", -i32::from(scale))
        .parse::<f64>()
        .map_err(|_| {
            LabelError::new(
                "ValueError",
                "decimal sum cannot be represented as a number",
            )
        })?;
    Ok(Scalar::F64(value))
}

/// The least (`max` false) or greatest of `values` as `group_min_max`
/// finds it: a later value replaces the one held only when strictly
/// smaller (or greater), so the first of equals, `0.0` or `-0.0`, stays.
fn extreme<T: PartialOrd + Copy>(values: impl Iterator<Item = T>, start: T, max: bool) -> T {
    values.fold(start, |held, v| {
        if (max && v > held) || (!max && v < held) {
            v
        } else {
            held
        }
    })
}

/// `group_sum` over objects: the first value, then `+` in row order.
fn object_sum(values: &[Obj]) -> Result<Obj, LabelError> {
    let mut acc = values[0].clone();
    for v in &values[1..] {
        acc = match (&acc, v) {
            (Obj::Str(a), Obj::Str(b)) => Obj::Str(format!("{a}{b}")),
            (Obj::Int(a), Obj::Int(b)) => Obj::Int(a.wrapping_add(*b)),
            (a, b) => {
                return Err(LabelError::new(
                    "TypeError",
                    format!(
                        "unsupported operand type(s) for +: '{}' and '{}'",
                        a.type_name(),
                        b.type_name()
                    ),
                ));
            }
        };
    }
    Ok(acc)
}

fn scalar_at(feat: &Column, row: usize) -> Scalar {
    match feat {
        Column::F32(v) => Scalar::F32(v[row]),
        Column::F64(v) => Scalar::F64(v[row]),
        Column::MF64(v) => Scalar::F64(v[row].unwrap_or(f64::NAN)),
        Column::I64(v) => Scalar::I64(v[row]),
        Column::MI64(v) => Scalar::I64(v[row].unwrap_or(0)),
        Column::Bool(v) => Scalar::Bool(v[row]),
        Column::MBool(v) => Scalar::Bool(v[row].unwrap_or(false)),
        Column::Time(_, v) => Scalar::Time(v[row].unwrap_or(i64::MIN)),
        other => Scalar::Obj(other.objects()[row].clone()),
    }
}

/// The groups' aggregates as the column `agg` returns.
fn column_of(op: Aggregation, feat: &Column, scalars: Vec<Scalar>) -> Result<Column, LabelError> {
    if op == Aggregation::Count {
        return Ok(Column::I64(
            scalars
                .into_iter()
                .map(|s| if let Scalar::I64(v) = s { v } else { 0 })
                .collect(),
        ));
    }
    if let Column::Time(unit, _) = feat {
        return Ok(Column::Time(
            *unit,
            scalars
                .into_iter()
                .map(|s| {
                    if let Scalar::Time(v) = s {
                        Some(v)
                    } else {
                        None
                    }
                })
                .collect(),
        ));
    }
    let mut f32s = Vec::new();
    let mut f64s = Vec::new();
    let mut i64s = Vec::new();
    let mut bools = Vec::new();
    let mut objs = Vec::new();
    for s in scalars {
        match s {
            Scalar::F32(v) => f32s.push(v),
            Scalar::F64(v) => f64s.push(v),
            Scalar::I64(v) => i64s.push(v),
            Scalar::Bool(v) => bools.push(v),
            Scalar::Time(v) => i64s.push(v),
            Scalar::Obj(v) => objs.push(v),
        }
    }
    Ok(if !objs.is_empty() {
        Column::Obj(objs)
    } else if !f32s.is_empty() {
        Column::F32(f32s)
    } else if !f64s.is_empty() {
        if matches!(feat, Column::MF64(_) | Column::MI64(_) | Column::MBool(_)) {
            Column::MF64(f64s.into_iter().map(Some).collect())
        } else {
            Column::F64(f64s)
        }
    } else if !bools.is_empty() {
        Column::Bool(bools)
    } else if !i64s.is_empty() {
        Column::I64(i64s)
    } else {
        // No group: an empty column, which the cast makes float32.
        Column::F32(Vec::new())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sum_of(feat: Column) -> Column {
        let n = feat.len();
        aggregate(Aggregation::Sum, &feat, &vec![0; n], 1, true)
            .unwrap()
            .0
    }

    #[test]
    fn float32_sums_compensate_as_pandas() {
        // A naive float32 sum stays at 2**24; pandas' Kahan sum doesn't.
        let feat = Column::F32(vec![16_777_216.0, 1.0, 1.0]);
        assert_eq!(sum_of(feat.clone()), Column::F32(vec![16_777_218.0]));
        let mean = aggregate(Aggregation::Avg, &feat, &[0, 0, 0], 1, true)
            .unwrap()
            .0;
        assert_eq!(mean, Column::F32(vec![5_592_406.0]));
    }

    #[test]
    fn a_batch_size_no_array_can_have_is_an_error_not_an_abort() {
        let feat = Column::F32(vec![1.0]);
        // usize::MAX values are past isize::MAX bytes; 2**59 of them, at
        // least 512 PiB, are past any address space.
        for (size, code) in [(usize::MAX, "ValueError"), (1 << 59, "MemoryError")] {
            for (op, filter_na) in [
                (Aggregation::Sum, true),
                (Aggregation::Max, true),
                (Aggregation::Max, false),
            ] {
                let error = aggregate(op, &feat, &[0], size, filter_na).unwrap_err();
                assert_eq!(error.code, code, "{op:?} of batch_size {size}");
            }
        }
    }

    #[test]
    fn integer_sums_wrap() {
        let feat = Column::I64(vec![i64::MAX, 1]);
        assert_eq!(sum_of(feat), Column::F32(vec![i64::MIN as f32]));
    }

    #[test]
    fn arrow_decimal_sum_is_exact_until_the_final_float32_cast() {
        let large = 10i128.pow(28);
        let feat = Column::Obj(vec![
            Obj::Decimal(large, 0),
            Obj::Decimal(1, 0),
            Obj::Decimal(-large, 0),
            Obj::None,
        ]);
        let (sum, mask) =
            aggregate_with_decimal(Aggregation::Sum, &feat, &[0, 0, 0, 1], 3, false, Some(0))
                .unwrap();
        assert_eq!(sum, Column::F32(vec![1.0, 0.0, 0.0]));
        assert_eq!(mask, vec![true, true, true]);
        assert!(aggregate(Aggregation::Sum, &feat, &[0, 0, 0, 1], 3, false).is_err());
    }

    #[test]
    fn arrow_decimal_sum_keeps_scale_and_large_intermediate_coefficients() {
        let large = 10i128.pow(38) - 1;
        for (values, scale, expected) in [
            (vec![large, large, -large, -large, 1], 0, 1.0),
            (vec![125, -25, 1], 2, 1.01),
            (vec![10, -5], -2, 500.0),
            (vec![1_000_000_059_604_644_775_390_625_000_001], 30, 1.0),
        ] {
            let batch = vec![0; values.len()];
            let feat = Column::Obj(
                values
                    .into_iter()
                    .map(|value| Obj::Decimal(value, scale))
                    .collect(),
            );
            let (sum, _) =
                aggregate_with_decimal(Aggregation::Sum, &feat, &batch, 1, true, Some(scale))
                    .unwrap();
            assert_eq!(sum, Column::F32(vec![expected]));
        }
    }

    #[test]
    fn missing_groups_are_zero_for_sums_and_masked_otherwise() {
        let feat = Column::MI64(vec![Some(2), None, Some(4)]);
        let (sum, mask) = aggregate(Aggregation::Sum, &feat, &[0, 1, 2], 3, false).unwrap();
        assert_eq!(sum, Column::F32(vec![2.0, 0.0, 4.0]));
        assert_eq!(mask, [true, true, true]);
        let (max, mask) = aggregate(Aggregation::Max, &feat, &[0, 1, 2], 3, false).unwrap();
        assert_eq!(mask, [true, false, true]);
        let Column::F32(max) = max else {
            panic!("float32")
        };
        assert!(max[1].is_nan() && max[0] == 2.0 && max[2] == 4.0);
    }

    #[test]
    fn distinct_lists_keep_the_first_of_equals() {
        let feat = Column::F64(vec![0.0, -0.0, 1.0]);
        let (out, _) = aggregate(Aggregation::ListDistinct, &feat, &[0, 0, 0], 1, true).unwrap();
        assert_eq!(
            out,
            Column::Obj(vec![Obj::List(vec![Obj::Float(0.0), Obj::Float(1.0)])])
        );
    }
}
