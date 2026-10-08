// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `PQueryPandasExecutor`: a query's labels over the rows the sampler
//! fetched, each step as pandas takes it, down to which side of a mismatched
//! `&` decides.

use std::cell::RefCell;
use std::cmp::Ordering;

use crate::LabelError;
use crate::aggregate::{Aggregation, aggregate};
use crate::column::{Column, Obj, TimeUnit};
use crate::date_offset::DateOffset;
use crate::query::{Constant, Node, Query, Typed, Window};
use sdm_schema::dt::Host;

/// One table's rows: its columns by name, all of one length.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Frame {
    pub columns: Vec<(String, Column)>,
}

impl Frame {
    pub fn len(&self) -> usize {
        self.columns.first().map_or(0, |(_, c)| c.len())
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn get(&self, name: &str) -> Option<&Column> {
        self.columns.iter().find(|(n, _)| n == name).map(|(_, c)| c)
    }
}

/// What the sampler hands the executor for one batch.
#[derive(Clone, Debug)]
pub struct Batch {
    pub feat: Vec<(String, Frame)>,
    pub arrow_decimal_scales: Vec<(String, i8)>,
    /// The time column of each table that has one.
    pub time: Vec<(String, Column)>,
    /// Each row's example.
    pub batch: Vec<(String, Vec<i64>)>,
    /// Each example's anchor time.
    pub anchor: Column,
    /// What reading a date constant takes from the Python process.
    pub host: Host,
    pub warnings: RefCell<Vec<crate::Warning>>,
}

impl Default for Batch {
    fn default() -> Self {
        Self {
            feat: Vec::new(),
            arrow_decimal_scales: Vec::new(),
            time: Vec::new(),
            batch: Vec::new(),
            anchor: Column::default(),
            host: Host::assumed(),
            warnings: RefCell::new(Vec::new()),
        }
    }
}

impl Batch {
    fn user_warnings<T>(
        &self,
        operation: impl FnOnce(&mut Vec<String>) -> Result<T, LabelError>,
    ) -> Result<T, LabelError> {
        let mut warnings = Vec::new();
        let result = operation(&mut warnings);
        self.warnings
            .borrow_mut()
            .extend(warnings.into_iter().map(crate::Warning::User));
        result
    }

    fn frame(&self, table: &str) -> Result<&Frame, LabelError> {
        self.feat
            .iter()
            .find(|(n, _)| n == table)
            .map(|(_, f)| f)
            .ok_or_else(|| LabelError::key_error(table))
    }

    fn time_of(&self, table: &str) -> Result<&Column, LabelError> {
        self.time
            .iter()
            .find(|(n, _)| n == table)
            .map(|(_, c)| c)
            .ok_or_else(|| LabelError::key_error(table))
    }

    fn batch_of(&self, table: &str) -> Result<&Vec<i64>, LabelError> {
        self.batch
            .iter()
            .find(|(n, _)| n == table)
            .map(|(_, b)| b)
            .ok_or_else(|| LabelError::key_error(table))
    }
}

/// A labelled result: the values and the mask of the examples they belong
/// to.
pub type Labelled = (Column, Vec<bool>);

/// numpy's `a & b` of two bool arrays: elementwise, a length-1 side
/// broadcast, other lengths refused.
fn and(a: &[bool], b: &[bool]) -> Result<Vec<bool>, LabelError> {
    match (a.len(), b.len()) {
        (x, y) if x == y => Ok(a.iter().zip(b).map(|(x, y)| *x && *y).collect()),
        (1, _) => Ok(b.iter().map(|y| a[0] && *y).collect()),
        (_, 1) => Ok(a.iter().map(|x| *x && b[0]).collect()),
        (x, y) => Err(LabelError::new(
            "ValueError",
            format!("operands could not be broadcast together with shapes ({x},) ({y},) "),
        )),
    }
}

/// numpy's `values[mask]` for a bool array: an empty mask selects nothing
/// from any array; another length is refused.
fn np_select<T: Clone>(values: &[T], mask: &[bool]) -> Result<Vec<T>, LabelError> {
    if mask.is_empty() {
        return Ok(Vec::new());
    }
    if mask.len() != values.len() {
        return Err(LabelError::new(
            "IndexError",
            format!(
                "boolean index did not match indexed array along axis 0; size of axis is {} but \
                 size of corresponding boolean axis is {}",
                values.len(),
                mask.len()
            ),
        ));
    }
    Ok(values
        .iter()
        .zip(mask)
        .filter(|(_, keep)| **keep)
        .map(|(v, _)| v.clone())
        .collect())
}

/// pandas' `series[mask]` for a bool array: the lengths must match.
fn boolean_index(column: &Column, mask: &[bool]) -> Result<Column, LabelError> {
    if mask.len() != column.len() {
        return Err(LabelError::new(
            "IndexError",
            format!(
                "Boolean index has wrong length: {} instead of {}",
                mask.len(),
                column.len()
            ),
        ));
    }
    Ok(column.filter(mask))
}

/// `execute_column`.
pub fn execute_column(fqn: &str, batch: &Batch, filter_na: bool) -> Result<Labelled, LabelError> {
    let parts: Vec<&str> = fqn.split('.').collect();
    let [table, column] = parts.as_slice() else {
        return Err(LabelError::new(
            "ValueError",
            if parts.len() > 2 {
                "too many values to unpack (expected 2)".to_string()
            } else {
                format!(
                    "not enough values to unpack (expected 2, got {})",
                    parts.len()
                )
            },
        ));
    };
    let frame = batch.frame(table)?;
    let mut out = if *column == "*" {
        Column::I64(vec![1; frame.len()])
    } else {
        frame
            .get(column)
            .cloned()
            .ok_or_else(|| LabelError::key_error(column))?
    };
    if out.is_float_dtype() {
        out = out
            .to_f32_with_warnings(&mut batch.warnings.borrow_mut())
            .map_err(|message| LabelError::new("ValueError", message))?;
    }
    let mask = out.notna();
    if !filter_na {
        return Ok((out, mask));
    }
    Ok((out.filter(&mask).to_primitive(), mask))
}

/// The constant cast to the left side's dtype,
/// `pd.Series([value], dtype=left.dtype).iloc[0]`, as a scalar to compare.
fn cast_scalar(value: &Obj, left: &Column) -> Result<Obj, LabelError> {
    let invalid_int = |text: &str| {
        LabelError::new(
            "ValueError",
            format!("invalid literal for int() with base 10: '{text}'"),
        )
    };
    let not_float = |text: &str| {
        LabelError::new(
            "ValueError",
            format!("could not convert string to float: '{text}'"),
        )
    };
    Ok(match left {
        Column::F32(_) => match value {
            Obj::Str(s) => {
                Obj::NpFloat32(crate::query::python_float(s).ok_or_else(|| not_float(s))? as f32)
            }
            Obj::Bool(b) => Obj::NpFloat32(f32::from(u8::from(*b))),
            Obj::Int(i) => Obj::NpFloat32(*i as f32),
            Obj::Float(f) => Obj::NpFloat32(*f as f32),
            other => other.clone(),
        },
        Column::F64(_) | Column::MF64(_) => match value {
            Obj::Str(s) => Obj::Float(crate::query::python_float(s).ok_or_else(|| not_float(s))?),
            Obj::Bool(b) => Obj::Float(f64::from(u8::from(*b))),
            Obj::Int(i) => Obj::Float(*i as f64),
            other => other.clone(),
        },
        Column::I64(_) | Column::MI64(_) => match value {
            Obj::Str(s) => Obj::Int(s.trim().parse::<i64>().map_err(|_| invalid_int(s))?),
            Obj::Bool(b) => Obj::Int(i64::from(*b)),
            Obj::Float(f) => Obj::Int(*f as i64),
            other => other.clone(),
        },
        Column::Bool(_) | Column::MBool(_) => match value {
            Obj::Int(i) => Obj::Bool(*i != 0),
            Obj::Float(f) => Obj::Bool(*f != 0.0),
            Obj::Str(s) => Obj::Bool(!s.is_empty()),
            other => other.clone(),
        },
        Column::Time(unit, _) => match value {
            Obj::Time(u, t) => {
                let ns = i128::from(*t) * i128::from(u.nanos());
                Obj::Time(*unit, ns.div_euclid(i128::from(unit.nanos())) as i64)
            }
            Obj::Int(i) => Obj::Time(*unit, *i),
            Obj::Str(s) => match crate::datetime::parse_timestamp(s) {
                Some(ns) => Obj::Time(TimeUnit::Ns, ns),
                None => {
                    return Err(LabelError::new(
                        "ValueError",
                        format!("Unknown datetime string format, unable to parse: {s}"),
                    ));
                }
            },
            other => other.clone(),
        },
        Column::Str(_) => match value {
            Obj::Str(s) => Obj::Str(s.clone()),
            Obj::Int(i) => Obj::Str(i.to_string()),
            Obj::Bool(b) => Obj::Str(if *b { "True" } else { "False" }.to_string()),
            other => other.clone(),
        },
        Column::Obj(_) => value.clone(),
    })
}

fn compare(op: &str, ord: Option<Ordering>) -> bool {
    match (op, ord) {
        ("=", Some(o)) => o == Ordering::Equal,
        ("!=", Some(o)) => o != Ordering::Equal,
        ("!=", None) => true,
        ("<=", Some(o)) => o != Ordering::Greater,
        (">=", Some(o)) => o != Ordering::Less,
        ("<", Some(o)) => o == Ordering::Less,
        (">", Some(o)) => o == Ordering::Greater,
        _ => false,
    }
}

/// `execute_rel_op(left, op, right)`.
pub fn execute_rel_op(
    left: &Column,
    op: &str,
    right: &Constant,
    host: &Host,
) -> Result<Column, LabelError> {
    execute_rel_op_with_warnings(left, op, right, host, &mut Vec::new())
}

fn execute_rel_op_with_warnings(
    left: &Column,
    op: &str,
    right: &Constant,
    host: &Host,
    warnings: &mut Vec<String>,
) -> Result<Column, LabelError> {
    if matches!(right.typed_with_warnings(host, warnings)?, Typed::None) {
        let notna = left.notna();
        return match op {
            "=" => Ok(Column::Bool(notna.iter().map(|x| !x).collect())),
            "!=" => Ok(Column::Bool(notna)),
            _ => Err(LabelError::new("AssertionError", "")),
        };
    }
    let value = match right.typed_with_warnings(host, warnings)? {
        Typed::None => return Err(LabelError::new("AssertionError", "")),
        Typed::Scalar(value) => value,
        Typed::List(_) => {
            return Err(LabelError::new(
                "ValueError",
                "Lengths must match to compare",
            ));
        }
    };
    if !matches!(op, "=" | "!=" | "<=" | ">=" | "<" | ">") {
        return Err(LabelError::new(
            "NotImplementedError",
            format!("Operator '{op}' not implemented"),
        ));
    }
    let left = if left.is_integer_dtype() && matches!(value, Obj::Float(_)) {
        left.int_to_f64()
    } else {
        left.clone()
    };
    let value = cast_scalar(&value, &left)?;
    let notna = left.notna();
    let objects = left.objects();
    let symbol = match op {
        "=" => "==",
        other => other,
    };
    let mut out = Vec::with_capacity(objects.len());
    for (item, present) in objects.iter().zip(&notna) {
        if !present {
            // fillna(False); a missing value is never unequal either.
            out.push(false);
            continue;
        }
        let ord = if matches!(op, "=" | "!=") {
            Some(if item.py_eq(&value) {
                Ordering::Equal
            } else {
                Ordering::Less
            })
        } else {
            item.py_cmp(&value, symbol)?
        };
        out.push(compare(op, ord));
    }
    Ok(Column::Bool(out))
}

/// `execute_member_op(left, IN, right)`.
pub fn execute_member_op(
    left: &Column,
    op: &str,
    right: &Constant,
    host: &Host,
) -> Result<Column, LabelError> {
    execute_member_op_with_warnings(left, op, right, host, &mut Vec::new())
}

fn execute_member_op_with_warnings(
    left: &Column,
    op: &str,
    right: &Constant,
    host: &Host,
    warnings: &mut Vec<String>,
) -> Result<Column, LabelError> {
    if op != "IN" {
        return Err(LabelError::new(
            "NotImplementedError",
            format!("Operator '{op}' not implemented"),
        ));
    }
    let values = match right.typed_with_warnings(host, warnings)? {
        Typed::List(items) => items,
        Typed::None => Vec::new(),
        Typed::Scalar(_) => {
            return Err(LabelError::new(
                "TypeError",
                "Index(...) must be called with a collection of some kind",
            ));
        }
    };
    let mut cast = Vec::new();
    let mut has_na = false;
    for item in &values {
        if let (Typed::Scalar(Obj::Float(f)), true) = (item, left.is_integer_dtype())
            && f.fract() != 0.0
        {
            return Err(match left {
                Column::I64(_) => {
                    LabelError::new("ValueError", "Trying to coerce float values to integers")
                }
                _ => LabelError::new(
                    "TypeError",
                    "cannot safely cast non-equivalent object to int64",
                ),
            });
        }
    }
    for item in values {
        match item {
            Typed::Scalar(v) => cast.push(cast_scalar(&v, left)?),
            _ => has_na = true,
        }
    }
    let notna = left.notna();
    let out = left
        .objects()
        .iter()
        .zip(&notna)
        .map(|(item, present)| {
            if !present {
                return has_na;
            }
            cast.iter().any(|v| item.py_eq(v))
        })
        .collect();
    Ok(Column::Bool(out))
}

/// `execute_bool_op(left, op, right)`, with pandas' alignment of two
/// lengths: a position only one side has is `False` on the left and
/// counts as `False` on the right.
pub fn execute_bool_op(
    left: &Column,
    op: &str,
    right: Option<&Column>,
) -> Result<Column, LabelError> {
    let bools = |c: &Column| {
        c.bools().ok_or_else(|| {
            LabelError::new(
                "TypeError",
                format!(
                    "unsupported operand type(s) for &: '{}' and 'bool'",
                    c.dtype()
                ),
            )
        })
    };
    let a = bools(left)?;
    if op == "NOT" {
        return Ok(Column::Bool(a.iter().map(|x| !x).collect()));
    }
    let Some(right) = right else {
        return Err(LabelError::new("AssertionError", ""));
    };
    if !matches!(op, "AND" | "OR") {
        return Err(LabelError::new(
            "NotImplementedError",
            format!("Operator '{op}' not implemented"),
        ));
    }
    let b = bools(right)?;
    let n = a.len().max(b.len());
    let out = (0..n)
        .map(|i| {
            let Some(&x) = a.get(i) else {
                return false;
            };
            let y = b.get(i).copied().unwrap_or(false);
            match op {
                "AND" => x && y,
                _ => x || y,
            }
        })
        .collect();
    Ok(Column::Bool(out))
}

/// `execute_aggregation_type`.
pub fn execute_aggregation_type(
    op: Aggregation,
    feat: &Column,
    batch: &[i64],
    batch_size: usize,
    filter_na: bool,
) -> Result<Labelled, LabelError> {
    aggregate(op, feat, batch, batch_size, filter_na)
}

/// The instants of a datetime column, in nanoseconds.
fn nanos(column: &Column) -> Result<Vec<Option<i128>>, LabelError> {
    match column {
        Column::Time(unit, values) => Ok(values
            .iter()
            .map(|v| v.map(|t| i128::from(t) * i128::from(unit.nanos())))
            .collect()),
        other => Err(LabelError::new(
            "TypeError",
            format!(
                "Invalid comparison between dtype={} and datetime64",
                other.dtype()
            ),
        )),
    }
}

/// `execute_aggregation`.
pub fn execute_aggregation(
    target: &Node,
    aggr: Aggregation,
    window: Option<&Window>,
    batch: &Batch,
    filter_na: bool,
) -> Result<Labelled, LabelError> {
    let fqn = target.target_column()?;
    let table = fqn.split('.').next().unwrap_or(fqn);
    let target_batch = batch.batch_of(table)?;
    let target_time = batch.time_of(table)?;
    let (target_feat, target_mask) = match target {
        Node::Column { fqn } => execute_column(fqn, batch, true)?,
        Node::Filter { target, condition } => execute_filter(target, condition, batch, true)?,
        _ => {
            return Err(LabelError::new("AssertionError", ""));
        }
    };
    let Some(window) = window else {
        return Err(LabelError::new("AssertionError", ""));
    };
    let Column::Time(anchor_unit, anchor) = &batch.anchor else {
        return Err(LabelError::new(
            "TypeError",
            "the anchor times are not datetimes",
        ));
    };
    let mut anchor_target = Vec::with_capacity(target_batch.len());
    for &b in target_batch {
        let at = usize::try_from(if b < 0 { b + anchor.len() as i64 } else { b })
            .ok()
            .filter(|&i| i < anchor.len())
            .ok_or_else(|| {
                LabelError::new("IndexError", "positional indexers are out-of-bounds")
            })?;
        anchor_target.push(anchor[at]);
    }
    let per_second = anchor_unit.per_second();
    let to_ns = |values: Vec<Option<i64>>| -> Vec<Option<i128>> {
        values
            .into_iter()
            .map(|v| v.map(|t| i128::from(t) * i128::from(anchor_unit.nanos())))
            .collect()
    };
    let times = nanos(target_time)?;
    if times.len() != anchor_target.len() {
        return Err(LabelError::new(
            "ValueError",
            "Can only compare identically-labeled Series objects",
        ));
    }
    let shifted = |offset: DateOffset| {
        offset
            .apply_array(&anchor_target, per_second)
            .map(to_ns)
            .map_err(|e| LabelError::new(e.code, e.message))
    };
    let end = shifted(window.end_offset())?;
    let mut time_filter: Vec<bool> = times
        .iter()
        .zip(&end)
        .map(|(t, e)| matches!((t, e), (Some(t), Some(e)) if t <= e))
        .collect();
    if let Some(start) = window.start_offset() {
        let start = shifted(start)?;
        for (keep, (t, s)) in time_filter.iter_mut().zip(times.iter().zip(&start)) {
            *keep = *keep && matches!((t, s), (Some(t), Some(s)) if t > s);
        }
    }
    let current = and(&target_mask, &time_filter)?;
    let kept_filter = np_select(&time_filter, &target_mask)?;
    let feat = boolean_index(&target_feat, &kept_filter)?;
    let rows = np_select(target_batch, &current)?;
    let decimal_scale = batch
        .arrow_decimal_scales
        .iter()
        .find(|(name, _)| name == fqn)
        .map(|(_, scale)| *scale);
    crate::aggregate::aggregate_with_warnings(
        aggr,
        &feat,
        &rows,
        anchor.len(),
        filter_na,
        decimal_scale,
        &mut batch.warnings.borrow_mut(),
    )
}

/// `execute_condition`.
pub fn execute_condition(
    target: &Node,
    op: &str,
    value: &Constant,
    batch: &Batch,
    filter_na: bool,
) -> Result<Labelled, LabelError> {
    let value_is_na = matches!(
        batch.user_warnings(|warnings| value.typed_with_warnings(&batch.host, warnings))?,
        Typed::None
    );
    let inner = filter_na && !value_is_na;
    let (left, mut mask) = match target {
        Node::Column { fqn } => execute_column(fqn, batch, inner)?,
        Node::Join { rhs_target } => execute_join(rhs_target, batch)?,
        Node::Aggregation {
            target,
            aggr,
            window,
        } => execute_aggregation(target, *aggr, window.as_ref(), batch, inner)?,
        _ => return Err(LabelError::new("AssertionError", "")),
    };
    if filter_na && value_is_na {
        mask = vec![true; left.len()];
    }
    let out = if op == "IN" {
        batch.user_warnings(|warnings| {
            execute_member_op_with_warnings(&left, op, value, &batch.host, warnings)
        })?
    } else {
        batch.user_warnings(|warnings| {
            execute_rel_op_with_warnings(&left, op, value, &batch.host, warnings)
        })?
    };
    Ok((out, mask))
}

/// `execute_join`: only an aggregation joins, and always without nulls.
pub fn execute_join(rhs_target: &Node, batch: &Batch) -> Result<Labelled, LabelError> {
    match rhs_target {
        Node::Aggregation {
            target,
            aggr,
            window,
        } => execute_aggregation(target, *aggr, window.as_ref(), batch, true),
        _ => Err(LabelError::new(
            "NotImplementedError",
            "Unexpected node nested in Join",
        )),
    }
}

fn condition_or_logical(
    node: &Node,
    batch: &Batch,
    filter_na: bool,
) -> Result<Labelled, LabelError> {
    match node {
        Node::Condition { target, op, value } => {
            execute_condition(target, op, value, batch, filter_na)
        }
        Node::Logical {
            left,
            bool_op,
            right,
        } => execute_logical_operation(left, bool_op, right.as_deref(), batch, filter_na),
        _ => Err(LabelError::new("AssertionError", "")),
    }
}

/// `execute_logical_operation`.
pub fn execute_logical_operation(
    left: &Node,
    bool_op: &str,
    right: Option<&Node>,
    batch: &Batch,
    filter_na: bool,
) -> Result<Labelled, LabelError> {
    let (left, mut mask) = condition_or_logical(left, batch, false)?;
    let right = match right {
        Some(node @ (Node::Condition { .. } | Node::Logical { .. })) => {
            Some(condition_or_logical(node, batch, false)?)
        }
        _ => None,
    };
    let out = execute_bool_op(&left, bool_op, right.as_ref().map(|(c, _)| c))?;
    if let Some((_, right_mask)) = &right {
        mask = and(&mask, right_mask)?;
    }
    if filter_na {
        return Ok((boolean_index(&out, &mask)?, mask));
    }
    Ok((out, mask))
}

/// `execute_filter`.
pub fn execute_filter(
    target: &Node,
    condition: &Node,
    batch: &Batch,
    filter_na: bool,
) -> Result<Labelled, LabelError> {
    let Node::Column { fqn } = target else {
        return Err(LabelError::new("AssertionError", ""));
    };
    let (out, mask) = execute_column(fqn, batch, false)?;
    let (kept, _) = condition_or_logical(condition, batch, false)?;
    let kept = kept
        .bools()
        .ok_or_else(|| LabelError::new("TypeError", "a filter's condition is not boolean"))?;
    let both = and(&kept, &mask)?;
    if filter_na {
        return Ok((boolean_index(&out, &both)?, both));
    }
    Ok((boolean_index(&out, &kept)?, both))
}

/// `execute_query`: the labels of a batch and the mask of the examples
/// that have one.
pub fn execute_query(query: &Query, batch: &Batch) -> Result<Labelled, LabelError> {
    execute_query_with(query, batch, true)
}

/// `execute_query(..., filter_na=filter_na)`: with `filter_na` false, as
/// the lagged targets are computed, a label for every example, missing where
/// it has none.
pub fn execute_query_with(
    query: &Query,
    batch: &Batch,
    filter_na: bool,
) -> Result<Labelled, LabelError> {
    let mut mask = match &query.entity {
        Node::Column { fqn } => execute_column(fqn, batch, false)?.1,
        Node::Filter { target, condition } => execute_filter(target, condition, batch, false)?.1,
        _ => return Err(LabelError::new("AssertionError", "")),
    };
    let (out, target_mask) = match &query.target {
        Node::Column { fqn } => execute_column(fqn, batch, filter_na)?,
        Node::Condition { target, op, value } => {
            execute_condition(target, op, value, batch, filter_na)?
        }
        Node::Aggregation {
            target,
            aggr,
            window,
        } => execute_aggregation(target, *aggr, window.as_ref(), batch, filter_na)?,
        Node::Join { rhs_target } => execute_join(rhs_target, batch)?,
        Node::Logical {
            left,
            bool_op,
            right,
        } => execute_logical_operation(left, bool_op, right.as_deref(), batch, filter_na)?,
        Node::Filter { .. } => return Err(LabelError::new("AssertionError", "")),
    };
    if let Some(whatif) = &query.whatif {
        let (assumed, _) = condition_or_logical(whatif, batch, false)?;
        let assumed = assumed
            .bools()
            .ok_or_else(|| LabelError::new("TypeError", "an ASSUMING condition is not boolean"))?;
        mask = and(&mask, &assumed)?;
    }
    if !filter_na {
        return Ok((out, and(&mask, &target_mask)?));
    }
    // out[mask[_mask]]: the entity mask at the examples the target labels.
    let at_labels = np_select(&mask, &target_mask)?;
    let out = boolean_index(&out, &at_labels)?;
    Ok((out, and(&mask, &target_mask)?))
}

#[cfg(test)]
mod warning_tests {
    use super::*;
    use crate::query::ConstantValue;

    #[test]
    fn query_filters_labels_before_combining_masks() {
        let query = Query {
            entity: Node::Column {
                fqn: "entities.id".into(),
            },
            target: Node::Column {
                fqn: "targets.value".into(),
            },
            whatif: None,
        };
        let batch = Batch {
            feat: vec![
                (
                    "entities".into(),
                    Frame {
                        columns: vec![("id".into(), Column::I64(vec![1, 2, 3]))],
                    },
                ),
                (
                    "targets".into(),
                    Frame {
                        columns: vec![("value".into(), Column::I64(vec![4, 5]))],
                    },
                ),
            ],
            ..Batch::default()
        };
        assert_eq!(
            execute_query_with(&query, &batch, true).unwrap_err(),
            LabelError::new(
                "IndexError",
                "boolean index did not match indexed array along axis 0; size of axis is 3 but size of corresponding boolean axis is 2"
            )
        );
        assert_eq!(
            execute_query_with(&query, &batch, false).unwrap_err(),
            LabelError::new(
                "ValueError",
                "operands could not be broadcast together with shapes (3,) (2,) "
            )
        );
    }

    fn date() -> Constant {
        Constant {
            value: ConstantValue::Text("29.02.2024".into()),
            dtype: Some("time".into()),
        }
    }

    fn batch() -> Batch {
        Batch {
            feat: vec![(
                "events".into(),
                Frame {
                    columns: vec![(
                        "when".into(),
                        Column::Time(TimeUnit::Ns, vec![Some(1_709_164_800_000_000_000)]),
                    )],
                },
            )],
            ..Batch::default()
        }
    }

    #[test]
    fn relational_condition_preserves_three_actual_constant_reads() {
        let batch = batch();
        let target = Node::Column {
            fqn: "events.when".into(),
        };
        let (output, _) = execute_condition(&target, "=", &date(), &batch, false).unwrap();
        assert_eq!(output, Column::Bool(vec![true]));
        let warnings = batch.warnings.borrow();
        assert_eq!(warnings.len(), 3);
        assert!(warnings.iter().all(|warning| {
            warning
                .message()
                .starts_with("Parsing dates in %d.%m.%Y format")
        }));
    }

    #[test]
    fn member_condition_preserves_two_actual_constant_reads() {
        let batch = batch();
        let target = Node::Column {
            fqn: "events.when".into(),
        };
        let value = Constant {
            value: ConstantValue::List(vec![date()]),
            dtype: Some("array[time]".into()),
        };
        execute_condition(&target, "IN", &value, &batch, false).unwrap();
        assert_eq!(batch.warnings.borrow().len(), 2);
    }

    #[test]
    fn missing_column_keeps_only_preceding_constant_warning() {
        let batch = batch();
        let target = Node::Column {
            fqn: "events.absent".into(),
        };
        assert!(execute_condition(&target, "=", &date(), &batch, false).is_err());
        assert_eq!(batch.warnings.borrow().len(), 1);
    }

    #[test]
    fn null_constant_short_circuits_without_date_warnings() {
        let batch = batch();
        let target = Node::Column {
            fqn: "events.when".into(),
        };
        let value = Constant {
            value: ConstantValue::Text("null".into()),
            dtype: None,
        };
        execute_condition(&target, "=", &value, &batch, false).unwrap();
        assert!(batch.warnings.borrow().is_empty());
    }

    #[test]
    fn refused_date_does_not_invent_a_format_warning() {
        let mut warnings = Vec::new();
        let value = Constant {
            value: ConstantValue::Text("30.02.2024".into()),
            dtype: Some("time".into()),
        };
        assert!(
            value
                .typed_with_warnings(&Host::assumed(), &mut warnings)
                .is_err()
        );
        assert!(warnings.is_empty());
    }
}
