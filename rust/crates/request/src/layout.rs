// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The flattened layout a result's frame is built from
//! (`PredictionResult.to_pandas`, `tests/rust_core/request/CONTRACT.md`,
//! "The layout").
//!
//! The rows are the items, each a mapping in `to_pandas`'s field order;
//! the columns are the frame's, in the order a field first appears, each
//! with the dtype `pd.DataFrame(rows)` infers for it; the rows are sorted
//! stably by `row_index`. The layout is that frame as `pa.Table.from_pandas`
//! writes it, and each field's metadata says what a host needs to build the
//! frame back from it alone: `sdm.dtype`, the column's pandas dtype, and,
//! for an `object` column, `sdm.cells`, its cells as fixture JSON.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::builder::{
    BooleanBuilder, Float64Builder, Int64Builder, LargeStringBuilder, StringBuilder, UInt64Builder,
};
use arrow_array::{
    Array, ArrayRef, BooleanArray, Float64Array, Int64Array, NullArray, RecordBatch, StructArray,
    TimestampMicrosecondArray, TimestampNanosecondArray, TimestampSecondArray, new_empty_array,
};
use arrow_schema::{DataType, Field, Fields, Schema};
use sdm_schema::pyval::PyVal;
use serde_json::{Map, Value, json};

use crate::result::{Item, Result_};
use crate::values::encode;

/// The field metadata keys a host reads the frame back with.
pub const DTYPE_KEY: &str = "sdm.dtype";
pub const CELLS_KEY: &str = "sdm.cells";

/// One column of the frame: its name, pandas dtype and cells as pandas
/// holds them.
#[derive(Clone, Debug)]
pub struct Column {
    pub name: String,
    pub dtype: String,
    pub cells: Vec<PyVal>,
}

const SCALAR_FIELDS: [&str; 4] = ["row_index", "id", "forecast_step", "prediction"];

/// The row `to_pandas` makes of one item.
fn row_of(item: &Item) -> Vec<(String, PyVal)> {
    let mut row: Vec<(String, PyVal)> = Vec::new();
    let put = |row: &mut Vec<(String, PyVal)>, key: String, value: PyVal| {
        if let Some(entry) = row.iter_mut().find(|(k, _)| *k == key) {
            entry.1 = value;
        } else {
            row.push((key, value));
        }
    };
    for name in SCALAR_FIELDS {
        let value = match name {
            "row_index" => item.row_index.clone(),
            "id" => item.id.clone(),
            "forecast_step" => item.forecast_step.clone(),
            _ => item.prediction.clone(),
        };
        if let Some(value) = value
            && !matches!(value, PyVal::None)
        {
            put(&mut row, name.to_string(), value);
        }
    }
    for (mapping, prefix) in [
        (&item.probabilities, "probability"),
        (&item.quantiles, "quantile"),
    ] {
        if let Some(pairs) = mapping {
            for (key, value) in pairs {
                put(
                    &mut row,
                    format!("{prefix}_{}", crate::pyops::py_str(key)),
                    value.clone(),
                );
            }
        }
    }
    for (values, name) in [(&item.scores, "scores"), (&item.embeddings, "embeddings")] {
        if let Some(values) = values {
            put(&mut row, name.to_string(), PyVal::List(values.clone()));
        }
    }
    if let Some(rankings) = &item.rankings
        && !rankings.is_empty()
    {
        put(
            &mut row,
            "rankings".to_string(),
            PyVal::List(rankings.iter().map(|e| PyVal::Dict(e.clone())).collect()),
        );
    }
    row
}

/// The columns `_empty_frame` gives a result with no rows.
fn empty_columns(result: &Result_) -> Vec<String> {
    empty_columns_with(result, |name| {
        Ok::<_, std::convert::Infallible>(result.outputs.iter().any(|output| output == name))
    })
    .unwrap_or_else(|error| match error {})
}

fn empty_columns_with<E>(
    result: &Result_,
    mut contains: impl FnMut(&str) -> Result<bool, E>,
) -> Result<Vec<String>, E> {
    let mut columns = vec!["row_index".to_string()];
    if result.names_its_rows {
        columns.push("id".to_string());
    }
    if result.names_its_step {
        columns.push("forecast_step".to_string());
    }
    for name in [
        "row_index",
        "id",
        "forecast_step",
        "prediction",
        "scores",
        "embeddings",
        "rankings",
    ] {
        if !matches!(name, "row_index" | "id" | "forecast_step") && contains(name)? {
            columns.push(name.to_string());
        }
    }
    Ok(columns)
}

/// A cell `pd.DataFrame(rows)` fills a missing key with.
fn missing() -> PyVal {
    PyVal::Float(f64::NAN)
}

pub fn requires_host_columns(result: &Result_) -> bool {
    fn held(value: &PyVal) -> bool {
        match value {
            PyVal::Opaque { kind, .. } => kind == "host_ref",
            PyVal::List(values) | PyVal::Tuple(values) => values.iter().any(held),
            PyVal::Dict(values) => values.iter().any(|(key, value)| held(key) || held(value)),
            _ => false,
        }
    }
    result.host_values.is_some()
        || result
            .items
            .iter()
            .flat_map(row_of)
            .any(|(_, value)| held(&value))
}

pub fn host_layout_with<E>(
    result: &Result_,
    infer: impl FnMut(&[PyVal]) -> Result<(String, usize), E>,
    order: impl FnMut(&str, bool, bool, bool) -> Result<Vec<usize>, E>,
    refuse: impl Fn(&str) -> E,
) -> Result<Value, E> {
    host_layout_with_protocol(
        result,
        infer,
        order,
        |_, _| Err(refuse("result requires host protocol facts")),
        &refuse,
    )
}

pub fn host_layout_with_protocol<E>(
    result: &Result_,
    mut infer: impl FnMut(&[PyVal]) -> Result<(String, usize), E>,
    mut order: impl FnMut(&str, bool, bool, bool) -> Result<Vec<usize>, E>,
    mut protocol: impl FnMut(&str, &[PyVal]) -> Result<PyVal, E>,
    refuse: impl Fn(&str) -> E,
) -> Result<Value, E> {
    let rows: Vec<_> = result.items.iter().map(row_of).collect();
    let mut names = if rows.is_empty() {
        empty_columns_with(result, |name| match &result.host_values {
            Some(values) => crate::result::protocol_bool(
                &mut protocol,
                "contains",
                &[values.outputs.clone(), PyVal::Str(name.to_string())],
                &refuse,
            ),
            None => Ok(result.outputs.iter().any(|output| output == name)),
        })?
    } else {
        Vec::new()
    };
    for row in &rows {
        for (name, _) in row {
            if !names.contains(name) {
                names.push(name.clone());
            }
        }
    }
    let mut columns = Vec::new();
    for name in names {
        let cells: Vec<_> = rows
            .iter()
            .map(|row| {
                row.iter()
                    .find(|(key, _)| *key == name)
                    .map_or_else(missing, |(_, value)| value.clone())
            })
            .collect();
        let (reference, length) = infer(&cells)?;
        if length != rows.len() {
            return Err(refuse("host column length differs from result rows"));
        }
        columns.push((name, reference));
    }
    let positions = match columns.iter().find(|(name, _)| name == "row_index") {
        Some((_, reference)) => order(reference, true, true, true)?,
        None => (0..rows.len()).collect(),
    };
    let mut seen = vec![false; rows.len()];
    if positions.len() != rows.len()
        || positions.iter().any(|&position| {
            seen.get_mut(position)
                .is_none_or(|seen| std::mem::replace(seen, true))
        })
    {
        return Err(refuse("host column order is not a complete permutation"));
    }
    Ok(json!({
        "rows": rows.len(),
        "columns": columns.into_iter().map(|(name, reference)| {
            json!({"name": name, "reference": reference, "positions": positions})
        }).collect::<Vec<_>>()
    }))
}

/// `result.to_pandas()`'s columns.
pub fn flatten(result: &Result_) -> Vec<Column> {
    if result.items.is_empty() {
        return empty_columns(result)
            .into_iter()
            .map(|name| Column {
                name,
                dtype: "object".to_string(),
                cells: Vec::new(),
            })
            .collect();
    }
    let rows: Vec<Vec<(String, PyVal)>> = result.items.iter().map(row_of).collect();
    let mut names: Vec<String> = Vec::new();
    for row in &rows {
        for (key, _) in row {
            if !names.contains(key) {
                names.push(key.clone());
            }
        }
    }
    let mut columns: Vec<Column> = names
        .iter()
        .map(|name| {
            let values: Vec<Option<PyVal>> = rows
                .iter()
                .map(|row| row.iter().find(|(k, _)| k == name).map(|(_, v)| v.clone()))
                .collect();
            infer_column(name, values)
        })
        .collect();
    if let Some(at) = columns.iter().position(|c| c.name == "row_index") {
        let keys: Vec<PyVal> = result
            .items
            .iter()
            .enumerate()
            .map(|(position, item)| {
                item.row_index_sort_key
                    .clone()
                    .unwrap_or_else(|| columns[at].cells[position].clone())
            })
            .collect();
        let order = stable_order(&keys);
        for column in &mut columns {
            column.cells = order.iter().map(|&i| column.cells[i].clone()).collect();
        }
    }
    columns
}

/// The positions sorted stably by a `row_index` column's values.
fn stable_order(cells: &[PyVal]) -> Vec<usize> {
    use std::cmp::Ordering;

    let key = |value: &PyVal| -> f64 {
        match crate::pyops::item(value) {
            PyVal::Float(value) => value,
            _ => f64::NAN,
        }
    };
    let integer_keys: Vec<_> = cells.iter().map(crate::pyops::integer_value).collect();
    let mut order: Vec<usize> = (0..cells.len()).collect();
    order.sort_by(|&left, &right| {
        match (&integer_keys[left], &integer_keys[right]) {
            (Some(left), Some(right)) => return left.cmp_finite(right),
            (Some(integer), None) => return compare_integer_float(integer, key(&cells[right])),
            (None, Some(integer)) => {
                return compare_integer_float(integer, key(&cells[left])).reverse();
            }
            _ => {}
        }
        let (left, right) = (key(&cells[left]), key(&cells[right]));
        match (left.is_nan(), right.is_nan()) {
            (true, true) => Ordering::Equal,
            (true, false) => Ordering::Greater,
            (false, true) => Ordering::Less,
            _ => left.partial_cmp(&right).unwrap_or(Ordering::Equal),
        }
    });
    order
}

fn compare_integer_float(integer: &crate::num::Dec, float: f64) -> std::cmp::Ordering {
    use std::cmp::Ordering;

    if float.is_nan() || float == f64::INFINITY {
        return Ordering::Less;
    }
    if float == f64::NEG_INFINITY {
        return Ordering::Greater;
    }
    let whole = crate::num::Dec::integer(&format!("{:.0}", float.trunc()))
        .expect("a finite truncated float has integer digits");
    integer.cmp_finite(&whole).then_with(|| {
        0.0f64
            .partial_cmp(&float.fract())
            .unwrap_or(Ordering::Equal)
    })
}

/// What one cell is, to `maybe_convert_objects`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Missing,
    NoneValue,
    Nan,
    Float,
    Int,
    Uint,
    Bool,
    Str,
    Stamp,
    StampTz,
    NaT,
    Object,
}

fn kind(value: &PyVal) -> Kind {
    match value {
        PyVal::None => Kind::NoneValue,
        PyVal::Float(x) if x.is_nan() => Kind::Nan,
        PyVal::Float(_) => Kind::Float,
        PyVal::Int(n) => {
            if *n >= i128::from(i64::MIN) && *n <= i128::from(i64::MAX) {
                Kind::Int
            } else if *n >= 0 && *n <= i128::from(u64::MAX) {
                Kind::Uint
            } else {
                Kind::Object
            }
        }
        PyVal::Bool(_) => Kind::Bool,
        PyVal::Str(_) => Kind::Str,
        PyVal::Timestamp(_) => Kind::Stamp,
        PyVal::TimestampTz(..) => Kind::StampTz,
        PyVal::NaT => Kind::NaT,
        PyVal::Numpy(name, inner) => match inner.as_ref() {
            PyVal::Bool(_) => Kind::Bool,
            PyVal::Float(x) if x.is_nan() => Kind::Nan,
            PyVal::Float(_) => Kind::Float,
            PyVal::Int(_) if name.starts_with("uint") => Kind::Uint,
            PyVal::Int(_) => Kind::Int,
            _ => Kind::Object,
        },
        _ => Kind::Object,
    }
}

/// `pd.DataFrame(rows)`'s column `name`: its dtype and cells.
fn infer_column(name: &str, values: Vec<Option<PyVal>>) -> Column {
    let cells: Vec<PyVal> = values
        .into_iter()
        .map(|v| v.unwrap_or_else(missing))
        .collect();
    let kinds: Vec<Kind> = cells.iter().map(kind).collect();
    let has = |k: Kind| kinds.contains(&k);
    let only = |allowed: &[Kind]| kinds.iter().all(|k| allowed.contains(k));
    let object = Column {
        name: name.to_string(),
        dtype: "object".to_string(),
        cells: cells.clone(),
    };
    if has(Kind::Object) {
        return object;
    }
    if has(Kind::Bool) {
        if only(&[Kind::Bool]) {
            return Column {
                name: name.to_string(),
                dtype: "bool".to_string(),
                cells: cells.iter().map(crate::pyops::item).collect(),
            };
        }
        return object;
    }
    if has(Kind::Str) {
        if only(&[Kind::Str, Kind::NoneValue, Kind::Nan, Kind::Missing]) {
            return Column {
                name: name.to_string(),
                dtype: "str".to_string(),
                cells: cells
                    .iter()
                    .map(|c| match c {
                        PyVal::Str(_) => c.clone(),
                        _ => PyVal::Float(f64::NAN),
                    })
                    .collect(),
            };
        }
        return object;
    }
    if has(Kind::Stamp) || has(Kind::StampTz) || has(Kind::NaT) {
        let nulls = [Kind::NoneValue, Kind::Nan, Kind::NaT];
        let aware = has(Kind::StampTz);
        if (aware && has(Kind::Stamp))
            || !kinds
                .iter()
                .all(|k| nulls.contains(k) || matches!(k, Kind::Stamp | Kind::StampTz))
        {
            return object;
        }
        let offsets: Vec<i64> = cells
            .iter()
            .filter_map(|c| match c {
                PyVal::TimestampTz(_, o) => Some(*o),
                _ => None,
            })
            .collect();
        if offsets.windows(2).any(|w| w[0] != w[1]) {
            return object;
        }
        let nanos = cells.iter().any(|c| match c {
            PyVal::Timestamp(w) | PyVal::TimestampTz(w, _) => w.nano % 1000 != 0,
            _ => false,
        });
        let any_stamp = has(Kind::Stamp) || aware;
        let unit = if !any_stamp {
            "s"
        } else if nanos {
            "ns"
        } else {
            "us"
        };
        let dtype = match offsets.first() {
            Some(0) => format!("datetime64[{unit}, UTC]"),
            Some(o) => format!("datetime64[{unit}, UTC{}]", offset_text(*o)),
            None => format!("datetime64[{unit}]"),
        };
        return Column {
            name: name.to_string(),
            dtype,
            cells: cells
                .iter()
                .map(|c| match c {
                    PyVal::Timestamp(_) | PyVal::TimestampTz(..) => c.clone(),
                    _ => PyVal::NaT,
                })
                .collect(),
        };
    }
    // Numbers and nulls.
    if only(&[Kind::NoneValue]) {
        return object;
    }
    if has(Kind::Uint) && has(Kind::Int) {
        let negative = cells
            .iter()
            .any(|c| matches!(crate::pyops::item(c), PyVal::Int(n) if n < 0));
        if negative && !(has(Kind::Float) || has(Kind::Nan)) {
            return object;
        }
    }
    let floating = has(Kind::Float) || has(Kind::Nan) || has(Kind::NoneValue) || has(Kind::Missing);
    if floating {
        return Column {
            name: name.to_string(),
            dtype: "float64".to_string(),
            cells: cells
                .iter()
                .map(|c| match crate::pyops::item(c) {
                    PyVal::Int(n) => PyVal::Float(n as f64),
                    PyVal::Float(x) => PyVal::Float(x),
                    _ => PyVal::Float(f64::NAN),
                })
                .collect(),
        };
    }
    let dtype = if has(Kind::Uint) { "uint64" } else { "int64" };
    Column {
        name: name.to_string(),
        dtype: dtype.to_string(),
        cells: cells.iter().map(crate::pyops::item).collect(),
    }
}

fn offset_text(seconds: i64) -> String {
    let sign = if seconds < 0 { '-' } else { '+' };
    let total = seconds.unsigned_abs();
    format!("{sign}{:02}:{:02}", total / 3600, total % 3600 / 60)
}

/// The Arrow array `pa.Table.from_pandas` makes of a column.
fn arrow_of(column: &Column) -> ArrayRef {
    let cells = &column.cells;
    match column.dtype.as_str() {
        "int64" => Arc::new(Int64Array::from(
            cells
                .iter()
                .map(|c| match c {
                    PyVal::Int(n) => i64::try_from(*n).ok(),
                    _ => None,
                })
                .collect::<Vec<_>>(),
        )),
        "uint64" => {
            let mut b = UInt64Builder::new();
            for c in cells {
                match c {
                    PyVal::Int(n) => b.append_option(u64::try_from(*n).ok()),
                    _ => b.append_null(),
                }
            }
            Arc::new(b.finish())
        }
        "float64" => Arc::new(Float64Array::from(
            cells
                .iter()
                .map(|c| match c {
                    PyVal::Float(x) if !x.is_nan() => Some(*x),
                    _ => None,
                })
                .collect::<Vec<_>>(),
        )),
        "bool" => Arc::new(BooleanArray::from(
            cells
                .iter()
                .map(|c| match c {
                    PyVal::Bool(b) => Some(*b),
                    _ => None,
                })
                .collect::<Vec<_>>(),
        )),
        "str" => {
            let mut b = LargeStringBuilder::new();
            for c in cells {
                match c {
                    PyVal::Str(s) => b.append_value(s),
                    _ => b.append_null(),
                }
            }
            Arc::new(b.finish())
        }
        "object" => infer_arrow(cells).unwrap_or_else(|| {
            let mut b = StringBuilder::new();
            for c in cells {
                if matches!(c, PyVal::None) {
                    b.append_null();
                } else {
                    b.append_value(crate::pyops::py_repr(c));
                }
            }
            Arc::new(b.finish())
        }),
        dtype if dtype.starts_with("datetime64[") => timestamp_array(cells, dtype),
        _ => Arc::new(NullArray::new(cells.len())),
    }
}

fn timestamp_array(cells: &[PyVal], dtype: &str) -> ArrayRef {
    let body = dtype
        .trim_start_matches("datetime64[")
        .trim_end_matches(']');
    let (unit, tz) = match body.split_once(", ") {
        Some((u, tz)) => (u, Some(tz.to_string())),
        None => (body, None),
    };
    let instants: Vec<Option<i128>> = cells
        .iter()
        .map(|c| match c {
            PyVal::Timestamp(w) => Some(w.epoch_nanos()),
            PyVal::TimestampTz(w, o) => Some(w.epoch_nanos() - i128::from(*o) * 1_000_000_000),
            _ => None,
        })
        .collect();
    let per: i128 = match unit {
        "s" => 1_000_000_000,
        "us" => 1_000,
        _ => 1,
    };
    let values: Vec<Option<i64>> = instants
        .iter()
        .map(|i| i.and_then(|n| i64::try_from(n.div_euclid(per)).ok()))
        .collect();
    let zone = tz.map(|t| {
        Arc::<str>::from(if t == "UTC" {
            "UTC".to_string()
        } else {
            t.trim_start_matches("UTC").to_string()
        })
    });
    match unit {
        "s" => Arc::new(TimestampSecondArray::from(values).with_timezone_opt(zone)),
        "us" => Arc::new(TimestampMicrosecondArray::from(values).with_timezone_opt(zone)),
        _ => Arc::new(TimestampNanosecondArray::from(values).with_timezone_opt(zone)),
    }
}

/// What pyarrow infers an `object` column's values as, where it can.
#[derive(Clone, Debug, PartialEq)]
enum Inferred {
    Null,
    Bool,
    Int,
    Float,
    Str,
    List(Box<Inferred>),
    Struct(Vec<(String, Inferred)>),
}

fn is_null_cell(value: &PyVal) -> bool {
    matches!(value, PyVal::None | PyVal::Na | PyVal::NaT)
        || matches!(value, PyVal::Float(x) if x.is_nan())
}

fn infer_type<'a>(values: impl Iterator<Item = &'a PyVal>) -> Option<Inferred> {
    let mut found = Inferred::Null;
    for value in values {
        if is_null_cell(value) {
            continue;
        }
        let this = match crate::pyops::item(value) {
            PyVal::Bool(_) => Inferred::Bool,
            PyVal::Int(n) if i64::try_from(n).is_ok() => Inferred::Int,
            PyVal::Float(_) => Inferred::Float,
            PyVal::Str(_) => Inferred::Str,
            PyVal::List(items) | PyVal::Tuple(items) => {
                Inferred::List(Box::new(infer_type(items.iter())?))
            }
            PyVal::Dict(pairs) => {
                let mut fields = Vec::new();
                for (k, v) in &pairs {
                    let PyVal::Str(key) = k else {
                        return None;
                    };
                    fields.push((key.clone(), infer_type(std::iter::once(v))?));
                }
                Inferred::Struct(fields)
            }
            _ => return None,
        };
        found = unify(found, this)?;
    }
    Some(found)
}

fn unify(a: Inferred, b: Inferred) -> Option<Inferred> {
    Some(match (a, b) {
        (Inferred::Null, x) | (x, Inferred::Null) => x,
        (Inferred::Int, Inferred::Float) | (Inferred::Float, Inferred::Int) => Inferred::Float,
        (Inferred::List(x), Inferred::List(y)) => Inferred::List(Box::new(unify(*x, *y)?)),
        (Inferred::Struct(mut x), Inferred::Struct(y)) => {
            for (name, t) in y {
                match x.iter_mut().find(|(n, _)| *n == name) {
                    Some(entry) => entry.1 = unify(entry.1.clone(), t)?,
                    None => x.push((name, t)),
                }
            }
            Inferred::Struct(x)
        }
        (x, y) if x == y => x,
        _ => return None,
    })
}

fn data_type(t: &Inferred) -> DataType {
    match t {
        Inferred::Null => DataType::Null,
        Inferred::Bool => DataType::Boolean,
        Inferred::Int => DataType::Int64,
        Inferred::Float => DataType::Float64,
        Inferred::Str => DataType::Utf8,
        Inferred::List(item) => DataType::List(Arc::new(Field::new("item", data_type(item), true))),
        Inferred::Struct(fields) => DataType::Struct(Fields::from(
            fields
                .iter()
                .map(|(n, t)| Field::new(n, data_type(t), true))
                .collect::<Vec<_>>(),
        )),
    }
}

fn build(t: &Inferred, values: &[Option<&PyVal>]) -> ArrayRef {
    match t {
        Inferred::Null => Arc::new(NullArray::new(values.len())),
        Inferred::Bool => {
            let mut b = BooleanBuilder::new();
            for v in values {
                match v.map(crate::pyops::item) {
                    Some(PyVal::Bool(x)) => b.append_value(x),
                    _ => b.append_null(),
                }
            }
            Arc::new(b.finish())
        }
        Inferred::Int => {
            let mut b = Int64Builder::new();
            for v in values {
                match v.map(crate::pyops::item) {
                    Some(PyVal::Int(n)) => b.append_option(i64::try_from(n).ok()),
                    _ => b.append_null(),
                }
            }
            Arc::new(b.finish())
        }
        Inferred::Float => {
            let mut b = Float64Builder::new();
            for v in values {
                match v.map(crate::pyops::item) {
                    Some(PyVal::Int(n)) => b.append_value(n as f64),
                    Some(PyVal::Float(x)) if !x.is_nan() => b.append_value(x),
                    _ => b.append_null(),
                }
            }
            Arc::new(b.finish())
        }
        Inferred::Str => {
            let mut b = StringBuilder::new();
            for v in values {
                match v {
                    Some(PyVal::Str(s)) => b.append_value(s),
                    _ => b.append_null(),
                }
            }
            Arc::new(b.finish())
        }
        Inferred::List(item) => {
            let mut offsets = vec![0i32];
            let mut flat: Vec<Option<&PyVal>> = Vec::new();
            let mut valid = Vec::new();
            for v in values {
                match v {
                    Some(PyVal::List(items) | PyVal::Tuple(items)) => {
                        flat.extend(
                            items
                                .iter()
                                .map(|i| if is_null_cell(i) { None } else { Some(i) }),
                        );
                        valid.push(true);
                    }
                    _ => valid.push(false),
                }
                offsets.push(flat.len() as i32);
            }
            let child = build(item, &flat);
            let field = Arc::new(Field::new("item", data_type(item), true));
            let nulls = arrow_buffer_nulls(&valid);
            Arc::new(arrow_array::ListArray::new(
                field,
                arrow_array::types::Int32Type::offsets(offsets),
                child,
                nulls,
            ))
        }
        Inferred::Struct(fields) => {
            let mut children = Vec::new();
            for (name, t) in fields {
                let column: Vec<Option<&PyVal>> = values
                    .iter()
                    .map(|v| match v {
                        Some(PyVal::Dict(pairs)) => pairs
                            .iter()
                            .find(|(k, _)| matches!(k, PyVal::Str(s) if s == name))
                            .map(|(_, v)| v)
                            .filter(|v| !is_null_cell(v)),
                        _ => None,
                    })
                    .collect();
                children.push((
                    Arc::new(Field::new(name, data_type(t), true)),
                    build(t, &column),
                ));
            }
            let valid: Vec<bool> = values
                .iter()
                .map(|v| matches!(v, Some(PyVal::Dict(_))))
                .collect();
            let (fields, arrays): (Vec<_>, Vec<_>) = children.into_iter().unzip();
            Arc::new(StructArray::new(
                Fields::from(fields),
                arrays,
                arrow_buffer_nulls(&valid),
            ))
        }
    }
}

trait Offsets {
    fn offsets(values: Vec<i32>) -> arrow_buffer::OffsetBuffer<i32>;
}

impl Offsets for arrow_array::types::Int32Type {
    fn offsets(values: Vec<i32>) -> arrow_buffer::OffsetBuffer<i32> {
        arrow_buffer::OffsetBuffer::new(arrow_buffer::ScalarBuffer::from(values))
    }
}

fn arrow_buffer_nulls(valid: &[bool]) -> Option<arrow_buffer::NullBuffer> {
    if valid.iter().all(|v| *v) {
        None
    } else {
        Some(arrow_buffer::NullBuffer::from(valid.to_vec()))
    }
}

/// An `object` column as pyarrow converts it, or `None` where it raises.
fn infer_arrow(cells: &[PyVal]) -> Option<ArrayRef> {
    let t = infer_type(cells.iter())?;
    let values: Vec<Option<&PyVal>> = cells
        .iter()
        .map(|c| if is_null_cell(c) { None } else { Some(c) })
        .collect();
    if cells.is_empty() {
        return Some(new_empty_array(&DataType::Null));
    }
    Some(build(&t, &values))
}

/// The layout of `result`: a record batch of the frame's columns, with
/// what a host needs to build the frame back in each field's metadata.
pub fn layout(result: &Result_) -> RecordBatch {
    let columns = flatten(result);
    let mut fields = Vec::new();
    let mut arrays = Vec::new();
    for column in &columns {
        let array = arrow_of(column);
        let mut metadata = HashMap::new();
        metadata.insert(DTYPE_KEY.to_string(), column.dtype.clone());
        if column.dtype == "object" {
            let cells: Vec<Value> = column.cells.iter().map(encode_deep).collect();
            metadata.insert(CELLS_KEY.to_string(), Value::Array(cells).to_string());
        }
        fields.push(
            Field::new(&column.name, array.data_type().clone(), true).with_metadata(metadata),
        );
        arrays.push(array);
    }
    let rows = columns.first().map_or(0, |c| c.cells.len());
    let schema = Arc::new(Schema::new(fields));
    RecordBatch::try_new_with_options(
        schema.clone(),
        arrays,
        &arrow_array::RecordBatchOptions::new().with_row_count(Some(rows)),
    )
    .unwrap_or_else(|_| RecordBatch::new_empty(schema))
}

// --- Results as plain data -----------------------------------------------------

/// A result read from its plain JSON form (the host's `result_json`).
pub fn decode_result(value: &Value) -> Result<Result_, String> {
    let map = value.as_object().ok_or("a result is an object")?;
    let text = |k: &str| map.get(k).and_then(Value::as_str).unwrap_or("").to_string();
    let items = map
        .get("items")
        .and_then(Value::as_array)
        .ok_or("a result has items")?
        .iter()
        .map(decode_item)
        .collect::<Result<Vec<_>, _>>()?;
    let metadata = match map.get("metadata") {
        None | Some(Value::Null) => PyVal::Dict(Vec::new()),
        Some(v) => match PyVal::decode(v)? {
            mapping @ PyVal::Dict(_) => mapping,
            ref reference @ PyVal::Opaque { ref kind, .. } if kind == "host_ref" => {
                reference.clone()
            }
            _ => return Err("metadata is a mapping".to_string()),
        },
    };
    let explanation = match map.get("explanation") {
        None | Some(Value::Null) => None,
        Some(v) => Some(PyVal::decode(v)?),
    };
    let outputs = map
        .get("outputs")
        .and_then(Value::as_array)
        .map(|o| {
            o.iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();
    let unscored = match map.get("unscored_candidates") {
        Some(Value::Array(items)) => items.iter().map(PyVal::decode).collect::<Result<_, _>>()?,
        _ => Vec::new(),
    };
    Ok(Result_ {
        host_values: match map.get("host_values") {
            None | Some(Value::Null) => None,
            Some(value) => {
                let fields = value.as_object().ok_or("host values must be an object")?;
                let held = |name: &str| {
                    PyVal::decode(fields.get(name).ok_or("missing host constructor field")?)
                };
                Some(crate::result::HostValues {
                    model: held("model")?,
                    request_id: held("request_id")?,
                    outputs: held("outputs")?,
                })
            }
        },
        model: text("model"),
        request_id: text("request_id"),
        items,
        metadata,
        explanation,
        outputs,
        unscored_candidates: unscored,
        names_its_rows: map
            .get("names_its_rows")
            .and_then(Value::as_bool)
            .unwrap_or(true),
        names_its_step: map
            .get("names_its_step")
            .and_then(Value::as_bool)
            .unwrap_or(false),
    })
}

fn pairs(value: &PyVal) -> Option<Vec<(PyVal, PyVal)>> {
    match value {
        PyVal::Dict(pairs) => Some(pairs.clone()),
        _ => None,
    }
}

fn list(value: &PyVal) -> Option<Vec<PyVal>> {
    match value {
        PyVal::List(items) | PyVal::Tuple(items) => Some(items.clone()),
        _ => None,
    }
}

fn decode_item(value: &Value) -> Result<Item, String> {
    let map = value.as_object().ok_or("an item is an object")?;
    let get = |k: &str| -> Result<Option<PyVal>, String> {
        match map.get(k) {
            None | Some(Value::Null) => Ok(None),
            Some(v) => PyVal::decode(v).map(Some),
        }
    };
    Ok(Item {
        row_index: get("row_index")?,
        row_index_sort_key: get("row_index_sort_key")?,
        id: get("id")?,
        forecast_step: get("forecast_step")?,
        prediction: get("prediction")?,
        probabilities: get("probabilities")?.as_ref().and_then(pairs),
        scores: get("scores")?.as_ref().and_then(list),
        rankings: get("rankings")?
            .as_ref()
            .and_then(list)
            .map(|entries| entries.iter().filter_map(pairs).collect()),
        embeddings: get("embeddings")?.as_ref().and_then(list),
        quantiles: get("quantiles")?.as_ref().and_then(pairs),
        explanation: get("explanation")?,
        metadata: get("metadata")?,
    })
}

/// Fixture JSON of a mapping, `$map` where a key is not text.
pub fn encode_mapping(pairs: &[(PyVal, PyVal)]) -> Value {
    if pairs.iter().all(|(k, _)| matches!(k, PyVal::Str(_))) {
        let mut out = Map::new();
        for (k, v) in pairs {
            out.insert(crate::pyops::py_str(k), encode_deep(v));
        }
        if out.keys().any(|k| k.starts_with('$')) && out.len() == 1 {
            return json!({"$dict": out});
        }
        return Value::Object(out);
    }
    json!({"$map": pairs.iter().map(|(k, v)| json!([encode_deep(k), encode_deep(v)])).collect::<Vec<_>>()})
}

/// Fixture JSON of a value, mappings with non-text keys as `$map`.
pub fn encode_deep(value: &PyVal) -> Value {
    match value {
        PyVal::Held {
            reference,
            owner,
            type_name,
            value,
        } => {
            let mut held =
                json!({"reference": {"$host_ref": reference}, "value": encode_deep(value)});
            if let Some(owner) = owner {
                held["owner"] = json!(owner);
            }
            if let Some(type_name) = type_name {
                held["type"] = json!(type_name);
            }
            json!({"$held": held})
        }
        PyVal::Dict(pairs) => encode_mapping(pairs),
        PyVal::List(items) => Value::Array(items.iter().map(encode_deep).collect()),
        PyVal::Tuple(items) => json!({"$tuple": items.iter().map(encode_deep).collect::<Vec<_>>()}),
        PyVal::Opaque {
            kind,
            type_name,
            repr,
        } => match kind.as_str() {
            "host_ref" => json!({"$host_ref": type_name}),
            sdm_schema::pyval::REFERS_BACK => match repr.parse::<u64>() {
                Ok(reference) => {
                    json!({"$refers_back": {"type": type_name, "reference": reference}})
                }
                Err(_) => json!({ "$refers_back": type_name }),
            },
            "container" => json!({ "$container": type_name }),
            "mapping" => json!({ "$mapping": type_name }),
            _ => json!({ "$object": { "type": type_name, "repr": repr } }),
        },
        other => encode(other),
    }
}

fn encode_item(item: &Item) -> Value {
    let mut out = Map::new();
    let mut put = |k: &str, v: Option<Value>| {
        if let Some(v) = v {
            out.insert(k.to_string(), v);
        }
    };
    put("row_index", item.row_index.as_ref().map(encode_deep));
    put(
        "row_index_sort_key",
        item.row_index_sort_key.as_ref().map(encode_deep),
    );
    put("id", item.id.as_ref().map(encode_deep));
    put(
        "forecast_step",
        item.forecast_step.as_ref().map(encode_deep),
    );
    put("prediction", item.prediction.as_ref().map(encode_deep));
    put(
        "probabilities",
        item.probabilities.as_ref().map(|p| encode_mapping(p)),
    );
    put(
        "scores",
        item.scores
            .as_ref()
            .map(|s| encode_deep(&PyVal::Tuple(s.clone()))),
    );
    put(
        "rankings",
        item.rankings.as_ref().map(|r| {
            encode_deep(&PyVal::Tuple(
                r.iter().map(|e| PyVal::Dict(e.clone())).collect(),
            ))
        }),
    );
    put(
        "embeddings",
        item.embeddings
            .as_ref()
            .map(|s| encode_deep(&PyVal::Tuple(s.clone()))),
    );
    put(
        "quantiles",
        item.quantiles.as_ref().map(|p| encode_mapping(p)),
    );
    put("explanation", item.explanation.as_ref().map(encode_deep));
    put("metadata", item.metadata.as_ref().map(encode_deep));
    Value::Object(out)
}

/// A result in its plain JSON form.
pub fn encode_result(result: &Result_) -> Value {
    let mut found = json!({
        "model": result.model,
        "request_id": result.request_id,
        "items": result.items.iter().map(encode_item).collect::<Vec<_>>(),
        "metadata": encode_deep(&result.metadata),
        "explanation": result.explanation.as_ref().map_or(Value::Null, encode_deep),
        "outputs": result.outputs,
        "unscored_candidates": result.unscored_candidates.iter().map(encode_deep).collect::<Vec<_>>(),
        "names_its_rows": result.names_its_rows,
        "names_its_step": result.names_its_step,
    });
    if let Some(values) = &result.host_values {
        found["host_values"] = json!({
            "model": encode_deep(&values.model),
            "request_id": encode_deep(&values.request_id),
            "outputs": encode_deep(&values.outputs),
        });
    }
    found
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_columns_use_native_field_order_and_apply_validated_positions() {
        use std::cell::RefCell;

        let result = Result_::new(
            "model",
            "request",
            vec![
                Item {
                    row_index: Some(PyVal::Int(2)),
                    prediction: Some(PyVal::Str("second".into())),
                    ..Item::default()
                },
                Item {
                    row_index: Some(PyVal::Int(1)),
                    forecast_step: Some(PyVal::Int(7)),
                    ..Item::default()
                },
            ],
            vec![],
        );
        let seen = RefCell::new(Vec::new());
        let found = host_layout_with(
            &result,
            |cells| {
                let mut seen = seen.borrow_mut();
                seen.push(cells.to_vec());
                Ok((seen.len().to_string(), cells.len()))
            },
            |reference, stable, ascending, missing_last| {
                assert_eq!(seen.borrow().len(), 3);
                assert_eq!(reference, "1");
                assert!(stable && ascending && missing_last);
                Ok(vec![1, 0])
            },
            str::to_string,
        )
        .unwrap();
        assert_eq!(found["rows"], 2);
        let columns = found["columns"].as_array().unwrap();
        assert_eq!(
            columns
                .iter()
                .map(|column| column["name"].as_str().unwrap())
                .collect::<Vec<_>>(),
            vec!["row_index", "prediction", "forecast_step"]
        );
        assert!(
            columns
                .iter()
                .all(|column| column["positions"] == json!([1, 0]))
        );
        assert!(matches!(seen.borrow()[1][1], PyVal::Float(value) if value.is_nan()));
    }

    #[test]
    fn host_columns_refuse_malformed_facts_and_skip_unneeded_order() {
        let result = Result_::new(
            "model",
            "request",
            vec![Item {
                row_index: Some(PyVal::Int(1)),
                ..Item::default()
            }],
            vec![],
        );
        for positions in [vec![], vec![1], vec![0, 0]] {
            let found = host_layout_with(
                &result,
                |_| Ok(("column".into(), 1)),
                |_, _, _, _| Ok(positions.clone()),
                str::to_string,
            );
            assert_eq!(
                found.unwrap_err(),
                "host column order is not a complete permutation"
            );
        }
        let found = host_layout_with(
            &result,
            |_| Ok(("column".into(), 2)),
            |_, _, _, _| panic!("length checked before ordering"),
            str::to_string,
        );
        assert_eq!(
            found.unwrap_err(),
            "host column length differs from result rows"
        );
        let result = Result_::new(
            "model",
            "request",
            vec![Item {
                prediction: Some(PyVal::Int(1)),
                ..Item::default()
            }],
            vec![],
        );
        let found = host_layout_with(
            &result,
            |_| Ok(("column".into(), 1)),
            |_, _, _, _| panic!("no index column to sort"),
            str::to_string,
        )
        .unwrap();
        assert_eq!(found["columns"][0]["positions"], json!([0]));
        assert!(!requires_host_columns(&result));
        assert_eq!(flatten(&result)[0].cells, vec![PyVal::Int(1)]);
    }

    #[test]
    fn row_order_keeps_exact_integers_and_places_missing_last() {
        let large = (1i128 << 63) + 1;
        let huge = PyVal::BigInt(
            "1606938044258990275541962092341162602522202993782792835301376".to_string(),
        );
        let cells = vec![
            huge.clone(),
            PyVal::Int(large),
            PyVal::None,
            PyVal::Int(large - 1),
            PyVal::Int(large),
            huge,
        ];
        assert_eq!(stable_order(&cells), vec![3, 1, 4, 0, 5, 2]);
    }

    #[test]
    fn mixed_numeric_order_uses_exact_integer_float_comparison() {
        let cells = vec![
            PyVal::Int((1i128 << 53) + 1),
            PyVal::Float((1u64 << 53) as f64),
            PyVal::Float(-1.5),
            PyVal::Int(-1),
            PyVal::Float(f64::NAN),
            PyVal::Float(f64::NEG_INFINITY),
            PyVal::Float(f64::INFINITY),
        ];
        assert_eq!(stable_order(&cells), vec![5, 2, 3, 1, 0, 6, 4]);
    }
}
