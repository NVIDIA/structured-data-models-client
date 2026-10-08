// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The seam between the request code's frames and the sampler's.
//!
//! The sampler reads a table as `pa.Table.from_pandas(frame)` writes it,
//! each column's pandas dtype in the `pandas` metadata, and gives back
//! frames of the same kind; the request code reads a column as the Python
//! values `series.tolist()` gives ([`Series`]). These are the two ways
//! across.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::builder::{
    BooleanBuilder, Float64Builder, Int64Builder, LargeStringBuilder, ListBuilder,
};
use arrow_array::{Array, ArrayRef, Decimal128Array, RecordBatch, RecordBatchOptions};
use arrow_schema::{DataType, Field, Schema};
use sdm_sampling::frame::{Dtype as SDtype, Frame, Series as SSeries};
use sdm_schema::pyval::PyVal;
use serde_json::{Value, json};

use crate::frames::{arrow_of, series_of};
use crate::num::Dec;
use crate::pyops::item;
use crate::series::{Dtype, Series, infer_dtype, is_na};

/// `str(dtype)` of a sampler column.
pub fn dtype_name(series: &SSeries) -> String {
    match &series.dtype {
        SDtype::Numpy(name) | SDtype::Extension(name) => name.clone(),
        SDtype::Object => "object".to_string(),
        SDtype::Category => "category".to_string(),
        SDtype::Arrow => format!(
            "{}[pyarrow]",
            sdm_schema::engine::infer::arrow_type_name(series.array.data_type())
        ),
    }
}

/// A sampler column as the values `tolist()` gives.
pub fn series_from(name: &str, series: &SSeries) -> Series {
    let dtype = dtype_name(series);
    let mut out = series_of(name, &series.array, &json!({ "dtype": dtype }));
    if let Some(cells) = series.object_values() {
        out.cells = cells;
        return out;
    }
    if matches!(
        series.array.data_type(),
        DataType::List(_) | DataType::LargeList(_) | DataType::FixedSizeList(..)
    ) {
        out.cells = list_cells(&series.array);
    }
    out
}

/// The cells of a list array: each a list of its items, `None` where
/// missing.
fn list_cells(array: &ArrayRef) -> Vec<PyVal> {
    let n = array.len();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        if array.is_null(i) {
            out.push(PyVal::None);
            continue;
        }
        let items: ArrayRef = match array.data_type() {
            DataType::List(_) => {
                arrow_array::cast::AsArray::as_list::<i32>(array.as_ref()).value(i)
            }
            DataType::LargeList(_) => {
                arrow_array::cast::AsArray::as_list::<i64>(array.as_ref()).value(i)
            }
            _ => arrow_array::cast::AsArray::as_fixed_size_list(array.as_ref()).value(i),
        };
        let cells = sdm_sampling::frame::object_values(&items)
            .unwrap_or_else(|| series_of("", &items, &json!({ "dtype": "object" })).cells);
        out.push(PyVal::List(cells));
    }
    out
}

/// A sampler frame as request columns, in its order.
pub fn columns_from(frame: &Frame) -> Vec<Series> {
    frame
        .names
        .iter()
        .zip(&frame.columns)
        .map(|(name, series)| series_from(name, series))
        .collect()
}

/// The Arrow array and the `numpy_type` `pa.Table.from_pandas` gives a
/// column, or why it cannot hold one.
pub fn pandas_array(series: &Series) -> Result<(ArrayRef, String), String> {
    match &series.dtype {
        Dtype::Object => object_array(&series.cells).map(|a| (a, "object".to_string())),
        Dtype::Category => {
            let (array, _) = arrow_of(series);
            Ok((array, "category".to_string()))
        }
        other => {
            let (array, _) = arrow_of(series);
            Ok((array, other.name()))
        }
    }
}

/// `pa.array(values)` of an `object` column: the one type its values share.
fn object_array(cells: &[PyVal]) -> Result<ArrayRef, String> {
    let present: Vec<PyVal> = cells.iter().filter(|v| !is_na(v)).map(item).collect();
    let content = infer_dtype(&present);
    let missing = |v: &PyVal| is_na(v);
    Ok(match content {
        "string" | "empty" => {
            let mut b = LargeStringBuilder::new();
            for v in cells {
                match item(v) {
                    PyVal::Str(s) if !missing(v) => b.append_value(s),
                    _ => b.append_null(),
                }
            }
            Arc::new(b.finish())
        }
        "integer" => {
            let mut b = Int64Builder::new();
            for v in cells {
                match item(v) {
                    PyVal::Int(n) if !missing(v) => match i64::try_from(n) {
                        Ok(n) => b.append_value(n),
                        Err(_) => return Err("an integer wider than int64".to_string()),
                    },
                    _ => b.append_null(),
                }
            }
            Arc::new(b.finish())
        }
        "floating" | "mixed-integer-float" => {
            let mut b = Float64Builder::new();
            for v in cells {
                match item(v) {
                    PyVal::Float(x) if !missing(v) => b.append_value(x),
                    PyVal::Int(n) if !missing(v) => b.append_value(n as f64),
                    _ => b.append_null(),
                }
            }
            Arc::new(b.finish())
        }
        "boolean" => {
            let mut b = BooleanBuilder::new();
            for v in cells {
                match item(v) {
                    PyVal::Bool(x) if !missing(v) => b.append_value(x),
                    _ => b.append_null(),
                }
            }
            Arc::new(b.finish())
        }
        "decimal" => decimal_array(cells)?,
        "date" | "datetime" | "datetime64" => time_array(cells)?,
        "timedelta" | "timedelta64" => {
            let values: Vec<Option<i64>> = cells
                .iter()
                .map(|v| match v {
                    PyVal::Timedelta(ns) if !missing(v) => Some(ns.div_euclid(1000)),
                    _ => None,
                })
                .collect();
            Arc::new(arrow_array::DurationMicrosecondArray::from(values))
        }
        "bytes" => {
            let values: Vec<Option<Vec<u8>>> = cells
                .iter()
                .map(|v| match v {
                    PyVal::Bytes(b) => Some(b.clone()),
                    _ => None,
                })
                .collect();
            Arc::new(arrow_array::BinaryArray::from_iter(values))
        }
        "mixed"
            if present
                .iter()
                .all(|v| matches!(v, PyVal::List(_) | PyVal::Tuple(_) | PyVal::NdArray(..))) =>
        {
            list_array(cells)?
        }
        "mixed"
            if present.iter().any(|value| {
                matches!(value, PyVal::List(_) | PyVal::Tuple(_) | PyVal::NdArray(..))
            }) =>
        {
            sdm_sampling::frame::encoded_objects(
                cells
                    .iter()
                    .map(|value| crate::layout::encode_deep(value).to_string())
                    .collect(),
            )
        }
        _ => {
            // Values Arrow holds as no one type: what the sampler does with
            // such a column (take its rows) needs no type, and what is then
            // read of its cells, a wire string, is each value's `str`.
            let mut b = LargeStringBuilder::new();
            for v in cells {
                if missing(v) {
                    b.append_null();
                } else {
                    b.append_value(crate::pyops::py_str(v));
                }
            }
            Arc::new(b.finish())
        }
    })
}

/// `pa.array` of dates and times: microseconds, at the one offset they
/// share, as UTC where they have one.
fn time_array(cells: &[PyVal]) -> Result<ArrayRef, String> {
    let mut zoned = false;
    let values: Vec<Option<i64>> = cells
        .iter()
        .map(|v| match v {
            PyVal::Date(y, m, d) => {
                Some((sdm_schema::dt::civil::Wall::date(*y, *m, *d).epoch_nanos() / 1000) as i64)
            }
            PyVal::DateTime(wall, None) | PyVal::Timestamp(wall) => {
                Some((wall.epoch_nanos() / 1000) as i64)
            }
            PyVal::DateTime(wall, Some(offset)) => {
                zoned = true;
                Some(((wall.epoch_nanos() - i128::from(*offset) * 1_000_000_000) / 1000) as i64)
            }
            PyVal::TimestampTz(wall, offset) => {
                zoned = true;
                Some(((wall.epoch_nanos() - i128::from(*offset) * 1_000_000_000) / 1000) as i64)
            }
            _ => None,
        })
        .collect();
    let array = arrow_array::TimestampMicrosecondArray::from(values);
    Ok(if zoned {
        Arc::new(array.with_timezone("UTC"))
    } else {
        Arc::new(array)
    })
}

/// `pa.array` of decimals: `decimal128` wide enough for every value.
fn decimal_array(cells: &[PyVal]) -> Result<ArrayRef, String> {
    let decs: Vec<Option<Dec>> = cells
        .iter()
        .map(|v| match item(v) {
            PyVal::Decimal(text) if !is_na(v) => Dec::parse(&text).filter(Dec::is_finite),
            _ => None,
        })
        .collect();
    let scale = decs
        .iter()
        .flatten()
        .map(|d| (-d.exponent).max(0))
        .max()
        .unwrap_or(0);
    let integral = decs
        .iter()
        .flatten()
        .map(|d| (d.digits.len() as i64 + d.exponent).max(1))
        .max()
        .unwrap_or(1);
    let precision = (integral + scale).clamp(1, 38);
    let values: Vec<Option<i128>> = decs
        .iter()
        .map(|d| {
            d.as_ref().and_then(|d| {
                let shift = d.exponent + scale;
                let digits = format!("{}{}", d.digits, "0".repeat(shift.max(0) as usize));
                digits
                    .parse::<i128>()
                    .ok()
                    .map(|m| if d.negative { -m } else { m })
            })
        })
        .collect();
    Decimal128Array::from(values)
        .with_precision_and_scale(precision as u8, scale as i8)
        .map(|a| Arc::new(a) as ArrayRef)
        .map_err(|e| e.to_string())
}

/// `pa.array` of lists of text, numbers or flags.
fn list_array(cells: &[PyVal]) -> Result<ArrayRef, String> {
    let items: Vec<PyVal> = cells
        .iter()
        .filter_map(|v| match v {
            PyVal::List(items) | PyVal::Tuple(items) | PyVal::NdArray(_, items) => {
                Some(items.clone())
            }
            _ => None,
        })
        .flatten()
        .collect();
    let lists = || {
        cells.iter().map(|v| match v {
            PyVal::List(items) | PyVal::Tuple(items) | PyVal::NdArray(_, items) => {
                Some(items.clone())
            }
            _ => None,
        })
    };
    match infer_dtype(&items) {
        "string" | "empty" => {
            let mut b = ListBuilder::new(LargeStringBuilder::new());
            for list in lists() {
                match list {
                    None => b.append(false),
                    Some(items) => {
                        for v in items {
                            match v {
                                PyVal::Str(s) => b.values().append_value(s),
                                _ => b.values().append_null(),
                            }
                        }
                        b.append(true);
                    }
                }
            }
            Ok(Arc::new(b.finish()))
        }
        "integer" => {
            let mut b = ListBuilder::new(Int64Builder::new());
            for list in lists() {
                match list {
                    None => b.append(false),
                    Some(items) => {
                        for v in items {
                            match item(&v) {
                                PyVal::Int(n) => b.values().append_option(i64::try_from(n).ok()),
                                _ => b.values().append_null(),
                            }
                        }
                        b.append(true);
                    }
                }
            }
            Ok(Arc::new(b.finish()))
        }
        "floating" | "mixed-integer-float" => {
            let mut b = ListBuilder::new(Float64Builder::new());
            for list in lists() {
                match list {
                    None => b.append(false),
                    Some(items) => {
                        for v in items {
                            match item(&v) {
                                PyVal::Float(x) => b.values().append_value(x),
                                PyVal::Int(n) => b.values().append_value(n as f64),
                                _ => b.values().append_null(),
                            }
                        }
                        b.append(true);
                    }
                }
            }
            Ok(Arc::new(b.finish()))
        }
        other => Err(format!(
            "an object column of lists of {other} values, which Arrow holds as no one type"
        )),
    }
}

/// `pa.Table.from_pandas(frame, preserve_index=False)` of request columns:
/// the batch, with the `pandas` metadata naming each column's dtype.
pub fn pandas_batch(columns: &[Series], rows: usize) -> Result<RecordBatch, String> {
    let mut fields = Vec::new();
    let mut arrays = Vec::new();
    let mut described = Vec::new();
    for series in columns {
        let name = series.name.clone().unwrap_or_default();
        let (array, numpy_type) = pandas_array(series).map_err(|why| format!("{name}: {why}"))?;
        fields.push(Field::new(&name, array.data_type().clone(), true));
        arrays.push(array);
        described.push(json!({
            "name": name,
            "field_name": name,
            "pandas_type": Value::Null,
            "numpy_type": numpy_type,
            "metadata": Value::Null,
        }));
    }
    let metadata: HashMap<String, String> = HashMap::from([(
        "pandas".to_string(),
        json!({"index_columns": [], "column_indexes": [], "columns": described}).to_string(),
    )]);
    let schema = Arc::new(Schema::new_with_metadata(fields, metadata));
    RecordBatch::try_new_with_options(
        schema,
        arrays,
        &RecordBatchOptions::new().with_row_count(Some(rows)),
    )
    .map_err(|e| e.to_string())
}

/// One sampler column of one value per row, as the inputs hold an argument
/// that is a `Series` (`capture_sampling.py`, `_series_table`).
pub fn series_batch(series: &SSeries) -> Result<RecordBatch, String> {
    let dtype = dtype_name(series);
    let metadata: HashMap<String, String> = HashMap::from([(
        "pandas".to_string(),
        json!({"index_columns": [], "column_indexes": [], "columns": [{
            "name": "value", "field_name": "value", "pandas_type": Value::Null,
            "numpy_type": dtype, "metadata": Value::Null,
        }]})
        .to_string(),
    )]);
    let field = Field::new("value", series.array.data_type().clone(), true);
    let schema = Arc::new(Schema::new_with_metadata(vec![field], metadata));
    RecordBatch::try_new(schema, vec![series.array.clone()]).map_err(|e| e.to_string())
}

/// A sampler frame's batch, each column's dtype in its metadata.
pub fn frame_batch(frame: &Frame) -> Result<RecordBatch, String> {
    let described: Vec<Value> = frame
        .names
        .iter()
        .zip(&frame.columns)
        .map(|(name, series)| {
            json!({"name": name, "field_name": name, "pandas_type": Value::Null,
                   "numpy_type": dtype_name(series), "metadata": Value::Null})
        })
        .collect();
    let metadata: HashMap<String, String> = HashMap::from([(
        "pandas".to_string(),
        json!({"index_columns": [], "column_indexes": [], "columns": described}).to_string(),
    )]);
    let fields: Vec<Field> = frame
        .names
        .iter()
        .zip(&frame.columns)
        .map(|(name, s)| Field::new(name, s.array.data_type().clone(), true))
        .collect();
    let schema = Arc::new(Schema::new_with_metadata(fields, metadata));
    RecordBatch::try_new_with_options(
        schema,
        frame.columns.iter().map(|s| s.array.clone()).collect(),
        &RecordBatchOptions::new().with_row_count(Some(frame.rows)),
    )
    .map_err(|e| e.to_string())
}

/// `pd.Series(values)`: the dtype pandas infers for a list of Python
/// values, and the values it then holds.
pub fn series_of_values(name: &str, values: Vec<PyVal>) -> Series {
    let present: Vec<PyVal> = values
        .iter()
        .filter(|v| !matches!(v, PyVal::None) && !is_na(v))
        .map(item)
        .collect();
    let has_missing = present.len() < values.len();
    let all = |f: fn(&PyVal) -> bool| !present.is_empty() && present.iter().all(f);
    let dtype = if present.is_empty() {
        if values.is_empty() || values.iter().all(|v| matches!(v, PyVal::None)) {
            Dtype::Object
        } else {
            Dtype::Float(64)
        }
    } else if all(|v| matches!(v, PyVal::Bool(_))) {
        if has_missing {
            Dtype::Object
        } else {
            Dtype::Bool
        }
    } else if all(|v| matches!(v, PyVal::Int(_))) {
        if has_missing {
            Dtype::Float(64)
        } else {
            let fits = present
                .iter()
                .all(|v| matches!(v, PyVal::Int(n) if i64::try_from(*n).is_ok()));
            if fits {
                Dtype::Int("int64".to_string())
            } else {
                Dtype::Object
            }
        }
    } else if all(|v| matches!(v, PyVal::Int(_) | PyVal::Float(_))) {
        Dtype::Float(64)
    } else if all(|v| matches!(v, PyVal::Str(_))) {
        Dtype::Str
    } else if all(|v| matches!(v, PyVal::Timestamp(_))) {
        Dtype::Datetime("us".to_string(), None)
    } else {
        Dtype::Object
    };
    let cells = values
        .into_iter()
        .map(|v| {
            let missing = matches!(v, PyVal::None) || is_na(&v);
            match &dtype {
                Dtype::Float(_) if missing => PyVal::Float(f64::NAN),
                Dtype::Float(_) => match item(&v) {
                    PyVal::Int(n) => PyVal::Float(n as f64),
                    other => other,
                },
                Dtype::Str if missing => PyVal::Float(f64::NAN),
                Dtype::Datetime(..) if missing => PyVal::NaT,
                Dtype::Object => v,
                _ => item(&v),
            }
        })
        .collect();
    Series::new(Some(name.to_string()), dtype, cells)
}

/// `sys.getsizeof(value)` of a Python scalar, as CPython 3.12 reports it.
fn getsizeof(value: &PyVal) -> usize {
    match value {
        PyVal::None => 16,
        PyVal::Bool(_) => 28,
        PyVal::Int(n) => {
            let bits = 128 - n.unsigned_abs().leading_zeros() as usize;
            if *n == 0 {
                28
            } else {
                24 + 4 * bits.div_ceil(30)
            }
        }
        PyVal::Float(_) => 24,
        PyVal::Str(s) => {
            let max = s.chars().map(|c| c as u32).max().unwrap_or(0);
            let n = s.chars().count();
            if max < 128 {
                41 + n
            } else if max < 256 {
                57 + n
            } else if max < 65_536 {
                58 + 2 * n
            } else {
                60 + 4 * n
            }
        }
        PyVal::Decimal(_) => 104,
        PyVal::Timestamp(_) | PyVal::TimestampTz(..) => 64,
        PyVal::List(items) | PyVal::Tuple(items) => 56 + 8 * items.len(),
        _ => 64,
    }
}

/// `series.memory_usage(deep=True, index=False)` of one column.
pub fn memory_usage(series: &Series) -> usize {
    let n = series.cells.len();
    match &series.dtype {
        Dtype::Object => 8 * n + series.cells.iter().map(getsizeof).sum::<usize>(),
        Dtype::Bool => n,
        Dtype::Int(name) => {
            let bits: usize = name
                .trim_start_matches("uint")
                .trim_start_matches("int")
                .parse()
                .unwrap_or(64);
            n * bits / 8
        }
        Dtype::Float(bits) => n * (*bits as usize) / 8,
        Dtype::Str | Dtype::String(_) => {
            let text: usize = series
                .cells
                .iter()
                .map(|v| match v {
                    PyVal::Str(s) => s.len(),
                    _ => 0,
                })
                .sum();
            8 * (n + 1) + text + n.div_ceil(8)
        }
        _ => 8 * n,
    }
}
