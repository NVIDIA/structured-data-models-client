// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Frames as the host hands them over (DV19): each a pandas `DataFrame`
//! described by `sdm_core.describe.describe_frames` as an Arrow table and a
//! tag per column naming its pandas dtype; an `object` column arrives as
//! text, one JSON-encoded cell per row, with the content
//! `pandas.api.types.infer_dtype` reports.
//!
//! The request code keeps frames in that form, takes rows and columns of
//! them as pandas would, and reads a column as a [`Series`] of the Python
//! values `series.tolist()` gives where it needs to look at them. A
//! fixture's `{"dtype", "values"}` columns are built into the same form,
//! as `describe_frames(build_frame(spec))` would describe them.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{
    Decimal32Type, Decimal64Type, Decimal128Type, Decimal256Type, DurationMicrosecondType,
    DurationMillisecondType, DurationNanosecondType, DurationSecondType, Float16Type, Float32Type,
    Float64Type, Int8Type, Int16Type, Int32Type, Int64Type, TimestampMicrosecondType,
    TimestampMillisecondType, TimestampNanosecondType, TimestampSecondType, UInt8Type, UInt16Type,
    UInt32Type, UInt64Type,
};
use arrow_array::{
    Array, ArrayRef, BinaryArray, BooleanArray, Date32Array, Date64Array, Decimal128Array,
    DictionaryArray, DurationMicrosecondArray, DurationMillisecondArray, DurationNanosecondArray,
    DurationSecondArray, Float32Array, Float64Array, Int8Array, Int16Array, Int32Array, Int64Array,
    LargeBinaryArray, LargeStringArray, RecordBatch, RecordBatchOptions, StringArray,
    Time32MillisecondArray, Time32SecondArray, Time64MicrosecondArray, Time64NanosecondArray,
    TimestampMicrosecondArray, TimestampMillisecondArray, TimestampNanosecondArray,
    TimestampSecondArray, UInt8Array, UInt16Array, UInt32Array, UInt64Array,
};
use arrow_schema::{DataType, Field, Schema, TimeUnit};
use sdm_schema::dt::civil::Wall;
use sdm_schema::pyval::PyVal;
use serde_json::{Map, Value, json};

use crate::num::Dec;
use crate::series::{Dtype, Series, fixed_offset, infer_dtype};

/// A frame: its columns as Arrow arrays, and each column's tag.
#[derive(Clone, Debug)]
pub struct Table {
    pub batch: RecordBatch,
    pub tags: Map<String, Value>,
    /// The rows of the frame the host described that this table holds, in
    /// order, where it was taken from one (`frame.iloc[rows]`); `None` for
    /// the frame itself, or one built here.
    pub rows: Option<Vec<usize>>,
}

impl Table {
    /// The table the host described as `batch` with `tags`.
    pub fn new(batch: RecordBatch, tags: Option<&Value>) -> Self {
        let tags = tags.and_then(Value::as_object).cloned().unwrap_or_default();
        Self {
            batch,
            tags,
            rows: None,
        }
    }

    /// `len(frame)`.
    pub fn len(&self) -> usize {
        self.batch.num_rows()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The column names, in order.
    pub fn columns(&self) -> Vec<String> {
        self.batch
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect()
    }

    /// Whether the frame has a column `name`.
    pub fn has(&self, name: &str) -> bool {
        self.batch.schema().index_of(name).is_ok()
    }

    /// A column's tag.
    pub fn tag(&self, name: &str) -> Value {
        self.tags.get(name).cloned().unwrap_or_else(|| {
            let data_type = self
                .batch
                .schema()
                .field_with_name(name)
                .map(|f| f.data_type().clone())
                .unwrap_or(DataType::Null);
            json!({ "dtype": default_dtype(&data_type) })
        })
    }

    /// The `repr` of a column's label that is not text, as the host's
    /// describer tags it (`name_repr`); `None` for a label that is text.
    pub fn label_not_text(&self, name: &str) -> Option<String> {
        self.tags
            .get(name)
            .and_then(|tag| tag.get("name_repr"))
            .and_then(Value::as_str)
            .map(str::to_string)
    }

    /// `repr(labels)` of a list of this frame's column labels, each text one
    /// quoted and every other as its `repr`.
    pub fn labels_repr(&self, names: &[String]) -> String {
        let parts: Vec<String> = names
            .iter()
            .map(|name| {
                self.label_not_text(name)
                    .unwrap_or_else(|| sdm_pql::pyrepr::str_repr(name))
            })
            .collect();
        format!("[{}]", parts.join(", "))
    }

    /// `label in frame.columns` for this frame's column `name` of `other`:
    /// the same text, and a label of the same kind.
    pub fn has_label_of(&self, other: &Table, name: &str) -> bool {
        self.has(name) && self.label_not_text(name) == other.label_not_text(name)
    }

    /// A column's pandas dtype.
    pub fn dtype(&self, name: &str) -> Dtype {
        Dtype::parse(
            self.tag(name)
                .get("dtype")
                .and_then(Value::as_str)
                .unwrap_or("object"),
        )
    }

    /// `frame[name]`, as the values `tolist()` gives.
    pub fn series(&self, name: &str) -> Option<Series> {
        let at = self.batch.schema().index_of(name).ok()?;
        let tag = self.tag(name);
        Some(series_of(name, self.batch.column(at), &tag))
    }

    /// `frame.iloc[rows]`.
    pub fn take(&self, rows: &[usize]) -> Table {
        let indices =
            arrow_array::UInt64Array::from(rows.iter().map(|&r| r as u64).collect::<Vec<_>>());
        let columns: Vec<ArrayRef> = self
            .batch
            .columns()
            .iter()
            .map(|c| {
                arrow_select::take::take(c.as_ref(), &indices, None)
                    .unwrap_or_else(|_| arrow_array::new_null_array(c.data_type(), rows.len()))
            })
            .collect();
        let batch = RecordBatch::try_new_with_options(
            self.batch.schema(),
            columns,
            &RecordBatchOptions::new().with_row_count(Some(rows.len())),
        )
        .unwrap_or_else(|_| RecordBatch::new_empty(self.batch.schema()));
        let taken = match &self.rows {
            Some(held) => rows
                .iter()
                .map(|&r| held.get(r).copied().unwrap_or(r))
                .collect(),
            None => rows.to_vec(),
        };
        Table {
            batch,
            tags: self.tags.clone(),
            rows: Some(taken),
        }
    }

    /// `frame.drop(columns=names)`, leaving any name it does not have.
    pub fn drop(&self, names: &[String]) -> Table {
        let keep: Vec<String> = self
            .columns()
            .into_iter()
            .filter(|c| !names.contains(c))
            .collect();
        self.select(&keep)
    }

    /// Whether the frame's columns are a pandas `MultiIndex`, which the
    /// host's describer says in each column's tag (`"multi_index": true`).
    pub fn multi_index(&self) -> bool {
        self.tags
            .values()
            .any(|tag| tag.get("multi_index").and_then(Value::as_bool) == Some(true))
    }

    /// `frame.columns = labels`, as the host's describer then describes the
    /// frame: each column keeps its values and dtype under its new name, a
    /// label that is not text tagged with its `repr` (`name_repr`), and
    /// every column of a `MultiIndex` tagged `"multi_index": true`. Two
    /// columns of one name share the first one's tag.
    pub fn relabelled(&self, labels: &[(String, Option<String>)], multi_index: bool) -> Table {
        let schema = self.batch.schema();
        let old = self.columns();
        let mut fields = Vec::new();
        let mut tags = Map::new();
        for (at, (name, repr)) in labels.iter().enumerate() {
            let Some(field) = schema.fields().get(at) else {
                break;
            };
            fields.push(field.as_ref().clone().with_name(name));
            let mut tag = match self.tag(&old[at]) {
                Value::Object(map) => map,
                _ => Map::new(),
            };
            tag.remove("name_repr");
            tag.remove("multi_index");
            if let Some(repr) = repr {
                tag.insert("name_repr".into(), json!(repr));
            }
            if multi_index {
                tag.insert("multi_index".into(), json!(true));
            }
            tags.entry(name.clone()).or_insert(Value::Object(tag));
        }
        let columns = self.batch.columns()[..fields.len()].to_vec();
        let schema = Arc::new(Schema::new(fields));
        let batch = RecordBatch::try_new_with_options(
            schema.clone(),
            columns,
            &RecordBatchOptions::new().with_row_count(Some(self.len())),
        )
        .unwrap_or_else(|_| RecordBatch::new_empty(schema));
        Table {
            batch,
            tags,
            rows: self.rows.clone(),
        }
    }

    /// `frame[names]`.
    pub fn select(&self, names: &[String]) -> Table {
        let schema = self.batch.schema();
        let mut fields = Vec::new();
        let mut columns = Vec::new();
        let mut tags = Map::new();
        for name in names {
            if let Ok(at) = schema.index_of(name) {
                fields.push(schema.field(at).clone());
                columns.push(self.batch.column(at).clone());
                if let Some(tag) = self.tags.get(name) {
                    tags.insert(name.clone(), tag.clone());
                }
            }
        }
        let schema = Arc::new(Schema::new(fields));
        let batch = RecordBatch::try_new_with_options(
            schema.clone(),
            columns,
            &RecordBatchOptions::new().with_row_count(Some(self.len())),
        )
        .unwrap_or_else(|_| RecordBatch::new_empty(schema));
        Table {
            batch,
            tags,
            rows: self.rows.clone(),
        }
    }

    /// A frame of one more column.
    pub fn with_series(&self, series: &Series) -> Table {
        let (array, tag) = arrow_of(series);
        let name = series.name.clone().unwrap_or_default();
        let schema = self.batch.schema();
        let mut fields: Vec<Field> = schema.fields().iter().map(|f| f.as_ref().clone()).collect();
        let mut columns = self.batch.columns().to_vec();
        match schema.index_of(&name) {
            Ok(at) => {
                fields[at] = Field::new(&name, array.data_type().clone(), true);
                columns[at] = array;
            }
            Err(_) => {
                fields.push(Field::new(&name, array.data_type().clone(), true));
                columns.push(array);
            }
        }
        let mut tags = self.tags.clone();
        tags.insert(name, tag);
        let schema = Arc::new(Schema::new(fields));
        let batch = RecordBatch::try_new_with_options(
            schema.clone(),
            columns,
            &RecordBatchOptions::new().with_row_count(Some(self.len())),
        )
        .unwrap_or_else(|_| RecordBatch::new_empty(schema));
        Table {
            batch,
            tags,
            rows: self.rows.clone(),
        }
    }

    /// The frame as the schema core reads one.
    pub fn schema_frame(&self) -> Result<sdm_schema::frame::Frame, String> {
        let tags = Value::Object(self.tags.clone());
        sdm_schema::frame::from_arrow(
            std::slice::from_ref(&self.batch),
            &self.batch.schema(),
            Some(&tags),
        )
    }

    /// The table's tags as the JSON a host writes them in.
    pub fn tags_json(&self) -> Value {
        Value::Object(self.tags.clone())
    }

    /// A frame of `columns`, each a [`Series`], with `rows` rows.
    pub fn from_series(columns: &[Series], rows: usize) -> Table {
        let mut fields = Vec::new();
        let mut arrays = Vec::new();
        let mut tags = Map::new();
        for series in columns {
            let (array, tag) = arrow_of(series);
            let name = series.name.clone().unwrap_or_default();
            fields.push(Field::new(&name, array.data_type().clone(), true));
            arrays.push(array);
            tags.insert(name, tag);
        }
        let schema = Arc::new(Schema::new(fields));
        let batch = RecordBatch::try_new_with_options(
            schema.clone(),
            arrays,
            &RecordBatchOptions::new().with_row_count(Some(rows)),
        )
        .unwrap_or_else(|_| RecordBatch::new_empty(schema));
        Table {
            batch,
            tags,
            rows: None,
        }
    }

    /// `build_frame(spec)` of `_fixture_ops.py`, as `describe_frames`
    /// describes it.
    pub fn from_spec(spec: &Value) -> Result<Table, String> {
        let columns = spec
            .get("columns")
            .and_then(Value::as_object)
            .ok_or("a frame has columns")?;
        let mut series = Vec::new();
        let mut rows = 0;
        for (name, column) in columns {
            let mut s = Series::from_spec(column)?;
            s.name = Some(name.clone());
            rows = s.len();
            series.push(s);
        }
        Ok(Table::from_series(&series, rows))
    }
}

/// The pandas dtype an untagged Arrow column stands for.
fn default_dtype(data_type: &DataType) -> String {
    match data_type {
        DataType::Boolean => "bool".to_string(),
        DataType::Int8 => "int8".to_string(),
        DataType::Int16 => "int16".to_string(),
        DataType::Int32 => "int32".to_string(),
        DataType::Int64 => "int64".to_string(),
        DataType::UInt8 => "uint8".to_string(),
        DataType::UInt16 => "uint16".to_string(),
        DataType::UInt32 => "uint32".to_string(),
        DataType::UInt64 => "uint64".to_string(),
        DataType::Float32 => "float32".to_string(),
        DataType::Float64 => "float64".to_string(),
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => "str".to_string(),
        DataType::Timestamp(unit, None) => format!("datetime64[{}]", unit_name(*unit)),
        DataType::Timestamp(unit, Some(tz)) => format!("datetime64[{}, {tz}]", unit_name(*unit)),
        DataType::Duration(unit) => format!("timedelta64[{}]", unit_name(*unit)),
        _ => "object".to_string(),
    }
}

fn unit_name(unit: TimeUnit) -> &'static str {
    match unit {
        TimeUnit::Second => "s",
        TimeUnit::Millisecond => "ms",
        TimeUnit::Microsecond => "us",
        TimeUnit::Nanosecond => "ns",
    }
}

fn unit_of(name: &str) -> TimeUnit {
    match name {
        "s" => TimeUnit::Second,
        "ms" => TimeUnit::Millisecond,
        "us" => TimeUnit::Microsecond,
        _ => TimeUnit::Nanosecond,
    }
}

fn per_unit(unit: TimeUnit) -> i128 {
    match unit {
        TimeUnit::Second => 1_000_000_000,
        TimeUnit::Millisecond => 1_000_000,
        TimeUnit::Microsecond => 1_000,
        TimeUnit::Nanosecond => 1,
    }
}

/// A missing cell of a column of `dtype`, as `tolist()` gives it.
pub fn missing_of(dtype: &Dtype) -> PyVal {
    match dtype {
        Dtype::Float(_) | Dtype::Str | Dtype::Category => PyVal::Float(f64::NAN),
        Dtype::Datetime(..) | Dtype::Timedelta(_) => PyVal::NaT,
        Dtype::Object => PyVal::None,
        _ => PyVal::Na,
    }
}

/// A column as the values `tolist()` gives.
pub fn series_of(name: &str, array: &ArrayRef, tag: &Value) -> Series {
    let dtype_name = tag.get("dtype").and_then(Value::as_str).unwrap_or("object");
    let dtype = Dtype::parse(dtype_name);
    let json_cells = tag.get("cells").and_then(Value::as_str) == Some("json");
    let cells = cells_of(array, &dtype, json_cells);
    Series::new(Some(name.to_string()), dtype, cells)
}

fn cells_of(array: &ArrayRef, dtype: &Dtype, json_cells: bool) -> Vec<PyVal> {
    let n = array.len();
    let null = missing_of(dtype);
    macro_rules! prim {
        ($t:ty, $f:expr) => {{
            let a = array.as_primitive::<$t>();
            (0..n)
                .map(|i| {
                    if a.is_null(i) {
                        null.clone()
                    } else {
                        $f(a.value(i))
                    }
                })
                .collect()
        }};
    }
    let int = |v: i128| PyVal::Int(v);
    match array.data_type() {
        DataType::Null => vec![null; n],
        DataType::Boolean => {
            let a = array.as_boolean();
            (0..n)
                .map(|i| {
                    if a.is_null(i) {
                        null.clone()
                    } else {
                        PyVal::Bool(a.value(i))
                    }
                })
                .collect()
        }
        DataType::Int8 => prim!(Int8Type, |v: i8| int(i128::from(v))),
        DataType::Int16 => prim!(Int16Type, |v: i16| int(i128::from(v))),
        DataType::Int32 => prim!(Int32Type, |v: i32| int(i128::from(v))),
        DataType::Int64 => prim!(Int64Type, |v: i64| int(i128::from(v))),
        DataType::UInt8 => prim!(UInt8Type, |v: u8| int(i128::from(v))),
        DataType::UInt16 => prim!(UInt16Type, |v: u16| int(i128::from(v))),
        DataType::UInt32 => prim!(UInt32Type, |v: u32| int(i128::from(v))),
        DataType::UInt64 => prim!(UInt64Type, |v: u64| int(i128::from(v))),
        DataType::Float16 => {
            let a = array.as_primitive::<Float16Type>();
            (0..n)
                .map(|i| {
                    if a.is_null(i) {
                        null.clone()
                    } else {
                        PyVal::Float(f64::from(a.value(i).to_f32()))
                    }
                })
                .collect()
        }
        DataType::Float32 => prim!(Float32Type, |v: f32| PyVal::Float(f64::from(v))),
        DataType::Float64 => prim!(Float64Type, PyVal::Float),
        DataType::Timestamp(unit, tz) => {
            let per = per_unit(*unit);
            let offset = tz.as_deref().and_then(fixed_offset);
            let stamp = |v: i64| {
                let instant = i128::from(v) * per;
                match offset {
                    Some(o) => PyVal::TimestampTz(
                        Wall::from_epoch_nanos(instant + i128::from(o) * 1_000_000_000),
                        o,
                    ),
                    None => PyVal::Timestamp(Wall::from_epoch_nanos(instant)),
                }
            };
            match unit {
                TimeUnit::Second => prim!(TimestampSecondType, stamp),
                TimeUnit::Millisecond => prim!(TimestampMillisecondType, stamp),
                TimeUnit::Microsecond => prim!(TimestampMicrosecondType, stamp),
                TimeUnit::Nanosecond => prim!(TimestampNanosecondType, stamp),
            }
        }
        DataType::Duration(unit) => {
            let per = per_unit(*unit) as i64;
            let delta = |v: i64| PyVal::Timedelta(v.saturating_mul(per));
            match unit {
                TimeUnit::Second => prim!(DurationSecondType, delta),
                TimeUnit::Millisecond => prim!(DurationMillisecondType, delta),
                TimeUnit::Microsecond => prim!(DurationMicrosecondType, delta),
                TimeUnit::Nanosecond => prim!(DurationNanosecondType, delta),
            }
        }
        DataType::Decimal32(_, scale) => {
            let scale = *scale;
            prim!(Decimal32Type, |value: i32| PyVal::Decimal(
                sdm_schema::frame::decimal_text(i128::from(value), scale)
            ))
        }
        DataType::Decimal64(_, scale) => {
            let scale = *scale;
            prim!(Decimal64Type, |value: i64| PyVal::Decimal(
                sdm_schema::frame::decimal_text(i128::from(value), scale)
            ))
        }
        DataType::Decimal128(_, scale) => {
            let scale = *scale;
            prim!(Decimal128Type, |v: i128| PyVal::Decimal(
                sdm_schema::frame::decimal_text(v, scale)
            ))
        }
        DataType::Decimal256(_, scale) => {
            let scale = *scale;
            prim!(Decimal256Type, |value: arrow_buffer::i256| PyVal::Decimal(
                sdm_schema::frame::decimal_digits_text(&value.to_string(), scale)
            ))
        }
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => {
            let texts: Vec<Option<String>> = match array.data_type() {
                DataType::Utf8 => array
                    .as_string::<i32>()
                    .iter()
                    .map(|s| s.map(str::to_string))
                    .collect(),
                DataType::LargeUtf8 => array
                    .as_string::<i64>()
                    .iter()
                    .map(|s| s.map(str::to_string))
                    .collect(),
                _ => array
                    .as_string_view()
                    .iter()
                    .map(|s| s.map(str::to_string))
                    .collect(),
            };
            texts
                .into_iter()
                .map(|t| match t {
                    None => null.clone(),
                    Some(text) if json_cells => serde_json::from_str::<Value>(&text)
                        .ok()
                        .and_then(|v| PyVal::decode(&v).ok())
                        .unwrap_or(PyVal::Str(text)),
                    Some(text) => PyVal::Str(text),
                })
                .collect()
        }
        DataType::Binary | DataType::LargeBinary => {
            let values: Vec<Option<Vec<u8>>> = match array.data_type() {
                DataType::Binary => array
                    .as_binary::<i32>()
                    .iter()
                    .map(|b| b.map(<[u8]>::to_vec))
                    .collect(),
                _ => array
                    .as_binary::<i64>()
                    .iter()
                    .map(|b| b.map(<[u8]>::to_vec))
                    .collect(),
            };
            values
                .into_iter()
                .map(|b| b.map_or_else(|| null.clone(), PyVal::Bytes))
                .collect()
        }
        DataType::Dictionary(..) => {
            let dict = array.as_any_dictionary();
            let values = cells_of(dict.values(), &Dtype::Object, false);
            let keys = dict.normalized_keys();
            (0..n)
                .map(|i| {
                    if array.is_null(i) {
                        PyVal::Float(f64::NAN)
                    } else {
                        values.get(keys[i]).cloned().unwrap_or(PyVal::None)
                    }
                })
                .collect()
        }
        DataType::Struct(fields) => {
            let structure = array.as_struct();
            let columns: Vec<Vec<PyVal>> = structure.columns().iter().map(nested_cells).collect();
            (0..n)
                .map(|position| {
                    if array.is_null(position) {
                        null.clone()
                    } else {
                        PyVal::Dict(
                            fields
                                .iter()
                                .zip(&columns)
                                .map(|(field, values)| {
                                    (PyVal::Str(field.name().clone()), values[position].clone())
                                })
                                .collect(),
                        )
                    }
                })
                .collect()
        }
        DataType::Map(..) => {
            let mapping = array.as_map();
            (0..n)
                .map(|position| {
                    if array.is_null(position) {
                        return null.clone();
                    }
                    let entries = mapping.value(position);
                    let keys = nested_cells(entries.column(0));
                    let values = nested_cells(entries.column(1));
                    PyVal::List(
                        keys.into_iter()
                            .zip(values)
                            .map(|(key, value)| PyVal::Tuple(vec![key, value]))
                            .collect(),
                    )
                })
                .collect()
        }
        DataType::List(_) | DataType::LargeList(_) | DataType::FixedSizeList(..) => (0..n)
            .map(|position| {
                if array.is_null(position) {
                    return null.clone();
                }
                let values = match array.data_type() {
                    DataType::List(_) => array.as_list::<i32>().value(position),
                    DataType::LargeList(_) => array.as_list::<i64>().value(position),
                    _ => array.as_fixed_size_list().value(position),
                };
                PyVal::List(nested_cells(&values))
            })
            .collect(),
        // A date, or a time of day, as the `datetime.date` or
        // `datetime.time` `tolist()` gives.
        DataType::Date32 | DataType::Date64 | DataType::Time32(_) | DataType::Time64(_) => (0..n)
            .map(|i| {
                if array.is_null(i) {
                    null.clone()
                } else {
                    sdm_schema::engine::result::value_at(array, i)
                }
            })
            .collect(),
        _ => vec![PyVal::None; n],
    }
}

fn nested_cells(array: &ArrayRef) -> Vec<PyVal> {
    if matches!(array.data_type(), DataType::Timestamp(..)) {
        return (0..array.len())
            .map(|position| sdm_schema::engine::result::value_at(array, position))
            .collect();
    }
    cells_of(array, &Dtype::Object, false)
}

/// The Arrow array and tag `describe_frames` gives a column.
pub fn arrow_of(series: &Series) -> (ArrayRef, Value) {
    let tag = json!({ "dtype": series.dtype.name() });
    let cells = &series.cells;
    let is_null = |v: &PyVal| crate::series::is_na(v);
    let ints = |v: &PyVal| -> Option<i128> {
        match crate::pyops::item(v) {
            PyVal::Int(n) => Some(n),
            PyVal::Bool(b) => Some(i128::from(b)),
            _ => None,
        }
    };
    let floats = |v: &PyVal| -> Option<f64> {
        if is_null(v) {
            return None;
        }
        match crate::pyops::item(v) {
            PyVal::Float(x) => Some(x),
            PyVal::Int(n) => Some(n as f64),
            _ => None,
        }
    };
    let array: ArrayRef = match &series.dtype {
        Dtype::Bool | Dtype::Nullable(_) if series.dtype.is_bool() => Arc::new(BooleanArray::from(
            cells
                .iter()
                .map(|v| match crate::pyops::item(v) {
                    PyVal::Bool(b) => Some(b),
                    _ => None,
                })
                .collect::<Vec<_>>(),
        )),
        Dtype::Int(name) | Dtype::Nullable(name) if series.dtype.is_integer() => {
            let values: Vec<Option<i128>> = cells
                .iter()
                .map(|v| if is_null(v) { None } else { ints(v) })
                .collect();
            int_array(&name.to_lowercase(), &values)
        }
        Dtype::Arrow(name) if series.dtype.is_integer() => {
            let values: Vec<Option<i128>> = cells
                .iter()
                .map(|v| if is_null(v) { None } else { ints(v) })
                .collect();
            int_array(name, &values)
        }
        Dtype::Float(32) | Dtype::Arrow(_)
            if series.dtype.float_itemsize() == 4 && series.dtype.is_float() =>
        {
            Arc::new(Float32Array::from(
                cells
                    .iter()
                    .map(|v| floats(v).map(|x| x as f32))
                    .collect::<Vec<_>>(),
            ))
        }
        _ if series.dtype.is_float() => Arc::new(Float64Array::from(
            cells.iter().map(floats).collect::<Vec<_>>(),
        )),
        Dtype::Datetime(unit, tz) => {
            let unit = unit_of(unit);
            let per = per_unit(unit);
            let values: Vec<Option<i64>> = cells
                .iter()
                .map(|v| match v {
                    PyVal::Timestamp(w) => i64::try_from(w.epoch_nanos().div_euclid(per)).ok(),
                    PyVal::TimestampTz(w, o) => i64::try_from(
                        (w.epoch_nanos() - i128::from(*o) * 1_000_000_000).div_euclid(per),
                    )
                    .ok(),
                    _ => None,
                })
                .collect();
            let zone: Option<Arc<str>> = tz.as_deref().map(Arc::from);
            match unit {
                TimeUnit::Second => {
                    Arc::new(TimestampSecondArray::from(values).with_timezone_opt(zone))
                }
                TimeUnit::Millisecond => {
                    Arc::new(TimestampMillisecondArray::from(values).with_timezone_opt(zone))
                }
                TimeUnit::Microsecond => {
                    Arc::new(TimestampMicrosecondArray::from(values).with_timezone_opt(zone))
                }
                TimeUnit::Nanosecond => {
                    Arc::new(TimestampNanosecondArray::from(values).with_timezone_opt(zone))
                }
            }
        }
        Dtype::Timedelta(unit) => {
            let unit = unit_of(unit);
            let per = per_unit(unit) as i64;
            let values: Vec<Option<i64>> = cells
                .iter()
                .map(|v| match v {
                    PyVal::Timedelta(n) => Some(n.div_euclid(per)),
                    _ => None,
                })
                .collect();
            match unit {
                TimeUnit::Second => Arc::new(DurationSecondArray::from(values)),
                TimeUnit::Millisecond => Arc::new(DurationMillisecondArray::from(values)),
                TimeUnit::Microsecond => Arc::new(DurationMicrosecondArray::from(values)),
                TimeUnit::Nanosecond => Arc::new(DurationNanosecondArray::from(values)),
            }
        }
        // `pd.ArrowDtype(pa.string())`, which `str()` names
        // `string[pyarrow]`: Arrow's `string`, as `pa.array` gives it.
        Dtype::String(name) if name == "string[pyarrow]" => Arc::new(StringArray::from(
            cells
                .iter()
                .map(|v| match v {
                    PyVal::Str(s) => Some(s.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>(),
        )),
        Dtype::Str | Dtype::String(_) => Arc::new(LargeStringArray::from(
            cells
                .iter()
                .map(|v| match v {
                    PyVal::Str(s) => Some(s.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>(),
        )),
        Dtype::Category => {
            let mut categories: Vec<PyVal> = Vec::new();
            let mut keys: Vec<Option<i8>> = Vec::new();
            for v in cells {
                if is_null(v) {
                    keys.push(None);
                    continue;
                }
                let at = match categories.iter().position(|c| c.eq_key() == v.eq_key()) {
                    Some(at) => at,
                    None => {
                        categories.push(v.clone());
                        categories.len() - 1
                    }
                };
                keys.push(i8::try_from(at).ok());
            }
            // pandas orders a categorical's categories by value.
            let mut order: Vec<usize> = (0..categories.len()).collect();
            order.sort_by(|&a, &b| compare_cells(&categories[a], &categories[b]));
            let rank: Vec<usize> = {
                let mut rank = vec![0; order.len()];
                for (r, &i) in order.iter().enumerate() {
                    rank[i] = r;
                }
                rank
            };
            let keys: Vec<Option<i8>> = keys
                .into_iter()
                .map(|k| k.and_then(|k| i8::try_from(rank[k as usize]).ok()))
                .collect();
            let sorted: Vec<PyVal> = order.iter().map(|&i| categories[i].clone()).collect();
            let values: ArrayRef = if sorted.iter().all(|v| matches!(v, PyVal::Str(_))) {
                Arc::new(LargeStringArray::from(
                    sorted.iter().map(crate::pyops::py_str).collect::<Vec<_>>(),
                ))
            } else {
                Arc::new(Int64Array::from(
                    sorted
                        .iter()
                        .map(|v| ints(v).and_then(|n| i64::try_from(n).ok()))
                        .collect::<Vec<_>>(),
                ))
            };
            Arc::new(
                DictionaryArray::<Int8Type>::try_new(Int8Array::from(keys), values)
                    .unwrap_or_else(|_| DictionaryArray::<Int8Type>::from_iter(Vec::<&str>::new())),
            )
        }
        Dtype::Arrow(_) if series.dtype.arrow_decimal().is_some() => {
            let (precision, scale) = series.dtype.arrow_decimal().unwrap_or((38, 0));
            let values: Vec<Option<i128>> = cells
                .iter()
                .map(|v| match v {
                    PyVal::Decimal(text) => Dec::parse(text).and_then(|d| scaled(&d, scale)),
                    _ => None,
                })
                .collect();
            Arc::new(
                Decimal128Array::from(values)
                    .with_precision_and_scale(precision as u8, scale as i8)
                    .unwrap_or_else(|_| Decimal128Array::from(Vec::<Option<i128>>::new())),
            )
        }
        Dtype::Arrow(name) if arrow_scalar_type(name).is_some() => {
            let data_type = arrow_scalar_type(name).expect("checked");
            arrow_scalars(&data_type, cells)
        }
        Dtype::Arrow(name) if name == "string" || name == "large_string" => {
            Arc::new(LargeStringArray::from(
                cells
                    .iter()
                    .map(|v| match v {
                        PyVal::Str(s) => Some(s.clone()),
                        _ => None,
                    })
                    .collect::<Vec<_>>(),
            ))
        }
        _ => {
            let content = infer_dtype(cells);
            let texts: Vec<Option<String>> = cells
                .iter()
                .map(|v| Some(sort_keys(&crate::layout::encode_deep(v)).to_string()))
                .collect();
            return (
                Arc::new(StringArray::from(texts)),
                json!({ "dtype": series.dtype.name(), "content": content, "cells": "json" }),
            );
        }
    };
    (array, tag)
}

/// The Arrow type of a `pd.ArrowDtype` of scalars whose name, as `str()`
/// writes it without `[pyarrow]`, is `name`: the types whose cells
/// [`cells_of`] reads as Python scalars and [`arrow_scalars`] writes back.
fn arrow_scalar_type(name: &str) -> Option<DataType> {
    let unit = |text: &str| -> Option<TimeUnit> {
        Some(match text {
            "s" => TimeUnit::Second,
            "ms" => TimeUnit::Millisecond,
            "us" => TimeUnit::Microsecond,
            "ns" => TimeUnit::Nanosecond,
            _ => return None,
        })
    };
    let inside = |prefix: &str| -> Option<&str> {
        name.strip_prefix(prefix)?
            .strip_prefix('[')?
            .strip_suffix(']')
    };
    let fixed = match name {
        "bool" => Some(DataType::Boolean),
        "date32[day]" => Some(DataType::Date32),
        "date64[ms]" => Some(DataType::Date64),
        "binary" => Some(DataType::Binary),
        "large_binary" => Some(DataType::LargeBinary),
        _ => None,
    };
    if fixed.is_some() {
        return fixed;
    }
    if let Some(given) = inside("timestamp") {
        let (unit_text, zone) = match given.split_once(", tz=") {
            Some((u, z)) => (u, Some(Arc::<str>::from(z))),
            None => (given, None),
        };
        return Some(DataType::Timestamp(unit(unit_text)?, zone));
    }
    if let Some(given) = inside("duration") {
        return Some(DataType::Duration(unit(given)?));
    }
    if let Some(given) = inside("time32") {
        return unit(given)
            .filter(|u| matches!(u, TimeUnit::Second | TimeUnit::Millisecond))
            .map(DataType::Time32);
    }
    unit(inside("time64")?)
        .filter(|u| matches!(u, TimeUnit::Microsecond | TimeUnit::Nanosecond))
        .map(DataType::Time64)
}

/// The cells of a `pd.ArrowDtype` column of `data_type`, as
/// `pa.array(series)` holds them: the inverse of [`cells_of`].
fn arrow_scalars(data_type: &DataType, cells: &[PyVal]) -> ArrayRef {
    let na = |v: &PyVal| crate::series::is_na(v);
    let micros_of_day = |v: &PyVal| -> Option<i64> {
        match v {
            PyVal::Time {
                hour,
                minute,
                second,
                micro,
                ..
            } => Some(
                (i64::from(*hour) * 3600 + i64::from(*minute) * 60 + i64::from(*second))
                    * 1_000_000
                    + i64::from(*micro),
            ),
            _ => None,
        }
    };
    let days = |v: &PyVal| -> Option<i64> {
        match v {
            PyVal::Date(y, m, d) => Some(sdm_schema::dt::civil::days_from_civil(*y, *m, *d)),
            _ => None,
        }
    };
    match data_type {
        DataType::Boolean => Arc::new(BooleanArray::from(
            cells
                .iter()
                .map(|v| match crate::pyops::item(v) {
                    PyVal::Bool(b) if !na(v) => Some(b),
                    _ => None,
                })
                .collect::<Vec<_>>(),
        )),
        DataType::Date32 => Arc::new(Date32Array::from(
            cells
                .iter()
                .map(|v| days(v).and_then(|d| i32::try_from(d).ok()))
                .collect::<Vec<_>>(),
        )),
        DataType::Date64 => Arc::new(Date64Array::from(
            cells
                .iter()
                .map(|v| days(v).map(|d| d * 86_400_000))
                .collect::<Vec<_>>(),
        )),
        DataType::Timestamp(unit, zone) => {
            let per = per_unit(*unit);
            let values: Vec<Option<i64>> = cells
                .iter()
                .map(|v| match v {
                    PyVal::Timestamp(w) => i64::try_from(w.epoch_nanos().div_euclid(per)).ok(),
                    PyVal::TimestampTz(w, o) => i64::try_from(
                        (w.epoch_nanos() - i128::from(*o) * 1_000_000_000).div_euclid(per),
                    )
                    .ok(),
                    _ => None,
                })
                .collect();
            let zone = zone.clone();
            match unit {
                TimeUnit::Second => {
                    Arc::new(TimestampSecondArray::from(values).with_timezone_opt(zone))
                }
                TimeUnit::Millisecond => {
                    Arc::new(TimestampMillisecondArray::from(values).with_timezone_opt(zone))
                }
                TimeUnit::Microsecond => {
                    Arc::new(TimestampMicrosecondArray::from(values).with_timezone_opt(zone))
                }
                TimeUnit::Nanosecond => {
                    Arc::new(TimestampNanosecondArray::from(values).with_timezone_opt(zone))
                }
            }
        }
        DataType::Duration(unit) => {
            let per = per_unit(*unit) as i64;
            let values: Vec<Option<i64>> = cells
                .iter()
                .map(|v| match v {
                    PyVal::Timedelta(ns) => Some(ns.div_euclid(per)),
                    _ => None,
                })
                .collect();
            match unit {
                TimeUnit::Second => Arc::new(DurationSecondArray::from(values)),
                TimeUnit::Millisecond => Arc::new(DurationMillisecondArray::from(values)),
                TimeUnit::Microsecond => Arc::new(DurationMicrosecondArray::from(values)),
                TimeUnit::Nanosecond => Arc::new(DurationNanosecondArray::from(values)),
            }
        }
        DataType::Time32(unit) => {
            let per: i64 = if matches!(unit, TimeUnit::Second) {
                1_000_000
            } else {
                1_000
            };
            let values: Vec<Option<i32>> = cells
                .iter()
                .map(|v| micros_of_day(v).and_then(|m| i32::try_from(m / per).ok()))
                .collect();
            if matches!(unit, TimeUnit::Second) {
                Arc::new(Time32SecondArray::from(values))
            } else {
                Arc::new(Time32MillisecondArray::from(values))
            }
        }
        DataType::Time64(unit) => {
            let values: Vec<Option<i64>> = cells
                .iter()
                .map(|v| {
                    micros_of_day(v).map(|m| {
                        if matches!(unit, TimeUnit::Nanosecond) {
                            m * 1_000
                        } else {
                            m
                        }
                    })
                })
                .collect();
            if matches!(unit, TimeUnit::Nanosecond) {
                Arc::new(Time64NanosecondArray::from(values))
            } else {
                Arc::new(Time64MicrosecondArray::from(values))
            }
        }
        DataType::LargeBinary => {
            Arc::new(LargeBinaryArray::from_iter(cells.iter().map(|v| match v {
                PyVal::Bytes(b) => Some(b.clone()),
                _ => None,
            })))
        }
        _ => Arc::new(BinaryArray::from_iter(cells.iter().map(|v| match v {
            PyVal::Bytes(b) => Some(b.clone()),
            _ => None,
        }))),
    }
}

/// JSON with every object's keys sorted, as `json.dumps(sort_keys=True)`.
fn sort_keys(value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            Value::Object(
                keys.into_iter()
                    .map(|k| (k.clone(), sort_keys(&map[k])))
                    .collect(),
            )
        }
        Value::Array(items) => Value::Array(items.iter().map(sort_keys).collect()),
        other => other.clone(),
    }
}

/// An exact decimal as the integer of `scale` places an Arrow decimal holds.
fn scaled(value: &Dec, scale: i32) -> Option<i128> {
    if !value.is_finite() {
        return None;
    }
    let shift = value.exponent + i64::from(scale);
    let digits = if shift >= 0 {
        format!("{}{}", value.digits, "0".repeat(shift as usize))
    } else {
        let cut = (-shift) as usize;
        if cut >= value.digits.len() {
            "0".to_string()
        } else {
            value.digits[..value.digits.len() - cut].to_string()
        }
    };
    let magnitude: i128 = digits.parse().ok()?;
    Some(if value.negative {
        -magnitude
    } else {
        magnitude
    })
}

fn int_array(name: &str, values: &[Option<i128>]) -> ArrayRef {
    macro_rules! of {
        ($a:ty, $t:ty) => {
            Arc::new(<$a>::from(
                values
                    .iter()
                    .map(|v| v.and_then(|n| <$t>::try_from(n).ok()))
                    .collect::<Vec<_>>(),
            ))
        };
    }
    match name {
        "int8" => of!(Int8Array, i8),
        "int16" => of!(Int16Array, i16),
        "int32" => of!(Int32Array, i32),
        "uint8" => of!(UInt8Array, u8),
        "uint16" => of!(UInt16Array, u16),
        "uint32" => of!(UInt32Array, u32),
        "uint64" => of!(UInt64Array, u64),
        _ => of!(Int64Array, i64),
    }
}

/// An order of two cells, as Python sorts like values.
pub fn compare_cells(a: &PyVal, b: &PyVal) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    match (crate::pyops::item(a), crate::pyops::item(b)) {
        (PyVal::Str(x), PyVal::Str(y)) => x.cmp(&y),
        (x, y) => {
            let fx = crate::pyops::py_float(&x).unwrap_or(f64::NAN);
            let fy = crate::pyops::py_float(&y).unwrap_or(f64::NAN);
            fx.partial_cmp(&fy).unwrap_or(Ordering::Equal)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nested_map_cells_keep_duplicate_keys_order_nulls_and_sliced_offsets() {
        use arrow_array::builder::{Int64Builder, MapBuilder, StringBuilder};

        let mut builder = MapBuilder::new(None, StringBuilder::new(), Int64Builder::new());
        builder.keys().append_value("sentinel");
        builder.values().append_value(0);
        builder.append(true).unwrap();
        for (key, value) in [("a", Some(2)), ("a", None), ("z", Some(4))] {
            builder.keys().append_value(key);
            builder.values().append_option(value);
        }
        builder.append(true).unwrap();
        builder.append(true).unwrap();
        builder.append(false).unwrap();
        let mapping = builder.finish();
        let DataType::Map(field, _) = mapping.data_type() else {
            panic!("map builder must produce a map");
        };
        for sorted in [false, true] {
            let array: ArrayRef = Arc::new(arrow_array::MapArray::new(
                field.clone(),
                mapping.offsets().clone(),
                mapping.entries().clone(),
                mapping.nulls().cloned(),
                sorted,
            ));
            assert_eq!(
                cells_of(&array.slice(1, 3), &Dtype::Object, false),
                vec![
                    PyVal::List(vec![
                        PyVal::Tuple(vec![PyVal::Str("a".into()), PyVal::Int(2)]),
                        PyVal::Tuple(vec![PyVal::Str("a".into()), PyVal::None]),
                        PyVal::Tuple(vec![PyVal::Str("z".into()), PyVal::Int(4)]),
                    ]),
                    PyVal::List(vec![]),
                    PyVal::None,
                ]
            );
        }
    }

    #[test]
    fn nested_struct_lists_and_dictionary_values_keep_their_cells() {
        use arrow_array::{ListArray, StructArray};
        use arrow_buffer::{NullBuffer, OffsetBuffer};

        let lists: ArrayRef = Arc::new(ListArray::new(
            Arc::new(Field::new("item", DataType::Int64, true)),
            OffsetBuffer::new(vec![0, 1, 3, 3, 3].into()),
            Arc::new(Int64Array::from(vec![Some(9), Some(1), None])),
            Some(NullBuffer::from(vec![true, true, true, false])),
        ));
        let fields = vec![
            Arc::new(Field::new("numbers", lists.data_type().clone(), true)),
            Arc::new(Field::new("label", DataType::Utf8, true)),
        ];
        let structure: ArrayRef = Arc::new(StructArray::new(
            fields.into(),
            vec![
                lists,
                Arc::new(StringArray::from(vec![
                    Some("skip"),
                    Some("one"),
                    None,
                    Some("masked"),
                ])),
            ],
            Some(NullBuffer::from(vec![true, true, true, false])),
        ));
        let expected = vec![
            PyVal::Dict(vec![
                (
                    PyVal::Str("numbers".into()),
                    PyVal::List(vec![PyVal::Int(1), PyVal::None]),
                ),
                (PyVal::Str("label".into()), PyVal::Str("one".into())),
            ]),
            PyVal::Dict(vec![
                (PyVal::Str("numbers".into()), PyVal::List(vec![])),
                (PyVal::Str("label".into()), PyVal::None),
            ]),
            PyVal::None,
        ];
        assert_eq!(
            cells_of(&structure.slice(1, 3), &Dtype::Object, false),
            expected
        );
        let dictionary: ArrayRef = Arc::new(
            DictionaryArray::<Int8Type>::try_new(
                Int8Array::from(vec![Some(2), Some(1), Some(3)]),
                structure,
            )
            .unwrap(),
        );
        assert_eq!(
            cells_of(&dictionary, &Dtype::Object, false),
            vec![expected[1].clone(), expected[0].clone(), PyVal::None]
        );
    }

    #[test]
    fn nested_temporal_cells_keep_arrow_scalar_classes() {
        use arrow_array::StructArray;

        let instant = 1_577_934_245_123_456;
        let columns: Vec<ArrayRef> = vec![
            Arc::new(Date32Array::from(vec![Some(18_263), None])),
            Arc::new(TimestampMicrosecondArray::from(vec![Some(instant), None])),
            Arc::new(TimestampNanosecondArray::from(vec![
                Some(instant * 1_000),
                None,
            ])),
            Arc::new(StringArray::from(vec![Some("a'b\"c"), None])),
        ];
        let fields: Vec<_> = ["day", "microsecond", "nanosecond", "label"]
            .iter()
            .zip(&columns)
            .map(|(name, column)| Arc::new(Field::new(*name, column.data_type().clone(), true)))
            .collect();
        let structure: ArrayRef = Arc::new(StructArray::new(fields.into(), columns, None));
        let cells = cells_of(&structure, &Dtype::Object, false);
        assert_eq!(crate::cells::cell_json_value(&cells[0], "string").unwrap(),
            PyVal::Str("{'day': datetime.date(2020, 1, 2), 'microsecond': datetime.datetime(2020, 1, 2, 3, 4, 5, 123456), 'nanosecond': '2020-01-02T03:04:05.123456Z', 'label': 'a\\'b\"c'}".into()));
        assert_eq!(
            crate::cells::cell_json_value(&cells[1], "string").unwrap(),
            PyVal::Str(
                "{'day': None, 'microsecond': None, 'nanosecond': None, 'label': None}".into()
            )
        );
    }

    #[test]
    fn nested_dictionary_null_keys_encode_as_null_cells() {
        use arrow_array::StructArray;

        let dictionary: ArrayRef = Arc::new(
            DictionaryArray::<Int8Type>::try_new(
                Int8Array::from(vec![None, Some(0)]),
                Arc::new(StringArray::from(vec!["value"])),
            )
            .unwrap(),
        );
        let structure: ArrayRef = Arc::new(StructArray::new(
            vec![Arc::new(Field::new(
                "label",
                dictionary.data_type().clone(),
                true,
            ))]
            .into(),
            vec![dictionary],
            None,
        ));
        let cells = cells_of(&structure, &Dtype::Object, false);
        assert_eq!(
            crate::cells::cell_json_value(&cells[0], "string").unwrap(),
            PyVal::Str("{'label': None}".into())
        );
        assert_eq!(
            crate::cells::cell_json_value(&cells[1], "string").unwrap(),
            PyVal::Str("{'label': 'value'}".into())
        );
    }

    #[test]
    fn nested_list_widths_and_fixed_size_lists_keep_null_and_empty_distinct() {
        use arrow_array::{FixedSizeListArray, LargeListArray, ListArray};
        use arrow_buffer::{NullBuffer, OffsetBuffer};

        let field = Arc::new(Field::new("item", DataType::Int64, true));
        let values: ArrayRef = Arc::new(Int64Array::from(vec![Some(9), Some(1), None]));
        let nulls = Some(NullBuffer::from(vec![true, true, true, false]));
        let arrays: Vec<ArrayRef> = vec![
            Arc::new(ListArray::new(
                field.clone(),
                OffsetBuffer::new(vec![0, 1, 3, 3, 3].into()),
                values.clone(),
                nulls.clone(),
            )),
            Arc::new(LargeListArray::new(
                field.clone(),
                OffsetBuffer::new(vec![0, 1, 3, 3, 3].into()),
                values,
                nulls,
            )),
        ];
        for array in arrays {
            assert_eq!(
                cells_of(&array.slice(1, 3), &Dtype::Object, false),
                vec![
                    PyVal::List(vec![PyVal::Int(1), PyVal::None]),
                    PyVal::List(vec![]),
                    PyVal::None
                ]
            );
        }
        let fixed: ArrayRef = Arc::new(FixedSizeListArray::new(
            field,
            2,
            Arc::new(Int64Array::from(vec![
                Some(9),
                Some(9),
                Some(1),
                None,
                Some(2),
                Some(3),
            ])),
            Some(NullBuffer::from(vec![true, true, false])),
        ));
        assert_eq!(
            cells_of(&fixed.slice(1, 2), &Dtype::Object, false),
            vec![PyVal::List(vec![PyVal::Int(1), PyVal::None]), PyVal::None]
        );
    }

    #[test]
    fn narrow_arrow_decimals_keep_their_numeric_cells() {
        let columns: Vec<(&str, ArrayRef)> = vec![
            (
                "decimal32(8, 2)",
                Arc::new(
                    arrow_array::Decimal32Array::from(vec![Some(125), None, Some(-5)])
                        .with_precision_and_scale(8, 2)
                        .unwrap(),
                ),
            ),
            (
                "decimal64(16, 2)",
                Arc::new(
                    arrow_array::Decimal64Array::from(vec![Some(125), None, Some(-5)])
                        .with_precision_and_scale(16, 2)
                        .unwrap(),
                ),
            ),
        ];
        for (dtype, array) in columns {
            let series = series_of(
                "amount",
                &array,
                &json!({"dtype": format!("{dtype}[pyarrow]")}),
            );
            assert_eq!(series.dtype, Dtype::Arrow(dtype.to_string()));
            assert_eq!(
                series.dtype.arrow_decimal(),
                Some((
                    if dtype.starts_with("decimal32") {
                        8
                    } else {
                        16
                    },
                    2
                ))
            );
            assert_eq!(
                series.cells,
                vec![
                    PyVal::Decimal("1.25".to_string()),
                    PyVal::Na,
                    PyVal::Decimal("-0.05".to_string())
                ]
            );
        }
    }

    /// A `pd.ArrowDtype` column read as Python values and written back is
    /// the array `describe_frames` handed over, of the same Arrow type, so
    /// the sampler reads what pandas holds.
    #[test]
    fn arrow_scalars_are_written_back_as_their_own_type() {
        let columns: Vec<(&str, ArrayRef)> = vec![
            ("bool", Arc::new(BooleanArray::from(vec![Some(true), None]))),
            ("string", Arc::new(StringArray::from(vec![Some("a"), None]))),
            (
                "large_string",
                Arc::new(LargeStringArray::from(vec![Some("a"), None])),
            ),
            (
                "date32[day]",
                Arc::new(Date32Array::from(vec![Some(18_262), None])),
            ),
            (
                "date64[ms]",
                Arc::new(Date64Array::from(vec![Some(1_577_836_800_000), None])),
            ),
            (
                "timestamp[s]",
                Arc::new(TimestampSecondArray::from(vec![Some(1_577_836_800), None])),
            ),
            (
                "timestamp[us]",
                Arc::new(TimestampMicrosecondArray::from(vec![
                    Some(1_577_836_800_000_000),
                    None,
                ])),
            ),
            (
                "timestamp[ns, tz=UTC]",
                Arc::new(
                    TimestampNanosecondArray::from(vec![Some(1_577_836_800_000_000_123), None])
                        .with_timezone("UTC"),
                ),
            ),
            (
                "duration[us]",
                Arc::new(DurationMicrosecondArray::from(vec![Some(90_000_000), None])),
            ),
            (
                "time64[us]",
                Arc::new(Time64MicrosecondArray::from(vec![
                    Some(3_723_000_004),
                    None,
                ])),
            ),
            (
                "time32[s]",
                Arc::new(Time32SecondArray::from(vec![Some(3_723), None])),
            ),
            ("float", Arc::new(Float32Array::from(vec![Some(1.5), None]))),
            (
                "binary",
                Arc::new(BinaryArray::from_iter(vec![Some(b"ab".to_vec()), None])),
            ),
        ];
        for (name, array) in columns {
            let tag = json!({ "dtype": format!("{name}[pyarrow]") });
            let series = series_of("c", &array, &tag);
            let (back, back_tag) = arrow_of(&series);
            assert_eq!(back.data_type(), array.data_type(), "{name}");
            assert_eq!(back.as_ref(), array.as_ref(), "{name}");
            assert_eq!(back_tag, tag, "{name}");
        }
    }

    /// `str(pd.ArrowDtype(type))` without `[pyarrow]`, read back as the type.
    #[test]
    fn arrow_scalar_types_are_read_from_their_names() {
        assert_eq!(
            arrow_scalar_type("timestamp[ms, tz=Europe/Berlin]"),
            Some(DataType::Timestamp(
                TimeUnit::Millisecond,
                Some(Arc::from("Europe/Berlin"))
            ))
        );
        assert_eq!(arrow_scalar_type("time32[us]"), None);
        assert_eq!(arrow_scalar_type("time64[s]"), None);
        assert_eq!(arrow_scalar_type("timestamp[d]"), None);
        assert_eq!(arrow_scalar_type("list<item: int64>"), None);
    }
}
