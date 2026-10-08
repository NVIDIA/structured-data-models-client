// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The frames the sampler builds, as pandas holds them: each column a
//! pandas dtype and its values as one Arrow array, the array
//! `pa.Table.from_pandas` would give of the column.
//!
//! The dtype is kept because it decides what pandas does with the column
//! later: how a key is hashed, how a label is computed, and what the host
//! gets back. A SQL result read with `to_pandas(types_mapper=pd.ArrowDtype)`
//! is [`Dtype::Arrow`]; the in-memory tables keep the numpy and extension
//! dtypes they were given.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{
    Float32Type, Float64Type, Int8Type, Int16Type, Int32Type, Int64Type, UInt8Type, UInt16Type,
    UInt32Type, UInt64Type,
};
use arrow_array::{Array, ArrayRef, StringArray, StructArray, UInt64Array, new_empty_array};
use arrow_schema::{DataType, Field};
use sdm_schema::pyval::PyVal;

use crate::driver::Raised;

pub(crate) mod decimal_lists;

/// Tagged object cells whose identities must survive Arrow row selection.
pub fn encoded_objects(values: Vec<String>) -> ArrayRef {
    let field = Arc::new(Field::new("sdm.python.object", DataType::Utf8, false));
    Arc::new(StructArray::from(vec![(
        field,
        Arc::new(StringArray::from(values)) as ArrayRef,
    )]))
}

/// Decode the internal object column without interpreting ordinary strings.
pub fn object_values(array: &ArrayRef) -> Option<Vec<PyVal>> {
    let DataType::Struct(fields) = array.data_type() else {
        return None;
    };
    if fields.len() != 1 || fields[0].name() != "sdm.python.object" {
        return None;
    }
    array
        .as_struct()
        .column(0)
        .as_string::<i32>()
        .iter()
        .map(|text| {
            serde_json::from_str(text?)
                .ok()
                .and_then(|value| PyVal::decode(&value).ok())
        })
        .collect()
}

/// A column's pandas dtype, as far as the sampler tells them apart.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Dtype {
    /// `pd.ArrowDtype` of the array's own type.
    Arrow,
    /// A numpy dtype by name, such as `int64`, `float64`, `bool` or
    /// `datetime64[ns]`.
    Numpy(String),
    /// A pandas extension dtype by name: `Int64`, `Float64`, `boolean`,
    /// `string` or `str`.
    Extension(String),
    /// `object`.
    Object,
    /// `category`: a pandas `Categorical`, whose array is an Arrow
    /// dictionary of its categories.
    Category,
}

impl Dtype {
    /// The dtype `pa.Table.from_pandas` records for a column as its
    /// `numpy_type`.
    pub fn from_numpy_type(name: &str) -> Self {
        if name.ends_with("[pyarrow]") {
            return Dtype::Arrow;
        }
        match name {
            "object" => Dtype::Object,
            "Int8" | "Int16" | "Int32" | "Int64" | "UInt8" | "UInt16" | "UInt32" | "UInt64"
            | "Float32" | "Float64" | "boolean" | "string" | "str" => {
                Dtype::Extension(name.to_string())
            }
            _ => Dtype::Numpy(name.to_string()),
        }
    }

    /// The dtype of a column `pa.Table.from_pandas` wrote as `array`, with
    /// `numpy_type` in its pandas metadata: a `Categorical` is written as a
    /// dictionary with the type of its codes, so the array tells it.
    pub fn of_column(numpy_type: Option<&str>, array: &dyn Array) -> Self {
        if matches!(array.data_type(), DataType::Dictionary(..)) {
            return Dtype::Category;
        }
        numpy_type.map_or(Dtype::Arrow, Dtype::from_numpy_type)
    }

    /// The tag the host reads the dtype back from (`capture_sampling.py`,
    /// `to_python`).
    pub fn tag(&self) -> String {
        match self {
            Dtype::Arrow => "arrow".to_string(),
            Dtype::Numpy(name) | Dtype::Extension(name) => name.clone(),
            Dtype::Object => "object".to_string(),
            Dtype::Category => "category".to_string(),
        }
    }

    /// The `numpy_type` pandas metadata gives the label executor's reader
    /// (`sdm_labels::arrow::column`) for a column of this dtype.
    pub fn numpy_type(&self, array: &dyn Array) -> String {
        match self {
            // The executor casts numeric ArrowDtype columns to numpy before
            // it computes with them, and compares the others as objects.
            Dtype::Arrow => match array.data_type() {
                DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => "string".to_string(),
                _ => "object".to_string(),
            },
            other => other.tag(),
        }
    }

    pub fn is_datetime(&self) -> bool {
        matches!(self, Dtype::Numpy(name) if name.starts_with("datetime64"))
    }
}

/// One column: its dtype and values.
#[derive(Clone, Debug)]
pub struct Series {
    pub dtype: Dtype,
    pub array: ArrayRef,
    original_values: Option<Vec<PyVal>>,
}

impl Series {
    pub fn new(dtype: Dtype, array: ArrayRef) -> Self {
        Self {
            dtype,
            array,
            original_values: None,
        }
    }

    pub fn with_original_values(mut self, values: Vec<PyVal>) -> Self {
        assert_eq!(self.len(), values.len());
        self.original_values = Some(values);
        self
    }

    pub fn object_values(&self) -> Option<Vec<PyVal>> {
        self.original_values
            .clone()
            .or_else(|| object_values(&self.array))
    }

    pub fn slice(&self, offset: usize, length: usize) -> Self {
        Self {
            dtype: self.dtype.clone(),
            array: self.array.slice(offset, length),
            original_values: self
                .original_values
                .as_ref()
                .map(|values| values[offset..offset + length].to_vec()),
        }
    }

    pub fn len(&self) -> usize {
        self.array.len()
    }

    pub fn is_empty(&self) -> bool {
        self.array.is_empty()
    }

    /// `series.iloc[positions]`.
    pub fn take(&self, positions: &[usize]) -> Result<Series, Raised> {
        Ok(Series {
            dtype: self.dtype.clone(),
            array: take(&self.array, positions)?,
            original_values: self.original_values.as_ref().map(|values| {
                positions
                    .iter()
                    .map(|position| values[*position].clone())
                    .collect()
            }),
        })
    }

    /// `pd.api.types.is_string_dtype(series)`, as the mapper asks it.
    pub fn is_string_dtype(&self) -> bool {
        let text = matches!(
            self.array.data_type(),
            DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View
        );
        match &self.dtype {
            Dtype::Arrow => text,
            Dtype::Extension(name) => name == "string" || name == "str",
            // An object column of text, which pandas infers.
            Dtype::Object => text,
            // pandas finds no `category` dtype a string dtype, whatever its
            // categories.
            Dtype::Numpy(_) | Dtype::Category => false,
        }
    }

    /// `pd.api.types.is_integer_dtype(series)`.
    pub fn is_integer_dtype(&self) -> bool {
        let ints = matches!(
            self.array.data_type(),
            DataType::Int8
                | DataType::Int16
                | DataType::Int32
                | DataType::Int64
                | DataType::UInt8
                | DataType::UInt16
                | DataType::UInt32
                | DataType::UInt64
        );
        match &self.dtype {
            Dtype::Object | Dtype::Category => false,
            _ => ints,
        }
    }
}

/// `array.take(positions)`.
pub fn take(array: &ArrayRef, positions: &[usize]) -> Result<ArrayRef, Raised> {
    let indices = UInt64Array::from_iter_values(positions.iter().map(|&p| p as u64));
    arrow_select::take::take(array.as_ref(), &indices, None)
        .map_err(|e| Raised::internal(format!("cannot take rows: {e}")))
}

/// `pd.concat` of columns of one dtype.
pub fn concat_series(parts: &[&Series]) -> Result<Series, Raised> {
    let Some(first) = parts.first() else {
        return Err(Raised::internal("no columns to concatenate"));
    };
    if parts.len() > 1
        && parts.iter().all(|series| series.dtype == Dtype::Object)
        && parts.iter().any(|series| series.original_values.is_some())
    {
        let cells = parts
            .iter()
            .flat_map(|series| {
                series.object_values().unwrap_or_else(|| {
                    (0..series.len())
                        .map(|position| {
                            sdm_schema::engine::result::value_at(&series.array, position)
                        })
                        .collect()
                })
            })
            .collect::<Vec<_>>();
        if decimal_lists::compatible(&cells) {
            return decimal_lists::series(&cells);
        }
    }
    // pandas keeps a dtype every part has; a part of no rows typed apart
    // (a frame a lookup of no ids gave) is left out, as it adds no values.
    let typed: Vec<&&Series> = parts
        .iter()
        .filter(|s| {
            !(s.is_empty()
                && (s.dtype != first.dtype || s.array.data_type() != first.array.data_type()))
        })
        .collect();
    let base = typed.first().map_or(first, |s| *s);
    if let Some(widened) = datetimes_at_one_unit(&typed) {
        let parts: Vec<&Series> = widened.iter().collect();
        return concat_series(&parts);
    }
    if typed
        .iter()
        .any(|s| s.dtype != base.dtype || s.array.data_type() != base.array.data_type())
    {
        return Err(Raised::internal(format!(
            "frames of different dtypes ({:?} and {:?}) are not concatenated here",
            base.dtype,
            typed.iter().map(|s| s.dtype.clone()).collect::<Vec<_>>()
        )));
    }
    if typed.len() == 1 {
        return Ok((*typed[0]).clone());
    }
    let arrays: Vec<&dyn Array> = typed.iter().map(|s| s.array.as_ref()).collect();
    let array = if arrays.is_empty() {
        new_empty_array(base.array.data_type())
    } else {
        arrow_select::concat::concat(&arrays)
            .map_err(|e| Raised::internal(format!("cannot concatenate: {e}")))?
    };
    Ok(Series {
        dtype: base.dtype.clone(),
        array,
        original_values: typed
            .iter()
            .any(|series| series.original_values.is_some())
            .then(|| {
                typed
                    .iter()
                    .flat_map(|series| {
                        series.original_values.clone().unwrap_or_else(|| {
                            (0..series.len())
                                .map(|position| {
                                    sdm_schema::engine::result::value_at(&series.array, position)
                                })
                                .collect()
                        })
                    })
                    .collect()
            }),
    })
}

/// Naive `datetime64` columns of more than one unit, each at the finest
/// of them, which is the unit `pd.concat` gives them all; `None` where the
/// parts are not that.
fn datetimes_at_one_unit(parts: &[&&Series]) -> Option<Vec<Series>> {
    use arrow_array::TimestampNanosecondArray;
    use arrow_schema::TimeUnit;
    let unit_of = |s: &Series| match s.array.data_type() {
        DataType::Timestamp(unit, None) if s.dtype.is_datetime() => Some(*unit),
        _ => None,
    };
    let units: Vec<TimeUnit> = parts.iter().map(|s| unit_of(s)).collect::<Option<_>>()?;
    if units.windows(2).all(|w| w[0] == w[1]) {
        return None;
    }
    let rank = |u: &TimeUnit| match u {
        TimeUnit::Second => 0,
        TimeUnit::Millisecond => 1,
        TimeUnit::Microsecond => 2,
        TimeUnit::Nanosecond => 3,
    };
    let finest = units.iter().max_by_key(|u| rank(u)).copied()?;
    let per = |u: &TimeUnit| -> i64 {
        match u {
            TimeUnit::Second => 1_000_000_000,
            TimeUnit::Millisecond => 1_000_000,
            TimeUnit::Microsecond => 1_000,
            TimeUnit::Nanosecond => 1,
        }
    };
    let name = match finest {
        TimeUnit::Second => "datetime64[s]",
        TimeUnit::Millisecond => "datetime64[ms]",
        TimeUnit::Microsecond => "datetime64[us]",
        TimeUnit::Nanosecond => "datetime64[ns]",
    };
    let mut out = Vec::new();
    for (part, unit) in parts.iter().zip(&units) {
        let factor = per(unit) / per(&finest);
        let values: Vec<Option<i64>> = stamps(part.array.as_ref())?
            .into_iter()
            .map(|v| v.and_then(|v| v.checked_mul(factor)))
            .collect();
        let nanos = TimestampNanosecondArray::from(values);
        let array: ArrayRef = match finest {
            TimeUnit::Nanosecond => Arc::new(nanos),
            TimeUnit::Microsecond => Arc::new(arrow_array::TimestampMicrosecondArray::from(
                nanos.iter().collect::<Vec<_>>(),
            )),
            TimeUnit::Millisecond => Arc::new(arrow_array::TimestampMillisecondArray::from(
                nanos.iter().collect::<Vec<_>>(),
            )),
            TimeUnit::Second => Arc::new(arrow_array::TimestampSecondArray::from(
                nanos.iter().collect::<Vec<_>>(),
            )),
        };
        out.push(Series::new(Dtype::Numpy(name.to_string()), array));
    }
    Some(out)
}

/// A timestamp array's values in its own unit, `None` where null.
fn stamps(array: &dyn Array) -> Option<Vec<Option<i64>>> {
    use arrow_array::types::{
        TimestampMicrosecondType, TimestampMillisecondType, TimestampNanosecondType,
        TimestampSecondType,
    };
    use arrow_schema::TimeUnit;
    macro_rules! read {
        ($t:ty) => {{
            let values = array.as_primitive::<$t>();
            Some(
                (0..array.len())
                    .map(|i| array.is_valid(i).then(|| values.value(i)))
                    .collect(),
            )
        }};
    }
    match array.data_type() {
        DataType::Timestamp(TimeUnit::Second, _) => read!(TimestampSecondType),
        DataType::Timestamp(TimeUnit::Millisecond, _) => read!(TimestampMillisecondType),
        DataType::Timestamp(TimeUnit::Microsecond, _) => read!(TimestampMicrosecondType),
        DataType::Timestamp(TimeUnit::Nanosecond, _) => read!(TimestampNanosecondType),
        _ => None,
    }
}

/// A table of rows: named columns of one length.
#[derive(Clone, Debug, Default)]
pub struct Frame {
    pub names: Vec<String>,
    pub columns: Vec<Series>,
    pub rows: usize,
}

impl Frame {
    pub fn new(names: Vec<String>, columns: Vec<Series>, rows: usize) -> Self {
        Self {
            names,
            columns,
            rows,
        }
    }

    pub fn len(&self) -> usize {
        self.rows
    }

    pub fn is_empty(&self) -> bool {
        self.rows == 0
    }

    pub fn position(&self, name: &str) -> Option<usize> {
        self.names.iter().position(|n| n == name)
    }

    /// `df[name]`, or the `KeyError` it raises.
    pub fn column(&self, name: &str) -> Result<&Series, Raised> {
        self.position(name)
            .map(|i| &self.columns[i])
            .ok_or_else(|| Raised::key_error(name))
    }

    /// `df[names]`, in the order given.
    pub fn select(&self, names: &[String]) -> Result<Frame, Raised> {
        let columns = names
            .iter()
            .map(|n| self.column(n).cloned())
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Frame::new(names.to_vec(), columns, self.rows))
    }

    /// `df[sorted(columns, key=df.columns.get_loc)]`: the columns in the
    /// frame's own order.
    pub fn select_in_order(&self, wanted: &[String]) -> Result<Frame, Raised> {
        for name in wanted {
            self.column(name)?;
        }
        let names: Vec<String> = self
            .names
            .iter()
            .filter(|n| wanted.contains(n))
            .cloned()
            .collect();
        self.select(&names)
    }

    /// `df.iloc[positions]`.
    pub fn take(&self, positions: &[usize]) -> Result<Frame, Raised> {
        let columns = self
            .columns
            .iter()
            .map(|c| c.take(positions))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Frame::new(self.names.clone(), columns, positions.len()))
    }

    /// `pd.concat(frames, ignore_index=True)` of frames of the same columns.
    pub fn concat(frames: &[&Frame]) -> Result<Frame, Raised> {
        let Some(first) = frames.first() else {
            return Err(Raised::internal("no frames to concatenate"));
        };
        if frames.len() == 1 {
            return Ok((*first).clone());
        }
        let mut columns = Vec::new();
        for (i, name) in first.names.iter().enumerate() {
            let parts = frames
                .iter()
                .map(|f| {
                    if f.names.get(i) == Some(name) {
                        Ok(&f.columns[i])
                    } else {
                        f.column(name)
                    }
                })
                .collect::<Result<Vec<_>, _>>()?;
            columns.push(concat_series(&parts)?);
        }
        let rows = frames.iter().map(|f| f.rows).sum();
        Ok(Frame::new(first.names.clone(), columns, rows))
    }

    /// The frame without the column `name`.
    pub fn without(&self, name: &str) -> Frame {
        let mut out = self.clone();
        if let Some(i) = out.position(name) {
            out.names.remove(i);
            out.columns.remove(i);
        }
        out
    }

    pub fn set(&mut self, name: &str, series: Series) {
        match self.position(name) {
            Some(i) => self.columns[i] = series,
            None => {
                self.names.push(name.to_string());
                self.columns.push(series);
            }
        }
    }
}

/// A dictionary array as the array of its values, each row its category
/// (null where the row is); `None` for an array that is no dictionary.
pub fn decoded(array: &dyn Array) -> Option<ArrayRef> {
    let dictionary = array.as_any_dictionary_opt()?;
    arrow_select::take::take(dictionary.values().as_ref(), dictionary.keys(), None).ok()
}

/// An integer array's values, each as `i64`, `None` where null; `None`
/// for an array of another type. A dictionary is read as its values.
pub fn ints(array: &dyn Array) -> Option<Vec<Option<i64>>> {
    if let Some(values) = decoded(array) {
        return ints(values.as_ref());
    }
    macro_rules! read {
        ($t:ty) => {{
            let values = array.as_primitive::<$t>();
            Some(
                (0..array.len())
                    .map(|i| array.is_valid(i).then(|| values.value(i) as i64))
                    .collect(),
            )
        }};
    }
    match array.data_type() {
        DataType::Int64 => read!(Int64Type),
        DataType::Int32 => read!(Int32Type),
        DataType::Int16 => read!(Int16Type),
        DataType::Int8 => read!(Int8Type),
        DataType::UInt64 => read!(UInt64Type),
        DataType::UInt32 => read!(UInt32Type),
        DataType::UInt16 => read!(UInt16Type),
        DataType::UInt8 => read!(UInt8Type),
        _ => None,
    }
}

/// A float array's values as `f64`, `None` where null. A dictionary is
/// read as its values.
pub fn floats(array: &dyn Array) -> Option<Vec<Option<f64>>> {
    if let Some(values) = decoded(array) {
        return floats(values.as_ref());
    }
    match array.data_type() {
        DataType::Float64 => {
            let values = array.as_primitive::<Float64Type>();
            Some(
                (0..array.len())
                    .map(|i| array.is_valid(i).then(|| values.value(i)))
                    .collect(),
            )
        }
        DataType::Float32 => {
            let values = array.as_primitive::<Float32Type>();
            Some(
                (0..array.len())
                    .map(|i| array.is_valid(i).then(|| f64::from(values.value(i))))
                    .collect(),
            )
        }
        _ => None,
    }
}

/// A text array's values, `None` where null. A dictionary is read as its
/// values.
pub fn texts(array: &dyn Array) -> Option<Vec<Option<String>>> {
    if let Some(values) = decoded(array) {
        return texts(values.as_ref());
    }
    let n = array.len();
    match array.data_type() {
        DataType::Utf8 => {
            let values = array.as_string::<i32>();
            Some(
                (0..n)
                    .map(|i| array.is_valid(i).then(|| values.value(i).to_string()))
                    .collect(),
            )
        }
        DataType::LargeUtf8 => {
            let values = array.as_string::<i64>();
            Some(
                (0..n)
                    .map(|i| array.is_valid(i).then(|| values.value(i).to_string()))
                    .collect(),
            )
        }
        DataType::Utf8View => {
            let values = array.as_string_view();
            Some(
                (0..n)
                    .map(|i| array.is_valid(i).then(|| values.value(i).to_string()))
                    .collect(),
            )
        }
        _ => None,
    }
}

/// An `int64` array.
pub fn int64_array(values: Vec<i64>) -> ArrayRef {
    Arc::new(arrow_array::Int64Array::from(values))
}
