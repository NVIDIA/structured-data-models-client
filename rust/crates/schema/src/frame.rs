// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A table of rows as the schema core reads it: each column a pandas dtype
//! and its cells, each cell the value `series.iloc[i]` gives in Python, so
//! that counting, comparing and writing a cell in a message come out as
//! pandas has them.
//!
//! The host hands each table over as Arrow arrays, with a tag per column
//! naming its pandas dtype; an `object` column, which has no Arrow type,
//! arrives as text, one JSON-encoded cell per row, with the content
//! `pandas.api.types.infer_dtype` reports (DV19).

use arrow_array::cast::AsArray;
use arrow_array::types::{
    Decimal32Type, Decimal64Type, Decimal128Type, Decimal256Type, Float16Type, Float32Type,
    Float64Type, Int8Type, Int16Type, Int32Type, Int64Type, UInt8Type, UInt16Type, UInt32Type,
    UInt64Type,
};
use arrow_array::{Array, ArrayRef, RecordBatch};
use arrow_schema::{DataType, TimeUnit};
use serde_json::Value;

use crate::dt::civil::Wall;
use crate::pyval::PyVal;

/// A column's pandas dtype, as the schema core needs to tell them apart.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PdType {
    /// `bool` or `boolean`.
    Bool,
    /// A signed or unsigned integer of `bits`; `name` as pandas spells it.
    Int { name: String, unsigned: bool },
    /// A float of `bits`.
    Float { name: String, bits: u32 },
    /// `datetime64[unit]`, maybe zoned.
    Datetime,
    /// An Arrow decimal dtype, retaining its width, precision and scale.
    Decimal {
        name: String,
        precision: u8,
        scale: i8,
    },
    /// `str`, `string` or `category`: text, without a numeric type.
    Text { name: String },
    /// `object`, with `infer_dtype`'s content label.
    Object { content: String },
    /// Anything else, by name: durations, dates, lists, binary.
    Other {
        name: String,
        scalar_only: bool,
        infer_dtype_error: Option<String>,
    },
}

impl PdType {
    pub fn name(&self) -> String {
        match self {
            PdType::Bool => "bool".to_string(),
            PdType::Int { name, .. } => name.clone(),
            PdType::Float { name, .. } => name.clone(),
            PdType::Datetime => "datetime64".to_string(),
            PdType::Decimal { name, .. } => name.clone(),
            PdType::Text { name } => name.clone(),
            PdType::Object { .. } => "object".to_string(),
            PdType::Other { name, .. } => name.clone(),
        }
    }
}

/// One column.
#[derive(Clone, Debug)]
pub struct Column {
    pub name: String,
    pub dtype: PdType,
    pub cells: Vec<PyVal>,
    /// `repr` of the name, where the frame's name was not text.
    pub name_repr: Option<String>,
}

/// A table of rows.
#[derive(Clone, Debug, Default)]
pub struct Frame {
    pub columns: Vec<Column>,
    pub rows: usize,
}

/// Whether `pd.isna` calls a cell missing.
pub fn missing(cell: &PyVal) -> bool {
    match cell {
        PyVal::Decimal(text) => {
            let t = text.trim_start_matches(['+', '-']).to_ascii_lowercase();
            t == "nan" || t == "snan"
        }
        other => other.is_missing(),
    }
}

impl Column {
    /// `series.isna().any()`.
    pub fn any_missing(&self) -> bool {
        self.cells.iter().any(missing)
    }

    /// `int(series.isna().sum())`.
    pub fn count_missing(&self) -> usize {
        self.cells.iter().filter(|c| missing(c)).count()
    }

    /// `series.count()`.
    pub fn count(&self) -> usize {
        self.cells.len() - self.count_missing()
    }

    /// `series.nunique(dropna=True)`.
    pub fn nunique(&self) -> crate::client::R<usize> {
        if let Some(message) = unique_kernel_refusal(&self.dtype.name()) {
            return Err(crate::client::Refusal::class(
                "ArrowNotImplementedError",
                message,
            ));
        }
        let mut seen = std::collections::HashSet::new();
        for cell in self.cells.iter().filter(|c| !missing(c)) {
            seen.insert(cell.eq_key());
        }
        Ok(seen.len())
    }
}

pub fn unique_kernel_refusal(dtype_name: &str) -> Option<String> {
    let arrow_type = dtype_name.strip_suffix("[pyarrow]")?;
    if arrow_type.starts_with("decimal32(")
        || arrow_type.starts_with("decimal64(")
        || arrow_type.starts_with("extension<")
        || arrow_type.starts_with("list<")
        || arrow_type.starts_with("large_list<")
        || arrow_type.starts_with("fixed_size_list<")
        || arrow_type.starts_with("map<")
        || arrow_type.starts_with("struct<")
        || arrow_type == "halffloat"
    {
        Some(format!(
            "Function 'unique' has no kernel matching input types ({arrow_type})"
        ))
    } else {
        None
    }
}

impl Frame {
    pub fn column(&self, name: &str) -> Option<&Column> {
        self.columns.iter().find(|c| c.name == name)
    }
}

fn wall_from(value: i64, unit: &TimeUnit) -> Wall {
    let nanos: i128 = match unit {
        TimeUnit::Second => i128::from(value) * 1_000_000_000,
        TimeUnit::Millisecond => i128::from(value) * 1_000_000,
        TimeUnit::Microsecond => i128::from(value) * 1000,
        TimeUnit::Nanosecond => i128::from(value),
    };
    Wall::from_epoch_nanos(nanos)
}

/// The text of a `decimal128` value at `scale`, as `str(Decimal)`.
pub fn decimal_text(value: i128, scale: i8) -> String {
    decimal_digits_text(&value.to_string(), scale)
}

/// Signed decimal digits at an Arrow scale, including Decimal256 values.
pub fn decimal_digits_text(value: &str, scale: i8) -> String {
    let negative = value.starts_with('-');
    let digits = value.trim_start_matches('-').to_string();
    let scale = i32::from(scale);
    let body = if scale <= 0 {
        if digits == "0" {
            if scale == 0 {
                "0".to_string()
            } else {
                format!("0E+{}", -scale)
            }
        } else if scale == 0 {
            digits
        } else {
            // A negative scale: digits times ten to the -scale.
            let exp = digits.len() as i32 - 1 - scale;
            let mut m = digits.clone();
            if m.len() > 1 {
                m.insert(1, '.');
            }
            format!("{m}E+{exp}")
        }
    } else if digits.len() as i32 - 1 - scale < -6 {
        // `str(Decimal)` writes a value this small in scientific notation,
        // `Decimal('1E-7')`, `Decimal('0E-7')`.
        let exp = digits.len() as i32 - 1 - scale;
        let mut m = digits.clone();
        if m.len() > 1 {
            m.insert(1, '.');
        }
        format!("{m}E{exp}")
    } else {
        let s = scale as usize;
        let padded = if digits.len() <= s {
            format!("{}{}", "0".repeat(s - digits.len() + 1), digits)
        } else {
            digits
        };
        let (int, frac) = padded.split_at(padded.len() - s);
        format!("{int}.{frac}")
    };
    if negative { format!("-{body}") } else { body }
}

fn numpy(name: &str, inner: PyVal) -> PyVal {
    PyVal::Numpy(name.to_string(), Box::new(inner))
}

/// The cells of `array`, as `iloc` gives them for a column of `dtype`.
fn cells_of(array: &ArrayRef, dtype: &PdType) -> Result<Vec<PyVal>, String> {
    let n = array.len();
    let mut out = Vec::with_capacity(n);
    let null = |dtype: &PdType| match dtype {
        PdType::Float { .. } | PdType::Text { .. } => PyVal::Float(f64::NAN),
        PdType::Datetime => PyVal::NaT,
        _ => PyVal::Na,
    };
    macro_rules! ints {
        ($t:ty, $name:expr) => {{
            let a = array.as_primitive::<$t>();
            for i in 0..n {
                out.push(if a.is_null(i) {
                    null(dtype)
                } else {
                    numpy($name, PyVal::Int(i128::from(a.value(i))))
                });
            }
        }};
    }
    macro_rules! decimals {
        ($decimal_type:ty, $scale:expr) => {{
            let values = array.as_primitive::<$decimal_type>();
            for index in 0..n {
                out.push(if values.is_null(index) {
                    PyVal::None
                } else {
                    PyVal::Decimal(decimal_digits_text(
                        &values.value(index).to_string(),
                        $scale,
                    ))
                });
            }
        }};
    }
    match array.data_type() {
        DataType::Boolean => {
            let a = array.as_boolean();
            for i in 0..n {
                out.push(if a.is_null(i) {
                    null(dtype)
                } else {
                    numpy("bool", PyVal::Bool(a.value(i)))
                });
            }
        }
        DataType::Int8 => ints!(Int8Type, "int8"),
        DataType::Int16 => ints!(Int16Type, "int16"),
        DataType::Int32 => ints!(Int32Type, "int32"),
        DataType::Int64 => ints!(Int64Type, "int64"),
        DataType::UInt8 => ints!(UInt8Type, "uint8"),
        DataType::UInt16 => ints!(UInt16Type, "uint16"),
        DataType::UInt32 => ints!(UInt32Type, "uint32"),
        DataType::UInt64 => ints!(UInt64Type, "uint64"),
        DataType::Float16 => {
            let a = array.as_primitive::<Float16Type>();
            for i in 0..n {
                out.push(if a.is_null(i) {
                    null(dtype)
                } else {
                    numpy("float16", PyVal::Float(f64::from(a.value(i).to_f32())))
                });
            }
        }
        DataType::Float32 => {
            let a = array.as_primitive::<Float32Type>();
            for i in 0..n {
                out.push(if a.is_null(i) {
                    null(dtype)
                } else {
                    numpy("float32", PyVal::Float(f64::from(a.value(i))))
                });
            }
        }
        DataType::Float64 => {
            let a = array.as_primitive::<Float64Type>();
            for i in 0..n {
                out.push(if a.is_null(i) {
                    null(dtype)
                } else {
                    numpy("float64", PyVal::Float(a.value(i)))
                });
            }
        }
        DataType::Timestamp(unit, _) => {
            let a = array.as_any();
            let get = |i: usize| -> Option<i64> {
                use arrow_array::types::*;
                match unit {
                    TimeUnit::Second => {
                        let a =
                            a.downcast_ref::<arrow_array::PrimitiveArray<TimestampSecondType>>()?;
                        (!a.is_null(i)).then(|| a.value(i))
                    }
                    TimeUnit::Millisecond => {
                        let a = a
                            .downcast_ref::<arrow_array::PrimitiveArray<TimestampMillisecondType>>(
                            )?;
                        (!a.is_null(i)).then(|| a.value(i))
                    }
                    TimeUnit::Microsecond => {
                        let a = a
                            .downcast_ref::<arrow_array::PrimitiveArray<TimestampMicrosecondType>>(
                            )?;
                        (!a.is_null(i)).then(|| a.value(i))
                    }
                    TimeUnit::Nanosecond => {
                        let a = a
                            .downcast_ref::<arrow_array::PrimitiveArray<TimestampNanosecondType>>(
                            )?;
                        (!a.is_null(i)).then(|| a.value(i))
                    }
                }
            };
            for i in 0..n {
                out.push(match get(i) {
                    Some(v) => PyVal::Timestamp(wall_from(v, unit)),
                    None => PyVal::NaT,
                });
            }
        }
        // `pd.ArrowDtype` dates, which `iloc` gives as `datetime.date`.
        DataType::Date32 | DataType::Date64 => {
            for i in 0..n {
                out.push(if array.is_null(i) {
                    null(dtype)
                } else {
                    crate::engine::result::value_at(array, i)
                });
            }
        }
        DataType::Decimal32(_, scale) => decimals!(Decimal32Type, *scale),
        DataType::Decimal64(_, scale) => decimals!(Decimal64Type, *scale),
        DataType::Decimal128(_, scale) => decimals!(Decimal128Type, *scale),
        DataType::Decimal256(_, scale) => decimals!(Decimal256Type, *scale),
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
            if matches!(
                dtype,
                PdType::Object { .. }
                    | PdType::Other {
                        scalar_only: false,
                        ..
                    }
            ) {
                for text in texts {
                    let Some(text) = text else {
                        out.push(PyVal::None);
                        continue;
                    };
                    let value: Value = serde_json::from_str(&text)
                        .map_err(|e| format!("an object cell that is not JSON: {e}"))?;
                    out.push(PyVal::decode(&value)?);
                }
            } else {
                for text in texts {
                    out.push(match text {
                        Some(t) => PyVal::Str(t),
                        None => null(dtype),
                    });
                }
            }
        }
        DataType::Dictionary(_, _) => {
            let values = arrow_cast_dictionary(array)?;
            out = values;
        }
        DataType::Null => {
            for _ in 0..n {
                out.push(PyVal::None);
            }
        }
        other => {
            let opaque = |kind: &str, type_name: &str| PyVal::Opaque {
                kind: kind.to_string(),
                type_name: type_name.to_string(),
                repr: String::new(),
            };
            for i in 0..n {
                out.push(if array.is_null(i) {
                    PyVal::None
                } else {
                    match other {
                        DataType::List(_)
                        | DataType::LargeList(_)
                        | DataType::FixedSizeList(..)
                        | DataType::ListView(_)
                        | DataType::Map(..) => opaque("container", "list"),
                        DataType::Struct(_) => opaque("mapping", "dict"),
                        DataType::Binary
                        | DataType::LargeBinary
                        | DataType::FixedSizeBinary(_)
                        | DataType::BinaryView => PyVal::Bytes(Vec::new()),
                        _ => opaque("object", &other.to_string()),
                    }
                });
            }
        }
    }
    Ok(out)
}

fn arrow_cast_dictionary(array: &ArrayRef) -> Result<Vec<PyVal>, String> {
    let dict = array.as_any_dictionary();
    let values = dict.values();
    let keys = dict.normalized_keys();
    let value_cells = cells_of(
        values,
        &PdType::Text {
            name: "str".to_string(),
        },
    )?;
    Ok((0..array.len())
        .map(|i| {
            if array.is_null(i) {
                PyVal::Float(f64::NAN)
            } else {
                value_cells.get(keys[i]).cloned().unwrap_or(PyVal::None)
            }
        })
        .collect())
}

/// The pandas dtype a tag names, with the Arrow type it arrived as.
fn dtype_of(tag: &Value, data_type: &DataType) -> Result<PdType, String> {
    let name = tag
        .get("dtype")
        .and_then(Value::as_str)
        .ok_or("a column's tag names its dtype")?
        .to_string();
    if tag.get("cells").and_then(Value::as_str) == Some("json") && name != "object" {
        return Ok(PdType::Other {
            name,
            scalar_only: false,
            infer_dtype_error: tag
                .get("infer_dtype_error")
                .map(|value| {
                    value
                        .as_str()
                        .map(str::to_string)
                        .ok_or("infer_dtype_error must be text")
                })
                .transpose()?,
        });
    }
    if name == "object" {
        let content = tag
            .get("content")
            .and_then(Value::as_str)
            .ok_or("an object column's tag names its content")?;
        return Ok(PdType::Object {
            content: content.to_string(),
        });
    }
    Ok(match data_type {
        DataType::Boolean => PdType::Bool,
        DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64 => PdType::Int {
            name,
            unsigned: false,
        },
        DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64 => PdType::Int {
            name,
            unsigned: true,
        },
        DataType::Float16 => PdType::Float {
            name: if name == "halffloat[pyarrow]" {
                name
            } else {
                "float16".into()
            },
            bits: 16,
        },
        DataType::Float32 => PdType::Float {
            name: "float32".into(),
            bits: 32,
        },
        DataType::Float64 => PdType::Float {
            name: "float64".into(),
            bits: 64,
        },
        // `pd.api.types.is_datetime64_any_dtype` holds for an Arrow date.
        DataType::Timestamp(..) | DataType::Date32 | DataType::Date64 => PdType::Datetime,
        DataType::Decimal32(p, s)
        | DataType::Decimal64(p, s)
        | DataType::Decimal128(p, s)
        | DataType::Decimal256(p, s)
            if name.ends_with("[pyarrow]") =>
        {
            PdType::Decimal {
                name,
                precision: *p,
                scale: *s,
            }
        }
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View | DataType::Dictionary(..) => {
            PdType::Text { name }
        }
        other => PdType::Other {
            name,
            infer_dtype_error: None,
            scalar_only: !matches!(
                other,
                DataType::List(_)
                    | DataType::LargeList(_)
                    | DataType::FixedSizeList(..)
                    | DataType::Binary
                    | DataType::LargeBinary
                    | DataType::FixedSizeBinary(_)
                    | DataType::BinaryView
                    | DataType::Struct(_)
                    | DataType::Map(..)
            ),
        },
    })
}

/// A frame from the host's Arrow batches and its columns' tags.
pub fn from_arrow(
    batches: &[RecordBatch],
    schema: &arrow_schema::Schema,
    tags: Option<&Value>,
) -> Result<Frame, String> {
    let mut columns = Vec::new();
    let mut rows = 0;
    for batch in batches {
        rows += batch.num_rows();
    }
    for (i, field) in schema.fields().iter().enumerate() {
        let tag = tags
            .and_then(|t| t.get(field.name()))
            .cloned()
            .unwrap_or(Value::Null);
        let dtype = if tag.is_null() {
            dtype_of(
                &serde_json::json!({"dtype": field.data_type().to_string()}),
                field.data_type(),
            )?
        } else {
            dtype_of(&tag, field.data_type())?
        };
        let mut cells = Vec::with_capacity(rows);
        for batch in batches {
            cells.extend(cells_of(batch.column(i), &dtype)?);
        }
        let name_repr = tag
            .get("name_repr")
            .and_then(Value::as_str)
            .map(str::to_string);
        columns.push(Column {
            name: field.name().clone(),
            dtype,
            cells,
            name_repr,
        });
    }
    Ok(Frame { columns, rows })
}

#[cfg(test)]
mod tests {
    use super::decimal_text;

    #[test]
    fn unsupported_arrow_unique_kernels_preserve_the_physical_type() {
        for name in [
            "decimal32(8, 2)[pyarrow]",
            "decimal64(16, -2)[pyarrow]",
            "extension<arrow.bool8>[pyarrow]",
            "extension<arrow.json>[pyarrow]",
            "extension<arrow.uuid>[pyarrow]",
            "list<item: int64>[pyarrow]",
            "large_list<item: int64>[pyarrow]",
            "fixed_size_list<item: int64>[2][pyarrow]",
            "list<item: list<item: int64>>[pyarrow]",
            "map<string, int64>[pyarrow]",
            "map<string, int64, keys_sorted>[pyarrow]",
            "struct<number: int64, label: string>[pyarrow]",
            "halffloat[pyarrow]",
        ] {
            let arrow_type = name.strip_suffix("[pyarrow]").unwrap();
            assert_eq!(
                super::unique_kernel_refusal(name).unwrap(),
                format!("Function 'unique' has no kernel matching input types ({arrow_type})")
            );
        }
        for name in [
            "object",
            "decimal32(8, 2)",
            "decimal128(38, 2)[pyarrow]",
            "int64[pyarrow]",
            "float16",
            "float32[pyarrow]",
        ] {
            assert!(super::unique_kernel_refusal(name).is_none());
        }
    }

    #[test]
    fn extension_inference_facts_do_not_replace_unique_refusals() {
        use std::sync::Arc;

        for name in ["bool8", "json", "uuid"] {
            let physical = format!("extension<arrow.{name}>");
            let array: super::ArrayRef =
                Arc::new(arrow_array::StringArray::from(vec!["true", "null"]));
            let schema = Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new(
                "value",
                arrow_schema::DataType::Utf8,
                true,
            )]));
            let batch = arrow_array::RecordBatch::try_new(schema.clone(), vec![array]).unwrap();
            for error in [None, Some(physical.clone())] {
                let mut tag = serde_json::json!({"value": {"dtype": format!("{physical}[pyarrow]"), "cells": "json"}});
                if let Some(message) = &error {
                    tag["value"]["infer_dtype_error"] = serde_json::json!(message);
                }
                let frame =
                    super::from_arrow(std::slice::from_ref(&batch), &schema, Some(&tag)).unwrap();
                let column = &frame.columns[0];
                let inferred = crate::client::infer_column(column).unwrap_err();
                assert_eq!(
                    inferred.raised_as(),
                    if error.is_some() {
                        "builtins.NotImplementedError"
                    } else {
                        "pyarrow.lib.ArrowNotImplementedError"
                    }
                );
                if let Some(message) = error {
                    assert_eq!(inferred.message, message);
                }
                let unique = column.nunique().unwrap_err();
                assert_eq!(unique.raised_as(), "pyarrow.lib.ArrowNotImplementedError");
                assert_eq!(
                    unique.message,
                    format!("Function 'unique' has no kernel matching input types ({physical})")
                );
                tag["value"]["infer_dtype_error"] = serde_json::json!(17);
                assert!(
                    super::from_arrow(std::slice::from_ref(&batch), &schema, Some(&tag)).is_err()
                );
            }
        }
    }

    #[test]
    fn half_float_unique_preserves_arrow_and_numpy_dtype_identity() {
        use std::sync::Arc;

        let array: super::ArrayRef = Arc::new(arrow_array::Float16Array::from_iter_values([
            0_u8.into(),
            1_u8.into(),
        ]));
        let schema = Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new(
            "item_id",
            arrow_schema::DataType::Float16,
            true,
        )]));
        let batch = arrow_array::RecordBatch::try_new(schema.clone(), vec![array]).unwrap();
        for name in ["halffloat[pyarrow]", "float16"] {
            let tag = serde_json::json!({"item_id": {"dtype": name}});
            let frame =
                super::from_arrow(std::slice::from_ref(&batch), &schema, Some(&tag)).unwrap();
            let column = &frame.columns[0];
            assert_eq!(column.dtype.name(), name);
            let inferred = crate::client::infer_column(column).unwrap();
            assert_eq!(inferred.dtype, "float32");
            let key = crate::client::infer_primary_key(&frame, &[inferred]);
            if name == "float16" {
                assert_eq!(key.unwrap(), Some("item_id".into()));
            } else {
                assert_eq!(
                    key.unwrap_err().raised_as(),
                    "pyarrow.lib.ArrowNotImplementedError"
                );
            }
        }
    }

    #[test]
    fn ordinary_float_names_keep_the_existing_normalization() {
        for (data_type, names, expected) in [
            (
                arrow_schema::DataType::Float32,
                ["float32", "Float32", "float[pyarrow]"],
                "float32",
            ),
            (
                arrow_schema::DataType::Float64,
                ["float64", "Float64", "double[pyarrow]"],
                "float64",
            ),
        ] {
            for name in names {
                let dtype =
                    super::dtype_of(&serde_json::json!({"dtype": name}), &data_type).unwrap();
                assert_eq!(dtype.name(), expected);
            }
        }
    }

    #[test]
    fn fixed_binary_cells_are_not_scalar_contract_values() {
        use std::sync::Arc;

        let array: super::ArrayRef = Arc::new(
            arrow_array::FixedSizeBinaryArray::try_from_sparse_iter_with_size(
                [Some(b"one".as_slice()), None].into_iter(),
                3,
            )
            .unwrap(),
        );
        let dtype = super::dtype_of(
            &serde_json::json!({"dtype": "fixed_size_binary[3][pyarrow]"}),
            array.data_type(),
        )
        .unwrap();
        let column = super::Column {
            name: "value".into(),
            dtype,
            cells: super::cells_of(
                &array,
                &super::PdType::Other {
                    name: "fixed_size_binary[3][pyarrow]".into(),
                    scalar_only: false,
                    infer_dtype_error: None,
                },
            )
            .unwrap(),
            name_repr: None,
        };
        let refusal = crate::client::require_scalar_cells(&column, "items.value").unwrap_err();
        assert_eq!(refusal.code, "UNSUPPORTED_REQUEST");
        assert!(
            refusal
                .message
                .starts_with("items.value holds raw bytes (bytes)")
        );
    }

    #[test]
    fn arrow_decimal_widths_preserve_cells_and_schema_inference() {
        use std::sync::Arc;

        use arrow_array::{Decimal32Array, Decimal64Array, Decimal128Array, Decimal256Array};

        for scale in [-2, 0, 2] {
            let arrays: Vec<super::ArrayRef> = vec![
                Arc::new(
                    Decimal32Array::from(vec![Some(125), None, Some(-5)])
                        .with_precision_and_scale(8, scale)
                        .unwrap(),
                ),
                Arc::new(
                    Decimal64Array::from(vec![Some(125), None, Some(-5)])
                        .with_precision_and_scale(16, scale)
                        .unwrap(),
                ),
                Arc::new(
                    Decimal128Array::from(vec![Some(125), None, Some(-5)])
                        .with_precision_and_scale(20, scale)
                        .unwrap(),
                ),
                Arc::new(
                    Decimal256Array::from(vec![
                        Some("125".parse().unwrap()),
                        None,
                        Some("-5".parse().unwrap()),
                    ])
                    .with_precision_and_scale(40, scale)
                    .unwrap(),
                ),
            ];
            for (width, array) in [32, 64, 128, 256].into_iter().zip(arrays) {
                let precision = match width {
                    32 => 8,
                    64 => 16,
                    128 => 20,
                    _ => 40,
                };
                let name = format!("decimal{width}({precision}, {scale})[pyarrow]");
                let schema = Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new(
                    "key",
                    array.data_type().clone(),
                    true,
                )]));
                let cells = [
                    super::PyVal::Decimal(decimal_text(125, scale)),
                    super::PyVal::None,
                    super::PyVal::Decimal(decimal_text(-5, scale)),
                ];
                for (offset, length) in [(0, 0), (1, 1), (0, 3)] {
                    let batch = super::RecordBatch::try_new(
                        schema.clone(),
                        vec![array.slice(offset, length)],
                    )
                    .unwrap();
                    let frame = super::from_arrow(
                        &[batch],
                        &schema,
                        Some(&serde_json::json!({"key": {"dtype": name}})),
                    )
                    .unwrap();
                    let column = &frame.columns[0];
                    assert_eq!(column.dtype.name(), name);
                    assert_eq!(column.cells, cells[offset..offset + length]);
                    assert_eq!(
                        crate::client::infer_tfm_dtype(column),
                        if scale > 0 { "float64" } else { "int64" }
                    );
                }
            }
        }
    }

    #[test]
    fn arrow_decimal256_keeps_values_beyond_i128() {
        use std::sync::Arc;

        let coefficient = "170141183460469231731687303715884105729";
        for (scale, positive, inferred) in [
            (0, "170141183460469231731687303715884105729", "string"),
            (2, "1701411834604692317316873037158841057.29", "float64"),
        ] {
            let array: super::ArrayRef = Arc::new(
                arrow_array::Decimal256Array::from(vec![
                    Some(coefficient.parse().unwrap()),
                    Some(format!("-{coefficient}").parse().unwrap()),
                ])
                .with_precision_and_scale(40, scale)
                .unwrap(),
            );
            let schema = Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new(
                "amount",
                array.data_type().clone(),
                false,
            )]));
            let batch = super::RecordBatch::try_new(schema.clone(), vec![array]).unwrap();
            let frame = super::from_arrow(&[batch], &schema, Some(&serde_json::json!({"amount": {"dtype": format!("decimal256(40, {scale})[pyarrow]")}}))).unwrap();
            assert_eq!(
                frame.columns[0].cells,
                vec![
                    super::PyVal::Decimal(positive.to_string()),
                    super::PyVal::Decimal(format!("-{positive}"))
                ]
            );
            assert_eq!(crate::client::infer_tfm_dtype(&frame.columns[0]), inferred);
        }
    }

    #[test]
    fn decimals_read_as_str_of_decimal_writes_them() {
        // `str(x)` of each `decimal128` value `pa.array(...).to_pylist()`
        // gives.
        for (value, scale, want) in [
            (0, 0, "0"),
            (1250, 2, "12.50"),
            (-5, 1, "-0.5"),
            (12345, 7, "0.0012345"),
            (1, 6, "0.000001"),
            (1, 7, "1E-7"),
            (0, 7, "0E-7"),
            (-15, 8, "-1.5E-7"),
            (5, -2, "5E+2"),
            (0, -2, "0E+2"),
        ] {
            assert_eq!(decimal_text(value, scale), want, "{value} at {scale}");
        }
    }
}
