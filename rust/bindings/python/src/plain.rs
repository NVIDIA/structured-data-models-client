// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Plain data between Python and the core
//! (`tests/rust_core/request/CONTRACT.md`, "Names"): `None`, `bool`, `int`,
//! `float`, `str`, `bytes`, pyarrow tables, batches and arrays, and lists,
//! tuples and `str`-keyed dicts of them. Anything else is refused with
//! `TypeError`.

use std::sync::Arc;

use arrow_array::{ArrayRef, RecordBatch, make_array};
use arrow_data::ArrayData;
use arrow_pyarrow::{FromPyArrow, ToPyArrow};
use arrow_schema::DataType;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple};
use sdm_core::plain::Plain;
use serde_json::Value;

fn pyarrow<'py>(py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyAny>> {
    py.import("pyarrow")?.getattr(name)
}

/// A Python value as plain data.
pub(crate) fn from_py(value: &Bound<'_, PyAny>) -> PyResult<Plain> {
    let py = value.py();
    if value.is_none() {
        return Ok(Plain::None);
    }
    if value.is_instance_of::<PyBool>() {
        return Ok(Plain::Bool(value.extract()?));
    }
    if value.is_instance_of::<PyInt>() {
        return Ok(match value.extract::<i64>() {
            Ok(i) => Plain::Int(i),
            Err(_) => Plain::BigInt(value.str()?.to_string()),
        });
    }
    if value.is_instance_of::<PyFloat>() {
        return Ok(Plain::Float(value.extract()?));
    }
    if value.is_instance_of::<PyString>() {
        return Ok(Plain::Str(value.extract()?));
    }
    if value.is_instance_of::<PyBytes>() {
        return Ok(Plain::Bytes(value.extract()?));
    }
    if value.is_instance_of::<PyList>() {
        return Ok(Plain::List(
            value
                .try_iter()?
                .map(|item| from_py(&item?))
                .collect::<PyResult<_>>()?,
        ));
    }
    if value.is_instance_of::<PyTuple>() {
        return Ok(Plain::Tuple(
            value
                .try_iter()?
                .map(|item| from_py(&item?))
                .collect::<PyResult<_>>()?,
        ));
    }
    if let Ok(dict) = value.cast::<PyDict>() {
        let mut items = Vec::with_capacity(dict.len());
        for (key, item) in dict.iter() {
            let key: String = key
                .extract()
                .map_err(|_| PyTypeError::new_err("plain data has only str keys"))?;
            items.push((key, from_py(&item)?));
        }
        return Ok(Plain::Dict(items));
    }
    if value.is_instance(&pyarrow(py, "Table")?)? {
        let (schema, batches) = crate::import_table(value)?;
        return Ok(Plain::Table(schema, batches));
    }
    if value.is_instance(&pyarrow(py, "RecordBatch")?)? {
        return Ok(Plain::Batch(RecordBatch::from_pyarrow_bound(value)?));
    }
    if value.is_instance(&pyarrow(py, "ChunkedArray")?)? {
        let data_type = DataType::from_pyarrow_bound(&value.getattr("type")?)?;
        let chunks = value
            .getattr("chunks")?
            .try_iter()?
            .map(|chunk| import_array(&chunk?))
            .collect::<PyResult<_>>()?;
        return Ok(Plain::Chunked(data_type, chunks));
    }
    if value.is_instance(&pyarrow(py, "Array")?)? {
        return Ok(Plain::Array(import_array(value)?));
    }
    Err(PyTypeError::new_err(format!(
        "{} is not plain data",
        value.get_type().name()?
    )))
}

fn import_array(array: &Bound<'_, PyAny>) -> PyResult<ArrayRef> {
    Ok(make_array(ArrayData::from_pyarrow_bound(array)?))
}

/// Plain data as a Python value.
pub(crate) fn to_py<'py>(py: Python<'py>, value: &Plain) -> PyResult<Bound<'py, PyAny>> {
    Ok(match value {
        Plain::None => py.None().into_bound(py),
        Plain::Bool(b) => PyBool::new(py, *b).to_owned().into_any(),
        Plain::Int(i) => i.into_pyobject(py)?.into_any(),
        Plain::BigInt(digits) => py
            .import("builtins")?
            .getattr("int")?
            .call1((digits.as_str(),))?,
        Plain::Float(f) => PyFloat::new(py, *f).into_any(),
        Plain::Str(s) => PyString::new(py, s).into_any(),
        Plain::Bytes(b) => PyBytes::new(py, b).into_any(),
        Plain::List(items) => PyList::new(
            py,
            items
                .iter()
                .map(|item| to_py(py, item))
                .collect::<PyResult<Vec<_>>>()?,
        )?
        .into_any(),
        Plain::Tuple(items) => PyTuple::new(
            py,
            items
                .iter()
                .map(|item| to_py(py, item))
                .collect::<PyResult<Vec<_>>>()?,
        )?
        .into_any(),
        Plain::Dict(items) => {
            let dict = PyDict::new(py);
            for (key, item) in items {
                dict.set_item(key, to_py(py, item)?)?;
            }
            dict.into_any()
        }
        Plain::Table(schema, batches) => {
            crate::export_table(py, Arc::clone(schema), batches.clone())?
        }
        Plain::Batch(batch) => batch.to_pyarrow(py)?,
        Plain::Array(array) => array.to_data().to_pyarrow(py)?,
        Plain::Chunked(data_type, chunks) => {
            let arrays = chunks
                .iter()
                .map(|chunk| chunk.to_data().to_pyarrow(py))
                .collect::<PyResult<Vec<_>>>()?;
            pyarrow(py, "chunked_array")?
                .call1((PyList::new(py, arrays)?, data_type.to_pyarrow(py)?))?
        }
    })
}

/// A JSON value as Python: objects as dicts, arrays as lists.
pub(crate) fn json_to_py<'py>(py: Python<'py>, value: &Value) -> PyResult<Bound<'py, PyAny>> {
    to_py(py, &Plain::from_json(value))
}
