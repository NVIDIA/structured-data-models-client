// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native presentation of the actual core-owned logical data.
//! No model schema, normalization, graph selection or JSON-body read lives
//! here. Exact builtin types construct only primitive native values.

use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString};
use sdm_host::LogicalValue;

pub(super) fn present(py: Python<'_>, value: &LogicalValue) -> PyResult<Py<PyAny>> {
    Ok(match value {
        LogicalValue::Null => py.None(),
        LogicalValue::Bool(value) => PyBool::new(py, *value).to_owned().into_any().unbind(),
        LogicalValue::Number(token) => {
            // This is the JSON number's native representation, preserving
            // integer precision and a float token such as 1.0 as a float.
            // Exact builtin classes cannot dispatch a caller constructor.
            let digits = token.strip_prefix('-').unwrap_or(token);
            let kind = if !digits.is_empty() && digits.bytes().all(|byte| byte.is_ascii_digit()) {
                py.get_type::<PyInt>()
            } else {
                py.get_type::<PyFloat>()
            };
            kind.call1((token,))?.unbind()
        }
        LogicalValue::Text(text) => PyString::new(py, text).into_any().unbind(),
        LogicalValue::Array(items) => {
            let result = PyList::empty(py);
            for item in items {
                result.append(present(py, item)?)?;
            }
            result.into_any().unbind()
        }
        LogicalValue::Object(members) => {
            let result = PyDict::new(py);
            for (name, value) in members {
                result.set_item(name, present(py, value)?)?;
            }
            result.into_any().unbind()
        }
    })
}
