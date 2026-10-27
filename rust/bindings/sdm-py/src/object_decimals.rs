// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Fixed object-Decimal constructor source observations, without callbacks.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use sdm_host::{ObjectDecimalContent, ObjectDecimalRead};

#[pyclass(module = "sdm_client._native")]
pub(crate) struct ObjectDecimalConstructor {
    inference: sdm_host::ObjectDecimalConstructor,
}

#[pymethods]
impl ObjectDecimalConstructor {
    #[new]
    fn new(rows: usize) -> Self {
        Self {
            inference: sdm_host::ObjectDecimalConstructor::new(rows),
        }
    }

    fn content(&mut self, py: Python<'_>, content: &str) -> PyResult<()> {
        let content = match content {
            "decimal" => ObjectDecimalContent::Decimal,
            "empty" => ObjectDecimalContent::Empty,
            _ => ObjectDecimalContent::Other,
        };
        crate::core_call(py, || self.inference.content(content))?
            .map_err(|_| PyValueError::new_err("a native Decimal content read is out of order"))
    }

    fn next_read(&self, py: Python<'_>) -> PyResult<(&'static str, usize)> {
        crate::core_call(py, || match self.inference.next() {
            ObjectDecimalRead::Content => ("content", 0),
            ObjectDecimalRead::Missing(row) => ("missing", row),
            ObjectDecimalRead::IsDecimal(row) => ("is_decimal", row),
            ObjectDecimalRead::Finite(row) => ("finite", row),
            ObjectDecimalRead::NegativeExponent(row) => ("negative_exponent", row),
            ObjectDecimalRead::Fractional(row) => ("fractional", row),
            ObjectDecimalRead::Wide(row) => ("wide", row),
            ObjectDecimalRead::AtLeastMinimum(row) => ("at_least_minimum", row),
            ObjectDecimalRead::AtMostMaximum(row) => ("at_most_maximum", row),
            ObjectDecimalRead::Complete => ("complete", 0),
        })
    }

    fn observe(&mut self, py: Python<'_>, value: bool) -> PyResult<()> {
        crate::core_call(py, || self.inference.observe(value))?
            .map_err(|_| PyValueError::new_err("a native Decimal fact arrived outside its read"))
    }

    fn is_decimal(&self, py: Python<'_>) -> PyResult<bool> {
        crate::core_call(py, || self.inference.is_decimal())
    }

    fn meaning(&self, py: Python<'_>) -> PyResult<(&'static str, bool)> {
        crate::core_call(py, || self.inference.meaning())?
            .map(|meaning| (meaning.dtype.wire_name(), meaning.untyped))
            .ok_or_else(|| PyValueError::new_err("native Decimal inference has not finished"))
    }
}
