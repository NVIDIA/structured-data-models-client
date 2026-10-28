// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Fixed object-Decimal constructor source observations, without callbacks.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use sdm_host::{FrameObjectContent, FrameObjectRead};

#[pyclass(module = "sdm_client._native")]
pub(crate) struct FrameKeyCandidates {
    candidates: Option<sdm_host::FrameKeyCandidates>,
}

#[pymethods]
impl FrameKeyCandidates {
    #[new]
    fn new(py: Python<'_>, meanings: &[u8], rows: u64) -> PyResult<Self> {
        let candidates = crate::core_call(py, || {
            sdm_host::frame_constructors::key_candidates(meanings, rows)
        })?
        .map_err(|_| {
            PyValueError::new_err("native key candidates need completed column meanings")
        })?;
        Ok(Self {
            candidates: Some(candidates),
        })
    }

    fn column(&self, py: Python<'_>) -> PyResult<Option<String>> {
        let held = self
            .candidates
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("native key populations have been consumed"))?;
        crate::core_call(py, || held.column().map(str::to_owned))
    }

    fn observe(&mut self, py: Python<'_>, distinct: u64, rows: u64) -> PyResult<()> {
        let held = self
            .candidates
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("native key populations have been consumed"))?;
        crate::core_call(py, || held.observe(distinct, rows))?.map_err(|_| {
            PyValueError::new_err("native key cardinality is outside its selected population")
        })
    }

    fn selected(&self, py: Python<'_>) -> PyResult<Option<String>> {
        let held = self
            .candidates
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("native key populations have been consumed"))?;
        crate::core_call(py, || held.selected().map(|name| name.map(str::to_owned)))?
            .map_err(|_| PyValueError::new_err("native key candidates have not been completed"))
    }

    fn finish(
        &mut self,
        py: Python<'_>,
        missing: Option<(bool, u64)>,
    ) -> PyResult<(Py<PyBytes>, Py<PyBytes>)> {
        let held = self
            .candidates
            .take()
            .ok_or_else(|| PyValueError::new_err("native key populations have been consumed"))?;
        let (nomination, notices) = crate::core_call(py, || {
            sdm_host::frame_constructors::finish_key_candidates(held, missing)
        })?
        .map_err(|_| {
            PyValueError::new_err("native nominee facts do not match its completed population")
        })?;
        Ok((
            PyBytes::new(py, &nomination).unbind(),
            PyBytes::new(py, &notices).unbind(),
        ))
    }
}

#[pyclass(module = "sdm_client._native")]
pub(crate) struct FrameObjectConstructor {
    inference: sdm_host::FrameObjectConstructor,
}

#[pymethods]
impl FrameObjectConstructor {
    #[new]
    fn new(rows: usize) -> Self {
        Self {
            inference: sdm_host::FrameObjectConstructor::new(rows),
        }
    }

    fn content(&mut self, py: Python<'_>, content: &str) -> PyResult<()> {
        let content = match content {
            "decimal" => FrameObjectContent::Decimal,
            "empty" => FrameObjectContent::Empty,
            "integer" => FrameObjectContent::Integer,
            "floating" | "mixed-integer-float" => FrameObjectContent::Floating,
            "boolean" => FrameObjectContent::Boolean,
            "date" | "datetime" | "datetime64" => FrameObjectContent::Date,
            _ => FrameObjectContent::Other,
        };
        crate::core_call(py, || self.inference.content(content))?
            .map_err(|_| PyValueError::new_err("a native Decimal content read is out of order"))
    }

    fn next_read(&self, py: Python<'_>) -> PyResult<(&'static str, usize)> {
        crate::core_call(py, || match self.inference.next() {
            FrameObjectRead::Content => ("content", 0),
            FrameObjectRead::Missing(row) => ("missing", row),
            FrameObjectRead::IsDecimal(row) => ("is_decimal", row),
            FrameObjectRead::WithinIntegerRange(row) => ("within_integer_range", row),
            FrameObjectRead::Finite(row) => ("finite", row),
            FrameObjectRead::NegativeExponent(row) => ("negative_exponent", row),
            FrameObjectRead::Fractional(row) => ("fractional", row),
            FrameObjectRead::Wide(row) => ("wide", row),
            FrameObjectRead::AtLeastMinimum(row) => ("at_least_minimum", row),
            FrameObjectRead::AtMostMaximum(row) => ("at_most_maximum", row),
            FrameObjectRead::Distinct => ("distinct", 0),
            FrameObjectRead::Complete => ("complete", 0),
        })
    }

    fn observe(&mut self, py: Python<'_>, value: bool) -> PyResult<()> {
        crate::core_call(py, || self.inference.observe(value))?
            .map_err(|_| PyValueError::new_err("a native Decimal fact arrived outside its read"))
    }

    fn distinct(&mut self, py: Python<'_>, distinct: u64) -> PyResult<()> {
        crate::core_call(py, || self.inference.distinct(distinct))?.map_err(|_| {
            PyValueError::new_err("native Decimal cardinality arrived outside its read")
        })
    }

    fn meaning(&self, py: Python<'_>) -> PyResult<Py<PyBytes>> {
        let meaning = crate::core_call(py, || self.inference.meaning())?
            .ok_or_else(|| PyValueError::new_err("native scalar inference has not finished"))?;
        let bytes = crate::core_call(py, || {
            sdm_host::frame_constructors::object_meaning_bytes(meaning)
        })?
        .map_err(|_| PyValueError::new_err("native scalar meaning serialization failed"))?;
        Ok(PyBytes::new(py, &bytes).unbind())
    }
}

fn native_checked<T>(py: Python<'_>, value: Result<T, sdm_contract::CoreError>) -> PyResult<T> {
    value.map_err(|error| {
        crate::core_error(
            py,
            sdm_host::write_error(&error, || Ok(sdm_contract::data::AttachmentId(0))),
        )
    })
}

/// Fixed aggregate samples only; this owner never stores a native callback.
#[pyclass(module = "sdm_client._native")]
pub(crate) struct FrameNativeConstructor {
    meaning: sdm_host::FrameNativeConstructor,
}

#[pymethods]
impl FrameNativeConstructor {
    #[new]
    fn new(
        py: Python<'_>,
        table: String,
        name: String,
        values: &[u8],
        declared: &[u8],
    ) -> PyResult<Self> {
        Ok(Self {
            meaning: native_checked(
                py,
                crate::core_call(py, || {
                    sdm_host::frame_constructors::native_column(table, name, values, declared)
                })?,
            )?,
        })
    }
    fn dtype(&self, py: Python<'_>) -> PyResult<Py<PyBytes>> {
        let bytes = native_checked(
            py,
            crate::core_call(py, || {
                sdm_host::frame_constructors::native_column_dtype_bytes(&self.meaning)
            })?,
        )?;
        Ok(PyBytes::new(py, &bytes).unbind())
    }
    fn timestamp_sample(&self, py: Python<'_>) -> PyResult<bool> {
        crate::core_call(py, || self.meaning.timestamp_sample())
    }
    fn separated_sample(&self, py: Python<'_>) -> PyResult<bool> {
        crate::core_call(py, || self.meaning.separated_sample())
    }
    fn element_sample(&self, py: Python<'_>) -> PyResult<bool> {
        crate::core_call(py, || self.meaning.element_sample())
    }
    fn element_count(&mut self, py: Python<'_>, parts: u64) -> PyResult<()> {
        native_checked(
            py,
            crate::core_call(py, || self.meaning.element_count(parts))?,
        )
    }
    fn distinct_sample(&self, py: Python<'_>) -> PyResult<bool> {
        crate::core_call(py, || self.meaning.distinct_sample())
    }
    fn timestamp_counts(&mut self, py: Python<'_>, parsed: u64, present: u64) -> PyResult<()> {
        native_checked(
            py,
            crate::core_call(py, || self.meaning.timestamp_counts(parsed, present))?,
        )
    }
    fn separator(&mut self, py: Python<'_>, points: Vec<u32>) -> PyResult<Option<String>> {
        Ok(native_checked(
            py,
            crate::core_call(py, || self.meaning.separator(&points))?,
        )?
        .map(|c| c.to_string()))
    }
    fn separated_counts(&mut self, py: Python<'_>, whole: u64, parts: u64) -> PyResult<()> {
        native_checked(
            py,
            crate::core_call(py, || self.meaning.separated_counts(whole, parts))?,
        )
    }
    fn distinct_count(&mut self, py: Python<'_>, distinct: u64) -> PyResult<bool> {
        native_checked(
            py,
            crate::core_call(py, || self.meaning.distinct_count(distinct))?,
        )
    }
    fn population_count(&mut self, py: Python<'_>, present: u64) -> PyResult<()> {
        native_checked(
            py,
            crate::core_call(py, || self.meaning.population_count(present))?,
        )
    }
    fn meaning(&self, py: Python<'_>) -> PyResult<Py<PyBytes>> {
        let bytes = native_checked(
            py,
            crate::core_call(py, || {
                sdm_host::frame_constructors::native_column_bytes(&self.meaning)
            })?,
        )?;
        Ok(PyBytes::new(py, &bytes).unbind())
    }
}
