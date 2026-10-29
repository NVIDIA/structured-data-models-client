// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Fixed final schema-advice and sample population bindings.

use std::collections::BTreeMap;

use pyo3::prelude::*;
use pyo3::types::PyBytes;

fn checked<T>(py: Python<'_>, value: Result<T, sdm_contract::CoreError>) -> PyResult<T> {
    value.map_err(|error| {
        crate::core_error(
            py,
            sdm_host::write_error(&error, || Ok(sdm_contract::data::AttachmentId(0))),
        )
    })
}

fn notices(
    py: Python<'_>,
    value: Result<Vec<sdm_contract::CoreWarning>, sdm_contract::CoreError>,
) -> PyResult<Py<PyBytes>> {
    let value = checked(py, value)?;
    let bytes = checked(
        py,
        crate::core_call(py, || sdm_host::frame_constructors::notices_bytes(&value))?,
    )?;
    Ok(PyBytes::new(py, &bytes).unbind())
}

#[pyclass(module = "sdm_client._native")]
pub(crate) struct FrameKeyAdvice {
    state: sdm_host::FrameKeyAdvice,
}

#[pymethods]
impl FrameKeyAdvice {
    #[new]
    fn new(py: Python<'_>, plan: &[u8], dates: BTreeMap<String, String>) -> PyResult<Self> {
        let state = checked(
            py,
            crate::core_call(py, || sdm_host::frame_constructors::key_advice(plan, dates))?,
        )?;
        Ok(Self { state })
    }

    fn declined_tables(&self, py: Python<'_>) -> PyResult<Vec<String>> {
        crate::core_call(py, || self.state.declined_tables())
    }

    fn begin_declined(&mut self, py: Python<'_>, name: String, empty: bool) -> PyResult<()> {
        checked(
            py,
            crate::core_call(py, || self.state.begin_declined(name, empty))?,
        )
    }

    fn declined_column(&self, py: Python<'_>) -> PyResult<Option<String>> {
        crate::core_call(py, || self.state.declined_column())
    }

    fn declined_distinct(&mut self, py: Python<'_>, distinct: u64, rows: u64) -> PyResult<bool> {
        checked(
            py,
            crate::core_call(py, || self.state.declined_distinct(distinct, rows))?,
        )
    }

    fn declined_quality(&mut self, py: Python<'_>, quality: &[u8]) -> PyResult<bool> {
        checked(
            py,
            crate::core_call(py, || {
                self.state
                    .declined_quality(sdm_host::frame_constructors::key_quality(quality)?)
            })?,
        )
    }

    fn declined_notice(&mut self, py: Python<'_>, rows: u64) -> PyResult<Py<PyBytes>> {
        notices(
            py,
            crate::core_call(py, || self.state.declined_notice(rows))?,
        )
    }

    fn keyless_tables(&self, py: Python<'_>) -> PyResult<Vec<String>> {
        crate::core_call(py, || self.state.keyless_tables())
    }

    fn keyless_columns(
        &mut self,
        py: Python<'_>,
        name: String,
        empty: bool,
    ) -> PyResult<Vec<String>> {
        checked(
            py,
            crate::core_call(py, || self.state.keyless_columns(name, empty))?,
        )
    }

    fn keyless_missing(&self, py: Python<'_>, missing: bool) -> PyResult<bool> {
        crate::core_call(py, || self.state.keyless_missing(missing))
    }

    fn keyless_quality(
        &mut self,
        py: Python<'_>,
        table: String,
        column: String,
        quality: &[u8],
    ) -> PyResult<()> {
        checked(
            py,
            crate::core_call(py, || {
                self.state.keyless_quality(
                    table,
                    column,
                    sdm_host::frame_constructors::key_quality(quality)?,
                )
            })?,
        )
    }

    fn keyless_notice(&mut self, py: Python<'_>, table: String) -> PyResult<Py<PyBytes>> {
        notices(
            py,
            crate::core_call(py, || self.state.keyless_notice(table))?,
        )
    }

    fn ambiguous_notices(&self, py: Python<'_>) -> PyResult<Py<PyBytes>> {
        notices(
            py,
            crate::core_call(py, || Ok(self.state.ambiguous_notices()))?,
        )
    }
}

#[pyfunction]
pub(crate) fn frame_sample_facts(py: Python<'_>, sample: &[u8]) -> PyResult<Py<PyBytes>> {
    let bytes = checked(
        py,
        crate::core_call(py, || sdm_host::frame_constructors::native_sample(sample))?,
    )?;
    Ok(PyBytes::new(py, &bytes).unbind())
}
