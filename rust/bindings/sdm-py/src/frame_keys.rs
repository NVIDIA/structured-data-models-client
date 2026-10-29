// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Fixed native key validation/adoption products, without native callbacks.

use pyo3::exceptions::PyValueError;
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

#[pyfunction]
pub(crate) fn gathered_entity_text(
    py: Python<'_>, source: &[u8], observed: &[u8], dtype: Vec<u32>,
) -> PyResult<Option<Vec<u32>>> {
    checked(py, crate::core_call(py, || {
        sdm_host::gathered_entity_bytes(source, observed, &dtype)
    })?)
}

#[pyfunction]
pub(crate) fn check_frame_cell(
    py: Python<'_>,
    described: &str,
    kind: &str,
    native_type: &str,
) -> PyResult<()> {
    checked(
        py,
        crate::core_call(py, || {
            sdm_host::check_frame_cell(described, kind, native_type)
        })?,
    )
}

#[pyclass(module = "sdm_client._native")]
pub(crate) struct FrameKeyAdoption {
    state: Option<sdm_host::FrameKeyAdoption>,
}

impl FrameKeyAdoption {
    fn held(&mut self) -> PyResult<&mut sdm_host::FrameKeyAdoption> {
        self.state
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("native key adoption has been consumed"))
    }
}

#[pymethods]
impl FrameKeyAdoption {
    #[new]
    fn new(py: Python<'_>, product: &[u8]) -> PyResult<Self> {
        let state = checked(
            py,
            crate::core_call(py, || sdm_host::frame_constructors::key_adoption(product))?,
        )?;
        Ok(Self { state: Some(state) })
    }

    fn validation_columns(&mut self, py: Python<'_>) -> PyResult<Option<Vec<String>>> {
        let state = self.held()?;
        crate::core_call(py, || state.validation_columns())
    }

    fn validation_empty(&mut self, py: Python<'_>, empty: bool) -> PyResult<()> {
        let state = self.held()?;
        crate::core_call(py, || state.validation_empty(empty))
    }

    fn validation_quality(&mut self, py: Python<'_>, quality: &[u8]) -> PyResult<()> {
        let state = self.held()?;
        checked(
            py,
            crate::core_call(py, || {
                state.validation_quality(sdm_host::frame_constructors::key_quality(quality)?)
            })?,
        )
    }

    fn missing_columns(&mut self, py: Python<'_>) -> PyResult<Vec<String>> {
        let state = self.held()?;
        crate::core_call(py, || state.missing_columns())
    }

    fn validation_needs_rows(&mut self, py: Python<'_>) -> PyResult<bool> {
        let state = self.held()?;
        crate::core_call(py, || state.validation_needs_rows())
    }

    fn validate(
        &mut self,
        py: Python<'_>,
        missing: Vec<String>,
        rows: Option<u64>,
    ) -> PyResult<()> {
        let state = self.held()?;
        checked(py, crate::core_call(py, || state.validate(missing, rows))?)
    }

    fn adoption_columns(&mut self, py: Python<'_>) -> PyResult<Option<Vec<String>>> {
        let state = self.held()?;
        checked(py, crate::core_call(py, || state.adoption_columns())?)
    }

    fn adopt(&mut self, py: Python<'_>, quality: Option<&[u8]>) -> PyResult<()> {
        let state = self.held()?;
        checked(
            py,
            crate::core_call(py, || {
                state.adopt(
                    quality
                        .map(sdm_host::frame_constructors::key_quality)
                        .transpose()?,
                )
            })?,
        )
    }

    fn discriminators(&mut self, py: Python<'_>) -> PyResult<Vec<String>> {
        let state = self.held()?;
        checked(py, crate::core_call(py, || state.discriminators())?)
    }

    fn discriminator(&mut self, py: Python<'_>, name: String, quality: &[u8]) -> PyResult<()> {
        let state = self.held()?;
        checked(
            py,
            crate::core_call(py, || {
                state.discriminator(name, sdm_host::frame_constructors::key_quality(quality)?)
            })?,
        )
    }

    fn usable_columns(&mut self, py: Python<'_>) -> PyResult<Vec<String>> {
        let state = self.held()?;
        checked(py, crate::core_call(py, || state.usable_columns())?)
    }

    fn usable(&mut self, py: Python<'_>, name: String, missing: Option<bool>) -> PyResult<()> {
        let state = self.held()?;
        checked(py, crate::core_call(py, || state.usable(name, missing))?)
    }

    fn needs_rows(&mut self, py: Python<'_>) -> PyResult<bool> {
        let state = self.held()?;
        crate::core_call(py, || state.needs_rows())
    }

    fn finish(
        &mut self,
        py: Python<'_>,
        rows: Option<u64>,
    ) -> PyResult<(Py<PyBytes>, Py<PyBytes>)> {
        let state = self
            .state
            .take()
            .ok_or_else(|| PyValueError::new_err("native key adoption has been consumed"))?;
        let (product, notices) = checked(
            py,
            crate::core_call(py, || {
                sdm_host::frame_constructors::finish_key_adoption(state, rows)
            })?,
        )?;
        Ok((
            PyBytes::new(py, &product).unbind(),
            PyBytes::new(py, &notices).unbind(),
        ))
    }
}
