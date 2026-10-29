// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Typed row-fact admission; all selection decisions run in the shared core.

use pyo3::prelude::*;

fn checked<T>(py: Python<'_>, value: Result<T, sdm_contract::CoreError>) -> PyResult<T> {
    value.map_err(|error| {
        crate::core_error(
            py,
            sdm_host::write_error(&error, || Ok(sdm_contract::data::AttachmentId(0))),
        )
    })
}

#[pyfunction]
pub(crate) fn tabular_indices(
    py: Python<'_>,
    keys: Vec<i64>,
    requested: Vec<Option<i64>>,
) -> PyResult<(Vec<usize>, Vec<usize>)> {
    let selected = crate::core_call(py, || {
        sdm_host::TabularIndexSelection::new(&keys, &requested)
    })?;
    Ok((selected.scored, selected.missing))
}

#[pyfunction]
pub(crate) fn tabular_matched_rows(
    py: Python<'_>,
    keys: Vec<i64>,
    requested: Vec<i64>,
) -> PyResult<Vec<usize>> {
    crate::core_call(py, || sdm_host::tabular_matched_rows(&keys, &requested))
}

#[pyfunction]
pub(crate) fn tabular_labelled_rows(
    py: Python<'_>,
    known: Vec<bool>,
    excluded: Vec<usize>,
) -> PyResult<Vec<usize>> {
    checked(
        py,
        crate::core_call(py, || sdm_host::tabular_labelled_rows(&known, &excluded))?,
    )
}

#[pyclass(module = "sdm_client._native")]
pub(crate) struct TabularContext {
    state: sdm_host::TabularContext,
}

#[pymethods]
impl TabularContext {
    #[new]
    #[pyo3(signature = (candidates, requested=None))]
    fn new(py: Python<'_>, candidates: Vec<usize>, requested: Option<usize>) -> PyResult<Self> {
        let state = checked(
            py,
            crate::core_call(py, || sdm_host::TabularContext::new(candidates, requested))?,
        )?;
        Ok(Self { state })
    }

    fn notice_rows(&self, py: Python<'_>) -> PyResult<Option<usize>> {
        crate::core_call(py, || self.state.notice_rows())
    }
    fn wanted(&self, py: Python<'_>) -> PyResult<usize> {
        crate::core_call(py, || self.state.wanted())
    }
    fn positions(&self, py: Python<'_>) -> PyResult<Option<Vec<usize>>> {
        crate::core_call(py, || self.state.positions())
    }
}
