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
    fn reduced(&self, py: Python<'_>) -> PyResult<bool> {
        crate::core_call(py, || self.state.reduced())
    }
    #[pyo3(signature = (labels=None))]
    fn positions(&self, py: Python<'_>, labels: Option<Vec<i64>>) -> PyResult<Option<Vec<usize>>> {
        checked(
            py,
            crate::core_call(py, || self.state.positions(labels.as_deref()))?,
        )
    }
}

#[pyclass(module = "sdm_client._native")]
pub(crate) struct TabularScoringKeys {
    state: sdm_host::TabularScoringKeys,
}

#[pymethods]
impl TabularScoringKeys {
    #[new]
    fn new(py: Python<'_>, keys: Vec<i64>) -> PyResult<Self> {
        Ok(Self {
            state: crate::core_call(py, || sdm_host::TabularScoringKeys::new(keys))?,
        })
    }
    fn needs_source(&self, py: Python<'_>) -> PyResult<bool> {
        crate::core_call(py, || self.state.needs_source())
    }
    fn source(&self, py: Python<'_>, keys: Vec<i64>) -> PyResult<TabularPopulation> {
        let state = checked(py, crate::core_call(py, || self.state.source(&keys))?)?;
        Ok(TabularPopulation { state })
    }
}

#[pyclass(module = "sdm_client._native")]
pub(crate) struct TabularPopulation {
    state: sdm_host::TabularPopulation,
}

#[pymethods]
impl TabularPopulation {
    fn positions(&self, py: Python<'_>) -> PyResult<Vec<usize>> {
        crate::core_call(py, || self.state.positions())
    }
    fn labelled_rows(&self, py: Python<'_>, known: Vec<bool>) -> PyResult<Vec<usize>> {
        checked(
            py,
            crate::core_call(py, || self.state.labelled_rows(&known))?,
        )
    }
    #[pyo3(signature = (candidates, requested=None))]
    fn context(
        &self,
        py: Python<'_>,
        candidates: Vec<usize>,
        requested: Option<usize>,
    ) -> PyResult<TabularContext> {
        let state = checked(
            py,
            crate::core_call(py, || self.state.context(&candidates, requested))?,
        )?;
        Ok(TabularContext { state })
    }
}
