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

#[pyfunction]
pub(crate) fn tabular_task_vocabulary(
    py: Python<'_>,
) -> PyResult<(Vec<String>, Vec<String>, String)> {
    let words = crate::core_call(py, sdm_host::tabular_task_vocabulary)?;
    Ok((words.stypes, words.tasks, words.categorical))
}

#[pyclass(module = "sdm_client._native")]
pub(crate) struct TabularTaskMeaning {
    state: sdm_host::TabularTaskMeaning,
}

#[pymethods]
impl TabularTaskMeaning {
    #[new]
    fn new(py: Python<'_>, target_supported: bool) -> PyResult<Self> {
        Ok(Self {
            state: checked(
                py,
                crate::core_call(py, || sdm_host::TabularTaskMeaning::new(target_supported))?,
            )?,
        })
    }
    #[pyo3(signature = (present, string=None, member=None))]
    fn stated(
        &self,
        py: Python<'_>,
        present: bool,
        string: Option<bool>,
        member: Option<bool>,
    ) -> PyResult<Self> {
        Ok(Self {
            state: checked(
                py,
                crate::core_call(py, || self.state.stated(present, string, member))?,
            )?,
        })
    }
    fn inferred(&self, py: Python<'_>, categorical: bool) -> PyResult<Self> {
        Ok(Self {
            state: checked(
                py,
                crate::core_call(py, || self.state.inferred(categorical))?,
            )?,
        })
    }
    fn original(&self, py: Python<'_>) -> PyResult<bool> {
        crate::core_call(py, || self.state.original())
    }
    fn kind(&self, py: Python<'_>) -> PyResult<Option<String>> {
        crate::core_call(py, || self.state.kind().map(str::to_owned))
    }
    fn numbers(&self, py: Python<'_>, cardinality: usize) -> PyResult<TabularTaskNumbers> {
        Ok(TabularTaskNumbers {
            state: checked(
                py,
                crate::core_call(py, || self.state.numbers(cardinality))?,
            )?,
        })
    }
}

#[pyclass(module = "sdm_client._native")]
pub(crate) struct TabularTaskNumbers {
    state: sdm_host::TabularTaskNumbers,
}

impl TabularTaskNumbers {
    fn number(
        &self,
        py: Python<'_>,
        ordinal: usize,
        value: sdm_host::TabularTaskNumber,
    ) -> PyResult<Self> {
        Ok(Self {
            state: checked(
                py,
                crate::core_call(py, || self.state.number(ordinal, value))?,
            )?,
        })
    }
}

#[pymethods]
impl TabularTaskNumbers {
    fn boolean(&self, py: Python<'_>, ordinal: usize) -> PyResult<Self> {
        self.number(py, ordinal, sdm_host::TabularTaskNumber::Boolean)
    }
    fn integral(&self, py: Python<'_>, ordinal: usize) -> PyResult<Self> {
        self.number(py, ordinal, sdm_host::TabularTaskNumber::Integral)
    }
    #[pyo3(signature = (ordinal, finite, equal=None, returned=None))]
    fn decimal(
        &self,
        py: Python<'_>,
        ordinal: usize,
        finite: bool,
        equal: Option<bool>,
        returned: Option<bool>,
    ) -> PyResult<Self> {
        self.number(
            py,
            ordinal,
            sdm_host::TabularTaskNumber::Decimal {
                finite,
                equal,
                returned,
            },
        )
    }
    fn real(&self, py: Python<'_>, ordinal: usize, value: f64) -> PyResult<Self> {
        self.number(
            py,
            ordinal,
            sdm_host::TabularTaskNumber::RealValue {
                bits: value.to_le_bytes(),
            },
        )
    }
    fn real_failure(
        &self,
        py: Python<'_>,
        ordinal: usize,
        overflow_error: bool,
        type_error: bool,
        value_error: bool,
    ) -> PyResult<Self> {
        self.number(
            py,
            ordinal,
            sdm_host::TabularTaskNumber::RealFailure {
                overflow_error,
                type_error,
                value_error,
            },
        )
    }
    fn other(&self, py: Python<'_>, ordinal: usize) -> PyResult<Self> {
        self.number(py, ordinal, sdm_host::TabularTaskNumber::Other)
    }
    fn exhausted(&self, py: Python<'_>, ordinal: usize) -> PyResult<Self> {
        Ok(Self {
            state: checked(py, crate::core_call(py, || self.state.exhausted(ordinal))?)?,
        })
    }
    fn kind(&self, py: Python<'_>) -> PyResult<Option<String>> {
        crate::core_call(py, || self.state.kind().map(str::to_owned))
    }
    fn propagates(&self, py: Python<'_>) -> PyResult<bool> {
        crate::core_call(py, || self.state.propagates())
    }
    fn notices(&self, py: Python<'_>) -> PyResult<Py<pyo3::types::PyBytes>> {
        let notices = crate::core_call(py, || self.state.notice().into_iter().collect::<Vec<_>>())?;
        let bytes = checked(
            py,
            crate::core_call(py, || sdm_host::frame_constructors::notices_bytes(&notices))?,
        )?;
        Ok(pyo3::types::PyBytes::new(py, &bytes).unbind())
    }
}
