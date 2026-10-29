// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Test-only typed access to core-owned Arrow schemas and plan witnesses.

use pyo3::prelude::*;
use pyo3::types::PyBytes;
use sdm_contract::data::AttachmentId;
use std::collections::BTreeMap;

#[pyfunction]
pub(crate) fn fetch_binding_witness<'py>(
    py: Python<'py>,
    request: &[u8],
    plan: &[u8],
    owners: BTreeMap<u32, Py<crate::ArrowStream>>,
) -> PyResult<Bound<'py, PyBytes>> {
    let schemas = owners
        .into_iter()
        .map(|(id, owner)| {
            (
                AttachmentId(id),
                std::sync::Arc::clone(&owner.borrow(py).schema),
            )
        })
        .collect();
    let bytes = crate::core_call(py, || {
        sdm_host::fetch_binding_bytes(request, plan, &schemas)
    })?
    .map_err(|error| {
        crate::core_error(py, sdm_host::write_error(&error, || Ok(AttachmentId(0))))
    })?;
    Ok(PyBytes::new(py, &bytes))
}
