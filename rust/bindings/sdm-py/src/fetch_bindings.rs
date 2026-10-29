// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Test-only typed access to core-owned Arrow schemas and plan witnesses.

use pyo3::prelude::*;
use pyo3::types::PyBytes;
use sdm_contract::data::{AttachmentId, DataRequest, FetchPlan};
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
    let value = crate::core_call(py, || {
        let request: DataRequest = serde_json::from_slice(request).map_err(|_| {
            sdm_contract::CoreError::protocol(
                sdm_contract::Sentence::InvalidMessage,
                "invalid fetch witness request",
            )
        })?;
        let plan: FetchPlan = serde_json::from_slice(plan).map_err(|_| {
            sdm_contract::CoreError::protocol(
                sdm_contract::Sentence::InvalidMessage,
                "invalid fetch witness plan",
            )
        })?;
        Ok::<_, sdm_contract::CoreError>(serde_json::json!({
            "owners":sdm_host::fetch_binding_schemas(&plan,&schemas)?,
            "witness":sdm_host::fetch_plan_witness(&request,&plan)?,
        }))
    })?
    .map_err(|error| {
        crate::core_error(py, sdm_host::write_error(&error, || Ok(AttachmentId(0))))
    })?;
    let bytes = serde_json::to_vec(&value)
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("fetch witness does not encode"))?;
    Ok(PyBytes::new(py, &bytes))
}
