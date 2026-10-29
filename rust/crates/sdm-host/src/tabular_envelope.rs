// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Codecs for typed complete widths and actual native column meanings.

use sdm_contract::{CoreError, Sentence};
use sdm_core::{TabularEnvelopeColumns, TabularEnvelopeDeclaration};

fn invalid() -> CoreError {
    CoreError::protocol(
        Sentence::InvalidMessage,
        "native tabular envelope data is not a closed typed product",
    )
}

pub fn columns(bytes: &[u8]) -> Result<TabularEnvelopeColumns, CoreError> {
    sdm_core::tabular_envelope_columns(serde_json::from_slice(bytes).map_err(|_| invalid())?)
}

pub fn declaration(
    columns: &TabularEnvelopeColumns,
    name: String,
    stype: &str,
) -> Result<TabularEnvelopeDeclaration, CoreError> {
    let stype =
        serde_json::from_value(serde_json::Value::String(stype.into())).map_err(|_| invalid())?;
    columns.declaration(name, stype)
}

pub fn finish(columns: &TabularEnvelopeColumns) -> Result<(Vec<u8>, Vec<u8>), CoreError> {
    let (value, notices) = columns.finish()?;
    Ok((
        serde_json::to_vec(&value).map_err(|_| invalid())?,
        crate::frame_constructors::notices_bytes(&notices)?,
    ))
}
