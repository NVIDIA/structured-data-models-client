// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The request fixture operations (`tests/rust_core/request/CONTRACT.md`,
//! "Fixtures"): a case's operation on its input, answered as JSON text
//! `{"output", "warnings"}` or `{"error", "warnings"}`.

use serde_json::{Map, Value};

pub use sdm_request::fixture::InputError;

/// `operation` on `input_json`, a fixture case's input.
pub fn fixture_op(operation: &str, input_json: &str) -> Result<String, InputError> {
    let input: Value = serde_json::from_str(input_json)
        .map_err(|e| InputError(format!("the input is not JSON: {e}")))?;
    let empty = Map::new();
    let input = input.as_object().unwrap_or(&empty);
    if let Some(answered) = sdm_request::fixture::fixture_op(operation, input) {
        return answered;
    }
    match operation {
        "client.run" => return crate::flow::client_run(input),
        "tabular.task" => return crate::flow::tabular_task(input),
        "decompose.parts" => return crate::flow::decompose_parts(input),
        "payload.feature_column" => return crate::flow::payload_feature_column(input),
        _ => {}
    }
    Err(InputError(format!(
        "no request fixture operation {operation:?}"
    )))
}
