// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Sampling the context a prediction is made from: the building blocks
//! (the graph store, the neighbor sampler, text normalization) and the
//! driver that puts them together ([`driver`]), as runs of the core.

pub mod driver;
pub mod fixture;
pub mod frame;
pub mod graph_store;
pub mod keys;
mod lower_table;
pub mod mapper;
pub mod neighbor;
pub mod sequences;
pub mod text;

use serde_json::{Value, json};

/// The `sampling.*` operations of the parity fixtures
/// (`tests/rust_core/sampling/CONTRACT.md`, Sampler fixtures): the case's
/// input in, `{"output": ...}` or `{"error": {"code", "message"}}` out, as
/// JSON text. `Err` is for an unknown operation or input that isn't a JSON
/// object.
pub fn fixture_op(operation: &str, input_json: &str) -> Result<String, String> {
    let input: Value =
        serde_json::from_str(input_json).map_err(|e| format!("the input is not JSON: {e}"))?;
    let Value::Object(input) = input else {
        return Err("the input is not a JSON object".to_string());
    };
    match fixture::run(operation, &input) {
        None => Err(format!("no sampling fixture operation {operation:?}")),
        // The output is JSON text already; see `fixture::run`.
        Some(Ok(output)) => Ok(format!("{{\"output\":{output}}}")),
        Some(Err(error)) => {
            Ok(json!({"error": {"code": error.code, "message": error.message}}).to_string())
        }
    }
}
