// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! An item's `embeddings`, read as the client's bindings read it
//! (`_float_tuple`, T2.11): an array of finite numbers, a number spelled as
//! text among them; anything else refused naming the field.

use sdm_wire::json;
use sdm_wire::response::PredictionResponse;

fn item(embeddings: &str) -> Result<PredictionResponse, sdm_wire::response::Refused> {
    let body = format!(
        r#"{{"id": "r", "model": "kumo-relational", "metadata": {{}},
            "predictions": [{{"row_index": 0, "embeddings": {embeddings}}}]}}"#
    );
    PredictionResponse::read(&json::read(body.as_bytes()).unwrap())
}

#[test]
fn embeddings_are_read_in_the_order_written() {
    let read = item(r#"[0.5, -2, "3.25", 1e-3]"#).unwrap();
    let first = &read.predictions[0];
    assert_eq!(first.embeddings, Some(vec![0.5, -2.0, 3.25, 0.001]));
    assert!(first.carried.is_empty());
    assert_eq!(item("[]").unwrap().predictions[0].embeddings, Some(vec![]));
    assert_eq!(item("null").unwrap().predictions[0].embeddings, None);
}

#[test]
fn embeddings_not_an_array_of_finite_numbers_are_refused() {
    for (given, reason) in [
        (r#""12""#, "not_a_sequence"),
        (r#"{"a": 1}"#, "not_a_sequence"),
        ("3", "not_a_sequence"),
        ("[true]", "not_a_number"),
        (r#"["x"]"#, "not_a_number"),
        ("[[1]]", "not_a_number"),
        (r#"["inf"]"#, "not_finite"),
    ] {
        let refused = item(given).map(|_| ()).unwrap_err();
        assert_eq!(refused.field, "prediction.embeddings", "{given}");
        assert_eq!(refused.reason, reason, "{given}");
    }
}
