// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Response field admission and typed values, without recorded output snapshots.

use sdm_wire::json;
use sdm_wire::response::PredictionResponse;
use serde_json::{Value, json};

fn envelope() -> Value {
    json!({"id":"response", "model":"kumo-relational", "metadata":{},
        "predictions":[{"id":"row", "row_index":0, "prediction":1.25}]})
}

fn read(body: Value) -> Result<PredictionResponse, sdm_wire::response::Refused> {
    PredictionResponse::read(&json::read(&serde_json::to_vec(&body).unwrap()).unwrap())
}

#[test]
fn typed_answers_preserve_values_and_wire_order() {
    let mut body = envelope();
    body["predictions"] = json!([
        {"id":"second", "row_index":1, "prediction":2.5, "scores":[0.25,0.75]},
        {"id":"first", "row_index":0.0, "prediction":1.25, "unused":true}
    ]);
    body["unused"] = json!([1, 2]);
    let got = read(body).unwrap();
    assert_eq!(got.id, "response");
    assert_eq!(got.model, "kumo-relational");
    assert_eq!(
        got.predictions
            .iter()
            .map(|v| v.row_index)
            .collect::<Vec<_>>(),
        [1, 0]
    );
    assert_eq!(got.predictions[0].id.as_deref(), Some("second"));
    assert_eq!(got.predictions[1].id.as_deref(), Some("first"));
    assert_eq!(got.predictions[0].scores, Some(vec![0.25, 0.75]));
    assert_eq!(
        got.predictions[1].prediction,
        Some(json::read(b"1.25").unwrap())
    );
    for empty in [json!([]), json!({})] {
        let mut body = envelope();
        body["predictions"] = empty;
        assert!(read(body).unwrap().predictions.is_empty());
    }
    for missing in [false, true] {
        let mut body = envelope();
        if missing {
            body["predictions"][0].as_object_mut().unwrap().remove("id");
        } else {
            body["predictions"][0]["id"] = Value::Null;
        }
        assert_eq!(read(body).unwrap().predictions[0].id, None);
    }
}

#[test]
fn malformed_fields_name_the_failed_contract() {
    assert_eq!(read(json!([])).unwrap_err().field, "response");
    for (field, invalid) in [
        ("id", json!(1)),
        ("model", json!(1)),
        ("predictions", json!("bad")),
        ("predictions", json!(1)),
        ("metadata", json!([])),
    ] {
        let mut body = envelope();
        body[field] = invalid;
        assert_eq!(read(body).unwrap_err().field, field);
    }
    for field in ["id", "model", "predictions", "metadata"] {
        let mut body = envelope();
        body.as_object_mut().unwrap().remove(field);
        let error = read(body).unwrap_err();
        assert_eq!(error.field, field);
        assert_eq!(error.reason, "missing");
    }
    for items in [json!({"x":1}), json!([1])] {
        let mut body = envelope();
        body["predictions"] = items;
        assert_eq!(read(body).unwrap_err().field, "prediction");
    }
    for (field, invalid) in [
        ("id", json!(1)),
        ("row_index", Value::Null),
        ("row_index", json!(0.25)),
        ("row_index", json!(true)),
        ("row_index", json!("0")),
        ("row_index", json!(1e100)),
        ("forecast_step", json!(0.25)),
    ] {
        let mut body = envelope();
        body["predictions"][0][field] = invalid;
        assert_eq!(read(body).unwrap_err().field, format!("prediction.{field}"));
    }
    let mut body = envelope();
    body["predictions"][0]
        .as_object_mut()
        .unwrap()
        .remove("row_index");
    let error = read(body).unwrap_err();
    assert_eq!(
        (error.field, error.reason),
        ("prediction.row_index", "missing")
    );
}
