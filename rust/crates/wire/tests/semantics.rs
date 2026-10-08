// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What the generated types keep and what they refuse.

use sdm_wire::{
    ArraysTableData, PredictionRequest, TableCellValue, TableData, TableSchemaPrimaryKey, Value,
    parse, parse_prediction_request,
};
use serde_json::json;

fn request() -> Value {
    json!({
        "model": "kumo-tabular",
        "task": {"kind": "regression", "target": {"column_name": "y", "dtype": "float64"}},
        "schema": {"instance_table": {"columns": {
            "id": {"dtype": "int64", "stype": "ID"},
            "y": {"dtype": "float64"}
        }}},
        "context": {"instance_table": {
            "format": "arrays", "columns": ["id", "y"], "rows": [[1, 2.0], [2, null]]
        }},
        "predict": {"instance_table": {
            "format": "arrow_ipc", "transport": "base64", "payload_base64": "QUJD"
        }},
        "output": {"fields": ["prediction"]}
    })
}

fn roundtrip(payload: &Value) -> Result<Value, String> {
    let request = parse_prediction_request(&payload.to_string()).map_err(|e| e.to_string())?;
    serde_json::to_value(request).map_err(|e| e.to_string())
}

#[test]
fn values_keep_their_json_type_and_absent_keys_stay_absent() {
    let payload = request();
    assert_eq!(roundtrip(&payload), Ok(payload.clone()));
    let back = roundtrip(&payload).expect("round-trips");
    assert!(back["context"]["instance_table"]["rows"][0][0].is_u64());
    assert!(back["context"]["instance_table"]["rows"][0][1].is_f64());
    assert!(back["context"]["instance_table"]["rows"][1][1].is_null());
    assert!(back.get("inference").is_none());
}

#[test]
fn floats_come_back_with_the_same_bits() {
    let mut payload = request();
    let cells: [f64; 6] = [
        0.1,
        1e300,
        -0.0,
        5e-324,
        0.30000000000000004,
        1.7976931348623157e308,
    ];
    payload["context"]["instance_table"]["rows"] =
        json!(cells.iter().map(|c| json!([1, c])).collect::<Vec<_>>());
    let back = roundtrip(&payload).expect("round-trips");
    for (i, cell) in cells.iter().enumerate() {
        let got = back["context"]["instance_table"]["rows"][i][1].as_f64();
        assert_eq!(got.map(f64::to_bits), Some(cell.to_bits()));
    }
}

#[test]
fn open_objects_keep_keys_the_contract_does_not_name() {
    let mut payload = request();
    payload["task"]["extra"] = json!({"n": 1, "x": 1.5, "list": [1, "a", null]});
    payload["schema"]["instance_table"]["columns"]["id"]["extra"] = json!(true);
    payload["output"]["extra"] = json!(null);
    assert_eq!(roundtrip(&payload), Ok(payload));
}

type Change = fn(&mut Value);

#[test]
fn what_the_contract_does_not_allow_is_refused() {
    let cases: [(&str, Change); 9] = [
        ("unknown top-level key", |p| p["extra"] = json!(1)),
        ("model outside the enum", |p| p["model"] = json!("other")),
        ("missing task", |p| {
            p.as_object_mut().map(|o| o.remove("task"));
        }),
        ("unknown table format", |p| {
            p["context"]["instance_table"]["format"] = json!("csv")
        }),
        ("table format not a string", |p| {
            p["context"]["instance_table"]["format"] = json!(1)
        }),
        ("unknown key in arrays data", |p| {
            p["context"]["instance_table"]["x"] = json!(1)
        }),
        ("null where the contract has no null", |p| {
            p["metadata"] = json!(null)
        }),
        ("an integer where a string goes", |p| {
            p["task"]["target"]["dtype"] = json!(1)
        }),
        ("a float where an integer goes", |p| {
            p["task"]["top_k"] = json!(1.5)
        }),
    ];
    for (case, change) in cases {
        let mut payload = request();
        change(&mut payload);
        assert!(roundtrip(&payload).is_err(), "{case} was accepted");
    }
}

#[test]
fn unions_parse_into_their_variants() {
    let mut payload = request();
    payload["schema"]["instance_table"]["primary_key"] = json!("id");
    payload["context"]["instance_table"]["rows"] = json!([[1, ["a", "b"]], ["x", true]]);
    let request: PredictionRequest = serde_json::from_value(payload).expect("a prediction request");
    assert_eq!(
        request.schema.instance_table.primary_key,
        Some(TableSchemaPrimaryKey::String("id".to_owned()))
    );
    let TableData::ArraysTableData(ArraysTableData { rows, .. }) = request.context.instance_table
    else {
        panic!("the context is arrays data");
    };
    assert_eq!(
        rows[0][1],
        TableCellValue::Array(vec!["a".to_owned(), "b".to_owned()])
    );
    assert_eq!(rows[1][1], TableCellValue::Boolean(true));
    assert!(matches!(
        request.predict.instance_table,
        TableData::ArrowIpcBase64TableData(_)
    ));
}

#[test]
fn numbers_that_cannot_come_back_unchanged_are_refused() {
    let text = request().to_string();
    let widest = text.replacen(
        r#""kind":"regression""#,
        r#""kind":"regression","big":18446744073709551615,"small":-9223372036854775808"#,
        1,
    );
    let widest: Value = serde_json::from_str(&widest).expect("JSON");
    let back = roundtrip(&widest).expect("the widest integers are accepted");
    assert_eq!(back, widest);
    assert_eq!(back["task"]["big"].as_u64(), Some(u64::MAX));
    assert_eq!(back["task"]["small"].as_i64(), Some(i64::MIN));
    for number in ["18446744073709551616", "-9223372036854775809", "1e400"] {
        let payload = text.replacen(
            r#""kind":"regression""#,
            &format!(r#""kind":"regression","big":{number}"#),
            1,
        );
        assert_ne!(payload, text);
        assert!(
            parse_prediction_request(&payload).is_err(),
            "{number} was accepted"
        );
    }
}

#[test]
fn every_wire_type_parses_with_the_integer_check() {
    let data = r#"{"format":"arrays","columns":["id"],"rows":[[18446744073709551615]]}"#;
    assert!(parse::<TableData>(data).is_ok());
    let wider = data.replace("18446744073709551615", "18446744073709551616");
    assert!(parse::<TableData>(&wider).is_err());
    // Deserializing directly skips the check, as the crate documents.
    assert!(serde_json::from_str::<TableData>(&wider).is_ok());
}
