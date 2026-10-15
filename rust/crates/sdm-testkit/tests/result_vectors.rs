// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The canonical form of a result (`sdm_testkit::values`) against
//! `contract/result-vectors.json`, which the Python runner's
//! `result_values` is held to as well (`contract/tests/test_result_vectors.py`):
//! each vector's statement, run on this host's DuckDB, gives the vector's
//! Arrow schema as pyarrow spells it, its cells, and its digests in the
//! order read and sorted.

use std::path::PathBuf;

use serde_json::Value;

use sdm_testkit::duck;
use sdm_testkit::values::{arrow_type, ascii_json, values};

fn vectors() -> Vec<Value> {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../contract/result-vectors.json");
    let file: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    assert_eq!(file["format"], "sdm-result-vectors/1");
    let vectors = file["vectors"].as_array().unwrap().clone();
    // Not vacuous: every kind the canonical form names, an empty result,
    // nanosecond times, an interval and a zoned timestamp.
    assert!(vectors.len() >= 12);
    vectors
}

#[test]
fn each_vector_reads_as_the_python_runner_reads_it() {
    for vector in vectors() {
        let name = vector["name"].as_str().unwrap();
        // A fresh database per vector, with its setup (the zone a zoned
        // timestamp is read in), as the Python test makes one.
        let connection = duck::open().unwrap();
        if let Some(setup) = vector.get("setup").and_then(Value::as_str) {
            connection.execute_batch(setup).unwrap();
        }
        let fetched = duck::run(&connection, vector["sql"].as_str().unwrap()).unwrap();
        let schema: Vec<Value> = fetched
            .schema
            .fields()
            .iter()
            .map(|f| serde_json::json!({"name": f.name(), "type": arrow_type(f.data_type())}))
            .collect();
        assert_eq!(Value::Array(schema), vector["arrow"], "{name}: schema");
        let fields: Vec<_> = fetched.schema.fields().iter().map(|f| f.as_ref()).collect();
        let got = values(&fields, &fetched.batches);
        let want = &vector["values"];
        for key in ["columns", "ordered", "unordered", "unencodable"] {
            assert_eq!(got[key], want[key], "{name}: {key}");
        }
        // This host has no driver description of DuckDB's type names.
        assert_eq!(got["types"], Value::Null, "{name}");
        // The digests are of the cells the vector spells out.
        let rows: Vec<String> = vector["rows"]
            .as_array()
            .unwrap()
            .iter()
            .map(ascii_json)
            .collect();
        let ordered = sdm_testkit::case::sha256(format!("[{}]", rows.join(",")).as_bytes());
        assert_eq!(Value::from(ordered), want["ordered"], "{name}: cells");
    }
}

#[test]
fn ascii_json_writes_as_python_s_json_dumps_does() {
    let value = serde_json::json!([
        "s",
        "caf\u{e9} \u{2603} \u{1f600}\u{1}\n\"\\\u{7f}\t\u{8}\u{c}\r"
    ]);
    // As `json.dumps(value, ensure_ascii=True, separators=(",", ":"))`
    // writes it, DEL escaped too.
    assert_eq!(
        ascii_json(&value),
        r#"["s","caf\u00e9 \u2603 \ud83d\ude00\u0001\n\"\\\u007f\t\b\f\r"]"#
    );
}
