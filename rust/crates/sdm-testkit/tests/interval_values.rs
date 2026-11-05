// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! part 2e: an interval's values in canonical form on this host
//! (`sdm_testkit::values`), held from outside the implementation (the
//! package's independent test author) to `contract/arrow-interval-vectors.json`,
//! which `contract/tests/test_arrow_interval_values.py` holds the Python
//! runner's `arrow_values` to as well, so the two encoders agree through it.
//!
//! The form (the file's `about`): an Arrow `month_day_nano_interval` is
//! `["interval", months, days, nanoseconds]`, months and days JSON integers,
//! nanoseconds a decimal string, a null interval `null`; alike in a list, a
//! struct, a map or a chosen union member. A vector with `sql` runs on this
//! host's DuckDB; one with `build` is built here as Arrow, as the Python
//! test builds it with pyarrow (the fields' limits; a `date64`, which the
//! form still does not hold and both encoders name unencodable). Arrow's
//! year-month and day-time intervals stay unencodable
//! (`tests/map_union_values.rs`).
//!
use std::path::PathBuf;
use std::sync::Arc;

use duckdb::arrow::array::{ArrayRef, Date64Array, IntervalMonthDayNanoArray};
use duckdb::arrow::datatypes::{Field, IntervalMonthDayNano, Schema};
use duckdb::arrow::record_batch::RecordBatch;
use serde_json::{Value, json};

use sdm_testkit::duck;
use sdm_testkit::values::{arrow_type, ascii_json, values};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

fn read_json(path: &str) -> Value {
    let text = std::fs::read_to_string(root().join(path)).unwrap();
    serde_json::from_str(&text).unwrap()
}

fn vectors() -> Vec<Value> {
    let file = read_json("contract/arrow-interval-vectors.json");
    assert_eq!(file["format"], "sdm-result-vectors/1");
    file["vectors"].as_array().unwrap().clone()
}

fn batch(name: &str, column: ArrayRef) -> RecordBatch {
    let schema = Schema::new(vec![Field::new(name, column.data_type().clone(), true)]);
    RecordBatch::try_new(Arc::new(schema), vec![column]).unwrap()
}

/// The vectors DuckDB cannot make, built as the Python test builds them.
fn built(name: &str) -> RecordBatch {
    match name {
        "intervals at their fields' limits" => batch(
            "v",
            Arc::new(IntervalMonthDayNanoArray::from(vec![
                Some(IntervalMonthDayNano::new(i32::MAX, i32::MIN, i64::MAX)),
                Some(IntervalMonthDayNano::new(-1, 0, i64::MIN)),
                None,
            ])),
        ),
        "a date64 stays unencodable" => {
            batch("d", Arc::new(Date64Array::from(vec![Some(0), None])))
        }
        other => panic!("no builder for the vector {other:?}"),
    }
}

/// The vector's Arrow schema as pyarrow spells it, and its values as this
/// host writes them.
fn read(vector: &Value) -> (Value, Value) {
    let (fields, batches): (Vec<Field>, Vec<RecordBatch>) = match vector.get("build") {
        Some(build) => {
            let made = built(build.as_str().unwrap());
            let fields = made
                .schema()
                .fields()
                .iter()
                .map(|f| f.as_ref().clone())
                .collect();
            (fields, vec![made])
        }
        None => {
            let connection = duck::open().unwrap();
            let fetched = duck::run(&connection, vector["sql"].as_str().unwrap()).unwrap();
            let fields = fetched
                .schema
                .fields()
                .iter()
                .map(|f| f.as_ref().clone())
                .collect();
            (fields, fetched.batches)
        }
    };
    let schema: Vec<Value> = fields
        .iter()
        .map(|f| json!({"name": f.name(), "type": arrow_type(f.data_type())}))
        .collect();
    let refs: Vec<&Field> = fields.iter().collect();
    (Value::Array(schema), values(&refs, &batches))
}

#[test]
fn the_vectors_are_what_they_say() {
    let all = vectors();
    assert!(all.len() >= 6);
    for vector in &all {
        let name = vector["name"].as_str().unwrap();
        assert!(
            vector.get("sql").is_some() != vector.get("build").is_some(),
            "{name}"
        );
        let mut rows: Vec<String> = vector["rows"]
            .as_array()
            .unwrap()
            .iter()
            .map(ascii_json)
            .collect();
        let ordered = sdm_testkit::case::sha256(format!("[{}]", rows.join(",")).as_bytes());
        assert_eq!(Value::from(ordered), vector["values"]["ordered"], "{name}");
        rows.sort();
        let unordered = sdm_testkit::case::sha256(format!("[{}]", rows.join(",")).as_bytes());
        assert_eq!(
            Value::from(unordered),
            vector["values"]["unordered"],
            "{name}"
        );
        let (schema, _) = read(vector);
        assert_eq!(schema, vector["arrow"], "{name}: schema");
    }
}

#[test]
fn each_interval_vector_reads_as_the_python_runner_reads_it() {
    let mut failures = Vec::new();
    for vector in vectors() {
        let name = vector["name"].as_str().unwrap();
        let (_, got) = read(&vector);
        let want = &vector["values"];
        let ok = ["columns", "ordered", "unordered", "unencodable"]
            .iter()
            .all(|key| got[*key] == want[*key])
            && got["types"] == Value::Null;
        if !ok {
            failures.push(format!("{name}: got {got}, want {want}"));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
