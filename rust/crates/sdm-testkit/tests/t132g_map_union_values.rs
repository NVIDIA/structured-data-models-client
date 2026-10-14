// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! T1.32g part 2c: a map's and a union's values in canonical form on this
//! host (`sdm_testkit::values`), held from outside the implementation (the
//! package's independent test author) to `contract/t132g-p2c-vectors.json`,
//! which `contract/tests/test_t132g_map_union_values.py` holds the Python
//! runner's `arrow_values` to as well, so the two encoders agree through it.
//!
//! The form (the file's `about`): a map is `["map", [[key, value], ...]]`
//! in Arrow's entry order, duplicates kept, a null map `null` and an empty
//! one `["map", []]`; a union is `["union", type code, member name,
//! cell]`, a null member keeping its tag. A vector with `sql` runs on this
//! host's DuckDB, as `tests/result_vectors.rs` runs `result-vectors.json`;
//! one with `build` is built here as Arrow (DuckDB writes no duplicate map
//! key and no dense union), as the Python test builds it with pyarrow. Each
//! must give the vector's Arrow schema as pyarrow spells it, its digests
//! and its unencodable names. An interval in a map or a chosen member has
//! its own cell since part 2e (`tests/t132g_interval_values.rs`).
//!
//! Gating: `contract/tests/t132g-pending.json`'s `pending["rust-host"]`
//! lists the vectors (by name) that failed here when the test was written.
//! Each must still fail (strict: one that passes must be taken out of the
//! list), and the list may only shrink: it stays within
//! `authored["rust-host"]`. The implementer removes names as they pass and
//! changes nothing else of this file or the vectors.

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::Arc;

use duckdb::arrow::array::{
    ArrayRef, Int32Array, Int32Builder, MapBuilder, StringArray, StringBuilder, UnionArray,
};
use duckdb::arrow::buffer::ScalarBuffer;
use duckdb::arrow::datatypes::{DataType, Field, Schema, UnionFields};
use duckdb::arrow::record_batch::RecordBatch;
use serde_json::{Value, json};

use sdm_testkit::duck;
use sdm_testkit::values::{arrow_type, ascii_json, values};

const PENDING: &str = "contract/tests/t132g-pending.json";

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

fn read_json(path: &str) -> Value {
    let text = std::fs::read_to_string(root().join(path)).unwrap();
    serde_json::from_str(&text).unwrap()
}

fn vectors() -> Vec<Value> {
    let file = read_json("contract/t132g-p2c-vectors.json");
    assert_eq!(file["format"], "sdm-result-vectors/1");
    file["vectors"].as_array().unwrap().clone()
}

fn names(value: &Value) -> BTreeSet<String> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_owned())
        .collect()
}

fn batch(name: &str, column: ArrayRef) -> RecordBatch {
    let schema = Schema::new(vec![Field::new(name, column.data_type().clone(), true)]);
    RecordBatch::try_new(Arc::new(schema), vec![column]).unwrap()
}

fn members() -> UnionFields {
    UnionFields::try_new(
        [3_i8, 7],
        [
            Field::new("i", DataType::Int32, true),
            Field::new("s", DataType::Utf8, true),
        ],
    )
    .unwrap()
}

/// The vectors DuckDB cannot make, built as the Python test builds them.
fn built(name: &str) -> RecordBatch {
    match name {
        "a map with a duplicate key" => {
            let mut map = MapBuilder::new(None, StringBuilder::new(), Int32Builder::new());
            for value in [1, 2] {
                map.keys().append_value("a");
                map.values().append_value(value);
            }
            map.append(true).unwrap();
            batch("m", Arc::new(map.finish()))
        }
        "a dense union with codes 3 and 7 and a null member" => {
            let union = UnionArray::try_new(
                members(),
                ScalarBuffer::from(vec![3_i8, 7, 7]),
                Some(ScalarBuffer::from(vec![0_i32, 0, 1])),
                vec![
                    Arc::new(Int32Array::from(vec![5])),
                    Arc::new(StringArray::from(vec![None, Some("x")])),
                ],
            )
            .unwrap();
            batch("u", Arc::new(union))
        }
        "a sparse union with codes 3 and 7 and a null member" => {
            let union = UnionArray::try_new(
                members(),
                ScalarBuffer::from(vec![7_i8, 3]),
                None,
                vec![
                    Arc::new(Int32Array::from(vec![Some(0), None])),
                    Arc::new(StringArray::from(vec![Some("y"), None])),
                ],
            )
            .unwrap();
            batch("u", Arc::new(union))
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
    assert!(all.len() >= 10);
    for vector in &all {
        let name = vector["name"].as_str().unwrap();
        assert!(
            vector.get("sql").is_some() != vector.get("build").is_some(),
            "{name}"
        );
        // Each vector's digests are its cells'.
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
        // Every schema, built or run, is the vector's.
        let (schema, _) = read(vector);
        assert_eq!(schema, vector["arrow"], "{name}: schema");
    }
}

#[test]
fn the_pending_list_only_shrinks() {
    let pending = read_json(PENDING);
    let now = names(&pending["pending"]["rust-host"]);
    let authored = names(&pending["authored"]["rust-host"]);
    assert!(
        now.is_subset(&authored),
        "pending names added: {:?}",
        now.difference(&authored)
    );
    // The rust-host list names part 2c's vectors and part 2e's
    // (`contract/t132g-p2e-vectors.json`, `tests/t132g_interval_values.rs`).
    let mut all: BTreeSet<String> = vectors()
        .iter()
        .map(|v| v["name"].as_str().unwrap().to_owned())
        .collect();
    for vector in read_json("contract/t132g-p2e-vectors.json")["vectors"]
        .as_array()
        .unwrap()
    {
        all.insert(vector["name"].as_str().unwrap().to_owned());
    }
    assert!(now.is_subset(&all), "pending names that are no vector");
}

#[test]
fn each_map_and_union_vector_reads_as_the_python_runner_reads_it() {
    let pending = names(&read_json(PENDING)["pending"]["rust-host"]);
    let mut failures = Vec::new();
    let mut passing_but_pending = Vec::new();
    for vector in vectors() {
        let name = vector["name"].as_str().unwrap();
        let (_, got) = read(&vector);
        let want = &vector["values"];
        let ok = ["columns", "ordered", "unordered", "unencodable"]
            .iter()
            .all(|key| got[*key] == want[*key])
            && got["types"] == Value::Null;
        match (ok, pending.contains(name)) {
            (true, true) => passing_but_pending.push(name.to_owned()),
            (false, false) => failures.push(format!("{name}: got {got}, want {want}")),
            _ => {}
        }
    }
    assert!(
        passing_but_pending.is_empty(),
        "these pass now: take them out of {PENDING}'s pending[\"rust-host\"]: \
         {passing_but_pending:?}"
    );
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Arrow's year-month and day-time intervals, which neither DuckDB nor
/// pyarrow writes, stay unencodable: part 2e gives a form to the
/// month-day-nano interval only.
#[test]
fn year_month_and_day_time_intervals_stay_unencodable() {
    use duckdb::arrow::array::{IntervalDayTimeArray, IntervalYearMonthArray};
    use duckdb::arrow::datatypes::IntervalDayTime;
    let months = batch(
        "m",
        Arc::new(IntervalYearMonthArray::from(vec![Some(14), None])),
    );
    let days = batch(
        "d",
        Arc::new(IntervalDayTimeArray::from(vec![
            Some(IntervalDayTime::new(1, 2)),
            None,
        ])),
    );
    for (made, name) in [
        (months, "arrow:month_interval"),
        (days, "arrow:day_time_interval"),
    ] {
        let fields: Vec<Field> = made
            .schema()
            .fields()
            .iter()
            .map(|f| f.as_ref().clone())
            .collect();
        let refs: Vec<&Field> = fields.iter().collect();
        let got = values(&refs, &[made]);
        assert_eq!(got["unencodable"], json!([name]), "{name}");
    }
}
