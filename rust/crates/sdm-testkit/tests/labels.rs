// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The labels on the pure-Rust host (T1.27, decision E19): each baseline
//! case's context labels, as the core computes them over this host's
//! fetches (the test-only `labels` operation, started in place of the
//! case's `materialize`), are the reference's exactly, as its payload's
//! context carries them in the committed reference observation:
//! `base.regression` and `decimal.target-38-18` (windowed sums at float32),
//! `base.binary` (a condition over a count) and `base.multiclass` (a static
//! text target). Each reference row is held, by its instance id, to the
//! core's example of that position: the entity, the anchor to the
//! microsecond the payload writes, and the label, compared exactly. The
//! reference leaves out of a static target's context the entities it is
//! asked to predict for (T1.28a's rule): an example missing from its rows
//! must be one of those, and the counts must agree with the payload's
//! `num_context_examples`. The Python host is held to the same rows by
//! `packages/structured-data-models-client/tests_test_ops/test_the_labels.py`.
//! The check fails closed on missing, empty or retyped evidence, seeded
//! below.

use std::path::PathBuf;

use arrow_array::cast::AsArray;
use arrow_array::types::{Float32Type, Int64Type, TimestampNanosecondType};
use arrow_array::{Array, RecordBatch};
use arrow_schema::DataType;
use sdm_contract::ops::{Operation, OperationResult};
use sdm_contract::test_ops::TestOperation;
use sdm_host::Attachment;
use sdm_testkit::host;
use sdm_testkit::{bases, case};
use serde_json::Value;

/// The cases whose labels are held, and what each label is.
const CASES: [&str; 4] = [
    "base.regression",
    "base.binary",
    "base.multiclass",
    "decimal.target-38-18",
];

fn contract() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../contract")
}

/// The committed reference observation of `case`.
fn reference(case: &str) -> Value {
    let directory = contract().join("reference");
    for entry in std::fs::read_dir(&directory).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
            continue;
        }
        for line in std::fs::read_to_string(&path).unwrap().lines() {
            let observed: Value = serde_json::from_str(line).unwrap();
            if observed["case"] == case {
                return observed;
            }
        }
    }
    Value::Null
}

/// One context example as the core gave it: the entity's key as the
/// payload writes an integer or text, the anchor as it writes a time, and
/// the label as JSON (a number's text read back exactly, a boolean, text).
#[derive(Debug, Clone, PartialEq)]
struct Example {
    entity: Value,
    anchor: String,
    target: Value,
}

/// The core's context examples for `case`, on this host.
fn core_examples(name: &str) -> Vec<Example> {
    let bases = bases::load(&contract().join("bases.yaml")).unwrap();
    let path = contract().join(format!("cases/s1/{name}.json"));
    let loaded = case::load(&[path]).unwrap().remove(0);
    let (observed, driven) = host::materialize_as(&loaded.case, &bases, &|materialize| {
        Operation::Test(TestOperation::Labels {
            materialize: Box::new(materialize),
        })
    })
    .unwrap();
    let Some((handle, Ok(OperationResult::Test(result)))) = driven else {
        panic!("{name}: the labels were not given: {:?}", observed.answer);
    };
    let Ok(Some(Attachment::Arrow(data))) = handle.take_attachment(result.data.unwrap()) else {
        panic!("{name}: the labels are not attached");
    };
    let batch = arrow_select::concat::concat_batches(&data.schema, &data.batches).unwrap();
    assert_eq!(batch.num_rows() as u64, result.rows);
    examples(&batch)
}

fn examples(batch: &RecordBatch) -> Vec<Example> {
    let entity = batch.column_by_name("entity").unwrap();
    let anchor = batch
        .column_by_name("anchor")
        .unwrap()
        .as_primitive::<TimestampNanosecondType>();
    let target = batch.column_by_name("target").unwrap();
    (0..batch.num_rows())
        .map(|row| Example {
            entity: match entity.data_type() {
                DataType::Int64 => Value::from(entity.as_primitive::<Int64Type>().value(row)),
                DataType::Utf8 => Value::from(entity.as_string::<i32>().value(row)),
                other => panic!("an entity key of {other}"),
            },
            anchor: sdm_request::payload::iso_microseconds(anchor.value(row)),
            target: match target.data_type() {
                DataType::Float32 => {
                    Value::from(f64::from(target.as_primitive::<Float32Type>().value(row)))
                }
                DataType::Int64 => Value::from(target.as_primitive::<Int64Type>().value(row)),
                DataType::Boolean => Value::from(target.as_boolean().value(row)),
                DataType::Utf8 => Value::from(target.as_string::<i32>().value(row)),
                other => panic!("a label of {other}"),
            },
        })
        .collect()
}

/// A payload cell as the core's examples hold it: `{"number": text}` read
/// as an integer where it is one, else as the float it is, exactly.
fn cell(value: &Value) -> Option<Value> {
    match value {
        Value::Object(number) => {
            let text = number.get("number")?.as_str()?;
            if let Ok(int) = text.parse::<i64>() {
                return Some(Value::from(int));
            }
            text.parse::<f64>().ok().map(Value::from)
        }
        Value::String(_) | Value::Bool(_) => Some(value.clone()),
        _ => None,
    }
}

/// Each way `core`'s examples depart from the reference's context in
/// `observed`, for a case asking for `indices`; missing, empty or retyped
/// evidence is a problem.
fn problems(observed: &Value, indices: &[Value], core: &[Example]) -> Vec<String> {
    let payload = &observed["public"]["payloads"][0]["parsed"];
    let Some(table) = payload["context"]["instance_table"].as_object() else {
        return vec!["the reference holds no context".into()];
    };
    let columns: Vec<&str> = table
        .get("columns")
        .and_then(Value::as_array)
        .map(|c| c.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    let at = |name: &str| columns.iter().position(|c| *c == name);
    let (Some(id), Some(entity), Some(anchor), Some(target)) = (
        at("instance_id"),
        at("__kumo_entity_ref_0"),
        at("__kumo_anchor_time"),
        at("TARGET"),
    ) else {
        return vec![format!("the reference's context has columns {columns:?}")];
    };
    let Some(rows) = table
        .get("rows")
        .and_then(Value::as_array)
        .filter(|r| !r.is_empty())
    else {
        return vec!["the reference's context holds no row".into()];
    };
    let Some(count) = cell(&payload["metadata"]["num_context_examples"])
        .and_then(|n| n.as_u64())
        .and_then(|n| usize::try_from(n).ok())
    else {
        return vec!["the reference names no context count".into()];
    };
    let mut found = Vec::new();
    if core.is_empty() {
        found.push("the core gave no label".into());
    }
    if core.len() != count {
        found.push(format!(
            "the core gave {} examples, the reference {count}",
            core.len()
        ));
    }
    let mut seen = vec![false; core.len()];
    for row in rows {
        let Some(position) = row
            .get(id)
            .and_then(cell)
            .and_then(|v| v.as_u64())
            .and_then(|v| usize::try_from(v).ok())
            .filter(|p| *p < core.len())
        else {
            found.push(format!("a reference row names no example: {row}"));
            continue;
        };
        seen[position] = true;
        let wanted = Example {
            entity: row.get(entity).and_then(cell).unwrap_or(Value::Null),
            anchor: row
                .get(anchor)
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned(),
            target: row.get(target).and_then(cell).unwrap_or(Value::Null),
        };
        if core[position] != wanted {
            found.push(format!(
                "example {position}: the core's {:?}, the reference's {wanted:?}",
                core[position]
            ));
        }
    }
    for (position, example) in core.iter().enumerate() {
        if !seen[position] && !indices.contains(&example.entity) {
            found.push(format!(
                "example {position} is not the reference's and not one predicted for"
            ));
        }
    }
    found
}

fn indices(name: &str) -> Vec<Value> {
    let path = contract().join(format!("cases/s1/{name}.json"));
    let document: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    document["input"]["materialize"]["indices"]
        .as_array()
        .cloned()
        .unwrap_or_default()
}

#[test]
fn the_labels_on_this_host_are_the_reference_s() {
    for name in CASES {
        let core = core_examples(name);
        let observed = reference(name);
        assert_eq!(
            problems(&observed, &indices(name), &core),
            Vec::<String>::new(),
            "{name}"
        );
    }
}

#[test]
fn the_label_check_fails_closed() {
    let name = "base.binary";
    let core = core_examples(name);
    let observed = reference(name);
    let indices = indices(name);
    assert!(problems(&observed, &indices, &core).is_empty());
    // A label flipped, an anchor moved, a label retyped, one dropped.
    let mut flipped = core.clone();
    flipped[3].target = Value::from(!flipped[3].target.as_bool().unwrap());
    assert!(!problems(&observed, &indices, &flipped).is_empty());
    let mut moved = core.clone();
    moved[0].anchor = "2024-01-30T00:00:00.000000Z".into();
    assert!(!problems(&observed, &indices, &moved).is_empty());
    let mut retyped = core.clone();
    retyped[1].target = Value::from(1);
    assert!(!problems(&observed, &indices, &retyped).is_empty());
    assert!(!problems(&observed, &indices, &core[1..]).is_empty());
    // No evidence: no core labels, no reference, a reference with no rows.
    assert!(!problems(&observed, &indices, &[]).is_empty());
    assert!(!problems(&Value::Null, &indices, &core).is_empty());
    let mut emptied = observed.clone();
    emptied["public"]["payloads"][0]["parsed"]["context"]["instance_table"]["rows"] =
        Value::Array(vec![]);
    assert!(!problems(&emptied, &indices, &core).is_empty());
    // An example the reference leaves out is one predicted for, only.
    let mut left_out = observed.clone();
    left_out["public"]["payloads"][0]["parsed"]["context"]["instance_table"]["rows"]
        .as_array_mut()
        .unwrap()
        .remove(0);
    assert!(!problems(&left_out, &[], &core).is_empty());
}
