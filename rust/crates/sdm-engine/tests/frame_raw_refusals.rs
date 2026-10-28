// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Raw guards are distinct from complete native source products.

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow_array::{
    ArrayRef, Int64Array, LargeStringArray, RecordBatch, StringArray,
    TimestampNanosecondArray, TimestampSecondArray,
};
use sdm_contract::data::strings::StringEncoding;
use sdm_contract::ops::HeldSchema;
use sdm_contract::{CoreError, Detail, Sentence};
use sdm_engine::frames::Snapshot;
use sdm_engine::graph::Graph;
use serde_json::json;

fn graph(key_type: &str, value_type: &str, stype: &str, timed: bool) -> Graph {
    let held: HeldSchema = serde_json::from_value(json!({"tables":[{
        "name":"rows","source":{"source":["rows"]},
        "columns":[
            {"name":"id","dtype":key_type,"stype":"ID","stype_before_key":null},
            {"name":"value","dtype":value_type,"stype":stype,"stype_before_key":null}
        ],"primary_key":["id"],"time_column":if timed {Some("value")} else {None}
    }],"links":[]})).unwrap();
    let description = sdm_schema::held::read(held, &BTreeMap::new(), true)
        .unwrap()
        .description;
    Graph::new(&description).unwrap()
}

fn rows(id: ArrayRef, value: ArrayRef) -> BTreeMap<String, RecordBatch> {
    BTreeMap::from([(
        "rows".into(),
        RecordBatch::try_from_iter([("id", id), ("value", value)]).unwrap(),
    )])
}

fn ids() -> ArrayRef {
    Arc::new(Int64Array::from(vec![1, 2]))
}

fn assert_rule(result: Result<Snapshot, CoreError>, rule: &str) {
    let error = match result {
        Ok(_) => panic!("unprepared raw input must retain its intended guard"),
        Err(error) => error,
    };
    assert_eq!(error.sentence, Sentence::UnsupportedOperation);
    assert_eq!(error.fields["rule"], Detail::text(rule));
}

#[test]
fn raw_text_keys_require_the_actual_encoding_product_for_both_offset_widths() {
    let graph = graph("string", "int64", "numerical", false);
    for id in [
        Arc::new(StringArray::from(vec!["a", "b"])) as ArrayRef,
        Arc::new(LargeStringArray::from(vec!["a", "b"])) as ArrayRef,
    ] {
        let input = rows(id, Arc::new(Int64Array::from(vec![10, 20])));
        assert_rule(Snapshot::new(&graph, input.clone()), "frame_text_key");
        Snapshot::new_native(
            &graph,
            input,
            BTreeMap::from([(
                "rows".into(),
                BTreeMap::from([("id".into(), StringEncoding::Utf8)]),
            )]),
        )
        .unwrap();
    }
}

#[test]
fn raw_text_and_untimed_timestamps_keep_separate_preparation_guards() {
    let graph = graph("int64", "string", "text", false);
    let input = rows(ids(), Arc::new(StringArray::from(vec!["A", "B"])));
    assert_rule(Snapshot::new(&graph, input.clone()), "frame_text_features");
    Snapshot::new_native(
        &graph,
        input,
        BTreeMap::from([(
            "rows".into(),
            BTreeMap::from([("value".into(), StringEncoding::Utf8)]),
        )]),
    )
    .unwrap();

    let graph = graph("int64", "timestamp[ns]", "timestamp", false);
    let input = rows(ids(), Arc::new(TimestampNanosecondArray::from(vec![1, 2])));
    assert_rule(
        Snapshot::new(&graph, input.clone()),
        "frame_untimed_timestamp_features",
    );
    Snapshot::new_native(&graph, input, BTreeMap::new()).unwrap();
}

#[test]
fn raw_time_type_and_range_guards_do_not_reject_prepared_native_counts() {
    let graph = graph("int64", "timestamp[ns]", "timestamp", true);
    assert_rule(
        Snapshot::new(&graph, rows(ids(), Arc::new(Int64Array::from(vec![1, 2])))),
        "frame_time_type",
    );
    assert_rule(
        Snapshot::new(
            &graph,
            rows(ids(), Arc::new(TimestampSecondArray::from(vec![1, i64::MAX]))),
        ),
        "frame_wide_timestamps",
    );
    let snapshot = Snapshot::new_native(
        &graph,
        rows(ids(), Arc::new(TimestampNanosecondArray::from(vec![1, 2]))),
        BTreeMap::new(),
    )
    .unwrap();
    assert_eq!(snapshot.time_bounds("rows").unwrap(), (Some(1), Some(2)));
}
