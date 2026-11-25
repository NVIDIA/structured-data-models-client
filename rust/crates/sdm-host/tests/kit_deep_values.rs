// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The SQL kit reads an answer's values on the adapter's grown stack, as
//! the core read a fetch's: a JSON value recurses once per level as it is
//! read, and a host's thread (a JVM's) may have a small stack, so a deep
//! value answered on such a thread is read, never overflowing it.

#![allow(
    clippy::disallowed_types,
    reason = "test code: Arrow's field metadata is a HashMap"
)]

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use arrow_array::{RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use sdm_contract::data::{AttachmentId, DataRequest, KeyData};
use sdm_host::kit::{Answered, HostKit, SqlChoice};

/// A JSON array nested `depth` levels deep.
fn nested(depth: usize) -> String {
    format!("{}{}", "[".repeat(depth), "]".repeat(depth))
}

fn postgres() -> SqlChoice {
    serde_json::from_value(serde_json::json!({
        "provider": "sql", "dialect": "postgres", "namespace": ["public"]
    }))
    .unwrap()
}

#[test]
fn a_deep_json_value_is_read_on_a_small_stack() {
    let depth = 1_900;
    // A thread with far less stack than reading the value takes.
    let read = std::thread::Builder::new()
        .stack_size(256 << 10)
        .spawn(move || {
            let kit = HostKit::new(postgres()).unwrap();
            let request: DataRequest = serde_json::from_value(serde_json::json!({
                "request": "scan", "table": {"source": ["t"]}, "columns": ["v"],
                "filters": [], "limit": null
            }))
            .unwrap();
            let keys: BTreeMap<AttachmentId, KeyData> = BTreeMap::new();
            let answered = kit.answer(&request, &keys, &mut |_fetch| {
                let field = Field::new("v", DataType::Utf8, true).with_metadata(HashMap::from([(
                    "ARROW:extension:name".to_owned(),
                    "arrow.json".to_owned(),
                )]));
                let schema = Arc::new(Schema::new(vec![field]));
                let column = Arc::new(StringArray::from(vec![Some(nested(depth))]));
                let batch = RecordBatch::try_new(Arc::clone(&schema), vec![column]).unwrap();
                Ok(Answered {
                    schema,
                    batches: vec![batch],
                })
            });
            let answered = answered.map_err(|_| "refused").unwrap();
            answered
                .batches
                .iter()
                .map(RecordBatch::num_rows)
                .sum::<usize>()
        })
        .unwrap()
        .join()
        .expect("the kit read the value without overflowing the thread's stack");
    assert_eq!(read, 1);
}
