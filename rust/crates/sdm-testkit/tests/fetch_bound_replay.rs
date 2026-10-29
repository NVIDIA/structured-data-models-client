// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A replay's local names may change only with its actual bound data.

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow_array::{ArrayRef, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use sdm_contract::data::AttachmentId;
use sdm_contract::run::FetchStep;
use sdm_testkit::{duck::Fetched, fetch_bindings, replay::Recording};
use serde_json::json;

fn step(attachment: u32) -> FetchStep {
    serde_json::from_value(json!({
        "id":1,
        "request":{"request":"lookup","table":{"source":["users"]},
          "key":["id"],"columns":["id"],"seeds":{"attachment":attachment+1,"columns":[
            {"name":"key","type":{"type":"int64"},"nullable":false},
            {"name":"batch","type":{"type":"int64"},"nullable":false}]}},
        "plan":{"plan":"sql","expect":{"expect":"discover"},"statements":[
          {"dialect":"postgres","text":"SELECT actual_bound_keys",
           "params":[{"param":"text","attachment":attachment}]}]},
    }))
    .unwrap()
}

fn keys(attachment: u32, value: &str) -> BTreeMap<AttachmentId, Fetched> {
    let schema = Arc::new(Schema::new(vec![Field::new("keys", DataType::Utf8, false)]));
    let values: ArrayRef = Arc::new(StringArray::from(vec![value]));
    let batch = RecordBatch::try_new(schema.clone(), vec![values]).unwrap();
    BTreeMap::from([(
        AttachmentId(attachment),
        Fetched {
            schema,
            batches: vec![batch],
        },
    )])
}

#[test]
fn replay_checks_actual_text_seed_content_before_local_id_correspondence() {
    let original = step(41);
    let bound = keys(41, "[{\"key\":9007199254740993,\"batch\":0}]");
    let recorded = fetch_bindings::observe(&original, 3, &bound).unwrap();
    let path = std::env::temp_dir().join(format!("sdm-bound-replay-{}.jsonl", std::process::id()));
    let line = json!({"format":"sdm-replay/1","case":"authored@postgres","fetches":[{
        "request":original.request,"plan":original.plan,"bindings":recorded,"statements":["SELECT actual_bound_keys"],
        "error":{"kind":"data_source","retryable":false}}]});
    std::fs::write(&path, format!("{line}\n")).unwrap();
    let recording = Recording::load(&path).unwrap();
    let renamed = step(4);
    let same = keys(4, "[{\"key\":9007199254740993,\"batch\":0}]");
    assert!(
        recording
            .replay("authored@postgres")
            .unwrap()
            .answer_bound(&renamed, 7, &same)
            .is_ok()
    );
    let changed = keys(4, "[{\"key\":9007199254740992,\"batch\":0}]");
    let retry = recording.replay("authored@postgres").unwrap();
    assert!(retry.answer_bound(&renamed, 7, &changed).is_err());
    assert!(!retry.finished());
    assert!(retry.answer_bound(&renamed, 7, &same).is_ok());
    let mut wrong_sql = renamed.clone();
    let sdm_contract::data::FetchPlan::Sql { statements, .. } = &mut wrong_sql.plan else {
        unreachable!()
    };
    statements[0].text.push_str(" WHERE false");
    assert!(
        recording
            .replay("authored@postgres")
            .unwrap()
            .answer_bound(&wrong_sql, 7, &same)
            .is_err()
    );
    let mut legacy = line;
    legacy["fetches"][0]
        .as_object_mut()
        .unwrap()
        .remove("bindings");
    std::fs::write(&path, format!("{legacy}\n")).unwrap();
    let legacy = Recording::load(&path).unwrap();
    assert!(
        legacy
            .replay("authored@postgres")
            .unwrap()
            .answer_bound(&renamed, 7, &same)
            .is_err()
    );
    std::fs::remove_file(path).unwrap();
}
