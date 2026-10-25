// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Zoned text policy is selected by the core, including bound key chunks.

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use sdm_contract::data::{AttachmentId, DataLayer, DataRequest, KeyData, ProviderChoice};
use sdm_data::Data;
use serde_json::json;

#[test]
fn postgres_alone_instructs_every_sql_result_to_encode_zoned_text_in_utc() {
    for request in [
        json!({"request":"describe", "tables":[], "part":"namespace"}),
        json!({"request":"describe", "tables":[{"source":["users"]}], "part":"columns"}),
        json!({"request":"preview", "table":{"source":["users"]}, "columns":["id"], "limit":2}),
        json!({"request":"count_rows", "table":{"source":["users"]}, "filters":[]}),
    ] {
        let request: DataRequest = serde_json::from_value(request).unwrap();
        for dialect in ["postgres", "duckdb", "sqlite", "databricks", "snowflake"] {
            let plan = Data::new().plan(&request, &ProviderChoice::sql(dialect)).unwrap();
            let written = serde_json::to_value(plan).unwrap();
            if dialect == "postgres" {
                assert_eq!(written["result_conversions"], json!([{"conversion":"zoned_text_utc"}]));
            } else {
                assert!(written.get("result_conversions").is_none(), "{dialect}");
            }
        }
    }
}

#[test]
fn postgres_key_text_chunks_preserve_the_whole_result_policy() {
    let request: DataRequest = serde_json::from_value(json!({
        "request":"lookup", "table":{"source":["users"]}, "key":["id"], "columns":["id"],
        "seeds":{"attachment":2,"columns":[
            {"name":"key","nullable":true,"type":{"type":"utf8"}},
            {"name":"batch","nullable":false,"type":{"type":"int64"}}
        ]}
    })).unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Utf8, true),
        Field::new("batch", DataType::Int64, false),
    ]));
    let keys = RecordBatch::try_new(Arc::clone(&schema), vec![
        Arc::new(StringArray::from(vec!["x".repeat(600_000), "y".repeat(600_000)])) as ArrayRef,
        Arc::new(Int64Array::from(vec![0, 1])),
    ]).unwrap();
    let provider = ProviderChoice::sql("postgres");
    let data = Data::new();
    let plan = data.plan(&request, &provider).unwrap();
    let bound = data.bind(&provider, &request, plan, &BTreeMap::from([
        (AttachmentId(2), KeyData { schema, batches: vec![keys] })
    ]), |value| value.to_string()).unwrap();
    assert_eq!(bound.fetches.len(), 2);
    for fetch in bound.fetches {
        assert_eq!(serde_json::to_value(fetch.plan).unwrap()["result_conversions"],
            json!([{"conversion":"zoned_text_utc"}]));
    }
}
