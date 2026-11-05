// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Zoned text policy is selected by the core, including bound key chunks.

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use sdm_contract::data::{
    AttachmentId, DataLayer, DataRequest, FetchPlan, KeyData, ProviderChoice, SqlResultConversion,
    SqlTimestampKind,
};
use sdm_contract::{Detail, Sentence};
use sdm_data::Data;
use serde_json::json;

fn provider(dialect: &str) -> ProviderChoice {
    let mut provider = ProviderChoice::sql(dialect);
    if let ProviderChoice::Sql { namespace, .. } = &mut provider {
        *namespace = match dialect {
            "postgres" => vec![Some("public".into())],
            "databricks" | "snowflake" => {
                vec![Some("catalog".into()), Some("schema".into())]
            }
            _ => Vec::new(),
        };
    }
    provider
}

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
            if dialect != "postgres"
                && matches!(
                    request,
                    DataRequest::Describe {
                        part: sdm_contract::data::DescribePart::Namespace,
                        ..
                    }
                )
            {
                continue;
            }
            let plan = Data::new().plan(&request, &provider(dialect)).unwrap();
            let written = serde_json::to_value(plan).unwrap();
            if dialect == "postgres" {
                assert_eq!(
                    written["result_conversions"],
                    json!([{"conversion":"zoned_text_utc"}])
                );
            } else {
                assert!(written.get("result_conversions").is_none(), "{dialect}");
            }
        }
    }
}

fn chunked_keys(kind: &str) -> (DataRequest, BTreeMap<AttachmentId, KeyData>) {
    let request: DataRequest = serde_json::from_value(json!({
        "request":kind, "table":{"source":["users"]}, "key":["id"], "columns":["id"],
        "seeds":{"attachment":2,"columns":[
            {"name":"key","nullable":true,"type":{"type":"utf8"}},
            {"name":"batch","nullable":false,"type":{"type":"int64"}}
        ]}
    }))
    .unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Utf8, true),
        Field::new("batch", DataType::Int64, false),
    ]));
    let keys = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(vec![
                "x".repeat(600_000),
                "y".repeat(600_000),
                "z".repeat(600_000),
            ])) as ArrayRef,
            Arc::new(Int64Array::from(vec![0, 1, 2])),
        ],
    )
    .unwrap();
    (
        request,
        BTreeMap::from([(
            AttachmentId(2),
            KeyData {
                schema,
                batches: vec![keys],
            },
        )]),
    )
}

#[test]
fn postgres_key_text_chunks_preserve_the_whole_result_policy() {
    let provider = provider("postgres");
    let data = Data::new();
    for kind in ["lookup", "neighbors"] {
        let (request, keys) = chunked_keys(kind);
        let plan = data.plan(&request, &provider).unwrap();
        let bound = data
            .bind(&provider, &request, plan, &keys, |value| value.to_string())
            .unwrap();
        assert_eq!(bound.fetches.len(), 3, "{kind}");
        for (offset, fetch) in bound.fetches.into_iter().enumerate() {
            let FetchPlan::Sql {
                statements,
                result_conversions,
                ..
            } = fetch.plan
            else {
                panic!("a Postgres fetch is SQL");
            };
            assert_eq!(result_conversions, vec![SqlResultConversion::ZonedTextUtc]);
            let batch = if offset == 0 {
                "(r.ordinality - 1)::bigint".to_owned()
            } else {
                format!("(r.ordinality - 1 + {offset})::bigint")
            };
            assert!(statements[0].text.contains(&batch), "{kind}: {offset}");
            assert_eq!(fetch.batches.len(), 1);
            let attachment = fetch.batches.values().next().unwrap();
            assert_eq!(attachment.batches.len(), 1);
            assert_eq!(attachment.batches[0].num_rows(), 1);
            let text = attachment.batches[0]
                .column(0)
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            let rows: serde_json::Value = serde_json::from_str(text.value(0)).unwrap();
            let expected = ["x", "y", "z"][offset].repeat(600_000);
            assert_eq!(rows, json!([expected]));
        }
    }
}

#[test]
fn postgres_does_not_accept_microseconds_by_also_naming_utc_text() {
    let provider = provider("postgres");
    let data = Data::new();
    let (request, keys) = chunked_keys("neighbors");
    for include_utc_text in [false, true] {
        let mut plan = data.plan(&request, &provider).unwrap();
        let FetchPlan::Sql {
            result_conversions, ..
        } = &mut plan
        else {
            panic!("a Postgres plan is SQL");
        };
        if !include_utc_text {
            result_conversions.clear();
        }
        result_conversions.push(SqlResultConversion::TimestampMicros {
            column: 1,
            name: "id".into(),
            source_kind: SqlTimestampKind::WallTime,
        });
        let error = data
            .bind(&provider, &request, plan, &keys, |value| value.to_string())
            .unwrap_err();
        assert_eq!(error.sentence, Sentence::DataRequestRefused);
        assert_eq!(
            error.fields.get("field"),
            Some(&Detail::text("timestamp_encoding"))
        );
        assert_eq!(
            error.fields.get("request"),
            Some(&Detail::text("neighbors"))
        );
    }
}
