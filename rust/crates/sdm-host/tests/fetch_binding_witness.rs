// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![cfg(feature = "test-ops")]

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow_schema::{DataType, Field, Schema};
use sdm_contract::data::{AttachmentId, DataRequest, Dialect, ProviderChoice};
use serde_json::json;

#[test]
fn only_an_independently_rendered_plan_can_change_local_seed_names() {
    let request: DataRequest = serde_json::from_value(json!({
        "request":"lookup", "table":{"source":["users"]},
        "key":["id"], "columns":["id"], "seeds":{"attachment":41,
        "columns":[{"name":"key", "type":{"type":"int64"}, "nullable":false},
                   {"name":"batch", "type":{"type":"int64"}, "nullable":false}]}
    }))
    .unwrap();
    let provider = ProviderChoice::Sql {
        dialect: Dialect::new("sqlite"),
        namespace: vec![],
        view: None,
        timestamp_encoding: None,
    };
    let plan = sdm_host::default_root()
        .data_access(&provider)
        .unwrap()
        .plan(&request)
        .unwrap();
    let witness = sdm_host::fetch_plan_witness(&request, &plan)
        .unwrap()
        .unwrap();
    assert_eq!(
        witness["issued_request"],
        serde_json::to_value(&request).unwrap()
    );
    assert_eq!(witness["issued_plan"], serde_json::to_value(&plan).unwrap());
    assert_eq!(witness["canonical_request"]["seeds"]["attachment"], 0);
    let mut corrupted = serde_json::to_value(&plan).unwrap();
    corrupted["statements"][0]["text"] = json!("SELECT 0 /* not the issued read */");
    let corrupted = serde_json::from_value(corrupted).unwrap();
    assert!(
        sdm_host::fetch_plan_witness(&request, &corrupted)
            .unwrap()
            .is_none()
    );
    let schema = Arc::new(Schema::new_with_metadata(
        vec![
            Field::new("key", DataType::Int64, false),
            Field::new("batch", DataType::Int64, false),
        ],
        [("actual".to_owned(), "owner schema".to_owned())]
            .into_iter()
            .collect(),
    ));
    let actual = BTreeMap::from([(AttachmentId(41), schema.clone())]);
    let captured = sdm_host::fetch_binding_schemas(&plan, &actual).unwrap();
    assert_eq!(captured[0]["metadata"]["actual"], "owner schema");
    let request_bytes = serde_json::to_vec(&request).unwrap();
    let plan_bytes = serde_json::to_vec(&plan).unwrap();
    let encoded = sdm_host::fetch_binding_bytes(&request_bytes, &plan_bytes, &actual).unwrap();
    let decoded: serde_json::Value = serde_json::from_slice(&encoded).unwrap();
    assert_eq!(decoded["owners"], captured);
    assert_eq!(decoded["witness"], witness);
    assert!(sdm_host::fetch_binding_bytes(b"not a request", &plan_bytes, &actual).is_err());
    assert!(sdm_host::fetch_binding_bytes(&request_bytes, b"not a plan", &actual).is_err());
    assert!(sdm_host::fetch_binding_bytes(&request_bytes, &plan_bytes, &BTreeMap::new()).is_err());
    assert!(sdm_host::fetch_binding_schemas(&plan, &BTreeMap::new()).is_err());
    assert!(
        sdm_host::fetch_binding_schemas(&plan, &BTreeMap::from([(AttachmentId(42), schema)]))
            .is_err()
    );
    let wrong = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Float64, false),
        Field::new("batch", DataType::Int64, false),
    ]));
    assert!(
        sdm_host::fetch_binding_schemas(&plan, &BTreeMap::from([(AttachmentId(41), wrong)]))
            .is_err()
    );
}
