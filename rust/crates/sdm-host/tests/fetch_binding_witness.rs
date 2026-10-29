// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![cfg(feature = "test-ops")]

use sdm_contract::data::{DataRequest, Dialect, ProviderChoice};
use serde_json::json;

#[test]
fn only_an_independently_rendered_plan_can_change_local_seed_names() {
    let request: DataRequest = serde_json::from_value(json!({
        "request":"lookup", "table":{"source":["users"]},
        "key":["id"], "columns":["id"], "seeds":{"attachment":41,
        "columns":[{"name":"__kumo_id__", "type":{"type":"int64"}, "nullable":false},
                   {"name":"__kumo_batch__", "type":{"type":"int64"}, "nullable":false}]}
    })).unwrap();
    let provider = ProviderChoice::Sql {dialect:Dialect::new("sqlite"),
        namespace:vec![],view:None,timestamp_encoding:None};
    let plan = sdm_host::default_root().data_access(&provider).unwrap().plan(&request).unwrap();
    let witness = sdm_host::fetch_plan_witness(&request, &plan).unwrap().unwrap();
    assert_eq!(witness["issued_request"], serde_json::to_value(&request).unwrap());
    assert_eq!(witness["issued_plan"], serde_json::to_value(&plan).unwrap());
    assert_eq!(witness["canonical_request"]["seeds"]["attachment"], 0);
    let mut corrupted = serde_json::to_value(&plan).unwrap();
    corrupted["statements"][0]["text"] = json!("SELECT 0 /* not the issued read */");
    let corrupted = serde_json::from_value(corrupted).unwrap();
    assert!(sdm_host::fetch_plan_witness(&request, &corrupted).unwrap().is_none());
}
