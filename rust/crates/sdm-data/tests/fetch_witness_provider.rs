// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![cfg(feature = "test-ops")]

use sdm_contract::data::{DataLayer, DataRequest, Dialect, FetchPlan, ProviderChoice, SqlParam};
use sdm_data::Data;
use serde_json::json;

#[test]
fn witness_candidates_are_owned_by_the_data_layer_and_keep_text_and_mixed_plans_literal() {
    let request: DataRequest = serde_json::from_value(json!({
        "request":"lookup", "table":{"source":["users"]},
        "key":["id"], "columns":["id"], "seeds":{"attachment":41,
        "columns":[{"name":"key", "type":{"type":"int64"}, "nullable":false},
                   {"name":"batch", "type":{"type":"int64"}, "nullable":false}]}
    }))
    .unwrap();
    for dialect in ["sqlite", "duckdb"] {
        let provider = ProviderChoice::Sql {
            dialect: Dialect::new(dialect),
            namespace: vec![],
            view: None,
            timestamp_encoding: None,
        };
        let plan = Data::new().plan(&request, &provider).unwrap();
        assert_eq!(sdm_data::fetch_witness_provider(&plan), Some(provider));
        let mut changed = plan.clone();
        let FetchPlan::Sql { statements, .. } = &mut changed else {
            panic!("SQL plan")
        };
        statements[0].params.push(SqlParam::Text {
            attachment: sdm_contract::data::AttachmentId(42),
        });
        assert!(sdm_data::fetch_witness_provider(&changed).is_none());
        let mut changed = plan;
        let FetchPlan::Sql { statements, .. } = &mut changed else {
            panic!("SQL plan")
        };
        statements[0].dialect = Dialect::new("postgres");
        assert!(sdm_data::fetch_witness_provider(&changed).is_none());
    }
}
