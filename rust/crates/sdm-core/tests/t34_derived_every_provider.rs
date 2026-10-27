// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A derived relation only of a provider declaring it reads one (contract
//! v28, T3.4 fix 1, round 1's Codex P1-5): enforced where every request is
//! routed, for a host provider as for a SQL one, so a provider that reads
//! only a relation's name never answers for a filter or columns it would
//! drop.

use std::sync::Arc;

use sdm_contract::data::{DERIVED_RULE, DataRequest, ProviderChoice};
use sdm_contract::{Detail, ErrorCode};
use sdm_core::CompositionRoot;
use serde_json::json;

fn host(derived: bool) -> ProviderChoice {
    serde_json::from_value(json!({
        "provider": "host",
        "capabilities": {
            "optional": [], "max_keys_per_request": null, "sampling": "seeded_stable",
            "max_seed": 18446744073709551615u64, "text_times": null, "null_order": null,
            "index_advice": null, "derived_relations": derived
        }
    }))
    .unwrap()
}

fn scan(table: serde_json::Value) -> DataRequest {
    serde_json::from_value(json!({
        "request": "scan", "table": table, "columns": ["x"], "filters": [], "limit": null
    }))
    .unwrap()
}

fn root() -> CompositionRoot {
    CompositionRoot::new(Arc::new(sdm_data::Data::new()))
}

#[test]
fn a_host_provider_not_declaring_derived_relations_is_never_asked_one() {
    let access = root().data_access(&host(false)).unwrap();
    for table in [
        json!({"source": ["t"], "filter": "x > 1"}),
        json!({"source": ["t"], "columns": [{"name": "x", "column": "x"}]}),
    ] {
        let Err(refused) = access.route(&scan(table)) else {
            panic!("a derived relation was routed to a provider that does not read one");
        };
        assert_eq!(refused.code, ErrorCode::UnsupportedRequest);
        assert_eq!(
            refused.fields.get("rule"),
            Some(&Detail::text(DERIVED_RULE))
        );
    }
    // A plain relation is routed as before.
    assert!(access.route(&scan(json!({"source": ["t"]}))).is_ok());
}

#[test]
fn a_host_provider_declaring_derived_relations_is_asked_one_whose_columns_hold() {
    let access = root().data_access(&host(true)).unwrap();
    assert!(
        access
            .route(&scan(json!({"source": ["t"], "filter": "x > 1"})))
            .is_ok()
    );
    let Err(refused) = access.route(&scan(json!({
        "source": ["t"], "columns": [{"name": "x", "column": "x", "expression": "1"}]
    }))) else {
        panic!("a column reading both a column and an expression was routed");
    };
    assert_eq!(refused.code, ErrorCode::InvalidRequest);
}
