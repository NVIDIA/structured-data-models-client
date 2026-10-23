// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Contract v28's Databricks vocabulary (T3.4, design choice 3): derived
//! relations, opaque SQL text the core carries but never reads, refused by
//! name for a provider that does not read them; the capabilities and
//! request fields the old client's warehouse samplers need; a provider
//! error saying the source refused an expression. Each optional, so a v27
//! message reads the same.

use sdm_contract::Detail;
use sdm_contract::data::{
    DERIVED_RULE, DataRequest, ProviderCapabilities, ProviderError, Relation, refuse_derived,
};
use serde_json::json;

fn read<T: serde::de::DeserializeOwned>(value: serde_json::Value) -> T {
    serde_json::from_value(value).unwrap()
}

fn written<T: serde::Serialize>(value: &T) -> serde_json::Value {
    serde_json::to_value(value).unwrap()
}

#[test]
fn a_derived_relation_round_trips_and_a_plain_one_writes_as_before() {
    let derived = json!({
        "source": ["c", "s", "orders"],
        "columns": [
            {"name": "order_id", "column": "id"},
            {"name": "net", "expression": "price - discount"}
        ],
        "filter": "status = 'shipped'"
    });
    let relation: Relation = read(derived.clone());
    assert!(relation.is_derived());
    assert_eq!(written(&relation), derived);
    let plain: Relation = read(json!({"source": ["t"]}));
    assert!(!plain.is_derived());
    assert_eq!(written(&plain), json!({"source": ["t"]}));
    assert_eq!(Relation::of(vec!["c".into(), "s".into()]).dotted(), "c.s");
}

#[test]
fn a_derived_relation_is_refused_by_name_where_it_is_not_read() {
    let request: DataRequest = read(json!({
        "request": "preview", "table": {"source": ["t"], "filter": "x > 1"},
        "columns": ["x"], "limit": 1
    }));
    let DataRequest::Preview { table, .. } = &request else {
        unreachable!()
    };
    let refused = refuse_derived(&request, "table", table, false).unwrap();
    assert_eq!(refused.fields["rule"], Detail::text(DERIVED_RULE));
    assert!(refuse_derived(&request, "table", table, true).is_none());
    // A column reading both a column and an expression, or named twice.
    for columns in [
        json!([{"name": "a", "column": "a", "expression": "a + 1"}]),
        json!([{"name": "a", "column": "a"}, {"name": "a", "column": "b"}]),
        json!([{"name": "a"}]),
    ] {
        let relation: Relation = read(json!({"source": ["t"], "columns": columns}));
        assert!(refuse_derived(&request, "table", &relation, true).is_some());
    }
}

#[test]
fn the_warehouse_samplers_fields_are_optional_and_round_trip() {
    let capabilities: ProviderCapabilities = read(json!({
        "optional": [], "max_keys_per_request": null, "sampling": "seeded_stable",
        "max_seed": 1, "text_times": null, "null_order": null, "index_advice": null,
        "derived_relations": true, "sample_rows": true, "neighbor_lookback": true
    }));
    assert!(capabilities.derived_relations && capabilities.sample_rows);
    assert!(capabilities.neighbor_lookback);
    let before: ProviderCapabilities = read(json!({
        "optional": [], "max_keys_per_request": null, "sampling": "seeded_stable",
        "max_seed": 1, "text_times": null, "null_order": null, "index_advice": null
    }));
    assert!(!before.derived_relations && !before.sample_rows && !before.neighbor_lookback);
    let sample: DataRequest = read(json!({
        "request": "sample_entities", "table": {"source": ["t"]}, "key": ["k"], "n": 2,
        "seed": {"t": "u64", "v": "1"}, "columns": ["k"], "filters": [], "rows": 100
    }));
    assert!(matches!(
        sample,
        DataRequest::SampleEntities {
            rows: Some(100),
            ..
        }
    ));
    let error: ProviderError = read(json!({
        "kind": "data_source", "retryable": false, "sqlstate": "42703",
        "vendor_code": null, "refuses_expression": true
    }));
    assert!(error.refuses_expression);
}
