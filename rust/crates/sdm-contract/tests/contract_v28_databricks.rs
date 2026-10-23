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

/// The namespace a description's tables were read under, where the run
/// read parts the provider left to the connection, travels with the
/// description, so materialize reads the same tables (round 2's Codex
/// P0-2); left out, none was read.
#[test]
fn a_description_carries_the_namespace_its_tables_were_read_under() {
    use sdm_contract::ops::SourceDescription;
    let described: SourceDescription = serde_json::from_value(serde_json::json!({
        "tables": [], "links": [], "namespace": ["main", "shop"]
    }))
    .unwrap();
    assert_eq!(
        described.namespace,
        Some(vec!["main".to_owned(), "shop".to_owned()])
    );
    let plain: SourceDescription =
        serde_json::from_value(serde_json::json!({"tables": [], "links": []})).unwrap();
    assert_eq!(plain.namespace, None);
    assert!(!serde_json::to_string(&plain).unwrap().contains("namespace"));
}

/// The columns a catalog read found, marked on a relation (round 2's
/// Claude P2-2), apart from a derived relation's own columns.
#[test]
fn a_relation_marks_the_columns_its_catalog_found() {
    use sdm_contract::data::Relation;
    let found: Relation = serde_json::from_value(serde_json::json!({
        "source": ["t"], "found": true, "columns": [{"name": "a", "column": "a"}]
    }))
    .unwrap();
    assert!(found.found);
    assert!(!Relation::named("t").found);
    assert!(
        !serde_json::to_string(&Relation::named("t"))
            .unwrap()
            .contains("found")
    );
}

/// A relation marked `found` holds the columns a catalog found, each read by
/// its own name, with no filter; one marked so that renames a column, reads
/// an expression or filters its rows is refused by name, never read as the
/// bare table (round 3's Codex P2-1).
#[test]
fn a_found_relation_is_its_catalog_s_columns_or_refused() {
    use sdm_contract::data::{DataRequest, Relation, refuse_derived};
    let scan = |table: serde_json::Value| -> (DataRequest, Relation) {
        let relation: Relation = serde_json::from_value(table).unwrap();
        let request: DataRequest = serde_json::from_value(serde_json::json!({
            "request": "scan", "table": relation, "columns": ["age"], "filters": [], "limit": null
        }))
        .unwrap();
        (request, relation)
    };
    let (request, found) = scan(serde_json::json!({
        "source": ["users"], "found": true, "columns": [{"name": "age", "column": "age"}]
    }));
    assert!(refuse_derived(&request, "table", &found, true).is_none());
    for contradicting in [
        serde_json::json!({"source": ["users"], "found": true,
            "columns": [{"name": "age", "column": "age"}], "filter": "age > 30"}),
        serde_json::json!({"source": ["users"], "found": true,
            "columns": [{"name": "years", "column": "age"}]}),
        serde_json::json!({"source": ["users"], "found": true,
            "columns": [{"name": "age", "expression": "age + 1"}]}),
    ] {
        let (request, relation) = scan(contradicting);
        let refused = refuse_derived(&request, "table", &relation, true)
            .expect("a found relation that is not its catalog's columns is refused");
        assert_eq!(refused.code, sdm_contract::ErrorCode::InvalidRequest);
    }
}
