// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Query validation across authored schema types and relationship names.
#[allow(
    dead_code,
    reason = "shared helpers include schemas used by other test binaries"
)]
mod pql_support;
use pql_support::{NoZones, facts, profile};
use sdm_contract::ops::{Dtype, QueryColumn, QuerySchema, Severity, Span, Stype};
use sdm_core::pql::resolve;
use serde_json::json;

fn shop() -> QuerySchema {
    serde_json::from_value(json!({
        "tables": [
            {"name":"orders", "primary_key":"order_id", "time_column":"ts", "end_time_column":null,
             "columns":[
                 {"name":"order_id","dtype":"int","stype":"ID"},
                 {"name":"user_id","dtype":"int","stype":"ID"},
                 {"name":"price","dtype":"float","stype":"numerical"},
                 {"name":"ts","dtype":"time","stype":"timestamp"}]},
            {"name":"users", "primary_key":"user_id", "time_column":null, "end_time_column":null,
             "columns":[
                 {"name":"user_id","dtype":"int","stype":"ID"},
                 {"name":"age","dtype":"int","stype":"numerical"},
                 {"name":"country","dtype":"string","stype":"categorical"}]}
        ],
        "links":[[{"table":"orders","column":"user_id"},{"table":"users","column":"user_id"}]]
    }))
    .unwrap()
}

const SUM: &str = "PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id";
fn accepts(schema: &QuerySchema, query: &str) {
    let result = resolve(query, schema, &profile("RELATIONAL"), &facts(), &NoZones).unwrap();
    assert!(result.query.is_some(), "{query}: {:?}", result.diagnostics);
    assert!(
        !result
            .diagnostics
            .iter()
            .any(|d| d.severity == Severity::Error)
    );
}

#[test]
fn regression_binary_and_categorical_queries_resolve() {
    for query in [
        SUM,
        "PREDICT COUNT(orders.*, 0, 30, days) > 0 FOR EACH users.user_id",
        "PREDICT users.country FOR EACH users.user_id",
    ] {
        accepts(&shop(), query);
    }
}

#[test]
fn key_types_and_renamed_relationships_resolve() {
    for dtype in [Dtype::Int, Dtype::String, Dtype::Float] {
        let mut schema = shop();
        schema.tables[0].columns[1].dtype = dtype;
        schema.tables[1].columns[0].dtype = dtype;
        accepts(&schema, SUM);
    }
    let mut schema = shop();
    schema.tables[0].columns[0].name = "oid".into();
    schema.tables[0].primary_key = Some("oid".into());
    schema.tables[0].columns[1].name = "buyer".into();
    schema.tables[1].columns[0].name = "uid".into();
    schema.tables[1].primary_key = Some("uid".into());
    schema.links[0][0].column = "buyer".into();
    schema.links[0][1].column = "uid".into();
    accepts(
        &schema,
        "PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.uid",
    );
}

#[test]
fn feature_types_and_declared_text_times_resolve() {
    for (name, dtype, stype) in [
        ("f", Dtype::Bool, Stype::Categorical),
        ("f", Dtype::Int, Stype::Categorical),
        ("tier", Dtype::Float, Stype::Categorical),
        ("f", Dtype::Time, Stype::Timestamp),
    ] {
        let mut schema = shop();
        schema.tables[1].columns.push(QueryColumn {
            name: name.into(),
            dtype,
            stype,
        });
        accepts(
            &schema,
            if name == "tier" {
                "PREDICT users.tier FOR EACH users.user_id"
            } else {
                SUM
            },
        );
    }
    for renamed in [false, true] {
        let mut schema = shop();
        schema.tables[0].columns[3].dtype = Dtype::String;
        if renamed {
            schema.tables[0].columns[3].name = "happened".into();
            schema.tables[0].time_column = Some("happened".into());
        }
        accepts(&schema, SUM);
    }
    let mut schema = shop();
    for name in ["f_date", "f_ts"] {
        schema.tables[1].columns.push(QueryColumn {
            name: name.into(),
            dtype: Dtype::Time,
            stype: Stype::Timestamp,
        });
    }
    schema.tables[1].time_column = Some("f_date".into());
    accepts(&schema, SUM);
}

#[test]
fn unknown_column_has_a_precise_diagnostic() {
    let result = resolve(
        "PREDICT SUM(orders.cost, 0, 30, days) FOR EACH users.user_id",
        &shop(),
        &profile("RELATIONAL"),
        &facts(),
        &NoZones,
    )
    .unwrap();
    assert!(result.query.is_none());
    let errors: Vec<_> = result
        .diagnostics
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .collect();
    assert_eq!(errors.len(), 1);
    assert_eq!(errors[0].code, "PQL_SCHEMA_UNKNOWN_COLUMN");
    assert_eq!(
        errors[0].span,
        Some(Span {
            start_row: 1,
            start_col: 12,
            end_row: 1,
            end_col: 12
        })
    );
}
