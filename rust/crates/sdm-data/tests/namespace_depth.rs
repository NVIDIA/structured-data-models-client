// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! One namespace model (design fix C, T3.4): each dialect reads a relation
//! of one part under as many namespace parts as its sources take, its
//! namespace depth, apart from the parts a relation may have; a part too
//! many is refused by name, and a part left to the connection (`null`)
//! before the parts given is read in the connection's current one.
//! DuckDB and SQLite take 0, Postgres 1 (schema), Databricks 2 (catalog,
//! schema).

use sdm_contract::Scalar;
use sdm_contract::data::{DataLayer, DataRequest, FetchPlan, ProviderChoice, SqlParam};
use sdm_data::Data;
use serde_json::json;

fn choice(dialect: &str, namespace: serde_json::Value) -> ProviderChoice {
    serde_json::from_value(json!({"provider": "sql", "dialect": dialect, "namespace": namespace}))
        .unwrap()
}

fn preview() -> DataRequest {
    serde_json::from_value(json!({
        "request": "preview", "table": {"source": ["t"]}, "columns": ["a"], "limit": 1
    }))
    .unwrap()
}

fn text(dialect: &str, namespace: serde_json::Value) -> Result<String, String> {
    match Data::new().plan(&preview(), &choice(dialect, namespace)) {
        Ok(FetchPlan::Sql { statements, .. }) => Ok(statements.last().unwrap().text.clone()),
        Ok(other) => Err(format!("{other:?}")),
        Err(error) => Err(format!("{error:?}")),
    }
}

/// The refusal names the namespace as the field it refuses.
fn refused_by_namespace(result: Result<String, String>) {
    let error = result.expect_err("planned a namespace past the dialect's depth");
    assert!(error.contains("namespace"), "{error}");
}

#[test]
fn each_dialect_reads_a_relation_under_its_namespace_depth() {
    // Depth 0: no namespace to read under.
    assert_eq!(
        text("duckdb", json!([])).unwrap(),
        "SELECT \"a\" FROM \"t\" LIMIT 1"
    );
    refused_by_namespace(text("duckdb", json!(["s"])));
    refused_by_namespace(text("sqlite", json!(["s"])));
    // Depth 1: the schema.
    assert!(
        text("postgres", json!(["s"]))
            .unwrap()
            .contains("FROM \"s\".\"t\"")
    );
    refused_by_namespace(text("postgres", json!(["c", "s"])));
    // Depth 2: the catalog and the schema, each quoted alone.
    assert!(
        text("databricks", json!(["c", "s"]))
            .unwrap()
            .contains("FROM `c`.`s`.`t`")
    );
    assert!(
        text("databricks", json!([null, "s"]))
            .unwrap()
            .contains("FROM `s`.`t`")
    );
    assert!(text("databricks", json!([])).unwrap().contains("FROM `t`"));
    refused_by_namespace(text("databricks", json!(["x", "c", "s"])));
    // A part given inside one left to the connection writes no relation.
    refused_by_namespace(text("databricks", json!(["c", null])));
}

/// A Databricks source naming its catalog alone (`DatabricksSource(conn,
/// catalog='main')`, round 1's P1-1): its namespace is read, the catalog
/// given bound and the schema the session's current one, which the core
/// then reads every relation under; deeper than two parts still refused.
#[test]
fn a_catalog_alone_reads_its_namespace_with_the_session_s_current_schema() {
    let request: DataRequest = serde_json::from_value(json!({
        "request": "describe", "tables": [], "part": "namespace"
    }))
    .unwrap();
    let Ok(FetchPlan::Sql { statements, .. }) =
        Data::new().plan(&request, &choice("databricks", json!(["main", null])))
    else {
        panic!("the namespace of a catalog alone is read");
    };
    let statement = statements.last().unwrap();
    assert!(
        statement.text.contains("SELECT 0 AS o, ? AS n"),
        "{}",
        statement.text
    );
    assert!(
        statement
            .text
            .contains("SELECT 1 AS o, current_schema() AS n"),
        "{}",
        statement.text
    );
    assert_eq!(
        statement.params,
        vec![SqlParam::Value {
            value: Scalar::Utf8("main".to_owned())
        }]
    );
    assert!(
        Data::new()
            .plan(&request, &choice("databricks", json!(["x", "main", null])))
            .is_err()
    );
}
