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

/// A Databricks source leaving any part to the connection, a leading one
/// included, has its namespace read first, as the reference reads
/// `current_catalog()` and `current_schema()` and names every table in
/// full, as its catalog names a table a foreign key refers to (round 2's
/// Claude P0-1); a dialect whose relations need no part given is left as
/// it is.
#[test]
fn a_databricks_source_leaving_any_part_to_the_connection_is_settled_first() {
    let data = Data::new();
    for namespace in [
        json!([]),
        json!([null, null]),
        json!([null, "shop"]),
        json!(["shop"]),
    ] {
        assert!(
            data.unsettled(&choice("databricks", namespace.clone())),
            "{namespace}"
        );
    }
    assert!(data.unsettled(&choice("databricks", json!(["main", null]))));
    assert!(!data.unsettled(&choice("databricks", json!(["main", "shop"]))));
    // A Postgres source given no schema is settled too, by its session's
    // current schema (T3.7); one given its schema is not.
    assert!(data.unsettled(&choice("postgres", json!([]))));
    assert!(data.unsettled(&choice("postgres", json!([null]))));
    assert!(!data.unsettled(&choice("postgres", json!(["shop"]))));
    assert!(!data.unsettled(&choice("duckdb", json!([]))));
    assert_eq!(
        data.settled(
            &choice("databricks", json!([null, "shop"])),
            vec!["main".into(), "shop".into()]
        ),
        choice("databricks", json!(["main", "shop"]))
    );
}

/// A table the core reads under a namespace of its own, named in full, is
/// read so whatever parts the provider leaves to the connection (round 3:
/// a table read where it was described).
#[test]
fn a_table_named_in_full_is_read_whatever_the_provider_leaves_out() {
    let request: DataRequest = serde_json::from_value(json!({
        "request": "count_rows", "table": {"source": ["main", "shop", "orders"]}, "filters": []
    }))
    .unwrap();
    let Ok(FetchPlan::Sql { statements, .. }) =
        Data::new().plan(&request, &choice("databricks", json!(["main", null])))
    else {
        panic!("a table named in full is read");
    };
    assert_eq!(
        statements.last().unwrap().text,
        "SELECT COUNT(*) FROM `main`.`shop`.`orders`"
    );
}

/// Namespace reads on Databricks already resolve both parts and align a
/// shorthand to the innermost part. Settlement must retain that answer.
#[test]
fn databricks_settlement_preserves_full_resolved_namespaces() {
    let data = Data::new();
    for (given, resolved) in [
        (json!([null]), vec!["main", "public"]),
        (json!([null, null]), vec!["main", "public"]),
        (json!(["shop"]), vec!["main", "shop"]),
        (json!([null, "shop"]), vec!["main", "shop"]),
        (json!(["kept", null]), vec!["kept", "public"]),
        (json!(["kept", "shop"]), vec!["kept", "shop"]),
    ] {
        let provider = choice("databricks", given.clone());
        let settled = data.settled(&provider, resolved.iter().map(|p| (*p).to_owned()).collect());
        assert_eq!(settled, choice("databricks", json!(resolved)), "{given}");
        let FetchPlan::Sql { statements, .. } = data.plan(&preview(), &settled).unwrap() else { panic!("SQL") };
        assert!(statements[0].text.contains(&format!("FROM `{}`.`{}`.`t`", resolved[0], resolved[1])));
        let request: DataRequest = serde_json::from_value(json!({"request": "describe", "part": "tables", "tables": []})).unwrap();
        let FetchPlan::Sql { statements, .. } = data.plan(&request, &settled).unwrap() else { panic!("SQL") };
        assert!(statements[0].text.contains(&format!("`{}`.information_schema.tables", resolved[0])));
        assert!(statements[0].params.contains(&SqlParam::Value { value: Scalar::Utf8(resolved[1].to_owned()) }));
    }
}

#[test]
fn snowflake_raw_session_settlement_fills_short_namespaces_without_truncation() {
    let data = Data::new();
    for (given, expected) in [
        (json!([null]), json!(["SESSION_DB", "SESSION_SCHEMA"])),
        (json!(["NAMED_DB"]), json!(["NAMED_DB", "SESSION_SCHEMA"])),
        (json!(["NAMED_DB", null]), json!(["NAMED_DB", "SESSION_SCHEMA"])),
        (json!([null, "NAMED_SCHEMA"]), json!(["SESSION_DB", "NAMED_SCHEMA"])),
    ] {
        assert_eq!(data.settled(&choice("snowflake", given), vec!["SESSION_DB".into(), "SESSION_SCHEMA".into()]), choice("snowflake", expected));
    }
    assert_eq!(data.settled(&choice("snowflake", json!(["NAMED_DB", null])), vec![]), choice("snowflake", json!(["NAMED_DB"])));
}
