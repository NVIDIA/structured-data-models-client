// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Lossless JDBC transport is planned by the core from the warehouse's result types.

use std::sync::Arc;

use arrow_array::{ArrayRef, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use sdm_contract::data::{DataLayer, DataRequest, FetchPlan, ProviderChoice, Relation};
use sdm_data::Data;

fn provider() -> ProviderChoice {
    serde_json::from_str(r#"{"provider":"sql","dialect":"databricks","namespace":["main","test"],"timestamp_encoding":"naive_microseconds"}"#).unwrap()
}

fn metadata(types: &[(&str, &str)]) -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("col_name", DataType::Utf8, false),
            Field::new("data_type", DataType::Utf8, false),
            Field::new("comment", DataType::Utf8, true),
        ])),
        vec![
            Arc::new(StringArray::from(
                types.iter().map(|(n, _)| *n).collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(StringArray::from(
                types.iter().map(|(_, t)| *t).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::new_null(types.len())),
        ],
    )
    .unwrap()
}

#[test]
fn ntz_projection_keeps_names_positions_and_all_other_types() {
    let request = DataRequest::Preview {
        table: Relation::of(vec!["main".into(), "test".into(), "events".into()]),
        columns: vec!["id".into(), "at`time".into(), "zoned".into()],
        limit: 5,
    };
    let choice = provider();
    let plan = Data::new().plan(&request, &choice).unwrap();
    let before = Data::new()
        .result_metadata(&choice, &request, &plan)
        .unwrap()
        .unwrap();
    let FetchPlan::Sql { statements, .. } = before else {
        panic!("SQL")
    };
    assert!(
        statements
            .last()
            .unwrap()
            .text
            .starts_with("DESCRIBE QUERY SELECT ")
    );
    let types = metadata(&[
        ("id", "bigint"),
        ("at`time", "timestamp_ntz"),
        ("zoned", "timestamp"),
    ]);
    let projected = Data::new()
        .project_result(&choice, &request, &plan, &[types])
        .unwrap();
    let value = serde_json::to_value(&projected).unwrap();
    assert_eq!(value["result_conversions"][0]["column"], 1);
    assert_eq!(value["result_conversions"][0]["name"], "at`time");
    assert_eq!(
        value["result_conversions"][0]["conversion"],
        "timestamp_micros"
    );
    let sql = value["statements"][0]["text"].as_str().unwrap();
    assert!(sql.contains("timestampdiff(MICROSECOND, TIMESTAMP_NTZ '1970-01-01 00:00:00', CAST(`at``time` AS TIMESTAMP_NTZ)) AS `at``time`"), "{sql}");
    assert!(sql.ends_with(" LIMIT 5"));
    assert!(sql.contains("`id`, "));
    assert!(sql.contains(", `zoned` FROM"));
    let native = ProviderChoice::sql("databricks");
    assert!(
        Data::new()
            .result_metadata(&native, &request, &plan)
            .unwrap()
            .is_none()
    );
}

#[test]
fn metadata_without_ntz_keeps_the_original_plan_exactly() {
    let request = DataRequest::Scan {
        table: Relation::of(vec!["main".into(), "test".into(), "events".into()]),
        columns: vec!["id".into()],
        filters: vec![],
        limit: None,
    };
    let choice = provider();
    let plan = Data::new().plan(&request, &choice).unwrap();
    assert_eq!(
        Data::new()
            .project_result(&choice, &request, &plan, &[metadata(&[("id", "bigint")])])
            .unwrap(),
        plan
    );
}
