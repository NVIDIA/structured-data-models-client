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
    serde_json::from_str(r#"{"provider":"sql","dialect":"databricks","namespace":["main","test"],"timestamp_encoding":"microseconds"}"#).unwrap()
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
fn instant_and_wall_projection_preserve_their_distinct_meaning_and_positions() {
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
    assert_eq!(value["result_conversions"][0]["source_kind"], "wall_time");
    assert_eq!(
        value["result_conversions"][0]["conversion"],
        "timestamp_micros"
    );
    assert_eq!(value["result_conversions"][1]["column"], 2);
    assert_eq!(value["result_conversions"][1]["name"], "zoned");
    assert_eq!(value["result_conversions"][1]["source_kind"], "instant");
    let sql = value["statements"][0]["text"].as_str().unwrap();
    assert!(
        sql.contains("DATEDIFF(CAST(`at``time` AS DATE), DATE '1970-01-01')"),
        "{sql}"
    );
    assert!(
        sql.contains("CAST(MINUTE(`at``time`) AS BIGINT) * 60000000"),
        "{sql}"
    );
    assert!(
        sql.contains("CAST(EXTRACT(SECOND FROM `at``time`) * 1000000 AS BIGINT)"),
        "{sql}"
    );
    assert!(sql.contains("UNIX_MICROS(`zoned`) AS `zoned`"), "{sql}");
    assert!(!sql.contains("TIMESTAMPDIFF(MICROSECOND"), "{sql}");
    assert!(sql.ends_with(" LIMIT 5"));
    assert!(sql.contains("`id`, "));
    assert!(!sql.contains("CAST(`zoned` AS TIMESTAMP_NTZ)"), "{sql}");
    let native = ProviderChoice::sql("databricks");
    assert!(
        Data::new()
            .result_metadata(&native, &request, &plan)
            .unwrap()
            .is_none()
    );
}

#[test]
fn precise_timestamp_names_are_selected_from_metadata_not_output_names() {
    let request = DataRequest::Preview {
        table: Relation::of(vec!["main".into(), "test".into(), "events".into()]),
        columns: vec!["naive_name".into(), "utc_name".into(), "id".into()],
        limit: 1,
    };
    let choice = provider();
    let plan = Data::new().plan(&request, &choice).unwrap();
    let types = metadata(&[
        ("naive_name", "TIMESTAMP_LTZ(6)"),
        ("utc_name", "TIMESTAMP_NTZ(6)"),
        ("id", "bigint"),
    ]);
    let projected = Data::new()
        .project_result(&choice, &request, &plan, &[types])
        .unwrap();
    let value = serde_json::to_value(&projected).unwrap();
    assert_eq!(value["result_conversions"].as_array().unwrap().len(), 2);
    assert_eq!(value["result_conversions"][0]["source_kind"], "instant");
    assert_eq!(value["result_conversions"][1]["source_kind"], "wall_time");
    let sql = value["statements"][0]["text"].as_str().unwrap();
    assert!(sql.contains("UNIX_MICROS(`naive_name`)"), "{sql}");
    assert!(sql.contains("DATEDIFF(CAST(`utc_name` AS DATE)"), "{sql}");
    assert!(sql.contains(", `id` FROM"), "{sql}");
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
