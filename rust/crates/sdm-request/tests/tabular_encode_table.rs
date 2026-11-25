// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use arrow_array::{Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use sdm_contract::direct::{TabularColumn, TabularColumns};
use sdm_contract::ops::{Dtype, Stype};
use sdm_request::tabular::{body, encode_table};
use std::sync::Arc;

fn plan() -> TabularColumns {
    TabularColumns {
        dtypes: vec![("target".into(), Dtype::Int64)],
        columns: vec![TabularColumn {
            name: "target".into(),
            dtype: Dtype::Int64,
            stype: Stype::Numerical,
        }],
        target: "target".into(),
    }
}

fn batch(name: &str, values: Vec<i64>) -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(name, DataType::Int64, false)])),
        vec![Arc::new(Int64Array::from(values))],
    )
    .unwrap()
}

#[test]
fn selected_table_encoding_does_not_require_an_unrelated_half() {
    let plan = plan();
    let good = vec![batch("target", vec![1, 2])];
    let unrelated = vec![batch("not_in_resolved_schema", vec![3])];
    let encoded = encode_table(&plan, &good, |v| v.to_string()).unwrap();
    assert_eq!(
        String::from_utf8(encoded.to_bytes()).unwrap(),
        r#"{"format":"arrays","columns":["target"],"rows":[[1],[2]]}"#
    );
    assert!(encode_table(&plan, &unrelated, |v| v.to_string()).is_err());
    assert!(
        body(&plan, &good, &unrelated, &[], "regression", &[], |v| v
            .to_string())
        .is_err()
    );
    assert!(
        body(&plan, &unrelated, &good, &[], "regression", &[], |v| v
            .to_string())
        .is_err()
    );
}

#[test]
fn body_preserves_both_tables_and_context_failure_precedence() {
    let plan = plan();
    let context = vec![batch("target", vec![1]), batch("target", vec![2])];
    let scoring = vec![batch("target", vec![3])];
    let payload = body(&plan, &context, &scoring, &[], "regression", &[], |v| {
        v.to_string()
    })
    .unwrap();
    assert_eq!(
        payload.get("context").unwrap().get("instance_table"),
        Some(&encode_table(&plan, &context, |v| v.to_string()).unwrap())
    );
    assert_eq!(
        payload.get("predict").unwrap().get("instance_table"),
        Some(&encode_table(&plan, &scoring, |v| v.to_string()).unwrap())
    );
    let bad_context = vec![batch("bad_context", vec![1])];
    let bad_scoring = vec![batch("bad_scoring", vec![2])];
    let expected = encode_table(&plan, &bad_context, |v| v.to_string()).unwrap_err();
    let actual = body(
        &plan,
        &bad_context,
        &bad_scoring,
        &[],
        "regression",
        &[],
        |v| v.to_string(),
    )
    .unwrap_err();
    assert_eq!(actual, expected);
}
