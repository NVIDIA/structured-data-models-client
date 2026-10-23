// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A column typed by its declaration first (T3.4 fix 1, round 1's P1-4):
//! a declaration the old client's `_to_dtype` reads as unsupported (a
//! Databricks struct, map or interval) is refused by name, rule
//! `column_type`, as a struct or map the values give is (register P1B-10,
//! P3B-1), whatever the values; any other declared type types the column.

use std::sync::Arc;

use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use sdm_contract::Detail;
use sdm_contract::catalog::CatalogColumn;
use sdm_contract::ops::Dtype;
use sdm_schema::infer::{self, TextShare, TextTimes, TextUnread};

struct NoTimes;

impl TextTimes for NoTimes {
    fn read(&self, values: &[Option<&str>], _utc: bool) -> Result<Vec<Option<i128>>, TextUnread> {
        Ok(vec![None; values.len()])
    }

    fn share(&self, _values: &[Option<&str>]) -> Result<TextShare, TextUnread> {
        Ok(TextShare {
            fewest: 0,
            listed: 0,
            most: 0,
            unshaped: 0,
        })
    }

    fn day_first_format(&self, _values: &[Option<&str>]) -> Option<String> {
        None
    }
}

fn previewed(declared: Option<Dtype>) -> Result<infer::Previewed, sdm_contract::CoreError> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("user_id", DataType::Int64, true),
        Field::new("profile", DataType::Utf8, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(vec![1, 2])) as ArrayRef,
            Arc::new(StringArray::from(vec!["a", "b"])),
        ],
    )
    .unwrap();
    let column = |name: &str, dtype: Option<Dtype>| CatalogColumn {
        name: name.to_owned(),
        declared_type: String::new(),
        not_null: false,
        primary_key: false,
        key_nullable: false,
        unique: false,
        dtype,
    };
    infer::preview(
        "users",
        &"users".into(),
        &[column("user_id", None), column("profile", declared)],
        &[],
        &schema,
        &[batch],
        &NoTimes,
        infer::Floats::NanValue,
    )
}

#[test]
fn a_declared_unsupported_type_is_refused_by_name_whatever_the_values() {
    let refused = previewed(Some(Dtype::Unsupported)).unwrap_err();
    assert_eq!(refused.fields["rule"], Detail::text("column_type"));
    assert!(previewed(Some(Dtype::String)).is_ok());
    assert!(previewed(None).is_ok());
}
