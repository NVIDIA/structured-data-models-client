// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A column typed by its declaration first ( fix 1, round 1's P1-4):
//! a declaration the old client's `_to_dtype` reads as unsupported (a
//! Databricks struct, map or interval) is refused by name, rule
//! `column_type`, as a struct or map the values give is (B-10,
//! P3B-1), whatever the values; any other declared type types the column.

use std::sync::Arc;

use arrow_array::{ArrayRef, Int64Array, NullArray, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use sdm_contract::catalog::CatalogColumn;
use sdm_contract::ops::Dtype;
use sdm_contract::{Detail, Scalar, Sentence};
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
    previewed_as(declared, Arc::new(StringArray::from(vec!["a", "b"])))
}

fn previewed_as(
    declared: Option<Dtype>,
    profile: ArrayRef,
) -> Result<infer::Previewed, sdm_contract::CoreError> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("user_id", DataType::Int64, true),
        Field::new("profile", profile.data_type().clone(), true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int64Array::from(vec![1, 2])) as ArrayRef, profile],
    )
    .unwrap();
    let column = |name: &str, dtype: Option<Dtype>| CatalogColumn {
        name: name.to_owned(),
        declared_type: match dtype {
            Some(Dtype::Unsupported) => "VARIANT".to_owned(),
            _ => String::new(),
        },
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

#[test]
fn a_refused_column_type_names_the_column_its_table_and_its_type() {
    let declared = previewed(Some(Dtype::Unsupported)).unwrap_err();
    let read = previewed_as(None, Arc::new(NullArray::new(2))).unwrap_err();
    for (refused, column_type) in [(declared, "VARIANT"), (read, "Null")] {
        assert_eq!(refused.sentence, Sentence::ColumnTypeUnsupported);
        assert_eq!(refused.fields["rule"], Detail::text("column_type"));
        assert_eq!(refused.fields["column"], Detail::text("profile"));
        assert_eq!(
            refused.fields["source"],
            Detail::List(vec![Scalar::Utf8("users".into())])
        );
        assert_eq!(refused.fields["column_type"], Detail::text(column_type));
    }
}
