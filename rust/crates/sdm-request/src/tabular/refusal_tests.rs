// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use std::sync::Arc;

use arrow_array::{Int64Array, StringArray};

#[test]
fn numeric_storage_guard_has_three_negative_readers_and_primitive_twins() {
    let strings: ArrayRef = Arc::new(StringArray::from(vec!["1"]));
    let errors = [
        raw(&strings, 0).unwrap_err(),
        cell(Scalar::Utf8("1".into()), Dtype::String, "value", |v| {
            v.to_string()
        })
        .unwrap_err(),
        cell(Scalar::Utf8("1".into()), Dtype::Int64, "value", |v| {
            v.to_string()
        })
        .unwrap_err(),
    ];
    for error in errors {
        assert_eq!(error.sentence, Sentence::NotYetPredict);
        assert_eq!(error.fields["rule"], Detail::text("direct_numeric_storage"));
    }
    let integers: ArrayRef = Arc::new(Int64Array::from(vec![1]));
    assert_eq!(raw(&integers, 0).unwrap(), Scalar::Int64(1));
    assert_eq!(
        cell(Scalar::UInt64(u64::MAX), Dtype::String, "value", |v| v
            .to_string())
        .unwrap(),
        Value::text(&u64::MAX.to_string())
    );
    assert_eq!(
        cell(Scalar::Bool(true), Dtype::String, "value", |v| v
            .to_string())
        .unwrap(),
        Value::text("True")
    );
}
