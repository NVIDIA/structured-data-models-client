// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! FR P3-8's admission edge (migration-plan.md section 8, T2.12's row),
//! held from outside the implementation by the package's test author: an
//! `arrow.bool8` tag under run-end encoding.
//!
//! Admission reads an `int8` field tagged `arrow.bool8` as `boolean`
//! (`sdm-host`'s `admission.rs`, its table's first row; held by
//! `t136_bool8.rs` at the top level and on every nested child DuckDB
//! writes) and decodes a `run_end_encoded<R, V>` column to `V`, admitted
//! (its last row). A run-end column whose values field carries the tag is
//! decoded by its values' type alone, so the tag is dropped and the column
//! is admitted as `int8`, its booleans read as numbers. No reference
//! writes one (DuckDB never does), so the expectation is admission's own
//! rule: the decoded column is the one the same rows give as a plain
//! `arrow.bool8` column, `boolean`, every value and null kept; an untagged
//! run-end `int8` stays `int8` (the control).
//!
//! Gating: `contract/tests/t212-pending.json`'s `pending` names, as
//! `rust:<test>`, the tests that failed when written; each must still
//! fail (strict), and the implementer removes each as it passes.

#![allow(
    clippy::disallowed_methods,
    clippy::disallowed_types,
    reason = "test code: reads the gating list, and Arrow's field metadata is a HashMap"
)]

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Int32Type;
use arrow_array::{Array, ArrayRef, Int8Array, Int32Array, RecordBatch, RunArray};
use arrow_schema::{DataType, Field, Schema};
use sdm_core::ArrowData;
use sdm_host::{admit, admit_from, admitted_type};
use serde_json::Value;

const NAME: &str = "ARROW:extension:name";
const METADATA: &str = "ARROW:extension:metadata";

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

fn pending(test: &str) -> bool {
    let text = std::fs::read_to_string(root().join("contract/tests/t212-pending.json")).unwrap();
    let gating: Value = serde_json::from_str(&text).unwrap();
    gating["pending"]
        .as_array()
        .unwrap()
        .iter()
        .any(|t| t == &Value::from(format!("rust:{test}")))
}

/// Runs `check`; a pending test must fail, any other must pass.
fn gated(test: &str, check: impl FnOnce() + std::panic::UnwindSafe) {
    let outcome = std::panic::catch_unwind(check);
    match (outcome, pending(test)) {
        (Ok(()), true) => panic!(
            "{test} passes now: take rust:{test} out of \
             contract/tests/t212-pending.json's pending"
        ),
        (Err(failure), false) => std::panic::resume_unwind(failure),
        _ => {}
    }
}

/// The values field of a run-end column: `int8`, tagged `arrow.bool8`
/// (with the spec's empty metadata) when `tagged`.
fn values_field(tagged: bool) -> Field {
    let field = Field::new("values", DataType::Int8, true);
    if !tagged {
        return field;
    }
    field.with_metadata(HashMap::from([
        (NAME.to_owned(), "arrow.bool8".to_owned()),
        (METADATA.to_owned(), String::new()),
    ]))
}

/// Six rows in four runs: 0 twice, 1, null, 2 twice; and what the same
/// rows read as a plain `arrow.bool8` column (any non-zero value true).
fn runs(tagged: bool) -> (ArrayRef, Vec<Option<bool>>) {
    let ends = Int32Array::from(vec![2, 3, 4, 6]);
    let values = Int8Array::from(vec![Some(0), Some(1), None, Some(2)]);
    let data_type = DataType::RunEndEncoded(
        Arc::new(Field::new("run_ends", DataType::Int32, false)),
        Arc::new(values_field(tagged)),
    );
    let plain = RunArray::<Int32Type>::try_new(&ends, &values).unwrap();
    let data = plain
        .to_data()
        .into_builder()
        .data_type(data_type)
        .build()
        .unwrap();
    let array: ArrayRef = Arc::new(RunArray::<Int32Type>::from(data));
    let read = vec![
        Some(false),
        Some(false),
        Some(true),
        None,
        Some(true),
        Some(true),
    ];
    (array, read)
}

fn data(array: ArrayRef) -> ArrowData {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "f",
        array.data_type().clone(),
        true,
    )]));
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![array]).unwrap();
    ArrowData::new(schema, vec![batch])
}

/// FR P3-8: a run-end column whose values carry `arrow.bool8` is admitted
/// as `boolean`, as the same rows given as a plain `arrow.bool8` column
/// are: its type, both ways a host admits (no vendor, and DuckDB's), and
/// every decoded value and null; the tag's keys are not carried onto a
/// field that holds booleans.
#[test]
fn a_bool8_tag_under_run_end_encoding_is_read_as_boolean() {
    gated(
        "a_bool8_tag_under_run_end_encoding_is_read_as_boolean",
        || {
            let (array, read) = runs(true);
            let wanted = admitted_type("f", array.data_type()).unwrap();
            assert_eq!(wanted, DataType::Boolean, "admitted as {wanted}");
            let data = data(array);
            for admitted in [admit(&data), admit_from(&data, Some("DuckDB"))] {
                let admitted = admitted.unwrap();
                let field = admitted.schema.field(0);
                assert_eq!(field.data_type(), &DataType::Boolean, "{field:?}");
                assert!(
                    !field.metadata().contains_key(NAME),
                    "the tag is not kept on a boolean: {field:?}"
                );
                let column = admitted.batches[0].column(0);
                let got: Vec<Option<bool>> = column.as_boolean().iter().collect();
                assert_eq!(got, read);
            }
        },
    );
}

/// The control: an untagged run-end `int8` column is admitted as `int8`,
/// its runs decoded.
#[test]
fn an_untagged_run_end_int8_stays_int8() {
    gated("an_untagged_run_end_int8_stays_int8", || {
        let (array, _) = runs(false);
        assert_eq!(
            admitted_type("f", array.data_type()).unwrap(),
            DataType::Int8
        );
        let data = data(array);
        for admitted in [admit(&data), admit_from(&data, Some("DuckDB"))] {
            let admitted = admitted.unwrap();
            let column = admitted.batches[0].column(0);
            let got: Vec<Option<i8>> = column
                .as_primitive::<arrow_array::types::Int8Type>()
                .iter()
                .collect();
            assert_eq!(got, vec![Some(0), Some(0), Some(1), None, Some(2), Some(2)]);
        }
    });
}
