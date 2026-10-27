// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Actual public selected-row/index Arrow inputs run through the shared engine.

use arrow_array::{ArrayRef, Int8Array, Int64Array};
use arrow_ipc::reader::FileReader;
use sdm_engine::index_context::{
    IndexOrderError, MissingValues, NumpyProducer, OriginalIndexFacts,
};
use serde_json::Value;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

fn rows(value: &Value) -> Vec<usize> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|n| n.as_u64().unwrap() as usize)
        .collect()
}

fn attachment(case: &Value, role: &str) -> ArrayRef {
    let descriptor = case["attachments"]
        .as_array()
        .unwrap()
        .iter()
        .find(|value| value["role"] == role)
        .unwrap();
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/index_context")
        .join(descriptor["path"].as_str().unwrap());
    let mut reader = FileReader::try_new(File::open(path).unwrap(), None).unwrap();
    let batch = reader.next().unwrap().unwrap();
    assert!(reader.next().is_none());
    batch.column(0).clone()
}

#[test]
fn public_selected_ordinals_reach_exact_original_context_order() {
    let recorded: Value = serde_json::from_str(include_str!("index_context/cases.json")).unwrap();
    assert_eq!(recorded["source_profile"], "numpy_253_x86_v3_gnu14");
    let cases = recorded["cases"].as_array().unwrap();
    assert!(
        cases.len() >= 30,
        "ordinary widths, nullable, category and temporal inputs must stay represented"
    );
    for case in cases {
        let values = attachment(case, "values");
        let count = case["source_rows"].as_u64().unwrap() as usize;
        let producer = NumpyProducer::Intel7Avx2Gnu14;
        let facts = match case["layout"].as_str().unwrap() {
            "categorical" => OriginalIndexFacts::categorical(
                values,
                attachment(case, "dictionary"),
                case["ordered"].as_bool().unwrap(),
                count,
                producer,
            ),
            "temporal" => OriginalIndexFacts::temporal(values, count),
            "range" => OriginalIndexFacts::range(values, count, producer),
            "numpy" => OriginalIndexFacts::numpy(
                values,
                count,
                if case["missing"] == "validity" {
                    MissingValues::Validity
                } else {
                    MissingValues::NanOrValidity
                },
                producer,
            ),
            other => panic!("unexpected recorded layout {other}"),
        }
        .unwrap_or_else(|error| panic!("{}: {error:?}", case["name"]));
        let selected = rows(&case["selected_original_rows"]);
        let expected = rows(&case["expected_context_original_rows"]);
        assert_eq!(facts.source_rows(), count);
        assert_eq!(
            facts.order_selected(&selected).unwrap(),
            expected,
            "{}",
            case["name"]
        );
    }
}

#[test]
fn original_row_bounds_and_snapshot_lengths_are_checked() {
    assert!(matches!(
        OriginalIndexFacts::numpy(
            Arc::new(Int64Array::from(vec![1, 2])),
            3,
            MissingValues::NanOrValidity,
            NumpyProducer::Scalar
        ),
        Err(IndexOrderError::RowCount {
            expected: 3,
            actual: 2
        })
    ));
    let facts = OriginalIndexFacts::numpy(
        Arc::new(Int64Array::from(vec![1, 2])),
        2,
        MissingValues::NanOrValidity,
        NumpyProducer::Scalar,
    )
    .unwrap();
    assert!(matches!(
        facts.order_selected(&[0, 2]),
        Err(IndexOrderError::RowBounds {
            row: 2,
            source_rows: 2
        })
    ));
}

#[test]
fn categorical_attachment_identity_and_unused_levels_are_retained() {
    let dictionary: ArrayRef = Arc::new(Int64Array::from_iter_values(0..126));
    let facts = OriginalIndexFacts::categorical(
        Arc::new(Int8Array::from(vec![1, -1, 0])),
        dictionary.clone(),
        false,
        3,
        NumpyProducer::Intel7Avx2Gnu14,
    )
    .unwrap();
    assert!(Arc::ptr_eq(facts.dictionary().unwrap(), &dictionary));
    assert_eq!(facts.dictionary().unwrap().len(), 126);
}
