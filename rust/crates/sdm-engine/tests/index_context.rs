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
fn actual_old_and_new_entry_profiles_preserve_their_public_context_orders() {
    let recorded: Value =
        serde_json::from_str(include_str!("index_context/profile_cases.json")).unwrap();
    let cases = recorded["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 4);
    for case in cases {
        assert_eq!(case["source_rows"], 400);
        assert_eq!(case["candidate_count"], 399);
        assert_eq!(case["missing_count"], 6);
        assert_eq!(case["nonmissing_count"], 294);
        let producer = match case["source_profile"].as_str().unwrap() {
            "numpy_226_avx2_intel5_entry_shared_gnu" => NumpyProducer::Intel5Avx2SharedGnu,
            "numpy_253_x86_v3_gnu14" => NumpyProducer::Intel7Avx2Gnu14,
            other => panic!("unexpected source-proven profile {other}"),
        };
        let facts = if case["layout"] == "categorical" {
            let facts = OriginalIndexFacts::categorical(
                attachment(case, "values"),
                attachment(case, "dictionary"),
                case["ordered"].as_bool().unwrap(),
                400,
                producer,
            )
            .unwrap();
            assert_eq!(
                facts.code_width(),
                Some(sdm_engine::index_codes::CodeWidth::I32)
            );
            assert_eq!(facts.dictionary().unwrap().len(), 32767);
            facts
        } else {
            OriginalIndexFacts::numpy(
                attachment(case, "values"),
                400,
                MissingValues::NanOrValidity,
                producer,
            )
            .unwrap()
        };
        let selected = rows(&case["selected_original_rows"]);
        let expected = rows(&case["expected_context_original_rows"]);
        assert_eq!(selected.len(), 300);
        assert_eq!(expected.len(), 300);
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
