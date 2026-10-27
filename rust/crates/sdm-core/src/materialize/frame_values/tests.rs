// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Authored before native profile/order and context range consumers.

use super::*;
use sdm_engine::words::{CasingProfile, UnicodeDataVersion};

#[test]
fn labelled_link_rows_preserve_typed_order_duplicates_and_decimal_scale() {
    use arrow_array::{ArrayRef, Int64Array};
    use sdm_engine::labels::{Labels, LinkValue};
    for values in [
        vec![
            vec![LinkValue::Int(9), LinkValue::Int(1)],
            vec![LinkValue::Int(9)],
        ],
        vec![
            vec![
                LinkValue::Text("red".into()),
                LinkValue::Text("blue".into()),
            ],
            vec![LinkValue::Text("red".into())],
        ],
        vec![
            vec![LinkValue::Decimal(900719925474099312, 2)],
            vec![LinkValue::Decimal(-1, 2)],
        ],
    ] {
        let labels = Labels::Lists(values);
        let keys: ArrayRef = Arc::new(Int64Array::from(vec![1, 2]));
        let rows = labelled_rows(&keys, &[10, 20], &labels).unwrap();
        let decoded = read_labels(&rows).unwrap();
        assert_eq!(decoded.labels, labels);
        assert_eq!(decoded.anchors, [10, 20]);
    }
}

#[test]
fn every_builtin_database_and_actual_result_storage_selects_core_casing() {
    for (database, version) in [
        ("13.0.0", UnicodeDataVersion::V13_0),
        ("14.0.0", UnicodeDataVersion::V14_0),
        ("15.0.0", UnicodeDataVersion::V15_0),
        ("15.1.0", UnicodeDataVersion::V15_1),
        ("16.0.0", UnicodeDataVersion::V16_0),
        ("17.0.0", UnicodeDataVersion::V17_0),
    ] {
        for target in [
            NativeStringTarget::NumpyUnicode,
            NativeStringTarget::PandasStringPython,
        ] {
            let facts = NativeTextFacts {
                target,
                python_unicode_database: database.to_owned(),
                arrow_simple_identity: None,
            };
            assert_eq!(
                casing_profile(&facts).unwrap(),
                CasingProfile::FullRoot(version)
            );
        }
    }
    let facts = NativeTextFacts {
        target: NativeStringTarget::PandasStringPyarrow,
        python_unicode_database: "17.0.0".to_owned(),
        arrow_simple_identity: Some("iσⰯᲉ꟎".to_owned()),
    };
    assert_eq!(
        casing_profile(&facts).unwrap(),
        CasingProfile::Simple(UnicodeDataVersion::V13_0)
    );
    assert!(
        casing_profile(&NativeTextFacts {
            arrow_simple_identity: None,
            ..facts
        })
        .is_err()
    );
}

#[test]
fn context_ranges_are_exact_core_selected_batches() {
    assert!(check_range(FrameBatchRange { start: 0, stop: 2 }, 3, 2).is_ok());
    assert!(check_range(FrameBatchRange { start: 2, stop: 3 }, 3, 2).is_ok());
    for range in [
        FrameBatchRange { start: 1, stop: 3 },
        FrameBatchRange { start: 0, stop: 1 },
        FrameBatchRange { start: 2, stop: 4 },
        FrameBatchRange { start: 3, stop: 3 },
    ] {
        assert!(check_range(range, 3, 2).is_err());
    }
}

#[test]
fn native_conversion_failure_prefix_identifies_the_actual_next_column() {
    let text = [
        ColumnRef {
            table: "users".into(),
            column: "note".into(),
        },
        ColumnRef {
            table: "orders".into(),
            column: "note".into(),
        },
    ];
    assert!(check_text_order(&text, &[text[0].clone()], Some(&text[1])).is_ok());
    assert!(check_text_order(&text, &[], Some(&text[1])).is_err());
    assert!(check_text_order(&text, &[text[1].clone()], Some(&text[0])).is_err());
    assert!(check_text_order(&text, &[text[0].clone()], None).is_err());
    assert!(check_text_order(&text, &text, None).is_ok());
}

#[test]
fn mixed_native_string_buffers_align_without_merging_adjacent_surrogate_points() {
    use arrow_array::{ArrayRef, BinaryArray, StringArray};
    use sdm_contract::data::strings::StringEncoding;
    use sdm_engine::strings::StringColumn;
    let paired = [0xed, 0xa0, 0xbd, 0xed, 0xb8, 0xba];
    let context: ArrayRef = Arc::new(BinaryArray::from(vec![paired.as_slice()]));
    let prediction: ArrayRef = Arc::new(StringArray::from(vec!["😺"]));
    let (context, prediction) = align_task_keys(
        context,
        StringEncoding::Utf8Surrogatepass,
        prediction,
        StringEncoding::Utf8,
    )
    .unwrap();
    assert_eq!(context.data_type(), prediction.data_type());
    let context = StringColumn::new(context.as_ref(), StringEncoding::Utf8Surrogatepass).unwrap();
    let prediction =
        StringColumn::new(prediction.as_ref(), StringEncoding::Utf8Surrogatepass).unwrap();
    assert_eq!(
        context
            .get(0)
            .unwrap()
            .unwrap()
            .points()
            .collect::<Vec<_>>(),
        [0xd83d, 0xde3a]
    );
    assert_eq!(
        prediction
            .get(0)
            .unwrap()
            .unwrap()
            .points()
            .collect::<Vec<_>>(),
        [0x1f63a]
    );
}

#[test]
fn encoded_part_identity_binds_its_plan_context_range_summary_and_wire_bytes() {
    let tables = vec!["users".to_owned()];
    let range = FrameBatchRange { start: 0, stop: 2 };
    let original = encoded_identity("plan", "context", range, 1, &tables, b"body").unwrap();
    for changed in [
        encoded_identity("other", "context", range, 1, &tables, b"body").unwrap(),
        encoded_identity("plan", "other", range, 1, &tables, b"body").unwrap(),
        encoded_identity(
            "plan",
            "context",
            FrameBatchRange { start: 2, stop: 4 },
            1,
            &tables,
            b"body",
        )
        .unwrap(),
        encoded_identity("plan", "context", range, 0, &tables, b"body").unwrap(),
        encoded_identity("plan", "context", range, 1, &[], b"body").unwrap(),
        encoded_identity("plan", "context", range, 1, &tables, b"changed").unwrap(),
    ] {
        assert_ne!(original, changed);
    }
}
