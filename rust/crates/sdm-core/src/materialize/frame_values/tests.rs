// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Authored before native profile/order and context range consumers.

use super::*;
use sdm_engine::words::{CasingProfile, UnicodeDataVersion};

#[test]
fn one_column_keeps_single_batch_owners_and_existing_other_shape_readers() {
    use arrow_array::{Int64Array, StringArray};
    use arrow_schema::{DataType, Field, Schema};

    let numbers: ArrayRef = Arc::new(Int64Array::from(vec![None, Some(4), Some(5)]));
    let text: ArrayRef = Arc::new(StringArray::from(vec![Some("left"), None, Some("right")]));
    for source in [numbers.slice(1, 2), text.slice(1, 2)] {
        let rows = RecordBatch::try_from_iter([("value", Arc::clone(&source))]).unwrap();
        let input = data(rows);
        let read = one_column(&input).unwrap();
        assert!(Arc::ptr_eq(&source, &read));
        assert_eq!(read.offset(), source.offset());
        assert_eq!(read.null_count(), source.null_count());
    }

    let first = RecordBatch::try_from_iter([("value", numbers.slice(0, 2))]).unwrap();
    let last = RecordBatch::try_from_iter([("value", numbers.slice(2, 1))]).unwrap();
    let schema = first.schema();
    let read = one_column(&ArrowData::new(Arc::clone(&schema), vec![first, last])).unwrap();
    assert_eq!(read.to_data(), numbers.to_data());
    let empty = one_column(&ArrowData::new(schema, vec![])).unwrap();
    assert_eq!(empty.len(), 0);
    assert_eq!(empty.data_type(), &DataType::Int64);

    let rows = RecordBatch::try_from_iter([("value", Arc::clone(&numbers))]).unwrap();
    let renamed = Arc::new(Schema::new(vec![Field::new(
        "renamed",
        DataType::Int64,
        true,
    )]));
    let read = one_column(&ArrowData::new(renamed, vec![rows.clone()])).unwrap();
    assert_eq!(read.to_data(), numbers.to_data());
    let wrong = Arc::new(Schema::new(vec![Field::new("value", DataType::Utf8, true)]));
    assert_eq!(
        one_column(&ArrowData::new(wrong, vec![rows]))
            .unwrap_err()
            .code,
        sdm_contract::ErrorCode::InternalError
    );
    for rows in [
        RecordBatch::new_empty(Arc::new(Schema::empty())),
        RecordBatch::try_from_iter([("one", Arc::clone(&numbers)), ("two", numbers)]).unwrap(),
    ] {
        assert_eq!(
            one_column(&data(rows)).unwrap_err().sentence,
            Sentence::InvalidMessage
        );
    }
}

#[test]
fn reached_frame_failures_keep_result_storage_small() {
    let error = FrameStageError::from(protocol("frame failure"));
    assert!(std::mem::size_of_val(&error) <= 32);
}

#[test]
fn local_label_calendar_bounds_widen_both_ends_before_exact_filtering() {
    use sdm_contract::pql::TimeUnit;
    let windows = label_windows(
        &[-1, 900_000_000],
        Offset::of(-1, TimeUnit::Months),
        Offset::of(1, TimeUnit::Months),
    );
    // Negative anchors floor toward minus infinity, as native // seconds.
    // A negative month's earliest bound spans 31 days; a positive month's
    // latest bound also spans 31 days, independent of the actual calendar.
    assert_eq!(windows[0].lower, Some(-1 - 31 * 86_400));
    assert_eq!(windows[0].upper, -1 + 31 * 86_400);
    assert_eq!(windows[1].lower, Some(-31 * 86_400));
    assert_eq!(windows[1].upper, 31 * 86_400);
}

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
        let decoded = read_labels(&rows, label_encoding(&labels)).unwrap();
        assert_eq!(decoded.labels, labels);
        assert_eq!(decoded.anchors, [10, 20]);
    }
}

#[test]
fn unsigned_link_labels_roundtrip_batch_extend_and_narrow_without_changing_digits() {
    use arrow_array::{Array, Int64Array};
    use sdm_engine::labels::{Labels, LinkValue};
    let values = [
        0,
        (1_u64 << 53) - 1,
        1_u64 << 53,
        (1_u64 << 53) + 1,
        i64::MAX as u64,
        (i64::MAX as u64) + 1,
        u64::MAX,
    ];
    let mut labels = Labels::Lists(vec![
        values[..3].iter().copied().map(LinkValue::UInt).collect(),
        vec![],
    ]);
    labels
        .extend(&Labels::Lists(vec![
            values[3..].iter().copied().map(LinkValue::UInt).collect(),
        ]))
        .unwrap();
    let keys: ArrayRef = Arc::new(Int64Array::from(vec![1, 2, 3]));
    let rows = labelled_rows(&keys, &[10, 20, 30], &labels).unwrap();
    let target = rows.column(2).as_list::<i32>();
    assert_eq!(target.values().data_type(), &DataType::UInt64);
    assert_eq!(target.values().null_count(), 0);
    assert_eq!(
        target
            .values()
            .as_primitive::<UInt64Type>()
            .values()
            .as_ref(),
        values
    );
    let context = read_labels(&rows, None).unwrap();
    assert_eq!(context.labels, labels);
    let narrowed = context.narrow(Some(2));
    let again = labelled_rows(&narrowed.keys, &narrowed.anchors, &narrowed.labels).unwrap();
    assert_eq!(
        read_labels(&again, None).unwrap().labels,
        labels.clone().truncated(2)
    );
    let empty: ArrayRef = Arc::new(Int64Array::from(Vec::<i64>::new()));
    let rows = labelled_rows(&empty, &[], &Labels::Lists(vec![])).unwrap();
    assert_eq!(
        read_labels(&rows, None).unwrap().labels,
        Labels::Lists(vec![])
    );
}

#[test]
fn unsigned_native_lists_reject_mixed_family_and_missing_child_evidence() {
    use arrow_array::types::UInt64Type;
    use arrow_array::{Int64Array, ListArray};
    use sdm_engine::labels::{Labels, LinkValue};
    let keys: ArrayRef = Arc::new(Int64Array::from(vec![1]));
    assert!(
        labelled_rows(
            &keys,
            &[10],
            &Labels::Lists(vec![vec![LinkValue::UInt(1), LinkValue::Int(1)]])
        )
        .is_err()
    );
    let forged: ArrayRef = Arc::new(ListArray::from_iter_primitive::<UInt64Type, _, _>([Some(
        vec![Some(u64::MAX), None],
    )]));
    assert_eq!(
        read_link_labels(&forged, None).unwrap_err().sentence,
        Sentence::InvalidMessage
    );
}

#[test]
fn neutral_casing_facts_select_only_supported_mathematical_maps() {
    for (database, version) in [
        ("13.0.0", UnicodeDataVersion::V13_0),
        ("14.0.0", UnicodeDataVersion::V14_0),
        ("15.0.0", UnicodeDataVersion::V15_0),
        ("15.1.0", UnicodeDataVersion::V15_1),
        ("16.0.0", UnicodeDataVersion::V16_0),
        ("17.0.0", UnicodeDataVersion::V17_0),
    ] {
        assert_eq!(
            casing_profile(&NativeTextFacts::FullRoot {
                unicode_version: database.to_owned(),
            })
            .unwrap(),
            CasingProfile::FullRoot(version),
        );
    }
    for (identity, version) in [
        ("iσⰯᲉ꟎", UnicodeDataVersion::V13_0),
        ("iσⱟᲉ꟎", UnicodeDataVersion::V14_0),
        ("iσⱟᲊ꟎", UnicodeDataVersion::V16_0),
        ("iσⱟᲊ꟏", UnicodeDataVersion::V17_0),
    ] {
        assert_eq!(
            casing_profile(&NativeTextFacts::Simple {
                identity: identity.into()
            })
            .unwrap(),
            CasingProfile::Simple(version),
        );
    }
    for facts in [
        NativeTextFacts::FullRoot {
            unicode_version: "12.0.0".into(),
        },
        NativeTextFacts::Simple {
            identity: "unknown".into(),
        },
        NativeTextFacts::Simple {
            identity: String::new(),
        },
    ] {
        assert_eq!(
            casing_profile(&facts).unwrap_err().sentence,
            Sentence::InvalidMessage
        );
    }
}

#[test]
fn neutral_mapping_facts_preserve_contextual_and_simple_lowercase() {
    use sdm_contract::scalar::unicode::UnicodeText;
    let input = UnicodeText::from("ΟΣ İ");
    for (facts, expected) in [
        (
            NativeTextFacts::FullRoot {
                unicode_version: "15.0.0".into(),
            },
            vec!["ος", "i\u{307}"],
        ),
        (
            NativeTextFacts::Simple {
                identity: "iσⱟᲊ꟏".into(),
            },
            vec!["οσ", "i"],
        ),
    ] {
        let words = sdm_engine::words::normalize_native(
            Some(input.as_slice()),
            casing_profile(&facts).unwrap(),
        );
        assert_eq!(
            words,
            expected
                .into_iter()
                .map(UnicodeText::from)
                .collect::<Vec<_>>()
        );
    }
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
    let logical = Value::text("logical body");
    let original =
        encoded_identity("plan", "context", range, 1, &tables, b"body", &logical).unwrap();
    for changed in [
        encoded_identity("other", "context", range, 1, &tables, b"body", &logical).unwrap(),
        encoded_identity("plan", "other", range, 1, &tables, b"body", &logical).unwrap(),
        encoded_identity(
            "plan",
            "context",
            FrameBatchRange { start: 2, stop: 4 },
            1,
            &tables,
            b"body",
            &logical,
        )
        .unwrap(),
        encoded_identity("plan", "context", range, 0, &tables, b"body", &logical).unwrap(),
        encoded_identity("plan", "context", range, 1, &[], b"body", &logical).unwrap(),
        encoded_identity("plan", "context", range, 1, &tables, b"changed", &logical).unwrap(),
        encoded_identity(
            "plan",
            "context",
            range,
            1,
            &tables,
            b"body",
            &Value::text("other logical body"),
        )
        .unwrap(),
    ] {
        assert_ne!(original, changed);
    }
}

#[test]
fn labelled_native_strings_bind_exact_target_encoding_and_keep_units_distinct() {
    use arrow_array::{ArrayRef, Int64Array};
    use sdm_contract::data::strings::StringEncoding;
    use sdm_contract::scalar::unicode::UnicodeText;
    use sdm_engine::labels::{Labels, LinkValue};
    let pair = UnicodeText::from_points([0xd83d, 0xde3a]).unwrap();
    let scalar = UnicodeText::from("😺");
    let lone = UnicodeText::from_points([0xd800]).unwrap();
    let keys: ArrayRef = Arc::new(Int64Array::from(vec![1, 2, 3]));
    for labels in [
        Labels::Text(vec![pair.clone(), scalar.clone(), lone.clone()]),
        Labels::Lists(vec![
            vec![LinkValue::Text(pair.clone())],
            vec![LinkValue::Text(scalar.clone())],
            vec![LinkValue::Text(lone.clone())],
        ]),
    ] {
        let encoding = label_encoding(&labels);
        assert_eq!(encoding, Some(StringEncoding::Utf8Surrogatepass));
        let rows = labelled_rows(&keys, &[10, 20, 30], &labels).unwrap();
        assert_eq!(read_labels(&rows, encoding).unwrap().labels, labels);
        assert!(read_labels(&rows, None).is_err());
        assert!(read_labels(&rows, Some(StringEncoding::Utf8)).is_err());
    }
}

#[test]
fn native_constructor_refusal_uses_the_reached_first_bad_element_and_dtype() {
    use sdm_contract::prepare::{BuiltinFailureClass, NativeFailure};
    let failure = FrameNativeFailure {
        builtin: NativeFailure {
            ancestry: vec![BuiltinFailureClass::ValueError],
        },
        key_error: false,
        structured_data_error: false,
        relational_error: false,
        deepest_module: Some("native.library".into()),
        message_casing: NativeTextFacts::FullRoot {
            unicode_version: "14.0.0".into(),
        },
    };
    let error = FrameStageError::Native(Box::new(FrameNativeFailureContext {
        site: NativeSite::Prediction {
            table: "users".into(),
        },
        failure,
        diagnostic: Some(NativeConstructionDiagnostic::Completed {
            first: Some(UncastableFrameInput {
                ordinal: 2,
                native_repr: "'bad'".into(),
            }),
            dtype_display: "string".into(),
        }),
        dropped_summary: None,
    }))
    .into_core();
    assert_eq!(
        error.fields["native_constructor_kind"],
        Detail::text("value_error")
    );
    assert_eq!(
        error.fields["native_constructor_message"],
        Detail::text(
            "'indices' must hold values matching the primary key of 'users', which is string; indices[2] is 'bad'"
        )
    );
}

#[test]
fn ordinary_batch_refusal_and_statistic_precedence_are_metadata_only() {
    for rows in ["0", "-1", "true", "1.5", "\"2\""] {
        let given =
            sdm_wire::json::read(format!("{{\"rows_per_call\":{rows}}}").as_bytes()).unwrap();
        let error = check_call(Some(&given), &[], 2, &FrameLimits::default()).unwrap_err();
        assert_eq!(error.sentence, Sentence::RowsPerCall);
        for value in [
            format!("{{\"rows_per_call\":{rows},\"prediction_statistic\":\"not-a-statistic\"}}"),
            format!("{{\"prediction_statistic\":\"not-a-statistic\",\"rows_per_call\":{rows}}}"),
        ] {
            let given = sdm_wire::json::read(value.as_bytes()).unwrap();
            let error = check_call(Some(&given), &[], 2, &FrameLimits::default()).unwrap_err();
            assert_eq!(error.sentence, Sentence::StatisticUnknown);
        }
    }
    for rows in ["null", "1", "200", "184467440737095516160"] {
        let given =
            sdm_wire::json::read(format!("{{\"rows_per_call\":{rows}}}").as_bytes()).unwrap();
        assert!(check_call(Some(&given), &[], 2, &FrameLimits::default()).is_ok());
    }
}
