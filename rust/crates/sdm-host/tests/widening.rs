// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Lossless widening at admission (decisions H16 and H27): each type the
//! reference accepts and the closed set holds in a wider form is admitted
//! with every value and null, and with the received type kept beside it; a
//! batch whose values pass the offset capacity is split between rows, and
//! only a single value past it is refused, by name. What is still refused
//! is generated in `admission.rs` and listed in register HOST-2.

use std::sync::Arc;

use arrow_array::builder::{
    FixedSizeBinaryBuilder, LargeListViewBuilder, ListViewBuilder, StringViewBuilder,
};
use arrow_array::cast::AsArray;
use arrow_array::types::{
    Decimal128Type, DurationMicrosecondType, Float32Type, Int8Type, Int16Type, Int32Type,
    Int64Type, Time64MicrosecondType, TimestampMicrosecondType,
};
use arrow_array::{
    Array, ArrayRef, BinaryArray, BinaryViewArray, Date64Array, Decimal32Array, Decimal64Array,
    DictionaryArray, DurationMillisecondArray, DurationSecondArray, Float16Array, Int8Array,
    Int16Array, Int32Array, Int64Array, LargeListArray, LargeStringArray, ListArray, RecordBatch,
    RunArray, StringArray, StringViewArray, Time32MillisecondArray, Time32SecondArray,
};
use arrow_buffer::{OffsetBuffer, ScalarBuffer};
use arrow_schema::{DataType, Field, TimeUnit};
use half::f16;
use sdm_contract::{CoreError, Detail, ErrorCode, ErrorKind, Scalar};
use sdm_core::ArrowData;
use sdm_host::{admit, admit_with_capacity};

fn data(column: ArrayRef) -> ArrowData {
    let batch = RecordBatch::try_from_iter([("c", column)]).unwrap();
    ArrowData::new(batch.schema(), vec![batch])
}

/// Admits one column, checking the admitted type and that the received
/// schema is kept; gives the admitted column.
fn admitted(column: ArrayRef, expected: &DataType) -> ArrayRef {
    let received = data(Arc::clone(&column));
    let out = admit(&received).unwrap();
    assert_eq!(out.schema.field(0).data_type(), expected);
    assert_eq!(out.received, received.schema, "the received schema is kept");
    assert_eq!(out.received.field(0).data_type(), column.data_type());
    let array = Arc::clone(out.batches[0].column(0));
    assert_eq!(array.len(), column.len());
    assert_eq!(array.logical_nulls(), column.logical_nulls(), "validity");
    array
}

#[test]
fn float16_widens_to_float32() {
    let half: ArrayRef = Arc::new(Float16Array::from(vec![
        Some(f16::from_f32(1.5)),
        None,
        Some(f16::NEG_INFINITY),
        Some(f16::from_f32(-0.0)),
        Some(f16::NAN),
    ]));
    let out = admitted(half, &DataType::Float32);
    let out = out.as_primitive::<Float32Type>();
    assert_eq!(out.value(0), 1.5);
    assert!(out.is_null(1));
    assert_eq!(out.value(2), f32::NEG_INFINITY);
    assert!(out.value(3) == 0.0 && out.value(3).is_sign_negative());
    assert!(out.value(4).is_nan());
}

#[test]
fn decimal32_and_decimal64_widen_to_decimal128() {
    let narrow: ArrayRef = Arc::new(
        Decimal32Array::from(vec![Some(999_999_999), None, Some(-1)])
            .with_precision_and_scale(9, 3)
            .unwrap(),
    );
    let out = admitted(narrow, &DataType::Decimal128(9, 3));
    let out = out.as_primitive::<Decimal128Type>();
    assert_eq!(
        out.iter().collect::<Vec<_>>(),
        [Some(999_999_999), None, Some(-1)]
    );
    let wide: ArrayRef = Arc::new(
        Decimal64Array::from(vec![Some(i64::MAX / 1000), Some(-5), None])
            .with_precision_and_scale(18, 18)
            .unwrap(),
    );
    let out = admitted(wide, &DataType::Decimal128(18, 18));
    let out = out.as_primitive::<Decimal128Type>();
    assert_eq!(
        out.iter().collect::<Vec<_>>(),
        [Some(i128::from(i64::MAX / 1000)), Some(-5), None]
    );
}

#[test]
fn durations_and_times_widen_to_microseconds() {
    let seconds: ArrayRef = Arc::new(DurationSecondArray::from(vec![Some(-2), None, Some(3)]));
    let out = admitted(seconds, &DataType::Duration(TimeUnit::Microsecond));
    assert_eq!(
        out.as_primitive::<DurationMicrosecondType>()
            .iter()
            .collect::<Vec<_>>(),
        [Some(-2_000_000), None, Some(3_000_000)]
    );
    let millis: ArrayRef = Arc::new(DurationMillisecondArray::from(vec![Some(1), None]));
    let out = admitted(millis, &DataType::Duration(TimeUnit::Microsecond));
    assert_eq!(
        out.as_primitive::<DurationMicrosecondType>()
            .iter()
            .collect::<Vec<_>>(),
        [Some(1_000), None]
    );
    // A duration past what an int64 of microseconds counts is refused.
    let far: ArrayRef = Arc::new(DurationSecondArray::from(vec![i64::MAX]));
    let err = admit(&data(far)).unwrap_err();
    assert_eq!(err.code, ErrorCode::ValueOutOfRange);

    let seconds: ArrayRef = Arc::new(Time32SecondArray::from(vec![Some(86_399), None]));
    let out = admitted(seconds, &DataType::Time64(TimeUnit::Microsecond));
    assert_eq!(
        out.as_primitive::<Time64MicrosecondType>()
            .iter()
            .collect::<Vec<_>>(),
        [Some(86_399_000_000), None]
    );
    let millis: ArrayRef = Arc::new(Time32MillisecondArray::from(vec![None, Some(1)]));
    let out = admitted(millis, &DataType::Time64(TimeUnit::Microsecond));
    assert_eq!(
        out.as_primitive::<Time64MicrosecondType>()
            .iter()
            .collect::<Vec<_>>(),
        [None, Some(1_000)]
    );
}

#[test]
fn date64_widens_to_a_timestamp_whatever_its_values() {
    // Not whole days included: a date64 is milliseconds, kept as they are.
    let dates: ArrayRef = Arc::new(Date64Array::from(vec![
        Some(86_400_000),
        None,
        Some(1),
        Some(-1),
    ]));
    let out = admitted(dates, &DataType::Timestamp(TimeUnit::Microsecond, None));
    assert_eq!(
        out.as_primitive::<TimestampMicrosecondType>()
            .iter()
            .collect::<Vec<_>>(),
        [Some(86_400_000_000), None, Some(1_000), Some(-1_000)]
    );
    let far: ArrayRef = Arc::new(Date64Array::from(vec![i64::MIN]));
    assert_eq!(
        admit(&data(far)).unwrap_err().code,
        ErrorCode::ValueOutOfRange
    );
}

#[test]
fn views_and_fixed_size_binary_become_offsets() {
    let mut text = StringViewBuilder::new();
    text.append_value("a string longer than twelve bytes");
    text.append_null();
    text.append_value("é");
    let text: ArrayRef = Arc::new(text.finish());
    let out = admitted(text, &DataType::Utf8);
    assert_eq!(
        out.as_string::<i32>().iter().collect::<Vec<_>>(),
        [Some("a string longer than twelve bytes"), None, Some("é")]
    );
    let bytes: ArrayRef = Arc::new(BinaryViewArray::from(vec![
        Some(&b"\x00"[..]),
        None,
        Some(&b""[..]),
    ]));
    let out = admitted(bytes, &DataType::Binary);
    assert_eq!(
        out.as_binary::<i32>().iter().collect::<Vec<_>>(),
        [Some(&b"\x00"[..]), None, Some(&b""[..])]
    );
    let mut fixed = FixedSizeBinaryBuilder::new(2);
    fixed.append_value(b"ab").unwrap();
    fixed.append_null();
    fixed.append_value(b"\xff\x00").unwrap();
    let fixed: ArrayRef = Arc::new(fixed.finish());
    let out = admitted(fixed, &DataType::Binary);
    assert_eq!(
        out.as_binary::<i32>().iter().collect::<Vec<_>>(),
        [Some(&b"ab"[..]), None, Some(&b"\xff\x00"[..])]
    );
}

#[test]
fn list_views_become_lists_in_row_order() {
    let item = Arc::new(Field::new("item", DataType::Int64, true));
    let list = |out: &ArrayRef| -> Vec<Option<Vec<Option<i64>>>> {
        out.as_list::<i32>()
            .iter()
            .map(|row| row.map(|v| v.as_primitive::<Int64Type>().iter().collect()))
            .collect()
    };
    let mut view = ListViewBuilder::new(arrow_array::builder::Int64Builder::new());
    view.append_value([Some(1), None]);
    view.append_null();
    view.append_value([Some(3)]);
    view.append_value([]);
    let view: ArrayRef = Arc::new(view.finish());
    let out = admitted(view, &DataType::List(Arc::clone(&item)));
    assert_eq!(
        list(&out),
        [
            Some(vec![Some(1), None]),
            None,
            Some(vec![Some(3)]),
            Some(vec![])
        ]
    );
    let mut large = LargeListViewBuilder::new(arrow_array::builder::Int64Builder::new());
    large.append_value([Some(7)]);
    large.append_null();
    let large: ArrayRef = Arc::new(large.finish());
    let out = admitted(large, &DataType::List(item));
    assert_eq!(list(&out), [Some(vec![Some(7)]), None]);
}

#[test]
fn a_dictionary_of_other_values_is_decoded_and_text_stays_a_dictionary() {
    let ints: DictionaryArray<Int32Type> = DictionaryArray::try_new(
        arrow_array::Int32Array::from(vec![Some(1), None, Some(0), Some(1)]),
        Arc::new(Int64Array::from(vec![10, 20])),
    )
    .unwrap();
    let out = admitted(Arc::new(ints), &DataType::Int64);
    assert_eq!(
        out.as_primitive::<Int64Type>().iter().collect::<Vec<_>>(),
        [Some(20), None, Some(10), Some(20)]
    );
    // Values that widen are decoded and widened.
    let halves: DictionaryArray<Int16Type> = DictionaryArray::try_new(
        arrow_array::Int16Array::from(vec![Some(0), None]),
        Arc::new(Float16Array::from(vec![f16::from_f32(0.5)])),
    )
    .unwrap();
    let out = admitted(Arc::new(halves), &DataType::Float32);
    assert_eq!(
        out.as_primitive::<Float32Type>().iter().collect::<Vec<_>>(),
        [Some(0.5), None]
    );
    // Text in a view stays a dictionary of utf8, its categorical meaning.
    let words: DictionaryArray<Int8Type> = DictionaryArray::try_new(
        arrow_array::Int8Array::from(vec![Some(1), Some(0), None]),
        Arc::new(StringViewArray::from(vec!["a", "b"])),
    )
    .unwrap();
    let out = admitted(
        Arc::new(words),
        &DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8)),
    );
    let out = out.as_dictionary::<Int8Type>();
    let values = out.values().as_string::<i32>();
    let read: Vec<Option<&str>> = out
        .keys()
        .iter()
        .map(|k| k.map(|k| values.value(usize::try_from(k).unwrap())))
        .collect();
    assert_eq!(read, [Some("b"), Some("a"), None]);
}

fn text(error: &CoreError, field: &str) -> String {
    match &error.fields[field] {
        Detail::Scalar(Scalar::Utf8(s)) => s.clone(),
        other => panic!("{field} is {other:?}"),
    }
}

#[test]
fn a_batch_past_the_offset_capacity_splits_between_rows() {
    let words = ["aaaa", "bbbb", "cc", "dddd", "e", "ffff"];
    let column: ArrayRef = Arc::new(LargeStringArray::from(
        words
            .iter()
            .map(|w| Some(*w))
            .chain([None])
            .collect::<Vec<_>>(),
    ));
    // At most 9 bytes of values per admitted batch.
    let out = admit_with_capacity(&data(column), 9).unwrap();
    assert!(out.batches.len() > 1);
    let mut read = Vec::new();
    for batch in &out.batches {
        let column = batch.column(0).as_string::<i32>();
        assert!(column.value_data().len() <= 9);
        assert!(*column.offsets().last().unwrap() <= 9);
        read.extend(column.iter().map(|v| v.map(str::to_owned)));
    }
    let expected: Vec<Option<String>> = words
        .iter()
        .map(|w| Some((*w).to_owned()))
        .chain([None])
        .collect();
    assert_eq!(read, expected);

    // A list's values split the same way, the child sliced with its rows.
    let list = LargeListArray::try_new(
        Arc::new(Field::new("item", DataType::Int64, true)),
        OffsetBuffer::new(ScalarBuffer::from(vec![0i64, 3, 6, 9])),
        Arc::new(Int64Array::from((0..9).collect::<Vec<i64>>())),
        None,
    )
    .unwrap();
    let out = admit_with_capacity(&data(Arc::new(list)), 4).unwrap();
    assert_eq!(out.batches.len(), 3);
    let flat: Vec<i64> = out
        .batches
        .iter()
        .flat_map(|b| {
            let list = b.column(0).as_list::<i32>();
            list.values().as_primitive::<Int64Type>().values().to_vec()
        })
        .collect();
    assert_eq!(flat, (0..9).collect::<Vec<i64>>());
}

#[test]
fn a_single_value_past_the_offset_capacity_is_refused_by_name() {
    let column: ArrayRef = Arc::new(LargeStringArray::from(vec!["ok", "far too long"]));
    let err = admit_with_capacity(&data(column), 9).unwrap_err();
    assert_eq!(
        (err.kind, err.code),
        (ErrorKind::Unsupported, ErrorCode::ValueOutOfRange)
    );
    assert_eq!(text(&err, "column"), "c");
    assert_eq!(text(&err, "arrow_type"), DataType::LargeUtf8.to_string());
    let mut view = StringViewBuilder::new();
    view.append_value("far too long");
    let err = admit_with_capacity(&data(Arc::new(view.finish())), 9).unwrap_err();
    assert_eq!(text(&err, "arrow_type"), DataType::Utf8View.to_string());
}

fn nullable(column: ArrayRef) -> ArrowData {
    let batch = RecordBatch::try_from_iter_with_nullable([("c", column, true)]).unwrap();
    ArrowData::new(batch.schema(), vec![batch])
}

/// Each admitted batch's `c`, read as strings, every batch's offsets
/// within `capacity`.
fn split_strings(out: &ArrowData, capacity: i32) -> Vec<Option<String>> {
    assert!(out.batches.len() > 1, "split between rows");
    out.batches
        .iter()
        .flat_map(|b| {
            let column = b.column(0).as_string::<i32>();
            assert!(*column.offsets().last().unwrap() <= capacity);
            column
                .iter()
                .map(|v| v.map(str::to_owned))
                .collect::<Vec<_>>()
        })
        .collect()
}

#[test]
fn decoded_runs_and_dictionaries_past_the_offset_capacity_split_between_rows() {
    // Small utf8 values, whose decoded rows would pass the capacity: the
    // admitted type is the values' own, so only the check before `take`
    // sees it.
    let values = StringArray::from(vec![Some("aaaa"), Some("bb"), None]);
    let runs = RunArray::<Int32Type>::try_new(&Int32Array::from(vec![3, 4, 6]), &values).unwrap();
    let out = admit_with_capacity(&nullable(Arc::new(runs)), 9).unwrap();
    let read = split_strings(&out, 9);
    let expected = ["aaaa", "aaaa", "aaaa", "bb"]
        .map(|v| Some(v.to_owned()))
        .into_iter()
        .chain([None, None])
        .collect::<Vec<_>>();
    assert_eq!(read, expected);

    // A dictionary of text stays a dictionary; one of bytes is decoded.
    let bytes: DictionaryArray<Int8Type> = DictionaryArray::try_new(
        Int8Array::from(vec![Some(0), None, Some(0), Some(1), Some(0)]),
        Arc::new(BinaryArray::from(vec![&b"aaaa"[..], &b"bbbb"[..]])),
    )
    .unwrap();
    let out = admit_with_capacity(&nullable(Arc::new(bytes)), 9).unwrap();
    assert!(out.batches.len() > 1);
    let mut read = Vec::new();
    for batch in &out.batches {
        let column = batch.column(0).as_binary::<i32>();
        assert!(*column.offsets().last().unwrap() <= 9);
        read.extend(column.iter().map(|v| v.map(<[u8]>::to_vec)));
    }
    let a = Some(b"aaaa".to_vec());
    assert_eq!(
        read,
        [a.clone(), None, a.clone(), Some(b"bbbb".to_vec()), a]
    );

    // A list's text inside the runs counts too.
    let lists = ListArray::try_new(
        Arc::new(Field::new("item", DataType::Utf8, true)),
        OffsetBuffer::new(ScalarBuffer::from(vec![0, 2])),
        Arc::new(StringArray::from(vec!["aaaa", "bbbb"])),
        None,
    )
    .unwrap();
    let runs = RunArray::<Int16Type>::try_new(&Int16Array::from(vec![2]), &lists).unwrap();
    let out = admit_with_capacity(&nullable(Arc::new(runs)), 9).unwrap();
    assert_eq!(out.batches.len(), 2);
    for batch in &out.batches {
        let list = batch.column(0).as_list::<i32>();
        assert_eq!(
            list.value(0).as_string::<i32>().iter().collect::<Vec<_>>(),
            [Some("aaaa"), Some("bbbb")]
        );
    }
}

#[test]
fn a_single_decoded_value_past_the_offset_capacity_is_refused_by_name() {
    let values = StringArray::from(vec!["far too long"]);
    let runs = RunArray::<Int64Type>::try_new(&Int64Array::from(vec![2]), &values).unwrap();
    let received = runs.data_type().clone();
    let err = admit_with_capacity(&nullable(Arc::new(runs)), 9).unwrap_err();
    assert_eq!(
        (err.kind, err.code),
        (ErrorKind::Unsupported, ErrorCode::ValueOutOfRange)
    );
    assert_eq!(text(&err, "column"), "c");
    assert_eq!(text(&err, "arrow_type"), received.to_string());
    let bytes: DictionaryArray<Int8Type> = DictionaryArray::try_new(
        Int8Array::from(vec![0]),
        Arc::new(BinaryArray::from(vec![&b"far too long"[..]])),
    )
    .unwrap();
    let received = bytes.data_type().clone();
    let err = admit_with_capacity(&nullable(Arc::new(bytes)), 9).unwrap_err();
    assert_eq!(err.code, ErrorCode::ValueOutOfRange);
    assert_eq!(text(&err, "arrow_type"), received.to_string());
}
