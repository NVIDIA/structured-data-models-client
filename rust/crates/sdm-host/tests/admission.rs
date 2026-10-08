// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The admission table: each rule keeps every value, null and field, an
//! overflowing value is refused by name, and every Arrow type family outside
//! the set is refused by name, a bounded representative of each.

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use arrow_array::builder::{Int64Builder, LargeStringBuilder, MapBuilder};
use arrow_array::cast::AsArray;
use arrow_array::types::{Int8Type, Int32Type};
use arrow_array::{
    Array, ArrayRef, DictionaryArray, FixedSizeListArray, Int32Array, Int64Array, LargeBinaryArray,
    LargeListArray, LargeStringArray, NullArray, RecordBatch, StructArray,
    TimestampMillisecondArray, TimestampSecondArray, new_empty_array,
};
use arrow_buffer::{OffsetBuffer, ScalarBuffer};
use arrow_schema::{
    DataType, Field, Fields, IntervalUnit, Schema, TimeUnit, UnionFields, UnionMode,
};
use sdm_contract::{CoreError, Detail, ErrorCode, ErrorKind, Scalar};
use sdm_core::ArrowData;
use sdm_host::{admit, admitted_type};

fn data(columns: Vec<(&str, ArrayRef)>) -> ArrowData {
    let batch = RecordBatch::try_from_iter(columns).unwrap();
    ArrowData {
        schema: batch.schema(),
        batches: vec![batch],
    }
}

fn text(error: &CoreError, field: &str) -> String {
    match &error.fields[field] {
        Detail::Scalar(Scalar::Utf8(s)) => s.clone(),
        other => panic!("{field} is {other:?}"),
    }
}

fn column(admitted: &ArrowData, name: &str) -> ArrayRef {
    let index = admitted.schema.index_of(name).unwrap();
    Arc::clone(admitted.batches[0].column(index))
}

#[test]
fn large_text_and_bytes_become_small_with_every_value_and_null() {
    let strings: ArrayRef = Arc::new(LargeStringArray::from(vec![Some("a"), None, Some("é")]));
    let bytes: ArrayRef = Arc::new(LargeBinaryArray::from(vec![
        Some(&b"\x00\xff"[..]),
        Some(&b""[..]),
        None,
    ]));
    let admitted = admit(&data(vec![("s", strings), ("b", bytes)])).unwrap();
    assert_eq!(admitted.schema.field(0).data_type(), &DataType::Utf8);
    assert_eq!(admitted.schema.field(1).data_type(), &DataType::Binary);
    let s = column(&admitted, "s");
    let s = s.as_string::<i32>();
    assert_eq!(s.iter().collect::<Vec<_>>(), [Some("a"), None, Some("é")]);
    let b = column(&admitted, "b");
    let b = b.as_binary::<i32>();
    assert_eq!(
        b.iter().collect::<Vec<_>>(),
        [Some(&b"\x00\xff"[..]), Some(&b""[..]), None]
    );
}

#[test]
fn seconds_and_milliseconds_become_microseconds_keeping_the_zone() {
    let seconds: ArrayRef = Arc::new(
        TimestampSecondArray::from(vec![Some(-1), None, Some(1_700_000_000)]).with_timezone("UTC"),
    );
    let millis: ArrayRef =
        Arc::new(TimestampMillisecondArray::from(vec![Some(1), Some(-1500)]).slice(0, 2));
    let admitted = admit(&data(vec![("s", seconds)])).unwrap();
    assert_eq!(
        admitted.schema.field(0).data_type(),
        &DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into()))
    );
    let s = column(&admitted, "s");
    let s = s.as_primitive::<arrow_array::types::TimestampMicrosecondType>();
    assert_eq!(
        s.iter().collect::<Vec<_>>(),
        [Some(-1_000_000), None, Some(1_700_000_000_000_000)]
    );
    let admitted = admit(&data(vec![("m", millis)])).unwrap();
    assert_eq!(
        admitted.schema.field(0).data_type(),
        &DataType::Timestamp(TimeUnit::Microsecond, None)
    );
    let m = column(&admitted, "m");
    let m = m.as_primitive::<arrow_array::types::TimestampMicrosecondType>();
    assert_eq!(m.values().to_vec(), [1_000, -1_500_000]);
}

#[test]
fn large_and_fixed_size_lists_become_lists() {
    let item = Arc::new(Field::new("item", DataType::LargeUtf8, true));
    let large = LargeListArray::try_new(
        Arc::clone(&item),
        OffsetBuffer::new(ScalarBuffer::from(vec![0i64, 2, 2, 3])),
        Arc::new(LargeStringArray::from(vec![Some("x"), None, Some("y")])),
        Some(vec![true, false, true].into()),
    )
    .unwrap();
    let fixed = FixedSizeListArray::try_new(
        Arc::new(Field::new("v", DataType::Int32, false)),
        2,
        Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5, 6])),
        Some(vec![true, false, true].into()),
    )
    .unwrap()
    .slice(1, 2);
    let admitted = admit(&data(vec![("large", Arc::new(large) as ArrayRef)])).unwrap();
    assert_eq!(
        admitted.schema.field(0).data_type(),
        &DataType::List(Arc::new(Field::new("item", DataType::Utf8, true)))
    );
    let large = column(&admitted, "large");
    let large = large.as_list::<i32>();
    assert_eq!(large.len(), 3);
    assert!(large.is_null(1));
    assert_eq!(
        large.value(0).as_string::<i32>().iter().collect::<Vec<_>>(),
        [Some("x"), None]
    );
    assert_eq!(large.value(2).as_string::<i32>().value(0), "y");
    let admitted = admit(&data(vec![("fixed", Arc::new(fixed) as ArrayRef)])).unwrap();
    assert_eq!(
        admitted.schema.field(0).data_type(),
        &DataType::List(Arc::new(Field::new("v", DataType::Int32, false)))
    );
    let fixed = column(&admitted, "fixed");
    let fixed = fixed.as_list::<i32>();
    assert!(fixed.is_null(0));
    assert_eq!(
        fixed.value(1).as_primitive::<Int32Type>().values().to_vec(),
        [5, 6]
    );
}

#[test]
fn nested_types_are_admitted_child_by_child() {
    let child: ArrayRef = Arc::new(LargeStringArray::from(vec![Some("k"), None]));
    let when: ArrayRef = Arc::new(TimestampMillisecondArray::from(vec![Some(7), Some(8)]));
    let record = StructArray::try_new(
        Fields::from(vec![
            Field::new("name", DataType::LargeUtf8, true),
            Field::new(
                "when",
                DataType::Timestamp(TimeUnit::Millisecond, None),
                false,
            ),
        ]),
        vec![child, when],
        Some(vec![true, false].into()),
    )
    .unwrap();
    let mut map = MapBuilder::new(None, LargeStringBuilder::new(), Int64Builder::new());
    map.keys().append_value("a");
    map.values().append_value(1);
    map.append(true).unwrap();
    map.append(false).unwrap();
    let dictionary: DictionaryArray<Int8Type> = vec![Some("p"), None]
        .into_iter()
        .collect::<DictionaryArray<Int8Type>>();
    let dictionary = dictionary.with_values(Arc::new(LargeStringArray::from(
        dictionary
            .values()
            .as_string::<i32>()
            .iter()
            .collect::<Vec<_>>(),
    )));
    let admitted = admit(&data(vec![
        ("record", Arc::new(record) as ArrayRef),
        ("map", Arc::new(map.finish()) as ArrayRef),
        ("dictionary", Arc::new(dictionary) as ArrayRef),
    ]))
    .unwrap();
    let DataType::Struct(fields) = admitted.schema.field(0).data_type() else {
        panic!("a struct stays a struct");
    };
    assert_eq!(fields[0].data_type(), &DataType::Utf8);
    assert_eq!(
        fields[1].data_type(),
        &DataType::Timestamp(TimeUnit::Microsecond, None)
    );
    assert!(!fields[1].is_nullable());
    let record = column(&admitted, "record");
    let record = record.as_struct();
    assert!(record.is_null(1));
    assert_eq!(record.column(0).as_string::<i32>().value(0), "k");
    let DataType::Map(entries, _) = admitted.schema.field(1).data_type() else {
        panic!("a map stays a map");
    };
    let DataType::Struct(kv) = entries.data_type() else {
        panic!("a map's entries are a struct");
    };
    assert_eq!(kv[0].data_type(), &DataType::Utf8);
    let map = column(&admitted, "map");
    let map = map.as_map();
    assert!(map.is_null(1));
    assert_eq!(map.keys().as_string::<i32>().value(0), "a");
    assert_eq!(
        admitted.schema.field(2).data_type(),
        &DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8))
    );
    let dictionary = column(&admitted, "dictionary");
    let dictionary = dictionary.as_dictionary::<Int8Type>();
    assert_eq!(
        dictionary.keys().iter().collect::<Vec<_>>(),
        [Some(0), None]
    );
}

#[test]
fn field_and_schema_metadata_and_admitted_types_are_kept() {
    let ints: ArrayRef = Arc::new(Int64Array::from(vec![Some(1), None]));
    let field = Field::new("i", DataType::Int64, true)
        .with_metadata(HashMap::from([("k".to_owned(), "v".to_owned())]));
    let schema = Arc::new(Schema::new_with_metadata(
        vec![field],
        HashMap::from([("pandas".to_owned(), "{}".to_owned())]),
    ));
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::clone(&ints)]).unwrap();
    let admitted = admit(&ArrowData {
        schema: Arc::clone(&schema),
        batches: vec![batch.clone(), batch],
    })
    .unwrap();
    assert_eq!(admitted.schema, schema);
    assert_eq!(admitted.batches.len(), 2);
    assert_eq!(admitted.batches[1].column(0).as_ref(), ints.as_ref());
}

fn assert_out_of_range(result: Result<ArrowData, CoreError>, column: &str) {
    let err = result.unwrap_err();
    assert_eq!(
        (err.kind, err.code),
        (ErrorKind::Unsupported, ErrorCode::ValueOutOfRange),
        "{err:?}"
    );
    assert_eq!(text(&err, "column"), column);
}

#[test]
fn a_value_the_admitted_type_cannot_hold_is_refused_by_name() {
    let far: ArrayRef = Arc::new(TimestampSecondArray::from(vec![
        Some(0),
        Some(i64::MAX / 1000),
    ]));
    assert_out_of_range(admit(&data(vec![("far", far)])), "far");
    let far: ArrayRef = Arc::new(TimestampMillisecondArray::from(vec![i64::MIN]));
    assert_out_of_range(admit(&data(vec![("far", far)])), "far");
    // A null slot's value is not a value, whatever it holds.
    let masked: ArrayRef = Arc::new(TimestampSecondArray::from(vec![Some(1), None]));
    let masked = masked
        .to_data()
        .into_builder()
        .buffers(vec![ScalarBuffer::from(vec![1i64, i64::MAX]).into_inner()])
        .build()
        .unwrap();
    let admitted = admit(&data(vec![("masked", arrow_array::make_array(masked))])).unwrap();
    assert_eq!(column(&admitted, "masked").null_count(), 1);

    // Offsets past int32, over a child of nulls, which costs no memory.
    let past = 1usize << 31;
    let item = Arc::new(Field::new("item", DataType::Null, true));
    let large = LargeListArray::try_new(
        Arc::clone(&item),
        OffsetBuffer::new(ScalarBuffer::from(vec![0i64, past as i64])),
        Arc::new(NullArray::new(past)),
        None,
    )
    .unwrap();
    assert_out_of_range(
        admit(&data(vec![("large", Arc::new(large) as ArrayRef)])),
        "large",
    );
    let fixed =
        FixedSizeListArray::try_new(item, 1 << 30, Arc::new(NullArray::new(past)), None).unwrap();
    assert_out_of_range(
        admit(&data(vec![("fixed", Arc::new(fixed) as ArrayRef)])),
        "fixed",
    );
    // Nested, the path names the child.
    let inner = Arc::new(Field::new(
        "t",
        DataType::Timestamp(TimeUnit::Second, None),
        true,
    ));
    let list = LargeListArray::try_new(
        inner,
        OffsetBuffer::new(ScalarBuffer::from(vec![0i64, 1])),
        Arc::new(TimestampSecondArray::from(vec![i64::MIN])),
        None,
    )
    .unwrap();
    assert_out_of_range(admit(&data(vec![("l", Arc::new(list) as ArrayRef)])), "l.t");
}

/// Whether the set, through admission, takes a family of Arrow types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Family {
    Taken,
    Outside,
}

/// The family of an Arrow type, and whether it is taken: every variant of
/// `DataType`, so a new variant fails to compile here until it is placed.
fn family(data_type: &DataType) -> (&'static str, Family) {
    use Family::{Outside, Taken};
    match data_type {
        DataType::Null => ("null", Taken),
        DataType::Boolean => ("boolean", Taken),
        DataType::Int8 => ("int8", Taken),
        DataType::Int16 => ("int16", Taken),
        DataType::Int32 => ("int32", Taken),
        DataType::Int64 => ("int64", Taken),
        DataType::UInt8 => ("uint8", Taken),
        DataType::UInt16 => ("uint16", Taken),
        DataType::UInt32 => ("uint32", Taken),
        DataType::UInt64 => ("uint64", Taken),
        DataType::Float16 => ("float16", Outside),
        DataType::Float32 => ("float32", Taken),
        DataType::Float64 => ("float64", Taken),
        DataType::Timestamp(..) => ("timestamp", Taken),
        DataType::Date32 => ("date32", Taken),
        DataType::Date64 => ("date64", Outside),
        DataType::Time32(_) => ("time32", Outside),
        DataType::Time64(_) => ("time64", Taken),
        DataType::Duration(_) => ("duration", Taken),
        DataType::Interval(_) => ("interval", Outside),
        DataType::Binary => ("binary", Taken),
        DataType::FixedSizeBinary(_) => ("fixed_size_binary", Outside),
        DataType::LargeBinary => ("large_binary", Taken),
        DataType::BinaryView => ("binary_view", Outside),
        DataType::Utf8 => ("utf8", Taken),
        DataType::LargeUtf8 => ("large_utf8", Taken),
        DataType::Utf8View => ("utf8_view", Outside),
        DataType::List(_) => ("list", Taken),
        DataType::ListView(_) => ("list_view", Outside),
        DataType::FixedSizeList(..) => ("fixed_size_list", Taken),
        DataType::LargeList(_) => ("large_list", Taken),
        DataType::LargeListView(_) => ("large_list_view", Outside),
        DataType::Struct(_) => ("struct", Taken),
        DataType::Union(..) => ("union", Outside),
        DataType::Dictionary(..) => ("dictionary", Taken),
        DataType::Decimal32(..) => ("decimal32", Outside),
        DataType::Decimal64(..) => ("decimal64", Outside),
        DataType::Decimal128(..) => ("decimal128", Taken),
        DataType::Decimal256(..) => ("decimal256", Taken),
        DataType::Map(..) => ("map", Taken),
        DataType::RunEndEncoded(..) => ("run_end_encoded", Outside),
    }
}

/// One bounded type of every family; for a family the set takes, a form it
/// takes.
fn every_family() -> Vec<DataType> {
    let item = |t: DataType| Arc::new(Field::new("item", t, true));
    let union = |mode| {
        DataType::Union(
            UnionFields::try_new(vec![0], vec![Field::new("a", DataType::Int32, true)]).unwrap(),
            mode,
        )
    };
    vec![
        DataType::Null,
        DataType::Boolean,
        DataType::Int8,
        DataType::Int16,
        DataType::Int32,
        DataType::Int64,
        DataType::UInt8,
        DataType::UInt16,
        DataType::UInt32,
        DataType::UInt64,
        DataType::Float16,
        DataType::Float32,
        DataType::Float64,
        DataType::Timestamp(TimeUnit::Millisecond, Some("UTC".into())),
        DataType::Date32,
        DataType::Date64,
        DataType::Time32(TimeUnit::Second),
        DataType::Time64(TimeUnit::Microsecond),
        DataType::Duration(TimeUnit::Nanosecond),
        DataType::Interval(IntervalUnit::MonthDayNano),
        DataType::Binary,
        DataType::FixedSizeBinary(4),
        DataType::LargeBinary,
        DataType::BinaryView,
        DataType::Utf8,
        DataType::LargeUtf8,
        DataType::Utf8View,
        DataType::List(item(DataType::Int32)),
        DataType::ListView(item(DataType::Int32)),
        DataType::FixedSizeList(item(DataType::Int32), 2),
        DataType::LargeList(item(DataType::Int32)),
        DataType::LargeListView(item(DataType::Int32)),
        DataType::Struct(Fields::from(vec![Field::new("a", DataType::Int32, true)])),
        union(UnionMode::Sparse),
        DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::LargeUtf8)),
        DataType::Decimal32(9, 2),
        DataType::Decimal64(18, 2),
        DataType::Decimal128(38, 10),
        DataType::Decimal256(76, 0),
        DataType::Map(
            Arc::new(Field::new(
                "entries",
                DataType::Struct(Fields::from(vec![
                    Field::new("key", DataType::Utf8, false),
                    Field::new("value", DataType::Int64, true),
                ])),
                false,
            )),
            false,
        ),
        DataType::RunEndEncoded(
            Arc::new(Field::new("run_ends", DataType::Int32, false)),
            Arc::new(Field::new("values", DataType::Utf8, true)),
        ),
    ]
}

/// Forms outside the set of families it takes in other forms.
fn outside_forms() -> Vec<DataType> {
    vec![
        DataType::Time64(TimeUnit::Second),
        DataType::Duration(TimeUnit::Second),
        DataType::Duration(TimeUnit::Millisecond),
        DataType::Time32(TimeUnit::Millisecond),
        DataType::Interval(IntervalUnit::YearMonth),
        DataType::Interval(IntervalUnit::DayTime),
        DataType::Union(
            UnionFields::try_new(vec![0], vec![Field::new("a", DataType::Int32, true)]).unwrap(),
            UnionMode::Dense,
        ),
        DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Int64)),
        DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8View)),
        DataType::Decimal128(0, 0),
        DataType::Decimal256(77, 0),
    ]
}

/// Whether Arrow builds arrays of `data_type`; some types are not valid
/// to build at all, and are refused as types.
fn buildable(data_type: &DataType) -> bool {
    !matches!(
        data_type,
        DataType::Time64(TimeUnit::Second | TimeUnit::Millisecond)
            | DataType::Time32(TimeUnit::Microsecond | TimeUnit::Nanosecond)
            | DataType::Decimal128(0, _)
            | DataType::Decimal256(77.., _)
    )
}

fn assert_refused(data_type: &DataType) {
    // As a type, and as a column of a bounded, empty array.
    let mut results = vec![admitted_type("col", data_type).map(|_| ())];
    if buildable(data_type) {
        results.push(admit(&data(vec![("col", new_empty_array(data_type))])).map(|_| ()));
    }
    for result in results {
        let err = result.unwrap_err();
        assert_eq!(
            (err.kind, err.code),
            (ErrorKind::Unsupported, ErrorCode::UnsupportedType),
            "{data_type}"
        );
        assert_eq!(text(&err, "column"), "col", "{data_type}");
        assert_eq!(text(&err, "arrow_type"), data_type.to_string());
    }
}

#[test]
fn every_family_outside_the_set_is_refused_by_name() {
    let families: BTreeSet<&str> = every_family().iter().map(|t| family(t).0).collect();
    assert_eq!(families.len(), 41, "one type of every family of DataType");
    for data_type in every_family() {
        match family(&data_type).1 {
            Family::Outside => assert_refused(&data_type),
            Family::Taken => {
                let admitted = admit(&data(vec![("col", new_empty_array(&data_type))]));
                assert!(admitted.is_ok(), "{data_type}: {admitted:?}");
            }
        }
    }
    for data_type in outside_forms() {
        assert_refused(&data_type);
    }
}

#[test]
fn a_refused_child_is_named_by_its_path() {
    let nested = DataType::LargeList(Arc::new(Field::new(
        "item",
        DataType::Struct(Fields::from(vec![Field::new("f", DataType::Float16, true)])),
        true,
    )));
    let err = admitted_type("col", &nested).unwrap_err();
    assert_eq!(err.code, ErrorCode::UnsupportedType);
    assert_eq!(text(&err, "column"), "col.item.f");
    assert_eq!(text(&err, "arrow_type"), "Float16");
}
