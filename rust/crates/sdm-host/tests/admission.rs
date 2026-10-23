// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The admission table: each rule keeps every value, null and field, an
//! overflowing value is refused by name (a time past int64 microseconds
//! kept as the microsecond infinity, for `materialize` to refuse), and every Arrow type family outside
//! the set is refused by name, a bounded representative of each. A run-end
//! array of every family the set takes is decoded through `HostRun` (E17).

#![allow(
    clippy::disallowed_methods,
    clippy::disallowed_types,
    reason = "test code: reads the register, and Arrow's field metadata is a HashMap"
)]

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::{Arc, mpsc};

use arrow_array::builder::{Int64Builder, LargeStringBuilder, MapBuilder, StringBuilder};
use arrow_array::cast::AsArray;
use arrow_array::types::{
    Int8Type, Int16Type, Int32Type, Int64Type, RunEndIndexType, TimestampMicrosecondType,
    UInt8Type, UInt16Type, UInt32Type, UInt64Type,
};
use arrow_array::{
    Array, ArrayRef, BinaryArray, BinaryViewArray, BooleanArray, Date32Array, Date64Array,
    Decimal32Array, Decimal64Array, Decimal128Array, Decimal256Array, DictionaryArray,
    DurationMillisecondArray, DurationNanosecondArray, DurationSecondArray, FixedSizeBinaryArray,
    FixedSizeListArray, Float16Array, Float32Array, Float64Array, Int32Array, Int64Array,
    LargeBinaryArray, LargeListArray, LargeListViewArray, LargeStringArray, ListArray,
    ListViewArray, NullArray, PrimitiveArray, RecordBatch, RunArray, StringArray, StringViewArray,
    StructArray, Time32SecondArray, Time64MicrosecondArray, TimestampMillisecondArray,
    TimestampSecondArray, UInt64Array, new_empty_array,
};
use arrow_buffer::{OffsetBuffer, ScalarBuffer, i256};
use arrow_schema::{
    DataType, Field, Fields, IntervalUnit, Schema, TimeUnit, UnionFields, UnionMode,
};
use half::f16;
use sdm_contract::data::{
    AttachmentId, DataRequest, Determinism, ProviderCapabilities, ProviderChoice,
};
use sdm_contract::ops::{OperationResult, QueryValidation};
use sdm_contract::run::{FetchOutcome, FetchReply};
use sdm_contract::{ControlMessage, CoreError, Detail, ErrorCode, ErrorKind, Scalar, Step};
use sdm_core::{ArrowData, Attachment, FetchResult, Reply, Run};
use sdm_host::{HostRun, ZoneResolver, admit, admit_with_capacity, admitted_type, encode, read};

fn data(columns: Vec<(&str, ArrayRef)>) -> ArrowData {
    let batch = RecordBatch::try_from_iter(columns).unwrap();
    ArrowData::new(batch.schema(), vec![batch])
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
    let admitted = admit(&ArrowData::new(
        Arc::clone(&schema),
        vec![batch.clone(), batch],
    ))
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

/// A `timestamp[s]` or `timestamp[ms]` value past int64 microseconds is
/// admitted as DuckDB's microsecond infinity on its side, which
/// `materialize` refuses as a time past the nanosecond range (DT-14), as
/// the reference describes the source and refuses there (T1.36 fix 3).
#[test]
fn a_time_past_int64_microseconds_is_admitted_as_the_microsecond_infinity() {
    let far: ArrayRef = Arc::new(TimestampSecondArray::from(vec![
        Some(0),
        Some(i64::MAX / 1000),
        None,
        Some(-(i64::MAX / 1000)),
        Some(9_223_372_036_854),
    ]));
    let admitted = admit(&data(vec![("far", far)])).unwrap();
    let got = column(&admitted, "far");
    let got = got.as_primitive::<TimestampMicrosecondType>();
    assert_eq!(
        got.iter().collect::<Vec<_>>(),
        [
            Some(0),
            Some(i64::MAX),
            None,
            Some(-i64::MAX),
            Some(9_223_372_036_854_000_000)
        ]
    );
    let far: ArrayRef = Arc::new(TimestampMillisecondArray::from(vec![i64::MIN, i64::MAX]));
    let admitted = admit(&data(vec![("far", far)])).unwrap();
    let got = column(&admitted, "far");
    let got = got.as_primitive::<TimestampMicrosecondType>();
    assert_eq!(got.values().to_vec(), [-i64::MAX, i64::MAX]);
}

#[test]
fn a_value_the_admitted_type_cannot_hold_is_refused_by_name() {
    let far: ArrayRef = Arc::new(DurationSecondArray::from(vec![
        Some(0),
        Some(i64::MAX / 1000),
    ]));
    assert_out_of_range(admit(&data(vec![("far", far)])), "far");
    let far: ArrayRef = Arc::new(DurationMillisecondArray::from(vec![i64::MIN]));
    assert_out_of_range(admit(&data(vec![("far", far)])), "far");
    // A null slot's value is not a value, whatever it holds.
    let masked: ArrayRef = Arc::new(DurationSecondArray::from(vec![Some(1), None]));
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
    // Two rows whose values together pass int32 split between them.
    let fixed =
        FixedSizeListArray::try_new(item, 1 << 30, Arc::new(NullArray::new(past)), None).unwrap();
    let admitted = admit(&data(vec![("fixed", Arc::new(fixed) as ArrayRef)])).unwrap();
    assert_eq!(admitted.batches.len(), 2);
    assert!(admitted.batches.iter().all(|b| b.num_rows() == 1));

    // Nested, the path names the child.
    let inner = Arc::new(Field::new("t", DataType::Duration(TimeUnit::Second), true));
    let list = LargeListArray::try_new(
        inner,
        OffsetBuffer::new(ScalarBuffer::from(vec![0i64, 1])),
        Arc::new(DurationSecondArray::from(vec![i64::MIN])),
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
        DataType::Float16 => ("float16", Taken),
        DataType::Float32 => ("float32", Taken),
        DataType::Float64 => ("float64", Taken),
        DataType::Timestamp(..) => ("timestamp", Taken),
        DataType::Date32 => ("date32", Taken),
        DataType::Date64 => ("date64", Taken),
        DataType::Time32(_) => ("time32", Taken),
        DataType::Time64(_) => ("time64", Taken),
        DataType::Duration(_) => ("duration", Taken),
        DataType::Interval(_) => ("interval", Outside),
        DataType::Binary => ("binary", Taken),
        DataType::FixedSizeBinary(_) => ("fixed_size_binary", Taken),
        DataType::LargeBinary => ("large_binary", Taken),
        DataType::BinaryView => ("binary_view", Taken),
        DataType::Utf8 => ("utf8", Taken),
        DataType::LargeUtf8 => ("large_utf8", Taken),
        DataType::Utf8View => ("utf8_view", Taken),
        DataType::List(_) => ("list", Taken),
        DataType::ListView(_) => ("list_view", Taken),
        DataType::FixedSizeList(..) => ("fixed_size_list", Taken),
        DataType::LargeList(_) => ("large_list", Taken),
        DataType::LargeListView(_) => ("large_list_view", Taken),
        DataType::Struct(_) => ("struct", Taken),
        DataType::Union(..) => ("union", Outside),
        DataType::Dictionary(..) => ("dictionary", Taken),
        DataType::Decimal32(..) => ("decimal32", Taken),
        DataType::Decimal64(..) => ("decimal64", Taken),
        DataType::Decimal128(..) => ("decimal128", Taken),
        DataType::Decimal256(..) => ("decimal256", Taken),
        DataType::Map(..) => ("map", Taken),
        DataType::RunEndEncoded(..) => ("run_end_encoded", Taken),
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
        DataType::Interval(IntervalUnit::YearMonth),
        DataType::Interval(IntervalUnit::DayTime),
        DataType::Union(
            UnionFields::try_new(vec![0], vec![Field::new("a", DataType::Int32, true)]).unwrap(),
            UnionMode::Dense,
        ),
        DataType::Dictionary(Box::new(DataType::Float32), Box::new(DataType::LargeUtf8)),
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
    ) && !matches!(data_type, DataType::Dictionary(index, _) if !index.is_integer())
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
    let interval = DataType::Interval(IntervalUnit::MonthDayNano);
    let nested = DataType::LargeList(Arc::new(Field::new(
        "item",
        DataType::Struct(Fields::from(vec![Field::new("f", interval.clone(), true)])),
        true,
    )));
    let err = admitted_type("col", &nested).unwrap_err();
    assert_eq!(err.code, ErrorCode::UnsupportedType);
    assert_eq!(text(&err, "column"), "col.item.f");
    assert_eq!(text(&err, "arrow_type"), interval.to_string());
    // A dictionary or run-end array decoded to values the set refuses
    // names those values.
    let dictionary = DataType::Dictionary(Box::new(DataType::Int8), Box::new(interval.clone()));
    let err = admitted_type("col", &dictionary).unwrap_err();
    assert_eq!(text(&err, "arrow_type"), interval.to_string());
    let runs = DataType::RunEndEncoded(
        Arc::new(Field::new("run_ends", DataType::Int64, false)),
        Arc::new(Field::new("values", interval.clone(), true)),
    );
    let err = admitted_type("col", &runs).unwrap_err();
    assert_eq!(err.code, ErrorCode::UnsupportedType);
    assert_eq!(text(&err, "arrow_type"), interval.to_string());
}

/// The backticked names in register HOST-2's entry: the type families it
/// lists as still refused.
fn host_2_families() -> BTreeSet<String> {
    let register = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../docs/design/rust-core/register.yaml"),
    )
    .unwrap();
    let start = register.find("- id: HOST-2\n").expect("register HOST-2");
    let entry = &register[start + 1..];
    let entry = &entry[..entry.find("\n- id: ").unwrap_or(entry.len())];
    entry
        .split('`')
        .skip(1)
        .step_by(2)
        .map(str::to_owned)
        .collect()
}

#[test]
fn the_families_still_refused_are_the_ones_register_host_2_lists() {
    // Generated from the family table above, so a widening or a new
    // refusal changes it, and then HOST-2 must say so.
    let refused: BTreeSet<String> = every_family()
        .iter()
        .map(family)
        .filter(|(_, taken)| *taken == Family::Outside)
        .map(|(name, _)| name.to_owned())
        .collect();
    assert_eq!(refused, host_2_families());
}

/// Three values of each family the set takes, the middle one null: one
/// sample array per family, and a dictionary of values it decodes.
fn samples() -> Vec<ArrayRef> {
    let ints = || vec![Some(1), None, Some(-2)];
    let lists = || {
        vec![
            Some(vec![Some(1), None]),
            None,
            Some(vec![Some(3), Some(4)]),
        ]
    };
    let bytes = || vec![Some(&b"\x00\xff"[..]), None, Some(&b""[..])];
    let words = || vec![Some("a"), None, Some("a string longer than twelve bytes")];
    let mut map = MapBuilder::new(None, StringBuilder::new(), Int64Builder::new());
    map.keys().append_value("k");
    map.values().append_null();
    map.append(true).unwrap();
    map.append(false).unwrap();
    map.append(true).unwrap();
    vec![
        Arc::new(NullArray::new(3)),
        Arc::new(BooleanArray::from(vec![Some(true), None, Some(false)])),
        Arc::new(PrimitiveArray::<Int8Type>::from(vec![
            Some(1),
            None,
            Some(-2),
        ])),
        Arc::new(PrimitiveArray::<Int16Type>::from(vec![
            Some(1),
            None,
            Some(-2),
        ])),
        Arc::new(Int32Array::from(ints())),
        Arc::new(Int64Array::from(vec![Some(i64::MAX), None, Some(i64::MIN)])),
        Arc::new(PrimitiveArray::<UInt8Type>::from(vec![
            Some(1),
            None,
            Some(2),
        ])),
        Arc::new(PrimitiveArray::<UInt16Type>::from(vec![
            Some(1),
            None,
            Some(2),
        ])),
        Arc::new(PrimitiveArray::<UInt32Type>::from(vec![
            Some(1),
            None,
            Some(2),
        ])),
        Arc::new(PrimitiveArray::<UInt64Type>::from(vec![
            Some(u64::MAX),
            None,
            Some(0),
        ])),
        Arc::new(Float16Array::from(vec![
            Some(f16::from_f32(1.5)),
            None,
            Some(f16::NEG_INFINITY),
        ])),
        Arc::new(Float32Array::from(vec![Some(1.5), None, Some(-0.0)])),
        Arc::new(Float64Array::from(vec![Some(1.5), None, Some(f64::NAN)])),
        Arc::new(
            TimestampMillisecondArray::from(vec![Some(1), None, Some(-1500)]).with_timezone("UTC"),
        ),
        Arc::new(Date32Array::from(ints())),
        Arc::new(Date64Array::from(vec![Some(86_400_000), None, Some(-1)])),
        Arc::new(Time32SecondArray::from(vec![Some(1), None, Some(86_399)])),
        Arc::new(Time64MicrosecondArray::from(vec![Some(1), None, Some(2)])),
        Arc::new(DurationNanosecondArray::from(vec![Some(1), None, Some(-2)])),
        Arc::new(BinaryArray::from(bytes())),
        Arc::new(
            FixedSizeBinaryArray::try_from_sparse_iter_with_size(
                vec![Some(b"abcd"), None, Some(b"\x00\x00\x00\xff")].into_iter(),
                4,
            )
            .unwrap(),
        ),
        Arc::new(LargeBinaryArray::from(bytes())),
        Arc::new(BinaryViewArray::from(bytes())),
        Arc::new(StringArray::from(words())),
        Arc::new(LargeStringArray::from(words())),
        Arc::new(StringViewArray::from(words())),
        Arc::new(ListArray::from_iter_primitive::<Int32Type, _, _>(lists())),
        Arc::new(ListViewArray::from_iter_primitive::<Int32Type, _, _>(
            lists(),
        )),
        Arc::new(FixedSizeListArray::from_iter_primitive::<Int32Type, _, _>(
            lists(),
            2,
        )),
        Arc::new(LargeListArray::from_iter_primitive::<Int32Type, _, _>(
            lists(),
        )),
        Arc::new(LargeListViewArray::from_iter_primitive::<Int32Type, _, _>(
            lists(),
        )),
        Arc::new(
            StructArray::try_new(
                Fields::from(vec![Field::new("a", DataType::Int32, true)]),
                vec![Arc::new(Int32Array::from(vec![Some(1), Some(2), None]))],
                Some(vec![true, false, true].into()),
            )
            .unwrap(),
        ),
        Arc::new(
            DictionaryArray::<Int32Type>::try_new(
                Int32Array::from(vec![Some(1), None, Some(1)]),
                Arc::new(LargeStringArray::from(vec!["p", "q"])),
            )
            .unwrap(),
        ),
        // A dictionary of other values, decoded under the runs.
        Arc::new(
            DictionaryArray::<Int8Type>::try_new(
                PrimitiveArray::<Int8Type>::from(vec![Some(0), Some(1), None]),
                Arc::new(Float16Array::from(vec![f16::from_f32(0.5), f16::ZERO])),
            )
            .unwrap(),
        ),
        Arc::new(
            Decimal32Array::from(ints())
                .with_precision_and_scale(9, 2)
                .unwrap(),
        ),
        Arc::new(
            Decimal64Array::from(vec![Some(1), None, Some(-2)])
                .with_precision_and_scale(18, 2)
                .unwrap(),
        ),
        Arc::new(
            Decimal128Array::from(vec![Some(1), None, Some(-2)])
                .with_precision_and_scale(38, 10)
                .unwrap(),
        ),
        Arc::new(
            Decimal256Array::from(vec![Some(i256::ONE), None, Some(i256::MINUS_ONE)])
                .with_precision_and_scale(76, 0)
                .unwrap(),
        ),
        Arc::new(map.finish()),
        // Runs under runs, decoded twice.
        Arc::new(
            RunArray::<Int16Type>::try_new(
                &PrimitiveArray::<Int16Type>::from(vec![1, 2, 3]),
                &Int32Array::from(ints()),
            )
            .unwrap(),
        ),
    ]
}

/// One nullable column, as a source declares it: `RecordBatch::try_from_iter`
/// would read a run-end array, whose nulls are its values', as having none.
fn column_c(array: ArrayRef) -> ArrowData {
    let batch = RecordBatch::try_from_iter_with_nullable([("c", array, true)]).unwrap();
    ArrowData::new(batch.schema(), vec![batch])
}

/// `values` in runs of one, two (the null) and three rows, with run ends
/// of type `R`.
fn runs<R: RunEndIndexType>(values: &ArrayRef) -> ArrayRef
where
    R::Native: From<i16>,
{
    let ends = PrimitiveArray::<R>::from_iter_values([1i16, 3, 6].map(R::Native::from));
    Arc::new(RunArray::<R>::try_new(&ends, values.as_ref()).unwrap())
}

/// The rows of `values` at `indices`, as plain values.
fn plain(values: &ArrayRef, indices: &[u64]) -> ArrayRef {
    arrow_select::take::take(values.as_ref(), &UInt64Array::from(indices.to_vec()), None).unwrap()
}

/// `data` handed to a run as fetched data through `HostRun`, as the run
/// receives it, admitted.
fn through_host_run(data: ArrowData) -> ArrowData {
    let (send, receive) = mpsc::channel();
    let host = ProviderChoice::Host {
        capabilities: ProviderCapabilities {
            derived_relations: false, neighbor_lookback: false, sample_rows: false,
            optional: BTreeSet::new(),
            max_keys_per_request: None,
            max_seed: u64::MAX,
            text_times: None,
            null_order: None,
            index_advice: None,
            sampling: Determinism::Unseeded,
            fallback: None,
        },
    };
    let access = sdm_host::default_root().data_access(&host).unwrap();
    let run = HostRun::with(
        Run::with_data(access, move |asker| async move {
            let reply = asker
                // A scan, whose answer's columns are the source's own.
                .fetch(DataRequest::Scan {
                    table: "t".into(),
                    columns: vec![],
                    filters: vec![],
                    limit: None,
                })?
                .await;
            if let Reply::Fetch(FetchResult::Data(data)) = reply {
                send.send(data).unwrap();
            }
            Ok(OperationResult::ValidateQuery(QueryValidation {
                accepted: true,
                diagnostics: vec![],
                query: None,
            }))
        }),
        Arc::new(ZoneResolver::new([])),
    );
    let ControlMessage::Step(Step::Fetch { requests }) = read(&run.next().unwrap()).unwrap() else {
        panic!("the run fetches");
    };
    let reply = ControlMessage::FetchReply(FetchReply {
        id: requests[0].id,
        outcome: FetchOutcome::Data {
            data: AttachmentId(1),
            continuation: None,
        },
    });
    run.reply(
        &encode(&reply).unwrap(),
        BTreeMap::from([(AttachmentId(1), Attachment::Arrow(data))]),
    )
    .unwrap();
    assert!(matches!(
        read(&run.next().unwrap()).unwrap(),
        ControlMessage::Step(Step::Done { .. })
    ));
    receive.recv().unwrap()
}

/// Whether each row is valid, its logical nulls read.
fn validity(array: &ArrayRef) -> Vec<bool> {
    let nulls = array.logical_nulls();
    (0..array.len())
        .map(|row| nulls.as_ref().is_none_or(|n| n.is_valid(row)))
        .collect()
}

/// A run-end column through `HostRun` matches its plain rows admitted:
/// the same admitted type, values and validity, the received type kept.
fn assert_decoded(ree: ArrayRef, rows: ArrayRef) {
    let received = column_c(Arc::clone(&ree));
    let out = through_host_run(received.clone());
    let expected = admit(&column_c(rows)).unwrap();
    let what = ree.data_type().to_string();
    assert_eq!(out.received, received.schema, "{what}: the received type");
    assert_eq!(out.schema, expected.schema, "{what}: the admitted type");
    assert_eq!(out.batches.len(), 1, "{what}");
    let (got, want) = (out.batches[0].column(0), expected.batches[0].column(0));
    assert_eq!(validity(got), validity(&ree), "{what}: validity");
    assert_eq!(got.as_ref(), want.as_ref(), "{what}: values");
}

#[test]
fn a_run_end_column_of_every_family_taken_is_decoded_through_host_run() {
    let taken: BTreeSet<&str> = every_family()
        .iter()
        .map(family)
        .filter(|(_, taken)| *taken == Family::Taken)
        .map(|(name, _)| name)
        .collect();
    let sampled: BTreeSet<&str> = samples()
        .iter()
        .map(|values| family(values.data_type()).0)
        .collect();
    assert_eq!(sampled, taken, "a sample of every family the set takes");
    for values in samples() {
        // Rows 0, 1 1 (null), 2 2 2; sliced, from inside the first long run
        // to inside the last.
        let rows = [0, 1, 1, 2, 2, 2];
        for ree in [
            runs::<Int16Type>(&values),
            runs::<Int32Type>(&values),
            runs::<Int64Type>(&values),
        ] {
            assert_decoded(Arc::clone(&ree), plain(&values, &rows));
            assert_decoded(ree.slice(2, 3), plain(&values, &rows[2..5]));
            assert_decoded(ree.slice(6, 0), plain(&values, &[]));
        }
    }
}

#[test]
fn a_sliced_run_end_column_keeps_only_its_rows() {
    let values: ArrayRef = Arc::new(LargeStringArray::from(vec![
        Some("a"),
        None,
        Some("b"),
        Some("c"),
    ]));
    let ends = Int64Array::from(vec![2, 3, 7, 8]);
    let ree = RunArray::<Int64Type>::try_new(&ends, values.as_ref()).unwrap();
    // Rows a a null b b b b c; the slice starts in a's run and ends in b's.
    let sliced: ArrayRef = Arc::new(ree.slice(1, 4));
    let out = through_host_run(column_c(sliced));
    assert_eq!(out.schema.field(0).data_type(), &DataType::Utf8);
    assert_eq!(
        out.batches[0]
            .column(0)
            .as_string::<i32>()
            .iter()
            .collect::<Vec<_>>(),
        [Some("a"), None, Some("b"), Some("b")]
    );

    // A batch past the offset capacity splits between rows, each part a
    // slice of the runs.
    let ree: ArrayRef = Arc::new(ree);
    let out = admit_with_capacity(&column_c(ree), 3).unwrap();
    assert!(out.batches.len() > 1);
    let read: Vec<Option<String>> = out
        .batches
        .iter()
        .flat_map(|b| {
            let column = b.column(0).as_string::<i32>();
            assert!(*column.offsets().last().unwrap() <= 3);
            column
                .iter()
                .map(|v| v.map(str::to_owned))
                .collect::<Vec<_>>()
        })
        .collect();
    let expected = ["a", "a", "", "b", "b", "b", "b", "c"];
    let expected: Vec<Option<String>> = expected
        .iter()
        .map(|v| (!v.is_empty()).then(|| (*v).to_owned()))
        .collect();
    assert_eq!(read, expected);
}

#[test]
fn a_run_end_column_declared_without_nulls_may_not_hold_a_null_run() {
    let values = Int32Array::from(vec![Some(1), None]);
    let ree = RunArray::<Int32Type>::try_new(&Int32Array::from(vec![1, 2]), &values).unwrap();
    let schema = Arc::new(Schema::new(vec![Field::new(
        "c",
        ree.data_type().clone(),
        false,
    )]));
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(ree)]).unwrap();
    let err = admit(&ArrowData::new(schema, vec![batch])).unwrap_err();
    assert_eq!(err.code, ErrorCode::ProviderSchemaMismatch, "{err:?}");
}
