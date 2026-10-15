// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The `arrow.bool8` reading's edges (T1.36 C13, carried from T1.33 part
//! 2's review), held from outside the implementation by the package's
//! independent test author.
//!
//! The canonical extension (Arrow's canonical extensions, "8-bit Boolean"):
//! storage `int8`, "false is denoted by the value 0, true can be specified
//! using any non-zero value; preferably 1". DuckDB 1.5.6 writes 1 for true
//! and 0 for false under `arrow_lossless_conversion`, and reads an
//! `arrow.bool8` it is handed with any non-zero value as true; pyarrow
//! 25.0.1's `Bool8Array` does the same. Probed 2026-10-14 with the
//! `venv-t133p3` interpreter:
//!
//! ```text
//! storage [2, -1, 0, 1, null] as pa.bool8()  -> to_pylist [True, True, False, True, None]
//! duckdb SELECT f::VARCHAR over that table     -> 'true', 'true', 'false', 'true', NULL
//! duckdb lossless BOOLEAN[]          -> list<l: extension<arrow.bool8>>, storage 1, 0, null
//! duckdb lossless STRUCT(a BOOLEAN)  -> struct<a: extension<arrow.bool8>>
//! duckdb lossless BOOLEAN[2]         -> fixed_size_list<: extension<arrow.bool8>>[2]
//! duckdb lossless MAP(INTEGER, BOOLEAN) -> map<int32, extension<arrow.bool8>>
//! duckdb lossless STRUCT(a BOOLEAN)[] -> list<l: struct<a: extension<arrow.bool8>>>
//! ```
//!
//! So every non-zero value is true, by the spec, which both readers
//! follow (DuckDB itself only ever writes 0 and 1). A field tagged
//! `arrow.bool8` whose storage is not `int8` is no `arrow.bool8`: it is
//! refused with `UNSUPPORTED_TYPE` naming the column and the type received,
//! never read as a boolean and never admitted as its storage with the tag
//! kept. A nested child is read child by child, as admission's module
//! documentation says of every nested type.

#![allow(
    clippy::disallowed_types,
    reason = "test code: Arrow's field metadata is a HashMap"
)]

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Int32Type;
use arrow_array::{
    Array, ArrayRef, DictionaryArray, FixedSizeListArray, Int8Array, Int16Array, Int32Array,
    LargeListArray, ListArray, MapArray, RecordBatch, StringArray, StructArray, UInt8Array,
};
use arrow_buffer::{NullBuffer, OffsetBuffer, ScalarBuffer};
use arrow_schema::{DataType, Field, FieldRef, Fields, Schema};
use sdm_contract::{CoreError, Detail, ErrorCode, ErrorKind, Scalar};
use sdm_core::ArrowData;
use sdm_host::{admit, admit_from, admitted_type};

const NAME: &str = "ARROW:extension:name";
const METADATA: &str = "ARROW:extension:metadata";

/// A field tagged `arrow.bool8` (with the empty metadata the spec gives
/// it), and one key of its own that admission must keep.
fn tagged(name: &str, storage: DataType, nullable: bool) -> Field {
    Field::new(name, storage, nullable).with_metadata(HashMap::from([
        (NAME.to_owned(), "arrow.bool8".to_owned()),
        (METADATA.to_owned(), String::new()),
        ("kept".to_owned(), "yes".to_owned()),
    ]))
}

/// One column `f` of `field`, holding `array`.
fn data(field: Field, array: ArrayRef) -> ArrowData {
    let schema = Arc::new(Schema::new(vec![field]));
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![array]).unwrap();
    ArrowData::new(schema, vec![batch])
}

/// Admitted both ways a host admits: with no opaque vendor, and as the
/// DuckDB provider's run does.
fn admitted(data: &ArrowData) -> [Result<ArrowData, CoreError>; 2] {
    [admit(data), admit_from(data, Some("DuckDB"))]
}

fn text(error: &CoreError, field: &str) -> String {
    match &error.fields[field] {
        Detail::Scalar(Scalar::Utf8(s)) => s.clone(),
        other => panic!("{field} is {other:?}"),
    }
}

/// The admitted field: boolean, its name and nullability kept, its own
/// metadata kept and the extension's two keys dropped.
fn assert_boolean_field(field: &Field, name: &str, nullable: bool) {
    assert_eq!(field.name(), name);
    assert_eq!(field.data_type(), &DataType::Boolean, "{field:?}");
    assert_eq!(field.is_nullable(), nullable, "{field:?}");
    assert!(
        !field.metadata().contains_key(NAME) && !field.metadata().contains_key(METADATA),
        "the extension's keys are dropped: {field:?}"
    );
    assert_eq!(
        field.metadata().get("kept").map(String::as_str),
        Some("yes"),
        "the field's own metadata is kept: {field:?}"
    );
}

fn bools(array: &dyn Array) -> Vec<Option<bool>> {
    array.as_boolean().iter().collect()
}

/// Storage values with a null slot over a non-zero value (5), and the
/// booleans pyarrow 25.0.1 and DuckDB 1.5.6 read them as (module docs):
/// any non-zero value true, the null slot null whatever its storage.
fn storage() -> (Int8Array, Vec<Option<bool>>) {
    let values = Int8Array::new(
        ScalarBuffer::from(vec![0i8, 1, 2, -1, i8::MAX, i8::MIN, 5]),
        Some(NullBuffer::from(vec![
            true, true, true, true, true, true, false,
        ])),
    );
    let read = vec![
        Some(false),
        Some(true),
        Some(true),
        Some(true),
        Some(true),
        Some(true),
        None,
    ];
    (values, read)
}

// --- values other than 0 and 1 ---------------------------------------------

/// P3 C13 (bool8 edges): an int8 value other than 0 or 1 (2, -1, 127,
/// -128) is true, as the canonical spec says and DuckDB and pyarrow read
/// it; a null stays null.
#[test]
fn a_bool8_value_other_than_0_or_1_is_true() {
    let (values, read) = storage();
    let data = data(tagged("f", DataType::Int8, true), Arc::new(values));
    for admitted in admitted(&data) {
        let admitted = admitted.unwrap();
        assert_boolean_field(admitted.schema.field(0), "f", true);
        assert_eq!(bools(admitted.batches[0].column(0).as_ref()), read);
    }
}

// --- the tag on a nested child -----------------------------------------------

/// P3 C13 (bool8 edges): a list whose item is `arrow.bool8` (DuckDB's
/// lossless BOOLEAN[]) is admitted as list<bool>, every value and null
/// kept, the item read as a top-level `arrow.bool8` is.
#[test]
fn a_bool8_list_item_is_read_as_boolean() {
    let (values, read) = storage();
    let item = Arc::new(tagged("l", DataType::Int8, true));
    let list = ListArray::try_new(
        Arc::clone(&item),
        OffsetBuffer::new(ScalarBuffer::from(vec![0i32, 3, 3, 7])),
        Arc::new(values),
        Some(NullBuffer::from(vec![true, false, true])),
    )
    .unwrap();
    let column = Field::new("f", list.data_type().clone(), true);
    let wanted = admitted_type("f", column.data_type()).unwrap();
    let DataType::List(admitted_item) = &wanted else {
        panic!("admitted as {wanted}");
    };
    assert_boolean_field(admitted_item, "l", true);
    for admitted in admitted(&data(column.clone(), Arc::new(list.clone()))) {
        let admitted = admitted.unwrap();
        assert_eq!(admitted.schema.field(0).data_type(), &wanted);
        let got = admitted.batches[0].column(0).as_list::<i32>().clone();
        assert_eq!(got.value_offsets(), &[0, 3, 3, 7]);
        assert_eq!(got.nulls(), list.nulls());
        assert_eq!(bools(got.values().as_ref()), read);
    }
}

/// P3 C13 (bool8 edges): a struct whose field is `arrow.bool8` (DuckDB's
/// lossless STRUCT(a BOOLEAN)) is admitted as struct<a: bool>, the
/// struct's other field and its nulls kept.
#[test]
fn a_bool8_struct_field_is_read_as_boolean() {
    let (values, read) = storage();
    let rows = values.len();
    let fields = Fields::from(vec![
        tagged("a", DataType::Int8, true),
        Field::new("b", DataType::Int32, false),
    ]);
    let r#struct = StructArray::try_new(
        fields.clone(),
        vec![
            Arc::new(values),
            Arc::new(Int32Array::from_iter_values(0..7)) as ArrayRef,
        ],
        None,
    )
    .unwrap();
    assert_eq!(rows, r#struct.len());
    let column = Field::new("f", DataType::Struct(fields), true);
    let wanted = admitted_type("f", column.data_type()).unwrap();
    let DataType::Struct(admitted_fields) = &wanted else {
        panic!("admitted as {wanted}");
    };
    assert_boolean_field(&admitted_fields[0], "a", true);
    assert_eq!(admitted_fields[1].data_type(), &DataType::Int32);
    for admitted in admitted(&data(column.clone(), Arc::new(r#struct.clone()))) {
        let admitted = admitted.unwrap();
        assert_eq!(admitted.schema.field(0).data_type(), &wanted);
        let got = admitted.batches[0].column(0).as_struct().clone();
        assert_eq!(bools(got.column(0).as_ref()), read);
        assert_eq!(got.column(1).as_ref(), r#struct.column(1).as_ref());
    }
}

/// P3 C13 (bool8 edges): the other nested forms DuckDB hands a lossless
/// boolean over in (module docs): BOOLEAN[2]'s fixed-size list, MAP(INTEGER,
/// BOOLEAN)'s map value, STRUCT(a BOOLEAN)[]'s list of structs, and a large
/// list. Each admitted with the child boolean, each value read.
#[test]
fn every_nested_form_duckdb_gives_reads_its_bool8_child_as_boolean() {
    let item: FieldRef = Arc::new(tagged("item", DataType::Int8, true));
    let values =
        || -> ArrayRef { Arc::new(Int8Array::from(vec![Some(0), Some(1), None, Some(2)])) };
    let read = vec![Some(false), Some(true), None, Some(true)];

    let fixed = FixedSizeListArray::try_new(Arc::clone(&item), 2, values(), None).unwrap();
    let large = LargeListArray::try_new(
        Arc::clone(&item),
        OffsetBuffer::new(ScalarBuffer::from(vec![0i64, 1, 4])),
        values(),
        None,
    )
    .unwrap();
    let inner = Fields::from(vec![tagged("a", DataType::Int8, true)]);
    let structs = StructArray::try_new(inner.clone(), vec![values()], None).unwrap();
    let list_of_structs = ListArray::try_new(
        Arc::new(Field::new("l", DataType::Struct(inner), true)),
        OffsetBuffer::new(ScalarBuffer::from(vec![0i32, 2, 4])),
        Arc::new(structs),
        None,
    )
    .unwrap();
    let entries = Fields::from(vec![
        Field::new("key", DataType::Int32, false),
        tagged("value", DataType::Int8, true),
    ]);
    let map = MapArray::try_new(
        Arc::new(Field::new(
            "entries",
            DataType::Struct(entries.clone()),
            false,
        )),
        OffsetBuffer::new(ScalarBuffer::from(vec![0i32, 3, 4])),
        StructArray::try_new(
            entries,
            vec![Arc::new(Int32Array::from(vec![1, 2, 3, 4])), values()],
            None,
        )
        .unwrap(),
        None,
        false,
    )
    .unwrap();

    let cases: Vec<(&str, ArrayRef)> = vec![
        ("fixed_size_list", Arc::new(fixed)),
        ("large_list", Arc::new(large)),
        ("list<struct>", Arc::new(list_of_structs)),
        ("map", Arc::new(map)),
    ];
    for (form, array) in cases {
        let column = Field::new("f", array.data_type().clone(), true);
        let wanted = admitted_type("f", column.data_type())
            .unwrap_or_else(|e| panic!("{form}: refused {e:?}"));
        for admitted in admitted(&data(column.clone(), Arc::clone(&array))) {
            let admitted = admitted.unwrap_or_else(|e| panic!("{form}: refused {e:?}"));
            assert_eq!(admitted.schema.field(0).data_type(), &wanted, "{form}");
            let got = admitted.batches[0].column(0);
            let leaf: ArrayRef = match got.data_type() {
                DataType::List(_) => {
                    let values = Arc::clone(got.as_list::<i32>().values());
                    match values.data_type() {
                        DataType::Struct(_) => Arc::clone(values.as_struct().column(0)),
                        _ => values,
                    }
                }
                DataType::Map(..) => Arc::clone(got.as_map().values()),
                other => panic!("{form}: admitted as {other}"),
            };
            assert_eq!(bools(leaf.as_ref()), read, "{form}");
        }
        let text = wanted.to_string();
        assert!(
            !text.contains("Int8") && !text.contains("arrow.bool8"),
            "{form}: the child is boolean, its tag dropped: {text}"
        );
    }
}

// --- the tag on a storage other than int8 --------------------------------------

/// The refusal by name: `UNSUPPORTED_TYPE`, the column's path, and the
/// type received there, which names the storage.
fn assert_refused_by_name(result: Result<ArrowData, CoreError>, column: &str, storage: &str) {
    let error = match result {
        Ok(admitted) => panic!(
            "{column}: a {storage} tagged arrow.bool8 was admitted as {:?}",
            admitted.schema
        ),
        Err(error) => error,
    };
    assert_eq!(
        (error.kind, error.code),
        (ErrorKind::Unsupported, ErrorCode::UnsupportedType),
        "{column}: {error:?}"
    );
    assert_eq!(text(&error, "column"), column, "{error:?}");
    let named = text(&error, "arrow_type");
    assert!(named.contains(storage), "{column}: arrow_type {named:?}");
}

/// P3 C13 (bool8 edges): `arrow.bool8` over int16, uint8 or utf8 is no
/// `arrow.bool8`: refused by name, never a boolean, never the storage with
/// the tag kept.
#[test]
fn a_bool8_tag_on_storage_other_than_int8_is_refused_by_name() {
    let cases: Vec<(DataType, ArrayRef, &str)> = vec![
        (
            DataType::Int16,
            Arc::new(Int16Array::from(vec![0, 1, 2])),
            "Int16",
        ),
        (
            DataType::UInt8,
            Arc::new(UInt8Array::from(vec![0, 1, 2])),
            "UInt8",
        ),
        (
            DataType::Utf8,
            Arc::new(StringArray::from(vec!["0", "1", "true"])),
            "Utf8",
        ),
    ];
    for (storage, array, name) in cases {
        let data = data(tagged("f", storage, true), array);
        for result in admitted(&data) {
            assert_refused_by_name(result, "f", name);
        }
    }
}

/// P3 C13 (bool8 edges): the same tag on a nested child's non-int8 storage
/// is refused naming the child's path.
#[test]
fn a_bool8_tag_on_a_nested_child_other_than_int8_is_refused_by_its_path() {
    let item = Arc::new(tagged("l", DataType::Int16, true));
    let list = ListArray::try_new(
        Arc::clone(&item),
        OffsetBuffer::new(ScalarBuffer::from(vec![0i32, 2])),
        Arc::new(Int16Array::from(vec![0, 2])),
        None,
    )
    .unwrap();
    let column = Field::new("f", list.data_type().clone(), true);
    for result in admitted(&data(column, Arc::new(list))) {
        assert_refused_by_name(result, "f.l", "Int16");
    }
    let fields = Fields::from(vec![tagged("a", DataType::Utf8, true)]);
    let r#struct = StructArray::try_new(
        fields.clone(),
        vec![Arc::new(StringArray::from(vec!["1"])) as ArrayRef],
        None,
    )
    .unwrap();
    let column = Field::new("f", DataType::Struct(fields), true);
    for result in admitted(&data(column, Arc::new(r#struct))) {
        assert_refused_by_name(result, "f.a", "Utf8");
    }
}

/// P3 C13 (bool8 edges): a dictionary of int8 tagged `arrow.bool8` (how
/// IPC dictionary-encodes an extension column) is either read as the
/// booleans it holds or refused by name; never decoded to an int8 column
/// that still carries the tag, which reads true as 1.
#[test]
fn a_dictionary_encoded_bool8_is_never_admitted_as_int8() {
    let dictionary: DictionaryArray<Int32Type> = DictionaryArray::try_new(
        Int32Array::from(vec![Some(0), Some(1), None, Some(2)]),
        Arc::new(Int8Array::from(vec![0, 1, -1])),
    )
    .unwrap();
    let field = tagged("f", dictionary.data_type().clone(), true);
    for result in admitted(&data(field, Arc::new(dictionary))) {
        match result {
            Ok(admitted) => {
                assert_boolean_field(admitted.schema.field(0), "f", true);
                assert_eq!(
                    bools(admitted.batches[0].column(0).as_ref()),
                    [Some(false), Some(true), None, Some(true)]
                );
            }
            Err(error) => {
                assert_eq!(error.code, ErrorCode::UnsupportedType, "{error:?}");
                assert_eq!(text(&error, "column"), "f");
            }
        }
    }
}
