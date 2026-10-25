// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use arrow_array::builder::{Int64Builder, MapBuilder, StringBuilder};
use arrow_array::types::{Int8Type, Int64Type};
use arrow_array::{
    Array, ArrayRef, DictionaryArray, Int8Array, Int64Array, ListArray, RecordBatch, StringArray,
    StructArray,
};
use arrow_buffer::{OffsetBuffer, ScalarBuffer};
use arrow_schema::{DataType, Field, Schema};

use super::{compact, logical_bytes};
use crate::run::ArrowData;

fn batch(array: ArrayRef) -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "a",
            array.data_type().clone(),
            true,
        )])),
        vec![array],
    )
    .unwrap()
}

#[test]
fn logical_slices_include_offsets_nulls_and_only_the_referenced_nested_rows() {
    let integer: ArrayRef =
        Arc::new(Int64Array::from(vec![Some(0), None, Some(2), Some(3)]).slice(1, 2));
    assert_eq!(logical_bytes(&batch(integer)).unwrap(), 17);

    let text: ArrayRef = Arc::new(
        StringArray::from(vec![
            Some("unrelated"),
            Some("é"),
            Some(""),
            None,
            Some("tail"),
        ])
        .slice(1, 3),
    );
    assert_eq!(logical_bytes(&batch(text)).unwrap(), 19);

    let list: ArrayRef = Arc::new(
        ListArray::from_iter_primitive::<Int64Type, _, _>(vec![
            Some(vec![Some(0), Some(1)]),
            None,
            Some(vec![Some(2), None]),
            Some(vec![Some(4)]),
        ])
        .slice(1, 2),
    );
    assert_eq!(logical_bytes(&batch(list)).unwrap(), 30);

    let mut map = MapBuilder::new(None, StringBuilder::new(), Int64Builder::new());
    for (key, value) in [
        ("unrelated", Some(0)),
        ("x", Some(1)),
        ("é", None),
        ("tail", Some(3)),
    ] {
        map.keys().append_value(key);
        map.values().append_option(value);
        map.append(true).unwrap();
    }
    let map: ArrayRef = Arc::new(map.finish().slice(1, 2));
    assert_eq!(logical_bytes(&batch(map)).unwrap(), 44);

    let child: ArrayRef = Arc::new(StringArray::from(vec!["unrelated", "a", "bb", "tail"]));
    let fields = vec![Arc::new(Field::new("s", child.data_type().clone(), false))];
    let structure: ArrayRef =
        Arc::new(StructArray::new(fields.into(), vec![child], None).slice(1, 2));
    assert_eq!(logical_bytes(&batch(structure)).unwrap(), 15);
}

#[test]
fn accepted_pages_detach_parent_buffers_including_sliced_dictionary_values() {
    let text = StringArray::from(vec![
        "x".repeat(100_000),
        "a".into(),
        "bb".into(),
        "tail".into(),
    ]);
    let values: ArrayRef = Arc::new(text.slice(1, 2));
    let keys = Int8Array::from_iter_values((0..1000).map(|i| (i % 2) as i8)).slice(500, 2);
    let dictionary: ArrayRef =
        Arc::new(DictionaryArray::<Int8Type>::try_new(keys, values).unwrap());
    let original = batch(dictionary);
    assert_eq!(logical_bytes(&original).unwrap(), 17);
    assert!(original.get_array_memory_size() > 100_000);
    let detached = compact(ArrowData::new(original.schema(), vec![original.clone()])).unwrap();
    assert_eq!(detached.batches[0], original);
    assert!(detached.batches[0].get_array_memory_size() < 1024);
    assert_eq!(logical_bytes(&detached.batches[0]).unwrap(), 17);
}

#[test]
fn nested_struct_ranges_use_arrow_array_slices_once() {
    let values: ArrayRef = Arc::new(StringArray::from(vec!["outside", "a", "bb", "tail"]));
    let field = Arc::new(Field::new("s", DataType::Utf8, false));
    // External valid data can retain a parent offset and unsliced children.
    let external = arrow_data::ArrayData::builder(DataType::Struct(vec![field].into()))
        .len(3)
        .offset(1)
        .add_child_data(values.to_data())
        .build()
        .unwrap();
    let structure = arrow_array::make_array(external);
    let lists = ListArray::new(
        Arc::new(Field::new("item", structure.data_type().clone(), false)),
        OffsetBuffer::new(ScalarBuffer::from(vec![0, 1, 3, 3])),
        structure,
        None,
    );
    // The child Struct starts at its row 1: "bb", "tail".
    assert_eq!(
        logical_bytes(&batch(Arc::new(lists.slice(1, 1)))).unwrap(),
        26
    );
    // An empty child range after preceding entries must not reslice them.
    assert_eq!(
        logical_bytes(&batch(Arc::new(lists.slice(2, 1)))).unwrap(),
        12
    );

    let mut builder = MapBuilder::new(None, StringBuilder::new(), Int64Builder::new());
    for key in ["outside", "a", "bb", "tail"] {
        builder.keys().append_value(key);
        builder.values().append_value(1);
        builder.append(true).unwrap();
    }
    let maps = builder.finish();
    let map_field = Arc::new(Field::new("m", maps.data_type().clone(), false));
    let structures = StructArray::new(vec![map_field].into(), vec![Arc::new(maps)], None);
    let nested = ListArray::new(
        Arc::new(Field::new("item", structures.data_type().clone(), false)),
        OffsetBuffer::new(ScalarBuffer::from(vec![0, 1, 3, 4])),
        Arc::new(structures),
        None,
    );
    // List offsets 8, map offsets 12, string offsets 12 + data 3, integers 16.
    assert_eq!(
        logical_bytes(&batch(Arc::new(nested.slice(1, 1)))).unwrap(),
        51
    );
    let slice = batch(Arc::new(nested.slice(1, 1)));
    let detached = compact(ArrowData::new(slice.schema(), vec![slice.clone()])).unwrap();
    assert_eq!(detached.batches[0], slice);
    assert_eq!(logical_bytes(&detached.batches[0]).unwrap(), 51);
}
