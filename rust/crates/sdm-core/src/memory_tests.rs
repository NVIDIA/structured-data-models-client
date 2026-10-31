// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Provider budgets charge returned rows even when Arrow retains a much larger parent.
use std::sync::Arc;

use arrow_array::builder::{Int32Builder, ListBuilder, MapBuilder, StringBuilder};
use arrow_array::types::Int32Type;
use arrow_array::{
    Array, ArrayRef, DictionaryArray, Int32Array, RecordBatch, StringArray, StructArray,
};
use arrow_schema::{Field, Schema};

fn bytes(array: ArrayRef) -> u64 {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "value",
        array.data_type().clone(),
        true,
    )]));
    crate::memory::batch_bytes(&RecordBatch::try_new(schema, vec![array]).unwrap()).unwrap()
}

#[test]
fn string_slice_does_not_charge_unreturned_parent_values() {
    let padding = "x".repeat(65_536);
    let parent = StringArray::from(vec![padding.as_str(), "ok", padding.as_str()]);
    let slice: ArrayRef = Arc::new(parent.slice(1, 1));
    assert!(slice.get_array_memory_size() > 100_000);
    assert_eq!(bytes(slice), bytes(Arc::new(StringArray::from(vec!["ok"]))));
}

fn lists(rows: &[&[&str]]) -> ArrayRef {
    let mut builder = ListBuilder::new(StringBuilder::new());
    for row in rows {
        for value in *row {
            builder.values().append_value(value);
        }
        builder.append(true);
    }
    Arc::new(builder.finish())
}

#[test]
fn nested_list_and_struct_slices_charge_only_selected_children() {
    let padding = "x".repeat(65_536);
    let parent = lists(&[&[&padding], &["a", "bb"], &[&padding]]);
    let field = Arc::new(Field::new("items", parent.data_type().clone(), true));
    let parent = StructArray::from(vec![(field.clone(), parent)]);
    let compact = StructArray::from(vec![(field, lists(&[&["a", "bb"]]))]);
    let slice: ArrayRef = Arc::new(parent.slice(1, 1));
    assert!(slice.get_array_memory_size() > 100_000);
    assert_eq!(bytes(slice), bytes(Arc::new(compact)));
}

fn maps(rows: &[&[(&str, i32)]]) -> ArrayRef {
    let mut builder = MapBuilder::new(None, StringBuilder::new(), Int32Builder::new());
    for row in rows {
        for (key, value) in *row {
            builder.keys().append_value(key);
            builder.values().append_value(*value);
        }
        builder.append(true).unwrap();
    }
    Arc::new(builder.finish())
}

#[test]
fn map_slice_does_not_charge_entries_outside_its_offsets() {
    let padding = "x".repeat(65_536);
    let parent = maps(&[&[(&padding, 1)], &[("key", 2)], &[(&padding, 3)]]);
    let slice = parent.slice(1, 1);
    assert!(slice.get_array_memory_size() > 100_000);
    assert_eq!(bytes(slice), bytes(maps(&[&[("key", 2)]])));
}

#[test]
fn dictionary_slice_charges_each_referenced_value_once() {
    let padding = "x".repeat(65_536);
    let parent = DictionaryArray::<Int32Type>::try_new(
        Int32Array::from(vec![0, 1, 2, 1, 3]),
        Arc::new(StringArray::from(vec![
            padding.as_str(),
            "ok",
            "yes",
            padding.as_str(),
        ])),
    )
    .unwrap();
    let compact = DictionaryArray::<Int32Type>::try_new(
        Int32Array::from(vec![0, 1, 0]),
        Arc::new(StringArray::from(vec!["ok", "yes"])),
    )
    .unwrap();
    let slice: ArrayRef = Arc::new(parent.slice(1, 3));
    assert!(slice.get_array_memory_size() > 100_000);
    assert_eq!(bytes(slice), bytes(Arc::new(compact)));
}
