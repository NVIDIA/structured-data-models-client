// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! One key meaning across ordinary and explicit native Unicode buffers.

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow_array::{ArrayRef, BinaryArray, LargeBinaryArray, RecordBatch, StringArray};
use sdm_contract::data::strings::StringEncoding;
use sdm_contract::scalar::unicode::UnicodeText;
use sdm_engine::frames::Snapshot;
use sdm_engine::graph::{Graph, Table};
use sdm_engine::keys::{self, Key};

#[test]
fn ordinary_native_and_large_offset_keys_share_equality_and_order() {
    let pair = [0xed, 0xa0, 0xbd, 0xed, 0xb8, 0xba];
    let scalar = "😺".as_bytes();
    let expected = vec![
        Some(Key::Text("ordinary".into())),
        Some(Key::Text(
            UnicodeText::from_utf8_surrogatepass(pair.to_vec()).unwrap(),
        )),
        Some(Key::Text("😺".into())),
        None,
    ];
    for array in [
        Arc::new(BinaryArray::from(vec![
            Some(b"ordinary".as_slice()),
            Some(pair.as_slice()),
            Some(scalar),
            None,
        ])) as ArrayRef,
        Arc::new(LargeBinaryArray::from(vec![
            Some(b"ordinary".as_slice()),
            Some(pair.as_slice()),
            Some(scalar),
            None,
        ])) as ArrayRef,
    ] {
        let keys = keys::read_strings(&array, StringEncoding::Utf8Surrogatepass).unwrap();
        assert_eq!(keys, expected);
        assert_ne!(keys[1], keys[2]);
        let ordinary: ArrayRef = Arc::new(StringArray::from(vec!["ordinary", "😺"]));
        assert_eq!(
            keys::read(&ordinary).unwrap(),
            [keys[0].clone(), keys[2].clone()]
        );
        // Arbitrary Binary is never silently admitted as a text key.
        assert!(keys::read(&array).is_err());
        assert!(keys::read_strings(&array, StringEncoding::Utf8).is_err());
    }
}

#[test]
fn malformed_native_key_bytes_fail_without_replacement_or_pair_merging() {
    for bytes in [&[0xed, 0xa0][..], &[0xc0, 0x80], &[0xf4, 0x90, 0x80, 0x80]] {
        let values: ArrayRef = Arc::new(BinaryArray::from(vec![bytes]));
        assert!(keys::read_strings(&values, StringEncoding::Utf8Surrogatepass).is_err());
    }
}

#[test]
fn native_snapshot_references_preserve_pair_scalar_and_literal_missing_spellings() {
    let pair = [0xed, 0xa0, 0xbd, 0xed, 0xb8, 0xba];
    let values: ArrayRef = Arc::new(BinaryArray::from(vec![
        pair.as_slice(),
        "😺".as_bytes(),
        b"<NA>".as_slice(),
    ]));
    let graph = Graph {
        tables: vec![Table {
            name: "users".into(),
            source: "input_users".into(),
            columns: vec![],
            primary_key: Some("id".into()),
            time_column: None,
        }],
        edges: vec![],
    };
    let rows = RecordBatch::try_from_iter([("id", Arc::clone(&values))]).unwrap();
    let snapshot = Snapshot::new_native(
        &graph,
        BTreeMap::from([("users".into(), rows)]),
        BTreeMap::from([(
            "users".into(),
            BTreeMap::from([("id".into(), StringEncoding::Utf8Surrogatepass)]),
        )]),
    )
    .unwrap();
    assert_eq!(
        snapshot
            .entity_rows_encoded(
                "users",
                &[(&values, Some(StringEncoding::Utf8Surrogatepass))]
            )
            .unwrap(),
        [0, 1, 2]
    );
    let ordinary: ArrayRef = Arc::new(StringArray::from(vec!["😺", "<NA>"]));
    assert_eq!(
        snapshot
            .entity_rows_encoded("users", &[(&ordinary, Some(StringEncoding::Utf8))])
            .unwrap(),
        [1, 2]
    );
    assert!(
        snapshot
            .entity_rows_encoded("users", &[(&values, None)])
            .is_err()
    );
}
