// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Complete native row identity is independent of scalar traversal topology.

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, Int64Array, RecordBatch, StringArray, TimestampNanosecondArray,
};
use sdm_contract::data::strings::StringEncoding;
use sdm_contract::ops::{HeldSchema, SourceDescription};
use sdm_engine::frame_identity::{IdentityColumn, encode};
use sdm_engine::frames::Snapshot;
use sdm_engine::graph::{Graph, NativeGraph};
use serde_json::json;

fn description() -> SourceDescription {
    let held: HeldSchema = serde_json::from_value(json!({"tables":[{
        "name":"history","source":{"source":["history"]},
        "columns":[
            {"name":"part","dtype":"string","stype":"ID","stype_before_key":null},
            {"name":"number","dtype":"int64","stype":"ID","stype_before_key":null},
            {"name":"time","dtype":"timestamp[ns]","stype":"timestamp","stype_before_key":null},
            {"name":"__kumo_existing","dtype":"int64","stype":"numerical","stype_before_key":null}
        ],"primary_key":["part","number"],"frame_key_form":"sequence","time_column":"time"
    }],"links":[]}))
    .unwrap();
    sdm_schema::held::read(held, &BTreeMap::new(), true)
        .unwrap()
        .description
}

fn rows() -> RecordBatch {
    RecordBatch::try_from_iter([
        (
            "part",
            Arc::new(StringArray::from(vec!["same", "same", "other", "last"])) as ArrayRef,
        ),
        (
            "number",
            Arc::new(Int64Array::from(vec![1, 1, 2, 3])) as ArrayRef,
        ),
        (
            "time",
            Arc::new(TimestampNanosecondArray::from(vec![
                None,
                Some(1),
                Some(2),
                None,
            ])) as ArrayRef,
        ),
        (
            "__kumo_existing",
            Arc::new(Int64Array::from(vec![10, 11, 12, 13])) as ArrayRef,
        ),
    ])
    .unwrap()
}

#[test]
fn native_topology_retains_compound_identity_and_source_features_without_sql_refusal() {
    let description = description();
    assert!(Graph::new(&description).is_err());
    let native = NativeGraph::new(&description).unwrap();
    assert_eq!(native.primary_keys["history"], ["part", "number"]);
    assert_eq!(
        native.topology.tables[0].primary_key.as_deref(),
        Some("__kumo_key_70617274_6e756d626572")
    );
    assert_eq!(
        native
            .topology
            .features(&native.topology.tables[0])
            .iter()
            .map(|(name, _)| name.as_str())
            .collect::<Vec<_>>(),
        ["part", "number", "time", "__kumo_existing"]
    );
    assert_eq!(description.tables[0].primary_key, ["part", "number"]);
}

#[test]
fn native_identity_duplicate_precedes_time_and_preserves_original_rows() {
    let native = NativeGraph::new(&description()).unwrap();
    let rows = rows();
    let identity = encode(
        &rows,
        &native.primary_keys["history"],
        &BTreeMap::from([("part".into(), StringEncoding::Utf8)]),
    )
    .unwrap();
    let snapshot = Snapshot::new_frame(
        &native,
        BTreeMap::from([("history".into(), rows)]),
        Some(BTreeMap::from([(
            "history".into(),
            BTreeMap::from([("part".into(), StringEncoding::Utf8)]),
        )])),
        BTreeMap::from([("history".into(), identity)]),
    )
    .unwrap();
    let dropped = &snapshot.dropped[0];
    assert_eq!(
        (
            dropped.null_keys,
            dropped.duplicate_keys,
            dropped.null_times
        ),
        (0, 1, 2)
    );
    assert_eq!(snapshot.time_bounds("history").unwrap(), (Some(2), Some(2)));
    let key: ArrayRef = Arc::new(StringArray::from(vec!["other\u{1f}2"]));
    assert_eq!(
        snapshot
            .entity_rows_encoded("history", &[(&key, Some(StringEncoding::Utf8))])
            .unwrap(),
        [2]
    );
    let lost: ArrayRef = Arc::new(StringArray::from(vec!["same\u{1f}1"]));
    assert!(
        snapshot
            .entity_rows_encoded("history", &[(&lost, Some(StringEncoding::Utf8))])
            .is_err()
    );
}

#[test]
fn standard_identity_codec_escapes_components_before_joining_and_keeps_order() {
    let rows = RecordBatch::try_from_iter([
        (
            "a",
            Arc::new(StringArray::from(vec!["a\u{1f}b", "a\\u", "", "é"])) as ArrayRef,
        ),
        (
            "b",
            Arc::new(Int64Array::from(vec![i64::MIN, i64::MAX, 0, 1])) as ArrayRef,
        ),
    ])
    .unwrap();
    let strings = BTreeMap::from([("a".into(), StringEncoding::Utf8)]);
    let result = encode(&rows, &["a".into(), "b".into()], &strings).unwrap();
    assert_eq!(result.encoding, StringEncoding::Utf8);
    let values = result
        .values
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(
        values.iter().collect::<Vec<_>>(),
        [
            Some("a\\ub\u{1f}-9223372036854775808"),
            Some("a\\\\u\u{1f}9223372036854775807"),
            Some("\u{1f}0"),
            Some("é\u{1f}1")
        ]
    );
    let reversed = encode(&rows, &["b".into(), "a".into()], &strings).unwrap();
    assert_ne!(result.values.to_data(), reversed.values.to_data());
}

#[test]
fn observed_native_identity_is_authoritative_and_shape_mismatches_are_refused() {
    let native = NativeGraph::new(&description()).unwrap();
    // Actual native replace/map/add overrides may legitimately return this
    // value instead of the standard rendering of the source components.
    let observed = IdentityColumn {
        values: Arc::new(StringArray::from(vec![
            "native-a", "native-b", "native-c", "native-d",
        ])),
        encoding: StringEncoding::Utf8,
    };
    let strings = Some(BTreeMap::from([(
        "history".into(),
        BTreeMap::from([("part".into(), StringEncoding::Utf8)]),
    )]));
    let snapshot = Snapshot::new_frame(
        &native,
        BTreeMap::from([("history".into(), rows())]),
        strings.clone(),
        BTreeMap::from([("history".into(), observed)]),
    )
    .unwrap();
    let key: ArrayRef = Arc::new(StringArray::from(vec!["native-c"]));
    assert_eq!(
        snapshot
            .entity_rows_encoded("history", &[(&key, Some(StringEncoding::Utf8))])
            .unwrap(),
        [2]
    );
    for values in [
        Arc::new(StringArray::from(vec!["short"])) as ArrayRef,
        Arc::new(Int64Array::from(vec![1, 2, 3, 4])) as ArrayRef,
    ] {
        assert!(
            Snapshot::new_frame(
                &native,
                BTreeMap::from([("history".into(), rows())]),
                strings.clone(),
                BTreeMap::from([(
                    "history".into(),
                    IdentityColumn {
                        values,
                        encoding: StringEncoding::Utf8
                    }
                )])
            )
            .is_err()
        );
    }
}

#[test]
fn standard_codec_rejects_null_components_in_declared_order_before_sanitization() {
    let rows = RecordBatch::try_from_iter([
        (
            "a",
            Arc::new(StringArray::from(vec![Some("ok"), None])) as ArrayRef,
        ),
        (
            "b",
            Arc::new(Int64Array::from(vec![None, Some(1)])) as ArrayRef,
        ),
    ])
    .unwrap();
    let strings = BTreeMap::from([("a".into(), StringEncoding::Utf8)]);
    for key in [vec!["a".into(), "b".into()], vec!["b".into(), "a".into()]] {
        let error = encode(&rows, &key, &strings).unwrap_err();
        assert_eq!(error.fields["column"], sdm_contract::Detail::text(&key[0]));
    }
}

#[test]
fn standard_codec_checks_all_approximate_types_before_any_component_nulls() {
    let rows = RecordBatch::try_from_iter([
        (
            "missing",
            Arc::new(Int64Array::from(vec![None, Some(1)])) as ArrayRef,
        ),
        (
            "approximate",
            Arc::new(arrow_array::Float64Array::from(vec![0.25, 0.5])) as ArrayRef,
        ),
    ])
    .unwrap();
    let error = encode(
        &rows,
        &["missing".into(), "approximate".into()],
        &BTreeMap::new(),
    )
    .unwrap_err();
    assert_eq!(
        error.fields["native_key_reason"],
        sdm_contract::Detail::text("approximate_component")
    );
    assert_eq!(
        error.fields["column"],
        sdm_contract::Detail::text("approximate")
    );
}

#[test]
fn standard_codec_keeps_surrogate_points_distinct_from_supplementary_scalars() {
    let values = [
        b"\xed\xa0\x80".as_slice(),
        b"\xed\xa0\xbd\xed\xb8\xba".as_slice(),
        "😺".as_bytes(),
    ];
    let rows = RecordBatch::try_from_iter([
        (
            "a",
            Arc::new(arrow_array::BinaryArray::from_iter_values(values)) as ArrayRef,
        ),
        ("b", Arc::new(Int64Array::from(vec![0, 0, 0])) as ArrayRef),
    ])
    .unwrap();
    let identity = encode(
        &rows,
        &["a".into(), "b".into()],
        &BTreeMap::from([("a".into(), StringEncoding::Utf8Surrogatepass)]),
    )
    .unwrap();
    assert_eq!(identity.encoding, StringEncoding::Utf8Surrogatepass);
    let text = sdm_engine::strings::StringColumn::new(identity.values.as_ref(), identity.encoding)
        .unwrap();
    let points = (0..3)
        .map(|row| text.get(row).unwrap().unwrap().points().collect::<Vec<_>>())
        .collect::<Vec<_>>();
    assert_eq!(
        points,
        [
            vec![0xd800, 0x1f, 0x30],
            vec![0xd83d, 0xde3a, 0x1f, 0x30],
            vec![0x1f63a, 0x1f, 0x30]
        ]
    );
}

#[test]
fn malformed_native_topology_cannot_hide_or_relabel_a_compound_identity() {
    let graph = NativeGraph::new(&description()).unwrap();
    for change in 0..3 {
        let mut malformed = graph.clone();
        match change {
            0 => malformed.topology.tables[0].primary_key = None,
            1 => malformed.primary_keys.get_mut("history").unwrap().reverse(),
            _ => {
                malformed.primary_keys.remove("history");
            }
        }
        assert!(Snapshot::new_frame(&malformed, BTreeMap::new(), None, BTreeMap::new()).is_err());
    }
}
