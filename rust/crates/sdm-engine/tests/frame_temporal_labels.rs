// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Source-first local label reads over the existing sanitized snapshot.

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Int64Type, TimestampNanosecondType};
use arrow_array::{ArrayRef, Int64Array, RecordBatch, TimestampNanosecondArray};
use sdm_engine::frames::{Snapshot, TemporalWindow};
use sdm_engine::graph::{Edge, Graph, Table};
use sdm_engine::labels::{self, Aggregation, Bounds, Values};

fn integers(values: Vec<i64>) -> ArrayRef {
    Arc::new(Int64Array::from(values))
}

fn snapshot() -> (Edge, Snapshot) {
    let edge = Edge {
        src: "orders".into(),
        fkey: "user_id".into(),
        dst: "users".into(),
    };
    let graph = Graph {
        tables: vec![
            Table {
                name: "users".into(),
                source: "input_users".into(),
                columns: vec![],
                primary_key: Some("id".into()),
                time_column: None,
            },
            Table {
                name: "orders".into(),
                source: "input_orders".into(),
                columns: vec![],
                primary_key: Some("id".into()),
                time_column: Some("time".into()),
            },
        ],
        edges: vec![edge.clone()],
    };
    let users = RecordBatch::try_from_iter([("id", integers(vec![1, 2, 3]))]).unwrap();
    let orders = RecordBatch::try_from_iter([
        (
            "id",
            Arc::new(Int64Array::from(vec![
                Some(10),
                Some(11),
                Some(12),
                Some(13),
                Some(14),
                Some(15),
                Some(16),
                Some(11),
                None,
            ])) as ArrayRef,
        ),
        ("user_id", integers(vec![1, 1, 1, 1, 1, 1, 2, 1, 1])),
        (
            "time",
            Arc::new(TimestampNanosecondArray::from(vec![
                Some(1_900_000_000),
                Some(800_000_000),
                Some(1_100_000_000),
                Some(1_300_000_000),
                Some(2_000_000_000),
                None,
                Some(1_200_000_000),
                Some(1_500_000_000),
                Some(1_700_000_000),
            ])) as ArrayRef,
        ),
    ])
    .unwrap();
    (
        edge,
        Snapshot::new(
            &graph,
            BTreeMap::from([("users".into(), users), ("orders".into(), orders)]),
        )
        .unwrap(),
    )
}

#[test]
fn local_labels_keep_coarse_second_order_and_original_rows_without_resampling() {
    let (edge, snapshot) = snapshot();
    let selected = snapshot
        .temporal_rows(
            &edge,
            &integers(vec![1, 2, 1, 3]),
            &[TemporalWindow {
                lower: Some(0),
                upper: 1,
            }; 4],
            &["id".into(), "time".into()],
        )
        .unwrap();
    // The C++ temporal hop sorts only floor seconds. Within second 1,
    // original source order remains 1.9, 1.1, 1.3, rather than exact time.
    assert_eq!(selected.source_rows, [0, 2, 3, 6, 0, 2, 3]);
    assert_eq!(selected.batch, [0, 0, 0, 1, 2, 2, 2]);
    assert_eq!(
        selected
            .rows
            .column(0)
            .as_primitive::<Int64Type>()
            .values()
            .as_ref(),
        &[10, 12, 13, 16, 10, 12, 13]
    );
    assert_eq!(selected.rows.num_columns(), 2);
    // Broad seconds deliberately include candidates outside the actual
    // label window. Existing label predicates reapply exact nanoseconds.
    let times = selected
        .rows
        .column(1)
        .as_primitive::<TimestampNanosecondType>()
        .iter()
        .collect::<Vec<_>>();
    let labels = labels::aggregate(
        Aggregation::Count,
        4,
        &selected.batch,
        &Values::Ones(selected.rows.num_rows()),
        &times,
        Bounds {
            starts: Some(&[1_150_000_000; 4]),
            ends: &[1_800_000_000; 4],
        },
    )
    .unwrap();
    assert_eq!(labels.values, [1.0, 1.0, 1.0, 0.0]);
    assert_eq!(labels.present, [true; 4]);
}

#[test]
fn unbounded_and_empty_windows_do_not_revive_sanitized_rows() {
    let (edge, snapshot) = snapshot();
    let selected = snapshot
        .temporal_rows(
            &edge,
            &integers(vec![1, 1]),
            &[
                TemporalWindow {
                    lower: None,
                    upper: 2,
                },
                TemporalWindow {
                    lower: Some(2),
                    upper: 1,
                },
            ],
            &["id".into()],
        )
        .unwrap();
    assert_eq!(selected.source_rows, [1, 0, 2, 3, 4]);
    assert_eq!(selected.batch, [0; 5]);
    assert_eq!(
        snapshot
            .dropped
            .iter()
            .map(|table| table.null_keys)
            .sum::<usize>(),
        1
    );
    assert_eq!(
        snapshot
            .dropped
            .iter()
            .map(|table| table.duplicate_keys)
            .sum::<usize>(),
        1
    );
    assert_eq!(
        snapshot
            .dropped
            .iter()
            .map(|table| table.null_times)
            .sum::<usize>(),
        1
    );
}

#[test]
fn a_temporal_self_edge_keeps_initial_seeds_then_new_nodes_per_example() {
    let edge = Edge {
        src: "nodes".into(),
        fkey: "parent".into(),
        dst: "nodes".into(),
    };
    let graph = Graph {
        tables: vec![Table {
            name: "nodes".into(),
            source: "input_nodes".into(),
            columns: vec![],
            primary_key: Some("id".into()),
            time_column: Some("time".into()),
        }],
        edges: vec![edge.clone()],
    };
    let rows = RecordBatch::try_from_iter([
        ("id", integers(vec![1, 2, 3])),
        ("parent", integers(vec![1, 1, 2])),
        (
            "time",
            Arc::new(TimestampNanosecondArray::from(vec![
                0,
                1_000_000_000,
                2_000_000_000,
            ])) as ArrayRef,
        ),
    ])
    .unwrap();
    let snapshot = Snapshot::new(&graph, BTreeMap::from([("nodes".into(), rows)])).unwrap();
    let selected = snapshot
        .temporal_rows(
            &edge,
            &integers(vec![1, 2]),
            &[TemporalWindow {
                lower: None,
                upper: 2,
            }; 2],
            &["id".into()],
        )
        .unwrap();
    // Native mapper starts with every seed and deduplicates only within
    // its example. Row 1 remains separately present in examples 0 and 1.
    assert_eq!(selected.source_rows, [0, 1, 1, 2]);
    assert_eq!(selected.batch, [0, 1, 0, 1]);
}

#[test]
fn malformed_window_alignment_or_edge_is_rejected() {
    let (edge, snapshot) = snapshot();
    assert!(
        snapshot
            .temporal_rows(&edge, &integers(vec![1]), &[], &[])
            .is_err()
    );
    let missing = Edge {
        fkey: "absent".into(),
        ..edge
    };
    assert!(
        snapshot
            .temporal_rows(
                &missing,
                &integers(vec![1]),
                &[TemporalWindow {
                    lower: None,
                    upper: 1
                }],
                &[]
            )
            .is_err()
    );
}

#[test]
fn temporal_link_labels_use_the_same_rows_exact_window_and_first_seen_order() {
    let (edge, snapshot) = snapshot();
    let selected = snapshot
        .temporal_rows(
            &edge,
            &integers(vec![1, 2, 3]),
            &[TemporalWindow {
                lower: Some(0),
                upper: 1,
            }; 3],
            &["id".into(), "time".into()],
        )
        .unwrap();
    let ids = selected
        .rows
        .column(0)
        .as_primitive::<Int64Type>()
        .iter()
        .collect();
    let times = selected
        .rows
        .column(1)
        .as_primitive::<TimestampNanosecondType>()
        .iter()
        .collect::<Vec<_>>();
    let labelled = sdm_engine::frame_labels::distinct(
        3,
        &selected.batch,
        &labels::Column::Int(ids),
        &times,
        Bounds {
            starts: Some(&[1_000_000_000; 3]),
            ends: &[1_800_000_000; 3],
        },
        None,
    )
    .unwrap();
    assert_eq!(labelled.mask, [true, true, false]);
    assert_eq!(
        labelled.labels,
        vec![
            vec![
                sdm_engine::frame_labels::LinkValue::Int(12),
                sdm_engine::frame_labels::LinkValue::Int(13)
            ],
            vec![sdm_engine::frame_labels::LinkValue::Int(16)],
        ]
    );
}
