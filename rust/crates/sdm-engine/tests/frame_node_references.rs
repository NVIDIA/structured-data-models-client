// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Concrete node references preserve task order and the existing walk stream.

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{ArrayRef, Int64Array, RecordBatch};
use sdm_contract::ErrorCode;
use sdm_engine::frames::{Snapshot, Walked};
use sdm_engine::graph::{Edge, Graph, Table};
use sdm_engine::random::Mt19937;

fn integers(values: Vec<i64>) -> ArrayRef {
    Arc::new(Int64Array::from(values))
}

fn snapshot() -> (Graph, Snapshot) {
    let graph = Graph {
        tables: ["users", "orders"]
            .into_iter()
            .map(|name| Table {
                name: name.to_owned(),
                source: name.into(),
                columns: vec![],
                primary_key: Some("id".to_owned()),
                time_column: None,
            })
            .collect(),
        edges: vec![Edge {
            src: "orders".to_owned(),
            fkey: "user_id".to_owned(),
            dst: "users".to_owned(),
        }],
    };
    // Row 1 is removed as a repeated primary key. A serialized reference
    // must never revive it even though its index is within the Arrow batch.
    let users = RecordBatch::try_from_iter([("id", integers(vec![1, 1, 3]))]).unwrap();
    let orders = RecordBatch::try_from_iter([
        ("id", integers((100..109).collect())),
        ("user_id", integers(vec![1, 1, 1, 2, 2, 2, 3, 3, 3])),
    ])
    .unwrap();
    let snapshot = Snapshot::new(
        &graph,
        BTreeMap::from([("users".to_owned(), users), ("orders".to_owned(), orders)]),
    )
    .unwrap();
    (graph, snapshot)
}

fn ids(walked: &Walked, table: &str) -> Vec<i64> {
    walked
        .tables
        .iter()
        .find(|found| found.name == table)
        .unwrap()
        .rows
        .column_by_name("id")
        .unwrap()
        .as_primitive::<Int64Type>()
        .values()
        .to_vec()
}

#[test]
fn resolved_rows_keep_repeated_entities_and_order_without_rewriting_keys() {
    let (graph, snapshot) = snapshot();
    let keys = integers(vec![3, 1, 3]);
    let rows = snapshot.entity_rows("users", &keys).unwrap();
    assert_eq!(rows, [2, 0, 2]);
    let mut rng = Mt19937::seeded(123);
    let before = rng.checkpoint();
    let walked = snapshot
        .walk_resolved(&graph, "users", &rows, &[0, 0, 0], &[-1], &mut rng)
        .unwrap();
    assert_eq!(ids(&walked, "users"), [3, 1, 3]);
    assert_eq!(
        ids(&walked, "orders"),
        [106, 107, 108, 100, 101, 102, 106, 107, 108]
    );
    assert_eq!(walked.tables[0].batch, [0, 1, 2]);
    assert_eq!(walked.tables[1].batch, [0, 0, 0, 1, 1, 1, 2, 2, 2]);
    // Native conversion consumes these core-selected original positions
    // from immutable owners. Sanitized row 1 is never revived or renumbered.
    assert_eq!(walked.tables[0].source_rows, [2, 0, 2]);
    assert_eq!(walked.tables[1].source_rows, [6, 7, 8, 0, 1, 2, 6, 7, 8]);
    assert_eq!(walked.links.len(), 1);
    assert_eq!(
        walked.links[0].pairs,
        [
            (0, 0),
            (1, 0),
            (2, 0),
            (3, 1),
            (4, 1),
            (5, 1),
            (6, 2),
            (7, 2),
            (8, 2)
        ]
    );
    assert_eq!(rng.checkpoint(), before);
}

#[test]
fn selected_source_positions_replay_raw_rows_without_sampling() {
    let (_, snapshot) = snapshot();
    let selected = snapshot
        .selected_rows("users", &[2, 0, 2], &["id".to_owned()])
        .unwrap();
    assert_eq!(
        selected
            .column(0)
            .as_primitive::<Int64Type>()
            .values()
            .as_ref(),
        &[3, 1, 3]
    );
    // A position must have survived the graph's actual sanitization.
    for rows in [&[1][..], &[3][..]] {
        assert_eq!(
            snapshot
                .selected_rows("users", rows, &["id".to_owned()])
                .unwrap_err()
                .code,
            ErrorCode::InvalidMessage
        );
    }
    assert!(snapshot.selected_rows("missing", &[], &[]).is_err());
    assert!(
        snapshot
            .selected_rows("users", &[0], &["missing".to_owned()])
            .is_err()
    );
}

#[test]
fn selected_link_pairs_bind_source_rows_and_example_ownership() {
    let (graph, snapshot) = snapshot();
    let edge = &graph.edges[0];
    let source = [6, 7, 8, 0, 1, 2, 6, 7, 8];
    let source_batch = [0, 0, 0, 1, 1, 1, 2, 2, 2];
    let target = [2, 0, 2];
    let target_batch = [0, 1, 2];
    let pairs = snapshot
        .selected_link_pairs(edge, &source, &source_batch, &target, &target_batch)
        .unwrap();
    assert_eq!(pairs, (0..9).map(|row| (row, row / 3)).collect::<Vec<_>>());
    assert!(
        snapshot
            .selected_link_pairs(edge, &source, &source_batch[..8], &target, &target_batch)
            .is_err()
    );
    assert!(
        snapshot
            .selected_link_pairs(edge, &[0], &[0], &[1], &[0])
            .is_err()
    );
    // A selected target from a different example cannot own the reference.
    assert!(
        snapshot
            .selected_link_pairs(edge, &[0], &[0], &[0], &[1])
            .unwrap()
            .is_empty()
    );
}

#[test]
fn malformed_or_removed_node_rows_are_rejected_before_neighbor_draws() {
    let (graph, snapshot) = snapshot();
    for rows in [&[2, 1][..], &[2, 3][..], &[usize::MAX][..]] {
        let mut rng = Mt19937::seeded(123);
        let before = rng.checkpoint();
        let error = snapshot
            .walk_resolved(&graph, "users", rows, &vec![0; rows.len()], &[2], &mut rng)
            .err()
            .expect("invalid node reference was accepted");
        assert_eq!(error.code, ErrorCode::InvalidMessage);
        assert_eq!(rng.checkpoint(), before);
    }
    let mut rng = Mt19937::seeded(123);
    let before = rng.checkpoint();
    assert!(
        snapshot
            .walk_resolved(&graph, "users", &[0], &[], &[2], &mut rng)
            .is_err()
    );
    assert_eq!(rng.checkpoint(), before);
}

#[test]
fn concrete_node_walk_shares_the_existing_sample_and_continuing_stream() {
    let (graph, snapshot) = snapshot();
    let keys = integers(vec![3, 1, 3]);
    let rows = snapshot.entity_rows("users", &keys).unwrap();
    let mut by_keys = Mt19937::seeded(123);
    let mut by_rows = Mt19937::seeded(123);
    for _ in 0..2 {
        let ordinary = snapshot
            .walk(&graph, "users", &keys, &[0, 0, 0], &[2], &mut by_keys)
            .unwrap();
        let resolved = snapshot
            .walk_resolved(&graph, "users", &rows, &[0, 0, 0], &[2], &mut by_rows)
            .unwrap();
        for table in ["users", "orders"] {
            assert_eq!(ids(&ordinary, table), ids(&resolved, table));
        }
        for (ordinary, resolved) in ordinary.tables.iter().zip(&resolved.tables) {
            assert_eq!(ordinary.source_rows, resolved.source_rows);
            assert_eq!(ordinary.batch, resolved.batch);
            assert_eq!(ordinary.source_rows.len(), ordinary.rows.num_rows());
        }
        assert_eq!(ordinary.links[0].pairs, resolved.links[0].pairs);
        assert_eq!(by_keys.checkpoint(), by_rows.checkpoint());
    }
}
