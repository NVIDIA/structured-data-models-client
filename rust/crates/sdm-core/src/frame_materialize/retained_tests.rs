// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Source-first lifetime and no-rebuild tests for the actual graph owner.

use super::retained::RetainedFrameGraph;
use super::*;
use arrow_array::Int64Array;
use sdm_contract::frames::materialize::{FrameNativeColumn, FrameNativeOrigin, FrameNativeTable};
use sdm_contract::ops::{ColumnRef, DescribedColumn, DescribedTable};

fn owner() -> RetainedFrameGraph {
    let description = SourceDescription {
        tables: vec![DescribedTable {
            name: "users".into(),
            source: "input_users".into(),
            columns: vec![DescribedColumn {
                name: "id".into(),
                received: sdm_contract::ColumnType::Int64,
                dtype: Dtype::Int,
                stype: Stype::Id,
                stype_before_key: None,
                nullable: false,
                unique: true,
                declared_type: None,
                exact_number: None,
            }],
            primary_key: vec!["id".into()],
            frame_key_form: None,
            time_column: None,
            time_column_inferred: None,
            duplicate_keys: None,
        }],
        links: vec![],
    };
    let graph = Graph::new(&description).unwrap();
    let rows =
        RecordBatch::try_from_iter([("id", Arc::new(Int64Array::from(vec![1, 2, 3])) as ArrayRef)])
            .unwrap();
    let tables = vec![(
        Relation::from("input_users"),
        ArrowData::new(rows.schema(), vec![rows.clone()]),
    )];
    let source = fingerprint(&description, &tables).unwrap();
    let snapshot = Snapshot::new(&graph, BTreeMap::from([("users".into(), rows)])).unwrap();
    let frames = Arc::new(Mutex::new(FrameRun {
        tables,
        metadata: Some(FrameCheckpointMetadata {
            source: source.clone(),
            description: description.clone(),
            neighbors: Mt19937::seeded(123).checkpoint(),
        }),
        snapshot: Some(snapshot),
        neighbors: Mt19937::seeded(123),
        entropy: None,
        retained_owner: false,
    }));
    RetainedFrameGraph::new(
        PreparedGraph {
            source: source.clone(),
            description,
            graph,
        },
        frames,
        vec![FrameNativeTable {
            source: "input_users".into(),
            columns: vec![FrameNativeColumn {
                name: "id".into(),
                origin: FrameNativeOrigin::BuiltinPrimitive,
                strings: None,
            }],
        }],
        source,
    )
}

#[test]
fn native_source_binding_is_core_resolved_independently_of_logical_aliases() {
    let owner = owner();
    let binding = owner
        .native_binding(&ColumnRef {
            table: "users".into(),
            column: "id".into(),
        })
        .unwrap();
    assert_eq!(binding.source, Relation::from("input_users"));
    assert_eq!(binding.column, "id");
    assert!(
        owner
            .native_binding(&ColumnRef {
                table: "input_users".into(),
                column: "id".into()
            })
            .is_err()
    );
    assert!(
        owner
            .native_binding(&ColumnRef {
                table: "users".into(),
                column: "missing".into()
            })
            .is_err()
    );
}

#[test]
fn every_rebound_batch_reuses_the_same_frame_and_prepared_graph_allocations() {
    let owner = owner();
    let prepared = owner.prepared();
    let frames = owner.frames();
    for _ in 0..12 {
        let lease = owner.try_lease().unwrap();
        assert!(Arc::ptr_eq(&prepared, &owner.prepared()));
        assert!(Arc::ptr_eq(&frames, &owner.frames()));
        let frame = lock(&frames);
        let entity = prepared.graph.table("users").unwrap();
        let keys: ArrayRef = Arc::new(Int64Array::from(vec![3, 1]));
        assert_eq!(frame.resolve_references(entity, &[&keys]).unwrap(), [2, 0]);
        drop(frame);
        drop(lease);
    }
}

#[test]
fn overlapping_owner_access_is_busy_and_does_not_change_neighbor_state() {
    let owner = owner();
    let before = lock(&owner.frames()).neighbors.checkpoint();
    let lease = owner.try_lease().unwrap();
    let another = owner.clone();
    let error = std::thread::spawn(move || {
        another
            .try_lease()
            .err()
            .expect("overlapping graph access succeeded")
    })
    .join()
    .unwrap();
    assert_eq!(error.code, sdm_contract::ErrorCode::RunBusy);
    assert_eq!(lock(&owner.frames()).neighbors.checkpoint(), before);
    drop(lease);
    assert!(owner.try_lease().is_ok());
}

#[test]
fn owner_and_active_lease_keep_rows_until_the_last_reference_is_released() {
    let owner = owner();
    let frames = Arc::downgrade(&owner.frames());
    let lease = owner.try_lease().unwrap();
    let second = owner.clone();
    drop(owner);
    drop(second);
    assert!(frames.upgrade().is_some());
    drop(lease);
    assert!(frames.upgrade().is_none());
}

#[test]
fn unwind_releases_the_owner_lease_without_resetting_neighbors() {
    let owner = owner();
    let before = lock(&owner.frames()).neighbors.checkpoint();
    let held = owner.clone();
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _lease = held.try_lease().unwrap();
        panic!("contained test unwind");
    }));
    assert!(owner.try_lease().is_ok());
    assert_eq!(lock(&owner.frames()).neighbors.checkpoint(), before);
}

#[test]
fn a_panic_while_mutating_frame_state_cannot_reuse_a_partly_updated_owner() {
    let owner = owner();
    let held = owner.clone();
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _lease = held.try_lease().unwrap();
        let frames = held.frames();
        let mut frames = lock(&frames);
        frames.neighbors = Mt19937::seeded(999);
        panic!("contained test frame-state unwind");
    }));
    let error = owner
        .try_lease()
        .err()
        .expect("a poisoned frame owner was reused");
    assert_eq!(error.code, sdm_contract::ErrorCode::RunPoisoned);
}
