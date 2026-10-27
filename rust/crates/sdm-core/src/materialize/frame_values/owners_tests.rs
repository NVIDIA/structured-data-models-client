// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Source-first immutable task/reference ownership, complexity and release.

use super::*;
use arrow_array::{Array, Int64Array};
use sdm_contract::pql::{EntityClause, QueryMode};

fn bound_graph(owner: crate::RetainedFrameGraph) -> BoundGraph {
    BoundGraph::take_retained(
        &mut BTreeMap::from([(AttachmentId(91), Attachment::FrameGraph(owner))]),
        AttachmentId(91),
    )
    .unwrap()
}

fn task(graph: crate::RetainedFrameGraph, predictions: usize) -> BoundTask {
    let graph = bound_graph(graph);
    let column = ColumnRef {
        table: "users".into(),
        column: "id".into(),
    };
    let node = PqlNode {
        dtype: None,
        stype: Some(Stype::Numerical),
        range: None,
        span: None,
        kind: NodeKind::Column {
            column: column.clone(),
        },
    };
    let query = ValidatedQuery {
        entity: node.clone(),
        target: node,
        assuming: None,
        entity_ids: None,
        problem_type: None,
        top_k: None,
        num_forecasts: 1,
        entity_clause: EntityClause::ForEach,
        mode: QueryMode::Inference,
        evaluate: false,
        explain: false,
    };
    let keys: ArrayRef = Arc::new(Int64Array::from(vec![1]));
    let anchors = vec![0];
    let labels = labels::Labels::Int(vec![1]);
    let context = data(labelled_rows(&keys, &anchors, &labels).unwrap());
    let predicted = data(
        RecordBatch::try_from_iter([(
            "entity",
            Arc::new(Int64Array::from_iter_values(
                (0..predictions).map(|row| (row % 3 + 1) as i64),
            )) as ArrayRef,
        )])
        .unwrap(),
    );
    let meaning = FrameTaskMeaning {
        version: FRAME_MATERIALIZATION_VERSION,
        source: graph.source.clone(),
        identity: "labelled-task-fixture".into(),
        query,
        source_query: "PREDICT users.id FOR EACH users.id".into(),
        facts: QueryFacts {
            entity_table: "users".into(),
            problem_type: None,
            windowed: false,
            num_forecasts: 1,
            text: None,
        },
        entity: column,
        mode: FrameRunMode::Fast,
        seed: Seed::from(7_u128),
        unseeded: false,
        prediction_anchor: 0,
        entity_anchor: false,
        no_time_marker: true,
        requested_context: Some(1),
        statistic: None,
        limits: FrameLimits::default(),
    };
    let value = CompletedFrameTask {
        identity: "completed-task-fixture".into(),
        labelled: LabelledFrameTask {
            graph: PreparedFrameGraph {
                version: FRAME_MATERIALIZATION_VERSION,
                source: graph.source.clone(),
                owner: AttachmentId(91),
                native_tables: graph.native.as_ref().clone(),
            },
            meaning,
            context: None,
            prediction_type: FramePredictionType::CoreKey {
                physical: sdm_contract::ColumnType::Int64,
            },
        },
        prediction: None,
        owner: Some(AttachmentId(92)),
        prediction_encoding: None,
        prediction_rows: predictions as u64,
        rows_per_call: 1,
        calls: predictions as u32,
    };
    let asked = Arc::new(Asked {
        entity_table: "users".into(),
        entity_column: "id".into(),
        target: Shape::Column("id".into()),
        range: None,
        task: Task::Regression,
        choosing: Vec::new(),
    });
    let rows = Arc::new(
        TaskRows::new(
            Arc::clone(&keys),
            anchors.clone(),
            one_column(&predicted).unwrap(),
            0,
        )
        .unwrap(),
    );
    BoundTask {
        labels: BoundLabels {
            graph,
            value: value.labelled.clone(),
            labelled: Arc::new(task::LabelledContext {
                keys,
                anchors,
                labels,
            }),
            asked,
            rows: context,
        },
        value,
        predicted,
        rows,
        owner: None,
    }
}

fn plan(task: BoundTask) -> BoundPlan {
    BoundPlan {
        value: FrameRequestPlan {
            version: FRAME_MATERIALIZATION_VERSION,
            identity: "plan-fixture".into(),
            task: task.value.clone(),
            inference: RelationalOptions {
                num_estimators: 1,
                column_shuffle: false,
                category_shuffle: false,
                hop_shuffle: false,
                class_shuffle: false,
                target_transforms: Vec::new(),
                output_type: "regression".into(),
            },
            excluded: Vec::new(),
            top_k: None,
            sampling_values: AttachmentId(93),
        },
        task,
        sampling: Value::Object(Vec::new()),
        deferred: options::values(None).unwrap(),
    }
}

#[test]
fn repeated_small_batches_retain_the_original_task_and_arrow_allocations() {
    let graph = crate::frame_materialize::retained_tests::owner();
    let bound = task(graph.clone(), 100_000);
    let owner = RetainedFrameTask::new(&bound);
    let metadata = bound.value.clone();
    let rows = Arc::clone(&bound.rows);
    let predicted = one_column(&bound.predicted).unwrap();
    drop(bound);
    // Binding does not read/hash/rebuild the 100,000 prediction rows.
    // Only TaskRows::batch concatenates the one selected prediction and
    // existing context. Increasing total predictions changes no bind work.
    for start in 0..64 {
        let mut rebound = metadata.clone();
        rebound.owner = Some(AttachmentId(200 + start));
        rebound.labelled.graph.owner = AttachmentId(400 + start);
        let rebound = owner.bind(bound_graph(graph.clone()), rebound).unwrap();
        assert!(Arc::ptr_eq(&rows, &rebound.rows));
        assert!(Arc::ptr_eq(&owner.data.labelled, &rebound.labels.labelled));
        assert!(Arc::ptr_eq(
            &predicted,
            &one_column(&rebound.predicted).unwrap()
        ));
        let selected = rebound
            .rows
            .batch(start as usize..start as usize + 1)
            .unwrap();
        assert_eq!(selected.keys.len(), 2);
        assert_eq!(selected.predicted.len(), 1);
    }
}

#[test]
fn task_metadata_cannot_rebind_another_meaning_or_same_source_rng_owner() {
    let graph = crate::frame_materialize::retained_tests::owner();
    let bound = task(graph.clone(), 2);
    let owner = RetainedFrameTask::new(&bound);
    let metadata = bound.value.clone();
    drop(bound);
    let mut altered = metadata.clone();
    altered.labelled.meaning.prediction_anchor = 9;
    assert!(owner.bind(bound_graph(graph.clone()), altered).is_err());
    let mut altered = metadata.clone();
    altered.rows_per_call = 2;
    assert!(owner.bind(bound_graph(graph.clone()), altered).is_err());
    let independent = crate::frame_materialize::retained_tests::owner();
    assert_eq!(independent.source(), graph.source());
    assert!(owner.bind(bound_graph(independent), metadata).is_err());
}

#[test]
fn complete_reference_owner_keeps_task_rows_alive_without_holding_a_graph_lease() {
    let graph = crate::frame_materialize::retained_tests::owner();
    let bound = plan(task(graph.clone(), 2));
    let nodes: ArrayRef = Arc::new(UInt64Array::from(vec![0, 1, 2]));
    let references =
        RetainedFrameReferences::new(&bound, Arc::clone(&nodes), "references-fixture".into());
    let rows = Arc::downgrade(&bound.task.rows);
    let weak_nodes = Arc::downgrade(&nodes);
    let metadata = bound.task.value.clone();
    drop(nodes);
    drop(bound);
    assert!(rows.upgrade().is_some());
    // Retained immutable task/reference owners carry no active lease.
    let rebound = references
        .data
        .task
        .bind(bound_graph(graph.clone()), metadata)
        .unwrap();
    let mut rebound = plan(rebound);
    rebound.value.task.owner = Some(AttachmentId(999));
    assert_eq!(
        references.validate(&rebound, "references-fixture").unwrap(),
        3
    );
    assert_eq!(references.rows().batches[0].column(0).len(), 3);
    assert!(
        references
            .validate(&rebound, "another-reference-result")
            .is_err()
    );
    rebound.value.identity = "another-plan".into();
    assert!(references.validate(&rebound, "references-fixture").is_err());
    drop(rebound);
    drop(references);
    assert!(rows.upgrade().is_none());
    assert!(weak_nodes.upgrade().is_none());
    assert!(graph.lease().is_ok());
}

#[test]
fn encoded_part_bodies_remain_independent_until_one_final_assembly() {
    let graph = crate::frame_materialize::retained_tests::owner();
    let plan = plan(task(graph, 64));
    let mut owners = Vec::new();
    for start in 0..64 {
        let part = EncodedFramePart {
            identity: format!("part-{start}"),
            context_identity: format!("context-{start}"),
            range: FrameBatchRange {
                start,
                stop: start + 1,
            },
            payload: AttachmentId(start + 1),
            answering: 0,
            tables: vec!["users".into()],
        };
        let logical = Arc::new(Value::text(&"x".repeat(16_384)));
        let bytes = logical.to_bytes();
        let allocation = bytes.as_ptr();
        let owner = RetainedFramePayload::new(
            &plan,
            &part,
            payload::Built {
                bytes,
                logical,
                answering: 0,
            },
        );
        assert_eq!(owner.wire().as_ptr(), allocation);
        owners.push((part, owner));
    }
    // An earlier body is never an input of a later encoding. Final assembly
    // validates small part metadata and traverses each body exactly once.
    for (part, owner) in &owners {
        assert!(owner.validate(&plan, part).is_ok());
        let clone = owner.clone();
        assert_eq!(clone.wire().as_ptr(), owner.wire().as_ptr());
        let mut rebound = part.clone();
        rebound.payload = AttachmentId(10_000 + part.range.start);
        assert!(clone.validate(&plan, &rebound).is_ok());
        rebound.range.stop += 1;
        assert!(clone.validate(&plan, &rebound).is_err());
    }
    let weak = Arc::downgrade(&owners[0].1.data);
    drop(owners);
    assert!(weak.upgrade().is_none());
}

#[test]
fn hashed_node_rows_cannot_bypass_the_complete_reference_constructor() {
    let graph = crate::frame_materialize::retained_tests::owner();
    let plan = plan(task(graph.clone(), 2));
    let before = crate::frame_materialize::lock(&graph.frames())
        .neighbors
        .checkpoint();
    // Matching row count and a genuine coherence hash still do not establish
    // that complete native conversion and source reference resolution ran.
    let nodes: ArrayRef = Arc::new(UInt64Array::from(vec![0, 1, 2]));
    let identity = reference_identity(&plan.value, &nodes).unwrap();
    let forged = data(RecordBatch::try_from_iter([("node", nodes)]).unwrap());
    let mut attachments = BTreeMap::from([(AttachmentId(66), Attachment::Arrow(forged))]);
    assert!(complete_reference_rows(&plan, &identity, AttachmentId(66), &mut attachments).is_err());
    assert_eq!(
        crate::frame_materialize::lock(&graph.frames())
            .neighbors
            .checkpoint(),
        before
    );
}
