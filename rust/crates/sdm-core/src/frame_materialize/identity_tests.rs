// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Source-only tests for the fixed value consumer, pending final composition.

use std::sync::Arc;

use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringArray};
use sdm_contract::frames::materialize::FrameBatchRange;

use super::identity::{Identity, ordered_coverage};
use crate::run::ArrowData;

fn rows(chunks: &[usize]) -> ArrowData {
    let batch = RecordBatch::try_from_iter([
        (
            "entity",
            Arc::new(Int64Array::from(vec![1, 2, 3])) as ArrayRef,
        ),
        (
            "target",
            Arc::new(StringArray::from(vec!["red", "blue", "red"])) as ArrayRef,
        ),
    ])
    .unwrap();
    let mut at = 0;
    let batches = chunks
        .iter()
        .map(|&length| {
            let part = batch.slice(at, length);
            at += length;
            part
        })
        .collect();
    assert_eq!(at, batch.num_rows());
    ArrowData::new(batch.schema(), batches)
}

#[test]
fn task_identity_ignores_attachment_ids_and_arrow_chunking() {
    let mut whole = Identity::new("task");
    whole.value(&("same-source", "same-query", 99_i64)).unwrap();
    whole.arrow(&rows(&[3])).unwrap();
    let mut chunked = Identity::new("task");
    chunked
        .value(&("same-source", "same-query", 99_i64))
        .unwrap();
    chunked.arrow(&rows(&[1, 0, 2])).unwrap();
    assert_eq!(whole.finish(), chunked.finish());
}

#[test]
fn task_identity_binds_actual_query_rows_and_anchors() {
    let original = |query: &str, anchor: i64, reverse: bool| {
        let mut identity = Identity::new("task");
        identity.value(&("source", query, anchor)).unwrap();
        let data = rows(&[3]);
        if reverse {
            let batch = &data.batches[0];
            let indices = arrow_array::UInt32Array::from(vec![2, 1, 0]);
            let mut changed = arrow_select::take::take_record_batch(batch, &indices).unwrap();
            // This changes a label as well as preserving the reversed rows.
            let target = Arc::new(StringArray::from(vec!["red", "blue", "other"])) as ArrayRef;
            let mut columns = changed.columns().to_vec();
            columns[1] = target;
            changed = RecordBatch::try_new(changed.schema(), columns).unwrap();
            identity
                .arrow(&ArrowData::new(changed.schema(), vec![changed]))
                .unwrap();
        } else {
            identity.arrow(&data).unwrap();
        }
        identity.finish()
    };
    let reference = original("PREDICT users.target FOR EACH users.id", 99, false);
    assert_ne!(
        reference,
        original("PREDICT users.other FOR EACH users.id", 99, false)
    );
    assert_ne!(
        reference,
        original("PREDICT users.target FOR EACH users.id", 100, false)
    );
    assert_ne!(
        reference,
        original("PREDICT users.target FOR EACH users.id", 99, true)
    );
}

#[test]
fn independently_useful_parts_require_exact_order_and_complete_coverage() {
    let ranges = [
        FrameBatchRange { start: 0, stop: 2 },
        FrameBatchRange { start: 2, stop: 3 },
    ];
    assert_eq!(ordered_coverage(&ranges, 3, 2, false).unwrap(), 3);
    assert_eq!(ordered_coverage(&ranges[..1], 3, 2, true).unwrap(), 2);
    assert!(ordered_coverage(&ranges[..1], 3, 2, false).is_err());
    assert!(ordered_coverage(&[ranges[1], ranges[0]], 3, 2, false).is_err());
    assert!(ordered_coverage(&[ranges[0], ranges[0]], 3, 2, false).is_err());
    assert!(ordered_coverage(&[FrameBatchRange { start: 0, stop: 3 }], 3, 2, false).is_err());
    assert!(ordered_coverage(&[FrameBatchRange { start: 0, stop: 0 }], 3, 2, true).is_err());
}

#[test]
fn identities_separate_domain_kind_and_length_delimited_components() {
    let hash = |domain: &str, fields: &[&[u8]]| {
        let mut identity = Identity::new(domain);
        for field in fields {
            identity.bytes(field);
        }
        identity.finish()
    };
    assert_ne!(hash("task", &[b"ab", b"c"]), hash("task", &[b"a", b"bc"]));
    assert_ne!(hash("task", &[b"same"]), hash("plan", &[b"same"]));
}
