// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use arrow_array::{Int64Array, UInt64Array};
use sdm_contract::data::Relation;
use sdm_contract::ops::DescribedColumn;

fn context() -> LookupContext {
    LookupContext {
        entity: Table {
            name: "people".into(),
            source: Relation::named("people"),
            columns: vec![DescribedColumn {
                name: "id".into(),
                received: ColumnType::Uint64,
                dtype: Dtype::Int64,
                stype: Stype::Id,
                stype_before_key: None,
                untyped: false,
                nullable: false,
                unique: false,
                declared_type: None,
                exact_number: None,
            }],
            primary_key: Some("id".into()),
            time_column: None,
        },
        key: "id".into(),
        scored: Arc::new(UInt64Array::from(vec![u64::MAX])),
        num_train: 1,
        range: sdm_contract::frames::materialize::FrameBatchRange { start: 7, stop: 8 },
    }
}

#[test]
fn actual_task_batch_controls_scored_refusal_before_any_business_column_exists() {
    let owner =
        RetainedLookupRows::new(Arc::new(context()), Relation::named("pinned"), ReplyId(19));
    let rows = RecordBatch::try_from_iter([
        (
            BATCH_COLUMN,
            Arc::new(Int64Array::from(vec![0, 1])) as ArrayRef,
        ),
        (
            MATCHES_COLUMN,
            Arc::new(Int64Array::from(vec![9, 2])) as ArrayRef,
        ),
    ])
    .unwrap();
    let error = owner.validate(&rows).unwrap_err();
    assert_eq!(error.sentence, Sentence::HeldBySeveralRowsOne);
    assert_eq!(
        error.fields["keys"],
        Detail::List(vec![Scalar::UInt64(u64::MAX)])
    );
    let context_only = rows.slice(0, 1);
    assert_eq!(owner.validate(&context_only).unwrap(), [0]);
}

#[test]
fn read_and_range_binding_are_complete_and_do_not_admit_a_same_table_other_read() {
    let context = Arc::new(context());
    let owner =
        RetainedLookupRows::new(Arc::clone(&context), Relation::named("pinned"), ReplyId(19));
    let meaning = owner.meaning(AttachmentId(8));
    assert_eq!(meaning.read, ReplyId(19));
    assert_eq!(meaning.source, Relation::named("pinned"));
    assert_eq!(meaning.range, context.range);
    assert_eq!(meaning.context_rows, 1);
    assert_eq!(meaning.scored_rows, 1);
    assert!(owner.matches(&meaning));
    let mut wrong = meaning.clone();
    wrong.read = ReplyId(20);
    assert!(!owner.matches(&wrong));
    wrong = meaning.clone();
    wrong.range.start = 6;
    assert!(!owner.matches(&wrong));
    wrong = meaning.clone();
    wrong.context_rows = 0;
    assert!(!owner.matches(&wrong));
    wrong = meaning;
    wrong.source = Relation::named("people");
    assert!(!owner.matches(&wrong));
}
