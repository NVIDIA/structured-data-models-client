// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Scored-key refusal uses actual task positions before native sanitation.

use super::*;
use arrow_array::{Int32Array, Int64Array, StringArray, UInt64Array};
use sdm_contract::data::Relation;
use sdm_contract::ops::DescribedColumn;

fn entity(unique: bool) -> Table {
    Table {
        name: "people".into(),
        source: Relation::named("physical_people"),
        columns: vec![DescribedColumn {
            name: "person_id".into(),
            untyped: false,
            received: ColumnType::Uint64,
            dtype: Dtype::Int64,
            stype: Stype::Id,
            stype_before_key: None,
            nullable: false,
            unique,
            declared_type: None,
            exact_number: None,
        }],
        primary_key: Some("person_id".into()),
        time_column: None,
    }
}

fn keys(values: Vec<u64>) -> ArrayRef {
    Arc::new(UInt64Array::from(values))
}

fn rows(positions: Vec<i64>, matches: Option<Vec<i64>>) -> RecordBatch {
    let mut columns = vec![(
        BATCH_COLUMN,
        Arc::new(Int64Array::from(positions)) as ArrayRef,
    )];
    if let Some(matches) = matches {
        columns.push((MATCHES_COLUMN, Arc::new(Int64Array::from(matches))));
    }
    RecordBatch::try_from_iter(columns).unwrap()
}

#[test]
fn duplicates_name_distinct_full_width_caller_values_in_scored_order() {
    let scored = keys(vec![u64::MAX, 9_007_199_254_740_993, u64::MAX]);
    let fetched = rows(vec![4, 0, 3, 2, 1, 4, 2], Some(vec![2, 9, 3, 2, 8, 2, 2]));
    let error = validate(&entity(false), "person_id", &scored, 2, &fetched).unwrap_err();
    assert_eq!(error.sentence, Sentence::HeldBySeveralRows);
    assert_eq!(
        error.fields,
        [
            ("field".into(), Detail::text("indices")),
            ("reason".into(), Detail::text("held_by_several_rows")),
            ("table".into(), Detail::text("people")),
            ("key".into(), Detail::text("person_id")),
            (
                "keys".into(),
                Detail::List(vec![
                    Scalar::UInt64(u64::MAX),
                    Scalar::UInt64(9_007_199_254_740_993)
                ]),
            ),
        ]
        .into()
    );
    assert!(error.cause.is_none());
    assert!(error.public_details.is_none());
}

#[test]
fn context_only_duplicates_do_not_refuse_or_reorder_fetched_rows() {
    let fetched = rows(vec![1, 0, 2], Some(vec![8, 9, 1]));
    assert_eq!(
        validate(&entity(false), "person_id", &keys(vec![77]), 2, &fetched).unwrap(),
        vec![1, 0, 2]
    );
}

#[test]
fn repeated_occurrences_of_one_scored_value_use_the_singular_sentence() {
    let error = validate(
        &entity(false),
        "person_id",
        &keys(vec![u64::MAX, u64::MAX]),
        1,
        &rows(vec![2, 1, 2], Some(vec![4, 4, 4])),
    )
    .unwrap_err();
    assert_eq!(error.sentence, Sentence::HeldBySeveralRowsOne);
    assert_eq!(
        error.fields["keys"],
        Detail::List(vec![Scalar::UInt64(u64::MAX)])
    );
    assert!(!error.fields.contains_key("shown"));
    assert!(!error.fields.contains_key("more"));
}

#[test]
fn more_than_five_repeated_keys_keep_all_keys_and_the_first_five_separately() {
    let values = vec![19, 2, 27, 4, 35, 6, 43];
    let all: Vec<Scalar> = values.iter().copied().map(Scalar::UInt64).collect();
    let error = validate(
        &entity(false),
        "person_id",
        &keys(values),
        2,
        &rows(vec![8, 6, 4, 2, 7, 5, 3], Some(vec![2; 7])),
    )
    .unwrap_err();
    assert_eq!(error.sentence, Sentence::HeldBySeveralRowsMany);
    assert_eq!(error.fields["keys"], Detail::List(all.clone()));
    assert_eq!(error.fields["shown"], Detail::List(all[..5].to_vec()));
    assert_eq!(error.fields["more"], Detail::Scalar(Scalar::UInt64(2)));
}

#[test]
fn declared_unique_key_may_omit_counts_but_a_present_count_is_still_read() {
    let scored = keys(vec![77]);
    assert_eq!(
        validate(
            &entity(true),
            "person_id",
            &scored,
            1,
            &rows(vec![1, 0], None)
        )
        .unwrap(),
        vec![1, 0]
    );
    let error = validate(
        &entity(true),
        "person_id",
        &scored,
        1,
        &rows(vec![1], Some(vec![2])),
    )
    .unwrap_err();
    assert_eq!(error.sentence, Sentence::HeldBySeveralRowsOne);
}

#[test]
fn missing_counts_are_not_inferred_unread_for_a_nonunique_key() {
    let error = validate(
        &entity(false),
        "person_id",
        &keys(vec![77]),
        1,
        &rows(vec![1], None),
    )
    .unwrap_err();
    assert_eq!(error.sentence, Sentence::ProviderSchemaMismatch);
    assert_eq!(error.fields["column"], Detail::text(MATCHES_COLUMN));
    assert_eq!(
        error.fields["reason"],
        Detail::text("an answer lacks a column it asked for")
    );
}

#[test]
fn missing_scored_rows_remain_the_later_callers_responsibility() {
    assert_eq!(
        validate(
            &entity(false),
            "person_id",
            &keys(vec![77]),
            1,
            &rows(vec![0], Some(vec![1])),
        )
        .unwrap(),
        vec![0]
    );
}

#[test]
fn invalid_seed_positions_precede_missing_counts_and_duplicate_refusal() {
    for position in [Some(-1), None, Some(2)] {
        let fetched = RecordBatch::try_from_iter([(
            BATCH_COLUMN,
            Arc::new(Int64Array::from(vec![position])) as ArrayRef,
        )])
        .unwrap();
        let error =
            validate(&entity(false), "person_id", &keys(vec![77]), 1, &fetched).unwrap_err();
        assert_eq!(error.sentence, Sentence::ProviderSchemaMismatch);
        assert_eq!(error.fields["column"], Detail::text(BATCH_COLUMN));
        assert_eq!(
            error.fields["reason"],
            Detail::text("a row names no seed asked for")
        );
    }
}

#[test]
fn helper_type_and_presence_errors_keep_the_original_diagnostics() {
    let wrong_batch = RecordBatch::try_from_iter([
        (
            BATCH_COLUMN,
            Arc::new(Int32Array::from(vec![0])) as ArrayRef,
        ),
        (MATCHES_COLUMN, Arc::new(Int64Array::from(vec![3]))),
    ])
    .unwrap();
    let wrong_matches = RecordBatch::try_from_iter([
        (
            BATCH_COLUMN,
            Arc::new(Int64Array::from(vec![0])) as ArrayRef,
        ),
        (MATCHES_COLUMN, Arc::new(StringArray::from(vec!["3"]))),
    ])
    .unwrap();
    let missing_batch = RecordBatch::try_from_iter([(
        MATCHES_COLUMN,
        Arc::new(Int64Array::from(vec![3])) as ArrayRef,
    )])
    .unwrap();
    for (fetched, column, reason) in [
        (wrong_batch, BATCH_COLUMN, "not int64"),
        (wrong_matches, MATCHES_COLUMN, "not int64"),
        (
            missing_batch,
            BATCH_COLUMN,
            "an answer lacks a column it asked for",
        ),
    ] {
        let error =
            validate(&entity(false), "person_id", &keys(vec![77]), 0, &fetched).unwrap_err();
        assert_eq!(error.sentence, Sentence::ProviderSchemaMismatch);
        assert_eq!(error.fields["check"], Detail::text("expected"));
        assert_eq!(error.fields["column"], Detail::text(column));
        assert_eq!(error.fields["reason"], Detail::text(reason));
    }
}
