// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The key rules (T1.32e), the implementer's own checks of `keys::hold`
//! against the reference's `_check_source_keys`; the test author's
//! vectors are `sdm-testkit`'s `t132e_key_inference.rs`.

use std::collections::BTreeSet;
use std::sync::Arc;

use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use sdm_contract::ops::{Dtype, Stype};
use sdm_contract::{ColumnType, Detail, ErrorCode, Scalar};
use sdm_schema::infer::{InferredColumn, InferredTable, TimeChoice};
use sdm_schema::keys::{self, Counted};

fn column(name: &str, unique: bool) -> InferredColumn {
    InferredColumn {
        name: name.into(),
        dtype: Dtype::Int,
        stype: Stype::Id,
        stype_before_key: None,
        received: ColumnType::Int64,
        nullable: true,
        unique,
        exact_number: None,
    }
}

/// A table keyed by `key`, holding `others` too.
fn table(name: &str, key: &str, others: &[&str], declared: bool) -> InferredTable {
    let mut columns = vec![column(key, declared)];
    columns.extend(others.iter().map(|o| column(o, false)));
    InferredTable {
        name: name.into(),
        source: name.into(),
        columns,
        left_out: vec![],
        unread_times: vec![],
        day_first: vec![],
        tied_keys: vec![],
        primary_key: vec![key.into()],
        time: TimeChoice::default(),
        time_column: None,
        unique: if declared {
            vec![BTreeSet::from([key.to_owned()])]
        } else {
            vec![]
        },
    }
}

fn counted(rows: u64, distinct: u64, missing: u64) -> Counted {
    Counted {
        rows,
        distinct,
        missing,
        example: (rows > distinct + missing).then(|| vec![Scalar::Int64(7)]),
    }
}

fn codes(held: &keys::Held) -> Vec<&str> {
    held.warnings.iter().map(|w| w.code.as_str()).collect()
}

fn none() -> BTreeSet<String> {
    BTreeSet::new()
}

#[test]
fn an_inferred_key_is_kept_at_99_percent_and_declined_under_it() {
    let mut tables = vec![table("payments", "payment_id", &[], false)];
    let held = keys::hold(&mut tables, &[(0, counted(1_000, 990, 0))], &none(), true).unwrap();
    assert_eq!(tables[0].primary_key, ["payment_id"]);
    assert_eq!(held.duplicates, [10]);
    assert_eq!(codes(&held), ["KEY_REPEATS"]);
    let held = keys::hold(&mut tables, &[(0, counted(1_000, 989, 0))], &none(), true).unwrap();
    assert!(tables[0].primary_key.is_empty());
    assert_eq!(held.duplicates, [0]);
    // Nobody else names the column: declined without a word.
    assert!(held.warnings.is_empty());
    // Nulls count against it as repeats do: 2 in 200 keep it, said; 10
    // drop it.
    let mut tables = vec![table("payments", "payment_id", &[], false)];
    let held = keys::hold(&mut tables, &[(0, counted(200, 198, 2))], &none(), true).unwrap();
    assert_eq!(tables[0].primary_key, ["payment_id"]);
    assert_eq!(codes(&held), ["KEY_NO_VALUE"]);
    keys::hold(&mut tables, &[(0, counted(200, 190, 10))], &none(), true).unwrap();
    assert!(tables[0].primary_key.is_empty());
}

#[test]
fn a_declined_key_another_table_names_is_said_when_nearly_a_key() {
    let tables = || {
        vec![
            table("accounts", "account_id", &[], false),
            table("payments", "payment_id", &["account_id"], false),
        ]
    };
    let mut both = tables();
    let held = keys::hold(&mut both, &[(0, counted(100, 95, 0))], &none(), true).unwrap();
    assert_eq!(codes(&held), ["KEY_DECLINED"]);
    let fields = &held.warnings[0].fields;
    assert_eq!(fields["example"], Detail::Scalar(Scalar::Int64(7)));
    // Under 90% distinct, or with the links given, nothing is said.
    let held = keys::hold(&mut tables(), &[(0, counted(100, 50, 0))], &none(), true).unwrap();
    assert!(held.warnings.is_empty());
    let held = keys::hold(&mut tables(), &[(0, counted(100, 95, 0))], &none(), false).unwrap();
    assert!(held.warnings.is_empty());
}

#[test]
fn a_named_key_with_a_null_or_too_many_repeats_is_refused() {
    let stated = BTreeSet::from(["payments".to_owned()]);
    let mut tables = vec![table("payments", "payment_id", &[], false)];
    let err = keys::hold(&mut tables, &[(0, counted(200, 199, 1))], &stated, true).unwrap_err();
    assert_eq!(err.code, ErrorCode::InvalidRequest);
    assert_eq!(err.fields["field"], Detail::text("primary_keys"));
    assert_eq!(err.fields["missing"], Detail::Scalar(Scalar::UInt64(1)));
    // Repeats within 1% are said; past it, refused.
    let held = keys::hold(&mut tables, &[(0, counted(1_000, 995, 0))], &stated, true).unwrap();
    assert_eq!(codes(&held), ["KEY_REPEATS"]);
    assert_eq!(held.duplicates, [5]);
    let err = keys::hold(&mut tables, &[(0, counted(1_000, 989, 0))], &stated, true).unwrap_err();
    assert_eq!(err.fields["repeats"], Detail::Scalar(Scalar::UInt64(11)));
}

/// A key of several columns, which only a caller names, is held as a
/// named key of one is: a row missing any part of it refused, repeats said
/// within 1% and refused past it, each naming its columns in order and
/// showing a repeated value's parts.
#[test]
fn a_named_key_of_several_columns_is_held_as_a_named_key_of_one() {
    let stated = BTreeSet::from(["lines".to_owned()]);
    let lines = || {
        let mut table = table("lines", "order_id", &["line"], false);
        table.primary_key = vec!["order_id".into(), "line".into()];
        vec![table]
    };
    let parts = Detail::List(vec![
        Scalar::Utf8("order_id".into()),
        Scalar::Utf8("line".into()),
    ]);
    let mut two = counted(1_000, 995, 0);
    two.example = Some(vec![Scalar::Int64(0), Scalar::Int64(1)]);
    let mut tables = lines();
    let held = keys::hold(&mut tables, &[(0, two.clone())], &stated, true).unwrap();
    assert_eq!(tables[0].primary_key, ["order_id", "line"]);
    assert_eq!(codes(&held), ["KEY_REPEATS"]);
    let fields = &held.warnings[0].fields;
    assert_eq!(fields["columns"], parts);
    assert!(!fields.contains_key("column"));
    assert_eq!(
        fields["example"],
        Detail::List(vec![Scalar::Int64(0), Scalar::Int64(1)])
    );
    two.distinct = 980;
    let err = keys::hold(&mut lines(), &[(0, two)], &stated, true).unwrap_err();
    assert_eq!(err.fields["columns"], parts);
    assert_eq!(err.fields["repeats"], Detail::Scalar(Scalar::UInt64(20)));
    let err = keys::hold(&mut lines(), &[(0, counted(60, 57, 3))], &stated, true).unwrap_err();
    assert_eq!(err.fields["missing"], Detail::Scalar(Scalar::UInt64(3)));
    assert_eq!(err.fields["columns"], parts);
}

#[test]
fn a_declared_key_with_nulls_is_said_and_its_repeats_held() {
    let mut tables = vec![table("payments", "payment_id", &[], true)];
    let held = keys::hold(&mut tables, &[(0, counted(200, 150, 50))], &none(), true).unwrap();
    assert_eq!(tables[0].primary_key, ["payment_id"]);
    assert_eq!(codes(&held), ["KEY_NO_VALUE"]);
    let err = keys::hold(&mut tables, &[(0, counted(1_000, 900, 0))], &none(), true).unwrap_err();
    assert_eq!(err.fields["field"], Detail::text("keys"));
    assert_eq!(
        err.fields["declared_unenforced"],
        Detail::Scalar(Scalar::Bool(false))
    );
}

#[test]
fn a_repeated_value_is_read_from_its_answer() {
    let batch = |array: ArrayRef| {
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(
                "k",
                array.data_type().clone(),
                true,
            )])),
            vec![array],
        )
        .unwrap()
    };
    assert_eq!(
        keys::example(&[batch(Arc::new(StringArray::from(vec!["k1"])))]),
        Some(vec![Scalar::Utf8("k1".into())])
    );
    assert_eq!(
        keys::example(&[batch(Arc::new(Int64Array::from(Vec::<i64>::new())))]),
        None
    );
    let two = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int64, true),
            Field::new("b", DataType::Int64, true),
        ])),
        vec![
            Arc::new(Int64Array::from(vec![1])),
            Arc::new(Int64Array::from(vec![2])),
        ],
    )
    .unwrap();
    // A key of several columns: its parts, in order.
    assert_eq!(
        keys::example(&[two]),
        Some(vec![Scalar::Int64(1), Scalar::Int64(2)])
    );
}
