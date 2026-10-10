// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The time column chosen among a table's timestamps (T1.21d): a stamp of a
//! load, of a birth or of the last edit passed over by its name, as the
//! reference's `choose_time_column` passes them over, where this build
//! refused them all before (`time_name`); a lone edit stamp kept as the
//! last resort, which the links settle.

use std::sync::Arc;

use arrow_array::{ArrayRef, Int64Array, RecordBatch, TimestampMicrosecondArray};
use arrow_schema::{DataType, Field, Schema, TimeUnit};
use sdm_contract::catalog::CatalogColumn;
use sdm_contract::ops::Stype;
use sdm_contract::{Detail, Scalar};
use sdm_schema::infer::{self, PassedOver, Taken};

/// Table `things` of `rows` rows: `thing_id`, then each named timestamp,
/// each row's value a day apart, or one value in every row when `constant`.
fn things(stamps: &[&str], rows: i64, constant: bool) -> Result<infer::InferredTable, String> {
    let mut fields = vec![Field::new("thing_id", DataType::Int64, true)];
    let mut arrays: Vec<ArrayRef> = vec![Arc::new(Int64Array::from_iter_values(0..rows))];
    for name in stamps {
        fields.push(Field::new(
            *name,
            DataType::Timestamp(TimeUnit::Microsecond, None),
            true,
        ));
        arrays.push(Arc::new(TimestampMicrosecondArray::from_iter_values(
            (0..rows)
                .map(|i| 1_700_000_000_000_000 + if constant { 0 } else { i * 86_400_000_000 }),
        )));
    }
    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(Arc::clone(&schema), arrays).unwrap();
    let catalog: Vec<CatalogColumn> = schema
        .fields()
        .iter()
        .map(|f| CatalogColumn {
            name: f.name().clone(),
            declared_type: String::new(),
            not_null: false,
            primary_key: false,
        })
        .collect();
    infer::table("things", &catalog, &[], &schema, &[batch]).map_err(
        |err| match &err.fields["rule"] {
            Detail::Scalar(Scalar::Utf8(rule)) => rule.clone(),
            other => panic!("{other:?}"),
        },
    )
}

fn inferred(stamps: &[&str], rows: i64) -> infer::InferredTable {
    things(stamps, rows, false).unwrap()
}

#[test]
fn stamps_of_a_load_a_birth_and_an_edit_are_passed_over_by_name() {
    let table = inferred(&["_loaded_at", "birth_date", "updated_at", "started"], 9);
    assert_eq!(table.time.column.as_deref(), Some("started"));
    assert_eq!(table.time.how, Some(Taken::Only));
    assert_eq!(
        table.time.refused,
        [
            ("_loaded_at".to_owned(), PassedOver::LoadStamp),
            ("birth_date".to_owned(), PassedOver::BirthDate),
            ("updated_at".to_owned(), PassedOver::EditStamp),
        ]
    );
    assert!(
        table
            .columns
            .iter()
            .skip(1)
            .all(|c| c.stype == Stype::Timestamp)
    );
}

#[test]
fn a_lone_edit_stamp_is_kept_as_the_last_resort() {
    let table = inferred(&["_loaded_at", "modified_date"], 9);
    assert_eq!(table.time.column.as_deref(), Some("modified_date"));
    assert_eq!(table.time.how, Some(Taken::EditStamp));
    assert_eq!(table.time.fallback.as_deref(), Some("modified_date"));
    // Settled by a link to the table: refused, and said with its reason.
    let linked = table.time.for_links(true);
    assert_eq!(linked.column, None);
    assert_eq!(
        linked
            .rejected()
            .map(|(c, _)| c.as_str())
            .collect::<Vec<_>>(),
        ["_loaded_at", "modified_date"]
    );
    // Two edit stamps and nothing else: none taken.
    let two = inferred(&["updated_at", "modified_date"], 9);
    assert_eq!(two.time.column, None);
    assert_eq!(two.time.fallback, None);
}

#[test]
fn what_the_choice_does_not_port_stays_refused_by_name() {
    // An edit stamp of one value is refused for its values, as the
    // reference refuses it before its name; a load stamp's values are not
    // read, so it is passed over by name.
    assert_eq!(
        things(&["modified_date"], 9, true).unwrap_err(),
        "time_values"
    );
    let loaded = things(&["_loaded_at"], 9, true).unwrap();
    assert_eq!(loaded.time.column, None);
    assert_eq!(
        loaded.time.refused,
        [("_loaded_at".to_owned(), PassedOver::LoadStamp)]
    );
    // Several timestamps left, and a preview that may not hold every row.
    assert_eq!(
        things(&["started", "ended"], 9, false).unwrap_err(),
        "time_several"
    );
    assert_eq!(
        things(&["updated_at"], 1_000, false).unwrap_err(),
        "time_whole_table"
    );
}
