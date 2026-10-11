// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The time column chosen among a table's timestamps: a stamp of a load, of
//! a birth or of the last edit passed over by its name, as the reference's
//! `choose_time_column` passes them over, where this build refused them all
//! before (`time_name`); a lone edit stamp kept as the last resort, which
//! the links settle (T1.21d). From T1.23, the rest of the choice: a
//! timestamp passed over for its values (`time_values`), the one creation
//! stamp or the earliest of several (`time_several`), a time of day passed
//! over as holding no time (`time_of_day`), and a list column left out
//! (`column_type`).

use std::sync::Arc;

use arrow_array::builder::{Int32Builder, ListBuilder};
use arrow_array::{
    ArrayRef, Date32Array, Int64Array, RecordBatch, Time64MicrosecondArray,
    TimestampMicrosecondArray,
};
use arrow_schema::{DataType, Field, Schema, TimeUnit};
use sdm_contract::catalog::CatalogColumn;
use sdm_contract::ops::Stype;
use sdm_contract::{Detail, Scalar};
use sdm_schema::infer::{self, LeftOut, PassedOver, Taken};

/// A table `things` of the columns given, in order, after its key.
fn of(columns: Vec<(&str, ArrayRef)>) -> Result<infer::InferredTable, String> {
    let rows = columns.first().map_or(0, |(_, a)| a.len());
    let mut fields = vec![Field::new("thing_id", DataType::Int64, true)];
    let mut arrays: Vec<ArrayRef> = vec![Arc::new(Int64Array::from_iter_values(
        0..i64::try_from(rows).unwrap(),
    ))];
    for (name, array) in columns {
        fields.push(Field::new(name, array.data_type().clone(), true));
        arrays.push(array);
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
    infer::table("things", "src_things", &catalog, &[], &schema, &[batch]).map_err(|err| match &err
        .fields["rule"]
    {
        Detail::Scalar(Scalar::Utf8(rule)) => rule.clone(),
        other => panic!("{other:?}"),
    })
}

/// Microseconds since the epoch of `days` after 2024-01-01.
fn day(days: i64) -> i64 {
    1_704_067_200_000_000 + days * 86_400_000_000
}

fn stamps(values: &[Option<i64>]) -> ArrayRef {
    Arc::new(TimestampMicrosecondArray::from(values.to_vec()))
}

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
    infer::table("things", "things", &catalog, &[], &schema, &[batch]).map_err(|err| {
        match &err.fields["rule"] {
            Detail::Scalar(Scalar::Utf8(rule)) => rule.clone(),
            other => panic!("{other:?}"),
        }
    })
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
fn a_timestamp_is_passed_over_for_its_values_and_why_is_kept() {
    // One value in every row: passed over, naming the value; a load stamp's
    // values are not read, so it is passed over by its name.
    let constant = things(&["modified_date"], 9, true).unwrap();
    assert_eq!(constant.time.column, None);
    assert_eq!(
        constant.time.refused,
        [(
            "modified_date".to_owned(),
            PassedOver::Constant {
                value: Some(Scalar::Timestamp {
                    unit: sdm_contract::Unit::Us,
                    zone: None,
                    value: 1_700_000_000_000_000
                }),
                missing: 0,
                rows: 9
            }
        )]
    );
    let loaded = things(&["_loaded_at"], 9, true).unwrap();
    assert_eq!(
        loaded.time.refused,
        [("_loaded_at".to_owned(), PassedOver::LoadStamp)]
    );
    // Held in fewer than half the rows; held in none; one value where held.
    let table = of(vec![
        (
            "sparse",
            stamps(&[Some(day(1)), None, None, Some(day(2)), None]),
        ),
        ("empty", stamps(&[None, None, None, None, None])),
        (
            "same",
            stamps(&[Some(day(3)), None, Some(day(3)), Some(day(3)), Some(day(3))]),
        ),
        (
            "kept",
            stamps(&[
                Some(day(1)),
                Some(day(2)),
                Some(day(3)),
                Some(day(4)),
                Some(day(5)),
            ]),
        ),
    ])
    .unwrap();
    assert_eq!(table.time.column.as_deref(), Some("kept"));
    assert_eq!(table.time.how, Some(Taken::Only));
    let reasons: Vec<(&str, &PassedOver)> = table
        .time
        .refused
        .iter()
        .map(|(c, w)| (c.as_str(), w))
        .collect();
    assert_eq!(
        reasons,
        [
            (
                "sparse",
                &PassedOver::Sparse {
                    missing: 3,
                    rows: 5
                }
            ),
            (
                "empty",
                &PassedOver::Constant {
                    value: None,
                    missing: 5,
                    rows: 5
                }
            ),
            (
                "same",
                &PassedOver::Constant {
                    value: Some(Scalar::Timestamp {
                        unit: sdm_contract::Unit::Us,
                        zone: None,
                        value: day(3)
                    }),
                    missing: 1,
                    rows: 5
                }
            ),
        ]
    );
    // A single row's one value is no evidence it never changes.
    let one = of(vec![("at", stamps(&[Some(day(1))]))]).unwrap();
    assert_eq!(one.time.column.as_deref(), Some("at"));
}

#[test]
fn of_several_the_creation_stamp_then_the_earliest_is_taken() {
    let rows = |shift: i64| stamps(&(0..6).map(|i| Some(day(i + shift))).collect::<Vec<_>>());
    // The one creation stamp, wherever the others start.
    let created = of(vec![("ended", rows(-9)), ("created_at", rows(0))]).unwrap();
    assert_eq!(created.time.column.as_deref(), Some("created_at"));
    assert_eq!(created.time.how, Some(Taken::Created));
    assert_eq!(created.time.among, ["ended", "created_at"]);
    // Otherwise the earliest, a date against a timestamp at its midnight.
    let dates: ArrayRef = Arc::new(Date32Array::from_iter_values(
        (0..6).map(|i| 19_723 + i), // 2024-01-01 onwards
    ));
    let earliest = of(vec![("shipped", rows(1)), ("ordered", dates)]).unwrap();
    assert_eq!(earliest.time.column.as_deref(), Some("ordered"));
    assert_eq!(earliest.time.how, Some(Taken::Earliest));
    assert_eq!(earliest.time.among, ["shipped", "ordered"]);
    // A tie: the first in table order.
    let tie = of(vec![("started", rows(0)), ("ended", rows(0))]).unwrap();
    assert_eq!(tie.time.column.as_deref(), Some("started"));
    // A placeholder is no start: 1900-01-01 and the epoch are passed by.
    let placeholder = -2_208_988_800_000_000; // 1900-01-01
    let held = |first: i64| {
        stamps(
            &std::iter::once(Some(first))
                .chain((1..6).map(|i| Some(day(i + 3))))
                .collect::<Vec<_>>(),
        )
    };
    let skipped = of(vec![
        ("cancelled", held(placeholder)),
        ("epoch", held(0)),
        ("placed", rows(0)),
    ])
    .unwrap();
    assert_eq!(skipped.time.column.as_deref(), Some("placed"));
    // Two creation stamps: the earliest of them, the others passed by.
    let two = of(vec![
        ("created_at", rows(2)),
        ("creation_ts", rows(1)),
        ("other", rows(-9)),
    ])
    .unwrap();
    assert_eq!(two.time.column.as_deref(), Some("creation_ts"));
    assert_eq!(two.time.among, ["created_at", "creation_ts"]);
}

#[test]
fn a_time_of_day_is_passed_over_as_holding_no_time_and_said() {
    let times: ArrayRef = Arc::new(Time64MicrosecondArray::from(vec![
        Some(1_800_000_000),
        Some(5_400_000_000),
        None,
        Some(1_800_000_000),
        Some(9_000_000_000),
        Some(12_600_000_000),
    ]));
    let table = of(vec![("tod", times)]).unwrap();
    assert_eq!(table.time.column, None);
    assert_eq!(
        table.time.refused,
        [(
            "tod".to_owned(),
            PassedOver::Constant {
                value: None,
                missing: 6,
                rows: 6
            }
        )]
    );
    let [unread] = table.unread_times.as_slice() else {
        panic!("{:?}", table.unread_times)
    };
    assert_eq!((unread.unread, unread.rows), (5, 6));
    let time = |value| Scalar::Time64 {
        unit: sdm_contract::Unit::Us,
        value,
    };
    assert_eq!(
        unread.examples,
        [
            time(1_800_000_000),
            time(5_400_000_000),
            time(9_000_000_000)
        ]
    );
}

#[test]
fn a_list_column_is_left_out_and_other_nested_types_stay_refused() {
    let mut lists = ListBuilder::new(Int32Builder::new());
    for i in 0..4 {
        lists.values().append_value(i);
        lists.append(i % 2 == 0);
    }
    let list: ArrayRef = Arc::new(lists.finish());
    let table = of(vec![("f", list)]).unwrap();
    assert_eq!(
        table.left_out,
        [LeftOut {
            name: "f".into(),
            dtype: sdm_contract::ops::Dtype::Intlist
        }]
    );
    assert_eq!(table.columns.len(), 1);
    assert_eq!(table.source, "src_things");
    // A struct column stays refused, by name.
    let fields = arrow_schema::Fields::from(vec![Field::new("a", DataType::Int64, true)]);
    let nested: ArrayRef = Arc::new(arrow_array::StructArray::new(
        fields,
        vec![Arc::new(Int64Array::from_iter_values(0..4)) as ArrayRef],
        None,
    ));
    assert_eq!(of(vec![("s", nested)]).unwrap_err(), "column_type");
}

#[test]
fn what_the_choice_does_not_port_stays_refused_by_name() {
    // A preview that may not hold every row.
    assert_eq!(
        things(&["updated_at"], 1_000, false).unwrap_err(),
        "time_whole_table"
    );
}
