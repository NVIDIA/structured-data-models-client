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
//! (`column_type`). From T1.32e, a preview that may not hold every row
//! reads the timestamps over the whole table (`time_whole_table`).

use std::sync::Arc;

use arrow_array::builder::{BinaryBuilder, BooleanBuilder, Int32Builder, ListBuilder};
use arrow_array::{
    ArrayRef, BinaryArray, Date32Array, DurationMicrosecondArray, Int64Array, RecordBatch,
    Time64MicrosecondArray, TimestampMicrosecondArray,
};
use arrow_schema::{DataType, Field, Schema, TimeUnit};
use sdm_contract::catalog::CatalogColumn;
use sdm_contract::ops::Stype;
use sdm_contract::{Detail, Scalar};
use sdm_schema::infer::{self, LeftOut, PassedOver, Taken, TextShare, TextTimes, TextUnread};

/// A reading of text as times for these tests, standing in for the
/// engine's (`sdm-engine::time::text`, held to its own vectors there):
/// `2024-01-DD` reads as that day's midnight, `no date` reads as no time,
/// an offset (`+01`) is held against offsets mixed, `code` is text outside
/// the shapes, counted at most only under a name that says time, and
/// anything else (`2024-02-DD`) is a value the reader does not decide,
/// counted in the share at most; the reading refuses both. `either` reads
/// as a time in one order of day and month only: counted among the values
/// read in either order, not at fewest, and refused by the reading as an
/// order left open.
struct Days;

impl Days {
    fn one(value: &str) -> Result<Option<(i128, Option<String>)>, TextUnread> {
        if value == "no date" {
            return Ok(None);
        }
        if value == "code" {
            return Err(TextUnread::Undecided);
        }
        if value == "either" {
            return Err(TextUnread::Order);
        }
        let (date, offset) = match value.split_once('+') {
            Some((date, offset)) => (date, Some(offset.to_owned())),
            None => (value, None),
        };
        let parts: Vec<i64> = date
            .split('-')
            .map(|p| p.parse().map_err(|_| TextUnread::Undecided))
            .collect::<Result<_, _>>()?;
        let [y, m, d] = parts[..] else {
            return Err(TextUnread::Undecided);
        };
        // Days from 2024-01-01 within 2024's January, enough here.
        if y != 2024 || m != 1 {
            return Err(TextUnread::Undecided);
        }
        Ok(Some((i128::from(day(d - 1)) * 1_000, offset)))
    }

    fn mixed<'a>(offsets: impl Iterator<Item = &'a Option<String>>) -> bool {
        offsets.collect::<std::collections::BTreeSet<_>>().len() > 1
    }
}

impl TextTimes for Days {
    fn read(&self, values: &[Option<&str>], utc: bool) -> Result<Vec<Option<i128>>, TextUnread> {
        let read = values
            .iter()
            .map(|value| value.map_or(Ok(None), Days::one))
            .collect::<Result<Vec<_>, _>>()?;
        if !utc && Days::mixed(read.iter().flatten().map(|(_, o)| o)) {
            return Err(TextUnread::MixedZones);
        }
        Ok(read.into_iter().map(|r| r.map(|(at, _)| at)).collect())
    }

    fn share(&self, values: &[Option<&str>]) -> Result<TextShare, TextUnread> {
        let read: Vec<_> = values.iter().flatten().map(|v| Days::one(v)).collect();
        if Days::mixed(read.iter().flatten().flatten().map(|(_, o)| o)) {
            return Err(TextUnread::MixedZones);
        }
        let codes = values.iter().flatten().filter(|v| **v == "code").count();
        let either = values.iter().flatten().filter(|v| **v == "either").count();
        let fewest = read.iter().filter(|r| matches!(r, Ok(Some(_)))).count();
        let open = read.iter().filter(|r| r.is_err()).count() - codes - either;
        Ok(TextShare {
            fewest,
            listed: fewest + either,
            most: fewest + either + open,
            unshaped: codes,
        })
    }

    fn day_first_format(&self, _: &[Option<&str>]) -> Option<String> {
        None
    }
}

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
            key_nullable: false,
            unique: false,
            dtype: None,
        })
        .collect();
    infer::table(
        "things",
        &"src_things".into(),
        &catalog,
        &[],
        &schema,
        &[batch],
        &Days,
    )
    .map_err(|err| match &err.fields["rule"] {
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
            key_nullable: false,
            unique: false,
            dtype: None,
        })
        .collect();
    infer::table(
        "things",
        &"things".into(),
        &catalog,
        &[],
        &schema,
        &[batch],
        &Days,
    )
    .map_err(|err| match &err.fields["rule"] {
        Detail::Scalar(Scalar::Utf8(rule)) => rule.clone(),
        other => panic!("{other:?}"),
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
    assert_eq!(table.source, "src_things".into());
    // A struct column stays refused, by name.
    let fields = arrow_schema::Fields::from(vec![Field::new("a", DataType::Int64, true)]);
    let nested: ArrayRef = Arc::new(arrow_array::StructArray::new(
        fields,
        vec![Arc::new(Int64Array::from_iter_values(0..4)) as ArrayRef],
        None,
    ));
    assert_eq!(of(vec![("s", nested)]).unwrap_err(), "column_type");
}

/// Binary, a duration and a list of binary have no scalar form on the
/// wire either: each is left out and named, with the data type the
/// reference's reading gives it (a list of binary is a list of text to
/// it). A list of any other item stays refused by name.
#[test]
fn binary_a_duration_and_a_list_of_binary_are_left_out_and_other_lists_refused() {
    let binary: ArrayRef = Arc::new(BinaryArray::from_iter(
        [Some(&b"a"[..]), None, Some(b"b"), Some(b"c")]
            .iter()
            .copied(),
    ));
    let duration: ArrayRef = Arc::new(DurationMicrosecondArray::from(vec![
        Some(1),
        None,
        Some(3),
        Some(4),
    ]));
    let mut blobs = ListBuilder::new(BinaryBuilder::new());
    for i in 0..4 {
        blobs.values().append_value(b"x");
        blobs.append(i != 1);
    }
    let blobs: ArrayRef = Arc::new(blobs.finish());
    let table = of(vec![("b", binary), ("d", duration), ("bl", blobs)]).unwrap();
    let dtype = |name: &str, dtype| LeftOut {
        name: name.into(),
        dtype,
    };
    use sdm_contract::ops::Dtype;
    assert_eq!(
        table.left_out,
        [
            dtype("b", Dtype::Binary),
            dtype("d", Dtype::Timedelta),
            dtype("bl", Dtype::Stringlist)
        ]
    );
    assert_eq!(table.columns.len(), 1, "only the key is described");
    // A list of booleans, and a list of lists, stay refused by name.
    let mut flags = ListBuilder::new(BooleanBuilder::new());
    for i in 0..4 {
        flags.values().append_value(i % 2 == 0);
        flags.append(true);
    }
    assert_eq!(
        of(vec![("l", Arc::new(flags.finish()) as ArrayRef)]).unwrap_err(),
        "column_type"
    );
    let mut nested = ListBuilder::new(ListBuilder::new(Int32Builder::new()));
    for i in 0..4 {
        nested.values().values().append_value(i);
        nested.values().append(true);
        nested.append(true);
    }
    assert_eq!(
        of(vec![("l", Arc::new(nested.finish()) as ArrayRef)]).unwrap_err(),
        "column_type"
    );
}

/// A preview of table `things`: `thing_id`, then each named timestamp
/// (`rows` rows, a day apart, or one value in every row when `constant`).
fn previewed(stamps: &[&str], rows: i64, constant: bool) -> infer::Previewed {
    let mut fields = vec![Field::new("thing_id", DataType::Int64, true)];
    let mut arrays: Vec<ArrayRef> = vec![Arc::new(Int64Array::from_iter_values(0..rows))];
    for name in stamps {
        fields.push(Field::new(
            *name,
            DataType::Timestamp(TimeUnit::Microsecond, None),
            true,
        ));
        arrays.push(Arc::new(TimestampMicrosecondArray::from_iter_values(
            (0..rows).map(|i| day(if constant { 0 } else { i })),
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
            key_nullable: false,
            unique: false,
            dtype: None,
        })
        .collect();
    infer::preview(
        "things",
        &"things".into(),
        &catalog,
        &[],
        &schema,
        &[batch],
        &Days,
        infer::Floats::NanValue,
    )
    .unwrap()
}

/// A reading of the whole table: `rows`, and each column's range.
fn whole(rows: usize, columns: &[(&str, usize, bool, Option<i64>)]) -> infer::WholeTable {
    infer::WholeTable {
        rows,
        columns: columns
            .iter()
            .map(|(name, held, same, start)| {
                (
                    (*name).to_owned(),
                    infer::ColumnRange {
                        held: *held,
                        same: *same,
                        low: start.map(|value| Scalar::Timestamp {
                            unit: sdm_contract::Unit::Us,
                            zone: None,
                            value,
                        }),
                        start: start.map(|us| i128::from(us) * 1_000),
                    },
                )
            })
            .collect(),
    }
}

fn internal(err: &sdm_contract::CoreError) -> bool {
    err.code == sdm_contract::ErrorCode::InternalError
}

#[test]
fn a_full_preview_asks_the_whole_table_about_its_timestamps() {
    // The reference asks its source only where the preview returned 1,000
    // rows, and only of the timestamps not passed over by their names.
    let stamps = ["_loaded_at", "birth_date", "updated_at", "started"];
    assert!(
        previewed(&stamps, 999, false)
            .whole_table_columns()
            .is_empty()
    );
    assert_eq!(
        previewed(&stamps, 1_000, false).whole_table_columns(),
        ["updated_at", "started"]
    );
    // No timestamp but load stamps and dates of birth: nothing is asked.
    assert!(
        previewed(&["_loaded_at", "birth_date"], 1_000, false)
            .whole_table_columns()
            .is_empty()
    );
    // `finish` without the whole table's reading, as `table` calls it,
    // which reads no source, takes a preview of every row only: an
    // internal error, the refusal `time_whole_table` gone.
    let err = infer::finish(previewed(&["updated_at"], 1_000, false), None).unwrap_err();
    assert!(internal(&err), "{err:?}");
    assert!(!err.fields.contains_key("rule"), "{err:?}");
}

#[test]
fn the_whole_table_decides_what_the_preview_cannot() {
    // One value in every row of the preview, another past it: taken.
    let late = whole(1_001, &[("happened", 1_001, false, Some(day(0)))]);
    let table = infer::finish(previewed(&["happened"], 1_000, true), Some(&late)).unwrap();
    assert_eq!(table.time_column.as_deref(), Some("happened"));
    // One value in every row of the whole table: passed over, the value
    // named the preview's first.
    let constant = whole(5_000, &[("happened", 5_000, true, Some(day(0)))]);
    let table = infer::finish(previewed(&["happened"], 1_000, true), Some(&constant)).unwrap();
    assert_eq!(table.time_column, None);
    assert_eq!(
        table.time.refused,
        [(
            "happened".to_owned(),
            PassedOver::Constant {
                value: Some(Scalar::Timestamp {
                    unit: sdm_contract::Unit::Us,
                    zone: None,
                    value: day(0),
                }),
                missing: 0,
                rows: 5_000,
            }
        )]
    );
    // Every row of the preview held, most of the whole table missing.
    let sparse = whole(5_000, &[("happened", 1_000, false, Some(day(0)))]);
    let table = infer::finish(previewed(&["happened"], 1_000, false), Some(&sparse)).unwrap();
    assert_eq!(
        table.time.refused,
        [(
            "happened".to_owned(),
            PassedOver::Sparse {
                missing: 4_000,
                rows: 5_000
            }
        )]
    );
}

#[test]
fn where_a_column_starts_is_read_over_the_whole_table() {
    // The preview starts both on day 0; past it `billed` starts earlier.
    let read = whole(
        1_001,
        &[
            ("shipped", 1_001, false, Some(day(0))),
            ("billed", 1_001, false, Some(day(-30))),
        ],
    );
    let table =
        infer::finish(previewed(&["shipped", "billed"], 1_000, false), Some(&read)).unwrap();
    assert_eq!(table.time_column.as_deref(), Some("billed"));
    assert_eq!(table.time.how, Some(Taken::Earliest));
    // A smallest value that is a placeholder (before 1901) is no start:
    // the preview's earliest real value is, as the reference reads it, so
    // the first in table order wins the tie.
    let placeholder = -2_208_988_800_000_000; // 1900-01-01
    let read = whole(
        5_000,
        &[
            ("shipped", 5_000, false, Some(day(0))),
            ("billed", 5_000, false, Some(placeholder)),
        ],
    );
    let table =
        infer::finish(previewed(&["shipped", "billed"], 1_000, false), Some(&read)).unwrap();
    assert_eq!(table.time_column.as_deref(), Some("shipped"));
}

#[test]
fn a_whole_table_reading_other_than_the_one_asked_fails_closed() {
    // Asked and missing, given where nothing is asked, naming other
    // columns, in another order, or empty: each an internal error, never a
    // choice from the preview.
    let two = ["shipped", "billed"];
    let missing = infer::finish(previewed(&two, 1_000, false), None).unwrap_err();
    assert!(internal(&missing), "{missing:?}");
    let given = whole(999, &[("shipped", 999, false, Some(day(0)))]);
    let unasked = infer::finish(previewed(&["shipped"], 999, false), Some(&given)).unwrap_err();
    assert!(internal(&unasked), "{unasked:?}");
    for columns in [
        vec![("shipped", 1_000, false, Some(day(0)))],
        vec![
            ("billed", 1_000, false, Some(day(0))),
            ("shipped", 1_000, false, Some(day(0))),
        ],
        vec![
            ("shipped", 1_000, false, Some(day(0))),
            ("other", 1_000, false, Some(day(0))),
        ],
        vec![],
    ] {
        let read = whole(1_000, &columns);
        let err = infer::finish(previewed(&two, 1_000, false), Some(&read)).unwrap_err();
        assert!(internal(&err), "{columns:?}: {err:?}");
    }
}

/// A one-row answer to `column_ranges`: the rows, then each column's low,
/// high and count.
fn ranges(rows: i64, columns: Vec<(ArrayRef, ArrayRef, i64)>) -> RecordBatch {
    let mut fields = vec![Field::new("count_star()", DataType::Int64, true)];
    let mut arrays: Vec<ArrayRef> = vec![Arc::new(Int64Array::from(vec![rows]))];
    for (at, (low, high, held)) in columns.into_iter().enumerate() {
        fields.push(Field::new(
            format!("min{at}"),
            low.data_type().clone(),
            true,
        ));
        fields.push(Field::new(
            format!("max{at}"),
            high.data_type().clone(),
            true,
        ));
        fields.push(Field::new(format!("count{at}"), DataType::Int64, true));
        arrays.extend([
            low,
            high,
            Arc::new(Int64Array::from(vec![held])) as ArrayRef,
        ]);
    }
    RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays).unwrap()
}

#[test]
fn the_whole_table_s_answer_is_read_by_column_kind() {
    let names = ["ts", "on", "at", "said"].map(str::to_owned);
    let batch = ranges(
        3_000,
        vec![
            (stamps(&[Some(day(0))]), stamps(&[Some(day(2))]), 3_000),
            (
                Arc::new(Date32Array::from(vec![Some(19_723)])),
                Arc::new(Date32Array::from(vec![Some(19_723)])),
                2_999,
            ),
            (
                Arc::new(Time64MicrosecondArray::from(vec![Some(28_800_000_000)])),
                Arc::new(Time64MicrosecondArray::from(vec![Some(28_900_000_000)])),
                10,
            ),
            (texts(&[Some("2024-01-05")]), texts(&[Some("code")]), 7),
        ],
    );
    let read = infer::whole_table(&names, &[batch], &Days).unwrap();
    assert_eq!(read.rows, 3_000);
    let [ts, on, at, said] = &read.columns[..] else {
        panic!("{read:?}");
    };
    assert_eq!((ts.1.held, ts.1.same), (3_000, false));
    assert_eq!(ts.1.start, Some(i128::from(day(0)) * 1_000));
    assert_eq!((on.1.held, on.1.same), (2_999, true));
    assert_eq!(on.1.low, Some(Scalar::Date32(19_723)));
    assert_eq!(on.1.start, Some(19_723 * 86_400_000_000_000));
    // A time of day: counted, its smallest value kept, no instant.
    assert_eq!((at.1.held, at.1.start), (10, None));
    assert!(matches!(at.1.low, Some(Scalar::Time64 { .. })), "{at:?}");
    // Text: its smallest as written, read as the choice reads text.
    assert_eq!(said.1.low, Some(Scalar::Utf8("2024-01-05".into())));
    assert_eq!(said.1.start, Some(i128::from(day(4)) * 1_000));
    // No value held: no low, no start.
    let none = ranges(3, vec![(stamps(&[None]), stamps(&[None]), 0)]);
    let read = infer::whole_table(&names[..1], &[none], &Days).unwrap();
    assert_eq!(
        (read.columns[0].1.low.clone(), read.columns[0].1.start),
        (None, None)
    );
}

#[test]
fn a_whole_table_answer_of_another_shape_is_refused() {
    let one = || vec![(stamps(&[Some(day(0))]), stamps(&[Some(day(1))]), 3)];
    let names = ["ts".to_owned()];
    let good = ranges(3, one());
    assert!(infer::whole_table(&names, std::slice::from_ref(&good), &Days).is_ok());
    // Two rows; no row; another number of columns; a null or negative
    // count: each an internal error.
    let two = RecordBatch::try_new(
        good.schema(),
        vec![
            Arc::new(Int64Array::from(vec![3, 3])) as ArrayRef,
            stamps(&[Some(day(0)), Some(day(0))]),
            stamps(&[Some(day(1)), Some(day(1))]),
            Arc::new(Int64Array::from(vec![3, 3])),
        ],
    )
    .unwrap();
    let empty = good.slice(0, 0);
    let mut cases = vec![
        vec![two],
        vec![empty],
        vec![good.clone(), good.clone()],
        vec![ranges(3, vec![])],
    ];
    let mut columns: Vec<ArrayRef> = good.columns().to_vec();
    columns[3] = Arc::new(Int64Array::from(vec![None::<i64>]));
    cases.push(vec![RecordBatch::try_new(good.schema(), columns).unwrap()]);
    let mut columns: Vec<ArrayRef> = good.columns().to_vec();
    columns[0] = Arc::new(Int64Array::from(vec![-1]));
    cases.push(vec![RecordBatch::try_new(good.schema(), columns).unwrap()]);
    for batches in cases {
        let err = infer::whole_table(&names, &batches, &Days).unwrap_err();
        assert!(internal(&err), "{err:?}");
    }
}

fn texts(values: &[Option<&str>]) -> ArrayRef {
    Arc::new(arrow_array::StringArray::from(values.to_vec()))
}

fn stype(table: &infer::InferredTable, name: &str) -> Stype {
    table.column(name).unwrap().stype
}

#[test]
fn text_is_a_timestamp_by_its_name_and_the_share_its_values_read() {
    // Every value reads: a timestamp whatever the name, and the time
    // column, the one left.
    let days = texts(&[Some("2024-01-02"), Some("2024-01-03"), Some("2024-01-04")]);
    let table = of(vec![("happened", Arc::clone(&days))]).unwrap();
    assert_eq!(stype(&table, "happened"), Stype::Timestamp);
    assert_eq!(table.time_column.as_deref(), Some("happened"));
    // One of three does not: 0.67 alone is no timestamp, and with a name
    // that says time (0.3), 0.97 is not either.
    let some = texts(&[Some("2024-01-02"), Some("no date"), Some("2024-01-04")]);
    assert_ne!(
        stype(
            &of(vec![("happened", Arc::clone(&some))]).unwrap(),
            "happened"
        ),
        Stype::Timestamp
    );
    assert_ne!(
        stype(&of(vec![("ts", Arc::clone(&some))]).unwrap(), "ts"),
        Stype::Timestamp
    );
    // Seven of ten with the name's 0.3 reaches 1.0, summed in floating
    // point as the reference sums it; six of ten does not.
    let mut seven: Vec<Option<&str>> = vec![Some("2024-01-02"); 7];
    seven.extend([Some("no date"); 3]);
    assert_eq!(
        stype(&of(vec![("ts", texts(&seven))]).unwrap(), "ts"),
        Stype::Timestamp
    );
    let mut six: Vec<Option<&str>> = vec![Some("2024-01-02"); 6];
    six.extend([Some("no date"); 4]);
    assert_ne!(
        stype(&of(vec![("ts", texts(&six))]).unwrap(), "ts"),
        Stype::Timestamp
    );
    // Empty text and missing values are not counted.
    let sparse = texts(&[Some("2024-01-02"), Some(""), None, Some("2024-01-04")]);
    assert_eq!(
        stype(&of(vec![("happened", sparse)]).unwrap(), "happened"),
        Stype::Timestamp
    );
}

#[test]
fn a_text_time_column_is_passed_over_for_its_text_and_chosen_by_its_instants() {
    // One value written in every row: passed over, the value its text.
    let same = texts(&[Some("2024-01-02"); 3]);
    let table = of(vec![("happened", same)]).unwrap();
    assert_eq!(table.time_column, None);
    assert_eq!(
        table.time.refused,
        [(
            "happened".to_owned(),
            PassedOver::Constant {
                value: Some(Scalar::Utf8("2024-01-02".into())),
                missing: 0,
                rows: 3
            }
        )]
    );
    // Of a text and a typed timestamp, the one starting earliest.
    let table = of(vec![
        ("later", stamps(&[Some(day(5)), Some(day(6)), Some(day(7))])),
        (
            "sooner",
            texts(&[Some("2024-01-02"), Some("2024-01-03"), Some("2024-01-09")]),
        ),
    ])
    .unwrap();
    assert_eq!(table.time_column.as_deref(), Some("sooner"));
    assert_eq!(table.time.how, Some(Taken::Earliest));
}

#[test]
fn text_the_reading_does_not_decide_is_text_under_any_name() {
    // A value the reading does not decide is no time, under a name that
    // says time too (the owner's item 24 of 2026-10-14; DT-15): text, not
    // refused.
    let undecided = texts(&[Some("2024-02-02"), Some("2024-01-03")]);
    assert_ne!(
        stype(&of(vec![("happened", undecided)]).unwrap(), "happened"),
        Stype::Timestamp
    );
    // Offsets mixed, which the reference refuses.
    let mixed = texts(&[Some("2024-01-02+01"), Some("2024-01-03+02")]);
    assert_eq!(
        of(vec![("happened", mixed)]).unwrap_err(),
        "text_time_zones"
    );
    // Undecided, but one value reads as no time and the name does not say
    // time: no timestamp at most, decided.
    let bounded = texts(&[Some("2024-02-02"), Some("no date")]);
    assert_ne!(
        stype(
            &of(vec![("happened", Arc::clone(&bounded))]).unwrap(),
            "happened"
        ),
        Stype::Timestamp
    );
    // With the name's 0.3, one in two reading as no time cannot reach 1.0.
    assert_ne!(
        stype(&of(vec![("ts", bounded)]).unwrap(), "ts"),
        Stype::Timestamp
    );
    // Nor one in ten, the values the list reads deciding it: text.
    let mut close: Vec<Option<&str>> = vec![Some("2024-02-02"); 9];
    close.push(Some("no date"));
    assert_ne!(
        stype(&of(vec![("ts", texts(&close))]).unwrap(), "ts"),
        Stype::Timestamp
    );
    // A text time column whose first 100 values read, and whose later
    // ones the reading does not decide: refused only when the choice asks
    // where it starts, among several.
    let mut long: Vec<Option<&str>> = vec![Some("2024-01-02"); 100];
    long.push(Some("2024-02-02"));
    let later = stamps(&(0..101).map(|i| Some(day(5 + i))).collect::<Vec<_>>());
    assert_eq!(
        of(vec![
            ("a_date", Arc::clone(&later)),
            ("b_date", texts(&long))
        ])
        .unwrap_err(),
        "text_time"
    );
    let alone = of(vec![("b_date", texts(&long))]).unwrap();
    assert_eq!(alone.time_column.as_deref(), Some("b_date"));
}

#[test]
fn text_outside_the_shapes_is_no_time_under_any_name() {
    // Codes under a name that says nothing of time: no timestamp, decided,
    // whatever another reader would make of them.
    let codes = texts(&[Some("code"), Some("code"), Some("2024-01-02")]);
    assert_ne!(
        stype(
            &of(vec![("account", Arc::clone(&codes))]).unwrap(),
            "account"
        ),
        Stype::Timestamp
    );
    // Nor under a name that says time (the owner's item 24 of
    // 2026-10-14; DT-17): text, not refused.
    assert_ne!(
        stype(&of(vec![("ts", codes)]).unwrap(), "ts"),
        Stype::Timestamp
    );
    // The values the list reads decide it either way.
    let mut most: Vec<Option<&str>> = vec![Some("2024-01-02"); 9];
    most.push(Some("code"));
    assert_eq!(
        stype(&of(vec![("ts", texts(&most))]).unwrap(), "ts"),
        Stype::Timestamp
    );
}

#[test]
fn an_order_the_values_leave_open_is_refused_where_it_decides_the_score() {
    // Values read as times in one order of day and month only: the score
    // is 0.3 at fewest and 1.3 in either order, so the order decides it.
    let open = texts(&[Some("either"); 10]);
    assert_eq!(
        of(vec![("ts", Arc::clone(&open))]).unwrap_err(),
        "text_time_order"
    );
    // Where the order does not decide the score, nothing is refused: one
    // in two read in either order cannot reach 1.0 under a name that does
    // not say time.
    let short = texts(&[Some("either"), Some("no date")]);
    assert_ne!(
        stype(&of(vec![("happened", short)]).unwrap(), "happened"),
        Stype::Timestamp
    );
    // A value outside the list as well: refused for it, `text_time`, as a
    // source's column with both is (the core's reader refuses a value
    // outside the list before an open order).
    let mut both: Vec<Option<&str>> = vec![Some("either"); 9];
    both.push(Some("2024-02-02"));
    assert_eq!(of(vec![("ts", texts(&both))]).unwrap_err(), "text_time");
    let mut coded: Vec<Option<&str>> = vec![Some("either"); 9];
    coded.push(Some("code"));
    assert_eq!(of(vec![("ts", texts(&coded))]).unwrap_err(), "text_time");
}

#[test]
fn a_whole_table_read_that_failed_keeps_what_the_preview_doubts() {
    // P1B-38 rejected by the owner: as the reference's choice when its
    // `full_range` raises. A column the preview passes over for its values
    // is kept, named unconfirmed, and taken only when no other is.
    let table = infer::finish_unchecked(previewed(&["happened"], 1_000, true)).unwrap();
    assert_eq!(table.time_column.as_deref(), Some("happened"));
    assert!(table.time.refused.is_empty());
    assert_eq!(
        table.time.unconfirmed,
        [(
            "happened".to_owned(),
            PassedOver::Constant {
                value: Some(Scalar::Timestamp {
                    unit: sdm_contract::Unit::Us,
                    zone: None,
                    value: day(0),
                }),
                missing: 0,
                rows: 1_000,
            }
        )]
    );
    // Beside a column the preview does not doubt, the doubted one is not
    // taken.
    let mut fields = vec![Field::new("thing_id", DataType::Int64, true)];
    let mut arrays: Vec<ArrayRef> = vec![Arc::new(Int64Array::from_iter_values(0..1_000))];
    for (name, constant) in [("happened", true), ("shipped", false)] {
        fields.push(Field::new(
            name,
            DataType::Timestamp(TimeUnit::Microsecond, None),
            true,
        ));
        arrays.push(Arc::new(TimestampMicrosecondArray::from_iter_values(
            (0..1_000).map(|i| day(if constant { 0 } else { i })),
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
            key_nullable: false,
            unique: false,
            dtype: None,
        })
        .collect();
    let both = infer::preview(
        "things",
        &"things".into(),
        &catalog,
        &[],
        &schema,
        &[batch],
        &Days,
        infer::Floats::NanValue,
    )
    .unwrap();
    let table = infer::finish_unchecked(both).unwrap();
    assert_eq!(table.time_column.as_deref(), Some("shipped"));
    assert_eq!(table.time.unconfirmed.len(), 1);
    // Under a full preview nothing was asked: as `finish` with no reading.
    let small = infer::finish_unchecked(previewed(&["happened"], 999, true)).unwrap();
    assert_eq!(small.time_column, None);
    assert!(small.time.unconfirmed.is_empty());
}
