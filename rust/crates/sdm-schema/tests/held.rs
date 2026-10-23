// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The schema as a host holds it, read into the schema the core works on
//! and written back (T3.16): each column's facts the host's, as handed; a
//! column with none given the facts the source was asked for, else read
//! by its data type and written back with none, else, where no data type
//! is to be read by, left out; a table or column held twice refused.

use std::collections::{BTreeMap, BTreeSet};

use sdm_contract::ops::{Dtype, HeldColumn, HeldSchema, HeldTable, SourceFacts, Stype};
use sdm_contract::{ColumnType, ErrorCode, Scalar};
use sdm_schema::held;

fn facts(dtype: Dtype, received: ColumnType) -> SourceFacts {
    SourceFacts {
        dtype,
        received,
        nullable: false,
        unique: true,
        declared_type: Some("HUGEINT".into()),
        exact_number: None,
    }
}

fn schema() -> HeldSchema {
    HeldSchema {
        tables: vec![HeldTable {
            name: "users".into(),
            source: "app_users".into(),
            columns: vec![
                HeldColumn {
                    name: "user_id".into(),
                    dtype: "int64".into(),
                    stype: Stype::Id,
                    stype_before_key: Some(Stype::Numerical),
                    source: Some(facts(Dtype::Int, ColumnType::Int64)),
                },
                HeldColumn {
                    name: "seen".into(),
                    dtype: "timestamp[us]".into(),
                    stype: Stype::Timestamp,
                    stype_before_key: None,
                    source: None,
                },
            ],
            primary_key: vec!["user_id".into()],
            time_column: Some("seen".into()),
            duplicate_keys: Some(Scalar::UInt64(3)),
        }],
        links: vec![],
    }
}

#[test]
fn facts_held_are_read_as_handed_and_written_back_whole() {
    let read = held::read(schema(), &BTreeMap::new(), true).unwrap();
    let users = &read.description.tables[0];
    assert_eq!(users.source, "app_users".into());
    assert_eq!(users.duplicate_keys, Some(Scalar::UInt64(3)));
    let key = &users.columns[0];
    assert_eq!(
        (key.dtype, &key.received, key.nullable, key.unique),
        (Dtype::Int, &ColumnType::Int64, false, true)
    );
    assert_eq!(key.declared_type.as_deref(), Some("HUGEINT"));
    assert_eq!(key.stype_before_key, Some(Stype::Numerical));
    // A column with no facts is read by its data type, and written back
    // with none.
    assert_eq!(users.columns[1].dtype, Dtype::Date);
    assert_eq!(
        read.unfacted,
        BTreeSet::from([("users".to_owned(), "seen".to_owned())])
    );
    assert_eq!(held::held(&read.description, &read.unfacted), schema());
}

#[test]
fn facts_the_source_gave_fill_a_column_that_held_none() {
    let found = BTreeMap::from([(
        ("users".to_owned(), "seen".to_owned()),
        facts(
            Dtype::Time,
            ColumnType::Time64 {
                unit: sdm_contract::Unit::Us,
            },
        ),
    )]);
    let read = held::read(schema(), &found, false).unwrap();
    assert!(read.unfacted.is_empty());
    // The source's data type, which tells a time of day from a date where
    // the wire's name does not.
    assert_eq!(read.description.tables[0].columns[1].dtype, Dtype::Time);
    let written = held::held(&read.description, &read.unfacted);
    assert_eq!(
        written.tables[0].columns[1]
            .source
            .as_ref()
            .map(|f| f.dtype),
        Some(Dtype::Time)
    );
}

#[test]
fn a_column_the_source_gave_nothing_for_is_left_out_where_none_is_read_by_type() {
    let read = held::read(schema(), &BTreeMap::new(), false).unwrap();
    let names: Vec<&str> = read.description.tables[0]
        .columns
        .iter()
        .map(|c| c.name.as_str())
        .collect();
    assert_eq!(names, ["user_id"]);
}

#[test]
fn a_data_type_no_name_gives_is_one_no_meaning_but_its_own_supports() {
    assert_eq!(held::wire_dtype("timestamp[us]"), Dtype::Date);
    assert_eq!(held::wire_dtype("int64"), Dtype::Int);
    assert_eq!(held::wire_dtype("float64"), Dtype::Float);
    assert_eq!(held::wire_dtype("decimal(9, 2)"), Dtype::Unsupported);
    for dtype in [
        Dtype::Bool,
        Dtype::Int,
        Dtype::Int32,
        Dtype::Float,
        Dtype::Float32,
        Dtype::String,
        Dtype::Date,
    ] {
        assert_eq!(
            held::wire_dtype(dtype.wire_name()).wire_name(),
            dtype.wire_name()
        );
    }
}

#[test]
fn a_table_or_column_held_twice_is_the_host_s_fault() {
    let mut twice = schema();
    twice.tables.push(twice.tables[0].clone());
    let refused = held::read(twice, &BTreeMap::new(), true).unwrap_err();
    assert_eq!(refused.code, ErrorCode::InvalidRequest);
    let mut twice = schema();
    let column = twice.tables[0].columns[0].clone();
    twice.tables[0].columns.push(column);
    let refused = held::read(twice, &BTreeMap::new(), true).unwrap_err();
    assert_eq!(refused.code, ErrorCode::InvalidRequest);
}
