// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The closed type set: every type in it comes in from Arrow and goes back
//! out unchanged, and a type outside it is refused by name.

#![allow(
    clippy::disallowed_methods,
    clippy::disallowed_types,
    reason = "test code: reads its vector file, and Arrow's field metadata is a HashMap"
)]

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use arrow_schema::{
    DataType, Field as ArrowField, Fields, IntervalUnit, TimeUnit, UnionFields, UnionMode,
};
use sdm_contract::types::cell_types;
use sdm_contract::{ColumnType, ErrorCode, ErrorKind, Field};

fn field(name: &str, data_type: DataType) -> ArrowField {
    ArrowField::new(name, data_type, true)
}

fn accepted() -> Vec<DataType> {
    let entries = Arc::new(ArrowField::new(
        "key_value",
        DataType::Struct(Fields::from(vec![
            ArrowField::new("key", DataType::Utf8, false),
            field("value", DataType::Int64),
        ])),
        false,
    ));
    vec![
        DataType::Null,
        DataType::Boolean,
        DataType::Int8,
        DataType::Int16,
        DataType::Int32,
        DataType::Int64,
        DataType::UInt8,
        DataType::UInt16,
        DataType::UInt32,
        DataType::UInt64,
        DataType::Float32,
        DataType::Float64,
        DataType::Decimal128(38, 0),
        DataType::Decimal128(38, 10),
        DataType::Decimal256(76, 38),
        DataType::Utf8,
        DataType::Binary,
        DataType::Date32,
        DataType::Timestamp(TimeUnit::Nanosecond, None),
        DataType::Timestamp(TimeUnit::Microsecond, Some("Europe/Berlin".into())),
        DataType::Duration(TimeUnit::Nanosecond),
        DataType::Duration(TimeUnit::Microsecond),
        DataType::Time64(TimeUnit::Microsecond),
        DataType::List(Arc::new(field("item", DataType::Utf8))),
        DataType::List(Arc::new(field(
            "element",
            DataType::List(Arc::new(field("item", DataType::Int32))),
        ))),
        DataType::Struct(Fields::from(vec![
            field("a", DataType::Int64),
            field(
                "b",
                DataType::List(Arc::new(field("item", DataType::Float64))),
            ),
        ])),
        DataType::Map(entries.clone(), false),
        DataType::Map(entries, true),
        DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
        DataType::Dictionary(Box::new(DataType::UInt8), Box::new(DataType::Utf8)),
    ]
}

fn refused() -> Vec<DataType> {
    vec![
        DataType::Float16,
        DataType::Decimal32(9, 2),
        DataType::Decimal64(18, 2),
        DataType::LargeUtf8,
        DataType::Utf8View,
        DataType::LargeBinary,
        DataType::BinaryView,
        DataType::FixedSizeBinary(16),
        DataType::Date64,
        DataType::Timestamp(TimeUnit::Second, None),
        DataType::Timestamp(TimeUnit::Millisecond, Some("UTC".into())),
        DataType::Time32(TimeUnit::Second),
        DataType::Time64(TimeUnit::Second),
        DataType::Duration(TimeUnit::Millisecond),
        DataType::Interval(IntervalUnit::MonthDayNano),
        DataType::LargeList(Arc::new(field("item", DataType::Int64))),
        DataType::FixedSizeList(Arc::new(field("item", DataType::Int64)), 3),
        DataType::ListView(Arc::new(field("item", DataType::Int64))),
        DataType::Union(
            UnionFields::try_new(vec![0], vec![field("a", DataType::Int64)]).unwrap(),
            UnionMode::Dense,
        ),
        DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::LargeUtf8)),
        DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Int64)),
        DataType::RunEndEncoded(
            Arc::new(ArrowField::new("run_ends", DataType::Int32, false)),
            Arc::new(field("values", DataType::Utf8)),
        ),
    ]
}

#[test]
fn every_type_in_the_set_comes_in_and_goes_out_unchanged() {
    for data_type in accepted() {
        let column_type = ColumnType::from_arrow("c", &data_type).unwrap();
        assert_eq!(column_type.to_arrow(), data_type);
        let text = serde_json::to_string(&column_type).unwrap();
        let back: ColumnType = serde_json::from_str(&text).unwrap();
        assert_eq!(back, column_type, "{text}");
    }
}

#[test]
fn field_names_nullability_and_metadata_are_kept() {
    let arrow = ArrowField::new("price", DataType::Decimal128(18, 2), false)
        .with_metadata(HashMap::from([("unit".to_owned(), "EUR".to_owned())]));
    let field = Field::from_arrow(&arrow).unwrap();
    assert_eq!(field.to_arrow(), arrow);
    let back: Field = serde_json::from_str(&serde_json::to_string(&field).unwrap()).unwrap();
    assert_eq!(back, field);
}

#[test]
fn a_type_outside_the_set_is_refused_by_name() {
    for data_type in refused() {
        let err = ColumnType::from_arrow("c", &data_type).unwrap_err();
        assert_eq!(err.kind, ErrorKind::Unsupported);
        assert_eq!(err.code, ErrorCode::UnsupportedType);
        let fields = serde_json::to_value(&err.fields).unwrap();
        assert_eq!(fields["column"]["v"], "c");
        assert_eq!(fields["arrow_type"]["v"], data_type.to_string());
    }
}

#[test]
fn a_nested_refusal_names_the_path() {
    let data_type = DataType::Struct(Fields::from(vec![field(
        "tags",
        DataType::List(Arc::new(field("item", DataType::LargeUtf8))),
    )]));
    let err = ColumnType::from_arrow("attrs", &data_type).unwrap_err();
    let fields = serde_json::to_value(&err.fields).unwrap();
    assert_eq!(fields["column"]["v"], "attrs.tags.item");
    assert_eq!(fields["arrow_type"]["v"], DataType::LargeUtf8.to_string());
}

#[test]
fn a_field_refuses_fields_its_type_does_not_have() {
    let good = r#"{"name":"a","type":{"type":"int64"},"nullable":true}"#;
    serde_json::from_str::<Field>(good).unwrap();
    for bad in [
        r#"{"name":"a","type":{"type":"int64","unit":"us"},"nullable":true}"#,
        r#"{"name":"a","type":{"type":"duration","unit":"us","zone":null},"nullable":true}"#,
        r#"{"name":"a","type":{"type":"large_utf8"},"nullable":true}"#,
        r#"{"name":"a","type":{"type":"int64"},"nullable":true,"dictionary":false}"#,
    ] {
        assert!(
            serde_json::from_str::<Field>(bad).is_err(),
            "{bad} was accepted"
        );
    }
}

#[test]
fn type_parameters_outside_the_limits_are_refused() {
    let good = [
        r#"{"type":"decimal128","precision":1,"scale":0}"#,
        r#"{"type":"decimal128","precision":38,"scale":38}"#,
        r#"{"type":"decimal128","precision":10,"scale":-3}"#,
        r#"{"type":"decimal256","precision":76,"scale":0}"#,
        r#"{"type":"timestamp","unit":"us","zone":"UTC"}"#,
        r#"{"type":"timestamp","unit":"us","zone":null}"#,
    ];
    let bad = [
        r#"{"type":"decimal128","precision":0,"scale":0}"#,
        r#"{"type":"decimal128","precision":39,"scale":0}"#,
        r#"{"type":"decimal128","precision":5,"scale":6}"#,
        r#"{"type":"decimal256","precision":0,"scale":0}"#,
        r#"{"type":"decimal256","precision":77,"scale":0}"#,
        r#"{"type":"decimal256","precision":3,"scale":4}"#,
        r#"{"type":"timestamp","unit":"us","zone":""}"#,
        // Nested: a list item and a struct child.
        r#"{"type":"list","item":{"name":"item","type":{"type":"decimal128","precision":0,"scale":0},"nullable":true}}"#,
        r#"{"type":"struct","fields":[{"name":"t","type":{"type":"timestamp","unit":"ns","zone":""},"nullable":true}]}"#,
    ];
    let field = |t: &str| format!(r#"{{"name":"a","type":{t},"nullable":true}}"#);
    for t in good {
        let read: Field = serde_json::from_str(&field(t)).unwrap();
        read.column_type.check().unwrap();
    }
    for t in bad {
        assert!(
            serde_json::from_str::<Field>(&field(t)).is_err(),
            "{t} was accepted"
        );
    }
}

#[test]
fn a_type_outside_the_limits_is_not_written() {
    let bad = [
        ColumnType::Decimal128 {
            precision: 0,
            scale: 0,
        },
        ColumnType::Decimal128 {
            precision: 5,
            scale: 6,
        },
        ColumnType::Decimal256 {
            precision: 77,
            scale: 0,
        },
        ColumnType::Timestamp {
            unit: sdm_contract::Unit::Us,
            zone: Some(String::new()),
        },
    ];
    for column_type in bad {
        assert!(column_type.check().is_err());
        let top = Field {
            name: "a".into(),
            column_type: column_type.clone(),
            nullable: true,
            metadata: Default::default(),
        };
        assert!(serde_json::to_string(&top).is_err(), "{column_type:?}");
        let nested = Field {
            name: "l".into(),
            column_type: ColumnType::List {
                item: Box::new(top),
            },
            nullable: true,
            metadata: Default::default(),
        };
        assert!(nested.column_type.check().is_err());
        assert!(
            serde_json::to_string(&nested).is_err(),
            "nested {column_type:?}"
        );
    }
}

#[test]
fn arrow_decimals_outside_the_limits_are_refused_and_an_empty_zone_is_none() {
    for data_type in [
        DataType::Decimal128(0, 0),
        DataType::Decimal128(39, 0),
        DataType::Decimal128(5, 6),
        DataType::Decimal256(77, 0),
    ] {
        let err = ColumnType::from_arrow("d", &data_type).unwrap_err();
        assert_eq!(err.code, ErrorCode::UnsupportedType, "{data_type}");
    }
    let empty = DataType::Timestamp(TimeUnit::Microsecond, Some("".into()));
    assert_eq!(
        ColumnType::from_arrow("t", &empty).unwrap(),
        ColumnType::Timestamp {
            unit: sdm_contract::Unit::Us,
            zone: None
        }
    );
}

#[test]
fn the_cell_vocabulary_is_the_scope_manifests() {
    // scope.schema.json's cell id pattern lists the types in this order.
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../docs/design/rust-core/phase-0/scope.schema.json");
    let schema: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let pattern = schema["properties"]["cells"]["items"]["properties"]["id"]["pattern"]
        .as_str()
        .unwrap();
    let types = pattern
        .strip_prefix("^(")
        .and_then(|p| p.split_once(")/"))
        .unwrap()
        .0;
    let from_manifest: Vec<&str> = types.split('|').collect();
    assert_eq!(cell_types(), from_manifest);
}
