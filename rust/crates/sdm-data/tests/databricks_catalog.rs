// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Databricks catalog interpretation from small authored information-schema rows.
use arrow_array::{ArrayRef, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use sdm_contract::catalog::{CatalogAnswer, KeyKind};
use sdm_contract::data::{DataLayer, DataRequest, DescribePart, ProviderChoice};
use sdm_contract::ops::Dtype;
use sdm_data::Data;
use std::sync::Arc;

fn batch(names: &[&str], rows: &[Vec<String>]) -> RecordBatch {
    let fields: Vec<_> = names
        .iter()
        .map(|name| Field::new(*name, DataType::Utf8, true))
        .collect();
    let arrays: Vec<ArrayRef> = (0..names.len())
        .map(|i| {
            Arc::new(StringArray::from(
                rows.iter().map(|r| r[i].as_str()).collect::<Vec<_>>(),
            )) as ArrayRef
        })
        .collect();
    RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays).unwrap()
}
fn rows(values: &[&[&str]]) -> Vec<Vec<String>> {
    values
        .iter()
        .map(|row| row.iter().map(|v| (*v).to_owned()).collect())
        .collect()
}
fn read(part: DescribePart, input: RecordBatch) -> Result<CatalogAnswer, sdm_contract::CoreError> {
    let request = DataRequest::Describe {
        tables: vec![
            serde_json::from_value(serde_json::json!({"source":["main","shop","orders"]})).unwrap(),
        ],
        part,
    };
    Data::new().read(&request, &ProviderChoice::sql("databricks"), &[input])
}
const COLUMNS: &[&str] = &[
    "column_name",
    "full_data_type",
    "is_nullable",
    "table_catalog",
    "table_schema",
    "table_name",
    "table_type",
];

#[test]
fn columns_preserve_declarations_order_nullability_and_type_meaning() {
    let cases = [
        ("tinyint", Dtype::Int),
        ("smallint", Dtype::Int),
        ("int", Dtype::Int),
        ("bigint", Dtype::Int),
        ("float", Dtype::Float),
        ("double", Dtype::Float),
        ("decimal(38,0)", Dtype::Int),
        ("decimal(38,6)", Dtype::Float),
        ("decimal(10,2)", Dtype::Float),
        ("string", Dtype::String),
        ("binary", Dtype::Binary),
        ("boolean", Dtype::Bool),
        ("date", Dtype::Date),
        ("timestamp", Dtype::Date),
        ("timestamp_ntz", Dtype::Date),
        ("array<int>", Dtype::Intlist),
        ("array<bigint>", Dtype::Intlist),
        ("array<float>", Dtype::Floatlist),
        ("array<double>", Dtype::Floatlist),
        ("array<string>", Dtype::Stringlist),
        ("map<string,int>", Dtype::Unsupported),
        ("struct<a:int>", Dtype::Unsupported),
        ("struct<age:int,country:string>", Dtype::Unsupported),
    ];
    let input: Vec<_> = cases
        .iter()
        .enumerate()
        .map(|(i, (declared, _))| {
            vec![
                format!("field {i}"),
                (*declared).into(),
                if i % 2 == 0 { "NO" } else { "YES" }.into(),
                "main".into(),
                "shop".into(),
                "orders".into(),
                "MANAGED".into(),
            ]
        })
        .collect();
    let CatalogAnswer::Columns(columns) =
        read(DescribePart::Columns, batch(COLUMNS, &input)).unwrap()
    else {
        panic!("columns")
    };
    assert_eq!(columns.len(), cases.len());
    for (i, (column, (declared, dtype))) in columns.iter().zip(cases).enumerate() {
        assert_eq!(column.name, format!("field {i}"));
        assert_eq!(column.declared_type, declared);
        assert_eq!(column.not_null, i % 2 == 0);
        assert_eq!(column.dtype, Some(dtype));
    }
}

#[test]
fn informational_single_column_keys_are_unenforced_and_composites_are_omitted() {
    let names = ["constraint_type", "constraint_name", "column_name"];
    let CatalogAnswer::Keys(empty) = read(DescribePart::Keys, batch(&names, &[])).unwrap() else {
        panic!("keys")
    };
    assert!(empty.is_empty());
    let input = rows(&[
        &["PRIMARY KEY", "pk", "id"],
        &["UNIQUE", "uq", "email"],
        &["UNIQUE", "composite", "country"],
        &["UNIQUE", "composite", "external_id"],
    ]);
    let CatalogAnswer::Keys(keys) = read(DescribePart::Keys, batch(&names, &input)).unwrap() else {
        panic!("keys")
    };
    assert_eq!(keys.len(), 2);
    assert!(keys.iter().all(|key| !key.enforced));
    assert!(
        keys.iter()
            .any(|key| key.kind == KeyKind::Primary && key.columns == ["id"])
    );
    assert!(
        keys.iter()
            .any(|key| key.kind == KeyKind::Unique && key.columns == ["email"])
    );
    // Composite constraints do not prove that either column alone is unique.
    assert!(keys.iter().all(|key| key.columns.len() == 1));
}

#[test]
fn foreign_keys_keep_distinct_roles_and_qualified_target() {
    let names = [
        "constraint_name",
        "source_column",
        "table_catalog",
        "table_schema",
        "table_name",
        "referenced_column",
    ];
    let CatalogAnswer::ForeignKeys(empty) =
        read(DescribePart::ForeignKeys, batch(&names, &[])).unwrap()
    else {
        panic!("foreign keys")
    };
    assert!(empty.is_empty());
    let input = rows(&[&["buyer_fk", "buyer", "other", "sales", "users", "user_id"]]);
    let CatalogAnswer::ForeignKeys(keys) =
        read(DescribePart::ForeignKeys, batch(&names, &input)).unwrap()
    else {
        panic!("foreign keys")
    };
    assert_eq!(keys.len(), 1);
    assert_eq!(keys[0].columns, ["buyer"]);
    assert_eq!(keys[0].referenced_columns, ["user_id"]);
    assert_eq!(keys[0].referenced_table, "other.sales.users");
}

#[test]
fn table_listing_preserves_names_and_order() {
    let CatalogAnswer::Tables(tables) = read(
        DescribePart::Tables,
        batch(&["table_name"], &rows(&[&["orders"], &["Odd.Table"]])),
    )
    .unwrap() else {
        panic!("tables")
    };
    assert_eq!(tables, ["orders", "Odd.Table"]);
}

#[test]
fn metric_view_is_refused_instead_of_read_as_physical_table() {
    let input = rows(&[&["id", "bigint", "NO", "main", "shop", "mv", "METRIC_VIEW"]]);
    let error = read(DescribePart::Columns, batch(COLUMNS, &input)).unwrap_err();
    assert!(error.message.to_lowercase().contains("metric view"));
}
