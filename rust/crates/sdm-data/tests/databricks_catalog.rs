// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The Databricks catalog reader over the warehouse's recorded catalog rows
//! (T3.4, the brief's exit 4): each catalog answer the old client read in
//! the capture, read by the `databricks` dialect into the contract's
//! `CatalogAnswer`, equal to how the old client read the same rows.
//!
//! The rows are `contract/fixtures/databricks/capture.json`'s, shaped as the
//! design's `sdm-capture/1` and recorded by `record.py` beside it from the
//! reference (43e55f4f) on a stand-in connection, until C-1's capture of
//! the same cases replaces them; each catalog statement's entry carries its
//! rows (`result.rows`, every column a string, as Unity Catalog's
//! `information_schema` gives them) and the old client's reading of them
//! (`reading`): `DatabricksTable._get_source_columns` (DB/table.py:212-316)
//! for a table's columns, its declared type and nullability, and the data
//! type `_to_dtype` (DB/table.py:442-497) fixes from the declaration, which
//! the old client types a column by before its values; its primary and
//! unique keys; `_get_source_foreign_keys` (DB/table.py:318-370) for its
//! foreign keys; and `Graph.from_databricks`'s listing (RE/graph.py:1199).

use std::sync::Arc;

use arrow_array::{ArrayRef, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use sdm_contract::catalog::{CatalogAnswer, KeyKind};
use sdm_contract::data::{DataLayer, DataRequest, DescribePart, ProviderChoice, Relation};
use sdm_contract::ops::Dtype;
use sdm_data::Data;
use serde_json::Value;

/// The capture, read at build time: the core's crates do no I/O, their
/// tests included.
const CAPTURE: &str = include_str!("../../../../contract/fixtures/databricks/capture.json");

/// The `databricks` dialect's choice, built from the contract's JSON so the
/// test reads whatever optional fields the choice gains.
fn databricks() -> ProviderChoice {
    serde_json::from_str(r#"{"provider":"sql","dialect":"databricks"}"#).unwrap()
}

/// Every catalog statement of the capture the old client read, with the
/// case it was recorded for.
fn read_statements() -> Vec<(String, Value)> {
    let capture: Value = serde_json::from_str(CAPTURE).unwrap();
    assert_eq!(capture["format"], "sdm-capture/1");
    let mut found = Vec::new();
    for case in capture["cases"].as_array().unwrap() {
        for statement in case["statements"].as_array().unwrap() {
            if statement.get("reading").is_some() {
                found.push((case["case"].as_str().unwrap().to_owned(), statement.clone()));
            }
        }
    }
    assert!(
        found.len() >= 30,
        "the capture holds {} catalog answers",
        found.len()
    );
    found
}

/// The recorded rows of a statement as the batch the driver fetched: each
/// column a string, named as the statement's `SELECT` names it.
fn batch(result: &Value) -> RecordBatch {
    let schema = result["schema"].as_array().unwrap();
    let rows = result["rows"].as_array().unwrap();
    let mut fields = Vec::new();
    let mut columns: Vec<ArrayRef> = Vec::new();
    for (at, field) in schema.iter().enumerate() {
        assert_eq!(field["type"], "string", "a catalog column not of strings");
        fields.push(Field::new(
            field["name"].as_str().unwrap(),
            DataType::Utf8,
            true,
        ));
        let values: Vec<Option<&str>> = rows.iter().map(|row| row[at].as_str()).collect();
        columns.push(Arc::new(StringArray::from(values)));
    }
    RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap()
}

/// The describe request the statement answered: one table, by catalog,
/// schema and the name the caller gave (its parameters), or none.
fn describe(part: DescribePart, statement: &Value) -> DataRequest {
    let parameters: Vec<&str> = statement["parameters"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| p.as_str().unwrap())
        .collect();
    let tables = match parameters.as_slice() {
        [_schema] => vec![],
        [schema, table] => {
            // From the contract's JSON, so the relation reads whatever
            // optional fields it gains (v28's columns and filter).
            let relation: Relation =
                serde_json::from_value(serde_json::json!({"source": ["main", schema, table]}))
                    .unwrap();
            vec![relation]
        }
        other => panic!("unexpected parameters {other:?}"),
    };
    DataRequest::Describe { tables, part }
}

/// The contract's data type for the old client's `Dtype` value.
fn dtype(value: &Value) -> Option<Dtype> {
    let name = value.as_str()?;
    Some(match name {
        "bool" => Dtype::Bool,
        "int" => Dtype::Int,
        "float" => Dtype::Float,
        "string" => Dtype::String,
        "binary" => Dtype::Binary,
        "date" => Dtype::Date,
        "intlist" => Dtype::Intlist,
        "floatlist" => Dtype::Floatlist,
        "stringlist" => Dtype::Stringlist,
        "unsupported" => Dtype::Unsupported,
        other => panic!("a data type the old client read and this test does not map: {other}"),
    })
}

fn read(request: &DataRequest, statement: &Value) -> CatalogAnswer {
    let batches = [batch(&statement["result"])];
    Data::new()
        .read(request, &databricks(), &batches)
        .unwrap_or_else(|error| panic!("{request:?}: refused: {error:?}"))
}

/// Each column's name, declared type, nullability and data type, as the old
/// client read them, in the catalog's order: a timestamp of either kind a
/// date, `DECIMAL(p,0)` an integer and any other scale a float, an array of
/// its element's list type, a map or struct unsupported (so the old client
/// refuses it by name).
#[test]
fn columns_read_as_the_old_client_reads_them() {
    let mut read_some = 0;
    for (case, statement) in read_statements() {
        let reading = &statement["reading"];
        if reading["part"] != "columns" {
            continue;
        }
        let request = describe(DescribePart::Columns, &statement);
        let CatalogAnswer::Columns(columns) = read(&request, &statement) else {
            panic!("{case}: not a columns answer");
        };
        let got: Vec<(String, String, bool, Option<Dtype>)> = columns
            .into_iter()
            .map(|c| (c.name, c.declared_type, c.not_null, c.dtype))
            .collect();
        let wanted: Vec<(String, String, bool, Option<Dtype>)> = reading["columns"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| {
                (
                    c["name"].as_str().unwrap().to_owned(),
                    c["declared_type"].as_str().unwrap().to_owned(),
                    c["not_null"].as_bool().unwrap(),
                    dtype(&c["dtype"]),
                )
            })
            .collect();
        assert_eq!(got, wanted, "{case}: {request:?}");
        read_some += 1;
    }
    assert!(read_some >= 15);
}

/// Each declared primary and unique key of one column, as the old client
/// keeps it (DB/table.py:271-300); none where the catalog declares none.
#[test]
fn keys_read_as_the_old_client_reads_them() {
    let mut declared = 0;
    for (case, statement) in read_statements() {
        let reading = &statement["reading"];
        if reading["part"] != "keys" {
            continue;
        }
        let request = describe(DescribePart::Keys, &statement);
        let CatalogAnswer::Keys(keys) = read(&request, &statement) else {
            panic!("{case}: not a keys answer");
        };
        let mut got: Vec<(String, Vec<String>)> = keys
            .into_iter()
            .map(|k| {
                let kind = match k.kind {
                    KeyKind::Primary => "PRIMARY KEY",
                    KeyKind::Unique => "UNIQUE",
                    other => panic!("{case}: a key read as {other:?}"),
                };
                (kind.to_owned(), k.columns)
            })
            .collect();
        let mut wanted: Vec<(String, Vec<String>)> = reading["keys"]
            .as_array()
            .unwrap()
            .iter()
            .map(|k| {
                (
                    k["kind"].as_str().unwrap().to_owned(),
                    serde_json::from_value(k["columns"].clone()).unwrap(),
                )
            })
            .collect();
        got.sort();
        wanted.sort();
        assert_eq!(got, wanted, "{case}: {request:?}");
        declared += wanted.len();
    }
    assert!(declared >= 2, "the capture declares {declared} keys");
}

/// Each declared foreign key of one column, its referenced table and
/// column, as the old client keeps it (DB/table.py:318-370): the table by
/// its name, or by catalog, schema and name as the old client writes it.
#[test]
fn foreign_keys_read_as_the_old_client_reads_them() {
    let mut declared = 0;
    for (case, statement) in read_statements() {
        let reading = &statement["reading"];
        if reading["part"] != "foreign_keys" {
            continue;
        }
        let request = describe(DescribePart::ForeignKeys, &statement);
        let CatalogAnswer::ForeignKeys(keys) = read(&request, &statement) else {
            panic!("{case}: not a foreign keys answer");
        };
        let wanted = reading["foreign_keys"].as_array().unwrap();
        assert_eq!(keys.len(), wanted.len(), "{case}: {keys:?}");
        for want in wanted {
            let table = want["referenced_table"].as_str().unwrap();
            let columns: Vec<String> = serde_json::from_value(want["columns"].clone()).unwrap();
            let referenced: Vec<String> =
                serde_json::from_value(want["referenced_columns"].clone()).unwrap();
            assert!(
                keys.iter().any(|k| {
                    k.columns == columns
                        && k.referenced_columns == referenced
                        && (k.referenced_table == table
                            || k.referenced_table == format!("main.shop.{table}"))
                }),
                "{case}: {want} not read: {keys:?}"
            );
        }
        declared += wanted.len();
    }
    assert!(declared >= 1, "the capture declares no foreign key");
}

/// The schema's tables, as `Graph.from_databricks` lists them with none
/// named (RE/graph.py:1199), in the catalog's order.
#[test]
fn the_schema_s_tables_read_as_the_old_client_lists_them() {
    let mut listed = 0;
    for (case, statement) in read_statements() {
        let reading = &statement["reading"];
        if reading["part"] != "tables" {
            continue;
        }
        let request = describe(DescribePart::Tables, &statement);
        let CatalogAnswer::Tables(tables) = read(&request, &statement) else {
            panic!("{case}: not a tables answer");
        };
        let wanted: Vec<String> = serde_json::from_value(reading["tables"].clone()).unwrap();
        assert_eq!(tables, wanted, "{case}");
        listed += 1;
    }
    assert!(listed >= 5);
}

/// A metric view named as a table is refused by name, as the old client
/// refuses it before reading a row (DB/table.py:254, `MetricViewAsTableError`):
/// a recorded columns answer with its relation's type made `METRIC_VIEW`.
#[test]
fn a_metric_view_named_as_a_table_is_refused_by_name() {
    let (_, statement) = read_statements()
        .into_iter()
        .find(|(case, s)| case == "base.binary" && s["reading"]["part"] == "columns")
        .unwrap();
    let mut seeded = statement.clone();
    for row in seeded["result"]["rows"].as_array_mut().unwrap() {
        let row = row.as_array_mut().unwrap();
        assert_eq!(row[6], "MANAGED");
        row[6] = Value::from("METRIC_VIEW");
    }
    let request = describe(DescribePart::Columns, &seeded);
    let refused = Data::new().read(&request, &databricks(), &[batch(&seeded["result"])]);
    let error = refused.expect_err("a metric view read as a table");
    let said = format!("{error:?}").to_lowercase();
    assert!(
        said.contains("metric_view") || said.contains("metric view"),
        "not refused as a metric view: {said}"
    );
    // The control: the same answer as recorded is read.
    read(&request, &statement);
}
