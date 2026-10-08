// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `Schema.from_source`'s run over a SQLite source of many tables: it asks
//! each statement once, and its work grows in proportion to the source.

use std::sync::Arc;
use std::time::{Duration, Instant};

use arrow_array::{ArrayRef, Float64Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use sdm_dialects::sqlite;
use sdm_protocol::{Outcome, QueryResult, Run, Step};
use sdm_schema::dt::Host;
use serde_json::{Value, json};

const ROWS: i64 = 200;
const COLUMNS: [(&str, &str); 5] = [
    ("id", "INTEGER"),
    ("amount", "REAL"),
    ("kind", "TEXT"),
    ("at", "TEXT"),
    ("count", "INTEGER"),
];

fn table_name(i: usize) -> String {
    format!("t{i:04}")
}

fn batch(columns: Vec<(&str, ArrayRef)>) -> Outcome {
    let fields: Vec<Field> = columns
        .iter()
        .map(|(n, a)| Field::new(*n, a.data_type().clone(), true))
        .collect();
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(fields)),
        columns.into_iter().map(|(_, a)| a).collect(),
    )
    .expect("a batch");
    Outcome::Data(QueryResult::from_batch(batch))
}

fn texts<'a>(values: impl IntoIterator<Item = &'a str>) -> ArrayRef {
    Arc::new(StringArray::from_iter_values(values))
}

fn ints(values: impl IntoIterator<Item = i64>) -> ArrayRef {
    Arc::new(Int64Array::from_iter_values(values))
}

fn no_rows(names: &[&str]) -> Outcome {
    let fields: Vec<Field> = names
        .iter()
        .map(|n| Field::new(*n, DataType::Int64, true))
        .collect();
    let batch = RecordBatch::new_empty(Arc::new(Schema::new(fields)));
    Outcome::Data(QueryResult::from_batch(batch))
}

/// A SQLite database of `tables` tables of [`COLUMNS`], `id` their primary
/// key, as `adbc_driver_sqlite` reads its catalog and rows.
struct Source {
    tables: usize,
}

impl Source {
    fn answer(&self, sql: &str) -> Outcome {
        let names: Vec<String> = (0..self.tables).map(table_name).collect();
        let quoted = |name: &str| format!("\"{name}\"");
        if sql == sqlite::list_tables().0 {
            return batch(vec![("name", texts(names.iter().map(String::as_str)))]);
        }
        let projections: Vec<String> = COLUMNS.iter().map(|(n, _)| quoted(n)).collect();
        for name in &names {
            let table = quoted(name);
            if sql == sqlite::table_info(&table).0 {
                let n = COLUMNS.len();
                return batch(vec![
                    ("cid", ints(0..n as i64)),
                    ("name", texts(COLUMNS.iter().map(|(n, _)| *n))),
                    ("type", texts(COLUMNS.iter().map(|(_, t)| *t))),
                    ("notnull", ints((0..n).map(|i| i64::from(i == 0)))),
                    ("dflt_value", Arc::new(Int64Array::from(vec![None; n]))),
                    ("pk", ints((0..n).map(|i| i64::from(i == 0)))),
                ]);
            }
            if sql == sqlite::index_list(&table).0 {
                return no_rows(&["seq", "name", "unique", "origin", "partial"]);
            }
            if sql == sqlite::foreign_key_list(&table).0 {
                return no_rows(&[
                    "id",
                    "seq",
                    "table",
                    "from",
                    "to",
                    "on_update",
                    "on_delete",
                    "match",
                ]);
            }
            if sql == sqlite::first_rows(&table, &projections, 1000).0 {
                let at: Vec<String> = (0..ROWS)
                    .map(|i| format!("2024-{:02}-{:02}", i % 12 + 1, i % 28 + 1))
                    .collect();
                return batch(vec![
                    ("id", ints(0..ROWS)),
                    (
                        "amount",
                        Arc::new(Float64Array::from_iter_values(
                            (0..ROWS).map(|i| i as f64 * 0.5),
                        )),
                    ),
                    (
                        "kind",
                        texts((0..ROWS).map(|i| ["a", "b", "c"][(i % 3) as usize])),
                    ),
                    ("at", texts(at.iter().map(String::as_str))),
                    ("count", ints((0..ROWS).map(|i| i % 7))),
                ]);
            }
        }
        panic!("the source has no answer for {sql:?}");
    }
}

/// The run over a source of `tables` tables, driven to its end: its schema,
/// the statements it asked, the steps it took, and how long it took.
fn drive(tables: usize) -> (Value, Vec<String>, usize, Duration) {
    let source = Source { tables };
    let (info, args) = sdm_schema::binding::source_args(
        "sqlite",
        &json!({"namespace": {}, "semantic_model": null, "trust_expressions": false}),
        &json!({"tables": null, "primary_keys": null, "time_columns": null,
                "relationships": null, "check_keys": true}),
    )
    .expect("readable arguments");
    let started = Instant::now();
    let mut run = sdm_schema::run::source_run(info, args, Host::assumed());
    let mut asked = Vec::new();
    let mut steps = 0;
    loop {
        match run.next().expect("a step") {
            Step::Done((schema, _)) => {
                let elapsed = started.elapsed();
                let schema = serde_json::from_str(&schema).expect("JSON");
                return (schema, asked, steps, elapsed);
            }
            Step::CallModel(_) => panic!("a schema run calls no model"),
            Step::RunQueries(queries) => {
                steps += 1;
                for query in queries {
                    let outcome = source.answer(&query.sql);
                    run.submit(query.id, outcome).expect("an asked query");
                    asked.push(query.sql);
                }
            }
        }
    }
}

#[test]
fn reads_every_table_asking_each_statement_once() {
    let (schema, asked, steps, _) = drive(120);
    let tables = schema["tables"].as_object().expect("the schema's tables");
    assert_eq!(tables.len(), 120, "{schema}");
    assert_eq!(tables["t0007"]["primary_key"], "id");
    assert_eq!(tables["t0007"]["time_column"], "at");
    // The listing, then each table's columns, indices, sample and foreign
    // keys: one step each.
    assert_eq!(asked.len(), 1 + 4 * 120);
    assert_eq!(steps, asked.len());
    let mut distinct = asked.clone();
    distinct.sort();
    distinct.dedup();
    assert_eq!(distinct.len(), asked.len(), "a statement was asked twice");
}

/// The run's work grows with the number of tables, not its square: ten
/// times the tables take far less than a hundred times as long.
#[test]
fn work_grows_in_proportion_to_the_source() {
    let fastest = |tables| (0..3).map(|_| drive(tables).3).min().expect("three runs");
    let small = fastest(30);
    let large = fastest(300);
    let ratio = large.as_secs_f64() / small.as_secs_f64();
    assert!(
        ratio < 30.0,
        "300 tables took {large:?}, {ratio:.1} times the {small:?} of 30 tables"
    );
}
