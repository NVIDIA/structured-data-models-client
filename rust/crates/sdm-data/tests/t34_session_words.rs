// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A Databricks table with a column named like a session word, read
//! through the table (T3.4 fix 1, round 1's Codex P0): Databricks reads
//! `user`, quoted or not, as the session user once a column of that name
//! is dropped or renamed, so the reference names each column its catalog
//! read through the table (`DatabricksTable._quoted_source_name`), and a
//! missing one fails the statement. The core names a table's columns the
//! catalog read on its relation, each under its own name, marked as the
//! columns found (`Relation.found`, round 2's Claude P2-2); the dialect
//! reads the table through them in every read where one is such a word,
//! and by its name otherwise, as the reference does.

use sdm_contract::data::{DataLayer, DataRequest, FetchPlan, ProviderChoice};
use sdm_data::Data;
use serde_json::{Value, json};

fn databricks() -> ProviderChoice {
    serde_json::from_value(json!({
        "provider": "sql", "dialect": "databricks", "namespace": ["cat", "sch"]
    }))
    .unwrap()
}

/// The table as the core names it once its catalog read `columns`.
fn found(columns: &[&str]) -> Value {
    json!({
        "source": ["logins"],
        "found": true,
        "columns": columns
            .iter()
            .map(|c| json!({"name": c, "column": c}))
            .collect::<Vec<_>>(),
    })
}

fn seeds() -> Value {
    json!({"attachment": 1, "columns": [
        {"name": "key", "type": {"type": "int64"}, "nullable": true},
        {"name": "batch", "type": {"type": "int64"}, "nullable": false}
    ]})
}

/// Every read of rows a table gets, of the table `table`.
fn reads(table: &Value) -> Vec<Value> {
    vec![
        json!({"request": "preview", "table": table, "columns": ["login_id", "USER"], "limit": 1000}),
        json!({"request": "scan", "table": table, "columns": ["USER"], "filters": [], "limit": null}),
        json!({"request": "count_rows", "table": table, "filters": []}),
        json!({"request": "column_ranges", "table": table, "columns": ["at"]}),
        json!({"request": "repeated_key", "table": table, "key": ["login_id"]}),
        json!({"request": "key_counts", "keys": [{"table": table, "columns": ["login_id"]}]}),
        json!({"request": "time_bounds", "columns": [{"table": table, "column": "at"}]}),
        json!({"request": "sample_entities", "table": table, "key": ["login_id"], "n": 10,
               "seed": {"t": "u64", "v": "42"}, "columns": ["login_id", "USER"], "filters": [],
               "key_dtype": "int", "rows": 100}),
        json!({"request": "neighbors", "table": table, "key": ["login_id"],
               "columns": ["login_id", "USER"], "seeds": seeds(), "window": null,
               "per_seed": null, "order": null}),
        json!({"request": "lookup", "table": table, "key": ["login_id"],
               "columns": ["login_id", "USER"], "seeds": seeds()}),
    ]
}

fn texts(request: Value) -> String {
    let request: DataRequest = serde_json::from_value(request).unwrap();
    match Data::new().plan(&request, &databricks()) {
        Ok(FetchPlan::Sql { statements, .. }) => statements
            .iter()
            .map(|s| s.text.clone())
            .collect::<Vec<_>>()
            .join("\n"),
        other => panic!("{} not planned: {other:?}", request.name()),
    }
}

#[test]
fn each_read_of_a_table_with_a_session_word_column_reads_it_through_the_table() {
    let through = "(SELECT source.`login_id`, source.`USER`, source.`at` \
                   FROM `cat`.`sch`.`logins` AS source)";
    for request in reads(&found(&["login_id", "USER", "at"])) {
        let name = request["request"].clone();
        let text = texts(request);
        assert!(text.contains(through), "{name}: {text}");
    }
}

#[test]
fn a_table_with_no_such_column_is_read_by_its_name_as_the_reference_reads_it() {
    for request in reads(&found(&["login_id", "who", "at"])) {
        let name = request["request"].clone();
        let text = texts(request);
        assert!(!text.contains("AS source"), "{name}: {text}");
        assert!(text.contains("`cat`.`sch`.`logins`"), "{name}: {text}");
    }
}

/// A derived relation a host names, its columns each its own source
/// column, is read as those columns: only the columns the catalog found
/// (`found`) are read by the table's name where no column needs them.
#[test]
fn a_host_s_derived_relation_is_read_as_its_columns() {
    let mut table = found(&["login_id", "who", "at"]);
    table.as_object_mut().unwrap().remove("found");
    let text = texts(json!({
        "request": "scan", "table": table, "columns": ["who"], "filters": [], "limit": null
    }));
    assert!(text.contains("AS source)"), "{text}");
}
