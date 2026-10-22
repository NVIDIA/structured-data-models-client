// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The stand-in's own tests (T3.3; design, choice 2), in process, over the
//! committed version-1 capture (`contract/tests/t33-capture-v1`: the shop
//! base's tables, a draw of `users` with a seed past 2^31, and a statement
//! with its recorded result): exact replay serves the recorded rows, a
//! request the old client never sent is answered on the local engine, the
//! recorded draw is served only for its own table, seed and fraction text,
//! a missing table is refused by name, and a catalog read is live only
//! (B1). The acceptance tests, through `sdm-standin` and the Python client,
//! are the test author's.

use std::collections::BTreeMap;
use std::path::PathBuf;

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use sdm_contract::run::FetchStep;
use sdm_testkit::standin::{Answer, B1, Capture, StandIn, normalised};
use serde_json::{Value, json};

fn capture() -> Capture {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../contract/tests/t33-capture-v1");
    Capture::read(&path).unwrap()
}

fn sql(text: &str, params: Value) -> Value {
    json!({
        "plan": "sql",
        "statements": [{"dialect": "databricks", "text": text, "params": params}],
        "expect": {"expect": "discover"},
    })
}

fn step(request: Value, plan: Value) -> FetchStep {
    serde_json::from_value(json!({"id": 1, "request": request, "plan": plan})).unwrap()
}

fn keys(answer: &Answer) -> Vec<i64> {
    let Answer::Data(fetched) = answer else {
        panic!("not rows");
    };
    let mut found: Vec<i64> = fetched
        .batches
        .iter()
        .flat_map(|batch| {
            let at = batch.schema().index_of("user_id").unwrap();
            let column = batch.column(at).as_primitive::<Int64Type>().clone();
            (0..column.len())
                .map(move |i| column.value(i))
                .collect::<Vec<_>>()
        })
        .collect();
    found.sort_unstable();
    found
}

const USERS: &str = "`{catalog}`.`{schema}`.`users`";

#[test]
fn a_recorded_statement_is_replayed_and_another_answered_locally() {
    let stand_in = StandIn::open(capture()).unwrap();
    let preview = |limit: u64| {
        step(
            json!({"request": "preview", "table": {"source": ["users"]},
                   "columns": ["user_id"], "limit": limit}),
            sql(
                &format!("SELECT `user_id`  FROM\n {USERS}\tLIMIT ?"),
                json!([{"param": "value", "value": {"t": "i64", "v": limit.to_string()}}]),
            ),
        )
    };
    let replayed = stand_in.fetch(&preview(3), &BTreeMap::new()).unwrap();
    assert_eq!(replayed.by, "replay");
    assert_eq!(keys(&replayed.answer), [1001, 1002, 1003]);
    let local = stand_in.fetch(&preview(2), &BTreeMap::new()).unwrap();
    assert_eq!(local.by, "local");
    assert_eq!(keys(&local.answer).len(), 2);
}

#[test]
fn a_draw_is_served_only_as_recorded() {
    let stand_in = StandIn::open(capture()).unwrap();
    let sample = |seed: u64, fraction: &str| {
        step(
            json!({"request": "sample_entities", "table": {"source": ["users"]},
                   "key": ["user_id"], "n": 2, "seed": {"t": "u64", "v": seed.to_string()},
                   "columns": ["user_id"], "filters": []}),
            sql(
                &format!(
                    "SELECT `user_id` FROM {USERS} TABLESAMPLE ({fraction} PERCENT) REPEATABLE ({seed}) LIMIT 2"
                ),
                json!([]),
            ),
        )
    };
    let drawn = stand_in
        .fetch(&sample(2_147_483_655, "12.5"), &BTreeMap::new())
        .unwrap();
    assert_eq!(drawn.by, "draw");
    assert_eq!(keys(&drawn.answer), [2, 5]);
    for (seed, fraction, named) in [
        (2_147_483_656, "12.5", "2147483656"),
        (2_147_483_655, "12.50", "12.50"),
    ] {
        let Err(refused) = stand_in.fetch(&sample(seed, fraction), &BTreeMap::new()) else {
            panic!("a draw asked otherwise was served");
        };
        assert!(refused.to_string().contains(named), "{refused}");
    }
}

#[test]
fn a_missing_table_is_refused_and_a_catalog_read_is_live_only() {
    let stand_in = StandIn::open(capture()).unwrap();
    let missing = step(
        json!({"request": "scan", "table": {"source": ["payments"]}, "columns": ["id"],
               "filters": [], "limit": null}),
        sql(
            "SELECT `id` FROM `{catalog}`.`{schema}`.`payments`",
            json!([]),
        ),
    );
    let refused = stand_in.fetch(&missing, &BTreeMap::new()).err().unwrap();
    assert!(refused.to_string().contains("payments"), "{refused}");
    let describe = step(
        json!({"request": "describe", "tables": [{"source": ["users"]}], "part": "columns"}),
        sql(
            "DESCRIBE TABLE EXTENDED `{catalog}`.`{schema}`.`users`",
            json!([]),
        ),
    );
    let answered = stand_in.fetch(&describe, &BTreeMap::new()).unwrap();
    assert!(matches!(answered.answer, Answer::LiveOnly(code) if code == B1));
}

#[test]
fn whitespace_outside_quotes_is_normalised() {
    assert_eq!(
        normalised("SELECT   `a  b`  FROM\n\n `{catalog}`.`t`\tLIMIT  ?"),
        "SELECT `a  b` FROM `{catalog}`.`t` LIMIT ?"
    );
}

/// The version-1 capture copied, its `capture.json` changed by `edit`.
fn edited(edit: impl FnOnce(&mut Value, &std::path::Path)) -> Capture {
    let source =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../contract/tests/t33-capture-v1");
    let target = std::env::temp_dir().join(format!(
        "sdm-standin-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(target.join("tables")).unwrap();
    for entry in std::fs::read_dir(source.join("tables")).unwrap() {
        let entry = entry.unwrap();
        std::fs::copy(entry.path(), target.join("tables").join(entry.file_name())).unwrap();
    }
    let mut document: Value =
        serde_json::from_str(&std::fs::read_to_string(source.join("capture.json")).unwrap())
            .unwrap();
    edit(&mut document, &target);
    std::fs::write(
        target.join("capture.json"),
        serde_json::to_vec(&document).unwrap(),
    )
    .unwrap();
    Capture::read(&target).unwrap()
}

/// `fetched` written as a capture table under `directory`, its name.
fn table_file(directory: &std::path::Path, fetched: &sdm_testkit::duck::Fetched) -> String {
    let bytes = sdm_testkit::standin::encode(fetched).unwrap();
    let name = format!("tables/{}.arrows", sdm_testkit::case::sha256(&bytes));
    std::fs::write(directory.join(&name), bytes).unwrap();
    name
}

#[test]
fn a_replayable_statement_over_a_dropped_table_is_refused_by_name() {
    // Review r2 P1: the request's tables are checked before replay.
    let capture = edited(|document, _| {
        let tables = document["tables"].as_array_mut().unwrap();
        tables.retain(|table| table["name"] != "users");
    });
    let stand_in = StandIn::open(capture).unwrap();
    let preview = step(
        json!({"request": "preview", "table": {"source": ["users"]},
               "columns": ["user_id"], "limit": 3}),
        sql(
            &format!("SELECT `user_id`\nFROM {USERS}\nLIMIT ?"),
            json!([{"param": "value", "value": {"t": "i64", "v": "3"}}]),
        ),
    );
    let refused = stand_in.fetch(&preview, &BTreeMap::new()).err().unwrap();
    assert!(refused.to_string().contains("users"), "{refused}");
}

#[test]
fn an_empty_recorded_table_and_an_unseeded_draw_are_served() {
    // Review r2 P2: a table of no rows loads; Snowflake's unseeded
    // `SAMPLE ROW (n ROWS)` draw is served as recorded.
    use duckdb::arrow::array::{ArrayRef, Int64Array};
    use duckdb::arrow::datatypes::{DataType, Field, Schema};
    use duckdb::arrow::record_batch::RecordBatch;
    use std::sync::Arc;

    let schema = Arc::new(Schema::new(vec![Field::new(
        "order_id",
        DataType::Int64,
        true,
    )]));
    let empty = sdm_testkit::duck::Fetched {
        schema: schema.clone(),
        batches: vec![
            RecordBatch::try_new(
                schema,
                vec![Arc::new(Int64Array::from(Vec::<i64>::new())) as ArrayRef],
            )
            .unwrap(),
        ],
    };
    let capture = edited(|document, directory| {
        let name = table_file(directory, &empty);
        for table in document["tables"].as_array_mut().unwrap() {
            if table["name"] == "orders" {
                table["data"] = json!(name);
            }
        }
        let draw = &mut document["draws"][0];
        draw["seed"] = Value::Null;
        draw["fraction"] = json!("2 ROWS");
    });
    let stand_in = StandIn::open(capture).unwrap();
    let scan = step(
        json!({"request": "scan", "table": {"source": ["orders"]}, "columns": ["order_id"],
               "filters": [], "limit": null}),
        sql(
            "SELECT `order_id` FROM `{catalog}`.`{schema}`.`orders`",
            json!([]),
        ),
    );
    let Answer::Data(rows) = stand_in.fetch(&scan, &BTreeMap::new()).unwrap().answer else {
        panic!("the empty table was not served");
    };
    assert_eq!(rows.batches.iter().map(|b| b.num_rows()).sum::<usize>(), 0);
    let sample = step(
        json!({"request": "sample_entities", "table": {"source": ["users"]},
               "key": ["user_id"], "n": 2, "seed": {"t": "u64", "v": "7"},
               "columns": ["user_id"], "filters": []}),
        sql(
            "SELECT \"user_id\" FROM \"{CATALOG}\".\"{SCHEMA}\".\"users\" SAMPLE ROW (2 ROWS)",
            json!([]),
        ),
    );
    let drawn = stand_in.fetch(&sample, &BTreeMap::new()).unwrap();
    assert_eq!(drawn.by, "draw");
    assert_eq!(keys(&drawn.answer), [2, 5]);
}

#[test]
fn a_draw_asked_with_another_request_seed_is_refused_whatever_the_plan_writes() {
    // Review r3 (Codex P1): the request's own seed is matched, not only the
    // seed the plan's sampling clause writes.
    let stand_in = StandIn::open(capture()).unwrap();
    let asked = step(
        json!({"request": "sample_entities", "table": {"source": ["users"]},
               "key": ["user_id"], "n": 2, "seed": {"t": "u64", "v": "2147483656"},
               "columns": ["user_id"], "filters": []}),
        sql(
            &format!(
                "SELECT `user_id` FROM {USERS} TABLESAMPLE (12.5 PERCENT) REPEATABLE (2147483655) LIMIT 2"
            ),
            json!([]),
        ),
    );
    let refused = stand_in.fetch(&asked, &BTreeMap::new()).err().unwrap();
    assert!(refused.to_string().contains("2147483656"), "{refused}");
}

#[test]
fn a_view_read_is_served_by_exact_replay_and_otherwise_live_only() {
    // Review r3 (Claude P2): a view is no recorded table; its read, which
    // the old client sent, is replayed, and the dropped-table refusal stays.
    use duckdb::arrow::array::{ArrayRef, Int64Array};
    use duckdb::arrow::datatypes::{DataType, Field, Schema};
    use duckdb::arrow::record_batch::RecordBatch;
    use std::sync::Arc;

    let schema = Arc::new(Schema::new(vec![Field::new(
        "user_id",
        DataType::Int64,
        true,
    )]));
    let rows = sdm_testkit::duck::Fetched {
        schema: schema.clone(),
        batches: vec![
            RecordBatch::try_new(
                schema,
                vec![Arc::new(Int64Array::from(vec![7_i64, 9])) as ArrayRef],
            )
            .unwrap(),
        ],
    };
    let view = "`{catalog}`.`{schema}`.`shop_view`";
    let capture = edited(|document, directory| {
        let name = table_file(directory, &rows);
        let statements = document["statements"].as_array_mut().unwrap();
        statements.push(
            json!({"text": format!("DESCRIBE TABLE EXTENDED {view} AS JSON"),
                               "parameters": [], "result": null}),
        );
        statements.push(json!({"text": format!("SELECT `user_id` FROM {view}"),
                               "parameters": [], "result": name}));
    });
    let stand_in = StandIn::open(capture).unwrap();
    let read = |text: String| {
        step(
            json!({"request": "scan", "table": {"source": ["shop_view"]}, "columns": ["user_id"],
                   "filters": [], "limit": null}),
            sql(&text, json!([])),
        )
    };
    let replayed = stand_in
        .fetch(
            &read(format!("SELECT `user_id` FROM {view}")),
            &BTreeMap::new(),
        )
        .unwrap();
    assert_eq!(replayed.by, "replay");
    assert_eq!(keys(&replayed.answer), [7, 9]);
    let other = stand_in
        .fetch(
            &read(format!("SELECT `user_id` FROM {view} LIMIT 1")),
            &BTreeMap::new(),
        )
        .unwrap();
    assert!(matches!(other.answer, Answer::LiveOnly(code) if code == B1));
}
