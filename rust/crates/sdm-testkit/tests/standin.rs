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
