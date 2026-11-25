// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Forecast execution through the Rust host and scripted model endpoint.
//!
//! Direct checks cover source-derived labels, forecast windows, result identity,
//! observation shape and bounded refusals. Expected labels come from the known
//! source rows, not a captured execution of either client.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use sdm_testkit::bases;
use sdm_testkit::case::Case;
use sdm_testkit::host::{self, Answer};
use serde_json::{Value, json};

#[path = "support/shop_assertions.rs"]
mod shop_assertions;
use shop_assertions::assert_shop_history;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

/// The calls the core made, from `predict`'s in-process answer: its
/// `calls` when it carries them, else its `body`, a list or one object.
fn calls(predicted: &Value) -> Vec<Value> {
    if let Value::Array(bodies) = &predicted["calls"] {
        return bodies.clone();
    }
    match &predicted["body"] {
        Value::Array(bodies) => bodies.clone(),
        Value::Null => vec![],
        body => vec![body.clone()],
    }
}

#[test]
fn invalid_forecasts_return_named_refusals() {
    let bases = bases::load(&root().join("contract/bases.yaml")).unwrap();
    for (name, pql, anchor) in [
        (
            "zero",
            "PREDICT SUM(orders.price, 0, 7, days) FORECAST 0 TIMEFRAMES FOR EACH users.user_id",
            "2024-03-01",
        ),
        (
            "too-many",
            "PREDICT SUM(orders.price, 0, 1, days) FORECAST 201 TIMEFRAMES FOR EACH users.user_id",
            "2024-03-01",
        ),
        (
            "outside-data",
            "PREDICT SUM(orders.price, 0, 7, days) FORECAST 3 TIMEFRAMES FOR EACH users.user_id",
            "2023-01-10",
        ),
        (
            "boolean",
            "PREDICT SUM(orders.price, 0, 30, days) > 10 FORECAST 2 TIMEFRAMES FOR EACH users.user_id",
            "2024-03-01",
        ),
        (
            "static",
            "PREDICT users.age FORECAST 2 TIMEFRAMES FOR EACH users.user_id",
            "2024-03-01",
        ),
    ] {
        let (mut input, _) = many_steps("3");
        input["id"] = json!(format!("forecast-refuse-{name}"));
        input["input"]["pql"] = json!(pql);
        input["input"]["materialize"]["anchor_time"] = json!(anchor);
        input["core"]["materialize"]["anchor_time"]["v"] = json!(if anchor == "2023-01-10" {
            "1673308800000000000"
        } else {
            "1709251200000000000"
        });
        let case: Case = serde_json::from_value(input).unwrap();
        let observed = host::run(&case, &bases).unwrap();
        match observed.answer {
            Answer::Error { code, .. } => assert_eq!(code, "INVALID_REQUEST", "{name}"),
            _ => panic!("{name}: expected a refusal"),
        }
    }
}

// --- the observation the case runner writes -------------------------------------

/// The JSON Schema type names `value` satisfies.
fn is_type(value: &Value, name: &str) -> bool {
    match name {
        "object" => value.is_object(),
        "array" => value.is_array(),
        "string" => value.is_string(),
        "integer" => value.is_i64() || value.is_u64(),
        "number" => value.is_number(),
        "boolean" => value.is_boolean(),
        "null" => value.is_null(),
        _ => false,
    }
}

/// `value` against `schema`'s `type`, `required`, `properties`,
/// `additionalProperties: false` and `items`: the keywords
/// `contract/observation.schema.json` gives `core.predicted`.
fn conforms(value: &Value, schema: &Value, at: &str, problems: &mut Vec<String>) {
    let types: Vec<&str> = match &schema["type"] {
        Value::String(one) => vec![one.as_str()],
        Value::Array(many) => many.iter().filter_map(Value::as_str).collect(),
        _ => vec![],
    };
    if !types.is_empty() && !types.iter().any(|t| is_type(value, t)) {
        problems.push(format!("{at} fails 'type' {types:?}"));
        return;
    }
    if let Value::Object(object) = value {
        for key in schema["required"].as_array().into_iter().flatten() {
            let key = key.as_str().unwrap();
            if !object.contains_key(key) {
                problems.push(format!("{at} lacks {key}"));
            }
        }
        for (key, item) in object {
            match schema["properties"].get(key) {
                Some(inner) => conforms(item, inner, &format!("{at}/{key}"), problems),
                None if schema["additionalProperties"] == false => {
                    problems.push(format!("{at}/{key} is not allowed"));
                }
                None => {}
            }
        }
    }
    if let (Value::Array(items), Some(inner)) = (value, schema.get("items")) {
        for (i, item) in items.iter().enumerate() {
            conforms(item, inner, &format!("{at}/{i}"), problems);
        }
    }
}

/// `cases` through `sdm-contract-run`, as CI runs it: each observation by
/// case id.
fn contract_run(name: &str, cases: &[Value]) -> BTreeMap<String, Value> {
    static RUN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let serial = RUN.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("forecast-{name}-{}-{serial}", std::process::id()));
    let into = dir.join("cases");
    std::fs::create_dir_all(&into).unwrap();
    for case in cases {
        let id = case["id"].as_str().unwrap();
        std::fs::write(into.join(format!("{id}.json")), case.to_string()).unwrap();
    }
    let out = dir.join("rust-host.jsonl");
    let done = Command::new(env!("CARGO_BIN_EXE_sdm-contract-run"))
        .current_dir(root())
        .stdin(Stdio::null())
        .arg("--out")
        .arg(&out)
        .arg(&into)
        .output()
        .unwrap();
    assert!(
        done.status.success(),
        "sdm-contract-run failed: {}",
        String::from_utf8_lossy(&done.stderr)
    );
    std::fs::read_to_string(&out)
        .unwrap()
        .lines()
        .map(|line| {
            let observation: Value = serde_json::from_str(line).unwrap();
            (
                observation["case"].as_str().unwrap().to_owned(),
                observation,
            )
        })
        .collect()
}

fn assert_daily_forecast(
    steps: i64,
    offset: i64,
    width: i64,
    early: bool,
    count: bool,
    filtered: bool,
    clause: &str,
) {
    let aggregate = if count { "COUNT" } else { "SUM" };
    let target = if filtered {
        "orders.* WHERE orders.price > 50"
    } else if count {
        "orders.*"
    } else {
        "orders.price"
    };
    let (mut case, _) = many_steps(&steps.to_string());
    let anchor = if early { "2023-01-10" } else { "2024-03-01" };
    let anchor_day = if early { 9 } else { 425 };
    case["input"]["materialize"]["anchor_time"] = json!(anchor);
    case["core"]["materialize"]["anchor_time"]["v"] = json!(if early {
        "1673308800000000000"
    } else {
        "1709251200000000000"
    });
    case["id"] = json!("forecast-observation-grid");
    case["input"]["pql"] = json!(format!(
        "PREDICT {aggregate}({target}, {offset}, {}, days) FORECAST {steps} TIMEFRAMES FOR EACH users.user_id{clause}",
        offset + width
    ));
    let observations = contract_run("direct-grid", &[case]);
    let observation = &observations["forecast-observation-grid"];
    assert_eq!(observation["outcome"]["result"], "success");
    let predicted = &observation["core"]["predicted"];
    let schema: Value = serde_json::from_str(
        &std::fs::read_to_string(root().join("contract/observation.schema.json")).unwrap(),
    )
    .unwrap();
    let mut problems = Vec::new();
    conforms(
        predicted,
        &schema["properties"]["core"]["properties"]["predicted"],
        "core/predicted",
        &mut problems,
    );
    assert!(problems.is_empty(), "{problems:?}");
    let calls = calls(predicted);
    assert_eq!(calls.len(), steps as usize);
    assert_eq!(&predicted["body"], calls.last().unwrap());
    let calendar = duckdb::Connection::open_in_memory().unwrap();
    for (step, body) in calls.iter().enumerate() {
        assert_shop_history(body);
        let start = offset + step as i64 * width;
        let end = start + width;
        // Forecast decomposition keeps the entity clause but does not carry
        // ASSUMING into step queries. Assert the existing behavior explicitly;
        // accepting this input is not evidence of counterfactual execution.
        let step_clause = if clause.starts_with(" ASSUMING") {
            ""
        } else {
            clause
        };
        assert_eq!(
            body["metadata"]["source_query"],
            format!(
                "PREDICT {aggregate}({target}, {start}, {end}, days) FOR EACH users.user_id{step_clause}"
            )
        );
        assert_eq!(body["task"]["kind"], "regression");
        assert_eq!(
            body["task"]["target"],
            json!({"column_name":"TARGET","dtype":"float32"})
        );
        let context = &body["context"]["instance_table"];
        assert_eq!(
            context["columns"],
            json!([
                "instance_id",
                "__kumo_entity_ref_0",
                "__kumo_anchor_time",
                "TARGET"
            ])
        );
        let mut grid = std::collections::BTreeSet::new();
        for row in context["rows"].as_array().unwrap() {
            let entity: i64 = row[1]["number"].as_str().unwrap().parse().unwrap();
            // Only calendar conversion uses DuckDB. Expected labels below are
            // ordinary arithmetic over the independently specified shop rows.
            let day: i64 = calendar
                .query_row(
                    "SELECT date_diff('day', DATE '2023-01-01', CAST(? AS TIMESTAMP))",
                    [row[2].as_str().unwrap()],
                    |r| r.get(0),
                )
                .unwrap();
            assert!(grid.insert((entity, day)), "duplicate training example");
            let expected: f64 = (0..80_i64)
                .filter(|i| {
                    i % 8 == entity
                        && i * 7 > day + start
                        && i * 7 <= day + end
                        && (!filtered || 10.0 + 1.5 * *i as f64 > 50.0)
                })
                .map(|i| if count { 1.0 } else { 10.0 + 1.5 * i as f64 })
                .sum();
            let actual: f64 = row[3]["number"].as_str().unwrap().parse().unwrap();
            assert_eq!(actual, expected, "step={step} entity={entity} day={day}");
        }
        // Training anchors walk backwards by the window width, leaving
        // enough source history for the full label window.
        let first_entity = if clause.starts_with(" WHERE") { 2 } else { 0 };
        let expected_grid: std::collections::BTreeSet<_> = (first_entity..8_i64)
            .flat_map(|entity| {
                (0..=((anchor_day - end) / width))
                    .map(move |n| (entity, anchor_day - end - n * width))
            })
            .collect();
        assert_eq!(grid, expected_grid);
        let rows = body["predict"]["instance_table"]["rows"]
            .as_array()
            .unwrap();
        assert_eq!(rows.len(), 2);
        for (entity, row) in rows.iter().enumerate() {
            assert_eq!(row[1], json!({"number":entity.to_string()}));
            assert_eq!(row[2], format!("{anchor}T00:00:00.000000Z"));
        }
    }
    assert_forecast_result(predicted, steps);
}

fn assert_forecast_result(predicted: &Value, steps: i64) {
    assert_eq!(
        predicted["result"]["columns"],
        json!([
            {"name":"row_index","type":"int64"},
            {"name":"id","type":"int64"},
            {"name":"forecast_step","type":"int64"},
            {"name":"prediction","type":"float64"},
            {"name":"answer_index","type":"int64"}
        ])
    );
    // The scripted endpoint returns 1.0. Output is entity-major, while
    // answer_index records step-major endpoint submission order.
    let expected: Vec<Value> = (0..2)
        .flat_map(|entity| {
            (1..=steps).map(move |step| {
                json!([
                    {"i64": entity.to_string()}, {"i64": entity.to_string()},
                    {"i64": step.to_string()}, {"f64_bits":"3ff0000000000000"},
                    {"i64": ((step - 1) * 2 + entity).to_string()}
                ])
            })
        })
        .collect();
    assert_eq!(predicted["result"]["rows"], json!(expected));
}

#[test]
fn sum_forecasts_preserve_windows_labels_and_result_identity() {
    assert_daily_forecast(3, 0, 7, false, false, false, "");
    assert_daily_forecast(1, 0, 30, false, false, false, "");
    assert_daily_forecast(2, 7, 7, false, false, false, "");
    assert_daily_forecast(3, 0, 3, true, false, false, "");
}

#[test]
fn count_forecasts_preserve_filtered_labels_and_empty_windows() {
    assert_daily_forecast(4, 0, 14, false, true, false, "");
    assert_daily_forecast(3, 0, 30, false, true, true, "");
}

#[test]
fn forecasts_filter_entities_and_accept_assumption_input() {
    assert_daily_forecast(3, 0, 7, false, false, false, " WHERE users.age > 21");
    assert_daily_forecast(
        3,
        0,
        7,
        false,
        false,
        false,
        " ASSUMING COUNT(orders.*, 0, 7, days) > 0",
    );
}

#[test]
fn nonadditive_forecasts_preserve_calendar_windows_and_labels() {
    let calendar = duckdb::Connection::open_in_memory().unwrap();
    for (aggregate, unit, width) in [
        ("AVG", "weeks", 2),
        ("MIN", "months", 1),
        ("MAX", "hours", 48),
    ] {
        let (mut case, _) = many_steps("2");
        let id = format!("forecast-{aggregate}");
        case["id"] = json!(id);
        case["input"]["pql"] = json!(format!(
            "PREDICT {aggregate}(orders.price, 0, {width}, {unit}) FORECAST 2 TIMEFRAMES FOR EACH users.user_id"
        ));
        let observations = contract_run(&id, &[case]);
        let observation = &observations[&id];
        assert_eq!(observation["outcome"]["result"], "success", "{aggregate}");
        let predicted = &observation["core"]["predicted"];
        let bodies = calls(predicted);
        assert_eq!(bodies.len(), 2);
        for (step, body) in bodies.iter().enumerate() {
            assert_shop_history(body);
            let start = step as i64 * width;
            let end = start + width;
            assert_eq!(
                body["metadata"]["source_query"],
                format!(
                    "PREDICT {aggregate}(orders.price, {start}, {end}, {unit}) FOR EACH users.user_id"
                )
            );
            // Independent source relation and aggregate, without invoking any
            // engine sampler or using its returned rows to choose expectations.
            // Empty AVG/MIN/MAX windows produce no labelled training example.
            // Sampler strides use the window measured from Jan1,1970:
            // Jan-Feb is31 days; Feb-Mar is28. Label bounds remain calendar months.
            let stride = if unit == "months" {
                if step == 0 { "31 days" } else { "28 days" }
            } else if unit == "weeks" {
                "7 days"
            } else {
                "1 hour"
            };
            let sql = format!(
                r#"
                WITH orders AS (
                    SELECT i % 8 AS entity, 10.0 + 1.5*i AS price,
                           TIMESTAMP '2023-01-01' + i * INTERVAL '7 days' AS ts
                    FROM range(80) AS source(i)
                ), anchors AS (
                    SELECT TIMESTAMP '2024-03-01' - {end} * INTERVAL '1 {unit}' - n*{width} * INTERVAL '{stride}' AS anchor_ts
                    FROM range(426) AS steps(n)
                )
                SELECT entity, strftime(anchor_ts, '%Y-%m-%dT%H:%M:%S.000000Z'),
                       CAST({aggregate}(price) AS DOUBLE)
                FROM anchors JOIN orders
                  ON ts > anchor_ts + {start} * INTERVAL '1 {unit}'
                 AND ts <= anchor_ts + {end} * INTERVAL '1 {unit}'
                WHERE anchor_ts >= TIMESTAMP '2023-01-01'
                GROUP BY entity, anchor_ts
            "#
            );
            let mut query = calendar.prepare(&sql).unwrap();
            let expected: BTreeMap<(i64, String), f64> = query
                .query_map([], |row| Ok(((row.get(0)?, row.get(1)?), row.get(2)?)))
                .unwrap()
                .map(Result::unwrap)
                .collect();
            assert!(!expected.is_empty());
            let rows = body["context"]["instance_table"]["rows"]
                .as_array()
                .unwrap();
            let actual: BTreeMap<(i64, String), f64> = rows
                .iter()
                .map(|row| {
                    (
                        (
                            row[1]["number"].as_str().unwrap().parse().unwrap(),
                            row[2].as_str().unwrap().to_owned(),
                        ),
                        row[3]["number"].as_str().unwrap().parse().unwrap(),
                    )
                })
                .collect();
            assert_eq!(actual.len(), rows.len(), "duplicate entity/anchor");
            assert_eq!(actual, expected, "{aggregate} step {step}");
            assert_eq!(
                body["predict"]["instance_table"]["rows"]
                    .as_array()
                    .unwrap()
                    .len(),
                2
            );
        }
        assert_forecast_result(predicted, 2);
    }
}

// --- a forecast of many steps is refused at once -------------------------------

const QUICK: Duration = Duration::from_secs(2);

/// The forecast `sum-days-3` asked over `n` steps, and the S0 case
/// `s0.refuse.task-forecast-zero` asked the same: each case as JSON.
fn many_steps(n: &str) -> (Value, Value) {
    // Build the boundary input from the maintained source case, not a
    // forecast output recording.
    let mut materialize: Value = serde_json::from_str(
        &std::fs::read_to_string(root().join("contract/cases/s1/base.regression.json")).unwrap(),
    )
    .unwrap();
    materialize["until"] = json!("predict");
    let pql = format!(
        "PREDICT SUM(orders.price, 0, 1, days) FORECAST {n} TIMEFRAMES FOR EACH users.user_id"
    );
    materialize["id"] = json!(format!("t136fc.steps-{n}"));
    materialize["input"]["pql"] = json!(pql);
    let text =
        std::fs::read_to_string(root().join("contract/cases/s0/s0.refuse.task-forecast-zero.json"))
            .unwrap();
    let mut validate: Value = serde_json::from_str(&text).unwrap();
    validate["id"] = json!(format!("t136fc.validate-{n}"));
    validate["input"]["pql"] = json!(pql);
    (materialize, validate)
}

/// The case run on this host, timed: its outcome and code.
fn timed(case: &Value) -> (Value, Duration) {
    let case: Case = serde_json::from_value(case.clone()).unwrap();
    let bases = bases::load(&root().join("contract/bases.yaml")).unwrap();
    let start = Instant::now();
    let observed = host::run(&case, &bases).unwrap();
    let took = start.elapsed();
    let (outcome, code) = observed.answer.outcome();
    (json!([outcome, code]), took)
}

#[test]
fn a_forecast_of_many_steps_is_refused_at_once() {
    // Past the grammar's most (10,000) the query itself is refused, at
    // validation and at materialize; at 10,000 and at 201, past the default
    // most_parts (200), materialize refuses it. Each INVALID_REQUEST within
    // two seconds of the same run at 201 steps, timed just before it: no
    // step is built for a refused forecast's count. A materialize run loads
    // its base and describes it first, which takes a second or more of a
    // debug build on a loaded machine, so the bound is over that run's own
    // time, three times over, not a fixed one.
    let (floor_case, _) = many_steps("201");
    for n in ["1000000000", "4294967297", "10001", "10000", "201"] {
        let (materialize, validate) = many_steps(n);
        let (_, floor) = timed(&floor_case);
        let (got, took) = timed(&materialize);
        assert_eq!(
            got,
            json!(["refusal", "INVALID_REQUEST"]),
            "materialize, {n} steps"
        );
        assert!(
            took < floor * 3 + QUICK,
            "materialize, {n} steps: {took:?}, at 201 steps {floor:?}"
        );
        let (got, took) = timed(&validate);
        let accepted = matches!(n, "10000" | "201");
        let want = if accepted {
            json!(["success", null])
        } else {
            json!(["refusal", "INVALID_REQUEST"])
        };
        // most_parts is materialize's: validation accepts what the grammar
        // allows, whatever the caller's limit.
        assert_eq!(got, want, "validate_query, {n} steps");
        assert!(took < QUICK, "validate_query, {n} steps: {took:?}");
    }
}
