// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! T2.9's exits on `rust-host` (the package's test author's): forecasting's
//! options, the design's T2.9 row (`num_forecasts`, `lag_timesteps`,
//! quantiles; `response_field` for quantiles; "forecast cases on rust-host
//! and python-core"). `contract/tests/test_t29_forecasting.py` holds
//! `python-core` to the same cases, and the cases and python-ref's lines to
//! the row.
//!
//! Each test runs its cases in process on DuckDB or SQLite (SQLite's data
//! loaded by the contract loader, a Python with DuckDB and pyarrow,
//! `SDM_PYTHON`, else `python3`, as T2.2's tests do), each through its
//! `until`, each model call answered by the scripted endpoint's rule, and
//! holds the run to python-ref's committed line for the same key: its
//! outcome (result and code), and where the reference answered, every
//! request the core wrote python-ref's (the `payload` rule: each but its
//! request id, in order, every number as its text). Postgres is held on
//! this host by CI's `cross-host (postgres)` job, which replays
//! python-core's fetches (KO2-20).
//!
//! - `num_forecasts`: FORECAST 1, 2 and 3 and a COUNT over months, one
//!   request a step; 0, -1, 1.5, 10,001, a static target and a condition
//!   refused.
//! - `lag_timesteps`: 0, 1 and 2 on a forecast, 2 on a regression, a
//!   binary classification and a static target, each request carrying its
//!   lagged targets; -1, 1.5, 2.0, true and "2" refused `INVALID_REQUEST`,
//!   never by the rule `lag_timesteps`.
//! - quantiles: each case naming `quantiles` in `outputs` asks for what it
//!   names in every request (the case's `outputs`, which this host passes
//!   to `predict`), answered or not.
//! - `response_field` for quantiles: each case whose answer carries
//!   quantiles (`input.quantiles`) answers through `predict`, never refused
//!   `response_field`, its result holding each row's quantiles as its
//!   step's call answered them: a map column whose keys are `0.1`, `0.5`
//!   and `0.9`, as this host records a map, each value the rule's for the
//!   row's step and position.
//!
//! - a lagged average with empty windows (`forecast.lag-average-empty-windows`,
//!   the orchestrator's decision under KO2-70): this host sends
//!   python-ref's requests but for the lags, each lag cell the average of
//!   that row's own window, computed here from the shop base's orders,
//!   where the reference packs the non-empty windows' values at the top of
//!   the column (`contract/tests/test_t29_forecasting.py` holds the
//!   reference's packing and the same computation's targets).
//!
//! Gating: `contract/tests/t29-pending.json`'s `pending` names, as
//! `rust:<test>`, the tests that failed when written. Each must still fail
//! (strict); once one passes it fails, naming itself, until it is taken
//! out.

#![allow(
    clippy::disallowed_methods,
    reason = "test code: reads the recorded cases and lines"
)]

use std::collections::BTreeMap;
use std::path::PathBuf;

use sdm_testkit::bases;
use sdm_testkit::case::{self, Case};
use sdm_testkit::host::{self, Answer, Observed};
use serde_json::Value;

const NUM_FORECASTS: [&str; 10] = [
    "forecast.steps-1",
    "forecast.steps-2",
    "forecast.steps-3",
    "forecast.steps-zero",
    "forecast.steps-negative",
    "forecast.steps-fraction",
    "forecast.steps-past-grammar",
    "forecast.static-target",
    "forecast.binary-target",
    "forecast.count-months",
];

const LAG_TIMESTEPS: [&str; 12] = [
    "forecast.lag-0",
    "forecast.lag-1",
    "forecast.lag-2",
    "forecast.lag-negative",
    "forecast.lag-fraction",
    "forecast.lag-whole-float",
    "forecast.lag-bool",
    "forecast.lag-text",
    "lag.regression-2",
    "lag.binary-2",
    "lag.static-2",
    "lag.static-negative",
];

const QUANTILES: [&str; 4] = [
    "forecast.quantiles",
    "forecast.quantiles-and-prediction",
    "forecast.quantiles-unanswered",
    "quantiles.regression",
];

const RESPONSE_FIELD: [&str; 3] = [
    "forecast.quantiles",
    "forecast.quantiles-and-prediction",
    "quantiles.regression",
];

/// The rules this package lifts: a case refused by one is the gap.
const LIFTED: [&str; 2] = ["lag_timesteps", "response_field"];

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

fn pending(test: &str) -> bool {
    let text = std::fs::read_to_string(root().join("contract/tests/t29-pending.json")).unwrap();
    let gating: Value = serde_json::from_str(&text).unwrap();
    gating["pending"]
        .as_array()
        .unwrap()
        .iter()
        .any(|t| t == &Value::from(format!("rust:{test}")))
}

/// Runs `check`; a pending test must fail, any other must pass.
fn gated(test: &str, check: impl FnOnce() + std::panic::UnwindSafe) {
    let outcome = std::panic::catch_unwind(check);
    match (outcome, pending(test)) {
        (Ok(()), true) => panic!(
            "{test} passes now: take rust:{test} out of \
             contract/tests/t29-pending.json's pending"
        ),
        (Err(failure), false) => std::panic::resume_unwind(failure),
        _ => {}
    }
}

fn load_case(id: &str) -> Case {
    let mut loaded = case::load(&[root().join(format!("contract/cases/s1/{id}.json"))]).unwrap();
    loaded.remove(0).case
}

/// python-ref's committed lines, by key.
fn reference() -> BTreeMap<String, Value> {
    let env = std::fs::read_to_string(root().join("contract/reference-env.txt")).unwrap();
    let commit = env
        .lines()
        .find_map(|l| l.strip_prefix("reference: "))
        .unwrap()
        .trim()
        .to_owned();
    let text =
        std::fs::read_to_string(root().join(format!("contract/reference/{commit}.jsonl"))).unwrap();
    text.lines()
        .map(|line| {
            let o: Value = serde_json::from_str(line).unwrap();
            (o["case"].as_str().unwrap().to_owned(), o)
        })
        .collect()
}

fn key(id: &str, database: &str) -> String {
    if database == "duckdb" {
        id.to_owned()
    } else {
        format!("{id}@{database}")
    }
}

/// Each case of `ids` run on `database` through this host: its key, the
/// case and what it observed, or why it could not run.
fn observed(ids: &[&str], database: &str) -> Vec<(String, Case, Result<Observed, String>)> {
    let bases = bases::load(&root().join("contract/bases.yaml")).unwrap();
    ids.iter()
        .map(|id| {
            let case = load_case(id);
            let seen = match database {
                "duckdb" => host::run(&case, &bases),
                "sqlite" => host::run_sqlite(&case, &bases, &root()),
                other => panic!("no {other} run here"),
            };
            (key(id, database), case, seen.map_err(|e| format!("{e:#}")))
        })
        .collect()
}

/// A request as the `payload` rule compares it: its request id left out.
fn without_request_id(request: &Value) -> Value {
    let mut request = request.clone();
    if let Some(metadata) = request.get_mut("metadata").and_then(Value::as_object_mut) {
        metadata.remove("request_id");
    }
    request
}

/// The requests python-ref's line records, as the `payload` rule reads
/// them.
fn their_requests(line: &Value) -> Vec<Value> {
    line["public"]["payloads"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| without_request_id(&p["parsed"]))
        .collect()
}

/// The requests this host's run sent, in order: every call's body where
/// there were several, else the one.
fn my_requests(answer: &Value) -> Vec<Value> {
    match answer.get("calls").and_then(Value::as_array) {
        Some(calls) => calls.iter().map(without_request_id).collect(),
        None => vec![without_request_id(&answer["body"])],
    }
}

/// Why this host's run of a case is not python-ref's line `theirs`, if it
/// is not: the outcome and code, a refusal by a rule this package lifts,
/// and where both answer, every request.
fn problem(key: &str, seen: &Observed, theirs: &Value) -> Option<String> {
    let (result, code) = seen.answer.outcome();
    let wanted = (
        theirs["outcome"]["result"].as_str().unwrap_or_default(),
        theirs["outcome"]["code"].as_str().map(str::to_owned),
    );
    if let Answer::Error { fields, .. } = &seen.answer
        && let Some(rule) = fields.get("rule")
        && LIFTED.contains(&rule.as_str())
    {
        return Some(format!("{key}: refused by the rule {rule}"));
    }
    if (result, code.clone()) != wanted {
        return Some(format!(
            "{key}: {result} {code:?} where python-ref is {} {:?}",
            wanted.0, wanted.1
        ));
    }
    if let Answer::Predicted(answer) = &seen.answer {
        let (mine, theirs) = (my_requests(answer), their_requests(theirs));
        if mine.len() != theirs.len() {
            return Some(format!(
                "{key}: {} requests where python-ref sent {}",
                mine.len(),
                theirs.len()
            ));
        }
        for (at, (m, t)) in mine.iter().zip(&theirs).enumerate() {
            if m != t {
                return Some(format!("{key}: request {at} is not python-ref's"));
            }
        }
    }
    None
}

fn as_the_reference(ids: &[&str], database: &str) {
    let reference = reference();
    let mut problems = Vec::new();
    for (key, _, seen) in observed(ids, database) {
        match seen {
            Ok(seen) => problems.extend(problem(&key, &seen, &reference[&key])),
            Err(error) => problems.push(format!("{key}: {error}")),
        }
    }
    assert!(problems.is_empty(), "{}", problems.join("\n"));
}

/// The quantiles the scripted endpoint gives row `index` of call `call`
/// (`contract/runners/scripted_endpoint.py`), each as its IEEE-754 bits.
fn spread(call: u64, index: u64) -> Vec<(String, String)> {
    let d = f64::from(u32::try_from((3 * call + 5 * index) % 9 + 1).unwrap());
    [("0.1", d - 1.0), ("0.5", d), ("0.9", d + 1.0)]
        .into_iter()
        .map(|(q, v)| (q.to_owned(), format!("{:016x}", v.to_bits())))
        .collect()
}

fn integer(cell: &Value) -> Option<u64> {
    cell.get("i64")?.as_str()?.parse().ok()
}

/// Why a predicted answer's result does not hold each row's quantiles as
/// its step's call answered them, if it does not.
fn quantile_problem(key: &str, answer: &Value) -> Option<String> {
    let result = &answer["result"];
    let names: Vec<&str> = result["columns"]
        .as_array()?
        .iter()
        .map(|c| c["name"].as_str().unwrap_or_default())
        .collect();
    let rows = result["rows"].as_array()?;
    let at = |name: &str| names.iter().position(|n| *n == name);
    let Some(row_index) = at("row_index") else {
        return Some(format!("{key}: no row_index column"));
    };
    let step = at("forecast_step");
    let keys = |cell: &Value| -> Option<Vec<String>> {
        cell.get("map")?
            .as_array()?
            .iter()
            .map(|pair| pair[0].as_str().map(str::to_owned))
            .collect()
    };
    let Some(column) = (0..names.len()).find(|&c| {
        !rows.is_empty()
            && rows.iter().all(|row| {
                keys(&row[c]).as_deref() == Some(&["0.1", "0.5", "0.9"].map(String::from)[..])
            })
    }) else {
        return Some(format!(
            "{key}: no column holds each row's quantiles (columns {names:?})"
        ));
    };
    for row in rows {
        let index = integer(&row[row_index])?;
        let call = step.and_then(|s| integer(&row[s])).map_or(0, |s| s - 1);
        let given: Vec<(String, String)> = row[column]["map"]
            .as_array()?
            .iter()
            .map(|pair| {
                (
                    pair[0].as_str().unwrap_or_default().to_owned(),
                    pair[1].as_str().unwrap_or_default().to_owned(),
                )
            })
            .collect();
        if given != spread(call, index) {
            return Some(format!(
                "{key}: row {index} of step {} holds {given:?}, not its call's quantiles",
                call + 1
            ));
        }
    }
    None
}

fn response_field(database: &str) {
    let reference = reference();
    let mut problems = Vec::new();
    for (key, _, seen) in observed(&RESPONSE_FIELD, database) {
        let seen = match seen {
            Ok(seen) => seen,
            Err(error) => {
                problems.push(format!("{key}: {error}"));
                continue;
            }
        };
        if let Some(problem) = problem(&key, &seen, &reference[&key]) {
            problems.push(problem);
            continue;
        }
        match &seen.answer {
            Answer::Predicted(answer) => problems.extend(quantile_problem(&key, answer)),
            other => problems.push(format!("{key}: {other:?}")),
        }
    }
    assert!(problems.is_empty(), "{}", problems.join("\n"));
}

#[test]
fn rust_host_num_forecasts_as_the_reference_on_duckdb() {
    gated("rust_host_num_forecasts_as_the_reference_on_duckdb", || {
        as_the_reference(&NUM_FORECASTS, "duckdb");
    });
}

#[test]
fn rust_host_num_forecasts_as_the_reference_on_sqlite() {
    gated("rust_host_num_forecasts_as_the_reference_on_sqlite", || {
        as_the_reference(&NUM_FORECASTS, "sqlite");
    });
}

#[test]
fn rust_host_lag_timesteps_as_the_reference_on_duckdb() {
    gated("rust_host_lag_timesteps_as_the_reference_on_duckdb", || {
        as_the_reference(&LAG_TIMESTEPS, "duckdb");
    });
}

#[test]
fn rust_host_lag_timesteps_as_the_reference_on_sqlite() {
    gated("rust_host_lag_timesteps_as_the_reference_on_sqlite", || {
        as_the_reference(&LAG_TIMESTEPS, "sqlite");
    });
}

#[test]
fn rust_host_quantiles_asked_as_the_reference_on_duckdb() {
    gated(
        "rust_host_quantiles_asked_as_the_reference_on_duckdb",
        || {
            as_the_reference(&QUANTILES, "duckdb");
        },
    );
}

#[test]
fn rust_host_quantiles_asked_as_the_reference_on_sqlite() {
    gated(
        "rust_host_quantiles_asked_as_the_reference_on_sqlite",
        || {
            as_the_reference(&QUANTILES, "sqlite");
        },
    );
}

#[test]
fn rust_host_response_field_for_quantiles_on_duckdb() {
    gated("rust_host_response_field_for_quantiles_on_duckdb", || {
        response_field("duckdb");
    });
}

#[test]
fn rust_host_response_field_for_quantiles_on_sqlite() {
    gated("rust_host_response_field_for_quantiles_on_sqlite", || {
        response_field("sqlite");
    });
}

/// A lagged average whose windows are sometimes empty (KO2-70).
const LAG_ALIGNED: &str = "forecast.lag-average-empty-windows";
/// Its query's window, in days, and its lags.
const WINDOW_DAYS: i64 = 28;
const LAG_STEPS: usize = 2;

/// The shop base's orders, as `contract/bases.yaml` builds them: order `i`
/// is user `i % 8`'s, at day `7 i` from 2023-01-01, priced `10 + 1.5 i`;
/// each as (user, day, price).
fn shop_orders() -> Vec<(i64, i64, f64)> {
    let text = std::fs::read_to_string(root().join("contract/bases.yaml")).unwrap();
    assert!(
        text.contains(
            "(i % 8)::BIGINT AS user_id, (10 + i * 1.5)::DOUBLE AS price, \
             TIMESTAMP '2023-01-01' + to_days((i * 7)::INTEGER) AS ts FROM range(80)"
        ),
        "the shop base is not the one this computation reads"
    );
    (0..80_i32)
        .map(|i| {
            (
                i64::from(i % 8),
                i64::from(7 * i),
                10.0 + 1.5 * f64::from(i),
            )
        })
        .collect()
}

/// Days from 1970-01-01 of a civil date.
fn civil(year: i64, month: i64, day: i64) -> i64 {
    let y = if month <= 2 { year - 1 } else { year };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = (month + 9) % 12;
    let doy = (153 * mp + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// A payload's anchor time (a midnight) as days from 2023-01-01.
fn day_of(text: &str) -> i64 {
    assert!(text.ends_with("T00:00:00.000000Z"), "{text}");
    let part = |r: std::ops::Range<usize>| text[r].parse::<i64>().unwrap();
    civil(part(0..4), part(5..7), part(8..10)) - civil(2023, 1, 1)
}

/// The average price of `user`'s orders after day `after` and up to day
/// `until`, or `None` where the window holds none.
fn window_average(orders: &[(i64, i64, f64)], user: i64, after: i64, until: i64) -> Option<f64> {
    let prices: Vec<f64> = orders
        .iter()
        .filter(|(u, day, _)| *u == user && after < *day && *day <= until)
        .map(|(_, _, price)| *price)
        .collect();
    (!prices.is_empty()).then(|| prices.iter().sum::<f64>() / prices.len() as f64)
}

/// A column's cells over the context rows then the rows to score, each
/// number as a float, an empty cell `None`.
fn cells(body: &Value, column: &str) -> Vec<Option<f64>> {
    let mut out = Vec::new();
    for half in ["context", "predict"] {
        let table = &body[half]["instance_table"];
        let Some(at) = table["columns"]
            .as_array()
            .unwrap()
            .iter()
            .position(|c| c == column)
        else {
            continue;
        };
        for row in table["rows"].as_array().unwrap() {
            out.push(row[at]["number"].as_str().map(|n| n.parse().unwrap()));
        }
    }
    out
}

/// For the request of forecast step `step` (from 0; its window days
/// `step * 28` to `(step + 1) * 28`), each instance row's lag `j` as the
/// shop base's orders give it, context rows then rows to score: the
/// average over (anchor - (j + 1) end + start, anchor - j end] of the user
/// `__kumo_entity_ref_0` names.
fn by_hand(body: &Value, step: i64, j: usize) -> Vec<Option<f64>> {
    let orders = shop_orders();
    let (start, end) = (step * WINDOW_DAYS, (step + 1) * WINDOW_DAYS);
    let back = i64::try_from(j).unwrap();
    let mut out = Vec::new();
    for half in ["context", "predict"] {
        let table = &body[half]["instance_table"];
        let columns = table["columns"].as_array().unwrap();
        let at = |name: &str| columns.iter().position(|c| c == name).unwrap();
        let (entity, anchor) = (at("__kumo_entity_ref_0"), at("__kumo_anchor_time"));
        for row in table["rows"].as_array().unwrap() {
            let user: i64 = row[entity]["number"].as_str().unwrap().parse().unwrap();
            let day = day_of(row[anchor].as_str().unwrap());
            out.push(window_average(
                &orders,
                user,
                day - (back + 1) * end + start,
                day - back * end,
            ));
        }
    }
    out
}

/// A request with `columns` taken out of both instance tables.
fn without(body: &Value, columns: &[String]) -> Value {
    let mut body = body.clone();
    for half in ["context", "predict"] {
        let table = &mut body[half]["instance_table"];
        let names = table["columns"].as_array().unwrap().clone();
        let keep: Vec<usize> = (0..names.len())
            .filter(|&i| !columns.iter().any(|c| names[i] == c.as_str()))
            .collect();
        table["columns"] = Value::Array(keep.iter().map(|&i| names[i].clone()).collect());
        let rows = table["rows"].as_array().unwrap().clone();
        table["rows"] = Value::Array(
            rows.iter()
                .map(|row| Value::Array(keep.iter().map(|&i| row[i].clone()).collect()))
                .collect(),
        );
    }
    body
}

fn lagged_average_on_its_own_rows(database: &str) {
    let reference = reference();
    let (key, _, seen) = observed(&[LAG_ALIGNED], database).remove(0);
    let seen = seen.unwrap_or_else(|e| panic!("{key}: {e}"));
    let Answer::Predicted(answer) = &seen.answer else {
        panic!("{key}: {:?}", seen.answer);
    };
    let (mine, theirs) = (my_requests(answer), their_requests(&reference[&key]));
    assert_eq!(mine.len(), theirs.len(), "{key}: requests");
    let lags: Vec<String> = (0..LAG_STEPS).map(|j| format!("__kumo_arl{j}__")).collect();
    let mut problems = Vec::new();
    for (step, (got, want)) in (0_i64..).zip(mine.iter().zip(&theirs)) {
        for (j, column) in lags.iter().enumerate() {
            let own = by_hand(want, step, j);
            if cells(got, column) != own {
                problems.push(format!(
                    "{key}: step {}: {column} is {:?}, not each row's own window {own:?}",
                    step + 1,
                    cells(got, column)
                ));
            }
        }
        if without(got, &lags) != without(want, &lags) {
            problems.push(format!(
                "{key}: step {}: the request but its lags is not python-ref's",
                step + 1
            ));
        }
    }
    assert!(problems.is_empty(), "{}", problems.join("\n"));
}

/// Control: the computation reads the reference's requests as the engine
/// does where nothing is dropped, and is not vacuous: python-ref's lags are
/// the non-empty windows packed at the top of three of the four columns,
/// and the rows scored get no second lag where their windows hold 46.0
/// and 47.5.
#[test]
fn the_reference_s_lagged_average_is_packed() {
    let reference = reference();
    for database in ["duckdb", "sqlite", "postgres"] {
        let theirs = their_requests(&reference[&key(LAG_ALIGNED, database)]);
        assert_eq!(theirs.len(), 2, "{database}");
        let mut packed = 0;
        for (step, body) in (0_i64..).zip(&theirs) {
            for j in 0..LAG_STEPS {
                let own = by_hand(body, step, j);
                let mut top: Vec<Option<f64>> =
                    own.iter().copied().filter(Option::is_some).collect();
                top.resize(own.len(), None);
                assert_eq!(cells(body, &format!("__kumo_arl{j}__")), top, "{database}");
                packed += usize::from(own != top);
            }
        }
        assert_eq!(packed, 3, "{database}");
        let own = by_hand(&theirs[0], 0, 1);
        assert_eq!(own[own.len() - 2..], [Some(46.0), Some(47.5)], "{database}");
    }
}

#[test]
fn rust_host_lagged_average_on_its_own_rows_on_duckdb() {
    gated("rust_host_lagged_average_on_its_own_rows_on_duckdb", || {
        lagged_average_on_its_own_rows("duckdb");
    });
}

#[test]
fn rust_host_lagged_average_on_its_own_rows_on_sqlite() {
    gated("rust_host_lagged_average_on_its_own_rows_on_sqlite", || {
        lagged_average_on_its_own_rows("sqlite");
    });
}
