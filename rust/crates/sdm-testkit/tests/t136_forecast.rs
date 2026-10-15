// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! T1.36 fix 21 (owner item 50): forecast decomposition moves into the
//! core, so a host with no Python answers a multi-step forecast. Held on
//! `rust-host` from outside the implementation (the independent test
//! author) to `contract/t136-forecast-vectors.json`, recorded from the
//! reference by `contract/tools/forecast_record.py`;
//! `contract/tests/test_t136_forecast.py` holds `python-core` and
//! `python-old` to the same vectors.
//!
//! Each vector's `case` (`base.regression`'s, its forecast question and
//! anchor) runs through this host as any case does: `describe_source`,
//! `materialize` and `predict` against the scripted endpoint. Where the
//! vector's `rust_host` is `reference`, the answer must be the reference's:
//! a refusal with its code; or a success whose model calls are the
//! reference's requests, as many and in the same order, each body's digest
//! as the `payload` rule compares it (the request id masked to its type,
//! each number by kind, sign and exact value, sorted keys) and the tables
//! it carries, and whose result holds the reference's `to_pandas()`
//! columns (`row_index`, `id`, `forecast_step`, `prediction`) by name with
//! the reference's rows, cell for cell. The calls are read from this
//! host's in-process answer (`host::run`'s `Answer::Predicted`): its
//! `calls`, every call's body in the order sent, when it carries them;
//! otherwise its `body`, a list of every call's or one call's object. The
//! observation the case runner writes is held apart, to python-core's
//! shape (`rust_host_s_observation_of_a_forecast_is_python_core_s`). Where it is `python-core` (a target or
//! entity filter, which the core does not port), the answer must be
//! python-core's refusal code as recorded. Fails today on every vector the
//! reference answers: the core refuses a forecast `UNSUPPORTED_REQUEST`
//! (rule `problem_type`). The implementer changes nothing of this file, the
//! vectors or the recorder.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use sdm_testkit::bases;
use sdm_testkit::case::Case;
use sdm_testkit::host::{self, Answer};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

fn vectors() -> Value {
    let text = std::fs::read_to_string(root().join("contract/t136-forecast-vectors.json")).unwrap();
    let all: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(all["format"], "sdm-forecast-vectors/1");
    assert_eq!(all["recorded_by"], "contract/tools/forecast_record.py");
    all
}

fn exclusions(all: &Value) -> Vec<Vec<String>> {
    all["payload_exclusions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| {
            p.as_array()
                .unwrap()
                .iter()
                .map(|s| s.as_str().unwrap().to_owned())
                .collect()
        })
        .collect()
}

/// A number token by kind, sign, significant digits and power of ten, as
/// the recorder's `number_form` reads it.
fn number_form(text: &str) -> Value {
    let integer = {
        let digits = text.strip_prefix('-').unwrap_or(text);
        !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit())
    };
    let kind = if integer { "integer" } else { "float" };
    let (negative, body) = match text.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, text.strip_prefix('+').unwrap_or(text)),
    };
    let (mantissa, power) = match body.split_once(['e', 'E']) {
        Some((m, p)) => (m, p.parse::<i64>().expect("an exponent")),
        None => (body, 0),
    };
    let (whole, fraction) = mantissa.split_once('.').unwrap_or((mantissa, ""));
    let digits = format!("{whole}{fraction}");
    let mut exponent = power - i64::try_from(fraction.len()).unwrap();
    let significant = digits.trim_start_matches('0');
    if significant.is_empty() {
        return json!([kind, negative, "0", 0]);
    }
    let stripped = significant.trim_end_matches('0');
    exponent += i64::try_from(significant.len() - stripped.len()).unwrap();
    json!([kind, negative, stripped, exponent])
}

/// Python's name of a parsed JSON value's type, as the recorder masks it.
fn type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "NoneType",
        Value::Bool(_) => "bool",
        Value::Number(_) => "int",
        Value::String(_) => "str",
        Value::Array(_) => "list",
        Value::Object(_) => "dict",
    }
}

fn masked(value: &Value, path: &[String]) -> Value {
    let Some((head, rest)) = path.split_first() else {
        return value.clone();
    };
    let Some(object) = value.as_object() else {
        return value.clone();
    };
    let Some(inner) = object.get(head) else {
        return value.clone();
    };
    let mut out = object.clone();
    let replaced = if rest.is_empty() {
        Value::from(format!("<{}>", type_name(inner)))
    } else {
        masked(inner, rest)
    };
    out.insert(head.clone(), replaced);
    Value::Object(out)
}

/// The body with each number as `number_form` reads it and every object's
/// keys sorted.
fn canonical(value: &Value) -> Value {
    match value {
        Value::Object(object) => {
            if object.len() == 1
                && let Some(Value::String(text)) = object.get("number")
            {
                return json!({"number": number_form(text)});
            }
            let sorted: BTreeMap<&String, &Value> = object.iter().collect();
            let mut out = Map::new();
            for (key, item) in sorted {
                out.insert(key.clone(), canonical(item));
            }
            Value::Object(out)
        }
        Value::Array(items) => Value::Array(items.iter().map(canonical).collect()),
        other => other.clone(),
    }
}

/// The recorder's `digest`: the SHA-256 of the canonical body.
fn digest(body: &Value, exclusions: &[Vec<String>]) -> String {
    let mut body = body.clone();
    for path in exclusions {
        body = masked(&body, path);
    }
    let text = serde_json::to_string(&canonical(&body)).unwrap();
    Sha256::digest(text.as_bytes())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// Each object of a body holding `columns` and `rows`, by its path: its
/// row count and null cells (the recorder's `tables_with_rows`).
fn tables_with_rows(value: &Value, at: &str, out: &mut Map<String, Value>) {
    if let Value::Object(object) = value {
        if let (Some(Value::Array(rows)), Some(Value::Array(_))) =
            (object.get("rows"), object.get("columns"))
        {
            let nulls = rows
                .iter()
                .filter_map(Value::as_array)
                .flatten()
                .filter(|c| c.is_null())
                .count();
            let key = if at.is_empty() { "." } else { at };
            out.insert(key.to_owned(), json!({"rows": rows.len(), "nulls": nulls}));
        }
        for (key, item) in object {
            let path = if at.is_empty() {
                key.clone()
            } else {
                format!("{at}.{key}")
            };
            tables_with_rows(item, &path, out);
        }
    }
}

fn payload(body: &Value, exclusions: &[Vec<String>]) -> Value {
    let mut tables = Map::new();
    tables_with_rows(body, "", &mut tables);
    json!({"digest": digest(body, exclusions), "tables": tables})
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

/// The core's result laid out as the reference's `to_pandas()` columns, by
/// name; a column it lacks is named as missing.
fn result_as(predicted: &Value, names: &[String]) -> Value {
    let columns: Vec<String> = predicted["result"]["columns"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|c| c["name"].as_str().unwrap_or_default().to_owned())
        .collect();
    let mut at = Vec::new();
    for name in names {
        match columns.iter().position(|c| c == name) {
            Some(i) => at.push(i),
            None => return json!(format!("no column {name} in {columns:?}")),
        }
    }
    let rows: Vec<Value> = predicted["result"]["rows"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|row| Value::Array(at.iter().map(|i| row[*i].clone()).collect()))
        .collect();
    Value::Array(rows)
}

/// What this host made of the case: its outcome, and for a success its
/// calls and its result as the reference's columns.
fn got(case: &Case, names: &[String], exclusions: &[Vec<String>]) -> Value {
    let contract = root().join("contract");
    let bases = bases::load(&contract.join("bases.yaml")).unwrap();
    let observed = match host::run(case, &bases) {
        Ok(observed) => observed,
        Err(failure) => return json!({"outcome": format!("host failure: {failure}")}),
    };
    match &observed.answer {
        Answer::Error { code, .. } => json!({"outcome": "refusal", "code": code}),
        Answer::Predicted(predicted) => json!({
            "outcome": "success",
            "code": null,
            "payloads": calls(predicted)
                .iter()
                .map(|b| payload(b, exclusions))
                .collect::<Vec<_>>(),
            "result": result_as(predicted, names),
        }),
        other => json!({"outcome": format!("no answer of predict: {other:?}")}),
    }
}

/// What the vector holds this host to.
fn wanted(vector: &Value) -> Value {
    if vector["rust_host"] == "python-core" {
        let outcome = &vector["python_core"]["outcome"];
        return json!({"outcome": outcome["result"], "code": outcome["code"]});
    }
    let reference = &vector["reference"];
    let outcome = &reference["outcome"];
    if outcome["result"] == "refusal" {
        return json!({"outcome": "refusal", "code": outcome["code"]});
    }
    json!({
        "outcome": "success",
        "code": null,
        "payloads": reference["payloads"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| json!({"digest": p["digest"], "tables": p["tables"]}))
            .collect::<Vec<_>>(),
        "result": reference["result"]["cells"],
    })
}

fn names(vector: &Value) -> Vec<String> {
    vector["reference"]["result"]["columns"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|c| c["name"].as_str().unwrap().to_owned())
        .collect()
}

fn run(filter: impl Fn(&Value) -> bool) {
    let all = vectors();
    let exclusions = exclusions(&all);
    let mut failures = Vec::new();
    let mut ran = 0;
    for vector in all["vectors"].as_array().unwrap() {
        if vector["rust_host"] == "none" || !filter(vector) {
            continue;
        }
        ran += 1;
        let id = vector["id"].as_str().unwrap();
        let case: Case = serde_json::from_value(vector["case"].clone()).unwrap();
        let want = wanted(vector);
        let got = got(&case, &names(vector), &exclusions);
        if got != want {
            let differ: Vec<&str> = ["outcome", "code", "payloads", "result"]
                .into_iter()
                .filter(|part| got.get(*part) != want.get(*part))
                .collect();
            failures.push(format!(
                "{id}: {differ:?} differ: got {} {} ({} calls), want {} {} ({} calls)",
                got["outcome"],
                got["code"],
                got["payloads"].as_array().map_or(0, Vec::len),
                want["outcome"],
                want["code"],
                want["payloads"].as_array().map_or(0, Vec::len),
            ));
        }
    }
    assert!(ran > 0, "no vectors");
    assert!(
        failures.is_empty(),
        "{} forecasts answered otherwise than the reference:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn rust_host_answers_a_forecast_as_the_reference() {
    run(|v| v["reference"]["outcome"]["result"] == "success");
}

#[test]
fn rust_host_refuses_a_forecast_as_the_reference() {
    run(|v| v["reference"]["outcome"]["result"] == "refusal");
}

#[test]
fn the_comparison_reads_this_host_s_call_as_the_recorder_does() {
    // The control: the one-step forecast's single part, asked as the plain
    // regression it is, is the reference's request for that step, read and
    // hashed here exactly as the forecasts are. It passes before the fix.
    let all = vectors();
    let exclusions = exclusions(&all);
    let vector = all["vectors"]
        .as_array()
        .unwrap()
        .iter()
        .find(|v| v["id"] == "sum-days-1")
        .unwrap();
    let mut case: Case = serde_json::from_value(vector["case"].clone()).unwrap();
    case.input.pql = case.input.pql.replace(" FORECAST 1 TIMEFRAMES", "");
    assert_eq!(
        case.input.pql,
        "PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id"
    );
    let got = got(
        &case,
        &[
            "row_index".to_owned(),
            "id".to_owned(),
            "prediction".to_owned(),
        ],
        &exclusions,
    );
    assert_eq!(got["outcome"], "success", "{got}");
    assert_eq!(
        got["payloads"],
        json!([{
            "digest": vector["reference"]["payloads"][0]["digest"],
            "tables": vector["reference"]["payloads"][0]["tables"],
        }])
    );
    // And a call whose body differs is caught.
    let mut seeded = vector["reference"]["payloads"][0].clone();
    seeded["digest"] = json!("0");
    assert_ne!(got["payloads"][0], seeded);
    // The reference's prediction column, cell for cell, beside its ids.
    let cells = vector["reference"]["result"]["cells"].as_array().unwrap();
    let want: Vec<Value> = cells
        .iter()
        .map(|row| json!([row[0], row[1], row[3]]))
        .collect();
    assert_eq!(got["result"], Value::Array(want));
}

#[test]
fn the_vectors_hold_several_steps_and_refusals() {
    let all = vectors();
    let mut steps = BTreeMap::new();
    let mut refusals = 0;
    for vector in all["vectors"].as_array().unwrap() {
        let reference = &vector["reference"];
        let case: Case = serde_json::from_value(vector["case"].clone()).unwrap();
        assert_eq!(case.input.pql, vector["pql"].as_str().unwrap());
        if reference["outcome"]["result"] == "success" {
            steps.insert(
                vector["id"].as_str().unwrap().to_owned(),
                reference["payloads"].as_array().unwrap().len(),
            );
            assert_eq!(
                names(vector),
                ["row_index", "id", "forecast_step", "prediction"]
            );
        } else {
            refusals += 1;
        }
    }
    assert_eq!(steps["sum-days-3"], 3);
    assert_eq!(steps["count-days-4"], 4);
    assert_eq!(steps["sum-days-1"], 1);
    assert!(refusals >= 5);
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

/// An observation's `core.predicted` as the vectors record python-core's
/// (`forecast_record.observed_predicted`).
fn observed_predicted(observation: &Value, exclusions: &[Vec<String>]) -> Value {
    let predicted = &observation["core"]["predicted"];
    let body = &predicted["body"];
    let kind = match body {
        Value::Object(_) => "dict",
        Value::Array(_) => "list",
        Value::Null => "NoneType",
        _ => "other",
    };
    json!({
        "outcome": {
            "result": observation["outcome"]["result"],
            "code": observation["outcome"]["code"],
        },
        "body_type": kind,
        "body": if body.is_object() { payload(body, exclusions) } else { Value::Null },
        "model": predicted["model"],
        "metadata": predicted["metadata"],
        "result": predicted["result"],
    })
}

/// `cases` through `sdm-contract-run`, as CI runs it: each observation by
/// case id.
fn contract_run(name: &str, cases: &[Value]) -> BTreeMap<String, Value> {
    let dir = std::env::temp_dir().join(format!("t136fc-{name}-{}", std::process::id()));
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

#[test]
fn rust_host_s_observation_of_a_forecast_is_python_core_s() {
    // What sdm-contract-run writes of each forecast the reference answers:
    // its core.predicted conforms to contract/observation.schema.json, and
    // is python-core's observation of the same case (recorded by
    // forecast_record.py's `observation` part): the body an object, the
    // same request (its digest and tables, the request id masked), model,
    // metadata and result. Fails at f744de6b: the body is a list.
    let all = vectors();
    let exclusions = exclusions(&all);
    let schema: Value = serde_json::from_str(
        &std::fs::read_to_string(root().join("contract/observation.schema.json")).unwrap(),
    )
    .unwrap();
    let subschema = &schema["properties"]["core"]["properties"]["predicted"];
    assert_eq!(subschema["properties"]["body"]["type"], "object");
    let held: Vec<&Value> = all["vectors"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|v| v.get("python_core_observation").is_some())
        .collect();
    assert_eq!(held.len(), 9);
    let observed = contract_run(
        "observation",
        &held.iter().map(|v| v["case"].clone()).collect::<Vec<_>>(),
    );
    let mut failures = Vec::new();
    for vector in held {
        let id = vector["case"]["id"].as_str().unwrap();
        let observation = &observed[id];
        let mut problems = Vec::new();
        conforms(
            &observation["core"]["predicted"],
            subschema,
            "core/predicted",
            &mut problems,
        );
        let got = observed_predicted(observation, &exclusions);
        let want = &vector["python_core_observation"];
        for part in [
            "outcome",
            "body_type",
            "body",
            "model",
            "metadata",
            "result",
        ] {
            if got[part] != want[part] {
                problems.push(format!("{part} is not python-core's"));
            }
        }
        if !problems.is_empty() {
            failures.push(format!("{id}: {}", problems.join("; ")));
        }
    }
    assert!(
        failures.is_empty(),
        "rust-host's observations differ from python-core's:\n{}",
        failures.join("\n")
    );
}

#[test]
fn python_core_s_recorded_observation_is_of_the_last_step_s_call() {
    // The recorded shape, read against the reference: python-core's
    // observation keeps the last call's body, which is the reference's last
    // request, and the merged result, one row per entity and step.
    for vector in vectors()["vectors"].as_array().unwrap() {
        let Some(observed) = vector.get("python_core_observation") else {
            continue;
        };
        let payloads = vector["reference"]["payloads"].as_array().unwrap();
        let last = payloads.last().unwrap();
        assert_eq!(
            observed["body"]["digest"], last["digest"],
            "{}",
            vector["id"]
        );
        assert_eq!(
            observed["body"]["tables"], last["tables"],
            "{}",
            vector["id"]
        );
        assert_eq!(
            observed["result"]["rows"].as_array().unwrap().len(),
            2 * payloads.len()
        );
    }
}

// --- a forecast of many steps is refused at once -------------------------------

const QUICK: Duration = Duration::from_secs(2);

/// The forecast `sum-days-3` asked over `n` steps, and the S0 case
/// `s0.refuse.task-forecast-zero` asked the same: each case as JSON.
fn many_steps(n: &str) -> (Value, Value) {
    let all = vectors();
    let mut materialize = all["vectors"]
        .as_array()
        .unwrap()
        .iter()
        .find(|v| v["id"] == "sum-days-3")
        .unwrap()["case"]
        .clone();
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
    // two seconds: no step is built for a refused forecast's count.
    for n in ["1000000000", "4294967297", "10001", "10000", "201"] {
        let (materialize, validate) = many_steps(n);
        let (got, took) = timed(&materialize);
        assert_eq!(
            got,
            json!(["refusal", "INVALID_REQUEST"]),
            "materialize, {n} steps"
        );
        assert!(took < QUICK, "materialize, {n} steps: {took:?}");
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
