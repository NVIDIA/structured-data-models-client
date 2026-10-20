// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! T2.11's exits on `rust-host` (the package's test author's), beside
//! `contract/tests/test_t211_sampling.py` and `test_t211_unseeded.py`,
//! which hold `python-core` to the same ones: the sampling options.
//!
//! - `rust_host_answers_each_case_as_the_reference_does_on_duckdb` and
//!   `_on_sqlite`: `sdm-contract-run` (with `--database sqlite`, each case
//!   loaded by the contract loader) runs each T2.11 case (`sample.*`): its
//!   outcome (result, and the code where the reference refuses) is
//!   python-ref's committed line's, the core operations the case names
//!   ran, and for a success the call the core wrote (`core.predicted.body`)
//!   is the request python-ref sent (the `payload` rule: each but its
//!   request id) and the result carries each column python-ref's does (an
//!   `embeddings` column where the endpoint answered embeddings: the
//!   `response_field` exit). rust-host's Postgres pass replays
//!   python-core's fetches (KO2-20), which `cross-host (postgres)` holds.
//! - `rust_host_draws_an_unseeded_sample_with_the_reference_s_properties_on_duckdb`
//!   and `_on_sqlite`: the unseeded cases (`contract/tests/t211-unseeded/`)
//!   judged by properties (KO2-76) against
//!   `contract/t211-unseeded-vectors.json`: a success through predict, the
//!   entities asked about as the reference asks them, as many context
//!   examples, each of a user, and every sampled row a row of its table.
//!
//! Gating: `contract/tests/t211-pending.json`'s `pending` names, as
//! `rust:<case key>` and `unseeded-rust:<case key>`, the keys that failed
//! when written; the cases are its `authored` keys of this host. Each
//! pending key must still fail (strict); once one passes it fails, naming
//! itself, until it is taken out.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use serde_json::Value;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

fn read_json(path: &Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

/// The pending file's `list` names with `prefix`, as case keys.
fn named(list: &str, prefix: &str) -> BTreeSet<String> {
    read_json(&root().join("contract/tests/t211-pending.json"))[list]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|n| n.as_str().unwrap().strip_prefix(prefix))
        .map(str::to_owned)
        .collect()
}

/// The case key of `id` on `database`, as the runners key it.
fn key(id: &str, database: &str) -> String {
    if database == "duckdb" {
        id.to_owned()
    } else {
        format!("{id}@{database}")
    }
}

/// The cases authored for `database` under `prefix`, by id.
fn case_ids(prefix: &str, database: &str) -> Vec<String> {
    named("authored", prefix)
        .into_iter()
        .filter_map(|k| match k.split_once('@') {
            Some((id, on)) if on == database => Some(id.to_owned()),
            None if database == "duckdb" => Some(k),
            _ => None,
        })
        .collect()
}

/// rust-host's observations of the cases `ids` in `dir` on `database`, by
/// key.
fn observe(dir_name: &str, ids: &[String], database: &str) -> BTreeMap<String, Value> {
    let dir = std::env::temp_dir().join(format!(
        "t211-{}-{database}-{}",
        dir_name.replace('/', "-"),
        std::process::id()
    ));
    std::fs::create_dir_all(dir.join("cases")).unwrap();
    for id in ids {
        std::fs::copy(
            root().join(format!("{dir_name}/{id}.json")),
            dir.join(format!("cases/{id}.json")),
        )
        .unwrap();
    }
    let out = dir.join("rust-host.jsonl");
    let mut arguments: Vec<String> = Vec::new();
    if database != "duckdb" {
        arguments.extend(["--database".to_owned(), database.to_owned()]);
    }
    arguments.extend([
        "--out".to_owned(),
        out.to_str().unwrap().to_owned(),
        dir.join("cases").to_str().unwrap().to_owned(),
    ]);
    let done = Command::new(env!("CARGO_BIN_EXE_sdm-contract-run"))
        .args(&arguments)
        .current_dir(root())
        .stdin(Stdio::null())
        .output()
        .unwrap();
    assert!(
        done.status.success(),
        "sdm-contract-run on {database} failed: {}{}",
        String::from_utf8_lossy(&done.stdout),
        String::from_utf8_lossy(&done.stderr)
    );
    let text = std::fs::read_to_string(&out).unwrap();
    std::fs::remove_dir_all(&dir).ok();
    text.lines()
        .map(|line| {
            let o: Value = serde_json::from_str(line).unwrap();
            (o["case"].as_str().unwrap().to_owned(), o)
        })
        .collect()
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
        .filter(|line| line.contains("\"sample."))
        .map(|line| {
            let o: Value = serde_json::from_str(line).unwrap();
            (o["case"].as_str().unwrap().to_owned(), o)
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

/// Two JSON numbers equal as numbers (an integer exactly, else as the
/// double each reads as), as the Python comparison holds them.
fn same_number(a: &serde_json::Number, b: &serde_json::Number) -> bool {
    if let (Some(x), Some(y)) = (a.as_i64(), b.as_i64()) {
        return x == y;
    }
    if let (Some(x), Some(y)) = (a.as_u64(), b.as_u64()) {
        return x == y;
    }
    a.as_f64() == b.as_f64()
}

/// Where `got` differs from `want`, a few paths.
fn differences(want: &Value, got: &Value, at: &str, out: &mut Vec<String>) {
    if out.len() >= 8 {
        return;
    }
    match (want, got) {
        (Value::Object(w), Value::Object(g)) => {
            let names: BTreeSet<&String> = w.keys().chain(g.keys()).collect();
            for name in names {
                differences(
                    w.get(name).unwrap_or(&Value::Null),
                    g.get(name).unwrap_or(&Value::Null),
                    &format!("{at}/{name}"),
                    out,
                );
            }
        }
        (Value::Array(w), Value::Array(g)) if w.len() == g.len() => {
            for (i, (a, b)) in w.iter().zip(g).enumerate() {
                differences(a, b, &format!("{at}[{i}]"), out);
            }
        }
        (Value::Array(w), Value::Array(g)) => out.push(format!(
            "{at}: {} items where the reference has {}",
            g.len(),
            w.len()
        )),
        (Value::Number(a), Value::Number(b)) if same_number(a, b) => {}
        _ if want == got => {}
        _ => out.push(format!("{at}: {got} where the reference has {want}")),
    }
}

/// The core operations a case names, in its `core`.
fn operations(id: &str) -> Vec<String> {
    read_json(&root().join(format!("contract/cases/s1/{id}.json")))["core"]
        .as_object()
        .unwrap()
        .keys()
        .cloned()
        .collect()
}

/// Why rust-host's `line` does not answer as python-ref's `wanted`, if it
/// does not.
fn problem(id: &str, line: &Value, wanted: &Value) -> Option<String> {
    if line["identity"]["runner"] != "rust-host" {
        return Some("not rust-host's".to_owned());
    }
    let ran: BTreeSet<&str> = line["identity"]["operations"]
        .as_array()
        .unwrap()
        .iter()
        .map(|o| o.as_str().unwrap())
        .collect();
    for operation in operations(id) {
        if !ran.contains(operation.as_str()) {
            return Some(format!("its core operation {operation} did not run"));
        }
    }
    let outcome = |o: &Value| (o["outcome"]["result"].clone(), o["outcome"]["code"].clone());
    if outcome(line) != outcome(wanted) {
        return Some(format!(
            "{:?} where python-ref has {:?}",
            outcome(line),
            outcome(wanted)
        ));
    }
    if wanted["outcome"]["result"] != "success" {
        return None;
    }
    let sent: Vec<Value> = wanted["public"]["payloads"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| without_request_id(&p["parsed"]))
        .collect();
    let predicted = &line["core"]["predicted"];
    let written: Vec<Value> = predicted
        .get("body")
        .filter(|b| !b.is_null())
        .map(|b| vec![without_request_id(b)])
        .unwrap_or_default();
    let mut found = Vec::new();
    differences(&Value::Array(sent), &Value::Array(written), "", &mut found);
    if !found.is_empty() {
        return Some(format!("the payload differs: {}", found.join("; ")));
    }
    let columns: BTreeSet<&str> = predicted["result"]["columns"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|c| c["name"].as_str())
        .collect();
    for column in wanted["public"]["result"]["columns"].as_array().unwrap() {
        let name = column["name"].as_str().unwrap();
        if !columns.contains(name) {
            return Some(format!("the result has no column {name}: {columns:?}"));
        }
    }
    None
}

/// Each T2.11 case on `database` through rust-host, held to python-ref's
/// line: a key not pending must answer as the reference does, a pending
/// key must still differ.
fn answers_as_the_reference(database: &str) {
    let ids = case_ids("rust:", database);
    assert!(!ids.is_empty(), "no T2.11 case authored on {database}");
    let observed = observe("contract/cases/s1", &ids, database);
    let reference = reference();
    let pending = named("pending", "rust:");
    let mut problems = Vec::new();
    for id in ids {
        let key = key(&id, database);
        let found = match (observed.get(&key), reference.get(&key)) {
            (None, _) => Some("not observed".to_owned()),
            (_, None) => Some("no python-ref line".to_owned()),
            (Some(line), Some(wanted)) => problem(&id, line, wanted),
        };
        match (found, pending.contains(&key)) {
            (Some(why), false) => problems.push(format!("{key}: {why}")),
            (None, true) => problems.push(format!(
                "{key} passes now: take rust:{key} out of \
                 contract/tests/t211-pending.json's pending"
            )),
            _ => {}
        }
    }
    assert!(problems.is_empty(), "{}", problems.join("\n"));
}

#[test]
fn rust_host_answers_each_case_as_the_reference_does_on_duckdb() {
    answers_as_the_reference("duckdb");
}

#[test]
fn rust_host_answers_each_case_as_the_reference_does_on_sqlite() {
    answers_as_the_reference("sqlite");
}

/// A value as a request writes it, read back as the recorder writes the
/// source's: a number (plain, or as the observation tags its token,
/// `{"number": "<token>"}`) as a float, anything else as itself.
fn written(value: &Value) -> Value {
    let token = value
        .as_object()
        .filter(|o| o.len() == 1)
        .and_then(|o| o.get("number"))
        .and_then(Value::as_str);
    match (token, value.as_f64()) {
        (Some(token), _) => token
            .parse::<f64>()
            .map_or_else(|_| value.clone(), |n| serde_json::json!(n)),
        (None, Some(number)) => serde_json::json!(number),
        (None, None) => value.clone(),
    }
}

/// Each feature column of a sampled table, as the recorder names them.
fn features(table: &str) -> Option<&'static [&'static str]> {
    match table {
        "users" => Some(&["user_id", "age", "country"]),
        "orders" => Some(&["price", "ts"]),
        "lines" => Some(&["qty"]),
        _ => None,
    }
}

/// Each property of an unseeded draw `line` breaks against the reference's
/// `wanted` properties and the case's `given` source.
fn unseeded_problems(line: &Value, wanted: &Value, given: &Value) -> Vec<String> {
    let mut found = Vec::new();
    if line["outcome"]["result"] != wanted["outcome"]["result"]
        || line["outcome"]["code"] != wanted["outcome"]["code"]
    {
        return vec![format!(
            "{} ({}) where the reference has {} ({})",
            line["outcome"]["result"],
            line["outcome"]["code"],
            wanted["outcome"]["result"],
            wanted["outcome"]["code"]
        )];
    }
    let body = &line["core"]["predicted"]["body"];
    let mut differ = Vec::new();
    differences(
        &wanted["predict"]["instance_table"],
        &body["predict"]["instance_table"],
        "/predict/instance_table",
        &mut differ,
    );
    found.extend(differ);
    let context = &body["context"]["instance_table"];
    let rows = context["rows"].as_array().cloned().unwrap_or_default();
    if Some(rows.len() as u64) != wanted["context_rows"].as_u64() {
        found.push(format!(
            "{} context examples where the reference draws {}",
            rows.len(),
            wanted["context_rows"]
        ));
    }
    let columns: Vec<&str> = context["columns"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .collect();
    let entities: BTreeSet<String> = given["entities"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| written(e).to_string())
        .collect();
    if let Some(at) = columns.iter().position(|c| *c == "__kumo_entity_ref_0") {
        for row in &rows {
            if !entities.contains(&written(&row[at]).to_string()) {
                found.push(format!("a context example of no user: {}", row[at]));
            }
        }
    }
    for section in ["context", "predict"] {
        let tables = body[section]["related_tables"]
            .as_object()
            .cloned()
            .unwrap_or_default();
        for (table, sampled) in tables {
            let Some(names) = features(&table) else {
                found.push(format!("{section} samples {table}, which it cannot"));
                continue;
            };
            let source: BTreeSet<String> = given["source"][&table]
                .as_array()
                .into_iter()
                .flatten()
                .map(|r| Value::Array(r.as_array().unwrap().iter().map(written).collect()).to_string())
                .collect();
            let held: Vec<&str> = sampled["columns"]
                .as_array()
                .unwrap()
                .iter()
                .filter_map(Value::as_str)
                .collect();
            let at: Vec<usize> = names
                .iter()
                .map(|n| held.iter().position(|c| c == n).unwrap_or(usize::MAX))
                .collect();
            for row in sampled["rows"].as_array().into_iter().flatten() {
                let values = Value::Array(
                    at.iter()
                        .map(|i| row.get(*i).map(written).unwrap_or(Value::Null))
                        .collect(),
                );
                if !source.contains(&values.to_string()) {
                    found.push(format!("{section} {table} row {values} is not in it"));
                }
            }
        }
    }
    found.truncate(8);
    found
}

/// The unseeded cases on `database` through rust-host, judged by their
/// properties (KO2-76).
fn draws_as_the_reference(database: &str) {
    let ids = case_ids("unseeded-rust:", database);
    assert!(!ids.is_empty(), "no unseeded case authored on {database}");
    let observed = observe("contract/tests/t211-unseeded", &ids, database);
    let vectors = read_json(&root().join("contract/t211-unseeded-vectors.json"));
    let pending = named("pending", "unseeded-rust:");
    let mut problems = Vec::new();
    for id in ids {
        let key = key(&id, database);
        let found = match observed.get(&key) {
            None => vec!["not observed".to_owned()],
            Some(line) => unseeded_problems(line, &vectors["keys"][&key], &vectors["cases"][&id]),
        };
        match (found.is_empty(), pending.contains(&key)) {
            (false, false) => problems.push(format!("{key}: {}", found.join("; "))),
            (true, true) => problems.push(format!(
                "{key} passes now: take unseeded-rust:{key} out of \
                 contract/tests/t211-pending.json's pending"
            )),
            _ => {}
        }
    }
    assert!(problems.is_empty(), "{}", problems.join("\n"));
}

#[test]
fn rust_host_draws_an_unseeded_sample_with_the_reference_s_properties_on_duckdb() {
    draws_as_the_reference("duckdb");
}

#[test]
fn rust_host_draws_an_unseeded_sample_with_the_reference_s_properties_on_sqlite() {
    draws_as_the_reference("sqlite");
}
