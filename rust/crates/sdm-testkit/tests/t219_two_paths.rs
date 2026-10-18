// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! T2.19's exits on `rust-host` (the package's test author's), beside
//! `contract/tests/test_t219_two_paths.py`, which holds `python-core` to
//! the same ones and the cases to `main`'s shapes: "a table reached by two
//! paths in one hop read only up to the slices it held when the hop
//! began", "one column linking two tables" and "a `from_source` diamond
//! compared at the payload".
//!
//! - `rust_host_answers_each_case_as_the_reference_does_on_duckdb` and
//!   `_on_sqlite`: `sdm-contract-run` (with `--database sqlite`, each case
//!   loaded by the contract loader) runs each T2.19 case (`shape.*`), its
//!   core operations through predict run, its outcome (result and code)
//!   python-ref's committed line's, and the call the core wrote
//!   (`core.predicted.body`) the request python-ref sent (the `payload`
//!   rule: each but its request id, drawn at random). rust-host's Postgres
//!   pass replays python-core's fetches (KO2-20), which `cross-host
//!   (postgres)` holds to python-core.
//!
//! Gating: `contract/tests/t219-pending.json`'s `pending` names, as
//! `rust:<case key>`, the keys that failed when written; the cases are its
//! `authored` `rust:` keys. Each pending key must still fail (strict);
//! once one passes it fails, naming itself, until it is taken out.

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

/// The pending file's `list` names of this host, as case keys.
fn named(list: &str) -> BTreeSet<String> {
    read_json(&root().join("contract/tests/t219-pending.json"))[list]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|n| n.as_str().unwrap().strip_prefix("rust:"))
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

/// The T2.19 cases authored for `database`, by id.
fn case_ids(database: &str) -> Vec<String> {
    named("authored")
        .into_iter()
        .filter_map(|k| match k.split_once('@') {
            Some((id, on)) if on == database => Some(id.to_owned()),
            None if database == "duckdb" => Some(k),
            _ => None,
        })
        .collect()
}

/// rust-host's observations of the T2.19 cases on `database`, by key.
fn observe(database: &str) -> BTreeMap<String, Value> {
    let dir = std::env::temp_dir().join(format!("t219-{database}-{}", std::process::id()));
    std::fs::create_dir_all(dir.join("cases")).unwrap();
    for id in case_ids(database) {
        std::fs::copy(
            root().join(format!("contract/cases/s1/{id}.json")),
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

/// Why rust-host's `line` does not answer as python-ref's `wanted`, if it
/// does not.
fn problem(line: &Value, wanted: &Value) -> Option<String> {
    if line["identity"]["runner"] != "rust-host" {
        return Some("not rust-host's".to_owned());
    }
    let ran: BTreeSet<&str> = line["identity"]["operations"]
        .as_array()
        .unwrap()
        .iter()
        .map(|o| o.as_str().unwrap())
        .collect();
    for operation in ["describe_source", "materialize", "predict"] {
        if !ran.contains(operation) {
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
    let sent: Vec<Value> = wanted["public"]["payloads"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| without_request_id(&p["parsed"]))
        .collect();
    let written: Vec<Value> = line["core"]["predicted"]
        .get("body")
        .filter(|b| !b.is_null())
        .map(|b| vec![without_request_id(b)])
        .unwrap_or_default();
    let mut found = Vec::new();
    differences(&Value::Array(sent), &Value::Array(written), "", &mut found);
    (!found.is_empty()).then(|| format!("the payload differs: {}", found.join("; ")))
}

/// Each T2.19 case on `database` through rust-host, held to python-ref's
/// line: a key not pending must answer as the reference does, a pending
/// key must still differ.
fn answers_as_the_reference(database: &str) {
    let ids = case_ids(database);
    assert!(!ids.is_empty(), "no T2.19 case authored on {database}");
    let observed = observe(database);
    let reference = reference();
    let pending = named("pending");
    let mut problems = Vec::new();
    for id in ids {
        let key = key(&id, database);
        let found = match (observed.get(&key), reference.get(&key)) {
            (None, _) => Some("not observed".to_owned()),
            (_, None) => Some("no python-ref line".to_owned()),
            (Some(line), Some(wanted)) => problem(line, wanted),
        };
        match (found, pending.contains(&key)) {
            (Some(why), false) => problems.push(format!("{key}: {why}")),
            (None, true) => problems.push(format!(
                "{key} passes now: take rust:{key} out of \
                 contract/tests/t219-pending.json's pending"
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
