// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! T2.12's exits on `rust-host` (the package's test author's): the schema
//! rules. `contract/tests/test_t212_schema_rules.py` holds `python-core`
//! to the same cases, to the key-type pair matrix for composite keys and
//! to FR P3-8's key-cast vectors.
//!
//! The cases are the `rule.*` S1 cases, one or more per rule the row lifts
//! (`float_nan`, `float_signed_zero`, `key_tie`, `key_name_order`,
//! `composite_key`), and the nine `edit.*` cases the row admits as
//! acceptance cases. Each test's line says what it proves:
//!
//! - "cases per rule" and "the nine `edit.*` cases acceptance": this host
//!   (`sdm-contract-run`, the test kit's DuckDB and dev-only `rusqlite`
//!   providers) runs each case on DuckDB and SQLite (the float cases on
//!   DuckDB only: SQLite cannot hold a NaN or -0.0), each of its core
//!   operations runs, and its outcome (class and code) is python-ref's
//!   committed line's, or an approved, permanent register entry names the
//!   key; where both reached `predict`, the request this host wrote is the
//!   one the scripted endpoint received from python-ref, its request id
//!   aside. Postgres is held on this host by CI's `cross-host (postgres)`
//!   job, which replays python-core's fetches.
//!
//! Gating: `contract/tests/t212-pending.json`'s `pending` names, as
//! `rust:<key>`, the keys that failed when written; each must still fail
//! (strict), and the implementer removes each as it passes.

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

/// The `rust:` names of a list of the pending file.
fn named(list: &str) -> BTreeSet<String> {
    read_json(&root().join("contract/tests/t212-pending.json"))[list]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|n| n.as_str().unwrap().strip_prefix("rust:"))
        .map(str::to_owned)
        .collect()
}

/// The T2.12 cases: the rule cases and the nine edit cases, by id.
fn cases() -> BTreeMap<String, Value> {
    let mut out = BTreeMap::new();
    for entry in std::fs::read_dir(root().join("contract/cases/s1")).unwrap() {
        let path = entry.unwrap().path();
        let name = path.file_name().unwrap().to_str().unwrap().to_owned();
        if name.starts_with("rule.") || name.starts_with("edit.") {
            let case = read_json(&path);
            out.insert(case["id"].as_str().unwrap().to_owned(), case);
        }
    }
    out
}

/// The databases a case runs on, as its file says (all three unless its
/// `databases` narrows them).
fn on(case: &Value, database: &str) -> bool {
    match case.get("databases") {
        Some(Value::Array(named)) => named.iter().any(|d| d == database),
        _ => true,
    }
}

fn key(id: &str, database: &str) -> String {
    if database == "duckdb" {
        id.to_owned()
    } else {
        format!("{id}@{database}")
    }
}

/// This host's observation of each case on `database`, by key.
fn observe(database: &str, ids: &[String]) -> BTreeMap<String, Value> {
    let dir = std::env::temp_dir().join(format!("t212-{database}-{}", std::process::id()));
    std::fs::create_dir_all(dir.join("cases")).unwrap();
    for id in ids {
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

/// The python-ref observations, by key.
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

/// The decision rows that approve only until a package lands, as
/// `scripts/rust_core/register.py`'s `TEMPORARY` names them.
fn temporary() -> Vec<String> {
    let text = std::fs::read_to_string(root().join("scripts/rust_core/register.py")).unwrap();
    let line = text
        .lines()
        .find_map(|l| l.strip_prefix("TEMPORARY = ("))
        .expect("register.py names TEMPORARY");
    line.trim_end_matches(')')
        .split(',')
        .map(|r| r.trim().trim_matches('\'').to_owned())
        .filter(|r| !r.is_empty())
        .collect()
}

/// The register's umbrellas: approved as E92's Phase 2 class, they excuse
/// no case (T1.33).
const UMBRELLAS: [&str; 4] = ["P1B-1", "P1B-2", "P1B-3", "P1B-4"];

/// The approved entry, no umbrella and citing no temporary approval, whose
/// text names `wanted` (a line key in its `cases` or `case_outcomes`):
/// each entry read from the register's text, block by block.
fn registered(wanted: &str) -> Option<String> {
    let text = std::fs::read_to_string(root().join("docs/design/rust-core/register.yaml")).unwrap();
    let temporary = temporary();
    for block in text.split("\n- id: ").skip(1) {
        let id = block.lines().next().unwrap_or_default().trim().to_owned();
        let approved = block.lines().any(|l| l.trim_end() == "  status: approved");
        let row = block
            .lines()
            .find(|l| l.starts_with("  decision_ref:"))
            .and_then(|l| l.split("row: ").nth(1))
            .map(|r| r.trim_end_matches('}').trim().to_owned());
        let names = block.split(|c: char| {
            c.is_whitespace() || matches!(c, ',' | '[' | ']' | '{' | '}' | '\'' | '"')
        });
        if approved
            && !UMBRELLAS.contains(&id.as_str())
            && !row.is_some_and(|r| temporary.contains(&r))
            && names.into_iter().any(|n| n == wanted)
        {
            return Some(id);
        }
    }
    None
}

fn without_request_id(request: &Value) -> Value {
    let mut request = request.clone();
    if let Some(metadata) = request.get_mut("metadata").and_then(Value::as_object_mut) {
        metadata.remove("request_id");
    }
    request
}

fn same_number(a: &serde_json::Number, b: &serde_json::Number) -> bool {
    if let (Some(x), Some(y)) = (a.as_i64(), b.as_i64()) {
        return x == y;
    }
    if let (Some(x), Some(y)) = (a.as_u64(), b.as_u64()) {
        return x == y;
    }
    a.as_f64() == b.as_f64()
}

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

/// Why this host's `line` for `key` fails the case, or None.
fn problem(key: &str, case: &Value, line: &Value, wanted: &Value) -> Option<String> {
    if line["identity"]["runner"] != "rust-host" {
        return Some("not rust-host's".to_owned());
    }
    let ran: BTreeSet<&str> = line["identity"]["operations"]
        .as_array()
        .unwrap()
        .iter()
        .map(|o| o.as_str().unwrap())
        .collect();
    for operation in case["core"].as_object().unwrap().keys() {
        if !ran.contains(operation.as_str()) {
            return Some(format!("its core operation {operation} did not run"));
        }
    }
    let outcome = |o: &Value| (o["outcome"]["result"].clone(), o["outcome"]["code"].clone());
    if outcome(line) != outcome(wanted) && registered(key).is_none() {
        return Some(format!(
            "{:?} where python-ref has {:?}, and no approved, permanent entry names it",
            outcome(line),
            outcome(wanted)
        ));
    }
    let written = line["core"]["predicted"]
        .get("body")
        .filter(|b| !b.is_null());
    if wanted["outcome"]["stage"] == "predict"
        && let Some(body) = written
    {
        let sent: Vec<Value> = wanted["public"]["payloads"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| without_request_id(&p["parsed"]))
            .collect();
        let mut found = Vec::new();
        differences(
            &Value::Array(sent),
            &Value::Array(vec![without_request_id(body)]),
            "",
            &mut found,
        );
        if !found.is_empty() {
            return Some(format!("the payload differs: {}", found.join("; ")));
        }
    }
    None
}

fn answers_as_the_reference(database: &str) {
    let all = cases();
    let ids: Vec<String> = all
        .iter()
        .filter(|(_, case)| on(case, database))
        .map(|(id, _)| id.clone())
        .collect();
    assert_eq!(
        ids.iter().filter(|id| id.starts_with("edit.")).count(),
        9,
        "the nine edit cases"
    );
    let authored = named("authored");
    for id in &ids {
        assert!(
            authored.contains(&key(id, database)),
            "{id}: not authored on {database}"
        );
    }
    let observed = observe(database, &ids);
    let reference = reference();
    let pending = named("pending");
    let mut problems = Vec::new();
    for id in ids {
        let key = key(&id, database);
        let found = match (observed.get(&key), reference.get(&key)) {
            (None, _) => Some("not observed".to_owned()),
            (_, None) => Some("no python-ref line".to_owned()),
            (Some(line), Some(wanted)) => problem(&key, &all[&id], line, wanted),
        };
        match (found, pending.contains(&key)) {
            (Some(why), false) => problems.push(format!("{key}: {why}")),
            (None, true) => problems.push(format!(
                "{key} passes now: take rust:{key} out of \
                 contract/tests/t212-pending.json's pending"
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

#[test]
fn the_pending_list_only_shrinks() {
    let authored = named("authored");
    assert!(named("pending").is_subset(&authored));
    assert!(!authored.is_empty());
}
